"""Decision lifecycle for the fixed post-draft R-event recovery action."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping

from history_memory.events import EventStore

from ..event_native_s0_policy import PreparedEventNativeS0
from ..policy import PolicyInputError
from .admission import admit_event, metadata_after_admission
from .config import (
    E1_RECOVERY_VERSION,
    TASK_GENERATION_LIMIT,
    ceil_fraction,
    parse_recovery_config,
    public_recovery_config,
)
from .gate import canonical, evaluate_gate
from .source import select_source_event


@dataclass
class PreparedEventNativeRecovery:
    """One prepared S0 decision plus its observable source-selection input."""

    memory: Any
    metadata: dict[str, Any]
    eligible_chunks: tuple[Any, ...]
    _base_prepared: PreparedEventNativeS0 = field(repr=False, compare=False)
    _store: EventStore = field(repr=False, compare=False)
    _tools: tuple[dict[str, Any], ...] = field(repr=False, compare=False)
    _shadow_features: Any = field(default=None, repr=False, compare=False)
    _features_observed: bool = field(default=False, repr=False, compare=False)
    _checked_result: dict[str, Any] | None = field(
        default=None, repr=False, compare=False
    )


class EventNativeRecoveryController:
    """Wrap an event-native S0 controller with the fixed E1 R-event action."""

    def __init__(self, base: Any, config: Mapping[str, Any]) -> None:
        if not callable(getattr(base, "prepare", None)) or not callable(
            getattr(base, "reconsider", None)
        ):
            raise TypeError(
                "post-draft recovery requires an event-native controller"
            )
        self.base = base
        self.config = parse_recovery_config(config)
        self.policy_config = base.policy_config
        self.kv_bytes_per_token = base.kv_bytes_per_token
        self.tokenizer = base.tokenizer
        self.packing = base.packing
        self.model_context = base.model_context
        self.required_task_generation_limit = TASK_GENERATION_LIMIT
        self._prepared: dict[
            tuple[str, str], PreparedEventNativeRecovery
        ] = {}
        self._recovery_counts: dict[str, int] = {}

    def prepare(
        self, payload: Mapping[str, Any], *, ratio: int, max_new_tokens: int
    ):
        base_prepared = self.base.prepare(
            payload, ratio=ratio, max_new_tokens=max_new_tokens
        )
        session_id = payload.get("session_id")
        decision_key = payload.get("decision_key")
        key = (session_id, decision_key)
        cached = self._prepared.get(key)
        if cached is not None:
            if cached._base_prepared is not base_prepared:
                raise PolicyInputError(
                    "Recovery decision cache differs from the S0 decision"
                )
            return cached
        messages = payload.get("messages")
        tools = payload.get("tools") or []
        store = EventStore.from_messages(session_id, messages)
        prepared = PreparedEventNativeRecovery(
            memory=base_prepared.memory,
            metadata=copy.deepcopy(base_prepared.metadata),
            eligible_chunks=tuple(base_prepared.eligible_chunks),
            _base_prepared=base_prepared,
            _store=store,
            _tools=tuple(copy.deepcopy(tools)),
        )
        prepared.metadata["route"] = {
            **copy.deepcopy(prepared.metadata["route"]),
            "baseline_identity": (
                prepared.metadata["route"]["baseline_identity"]
                + "+e1-post-draft-event-recovery"
            ),
            "recovery_enabled": self.config["gate"] != "disabled",
            "max_generations_per_decision": (
                1 if self.config["gate"] == "disabled" else 2
            ),
        }
        prepared.metadata["post_draft_recovery_config"] = (
            public_recovery_config(self.config)
        )
        self._prepared[key] = prepared
        return prepared

    def observe_draft_features(
        self, *, session_id: str, decision_key: str, shadow_features: Any
    ) -> None:
        """Bind generation-time scalar features to the exact held draft."""

        prepared = self._prepared.get((session_id, decision_key))
        if prepared is None:
            raise PolicyInputError(
                "Draft features arrived before recovery preparation"
            )
        snapshot = copy.deepcopy(shadow_features)
        if prepared._features_observed:
            if canonical(snapshot) != canonical(prepared._shadow_features):
                raise PolicyInputError(
                    "Different draft features reused one decision key"
                )
            return
        if prepared._checked_result is not None:
            raise PolicyInputError(
                "Draft features arrived after recovery reconsideration"
            )
        prepared._shadow_features = snapshot
        prepared._features_observed = True

    def reconsider(
        self,
        prepared: PreparedEventNativeRecovery,
        draft_tool_calls: Any,
        *,
        draft_text: str,
        parse_error: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(prepared, PreparedEventNativeRecovery):
            raise PolicyInputError(
                "Prepared decision belongs to another controller"
            )
        base_result = self.base.reconsider(
            prepared._base_prepared,
            draft_tool_calls,
            draft_text=draft_text,
            parse_error=parse_error,
        )
        if prepared._checked_result is not None:
            return _copy_result(prepared._checked_result)

        decision_index = prepared.metadata["decision_index"]
        decision = _decision(decision_index, self.config["gate"])
        if self.config["gate"] == "disabled":
            decision.update(status="no_op", reason="recovery_disabled")
            return self._finish(prepared, base_result, decision)

        gate = self._gate(prepared)
        decision["gate"] = gate
        if not gate["triggered"]:
            decision.update(status="abstain", reason=gate["reason"])
            return self._finish(prepared, base_result, decision)

        recovered = self._recovery_counts.get(prepared._store.session_id, 0)
        quota = ceil_fraction(decision_index, self.config["quota"])
        decision["quota"] = {
            "policy": "online_cumulative_ceil_fraction_v1",
            "decision_index": decision_index,
            "recovery_count_before": recovered,
            "recovery_limit_at_decision": quota,
            "task_generation_limit": self.config["task_generation_limit"],
            "projected_generation_calls_with_regeneration": (
                decision_index + recovered + 1
            ),
        }
        if recovered >= quota:
            decision.update(
                status="abstain", reason="online_recovery_quota_exhausted"
            )
            return self._finish(prepared, base_result, decision)
        if (
            decision_index + recovered + 1
            > self.config["task_generation_limit"]
        ):
            decision.update(
                status="abstain", reason="shared_task_generation_limit"
            )
            return self._finish(prepared, base_result, decision)

        candidate, source = self._select_event(
            prepared, draft_tool_calls, draft_text=draft_text
        )
        decision["source"] = source
        decision["candidate_event_id"] = candidate
        if candidate is None:
            decision.update(status="abstain", reason=source["reason"])
            return self._finish(prepared, base_result, decision)

        admitted = self._admit(prepared, candidate)
        decision["allocation"] = admitted["receipt"]
        if admitted["measure"] is None:
            decision.update(
                status="abstain", reason="candidate_not_admitted_under_b0"
            )
            return self._finish(prepared, base_result, decision)

        measure = admitted["measure"]
        metadata = self._metadata_after_admission(
            prepared, measure, candidate, admitted["receipt"]
        )
        self._recovery_counts[prepared._store.session_id] = recovered + 1
        decision.update(
            status="recover",
            reason="gate_triggered_event_admitted",
            upgrade_count=1,
            upgraded_event_id=candidate,
            regeneration_allowed=True,
        )
        metadata["exact_recovery"] = copy.deepcopy(decision)
        result = {
            "regenerate": True,
            "memory": measure.memory,
            "metadata": metadata,
            "decision": copy.deepcopy(decision),
        }
        prepared._checked_result = _copy_result(result)
        return result

    def _gate(self, prepared: PreparedEventNativeRecovery) -> dict[str, Any]:
        store = getattr(prepared, "_store", None)
        metadata = getattr(prepared, "metadata", {})
        return evaluate_gate(
            self.config,
            session_id=getattr(store, "session_id", None),
            decision_key=metadata.get("decision_key"),
            shadow_features=prepared._shadow_features,
        )

    def _select_event(self, prepared, draft_tool_calls, *, draft_text):
        return select_source_event(
            prepared, draft_tool_calls, draft_text=draft_text
        )

    def _admit(self, prepared, candidate):
        return admit_event(self.base, prepared, candidate)

    def _metadata_after_admission(
        self, prepared, measure, candidate, receipt
    ):
        return metadata_after_admission(
            self.base, prepared, measure, candidate, receipt
        )

    def _finish(self, prepared, base_result, decision):
        decision.setdefault("regeneration_allowed", False)
        metadata = copy.deepcopy(base_result["metadata"])
        metadata["post_draft_recovery_config"] = public_recovery_config(
            self.config
        )
        metadata["exact_recovery"] = copy.deepcopy(decision)
        metadata["post_draft_exact_recovery_applied"] = False
        result = {
            "regenerate": False,
            "memory": base_result["memory"],
            "metadata": metadata,
            "decision": copy.deepcopy(decision),
        }
        prepared._checked_result = _copy_result(result)
        return result


def _decision(decision_index: int, gate: str) -> dict[str, Any]:
    return {
        "version": E1_RECOVERY_VERSION,
        "status": "started",
        "reason": None,
        "gate_type": gate,
        "decision_index": decision_index,
        "candidate_event_id": None,
        "bindings": [],
        "judges_action_correctness": False,
        "uses_gold_future_or_tool_result": False,
        "upgrade_count": 0,
        "upgraded_event_id": None,
        "regeneration_allowed": False,
    }


def _copy_result(value):
    return {
        "regenerate": value["regenerate"],
        "memory": value["memory"],
        "metadata": copy.deepcopy(value["metadata"]),
        "decision": copy.deepcopy(value["decision"]),
    }
