"""Stateful CPU policy and packing for event-native inference requests."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from history_memory.events import EventStore
from history_memory.packing import (
    MemoryView,
    PackedMemory,
    PackingBudgetError,
    encode_event_chunks,
    native_ids,
    pack_memory,
    raw_workspace_messages,
    select_view,
    visible_message,
)

from .policy import BudgetExceeded, ConversationMemory, PolicyInputError, RuntimeConfig


EVENT_NATIVE_POLICY_VERSION = "event-native-policy-v1"
PREPARATION_SOURCE_COMMIT = "ad277e785f6c09186ebabcf8d3bbd39b96d62f0f"
POLICY_SOURCE_COMMIT = "affe0e3bd29cce06beadd5a67b1e629f8ca77022"
HISTORY_BUDGET_DEFINITION = (
    "gist plus charged raw after subtracting the fixed current-input baseline"
)
WORKSPACE_BUDGET_DEFINITION = "incremental native evidence packet"
CURRENT_INPUT_BASELINE = (
    "source system messages plus latest user source message plus last visible "
    "source message, deduplicated; same tools and generation prompt"
)
_SUPPORTED_POLICY_MODES = frozenset({"protect", "recover_once", "persistent"})
_SUPPORTED_VIEW_MODES = frozenset({"static", "policy"})
_PACKING_FIELDS = (
    "ratios",
    "recent_tool_events",
    "max_chunk_tokens",
    "chunk_overlap",
    "max_chunks",
    "max_encoder_tokens",
    "max_system_tokens",
    "max_workspace_tokens",
    "max_target_tokens",
    "max_sequence_tokens",
)
_POLICY_FIELDS = (
    "mode",
    "history_budget_bytes",
    "workspace_budget_bytes",
    "lease_decisions",
    "max_retrieved_events",
    "kv_bytes_per_token",
    "source_commit",
    "history_budget_definition",
    "workspace_budget_definition",
    "current_input_baseline",
)


@dataclass(frozen=True)
class PreparedEventNative:
    memory: PackedMemory
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _PackingConfig:
    ratios: tuple[int, ...]
    recent_tool_events: int
    max_chunk_tokens: int
    chunk_overlap: int
    max_chunks: int
    max_encoder_tokens: int
    max_system_tokens: int
    max_workspace_tokens: int
    max_target_tokens: int
    max_sequence_tokens: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> _PackingConfig:
        if not isinstance(value, Mapping):
            raise TypeError("packing must be a mapping")
        missing = [name for name in _PACKING_FIELDS if name not in value]
        if missing:
            raise ValueError(f"packing lacks required fields: {missing!r}")
        raw_ratios = value["ratios"]
        if (
            not isinstance(raw_ratios, Sequence)
            or isinstance(raw_ratios, (str, bytes, bytearray))
            or not raw_ratios
        ):
            raise ValueError("packing.ratios must be a nonempty sequence")
        ratios = tuple(raw_ratios)
        if any(not _positive_int(item) for item in ratios):
            raise ValueError("packing.ratios must contain positive integers")
        if len(set(ratios)) != len(ratios):
            raise ValueError("packing.ratios must not contain duplicates")
        kwargs = {name: value[name] for name in _PACKING_FIELDS if name != "ratios"}
        for name in (
            "max_chunk_tokens",
            "max_chunks",
            "max_encoder_tokens",
            "max_system_tokens",
            "max_workspace_tokens",
            "max_target_tokens",
            "max_sequence_tokens",
        ):
            if not _positive_int(kwargs[name]):
                raise ValueError(f"packing.{name} must be a positive integer")
        if not _nonnegative_int(kwargs["recent_tool_events"]):
            raise ValueError("packing.recent_tool_events must be a nonnegative integer")
        if (
            not _nonnegative_int(kwargs["chunk_overlap"])
            or kwargs["chunk_overlap"] >= kwargs["max_chunk_tokens"]
        ):
            raise ValueError("Require packing.max_chunk_tokens > chunk_overlap >= 0")
        return cls(ratios=ratios, **kwargs)


@dataclass
class _SessionState:
    message_json: tuple[str, ...]
    tools_json: str
    decision_index: int
    policy_memory: ConversationMemory | None
    decisions: dict[str, tuple[tuple[Any, ...], PreparedEventNative]]


class EventNativeController:
    """Apply A selection and B token/byte budgets before event-native decode."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        packing: dict[str, Any],
        policy: dict[str, Any],
        view_mode: str,
    ) -> None:
        if not callable(getattr(tokenizer, "apply_chat_template", None)):
            raise TypeError("tokenizer must expose apply_chat_template")
        if view_mode not in _SUPPORTED_VIEW_MODES:
            raise ValueError(f"view_mode must be one of {sorted(_SUPPORTED_VIEW_MODES)!r}")
        self.tokenizer = tokenizer
        self.packing = _PackingConfig.from_mapping(packing)
        self.policy_config, self.kv_bytes_per_token = self._parse_policy(policy)
        self.view_mode = view_mode
        self._sessions: dict[str, _SessionState] = {}

    def prepare(
        self,
        payload: Mapping[str, Any],
        *,
        ratio: int,
        max_new_tokens: int,
    ) -> PreparedEventNative:
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        allowed = {"session_id", "decision_key", "messages", "tools"}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise PolicyInputError(
                f"Targets and unknown request fields are forbidden: {unknown!r}"
            )
        if not _positive_int(ratio) or ratio not in self.packing.ratios:
            raise ValueError(
                f"ratio must be one of checkpoint packing ratios {self.packing.ratios!r}"
            )
        if not _positive_int(max_new_tokens):
            raise ValueError("max_new_tokens must be a positive integer")
        if max_new_tokens > self.packing.max_target_tokens:
            raise PackingBudgetError(
                f"Generation needs up to {max_new_tokens} tokens; budget is "
                f"{self.packing.max_target_tokens}"
            )

        session_id = payload.get("session_id")
        decision_key = payload.get("decision_key")
        if not isinstance(session_id, str) or not session_id:
            raise PolicyInputError("An explicit nonempty session_id is required")
        if not isinstance(decision_key, str) or not decision_key:
            raise PolicyInputError("An explicit nonempty decision_key is required")
        raw_messages = payload.get("messages")
        if (
            not isinstance(raw_messages, Sequence)
            or isinstance(raw_messages, (str, bytes, bytearray))
            or not raw_messages
        ):
            raise PolicyInputError("messages must be a nonempty sequence")
        if any(not isinstance(message, Mapping) for message in raw_messages):
            raise PolicyInputError("messages must contain mappings")
        raw_tools = payload.get("tools", ())
        if raw_tools is None:
            raw_tools = ()
        if (
            not isinstance(raw_tools, Sequence)
            or isinstance(raw_tools, (str, bytes, bytearray))
            or any(not isinstance(tool, Mapping) for tool in raw_tools)
        ):
            raise PolicyInputError("tools must be a sequence of mappings")
        tools = tuple(_json_snapshot(tool) for tool in raw_tools)
        tools_json = _canonical_json(tools)
        store = EventStore.from_messages(session_id, raw_messages)
        message_json = tuple(message.json_text for message in store.messages)
        signature = (message_json, tools_json, ratio, max_new_tokens)

        state = self._sessions.get(session_id)
        if state is not None:
            if state.tools_json != tools_json:
                raise PolicyInputError(
                    "Tools changed within a session; use a new explicit session_id"
                )
            self._validate_monotone_prefix(state.message_json, message_json)
            cached = state.decisions.get(decision_key)
            if cached is not None:
                old_signature, prepared = cached
                if old_signature != signature:
                    raise PolicyInputError(
                        f"decision_key {decision_key!r} was reused with different input"
                    )
                return _copy_prepared(prepared)

        static_view = select_view(
            store,
            recent_tool_events=self.packing.recent_tool_events,
        )
        planning_ratio = min(self.packing.ratios)
        staged_policy_memory = self._staged_policy_memory(state, session_id)
        if self.view_mode == "policy":
            view, selection_metadata = self._select_policy_view(
                store,
                static_view,
                tools,
                decision_key,
                staged_policy_memory,
                planning_ratio,
            )
            decision_index = int(selection_metadata["decision_index"])
        else:
            view = static_view
            decision_index = (state.decision_index if state is not None else 0) + 1
            selection_metadata = self._static_selection_metadata(
                view,
                decision_index,
            )

        self._require_full_coverage(store, view)
        memory = pack_memory(
            store,
            view,
            self.tokenizer,
            tools=tools,
            max_chunk_tokens=self.packing.max_chunk_tokens,
            chunk_overlap=self.packing.chunk_overlap,
            max_chunks=self.packing.max_chunks,
            max_raw_tokens=(
                self.packing.max_system_tokens + self.packing.max_workspace_tokens
            ),
        )
        costs_by_ratio = self._validate_packed_limits(
            store,
            static_view,
            view,
            memory,
            tools,
            max_new_tokens,
        )
        metadata = self._metadata(
            store,
            decision_key,
            static_view,
            view,
            memory,
            ratio,
            max_new_tokens,
            planning_ratio,
            selection_metadata,
            costs_by_ratio,
        )
        prepared = PreparedEventNative(memory=memory, metadata=metadata)

        decisions = dict(state.decisions) if state is not None else {}
        decisions[decision_key] = (signature, _copy_prepared(prepared))
        self._sessions[session_id] = _SessionState(
            message_json=message_json,
            tools_json=tools_json,
            decision_index=decision_index,
            policy_memory=(
                staged_policy_memory
                if self.view_mode == "policy"
                else (state.policy_memory if state is not None else None)
            ),
            decisions=decisions,
        )
        return prepared

    @staticmethod
    def _parse_policy(
        policy: Mapping[str, Any],
    ) -> tuple[RuntimeConfig, int]:
        if not isinstance(policy, Mapping):
            raise TypeError("policy must be a mapping")
        missing = [name for name in _POLICY_FIELDS if name not in policy]
        if missing:
            raise ValueError(f"policy lacks required fields: {missing!r}")
        mode = policy["mode"]
        if mode not in _SUPPORTED_POLICY_MODES:
            raise ValueError(
                "event-native rendering supports only protect, recover_once, "
                "and persistent policy modes"
            )
        for name in (
            "history_budget_bytes",
            "workspace_budget_bytes",
            "lease_decisions",
            "max_retrieved_events",
        ):
            if not _nonnegative_int(policy[name]):
                raise ValueError(f"policy.{name} must be a nonnegative integer")
        if not _positive_int(policy["kv_bytes_per_token"]):
            raise ValueError("policy.kv_bytes_per_token must be a positive integer")
        expected_metadata = {
            "source_commit": POLICY_SOURCE_COMMIT,
            "history_budget_definition": HISTORY_BUDGET_DEFINITION,
            "workspace_budget_definition": WORKSPACE_BUDGET_DEFINITION,
            "current_input_baseline": CURRENT_INPUT_BASELINE,
        }
        for name, expected in expected_metadata.items():
            if policy[name] != expected:
                raise ValueError(
                    f"policy.{name}={policy[name]!r}; expected {expected!r}"
                )
        config = RuntimeConfig(
            mode=mode,
            history_budget_bytes=policy["history_budget_bytes"],
            workspace_budget_bytes=policy["workspace_budget_bytes"],
            lease_decisions=policy["lease_decisions"],
            max_retrieved_events=policy["max_retrieved_events"],
        )
        return config, policy["kv_bytes_per_token"]

    def _staged_policy_memory(
        self,
        state: _SessionState | None,
        session_id: str,
    ) -> ConversationMemory:
        if state is not None and state.policy_memory is not None:
            return copy.deepcopy(state.policy_memory)
        return ConversationMemory(session_id, self.policy_config)

    def _select_policy_view(
        self,
        store: EventStore,
        static_view: MemoryView,
        tools: Sequence[Mapping[str, Any]],
        decision_key: str,
        memory: ConversationMemory,
        planning_ratio: int,
    ) -> tuple[MemoryView, dict[str, Any]]:
        subcap = min(
            self.policy_config.history_budget_bytes,
            self.policy_config.workspace_budget_bytes,
        )
        measured_cache: dict[tuple[str, ...], tuple[int, int, int, int]] = {}

        def measured(event_ids: tuple[str, ...]) -> tuple[int, int, int, int]:
            wanted = set(event_ids)
            ordered = tuple(
                event.event_id for event in store.events if event.event_id in wanted
            )
            if ordered not in measured_cache:
                candidate = select_view(
                    store,
                    recent_tool_events=self.packing.recent_tool_events,
                    restored_event_ids=ordered,
                )
                evidence_tokens = _evidence_increment_tokens(
                    store,
                    static_view,
                    candidate,
                    self.tokenizer,
                    tools,
                )
                gist_tokens, raw_tokens = _history_components(
                    store,
                    candidate,
                    self.tokenizer,
                    ratio=planning_ratio,
                    max_chunk_tokens=self.packing.max_chunk_tokens,
                    chunk_overlap=self.packing.chunk_overlap,
                    tools=tools,
                )
                evidence_bytes = evidence_tokens * self.kv_bytes_per_token
                history_bytes = (gist_tokens + raw_tokens) * self.kv_bytes_per_token
                dual_cost = max(
                    evidence_bytes,
                    max(
                        0,
                        history_bytes
                        - self.policy_config.history_budget_bytes
                        + subcap,
                    ),
                )
                measured_cache[ordered] = (
                    dual_cost,
                    evidence_bytes,
                    history_bytes,
                    evidence_tokens,
                )
            return measured_cache[ordered]

        selection = memory.prepare(
            store,
            lambda event_ids: measured(event_ids)[0],
            set(static_view.raw_event_ids),
            decision_key,
        )
        view = select_view(
            store,
            recent_tool_events=self.packing.recent_tool_events,
            restored_event_ids=selection.selected_event_ids,
        )
        dual_cost, evidence_bytes, history_bytes, evidence_tokens = measured(
            selection.selected_event_ids
        )
        if evidence_bytes > self.policy_config.workspace_budget_bytes:
            raise BudgetExceeded(
                required_event_ids=selection.selected_event_ids,
                required_cost=evidence_bytes,
                budget=self.policy_config.workspace_budget_bytes,
            )
        if history_bytes > self.policy_config.history_budget_bytes:
            raise BudgetExceeded(
                required_event_ids=selection.selected_event_ids,
                required_cost=history_bytes,
                budget=self.policy_config.history_budget_bytes,
            )
        metadata = {
            **dict(selection.metadata),
            "planning_ratio": planning_ratio,
            "evidence_tokens": evidence_tokens,
            "evidence_bytes": evidence_bytes,
            "history_bytes": history_bytes,
            "dual_constraint_cost_bytes": dual_cost,
            "protected_event_ids": list(selection.protected_event_ids),
            "retrieved_event_ids": list(selection.retrieved_event_ids),
            "retained_event_ids": list(selection.retained_event_ids),
            "selected_event_ids": list(selection.selected_event_ids),
        }
        metadata["selection_reasons"] = _selection_reasons(metadata)
        metadata["lease_reasons"] = {
            "retained_active_lease_event_ids": metadata["retained_event_ids"],
            "expired_event_ids": list(metadata["expired_lease_event_ids"]),
            "revision_cancelled_event_ids": list(
                metadata["revision_cancelled_event_ids"]
            ),
            "acquired_from_direct_source_event_ids": (
                metadata["retrieved_event_ids"]
                if self.policy_config.mode == "persistent"
                else []
            ),
        }
        return view, metadata

    def _static_selection_metadata(
        self,
        view: MemoryView,
        decision_index: int,
    ) -> dict[str, Any]:
        return {
            "mode": self.policy_config.mode,
            "decision_index": decision_index,
            "budget_bytes": min(
                self.policy_config.history_budget_bytes,
                self.policy_config.workspace_budget_bytes,
            ),
            "selected_cost_bytes": 0,
            "mandatory_event_ids": [],
            "direct_source_candidate_ids": [],
            "retrieval_policy_version": "direct-source-v0",
            "skipped_for_budget": [],
            "expired_lease_event_ids": [],
            "revision_cancelled_event_ids": [],
            "raw_pending_event_ids": [],
            "protected_event_ids": list(view.raw_event_ids),
            "retrieved_event_ids": [],
            "retained_event_ids": [],
            "selected_event_ids": [],
            "selection_reasons": {
                event_id: ["static_base_view"] for event_id in view.raw_event_ids
            },
            "lease_reasons": {
                "retained_active_lease_event_ids": [],
                "expired_event_ids": [],
                "revision_cancelled_event_ids": [],
                "acquired_from_direct_source_event_ids": [],
            },
        }

    def _validate_packed_limits(
        self,
        store: EventStore,
        static_view: MemoryView,
        view: MemoryView,
        memory: PackedMemory,
        tools: Sequence[Mapping[str, Any]],
        max_new_tokens: int,
    ) -> dict[str, dict[str, int]]:
        if len(memory.system_input_ids) > self.packing.max_system_tokens:
            raise PackingBudgetError(
                f"System/tools need {len(memory.system_input_ids)} tokens; "
                f"budget is {self.packing.max_system_tokens}"
            )
        if len(memory.workspace_input_ids) > self.packing.max_workspace_tokens:
            raise PackingBudgetError(
                f"Workspace needs {len(memory.workspace_input_ids)} tokens; "
                f"budget is {self.packing.max_workspace_tokens}"
            )
        encoder_tokens = sum(len(chunk.token_ids) for chunk in memory.chunks)
        if encoder_tokens > self.packing.max_encoder_tokens:
            raise PackingBudgetError(
                f"Encoder chunks need {encoder_tokens} tokens; "
                f"budget is {self.packing.max_encoder_tokens}"
            )
        evidence_tokens = _evidence_increment_tokens(
            store,
            static_view,
            view,
            self.tokenizer,
            tools,
        )
        evidence_bytes = evidence_tokens * self.kv_bytes_per_token
        if evidence_bytes > self.policy_config.workspace_budget_bytes:
            raise PackingBudgetError(
                f"Incremental evidence needs {evidence_bytes} planned KV bytes; "
                f"budget is {self.policy_config.workspace_budget_bytes}"
            )

        result: dict[str, dict[str, int]] = {}
        for candidate_ratio in self.packing.ratios:
            costs = memory.costs(candidate_ratio)
            sequence_tokens = costs["resident_kv_tokens"] + max_new_tokens
            if sequence_tokens > self.packing.max_sequence_tokens:
                raise PackingBudgetError(
                    "Resident memory plus requested generation need "
                    f"{sequence_tokens} tokens at ratio {candidate_ratio}; "
                    f"budget is {self.packing.max_sequence_tokens}"
                )
            gist_tokens, raw_tokens = _history_components(
                store,
                view,
                self.tokenizer,
                ratio=candidate_ratio,
                max_chunk_tokens=self.packing.max_chunk_tokens,
                chunk_overlap=self.packing.chunk_overlap,
                tools=tools,
            )
            history_tokens = gist_tokens + raw_tokens
            history_bytes = history_tokens * self.kv_bytes_per_token
            if history_bytes > self.policy_config.history_budget_bytes:
                raise PackingBudgetError(
                    "Historical gist plus charged raw need "
                    f"{history_bytes} planned KV bytes at ratio "
                    f"{candidate_ratio}; budget is "
                    f"{self.policy_config.history_budget_bytes}"
                )
            result[str(candidate_ratio)] = {
                **costs,
                "max_new_tokens": max_new_tokens,
                "sequence_tokens": sequence_tokens,
                "history_gist_tokens": gist_tokens,
                "history_raw_tokens": raw_tokens,
                "history_total_tokens": history_tokens,
                "history_bytes": history_bytes,
                "evidence_tokens": evidence_tokens,
                "evidence_bytes": evidence_bytes,
                "history_budget_bytes": self.policy_config.history_budget_bytes,
                "workspace_budget_bytes": self.policy_config.workspace_budget_bytes,
            }
        return result

    def _metadata(
        self,
        store: EventStore,
        decision_key: str,
        static_view: MemoryView,
        view: MemoryView,
        memory: PackedMemory,
        ratio: int,
        max_new_tokens: int,
        planning_ratio: int,
        selection: Mapping[str, Any],
        costs_by_ratio: Mapping[str, Mapping[str, int]],
    ) -> dict[str, Any]:
        baseline_indices = _current_input_baseline_indices(store)
        return {
            "event_native_policy_version": EVENT_NATIVE_POLICY_VERSION,
            "preparation_source_commit": PREPARATION_SOURCE_COMMIT,
            "policy_source_commit": POLICY_SOURCE_COMMIT,
            "session_id": store.session_id,
            "decision_key": decision_key,
            "decision_index": int(selection["decision_index"]),
            "view_mode": self.view_mode,
            "mode": self.policy_config.mode,
            "planning_ratio": planning_ratio,
            "requested_ratio": ratio,
            "max_new_tokens": max_new_tokens,
            "accounting_scope": "planned KV geometry from checkpoint metadata; not measured HBM",
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "history_budget_definition": HISTORY_BUDGET_DEFINITION,
            "workspace_budget_definition": WORKSPACE_BUDGET_DEFINITION,
            "current_input_baseline": CURRENT_INPUT_BASELINE,
            "common_input_source_indices": list(baseline_indices),
            "raw_source_indices": list(memory.raw_source_indices),
            "full_source_indices": list(range(len(store.messages))),
            "full_source_coverage": True,
            "base_view": _view_dict(static_view),
            "view": _view_dict(view),
            "per_ratio": copy.deepcopy(dict(costs_by_ratio)),
            "selection": (
                copy.deepcopy(dict(selection)) if self.view_mode == "policy" else {}
            ),
            "recovery_stage": "pre_draft_direct_source_v0",
            "post_draft_exact_recovery_applied": False,
        }

    @staticmethod
    def _require_full_coverage(store: EventStore, view: MemoryView) -> None:
        view.validate(store)
        known = {event.event_id for event in store.events}
        gist = set(view.gist_event_ids)
        raw = set(view.raw_event_ids)
        if gist & raw or gist | raw != known:
            raise PolicyInputError("Memory view does not cover every visible event exactly once")

    @staticmethod
    def _validate_monotone_prefix(
        previous: tuple[str, ...],
        current: tuple[str, ...],
    ) -> None:
        if len(current) < len(previous) or current[: len(previous)] != previous:
            raise PolicyInputError(
                "Observable history was truncated or rewritten; use a new "
                "explicit session_id"
            )


def _history_components(
    store: EventStore,
    view: MemoryView,
    tokenizer: Any,
    *,
    ratio: int,
    max_chunk_tokens: int,
    chunk_overlap: int,
    tools: Sequence[Mapping[str, Any]],
) -> tuple[int, int]:
    """B/ad277e7 fixed-baseline history accounting."""

    full_messages = raw_workspace_messages(store, view)
    full_tokens = len(
        native_ids(tokenizer, full_messages, tools=tools, generation=True)
    )
    baseline_indices = _current_input_baseline_indices(store)
    baseline_messages = [
        visible_message(store.messages[index]) for index in baseline_indices
    ]
    baseline_tokens = (
        len(native_ids(tokenizer, baseline_messages, tools=tools, generation=True))
        if baseline_messages
        else 0
    )
    raw_tokens = max(0, full_tokens - baseline_tokens)
    gist_tokens = 0
    for event in store.events:
        if event.event_id not in view.gist_event_ids:
            continue
        chunks = encode_event_chunks(
            store,
            event.event_id,
            tokenizer,
            max_chunk_tokens=max_chunk_tokens,
            chunk_overlap=chunk_overlap,
        )
        gist_tokens += sum(math.ceil(len(chunk.token_ids) / ratio) for chunk in chunks)
    return gist_tokens, raw_tokens


def _evidence_increment_tokens(
    store: EventStore,
    base_view: MemoryView,
    candidate_view: MemoryView,
    tokenizer: Any,
    tools: Sequence[Mapping[str, Any]],
) -> int:
    base = native_ids(
        tokenizer,
        raw_workspace_messages(store, base_view),
        tools=tools,
        generation=True,
    )
    candidate = native_ids(
        tokenizer,
        raw_workspace_messages(store, candidate_view),
        tools=tools,
        generation=True,
    )
    return max(0, len(candidate) - len(base))


def _current_input_baseline_indices(store: EventStore) -> tuple[int, ...]:
    indices = {
        index
        for event in store.events
        if event.kind == "instruction"
        for index in event.source_indices
    }
    users = [event for event in store.events if event.kind == "user"]
    if users:
        indices.add(max(users[-1].source_indices))
    if store.messages:
        indices.add(len(store.messages) - 1)
    return tuple(sorted(indices))


def _selection_reasons(metadata: Mapping[str, Any]) -> dict[str, list[str]]:
    reasons: dict[str, list[str]] = {}

    def add(event_ids: Sequence[str], reason: str) -> None:
        for event_id in event_ids:
            reasons.setdefault(event_id, []).append(reason)

    mandatory = list(metadata["mandatory_event_ids"])
    protected = list(metadata["protected_event_ids"])
    add(mandatory, "mandatory")
    add([event_id for event_id in protected if event_id not in mandatory], "recent_complete_tool")
    add(list(metadata["retrieved_event_ids"]), "direct_source")
    add(list(metadata["retained_event_ids"]), "active_lease")
    return reasons


def _view_dict(view: MemoryView) -> dict[str, list[str]]:
    return {
        "gist_event_ids": list(view.gist_event_ids),
        "raw_event_ids": list(view.raw_event_ids),
        "evidence_event_ids": list(view.evidence_event_ids),
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_snapshot(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(_canonical_json(dict(value)))


def _copy_prepared(value: PreparedEventNative) -> PreparedEventNative:
    return PreparedEventNative(value.memory, copy.deepcopy(value.metadata))


def _positive_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _nonnegative_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


__all__ = ["EventNativeController", "PreparedEventNative"]
