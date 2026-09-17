"""Finite event-native controller routing with explicit baseline identities."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from history_memory.events import EventStore
from history_memory.packing import PackedMemory, PackingBudgetError

from .always_compress import ALWAYS_COMPRESSION_POLICY
from .event_native_always import (
    NATIVE_ALWAYS_CANONICAL_MODES,
    NATIVE_ALWAYS_IMPLEMENTATION_PROFILE,
    NATIVE_ALWAYS_ROUTE_MODES,
    NATIVE_S0_MODE,
)
from .event_native_exact_policy import EventNativeExactController
from .event_native_policy import (
    EventNativeController,
    PolicyInputError,
    _PackingConfig,
    _canonical_json,
    _json_snapshot,
)
from .event_native_raw import build_raw_control


EVENT_NATIVE_CONTROL_VERSION = "a-event-native-finite-control-v1"

# Keep the frozen finite-route catalog stable.  New pre-B routes are an
# explicit opt-in namespace so callers that snapshot this set do not silently
# broaden their method contract.
FINITE_VIEW_MODES = frozenset(
    {
        "capacity_protect",
        "capacity_exact_once",
        "capacity_exact_persistent",
        "full_exact_shared",
        "capacity_exact_no_gist",
        "full_original",
        "static",
    }
)
ALWAYS_COMPRESS_VIEW_MODES = NATIVE_ALWAYS_ROUTE_MODES
ALL_VIEW_MODES = FINITE_VIEW_MODES | ALWAYS_COMPRESS_VIEW_MODES
ONE_PASS_VIEW_MODES = frozenset(
    {
        "capacity_protect",
        "full_original",
        "static",
    }
)

_EXACT_VIEW_MODES = ALL_VIEW_MODES - {"full_original", "static"}
_ROUTES = {
    "full_original": ("Full-original", False, 1),
    "full_exact_shared": ("Full-shared", True, 2),
    "capacity_protect": ("C2KV-protect", False, 1),
    "capacity_exact_once": ("C2KV-recover-once", True, 2),
    "capacity_exact_persistent": ("C2KV-persistent", True, 2),
    "capacity_exact_no_gist": ("NoGist-budgeted", True, 2),
    "static": ("event-native-training-static", False, 1),
    "ac_gist_static": ("AC-gist-static", False, 1),
    "ac_protect": ("AC-protect", False, 1),
    "ac_exact_once": ("AC-exact-once", True, 2),
    "ac_exact_persistent": ("AC-exact-persistent", True, 2),
    "ac_full_shared": ("AC-Full-shared-control", True, 2),
    "raw_exact_shared": ("Raw-exact-shared", True, 2),
    NATIVE_S0_MODE: ("S0-native-event-lexical-raw-reserve-failed-operation", False, 1),
}


def describe_event_native_route(
    view_mode: str,
    *,
    compression_policy: str | None = None,
    history_view_protocol: str = "fixed-budget-main",
) -> dict[str, Any]:
    """Return the stable engineering identity of one finite runtime route."""

    try:
        baseline_identity, recovery_enabled, max_generations = _ROUTES[view_mode]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"view_mode must be one of {sorted(ALL_VIEW_MODES)!r}"
        ) from error
    always_compress = view_mode in NATIVE_ALWAYS_ROUTE_MODES
    if always_compress and compression_policy != ALWAYS_COMPRESSION_POLICY:
        raise ValueError(
            "Always-compress routes require compression_policy='always-compress-v1'"
        )
    if not always_compress and compression_policy is not None:
        raise ValueError("compression_policy requires a new always-compress route")
    if history_view_protocol != "fixed-budget-main":
        raise ValueError(
            "event-native P0 supports history_view_protocol='fixed-budget-main'"
        )
    result = {
        "view_mode": view_mode,
        "baseline_identity": baseline_identity,
        "recovery_enabled": recovery_enabled,
        "max_generations_per_decision": max_generations,
        # The historical 1088 legacy route remains in the existing proxy.  In
        # particular, B's training-static view must never be relabeled legacy.
        "legacy_1088_equivalent": False,
    }
    if always_compress:
        result.update(
            compression_policy=ALWAYS_COMPRESSION_POLICY,
            history_view_protocol=history_view_protocol,
            implementation_profile=NATIVE_ALWAYS_IMPLEMENTATION_PROFILE,
            canonical_mode=NATIVE_ALWAYS_CANONICAL_MODES[view_mode],
            training_static=False,
        )
    return result


def build_event_native_controller(
    tokenizer: Any,
    *,
    packing: Mapping[str, Any],
    policy: Mapping[str, Any],
    view_mode: str,
    model_context: int | None = None,
    compression_policy: str | None = None,
    history_view_protocol: str = "fixed-budget-main",
    s0_config: Mapping[str, Any] | None = None,
) -> Any:
    """Build a finite controller without importing the CLI/server module."""

    if s0_config is not None and "gp_experiments" in s0_config:
        from .recovery.experiment import GPRecoveryController

        if view_mode != NATIVE_S0_MODE:
            raise ValueError("G--P switches require the current native S0 route")
        config = dict(s0_config)
        switches = config.pop("gp_experiments")
        detector = config.pop("post_draft_recovery", None)
        if detector is None:
            raise ValueError("G--P requires the current post_draft_recovery config")
        controller = build_event_native_controller(
            tokenizer, view_mode=view_mode, packing=packing, policy=policy,
            model_context=model_context, compression_policy=compression_policy,
            history_view_protocol=history_view_protocol, s0_config=config)
        return GPRecoveryController(controller, detector, switches)

    if s0_config is not None and "post_draft_recovery" in s0_config:
        from .event_native_recovery import wrap_with_event_native_recovery

        recovery_config = s0_config["post_draft_recovery"]
        config = dict(s0_config)
        del config["post_draft_recovery"]
        controller = build_event_native_controller(
            tokenizer,
            view_mode=view_mode,
            packing=packing,
            policy=policy,
            model_context=model_context,
            compression_policy=compression_policy,
            history_view_protocol=history_view_protocol,
            s0_config=config,
        )
        return wrap_with_event_native_recovery(controller, recovery_config)

    if s0_config is not None and "raw_warmup_policy" in s0_config:
        from .raw_warmup import RAW_WARMUP_POLICY, wrap_with_raw_warmup
        if s0_config["raw_warmup_policy"] != RAW_WARMUP_POLICY:
            raise ValueError("Unknown raw warmup policy")
        config = dict(s0_config)
        del config["raw_warmup_policy"]
        controller = build_event_native_controller(
            tokenizer, view_mode=view_mode, packing=packing, policy=policy,
            model_context=model_context, compression_policy=compression_policy,
            history_view_protocol=history_view_protocol, s0_config=config)
        return wrap_with_raw_warmup(controller)

    if s0_config is not None and "stalled_operation_policy" in s0_config:
        from .stalled_operation import (
            STALLED_OPERATION_POLICY,
            enable_stalled_operation,
        )
        if s0_config["stalled_operation_policy"] != STALLED_OPERATION_POLICY:
            raise ValueError("Unknown stalled-operation policy")
        config = dict(s0_config)
        del config["stalled_operation_policy"]
        controller = build_event_native_controller(
            tokenizer,
            view_mode=view_mode,
            packing=packing,
            policy=policy,
            model_context=model_context,
            compression_policy=compression_policy,
            history_view_protocol=history_view_protocol,
            s0_config=config,
        )
        return enable_stalled_operation(controller)

    if s0_config is not None and "compact_first_failure_policy" in s0_config:
        from .compact_first_failure import (
            COMPACT_FIRST_FAILURE_POLICY,
            enable_compact_first_failure,
        )
        if (
            s0_config["compact_first_failure_policy"]
            != COMPACT_FIRST_FAILURE_POLICY
        ):
            raise ValueError("Unknown compact first-failure policy")
        config = dict(s0_config)
        del config["compact_first_failure_policy"]
        controller = build_event_native_controller(
            tokenizer,
            view_mode=view_mode,
            packing=packing,
            policy=policy,
            model_context=model_context,
            compression_policy=compression_policy,
            history_view_protocol=history_view_protocol,
            s0_config=config,
        )
        return enable_compact_first_failure(controller)

    if view_mode not in ALL_VIEW_MODES:
        raise ValueError(
            f"view_mode must be one of {sorted(ALL_VIEW_MODES)!r}"
        )
    describe_event_native_route(
        view_mode,
        compression_policy=compression_policy,
        history_view_protocol=history_view_protocol,
    )
    if view_mode == NATIVE_S0_MODE:
        from .event_native_s0_policy import EventNativeS0Controller
        if s0_config is None:
            raise ValueError('Native S0 requires an explicit controller config')
        observed_slot_key = "observed_entity_slot_policy"
        if observed_slot_key in s0_config:
            from .observed_entity_slot import (
                OBSERVED_ENTITY_SLOT_POLICY,
                ObservedEntitySlotS0Controller,
            )
            from .observed_entity_slot_revision import (
                MISSING_REQUIRED_REFERENCE_POLICY,
                RevisionObservedEntitySlotS0Controller,
            )
            from .same_event_reference import (
                SAME_EVENT_REFERENCE_POLICY,
                SameEventReferenceS0Controller,
            )
            from .same_event_bridge_only import (
                SAME_EVENT_BRIDGE_ONLY_POLICY,
                SameEventBridgeOnlyS0Controller,
            )
            from .result_key_bridge import (
                RESULT_KEY_BRIDGE_POLICY,
                ResultKeyBridgeS0Controller,
            )
            candidate_policy = s0_config[observed_slot_key]
            base_s0_config = dict(s0_config)
            del base_s0_config[observed_slot_key]
            if candidate_policy == OBSERVED_ENTITY_SLOT_POLICY:
                return ObservedEntitySlotS0Controller(
                    tokenizer,
                    packing=packing,
                    policy=policy,
                    model_context=model_context,
                    s0_config=base_s0_config,
                    observed_entity_slot_policy=candidate_policy,
                )
            if candidate_policy == MISSING_REQUIRED_REFERENCE_POLICY:
                return RevisionObservedEntitySlotS0Controller(
                    tokenizer,
                    packing=packing,
                    policy=policy,
                    model_context=model_context,
                    s0_config=base_s0_config,
                    observed_entity_slot_revision_policy=candidate_policy,
                )
            if candidate_policy == SAME_EVENT_REFERENCE_POLICY:
                return SameEventReferenceS0Controller(
                    tokenizer,
                    packing=packing,
                    policy=policy,
                    model_context=model_context,
                    s0_config=base_s0_config,
                    same_event_reference_policy=candidate_policy,
                )
            if candidate_policy == SAME_EVENT_BRIDGE_ONLY_POLICY:
                return SameEventBridgeOnlyS0Controller(
                    tokenizer,
                    packing=packing,
                    policy=policy,
                    model_context=model_context,
                    s0_config=base_s0_config,
                    same_event_bridge_only_policy=candidate_policy,
                )
            if candidate_policy == RESULT_KEY_BRIDGE_POLICY:
                return ResultKeyBridgeS0Controller(
                    tokenizer,
                    packing=packing,
                    policy=policy,
                    model_context=model_context,
                    s0_config=base_s0_config,
                    result_key_bridge_policy=candidate_policy,
                )
            raise ValueError("Unknown observed entity slot policy")
        return EventNativeS0Controller(tokenizer, packing=packing, policy=policy,
            model_context=model_context, s0_config=s0_config)
    if view_mode in _EXACT_VIEW_MODES:
        return EventNativeExactController(
            tokenizer,
            packing=packing,
            policy=policy,
            mode=view_mode,
            model_context=model_context,
            compression_policy=compression_policy,
            history_view_protocol=history_view_protocol,
        )
    return EventNativeOnePassController(
        tokenizer,
        packing=packing,
        policy=policy,
        view_mode=view_mode,
        model_context=model_context,
    )


@dataclass
class PreparedEventNativeOnePass:
    """A prepared non-recovery decision owned by one controller."""

    memory: PackedMemory
    metadata: dict[str, Any]
    _owner: object = field(repr=False, compare=False)
    _session_id: str = field(repr=False, compare=False)
    _decision_key: str = field(repr=False, compare=False)
    _checked_signature: str | None = field(default=None, repr=False, compare=False)
    _checked_result: dict[str, Any] | None = field(
        default=None, repr=False, compare=False
    )


@dataclass
class _OnePassSessionState:
    message_json: tuple[str, ...]
    tools_json: str
    decision_index: int
    decisions: dict[str, tuple[tuple[Any, ...], PreparedEventNativeOnePass]]
    active_decision_key: str


class EventNativeOnePassController:
    """Adapt Full-original and training-static to the finite runner contract."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        packing: Mapping[str, Any],
        policy: Mapping[str, Any],
        view_mode: str,
        model_context: int | None = None,
    ) -> None:
        if not callable(getattr(tokenizer, "apply_chat_template", None)):
            raise TypeError("tokenizer must expose apply_chat_template")
        if view_mode not in {"full_original", "static"}:
            raise ValueError("one-pass view_mode must be full_original or static")
        if model_context is not None and not _positive_int(model_context):
            raise ValueError("model_context must be a positive integer or None")
        self.tokenizer = tokenizer
        self.packing = _PackingConfig.from_mapping(packing)
        # Parse once so controls reject a malformed checkpoint policy even when
        # Full-original does not consume its history/workspace byte budgets.
        EventNativeController._parse_policy(policy)
        self.policy = _json_snapshot(policy)
        self.view_mode = view_mode
        self.model_context = model_context
        self._owner = object()
        self._sessions: dict[str, _OnePassSessionState] = {}

    def prepare(
        self,
        payload: Mapping[str, Any],
        *,
        ratio: int,
        max_new_tokens: int,
    ) -> PreparedEventNativeOnePass:
        session_id, decision_key, store, tools, tools_json, message_json = (
            self._validate_request(payload, ratio, max_new_tokens)
        )
        signature = (message_json, tools_json, ratio, max_new_tokens)
        state = self._sessions.get(session_id)
        if state is not None:
            if state.tools_json != tools_json:
                raise PolicyInputError(
                    "Tools changed within a session; use a new explicit session_id"
                )
            EventNativeController._validate_monotone_prefix(
                state.message_json, message_json
            )
            cached = state.decisions.get(decision_key)
            if cached is not None:
                old_signature, prepared = cached
                if old_signature != signature:
                    raise PolicyInputError(
                        f"decision_key {decision_key!r} was reused with different input"
                    )
                return prepared

        decision_index = (state.decision_index if state is not None else 0) + 1
        packing = self._effective_packing_mapping()
        if self.view_mode == "static":
            # Static selection has no evolving policy state.  A fresh instance
            # preserves its established packing contract while the adapter owns
            # the multi-decision clock and identity checks.
            inner = EventNativeController(
                self.tokenizer,
                packing=packing,
                policy=self.policy,
                view_mode="static",
            )
            base = inner.prepare(
                payload, ratio=ratio, max_new_tokens=max_new_tokens
            )
        else:
            base = build_raw_control(
                store,
                self.tokenizer,
                packing=packing,
                policy=self.policy,
                mode="full_original",
                max_new_tokens=max_new_tokens,
                tools=tools,
            )
        self._require_model_context(base.memory, ratio, max_new_tokens)

        metadata = copy.deepcopy(base.metadata)
        metadata.update(
            {
                "event_native_control_version": EVENT_NATIVE_CONTROL_VERSION,
                "session_id": session_id,
                "decision_key": decision_key,
                "decision_index": decision_index,
                "view_mode": self.view_mode,
                "route": describe_event_native_route(self.view_mode),
                "configured_max_sequence_tokens": self.packing.max_sequence_tokens,
                "effective_max_sequence_tokens": packing["max_sequence_tokens"],
                "model_context": self.model_context,
                "pre_draft_retrieval": False,
                "recovery_stage": "one_pass_recovery_disabled",
                "post_draft_exact_recovery_applied": False,
            }
        )
        prepared = PreparedEventNativeOnePass(
            memory=base.memory,
            metadata=metadata,
            _owner=self._owner,
            _session_id=session_id,
            _decision_key=decision_key,
        )
        decisions = dict(state.decisions) if state is not None else {}
        decisions[decision_key] = (signature, prepared)
        self._sessions[session_id] = _OnePassSessionState(
            message_json=message_json,
            tools_json=tools_json,
            decision_index=decision_index,
            decisions=decisions,
            active_decision_key=decision_key,
        )
        return prepared

    def reconsider(
        self,
        prepared: PreparedEventNativeOnePass,
        draft_tool_calls: Any,
        *,
        draft_text: str,
        parse_error: str | None = None,
    ) -> dict[str, Any]:
        if (
            not isinstance(prepared, PreparedEventNativeOnePass)
            or prepared._owner is not self._owner
        ):
            raise PolicyInputError(
                "Prepared decision belongs to another one-pass controller"
            )
        if not isinstance(draft_text, str):
            raise TypeError("draft_text must be a string")
        if parse_error is not None and not isinstance(parse_error, str):
            raise TypeError("parse_error must be a string or None")
        signature = _canonical_json(
            {
                "draft_text": draft_text,
                "draft_tool_calls": draft_tool_calls,
                "parse_error": parse_error,
            }
        )
        if prepared._checked_result is not None:
            if signature != prepared._checked_signature:
                raise PolicyInputError(
                    "A prepared decision cannot inspect a second different draft"
                )
            return _copy_reconsideration(prepared._checked_result)

        state = self._sessions.get(prepared._session_id)
        if state is None or state.active_decision_key != prepared._decision_key:
            raise PolicyInputError("Prepared decision is stale for the active prefix")

        decision = {
            "version": EVENT_NATIVE_CONTROL_VERSION,
            "status": "no_op",
            "reason": "recovery_disabled",
            "gap_type": None,
            "candidate_event_id": None,
            "bindings": [],
            "judges_action_correctness": False,
            "decision_index": prepared.metadata["decision_index"],
            "upgrade_count": 0,
            "regeneration_allowed": False,
            "upgraded_event_id": None,
        }
        metadata = copy.deepcopy(prepared.metadata)
        metadata["exact_recovery"] = copy.deepcopy(decision)
        metadata["post_draft_exact_recovery_applied"] = False
        result = {
            "regenerate": False,
            "memory": prepared.memory,
            "metadata": metadata,
            "decision": copy.deepcopy(decision),
        }
        prepared._checked_signature = signature
        prepared._checked_result = _copy_reconsideration(result)
        return result

    def _validate_request(
        self,
        payload: Mapping[str, Any],
        ratio: int,
        max_new_tokens: int,
    ) -> tuple[
        str,
        str,
        EventStore,
        tuple[dict[str, Any], ...],
        str,
        tuple[str, ...],
    ]:
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        allowed = {"session_id", "decision_key", "messages", "tools"}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise PolicyInputError(
                f"Targets and privileged request fields are forbidden: {unknown!r}"
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
        messages = payload.get("messages")
        if (
            not isinstance(messages, Sequence)
            or isinstance(messages, (str, bytes, bytearray))
            or not messages
            or any(not isinstance(message, Mapping) for message in messages)
        ):
            raise PolicyInputError("messages must be a nonempty sequence of mappings")
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
        store = EventStore.from_messages(session_id, messages)
        message_json = tuple(message.json_text for message in store.messages)
        return session_id, decision_key, store, tools, tools_json, message_json

    def _effective_packing_mapping(self) -> dict[str, Any]:
        value = {
            "ratios": list(self.packing.ratios),
            "recent_tool_events": self.packing.recent_tool_events,
            "max_chunk_tokens": self.packing.max_chunk_tokens,
            "chunk_overlap": self.packing.chunk_overlap,
            "max_chunks": self.packing.max_chunks,
            "max_encoder_tokens": self.packing.max_encoder_tokens,
            "max_system_tokens": self.packing.max_system_tokens,
            "max_workspace_tokens": self.packing.max_workspace_tokens,
            "max_target_tokens": self.packing.max_target_tokens,
            "max_sequence_tokens": self.packing.max_sequence_tokens,
        }
        if self.model_context is not None:
            value["max_sequence_tokens"] = min(
                value["max_sequence_tokens"], self.model_context
            )
        return value

    def _require_model_context(
        self,
        memory: PackedMemory,
        ratio: int,
        max_new_tokens: int,
    ) -> None:
        if self.model_context is None:
            return
        logical_end = (
            memory.workspace_position_start
            + len(memory.workspace_input_ids)
            + max_new_tokens
        )
        physical_end = memory.costs(ratio)["resident_kv_tokens"] + max_new_tokens
        required = max(logical_end, physical_end)
        if required > self.model_context:
            raise PackingBudgetError(
                f"Prepared request needs {required} model positions; model context is "
                f"{self.model_context}"
            )


def _copy_reconsideration(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "regenerate": bool(value["regenerate"]),
        "memory": value["memory"],
        "metadata": copy.deepcopy(value["metadata"]),
        "decision": copy.deepcopy(value["decision"]),
    }


def _positive_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


__all__ = [
    "ALL_VIEW_MODES",
    "ALWAYS_COMPRESS_VIEW_MODES",
    "EVENT_NATIVE_CONTROL_VERSION",
    "FINITE_VIEW_MODES",
    "ONE_PASS_VIEW_MODES",
    "EventNativeOnePassController",
    "PreparedEventNativeOnePass",
    "build_event_native_controller",
    "describe_event_native_route",
]
