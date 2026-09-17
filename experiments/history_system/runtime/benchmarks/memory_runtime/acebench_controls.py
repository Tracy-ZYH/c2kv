"""Finite event-native controllers for ACEBench receipt-backed history."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from history_memory.events import EventStore
from history_memory.packing import PackingBudgetError

from .acebench_source import (
    ACE_SOURCE_VERSION,
    ace_source_prefix_signature,
    build_ace_event_store,
)
from .event_native_controls import EventNativeOnePassController
from .event_native_exact_policy import EventNativeExactController
from .event_native_policy import _canonical_json, _json_snapshot
from .policy import PolicyInputError


ACEBENCH_VIEW_MODES = frozenset(
    {
        "capacity_protect",
        "capacity_exact_once",
        "capacity_exact_persistent",
        "full_exact_shared",
        "capacity_exact_no_gist",
        "full_original",
    }
)
_ACEBENCH_EXACT_VIEW_MODES = ACEBENCH_VIEW_MODES - {"full_original"}


def _positive_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _validate_ace_request(
    owner: Any,
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
    allowed = {
        "session_id",
        "decision_key",
        "messages",
        "tools",
        "c2kv_ace_source",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise PolicyInputError(
            f"Targets and privileged request fields are forbidden: {unknown!r}"
        )
    if not _positive_int(ratio) or ratio not in owner.packing.ratios:
        raise ValueError(
            f"ratio must be one of checkpoint packing ratios {owner.packing.ratios!r}"
        )
    if not _positive_int(max_new_tokens):
        raise ValueError("max_new_tokens must be a positive integer")
    if max_new_tokens > owner.packing.max_target_tokens:
        raise PackingBudgetError(
            f"Generation needs up to {max_new_tokens} tokens; budget is "
            f"{owner.packing.max_target_tokens}"
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
    try:
        store = build_ace_event_store(
            session_id, messages, payload.get("c2kv_ace_source")
        )
        message_json = ace_source_prefix_signature(
            messages, payload.get("c2kv_ace_source")
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise PolicyInputError(str(error)) from error
    return session_id, decision_key, store, tools, tools_json, message_json


def _add_ace_metadata(prepared: Any) -> Any:
    prepared.metadata["source_profile"] = ACE_SOURCE_VERSION
    prepared.metadata["acebench_events"] = [
        {
            "event_id": event.event_id,
            "kind": event.kind,
            "complete": event.complete,
            "source_indices": list(event.source_indices),
            "submitted_call_count": len(event.tool_call_ids),
        }
        for event in prepared._store.events
    ] if hasattr(prepared, "_store") else prepared.metadata.get("acebench_events", [])
    return prepared


class AceEventNativeExactController(EventNativeExactController):
    """Use ACE action batches instead of native OpenAI call/result binding."""

    def _validate_request(self, payload, ratio, max_new_tokens):
        return _validate_ace_request(self, payload, ratio, max_new_tokens)

    def prepare(self, payload, *, ratio, max_new_tokens):
        prepared = super().prepare(
            payload, ratio=ratio, max_new_tokens=max_new_tokens
        )
        return _add_ace_metadata(prepared)


class AceEventNativeOnePassController(EventNativeOnePassController):
    """ACE Full-original adapter with the same decision identity checks."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        packing: Mapping[str, Any],
        policy: Mapping[str, Any],
        view_mode: str,
        model_context: int | None = None,
    ) -> None:
        if view_mode != "full_original":
            raise ValueError("ACEBench one-pass view_mode must be full_original")
        super().__init__(
            tokenizer,
            packing=packing,
            policy=policy,
            view_mode=view_mode,
            model_context=model_context,
        )

    def _validate_request(self, payload, ratio, max_new_tokens):
        return _validate_ace_request(self, payload, ratio, max_new_tokens)

    def prepare(self, payload, *, ratio, max_new_tokens):
        prepared = super().prepare(
            payload, ratio=ratio, max_new_tokens=max_new_tokens
        )
        prepared.metadata["source_profile"] = ACE_SOURCE_VERSION
        # One-pass prepared values do not retain the EventStore, so rebuild the
        # small immutable CPU representation only for trace classification.
        store = build_ace_event_store(
            payload["session_id"],
            payload["messages"],
            payload["c2kv_ace_source"],
        )
        prepared.metadata["acebench_events"] = [
            {
                "event_id": event.event_id,
                "kind": event.kind,
                "complete": event.complete,
                "source_indices": list(event.source_indices),
                "submitted_call_count": len(event.tool_call_ids),
            }
            for event in store.events
        ]
        return prepared


def build_acebench_controller(
    tokenizer: Any,
    *,
    packing: Mapping[str, Any],
    policy: Mapping[str, Any],
    view_mode: str,
    model_context: int | None = None,
) -> Any:
    """Build an ACE controller; training-static has no textual-action contract."""

    if view_mode not in ACEBENCH_VIEW_MODES:
        raise ValueError(
            f"ACEBench view_mode must be one of {sorted(ACEBENCH_VIEW_MODES)!r}"
        )
    if view_mode in _ACEBENCH_EXACT_VIEW_MODES:
        return AceEventNativeExactController(
            tokenizer,
            packing=packing,
            policy=policy,
            mode=view_mode,
            model_context=model_context,
        )
    return AceEventNativeOnePassController(
        tokenizer,
        packing=packing,
        policy=policy,
        view_mode=view_mode,
        model_context=model_context,
    )


__all__ = [
    "ACEBENCH_VIEW_MODES",
    "AceEventNativeExactController",
    "AceEventNativeOnePassController",
    "build_acebench_controller",
]
