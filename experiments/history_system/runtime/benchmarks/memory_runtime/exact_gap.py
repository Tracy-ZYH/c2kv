"""Conservative exact-source retrieval for an unsubmitted native tool draft.

The detector reports missing visible provenance, never action correctness.
Only events in the caller's immutable observable prefix can be sources.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

from history_memory.events import EventStore

VERSION = "exact-source-gap-v2"
_IDENTIFIER_PUNCTUATION = frozenset("_./:@\\-")


@dataclass(frozen=True)
class Binding:
    argument_path: str
    value: str
    visible: bool
    source_event_ids: tuple[str, ...]

    def metadata(self) -> dict:
        # The original draft is captured separately when request capture is on.
        return {
            "argument_path": self.argument_path,
            "value_sha256": hashlib.sha256(self.value.encode("utf-8")).hexdigest(),
            "visible": self.visible,
            "source_event_ids": list(self.source_event_ids),
        }


@dataclass(frozen=True)
class GapDecision:
    status: str
    reason: str
    bindings: tuple[Binding, ...] = ()
    event_id: str | None = None

    def metadata(self) -> dict:
        return {
            "version": VERSION, "status": self.status, "reason": self.reason,
            "gap_type": "binding" if self.status == "gap" else None,
            "candidate_event_id": self.event_id,
            "bindings": [binding.metadata() for binding in self.bindings],
            "judges_action_correctness": False,
        }


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON argument key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Non-JSON numeric constant: {value}")


def _json(value: str):
    return json.loads(value, object_pairs_hook=_no_duplicates,
                      parse_constant=_reject_constant)


def _string_leaves(value: Any, path: str = ""):
    if isinstance(value, str):
        if value:
            yield path, value
    elif isinstance(value, dict):
        for key, child in value.items():
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            yield from _string_leaves(child, f"{path}/{escaped}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _string_leaves(child, f"{path}/{index}")


def _identifier_character(value: str) -> bool:
    return value.isalnum() or value in _IDENTIFIER_PUNCTUATION


def contains_exact_literal(text: str, value: str) -> bool:
    """Find a case-sensitive literal without matching part of an identifier."""
    if not value:
        return False
    start = 0
    while (index := text.find(value, start)) >= 0:
        end = index + len(value)
        left_ok = index == 0 or not _identifier_character(text[index - 1])
        right_ok = end == len(text) or not _identifier_character(text[end])
        # A sentence-final dot/colon delimits a literal; an extension such as
        # file.txt.bak or a path continuation still belongs to the identifier.
        if (not right_ok and text[end] in ".:"
                and (end + 1 == len(text) or not _identifier_character(text[end + 1]))):
            right_ok = True
        if left_ok and right_ok:
            return True
        start = index + 1
    return False


def _content_matches(content: Any, value: str) -> bool:
    if isinstance(content, str):
        try:
            decoded = _json(content)
        except (ValueError, TypeError):
            return contains_exact_literal(content, value)
        # Structured tool results often wrap prose in a JSON string value.
        # Apply the same literal boundaries as plain text, excluding JSON keys.
        return any(contains_exact_literal(literal, value)
                   for _, literal in _string_leaves(decoded))
    if isinstance(content, list):
        return any(
            isinstance(part, dict) and isinstance(part.get("text"), str)
            and _content_matches(part["text"], value) for part in content
        )
    return False


def message_has_literal(message: dict, value: str) -> bool:
    """Search source content and argument values, excluding metadata and keys."""
    if _content_matches(message.get("content"), value):
        return True
    for call in message.get("tool_calls") or []:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict):
            continue
        arguments = function.get("arguments")
        try:
            parsed = _json(arguments) if isinstance(arguments, str) else arguments
        except (ValueError, TypeError):
            continue
        if any(literal == value for _, literal in _string_leaves(parsed)):
            return True
    return False


def detect_exact_source_gap(
    store: EventStore, *, visible_source_indices: Iterable[int],
    source_cutoff: int, draft_tool_calls: Any,
) -> GapDecision:
    """Return no_op, abstain, or one source event for one evidence upgrade.

    Visibility is tracked per original message, so a partial tool group in the
    raw suffix is visible even if that complete event crosses the boundary.
    Candidate sources must be complete and wholly before the raw suffix.
    """
    if not isinstance(store, EventStore):
        raise TypeError("store must be an EventStore")
    if type(source_cutoff) is not int or not 0 <= source_cutoff <= len(store.messages):
        raise ValueError("source_cutoff must index the observable prefix")
    visible = frozenset(visible_source_indices)
    if any(type(index) is not int or not 0 <= index < len(store.messages) for index in visible):
        raise ValueError("Visible source index is outside the observable prefix")
    if draft_tool_calls is None or draft_tool_calls == []:
        return GapDecision("no_op", "no_native_tool_calls")
    if not isinstance(draft_tool_calls, list):
        return GapDecision("abstain", "malformed_arguments")

    references = []
    try:
        for index, call in enumerate(draft_tool_calls):
            arguments = call["function"]["arguments"]
            parsed = _json(arguments) if isinstance(arguments, str) else arguments
            if not isinstance(parsed, dict):
                raise ValueError("Native function arguments must be a JSON object")
            references.extend(_string_leaves(parsed, f"/tool_calls/{index}/function/arguments"))
    except (KeyError, ValueError, TypeError):
        return GapDecision("abstain", "malformed_arguments")
    if not references:
        return GapDecision("no_op", "no_string_bindings")

    native = [message.to_dict() for message in store.messages]
    bindings = []
    for argument_path, value in references:
        is_visible = any(message_has_literal(native[index], value) for index in visible)
        sources = () if is_visible else tuple(
            event.event_id for event in store.events
            if event.complete and max(event.source_indices) < source_cutoff
            and any(message_has_literal(native[index], value) for index in event.source_indices)
        )
        bindings.append(Binding(argument_path, value, is_visible, sources))
    missing = [binding for binding in bindings if not binding.visible]
    frozen = tuple(bindings)
    if not missing:
        return GapDecision("no_op", "all_bindings_visible", frozen)
    if any(not binding.source_event_ids for binding in missing):
        return GapDecision("abstain", "missing_source", frozen)
    if any(len(binding.source_event_ids) != 1 for binding in missing):
        return GapDecision("abstain", "ambiguous_sources", frozen)
    sources = {binding.source_event_ids[0] for binding in missing}
    if len(sources) != 1:
        return GapDecision("abstain", "multiple_source_events", frozen)
    return GapDecision("gap", "missing_unique_complete_source", frozen, next(iter(sources)))
