"""Stable encoder-unit boundaries for event-native history memory.

The source ``EventStore`` remains the provenance authority. This module only
groups already complete, observable events and finds exact record spans inside
their text fields; it never rewrites the source snapshots.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable

from .events import EventStore


ENCODING_SCOPES = frozenset(
    {
        "current",
        "event",
        "record",
        "adjacent_pair",
        "record_bound",
        "record_bound_structural",
    }
)


@dataclass(frozen=True)
class ExactTextSpan:
    """One exact source substring and its JSON path when applicable."""

    text: str
    char_start: int
    char_end: int
    field_path: tuple[str | int, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.text, str)
            or isinstance(self.char_start, bool)
            or not isinstance(self.char_start, int)
            or isinstance(self.char_end, bool)
            or not isinstance(self.char_end, int)
            or self.char_start < 0
            or self.char_end < self.char_start
        ):
            raise ValueError("ExactTextSpan requires a valid source character range")


@dataclass(frozen=True)
class EventRecordSpan:
    """One exact record bound to a scalar container in a source message."""

    source_index: int
    container_path: tuple[str | int, ...]
    span: ExactTextSpan


@dataclass(frozen=True)
class EncodingScopePlan:
    """Atomic event groups plus events that cannot yet enter this scope."""

    scope: str
    event_groups: tuple[tuple[str, ...], ...]
    pending_event_ids: tuple[str, ...]

    @property
    def compressible_event_ids(self) -> tuple[str, ...]:
        return tuple(event_id for group in self.event_groups for event_id in group)


def validate_encoding_scope(value: str) -> str:
    if value not in ENCODING_SCOPES:
        raise ValueError(f"encoding_scope must be one of {sorted(ENCODING_SCOPES)!r}")
    return value


def plan_encoding_scope(
    store: EventStore,
    event_ids: Iterable[str],
    scope: str,
) -> EncodingScopePlan:
    """Plan stable units in source order without crossing an incomplete event."""

    scope = validate_encoding_scope(scope)
    selected = set(event_ids)
    ordered = []
    for event in store.events:
        if event.event_id not in selected:
            continue
        if not event.complete or event.kind == "instruction":
            raise ValueError("Encoding scopes accept only complete non-instruction events")
        ordered.append(event.event_id)
    unknown = selected - set(ordered)
    if unknown:
        raise ValueError(
            f"Encoding-scope event is not in the visible prefix: {sorted(unknown)!r}"
        )

    if scope == "adjacent_pair":
        pair_count = len(ordered) // 2
        groups = tuple(
            (ordered[index * 2], ordered[index * 2 + 1])
            for index in range(pair_count)
        )
        return EncodingScopePlan(scope, groups, tuple(ordered[pair_count * 2 :]))

    if scope == "record":
        groups = []
        pending = []
        for event_id in ordered:
            if event_record_spans(store, event_id):
                groups.append((event_id,))
            else:
                pending.append(event_id)
        return EncodingScopePlan(scope, tuple(groups), tuple(pending))

    return EncodingScopePlan(scope, tuple((event_id,) for event_id in ordered), ())


def structural_tool_result_record_spans(
    store: EventStore, event_id: str
) -> tuple[EventRecordSpan, ...]:
    """Return records only for an unambiguous multi-record tool-result event.

    The structural G scope is intentionally conservative.  Every message in
    the event must be either a content-free assistant producer or a matched
    tool result, and every result must be a JSON value with at least two
    object records in one array.  Anything else falls back to ``current``
    event encoding rather than partially representing an event.
    """

    event = store.event(event_id)
    if not event.complete or event.kind != "tool_event":
        return ()

    calls_by_id: dict[str, tuple[int, dict]] = {}
    results = []
    for source_index in event.source_indices:
        message = store.messages[source_index].to_dict()
        role = message.get("role")
        if role == "assistant":
            content = message.get("content")
            if content is not None and content != "":
                return ()
            calls = message.get("tool_calls")
            if not isinstance(calls, list) or not calls:
                return ()
            for call in calls:
                if not isinstance(call, dict):
                    return ()
                call_id = call.get("id")
                if not isinstance(call_id, str) or call_id in calls_by_id:
                    return ()
                calls_by_id[call_id] = (source_index, call)
            continue
        if role != "tool":
            return ()
        results.append((source_index, message))

    if not calls_by_id or len(results) != len(calls_by_id):
        return ()

    records = []
    seen_call_ids = set()
    for source_index, message in results:
        call_id = message.get("tool_call_id")
        content = message.get("content")
        if (
            not isinstance(call_id, str)
            or call_id not in calls_by_id
            or call_id in seen_call_ids
            or not isinstance(content, str)
            or not content
        ):
            return ()
        spans = _multi_json_record_spans(content)
        if not spans:
            return ()
        seen_call_ids.add(call_id)
        records.extend(
            EventRecordSpan(source_index, ("content",), span) for span in spans
        )
    if seen_call_ids != set(calls_by_id):
        return ()
    return tuple(records)


def event_record_spans(
    store: EventStore, event_id: str
) -> tuple[EventRecordSpan, ...]:
    """Return exact records from content and serialized tool-call arguments."""

    event = store.event(event_id)
    if not event.complete or event.kind == "instruction":
        raise ValueError("Record encoding accepts only complete non-instruction events")
    records = []
    for source_index in event.source_indices:
        message = store.messages[source_index].to_dict()
        content = message.get("content")
        if isinstance(content, str) and content:
            spans = extract_record_spans(content)
            if not spans:
                return ()
            records.extend(
                EventRecordSpan(source_index, ("content",), span)
                for span in spans
            )
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call_index, call in enumerate(calls):
            function = call.get("function") if isinstance(call, dict) else None
            arguments = function.get("arguments") if isinstance(function, dict) else None
            if not isinstance(arguments, str) or not arguments:
                continue
            spans = extract_record_spans(arguments)
            if not spans:
                return ()
            records.extend(
                EventRecordSpan(
                    source_index,
                    ("tool_calls", call_index, "function", "arguments"),
                    span,
                )
                for span in spans
            )
    return tuple(records)


def extract_record_spans(text: str) -> tuple[ExactTextSpan, ...]:
    """Extract exact top-level JSON records or blank-line paragraphs.

    Arrays split only when every element is an object. For an object, every
    immediate list field whose nonempty elements are all objects contributes
    those elements. Otherwise the complete JSON value is one record. Text that
    looks like malformed or mixed JSON is rejected rather than relabelled as
    prose.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    start = _skip_ws(text, 0)
    if start == len(text):
        return ()
    decoder = json.JSONDecoder()
    try:
        value, end = decoder.raw_decode(text, start)
    except json.JSONDecodeError:
        if "{" in text or "[" in text:
            return ()
        return _paragraph_spans(text)
    if _skip_ws(text, end) != len(text):
        if isinstance(value, (dict, list)) or "{" in text or "[" in text:
            return ()
        return _paragraph_spans(text)

    if isinstance(value, list):
        if value and all(isinstance(item, dict) for item in value):
            return tuple(
                ExactTextSpan(text[a:b], a, b, (index,))
                for index, (a, b) in enumerate(_array_item_ranges(text, start, end))
            )
        return (ExactTextSpan(text[start:end], start, end, ()),)

    if isinstance(value, dict):
        child_records = []
        for key, value_start, value_end, child in _object_value_ranges(
            text, start, end
        ):
            if not isinstance(child, list) or not child or not all(
                isinstance(item, dict) for item in child
            ):
                continue
            child_records.extend(
                ExactTextSpan(text[a:b], a, b, (key, index))
                for index, (a, b) in enumerate(
                    _array_item_ranges(text, value_start, value_end)
                )
            )
        if child_records:
            return tuple(child_records)
        return (ExactTextSpan(text[start:end], start, end, ()),)

    return (ExactTextSpan(text[start:end], start, end, ()),)


def record_common_spans(
    text: str, record: ExactTextSpan
) -> tuple[ExactTextSpan, ...]:
    """Return exact top-level values shared by records in the same object.

    Record-array values are omitted so one record never smuggles sibling
    records into its encoder input.  Each returned ``text`` is an exact source
    substring; paths are mechanically read from the parsed source object.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if (
        len(record.field_path) != 2
        or not isinstance(record.field_path[0], str)
        or not isinstance(record.field_path[1], int)
    ):
        return ()
    start = _skip_ws(text, 0)
    decoder = json.JSONDecoder()
    try:
        value, end = decoder.raw_decode(text, start)
    except json.JSONDecodeError:
        return ()
    if not isinstance(value, dict) or _skip_ws(text, end) != len(text):
        return ()

    common = []
    for key, value_start, value_end, child in _object_value_ranges(
        text, start, end
    ):
        if (
            isinstance(child, list)
            and child
            and all(isinstance(item, dict) for item in child)
        ):
            continue
        common.append(
            ExactTextSpan(
                text[value_start:value_end],
                value_start,
                value_end,
                (key,),
            )
        )
    return tuple(common)


def _multi_json_record_spans(text: str) -> tuple[ExactTextSpan, ...]:
    """Recognize JSON arrays that visibly contain multiple object records."""

    start = _skip_ws(text, 0)
    decoder = json.JSONDecoder()
    try:
        value, end = decoder.raw_decode(text, start)
    except json.JSONDecodeError:
        return ()
    if _skip_ws(text, end) != len(text):
        return ()
    if isinstance(value, list):
        if len(value) < 2 or not all(isinstance(item, dict) for item in value):
            return ()
        return tuple(
            ExactTextSpan(text[a:b], a, b, (index,))
            for index, (a, b) in enumerate(_array_item_ranges(text, start, end))
        )
    if not isinstance(value, dict):
        return ()

    records = []
    has_multi_record_array = False
    for key, value_start, value_end, child in _object_value_ranges(text, start, end):
        if not isinstance(child, list) or not child or not all(
            isinstance(item, dict) for item in child
        ):
            continue
        has_multi_record_array = has_multi_record_array or len(child) >= 2
        records.extend(
            ExactTextSpan(text[a:b], a, b, (key, index))
            for index, (a, b) in enumerate(
                _array_item_ranges(text, value_start, value_end)
            )
        )
    return tuple(records) if has_multi_record_array else ()


def _array_item_ranges(text: str, start: int, end: int) -> tuple[tuple[int, int], ...]:
    decoder = json.JSONDecoder()
    cursor = _skip_ws(text, start + 1)
    ranges = []
    while cursor < end and text[cursor] != "]":
        _, item_end = decoder.raw_decode(text, cursor)
        ranges.append((cursor, item_end))
        cursor = _skip_ws(text, item_end)
        if cursor < end and text[cursor] == ",":
            cursor = _skip_ws(text, cursor + 1)
        elif cursor < end and text[cursor] != "]":
            raise ValueError("Invalid JSON array boundary")
    return tuple(ranges)


def _object_value_ranges(text: str, start: int, end: int):
    decoder = json.JSONDecoder()
    cursor = _skip_ws(text, start + 1)
    values = []
    while cursor < end and text[cursor] != "}":
        key, key_end = decoder.raw_decode(text, cursor)
        if not isinstance(key, str):
            raise ValueError("Invalid JSON object key")
        cursor = _skip_ws(text, key_end)
        if cursor >= end or text[cursor] != ":":
            raise ValueError("Invalid JSON object boundary")
        value_start = _skip_ws(text, cursor + 1)
        value, value_end = decoder.raw_decode(text, value_start)
        values.append((key, value_start, value_end, value))
        cursor = _skip_ws(text, value_end)
        if cursor < end and text[cursor] == ",":
            cursor = _skip_ws(text, cursor + 1)
        elif cursor < end and text[cursor] != "}":
            raise ValueError("Invalid JSON object boundary")
    return tuple(values)


def _paragraph_spans(text: str) -> tuple[ExactTextSpan, ...]:
    spans = []
    cursor = 0
    separators = list(re.finditer(r"(?:\r?\n)[ \t]*(?:\r?\n)+", text))
    for separator in [*separators, None]:
        end = len(text) if separator is None else separator.start()
        content_start = cursor
        while content_start < end and text[content_start].isspace():
            content_start += 1
        content_end = end
        while content_end > content_start and text[content_end - 1].isspace():
            content_end -= 1
        if content_start < content_end:
            spans.append(
                ExactTextSpan(
                    text[content_start:content_end],
                    content_start,
                    content_end,
                    (len(spans),),
                )
            )
        cursor = len(text) if separator is None else separator.end()
    return tuple(spans)


def _skip_ws(text: str, start: int) -> int:
    while start < len(text) and text[start] in " \t\r\n":
        start += 1
    return start


__all__ = [
    "ENCODING_SCOPES",
    "EncodingScopePlan",
    "EventRecordSpan",
    "ExactTextSpan",
    "event_record_spans",
    "extract_record_spans",
    "plan_encoding_scope",
    "record_common_spans",
    "structural_tool_result_record_spans",
    "validate_encoding_scope",
]
