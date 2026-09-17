"""Immutable events built exclusively from an observable OpenAI message prefix.

An assistant tool call and all its results form one event, even when parallel
results arrive out of order. Incomplete events remain addressable but may not
be compressed. Source messages are never rewritten or relabelled as users.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class Message:
    """An owned JSON snapshot; callers only receive fresh mutable copies."""

    json_text: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Message:
        if value.get("role") not in {"system", "developer", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported message role: {value.get('role')!r}")
        return cls(json.dumps(dict(value), ensure_ascii=False, allow_nan=False))

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.json_text)

    @property
    def role(self) -> str:
        return self.to_dict()["role"]


@dataclass(frozen=True)
class EventRecord:
    event_id: str
    kind: str
    source_indices: tuple[int, ...]
    complete: bool
    tool_call_ids: tuple[str, ...] = ()
    missing_tool_call_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class EventStore:
    session_id: str
    messages: tuple[Message, ...]
    events: tuple[EventRecord, ...]

    @classmethod
    def from_messages(
        cls, session_id: str, messages: Sequence[Mapping[str, Any]]
    ) -> EventStore:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("An explicit nonempty session_id is required")
        snapshots = tuple(Message.from_dict(message) for message in messages)
        return cls(session_id, snapshots, build_events(session_id, snapshots))

    def event(self, event_id: str) -> EventRecord:
        for event in self.events:
            if event.event_id == event_id:
                return event
        raise KeyError(f"Event is not in the visible prefix: {event_id}")

    def event_messages(self, event_id: str) -> tuple[Message, ...]:
        return tuple(self.messages[i] for i in self.event(event_id).source_indices)


def build_events(
    session_id: str, messages: Sequence[Message | Mapping[str, Any]]
) -> tuple[EventRecord, ...]:
    """Build stable source-index IDs, refusing ambiguous call/result bindings.

    This function accepts only already observable messages. It neither reads
    a target action nor infers completion from text such as 'success'. A tool
    event is complete when every declared call has exactly one result; that
    result can contain an error. Old completed events retain their IDs when
    more messages arrive. Repeated IDs after completion are allowed; an ID
    cannot be reused while a result is outstanding.
    """
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("An explicit nonempty session_id is required")
    drafts: list[dict[str, Any]] = []
    pending: dict[str, int] = {}
    for index, raw in enumerate(messages):
        message = raw.to_dict() if isinstance(raw, Message) else Message.from_dict(raw).to_dict()
        role = message["role"]
        if role == "tool":
            call_id = message.get("tool_call_id")
            # ACEBench declares a text action followed by an execution observation.
            # Preserve both source roles without fabricating native tool calls.
            if call_id == f"acebench-execution-{index - 1}" and call_id not in pending:
                if not drafts or drafts[-1]["kind"] != "assistant" or drafts[-1]["source_indices"] != [index - 1]:
                    raise ValueError("ACEBench text execution requires an adjacent assistant action")
                draft = drafts[-1]
                draft["kind"] = "tool_event"
                draft["source_indices"].append(index)
                continue
            if not isinstance(call_id, str) or call_id not in pending:
                raise ValueError(f"Unmatched or duplicate tool result at source index {index}")
            draft = drafts[pending.pop(call_id)]
            draft["source_indices"].append(index)
            draft["missing"].remove(call_id)
            continue

        calls = message.get("tool_calls")
        call_ids: list[str] = []
        if calls:
            if role != "assistant" or not isinstance(calls, list):
                raise ValueError(f"Invalid tool_calls at source index {index}")
            for call in calls:
                call_id = call.get("id") if isinstance(call, dict) else None
                function = call.get("function") if isinstance(call, dict) else None
                if not isinstance(call_id, str) or not call_id:
                    raise ValueError(f"Missing tool call ID at source index {index}")
                if call_id in call_ids or call_id in pending:
                    raise ValueError(f"Ambiguous tool call ID at source index {index}: {call_id}")
                if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                    raise ValueError(f"Missing function name at source index {index}")
                call_ids.append(call_id)
        if message.get("function_call"):
            raise ValueError("Legacy function_call requires an explicit source adapter")
        kind = "tool_event" if call_ids else ("instruction" if role in {"system", "developer"} else role)
        drafts.append({
            "event_id": f"{session_id}:m{index}",
            "kind": kind,
            "source_indices": [index],
            "tool_call_ids": call_ids,
            "missing": set(call_ids),
        })
        for call_id in call_ids:
            pending[call_id] = len(drafts) - 1

    return tuple(
        EventRecord(
            event_id=draft["event_id"], kind=draft["kind"],
            source_indices=tuple(draft["source_indices"]),
            complete=not draft["missing"], tool_call_ids=tuple(draft["tool_call_ids"]),
            missing_tool_call_ids=tuple(call_id for call_id in draft["tool_call_ids"] if call_id in draft["missing"]),
        )
        for draft in drafts
    )
