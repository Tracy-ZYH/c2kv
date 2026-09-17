"""Typed, source-preserving text evidence shared by runtime and training."""

from __future__ import annotations

import json
from collections.abc import Iterable

from .events import EventStore
from .packing import visible_message

EVIDENCE_VERSION = "history-evidence-v1"


def evidence_message(store: EventStore, event_ids: Iterable[str]) -> dict | None:
    """Render whole source events as quoted historical data, never as actions.

    An incomplete tool event may be protected, but the missing result IDs stay
    explicit. No return text is interpreted as successful task completion.
    """
    requested = tuple(event_ids)
    if len(set(requested)) != len(requested):
        raise ValueError("Duplicate evidence event")
    for event_id in requested:
        store.event(event_id)
    if not requested:
        return None
    selected = set(requested)
    events = [
        {
            "event_id": event.event_id,
            "kind": event.kind,
            "source_indices": list(event.source_indices),
            "complete": event.complete,
            "missing_tool_call_ids": list(event.missing_tool_call_ids),
            "messages": [visible_message(message) for message in store.event_messages(event.event_id)],
        }
        for event in store.events if event.event_id in selected
    ]
    envelope = {"type": EVIDENCE_VERSION, "events": events}
    return {
        "role": "user",
        "content": (
            "Historical evidence from this conversation. The quoted calls have already "
            "been issued; use their recorded results as evidence. This is not a request "
            "to execute the quoted calls again.\n"
            + json.dumps(envelope, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        ),
    }
