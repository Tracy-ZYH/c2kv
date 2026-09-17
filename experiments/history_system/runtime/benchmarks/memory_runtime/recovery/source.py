"""Observable source selection for post-draft event recovery."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from history_memory.events import EventStore

from ..policy import (
    _anchors,
    _argument_strings,
    _event_text,
    _is_explicit_revision,
    _tokens,
)


def select_source_event(prepared: Any, draft_tool_calls: Any, *, draft_text: str):
    receipt = {
        "policy": "current-goal-plus-held-draft-visible-event-lexical-v2",
        "reason": "no_held_draft_query",
        "ranked_candidate_event_ids": [],
        "selected_event": None,
        "current_prefix_only": True,
        "draft_arguments_serialized_in_receipt": False,
        "draft_text_serialized_in_receipt": False,
        "uses_gold_future_or_tool_result": False,
        "staleness_policy": "explicit-current-revision-abstain-v1",
        "candidate_scope": (
            "complete non-instruction events before the live raw suffix"
        ),
        "fabricated_tool_execution": False,
    }
    calls = valid_tool_calls(draft_tool_calls)
    if not isinstance(draft_text, str):
        raise TypeError("draft_text must be a string")
    if not draft_text.strip() and not calls:
        return None, receipt
    users = [event for event in prepared._store.events if event.kind == "user"]
    current_user = users[-1] if users else None
    if current_user is None:
        receipt["reason"] = "no_current_user_goal"
        return None, receipt
    if _is_explicit_revision(prepared._store, current_user):
        receipt["reason"] = "current_goal_is_explicit_revision"
        return None, receipt

    excluded = set(prepared.memory.view.raw_event_ids)
    excluded.update(prepared.memory.view.mandatory_raw_event_ids)
    excluded.update(prepared.metadata.get("revision_cancelled_event_ids") or ())
    eligible = set(
        prepared.metadata["eligible_extraction"]["eligible_event_ids"]
    )
    ranked = rank_visible_source_events(
        prepared._store,
        current_user=current_user,
        draft_text=draft_text,
        draft_tool_calls=calls,
        eligible=eligible,
        excluded=excluded,
    )
    receipt["ranked_candidate_event_ids"] = list(ranked)
    if not ranked:
        receipt["reason"] = "no_complete_unseen_compatible_event"
        return None, receipt
    receipt["selected_event"] = source_event_receipt(prepared, ranked[0])
    receipt["reason"] = "highest_ranked_complete_unseen_event"
    return ranked[0], receipt


def source_event_receipt(prepared: Any, candidate: str) -> dict[str, Any]:
    event = prepared._store.event(candidate)
    source_messages = prepared._store.event_messages(event.event_id)
    return {
        "event_id": event.event_id,
        "kind": event.kind,
        "source_indices": list(event.source_indices),
        "source_roles": [message.role for message in source_messages],
        "source_message_sha256": [
            hashlib.sha256(message.json_text.encode("utf-8")).hexdigest()
            for message in source_messages
        ],
        "complete": event.complete,
        "already_raw_visible": False,
        "revision_cancelled": False,
    }


def rank_visible_source_events(
    store: EventStore,
    *,
    current_user: Any,
    draft_text: str,
    draft_tool_calls: list[dict[str, Any]],
    eligible: set[str],
    excluded: set[str],
) -> tuple[str, ...]:
    """Rank complete visible events without assuming a tool-call transport."""

    query_parts = [_event_text(store, current_user), draft_text]
    exact_parts = []
    for call in draft_tool_calls:
        function = call["function"]
        query_parts.append(function["name"])
        arguments = _argument_strings(function["arguments"])
        query_parts.extend(arguments)
        exact_parts.extend(arguments)
    query = "\n".join(part for part in query_parts if part)
    query_tokens = _tokens(query)
    anchors = _anchors(query)
    anchors.update(
        normalized
        for part in exact_parts
        if len(normalized := " ".join(part.casefold().split())) >= 2
    )
    if not query_tokens and not anchors:
        return ()

    ranked: list[tuple[float, int, str]] = []
    for event in store.events:
        if (
            event.event_id not in eligible
            or event.event_id in excluded
            or not event.complete
            or event.kind == "instruction"
            or event.event_id == current_user.event_id
        ):
            continue
        text = _event_text(store, event)
        lower = text.casefold()
        exact_hits = sum(1 for anchor in anchors if anchor in lower)
        event_tokens = _tokens(text)
        overlap = query_tokens & event_tokens
        if not exact_hits and not overlap:
            continue
        union = query_tokens | event_tokens
        lexical = len(overlap) / len(union) if union else 0.0
        kind_bonus = 8.0 if event.kind == "tool_event" else 0.0
        score = exact_hits * 100.0 + len(overlap) * 4.0 + lexical + kind_bonus
        ranked.append((score, max(event.source_indices), event.event_id))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return tuple(event_id for _, _, event_id in ranked)


def valid_tool_calls(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)) or not value:
        return []
    calls = []
    for index, call in enumerate(value):
        if not isinstance(call, Mapping):
            return []
        function = call.get("function")
        if not isinstance(function, Mapping) or not isinstance(
            function.get("name"), str
        ):
            return []
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            return []
        try:
            json.loads(arguments)
        except json.JSONDecodeError:
            return []
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id:
            call_id = f"held-draft-{index}"
        calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": function["name"],
                    "arguments": arguments,
                },
            }
        )
    return calls
