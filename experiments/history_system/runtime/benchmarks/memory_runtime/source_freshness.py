"""Bounded current-goal recency preference for the lexical source candidate.

Exact calls are an observable grouping heuristic, not semantic resource IDs.
This changes retrieval priority only; it does not delete history or certify
completion, prevent retries, or prefer successful results over later errors.
"""

from __future__ import annotations

import json

from history_memory.events import EventStore


FRESHNESS_VERSION = "current-goal-same-call-freshness-v1"
# Exact names from the development benchmark's supplied tool schemas.
CONTEXT_BARRIERS = frozenset({
    "cd", "authenticate_twitter", "logout", "message_login", "ticket_login",
    "trading_login", "trading_logout",
})


def refresh_source_ids(store: EventStore, requested, candidate_ids):
    """Replace a requested singleton event with its latest eligible exact call.

Both events must belong to the current goal and the same interval between
context-changing calls. The newest complete match must be in the fitted pool;
an intermediate match is never substituted for an unavailable newest event.
Normal native admission still decides whether a requested event fits.
"""
    pool = set(candidate_ids)
    records, goal, interval = {}, None, 0
    for event in store.events:
        if event.kind == "user":
            goal = event.event_id
        calls = [call for snapshot in store.event_messages(event.event_id)
                 for call in (snapshot.to_dict().get("tool_calls") or ())]
        barrier = any(call["function"]["name"] in CONTEXT_BARRIERS for call in calls)
        if barrier:
            interval += 1
        if event.kind != "tool_event" or not event.complete or len(calls) != 1 or barrier:
            continue
        function = calls[0]["function"]
        arguments = function.get("arguments")
        try:
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                continue
            key = (function["name"], json.dumps(arguments, sort_keys=True,
                   ensure_ascii=False, separators=(",", ":"), allow_nan=False))
        except (ValueError, TypeError):
            continue
        records[event.event_id] = {"goal": goal, "interval": interval, "key": key}

    output, decisions = [], []
    for old in requested:
        target, reason = old, "no_later_complete_same_call"
        record = records.get(old)
        if record is None:
            reason = "not_singleton_exact_call_or_context_barrier"
        elif record["goal"] != goal:
            reason = "outside_current_goal"
        else:
            matches = [event_id for event_id, row in records.items()
                       if row["goal"] == goal and row["key"] == record["key"]]
            latest = matches[-1]
            if latest != old:
                if records[latest]["interval"] != record["interval"]:
                    reason = "context_barrier_between_calls"
                elif latest not in pool:
                    reason = "latest_not_in_fitted_pool"
                else:
                    target, reason = latest, "latest_current_goal_same_call"
        duplicate = target in output
        if not duplicate:
            output.append(target)
        decisions.append({"original_event_id": old, "requested_event_id": target,
                          "reason": reason, "deduplicated": duplicate})
    return tuple(output), {"version": FRESHNESS_VERSION,
        "current_goal_event_id": goal, "original_requested_event_ids": list(requested),
        "requested_event_ids": output, "decisions": decisions,
        "context_barrier_names": sorted(CONTEXT_BARRIERS),
        "budget_fallback": "ordinary admission; no fallback to the older event"}
