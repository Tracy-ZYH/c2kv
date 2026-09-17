"""Pure-CPU, budgeted raw-history recency view for the Full renderer."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from history_memory.events import EventStore


def _count(token_counter, messages, tools):
    value = token_counter(messages, tools)
    if type(value) is not int or value < 0:
        raise ValueError("token_counter must return a nonnegative integer")
    return value


def _render(render, source):
    rendered, counts = render([dict(message) for message in source])
    if not isinstance(rendered, list) or not isinstance(counts, dict):
        raise ValueError("render must return (list, dict)")
    if any(not isinstance(message, dict) for message in rendered):
        raise ValueError("render returned a non-message item")
    return rendered, counts


def build_raw_recency_view(
    store: EventStore,
    render: Callable[[list[dict[str, Any]]], tuple[list, dict]],
    token_counter: Callable[[list[dict[str, Any]], Any], int],
    tools: Any,
    bytes_per_kv_token: int,
    history_budget_bytes: int,
) -> tuple[list, dict, dict]:
    """Select whole source events by recency, then use the Full training renderer.

    Common live input is the original instruction set plus the source suffix
    after the latest assistant. Events crossing that boundary and incomplete
    events are mandatory and remain byte-for-byte source records.
    """
    if not isinstance(store, EventStore):
        raise TypeError("store must be an EventStore")
    if not callable(render) or not callable(token_counter):
        raise TypeError("render and token_counter must be callable")
    if type(bytes_per_kv_token) is not int or bytes_per_kv_token <= 0:
        raise ValueError("bytes_per_kv_token must be a positive integer")
    if type(history_budget_bytes) is not int or history_budget_bytes < 0:
        raise ValueError("history_budget_bytes must be a nonnegative integer")

    source = [message.to_dict() for message in store.messages]
    last_anchor = max(
        (index for index, message in enumerate(source)
         if message.get("role") in {"user", "tool"}),
        default=-1,
    )
    source_cutoff = next(
        (index + 1 for index in range(last_anchor, -1, -1)
         if source[index].get("role") == "assistant"),
        0,
    )
    suffix_indices = set(range(source_cutoff, len(source)))
    instruction_indices = {
        index for event in store.events if event.kind == "instruction"
        for index in event.source_indices
    }
    common_indices = instruction_indices | suffix_indices
    raw_visible_ids = tuple(
        event.event_id for event in store.events
        if set(event.source_indices) <= common_indices
    )
    mandatory_ids = tuple(
        event.event_id for event in store.events
        if (event.kind == "instruction" or not event.complete
            or bool(set(event.source_indices) & suffix_indices))
    )
    mandatory_set = set(mandatory_ids)
    selected_indices = set(common_indices)
    for event_id in mandatory_ids:
        selected_indices.update(store.event(event_id).source_indices)

    def measure(indices):
        selected_source = [source[index] for index in sorted(indices)]
        rendered, counts = _render(render, selected_source)
        return rendered, counts, _count(token_counter, rendered, tools)

    _, _, common_tokens = measure(common_indices)
    full_indices = set(range(len(source)))
    full_messages, full_counts, full_tokens = measure(full_indices)
    full_history_tokens = full_tokens - common_tokens
    if full_history_tokens < 0:
        raise ValueError("Full raw history token delta became negative")

    selected_optional = set()
    skipped = []
    if full_history_tokens * bytes_per_kv_token <= history_budget_bytes:
        selected_indices = full_indices
        selected_optional = {
            event.event_id for event in store.events
            if event.event_id not in mandatory_set and event.kind != "instruction"
        }
        rendered, counts, total_tokens = full_messages, full_counts, full_tokens
    else:
        rendered, counts, total_tokens = measure(selected_indices)
        mandatory_tokens = total_tokens - common_tokens
        if mandatory_tokens < 0:
            raise ValueError("Mandatory raw history token delta became negative")
        mandatory_bytes = mandatory_tokens * bytes_per_kv_token
        if mandatory_bytes > history_budget_bytes:
            raise ValueError(
                f"Mandatory complete/current events need {mandatory_bytes} bytes; "
                f"budget is {history_budget_bytes}"
            )

        for event in reversed(store.events):
            if event.event_id in mandatory_set or event.kind == "instruction":
                continue
            if not event.complete:
                raise ValueError("Incomplete event escaped mandatory raw selection")
            trial_indices = selected_indices | set(event.source_indices)
            _, _, trial_tokens = measure(trial_indices)
            trial_history_tokens = trial_tokens - common_tokens
            if trial_history_tokens < 0:
                raise ValueError("Raw history token delta became negative")
            trial_bytes = trial_history_tokens * bytes_per_kv_token
            if trial_bytes <= history_budget_bytes:
                selected_indices = trial_indices
                selected_optional.add(event.event_id)
            else:
                skipped.append({
                    "event_id": event.event_id,
                    "reason": "raw_recency_history_budget",
                    "candidate_cost_bytes": trial_bytes,
                })

        rendered, counts, total_tokens = measure(selected_indices)
    raw_history_tokens = total_tokens - common_tokens
    active_history_bytes = raw_history_tokens * bytes_per_kv_token
    if raw_history_tokens < 0 or active_history_bytes > history_budget_bytes:
        raise ValueError("Final raw-recency view violates its frozen history budget")
    selected_ids = tuple(
        event.event_id for event in store.events
        if set(event.source_indices) <= selected_indices
    )
    skipped.reverse()

    selected_positions = sorted(selected_indices)
    fallback_current_raw = sum(
        source[index].get("role") != "system"
        for index in selected_indices if index >= source_cutoff
    )
    count_defaults = {
        "system_raw": sum(message.get("role") == "system" for message in rendered),
        "history_raw": sum(
            source[index].get("role") != "system"
            for index in selected_indices if index < source_cutoff),
        "current_raw": fallback_current_raw,
        "compressed": 0,
        "gist_tokens": 0,
        "original_tokens": 0,
        "n_gist_messages": 0,
        "compressed_records": [],
        "doc_packing": "message",
        "n_docs": 0,
        "dropped_docs": 0,
        "current_start_out_index": len(rendered) - fallback_current_raw,
        "history_packed_original_tokens": None,
        "history_dropped_original_tokens": None,
        "history_packed_candidate_doc_count": None,
        "history_retained_fraction": None,
    }
    for key, value in count_defaults.items():
        counts.setdefault(key, value)
    metadata = {
        "mode": "raw_recency",
        "bytes_per_kv_token": bytes_per_kv_token,
        "history_budget_bytes": history_budget_bytes,
        "budget_applies": True,
        "active_history_bytes": active_history_bytes,
        "evidence_bytes": 0,
        "gist_tokens": 0,
        "raw_history_tokens": raw_history_tokens,
        "common_raw_prompt_tokens": common_tokens,
        "total_raw_prompt_tokens": total_tokens,
        "selected_event_ids": list(selected_ids),
        "recency_selected_event_ids": [
            event.event_id for event in store.events
            if event.event_id in selected_optional
        ],
        "raw_visible_event_ids": list(raw_visible_ids),
        "mandatory_event_ids": list(mandatory_ids),
        "selected_source_indices": selected_positions,
        "common_source_indices": sorted(common_indices),
        "skipped_event_ids": [item["event_id"] for item in skipped],
        "skipped_for_budget": skipped,
        "evicted_event_ids": [item["event_id"] for item in skipped],
        "eviction_reason": "raw_recency_history_budget" if skipped else None,
        "raw_pending_event_ids": [
            event.event_id for event in store.events if not event.complete
        ],
        "missing_tool_call_ids": {
            event.event_id: list(event.missing_tool_call_ids)
            for event in store.events if event.missing_tool_call_ids
        },
        "protected_event_ids": list(mandatory_ids),
        "retrieved_event_ids": [],
        "retained_event_ids": [],
        "block_refs": [],
        "evicted_gist_keys": [],
        "evidence_out_index": None,
        "policy": {"selection": "complete_events_newest_first"},
    }
    if not skipped and selected_indices == full_indices:
        if rendered != full_messages:
            raise ValueError("All-fit raw-recency view differs from Full rendering")
        metadata["all_history_fits"] = True
    else:
        metadata["all_history_fits"] = False
    return rendered, counts, metadata
