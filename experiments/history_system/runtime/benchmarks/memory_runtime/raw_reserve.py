"""Spend unused capacity on at most one complete representation-backed raw event."""
import copy

from history_memory.events import EventStore
from .adapter import raw_source_cutoff
from .always_compress import coverage_accounting, ratio_accounting

RESERVE_VERSION = "gist-preserving-raw-reserve-v1"
SUMMARY_RESERVE_VERSION = "representation-preserving-raw-reserve-v2"

SUMMARY_ROUTE = "text_summary_native_needs_lexical_raw_reserve_failed_operation"
DEPENDENCY_PACKET_ROUTE = "ac_native_dependency_packet_lexical_raw_reserve_failed_operation"


def reserve_older_raw(source, messages, counts, tools, token_counter, full_messages):
    summary_route = counts["memory_runtime"]["route_mode"] == SUMMARY_ROUTE
    condition = ("recent_representation_backed_event" if summary_route
                 else "recent_gist_backed_event")
    out, updated = copy.deepcopy(messages), copy.deepcopy(counts)
    base, meta = counts["memory_runtime"], updated["memory_runtime"]
    if base["route_mode"] not in {"ac_native_needs_lexical_raw_reserve",
                                   "ac_native_needs_lexical_raw_reserve_failed_operation",
                                   DEPENDENCY_PACKET_ROUTE,
                                   "raw_native_needs_lexical_raw_reserve_failed_operation",
                                   SUMMARY_ROUTE}:
        raise ValueError("Raw reserve requires its dedicated lexical runtime route")
    full = full_messages
    shift = len(full)-len(source)
    if shift not in (0, 1):
        raise ValueError("Unexpected native Full renderer source shift")
    selected = set(base["selected_source_indices"])
    representation_positions = set(base.get("representation_out_indices") or [])
    packet_index = (base.get("dependency_packet") or {}).get("out_index")
    non_native_positions = representation_positions | ({packet_index} if packet_index is not None else set())
    raw = [m for i, m in enumerate(messages)
           if not m.get("c2kv_key_hash") and i not in non_native_positions]
    expected = [m for i, m in enumerate(full) if i < shift or i-shift in selected]
    billed = [m for m in messages if not m.get("c2kv_key_hash")]
    if raw != expected or token_counter(billed, tools) != base["total_raw_prompt_tokens"]:
        raise ValueError("Original native view does not match its source and token ledger")
    store = EventStore.from_messages(base["task_id"], source)
    eligible = set(base["source_coverage"]["eligible_source_indices"])
    coverage_before = base["source_coverage"]
    representation_backed = set(
        coverage_before.get("summary_input_fully_retained_source_indices")
        if base["route_mode"] == SUMMARY_ROUTE
        else coverage_before.get("gist_fully_represented_source_indices") or [])
    candidates = [event for event in store.events if event.kind == "tool_event" and event.complete
        and set(event.source_indices) <= eligible & representation_backed
        and not set(event.source_indices) <= selected
        and event.event_id not in base["protected_event_ids"]]
    event = max(candidates, key=lambda e: max(e.source_indices)) if candidates else None
    budget = min(base["history_budget_bytes"], base["workspace_budget_bytes"])
    unit = base["bytes_per_kv_token"]
    receipt = {"version": SUMMARY_RESERVE_VERSION if summary_route else RESERVE_VERSION,
        "condition": condition,
        "maximum_extra_events": 1, "original_selected_source_indices": sorted(selected),
        "candidate_event_id": event.event_id if event else None, "admitted_event_id": None,
        "added_source_indices": [], "extra_raw_tokens": 0, "extra_kv_equivalent_bytes": 0,
        "unused_capacity_before_bytes": budget-base["active_history_bytes"],
        "status": "no_eligible_extra_event", "existing_raw_messages_unchanged": True,
        "protected_events_unchanged": True, "no_fallback_search": True,
        "scope": ("One complete older representation-backed event may spend unused B/W capacity. Original lexical requests, all retained representation messages and existing raw stay intact; extra raw is separately charged. No prediction, generation, hidden state, or future response enters selection."
            if summary_route else
            "One complete older gist-backed event may spend unused B/W capacity. Original lexical requests, all retained gist and existing raw stay intact; extra raw is separately charged. No prediction, generation, hidden state, or future response enters selection.")}
    if summary_route:
        receipt["representation_messages_unchanged"] = True
    else:
        receipt["gist_messages_unchanged"] = True
    if event:
        added = set(event.source_indices)-selected
        trial = selected | set(event.source_indices)
        cutoff = raw_source_cutoff(source)
        indices = [i for i in sorted(trial) if i < cutoff and source[i].get("role") != "system"]
        old_end = counts["current_start_out_index"]
        old_positions = base["pre_generation_workspace"]["native_workspace_out_indices"]
        old_start = min(old_positions) if old_positions else old_end
        if old_positions != list(range(old_start, old_end)):
            raise ValueError("Native workspace is not one contiguous source-ordered region")
        new_raw_rows = [copy.deepcopy(full[i+shift]) for i in indices]
        candidate = copy.deepcopy(messages[:old_start])+new_raw_rows+copy.deepcopy(messages[old_end:])
        candidate_native = [m for i, m in enumerate(candidate)
            if not m.get("c2kv_key_hash") and i not in non_native_positions]
        expected_trial = [m for i, m in enumerate(full) if i < shift or i-shift in trial]
        if candidate_native != expected_trial:
            raise ValueError("Extra-event rendering changed source order or existing raw")
        tokens = token_counter([m for m in candidate if not m.get("c2kv_key_hash")], tools)
        extra = tokens-base["total_raw_prompt_tokens"]
        if extra <= 0:
            raise ValueError("A nonempty complete extra event must add raw tokens")
        active_bytes = base["active_history_bytes"]+extra*unit
        receipt.update(candidate_extra_raw_tokens=extra, candidate_active_history_bytes=active_bytes)
        if active_bytes > budget:
            receipt["status"] = "extra_event_over_budget"
        else:
            out = candidate
            if base["route_mode"] == SUMMARY_ROUTE:
                from memory_runtime.source_needs_runtime import summary_coverage_accounting
                coverage = summary_coverage_accounting(
                    eligible_sources=eligible, raw_sources=trial,
                    retained_summaries=base["representation_refs"],
                    packing_fragments=counts["history_packing_fragments"])
                if (coverage["summary_input_fully_retained_source_indices"]
                        != coverage_before["summary_input_fully_retained_source_indices"]):
                    raise ValueError("Raw reserve changed summary input provenance")
            else:
                coverage = coverage_accounting(eligible_sources=eligible, raw_sources=trial,
                    retained_blocks=base["block_refs"],
                    packing_fragments=counts.get("history_packing_fragments") if base["block_refs"] else [])
                if (coverage["unrepresented_source_indices"]
                        != base["source_coverage"]["unrepresented_source_indices"]):
                    raise ValueError("A fully gist-backed extra event changed complete coverage")
            new_end = old_start+len(indices)
            updated["current_start_out_index"] = new_end
            meta["pre_generation_workspace"].update(restored_source_indices=indices,
                native_workspace_out_indices=list(range(old_start, new_end)))
            meta.update(selected_source_indices=sorted(trial), source_coverage=coverage,
                selected_event_ids=[e.event_id for e in store.events if set(e.source_indices) <= trial],
                total_raw_prompt_tokens=tokens, raw_history_tokens=base["raw_history_tokens"]+extra,
                active_history_bytes=active_bytes, evidence_bytes=base["evidence_bytes"]+extra*unit)
            receipt.update(status="extra_event_admitted", admitted_event_id=event.event_id,
                added_source_indices=sorted(added), extra_raw_tokens=extra,
                extra_kv_equivalent_bytes=extra*unit)
            updated["history_raw"] = counts["history_raw"] + len(added)
    if base["route_mode"] == SUMMARY_ROUTE:
        if ([out[i] for i in sorted(representation_positions)]
                != [messages[i] for i in sorted(representation_positions)]):
            raise ValueError("Raw reserve changed retained summaries")
    elif ([m for m in out if m.get("c2kv_key_hash")]
          != [m for m in messages if m.get("c2kv_key_hash")]):
        raise ValueError("Raw reserve changed retained gist")
    meta["raw_reserve"] = receipt
    meta["byte_geometry_verified_by_backend"] = False
    if "raw_prompt_tokens_verified_by_backend" in meta:
        meta["raw_prompt_tokens_verified_by_backend"] = False
    meta["compression_ratio"] = ratio_accounting(
        {"active_history_bytes": base["compression_ratio"]["full_history_bytes"],
         "common_raw_prompt_tokens": base["common_raw_prompt_tokens"]}, meta)
    meta["compression_ratio"]["includes_coverage_loss"] = (
        None if base["route_mode"] == SUMMARY_ROUTE
        else bool(meta["source_coverage"]["unrepresented_source_indices"]))
    if (meta["active_history_bytes"] != (meta["raw_history_tokens"]+meta["gist_tokens"])*unit
            or meta["active_history_bytes"] > budget):
        raise ValueError("Raw-reserve accounting exceeds the frozen B/W contract")
    return out, updated
