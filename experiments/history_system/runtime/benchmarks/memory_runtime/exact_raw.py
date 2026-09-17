"""Fill a NoGist raw history body around the shared exact evidence packet."""
from __future__ import annotations

import copy

from history_memory.evidence import evidence_message


def build_exact_raw_view(
        store, full_messages, full_counts, *, source_cutoff, evidence_event_ids,
        token_counter, tools, bytes_per_kv_token, history_budget_bytes,
        workspace_budget_bytes, reference_common_tokens):
    """Select complete raw events by recency under the measured joint B/W caps.

    The original Full training renderer is message-preserving (apart from its
    optional default system message). Selecting its already-rendered records
    preserves that dialect and keeps the original live boundary fixed. Evidence
    has priority; every call refills the raw body from the same frozen prefix.
    """
    source = [message.to_dict() for message in store.messages]
    shift = int(not any(message.get("role") == "system" for message in source))
    if len(full_messages) != len(source) + shift or any(
            message.get("c2kv_key_hash") for message in full_messages):
        raise ValueError("NoGist requires the unmodified message-preserving Full renderer")
    if type(source_cutoff) is not int or not 0 <= source_cutoff <= len(source):
        raise ValueError("Invalid original source cutoff")
    if any(type(value) is not int or value < 0 for value in (
            history_budget_bytes, workspace_budget_bytes, reference_common_tokens)):
        raise ValueError("NoGist budgets and reference count must be nonnegative integers")
    if type(bytes_per_kv_token) is not int or bytes_per_kv_token <= 0:
        raise ValueError("Invalid KV byte geometry")
    full = copy.deepcopy(full_messages)
    cutoff = source_cutoff + shift
    common_positions = {index for index, message in enumerate(full)
                        if index >= cutoff or message.get("role") in {"system", "developer"}}
    common_sources = {index - shift for index in common_positions if index >= shift}
    selected_evidence = tuple(evidence_event_ids)
    packet = evidence_message(store, selected_evidence)
    evidence_sources = {index for event_id in selected_evidence
                        for index in store.event(event_id).source_indices}
    evidence_set = set(selected_evidence)

    def count(messages):
        value = token_counter(messages, tools)
        if type(value) is not int or value < 0:
            raise ValueError("token_counter must return a nonnegative integer")
        return value

    def measure(raw_ids):
        raw_sources = {index for event_id in raw_ids
                       for index in store.event(event_id).source_indices}
        positions = common_positions | {index + shift for index in raw_sources}
        before = [full[index] for index in sorted(positions) if index < cutoff]
        suffix = [full[index] for index in sorted(positions) if index >= cutoff]
        without_evidence = before + suffix
        out = before + ([packet] if packet is not None else []) + suffix
        total_tokens = count(out)
        base_tokens = count(without_evidence)
        history_bytes = (total_tokens - reference_common_tokens) * bytes_per_kv_token
        evidence_bytes = (total_tokens - base_tokens) * bytes_per_kv_token
        if history_bytes < 0 or evidence_bytes < 0:
            raise ValueError("NoGist measured history/evidence delta became negative")
        return (out, positions, len(before), total_tokens, base_tokens,
                history_bytes, evidence_bytes, raw_sources)

    common = [full[index] for index in sorted(common_positions)]
    if count(common) != reference_common_tokens:
        raise ValueError("NoGist common view differs from the shared controller reference")
    empty = measure(())
    if empty[5] > history_budget_bytes or empty[6] > workspace_budget_bytes:
        raise ValueError("Shared evidence alone exceeds the NoGist B/W caps")
    selection_evidence_bytes = empty[6]
    for event in store.events:
        if not event.complete and event.event_id not in evidence_set and not (
                set(event.source_indices) <= common_sources):
            raise ValueError("An incomplete event escaped shared evidence protection")

    candidates = [event for event in store.events
                  if event.kind != "instruction" and event.complete
                  and max(event.source_indices) < source_cutoff
                  and event.event_id not in evidence_set]
    selected_raw = set()
    skipped = []
    for event in reversed(candidates):
        trial = measure(selected_raw | {event.event_id})
        if trial[5] <= history_budget_bytes and trial[6] <= workspace_budget_bytes:
            selected_raw.add(event.event_id)
        else:
            skipped.append({
                "event_id": event.event_id,
                "reason": "joint_history_budget" if trial[5] > history_budget_bytes
                          else "actual_evidence_workspace_budget",
                "candidate_history_bytes": trial[5], "candidate_evidence_bytes": trial[6],
            })
    out, positions, before_count, total, without_e, active, actual_e, raw_sources = measure(selected_raw)
    selected_ids = [event.event_id for event in store.events if event.event_id in selected_raw]
    visible = common_sources | raw_sources | evidence_sources
    counts = dict(full_counts)
    counts.update(
        system_raw=sum(message.get("role") == "system" for message in out),
        history_raw=sum(full[index].get("role") != "system" for index in positions if index < cutoff)
                    + int(packet is not None),
        current_raw=sum(full[index].get("role") != "system" for index in positions if index >= cutoff),
        current_start_out_index=before_count + int(packet is not None),
        compressed=0, compressed_records=[], gist_tokens=0, original_tokens=0,
        n_gist_messages=0, n_docs=0, dropped_docs=0,
        history_packed_original_tokens=None, history_dropped_original_tokens=None,
        history_packed_candidate_doc_count=None, history_retained_fraction=None)
    metadata = {
        "raw_body_policy": "complete_history_events_newest_first_after_evidence",
        "source_cutoff": source_cutoff,
        "raw_history_event_ids": selected_ids,
        "raw_history_source_indices": sorted(raw_sources),
        "visible_source_indices": sorted(visible),
        "common_source_indices": sorted(common_sources),
        "evicted_raw_event_ids": [event.event_id for event in candidates
                                  if event.event_id not in selected_raw],
        "raw_skipped_for_budget": list(reversed(skipped)),
        "active_history_bytes": active, "evidence_bytes": actual_e,
        "auxiliary_selection_bytes": selection_evidence_bytes,
        "raw_body_bytes": (without_e - reference_common_tokens) * bytes_per_kv_token,
        "raw_history_tokens": total - reference_common_tokens,
        "common_raw_prompt_tokens": reference_common_tokens,
        "total_raw_prompt_tokens": total,
        "evidence_out_index": before_count if packet is not None else None,
    }
    return out, counts, metadata
