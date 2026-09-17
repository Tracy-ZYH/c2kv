"""B0 admission and metadata updates for one selected recovery event."""

from __future__ import annotations

import copy
from typing import Any


def admit_event(base: Any, prepared: Any, candidate: str) -> dict[str, Any]:
    store, tools = prepared._store, prepared._tools
    raw = list(prepared.memory.view.raw_event_ids)
    gist = list(prepared.memory.view.gist_event_ids)
    mandatory = set(prepared.memory.view.mandatory_raw_event_ids)
    protected = mandatory | set(prepared.metadata.get("protected_event_ids") or ())
    eligible = prepared.metadata["eligible_extraction"]["eligible_event_ids"]
    common = prepared.metadata["common_raw_prompt_tokens"]
    max_new = prepared.metadata["max_new_tokens"]
    derived = tuple(
        prepared.metadata.get("derived_workspace_prefix_messages") or ()
    )

    demotion_priority = []
    demotion_priority.extend(
        prepared.metadata.get("recency_selected_event_ids") or ()
    )
    already_admitted = (
        prepared.metadata.get("source_needs", {}).get("admitted_event_ids") or ()
    )
    demotion_priority.extend(reversed(already_admitted))
    known = set(demotion_priority)
    generic = [
        event
        for event in store.events
        if event.event_id in set(raw)
        and event.event_id not in known
        and event.event_id not in protected
        and event.event_id in set(gist)
    ]
    generic.sort(key=lambda event: (max(event.source_indices), event.event_id))
    demotion_priority.extend(event.event_id for event in generic)
    demoted = []
    released = []

    def trial():
        measure = base._try_measure(
            store,
            tools,
            (*raw, candidate),
            mandatory,
            gist,
            eligible,
            common,
            max_new,
            derived_messages=derived,
        )
        return measure if measure is not None and not measure.reasons else None

    measure = trial()
    for event_id in demotion_priority:
        if measure is not None:
            break
        if (
            event_id in raw
            and event_id in gist
            and event_id not in protected
            and event_id != candidate
        ):
            raw.remove(event_id)
            demoted.append(event_id)
            measure = trial()

    reserved = prepared.metadata.get("gist_reservation", {}).get(
        "reserved_event_id"
    )
    gist_priority = list(
        prepared.metadata.get("gist_refill_priority_event_ids") or gist
    )
    for event_id in reversed(gist_priority):
        if measure is not None:
            break
        if event_id in gist and event_id != reserved:
            gist.remove(event_id)
            released.append(event_id)
            measure = trial()

    receipt = {
        "policy": "r-event-b0-repack-v1",
        "status": "admitted" if measure is not None else "abstained",
        "candidate_event_id": candidate,
        "demoted_raw_event_ids": demoted,
        "released_gist_event_ids": released,
        "mandatory_raw_event_ids_unchanged": True,
        "protected_event_ids_unchanged": True,
        "minimum_gist_event_id": reserved,
        "minimum_gist_preserved": reserved is None or reserved in gist,
        "b0_rechecked_after_all_changes": True,
        "admission_failures": (
            [] if measure is not None else ["declared_b0_or_packing_limit"]
        ),
    }
    return {"measure": measure, "receipt": receipt}


def metadata_after_admission(
    base: Any,
    prepared: Any,
    measure: Any,
    candidate: str,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    metadata = copy.deepcopy(prepared.metadata)
    store = prepared._store
    ratio = metadata["requested_ratio"]
    view = measure.memory.view
    raw = set(view.raw_event_ids)
    gist = set(view.gist_event_ids)
    eligible = metadata["eligible_extraction"]["eligible_event_ids"]
    eligible_sources = frozenset(
        metadata["eligible_extraction"]["eligible_source_indices"]
    )
    coverage, packing_fragments, retained_fragments = base._coverage(
        store,
        eligible,
        eligible_sources,
        prepared.eligible_chunks,
        measure.memory,
    )
    selected = [
        event.event_id for event in store.events if event.event_id in raw
    ]
    omitted = [
        event.event_id
        for event in store.events
        if event.event_id not in raw and event.event_id not in gist
    ]
    metadata.update(
        raw_prompt_tokens=measure.raw_prompt_tokens,
        actual_raw_history_tokens=measure.raw_history_tokens,
        actual_gist_tokens=measure.per_ratio[str(ratio)]["history_gist_tokens"],
        actual_history_bytes=measure.per_ratio[str(ratio)]["history_bytes"],
        logical_sequence_tokens=measure.logical_sequence_tokens,
        per_ratio=copy.deepcopy(measure.per_ratio),
        raw_source_indices=list(measure.memory.raw_source_indices),
        raw_event_ids=list(view.raw_event_ids),
        gist_event_ids=list(view.gist_event_ids),
        omitted_event_ids=omitted,
        selected_event_ids=selected,
        selected_source_indices=list(measure.memory.raw_source_indices),
        raw_evidence_event_ids=list(
            dict.fromkeys(
                [*(metadata.get("raw_evidence_event_ids") or ()),
                 *([candidate] if candidate is not None else [])]
            )
        ),
        source_coverage=coverage,
        history_packing_fragments=packing_fragments,
        retained_history_packing_fragments=retained_fragments,
        full_source_coverage=coverage["complete_history_coverage"],
        compression_ratio=base._compression_ratio(
            metadata["same_prefix_full_reference"], measure, coverage, ratio
        ),
        post_draft_exact_recovery_applied=True,
        recovery_stage="post_draft_pre_tool_e1",
    )
    metadata["retrieved_event_ids"] = [
        event_id
        for event_id in metadata.get("retrieved_event_ids") or ()
        if event_id in raw
    ]
    metadata["recency_selected_event_ids"] = [
        event_id
        for event_id in metadata.get("recency_selected_event_ids") or ()
        if event_id in raw
    ]
    source_needs = metadata.get("source_needs")
    if isinstance(source_needs, dict):
        source_needs["admitted_event_ids"] = [
            event_id
            for event_id in source_needs.get("admitted_event_ids") or ()
            if event_id in raw
        ]
    extraction = metadata["eligible_extraction"]
    extraction.update(
        retained_event_ids=list(view.gist_event_ids),
        retained_chunk_count=len(measure.memory.chunks),
        retained_presented_encoder_tokens=sum(
            len(chunk.token_ids) for chunk in measure.memory.chunks
        ),
        omitted_active_gist_event_ids=[
            event_id for event_id in eligible if event_id not in gist
        ],
        raw_gist_overlap_event_ids=[
            event_id
            for event_id in eligible
            if event_id in raw and event_id in gist
        ],
    )
    metadata["gist_refilled_event_ids"] = list(view.gist_event_ids)
    metadata["min_gist_reservation_met"] = (
        bool(view.gist_event_ids) if eligible else None
    )
    metadata["post_draft_recovery_allocation"] = copy.deepcopy(receipt)
    return metadata
