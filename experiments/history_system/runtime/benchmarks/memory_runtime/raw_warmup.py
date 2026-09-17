"""Retain full history as raw when it fits the declared hybrid history budget."""
from __future__ import annotations

import copy

from .event_native_s0_policy import PreparedEventNativeS0


RAW_WARMUP_POLICY = "full-history-if-fits-v1"


def apply_raw_warmup(controller, prepared, store, tools, *, ratio, max_new_tokens):
    metadata = copy.deepcopy(prepared.metadata)
    budget = min(controller.policy_config.history_budget_bytes,
                 controller.policy_config.workspace_budget_bytes)
    full_bytes = metadata["same_prefix_full_reference"]["full_history_bytes"]
    receipt = {"policy":RAW_WARMUP_POLICY, "full_history_bytes":full_bytes,
               "history_budget_bytes":budget, "status":"full_history_over_budget",
               "eligible_extraction_preserved":True, "threshold_source":"same-prefix native Full renderer"}
    if full_bytes > budget:
        metadata["raw_warmup"] = receipt
        return PreparedEventNativeS0(memory=prepared.memory, metadata=metadata,
            eligible_chunks=prepared.eligible_chunks, _owner=controller._owner,
            _session_id=store.session_id, _decision_key=prepared._decision_key)
    all_raw = tuple(event.event_id for event in store.events)
    eligible = metadata["eligible_extraction"]["eligible_event_ids"]
    measured = controller._try_measure(store, tools, all_raw,
        prepared.memory.view.mandatory_raw_event_ids, (), eligible,
        metadata["common_raw_prompt_tokens"], max_new_tokens, derived_messages=())
    if measured is None or measured.reasons:
        receipt.update(status="full_history_native_capacity_failure",
                       reasons=["packing_budget_exceeded"] if measured is None else list(measured.reasons))
        metadata["raw_warmup"] = receipt
        return PreparedEventNativeS0(memory=prepared.memory, metadata=metadata,
            eligible_chunks=prepared.eligible_chunks, _owner=controller._owner,
            _session_id=store.session_id, _decision_key=prepared._decision_key)
    eligible_indices = metadata["eligible_extraction"]["eligible_source_indices"]
    coverage, packing_fragments, retained_fragments = controller._coverage(
        store, eligible, frozenset(eligible_indices), prepared.eligible_chunks, measured.memory)
    receipt.update(status="full_raw_admitted", active_history_bytes=measured.per_ratio[str(ratio)]["history_bytes"],
        previous_active_history_bytes=metadata["actual_history_bytes"],
        dropped_gist_event_ids=list(prepared.memory.view.gist_event_ids),
        dropped_derived_workspace_count=metadata.get("derived_workspace_prefix_message_count", len(metadata.get("derived_workspace_prefix_messages", []))))
    metadata.update(
        raw_warmup=receipt, raw_prompt_tokens=measured.raw_prompt_tokens,
        actual_raw_history_tokens=measured.raw_history_tokens,
        actual_history_bytes=measured.per_ratio[str(ratio)]["history_bytes"],
        per_ratio=copy.deepcopy(measured.per_ratio), logical_sequence_tokens=measured.logical_sequence_tokens,
        raw_source_indices=list(measured.memory.raw_source_indices), raw_event_ids=list(all_raw),
        gist_event_ids=[], omitted_event_ids=[], evidence_event_ids=[],
        raw_evidence_event_ids=[], selected_event_ids=list(all_raw),
        selected_source_indices=list(measured.memory.raw_source_indices),
        retrieved_event_ids=[], retained_event_ids=[], recency_selected_event_ids=[],
        derived_workspace_prefix_messages=[], derived_workspace_prefix_message_count=0,
        derived_workspace_source_indices=[], source_coverage=coverage,
        actual_gist_tokens=0, history_packing_fragments=packing_fragments,
        retained_history_packing_fragments=retained_fragments,
        full_source_coverage=coverage["complete_history_coverage"],
        gist_refilled_event_ids=[], skipped_gist_refill_events=[],
        min_gist_reservation_required=False, min_gist_reservation_met=None,
        compression_ratio=controller._compression_ratio(metadata["same_prefix_full_reference"], measured, coverage, ratio),
    )
    metadata["eligible_extraction"].update(retained_event_ids=[], retained_chunk_count=0,
        retained_presented_encoder_tokens=0, omitted_active_gist_event_ids=list(eligible),
        raw_gist_overlap_event_ids=[])
    # Selection and cue receipts describe the displaced S0 proposal, not the final view.
    for key in (
        "raw_reserve",
        "failed_operation_cue",
        "stalled_operation_ledger",
        "compact_first_failure_cue",
        "same_event_reference",
        "source_needs",
        "gist_reservation",
    ):
        if key in metadata:
            metadata[key] = {"status":"replaced_by_full_raw_warmup", "proposal_receipt":metadata[key]}
    return PreparedEventNativeS0(memory=measured.memory, metadata=metadata,
        eligible_chunks=prepared.eligible_chunks, _owner=controller._owner,
        _session_id=store.session_id, _decision_key=prepared._decision_key)


def wrap_with_raw_warmup(controller):
    original = controller._prepare_view

    def prepare(store, tools, *, ratio, max_new_tokens, decision_key, decision_index):
        prepared = original(store, tools, ratio=ratio, max_new_tokens=max_new_tokens,
                            decision_key=decision_key, decision_index=decision_index)
        return apply_raw_warmup(controller, prepared, store, tools, ratio=ratio, max_new_tokens=max_new_tokens)

    controller._prepare_view = prepare
    controller.raw_warmup_policy = RAW_WARMUP_POLICY
    return controller
