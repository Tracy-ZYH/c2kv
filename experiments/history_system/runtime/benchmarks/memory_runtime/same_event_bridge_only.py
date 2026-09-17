"""Default-off SAME bridge that leaves non-bridge S0 inputs unchanged.

The source/event selector is the validated single-complete-event relation from
``same_event_reference``.  Unlike the earlier joint candidate, this controller
does not render the inherited missing-required result fields.  It emits only
the exact call-argument field whose value is proven by one same-typed result
scalar in that same successful event.
"""

from __future__ import annotations

import copy
from typing import Any

from benchmarks.memory_runtime.event_native_s0_policy import (
    EventNativeS0Controller,
    PreparedEventNativeS0,
)
from benchmarks.memory_runtime.observed_entity_slot import ObservedEntitySlot, slot_message
from benchmarks.memory_runtime.same_event_reference import (
    SameEventReference,
    SameEventReferenceSelection,
    select_same_event_reference,
)
from history_memory.events import EventStore


SAME_EVENT_BRIDGE_ONLY_POLICY = "same-complete-event-reference-bridge-only-v1"
SAME_EVENT_BRIDGE_ONLY_VERSION = "a-native-same-complete-event-reference-bridge-only-v1"


class SameEventBridgeOnlyS0Controller(EventNativeS0Controller):
    """Apply one exact SAME relation directly to an original S0 preparation."""

    def __init__(
        self,
        *args: Any,
        same_event_bridge_only_policy: str,
        **kwargs: Any,
    ) -> None:
        if same_event_bridge_only_policy != SAME_EVENT_BRIDGE_ONLY_POLICY:
            raise ValueError("Unknown same-event bridge-only policy")
        super().__init__(*args, **kwargs)
        self.observed_entity_slot_policy = same_event_bridge_only_policy
        self.same_event_bridge_only_policy = same_event_bridge_only_policy
        self.same_event_bridge_only_version = SAME_EVENT_BRIDGE_ONLY_VERSION

    def _select_bridge_reference(
        self,
        store: EventStore,
        tools: tuple[dict[str, Any], ...],
        *,
        raw_source_indices: tuple[int, ...],
    ) -> SameEventReferenceSelection:
        """Select the exact equality bridge used by the frozen v1 policy."""

        return select_same_event_reference(
            store,
            tools,
            raw_source_indices=raw_source_indices,
        )

    def _bridge_receipt_semantics(
        self, selection: SameEventReferenceSelection
    ) -> dict[str, Any]:
        """Return relation claims for the frozen equality selector."""

        return {
            "single_complete_event_only": True,
            "same_type_scalar_equality_required": True,
            "global_alias_table_used": False,
            "cross_event_join_used": False,
            "call_argument_is_value_source": False,
            "result_scalar_is_value_source": True,
        }

    def _bridge_reference_receipt_fields(
        self, reference: SameEventReference
    ) -> dict[str, Any]:
        """Describe the call-argument/result equality relation."""

        return {
            "call_argument_field": reference.field,
            "result_field": reference.result_field,
            "result_path": [reference.result_field],
        }

    def _prepare_view(
        self,
        store: EventStore,
        tools: tuple[dict[str, Any], ...],
        *,
        ratio: int,
        max_new_tokens: int,
        decision_key: str,
        decision_index: int,
    ) -> PreparedEventNativeS0:
        prepared = super()._prepare_view(
            store,
            tools,
            ratio=ratio,
            max_new_tokens=max_new_tokens,
            decision_key=decision_key,
            decision_index=decision_index,
        )
        return self._apply_same_event_bridge(
            prepared,
            store,
            tools,
            ratio=ratio,
            max_new_tokens=max_new_tokens,
        )

    def _apply_same_event_bridge(
        self,
        prepared: PreparedEventNativeS0,
        store: EventStore,
        tools: tuple[dict[str, Any], ...],
        *,
        ratio: int,
        max_new_tokens: int,
    ) -> PreparedEventNativeS0:
        selection = self._select_bridge_reference(
            store,
            tools,
            raw_source_indices=prepared.memory.raw_source_indices,
        )
        reference = selection.reference
        receipt: dict[str, Any] = {
            "version": self.same_event_bridge_only_version,
            "policy": self.same_event_bridge_only_policy,
            "selection_status": selection.status,
            "status": selection.status,
            "selection_anchor_policy": selection.base.status,
            "base_source_event_id": selection.base.event_id,
            "base_result_source_index": selection.base.result_source_index,
            "candidate_pair_count": selection.candidate_pair_count,
            "event_call_count": selection.single_call_count,
            "event_result_count": selection.single_result_count,
            "current_user_override_fields": list(
                selection.current_user_override_fields
            ),
            "inherited_missing_required_fields_rendered": False,
            "no_bridge_input_semantics": "exact-original-s0-packed-input",
            "values_serialized_in_receipt": False,
            "whole_field_only": True,
            "truncation_allowed": False,
            "uses_gold_future_or_hidden_state": False,
            "incremental_raw_tokens": 0,
            "extra_bytes": 0,
        }
        receipt.update(self._bridge_receipt_semantics(selection))
        metadata = copy.deepcopy(prepared.metadata)
        if reference is None:
            metadata["same_event_reference"] = receipt
            return PreparedEventNativeS0(
                memory=prepared.memory,
                metadata=metadata,
                eligible_chunks=prepared.eligible_chunks,
                _owner=self._owner,
                _session_id=store.session_id,
                _decision_key=prepared._decision_key,
            )

        receipt.update(
            source_event_id=reference.event_id,
            source_event_source_indices=list(reference.event_source_indices),
            call_source_index=reference.call_source_index,
            result_source_index=reference.result_source_index,
            call_source_message_sha256=reference.call_source_message_sha256,
            result_source_message_sha256=reference.result_source_message_sha256,
            producer_tool=reference.producer_tool,
            value_sha256=reference.value_sha256,
            value_type=type(reference.value).__name__,
            required_consumer_tools=list(reference.required_consumer_tools),
            source_result_present_in_raw_workspace=False,
        )
        receipt.update(self._bridge_reference_receipt_fields(reference))
        message = slot_message([ObservedEntitySlot(reference.field, reference.value)])
        assert message is not None
        base_derived = tuple(
            prepared.metadata.get("derived_workspace_prefix_messages") or ()
        )
        measure = self._try_measure(
            store,
            tools,
            prepared.memory.view.raw_event_ids,
            prepared.memory.view.mandatory_raw_event_ids,
            prepared.memory.view.gist_event_ids,
            prepared.metadata["eligible_extraction"]["eligible_event_ids"],
            prepared.metadata["common_raw_prompt_tokens"],
            max_new_tokens,
            derived_messages=(message, *base_derived),
        )
        if measure is None or measure.reasons:
            receipt.update(
                status="whole_field_over_budget",
                admission_failures=(
                    ["packing_budget_exceeded"]
                    if measure is None
                    else list(measure.reasons)
                ),
            )
            metadata["same_event_reference"] = receipt
            return PreparedEventNativeS0(
                memory=prepared.memory,
                metadata=metadata,
                eligible_chunks=prepared.eligible_chunks,
                _owner=self._owner,
                _session_id=store.session_id,
                _decision_key=prepared._decision_key,
            )

        incremental = measure.raw_prompt_tokens - prepared.metadata["raw_prompt_tokens"]
        if incremental <= 0:
            raise RuntimeError("Same-event bridge added no workspace tokens")
        receipt.update(
            status="admitted",
            admitted_field_name=reference.field,
            incremental_raw_tokens=incremental,
            extra_bytes=incremental * self.kv_bytes_per_token,
            candidate_active_history_bytes=measure.per_ratio[str(ratio)][
                "history_bytes"
            ],
            budget_bytes=min(
                self.policy_config.history_budget_bytes,
                self.policy_config.workspace_budget_bytes,
            ),
        )
        metadata.update(
            raw_prompt_tokens=measure.raw_prompt_tokens,
            actual_raw_history_tokens=measure.raw_history_tokens,
            actual_history_bytes=measure.per_ratio[str(ratio)]["history_bytes"],
            logical_sequence_tokens=measure.logical_sequence_tokens,
            per_ratio=copy.deepcopy(measure.per_ratio),
        )
        metadata["same_event_reference"] = receipt
        metadata["derived_workspace_prefix_message_count"] = len(base_derived) + 1
        metadata["compression_ratio"] = self._compression_ratio(
            metadata["same_prefix_full_reference"],
            measure,
            metadata["source_coverage"],
            ratio,
        )
        return PreparedEventNativeS0(
            memory=measure.memory,
            metadata=metadata,
            eligible_chunks=prepared.eligible_chunks,
            _owner=self._owner,
            _session_id=store.session_id,
            _decision_key=prepared._decision_key,
        )


__all__ = [
    "SAME_EVENT_BRIDGE_ONLY_POLICY",
    "SAME_EVENT_BRIDGE_ONLY_VERSION",
    "SameEventBridgeOnlyS0Controller",
]
