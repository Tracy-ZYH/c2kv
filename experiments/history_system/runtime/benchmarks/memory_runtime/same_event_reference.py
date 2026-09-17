"""Default-off bridge for exact reference names within one complete tool event.

The existing missing-required-reference selector owns the source row.  This
module may add one field name from that row's single tool call when exactly one
same-typed scalar in the paired result proves the value relation.  It never
joins values across events and never uses an alias table.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from benchmarks.memory_runtime.dependency_packet import _tool_observations
from benchmarks.memory_runtime.event_native_s0_policy import PreparedEventNativeS0
from benchmarks.memory_runtime.observed_entity_slot import (
    ObservedEntitySlot,
    _complete_scalar,
    _source_message_sha256,
    slot_message,
)
from benchmarks.memory_runtime.observed_entity_slot_revision import (
    MISSING_REQUIRED_REFERENCE_POLICY,
    RevisionObservedEntitySlotS0Controller,
    RevisionSelection,
    RevisionSlot,
    _current_user_overrides,
    _tool_required_consumers,
    select_revision_slots,
)
from history_memory.events import EventStore


SAME_EVENT_REFERENCE_POLICY = "same-complete-event-reference-v1"
SAME_EVENT_REFERENCE_VERSION = "a-native-same-complete-event-reference-v1"


def _value_sha256(value: Any) -> str:
    # Match the earlier task120 evidence audit so the source value can be
    # joined by receipt without exposing it.  The receipt separately records
    # the exact scalar type required by the equality check.
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SameEventReference:
    field: str
    value: str | int | float | bool
    required_consumer_tools: tuple[str, ...]
    result_field: str
    event_id: str
    event_source_indices: tuple[int, ...]
    call_source_index: int
    result_source_index: int
    call_source_message_sha256: str
    result_source_message_sha256: str
    producer_tool: str
    value_sha256: str


@dataclass(frozen=True)
class SameEventReferenceSelection:
    base: RevisionSelection
    reference: SameEventReference | None
    status: str
    candidate_pair_count: int
    single_call_count: int | None
    single_result_count: int | None
    current_user_override_fields: tuple[str, ...] = ()


def _event_call_result_counts(
    store: EventStore, event_id: str
) -> tuple[int, int]:
    messages = [message.to_dict() for message in store.event_messages(event_id)]
    call_count = sum(len(message.get("tool_calls") or ()) for message in messages)
    result_count = sum(message.get("role") == "tool" for message in messages)
    return call_count, result_count


def select_same_event_reference(
    store: EventStore,
    tools: Sequence[Mapping[str, Any]],
    *,
    raw_source_indices: Sequence[int],
) -> SameEventReferenceSelection:
    """Supplement the row selected by missing-required-reference, or abstain."""

    base = select_revision_slots(
        store,
        tools,
        raw_source_indices=raw_source_indices,
        policy=MISSING_REQUIRED_REFERENCE_POLICY,
    )
    if base.event_id is None or base.result_source_index is None or not base.slots:
        return SameEventReferenceSelection(base, None, "base_row_not_selected", 0, None, None)
    if base.result_source_index in set(raw_source_indices):
        return SameEventReferenceSelection(base, None, "result_already_raw_visible", 0, None, None)

    event = store.event(base.event_id)
    call_count, result_count = _event_call_result_counts(store, base.event_id)
    if not event.complete:
        return SameEventReferenceSelection(
            base, None, "event_incomplete", 0, call_count, result_count
        )
    if call_count != 1 or result_count != 1:
        return SameEventReferenceSelection(
            base, None, "not_single_call_single_result", 0, call_count, result_count
        )

    observations = [
        observation
        for observation in _tool_observations(store)
        if observation["event_id"] == base.event_id
        and observation["result_source_index"] == base.result_source_index
    ]
    if len(observations) != 1:
        return SameEventReferenceSelection(
            base, None, "observation_pair_not_unique", 0, call_count, result_count
        )
    observation = observations[0]
    arguments = observation["arguments"]
    result = observation["result"]
    if not isinstance(arguments, Mapping) or not isinstance(result, Mapping):
        return SameEventReferenceSelection(
            base, None, "call_or_result_not_object", 0, call_count, result_count
        )

    required_consumers = _tool_required_consumers(tools)
    override_fields = _current_user_overrides(store, tuple(required_consumers))
    producer_tool = str(observation["tool"])
    candidate_pairs: list[tuple[str, Any, tuple[str, ...], str]] = []
    bridge_override_fields: set[str] = set()
    for argument_field, argument_value in arguments.items():
        if not isinstance(argument_field, str) or argument_field in result:
            continue
        consumers = tuple(
            name
            for name in required_consumers.get(argument_field, ())
            if name != producer_tool
        )
        if not consumers:
            continue
        if argument_field.casefold() in override_fields:
            bridge_override_fields.add(argument_field)
            continue
        if not _complete_scalar(argument_value):
            continue
        for result_field, result_value in result.items():
            if (
                isinstance(result_field, str)
                and _complete_scalar(result_value)
                and type(argument_value) is type(result_value)
                and argument_value == result_value
            ):
                # The result is the value source.  The call argument supplies
                # only the exact consumer field name relation.
                candidate_pairs.append(
                    (argument_field, result_value, consumers, result_field)
                )

    if len(candidate_pairs) != 1:
        return SameEventReferenceSelection(
            base,
            None,
            (
                "current_user_binding_precedence"
                if bridge_override_fields and not candidate_pairs
                else "no_unique_same_type_scalar_pair"
                if not candidate_pairs
                else "ambiguous_scalar_pairs"
            ),
            len(candidate_pairs),
            call_count,
            result_count,
            tuple(sorted(bridge_override_fields, key=lambda value: (value.casefold(), value))),
        )

    field, result_value, consumers, result_field = candidate_pairs[0]
    reference = SameEventReference(
        field=field,
        value=result_value,
        required_consumer_tools=consumers,
        result_field=result_field,
        event_id=base.event_id,
        event_source_indices=tuple(event.source_indices),
        call_source_index=int(observation["call_source_index"]),
        result_source_index=int(observation["result_source_index"]),
        call_source_message_sha256=_source_message_sha256(
            store, int(observation["call_source_index"])
        ),
        result_source_message_sha256=_source_message_sha256(
            store, int(observation["result_source_index"])
        ),
        producer_tool=producer_tool,
        value_sha256=_value_sha256(result_value),
    )
    return SameEventReferenceSelection(
        base,
        reference,
        "selected",
        1,
        call_count,
        result_count,
        tuple(sorted(bridge_override_fields, key=lambda value: (value.casefold(), value))),
    )


class SameEventReferenceS0Controller(RevisionObservedEntitySlotS0Controller):
    """Add one provenance-bound bridge after the existing row admission."""

    def __init__(
        self,
        *args: Any,
        same_event_reference_policy: str,
        **kwargs: Any,
    ) -> None:
        if same_event_reference_policy != SAME_EVENT_REFERENCE_POLICY:
            raise ValueError("Unknown same-event reference policy")
        super().__init__(
            *args,
            observed_entity_slot_revision_policy=MISSING_REQUIRED_REFERENCE_POLICY,
            **kwargs,
        )
        self.observed_entity_slot_policy = same_event_reference_policy
        self.same_event_reference_policy = same_event_reference_policy

    def _apply_observed_entity_slots(
        self,
        prepared: PreparedEventNativeS0,
        store: EventStore,
        tools: tuple[dict[str, Any], ...],
        *,
        ratio: int,
        max_new_tokens: int,
    ) -> PreparedEventNativeS0:
        base_selected = super()._apply_observed_entity_slots(
            prepared,
            store,
            tools,
            ratio=ratio,
            max_new_tokens=max_new_tokens,
        )
        selection = select_same_event_reference(
            store,
            tools,
            raw_source_indices=prepared.memory.raw_source_indices,
        )
        reference = selection.reference
        receipt: dict[str, Any] = {
            "version": SAME_EVENT_REFERENCE_VERSION,
            "policy": self.same_event_reference_policy,
            "selection_status": selection.status,
            "status": selection.status,
            "base_policy": MISSING_REQUIRED_REFERENCE_POLICY,
            "base_source_event_id": selection.base.event_id,
            "base_result_source_index": selection.base.result_source_index,
            "candidate_pair_count": selection.candidate_pair_count,
            "event_call_count": selection.single_call_count,
            "event_result_count": selection.single_result_count,
            "current_user_override_fields": list(
                selection.current_user_override_fields
            ),
            "single_complete_event_only": True,
            "same_type_scalar_equality_required": True,
            "global_alias_table_used": False,
            "cross_event_join_used": False,
            "call_argument_is_value_source": False,
            "result_scalar_is_value_source": True,
            "values_serialized_in_receipt": False,
            "whole_field_only": True,
            "truncation_allowed": False,
            "uses_gold_future_or_hidden_state": False,
            "incremental_raw_tokens": 0,
            "extra_bytes": 0,
        }
        metadata = copy.deepcopy(base_selected.metadata)
        if reference is None:
            metadata["same_event_reference"] = receipt
            return PreparedEventNativeS0(
                memory=base_selected.memory,
                metadata=metadata,
                eligible_chunks=base_selected.eligible_chunks,
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
            call_argument_field=reference.field,
            result_field=reference.result_field,
            result_path=[reference.result_field],
            value_sha256=reference.value_sha256,
            value_type=type(reference.value).__name__,
            required_consumer_tools=list(reference.required_consumer_tools),
            source_result_present_in_raw_workspace=False,
        )

        base_receipt = base_selected.metadata["observed_entity_slot_revision"]
        admitted_names = set(base_receipt["admitted_field_names"])
        admitted_base = [
            slot for slot in selection.base.slots if slot.field in admitted_names
        ]
        if not admitted_base:
            receipt["status"] = "base_fields_not_admitted"
            metadata["same_event_reference"] = receipt
            return PreparedEventNativeS0(
                memory=base_selected.memory,
                metadata=metadata,
                eligible_chunks=base_selected.eligible_chunks,
                _owner=self._owner,
                _session_id=store.session_id,
                _decision_key=prepared._decision_key,
            )

        combined = sorted(
            (
                *admitted_base,
                RevisionSlot(
                    reference.field,
                    reference.value,
                    reference.required_consumer_tools,
                ),
            ),
            key=lambda slot: (slot.field.casefold(), slot.field),
        )
        message = slot_message(
            [ObservedEntitySlot(slot.field, slot.value) for slot in combined]
        )
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
                memory=base_selected.memory,
                metadata=metadata,
                eligible_chunks=base_selected.eligible_chunks,
                _owner=self._owner,
                _session_id=store.session_id,
                _decision_key=prepared._decision_key,
            )

        incremental_total = (
            measure.raw_prompt_tokens - prepared.metadata["raw_prompt_tokens"]
        )
        incremental_bridge = (
            measure.raw_prompt_tokens - base_selected.metadata["raw_prompt_tokens"]
        )
        if incremental_bridge <= 0:
            raise RuntimeError("Same-event reference added no workspace tokens")
        receipt.update(
            status="admitted",
            admitted_field_name=reference.field,
            combined_admitted_field_names=[slot.field for slot in combined],
            incremental_raw_tokens=incremental_bridge,
            total_slot_incremental_raw_tokens=incremental_total,
            extra_bytes=incremental_bridge * self.kv_bytes_per_token,
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
    "SAME_EVENT_REFERENCE_POLICY",
    "SAME_EVENT_REFERENCE_VERSION",
    "SameEventReference",
    "SameEventReferenceS0Controller",
    "SameEventReferenceSelection",
    "select_same_event_reference",
]
