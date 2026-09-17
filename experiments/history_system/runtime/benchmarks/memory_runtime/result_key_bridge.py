"""Default-off bridge for generated result keys required by another tool.

The frozen equality bridge remains the first selector.  When it has no scalar
equality candidate, this policy may copy one top-level result scalar whose
exact key is a required argument of another tool.  The successful source must
still be one complete event with exactly one call and one result.
"""

from __future__ import annotations

from typing import Any

from benchmarks.memory_runtime.dependency_packet import _tool_observations
from benchmarks.memory_runtime.observed_entity_slot import _source_message_sha256
from benchmarks.memory_runtime.same_event_bridge_only import (
    SAME_EVENT_BRIDGE_ONLY_POLICY,
    SameEventBridgeOnlyS0Controller,
)
from benchmarks.memory_runtime.same_event_reference import (
    SameEventReference,
    SameEventReferenceSelection,
    _value_sha256,
    select_same_event_reference,
)
from history_memory.events import EventStore


RESULT_KEY_BRIDGE_POLICY = "same-complete-event-result-key-bridge-v1"
RESULT_KEY_BRIDGE_VERSION = "a-native-same-complete-event-result-key-bridge-v1"
_DIRECT_SELECTED_STATUS = "selected_direct_result_key_required_consumer"


def select_result_key_bridge_reference(
    store: EventStore,
    tools: tuple[dict[str, Any], ...],
    *,
    raw_source_indices: tuple[int, ...],
) -> SameEventReferenceSelection:
    """Prefer equality; otherwise select one exact result-key dependency."""

    equality = select_same_event_reference(
        store,
        tools,
        raw_source_indices=raw_source_indices,
    )
    if (
        equality.reference is None
        and not equality.current_user_override_fields
        and equality.base.current_user_override_fields
    ):
        equality = SameEventReferenceSelection(
            equality.base,
            None,
            equality.status,
            equality.candidate_pair_count,
            equality.single_call_count,
            equality.single_result_count,
            equality.base.current_user_override_fields,
        )
    if equality.reference is not None:
        return equality
    if equality.status != "no_unique_same_type_scalar_pair":
        return equality

    slots = equality.base.slots
    if len(slots) != 1:
        return SameEventReferenceSelection(
            equality.base,
            None,
            (
                "no_direct_result_key_required_consumer"
                if not slots
                else "ambiguous_direct_result_key_required_consumers"
            ),
            equality.candidate_pair_count,
            equality.single_call_count,
            equality.single_result_count,
            equality.current_user_override_fields,
        )

    observations = [
        observation
        for observation in _tool_observations(store)
        if observation["event_id"] == equality.base.event_id
        and observation["result_source_index"] == equality.base.result_source_index
    ]
    if len(observations) != 1:
        return SameEventReferenceSelection(
            equality.base,
            None,
            "direct_result_observation_not_unique",
            equality.candidate_pair_count,
            equality.single_call_count,
            equality.single_result_count,
            equality.current_user_override_fields,
        )

    slot = slots[0]
    observation = observations[0]
    reference = SameEventReference(
        field=slot.field,
        value=slot.value,
        required_consumer_tools=slot.required_consumer_tools,
        result_field=slot.field,
        event_id=str(equality.base.event_id),
        event_source_indices=tuple(equality.base.event_source_indices),
        call_source_index=int(observation["call_source_index"]),
        result_source_index=int(observation["result_source_index"]),
        call_source_message_sha256=_source_message_sha256(
            store, int(observation["call_source_index"])
        ),
        result_source_message_sha256=_source_message_sha256(
            store, int(observation["result_source_index"])
        ),
        producer_tool=str(observation["tool"]),
        value_sha256=_value_sha256(slot.value),
    )
    return SameEventReferenceSelection(
        equality.base,
        reference,
        _DIRECT_SELECTED_STATUS,
        equality.candidate_pair_count,
        equality.single_call_count,
        equality.single_result_count,
        equality.current_user_override_fields,
    )


class ResultKeyBridgeS0Controller(SameEventBridgeOnlyS0Controller):
    """Add the direct result-key fallback behind an independent policy flag."""

    def __init__(
        self,
        *args: Any,
        result_key_bridge_policy: str,
        **kwargs: Any,
    ) -> None:
        if result_key_bridge_policy != RESULT_KEY_BRIDGE_POLICY:
            raise ValueError("Unknown result-key bridge policy")
        super().__init__(
            *args,
            same_event_bridge_only_policy=SAME_EVENT_BRIDGE_ONLY_POLICY,
            **kwargs,
        )
        self.observed_entity_slot_policy = result_key_bridge_policy
        self.same_event_bridge_only_policy = result_key_bridge_policy
        self.same_event_bridge_only_version = RESULT_KEY_BRIDGE_VERSION

    def _select_bridge_reference(
        self,
        store: EventStore,
        tools: tuple[dict[str, Any], ...],
        *,
        raw_source_indices: tuple[int, ...],
    ) -> SameEventReferenceSelection:
        return select_result_key_bridge_reference(
            store,
            tools,
            raw_source_indices=raw_source_indices,
        )

    def _bridge_receipt_semantics(
        self, selection: SameEventReferenceSelection
    ) -> dict[str, Any]:
        direct = selection.status == _DIRECT_SELECTED_STATUS
        equality = selection.reference is not None and not direct
        direct_evaluated = selection.status in {
            _DIRECT_SELECTED_STATUS,
            "no_direct_result_key_required_consumer",
            "ambiguous_direct_result_key_required_consumers",
            "direct_result_observation_not_unique",
        }
        return {
            "relation": (
                "direct_result_key_required_consumer"
                if direct
                else "same_type_scalar_equality"
                if equality
                else None
            ),
            "single_complete_event_only": True,
            "same_type_scalar_equality_required": equality,
            "direct_result_key_required_consumer_schema": direct,
            "global_alias_table_used": False,
            "cross_event_join_used": False,
            "call_argument_field_relation_used": equality,
            "call_argument_is_value_source": False,
            "result_scalar_is_value_source": True,
            "direct_result_key_candidate_count": (
                len(selection.base.slots) if direct_evaluated else 0
            ),
        }

    def _bridge_reference_receipt_fields(
        self, reference: SameEventReference
    ) -> dict[str, Any]:
        if reference.field == reference.result_field:
            return {
                "consumer_argument_field": reference.field,
                "result_field": reference.result_field,
                "result_path": [reference.result_field],
            }
        return super()._bridge_reference_receipt_fields(reference)


__all__ = [
    "RESULT_KEY_BRIDGE_POLICY",
    "RESULT_KEY_BRIDGE_VERSION",
    "ResultKeyBridgeS0Controller",
    "select_result_key_bridge_reference",
]
