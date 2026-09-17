"""Offline M5 revision: preserve missing, schema-traceable reference fields.

This module is intentionally outside the production runtime.  It subclasses the
frozen v1 candidate for CPU replay without changing that candidate or its package.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from benchmarks.memory_runtime.dependency_packet import (
    _json_bindings,
    _text_bindings,
    _tool_observations,
    _tool_slots,
)
from benchmarks.memory_runtime.failed_operation import failed as result_reports_failure
from benchmarks.memory_runtime.observed_entity_slot import (
    OBSERVED_ENTITY_SLOT_POLICY,
    ObservedEntitySlot,
    ObservedEntitySlotS0Controller,
    _complete_scalar,
    _source_message_sha256,
    slot_message,
)
from benchmarks.memory_runtime.event_native_s0_policy import PreparedEventNativeS0
from history_memory.events import EventStore


MISSING_SOURCE_ONLY_POLICY = "native-observed-entity-slot-missing-result-v1"
MISSING_REQUIRED_REFERENCE_POLICY = (
    "native-observed-entity-slot-missing-required-reference-v1"
)
REVISION_VERSION = "a-native-observed-entity-slot-missing-source-revision-v1"
REVISION_POLICIES = frozenset(
    {MISSING_SOURCE_ONLY_POLICY, MISSING_REQUIRED_REFERENCE_POLICY}
)


@dataclass(frozen=True)
class RevisionSlot:
    field: str
    value: str | int | float | bool
    required_consumer_tools: tuple[str, ...]


@dataclass(frozen=True)
class RevisionSelection:
    event_id: str | None
    event_source_indices: tuple[int, ...]
    result_source_index: int | None
    source_message_sha256: str | None
    producer_tool: str | None
    slots: tuple[RevisionSlot, ...]
    current_user_override_fields: tuple[str, ...]
    visible_result_observations_skipped: int
    non_reference_field_names_skipped: tuple[str, ...]
    status: str


def _tool_required_consumers(
    tools: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[str, ...]]:
    """Map exact schema field names to tools that require them as inputs."""

    consumers: dict[str, set[str]] = {}
    for raw_tool in tools:
        function = raw_tool.get("function", raw_tool)
        if not isinstance(function, Mapping):
            continue
        name = function.get("name")
        parameters = function.get("parameters")
        if not isinstance(name, str) or not isinstance(parameters, Mapping):
            continue
        properties = parameters.get("properties")
        required = parameters.get("required", ())
        if not isinstance(properties, Mapping) or not isinstance(required, Sequence):
            continue
        for field in required:
            if isinstance(field, str) and field in properties:
                consumers.setdefault(field, set()).add(name)
    return {
        field: tuple(sorted(names, key=lambda value: (value.casefold(), value)))
        for field, names in consumers.items()
    }


def _current_user_overrides(
    store: EventStore, slot_names: Sequence[str]
) -> set[str]:
    users = [event for event in store.events if event.kind == "user"]
    current_user = users[-1] if users else None
    if current_user is None:
        return set()
    source_index = current_user.source_indices[0]
    content = store.messages[source_index].to_dict().get("content")
    if not isinstance(content, str):
        return set()
    bindings = _json_bindings(content, slot_names) + _text_bindings(content, slot_names)
    return {
        binding["field"].casefold()
        for binding in bindings
        if isinstance(binding.get("field"), str)
    }


def select_revision_slots(
    store: EventStore,
    tools: Sequence[Mapping[str, Any]],
    *,
    raw_source_indices: Sequence[int],
    policy: str,
) -> RevisionSelection:
    """Select only values whose complete result message is absent from raw input.

    The stricter policy additionally requires an exact-name, required input slot
    on a different tool.  This is a schema relation, not an identifier-name list.
    """

    if policy not in REVISION_POLICIES:
        raise ValueError("Unknown observed entity slot revision policy")
    slot_names, _ = _tool_slots(tools)
    folded_names = {name.casefold() for name in slot_names}
    overrides = _current_user_overrides(store, slot_names)
    raw_indices = set(raw_source_indices)
    required_consumers = _tool_required_consumers(tools)
    visible_skipped = 0
    non_reference_skipped: set[str] = set()

    for observation in reversed(list(_tool_observations(store))):
        result = observation["result"]
        if result_reports_failure(result) or not isinstance(result, Mapping):
            continue
        matching: list[RevisionSlot] = []
        for field, value in result.items():
            if (
                not isinstance(field, str)
                or field.casefold() not in folded_names
                or not _complete_scalar(value)
            ):
                continue
            consumers = tuple(
                name
                for name in required_consumers.get(field, ())
                if name != str(observation["tool"])
            )
            if policy == MISSING_REQUIRED_REFERENCE_POLICY and not consumers:
                non_reference_skipped.add(field)
                continue
            matching.append(RevisionSlot(field, value, consumers))
        if not matching:
            continue
        matching.sort(key=lambda slot: (slot.field.casefold(), slot.field))
        selected = tuple(
            slot for slot in matching if slot.field.casefold() not in overrides
        )
        if observation["result_source_index"] in raw_indices:
            # The newest successful source row wins before visibility gating.
            # Never fall back to an older value merely because the newest row is
            # already present in the raw workspace.
            visible_skipped += 1
            return RevisionSelection(
                event_id=observation["event_id"],
                event_source_indices=tuple(observation["event_source_indices"]),
                result_source_index=observation["result_source_index"],
                source_message_sha256=_source_message_sha256(
                    store, observation["result_source_index"]
                ),
                producer_tool=str(observation["tool"]),
                slots=(),
                current_user_override_fields=tuple(
                    slot.field for slot in matching if slot.field.casefold() in overrides
                ),
                visible_result_observations_skipped=visible_skipped,
                non_reference_field_names_skipped=tuple(
                    sorted(
                        non_reference_skipped,
                        key=lambda value: (value.casefold(), value),
                    )
                ),
                status="latest_matching_result_already_raw_visible",
            )
        return RevisionSelection(
            event_id=observation["event_id"],
            event_source_indices=tuple(observation["event_source_indices"]),
            result_source_index=observation["result_source_index"],
            source_message_sha256=_source_message_sha256(
                store, observation["result_source_index"]
            ),
            producer_tool=str(observation["tool"]),
            slots=selected,
            current_user_override_fields=tuple(
                slot.field for slot in matching if slot.field.casefold() in overrides
            ),
            visible_result_observations_skipped=visible_skipped,
            non_reference_field_names_skipped=tuple(
                sorted(non_reference_skipped, key=lambda value: (value.casefold(), value))
            ),
            status=("selected" if selected else "current_user_binding_precedence"),
        )
    return RevisionSelection(
        event_id=None,
        event_source_indices=(),
        result_source_index=None,
        source_message_sha256=None,
        producer_tool=None,
        slots=(),
        current_user_override_fields=(),
        visible_result_observations_skipped=visible_skipped,
        non_reference_field_names_skipped=tuple(
            sorted(non_reference_skipped, key=lambda value: (value.casefold(), value))
        ),
        status=(
            "matching_results_already_raw_visible"
            if visible_skipped and not non_reference_skipped
            else "no_missing_qualified_observation"
        ),
    )


class RevisionObservedEntitySlotS0Controller(ObservedEntitySlotS0Controller):
    """CPU-only candidate layered on the frozen v1 controller implementation."""

    def __init__(
        self,
        *args: Any,
        observed_entity_slot_revision_policy: str,
        **kwargs: Any,
    ) -> None:
        if observed_entity_slot_revision_policy not in REVISION_POLICIES:
            raise ValueError("Unknown observed entity slot revision policy")
        super().__init__(
            *args,
            observed_entity_slot_policy=OBSERVED_ENTITY_SLOT_POLICY,
            **kwargs,
        )
        self.observed_entity_slot_policy = observed_entity_slot_revision_policy
        self.observed_entity_slot_revision_policy = observed_entity_slot_revision_policy

    def _apply_observed_entity_slots(
        self,
        prepared: PreparedEventNativeS0,
        store: EventStore,
        tools: tuple[dict[str, Any], ...],
        *,
        ratio: int,
        max_new_tokens: int,
    ) -> PreparedEventNativeS0:
        selection = select_revision_slots(
            store,
            tools,
            raw_source_indices=prepared.memory.raw_source_indices,
            policy=self.observed_entity_slot_revision_policy,
        )
        base_metadata = prepared.metadata
        base_derived = tuple(base_metadata.get("derived_workspace_prefix_messages") or ())
        admitted: list[RevisionSlot] = []
        skipped: list[dict[str, Any]] = [
            {"field": field, "reason": "current_user_binding_precedence"}
            for field in selection.current_user_override_fields
        ]
        final_measure = None
        for slot in selection.slots:
            message = slot_message(
                [ObservedEntitySlot(item.field, item.value) for item in (*admitted, slot)]
            )
            assert message is not None
            candidate = self._try_measure(
                store,
                tools,
                prepared.memory.view.raw_event_ids,
                prepared.memory.view.mandatory_raw_event_ids,
                prepared.memory.view.gist_event_ids,
                base_metadata["eligible_extraction"]["eligible_event_ids"],
                base_metadata["common_raw_prompt_tokens"],
                max_new_tokens,
                derived_messages=(message, *base_derived),
            )
            if candidate is None or candidate.reasons:
                skipped.append(
                    {
                        "field": slot.field,
                        "reason": "whole_field_over_budget",
                        "admission_failures": (
                            ["packing_budget_exceeded"]
                            if candidate is None
                            else list(candidate.reasons)
                        ),
                    }
                )
                continue
            admitted.append(slot)
            final_measure = candidate

        receipt = {
            "version": REVISION_VERSION,
            "policy": self.observed_entity_slot_revision_policy,
            "selection_status": selection.status,
            "status": (
                "admitted"
                if admitted
                else "over_budget"
                if selection.slots
                else selection.status
            ),
            "source_event_id": selection.event_id,
            "source_event_source_indices": list(selection.event_source_indices),
            "result_source_index": selection.result_source_index,
            "source_result_present_in_raw_workspace": (
                selection.result_source_index in prepared.memory.raw_source_indices
                if selection.result_source_index is not None
                else None
            ),
            "source_message_role": (
                store.messages[selection.result_source_index].role
                if selection.result_source_index is not None
                else None
            ),
            "source_message_sha256": selection.source_message_sha256,
            "producer_tool": selection.producer_tool,
            "source_semantics": (
                "complete observed tool-result message is supplemented only when its "
                "result_source_index is absent from current raw_source_indices"
            ),
            "qualification_semantics": (
                "exact field-name match to a required input schema slot on a different tool"
                if self.observed_entity_slot_revision_policy
                == MISSING_REQUIRED_REFERENCE_POLICY
                else "exact field-name match to any visible tool input schema slot"
            ),
            "user_override_semantics": (
                "an explicit binding in the latest visible user source message has precedence"
            ),
            "eligible_field_names": [slot.field for slot in selection.slots],
            "required_consumer_tools_by_field": {
                slot.field: list(slot.required_consumer_tools) for slot in selection.slots
            },
            "admitted_field_names": [slot.field for slot in admitted],
            "visible_result_observations_skipped": (
                selection.visible_result_observations_skipped
            ),
            "non_reference_field_names_skipped": list(
                selection.non_reference_field_names_skipped
            ),
            "skipped_fields": skipped,
            "incremental_raw_tokens": 0,
            "extra_bytes": 0,
            "derived_workspace_message_count": int(bool(admitted)),
            "existing_derived_workspace_message_count": len(base_derived),
            "values_serialized_in_receipt": False,
            "whole_field_only": True,
            "truncation_allowed": False,
            "uses_gold_future_or_hidden_state": False,
        }
        metadata = copy.deepcopy(base_metadata)
        metadata["observed_entity_slot_revision"] = receipt
        if not admitted:
            return PreparedEventNativeS0(
                memory=prepared.memory,
                metadata=metadata,
                eligible_chunks=prepared.eligible_chunks,
                _owner=self._owner,
                _session_id=store.session_id,
                _decision_key=prepared._decision_key,
            )

        assert final_measure is not None
        incremental = final_measure.raw_prompt_tokens - base_metadata["raw_prompt_tokens"]
        if incremental <= 0:
            raise RuntimeError("Observed entity slot revision added no workspace tokens")
        receipt.update(
            incremental_raw_tokens=incremental,
            extra_bytes=incremental * self.kv_bytes_per_token,
            candidate_active_history_bytes=final_measure.per_ratio[str(ratio)][
                "history_bytes"
            ],
            budget_bytes=min(
                self.policy_config.history_budget_bytes,
                self.policy_config.workspace_budget_bytes,
            ),
        )
        metadata.update(
            raw_prompt_tokens=final_measure.raw_prompt_tokens,
            actual_raw_history_tokens=final_measure.raw_history_tokens,
            actual_history_bytes=final_measure.per_ratio[str(ratio)]["history_bytes"],
            logical_sequence_tokens=final_measure.logical_sequence_tokens,
            per_ratio=copy.deepcopy(final_measure.per_ratio),
        )
        metadata["observed_entity_slot_revision"] = receipt
        metadata["derived_workspace_prefix_message_count"] = len(base_derived) + 1
        metadata["observed_entity_slot_revision"][
            "workspace_payload_serialized_in_metadata"
        ] = False
        metadata["compression_ratio"] = self._compression_ratio(
            metadata["same_prefix_full_reference"],
            final_measure,
            metadata["source_coverage"],
            ratio,
        )
        return PreparedEventNativeS0(
            memory=final_measure.memory,
            metadata=metadata,
            eligible_chunks=prepared.eligible_chunks,
            _owner=self._owner,
            _session_id=store.session_id,
            _decision_key=prepared._decision_key,
        )


__all__ = [
    "MISSING_REQUIRED_REFERENCE_POLICY",
    "MISSING_SOURCE_ONLY_POLICY",
    "REVISION_POLICIES",
    "REVISION_VERSION",
    "RevisionObservedEntitySlotS0Controller",
    "RevisionSelection",
    "RevisionSlot",
    "select_revision_slots",
]
