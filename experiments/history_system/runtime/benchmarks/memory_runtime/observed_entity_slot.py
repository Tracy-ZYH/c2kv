"""Opt-in observed entity slots for the native S0 pre-generation workspace.

The candidate copies complete scalar values from one successful observable tool
result.  It uses the existing dependency-packet schema matching primitives but
does not build or admit a dependency packet.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from history_memory.events import EventStore

from .dependency_packet import (
    _json_bindings,
    _text_bindings,
    _tool_observations,
    _tool_slots,
)
from .event_native_s0_policy import EventNativeS0Controller, PreparedEventNativeS0
from .failed_operation import failed as result_reports_failure


OBSERVED_ENTITY_SLOT_POLICY = "native-observed-entity-slot-v1"
OBSERVED_ENTITY_SLOT_VERSION = "a-native-observed-entity-slot-v1"


@dataclass(frozen=True)
class ObservedEntitySlot:
    field: str
    value: str | int | float | bool


@dataclass(frozen=True)
class ObservedEntitySlotSelection:
    event_id: str | None
    event_source_indices: tuple[int, ...]
    result_source_index: int | None
    source_message_sha256: str | None
    slots: tuple[ObservedEntitySlot, ...]
    current_user_override_fields: tuple[str, ...]
    status: str


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _source_message_sha256(store: EventStore, source_index: int) -> str:
    payload = _canonical(store.messages[source_index].to_dict()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _complete_scalar(value: Any) -> bool:
    if isinstance(value, bool) or isinstance(value, (str, int)):
        return True
    return isinstance(value, float) and math.isfinite(value)


def _slot_text(slot: ObservedEntitySlot) -> str:
    # Preserve ordinary identifier strings byte-for-byte.  JSON syntax keeps
    # the remaining scalar types and strings containing line boundaries whole.
    value = slot.value
    rendered = (
        value
        if isinstance(value, str) and "\n" not in value and "\r" not in value
        else _canonical(value)
    )
    return f"{slot.field}={rendered}"


def slot_message(slots: Sequence[ObservedEntitySlot]) -> dict[str, str] | None:
    if not slots:
        return None
    return {"role": "user", "content": "\n".join(_slot_text(slot) for slot in slots)}


def select_observed_entity_slots(
    store: EventStore,
    tools: Sequence[Mapping[str, Any]],
) -> ObservedEntitySlotSelection:
    """Select schema-matched top-level scalars without task-specific rules."""
    slot_names, _ = _tool_slots(tools)
    folded_names = {name.casefold() for name in slot_names}
    users = [event for event in store.events if event.kind == "user"]
    current_user = users[-1] if users else None
    override_fields: set[str] = set()
    if current_user is not None:
        source_index = current_user.source_indices[0]
        content = store.messages[source_index].to_dict().get("content")
        if isinstance(content, str):
            bindings = _json_bindings(content, slot_names) + _text_bindings(content, slot_names)
            override_fields = {
                binding["field"].casefold()
                for binding in bindings
                if isinstance(binding.get("field"), str)
            }

    observations = list(_tool_observations(store))
    for observation in reversed(observations):
        result = observation["result"]
        if result_reports_failure(result) or not isinstance(result, Mapping):
            continue
        matching = [
            ObservedEntitySlot(str(field), value)
            for field, value in result.items()
            if isinstance(field, str)
            and field.casefold() in folded_names
            and _complete_scalar(value)
        ]
        if not matching:
            continue
        matching.sort(key=lambda slot: (slot.field.casefold(), slot.field))
        selected = tuple(
            slot for slot in matching if slot.field.casefold() not in override_fields
        )
        return ObservedEntitySlotSelection(
            event_id=observation["event_id"],
            event_source_indices=tuple(observation["event_source_indices"]),
            result_source_index=observation["result_source_index"],
            source_message_sha256=_source_message_sha256(
                store, observation["result_source_index"]
            ),
            slots=selected,
            current_user_override_fields=tuple(
                slot.field for slot in matching if slot.field.casefold() in override_fields
            ),
            status=("selected" if selected else "current_user_binding_precedence"),
        )
    return ObservedEntitySlotSelection(
        event_id=None,
        event_source_indices=(),
        result_source_index=None,
        source_message_sha256=None,
        slots=(),
        current_user_override_fields=(),
        status="no_matching_successful_observation",
    )


class ObservedEntitySlotS0Controller(EventNativeS0Controller):
    """Native S0 with one explicitly enabled, B0-charged M5 candidate."""

    def __init__(self, *args, observed_entity_slot_policy: str | None = None, **kwargs):
        if observed_entity_slot_policy not in {None, OBSERVED_ENTITY_SLOT_POLICY}:
            raise ValueError("Unknown observed entity slot policy")
        history_representation = kwargs.get("history_representation", "c2kv")
        if observed_entity_slot_policy is not None and history_representation != "c2kv":
            raise ValueError("Observed entity slots require the C2KV S0 representation")
        super().__init__(*args, **kwargs)
        self.observed_entity_slot_policy = observed_entity_slot_policy

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
        if self.observed_entity_slot_policy is None:
            return prepared
        return self._apply_observed_entity_slots(
            prepared, store, tools, ratio=ratio, max_new_tokens=max_new_tokens
        )

    def _apply_observed_entity_slots(
        self,
        prepared: PreparedEventNativeS0,
        store: EventStore,
        tools: tuple[dict[str, Any], ...],
        *,
        ratio: int,
        max_new_tokens: int,
    ) -> PreparedEventNativeS0:
        selection = select_observed_entity_slots(store, tools)
        base_metadata = prepared.metadata
        base_derived = tuple(base_metadata.get("derived_workspace_prefix_messages") or ())
        admitted: list[ObservedEntitySlot] = []
        skipped: list[dict[str, Any]] = [
            {"field": field, "reason": "current_user_binding_precedence"}
            for field in selection.current_user_override_fields
        ]
        final_measure = None
        for slot in selection.slots:
            trial_slots = (*admitted, slot)
            message = slot_message(trial_slots)
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
            "version": OBSERVED_ENTITY_SLOT_VERSION,
            "policy": self.observed_entity_slot_policy,
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
            "source_message_role": (
                store.messages[selection.result_source_index].role
                if selection.result_source_index is not None
                else None
            ),
            "source_message_sha256": selection.source_message_sha256,
            "source_semantics": (
                "complete visible tool-result message paired with its observable tool event"
            ),
            "user_override_semantics": (
                "an explicit binding in the latest visible user source message has precedence"
            ),
            "eligible_field_names": [slot.field for slot in selection.slots],
            "admitted_field_names": [slot.field for slot in admitted],
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
        metadata["observed_entity_slot"] = receipt
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
            raise RuntimeError("Observed entity slots added no native workspace tokens")
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
        metadata["observed_entity_slot"] = receipt
        metadata["derived_workspace_prefix_message_count"] = len(base_derived) + 1
        metadata["observed_entity_slot"]["workspace_payload_serialized_in_metadata"] = False
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
    "OBSERVED_ENTITY_SLOT_POLICY",
    "OBSERVED_ENTITY_SLOT_VERSION",
    "ObservedEntitySlot",
    "ObservedEntitySlotS0Controller",
    "ObservedEntitySlotSelection",
    "select_observed_entity_slots",
    "slot_message",
]
