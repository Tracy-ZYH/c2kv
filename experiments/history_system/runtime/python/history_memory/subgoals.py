"""Observable, prefix-stable subgoal groups for history packing.

The lifecycle is derived only from an immutable :class:`EventStore`.  A
subgoal starts when an assistant emits exactly one ``Subgoal: ...`` content
line together with at least one native tool call.  A later valid declaration
or user turn closes the grouping interval, but neither boundary proves
semantic task success.  Tool events must be complete before any of their
source messages may be offered to a compressor.

This module deliberately does not render prompts or call a model.  Legacy and
event-native renderers can consume :func:`group_subgoal_sources` and retain
their existing message dialect while preserving original source indices.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from .events import EventRecord, EventStore


SUBGOAL_LIFECYCLE_VERSION = "observable-subgoal-lifecycle-v1"
SUBGOAL_GROUPING_VERSION = "observable-subgoal-source-groups-v1"

ACTIVE = "active"
CLOSED_BY_NEXT_DECLARATION = "closed_by_next_declaration"
CLOSED_BY_USER_TURN = "closed_by_user_turn"
UNASSIGNED = "unassigned"
SEMANTIC_COMPLETION_UNKNOWN = "unknown"


@dataclass(frozen=True)
class SubgoalProtocolViolation:
    """A visible ``Subgoal:`` candidate that cannot define a lifecycle edge."""

    source_index: int
    event_id: str
    reason: str

    def metadata(self) -> dict[str, Any]:
        return {
            "source_index": self.source_index,
            "event_id": self.event_id,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SubgoalRecord:
    """One source-owned subgoal interval.

    ``archivable`` is a storage property.  It never means that an evaluator or
    environment proved the subgoal correct.  Active intervals can still
    contribute completed source events to compression groups.
    """

    subgoal_id: str
    ordinal: int
    label: str
    goal_event_id: str | None
    declaration_event_id: str
    declaration_source_index: int
    event_ids: tuple[str, ...]
    source_indices: tuple[int, ...]
    lifecycle_status: str = ACTIVE
    boundary_event_id: str | None = None
    boundary_source_index: int | None = None
    pending_event_ids: tuple[str, ...] = ()
    archivable: bool = False

    @property
    def controller_completion_claimed(self) -> bool:
        return self.lifecycle_status == CLOSED_BY_NEXT_DECLARATION

    def metadata(self) -> dict[str, Any]:
        return {
            "version": SUBGOAL_LIFECYCLE_VERSION,
            "subgoal_id": self.subgoal_id,
            "ordinal": self.ordinal,
            "label": self.label,
            "goal_event_id": self.goal_event_id,
            "declaration_event_id": self.declaration_event_id,
            "declaration_source_index": self.declaration_source_index,
            "event_ids": list(self.event_ids),
            "source_indices": list(self.source_indices),
            "lifecycle_status": self.lifecycle_status,
            "boundary_event_id": self.boundary_event_id,
            "boundary_source_index": self.boundary_source_index,
            "pending_event_ids": list(self.pending_event_ids),
            "archivable": self.archivable,
            "controller_completion_claimed": self.controller_completion_claimed,
            "semantic_completion": SEMANTIC_COMPLETION_UNKNOWN,
        }


@dataclass(frozen=True)
class SubgoalLedger:
    """Deterministic lifecycle view rebuilt from one observable prefix."""

    session_id: str
    records: tuple[SubgoalRecord, ...]
    violations: tuple[SubgoalProtocolViolation, ...] = ()

    def record(self, subgoal_id: str) -> SubgoalRecord:
        for record in self.records:
            if record.subgoal_id == subgoal_id:
                return record
        raise KeyError(f"Unknown subgoal id: {subgoal_id}")

    def metadata(self) -> dict[str, Any]:
        return {
            "version": SUBGOAL_LIFECYCLE_VERSION,
            "session_id": self.session_id,
            "records": [record.metadata() for record in self.records],
            "violations": [violation.metadata() for violation in self.violations],
            "n_started": len(self.records),
            "n_active": sum(record.lifecycle_status == ACTIVE for record in self.records),
            "n_closed_by_next_declaration": sum(
                record.lifecycle_status == CLOSED_BY_NEXT_DECLARATION
                for record in self.records
            ),
            "n_closed_by_user_turn": sum(
                record.lifecycle_status == CLOSED_BY_USER_TURN
                for record in self.records
            ),
            "n_archivable": sum(record.archivable for record in self.records),
        }


@dataclass(frozen=True)
class SubgoalSourceGroup:
    """An exact subset of original sources offered as one packing group."""

    group_id: str
    subgoal_id: str | None
    goal_event_id: str | None
    label: str | None
    lifecycle_status: str
    source_indices: tuple[int, ...]
    event_ids: tuple[str, ...]
    pending_event_ids: tuple[str, ...] = ()
    archivable: bool = False
    controller_completion_claimed: bool = False

    def source_messages(self, store: EventStore) -> tuple[dict[str, Any], ...]:
        """Return fresh copies in the exact original source order."""

        if not self.group_id.startswith(f"{store.session_id}:group:"):
            raise ValueError("Source group belongs to a different EventStore session")
        return tuple(store.messages[index].to_dict() for index in self.source_indices)

    def metadata(self) -> dict[str, Any]:
        return {
            "version": SUBGOAL_GROUPING_VERSION,
            "group_id": self.group_id,
            "subgoal_id": self.subgoal_id,
            "goal_event_id": self.goal_event_id,
            "label": self.label,
            "lifecycle_status": self.lifecycle_status,
            "source_indices": list(self.source_indices),
            "event_ids": list(self.event_ids),
            "pending_event_ids": list(self.pending_event_ids),
            "archivable": self.archivable,
            "controller_completion_claimed": self.controller_completion_claimed,
            "semantic_completion": SEMANTIC_COMPLETION_UNKNOWN,
            "source_identity": "original_event_store_indices",
        }
def _declaration_candidate(message: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """Return ``(label, violation_reason)`` for an assistant message."""

    if message.get("role") != "assistant":
        return None, None
    content = message.get("content")
    if not isinstance(content, str):
        return None, None
    stripped = content.strip()
    if not stripped.startswith("Subgoal:"):
        return None, None
    if "\n" in stripped or "\r" in stripped:
        return None, "subgoal_content_must_be_one_line"
    label = stripped[len("Subgoal:") :].strip()
    if not label:
        return None, "subgoal_label_is_empty"
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        return None, "subgoal_declaration_requires_native_tool_call"
    return label, None


def _event_maps(store: EventStore) -> tuple[dict[int, EventRecord], dict[str, EventRecord]]:
    by_source: dict[int, EventRecord] = {}
    by_id: dict[str, EventRecord] = {}
    for event in store.events:
        if event.event_id in by_id:
            raise ValueError(f"Duplicate event id: {event.event_id}")
        by_id[event.event_id] = event
        for source_index in event.source_indices:
            if source_index in by_source:
                raise ValueError(f"Source index belongs to multiple events: {source_index}")
            by_source[source_index] = event
    expected = set(range(len(store.messages)))
    if set(by_source) != expected:
        raise ValueError("EventStore does not own every source message exactly once")
    return by_source, by_id


def _finalize_record(
    record: SubgoalRecord,
    events: Sequence[EventRecord],
    *,
    lifecycle_status: str,
    boundary_event_id: str | None,
    boundary_source_index: int | None,
) -> SubgoalRecord:
    event_ids = tuple(event.event_id for event in events)
    source_indices = tuple(
        source_index for event in events for source_index in event.source_indices
    )
    pending = tuple(event.event_id for event in events if not event.complete)
    return replace(
        record,
        event_ids=event_ids,
        source_indices=source_indices,
        lifecycle_status=lifecycle_status,
        boundary_event_id=boundary_event_id,
        boundary_source_index=boundary_source_index,
        pending_event_ids=pending,
        archivable=lifecycle_status != ACTIVE and not pending,
    )


def build_subgoal_ledger(store: EventStore) -> SubgoalLedger:
    """Build source-owned subgoal intervals without hidden or evaluator state."""

    if not isinstance(store, EventStore):
        raise TypeError("store must be an EventStore")
    _event_maps(store)
    records: list[SubgoalRecord] = []
    violations: list[SubgoalProtocolViolation] = []
    active: SubgoalRecord | None = None
    active_events: list[EventRecord] = []
    current_goal_event_id: str | None = None

    def close_active(
        status: str, boundary_event_id: str, boundary_source_index: int
    ) -> None:
        nonlocal active, active_events
        if active is None:
            return
        records.append(
            _finalize_record(
                active,
                active_events,
                lifecycle_status=status,
                boundary_event_id=boundary_event_id,
                boundary_source_index=boundary_source_index,
            )
        )
        active = None
        active_events = []

    for event in store.events:
        first_source = min(event.source_indices)
        first_message = store.messages[first_source].to_dict()

        if event.kind == "instruction":
            continue
        if event.kind == "user":
            close_active(CLOSED_BY_USER_TURN, event.event_id, first_source)
            current_goal_event_id = event.event_id
            continue

        label, violation = _declaration_candidate(first_message)
        if violation is not None:
            violations.append(
                SubgoalProtocolViolation(first_source, event.event_id, violation)
            )

        if label is not None:
            close_active(CLOSED_BY_NEXT_DECLARATION, event.event_id, first_source)
            active = SubgoalRecord(
                subgoal_id=f"{store.session_id}:sg{first_source}",
                ordinal=len(records) + 1,
                label=label,
                goal_event_id=current_goal_event_id,
                declaration_event_id=event.event_id,
                declaration_source_index=first_source,
                event_ids=(),
                source_indices=(),
            )
            active_events = [event]
        elif active is not None:
            active_events.append(event)

    if active is not None:
        records.append(
            _finalize_record(
                active,
                active_events,
                lifecycle_status=ACTIVE,
                boundary_event_id=None,
                boundary_source_index=None,
            )
        )
    return SubgoalLedger(store.session_id, tuple(records), tuple(violations))


def _validate_requested_sources(
    store: EventStore,
    source_indices: Sequence[int],
    by_source: Mapping[int, EventRecord],
    *,
    require_complete_events: bool,
) -> tuple[int, ...]:
    selected = tuple(source_indices)
    if any(type(index) is not int for index in selected):
        raise TypeError("source_indices must contain integers")
    if any(left >= right for left, right in zip(selected, selected[1:])):
        raise ValueError("source_indices must be strictly increasing and unique")
    if any(index < 0 or index >= len(store.messages) for index in selected):
        raise ValueError("source_indices must refer to the visible EventStore prefix")
    if require_complete_events:
        pending = sorted(
            {by_source[index].event_id for index in selected if not by_source[index].complete}
        )
        if pending:
            raise ValueError(
                "Pending tool events cannot be packed: " + ", ".join(pending)
            )
    return selected


def group_subgoal_sources(
    store: EventStore,
    source_indices: Sequence[int],
    *,
    ledger: SubgoalLedger | None = None,
    require_complete_events: bool = True,
) -> tuple[SubgoalSourceGroup, ...]:
    """Group an exact ordered source subset for a legacy or native renderer.

    The flattened output ``source_indices`` is identical to the input.  Active
    subgoals are included when the selected source occurrences belong to
    complete events; this lets always-compress process an active historical
    prefix without treating it as archivable.  Sources outside a valid
    subgoal are grouped by their immutable EventStore event id.
    """

    if not isinstance(store, EventStore):
        raise TypeError("store must be an EventStore")
    by_source, _ = _event_maps(store)
    selected = _validate_requested_sources(
        store,
        source_indices,
        by_source,
        require_complete_events=require_complete_events,
    )
    if ledger is None:
        ledger = build_subgoal_ledger(store)
    if ledger.session_id != store.session_id:
        raise ValueError("SubgoalLedger belongs to a different EventStore session")

    record_by_source: dict[int, SubgoalRecord] = {}
    for record in ledger.records:
        for source_index in record.source_indices:
            if source_index in record_by_source:
                raise ValueError(f"Source index belongs to multiple subgoals: {source_index}")
            record_by_source[source_index] = record

    group_keys: list[tuple[str, str]] = []
    grouped_sources: dict[tuple[str, str], list[int]] = {}
    for source_index in selected:
        record = record_by_source.get(source_index)
        key = (
            ("subgoal", record.subgoal_id)
            if record is not None
            else ("event", by_source[source_index].event_id)
        )
        if key not in grouped_sources:
            group_keys.append(key)
            grouped_sources[key] = []
        grouped_sources[key].append(source_index)

    groups: list[SubgoalSourceGroup] = []
    records_by_id = {record.subgoal_id: record for record in ledger.records}
    for kind, owner_id in group_keys:
        indices = tuple(grouped_sources[(kind, owner_id)])
        event_ids = tuple(
            dict.fromkeys(by_source[index].event_id for index in indices)
        )
        if kind == "subgoal":
            record = records_by_id[owner_id]
            groups.append(
                SubgoalSourceGroup(
                    group_id=f"{store.session_id}:group:{record.subgoal_id}",
                    subgoal_id=record.subgoal_id,
                    goal_event_id=record.goal_event_id,
                    label=record.label,
                    lifecycle_status=record.lifecycle_status,
                    source_indices=indices,
                    event_ids=event_ids,
                    pending_event_ids=record.pending_event_ids,
                    archivable=record.archivable,
                    controller_completion_claimed=record.controller_completion_claimed,
                )
            )
        else:
            groups.append(
                SubgoalSourceGroup(
                    group_id=f"{store.session_id}:group:{owner_id}",
                    subgoal_id=None,
                    goal_event_id=None,
                    label=None,
                    lifecycle_status=UNASSIGNED,
                    source_indices=indices,
                    event_ids=event_ids,
                )
            )

    flattened = tuple(index for group in groups for index in group.source_indices)
    if flattened != selected:
        raise RuntimeError("Subgoal grouping changed original source order or identity")
    return tuple(groups)


def group_subgoal_history(
    store: EventStore,
    source_cutoff: int,
    eligible_source_indices: Sequence[int],
    *,
    ledger: SubgoalLedger | None = None,
) -> tuple[SubgoalSourceGroup, ...]:
    """Validate a legacy history boundary and group caller-selected sources.

    Selection remains the runtime allocator's responsibility.  This adapter
    only proves that every supplied source is before ``source_cutoff`` and is
    owned by a complete event before delegating to
    :func:`group_subgoal_sources`.
    """

    if type(source_cutoff) is not int or not 0 <= source_cutoff <= len(store.messages):
        raise ValueError("source_cutoff must index the visible EventStore prefix")
    if any(type(index) is not int or index >= source_cutoff
           for index in eligible_source_indices):
        raise ValueError("eligible_source_indices must be strictly before source_cutoff")
    return group_subgoal_sources(
        store,
        eligible_source_indices,
        ledger=ledger,
        require_complete_events=True,
    )


__all__ = [
    "ACTIVE",
    "CLOSED_BY_NEXT_DECLARATION",
    "CLOSED_BY_USER_TURN",
    "SEMANTIC_COMPLETION_UNKNOWN",
    "SUBGOAL_GROUPING_VERSION",
    "SUBGOAL_LIFECYCLE_VERSION",
    "UNASSIGNED",
    "SubgoalLedger",
    "SubgoalProtocolViolation",
    "SubgoalRecord",
    "SubgoalSourceGroup",
    "build_subgoal_ledger",
    "group_subgoal_history",
    "group_subgoal_sources",
]
