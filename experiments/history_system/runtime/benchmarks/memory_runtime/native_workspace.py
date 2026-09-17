"""Matched pre-generation workspace views for the legacy 1088 renderer."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from history_memory.events import EventStore


WORKSPACE_VERSION = "pre-generation-native-workspace-v1"


@dataclass(frozen=True)
class NativeWorkspace:
    """D9 candidates mapped to the same-prefix Full training dialect."""

    event_id: str | None
    goal_event_id: str | None = None
    event_sources: tuple[tuple[str, tuple[int, ...]], ...] = ()
    rendered_hidden_sources: tuple[tuple[int, dict[str, Any]], ...] = ()

    def select(self, selected_event_ids: Sequence[str]) -> NativeWorkspaceSelection:
        """Return only D9 rows whose owning events the incumbent selected."""
        selected = set(selected_event_ids)
        event_ids = tuple(event_id for event_id, _ in self.event_sources
                          if event_id in selected)
        source_indices = tuple(sorted({
            index for event_id, indices in self.event_sources
            if event_id in selected for index in indices
        }))
        hidden_by_index = dict(self.rendered_hidden_sources)
        restored = tuple(index for index in source_indices if index in hidden_by_index)
        common = tuple(index for index in source_indices if index not in hidden_by_index)
        return NativeWorkspaceSelection(
            event_ids=event_ids,
            source_indices=source_indices,
            restored_source_indices=restored,
            common_source_indices=common,
            messages=tuple(copy.deepcopy(hidden_by_index[index]) for index in restored),
        )

    def metadata(self, selected_event_ids: Sequence[str] = ()) -> dict[str, Any]:
        native = self.select(selected_event_ids)
        return {
            "version": WORKSPACE_VERSION,
            "candidate_event_id": self.event_id,
            "candidate_tool_event_id": self.event_id,
            "candidate_goal_event_id": self.goal_event_id,
            "candidate_event_ids": [event_id for event_id, _ in self.event_sources],
            "source_indices": list(native.source_indices),
            "native_source_indices": list(native.source_indices),
            "restored_source_indices": list(native.restored_source_indices),
            "already_common_source_indices": list(native.common_source_indices),
            "source_status": "observed_complete_tool_event" if self.event_id else "no_complete_tool_event",
            "renderer_source": "same-prefix Full training renderer",
            "uses_gold_or_future_state": False,
        }


@dataclass(frozen=True)
class NativeWorkspaceSelection:
    """Selected D9 rows in original source order."""

    event_ids: tuple[str, ...] = ()
    source_indices: tuple[int, ...] = ()
    restored_source_indices: tuple[int, ...] = ()
    common_source_indices: tuple[int, ...] = ()
    messages: tuple[dict[str, Any], ...] = ()


def plan_native_workspace(
    store: EventStore,
    *,
    source_cutoff: int,
    full_messages: Sequence[Mapping[str, Any]],
    full_cutoff: int,
) -> NativeWorkspace:
    """Restore only source rows absent from the existing common raw suffix.

    The legacy Full renderer flattens tool results to user messages and native
    tool calls to its trained action text.  Copying from that rendered view,
    instead of from the OpenAI source objects, preserves the checkpoint's raw
    dialect.  Source rows at or after ``source_cutoff`` stay in their existing
    suffix positions and are never duplicated.
    """
    if not isinstance(store, EventStore):
        raise TypeError("store must be an EventStore")
    if type(source_cutoff) is not int or not 0 <= source_cutoff <= len(store.messages):
        raise ValueError("source_cutoff must index the observable prefix")
    if type(full_cutoff) is not int or not 0 <= full_cutoff <= len(full_messages):
        raise ValueError("full_cutoff must index the Full rendered prefix")

    shift = len(full_messages) - len(store.messages)
    if shift not in {0, 1} or full_cutoff != source_cutoff + shift:
        raise ValueError("Full renderer does not preserve the legacy source boundary")
    if shift and (not full_messages or full_messages[0].get("role") != "system"):
        raise ValueError("Full renderer source shift is not its default system message")

    tool = next(
        (
            event
            for event in reversed(store.events)
            if event.kind == "tool_event"
            and event.complete
            and any(index < source_cutoff for index in event.source_indices)
        ),
        None,
    )
    users = [event for event in store.events if event.kind == "user"]
    goal = users[-1] if users else None
    candidates = {event.event_id: event for event in (tool, goal) if event is not None}
    ordered = tuple(sorted(candidates.values(), key=lambda event: min(event.source_indices)))
    hidden = tuple(sorted({
        index for event in ordered for index in event.source_indices
        if index < source_cutoff
    }))
    rendered = tuple(
        (index, copy.deepcopy(dict(full_messages[index + shift]))) for index in hidden
    )
    if len(rendered) != len(hidden):
        raise RuntimeError("Native workspace restoration lost a source row")
    return NativeWorkspace(
        event_id=tool.event_id if tool else None,
        goal_event_id=goal.event_id if goal else None,
        event_sources=tuple((event.event_id, tuple(event.source_indices)) for event in ordered),
        rendered_hidden_sources=rendered,
    )


__all__ = [
    "NativeWorkspace", "NativeWorkspaceSelection", "WORKSPACE_VERSION",
    "plan_native_workspace",
]
