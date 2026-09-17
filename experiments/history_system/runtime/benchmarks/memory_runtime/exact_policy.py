"""State for one-shot exact-source evidence upgrades.

The first decision view is composed from the existing protection policy. A
later exact-source detector may add one complete event to that same decision
without advancing the decision clock a second time.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

_shared_path = str(Path(__file__).resolve().parents[2] / "python")
sys.path.insert(0, _shared_path)
try:
    from history_memory.events import EventRecord, EventStore
    from .policy import (
        BudgetExceeded,
        ConversationMemory,
        PolicyInputError,
        RuntimeConfig,
        Selection,
        _checked_cost,
        _is_explicit_revision,
        _ordered_ids,
        _skip,
        _try_add,
    )
finally:
    sys.path.remove(_shared_path)


EXACT_POLICY_VERSION = "exact-source-recovery-v1"


@dataclass(frozen=True)
class _ExactLease:
    expires_at_decision: int


@dataclass(frozen=True)
class DecisionHandle:
    """Immutable capability for at most one upgrade of the active decision."""

    selection: Selection
    store: EventStore
    visible_event_ids: frozenset[str]
    decision_index: int
    decision_key: str
    _cost_fn: Callable[[tuple[str, ...]], int] = field(repr=False, compare=False)
    _signature: tuple[Any, ...] = field(repr=False, compare=False)
    _owner: object = field(repr=False, compare=False)
    _selection_budget_bytes: int = field(repr=False, compare=False)


@dataclass
class ExactRecoveryMemory:
    """Per-session exact upgrade and lease state for the first recovery arms."""

    session_id: str
    config: RuntimeConfig
    _protect: ConversationMemory = field(init=False, repr=False)
    _owner: object = field(default_factory=object, init=False, repr=False)
    _decision_index: int = field(default=0, init=False, repr=False)
    _last_user_event_id: str | None = field(default=None, init=False, repr=False)
    _leases: dict[str, _ExactLease] = field(default_factory=dict, init=False, repr=False)
    _decisions: dict[str, tuple[tuple[Any, ...], DecisionHandle]] = field(
        default_factory=dict, init=False, repr=False
    )
    _upgrades: dict[str, tuple[str, Selection]] = field(
        default_factory=dict, init=False, repr=False
    )
    _active_handle: DecisionHandle | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError("An explicit nonempty session_id is required")
        if not isinstance(self.config, RuntimeConfig):
            raise TypeError("config must be a RuntimeConfig")
        if self.config.mode not in {"recover_once", "persistent"}:
            raise ValueError(
                "ExactRecoveryMemory requires recover_once or persistent mode"
            )
        self._protect = ConversationMemory(
            self.session_id, replace(self.config, mode="protect")
        )

    def prepare_decision(
        self,
        store: EventStore,
        cost_fn: Callable[[tuple[str, ...]], int],
        visible_event_ids: set[str],
        decision_key: str,
        *,
        selection_budget_bytes: int | None = None,
    ) -> DecisionHandle:
        """Prepare protection plus retained leases and advance one decision."""
        if not isinstance(store, EventStore):
            raise TypeError("store must be an EventStore")
        if store.session_id != self.session_id:
            raise PolicyInputError(
                f"Session mismatch: memory={self.session_id!r}, store={store.session_id!r}"
            )
        if not callable(cost_fn):
            raise TypeError("cost_fn must be callable")
        if not isinstance(decision_key, str) or not decision_key:
            raise PolicyInputError("An explicit nonempty decision_key is required")
        try:
            visible = frozenset(visible_event_ids)
        except TypeError as exc:
            raise PolicyInputError(
                "visible_event_ids must be an iterable of event IDs"
            ) from exc
        if any(not isinstance(event_id, str) for event_id in visible):
            raise PolicyInputError("visible_event_ids must contain only strings")

        admission_budget = self._selection_budget
        if selection_budget_bytes is not None:
            if (type(selection_budget_bytes) is not int
                    or not 0 <= selection_budget_bytes <= admission_budget):
                raise ValueError("selection_budget_bytes must fit the policy B/W limit")
            admission_budget = selection_budget_bytes
        message_json = tuple(message.json_text for message in store.messages)
        signature = (message_json, tuple(sorted(visible)))
        if selection_budget_bytes is not None:
            signature += (admission_budget,)

        # Stage the composed protection controller so a later validation error
        # cannot partially advance its clock.
        staged_protect = copy.deepcopy(self._protect)
        if selection_budget_bytes is not None:
            staged_protect.config = replace(
                staged_protect.config, history_budget_bytes=admission_budget,
                workspace_budget_bytes=admission_budget,
            )
        base = staged_protect.prepare(store, cost_fn, set(visible), decision_key)
        cached = self._decisions.get(decision_key)
        if cached is not None:
            old_signature, handle = cached
            if old_signature != signature:
                raise PolicyInputError(
                    f"decision_key {decision_key!r} was reused with different observable input"
                )
            return handle

        decision_index = base.metadata.get("decision_index")
        if (type(decision_index) is not int
                or decision_index != self._decision_index + 1):
            raise RuntimeError("Protection and exact-recovery decision clocks diverged")

        event_by_id = {event.event_id: event for event in store.events}
        leases = dict(self._leases)
        expired = _ordered_ids(
            store,
            (
                event_id
                for event_id, lease in leases.items()
                if decision_index >= lease.expires_at_decision
            ),
        )
        for event_id in expired:
            leases.pop(event_id, None)

        user_events = [event for event in store.events if event.kind == "user"]
        current_user: EventRecord | None = user_events[-1] if user_events else None
        revision_cancelled: tuple[str, ...] = ()
        if (
            current_user is not None
            and self._last_user_event_id is not None
            and current_user.event_id != self._last_user_event_id
            and _is_explicit_revision(store, current_user)
        ):
            revision_cancelled = _ordered_ids(store, leases)
            leases.clear()

        selected = set(base.selected_event_ids)
        retained: set[str] = set()
        skipped: list[dict[str, Any]] = []
        dropped: list[str] = []
        if self.config.mode == "persistent":
            for event_id in _ordered_ids(store, leases):
                event = event_by_id.get(event_id)
                if event is None or not event.complete:
                    leases.pop(event_id, None)
                    dropped.append(event_id)
                    continue
                # Raw visibility suppresses a duplicate packet but does not
                # refresh or pause the lease clock.
                if event_id in visible or event_id in selected:
                    continue
                if _try_add(
                    store, cost_fn, selected, event_id, admission_budget
                ):
                    retained.add(event_id)
                else:
                    skipped.append(
                        _skip(event_id, "active_exact_lease", store, cost_fn, selected)
                    )

        selected_ids = _ordered_ids(store, selected)
        final_cost = _checked_cost(cost_fn, selected_ids)
        selection = Selection(
            protected_event_ids=base.protected_event_ids,
            retrieved_event_ids=(),
            retained_event_ids=_ordered_ids(store, retained),
            selected_event_ids=selected_ids,
            metadata={
                "mode": self.config.mode,
                "exact_policy_version": EXACT_POLICY_VERSION,
                "decision_index": decision_index,
                "budget_bytes": admission_budget,
                "selected_cost_bytes": final_cost,
                "base_protected_event_ids": base.protected_event_ids,
                "pre_draft_retrieval": False,
                "expired_lease_event_ids": expired,
                "revision_cancelled_event_ids": revision_cancelled,
                "dropped_lease_event_ids": tuple(dropped),
                "skipped_for_budget": tuple(skipped),
                "upgrade": {
                    "status": "not_requested",
                    "event_id": None,
                    "upgrade_count": 0,
                },
                "protect_policy": dict(base.metadata),
            },
        )
        handle = DecisionHandle(
            selection=selection,
            store=store,
            visible_event_ids=visible,
            decision_index=decision_index,
            decision_key=decision_key,
            _cost_fn=cost_fn,
            _signature=signature,
            _owner=self._owner,
            _selection_budget_bytes=admission_budget,
        )

        self._protect = staged_protect
        self._decision_index = decision_index
        self._last_user_event_id = (
            current_user.event_id if current_user is not None else None
        )
        self._leases = leases
        self._decisions[decision_key] = (signature, handle)
        self._active_handle = handle
        return handle

    def upgrade_decision(self, handle: DecisionHandle, event_id: str) -> Selection:
        """Add one complete hidden event without advancing the decision clock."""
        if not isinstance(handle, DecisionHandle):
            raise TypeError("handle must be a DecisionHandle")
        if handle._owner is not self._owner:
            raise PolicyInputError("DecisionHandle belongs to a different memory")
        if handle is not self._active_handle:
            raise PolicyInputError("DecisionHandle is stale for the active prefix")
        cached = self._decisions.get(handle.decision_key)
        if cached is None or cached[1] is not handle:
            raise PolicyInputError("DecisionHandle is not registered by this memory")
        if not isinstance(event_id, str) or not event_id:
            raise PolicyInputError("event_id must be a nonempty string")

        previous = self._upgrades.get(handle.decision_key)
        if previous is not None:
            previous_event_id, selection = previous
            if event_id != previous_event_id:
                raise PolicyInputError(
                    "A decision cannot be upgraded with a second event"
                )
            return selection

        try:
            event = handle.store.event(event_id)
        except KeyError as exc:
            raise PolicyInputError(
                f"Upgrade event is outside the observable prefix: {event_id!r}"
            ) from exc
        if not event.complete:
            raise PolicyInputError("Upgrade event must be complete")
        if event_id in handle.visible_event_ids:
            raise PolicyInputError("Upgrade event already has exact raw visibility")
        if event_id in handle.selection.selected_event_ids:
            raise PolicyInputError("Upgrade event is already present in selected evidence")

        selected = set(handle.selection.selected_event_ids)
        selected.add(event_id)
        selected_ids = _ordered_ids(handle.store, selected)
        required_cost = _checked_cost(handle._cost_fn, selected_ids)
        if required_cost > handle._selection_budget_bytes:
            raise BudgetExceeded(
                required_event_ids=selected_ids,
                required_cost=required_cost,
                budget=handle._selection_budget_bytes,
            )

        lease_expiry = None
        leases = dict(self._leases)
        if self.config.mode == "persistent" and self.config.lease_decisions:
            lease_expiry = handle.decision_index + self.config.lease_decisions
            leases[event_id] = _ExactLease(expires_at_decision=lease_expiry)
        metadata = dict(handle.selection.metadata)
        metadata.update(
            selected_cost_bytes=required_cost,
            upgrade={
                "status": "admitted",
                "event_id": event_id,
                "upgrade_count": 1,
                "decision_index": handle.decision_index,
                "lease_expires_at_decision": lease_expiry,
            },
        )
        selection = Selection(
            protected_event_ids=handle.selection.protected_event_ids,
            retrieved_event_ids=(event_id,),
            retained_event_ids=handle.selection.retained_event_ids,
            selected_event_ids=selected_ids,
            metadata=metadata,
        )
        self._leases = leases
        self._upgrades[handle.decision_key] = (event_id, selection)
        return selection

    @property
    def _selection_budget(self) -> int:
        return min(
            self.config.history_budget_bytes,
            self.config.workspace_budget_bytes,
        )
