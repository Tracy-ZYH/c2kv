"""Deterministic CPU policy for selecting exact conversation events.

The policy sees only an :class:`EventStore` built from the observable prefix.
It returns event identifiers; rendering and byte/token measurement belong to
the caller.  Budgets are checked by calling ``cost_fn`` on every complete
candidate set, because packet costs need not be additive per event.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from history_memory.events import EventRecord, EventStore


class PolicyInputError(ValueError):
    """The caller supplied ambiguous, unsafe, or state-inconsistent input."""


class BudgetExceeded(ValueError):
    """The events required for a valid next decision do not fit the budget."""

    def __init__(
        self, *, required_event_ids: tuple[str, ...], required_cost: int, budget: int
    ) -> None:
        self.required_event_ids = required_event_ids
        self.required_cost = required_cost
        self.budget = budget
        super().__init__(
            "Mandatory complete events need "
            f"{required_cost} bytes; budget is {budget}: {required_event_ids!r}"
        )


_MODES = frozenset({"protect", "recover_once", "persistent", "no_gist", "full_shared"})


@dataclass(frozen=True)
class RuntimeConfig:
    mode: str = "protect"
    history_budget_bytes: int = 65_536
    workspace_budget_bytes: int = 16_384
    lease_decisions: int = 3
    max_retrieved_events: int = 2

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ValueError(f"Unsupported memory policy mode: {self.mode!r}")
        for name in (
            "history_budget_bytes",
            "workspace_budget_bytes",
            "lease_decisions",
            "max_retrieved_events",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")


@dataclass(frozen=True)
class Selection:
    protected_event_ids: tuple[str, ...]
    retrieved_event_ids: tuple[str, ...]
    retained_event_ids: tuple[str, ...]
    selected_event_ids: tuple[str, ...]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class _Lease:
    expires_at_decision: int


@dataclass
class ConversationMemory:
    """Per-session online selection state.

    ``decision_key`` is an idempotency key bound to one exact observable
    prefix and one visible-event set.  A key reused with different input is a
    caller error.  ``lease_decisions`` includes the acquisition decision, so a
    lease of three is available for the acquisition and two later decisions
    unless a later direct reference refreshes it.
    """

    session_id: str
    config: RuntimeConfig = field(default_factory=RuntimeConfig)
    _known_message_json: tuple[str, ...] = field(default=(), init=False, repr=False)
    _decision_index: int = field(default=0, init=False, repr=False)
    _last_user_event_id: str | None = field(default=None, init=False, repr=False)
    _leases: dict[str, _Lease] = field(default_factory=dict, init=False, repr=False)
    _decisions: dict[str, tuple[tuple[Any, ...], Selection]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError("An explicit nonempty session_id is required")
        if not isinstance(self.config, RuntimeConfig):
            raise TypeError("config must be a RuntimeConfig")

    def prepare(
        self,
        store: EventStore,
        cost_fn: Callable[[tuple[str, ...]], int],
        visible_event_ids: set[str],
        decision_key: str,
        **forbidden_inputs: Any,
    ) -> Selection:
        """Select whole observable events for one model decision.

        ``visible_event_ids`` means every currently known source message for
        that event is already present in the caller's exact raw tail.  This can
        include an incomplete tool event: a future result is not part of the
        observable prefix yet.  Gold labels and target actions are deliberately
        outside this API.
        """
        if forbidden_inputs:
            names = ", ".join(sorted(forbidden_inputs))
            raise PolicyInputError(f"Privileged or unknown policy inputs are forbidden: {names}")
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

        known_ids = {event.event_id for event in store.events}
        try:
            visible = frozenset(visible_event_ids)
        except TypeError as exc:
            raise PolicyInputError("visible_event_ids must be an iterable of event IDs") from exc
        if any(not isinstance(event_id, str) for event_id in visible):
            raise PolicyInputError("visible_event_ids must contain only strings")
        unknown_visible = visible - known_ids
        if unknown_visible:
            raise PolicyInputError(
                f"Visible events are outside the observable prefix: {sorted(unknown_visible)!r}"
            )

        message_json = tuple(message.json_text for message in store.messages)
        self._validate_monotone_prefix(message_json)
        signature = (message_json, tuple(sorted(visible)))
        cached = self._decisions.get(decision_key)
        if cached is not None:
            old_signature, selection = cached
            if old_signature != signature:
                raise PolicyInputError(
                    f"decision_key {decision_key!r} was reused with different observable input"
                )
            return selection

        decision_index = self._decision_index + 1
        event_by_id = {event.event_id: event for event in store.events}
        raw_visible = visible
        raw_pending = tuple(
            event.event_id
            for event in store.events
            if event.event_id in visible and not event.complete
        )
        user_events = [event for event in store.events if event.kind == "user"]
        current_user = user_events[-1] if user_events else None

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

        revision_cancelled: tuple[str, ...] = ()
        if (
            current_user is not None
            and self._last_user_event_id is not None
            and current_user.event_id != self._last_user_event_id
            and _is_explicit_revision(store, current_user)
        ):
            revision_cancelled = _ordered_ids(store, leases)
            leases.clear()

        mandatory = {
            event.event_id
            for event in store.events
            if not event.complete and event.event_id not in raw_visible
        }
        if current_user is not None and current_user.event_id not in raw_visible:
            mandatory.add(current_user.event_id)

        budget = (
            self.config.history_budget_bytes
            if self.config.mode == "no_gist"
            else min(
                self.config.history_budget_bytes,
                self.config.workspace_budget_bytes,
            )
        )
        selected = set(mandatory)
        mandatory_ids = _ordered_ids(store, mandatory)
        required_cost = _checked_cost(cost_fn, mandatory_ids)
        if required_cost > budget:
            raise BudgetExceeded(
                required_event_ids=mandatory_ids,
                required_cost=required_cost,
                budget=budget,
            )

        protected = set(mandatory)
        skipped: list[dict[str, Any]] = []

        recent_complete_tools = [
            event
            for event in store.events
            if event.kind == "tool_event"
            and event.complete
            and event.event_id not in raw_visible
            and event.event_id not in selected
        ]
        if recent_complete_tools:
            recent = recent_complete_tools[-1].event_id
            if _try_add(store, cost_fn, selected, recent, budget):
                protected.add(recent)
            else:
                skipped.append(_skip(recent, "recent_complete_tool", store, cost_fn, selected))

        retained: set[str] = set()
        if self.config.mode in {"persistent", "no_gist"}:
            for event_id in _ordered_ids(store, leases):
                if event_id in raw_visible or event_id in selected:
                    continue
                if event_id not in event_by_id or not event_by_id[event_id].complete:
                    leases.pop(event_id, None)
                    continue
                if _try_add(store, cost_fn, selected, event_id, budget):
                    retained.add(event_id)
                else:
                    skipped.append(_skip(event_id, "active_lease", store, cost_fn, selected))

        retrieved: set[str] = set()
        matches: tuple[str, ...] = ()
        if self.config.mode in {"recover_once", "persistent", "no_gist"}:
            matches = _direct_source_candidates(
                store,
                current_user=current_user,
                excluded=raw_visible | protected,
            )
            triggered_selected: set[str] = set()
            direct_slots = 0
            for event_id in matches:
                if direct_slots >= self.config.max_retrieved_events:
                    break
                if event_id in selected:
                    triggered_selected.add(event_id)
                    direct_slots += 1
                    continue
                if _try_add(store, cost_fn, selected, event_id, budget):
                    retrieved.add(event_id)
                    triggered_selected.add(event_id)
                    direct_slots += 1
                else:
                    skipped.append(_skip(event_id, "direct_source", store, cost_fn, selected))

            if (
                self.config.mode in {"persistent", "no_gist"}
                and self.config.lease_decisions
            ):
                expiry = decision_index + self.config.lease_decisions
                for event_id in triggered_selected:
                    leases[event_id] = _Lease(expires_at_decision=expiry)

        if self.config.mode == "no_gist":
            for event in reversed(store.events):
                event_id = event.event_id
                if event_id in selected or event_id in raw_visible:
                    continue
                if _try_add(store, cost_fn, selected, event_id, budget):
                    continue
                skipped.append(_skip(event_id, "no_gist_history", store, cost_fn, selected))

        selected_ids = _ordered_ids(store, selected)
        final_cost = _checked_cost(cost_fn, selected_ids)
        selection = Selection(
            protected_event_ids=_ordered_ids(store, protected),
            retrieved_event_ids=_ordered_ids(store, retrieved),
            retained_event_ids=_ordered_ids(store, retained),
            selected_event_ids=selected_ids,
            metadata={
                "mode": self.config.mode,
                "decision_index": decision_index,
                "budget_bytes": budget,
                "selected_cost_bytes": final_cost,
                "mandatory_event_ids": mandatory_ids,
                "direct_source_candidate_ids": matches,
                "retrieval_policy_version": "direct-source-v0",
                "skipped_for_budget": tuple(skipped),
                "expired_lease_event_ids": expired,
                "revision_cancelled_event_ids": revision_cancelled,
                "raw_pending_event_ids": raw_pending,
                "full_shared_history_delegated": self.config.mode == "full_shared",
            },
        )

        # Commit state only after the complete decision has passed all checks.
        self._known_message_json = message_json
        self._decision_index = decision_index
        self._last_user_event_id = current_user.event_id if current_user else None
        self._leases = leases
        self._decisions[decision_key] = (signature, selection)
        return selection

    def _validate_monotone_prefix(self, message_json: tuple[str, ...]) -> None:
        if not self._known_message_json:
            return
        if (
            len(message_json) < len(self._known_message_json)
            or message_json[: len(self._known_message_json)] != self._known_message_json
        ):
            raise PolicyInputError(
                "Observable history was truncated or rewritten; create a new "
                "ConversationMemory instance with an explicit session identity"
            )


def _ordered_ids(store: EventStore, event_ids: Iterable[str]) -> tuple[str, ...]:
    wanted = set(event_ids)
    return tuple(event.event_id for event in store.events if event.event_id in wanted)


def _checked_cost(
    cost_fn: Callable[[tuple[str, ...]], int], event_ids: tuple[str, ...]
) -> int:
    try:
        value = cost_fn(event_ids)
    except Exception as exc:
        raise PolicyInputError("cost_fn failed for a complete event set") from exc
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PolicyInputError("cost_fn must return a nonnegative integer byte count")
    return value


def _try_add(
    store: EventStore,
    cost_fn: Callable[[tuple[str, ...]], int],
    selected: set[str],
    event_id: str,
    budget: int,
) -> bool:
    trial = set(selected)
    trial.add(event_id)
    if _checked_cost(cost_fn, _ordered_ids(store, trial)) > budget:
        return False
    selected.add(event_id)
    return True


def _skip(
    event_id: str,
    reason: str,
    store: EventStore,
    cost_fn: Callable[[tuple[str, ...]], int],
    selected: set[str],
) -> dict[str, Any]:
    trial = set(selected)
    trial.add(event_id)
    return {
        "event_id": event_id,
        "reason": reason,
        "candidate_cost_bytes": _checked_cost(cost_fn, _ordered_ids(store, trial)),
    }


_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/@\\-]{2,}")
_QUOTED_RE = re.compile(r"[\"']([^\"'\r\n]{3,})[\"']")
_REVISION_RE = re.compile(
    r"(?:\b(?:actually|instead|revise|revision|correction|replace|"
    r"change\s+(?:it|this|that|to|from|the\s+\w+)|correct\s+(?:it|this|that)|"
    r"ignore\s+(?:the\s+)?previous|new\s+requirement)\b|"
    r"改成|改为|更正|修订|换成|忽略之前|不要.*了)",
    re.IGNORECASE,
)
_STOP_WORDS = frozenset(
    {
        "about", "after", "again", "also", "and", "are", "been", "before",
        "can", "check", "could", "earlier", "for", "from", "have", "into",
        "just", "last", "please", "result", "that", "the", "then", "this",
        "tool", "use", "was", "were", "what", "when", "where", "with", "would",
    }
)


def _is_explicit_revision(store: EventStore, event: EventRecord) -> bool:
    return bool(_REVISION_RE.search(_event_text(store, event)))


def _direct_source_candidates(
    store: EventStore,
    *,
    current_user: EventRecord | None,
    excluded: Iterable[str],
) -> tuple[str, ...]:
    if current_user is None:
        return ()
    excluded_ids = set(excluded)
    current_start = min(current_user.source_indices)
    query_parts = [_event_text(store, current_user)]
    argument_parts: list[str] = []
    for event in store.events:
        if event.kind != "tool_event" or max(event.source_indices) < current_start:
            continue
        for message in store.event_messages(event.event_id):
            value = message.to_dict()
            for call in value.get("tool_calls") or ():
                function = call.get("function") if isinstance(call, dict) else None
                if isinstance(function, dict):
                    argument_parts.extend(_argument_strings(function.get("arguments")))
    query_parts.extend(argument_parts)
    query = "\n".join(part for part in query_parts if part)
    query_tokens = _tokens(query)
    anchors = _anchors(query)
    anchors.update(
        normalized
        for part in argument_parts
        if len(normalized := " ".join(part.casefold().split())) >= 2
    )
    if not query_tokens and not anchors:
        return ()

    ranked: list[tuple[float, int, str]] = []
    for event in store.events:
        if (
            event.event_id in excluded_ids
            or not event.complete
            or event.kind == "instruction"
            or event.event_id == current_user.event_id
        ):
            continue
        text = _event_text(store, event)
        lower = text.casefold()
        exact_hits = sum(1 for anchor in anchors if anchor in lower)
        event_tokens = _tokens(text)
        overlap = query_tokens & event_tokens
        if not exact_hits and not overlap:
            continue
        union = query_tokens | event_tokens
        lexical = len(overlap) / len(union) if union else 0.0
        kind_bonus = 8.0 if event.kind == "tool_event" else 0.0
        score = exact_hits * 100.0 + len(overlap) * 4.0 + lexical + kind_bonus
        ranked.append((score, max(event.source_indices), event.event_id))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return tuple(event_id for _, _, event_id in ranked)


def _event_text(store: EventStore, event: EventRecord) -> str:
    parts: list[str] = []
    for message in store.event_messages(event.event_id):
        value = message.to_dict()
        content = value.get("content")
        if isinstance(content, str):
            parts.append(content)
        for call in value.get("tool_calls") or ():
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if isinstance(name, str):
                parts.append(name)
            parts.extend(_argument_strings(function.get("arguments")))
    return "\n".join(parts)


def _argument_strings(arguments: Any) -> list[str]:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            return [arguments]
    values: list[str] = []
    if isinstance(arguments, Mapping):
        for value in arguments.values():
            values.extend(_argument_strings(value))
    elif isinstance(arguments, (list, tuple)):
        for value in arguments:
            values.extend(_argument_strings(value))
    elif isinstance(arguments, (str, int, float)) and not isinstance(arguments, bool):
        values.append(str(arguments))
    return values


def _tokens(text: str) -> set[str]:
    return {
        token
        for raw in _TOKEN_RE.findall(text.casefold())
        if len(token := raw.strip("._:/@\\-")) >= 3 and token not in _STOP_WORDS
    }


def _anchors(text: str) -> set[str]:
    anchors = {match.casefold() for match in _QUOTED_RE.findall(text)}
    for raw in _TOKEN_RE.findall(text):
        token = raw.strip("._:/@\\-")
        if (
            len(token) >= 4
            and (
                any(character.isdigit() for character in token)
                or any(character in raw for character in "_./:@\\-")
            )
        ):
            anchors.add(token.casefold())
    return anchors
