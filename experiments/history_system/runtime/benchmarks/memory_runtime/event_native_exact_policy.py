"""Two-phase exact-source recovery for event-native inference views."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Sequence

from history_memory.events import EventStore
from history_memory.packing import (
    MemoryView,
    PackedMemory,
    PackingBudgetError,
    encode_event_chunks,
    pack_memory,
    select_view,
)

from .always_compress import (
    ALWAYS_COMPRESSION_POLICY,
    CapacityInfeasible,
    coverage_accounting,
)
from .event_native_always import (
    NATIVE_ALWAYS_CANONICAL_MODES,
    NATIVE_ALWAYS_IMPLEMENTATION_PROFILE,
    NATIVE_ALWAYS_ROUTE_MODES,
)
from .event_native_policy import (
    EVENT_NATIVE_POLICY_VERSION,
    EventNativeController,
    PreparedEventNative,
    _PackingConfig,
    _canonical_json,
    _evidence_increment_tokens,
    _history_components,
    _json_snapshot,
)
from .event_native_raw import RuntimeMemoryView, build_raw_control
from .exact_gap import GapDecision, detect_exact_source_gap
from .exact_policy import DecisionHandle, ExactRecoveryMemory
from .policy import BudgetExceeded, PolicyInputError, RuntimeConfig, Selection


EVENT_NATIVE_EXACT_VERSION = "a-event-native-exact-source-v1"
BUDGETED_GIST_LAYOUT = "event-native-budgeted-gist-v1"
ALWAYS_COMPRESS_GIST_LAYOUT = "event-native-always-compress-gist-v1"
_ALWAYS_GIST_MODES = frozenset(
    {"ac_gist_static", "ac_protect", "ac_exact_once", "ac_exact_persistent"}
)
_ALWAYS_RECOVERY_DISABLED = frozenset({"ac_gist_static", "ac_protect"})
_MODES = frozenset(
    {
        "capacity_protect",
        "capacity_exact_once",
        "capacity_exact_persistent",
        "full_exact_shared",
        "capacity_exact_no_gist",
        *NATIVE_ALWAYS_ROUTE_MODES,
    }
)


@dataclass
class PreparedEventNativeExact:
    """One active first-draft decision that may be reconsidered once."""

    memory: PackedMemory
    metadata: dict[str, Any]
    _owner: object = field(repr=False, compare=False)
    _store: EventStore = field(repr=False, compare=False)
    _decision: DecisionHandle = field(repr=False, compare=False)
    _policy_visible_event_ids: frozenset[str] = field(repr=False, compare=False)
    _actual_visible_source_indices: frozenset[int] = field(repr=False, compare=False)
    _cost_fn: Callable[[tuple[str, ...]], int] = field(repr=False, compare=False)
    _plan_cache: dict[tuple[str, ...], "_GistPlan"] = field(
        repr=False, compare=False
    )
    _tools: tuple[dict[str, Any], ...] = field(repr=False, compare=False)
    _ratio: int = field(repr=False, compare=False)
    _max_new_tokens: int = field(repr=False, compare=False)
    _static_view: MemoryView = field(repr=False, compare=False)
    _full_memory: PackedMemory = field(repr=False, compare=False)
    _full_history_tokens: int = field(repr=False, compare=False)
    _full_history_bytes: int = field(repr=False, compare=False)
    _selection_budget_bytes: int | None = field(repr=False, compare=False)
    _checked_signature: str | None = field(default=None, repr=False, compare=False)
    _checked_result: dict[str, Any] | None = field(
        default=None, repr=False, compare=False
    )


@dataclass
class _SessionState:
    message_json: tuple[str, ...]
    tools_json: str
    exact_memory: ExactRecoveryMemory
    decisions: dict[str, tuple[tuple[Any, ...], PreparedEventNativeExact]]
    active_decision_key: str


@dataclass(frozen=True)
class _ViewMeasure:
    memory: PackedMemory
    per_ratio: dict[str, dict[str, int]]
    evidence_tokens: int
    evidence_bytes: int
    logical_sequence_tokens: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class _GistPlan:
    memory: PackedMemory
    metadata: dict[str, Any]
    selection_cost_bytes: int
    admissible: bool


class EventNativeExactController:
    """Keep exact recovery state around immutable event-native request prefixes."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        packing: Mapping[str, Any],
        policy: Mapping[str, Any],
        mode: str,
        model_context: int | None = None,
        compression_policy: str | None = None,
        history_view_protocol: str = "fixed-budget-main",
    ) -> None:
        if not callable(getattr(tokenizer, "apply_chat_template", None)):
            raise TypeError("tokenizer must expose apply_chat_template")
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {sorted(_MODES)!r}")
        if model_context is not None and not _positive_int(model_context):
            raise ValueError("model_context must be a positive integer or None")
        route_mode = mode
        always_compress = route_mode in NATIVE_ALWAYS_ROUTE_MODES
        if always_compress and compression_policy != ALWAYS_COMPRESSION_POLICY:
            raise ValueError(
                "Always-compress routes require compression_policy='always-compress-v1'"
            )
        if not always_compress and compression_policy is not None:
            raise ValueError("compression_policy requires a new always-compress route")
        if history_view_protocol != "fixed-budget-main":
            raise ValueError(
                "event-native P0 supports history_view_protocol='fixed-budget-main'"
            )
        mode = NATIVE_ALWAYS_CANONICAL_MODES.get(route_mode, route_mode)
        self.tokenizer = tokenizer
        self.packing = _PackingConfig.from_mapping(packing)
        base_policy, self.kv_bytes_per_token = EventNativeController._parse_policy(
            policy
        )
        exact_mode = (
            "recover_once"
            if mode in {"capacity_protect", "capacity_exact_once"}
            else "persistent"
        )
        self.policy_config = replace(base_policy, mode=exact_mode)
        self.policy = _json_snapshot(policy)
        self.mode = mode
        self.route_mode = route_mode
        self.always_compress = always_compress
        self.compression_policy = compression_policy
        self.history_view_protocol = history_view_protocol
        self.model_context = model_context
        self._owner = object()
        self._sessions: dict[str, _SessionState] = {}

    @property
    def _requires_min_gist(self) -> bool:
        return self.route_mode in _ALWAYS_GIST_MODES

    def prepare(
        self,
        payload: Mapping[str, Any],
        *,
        ratio: int,
        max_new_tokens: int,
    ) -> PreparedEventNativeExact:
        session_id, decision_key, store, tools, tools_json, message_json = (
            self._validate_request(payload, ratio, max_new_tokens)
        )
        signature = (message_json, tools_json, ratio, max_new_tokens)
        state = self._sessions.get(session_id)
        if state is not None:
            if state.tools_json != tools_json:
                raise PolicyInputError(
                    "Tools changed within a session; use a new explicit session_id"
                )
            EventNativeController._validate_monotone_prefix(
                state.message_json, message_json
            )
            cached = state.decisions.get(decision_key)
            if cached is not None:
                old_signature, prepared = cached
                if old_signature != signature:
                    raise PolicyInputError(
                        f"decision_key {decision_key!r} was reused with different input"
                    )
                return prepared

        static_view = select_view(
            store, recent_tool_events=self.packing.recent_tool_events
        )
        full_memory, full_history_tokens, full_history_bytes = self._measure_full(
            store, tools, max_new_tokens
        )
        eligible_event_ids = tuple(static_view.gist_event_ids)
        activated = (
            bool(eligible_event_ids)
            if self.always_compress
            else full_history_bytes > self.policy_config.history_budget_bytes
        )
        capacity_gate = {
            "rule": "full_native_history_exceeds_fixed_source_baseline_budget",
            "full_history_tokens": full_history_tokens,
            "full_history_bytes": full_history_bytes,
            "history_budget_bytes": self.policy_config.history_budget_bytes,
            "activated": activated,
            "full_identity_bypass": not activated,
        }
        if self.always_compress:
            capacity_gate.update(
                rule="eligible_native_completed_history_always_compressed",
                eligibility_stage=(
                    "observable EventStore plus frozen select_view before "
                    "max_chunks and byte-budget admission"
                ),
                eligible_event_ids=list(eligible_event_ids),
                eligible_event_count=len(eligible_event_ids),
                full_identity_bypass=False,
                natural_zero_eligible_raw=not activated,
            )

        staged = (
            copy.deepcopy(state.exact_memory)
            if state is not None
            else ExactRecoveryMemory(session_id, self.policy_config)
        )
        plan_cache: dict[tuple[str, ...], _GistPlan] = {}
        if not activated:
            policy_visible = frozenset(event.event_id for event in store.events)

            def no_packet_cost(event_ids: tuple[str, ...]) -> int:
                if event_ids:
                    raise PolicyInputError(
                        "Full-visible history needs no exact evidence packet"
                    )
                return 0

            cost_fn = no_packet_cost
            decision = staged.prepare_decision(
                store, cost_fn, set(policy_visible), decision_key
            )
            if decision.selection.selected_event_ids:
                raise RuntimeError("Full identity selected an auxiliary event")
            raw = build_raw_control(
                store,
                self.tokenizer,
                packing=self._raw_packing_mapping(),
                policy=self.policy,
                mode="full_original",
                max_new_tokens=max_new_tokens,
                tools=tools,
            )
            memory = raw.memory
            self._require_model_context(memory, max_new_tokens)
            base_metadata = copy.deepcopy(raw.metadata)
        else:
            policy_visible = frozenset(
                event.event_id for event in store.events
            ) if self.route_mode == "ac_gist_static" else frozenset(
                static_view.raw_event_ids
            )
            selection_budget = min(
                self.policy_config.history_budget_bytes,
                self.policy_config.workspace_budget_bytes,
            )

            def infeasible_candidate(
                base: _GistPlan, error: CapacityInfeasible
            ) -> _GistPlan:
                metadata = copy.deepcopy(base.metadata)
                metadata.update(
                    admission_failures=["capacity_infeasible"],
                    admission_failure_detail=str(error),
                    selection_cost_bytes=selection_budget + 1,
                )
                return _GistPlan(
                    memory=base.memory,
                    metadata=metadata,
                    selection_cost_bytes=selection_budget + 1,
                    admissible=False,
                )

            def plan(event_ids: tuple[str, ...]) -> _GistPlan:
                ordered = _ordered_ids(store, event_ids)
                if ordered not in plan_cache:
                    try:
                        plan_cache[ordered] = self._plan_budgeted_gist(
                            store,
                            static_view,
                            ordered,
                            tools,
                            max_new_tokens,
                        )
                    except CapacityInfeasible as error:
                        base = plan_cache.get(())
                        if not ordered or base is None:
                            raise
                        plan_cache[ordered] = infeasible_candidate(base, error)
                return plan_cache[ordered]

            empty_plan = plan(())
            if not empty_plan.admissible:
                error_type = CapacityInfeasible if self._requires_min_gist else PackingBudgetError
                raise error_type(
                    "Mandatory event-native view exceeds a frozen packing or byte budget: "
                    f"{empty_plan.metadata['admission_failures']!r}"
                )

            def cost_fn(event_ids: tuple[str, ...]) -> int:
                candidate = plan(event_ids)
                if not candidate.admissible:
                    return max(selection_budget + 1, candidate.selection_cost_bytes)
                if self._requires_min_gist:
                    return candidate.selection_cost_bytes
                return candidate.selection_cost_bytes

            decision_kwargs = (
                {"selection_budget_bytes": selection_budget}
                if self.always_compress
                else {}
            )
            try:
                decision = staged.prepare_decision(
                    store, cost_fn, set(policy_visible), decision_key, **decision_kwargs
                )
            except BudgetExceeded as error:
                if not self.always_compress:
                    raise
                raise CapacityInfeasible(
                    "Necessary event-native raw/evidence input and one complete "
                    f"gist block exceed the fixed budget: {error}"
                ) from error
            chosen = plan(decision.selection.selected_event_ids)
            if not chosen.admissible:
                error_type = (
                    CapacityInfeasible if self._requires_min_gist else PackingBudgetError
                )
                raise error_type(
                    "Selected exact evidence has no admissible event-native view"
                )
            if self._requires_min_gist:
                reservation_event_id = chosen.metadata[
                    "min_gist_reservation_event_id"
                ]
                selected_ids = _ordered_ids(
                    store, decision.selection.selected_event_ids
                )
                plan_cache = {selected_ids: chosen}

                def fixed_plan(event_ids: tuple[str, ...]) -> _GistPlan:
                    ordered = _ordered_ids(store, event_ids)
                    if ordered not in plan_cache:
                        try:
                            plan_cache[ordered] = self._plan_budgeted_gist(
                                store,
                                static_view,
                                ordered,
                                tools,
                                max_new_tokens,
                                reservation_event_id=reservation_event_id,
                            )
                        except CapacityInfeasible as error:
                            plan_cache[ordered] = infeasible_candidate(chosen, error)
                    return plan_cache[ordered]

                def fixed_cost_fn(event_ids: tuple[str, ...]) -> int:
                    candidate = fixed_plan(event_ids)
                    if not candidate.admissible:
                        return max(
                            selection_budget + 1,
                            candidate.selection_cost_bytes,
                        )
                    return candidate.selection_cost_bytes

                cost_fn = fixed_cost_fn
                decision = replace(decision, _cost_fn=fixed_cost_fn)
                cached_signature, _ = staged._decisions[decision_key]
                staged._decisions[decision_key] = (cached_signature, decision)
                staged._active_handle = decision
            memory, base_metadata = self._render_representation(
                store,
                static_view,
                decision.selection.selected_event_ids,
                chosen,
                tools,
                max_new_tokens,
            )

        actual_visible_sources = frozenset(memory.raw_source_indices)
        base_metadata = self._annotate_always_compress(
            store,
            static_view,
            memory,
            base_metadata,
            full_memory,
            full_history_tokens,
            full_history_bytes,
            tools,
            ratio,
        )
        metadata = self._compose_metadata(
            store,
            decision_key,
            ratio,
            max_new_tokens,
            decision.selection,
            base_metadata,
            capacity_gate,
            actual_visible_sources,
            policy_visible,
            recovery_stage="first_draft",
        )
        prepared = PreparedEventNativeExact(
            memory=memory,
            metadata=metadata,
            _owner=self._owner,
            _store=store,
            _decision=decision,
            _policy_visible_event_ids=policy_visible,
            _actual_visible_source_indices=actual_visible_sources,
            _cost_fn=cost_fn,
            _plan_cache=plan_cache,
            _tools=tools,
            _ratio=ratio,
            _max_new_tokens=max_new_tokens,
            _static_view=static_view,
            _full_memory=full_memory,
            _full_history_tokens=full_history_tokens,
            _full_history_bytes=full_history_bytes,
            _selection_budget_bytes=(
                selection_budget if activated and self.always_compress else None
            ),
        )
        decisions = dict(state.decisions) if state is not None else {}
        decisions[decision_key] = (signature, prepared)
        self._sessions[session_id] = _SessionState(
            message_json=message_json,
            tools_json=tools_json,
            exact_memory=staged,
            decisions=decisions,
            active_decision_key=decision_key,
        )
        return prepared

    def reconsider(
        self,
        prepared: PreparedEventNativeExact,
        draft_tool_calls: Any,
        *,
        draft_text: str,
        parse_error: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(prepared, PreparedEventNativeExact) or prepared._owner is not self._owner:
            raise PolicyInputError("Prepared decision belongs to another exact controller")
        if not isinstance(draft_text, str):
            raise TypeError("draft_text must be a string")
        if parse_error is not None and not isinstance(parse_error, str):
            raise TypeError("parse_error must be a string or None")
        signature = _canonical_json(
            {
                "draft_text": draft_text,
                "draft_tool_calls": draft_tool_calls,
                "parse_error": parse_error,
            }
        )
        if prepared._checked_result is not None:
            if signature != prepared._checked_signature:
                raise PolicyInputError(
                    "A prepared decision cannot inspect a second different draft"
                )
            return _copy_reconsideration(prepared._checked_result)

        state = self._sessions.get(prepared._store.session_id)
        decision_key = prepared._decision.decision_key
        if state is None or state.active_decision_key != decision_key:
            raise PolicyInputError("Prepared decision is stale for the active prefix")

        gap: GapDecision | None = None
        if self.mode == "capacity_protect" or self.route_mode in _ALWAYS_RECOVERY_DISABLED:
            trace = {
                "version": EVENT_NATIVE_EXACT_VERSION,
                "status": "no_op",
                "reason": "recovery_disabled",
                "gap_type": None,
                "candidate_event_id": None,
                "bindings": [],
                "judges_action_correctness": False,
            }
        elif parse_error is not None:
            gap = GapDecision("abstain", "draft_parse_error")
            trace = {**gap.metadata(), "parse_error": parse_error}
        else:
            gap = detect_exact_source_gap(
                prepared._store,
                visible_source_indices=prepared._actual_visible_source_indices,
                source_cutoff=len(prepared._store.messages),
                draft_tool_calls=draft_tool_calls,
            )
            trace = gap.metadata()
        trace.update(
            decision_index=prepared._decision.decision_index,
            upgrade_count=0,
            regeneration_allowed=False,
            upgraded_event_id=None,
        )

        regenerate = False
        memory = prepared.memory
        metadata = copy.deepcopy(prepared.metadata)
        if gap is not None and gap.status == "gap":
            if gap.event_id is None:
                raise RuntimeError("Exact gap omitted its candidate event")
            if any(
                index in prepared._actual_visible_source_indices
                for index in prepared._store.event(gap.event_id).source_indices
            ):
                raise RuntimeError("Exact detector selected an actually visible source")
            staged = copy.deepcopy(state.exact_memory)
            staged_handle = staged.prepare_decision(
                prepared._store,
                prepared._cost_fn,
                set(prepared._policy_visible_event_ids),
                decision_key,
                **(
                    {"selection_budget_bytes": prepared._selection_budget_bytes}
                    if prepared._selection_budget_bytes is not None
                    else {}
                ),
            )
            try:
                upgraded = staged.upgrade_decision(staged_handle, gap.event_id)
                static_view = select_view(
                    prepared._store,
                    recent_tool_events=self.packing.recent_tool_events,
                )
                plan = self._cached_or_new_plan(
                    prepared,
                    static_view,
                    upgraded.selected_event_ids,
                )
                if not plan.admissible:
                    raise PackingBudgetError(
                        "Upgraded exact evidence has no admissible event-native view"
                    )
                memory, base_metadata = self._render_representation(
                    prepared._store,
                    static_view,
                    upgraded.selected_event_ids,
                    plan,
                    prepared._tools,
                    prepared._max_new_tokens,
                )
            except (BudgetExceeded, PackingBudgetError) as error:
                trace.update(status="abstain", reason="budget_exhausted")
                if isinstance(error, BudgetExceeded):
                    trace.update(
                        required_cost_bytes=error.required_cost,
                        budget_bytes=error.budget,
                    )
            else:
                state.exact_memory = staged
                actual_visible = frozenset(memory.raw_source_indices)
                base_metadata = self._annotate_always_compress(
                    prepared._store,
                    prepared._static_view,
                    memory,
                    base_metadata,
                    prepared._full_memory,
                    prepared._full_history_tokens,
                    prepared._full_history_bytes,
                    prepared._tools,
                    prepared._ratio,
                )
                metadata = self._compose_metadata(
                    prepared._store,
                    decision_key,
                    prepared._ratio,
                    prepared._max_new_tokens,
                    upgraded,
                    base_metadata,
                    prepared.metadata["capacity_gate"],
                    actual_visible,
                    prepared._policy_visible_event_ids,
                    recovery_stage="post_draft_upgraded",
                )
                trace.update(
                    upgrade_count=1,
                    regeneration_allowed=True,
                    upgraded_event_id=gap.event_id,
                )
                regenerate = True

        metadata["exact_recovery"] = copy.deepcopy(trace)
        metadata["post_draft_exact_recovery_applied"] = regenerate
        result = {
            "regenerate": regenerate,
            "memory": memory,
            "metadata": metadata,
            "decision": copy.deepcopy(trace),
        }
        prepared._checked_signature = signature
        prepared._checked_result = _copy_reconsideration(result)
        return result

    def _cached_or_new_plan(
        self,
        prepared: PreparedEventNativeExact,
        static_view: MemoryView,
        event_ids: Sequence[str],
    ) -> _GistPlan:
        ordered = _ordered_ids(prepared._store, event_ids)
        plan = prepared._plan_cache.get(ordered)
        if plan is None:
            plan = self._plan_budgeted_gist(
                prepared._store,
                static_view,
                ordered,
                prepared._tools,
                prepared._max_new_tokens,
            )
            prepared._plan_cache[ordered] = plan
        return plan

    def _render_representation(
        self,
        store: EventStore,
        static_view: MemoryView,
        evidence_ids: Sequence[str],
        gist_plan: _GistPlan,
        tools: tuple[dict[str, Any], ...],
        max_new_tokens: int,
    ) -> tuple[PackedMemory, dict[str, Any]]:
        if self.mode in {
            "capacity_protect",
            "capacity_exact_once",
            "capacity_exact_persistent",
        }:
            return gist_plan.memory, copy.deepcopy(gist_plan.metadata)
        raw_mode = (
            "full_shared"
            if self.mode == "full_exact_shared"
            else "no_gist"
        )
        raw = build_raw_control(
            store,
            self.tokenizer,
            packing=self._raw_packing_mapping(),
            policy=self.policy,
            mode=raw_mode,
            evidence_event_ids=tuple(evidence_ids),
            max_new_tokens=max_new_tokens,
            tools=tools,
        )
        self._require_model_context(raw.memory, max_new_tokens)
        metadata = copy.deepcopy(raw.metadata)
        actual = dict(metadata.get("actual_costs") or {})
        metadata["per_ratio"] = {
            str(ratio): copy.deepcopy(actual) for ratio in self.packing.ratios
        }
        metadata["common_budgeted_gist_selection_cost_bytes"] = (
            gist_plan.selection_cost_bytes
        )
        return raw.memory, metadata

    def _plan_budgeted_gist(
        self,
        store: EventStore,
        static_view: MemoryView,
        evidence_ids: tuple[str, ...],
        tools: tuple[dict[str, Any], ...],
        max_new_tokens: int,
        *,
        reservation_event_id: str | None = None,
    ) -> _GistPlan:
        if self._requires_min_gist:
            return self._plan_always_compress_gist(
                store,
                static_view,
                evidence_ids,
                tools,
                max_new_tokens,
                reservation_event_id=reservation_event_id,
            )
        static_raw = set(static_view.raw_event_ids)
        evidence = set(evidence_ids)
        if evidence & static_raw:
            raise PolicyInputError(
                "Exact evidence must be hidden from the frozen static raw view"
            )
        static_gist = tuple(static_view.gist_event_ids)
        if not evidence <= set(static_gist):
            raise PolicyInputError(
                "Exact evidence must name complete events in the frozen static gist"
            )
        raw_ids = _ordered_ids(store, static_raw | evidence)
        remaining = tuple(event_id for event_id in static_gist if event_id not in evidence)
        mandatory_ids = raw_ids
        base_view = RuntimeMemoryView(
            gist_event_ids=(),
            raw_event_ids=raw_ids,
            evidence_event_ids=evidence_ids,
            omitted_event_ids=remaining,
            mandatory_raw_event_ids=mandatory_ids,
            raw_control_layout=BUDGETED_GIST_LAYOUT,
        )
        base_measure = self._measure_view(
            store, static_view, base_view, tools, max_new_tokens
        )
        selection_budget = min(
            self.policy_config.history_budget_bytes,
            self.policy_config.workspace_budget_bytes,
        )
        raw_history_bytes = max(
            row["history_bytes"] for row in base_measure.per_ratio.values()
        )
        selection_cost = max(
            base_measure.evidence_bytes,
            max(
                0,
                raw_history_bytes
                - self.policy_config.history_budget_bytes
                + selection_budget,
            ),
        )
        if base_measure.reasons:
            selection_cost = max(selection_cost, selection_budget + 1)

        view = base_view
        measure = base_measure
        priority: list[str] = []
        if static_gist and static_gist[0] not in evidence:
            priority.append(static_gist[0])
        priority.extend(
            event.event_id
            for event in sorted(
                (
                    store.event(event_id)
                    for event_id in static_gist
                    if event_id not in evidence and event_id not in priority
                ),
                key=lambda event: max(event.source_indices),
                reverse=True,
            )
        )
        retained: list[str] = []
        skipped: list[dict[str, Any]] = []
        if not base_measure.reasons:
            for event_id in priority:
                candidate_gist = _ordered_ids(
                    store, set(view.gist_event_ids) | {event_id}
                )
                candidate = replace(
                    view,
                    gist_event_ids=candidate_gist,
                    omitted_event_ids=tuple(
                        item for item in view.omitted_event_ids if item != event_id
                    ),
                )
                try:
                    candidate_measure = self._measure_view(
                        store, static_view, candidate, tools, max_new_tokens
                    )
                except PackingBudgetError as error:
                    skipped.append(
                        {
                            "event_id": event_id,
                            "reasons": ["packing_budget_exceeded"],
                            "detail": str(error),
                        }
                    )
                    continue
                if candidate_measure.reasons:
                    skipped.append(
                        {
                            "event_id": event_id,
                            "reasons": list(candidate_measure.reasons),
                        }
                    )
                    continue
                view = candidate
                measure = candidate_measure
                retained.append(event_id)

        metadata = {
            "raw_control_layout": BUDGETED_GIST_LAYOUT,
            "raw_event_ids": list(view.raw_event_ids),
            "gist_event_ids": list(view.gist_event_ids),
            "omitted_event_ids": list(view.omitted_event_ids),
            "mandatory_raw_event_ids": list(view.mandatory_raw_event_ids),
            "static_raw_event_ids": list(static_view.raw_event_ids),
            "evidence_event_ids": list(view.evidence_event_ids),
            "shared_evidence_event_ids": list(view.evidence_event_ids),
            "raw_source_indices": list(measure.memory.raw_source_indices),
            "full_source_indices": list(range(len(store.messages))),
            "full_source_coverage": not view.omitted_event_ids,
            "gist_refill_priority_event_ids": priority,
            "gist_refilled_event_ids": retained,
            "skipped_gist_refill_events": skipped,
            "event_priority_mapping": (
                "whole event: oldest frozen pre-E gist event first when it is not E; "
                "then remaining frozen pre-E gist events newest-first"
            ),
            "legacy_1088_block_parity": False,
            "atomic_packing_unit": "whole_event_all_encoder_chunks",
            "selection_cost_bytes": selection_cost,
            "evidence_tokens": measure.evidence_tokens,
            "evidence_bytes": measure.evidence_bytes,
            "history_bytes": max(
                row["history_bytes"] for row in measure.per_ratio.values()
            ),
            "sequence_tokens": max(
                row["sequence_tokens"] for row in measure.per_ratio.values()
            ),
            "logical_sequence_tokens": measure.logical_sequence_tokens,
            "per_ratio": copy.deepcopy(measure.per_ratio),
            "admission_failures": list(base_measure.reasons),
            "reflow_logical_positions": bool(
                view.gist_event_ids or view.omitted_event_ids
            ),
        }
        return _GistPlan(
            memory=measure.memory,
            metadata=metadata,
            selection_cost_bytes=selection_cost,
            admissible=not base_measure.reasons,
        )

    def _plan_always_compress_gist(
        self,
        store: EventStore,
        static_view: MemoryView,
        evidence_ids: tuple[str, ...],
        tools: tuple[dict[str, Any], ...],
        max_new_tokens: int,
        *,
        reservation_event_id: str | None = None,
    ) -> _GistPlan:
        """Reserve one complete native gist event with necessary raw evidence."""

        static_raw = set(static_view.raw_event_ids)
        static_gist = tuple(static_view.gist_event_ids)
        evidence = set(evidence_ids)
        if not static_gist:
            raise CapacityInfeasible(
                "Always-compress gist planning requires eligible completed history"
            )
        if evidence & static_raw:
            raise PolicyInputError(
                "Exact evidence must be hidden from the frozen static raw view"
            )
        if not evidence <= set(static_gist):
            raise PolicyInputError(
                "Exact evidence must name complete events in the frozen static gist"
            )

        priority = self._gist_priority(store, static_gist)
        if reservation_event_id is not None:
            if reservation_event_id not in static_gist:
                raise PolicyInputError(
                    "The fixed minimum-gist reservation is outside the frozen "
                    "pre-E gist candidates"
                )
            reservation_candidates = [reservation_event_id]
        else:
            reservation_candidates = priority
        chosen_reservation_event_id = None
        reservation_measure = None
        reservation_skips: list[dict[str, Any]] = []
        raw_ids = _ordered_ids(store, static_raw | evidence)
        for event_id in reservation_candidates:
            reservation_view = RuntimeMemoryView(
                gist_event_ids=(event_id,),
                raw_event_ids=raw_ids,
                evidence_event_ids=evidence_ids,
                omitted_event_ids=tuple(
                    item
                    for item in static_gist
                    if item != event_id and item not in evidence
                ),
                mandatory_raw_event_ids=raw_ids,
                raw_control_layout=ALWAYS_COMPRESS_GIST_LAYOUT,
            )
            try:
                candidate_measure = self._measure_view(
                    store, static_view, reservation_view, tools, max_new_tokens
                )
            except PackingBudgetError as error:
                reservation_skips.append(
                    {
                        "event_id": event_id,
                        "reasons": ["packing_budget_exceeded"],
                        "detail": str(error),
                    }
                )
                continue
            if candidate_measure.reasons:
                reservation_skips.append(
                    {
                        "event_id": event_id,
                        "reasons": list(candidate_measure.reasons),
                    }
                )
                continue
            chosen_reservation_event_id = event_id
            reservation_measure = candidate_measure
            break
        if chosen_reservation_event_id is None or reservation_measure is None:
            raise CapacityInfeasible(
                "No complete eligible native gist block fits with mandatory input: "
                f"{reservation_skips!r}"
            )

        gist_ids = (chosen_reservation_event_id,)
        represented = set(raw_ids) | set(gist_ids)
        view = RuntimeMemoryView(
            gist_event_ids=gist_ids,
            raw_event_ids=raw_ids,
            evidence_event_ids=evidence_ids,
            omitted_event_ids=tuple(
                event_id for event_id in static_gist if event_id not in represented
            ),
            mandatory_raw_event_ids=raw_ids,
            raw_control_layout=ALWAYS_COMPRESS_GIST_LAYOUT,
        )
        measure = reservation_measure
        admission_failures: list[str] = []
        packing_detail = None

        retained = [chosen_reservation_event_id]
        skipped: list[dict[str, Any]] = []
        if not admission_failures:
            for event_id in priority:
                if event_id == chosen_reservation_event_id or event_id in evidence:
                    continue
                candidate = replace(
                    view,
                    gist_event_ids=_ordered_ids(
                        store, set(view.gist_event_ids) | {event_id}
                    ),
                    omitted_event_ids=tuple(
                        item for item in view.omitted_event_ids if item != event_id
                    ),
                )
                try:
                    candidate_measure = self._measure_view(
                        store, static_view, candidate, tools, max_new_tokens
                    )
                except PackingBudgetError as error:
                    skipped.append(
                        {
                            "event_id": event_id,
                            "reasons": ["packing_budget_exceeded"],
                            "detail": str(error),
                        }
                    )
                    continue
                if candidate_measure.reasons:
                    skipped.append(
                        {
                            "event_id": event_id,
                            "reasons": list(candidate_measure.reasons),
                        }
                    )
                    continue
                view = candidate
                measure = candidate_measure
                retained.append(event_id)

        history_bytes = max(
            row["history_bytes"] for row in measure.per_ratio.values()
        )
        selection_budget = min(
            self.policy_config.history_budget_bytes,
            self.policy_config.workspace_budget_bytes,
        )
        selection_cost = max(
            measure.evidence_bytes,
            max(
                0,
                history_bytes
                - self.policy_config.history_budget_bytes
                + selection_budget,
            ),
        )
        metadata = {
            "raw_control_layout": ALWAYS_COMPRESS_GIST_LAYOUT,
            "raw_event_ids": list(view.raw_event_ids),
            "gist_event_ids": list(view.gist_event_ids),
            "omitted_event_ids": list(view.omitted_event_ids),
            "mandatory_raw_event_ids": list(view.mandatory_raw_event_ids),
            "static_raw_event_ids": list(static_view.raw_event_ids),
            "evidence_event_ids": list(view.evidence_event_ids),
            "shared_evidence_event_ids": list(view.evidence_event_ids),
            "raw_source_indices": list(measure.memory.raw_source_indices),
            "full_source_indices": list(range(len(store.messages))),
            "full_source_coverage": False,
            "gist_refill_priority_event_ids": priority,
            "gist_refilled_event_ids": retained,
            "skipped_gist_refill_events": skipped,
            "min_gist_reservation_required": True,
            "min_gist_reservation_event_id": chosen_reservation_event_id,
            "min_gist_reservation_met": (
                chosen_reservation_event_id in view.gist_event_ids
            ),
            "min_gist_reservation_history_bytes": max(
                row["history_gist_tokens"] * self.kv_bytes_per_token
                for row in reservation_measure.per_ratio.values()
            ),
            "min_gist_reservation_skipped_events": reservation_skips,
            "event_priority_mapping": (
                "whole event: oldest frozen pre-E gist event first; then remaining "
                "frozen pre-E gist events newest-first"
            ),
            "legacy_1088_block_parity": False,
            "atomic_packing_unit": "whole_event_all_encoder_chunks",
            "selection_cost_bytes": selection_cost,
            "evidence_tokens": measure.evidence_tokens,
            "evidence_bytes": measure.evidence_bytes,
            "history_bytes": history_bytes,
            "sequence_tokens": max(
                row["sequence_tokens"] for row in measure.per_ratio.values()
            ),
            "logical_sequence_tokens": measure.logical_sequence_tokens,
            "per_ratio": copy.deepcopy(measure.per_ratio),
            "admission_failures": admission_failures,
            "reflow_logical_positions": True,
        }
        if packing_detail is not None:
            metadata["admission_failure_detail"] = packing_detail
        return _GistPlan(
            memory=measure.memory,
            metadata=metadata,
            selection_cost_bytes=selection_cost,
            admissible=not admission_failures,
        )

    @staticmethod
    def _gist_priority(store: EventStore, event_ids: Sequence[str]) -> list[str]:
        ordered = list(event_ids)
        if not ordered:
            return []
        return [ordered[0], *(
            event.event_id
            for event in sorted(
                (store.event(event_id) for event_id in ordered[1:]),
                key=lambda event: max(event.source_indices),
                reverse=True,
            )
        )]

    def _annotate_always_compress(
        self,
        store: EventStore,
        static_view: MemoryView,
        memory: PackedMemory,
        base: Mapping[str, Any],
        full_memory: PackedMemory,
        full_history_tokens: int,
        full_history_bytes: int,
        tools: tuple[dict[str, Any], ...],
        ratio: int,
    ) -> dict[str, Any]:
        metadata = copy.deepcopy(dict(base))
        if not self.always_compress:
            return metadata

        eligible_event_ids = tuple(static_view.gist_event_ids)
        eligible_sources = frozenset(
            index
            for event_id in eligible_event_ids
            for index in store.event(event_id).source_indices
        )

        def fragment(chunk: Any) -> dict[str, Any]:
            fragment_id = (
                f"{chunk.event_id}:{chunk.part_index}:"
                f"{chunk.source_token_start}:{chunk.source_token_end}"
            )
            return {
                "fragment_id": fragment_id,
                "packing_fragment_id": fragment_id,
                "event_id": chunk.event_id,
                "source_indices": list(chunk.source_indices),
                "encoder_input_tokens": len(chunk.token_ids),
            }

        packing_fragments = [
            fragment(chunk)
            for event_id in eligible_event_ids
            for chunk in encode_event_chunks(
                store,
                event_id,
                self.tokenizer,
                max_chunk_tokens=self.packing.max_chunk_tokens,
                chunk_overlap=self.packing.chunk_overlap,
            )
        ]
        retained_fragments = [fragment(chunk) for chunk in memory.chunks]
        coverage = coverage_accounting(
            eligible_sources=eligible_sources,
            raw_sources=set(memory.raw_source_indices),
            retained_blocks=retained_fragments,
            packing_fragments=packing_fragments,
        )
        retained_event_ids = set(memory.view.gist_event_ids) | set(
            memory.view.raw_event_ids
        )
        budget_omitted_event_ids = [
            event_id
            for event_id in eligible_event_ids
            if event_id not in retained_event_ids
        ]
        packing_skip_rows = [
            *metadata.get("min_gist_reservation_skipped_events", ()),
            *metadata.get("skipped_gist_refill_events", ()),
        ]
        native_packing_omitted = {
            row.get("event_id")
            for row in packing_skip_rows
            if "packing_budget_exceeded" in row.get("reasons", ())
        } & set(budget_omitted_event_ids)
        native_packing_omitted_event_ids = [
            event_id
            for event_id in eligible_event_ids
            if event_id in native_packing_omitted
        ]
        budget_omitted_event_ids = [
            event_id
            for event_id in budget_omitted_event_ids
            if event_id not in native_packing_omitted
        ]

        def source_indices(event_ids: Sequence[str]) -> list[int]:
            return sorted(
                {
                    index
                    for event_id in event_ids
                    for index in store.event(event_id).source_indices
                    if index in eligible_sources
                }
            )

        coverage.update(
            {
                "eligible_event_ids": list(eligible_event_ids),
                "eligibility_stage": (
                    "observable EventStore plus frozen select_view before "
                    "max_chunks and byte-budget admission"
                ),
                "raw_event_ids": [
                    event_id
                    for event_id in eligible_event_ids
                    if event_id in memory.view.raw_event_ids
                ],
                "gist_event_ids": [
                    event_id
                    for event_id in eligible_event_ids
                    if event_id in memory.view.gist_event_ids
                ],
                "raw_gist_overlap_event_ids": [
                    event_id
                    for event_id in eligible_event_ids
                    if event_id in memory.view.raw_event_ids
                    and event_id in memory.view.gist_event_ids
                ],
                "native_packing_omitted_event_ids": (
                    native_packing_omitted_event_ids
                ),
                "native_packing_omitted_source_indices": source_indices(
                    native_packing_omitted_event_ids
                ),
                "budget_omitted_event_ids": budget_omitted_event_ids,
                "budget_omitted_source_indices": source_indices(
                    budget_omitted_event_ids
                ),
                "omission_scope": (
                    "eligible source occurrences; distinct events are never merged by text"
                ),
            }
        )

        gist_tokens, raw_history_tokens = _history_components(
            store,
            memory.view,
            self.tokenizer,
            ratio=ratio,
            max_chunk_tokens=self.packing.max_chunk_tokens,
            chunk_overlap=self.packing.chunk_overlap,
            tools=tools,
        )
        active_history_tokens = gist_tokens + raw_history_tokens
        active_history_bytes = active_history_tokens * self.kv_bytes_per_token
        full_prompt_tokens = len(full_memory.system_input_ids) + len(
            full_memory.workspace_input_ids
        )
        common_live_tokens = max(0, full_prompt_tokens - full_history_tokens)
        common_live_bytes = common_live_tokens * self.kv_bytes_per_token
        compression_ratio = {
            "schema": "a-same-prefix-compression-ratio-v1",
            "denominator_source": (
                "Full renderer on this observable prefix; no Full generation"
            ),
            "full_history_bytes": full_history_bytes,
            "common_live_bytes": common_live_bytes,
            "active_history_bytes": active_history_bytes,
            "active_gist_bytes": gist_tokens * self.kv_bytes_per_token,
            "active_raw_history_bytes": (
                raw_history_tokens * self.kv_bytes_per_token
            ),
            "n_history": (
                full_history_bytes / active_history_bytes
                if full_history_bytes and active_history_bytes
                else None
            ),
            "n_total": (
                (common_live_bytes + full_history_bytes)
                / (common_live_bytes + active_history_bytes)
                if common_live_bytes + active_history_bytes
                else None
            ),
            "configured_gist_ratio": ratio,
            "actual_source_to_gist_ratio": (
                coverage["retained_encoder_input_tokens"] / gist_tokens
                if gist_tokens and coverage["retained_encoder_input_tokens"]
                is not None
                else None
            ),
            "actual_source_to_gist_scope": (
                "retained native encoder inputs including event-envelope templates"
            ),
            "includes_coverage_loss": not coverage["complete_history_coverage"],
        }
        metadata.update(
            {
                "route_mode": self.route_mode,
                "canonical_mode": self.mode,
                "compression_policy": ALWAYS_COMPRESSION_POLICY,
                "implementation_profile": NATIVE_ALWAYS_IMPLEMENTATION_PROFILE,
                "history_view_protocol": self.history_view_protocol,
                "no_eligible_history": not eligible_event_ids,
                "source_coverage": coverage,
                "compression_ratio": compression_ratio,
                "same_prefix_full_reference": {
                    "render_only": True,
                    "generation_performed": False,
                    "full_prompt_tokens": full_prompt_tokens,
                    "full_history_tokens": full_history_tokens,
                    "full_history_bytes": full_history_bytes,
                    "common_live_tokens": common_live_tokens,
                    "common_live_bytes": common_live_bytes,
                },
                "configured_gist_ratio": ratio,
                "actual_gist_tokens": gist_tokens,
                "actual_raw_history_tokens": raw_history_tokens,
                "actual_history_bytes": active_history_bytes,
                "full_source_coverage": coverage["complete_history_coverage"],
                "min_gist_reservation_required": self._requires_min_gist
                and bool(eligible_event_ids),
                "min_gist_reservation_met": (
                    bool(memory.view.gist_event_ids)
                    if self._requires_min_gist and eligible_event_ids
                    else None
                ),
            }
        )
        return metadata

    def _measure_view(
        self,
        store: EventStore,
        static_view: MemoryView,
        view: RuntimeMemoryView,
        tools: tuple[dict[str, Any], ...],
        max_new_tokens: int,
    ) -> _ViewMeasure:
        memory = pack_memory(
            store,
            view,
            self.tokenizer,
            tools=tools,
            max_chunk_tokens=self.packing.max_chunk_tokens,
            chunk_overlap=self.packing.chunk_overlap,
            max_chunks=self.packing.max_chunks,
        )
        reasons: list[str] = []
        system_tokens = len(memory.system_input_ids)
        workspace_tokens = len(memory.workspace_input_ids)
        encoder_tokens = sum(len(chunk.token_ids) for chunk in memory.chunks)
        if system_tokens > self.packing.max_system_tokens:
            reasons.append("system_budget")
        if workspace_tokens > self.packing.max_workspace_tokens:
            reasons.append("workspace_token_budget")
        if encoder_tokens > self.packing.max_encoder_tokens:
            reasons.append("encoder_budget")
        evidence_tokens = _evidence_increment_tokens(
            store,
            static_view,
            view,
            self.tokenizer,
            tools,
        )
        evidence_bytes = evidence_tokens * self.kv_bytes_per_token
        if evidence_bytes > self.policy_config.workspace_budget_bytes:
            reasons.append("workspace_byte_budget")
        logical_sequence = (
            memory.workspace_position_start
            + len(memory.workspace_input_ids)
            + max_new_tokens
        )
        if self.model_context is not None and logical_sequence > self.model_context:
            reasons.append("model_logical_context")

        per_ratio: dict[str, dict[str, int]] = {}
        for ratio in self.packing.ratios:
            costs = memory.costs(ratio)
            sequence_tokens = costs["resident_kv_tokens"] + max_new_tokens
            gist_tokens, raw_tokens = _history_components(
                store,
                view,
                self.tokenizer,
                ratio=ratio,
                max_chunk_tokens=self.packing.max_chunk_tokens,
                chunk_overlap=self.packing.chunk_overlap,
                tools=tools,
            )
            history_tokens = gist_tokens + raw_tokens
            history_bytes = history_tokens * self.kv_bytes_per_token
            if sequence_tokens > self.packing.max_sequence_tokens:
                reasons.append(f"physical_sequence_budget:{ratio}")
            if history_bytes > self.policy_config.history_budget_bytes:
                reasons.append(f"history_byte_budget:{ratio}")
            per_ratio[str(ratio)] = {
                **costs,
                "max_new_tokens": max_new_tokens,
                "sequence_tokens": sequence_tokens,
                "logical_sequence_tokens": logical_sequence,
                "history_gist_tokens": gist_tokens,
                "history_raw_tokens": raw_tokens,
                "history_total_tokens": history_tokens,
                "history_bytes": history_bytes,
                "evidence_tokens": evidence_tokens,
                "evidence_bytes": evidence_bytes,
                "history_budget_bytes": self.policy_config.history_budget_bytes,
                "workspace_budget_bytes": self.policy_config.workspace_budget_bytes,
            }
        return _ViewMeasure(
            memory=memory,
            per_ratio=per_ratio,
            evidence_tokens=evidence_tokens,
            evidence_bytes=evidence_bytes,
            logical_sequence_tokens=logical_sequence,
            reasons=tuple(dict.fromkeys(reasons)),
        )

    def _measure_full(
        self,
        store: EventStore,
        tools: tuple[dict[str, Any], ...],
        max_new_tokens: int,
    ) -> tuple[PackedMemory, int, int]:
        all_ids = tuple(event.event_id for event in store.events)
        view = MemoryView(gist_event_ids=(), raw_event_ids=all_ids)
        memory = pack_memory(store, view, self.tokenizer, tools=tools)
        _, raw_tokens = _history_components(
            store,
            view,
            self.tokenizer,
            ratio=min(self.packing.ratios),
            max_chunk_tokens=self.packing.max_chunk_tokens,
            chunk_overlap=self.packing.chunk_overlap,
            tools=tools,
        )
        return memory, raw_tokens, raw_tokens * self.kv_bytes_per_token

    def _compose_metadata(
        self,
        store: EventStore,
        decision_key: str,
        ratio: int,
        max_new_tokens: int,
        selection: Selection,
        base: Mapping[str, Any],
        capacity_gate: Mapping[str, Any],
        actual_visible_sources: frozenset[int],
        policy_visible_event_ids: frozenset[str],
        *,
        recovery_stage: str,
    ) -> dict[str, Any]:
        selection_metadata = copy.deepcopy(dict(selection.metadata))
        metadata = copy.deepcopy(dict(base))
        metadata.update(
            {
                "event_native_exact_version": EVENT_NATIVE_EXACT_VERSION,
                "event_native_policy_version": EVENT_NATIVE_POLICY_VERSION,
                "session_id": store.session_id,
                "decision_key": decision_key,
                "decision_index": selection_metadata["decision_index"],
                "mode": self.mode,
                "requested_ratio": ratio,
                "max_new_tokens": max_new_tokens,
                "kv_bytes_per_token": self.kv_bytes_per_token,
                "history_budget_bytes": self.policy_config.history_budget_bytes,
                "workspace_budget_bytes": self.policy_config.workspace_budget_bytes,
                "accounting_scope": (
                    "actual event-native tokenization with planned checkpoint KV "
                    "geometry; not measured HBM"
                ),
                "configured_max_sequence_tokens": self.packing.max_sequence_tokens,
                "effective_raw_max_sequence_tokens": min(
                    self.packing.max_sequence_tokens,
                    self.model_context
                    if self.model_context is not None
                    else self.packing.max_sequence_tokens,
                ),
                "model_context": self.model_context,
                "capacity_gate": copy.deepcopy(dict(capacity_gate)),
                "pre_draft_retrieval": False,
                "selected_event_ids": list(selection.selected_event_ids),
                "protected_event_ids": list(selection.protected_event_ids),
                "retrieved_event_ids": list(selection.retrieved_event_ids),
                "retained_event_ids": list(selection.retained_event_ids),
                "expired_lease_event_ids": list(
                    selection_metadata.get("expired_lease_event_ids", ())
                ),
                "revision_cancelled_event_ids": list(
                    selection_metadata.get("revision_cancelled_event_ids", ())
                ),
                "actual_visible_source_indices": sorted(actual_visible_sources),
                "policy_visible_event_ids": _ordered_ids(
                    store, policy_visible_event_ids
                ),
                "selection": selection_metadata,
                "recovery_stage": recovery_stage,
                "post_draft_exact_recovery_applied": (
                    recovery_stage == "post_draft_upgraded"
                ),
            }
        )
        return metadata

    def _require_model_context(
        self, memory: PackedMemory, max_new_tokens: int
    ) -> None:
        if self.model_context is None:
            return
        logical_sequence = (
            memory.workspace_position_start
            + len(memory.workspace_input_ids)
            + max_new_tokens
        )
        if logical_sequence > self.model_context:
            raise PackingBudgetError(
                f"Logical sequence needs {logical_sequence} positions; model context is "
                f"{self.model_context}"
            )

    def _validate_request(
        self,
        payload: Mapping[str, Any],
        ratio: int,
        max_new_tokens: int,
    ) -> tuple[
        str,
        str,
        EventStore,
        tuple[dict[str, Any], ...],
        str,
        tuple[str, ...],
    ]:
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        allowed = {"session_id", "decision_key", "messages", "tools"}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise PolicyInputError(
                f"Targets and privileged request fields are forbidden: {unknown!r}"
            )
        if not _positive_int(ratio) or ratio not in self.packing.ratios:
            raise ValueError(
                f"ratio must be one of checkpoint packing ratios {self.packing.ratios!r}"
            )
        if not _positive_int(max_new_tokens):
            raise ValueError("max_new_tokens must be a positive integer")
        if max_new_tokens > self.packing.max_target_tokens:
            raise PackingBudgetError(
                f"Generation needs up to {max_new_tokens} tokens; budget is "
                f"{self.packing.max_target_tokens}"
            )
        session_id = payload.get("session_id")
        decision_key = payload.get("decision_key")
        if not isinstance(session_id, str) or not session_id:
            raise PolicyInputError("An explicit nonempty session_id is required")
        if not isinstance(decision_key, str) or not decision_key:
            raise PolicyInputError("An explicit nonempty decision_key is required")
        messages = payload.get("messages")
        if (
            not isinstance(messages, Sequence)
            or isinstance(messages, (str, bytes, bytearray))
            or not messages
            or any(not isinstance(message, Mapping) for message in messages)
        ):
            raise PolicyInputError("messages must be a nonempty sequence of mappings")
        raw_tools = payload.get("tools", ())
        if raw_tools is None:
            raw_tools = ()
        if (
            not isinstance(raw_tools, Sequence)
            or isinstance(raw_tools, (str, bytes, bytearray))
            or any(not isinstance(tool, Mapping) for tool in raw_tools)
        ):
            raise PolicyInputError("tools must be a sequence of mappings")
        tools = tuple(_json_snapshot(tool) for tool in raw_tools)
        tools_json = _canonical_json(tools)
        store = EventStore.from_messages(session_id, messages)
        message_json = tuple(message.json_text for message in store.messages)
        return session_id, decision_key, store, tools, tools_json, message_json

    def _packing_mapping(self) -> dict[str, Any]:
        return {
            "ratios": list(self.packing.ratios),
            "recent_tool_events": self.packing.recent_tool_events,
            "max_chunk_tokens": self.packing.max_chunk_tokens,
            "chunk_overlap": self.packing.chunk_overlap,
            "max_chunks": self.packing.max_chunks,
            "max_encoder_tokens": self.packing.max_encoder_tokens,
            "max_system_tokens": self.packing.max_system_tokens,
            "max_workspace_tokens": self.packing.max_workspace_tokens,
            "max_target_tokens": self.packing.max_target_tokens,
            "max_sequence_tokens": self.packing.max_sequence_tokens,
        }

    def _raw_packing_mapping(self) -> dict[str, Any]:
        value = self._packing_mapping()
        if self.model_context is not None:
            value["max_sequence_tokens"] = min(
                value["max_sequence_tokens"], self.model_context
            )
        return value


def _ordered_ids(store: EventStore, event_ids: Sequence[str] | set[str]) -> tuple[str, ...]:
    selected = set(event_ids)
    known = {event.event_id for event in store.events}
    unknown = selected - known
    if unknown:
        raise PolicyInputError(
            f"Event is outside the observable prefix: {sorted(unknown)!r}"
        )
    return tuple(event.event_id for event in store.events if event.event_id in selected)


def _copy_reconsideration(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "regenerate": bool(value["regenerate"]),
        "memory": value["memory"],
        "metadata": copy.deepcopy(value["metadata"]),
        "decision": copy.deepcopy(value["decision"]),
    }


def _positive_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


__all__ = ["EventNativeExactController", "PreparedEventNativeExact"]
