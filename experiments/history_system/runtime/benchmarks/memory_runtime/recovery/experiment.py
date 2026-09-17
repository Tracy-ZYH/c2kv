"""G--P recovery using observed source units and append-only interventions."""
from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, replace
from types import SimpleNamespace

from history_memory.events import EventStore

from ..always_compress import CapacityInfeasible
from ..policy import PolicyInputError, _event_text
from .admission import metadata_after_admission
from .experiment_config import parse_gp_config
from .gate import canonical
from .orchestrator import EventNativeRecoveryController, _copy_result


@dataclass(frozen=True)
class Lease:
    unit: object
    decision_index: int
    user_turn: str


class GPRecoveryController(EventNativeRecoveryController):
    """Each decision may append new evidence, regenerate, then commit once.

    The archive is always the caller's observed prefix. Exact copies have only
    static protection; releasing them restores the base compression policy and
    makes their original source eligible for another recovery.
    """

    def __init__(self, base, detector_config, switches, *, backends=None):
        from .selection import prepare_selection_dependencies, load_candidate_scorer

        super().__init__(base, detector_config)
        self.gp = parse_gp_config(switches)
        self.max_recovery_rounds = self.gp["R"]
        self.set_protocol = self.gp.get("selection_protocol") == "evidence_sets_v1"
        self.trained_selector = None
        if self.set_protocol:
            from .local_selection_models import LocalSelectionModels
            self.backends = backends if backends is not None else LocalSelectionModels(self.gp.get("local_models") or {})
            if self.gp.get("selector_artifact"):
                from .set_models import load_set_selector, validate_selector_score_models
                self.trained_selector = load_set_selector(self.gp["selector_artifact"])
                expected = "reranker_calibrator" if self.gp.get("set_selector") == "reranker" else self.gp.get("set_selector")
                if self.trained_selector.kind != expected:
                    raise ValueError("selector artifact target differs from set_selector")
                if expected != "risk":
                    validate_selector_score_models(
                        self.trained_selector,
                        self.backends,
                        semantic_query_overflow_policy=self.gp.get(
                            "semantic_query_overflow_policy", "error"
                        ),
                    )
            if self.gp.get("set_selector") == "legacy_prefill":
                if self.config.get("gate") != "prefill_linear_head":
                    raise ValueError(
                        "legacy_prefill requires the frozen prefill_linear_head detector"
                    )
        else:
            self.backends = prepare_selection_dependencies(self.gp, backends)
        if self.gp["D"] == "supervised":
            self.gp["candidate_scorer"] = load_candidate_scorer(self.gp["candidate_scorer"])
        inner = base
        while hasattr(inner, "base"):
            inner = inner.base
        inner.encoding_scope = self.gp["G"]
        self._packer = inner
        self._leases = {}

    def prepare(self, payload, *, ratio, max_new_tokens):
        from .evidence_units import render_units

        clean_payload = dict(payload)
        explicit_turn = clean_payload.pop("user_turn_id", None)
        store = EventStore.from_messages(clean_payload["session_id"], clean_payload["messages"])
        turn = self._user_turn(clean_payload, store, explicit_turn)
        state = self._packer._sessions.get(store.session_id)
        next_index = state.decision_index + 1 if state else 1
        context = SimpleNamespace(metadata={"decision_index": next_index}, _gp_turn=turn)
        protected = [lease.unit for lease in self._leases.get(store.session_id, [])
                     if self._is_protected(lease, context)]
        protected_messages = render_units(protected, store, self.gp["P"], self.gp["order"]) if protected else ()
        cached = self._prepared.get((store.session_id, clean_payload["decision_key"]))
        if cached is not None and hasattr(cached, "_gp_protected_messages"):
            protected_messages = cached._gp_protected_messages
        self._packer.protected_recovery_messages = protected_messages
        reserve = self.gp.get("recovery_reserve_tokens", 0)
        original_policy = self._packer.policy_config
        reserve_bytes = reserve * self.kv_bytes_per_token
        if reserve_bytes:
            if reserve_bytes >= min(original_policy.history_budget_bytes, original_policy.workspace_budget_bytes):
                raise CapacityInfeasible("Recovery reservation consumes the entire history budget")
            self._packer.policy_config = replace(original_policy,
                history_budget_bytes=original_policy.history_budget_bytes - reserve_bytes,
                workspace_budget_bytes=original_policy.workspace_budget_bytes - reserve_bytes)
        try:
            prepared = super().prepare(clean_payload, ratio=ratio, max_new_tokens=max_new_tokens)
        finally:
            self._packer.policy_config = original_policy
        if hasattr(prepared, "_gp_round"):
            if prepared._gp_turn != turn:
                raise PolicyInputError("decision key reused with a different user turn")
            return prepared
        prepared._gp_round = 0
        prepared._gp_turn = turn
        prepared._gp_protected_messages = protected_messages
        prepared._gp_signature = None
        prepared._gp_base_checked = False
        prepared._gp_base_derived = tuple(
            message for message in prepared.metadata.get("derived_workspace_prefix_messages") or ()
            if message not in protected_messages)
        prepared._gp_visible = []
        if self.set_protocol:
            prepared._set_draft_logprobs = ()
            prepared.metadata["gp_reservation"] = {
                "requested_raw_tokens": reserve, "reserved_bytes": reserve_bytes,
                "first_draft_history_budget_bytes": original_policy.history_budget_bytes - reserve_bytes,
                "total_history_budget_bytes": original_policy.history_budget_bytes,
                "allocation_time": "before_first_draft", "append_evicts_existing_content": False}
            prepared.metadata["history_budget_bytes"] = original_policy.history_budget_bytes
            prepared.metadata["workspace_budget_bytes"] = original_policy.workspace_budget_bytes
            prepared.metadata["shared_allocation_budget_bytes"] = min(
                original_policy.history_budget_bytes, original_policy.workspace_budget_bytes)
            for measure in prepared.metadata.get("per_ratio", {}).values():
                measure["history_budget_bytes"] = original_policy.history_budget_bytes
                measure["workspace_budget_bytes"] = original_policy.workspace_budget_bytes
        prepared.metadata["gp_experiments"] = copy.deepcopy(self.gp)
        prepared.metadata["gp_recovery_limits"] = {
            "check_each_decision": True, "legacy_e1_quota_applied": False,
            "rounds_per_decision": self.max_recovery_rounds,
            "task_generation_limit": self.required_task_generation_limit,
        }
        prepared.metadata["route"].update(
            baseline_identity=prepared.metadata["route"]["baseline_identity"] + "+gp",
            max_generations_per_decision=1 + self.max_recovery_rounds,
            recovery_enabled=True,
        )
        sid = prepared._store.session_id
        retained, expired = [], []
        for lease in self._leases.get(sid, []):
            (retained if self._is_protected(lease, prepared) else expired).append(lease)
        self._leases[sid] = retained
        active = self._not_raw_visible([lease.unit for lease in retained], prepared)
        prepared.metadata["gp_lifecycle"] = {
            "policy": self.gp["L"], "user_turn": turn,
            "protected_unit_ids": [lease.unit.unit_id for lease in retained],
            "released_unit_ids": [lease.unit.unit_id for lease in expired],
            "released_sources_return_to_normal_compression": True,
            "archive_contains_recovery_copies": False,
        }
        if protected_messages:
            measure, receipt, derived = self._append_measure(prepared, active)
            if measure is None:
                raise CapacityInfeasible("Protected G--P evidence cannot fit the declared history budget")
            self._apply_measure(prepared, measure, receipt, derived)
            prepared._gp_visible = active
        return prepared

    def _is_protected(self, lease, prepared):
        rule = self.gp["L"]
        if rule == "task":
            return True
        if rule == "user_turn":
            return lease.user_turn == prepared._gp_turn
        if rule == "two_decisions":
            return prepared.metadata["decision_index"] <= lease.decision_index + 2
        return prepared.metadata["decision_index"] == lease.decision_index

    @staticmethod
    def _user_turn(payload, store, explicit):
        if explicit is not None:
            if not isinstance(explicit, (str, int)) or isinstance(explicit, bool):
                raise ValueError("user_turn_id must be a string or integer")
            return str(explicit)
        match = re.match(r"turn-([^/]+)/", payload["decision_key"])
        if match:
            return match.group(1)
        users = [event.event_id for event in store.events if event.kind == "user"]
        return users[-1] if users else "initial"

    def reconsider(self, prepared, draft_tool_calls, *, draft_text, parse_error=None):
        from .evidence_units import build_catalog, expand_units, deduplicate_units, unit_is_covered
        from .selection import select_candidates

        key = (prepared._store.session_id, prepared.metadata["decision_key"])
        if self._prepared.get(key) is not prepared:
            raise PolicyInputError("Prepared decision belongs to another G--P controller")
        signature = canonical([draft_tool_calls, draft_text, parse_error])
        if prepared._checked_result is not None:
            if signature != prepared._gp_signature:
                raise PolicyInputError("A recovery round cannot inspect two different drafts")
            return _copy_result(prepared._checked_result)
        if not prepared._gp_base_checked:
            self.base.reconsider(prepared._base_prepared, draft_tool_calls,
                                 draft_text=draft_text, parse_error=parse_error)
            prepared._gp_base_checked = True
        prepared._gp_signature = signature
        decision = {
            "version": self.gp["schema"], "status": "abstain", "reason": None,
            "decision_index": prepared.metadata["decision_index"],
            "recovery_round": prepared._gp_round + 1, "gate_type": self.gp["D"],
            "uses_gold_future_or_tool_result": False, "judges_action_correctness": False,
            "regeneration_allowed": False, "upgrade_count": 0,
            "selected_unit_count": 0, "appended_unit_count": 0,
        }
        if self.set_protocol:
            decision["candidate_supply"] = {
                "schema": "recovery-candidate-supply-v2", "stage": "not_evaluated",
                "n_archive_units": None, "n_retrieved": 0, "n_after_source_check": 0,
                "n_feasible": 0, "n_presented_to_selector": 0,
                "selected_ids": [], "actually_appended_ids": [], "rejection_reason": None,
            }
        if prepared._gp_round >= self.max_recovery_rounds:
            return self._finish_gp(prepared, decision, "recovery_round_limit")
        if self.set_protocol:
            return self._reconsider_sets(prepared, draft_tool_calls, draft_text, parse_error, decision)
        detector_modes = {"detector", "detector_llm", "candidate_or_detector"}
        calibration_telemetry = bool(
            self.gp.get("detector_calibration_telemetry", False)
        )
        detector_gate = None
        if self.gp["D"] in detector_modes or calibration_telemetry:
            detector_gate = self._detector_gate(prepared)
        if self.gp["D"] in {"detector", "detector_llm"}:
            decision["gate"] = detector_gate
            if not detector_gate["triggered"] and not calibration_telemetry:
                # Preserve the legacy detector path: an abstain does not build
                # a catalog or run any B0 feasibility probes by default.
                return self._finish_gp(prepared, decision, detector_gate["reason"])
        recovered = self._recovery_counts.get(prepared._store.session_id, 0)
        catalog = build_catalog(prepared._store, self.tokenizer, self.gp["U"])
        eligible = set(prepared.metadata["eligible_extraction"]["eligible_event_ids"])
        cancelled = set(prepared.metadata.get("revision_cancelled_event_ids") or ())
        candidates = self._not_raw_visible(
            [unit for unit in catalog if unit.event_id in eligible and unit.event_id not in cancelled], prepared)
        candidates = [unit for unit in candidates if not unit_is_covered(unit, prepared._gp_visible)]
        decision["candidate_availability"] = {
            "schema": "a-history-legal-candidate-availability-v1",
            "candidate_available": bool(candidates),
            "legal_candidate_count": len(candidates),
            "stage": "after_eligibility_raw_visibility_and_current_decision_dedup",
            "model_calls": 0,
        }
        if calibration_telemetry:
            candidate_feasibility = self._candidate_feasibility(
                prepared, candidates, catalog, cancelled
            )
            decision["candidate_feasibility"] = candidate_feasibility
            decision["calibration_telemetry"] = self._calibration_telemetry(
                detector_gate, candidate_feasibility
            )
        if self.gp["D"] in {"detector", "detector_llm"}:
            if not detector_gate["triggered"]:
                return self._finish_gp(prepared, decision, detector_gate["reason"])
        elif self.gp["D"] == "candidate_or_detector":
            decision["gate"] = self._candidate_or_detector_gate(
                detector_gate, bool(candidates)
            )
        if prepared.metadata["decision_index"] + recovered + 1 > self.required_task_generation_limit:
            return self._finish_gp(prepared, decision, "shared_task_generation_limit")
        if not candidates:
            return self._finish_gp(prepared, decision, "no_new_source_units")
        if self.gp["D"] == "candidate_rule":
            decision["gate"] = {"type": "candidate_rule", "triggered": True,
                                "reason": "available_source_units"}
        users = [event for event in prepared._store.events if event.kind == "user"]
        goal = _event_text(prepared._store, users[-1]) if users else ""
        selected, selection_receipt = select_candidates(
            candidates, goal=goal, draft_text=draft_text, draft_tool_calls=draft_tool_calls,
            config=self.gp, backends=self.backends)
        decision["selection"] = selection_receipt
        decision["selected_unit_ids"] = [unit.unit_id for unit in selected]
        decision["selected_unit_count"] = len(selected)
        if self.gp["D"] in {"joint_llm", "supervised"}:
            decision["gate"] = {"type": self.gp["D"], "triggered": bool(selected),
                                "reason": "nonempty_selected_set" if selected else "selector_empty_set"}
        if not selected:
            return self._finish_gp(prepared, decision, "selector_empty_set")
        expanded = expand_units(selected, catalog, prepared._store, self.gp["B"])
        expanded = self._not_raw_visible(expanded, prepared)
        expanded = [unit for unit in expanded if not unit_is_covered(unit, prepared._gp_visible)
                    and unit.event_id not in cancelled]
        expanded = deduplicate_units(expanded)
        if not expanded:
            return self._finish_gp(prepared, decision, "no_new_source_units")
        decision["proposed_unit_count"] = len(expanded)
        active = [*prepared._gp_visible, *expanded]
        measure, receipt, derived = self._append_measure(prepared, active)
        decision["allocation"] = receipt
        if measure is None:
            return self._finish_gp(prepared, decision, "selected_set_not_admitted_under_b0")
        self._apply_measure(prepared, measure, receipt, derived)
        prepared._gp_visible = active
        prepared._gp_round += 1
        self._recovery_counts[prepared._store.session_id] = recovered + 1
        leases = self._leases.setdefault(prepared._store.session_id, [])
        known = {lease.unit.unit_id for lease in leases}
        leases.extend(Lease(unit, prepared.metadata["decision_index"], prepared._gp_turn)
                      for unit in expanded if unit.unit_id not in known)
        prepared.metadata["gp_lifecycle"]["protected_unit_ids"] = [lease.unit.unit_id for lease in leases]
        decision.update(
            status="recover", reason="selected_source_units_appended",
            appended_unit_ids=[unit.unit_id for unit in expanded],
            appended_source_event_ids=list(dict.fromkeys(unit.event_id for unit in expanded)),
            selected_unit_count=len(selected), appended_unit_count=len(expanded),
            upgrade_count=len(expanded), regeneration_allowed=True,
            task_recovery_round_count=recovered + 1,
            appended_units=[unit.to_receipt() for unit in expanded],
        )
        return self._finish_gp(prepared, decision, decision["reason"], regenerate=True)

    def observe_selection_draft(self, *, session_id, decision_key, token_logprobs):
        """Bind the existing actor token scores without another forward pass."""
        if not self.set_protocol:
            return
        prepared = self._prepared.get((session_id, decision_key))
        if prepared is None:
            raise PolicyInputError("Selection features arrived before prepare")
        prepared._set_draft_logprobs = tuple(token_logprobs)

    def _reconsider_sets(self, prepared, draft_tool_calls, draft_text, parse_error, decision):
        from .set_protocol import context_from_prepared, build_legal_sets, context_digest
        from .set_retrieval import supply_candidates
        from .set_selectors import select_evidence_set, public_candidates

        recovered = self._recovery_counts.get(prepared._store.session_id, 0)
        if prepared.metadata["decision_index"] + recovered + 1 > self.required_task_generation_limit:
            return self._finish_gp(prepared, decision, "shared_task_generation_limit")
        context = context_from_prepared(prepared, draft_tool_calls, draft_text, parse_error)
        def admissible(units):
            measure, receipt, _ = self._append_measure(prepared, [*prepared._gp_visible, *units])
            return measure is not None, receipt
        candidates, supply = supply_candidates(prepared, self.tokenizer, self.gp, context,
            admissible, self.backends)
        actions, rejected_actions = build_legal_sets(candidates, admissible,
            maximum=min(self.gp.get("selector_max_units", self.gp["K"]), self.gp["K"]))
        legacy_gate = (
            self._detector_gate(prepared)
            if self.gp.get("set_selector") == "legacy_prefill"
            else None
        )
        chosen, selection = select_evidence_set(
            context,
            candidates,
            actions,
            self.gp,
            models=self.backends,
            tokenizer=self.tokenizer,
            trained=self.trained_selector,
            legacy_gate=legacy_gate,
        )
        supply.update(selected_ids=list(chosen), actually_appended_ids=[],
            rejection_reason=None if chosen else selection["reason"],
            legal_action_count=len(actions), rejected_actions=rejected_actions)
        gate = {
            "type": self.gp.get("set_selector", "candidate_rule"),
            "triggered": bool(chosen),
            "reason": selection["reason"],
            "legacy_risk_gate_applied": False,
        }
        if legacy_gate is not None:
            gate = copy.deepcopy(legacy_gate)
            gate.update(
                legacy_risk_gate_applied=True,
                selector="legacy_prefill",
                evidence_selected=bool(chosen),
            )
        decision.update(version="a-history-gp-evidence-sets-v1", candidate_supply=supply,
            selection=selection, selected_unit_ids=list(chosen), selected_unit_count=len(chosen),
            candidate_availability={"candidate_available": bool(candidates),
                "legal_candidate_count": len(candidates), "stage": "after_real_b0_feasibility"},
            gate=gate,
            context_sha256=context_digest(context))
        drain = getattr(self.backends, "drain_receipts", None)
        if callable(drain):
            decision["selection_model_calls"] = drain()
        if self.gp.get("export_selection_state", False):
            task_id = prepared._store.session_id
            if task_id.startswith("bfcl/"):
                task_id = task_id.split("/")[1]
            risk = self._detector_gate(prepared)
            score = risk.get("score", risk.get("value"))
            risk_bucket = "unknown" if score is None else "high" if risk["triggered"] else "low"
            decision["selection_state"] = {
                "schema": "t02-state-candidate-v1", "state_id": context_digest(context),
                "benchmark": "bfcl" if prepared._store.session_id.startswith("bfcl/") else "unknown",
                "task_id": task_id, "task_group_id": task_id,
                "decision_key": context["decision_key"], "draft_kind": "stop" if context["is_stop"] else "call",
                "set_selector": self.gp.get("set_selector", "candidate_rule"),
                "risk_bucket": risk_bucket, "observed_risk": risk, "q": context,
                "draft": {"text": draft_text, "tool_calls": list(draft_tool_calls), "parse_ok": parse_error is None},
                "candidates": [{**row, "candidate_id": row["unit_id"], "source_id": row["event_id"], "rank": rank}
                    for rank, row in enumerate(public_candidates(candidates), 1)],
                "allowed_actions": [{"action_id": index, "candidate_ids": list(action)} for index, action in enumerate(actions)],
                "selected_ids": list(chosen), "state_scope": "observations_only_not_backend_snapshot",
            }
            if self.gp.get("set_selector") == "local_llm":
                decision["selection_state"]["local_llm_selected_ids"] = list(chosen)
        if getattr(prepared, "_t02_hold_selection", False):
            prepared._t02_selection = (candidates, actions, copy.deepcopy(decision))
            return {"regenerate": False, "memory": prepared.memory,
                    "metadata": prepared.metadata, "decision": copy.deepcopy(decision)}
        return self._commit_evidence_set(prepared, candidates, chosen, decision)

    def hold_selection(self, prepared, draft_tool_calls, *, draft_text, parse_error=None):
        """Expose the actual legal catalog before any evidence is appended."""
        if not self.set_protocol or not self.gp.get("export_selection_state"):
            raise PolicyInputError("T02 collection requires exported evidence-set states")
        if prepared._checked_result is not None:
            raise PolicyInputError("Cannot hold an already committed recovery decision")
        prepared._t02_hold_selection = True
        try:
            result = self.reconsider(prepared, draft_tool_calls,
                draft_text=draft_text, parse_error=parse_error)
        finally:
            prepared._t02_hold_selection = False
        return result

    def commit_held_selection(self, prepared, candidate_ids):
        """Apply one frozen legal action without reranking or changing the draft."""
        if prepared._checked_result is not None:
            raise PolicyInputError("A held decision can only be committed once")
        frozen = getattr(prepared, "_t02_selection", None)
        if frozen is None:
            raise PolicyInputError("No held evidence-set catalog exists")
        candidates, actions, original = frozen
        chosen = tuple(candidate_ids)
        if chosen not in [tuple(action) for action in actions]:
            raise PolicyInputError("T02 intervention is outside the frozen legal actions")
        decision = copy.deepcopy(original)
        reason = "t02_forced_legal_set" if chosen else "t02_submit_original_draft"
        decision["selection"].update(selected_ids=list(chosen),
            action_id=[tuple(action) for action in actions].index(chosen), reason=reason,
            t02_intervention=True)
        decision.update(selected_unit_ids=list(chosen), selected_unit_count=len(chosen))
        decision["candidate_supply"].update(selected_ids=list(chosen),
            actually_appended_ids=[], rejection_reason=None if chosen else reason)
        decision["gate"].update(triggered=bool(chosen), reason=reason)
        return self._commit_evidence_set(prepared, candidates, chosen, decision)

    def _commit_evidence_set(self, prepared, candidates, chosen, decision):
        recovered = self._recovery_counts.get(prepared._store.session_id, 0)
        supply = decision["candidate_supply"]
        if not chosen:
            return self._finish_gp(prepared, decision, decision["selection"]["reason"])
        by_id = {row["unit_id"]: row["unit"] for row in candidates}
        selected = [by_id[identifier] for identifier in chosen]
        active = [*prepared._gp_visible, *selected]
        measure, allocation, derived = self._append_measure(prepared, active)
        decision["allocation"] = allocation
        if measure is None:
            # No actor state has changed. Admission is checked again at commit.
            supply["rejection_reason"] = "selected_set_not_admitted_under_b0"
            return self._finish_gp(prepared, decision, supply["rejection_reason"])
        self._apply_measure(prepared, measure, allocation, derived)
        prepared._gp_visible = active
        prepared._gp_round += 1
        self._recovery_counts[prepared._store.session_id] = recovered + 1
        leases = self._leases.setdefault(prepared._store.session_id, [])
        known = {lease.unit.unit_id for lease in leases}
        leases.extend(Lease(unit, prepared.metadata["decision_index"], prepared._gp_turn)
            for unit in selected if unit.unit_id not in known)
        prepared.metadata["gp_lifecycle"]["protected_unit_ids"] = [lease.unit.unit_id for lease in leases]
        supply["actually_appended_ids"] = list(chosen)
        decision.update(status="recover", reason="selected_source_units_appended", upgrade_count=len(selected),
            appended_unit_ids=list(chosen), appended_source_event_ids=list(dict.fromkeys(unit.event_id for unit in selected)),
            proposed_unit_count=len(selected), appended_unit_count=len(selected), regeneration_allowed=True,
            task_recovery_round_count=recovered + 1, appended_units=[unit.to_receipt() for unit in selected])
        return self._finish_gp(prepared, decision, decision["reason"], regenerate=True)

    def advance_recovery(self, prepared, *, shadow_features):
        if prepared._checked_result is None or not prepared._checked_result["regenerate"]:
            raise PolicyInputError("Only an admitted recovery may advance to a new draft")
        prepared._checked_result = None
        prepared._gp_signature = None
        prepared._shadow_features = copy.deepcopy(shadow_features)
        prepared._features_observed = True

    def _detector_gate(self, prepared):
        """Evaluate the frozen detector, optionally replacing only its threshold."""

        from .gate import evaluate_gate

        detector = copy.deepcopy(self.config)
        override = self.gp.get("detector_threshold")
        gate_type = detector["gate"]
        if gate_type == "disabled":
            if override is not None:
                raise ValueError("detector_threshold requires a threshold-based detector")
            return {"type": "disabled", "triggered": False, "reason": "recovery_disabled"}
        base_threshold = None
        if override is not None:
            threshold = float(override)
            if not math.isfinite(threshold):
                raise ValueError("detector_threshold must be finite")
            if gate_type == "first_name_margin":
                if threshold < 0:
                    raise ValueError("first_name_margin detector_threshold must be nonnegative")
                base_threshold = detector["margin_calibration"]["threshold"]
                detector["margin_calibration"]["threshold"] = threshold
            elif gate_type == "prefill_linear_head":
                if not 0 <= threshold <= 1:
                    raise ValueError("prefill detector_threshold must be between zero and one")
                base_threshold = detector["prefill_head"]["threshold"]
                detector["prefill_head"]["threshold"] = threshold
            else:
                raise ValueError("detector_threshold requires a threshold-based detector")
        result = evaluate_gate(
            detector,
            session_id=prepared._store.session_id,
            decision_key=prepared.metadata["decision_key"],
            shadow_features=prepared._shadow_features,
        )
        if override is not None:
            result["threshold_source"] = "gp_experiments.detector_threshold"
            result["base_detector_threshold"] = base_threshold
        return result

    def _candidate_feasibility(self, prepared, candidates, catalog, cancelled):
        """Measure whether any legal single candidate can pass the real B0 packer."""

        from .evidence_units import expand_units, deduplicate_units, unit_is_covered

        feasible_ids = []
        for candidate in candidates:
            expanded = expand_units([candidate], catalog, prepared._store, self.gp["B"])
            expanded = self._not_raw_visible(expanded, prepared)
            expanded = [
                unit
                for unit in expanded
                if not unit_is_covered(unit, prepared._gp_visible)
                and unit.event_id not in cancelled
            ]
            expanded = deduplicate_units(expanded)
            if not expanded:
                continue
            active = [*prepared._gp_visible, *expanded]
            measure, _receipt, _derived = self._append_measure(prepared, active)
            if measure is not None:
                feasible_ids.append(candidate.unit_id)
        return {
            "schema": "a-history-candidate-feasibility-v1",
            "candidate_available": bool(candidates),
            "legal_candidate_count": len(candidates),
            "candidate_feasible": bool(feasible_ids),
            "b0_admissible_candidate_count": len(feasible_ids),
            "b0_admissible_candidate_ids": feasible_ids,
            "availability_stage": (
                "after_eligibility_raw_visibility_and_current_decision_dedup"
            ),
            "feasibility_stage": "single_candidate_expansion_after_real_b0_admission",
            "admission_policy": "gp-append-only-b0-v1",
            "model_calls": 0,
        }

    @staticmethod
    def _candidate_or_detector_gate(detector_gate, candidate_feasible):
        candidate_condition = {
            "type": "candidate_rule",
            "triggered": candidate_feasible,
            "reason": "available_source_units" if candidate_feasible else "no_new_source_units",
        }
        raw_or = candidate_feasible or detector_gate["triggered"]
        append_triggered = candidate_feasible and raw_or
        return {
            "type": "candidate_or_detector",
            "triggered": append_triggered,
            "reason": "available_source_units" if append_triggered else "no_new_source_units",
            "composition": "candidate_rule_or_detector_with_legal_candidate_requirement",
            "raw_or_triggered": raw_or,
            "legal_candidate_required_for_append": True,
            "conditions": {
                "candidate_rule": candidate_condition,
                "detector": copy.deepcopy(detector_gate),
            },
            "append_action_equivalent_to_candidate_rule": True,
            "detector_can_change_append_decision": False,
            "equivalence_reason": (
                "candidate_rule already triggers for every state with a legal candidate"
            ),
        }

    @staticmethod
    def _calibration_telemetry(detector_gate, candidate_feasibility):
        score = detector_gate.get("score", detector_gate.get("value"))
        return {
            "schema": "a-history-detector-feasibility-observation-v1",
            "detector_type": detector_gate["type"],
            "detector_score": score,
            "detector_score_available": score is not None,
            "detector_direction": detector_gate.get("direction"),
            "detector_feature": detector_gate.get("feature"),
            "evaluated_threshold": detector_gate.get("threshold"),
            "candidate_available": candidate_feasibility["candidate_available"],
            "candidate_feasible": candidate_feasibility["candidate_feasible"],
            "legal_candidate_count": candidate_feasibility["legal_candidate_count"],
            "b0_admissible_candidate_count": candidate_feasibility[
                "b0_admissible_candidate_count"
            ],
            "feasibility_stage": candidate_feasibility["feasibility_stage"],
            "model_calls": 0,
            "uses_gold_future_or_tool_result": False,
        }

    @staticmethod
    def _not_raw_visible(units, prepared):
        raw_sources = set(prepared.memory.raw_source_indices)
        return [unit for unit in units if not set(unit.source_indices) <= raw_sources]

    def _append_measure(self, prepared, units):
        from .evidence_units import render_units

        rendered = render_units(units, prepared._store, self.gp["P"], self.gp["order"]) if units else ()
        derived = (*prepared._gp_base_derived, *rendered)
        view = prepared.memory.view
        protected_messages = getattr(self._packer, "protected_recovery_messages", ())
        self._packer.protected_recovery_messages = ()
        try:
            measure = self._packer._try_measure(
                prepared._store, prepared._tools, view.raw_event_ids, view.mandatory_raw_event_ids,
                view.gist_event_ids, prepared.metadata["eligible_extraction"]["eligible_event_ids"],
                prepared.metadata["common_raw_prompt_tokens"], prepared.metadata["max_new_tokens"],
                derived_messages=derived)
        finally:
            self._packer.protected_recovery_messages = protected_messages
        reasons = list(measure.reasons) if measure is not None else ["packing_limit"]
        if reasons:
            measure = None
        receipt = {
            "policy": "gp-append-only-b0-v1", "status": "admitted" if measure else "abstained",
            "unit_ids": [unit.unit_id for unit in units], "admission_failures": reasons,
            "raw_event_ids_unchanged": True, "gist_event_ids_unchanged": True,
            "b0_rechecked_after_all_changes": True, "all_wrappers_charged": True,
        }
        return measure, receipt, derived

    def _apply_measure(self, prepared, measure, receipt, derived):
        metadata = metadata_after_admission(self._packer, prepared, measure, None, receipt)
        metadata["derived_workspace_prefix_messages"] = copy.deepcopy(list(derived))
        metadata["recovery_stage"] = "post_draft_pre_tool_gp"
        # Spans are exact evidence but do not pretend to cover a full raw event.
        metadata["gp_exact_unit_ids"] = list(receipt["unit_ids"])
        prepared.memory, prepared.metadata = measure.memory, metadata

    @staticmethod
    def _finish_gp(prepared, decision, reason, *, regenerate=False):
        decision["reason"] = reason
        supply = decision.get("candidate_supply")
        if supply is not None and not regenerate:
            supply["rejection_reason"] = reason
        decision.setdefault("gate", {"type": decision["gate_type"], "triggered": False, "reason": reason})
        metadata = copy.deepcopy(prepared.metadata)
        metadata["exact_recovery"] = copy.deepcopy(decision)
        metadata["post_draft_exact_recovery_applied"] = regenerate
        result = {"regenerate": regenerate, "memory": prepared.memory,
                  "metadata": metadata, "decision": copy.deepcopy(decision)}
        prepared._checked_result = _copy_result(result)
        return result
