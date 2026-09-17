"""Shared native workspace allocation for C2KV and raw source-needs controls."""

from __future__ import annotations

import copy
import json
import time
from typing import Any

from .adapter import RuntimeAdapter, RUNTIME_VERSION, raw_source_cutoff
from .always_compress import (
    CapacityInfeasible, coverage_accounting, eligible_history_sources, ratio_accounting,
)
from .capacity import measure_full_history
from .dependency_packet import (
    TOP_LEVEL_SCHEMA_SLOT_PRIORITY,
    VERSION as DEPENDENCY_PACKET_VERSION,
    build_dependency_packet,
    drop_lowest_fact,
    fit_dependency_packet,
    packet_message,
)
from .policy import PolicyInputError
from .observed_state import STATE_VERSION, build_observed_state, fit_observed_state, state_message
from . import native_source_needs
from .source_needs import (
    NEEDS_VERSION, SourceRequest, build_needs_input, fit_prediction_input,
    lexical_source_ids, parse_source_request, prediction_messages,
)
from .tokenization import TOOL_SCHEMA_PROFILE
from .text_summary import VERSION as SUMMARY_VERSION
from history_memory.events import EventStore
from history_memory.subgoals import build_subgoal_ledger


SOURCE_NEEDS_ROUTES = {
    "ac_native_needs_lexical": ("c2kv", "lexical"),
    "ac_native_needs_lexical_fresh": ("c2kv", "lexical"),
    "ac_native_needs_lexical_pruned": ("c2kv", "lexical"),
    "ac_native_needs_lexical_narration": ("c2kv", "lexical"),
    "ac_native_needs_lexical_raw_reserve": ("c2kv", "lexical"),
    "ac_native_needs_lexical_raw_reserve_failed_operation": ("c2kv", "lexical"),
    "ac_native_dependency_packet_lexical_raw_reserve_failed_operation":
        ("c2kv", "lexical"),
    "ac_native_needs_typed": ("c2kv", "typed"),
    "raw_native_needs_lexical": ("raw", "lexical"),
    "raw_native_needs_lexical_fresh": ("raw", "lexical"),
    "raw_native_needs_lexical_raw_reserve_failed_operation": ("raw", "lexical"),
    "raw_native_needs_typed": ("raw", "typed"),
    "ac_native_needs_tool": ("c2kv", "tool"),
    "raw_native_needs_tool": ("raw", "tool"),
    "ac_native_needs_none": ("c2kv", "none"),
    "raw_native_needs_none": ("raw", "none"),
    "ac_native_state_none": ("c2kv", "none"),
    "raw_native_state_none": ("raw", "none"),
    "text_summary_native_needs_lexical_raw_reserve_failed_operation":
        ("text_summary", "lexical"),
}
STATE_ROUTES = frozenset({"ac_native_state_none", "raw_native_state_none"})
DEPENDENCY_PACKET_ROUTES = frozenset({
    "ac_native_dependency_packet_lexical_raw_reserve_failed_operation",
})
BOUNDED_LATEST_TOOL_ROUTES = frozenset({
    "ac_native_needs_lexical_raw_reserve_failed_operation",
    "ac_native_dependency_packet_lexical_raw_reserve_failed_operation",
    "raw_native_needs_lexical_raw_reserve_failed_operation",
    "text_summary_native_needs_lexical_raw_reserve_failed_operation",
})
LATEST_TOOL_PROTECTION_VERSION = "latest-complete-tool-protection-v1"
REQUESTED_FULL_EVENTS_SPARE_RAW_POLICY = "requested-full-events-v1"
DEPENDENCY_PACKET_SPARE_RAW_VERSION = "dependency-packet-spare-raw-v1"


def summary_coverage_accounting(*, eligible_sources, raw_sources,
                                retained_summaries, packing_fragments):
    """Report summary-input provenance without claiming semantic preservation."""
    fragments = list(packing_fragments)
    kept_ids = {record["packing_fragment_id"] for record in retained_summaries}
    candidates = set().union(*(set(row["source_indices"]) for row in fragments)) if fragments else set()
    touched = (set().union(*(set(row["source_indices"])
        for row in retained_summaries)) if retained_summaries else set()) & set(eligible_sources)
    fully_retained = set()
    for source_index in eligible_sources:
        relevant = [row for row in fragments if source_index in row["source_indices"]]
        if relevant and all(row["fragment_id"] in kept_ids for row in relevant):
            fully_retained.add(source_index)
    exact_raw = set(raw_sources) & set(eligible_sources)
    surviving = [row for row in fragments if row["fragment_id"] in kept_ids]
    packed_tokens = sum(int(row["encoder_input_tokens"]) for row in fragments)
    retained_tokens = sum(int(row["encoder_input_tokens"]) for row in surviving)
    return {
        "schema": "a-text-summary-source-provenance-v1",
        "representation": "text_summary",
        "packing_ledger_available": True,
        "eligible_source_indices": sorted(eligible_sources),
        "exact_raw_source_indices": sorted(exact_raw),
        "exact_raw_unrepresented_source_indices": sorted(set(eligible_sources) - exact_raw),
        "exact_raw_source_fraction": (len(exact_raw) / len(eligible_sources)
                                      if eligible_sources else None),
        "complete_exact_raw_coverage": set(eligible_sources) <= exact_raw,
        "summary_input_touched_source_indices": sorted(touched),
        "summary_input_fully_retained_source_indices": sorted(fully_retained),
        "summary_input_missing_source_indices": sorted(set(eligible_sources) - fully_retained),
        "raw_summary_input_overlap_source_indices": sorted(exact_raw & touched),
        "semantic_fidelity": "unknown",
        "semantically_represented_source_indices": None,
        "semantically_unrepresented_source_indices": None,
        "unrepresented_source_indices": None,
        "complete_history_coverage": None,
        "fully_represented_source_fraction": None,
        "native_unencoded_source_indices": sorted(set(eligible_sources) - candidates),
        "fitted_fragment_count": len(fragments),
        "retained_fragment_count": len(surviving),
        "fitted_encoder_input_tokens": packed_tokens,
        "retained_encoder_input_tokens": retained_tokens,
        "fitted_fragment_retained_token_fraction": (retained_tokens / packed_tokens
                                                     if packed_tokens else None),
        "token_weight_scope":
            "fitted summary encoder inputs including templates; not raw history tokens",
        "coverage_scope":
            "exact raw visibility and summary input provenance only; semantic coverage is unknown",
    }


class SourceNeedsRuntime(RuntimeAdapter):
    """One controller; representation-specific filling of remaining history.

    Goal, latest complete action/observation, pending events and the common
    suffix are protected. Up to two requested older events precede history
    filling. C2KV reserves a complete gist block and fills with gist; raw fills
    with complete original events. All admission uses the actual native view.
    """

    def __init__(self, config, token_counter):
        route = config["mode"]
        self.history_representation, self.source_needs_strategy = SOURCE_NEEDS_ROUTES[route]
        canonical = "ac_native_workspace" if self.history_representation == "c2kv" else "raw_recency"
        super().__init__({**config, "mode": canonical}, token_counter)
        self.route_mode = route
        self.history_organization = config.get("history_organization", "turn")
        self.actor_prompt_protocol = config.get("actor_prompt_protocol")
        if self.history_organization not in {"turn", "subgoal-v1"}:
            raise ValueError("Unknown history organization")
        subgoal_actor_protocols = {
            "native-subgoal-note-v1", "native-subgoal-note-v2"}
        if self.actor_prompt_protocol not in {None, *subgoal_actor_protocols}:
            raise ValueError("Unknown actor prompt protocol")
        if self.history_organization == "subgoal-v1":
            if (self.history_representation != "c2kv"
                    or self.actor_prompt_protocol not in subgoal_actor_protocols):
                raise ValueError("Subgoal organization requires C2KV and its actor protocol")
        proposal_key = "typed_subgoal_proposal_policy"
        self.typed_subgoal_proposal_policy = config.get(proposal_key)
        if self.typed_subgoal_proposal_policy not in {None, "post-action-internal-v1"}:
            raise ValueError("Unknown typed subgoal proposal policy")
        if self.typed_subgoal_proposal_policy is not None and (
                self.history_organization != "subgoal-v1"
                or self.actor_prompt_protocol != "native-subgoal-note-v2"):
            raise ValueError(
                "Typed subgoal proposal requires subgoal-v1 and native-subgoal-note-v2")
        if self.history_representation == "text_summary":
            keys = ("summary_model", "summary_prompt_token_cap", "summary_attempts_per_task")
            if any(key not in config for key in keys):
                raise ValueError("Text-summary route requires its explicit model and resource contract")
            self.summary_config = {key: config[key] for key in keys}
        self.observed_state_enabled = route in STATE_ROUTES
        self.dependency_packet_enabled = route in DEPENDENCY_PACKET_ROUTES
        packet_priority_key = "dependency_packet_field_priority_policy"
        self.dependency_packet_field_priority_policy = config.get(packet_priority_key)
        if (packet_priority_key in config and not self.dependency_packet_enabled):
            raise ValueError(
                "Dependency-packet field priority requires a dependency-packet route")
        if self.dependency_packet_field_priority_policy not in {
                None, TOP_LEVEL_SCHEMA_SLOT_PRIORITY}:
            raise ValueError("Unknown dependency-packet field priority policy")
        spare_raw_key = "dependency_packet_spare_raw_policy"
        self.dependency_packet_spare_raw_policy = config.get(spare_raw_key)
        if spare_raw_key in config and not self.dependency_packet_enabled:
            raise ValueError(
                "Dependency-packet spare raw requires a dependency-packet route")
        if self.dependency_packet_spare_raw_policy not in {
                None, REQUESTED_FULL_EVENTS_SPARE_RAW_POLICY}:
            raise ValueError("Unknown dependency-packet spare raw policy")
        self.state_prompt_token_cap = config.get("state_prompt_token_cap")
        if self.observed_state_enabled and (type(self.state_prompt_token_cap) is not int
                                            or self.state_prompt_token_cap <= 0):
            raise ValueError("Observed-state routes require an explicit positive state prompt cap")
        if self.dependency_packet_enabled:
            packet_keys = (
                "dependency_packet_prompt_token_cap", "dependency_packet_max_entities",
                "dependency_packet_max_fields", "dependency_packet_max_atom_chars",
                "dependency_packet_max_arguments_chars",
            )
            if any(type(config.get(key)) is not int or config[key] <= 0 for key in packet_keys):
                raise ValueError(
                    "Dependency-packet routes require explicit positive integer limits")
            self.dependency_packet_config = {key: config[key] for key in packet_keys}
            if packet_priority_key in config:
                self.dependency_packet_config[packet_priority_key] = (
                    self.dependency_packet_field_priority_policy)
        if not self.always_compress or self.history_view_protocol != "fixed-budget-main":
            raise ValueError("Source-needs routes require always-compress-v1 and fixed-budget-main")
        if config.get("max_retrieved_events") != 2 or config.get("lease_decisions") != 0:
            raise ValueError("Source-needs v1 requires two source slots and no cross-decision lease")
        self.source_index_max_events = config["source_index_max_events"]
        self.predictor_prompt_token_cap = config["predictor_prompt_token_cap"]
        self.predictor_completion_token_cap = config["predictor_completion_token_cap"]
        self.latest_complete_tool_protection = config.get(
            "latest_complete_tool_protection", "required")
        self._latest_complete_tool_protection_explicit = (
            "latest_complete_tool_protection" in config)
        if self.latest_complete_tool_protection not in {"required", "budgeted"}:
            raise ValueError("Unknown latest_complete_tool_protection policy")
        if (self.latest_complete_tool_protection == "budgeted"
                and route not in BOUNDED_LATEST_TOOL_ROUTES):
            raise ValueError(
                "Budgeted latest-complete-tool protection requires a dedicated "
                "raw-reserve failed-operation route")
        if type(self.source_index_max_events) is not int or not 0 < self.source_index_max_events <= 12:
            raise ValueError("Source index must contain between one and twelve events")
        for value in (self.predictor_prompt_token_cap, self.predictor_completion_token_cap):
            if type(value) is not int or value <= 0:
                raise ValueError("Predictor caps must be positive integers")
        self._observed_prefixes = {}

    def apply(self, messages, assembled, counts, eval_context, tools=None, *,
              render_full=None, render_compressed=None, render_summary=None,
              source_predictor=None):
        with self._lock:
            output, updated = self._apply_needs(messages, assembled, counts, eval_context, tools,
                                                render_compressed, source_predictor,
                                                render_summary=render_summary)
            if self.route_mode == "ac_native_needs_lexical_pruned":
                from .duplicate_gist import prune_duplicate_gist
                started = time.perf_counter()
                output, updated = prune_duplicate_gist(output, updated, tools, self._token_counter)
                elapsed = time.perf_counter() - started
                updated["memory_runtime"]["gist_pruning"]["controller_wall_sec"] = elapsed
                updated["memory_runtime"]["controller_wall_sec"] += elapsed
            if self.route_mode == "ac_native_needs_lexical_narration":
                from .raw_narration import prune_older_narration
                started = time.perf_counter()
                output, updated = prune_older_narration(
                    messages, output, updated, tools, self._token_counter)
                elapsed = time.perf_counter() - started
                updated["memory_runtime"]["raw_narration"]["controller_wall_sec"] = elapsed
                updated["memory_runtime"]["controller_wall_sec"] += elapsed
            if self.route_mode in {"ac_native_needs_lexical_raw_reserve",
                                   "ac_native_needs_lexical_raw_reserve_failed_operation",
                                   "ac_native_dependency_packet_lexical_raw_reserve_failed_operation",
                                   "raw_native_needs_lexical_raw_reserve_failed_operation",
                                   "text_summary_native_needs_lexical_raw_reserve_failed_operation"}:
                from .raw_reserve import reserve_older_raw
                started = time.perf_counter()
                output, updated = reserve_older_raw(
                    messages, output, updated, tools, self._token_counter, assembled)
                elapsed = time.perf_counter() - started
                updated["memory_runtime"]["raw_reserve"]["controller_wall_sec"] = elapsed
                updated["memory_runtime"]["controller_wall_sec"] += elapsed
            if self.route_mode in {"ac_native_needs_lexical_raw_reserve_failed_operation",
                                   "ac_native_dependency_packet_lexical_raw_reserve_failed_operation",
                                   "raw_native_needs_lexical_raw_reserve_failed_operation",
                                   "text_summary_native_needs_lexical_raw_reserve_failed_operation"}:
                from .failed_operation import apply_condition
                started = time.perf_counter()
                names = [tool["function"]["name"] for tool in tools or []]
                if len(names) != len(set(names)):
                    raise PolicyInputError("Failed-operation observations require unique tool names")
                store = EventStore.from_messages(updated["memory_runtime"]["task_id"], messages)
                output, updated = apply_condition(output, updated, store, tools,
                    self._token_counter, "failed_operation_cue")
                elapsed = time.perf_counter() - started
                updated["memory_runtime"]["failed_operation_cue"]["controller_wall_sec"] = elapsed
                updated["memory_runtime"]["controller_wall_sec"] += elapsed
            if self.dependency_packet_spare_raw_policy is not None:
                started = time.perf_counter()
                output, updated = self._apply_dependency_packet_spare_raw(
                    messages, output, updated, tools, assembled)
                elapsed = time.perf_counter() - started
                updated["memory_runtime"]["dependency_packet_spare_raw"][
                    "controller_wall_sec"] = elapsed
                updated["memory_runtime"]["controller_wall_sec"] += elapsed
            return output, updated

    def _apply_dependency_packet_spare_raw(
            self, source, output, counts, tools, full_messages):
        """Spend only final-view spare B0 on requested complete raw events.

        This runs after the existing raw-reserve and failed-operation stages so
        their exact messages remain present.  It rebuilds only the contiguous
        native history region from the same-prefix Full renderer.
        """
        if (not self.dependency_packet_enabled
                or self.dependency_packet_spare_raw_policy
                != REQUESTED_FULL_EVENTS_SPARE_RAW_POLICY):
            raise PolicyInputError("Spare raw requires its explicit dependency-packet policy")
        out, updated = copy.deepcopy(output), copy.deepcopy(counts)
        meta = updated["memory_runtime"]
        workspace = meta["pre_generation_workspace"]
        source_cutoff = raw_source_cutoff(source)
        shift = len(full_messages) - len(source)
        if shift not in (0, 1):
            raise PolicyInputError("Unexpected same-prefix Full source shift")

        selected = set(meta["selected_source_indices"])
        restored = list(workspace["restored_source_indices"])
        positions = list(workspace["native_workspace_out_indices"])
        current_end = int(updated["current_start_out_index"])
        original_current_end = current_end
        current_start = min(positions) if positions else current_end
        if positions != list(range(current_start, current_end)):
            raise PolicyInputError("Native workspace is not one contiguous source-ordered region")

        def native_source_indices(indices):
            return [index for index in sorted(indices)
                    if index < source_cutoff and source[index].get("role") != "system"]

        def native_rows(indices):
            return [dict(full_messages[index + shift])
                    for index in native_source_indices(indices)]

        expected_restored = native_source_indices(selected)
        if restored != expected_restored or out[current_start:current_end] != native_rows(selected):
            raise PolicyInputError("Final native workspace disagrees with its source ledger")
        if any(message.get("c2kv_key_hash") for message in full_messages):
            raise PolicyInputError("Spare raw requires a raw same-prefix Full renderer")

        common_tokens = int(meta["common_raw_prompt_tokens"])
        gist_tokens = int(meta["gist_tokens"])
        unit = int(meta["bytes_per_kv_token"])
        budget = min(int(meta["history_budget_bytes"]), int(meta["workspace_budget_bytes"]))

        def count_and_active(view):
            raw_tokens = self._token_counter(
                [message for message in view if not message.get("c2kv_key_hash")], tools)
            active = (raw_tokens - common_tokens + gist_tokens) * unit
            return raw_tokens, active

        current_tokens, current_active = count_and_active(out)
        if (current_tokens != int(meta["total_raw_prompt_tokens"])
                or current_active != int(meta["active_history_bytes"])
                or current_active > budget):
            raise PolicyInputError("Final S1 view disagrees with its exact B0 ledger")

        store = EventStore.from_messages(meta["task_id"], source)
        eligible = set(meta["source_coverage"]["eligible_source_indices"])
        requested_ids = list(meta["source_needs"]["requested_event_ids"])
        if len(requested_ids) != len(set(requested_ids)):
            raise PolicyInputError("Requested spare-raw event IDs must be unique")
        represented = set((meta.get("dependency_packet") or {}).get(
            "represented_source_ids") or [])
        items, added_event_ids = [], []
        original_selected = set(selected)
        for request_rank, event_id in enumerate(requested_ids):
            event = store.event(event_id)
            event_sources = set(event.source_indices)
            if (not event.complete or event.kind == "instruction"
                    or not event_sources or not event_sources <= eligible):
                raise PolicyInputError(
                    "Requested spare-raw event is not complete eligible history")
            before = current_active
            item = {
                "request_rank": request_rank,
                "event_id": event.event_id,
                "source_indices": list(event.source_indices),
                "active_history_bytes_before": before,
                "budget_bytes": budget,
                "packet_represented": event.event_id in represented,
            }
            if event_sources <= selected:
                item.update(
                    status="already_raw", added_source_indices=[],
                    candidate_active_history_bytes=before,
                    active_history_bytes_after=before,
                    incremental_raw_tokens=0, incremental_bytes=0)
                items.append(item)
                continue

            trial_selected = selected | event_sources
            trial_native = native_rows(trial_selected)
            trial = (copy.deepcopy(out[:current_start]) + trial_native
                     + copy.deepcopy(out[current_end:]))
            trial_tokens, trial_active = count_and_active(trial)
            delta_tokens = trial_tokens - current_tokens
            delta_bytes = trial_active - current_active
            added_sources = sorted(event_sources - selected)
            if (not added_sources or delta_tokens <= 0
                    or delta_bytes != delta_tokens * unit):
                raise PolicyInputError("Complete spare-raw event did not add exact raw tokens")
            item.update(
                added_source_indices=added_sources,
                candidate_active_history_bytes=trial_active,
                incremental_raw_tokens=delta_tokens,
                incremental_bytes=delta_bytes)
            if trial_active > budget:
                item.update(status="over_budget", active_history_bytes_after=before)
                items.append(item)
                continue
            item.update(status="added", active_history_bytes_after=trial_active)
            items.append(item)
            added_event_ids.append(event.event_id)
            out, selected = trial, trial_selected
            current_tokens, current_active = trial_tokens, trial_active
            current_end = current_start + len(trial_native)

        final_restored = native_source_indices(selected)
        if out[current_start:current_end] != native_rows(selected):
            raise PolicyInputError("Spare-raw output lost native source order")
        if (out[:current_start] != output[:current_start]
                or out[current_end:] != output[original_current_end:]):
            raise PolicyInputError("Spare raw changed a non-native final-view message")
        if ([message for message in out if message.get("c2kv_key_hash")]
                != [message for message in output if message.get("c2kv_key_hash")]):
            raise PolicyInputError("Spare raw changed retained gist messages")

        packing_fragments = (updated.get("history_packing_fragments")
                             if meta["block_refs"] else [])
        coverage = coverage_accounting(
            eligible_sources=frozenset(eligible), raw_sources=selected,
            retained_blocks=meta["block_refs"], packing_fragments=packing_fragments)
        raw_requested = {event_id for event_id in requested_ids
                         if set(store.event(event_id).source_indices) <= selected}
        admitted = [event_id for event_id in requested_ids
                    if event_id in represented or event_id in raw_requested]
        item_by_id = {item["event_id"]: item for item in items}
        skipped = []
        for event_id in requested_ids:
            if event_id in admitted:
                continue
            item = item_by_id[event_id]
            if item["status"] != "over_budget":
                raise PolicyInputError("Requested event has no honest admission outcome")
            skipped.append({
                "event_id": event_id,
                "reason": "dependency_packet_no_retained_fact_and_requested_full_event_budget",
                "candidate_history_bytes": item["candidate_active_history_bytes"],
            })

        added_sources = sorted(selected - original_selected)
        extra_tokens = current_tokens - int(meta["total_raw_prompt_tokens"])
        extra_bytes = current_active - int(meta["active_history_bytes"])
        if extra_bytes != extra_tokens * unit:
            raise PolicyInputError("Spare-raw byte geometry mismatch")
        meta.update(
            selected_source_indices=sorted(selected),
            selected_event_ids=[event.event_id for event in store.events
                                if set(event.source_indices) <= selected],
            retrieved_event_ids=admitted,
            source_coverage=coverage,
            total_raw_prompt_tokens=current_tokens,
            raw_history_tokens=current_tokens - common_tokens,
            active_history_bytes=current_active,
            evidence_bytes=(current_tokens - common_tokens) * unit,
            byte_geometry_verified_by_backend=False,
        )
        if "native_evidence_bytes" in meta:
            meta["native_evidence_bytes"] += extra_bytes
        if "raw_prompt_tokens_verified_by_backend" in meta:
            meta["raw_prompt_tokens_verified_by_backend"] = False
        workspace.update(
            restored_source_indices=final_restored,
            native_workspace_out_indices=list(range(current_start, current_end)))
        updated["current_start_out_index"] = current_end
        updated["history_raw"] += len(final_restored) - len(restored)
        meta["source_needs"].update(
            admitted_event_ids=admitted,
            skipped_for_budget=skipped,
            admission_rule="existing_final_view_then_requested_full_events_from_spare_budget",
            admission_cost="actual_final_view_raw_tokens_plus_retained_gist")
        meta["dependency_packet_spare_raw"] = {
            "version": DEPENDENCY_PACKET_SPARE_RAW_VERSION,
            "policy": self.dependency_packet_spare_raw_policy,
            "selection_order": "recorded requested_event_ids order",
            "source_scope": "complete eligible EventStore history in the observable prefix",
            "budget_bytes": budget,
            "original_selected_source_indices": sorted(original_selected),
            "final_selected_source_indices": sorted(selected),
            "added_event_ids": added_event_ids,
            "added_source_indices": added_sources,
            "incremental_raw_tokens": extra_tokens,
            "incremental_bytes": extra_bytes,
            "active_history_bytes_before": int(counts["memory_runtime"]["active_history_bytes"]),
            "active_history_bytes_after": current_active,
            "items": items,
            "existing_gist_packet_state_raw_and_failed_cue_unchanged": True,
            "uses_gold_future_s0_or_hidden_state": False,
        }
        meta["compression_ratio"] = ratio_accounting(
            {"active_history_bytes": meta["compression_ratio"]["full_history_bytes"],
             "common_raw_prompt_tokens": common_tokens}, meta)
        meta["compression_ratio"]["includes_coverage_loss"] = bool(
            coverage["unrepresented_source_indices"])
        if current_active > budget:
            raise PolicyInputError("Spare raw exceeded the exact final B0 budget")
        return out, updated

    def _apply_needs(self, messages, full_messages, full_counts, context, tools,
                     render_compressed, source_predictor, *, render_summary=None,
                     source_request_override=None):
        started = time.perf_counter()
        if not isinstance(context, dict) or any(
                key in context for key in ("gold", "oracle", "target_action", "hidden_state")):
            raise PolicyInputError("Source-needs requires observable benchmark context")
        task_id = context.get("task_id")
        attempt = context.get("attempt_id", context.get("attempt"))
        if not isinstance(task_id, str) or not task_id or attempt is None:
            raise PolicyInputError("Require explicit task_id and attempt")
        if context.get("run_id", self.run_id) != self.run_id:
            raise PolicyInputError("Source-needs run identity mismatch")
        decision = context.get("decision_id")
        if decision is None:
            if not {"user_turn", "step"} <= context.keys():
                raise PolicyInputError("Require decision_id or user_turn and step")
            decision = json.dumps([context["user_turn"], context["step"]])
        state_key = (self.run_id, task_id, str(attempt))
        signature = tuple(json.dumps(m, sort_keys=True, ensure_ascii=False) for m in messages)
        previous = self._observed_prefixes.get(state_key, ())
        if signature[:len(previous)] != previous:
            raise PolicyInputError("Source-needs input must extend its observed prefix")
        store = EventStore.from_messages(task_id, messages)
        if any(message.get("role") == "developer" for message in messages):
            raise PolicyInputError("Source-needs v1 supports the legacy system/user/tool profile")
        source_cutoff = raw_source_cutoff(messages)
        full_cutoff = int(full_counts["current_start_out_index"])
        shift = len(full_messages) - len(messages)
        if shift not in (0, 1) or full_cutoff != source_cutoff + shift:
            raise PolicyInputError("Full renderer did not preserve source row boundaries")
        if any(message.get("c2kv_key_hash") for message in full_messages):
            raise PolicyInputError("Source-needs requires the same-prefix raw Full reference")
        common_prefix = [dict(m) for m in full_messages[:full_cutoff] if m.get("role") == "system"]
        suffix = [dict(m) for m in full_messages[full_cutoff:]]
        common = common_prefix + suffix
        common_sources = set(range(source_cutoff, len(messages))) | {
            i for event in store.events if event.kind == "instruction" for i in event.source_indices}
        full = measure_full_history(full_messages, full_counts, self._token_counter,
                                    tools, self.bytes_per_kv_token)
        common_tokens = self._token_counter(common, tools)
        if common_tokens != full["common_raw_prompt_tokens"]:
            raise PolicyInputError("Source-needs changed the Full common-input boundary")
        eligible = eligible_history_sources(store, source_cutoff)
        users = [event for event in store.events if event.kind == "user"]
        complete_tools = [event for event in store.events if event.kind == "tool_event" and event.complete]
        latest_complete_tool = complete_tools[-1] if complete_tools else None
        protected_ids = {event.event_id for event in store.events
                         if not event.complete or set(event.source_indices) & set(range(source_cutoff, len(messages)))}
        if users:
            protected_ids.add(users[-1].event_id)
        if latest_complete_tool:
            protected_ids.add(latest_complete_tool.event_id)
        selected_sources = common_sources | {
            i for event in store.events if event.event_id in protected_ids for i in event.source_indices}

        def select_sources(current_selected_sources, *, recent_tool_event_visible=True):
            current_visible_ids = {event.event_id for event in store.events
                                   if set(event.source_indices) <= current_selected_sources}
            needs_input = build_needs_input(
                store, tools or [], excluded_event_ids=current_visible_ids,
                max_candidates=self.source_index_max_events,
                recent_tool_event_visible=recent_tool_event_visible,
                include_user_events=self.dependency_packet_enabled)
            fit_input = (native_source_needs.fit_prediction_input
                         if self.source_needs_strategy == "tool" else fit_prediction_input)
            current_fitted, current_receipt = fit_input(
                needs_input, self._token_counter,
                max_prompt_tokens=self.predictor_prompt_token_cap)
            current_request = SourceRequest((), (), current_receipt["status"])
            current_freshness = None
            current_prediction_wall_sec = 0.0
            if current_fitted is not None and current_receipt["status"] == "ready":
                if self.source_needs_strategy == "lexical":
                    current_request = SourceRequest(
                        lexical_source_ids(store, current_fitted), (), "lexical_sources_ranked")
                    if self.route_mode.endswith("_lexical_fresh"):
                        from .source_freshness import refresh_source_ids
                        ids, current_freshness = refresh_source_ids(
                            store, current_request.source_ids,
                            [entry["source_id"] for entry in current_fitted["index"]])
                        current_request = SourceRequest(
                            ids, (), "lexical_sources_refreshed")
                elif self.source_needs_strategy == "typed":
                    if not callable(source_predictor):
                        raise PolicyInputError(
                            "Typed source-needs requires an accounted predictor callback")
                    prediction_started = time.perf_counter()
                    prediction = source_predictor(
                        prediction_messages(current_fitted), current_receipt["prompt_tokens"])
                    current_prediction_wall_sec = time.perf_counter() - prediction_started
                    current_request = parse_source_request(
                        None if prediction.get("tool_calls") else prediction.get("content"),
                        current_fitted)
                elif self.source_needs_strategy == "tool":
                    if not callable(source_predictor):
                        raise PolicyInputError(
                            "Tool source-needs requires an accounted predictor callback")
                    prediction_started = time.perf_counter()
                    prediction = source_predictor(
                        native_source_needs.prediction_messages(current_fitted),
                        current_receipt["prompt_tokens"],
                        tools=native_source_needs.prediction_tools(current_fitted))
                    current_prediction_wall_sec = time.perf_counter() - prediction_started
                    current_request = native_source_needs.parse_prediction(
                        prediction, current_fitted)
                else:
                    current_request = SourceRequest((), (), "no_retrieval_control")
            return (current_visible_ids, current_fitted, current_receipt, current_request,
                    current_freshness, current_prediction_wall_sec)

        (visible_ids, fitted, input_receipt, requested,
         freshness_receipt, prediction_wall_sec) = select_sources(selected_sources)

        if source_request_override is not None:
            if not getattr(self, "supports_actor_evidence", False):
                raise PolicyInputError("Explicit source requests require the actor-evidence route")
            if not isinstance(source_request_override, SourceRequest):
                raise PolicyInputError("Explicit source request must already be parsed")
            ids = source_request_override.source_ids
            allowed = {event.event_id for event in complete_tools
                       if all(index < source_cutoff for index in event.source_indices)}
            if not 0 < len(ids) <= 2 or len(set(ids)) != len(ids) or not set(ids) <= allowed:
                raise PolicyInputError("Explicit source request contains unavailable events")
            requested = source_request_override
            fitted = {"index": [{"source_id": event_id} for event_id in ids]}
            input_receipt = {"status": "actor_request_validated", "prompt_tokens": 0,
                "scope": "Request pool and model cost are recorded by actor_evidence"}

        compressed_counts, records = full_counts, []
        summary_result = None
        compressed_assembly_sec = 0.0
        representation_assembly_sec = 0.0
        if self.history_representation == "c2kv" and eligible:
            if not callable(render_compressed):
                raise PolicyInputError("C2KV source-needs requires a compressed renderer")
            compressed_started = time.perf_counter()
            compressed_messages, compressed_counts = render_compressed(messages)
            compressed_assembly_sec = time.perf_counter() - compressed_started
            if [dict(m) for m in compressed_messages[int(compressed_counts["current_start_out_index"]):]
                    if not m.get("c2kv_key_hash")] != suffix:
                raise PolicyInputError("Compressed renderer changed the native live suffix")
            records = list(compressed_counts.get("compressed_records") or [])
            if not records:
                raise CapacityInfeasible("Eligible history produced no complete gist block")
        elif self.history_representation == "text_summary" and eligible:
            if not callable(render_summary):
                raise PolicyInputError("Text-summary source-needs requires a summary renderer")
            summary_started = time.perf_counter()
            summary_result = render_summary(messages)
            representation_assembly_sec = time.perf_counter() - summary_started
            if not isinstance(summary_result, dict):
                raise PolicyInputError("Summary renderer returned no provenance record")
            records = list(summary_result.get("records") or [])
            fragments = list(summary_result.get("history_packing_fragments") or [])
            fragment_by_id = {row.get("fragment_id"): row for row in fragments}
            if len(fragment_by_id) != len(fragments):
                raise PolicyInputError("Summary packing fragment ids must be unique")
            for record in records:
                required = {"summary_key", "packing_fragment_id", "source_indices",
                            "encoder_input_tokens", "source_content_sha256", "message",
                            "completion_cap", "finish_reason"}
                if not required <= record.keys():
                    raise PolicyInputError("Summary renderer omitted required provenance")
                fragment = fragment_by_id.get(record["packing_fragment_id"])
                if (fragment is None
                        or list(record["source_indices"]) != list(fragment.get("source_indices") or [])
                        or record["encoder_input_tokens"] != fragment.get("encoder_input_tokens")):
                    raise PolicyInputError("Summary record disagrees with its packing fragment")
                if (not record["source_indices"]
                        or any(type(index) is not int or index < 0 or index >= source_cutoff
                               or messages[index].get("role") == "system"
                               for index in record["source_indices"])):
                    raise PolicyInputError("Summary record has invalid original source indices")
                message = record["message"]
                if (not isinstance(message, dict) or message.get("role") != "user"
                        or not isinstance(message.get("content"), str)
                        or not message["content"].strip() or "c2kv_key_hash" in message):
                    raise PolicyInputError("Summary carrier must be ordinary nonempty user text")
            if not records:
                raise CapacityInfeasible("Eligible history produced no complete text summary")

        state_fitted, state_receipt, state_rows, state_budget_dropped = None, None, [], []
        if self.observed_state_enabled:
            state_fitted, state_receipt = fit_observed_state(
                build_observed_state(store), self._token_counter,
                max_prompt_tokens=self.state_prompt_token_cap)
            if state_fitted and state_fitted["calls"]:
                state_rows = [state_message(state_fitted)]

        packet_fitted, packet_receipt, packet_rows, packet_budget_dropped = None, None, [], []

        def prepare_dependency_packet():
            if not self.dependency_packet_enabled:
                return None, None, []
            candidate = build_dependency_packet(
                store, requested.source_ids, tools or [],
                max_entities=self.dependency_packet_config["dependency_packet_max_entities"],
                max_fields=self.dependency_packet_config["dependency_packet_max_fields"],
                max_atom_chars=self.dependency_packet_config["dependency_packet_max_atom_chars"],
                max_arguments_chars=self.dependency_packet_config[
                    "dependency_packet_max_arguments_chars"],
                field_priority_policy=self.dependency_packet_field_priority_policy,
            )
            fitted_packet, receipt = fit_dependency_packet(
                candidate, self._token_counter,
                max_prompt_tokens=self.dependency_packet_config[
                    "dependency_packet_prompt_token_cap"],
            )
            message = packet_message(fitted_packet)
            return fitted_packet, receipt, [message] if message is not None else []

        packet_fitted, packet_receipt, packet_rows = prepare_dependency_packet()

        def native_rows(indices):
            return [dict(full_messages[i + shift]) for i in sorted(indices)
                    if i < source_cutoff and messages[i].get("role") != "system"]

        def representation_rows(indices):
            if self.history_representation != "text_summary":
                return []
            return [dict(records[index]["message"]) for index in sorted(indices)]

        def raw_history_bytes(indices, representation_indices=()):
            trial = (common_prefix + representation_rows(representation_indices)
                     + packet_rows + state_rows + native_rows(indices) + suffix)
            count = self._token_counter(trial, tools) - common_tokens
            if count < 0:
                raise PolicyInputError("Source-needs raw-history token delta became negative")
            return count * self.bytes_per_kv_token

        def active_history_bytes(indices, representation_indices=()):
            active = raw_history_bytes(indices, representation_indices)
            if self.history_representation == "c2kv":
                active += sum(int(records[index]["record"]["gist_len"])
                              for index in representation_indices) * self.bytes_per_kv_token
            return active

        budget = min(self.config.history_budget_bytes, self.config.workspace_budget_bytes)
        priority = ([0] + list(range(len(records) - 1, 0, -1))) if records else []
        latest_tool_optional = bool(latest_complete_tool and all(
            index < source_cutoff and index not in common_sources
            for index in latest_complete_tool.source_indices))
        latest_tool_receipt = {
            "version": LATEST_TOOL_PROTECTION_VERSION,
            "policy": self.latest_complete_tool_protection,
            "status": ("none" if latest_complete_tool is None else
                       "admitted" if latest_tool_optional else "mandatory-common"),
            "event_id": latest_complete_tool.event_id if latest_complete_tool else None,
            "source_indices": (list(latest_complete_tool.source_indices)
                               if latest_complete_tool else []),
            "initially_protected": latest_complete_tool is not None,
            "selector_recomputed_after_skip": False,
            "indexed_after_skip": False,
            "budget_bytes": budget,
            "candidate_required_native_bytes": None,
            "candidate_active_history_bytes": None,
            "reason": ("no_complete_tool_event" if latest_complete_tool is None else
                       "latest_complete_tool_is_optional_historical_raw"
                       if latest_tool_optional else
                       "latest_complete_tool_is_in_the_mandatory_common_suffix"),
        }
        latest_tool_dropped = False
        while True:
            required_bytes = raw_history_bytes(selected_sources)
            reserved_index, reserved_bytes = None, 0
            representation_candidate_bytes = []
            for index in priority:
                candidate = active_history_bytes(selected_sources, {index})
                representation_candidate_bytes.append(candidate)
                size = candidate - required_bytes
                if size > 0 and candidate <= budget:
                    reserved_index, reserved_bytes = index, size
                    break
            if required_bytes <= budget and (not records or reserved_index is not None):
                if latest_tool_optional and not latest_tool_dropped:
                    latest_tool_receipt.update(
                        status="admitted",
                        candidate_required_native_bytes=required_bytes,
                        candidate_active_history_bytes=(
                            active_history_bytes(selected_sources, {reserved_index})
                            if reserved_index is not None else required_bytes),
                        reason="protected_native_and_required_representation_fit")
                break
            if not state_rows:
                if (self.latest_complete_tool_protection == "budgeted"
                        and latest_tool_optional and not latest_tool_dropped):
                    if self.source_needs_strategy != "lexical":
                        raise PolicyInputError(
                            "Budgeted latest-complete-tool protection must recompute lexical needs")
                    latest_tool_receipt.update(
                        status="skipped",
                        candidate_required_native_bytes=required_bytes,
                        candidate_active_history_bytes=(
                            min(representation_candidate_bytes)
                            if representation_candidate_bytes else required_bytes),
                        reason=("protected_native_over_budget" if required_bytes > budget else
                                "required_representation_over_budget"))
                    protected_ids.remove(latest_complete_tool.event_id)
                    selected_sources = common_sources | {
                        index for event in store.events if event.event_id in protected_ids
                        for index in event.source_indices}
                    (visible_ids, fitted, input_receipt, requested,
                     freshness_receipt, prediction_wall_sec) = select_sources(
                        selected_sources, recent_tool_event_visible=False)
                    packet_fitted, packet_receipt, packet_rows = prepare_dependency_packet()
                    fitted_ids = ([entry["source_id"] for entry in fitted["index"]]
                                  if fitted else [])
                    latest_tool_receipt.update(
                        selector_recomputed_after_skip=True,
                        indexed_after_skip=latest_complete_tool.event_id in fitted_ids)
                    latest_tool_dropped = True
                    continue
                if packet_rows:
                    packet_fitted, dropped_fact = drop_lowest_fact(packet_fitted)
                    dropped_source = dropped_fact["source"]
                    packet_budget_dropped.append({
                        "event_id": dropped_source["event_id"],
                        "source_index": dropped_source.get(
                            "result_source_index", dropped_source.get("source_index")),
                        "path": (dropped_fact.get("field") or {}).get("path"),
                        "kind": dropped_fact["kind"],
                    })
                    message = packet_message(packet_fitted)
                    packet_rows = [message] if message is not None else []
                    continue
                raise CapacityInfeasible(
                    "Protected native workspace and required representation exceed the history budget")
            state_budget_dropped.append(state_fitted["calls"].pop()["latest_result"]["source_id"])
            state_fitted["omitted_distinct_calls"] += 1
            state_rows = [state_message(state_fitted)] if state_fitted["calls"] else []
        admitted, skipped = [], []
        if self.dependency_packet_enabled:
            represented = set(packet_fitted["represented_source_ids"] if packet_fitted else ())
            admitted = [event_id for event_id in requested.source_ids if event_id in represented]
            skipped = [{"event_id": event_id, "reason": "dependency_packet_no_retained_fact"}
                       for event_id in requested.source_ids if event_id not in represented]
        else:
            for event_id in requested.source_ids:
                trial = selected_sources | set(store.event(event_id).source_indices)
                trial_bytes = active_history_bytes(
                    trial, {reserved_index} if reserved_index is not None else set())
                if trial_bytes <= budget:
                    selected_sources = trial
                    admitted.append(event_id)
                else:
                    skipped.append({"event_id": event_id, "reason": "requested_source_budget",
                                    "candidate_history_bytes": trial_bytes})
        selected_representation_indices, recency_ids = set(), []
        if records:
            order = [reserved_index] + [index for index in priority if index != reserved_index]
            for index in order:
                trial = selected_representation_indices | {index}
                if active_history_bytes(selected_sources, trial) <= budget:
                    selected_representation_indices = trial
        else:
            for event in reversed(store.events):
                if not event.complete or set(event.source_indices) <= selected_sources:
                    continue
                trial = selected_sources | set(event.source_indices)
                if raw_history_bytes(trial) <= budget:
                    selected_sources = trial
                    recency_ids.append(event.event_id)

        out, retained_records, blocks, representation_refs = list(common_prefix), [], [], []
        representation_out_indices = []
        for index, record in enumerate(records):
            if index not in selected_representation_indices:
                continue
            out_index = len(out)
            representation_out_indices.append(out_index)
            if self.history_representation == "c2kv":
                retained_records.append({**record, "out_index": out_index})
                out.append(dict(compressed_messages[record["out_index"]]))
                blocks.append({"key_hash": record["record"]["key_hash"],
                    "source_indices": [i - shift for i in record["source_indices"]],
                    "gist_tokens": int(record["record"]["gist_len"]),
                    **({"packing_fragment_id": record["packing_fragment_id"]}
                       if "packing_fragment_id" in record else {})})
            else:
                retained_records.append({**record, "out_index": out_index})
                out.append(dict(record["message"]))
                representation_refs.append({key: record[key] for key in
                    ("summary_key", "packing_fragment_id", "source_indices",
                     "encoder_input_tokens", "source_content_sha256", "completion_cap",
                     "finish_reason")})
        dependency_packet_out_index = len(out) if packet_rows else None
        out.extend(packet_rows)
        state_out_index = len(out) if state_rows else None
        out.extend(state_rows)
        native_out_start = len(out)
        restored = native_rows(selected_sources)
        out.extend(restored)
        current_out_index = len(out)
        out.extend(suffix)
        raw_tokens = self._token_counter([m for m in out if not m.get("c2kv_key_hash")], tools)
        gist_tokens = sum(block["gist_tokens"] for block in blocks)
        active_bytes = (raw_tokens - common_tokens + gist_tokens) * self.bytes_per_kv_token
        if active_bytes > budget or (self.history_representation == "c2kv" and eligible and not gist_tokens):
            raise CapacityInfeasible("Final source-needs view violates native B/W or gist reservation")
        if self.history_representation == "text_summary":
            coverage = summary_coverage_accounting(
                eligible_sources=eligible, raw_sources=selected_sources,
                retained_summaries=representation_refs,
                packing_fragments=summary_result["history_packing_fragments"] if summary_result else [])
        else:
            coverage = coverage_accounting(eligible_sources=eligible, raw_sources=selected_sources,
                retained_blocks=blocks, packing_fragments=(compressed_counts.get("history_packing_fragments")
                    if records else []))
        selected_ids = [event.event_id for event in store.events
                        if set(event.source_indices) <= selected_sources]
        metadata = {
            "version": RUNTIME_VERSION, "mode": self.mode, "route_mode": self.route_mode,
            "run_id": self.run_id, "task_id": task_id, "attempt_id": attempt, "decision_id": str(decision),
            "compression_policy": self.compression_policy, "history_view_protocol": self.history_view_protocol,
            "tool_schema_profile": TOOL_SCHEMA_PROFILE, "c2kv_tools_dump_expected": "full",
            "bytes_per_kv_token": self.bytes_per_kv_token, "byte_geometry_verified_by_backend": False,
            "history_budget_bytes": self.config.history_budget_bytes,
            "workspace_budget_bytes": self.config.workspace_budget_bytes,
            "budget_applies": True, "workspace_budget_applies": True,
            "active_history_bytes": active_bytes, "evidence_bytes": (raw_tokens - common_tokens) * self.bytes_per_kv_token,
            "gist_tokens": gist_tokens, "raw_history_tokens": raw_tokens - common_tokens,
            "total_raw_prompt_tokens": raw_tokens, "common_raw_prompt_tokens": common_tokens,
            "selected_event_ids": selected_ids, "selected_source_indices": sorted(selected_sources),
            "common_source_indices": sorted(common_sources), "protected_event_ids": sorted(protected_ids),
            "retrieved_event_ids": admitted, "retained_event_ids": [], "recency_selected_event_ids": recency_ids,
            "block_refs": blocks, "source_coverage": coverage, "evidence_out_index": None,
            "no_eligible_history": not eligible, "eligible_history_count": len(eligible), "full_identity_bypass": False,
            "pre_generation_workspace": {"version": "source-needs-native-workspace-v1",
                "renderer": "same-prefix Full training renderer", "post_draft_regeneration": False,
                "restored_source_indices": [i for i in sorted(selected_sources) if i < source_cutoff
                                             and messages[i].get("role") != "system"],
                "native_workspace_out_indices": list(range(native_out_start, current_out_index)),
                "dependency_packet_out_index": dependency_packet_out_index},
            "source_needs": {"version": native_source_needs.TOOL_NEEDS_VERSION
                if self.source_needs_strategy == "tool" else NEEDS_VERSION, "strategy": self.source_needs_strategy,
                "history_representation": self.history_representation,
                "input_receipt": input_receipt,
                "candidate_event_ids": [entry["source_id"] for entry in fitted["index"]] if fitted else [],
                "prediction_status": requested.status, "requested_event_ids": list(requested.source_ids),
                "typed_needs": [{"kind": kind, "source_ids": list(ids)} for kind, ids in requested.needs],
                "admitted_event_ids": admitted, "skipped_for_budget": skipped,
                "predictor_wall_sec": prediction_wall_sec,
                "admission_rule": "protected_native_then_requested_then_representation_fill",
                "admission_cost": ("actual_whole_trial_raw_tokens"
                    if self.history_representation == "text_summary"
                    else "actual_native_raw_plus_retained_gist")},
            "compressed_assembly_wall_sec": compressed_assembly_sec,
        }
        if self._latest_complete_tool_protection_explicit:
            metadata["latest_complete_tool_protection"] = latest_tool_receipt
        if self.actor_prompt_protocol is not None:
            metadata["history_organization"] = self.history_organization
            metadata["actor_prompt_protocol"] = self.actor_prompt_protocol
        if self.typed_subgoal_proposal_policy is not None:
            metadata["typed_subgoal_proposal_policy"] = self.typed_subgoal_proposal_policy
        if self.history_organization == "subgoal-v1":
            subgoal_organization = compressed_counts.get("subgoal_organization")
            if subgoal_organization is None and not eligible:
                subgoal_organization = build_subgoal_ledger(store).metadata()
            metadata["subgoal_organization"] = subgoal_organization
            metadata["retained_subgoal_groups"] = [
                row["subgoal_group"] for row in retained_records
                if "subgoal_group" in row]
        if self.history_representation != "text_summary":
            metadata["reserved_gist_bytes"] = reserved_bytes
            metadata["gist_reservation"] = {"required": bool(records),
                "reserved_gist_bytes": reserved_bytes, "satisfied": not records or gist_tokens > 0}
        if self.dependency_packet_enabled:
            without_packet = (out[:dependency_packet_out_index] + out[dependency_packet_out_index + 1:]
                              if dependency_packet_out_index is not None else out)
            packet_tokens = raw_tokens - self._token_counter(
                [message for message in without_packet if not message.get("c2kv_key_hash")], tools)
            if packet_tokens < 0:
                raise PolicyInputError("Dependency-packet incremental cost became negative")
            metadata["dependency_packet_bytes"] = packet_tokens * self.bytes_per_kv_token
            metadata["native_evidence_bytes"] = metadata["evidence_bytes"] - metadata[
                "dependency_packet_bytes"]
            metadata["dependency_packet"] = {
                "version": DEPENDENCY_PACKET_VERSION,
                **self.dependency_packet_config,
                "fit_receipt": packet_receipt,
                "dropped_for_workspace": packet_budget_dropped,
                "requested_source_ids": list(requested.source_ids),
                "represented_source_ids": list(
                    packet_fitted["represented_source_ids"] if packet_fitted else ()),
                "retained_fact_count": len(packet_fitted["facts"]) if packet_fitted else 0,
                "omitted": packet_fitted["omitted"] if packet_fitted else None,
                "fact_provenance": packet_fitted["facts"] if packet_fitted else [],
                "out_index": dependency_packet_out_index,
                "incremental_raw_tokens": packet_tokens,
                "counts_as_complete_event_coverage": False,
                "selector": "existing bounded lexical source-needs selector",
                "extractor_source": "preceding observable EventStore prefix",
                "uses_gold_future_or_hidden_state": False,
                "admission_rule": (
                    "current/protected raw, dependency packet, and one gist reservation share min(B,W); "
                    "drop lowest-ranked whole facts if needed"),
            }
        if self.history_representation == "text_summary":
            metadata.update(representation_refs=representation_refs,
                representation_out_indices=representation_out_indices,
                reserved_representation_bytes=reserved_bytes,
                representation_reservation={"required": bool(records),
                    "representation": self.history_representation,
                    "reserved_bytes": reserved_bytes,
                    "satisfied": not records or bool(representation_out_indices)},
                representation_assembly_wall_sec=representation_assembly_sec)
            metadata["pre_generation_workspace"]["representation_out_indices"] = representation_out_indices
            summary_receipt = summary_result or {
                "version": SUMMARY_VERSION, "dropped_docs": 0,
                "lookups": [], "producer_calls": 0, "wall_sec": 0.0,
                "source_scope": "preceding observed prefix; no eligible history",
                "coverage_scope": "no eligible history"}
            metadata["text_summary"] = {
                key: summary_receipt.get(key) for key in
                ("version", "dropped_docs", "lookups", "producer_calls", "wall_sec",
                 "source_scope", "coverage_scope")}
            metadata["text_summary"].update(
                candidate_summary_count=len(records),
                retained_summary_count=len(retained_records),
                retained_summary_keys=[record["summary_key"] for record in retained_records],
                semantic_fidelity="unknown")
        metadata["compression_ratio"] = ratio_accounting(full, metadata)
        if freshness_receipt is not None:
            metadata["source_needs"]["freshness"] = freshness_receipt
        metadata["compression_ratio"]["includes_coverage_loss"] = (
            None if self.history_representation == "text_summary"
            else bool(coverage["unrepresented_source_indices"]))
        if self.observed_state_enabled:
            without_state = (out[:state_out_index] + out[state_out_index + 1:]
                             if state_out_index is not None else out)
            state_tokens = raw_tokens - self._token_counter(
                [message for message in without_state if not message.get("c2kv_key_hash")], tools)
            if state_tokens < 0:
                raise PolicyInputError("Observed-state incremental cost became negative")
            metadata["state_bytes"] = state_tokens * self.bytes_per_kv_token
            metadata["native_evidence_bytes"] = metadata["evidence_bytes"] - metadata["state_bytes"]
            metadata["observed_state"] = {"version": STATE_VERSION,
                "state_prompt_token_cap": self.state_prompt_token_cap,
                "fit_receipt": state_receipt, "dropped_for_workspace": state_budget_dropped,
                "admitted_calls": len(state_fitted["calls"]) if state_fitted else 0,
                "admitted_source_ids": [row["latest_result"]["source_id"] for row in state_fitted["calls"]]
                    if state_fitted else [],
                "state_out_index": state_out_index, "incremental_raw_tokens": state_tokens,
                "state_counts_as_complete_event_coverage": False,
                "admission_rule": "bounded_state_then_native_and_one_gist_reservation; drop_oldest_state_rows_if_infeasible"}
        original_tokens = sum(
            int(record["record"].get("original_seq_len", 0))
            if self.history_representation == "c2kv"
            else int(record["encoder_input_tokens"])
            for record in retained_records)
        updated = {**compressed_counts, "memory_runtime": metadata,
            "compressed_records": (retained_records
                                   if self.history_representation == "c2kv" else []),
            "gist_tokens": gist_tokens,
            "original_tokens": original_tokens, "n_gist_messages": len(blocks),
            "compressed": len(blocks), "n_docs": len(retained_records),
            "current_start_out_index": current_out_index,
            "history_raw": (len(restored) + len(packet_rows) + len(state_rows)
                            + len(representation_refs)),
            "dropped_docs": ((int(summary_result.get("dropped_docs", 0))
                              if summary_result else int(compressed_counts.get("dropped_docs", 0)))
                             + len(records) - len(retained_records))}
        if self.history_representation == "text_summary":
            updated["summary_records"] = retained_records
            updated["n_summary_messages"] = len(representation_refs)
            fragments = summary_result["history_packing_fragments"] if summary_result else []
            updated["history_packing_fragments"] = fragments
            updated["history_packed_original_tokens"] = sum(
                int(row["encoder_input_tokens"]) for row in fragments)
            updated["history_packed_candidate_doc_count"] = len(fragments)
        packed = updated.get("history_packed_original_tokens")
        if packed is not None:
            updated["history_dropped_original_tokens"] = packed - original_tokens
            updated["history_retained_fraction"] = original_tokens / packed if packed else None
        self._observed_prefixes[state_key] = signature
        metadata["controller_wall_sec"] = (time.perf_counter() - started - prediction_wall_sec
                                           - compressed_assembly_sec - representation_assembly_sec)
        return out, updated
