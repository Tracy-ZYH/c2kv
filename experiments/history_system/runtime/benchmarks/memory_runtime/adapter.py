"""Bridge immutable events and a measured budget into the existing proxy.

The 1088 adapter leaves the legacy turn encoder intact. Recovery is an exact
text packet; all raw suffix tokens are rebuilt by the normal chat request.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable
from .tokenization import TOOL_SCHEMA_PROFILE, serving_tools
from .always_compress import (
    ALWAYS_COMPRESSION_POLICY, ALWAYS_ROUTE_MODES, HISTORY_VIEW_PROTOCOLS,
    WORKSPACE_ROUTE_RENDERERS, CapacityInfeasible, NativeCoverageUnsupported,
    coverage_accounting, eligible_history_sources, ratio_accounting,
)

_shared_path = str(Path(__file__).resolve().parents[2] / "python")
sys.path.insert(0, _shared_path)
try:
    from history_memory.evidence import EVIDENCE_VERSION, evidence_message
    from history_memory.events import EventStore
    from history_memory.packing import native_ids
    from .policy import BudgetExceeded, ConversationMemory, PolicyInputError, RuntimeConfig
finally:
    # python/agent is a different training package from the harness agent/
    # namespace. Loading this optional interface must not change its imports.
    sys.path.remove(_shared_path)

RUNTIME_VERSION = "a-event-runtime-v1"
EXACT_MODES = {
    "capacity_exact_once", "capacity_exact_persistent", "full_exact_shared",
    "capacity_exact_no_gist",
}
CAPACITY_MODES = {"capacity_protect", "full_capacity_aux"} | EXACT_MODES
MODES = {"legacy", "protect", "recover_once", "persistent", "no_gist", "full_shared", "raw_recency"} | CAPACITY_MODES


def raw_source_cutoff(messages):
    """Original source boundary corresponding to the legacy live input suffix."""
    last_anchor = max((i for i, message in enumerate(messages)
                       if message.get("role") in {"user", "tool"}), default=-1)
    return next((i + 1 for i in range(last_anchor, -1, -1)
                 if messages[i].get("role") == "assistant"), 0)


@dataclass
class PreparedExact:
    owner: Any
    memory: Any
    decision: Any
    source_cutoff: int
    visible_source_indices: frozenset[int]
    render: Callable | None
    messages: list
    counts: dict
    checked_signature: str | None = None
    checked_result: dict | None = None


class RuntimeAdapter:
    def __init__(self, config: dict, token_counter: Callable[[list, Any], int]):
        self.route_mode = config["mode"]
        self.mode = ALWAYS_ROUTE_MODES.get(self.route_mode, self.route_mode)
        if self.mode not in MODES:
            raise ValueError(f"Unknown runtime mode: {self.mode}")
        self.compression_policy = config.get(
            "compression_policy",
            ALWAYS_COMPRESSION_POLICY if self.route_mode in ALWAYS_ROUTE_MODES else None,
        )
        if self.compression_policy not in (None, ALWAYS_COMPRESSION_POLICY):
            raise ValueError("Unknown compression_policy")
        self.always_compress = self.compression_policy == ALWAYS_COMPRESSION_POLICY
        self.acquire_for_next = self.route_mode == "ac_acquire_for_next"
        self.workspace_renderer = WORKSPACE_ROUTE_RENDERERS.get(self.route_mode)
        if self.route_mode in ALWAYS_ROUTE_MODES and not self.always_compress:
            raise ValueError("Always-compress routes require their explicit policy")
        if self.always_compress and self.route_mode not in {*ALWAYS_ROUTE_MODES, "raw_recency"}:
            raise ValueError("Use a new route identity for the always-compress policy")
        self.history_view_protocol = config.get("history_view_protocol", "fixed-budget-main")
        if self.history_view_protocol not in HISTORY_VIEW_PROTOCOLS:
            raise ValueError("Unknown history_view_protocol")
        if not self.always_compress and "history_view_protocol" in config:
            raise ValueError("history_view_protocol belongs to the new always-compress routes")
        self.run_id = config.get("run_id")
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("Runtime config requires an explicit run_id")
        self.bytes_per_kv_token = config["bytes_per_kv_token"]
        if type(self.bytes_per_kv_token) is not int or self.bytes_per_kv_token <= 0:
            raise ValueError("bytes_per_kv_token must be a positive measured integer")
        self.config = RuntimeConfig(
            mode=("protect" if self.mode in {"legacy", "raw_recency", "capacity_protect"}
                  else "recover_once" if self.mode == "capacity_exact_once"
                  else "persistent" if self.mode in {
                      "capacity_exact_persistent", "full_exact_shared", "capacity_exact_no_gist"}
                  else "full_shared" if self.mode == "full_capacity_aux" else self.mode),
            history_budget_bytes=config["history_budget_bytes"],
            workspace_budget_bytes=config["workspace_budget_bytes"],
            lease_decisions=config.get("lease_decisions", 3),
            max_retrieved_events=config.get("max_retrieved_events", 2),
        )
        self._token_counter = token_counter
        self._states: dict[tuple, ConversationMemory] = {}
        self._exact_states: dict[tuple, Any] = {}
        self._lock = threading.RLock()

    @property
    def supports_exact_recovery(self):
        return self.mode in EXACT_MODES

    def _exact_memory(self, state_key, task_id):
        from .exact_policy import ExactRecoveryMemory
        if state_key not in self._exact_states:
            self._exact_states[state_key] = ExactRecoveryMemory(task_id, self.config)
        return self._exact_states[state_key]

    @classmethod
    def from_config(cls, path: str, tokenizer_path: str) -> "RuntimeAdapter":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

        def count(messages, tools):
            # Recent Transformers versions return BatchEncoding by default;
            # its len() is the number of fields, not the token sequence length.
            return len(native_ids(tokenizer, messages, tools=serving_tools(tools), generation=True))

        config = json.loads(Path(path).read_text(encoding="utf-8"))
        from .source_needs_runtime import SOURCE_NEEDS_ROUTES, SourceNeedsRuntime
        from .actor_evidence import ACTOR_EVIDENCE_ROUTES, ActorEvidenceRuntime
        if config.get("mode") in ACTOR_EVIDENCE_ROUTES:
            runtime = ActorEvidenceRuntime(config, count)
            runtime.tokenizer = tokenizer
            return runtime
        runtime_type = SourceNeedsRuntime if config.get("mode") in SOURCE_NEEDS_ROUTES else cls
        runtime = runtime_type(config, count)
        runtime.tokenizer = tokenizer
        return runtime

    def apply(self, messages, assembled, counts, eval_context, tools=None, *,
              render_full=None, render_compressed=None):
        with self._lock:
            return self._apply(messages, assembled, counts, eval_context, tools,
                               render_full, render_compressed)

    def prepare_exact(self, messages, assembled, counts, eval_context, tools=None, *,
                      render_compressed):
        if not self.supports_exact_recovery:
            raise PolicyInputError("prepare_exact requires an exact recovery mode")
        prepared = []
        with self._lock:
            out, updated = self._apply(messages, assembled, counts, eval_context, tools,
                                       None, render_compressed, on_prepared=prepared.append)
        if len(prepared) != 1:
            raise RuntimeError("Exact runtime did not prepare one request-local decision")
        return out, updated, prepared[0]

    def reconsider(self, prepared: PreparedExact, draft_tool_calls):
        from .exact_gap import detect_exact_source_gap
        from .policy import BudgetExceeded
        with self._lock:
            if not isinstance(prepared, PreparedExact) or prepared.owner is not self:
                raise PolicyInputError("Prepared decision belongs to another runtime")
            signature = json.dumps(draft_tool_calls, sort_keys=True, ensure_ascii=False)
            if prepared.checked_result is not None:
                if signature != prepared.checked_signature:
                    raise PolicyInputError("A prepared decision cannot inspect a second different draft")
                return prepared.checked_result
            started = time.perf_counter()
            gap = detect_exact_source_gap(
                prepared.decision.store, visible_source_indices=prepared.visible_source_indices,
                source_cutoff=prepared.source_cutoff, draft_tool_calls=draft_tool_calls)
            trace = {**gap.metadata(), "decision_index": prepared.decision.decision_index,
                     "upgrade_count": 0, "regeneration_allowed": False}
            regenerate = False
            if gap.status == "gap":
                if prepared.render is None:
                    raise PolicyInputError("Full-visible decision cannot need hidden evidence")
                try:
                    selection = prepared.memory.upgrade_decision(prepared.decision, gap.event_id)
                except BudgetExceeded as error:
                    trace.update(status="abstain", reason="budget_exhausted",
                                 required_cost_bytes=error.required_cost, budget_bytes=error.budget)
                else:
                    if self.acquire_for_next:
                        # Admission uses the same tokenized cost and reserved-gist
                        # limit as the incumbent. Only its lease state advances;
                        # the unconsumed evidence view is never rendered or logged
                        # as the current/final model input.
                        trace.update(
                            admitted_event_id=gap.event_id,
                            admission_count=1,
                            deferred_lease_acquisition_count=1,
                            current_view_upgrade_count=0,
                            actual_regeneration_count=0,
                            lease_expires_at_decision=selection.metadata["upgrade"][
                                "lease_expires_at_decision"],
                            evidence_consumption="next_decision_subject_to_revalidation",
                            final_action_source="original_draft",
                            method_revision="acquire-for-next-v1",
                        )
                    else:
                        old_raw_ids = prepared.counts["memory_runtime"].get("raw_history_event_ids")
                        prepared.messages, prepared.counts = prepared.render(selection, started)
                        if old_raw_ids is not None:
                            new_raw_ids = set(prepared.counts["memory_runtime"]["raw_history_event_ids"])
                            prepared.counts["memory_runtime"]["raw_body_evicted_on_upgrade"] = [
                                event_id for event_id in old_raw_ids if event_id not in new_raw_ids]
                        trace.update(upgrade_count=1, regeneration_allowed=True,
                                     upgraded_event_id=gap.event_id)
                        regenerate = True
            trace["controller_wall_sec"] = time.perf_counter() - started
            prepared.counts["memory_runtime"]["exact_recovery"] = trace
            result = {"regenerate": regenerate, "messages": prepared.messages,
                      "counts": prepared.counts, "decision": trace}
            prepared.checked_signature = signature
            prepared.checked_result = result
            return result

    def _apply(self, messages, assembled, counts, eval_context, tools,
               render_full, render_compressed, on_prepared=None):
        started = time.perf_counter()
        compressed_assembly_sec = 0.0
        capacity_gate = None
        full_aux = self.mode in {"full_capacity_aux", "full_exact_shared"}
        exact_no_gist = self.mode == "capacity_exact_no_gist"
        keep_full_reference = exact_no_gist or self.workspace_renderer is not None
        full_raw_reference = assembled if keep_full_reference else None
        full_raw_counts = counts if keep_full_reference else None
        budget_applies = self.mode not in {
            "full_shared", "full_capacity_aux", "full_exact_shared"}
        coverage_diagnostic = self.always_compress and (
            self.history_view_protocol == "coverage-preserving-diagnostic")
        if coverage_diagnostic and not full_aux and not exact_no_gist and self.mode != "raw_recency":
            budget_applies = False
        gate_name = "auxiliary_gate" if full_aux else "capacity_gate"
        activation_name = "auxiliary_activated" if full_aux else "compression_activated"
        if not isinstance(eval_context, dict):
            raise PolicyInputError("A runtime needs explicit benchmark request context")
        if any(key in eval_context for key in ("gold", "oracle", "target_action", "hidden_state")):
            raise PolicyInputError("Privileged labels cannot enter runtime context")
        task_id = eval_context.get("task_id")
        run_id = eval_context.get("run_id", self.run_id)
        attempt = eval_context.get("attempt_id", eval_context.get("attempt"))
        if not isinstance(task_id, str) or not task_id or run_id != self.run_id or attempt is None:
            raise PolicyInputError("Require task_id, matching run_id, and explicit attempt/attempt_id")
        decision = eval_context.get("decision_id")
        if decision is None:
            if not {"user_turn", "step"} <= eval_context.keys():
                raise PolicyInputError("Require decision_id or explicit user_turn and step")
            decision = json.dumps([eval_context["user_turn"], eval_context["step"]])
        state_key = (run_id, task_id, str(attempt))
        # IDs use the explicit task identity so auxiliary text is identical
        # across paired arms. The state registry also isolates run and attempt.
        store = EventStore.from_messages(task_id, messages)
        if self.mode in {"full_exact_shared", "capacity_exact_no_gist"} and any(
            message.role == "developer" for message in store.messages
        ):
            raise PolicyInputError(
                f"{self.mode} does not support developer-role messages"
            )
        source_cutoff = raw_source_cutoff(messages)
        eligible_sources = eligible_history_sources(store, source_cutoff)
        full = None
        def annotate_always(metadata, *, records=(), raw_sources=(), packing_fragments=None):
            if not self.always_compress:
                return
            metadata.update(
                route_mode=self.route_mode,
                compression_policy=ALWAYS_COMPRESSION_POLICY,
                history_view_protocol=self.history_view_protocol,
                full_identity_bypass=False,
                no_eligible_history=not eligible_sources,
                eligible_history_count=len(eligible_sources),
            )
            if full is None:
                raise ValueError("Always-compress accounting requires a same-prefix Full view")
            coverage = coverage_accounting(
                eligible_sources=eligible_sources, raw_sources=set(raw_sources),
                retained_blocks=records, packing_fragments=packing_fragments,
            )
            metadata["source_coverage"] = coverage
            metadata["compression_ratio"] = ratio_accounting(full, metadata)
            missing = coverage["unrepresented_source_indices"]
            metadata["compression_ratio"]["includes_coverage_loss"] = (
                bool(missing) if missing is not None else None)
        exact_memory = self._exact_memory(state_key, task_id) if self.supports_exact_recovery else None
        if self.mode in CAPACITY_MODES or self.always_compress:
            from .capacity import measure_full_history

            gate_started = time.perf_counter()
            full = measure_full_history(assembled, counts, self._token_counter,
                                        tools, self.bytes_per_kv_token)
            activated = (bool(eligible_sources) if self.always_compress else
                         full["active_history_bytes"] > self.config.history_budget_bytes)
            capacity_gate = {
                "rule": "eligible_history_always_compressed" if self.always_compress else "full_history_exceeds_budget",
                "full_raw_prompt_tokens": full["total_raw_prompt_tokens"],
                "full_raw_history_tokens": full["raw_history_tokens"],
                "full_history_bytes": full["active_history_bytes"],
                "history_budget_bytes": self.config.history_budget_bytes,
                activation_name: activated,
                "wall_sec": time.perf_counter() - gate_started,
            }
            if not activated:
                exact_decision = None
                if exact_memory is not None:
                    def no_packet_cost(event_ids):
                        if event_ids:
                            raise PolicyInputError("Full-visible history needs no auxiliary packet")
                        return 0
                    exact_decision = exact_memory.prepare_decision(
                        store, no_packet_cost, {event.event_id for event in store.events}, str(decision))
                updated = dict(counts)
                updated["memory_runtime"] = {
                    **full, "version": RUNTIME_VERSION, "evidence_version": None,
                    "tool_schema_profile": TOOL_SCHEMA_PROFILE, "c2kv_tools_dump_expected": "full",
                    "mode": self.mode, "run_id": run_id, "task_id": task_id,
                    "attempt_id": attempt, "decision_id": str(decision),
                    "bytes_per_kv_token": self.bytes_per_kv_token,
                    "byte_geometry_verified_by_backend": False,
                    "history_budget_bytes": self.config.history_budget_bytes,
                    "workspace_budget_bytes": self.config.workspace_budget_bytes,
                    "workspace_budget_applies": False, "budget_applies": budget_applies,
                    "evidence_bytes": 0, "gist_tokens": 0,
                    "selected_event_ids": [event.event_id for event in store.events],
                    "selected_source_indices": list(range(len(messages))),
                    "protected_event_ids": [], "retrieved_event_ids": [], "retained_event_ids": [],
                    "block_refs": [], "overlapping_gist_keys": [], "evicted_gist_keys": [],
                    "eviction_reason": None, "evidence_out_index": None,
                    "policy": {"selection": "no_eligible_history" if self.always_compress else "full_history_within_budget"},
                    "suffix_policy": "fresh_full_renderer",
                    gate_name: capacity_gate, "compressed_assembly_wall_sec": 0.0,
                    "controller_wall_sec": time.perf_counter() - started,
                }
                annotate_always(updated["memory_runtime"], raw_sources=range(len(messages)), packing_fragments=[])
                if self.always_compress:
                    updated["memory_runtime"]["gist_reservation"] = {
                        "required": False, "reserved_gist_bytes": 0,
                        "selection_budget_bytes": min(self.config.history_budget_bytes, self.config.workspace_budget_bytes),
                        "satisfied": True,
                    }
                if self.workspace_renderer is not None:
                    updated["memory_runtime"]["pre_generation_workspace"] = {
                        "version": "pre-generation-native-workspace-v1",
                        "renderer": self.workspace_renderer,
                        "matched_control_route": (
                            "ac_native_workspace" if self.route_mode == "ac_packet_workspace"
                            else "ac_packet_workspace"
                        ),
                        "status": "no_eligible_history",
                        "candidate_event_id": None,
                        "candidate_tool_event_id": None,
                        "candidate_goal_event_id": None,
                        "candidate_event_ids": [],
                        "selected_event_id": None,
                        "selected_native_event_ids": [],
                        "native_source_indices": [],
                        "restored_source_indices": [],
                        "already_common_source_indices": [],
                        "common_admission_cost_bytes": 0,
                        "rendered_workspace_bytes": 0,
                        "post_draft_regeneration": False,
                        "uses_gold_or_future_state": False,
                    }
                if exact_decision is not None:
                    updated["memory_runtime"]["policy"].update(exact_decision.selection.metadata)
                    if on_prepared is not None:
                        on_prepared(PreparedExact(
                            self, exact_memory, exact_decision, source_cutoff,
                            frozenset(range(len(messages))), None, assembled, updated))
                return assembled, updated
            if not full_aux and not exact_no_gist and self.mode != "raw_recency":
                if render_compressed is None:
                    raise PolicyInputError("capacity_protect requires a lazy compressed renderer above budget")
                assembly_started = time.perf_counter()
                assembled, counts = render_compressed(messages)
                compressed_assembly_sec = time.perf_counter() - assembly_started
        if self.mode == "raw_recency":
            if render_full is None:
                raise PolicyInputError("raw_recency requires the existing Full renderer")
            from .raw_recency import build_raw_recency_view

            out, updated, metadata = build_raw_recency_view(
                store, render=render_full, token_counter=self._token_counter,
                tools=tools, bytes_per_kv_token=self.bytes_per_kv_token,
                history_budget_bytes=self.config.history_budget_bytes,
            )
            metadata.update({
                "version": RUNTIME_VERSION, "evidence_version": None,
                "tool_schema_profile": TOOL_SCHEMA_PROFILE, "c2kv_tools_dump_expected": "full",
                "mode": self.mode, "run_id": run_id, "task_id": task_id,
                "attempt_id": attempt, "decision_id": str(decision),
                "bytes_per_kv_token": self.bytes_per_kv_token,
                "byte_geometry_verified_by_backend": False,
                "history_budget_bytes": self.config.history_budget_bytes,
                "workspace_budget_bytes": self.config.workspace_budget_bytes,
                "workspace_budget_applies": False, "budget_applies": True,
                "evidence_bytes": 0, "gist_tokens": 0,
                "retrieved_event_ids": [], "retained_event_ids": [],
                "block_refs": [], "overlapping_gist_keys": [], "evicted_gist_keys": [],
                "evidence_out_index": None,
                "suffix_policy": "fresh_full_renderer_with_complete_raw_events",
                "controller_wall_sec": time.perf_counter() - started,
            })
            updated["memory_runtime"] = metadata
            annotate_always(metadata, raw_sources=metadata["selected_source_indices"], packing_fragments=[])
            return out, updated
        # The legacy proxy keeps the entire input block after the preceding
        # assistant, including consecutive user messages / parallel returns.
        cutoff = int(counts["current_start_out_index"])
        base = [dict(message) for message in assembled]
        raw = [message for message in base if not message.get("c2kv_key_hash")]
        prefix_raw = [message for message in base[:cutoff] if not message.get("c2kv_key_hash")]
        common_prefix = [message for message in prefix_raw if message.get("role") == "system"]
        common = common_prefix + [message for message in base[cutoff:] if not message.get("c2kv_key_hash")]
        common_tokens = self._token_counter(common, tools)

        workspace = None
        if self.workspace_renderer is not None:
            from .native_workspace import plan_native_workspace

            full_cutoff = int(full_raw_counts["current_start_out_index"])
            full_suffix = [dict(message) for message in full_raw_reference[full_cutoff:]]
            base_suffix = [dict(message) for message in base[cutoff:]
                           if not message.get("c2kv_key_hash")]
            if base_suffix != full_suffix:
                raise PolicyInputError(
                    "Compressed current suffix differs from the same-prefix Full renderer")
            workspace = plan_native_workspace(
                store, source_cutoff=source_cutoff,
                full_messages=full_raw_reference, full_cutoff=full_cutoff)

        if self.mode == "no_gist" or exact_no_gist:
            base = common[:]
            cutoff = len(common_prefix)
            raw = common[:]
        elif budget_applies and any(m.get("role") != "system" for m in prefix_raw):
            raise PolicyInputError("1088 budget adapter requires plain gist history, without hybrid raw history")

        visible = {
            event.event_id for event in store.events
            if self.mode == "full_shared" or event.kind == "instruction"
            or all(index >= source_cutoff for index in event.source_indices)
        }
        packets: dict[tuple, dict | None] = {}
        costs: dict[tuple, int] = {}
        # Match protection's auxiliary selection cost at the common raw suffix.
        # Full history remains in the actual view and is measured separately.
        cost_base = common if full_aux else base
        cost_cutoff = len(common_prefix) if full_aux else cutoff
        cost_raw = common if full_aux else raw

        def packet_for(event_ids):
            ids = tuple(event_ids)
            if ids not in packets:
                packet = evidence_message(store, ids)
                packets[ids] = packet
            return packets[ids]

        def rendered_workspace(event_ids, renderer):
            ids = tuple(event_ids)
            native = workspace.select(ids) if workspace is not None else None
            packet_ids = ids
            if renderer == "native_workspace" and native is not None:
                native_ids = set(native.event_ids)
                packet_ids = tuple(event_id for event_id in ids if event_id not in native_ids)
            else:
                native = workspace.select(()) if workspace is not None else None
            packet = packet_for(packet_ids)
            view = [m for m in cost_base[:cost_cutoff] if not m.get("c2kv_key_hash")]
            if packet is not None:
                view.append(packet)
            view.extend(dict(message) for message in (native.messages if native else ()))
            view.extend(m for m in cost_base[cost_cutoff:] if not m.get("c2kv_key_hash"))
            return view, packet_ids, packet, native

        def workspace_cost(event_ids, renderer="evidence_packet"):
            ids = (renderer, *tuple(event_ids))
            if ids not in costs:
                view, _, _, _ = rendered_workspace(event_ids, renderer)
                # SGLang removes gist carriers before raw chat templating.
                # Measure the complete raw view, including template boundaries.
                delta = self._token_counter(view, tools) - self._token_counter(cost_raw, tools)
                if delta < 0:
                    raise ValueError("Workspace renderer produced a negative token delta")
                costs[ids] = delta * self.bytes_per_kv_token
            return costs[ids]

        def packet_cost(event_ids):
            return workspace_cost(event_ids, "evidence_packet")

        def selection_cost(event_ids):
            if self.workspace_renderer is None:
                return packet_cost(event_ids)
            return max(
                workspace_cost(event_ids, "evidence_packet"),
                workspace_cost(event_ids, "native_workspace"),
            )

        native_records = list(counts.get("compressed_records") or [])
        reserved_gist_index = None
        reserved_gist_bytes = 0
        always_gist = self.always_compress and not full_aux and not exact_no_gist
        admission_budget = min(self.config.history_budget_bytes, self.config.workspace_budget_bytes)
        if always_gist and eligible_sources:
            if not native_records:
                raise CapacityInfeasible("Eligible history produced no native gist block")
            if coverage_diagnostic:
                if int(counts.get("dropped_docs") or 0):
                    raise NativeCoverageUnsupported("Native packing dropped history before the coverage-preserving view")
                if counts.get("history_packing_fragments") is None:
                    raise NativeCoverageUnsupported("Complete coverage requires the native packing fragment ledger")
            else:
                users = [event for event in store.events if event.kind == "user"]
                mandatory = {
                    event.event_id for event in store.events
                    if not event.complete and event.event_id not in visible
                }
                if users and users[-1].event_id not in visible:
                    mandatory.add(users[-1].event_id)
                mandatory_ids = tuple(event.event_id for event in store.events if event.event_id in mandatory)
                mandatory_bytes = selection_cost(mandatory_ids) if self.mode != "legacy" else 0
                raw_base_bytes = (self._token_counter(raw, tools) - common_tokens) * self.bytes_per_kv_token
                priority = [0] + list(range(len(native_records) - 1, 0, -1))
                for index in priority:
                    gist_bytes = int(native_records[index]["record"]["gist_len"]) * self.bytes_per_kv_token
                    if gist_bytes > 0 and gist_bytes + raw_base_bytes + mandatory_bytes <= self.config.history_budget_bytes:
                        reserved_gist_index, reserved_gist_bytes = index, gist_bytes
                        break
                if reserved_gist_index is None:
                    raise CapacityInfeasible("Necessary raw input and one complete gist block exceed B")
                admission_budget = min(self.config.workspace_budget_bytes,
                                       self.config.history_budget_bytes - reserved_gist_bytes - raw_base_bytes)
        selection = None
        exact_decision = None
        if self.mode == "legacy":
            selected_ids = ()
        elif exact_memory is not None:
            options = {"selection_budget_bytes": admission_budget} if always_gist and not coverage_diagnostic else {}
            try:
                exact_decision = exact_memory.prepare_decision(store, selection_cost, visible, str(decision), **options)
            except BudgetExceeded as error:
                if always_gist:
                    raise CapacityInfeasible(str(error)) from error
                raise
            selection = exact_decision.selection
            selected_ids = selection.selected_event_ids
        else:
            memory = self._states.setdefault(state_key, ConversationMemory(task_id, self.config))
            if always_gist and not coverage_diagnostic:
                memory.config = replace(self.config, history_budget_bytes=admission_budget,
                                        workspace_budget_bytes=admission_budget)
            try:
                selection = memory.prepare(store, selection_cost, visible, str(decision))
            except BudgetExceeded as error:
                if always_gist:
                    raise CapacityInfeasible(str(error)) from error
                raise
            selected_ids = selection.selected_event_ids
        def render_selected(selection, selected_ids, *, phase_started=started,
                            assembly_sec=compressed_assembly_sec):
            renderer = self.workspace_renderer or "evidence_packet"
            rendered_raw, packet_ids, packet, native = rendered_workspace(
                selected_ids, renderer)
            restored = native.messages if native is not None else ()
            evidence_bytes = workspace_cost(selected_ids, renderer)
            allocation_bytes = (selection_cost(selected_ids)
                                if self.workspace_renderer is not None else evidence_bytes)

            records = list(counts.get("compressed_records") or [])
            kept_indices = set()
            remaining = self.config.history_budget_bytes - allocation_bytes
            if always_gist and coverage_diagnostic:
                remaining = sum(int(item["record"]["gist_len"]) * self.bytes_per_kv_token for item in records)
            # Preserve the legacy doc-0 anchor, then prefer newer complete blocks.
            priority = ([0] + list(range(len(records) - 1, 0, -1))) if records else []
            if reserved_gist_index is not None:
                priority = [reserved_gist_index] + [i for i in priority if i != reserved_gist_index]
            for index in priority:
                size = int(records[index]["record"]["gist_len"]) * self.bytes_per_kv_token
                if size <= remaining:
                    kept_indices.add(index)
                    remaining -= size
            if always_gist and eligible_sources and not kept_indices:
                raise CapacityInfeasible("Evidence admission would remove the last gist block")
            keep_out = {records[index]["out_index"] for index in kept_indices}
            discarded = [record for i, record in enumerate(records) if i not in kept_indices]
            out = []
            old_to_new = {}
            packet_index = None
            native_workspace_indices = []
            for index, message in enumerate(base):
                if index == cutoff and packet is not None:
                    packet_index = len(out)
                    out.append(packet)
                if index == cutoff and restored:
                    native_workspace_indices.extend(range(len(out), len(out) + len(restored)))
                    out.extend(dict(item) for item in restored)
                if message.get("c2kv_key_hash") and index not in keep_out:
                    continue
                old_to_new[index] = len(out)
                out.append(message)
            if cutoff == len(base) and packet is not None:
                packet_index = len(out)
                out.append(packet)
            if cutoff == len(base) and restored:
                native_workspace_indices.extend(range(len(out), len(out) + len(restored)))
                out.extend(dict(item) for item in restored)
            retained_records = []
            for index, record in enumerate(records):
                if index in kept_indices:
                    retained_records.append({**record, "out_index": old_to_new[record["out_index"]]})
            gist_tokens = sum(int(record["record"]["gist_len"]) for record in retained_records)
            original_tokens = sum(int(record["record"].get("original_seq_len", 0)) for record in retained_records)
            full_raw_tokens = self._token_counter([m for m in out if not m.get("c2kv_key_hash")], tools)
            raw_history_tokens = full_raw_tokens - common_tokens
            if raw_history_tokens < 0:
                raise ValueError("Raw history accounting became negative")
            history_bytes = (gist_tokens + raw_history_tokens) * self.bytes_per_kv_token
            if budget_applies and history_bytes > self.config.history_budget_bytes:
                raise ValueError("Measured active history exceeds the frozen byte cap")
            selection_evidence_bytes = evidence_bytes
            if full_aux:
                evidence_bytes = (full_raw_tokens - self._token_counter(raw, tools)) * self.bytes_per_kv_token
                if evidence_bytes < 0 or evidence_bytes > self.config.workspace_budget_bytes:
                    raise ValueError("Full auxiliary packet exceeds the measured workspace byte cap")
            inserted_system = not any(m.get("role") == "system" for m in messages)
            blocks = [{
                "key_hash": record["record"]["key_hash"],
                "source_indices": [i - int(inserted_system) for i in record["source_indices"]],
                "gist_tokens": int(record["record"]["gist_len"]),
                **({"packing_fragment_id": record["packing_fragment_id"]}
                   if "packing_fragment_id" in record else {}),
            } for record in retained_records]
            selected_sources = {i for event_id in selected_ids for i in store.event(event_id).source_indices}
            metadata = {
                "version": RUNTIME_VERSION, "evidence_version": EVIDENCE_VERSION,
                "tool_schema_profile": TOOL_SCHEMA_PROFILE, "c2kv_tools_dump_expected": "full",
                "mode": self.mode, "run_id": run_id, "task_id": task_id,
                "attempt_id": attempt, "decision_id": str(decision),
                "bytes_per_kv_token": self.bytes_per_kv_token,
                "byte_geometry_verified_by_backend": False,
                "history_budget_bytes": self.config.history_budget_bytes,
                "workspace_budget_bytes": self.config.workspace_budget_bytes,
                "budget_applies": budget_applies,
                "active_history_bytes": history_bytes,
                "evidence_bytes": evidence_bytes, "gist_tokens": gist_tokens,
                "raw_history_tokens": raw_history_tokens,
                "common_raw_prompt_tokens": common_tokens,
                "total_raw_prompt_tokens": full_raw_tokens,
                "selected_event_ids": list(selected_ids),
                "protected_event_ids": list(selection.protected_event_ids) if selection else [],
                "retrieved_event_ids": list(selection.retrieved_event_ids) if selection else [],
                "retained_event_ids": list(selection.retained_event_ids) if selection else [],
                "policy": dict(selection.metadata) if selection else {},
                "block_refs": blocks,
                "overlapping_gist_keys": [b["key_hash"] for b in blocks if selected_sources.intersection(b["source_indices"])],
                "evicted_gist_keys": [record["record"]["key_hash"] for record in discarded],
                "eviction_reason": "absolute_history_budget" if discarded else None,
                "evidence_out_index": packet_index,
                "suffix_policy": "fresh_raw_suffix_with_cached_unchanged_gist",
                "controller_wall_sec": time.perf_counter() - phase_started - assembly_sec,
            }
            if self.always_compress:
                metadata["gist_reservation"] = {
                    "required": bool(always_gist and eligible_sources),
                    "reserved_gist_bytes": reserved_gist_bytes,
                    "selection_budget_bytes": admission_budget,
                    "satisfied": not always_gist or not eligible_sources or gist_tokens > 0,
                }
            if self.workspace_renderer is not None:
                packet_render_cost = workspace_cost(selected_ids, "evidence_packet")
                native_render_cost = workspace_cost(selected_ids, "native_workspace")
                selected_workspace_event = (
                    workspace.event_id if workspace.event_id in selected_ids else None)
                selected_workspace_ids = workspace.select(selected_ids).event_ids
                metadata["pre_generation_workspace"] = {
                    **workspace.metadata(native.event_ids if native is not None else ()),
                    "renderer": self.workspace_renderer,
                    "matched_control_route": (
                        "ac_native_workspace" if self.route_mode == "ac_packet_workspace"
                        else "ac_packet_workspace"
                    ),
                    "status": "selected" if selected_workspace_ids else "no_op",
                    "selected_event_id": selected_workspace_event,
                    "selected_workspace_event_ids": list(selected_workspace_ids),
                    "selected_event_ids": list(selected_ids),
                    "selected_native_event_ids": list(native.event_ids if native else ()),
                    "evidence_packet_event_ids": list(packet_ids),
                    "native_workspace_out_indices": native_workspace_indices,
                    "packet_render_cost_bytes": packet_render_cost,
                    "native_render_cost_bytes": native_render_cost,
                    "common_admission_cost_bytes": max(packet_render_cost, native_render_cost),
                    "rendered_workspace_bytes": evidence_bytes,
                    "allocation_bytes": allocation_bytes,
                    "post_draft_regeneration": False,
                }
            if capacity_gate is not None:
                metadata[gate_name] = capacity_gate
                metadata["compressed_assembly_wall_sec"] = assembly_sec
                metadata["workspace_budget_applies"] = True
            if full_aux:
                metadata.update({
                    "auxiliary_reference_mode": (
                        "capacity_exact_persistent"
                        if self.mode == "full_exact_shared"
                        else "capacity_protect"
                    ),
                    "auxiliary_selection_bytes": selection_evidence_bytes,
                    "full_raw_visible_event_ids": [event.event_id for event in store.events],
                    "retrieval_noop_reason": "all_source_events_present_in_full_raw_history",
                    "suffix_policy": "fresh_full_renderer_with_capacity_matched_auxiliary",
                })
            updated = dict(counts)
            if self.always_compress:
                metadata["reserved_gist_bytes"] = reserved_gist_bytes
                metadata["evidence_admission_budget_bytes"] = admission_budget
            updated.update({
                "memory_runtime": metadata, "compressed_records": retained_records,
                "gist_tokens": gist_tokens, "original_tokens": original_tokens,
                "n_gist_messages": len(retained_records), "compressed": len(retained_records),
                "n_docs": len(retained_records),
                "dropped_docs": int(counts.get("dropped_docs", 0)) + len(discarded),
                "current_start_out_index": old_to_new.get(cutoff, len(out)),
                "history_raw": sum(m.get("role") != "system" and not m.get("c2kv_key_hash") for m in out[:old_to_new.get(cutoff, len(out))]),
            })
            denominator = updated.get("history_packed_original_tokens")
            if denominator is not None:
                updated["history_dropped_original_tokens"] = denominator - original_tokens
                updated["history_retained_fraction"] = original_tokens / denominator if denominator else None
            if exact_no_gist:
                from .exact_raw import build_exact_raw_view

                out, raw_counts, raw_metadata = build_exact_raw_view(
                    store, full_raw_reference, full_raw_counts,
                    source_cutoff=source_cutoff, evidence_event_ids=selected_ids,
                    token_counter=self._token_counter, tools=tools,
                    bytes_per_kv_token=self.bytes_per_kv_token,
                    history_budget_bytes=self.config.history_budget_bytes,
                    workspace_budget_bytes=self.config.workspace_budget_bytes,
                    reference_common_tokens=common_tokens)
                if raw_metadata["auxiliary_selection_bytes"] != selection_evidence_bytes:
                    raise ValueError("NoGist evidence admission differs from shared reference")
                raw_counts.pop("memory_runtime", None)
                updated.update(raw_counts)
                metadata.update(raw_metadata)
                metadata["suffix_policy"] = "fresh_full_dialect_raw_body_with_shared_exact_evidence"
                metadata["controller_wall_sec"] = time.perf_counter() - phase_started - assembly_sec
            actual_raw_sources = (
                set(range(len(messages))) if full_aux else
                set(metadata.get("visible_source_indices", ())) if exact_no_gist else
                set(range(source_cutoff, len(messages))) | selected_sources
            )
            annotate_always(metadata, records=blocks, raw_sources=actual_raw_sources,
                            packing_fragments=counts.get("history_packing_fragments", [] if exact_no_gist or full_aux else None))
            if always_gist and coverage_diagnostic and metadata["source_coverage"]["complete_history_coverage"] is not True:
                raise NativeCoverageUnsupported("Native gist and raw union do not cover all eligible source occurrences")
            return out, updated

        out, updated = render_selected(selection, selected_ids)
        if exact_decision is not None and on_prepared is not None:
            if self.mode == "full_exact_shared":
                visible_sources = set(range(len(messages)))
            elif exact_no_gist:
                visible_sources = set(updated["memory_runtime"]["visible_source_indices"])
            else:
                visible_sources = set(range(source_cutoff, len(messages)))
                visible_sources.update(
                    index for event in store.events if event.kind == "instruction"
                    for index in event.source_indices)
                visible_sources.update(
                    index for event_id in selected_ids
                    for index in store.event(event_id).source_indices)
            on_prepared(PreparedExact(
                self, exact_memory, exact_decision, source_cutoff, frozenset(visible_sources),
                lambda chosen, phase_started: render_selected(
                    chosen, chosen.selected_event_ids, phase_started=phase_started, assembly_sec=0.0),
                out, updated))
        return out, updated
