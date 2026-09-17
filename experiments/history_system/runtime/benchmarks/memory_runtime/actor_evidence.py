"""Let one actor request bounded exact history before submitting its action."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import time

from .source_needs_runtime import SourceNeedsRuntime
from history_memory.events import EventStore
from .always_compress import CapacityInfeasible, ratio_accounting
from .capacity import measure_full_history
from .native_source_needs import TOOL_NAME, prediction_tools, parse_prediction
from .policy import PolicyInputError
from .source_needs import build_needs_input

ACTOR_EVIDENCE_ROUTES = {"ac_actor_evidence_once": "ac_native_needs_lexical",
                         "raw_actor_evidence_once": "raw_native_needs_lexical"}
VERSION = "actor-evidence-once-v1"


def evidence_tool(index):
    context = {"index": index}
    tool = prediction_tools(context)[0]
    compact = [{"source_id": row["source_id"], "calls": [
        {"name": call["name"], "arguments": call["argument_anchors"]} for call in row["calls"]],
        "result_fields": [result["fields"] for result in row["results"]]} for row in index]
    tool["function"]["description"] = (
        "Retrieve exact contents of older observed tool events needed before your next action or final answer. "
        "Use at most once. These are history reads, not application actions. "
        "Select only source_ids from this metadata index: "
        + json.dumps(compact, ensure_ascii=False, separators=(",", ":")))
    return tool


@dataclass
class PreparedActorEvidence:
    owner: object
    messages: list
    full_messages: list
    full_counts: dict
    context: dict
    application_tools: list
    offered_tools: list
    index: list
    render_compressed: object
    first_counts: dict
    completed: bool = False


class ActorEvidenceRuntime(SourceNeedsRuntime):
    supports_actor_evidence = True

    def __init__(self, config, token_counter):
        self.parent_config = {**config, "mode": ACTOR_EVIDENCE_ROUTES[config["mode"]]}
        super().__init__(self.parent_config, token_counter)
        self.route_mode = config["mode"]
        self.actor_evidence_schema_token_cap = config["actor_evidence_schema_token_cap"]
        if type(self.actor_evidence_schema_token_cap) is not int or self.actor_evidence_schema_token_cap <= 0:
            raise ValueError("Actor evidence requires a positive explicit schema token cap")

    def prepare_actor(self, messages, full_messages, full_counts, context, tools, *, render_compressed):
        with self._lock:
            started = time.perf_counter()
            tools = deepcopy(tools or [])
            if any(tool.get("function", tool).get("name") == TOOL_NAME for tool in tools):
                raise PolicyInputError("Application schema collides with the internal evidence tool")
            compressed, extraction_wall = [], [0.0]
            def render(source):
                if source != messages:
                    raise PolicyInputError("Evidence preparation cannot change the observed source prefix")
                if not compressed:
                    begin = time.perf_counter()
                    compressed.append(render_compressed(source))
                    extraction_wall[0] += time.perf_counter() - begin
                return deepcopy(compressed[0])
            base, base_counts = super().apply(messages, full_messages, full_counts, context, tools,
                render_compressed=render)
            store = EventStore.from_messages(context["task_id"], messages)
            visible = set(base_counts["memory_runtime"]["selected_source_indices"])
            excluded = [event.event_id for event in store.events if set(event.source_indices) <= visible]
            index = build_needs_input(store, tools, excluded_event_ids=excluded,
                                     max_candidates=self.source_index_max_events)["index"]
            def raw_count(rows, schema):
                return self._token_counter([row for row in rows if not row.get("c2kv_key_hash")], schema)
            out, counts, offered_tools = base, base_counts, tools
            dropped, overhead = [], 0
            status = "no_missing_raw_candidates" if not index else "offer_not_admitted"
            while index:
                trial_tools = tools + [evidence_tool(index)]
                delta = raw_count(base, trial_tools) - raw_count(base, tools)
                if delta < 0:
                    raise PolicyInputError("Internal schema has a negative measured token cost")
                if delta > self.actor_evidence_schema_token_cap:
                    dropped.append({"source_id": index.pop()["source_id"], "reason": "schema_token_cap"})
                    continue
                charge = delta * self.bytes_per_kv_token
                reduced = {**self.parent_config,
                    "history_budget_bytes": self.config.history_budget_bytes - charge,
                    "workspace_budget_bytes": self.config.workspace_budget_bytes - charge}
                if min(reduced["history_budget_bytes"], reduced["workspace_budget_bytes"]) <= 0:
                    dropped.append({"source_id": index.pop()["source_id"], "reason": "no_history_capacity"})
                    continue
                try:
                    allocator = SourceNeedsRuntime(reduced, self._token_counter)
                    trial_out, trial_counts = allocator.apply(messages, full_messages, full_counts, context, tools,
                        render_compressed=render)
                except CapacityInfeasible:
                    dropped.append({"source_id": index.pop()["source_id"], "reason": "protected_workspace_and_gist"})
                    continue
                actual_delta = raw_count(trial_out, trial_tools) - raw_count(trial_out, tools)
                if actual_delta != delta:
                    raise PolicyInputError("Schema charge changed when the actor workspace was allocated")
                raw_visible = set(trial_counts["memory_runtime"]["selected_source_indices"])
                redundant = [entry["source_id"] for entry in index
                             if set(store.event(entry["source_id"]).source_indices) <= raw_visible]
                if redundant:
                    dropped.extend({"source_id": event_id, "reason": "raw_visible_after_allocation"} for event_id in redundant)
                    index = [entry for entry in index if entry["source_id"] not in redundant]
                    continue
                out, counts, offered_tools, overhead = trial_out, trial_counts, trial_tools, delta
                status = "offered"
                break
            meta = counts["memory_runtime"]
            meta.update(route_mode=self.route_mode, history_budget_bytes=self.config.history_budget_bytes,
                        workspace_budget_bytes=self.config.workspace_budget_bytes)
            if overhead:
                for field in ("active_history_bytes", "evidence_bytes"):
                    meta[field] += overhead * self.bytes_per_kv_token
                for field in ("raw_history_tokens", "total_raw_prompt_tokens"):
                    meta[field] += overhead
                full = measure_full_history(full_messages, full_counts, self._token_counter, tools, self.bytes_per_kv_token)
                meta["compression_ratio"] = ratio_accounting(full, meta)
                meta["compression_ratio"]["includes_coverage_loss"] = bool(meta["source_coverage"]["unrepresented_source_indices"])
            if raw_count(out, offered_tools) != meta["total_raw_prompt_tokens"]:
                raise PolicyInputError("Actor evidence prompt accounting disagrees with actual tools")
            if meta["active_history_bytes"] > min(self.config.history_budget_bytes, self.config.workspace_budget_bytes):
                raise CapacityInfeasible("Actor evidence schema was not charged to B/W")
            meta["actor_evidence"] = {"version": VERSION, "status": status,
                "candidate_event_ids": [entry["source_id"] for entry in index] if status == "offered" else [],
                "index": deepcopy(index) if status == "offered" else [], "dropped_candidates": dropped,
                "schema_tokens": overhead, "schema_bytes": overhead * self.bytes_per_kv_token,
                "schema_counts_toward_history_and_workspace": True,
                "index_excludes_raw_visible_sources": True, "maximum_internal_requests": 1,
                "maximum_requested_sources": 2, "application_tools_changed": False,
                "base_lexical_active_history_bytes": base_counts["memory_runtime"]["active_history_bytes"]}
            meta["compressed_assembly_wall_sec"] = extraction_wall[0]
            meta["controller_wall_sec"] = time.perf_counter() - started - extraction_wall[0]
            prepared = PreparedActorEvidence(self, deepcopy(messages), deepcopy(full_messages), deepcopy(full_counts),
                deepcopy(context), tools, offered_tools, deepcopy(index) if status == "offered" else [],
                render, deepcopy(counts))
            return out, counts, prepared

    def reconsider_actor(self, prepared, response):
        with self._lock:
            if not isinstance(prepared, PreparedActorEvidence) or prepared.owner is not self or prepared.completed:
                raise PolicyInputError("Actor evidence decision is foreign or already resolved")
            prepared.completed = True
            calls = response.get("tool_calls") or []
            internal = [call for call in calls if call.get("function", {}).get("name") == TOOL_NAME]
            if not internal:
                return None
            if not prepared.index:
                raise PolicyInputError("Actor requested an evidence tool that was not offered")
            requested = parse_prediction({"tool_calls": internal}, {"index": prepared.index})
            if not requested.source_ids:
                raise PolicyInputError("Invalid internal evidence request")
            started = time.perf_counter()
            out, counts = self._apply_needs(prepared.messages, prepared.full_messages, prepared.full_counts,
                prepared.context, prepared.application_tools, prepared.render_compressed, None,
                source_request_override=requested)
            meta = counts["memory_runtime"]
            meta["actor_evidence"] = {**prepared.first_counts["memory_runtime"]["actor_evidence"],
                "status": "requested", "requested_event_ids": list(requested.source_ids),
                "admitted_event_ids": meta["source_needs"]["admitted_event_ids"],
                "skipped_for_budget": meta["source_needs"]["skipped_for_budget"],
                "mixed_application_calls_discarded": len(calls) - len(internal),
                "request_submitted_to_executor": False,
                "initial_schema_tokens": prepared.first_counts["memory_runtime"]["actor_evidence"]["schema_tokens"],
                "schema_tokens": 0, "schema_bytes": 0, "final_schema_tokens": 0,
                "recovery_rule": "reallocate once at original B/W, then regenerate with original application tools"}
            meta["source_needs"]["strategy"] = "actor_native_request"
            meta["controller_wall_sec"] = (prepared.first_counts["memory_runtime"]["controller_wall_sec"]
                                            + time.perf_counter() - started)
            meta["compressed_assembly_wall_sec"] = prepared.first_counts["memory_runtime"]["compressed_assembly_wall_sec"]
            return out, counts
