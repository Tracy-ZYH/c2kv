"""Bind an explicit A policy to its detector, controller, and lease implementation.

This is a source identity check for one local implementation. Checkpoint packing,
tokenization, model geometry, sampling, and task selection remain separate run
contracts; a matching method contract does not establish held-out readiness.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


METHOD_SCHEMA = "a-event-native-method-contract-v1"
METHOD_ID = "a-event-native-exact-source-baseline-v1"
ROOT = Path(__file__).resolve().parents[2]

# Fixed relative paths only: a policy file cannot request arbitrary file reads.
_SOURCE_ROLES = {
    "benchmarks/memory_runtime/policy.py": "base protection, byte admission, and revision recognition",
    "benchmarks/memory_runtime/exact_gap.py": "exact-source detector and literal matching",
    "benchmarks/memory_runtime/exact_policy.py": "one-event upgrade and evidence lease lifecycle",
    "benchmarks/memory_runtime/event_native_policy.py": "policy schema, token accounting, and packing interface",
    "benchmarks/memory_runtime/event_native_exact_policy.py": "capacity gate, gist refill, and post-draft recovery",
    "benchmarks/memory_runtime/event_native_raw.py": "Full and NoGist representation and raw refill",
    "benchmarks/memory_runtime/event_native_controls.py": "finite route definitions and controller selection",
    "benchmarks/memory_runtime/event_native_draft.py": "native draft parsing before exact-source detection",
    "benchmarks/memory_runtime/event_native_step.py": "one-regeneration orchestration and final-only response",
    "benchmarks/memory_runtime/event_native_eval_policy.py": "A parameter override and method admission",
    "benchmarks/memory_runtime/event_native_method_contract.py": "method identity and behavior catalog",
}
_VERSION_FIELDS = {
    "benchmarks/memory_runtime/exact_gap.py": ("VERSION",),
    "benchmarks/memory_runtime/exact_policy.py": ("EXACT_POLICY_VERSION",),
    "benchmarks/memory_runtime/event_native_policy.py": ("EVENT_NATIVE_POLICY_VERSION",),
    "benchmarks/memory_runtime/event_native_exact_policy.py": ("EVENT_NATIVE_EXACT_VERSION", "BUDGETED_GIST_LAYOUT"),
    "benchmarks/memory_runtime/event_native_raw.py": ("RAW_CONTROL_VERSION",),
    "benchmarks/memory_runtime/event_native_controls.py": ("EVENT_NATIVE_CONTROL_VERSION",),
    "benchmarks/memory_runtime/event_native_draft.py": ("NATIVE_DRAFT_VERSION",),
}

_BEHAVIOR = {
    "detector": {
        "input": "observable EventStore, actual raw source indices, and one unsubmitted native draft",
        "binding": "nonempty JSON-object argument string leaves; case-sensitive exact literal boundaries; JSON keys excluded",
        "candidate": "all missing bindings must have one complete source each and resolve to the same event before source_cutoff",
        "judges_action_correctness": False,
        "pre_draft_lexical_retrieval": False,
        "status_reasons": {
            "no_op": ["no_native_tool_calls", "no_string_bindings", "all_bindings_visible"],
            "abstain": ["malformed_arguments", "missing_source", "ambiguous_sources", "multiple_source_events"],
            "gap": ["missing_unique_complete_source"],
        },
        "controller_status_reasons": {
            "no_op": ["recovery_disabled"],
            "abstain": ["draft_parse_error", "budget_exhausted"],
        },
    },
    "recovery": {
        "max_events_per_upgrade": 1,
        "max_regenerations_per_decision": 1,
        "same_handle_same_source": "idempotent; no additional decision-clock tick",
        "second_source_or_different_draft": "rejected",
        "admission": "complete evidence packet must fit min(B,W), then actual representation and checkpoint limits",
        "submitted_response": "only the final generation; discarded draft tool calls are never executed",
        "cost": "all actual generation attempts are journaled; incomplete work retains unknown usage",
    },
    "protection_and_refill": {
        "protection_order": ["non-visible incomplete events and latest user", "latest non-visible complete tool event", "active exact leases in event order"],
        "gist_refill_order": "first static gist event, then remaining static gist events by descending last source index",
        "no_gist_refill_order": "after mandatory raw and shared evidence, descending last source index; skip candidates that do not fit",
        "budget_skip_reasons": ["recent_complete_tool", "active_exact_lease"],
        "gist_admission_reasons": ["system_budget", "workspace_token_budget", "encoder_budget", "workspace_byte_budget", "model_logical_context", "packing_budget_exceeded"],
        "ratio_qualified_gist_admission_reasons": ["physical_sequence_budget:{ratio}", "history_byte_budget:{ratio}"],
        "raw_admission_reasons": ["history_budget", "workspace_budget", "sequence_budget"],
        "diagnostic_detail_fields": ["skipped_gist_refill_events[].detail", "exact_recovery.parse_error", "response.native_parse_reason"],
        "diagnostic_detail_scope": "free-form parsing/exception detail; not an additional detector decision code",
    },
    "lease_release": {
        "clock": "one tick per new decision, including Full-identity bypass; repeated decision is idempotent",
        "acquisition": "persistent modes only, if lease_decisions is nonzero; expiry = acquisition decision + lease_decisions",
        "expiry": "remove pin when current decision >= expiry; acquisition counts toward the lease duration",
        "once": "no cross-decision lease; later base protection can independently select the same source",
        "raw_visibility": "avoid duplicate evidence without refreshing or pausing expiry",
        "revision": "a new latest-user event matching the fixed revision recognizer clears all exact leases",
        "missing_or_incomplete_source": "drop the lease",
        "budget_pressure": "skip a pin that does not fit this decision; preserve its original expiry",
        "session_end": "runner.close releases generator cache; controller state remains isolated by session ID until finite process teardown",
        "release_metadata_fields": ["expired_lease_event_ids", "revision_cancelled_event_ids", "dropped_lease_event_ids"],
    },
    "routes": {
        "full_original": {"history_budget": "unused", "workspace_budget": "unused", "lease": False, "recovery": False},
        "full_exact_shared": {"history_budget": "auxiliary_gate_and_selection_only", "workspace_budget": "evidence_limit", "lease": True, "recovery": True},
        "capacity_protect": {"history_budget": "history_limit", "workspace_budget": "evidence_limit", "lease": False, "recovery": False},
        "capacity_exact_once": {"history_budget": "history_limit", "workspace_budget": "evidence_limit", "lease": False, "recovery": True},
        "capacity_exact_persistent": {"history_budget": "history_limit", "workspace_budget": "evidence_limit", "lease": True, "recovery": True},
        "capacity_exact_no_gist": {"history_budget": "history_limit", "workspace_budget": "evidence_limit", "lease": True, "recovery": True},
    },
    "excluded_routes": ["event-native training-static", "historical pre-draft lexical recovery", "legacy 1088 proxy"],
    "separate_run_contracts": ["checkpoint weights and profile", "event/packing/tokenizer interface", "KV geometry and dtype", "ratio and model context", "decode/cache strategy", "sampling and total run budget", "task split and official scorer"],
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def method_sha256(contract: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(contract).encode("utf-8")).hexdigest()


def current_method_contract() -> dict[str, Any]:
    """Describe the current supported method without importing model modules."""
    sources, versions = [], {}
    for relative, role in sorted(_SOURCE_ROLES.items()):
        source = (ROOT / relative).read_bytes()
        sources.append({"path": relative, "sha256": hashlib.sha256(source).hexdigest(), "role": role})
        names = _VERSION_FIELDS.get(relative, ())
        if names:
            assignments = {}
            for node in ast.parse(source.decode("utf-8-sig"), filename=relative).body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in names:
                            assignments[target.id] = ast.literal_eval(node.value)
            if set(assignments) != set(names) or any(type(value) is not str for value in assignments.values()):
                raise ValueError(f"method version constants are missing or invalid in {relative}")
            versions[relative] = assignments
    return {
        "schema": METHOD_SCHEMA, "method_id": METHOD_ID,
        "module_versions": versions, "source_allowlist": sources,
        "behavior_catalog": copy.deepcopy(_BEHAVIOR),
    }


def validate_method_contract(value: Any) -> dict[str, Any]:
    """Reject policy/source/catalog drift before the inference entry point loads weights."""
    if not isinstance(value, Mapping):
        raise TypeError("method_contract must be a mapping")
    expected = current_method_contract()
    try:
        actual_json = _canonical(dict(value))
    except (TypeError, ValueError) as error:
        raise ValueError("method_contract must contain finite JSON data") from error
    if actual_json != _canonical(expected):
        raise ValueError("method_contract differs from the current A method source, versions, or behavior catalog")
    return copy.deepcopy(expected)
