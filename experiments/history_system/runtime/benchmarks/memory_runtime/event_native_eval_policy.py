"""Resolve an explicit A evaluation policy without changing checkpoint contracts."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .event_native_always import NATIVE_ALWAYS_ROUTE_MODES

EVAL_POLICY_SCHEMA = "a-event-native-eval-policy-v1"
EVAL_POLICY_V2_SCHEMA = "a-event-native-eval-policy-v2"
RUNTIME_POLICY_SCHEMA = "a-event-native-runtime-policy-v1"
EVAL_POLICY_FIELDS = frozenset(
    {
        "history_budget_bytes",
        "workspace_budget_bytes",
        "lease_decisions",
        "max_retrieved_events",
    }
)
OVERRIDE_VIEW_MODES = frozenset(
    {
        "capacity_protect",
        "capacity_exact_once",
        "capacity_exact_persistent",
        "full_exact_shared",
        "capacity_exact_no_gist",
        "full_original",
    }
)
ALWAYS_VIEW_MODES = NATIVE_ALWAYS_ROUTE_MODES
KNOWN_VIEW_MODES = OVERRIDE_VIEW_MODES | ALWAYS_VIEW_MODES | {"static"}
RECOVERY_VIEW_MODES = frozenset(
    {
        "capacity_exact_once",
        "capacity_exact_persistent",
        "full_exact_shared",
        "capacity_exact_no_gist",
        "ac_exact_once",
        "ac_exact_persistent",
        "ac_full_shared",
        "raw_exact_shared",
    }
)
PERSISTENT_EVIDENCE_VIEW_MODES = frozenset(
    {
        "capacity_exact_persistent",
        "full_exact_shared",
        "capacity_exact_no_gist",
        "ac_exact_persistent",
        "ac_full_shared",
        "raw_exact_shared",
    }
)


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _validate_eval_policy(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("evaluation policy must be a mapping")
    required = {"schema", "policy_id", "policy"}
    schema = value.get("schema")
    if schema == EVAL_POLICY_V2_SCHEMA:
        required.add("method_contract")
    if set(value) != required:
        raise ValueError(
            "evaluation policy must contain exactly " + ", ".join(sorted(required))
        )
    if schema not in (EVAL_POLICY_SCHEMA, EVAL_POLICY_V2_SCHEMA):
        raise ValueError(f"evaluation policy schema must be {EVAL_POLICY_SCHEMA!r} or {EVAL_POLICY_V2_SCHEMA!r}")
    policy_id = value["policy_id"]
    if not isinstance(policy_id, str) or not policy_id.strip():
        raise ValueError("evaluation policy policy_id must be a nonempty string")
    policy = value["policy"]
    if not isinstance(policy, Mapping):
        raise TypeError("evaluation policy policy must be a mapping")
    if set(policy) != EVAL_POLICY_FIELDS:
        raise ValueError(
            "evaluation policy policy must contain exactly "
            + ", ".join(sorted(EVAL_POLICY_FIELDS))
        )
    for field in sorted(EVAL_POLICY_FIELDS):
        if type(policy[field]) is not int or policy[field] < 0:
            raise ValueError(
                f"evaluation policy policy.{field} must be a nonnegative integer"
            )
    validated = {
        "schema": schema,
        "policy_id": policy_id,
        "policy": {field: policy[field] for field in sorted(EVAL_POLICY_FIELDS)},
    }
    if schema == EVAL_POLICY_V2_SCHEMA:
        from .event_native_method_contract import validate_method_contract
        validated["method_contract"] = validate_method_contract(value["method_contract"])
    return validated


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _recorded_source_path(source_path: str | os.PathLike[str] | None) -> str | None:
    if source_path is None:
        return None
    try:
        result = os.fspath(source_path)
    except TypeError as error:
        raise TypeError("source_path must be a string, path-like object, or None") from error
    if not isinstance(result, str):
        raise TypeError("source_path must resolve to a string path")
    if not result:
        raise ValueError("source_path must be nonempty when provided")
    return result


def _runtime_semantics(view_mode: str) -> tuple[int, dict[str, str]]:
    runtime_recovery_cap = 1 if view_mode in RECOVERY_VIEW_MODES else 0
    return runtime_recovery_cap, {
        "history_budget_bytes": (
            "unused"
            if view_mode == "full_original"
            else "auxiliary_gate_and_selection_only"
            if view_mode in {"full_exact_shared", "ac_full_shared"}
            else "history_limit"
        ),
        "workspace_budget_bytes": (
            "unused" if view_mode == "full_original" else "evidence_limit"
        ),
        "lease_decisions": (
            "retained_evidence_lifetime"
            if view_mode in PERSISTENT_EVIDENCE_VIEW_MODES
            else "unused"
        ),
        "max_retrieved_events": (
            "fixed_single_event_upgrade"
            if view_mode in RECOVERY_VIEW_MODES
            else "unused"
        ),
    }


def load_eval_policy(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load and strictly validate one external A evaluation-policy JSON file."""

    payload = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_object_without_duplicate_keys,
    )
    return _validate_eval_policy(payload)


def resolve_event_native_eval_policy(
    profile: Mapping[str, Any],
    *,
    view_mode: str,
    policy_override: Mapping[str, Any] | None = None,
    source_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Validate the checkpoint policy and optionally replace four A eval fields."""

    if not isinstance(profile, Mapping):
        raise TypeError("profile must be a mapping")
    training_policy = profile.get("policy_contract")
    if not isinstance(training_policy, Mapping):
        raise TypeError("profile.policy_contract must be a mapping")
    training_policy = copy.deepcopy(dict(training_policy))

    # Keep import-time dependencies stdlib-only and avoid importing event_native,
    # which owns the checkpoint-facing caller of this resolver.
    from .event_native_policy import EventNativeController

    EventNativeController._parse_policy(training_policy)

    if not isinstance(view_mode, str) or view_mode not in KNOWN_VIEW_MODES:
        raise ValueError(f"view_mode must be one of {sorted(KNOWN_VIEW_MODES)!r}")
    runtime_recovery_cap, field_roles = _runtime_semantics(view_mode)

    recorded_path = _recorded_source_path(source_path)
    if policy_override is None:
        if recorded_path is not None:
            raise ValueError("source_path requires policy_override")
        effective_policy = copy.deepcopy(training_policy)
        EventNativeController._parse_policy(effective_policy)
        return {
            "schema": RUNTIME_POLICY_SCHEMA,
            "source": "checkpoint_training_policy",
            "policy_id": None,
            "policy_sha256": None,
            "eval_policy": None,
            "source_path": None,
            "effective_policy": copy.deepcopy(effective_policy),
            "runtime_recovery_cap": runtime_recovery_cap,
            "field_roles": copy.deepcopy(field_roles),
        }

    if view_mode == "static":
        raise ValueError("static view_mode cannot use an explicit evaluation policy")
    if view_mode not in OVERRIDE_VIEW_MODES | ALWAYS_VIEW_MODES:
        raise ValueError(
            "evaluation policy override requires one of "
            f"{sorted(OVERRIDE_VIEW_MODES | ALWAYS_VIEW_MODES)!r}"
        )

    eval_policy = _validate_eval_policy(policy_override)
    if (
        view_mode in ALWAYS_VIEW_MODES
        and eval_policy["schema"] == EVAL_POLICY_V2_SCHEMA
    ):
        raise ValueError(
            "always-compress routes cannot use the frozen legacy v2 method_contract; "
            "use the dedicated v1 policy plus explicit compression_policy identity"
        )
    if (
        view_mode in RECOVERY_VIEW_MODES
        and eval_policy["policy"]["max_retrieved_events"] != 1
    ):
        raise ValueError(
            "explicit evaluation policy requires max_retrieved_events=1 for "
            "event-native recovery routes"
        )
    effective_policy = copy.deepcopy(training_policy)
    for field in EVAL_POLICY_FIELDS:
        effective_policy[field] = eval_policy["policy"][field]
    EventNativeController._parse_policy(effective_policy)

    resolved = {
        "schema": RUNTIME_POLICY_SCHEMA,
        "source": "explicit_eval_policy",
        "policy_id": eval_policy["policy_id"],
        "policy_sha256": _canonical_sha256(eval_policy),
        "eval_policy": copy.deepcopy(eval_policy),
        "source_path": recorded_path,
        "effective_policy": copy.deepcopy(effective_policy),
        "runtime_recovery_cap": runtime_recovery_cap,
        "field_roles": copy.deepcopy(field_roles),
    }
    if eval_policy["schema"] == EVAL_POLICY_V2_SCHEMA:
        from .event_native_method_contract import method_sha256
        resolved["method_contract"] = copy.deepcopy(eval_policy["method_contract"])
        resolved["method_sha256"] = method_sha256(eval_policy["method_contract"])
    return resolved
