"""Connect a checkpoint-bound overlap audit to an actual BFCL invocation.

This is an execution check for the audit's exact exclusions. It does not select
tasks, freeze a formal split, certify weights, or establish contamination freedom.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .audit_b_training_overlap import AUDIT_SCHEMA


QUESTION_FILENAME = "BFCL_v4_multi_turn_base.json"


def _identity(path: Path) -> dict:
    payload = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload)}


def _same_content(actual: dict, expected: dict) -> bool:
    return isinstance(expected, dict) and all(
        actual[key] == expected.get(key) for key in ("sha256", "bytes")
    )


def _ids(value, label: str) -> list[str]:
    if (not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value)
            or len(value) != len(set(value))):
        raise ValueError(f"{label} must contain unique nonempty task IDs")
    return value


def validate_overlap_admission(
    audit_path: Path, ready: dict, question_path: Path, *,
    expected_audit_sha256: str | None = None,
) -> dict:
    """Validate actual task inputs and checkpoint metadata without model work."""
    audit_path, question_path = Path(audit_path), Path(question_path)
    payload = audit_path.read_bytes()
    audit_identity = {"path": str(audit_path.resolve()), "sha256": hashlib.sha256(payload).hexdigest(),
                      "bytes": len(payload)}
    if expected_audit_sha256 is not None and audit_identity["sha256"] != expected_audit_sha256:
        raise ValueError("overlap audit changed after parent admission")
    audit = json.loads(payload)
    if not isinstance(audit, dict) or audit.get("schema") != AUDIT_SCHEMA:
        raise ValueError("a BFCL prepared-corpus overlap audit is required")
    if (audit.get("status") != "checkpoint_bound"
            or audit.get("validation") != {"status": "passed", "corpus_contract": "passed",
                                           "checkpoint_contract": "passed"}):
        raise ValueError("overlap admission requires a passed checkpoint_bound audit")
    binding = audit.get("checkpoint_binding", {})
    corpus_identity = audit.get("corpus", {}).get("manifest", {}).get("sha256")
    if (not isinstance(corpus_identity, str) or len(corpus_identity) != 64
            or binding.get("status") != "bound" or binding.get("corpus_identity") != corpus_identity):
        raise ValueError("overlap audit has inconsistent corpus/checkpoint binding")

    selected = _ids(ready.get("allowed_task_ids"), "server task selection")
    if not selected or any(not task.startswith("multi_turn_base_")
                           or not task.removeprefix("multi_turn_base_").isdigit() for task in selected):
        raise ValueError("overlap admission requires explicit multi_turn_base task IDs")
    eligible = set(_ids(audit.get("eligible_candidate_ids"), "eligible candidates"))
    excluded = set(_ids(audit.get("excluded_candidate_ids"), "excluded candidates"))
    candidate_overlap = audit.get("overlap", {}).get("candidate", {})
    matched = set()
    for field in ("canonical_exact_task_ids", "normalized_exact_user_prompts"):
        matched.update(_ids(candidate_overlap.get(field, {}).get("matched_task_ids"), field))
    if eligible & excluded or not matched <= excluded:
        raise ValueError("overlap audit has inconsistent eligible/excluded candidate IDs")
    if not set(selected) <= eligible or set(selected) & (excluded | matched):
        raise ValueError("server selected an excluded or unaudited task")

    source_identity = _identity(question_path)
    if not _same_content(source_identity, audit.get("bfcl_source")):
        raise ValueError("actual BFCL question source differs from the overlap audit")
    # This also catches duplicate serialized rows in older audit inputs. A
    # selected ID must resolve to exactly one task in the file used by BFCL.
    source_ids = [json.loads(line)["id"] for line in question_path.read_bytes().splitlines() if line.strip()]
    if any(source_ids.count(task) != 1 for task in selected):
        raise ValueError("selected task IDs must occur exactly once in the BFCL question source")

    profile = ready.get("checkpoint", {})
    checkpoint_path = profile.get("checkpoint") or ready.get("checkpoint_path")
    if not isinstance(checkpoint_path, str) or not checkpoint_path:
        raise ValueError("server manifest lacks a checkpoint metadata directory")
    checkpoint = Path(checkpoint_path)
    declared_path = ready.get("checkpoint_path")
    if declared_path is not None and Path(declared_path).resolve() != checkpoint.resolve():
        raise ValueError("server checkpoint paths disagree")
    if profile.get("corpus_identity") != corpus_identity:
        raise ValueError("server checkpoint corpus identity differs from the overlap audit")
    config_identity = _identity(checkpoint / "config.json")
    state_identity = _identity(checkpoint / "trainer_state.json")
    matching = [entry for entry in binding.get("checkpoints", [])
                if _same_content(config_identity, entry.get("config"))
                and _same_content(state_identity, entry.get("trainer_state"))]
    if not matching:
        raise ValueError("server checkpoint metadata differs from audited config/trainer_state")
    if any(entry.get("corpus_identity") != corpus_identity
           or entry.get("arm") != profile.get("training_arm") for entry in matching):
        raise ValueError("server checkpoint arm or corpus identity differs from audited metadata")

    return {
        "schema": "a-bfcl-overlap-admission-v1", "status": "passed",
        "audit": audit_identity, "task_ids": list(selected), "bfcl_source": source_identity,
        "checkpoint": {"config": config_identity, "trainer_state": state_identity,
                       "corpus_identity": corpus_identity, "training_arm": profile["training_arm"]},
        "formal_split_frozen": False,
        "scope": "Exact-overlap execution admission; formal split, weights, policy selection, and near/semantic contamination are separate contracts.",
    }


def official_question_path() -> Path:
    """Resolve the data file from the package actually imported by the worker."""
    from bfcl_eval.constants.eval_config import PROMPT_PATH, VERSION_PREFIX

    filename = f"{VERSION_PREFIX}_multi_turn_base.json"
    if filename != QUESTION_FILENAME:
        raise ValueError("imported BFCL version differs from the overlap admission input contract")
    return Path(PROMPT_PATH) / filename
