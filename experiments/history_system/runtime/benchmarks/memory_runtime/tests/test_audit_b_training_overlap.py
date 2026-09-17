"""CPU acceptance tests for the B-corpus formal-overlap audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.memory_runtime.audit_b_training_overlap import audit_formal_corpus


SCHEMA_VERSION = "history-memory-paired-v1"
PROFILE = "history-event-base-query-v1"


def _json_bytes(value: object) -> bytes:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return (payload + "\n").encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(_json_bytes(row) for row in rows))


def _integrity(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def _decision_id(session: dict, source_message_index: int) -> str:
    identity = json.dumps(
        [
            session["source"],
            session["session_id"],
            session["task_id"],
            session["template_id"],
            source_message_index,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "decision-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _paired_row(session: dict, source_message_index: int = 2) -> dict:
    target = session["messages"][source_message_index]
    target_sha256 = hashlib.sha256(
        json.dumps(
            target, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "session_index": session["session_index"],
        "session_key": json.dumps(
            [session["source"], session["session_id"]], ensure_ascii=False, separators=(",", ":")
        ),
        "decision_id": _decision_id(session, source_message_index),
        "decision_index": 0,
        "source_message_index": source_message_index,
        "source": session["source"],
        "split": session["split"],
        "ratio": 4,
        "weight": 1.0,
        "repetition_index": 0,
        "target_sha256": target_sha256,
        "arms": {"C": {"view": {}}, "B": {"view": {}}},
    }


def _refresh_integrity(corpus_dir: Path) -> None:
    manifest_path = corpus_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest["files"]
    manifest["file_integrity"] = {
        filename: _integrity(corpus_dir / filename)
        for filename in (files["sessions"], files["paired_decisions"])
    }
    _write_json(manifest_path, manifest)


def _write_corpus(tmp_path: Path) -> tuple[Path, str]:
    corpus_dir = tmp_path / "prepared-corpus"
    sessions = [
        {
            "schema_version": SCHEMA_VERSION,
            "session_index": 0,
            "source": "unknown-importer-v99",
            "session_id": "opaque-session-1",
            "task_id": "Ｍulti—turn BASE 01",
            "template_id": "opaque-template",
            "split": "train",
            "tools": [],
            "messages": [
                {"role": "system", "content": "Use the supplied tools."},
                {"role": "user", "content": "Make   a REMINDER"},
                {"role": "assistant", "content": "I will create it."},
                {"role": "user", "content": "Future-only request"},
                {"role": "assistant", "content": "This must not enter the first prefix."},
            ],
        },
        {
            "schema_version": SCHEMA_VERSION,
            "session_index": 1,
            "source": "another-unknown-source",
            "session_id": "opaque-session-2",
            "task_id": "Development Task",
            "template_id": "opaque-template-2",
            "split": "train",
            "tools": [],
            "messages": [
                {"role": "system", "content": "Use the supplied tools."},
                {"role": "user", "content": "Development-only request"},
                {"role": "assistant", "content": "I will create it."},
            ],
        },
    ]
    records = [_paired_row(session) for session in sessions]
    _write_jsonl(corpus_dir / "sessions.jsonl", sessions)
    _write_jsonl(corpus_dir / "paired_decisions.jsonl", records)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "packing_version": "history-event-v1",
        "raw_layout_profile": "event-native-evidence-v1",
        "evidence_version": "history-evidence-v1",
        "files": {
            "sessions": "sessions.jsonl",
            "paired_decisions": "paired_decisions.jsonl",
        },
        "file_integrity": {
            "sessions.jsonl": _integrity(corpus_dir / "sessions.jsonl"),
            "paired_decisions.jsonl": _integrity(corpus_dir / "paired_decisions.jsonl"),
        },
        "arms": ["C", "B"],
        "tokenizer": {"sha256": "metadata-only-tokenizer-fixture"},
        "packing": {"ratios": [4]},
        "policy": {},
        "counts": {
            "sessions_written": len(sessions),
            "base_decisions_written": len(records),
            "paired_exposures_written": len(records),
        },
        "allow_unchanged_b": False,
    }
    _write_json(corpus_dir / "manifest.json", manifest)
    manifest_sha256 = hashlib.sha256((corpus_dir / "manifest.json").read_bytes()).hexdigest()
    return corpus_dir, manifest_sha256


def _write_bfcl_tasks(tmp_path: Path) -> Path:
    path = tmp_path / "BFCL_v4_multi_turn_base.json"
    _write_jsonl(
        path,
        [
            {
                "id": "multi_turn_base_01",
                "question": [[{"role": "user", "content": "Canonical-id only"}]],
            },
            {
                "id": "prompt-only",
                "question": [[{"role": "user", "content": " make a    reminder "}]],
            },
            {
                "id": "future-only",
                "question": [[{"role": "user", "content": "Future-only request"}]],
            },
            {
                "id": "bfcl:multi_turn_base_01",
                "question": [[{"role": "user", "content": "Prefixed-id only"}]],
            },
            {
                "id": "development-task",
                "question": [[{"role": "user", "content": "Development candidate ID only"}]],
            },
            {
                "id": "unseen",
                "question": [[{"role": "user", "content": "Never used"}]],
            },
        ],
    )
    return path


def _write_checkpoint(
    tmp_path: Path,
    identity: str,
    *,
    config_identity: str | None = None,
    trainer_identity: str | None = None,
    config_arm: str = "B",
    trainer_arm: str = "B",
    config_profile: str = PROFILE,
    trainer_profile: str = PROFILE,
) -> Path:
    checkpoint = tmp_path / "checkpoint-1"
    _write_json(
        checkpoint / "config.json",
        {
            "history_memory_corpus_identity": config_identity or identity,
            "history_memory_arm": config_arm,
            "history_memory_training_profile": config_profile,
            "history_memory_seed": 17,
            "history_memory_packing_version": "history-event-v1",
            "history_memory_raw_layout": "event-native-evidence-v1",
            "history_memory_evidence_version": "history-evidence-v1",
        },
    )
    _write_json(
        checkpoint / "trainer_state.json",
        {
            "contract": {
                "corpus_identity": trainer_identity or identity,
                "arm": trainer_arm,
                "profile": trainer_profile,
                "seed": 17,
            },
            "training_profile": trainer_profile,
        },
    )
    return checkpoint


def _audit(corpus_dir: Path, tasks_path: Path, *, checkpoints=()):
    return audit_formal_corpus(
        corpus_dir,
        bfcl_tasks_path=tasks_path,
        candidate_ids=[
            "multi_turn_base_01",
            "prompt-only",
            "future-only",
            "bfcl:multi_turn_base_01",
            "development-task",
            "unseen",
        ],
        dev_task_ids=["development-task"],
        checkpoint_dirs=checkpoints,
    )


def test_binds_checkpoint_and_audits_all_sources_prefix_locally(tmp_path: Path) -> None:
    corpus_dir, manifest_sha256 = _write_corpus(tmp_path)
    checkpoint = _write_checkpoint(tmp_path, manifest_sha256)
    report = _audit(corpus_dir, _write_bfcl_tasks(tmp_path), checkpoints=[checkpoint])

    assert report["schema"] == "c2kv-b-training-overlap-v1"
    assert report["status"] == "checkpoint_bound"
    assert report["formal_split_frozen"] is False
    assert report["interpretation"] == "exact_match_found"
    assert "corpus" in report and "training_prefixes" in report and "checkpoint_binding" in report
    assert set(report["corpus"]) == {"manifest", "files", "contract"}
    assert report["checkpoint_binding"]["status"] == "bound"
    assert len(report["checkpoint_binding"]["checkpoints"]) == 1
    assert report["overlap"]["candidate"]["interpretation"] == "exact_match_found"
    canonical_ids = report["overlap"]["candidate"]["canonical_exact_task_ids"]
    prompt_ids = report["overlap"]["candidate"]["normalized_exact_user_prompts"]
    assert set(canonical_ids["matched_task_ids"]) == {
        "development-task",
        "multi_turn_base_01",
    }
    assert set(prompt_ids["matched_task_ids"]) == {"prompt-only"}
    assert prompt_ids["matches"][0]["training_prefix_hits"][0]["source"] == "unknown-importer-v99"
    assert report["training_prefixes"]["future_turns_scanned"] is False
    assert report["source_alias_diagnostic"]["used_as_overlap_filter"] is False
    development_ids = report["overlap"]["development"]["canonical_exact_task_ids"]
    assert set(development_ids["matched_task_ids"]) == {"development-task"}
    assert "development-task" in report["excluded_candidate_ids"]
    assert {"future-only", "bfcl:multi_turn_base_01", "unseen"} <= set(report["eligible_candidate_ids"])
    assert "development-task" not in report["eligible_candidate_ids"]


def test_corpus_without_checkpoint_is_explicitly_corpus_only(tmp_path: Path) -> None:
    corpus_dir, _ = _write_corpus(tmp_path)
    report = _audit(corpus_dir, _write_bfcl_tasks(tmp_path))

    assert report["status"] == "corpus_only"
    assert report["formal_split_frozen"] is False
    assert report["checkpoint_binding"] == {
        "status": "not_provided",
        "corpus_identity": report["corpus"]["manifest"]["sha256"],
        "checkpoints": [],
    }


def test_rejects_bound_file_integrity_mismatch(tmp_path: Path) -> None:
    corpus_dir, _ = _write_corpus(tmp_path)
    tasks_path = _write_bfcl_tasks(tmp_path)
    with (corpus_dir / "sessions.jsonl").open("ab") as handle:
        handle.write(b"\n")

    with pytest.raises(ValueError, match="integrity"):
        _audit(corpus_dir, tasks_path)


@pytest.mark.parametrize(
    "checkpoint_kwargs",
    [
        {"config_identity": "another-manifest"},
        {"trainer_identity": "another-manifest"},
        {"config_arm": "C"},
        {"trainer_arm": "C"},
        {"config_profile": "another-profile"},
        {"trainer_profile": "another-profile"},
    ],
    ids=["config-manifest", "trainer-manifest", "config-arm", "trainer-arm", "config-profile", "trainer-profile"],
)
def test_rejects_checkpoint_manifest_arm_or_profile_mismatch(
    tmp_path: Path, checkpoint_kwargs: dict[str, str]
) -> None:
    corpus_dir, manifest_sha256 = _write_corpus(tmp_path)
    checkpoint = _write_checkpoint(tmp_path, manifest_sha256, **checkpoint_kwargs)

    with pytest.raises(ValueError):
        _audit(corpus_dir, _write_bfcl_tasks(tmp_path), checkpoints=[checkpoint])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("session_index", 9),
        ("session_key", "[\\\"unknown-importer-v99\\\",\\\"wrong-session\\\"]"),
        ("source_message_index", 1),
        ("decision_id", "decision-not-bound-to-session"),
    ],
    ids=["session-index", "session-reference", "source-message-index", "decision-binding"],
)
def test_rejects_invalid_paired_session_reference_or_index(
    tmp_path: Path, field: str, value: object
) -> None:
    corpus_dir, _ = _write_corpus(tmp_path)
    records_path = corpus_dir / "paired_decisions.jsonl"
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    records[0][field] = value
    _write_jsonl(records_path, records)
    _refresh_integrity(corpus_dir)

    with pytest.raises(ValueError):
        _audit(corpus_dir, _write_bfcl_tasks(tmp_path))
