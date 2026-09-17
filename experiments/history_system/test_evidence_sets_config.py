"""Configuration-contract tests for the evidence_sets_v1 generator."""

from __future__ import annotations

import json
import importlib
import subprocess
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import evidence_sets
from current import RUNTIME, _configure_controller, load_config


def _artifact(tmp_path: Path, selector: str) -> Path:
    _configure_controller(evidence_sets._base_controller(), {})
    runtime_python = str(RUNTIME / "python")
    if runtime_python not in sys.path:
        sys.path.insert(0, runtime_python)
    models = importlib.import_module("_c2kv_active_recovery.set_models")
    kind = {
        "risk": "c1_risk_logistic",
        "gain_turn": "c4_gain_turn",
        "gain_task": "c4_gain_task",
        "reranker": "t01_relevance_logistic",
    }[selector]
    if selector == "risk":
        feature_contract = {
            "schema": models.C1_FEATURE_SCHEMA,
            "feature_names": list(models.C1_FEATURE_NAMES),
        }
    else:
        config, _ = evidence_sets.build_config(
            history="H0", selector="candidate_rule"
        )
        score_model_contract = {
            role: importlib.import_module(
                "_c2kv_active_recovery.local_selection_models"
            ).LocalSelectionModels(config["local_models"]).public_config()[role]
            for role in ("embedding", "reranker")
        }
        names = (
            models.T01_FEATURE_NAMES if selector == "reranker" else models.C4_FEATURE_NAMES
        )
        schema = (
            models.T01_FEATURE_SCHEMA if selector == "reranker" else models.C4_FEATURE_SCHEMA
        )
        feature_contract = {
            "schema": schema,
            "feature_names": list(names),
            "score_model_contract": score_model_contract,
        }
        if selector != "reranker":
            feature_contract["tokenizer_contract"] = {"fixture": True}
            feature_contract["semantic_query_overflow_policy"] = "error"
    artifact = {
        "schema": "c2kv-recovery-set-model-v1",
        "model_kind": kind,
        "feature_contract": feature_contract,
        "components": {},
        "provenance": {"fixture": True},
    }
    if selector.startswith("gain_"):
        artifact["target"] = "delta_turn" if selector == "gain_turn" else "delta_task"
    artifact["artifact_sha256"] = models.artifact_sha256(artifact)
    path = tmp_path / "selector.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("controller", "selector"), tuple(evidence_sets.CONTROLLER_VARIANTS.items())
)
def test_h0_c0_c5_build_through_active_controller(
    tmp_path: Path, controller: str, selector: str
) -> None:
    artifact = _artifact(tmp_path, selector) if selector in evidence_sets.TRAINED_SELECTORS else None
    config, hashes = evidence_sets.build_config(
        history="H0", selector=selector, selector_artifact=artifact
    )
    current = load_config()
    base_path = RUNTIME / current["runtime"]["controller"]
    built = _configure_controller(
        json.loads(base_path.read_text(encoding="utf-8")), config
    )

    assert built["gp_experiments"] == config
    assert hashes == {}
    assert config["G"] == "current"
    assert config["selection_protocol"] == "evidence_sets_v1"
    assert config["set_selector"] == selector
    assert "selector" not in config


def test_fixed_defaults_and_readable_limit_options() -> None:
    config, _ = evidence_sets.build_config(
        history="H0",
        selector="candidate_rule",
        candidate_pool_size=7,
        selected_evidence_max=1,
        fallback_256=True,
        reserve_tokens=512,
    )
    assert {
        key: config[key]
        for key in ("U", "B", "Q", "K", "L", "R", "D", "P", "order", "candidate_limit")
    } == {
        "U": "tokens_1024",
        "B": "source",
        "Q": "archive_rrf",
        "K": 1,
        "L": "next_decision",
        "R": 1,
        "D": "candidate_rule",
        "P": "quoted",
        "order": "chronological",
        "candidate_limit": 7,
    }
    assert config["fallback_unit"] == "tokens_256"
    assert config["recovery_reserve_tokens"] == 512
    assert "selector_max_units" not in config
    assert config["local_models"]["selector"] == {
        "model_name_or_path": "Qwen/Qwen3-4B-Instruct-2507",
        "revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
        "device": "cpu",
        "local_files_only": True,
    }
    assert config["local_models"]["reranker"]["batch_size"] == 1
    local_models = importlib.import_module(
        "_c2kv_active_recovery.local_selection_models"
    )
    assert evidence_sets.MODEL_DEFAULTS["selector"] == {
        "model_name_or_path": local_models.SELECTOR_MODEL,
        "revision": local_models.SELECTOR_REVISION,
    }


def test_c4_task_target_is_an_explicit_selector_variant(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, "gain_task")
    config, _ = evidence_sets.build_config(
        history="H0", selector="gain_task", selector_artifact=artifact
    )
    assert evidence_sets.CONTROLLER_VARIANTS["C4"] == "gain_turn"
    assert evidence_sets.CONTROLLER_VARIANTS["C5"] == "parameter_source"
    assert config["set_selector"] == "gain_task"


def test_legacy_prefill_is_an_explicit_untrained_compatibility_selector(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "legacy-prefill.json"
    assert evidence_sets.main(
        [
            "--history", "H0",
            "--selector", "legacy_prefill",
            "--out", str(out),
        ]
    ) == 0
    config = json.loads(out.read_text(encoding="utf-8"))
    receipt = json.loads(capsys.readouterr().out)

    assert config["set_selector"] == "legacy_prefill"
    assert "selector_artifact" not in config
    assert receipt["legacy_prefill_source"] == {
        "compatibility_mode": True,
        "training_performed": False,
        "gate": "prefill_linear_head",
        "head_artifact_sha256": "4bd01ef68206cbc3902da49dd0ba288b41f9c92aa8f5a2473d5ffdfdb9d97c57",
        "layer": 34,
        "threshold": 0.997483851175923,
    }


def test_t01_artifact_is_allowed_for_reranker(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, "reranker")
    config, _ = evidence_sets.build_config(
        history="H0", selector="reranker", selector_artifact=artifact
    )
    assert config["selector_artifact"]["model_kind"] == "t01_relevance_logistic"


def test_h1_is_bound_to_frozen_g04_source_hashes() -> None:
    config, hashes = evidence_sets.build_config(
        history="H1", selector="parameter_source"
    )
    assert config["G"] == "record_bound"
    assert hashes == evidence_sets.H1_SOURCE_HASHES


def test_h1_rejects_a_changed_frozen_source_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    changed = dict(evidence_sets.H1_SOURCE_HASHES)
    changed["python/history_memory/packing.py"] = "0" * 64
    monkeypatch.setattr(evidence_sets, "H1_SOURCE_HASHES", changed)
    with pytest.raises(RuntimeError, match="frozen G04 source changed"):
        evidence_sets.build_config(history="H1", selector="candidate_rule")


def test_trained_selector_requires_a_real_artifact(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires --selector-artifact"):
        evidence_sets.build_config(history="H0", selector="risk")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        evidence_sets.build_config(
            history="H0", selector="risk", selector_artifact=tmp_path / "missing.json"
        )


def test_artifact_is_validated_and_embedded_with_source_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artifact = _artifact(tmp_path, "risk")
    out = tmp_path / "risk.json"
    assert evidence_sets.main(
        [
            "--history", "H0", "--controller", "C1", "--out", str(out),
            "--selector-artifact", str(artifact),
        ]
    ) == 0
    config = json.loads(out.read_text(encoding="utf-8"))
    receipt = json.loads(capsys.readouterr().out)
    assert isinstance(config["selector_artifact"], dict)
    assert config["selector_artifact"]["model_kind"] == "c1_risk_logistic"
    assert receipt["selector_artifact_source"] == evidence_sets._artifact_receipt(artifact)


def test_trained_artifact_cli_runs_as_a_subprocess_without_model_calls(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path, "risk")
    out = tmp_path / "subprocess-risk.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(HERE / "evidence_sets.py"),
            "--history", "H0",
            "--controller", "C1",
            "--out", str(out),
            "--selector-artifact", str(artifact),
        ],
        cwd=HERE,
        check=True,
        capture_output=True,
        text=True,
    )
    receipt = json.loads(completed.stdout)
    config = json.loads(out.read_text(encoding="utf-8"))
    assert receipt["model_calls"] == 0
    assert receipt["selector_artifact_source"]["source_path"] == str(
        artifact.resolve()
    )
    assert config["selector_artifact"]["model_kind"] == "c1_risk_logistic"


def test_custom_local_model_paths_remain_offline(tmp_path: Path) -> None:
    model_path = tmp_path / "embedding-snapshot"
    config, _ = evidence_sets.build_config(
        history="H0",
        selector="candidate_rule",
        embedding_model=str(model_path),
        embedding_device="npu:3",
    )
    embedding = config["local_models"]["embedding"]
    assert embedding == {
        "model_name_or_path": str(model_path),
        "revision": None,
        "device": "npu:3",
        "local_files_only": True,
    }
    assert all(
        role["local_files_only"] for role in config["local_models"].values()
    )


def test_reranker_batch_size_is_explicit_and_validated() -> None:
    config, _ = evidence_sets.build_config(
        history="H0",
        selector="reranker",
        reranker_batch_size=1,
    )
    assert config["local_models"]["reranker"]["batch_size"] == 1
    with pytest.raises(ValueError, match="positive integer"):
        evidence_sets.build_config(
            history="H0",
            selector="reranker",
            reranker_batch_size=0,
        )


def test_c3_can_bind_complete_plain_4b_local_path(tmp_path: Path) -> None:
    selector_path = tmp_path / "Qwen3-4B-Instruct-2507"
    config, _ = evidence_sets.build_config(
        history="H0",
        selector="local_llm",
        selector_model=str(selector_path),
        selector_device="npu:0",
    )
    assert config["local_models"]["selector"] == {
        "model_name_or_path": str(selector_path),
        "revision": None,
        "device": "npu:0",
        "local_files_only": True,
    }


def test_cli_writes_gp_json_and_reports_concrete_current_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "h0-c2.json"
    checkpoint = tmp_path / "checkpoint-1000"
    run_out = tmp_path / "run"
    assert evidence_sets.main(
        [
            "--history", "H0",
            "--controller", "C2",
            "--out", str(out),
            "--candidate-pool-size", "8",
            "--selected-evidence-max", "4",
            "--checkpoint", str(checkpoint),
            "--run-out", str(run_out),
            "--task-id", "test_task",
            "--sglang-backend-url", "http://127.0.0.1:36100",
        ]
    ) == 0
    value = json.loads(out.read_text(encoding="utf-8"))
    receipt = json.loads(capsys.readouterr().out)
    assert value["set_selector"] == "reranker"
    assert value["candidate_limit"] == 8
    assert value["K"] == 4
    command = receipt["current_command"]
    assert command[1] == str(HERE / "current.py")
    assert command[2] == "preview"
    assert command[command.index("--gp-config") + 1] == str(out.resolve())
    assert receipt["freeze_controller"] == str(run_out.resolve() / "gp.controller.json")
    assert receipt["model_calls"] == 0


def test_cli_is_idempotent_but_does_not_overwrite_different_config(tmp_path: Path) -> None:
    out = tmp_path / "config.json"
    args = ["--history", "H0", "--controller", "C0", "--out", str(out)]
    assert evidence_sets.main(args) == 0
    assert evidence_sets.main(args) == 0
    with pytest.raises(FileExistsError):
        evidence_sets.main(
            ["--history", "H0", "--controller", "C3", "--out", str(out)]
        )


@pytest.mark.parametrize("value", (0, 9))
def test_candidate_pool_never_silently_truncates(value: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 8"):
        evidence_sets.build_config(
            history="H0", selector="candidate_rule", candidate_pool_size=value
        )
