from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest import mock

import pytest

import run_c1


class FinishedProcess:
    def __init__(self) -> None:
        self.returncode = 0

    def poll(self) -> int:
        return self.returncode

    def wait(self, timeout: float) -> int:
        assert timeout > 0
        return self.returncode


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        out=tmp_path / "run",
        benchmark="bfcl",
        portable_root=tmp_path,
        task_timeout=10,
        method="proposed",
        detector="t02_risk",
    )


def _write_finished_task(task_out: Path, *, risk_available: bool | None) -> None:
    server = task_out / "server"
    official = task_out / "bfcl"
    server.mkdir(parents=True)
    official.mkdir()
    (server / "ready.json").write_text("{}\n", encoding="utf-8")
    (server / "final.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "stop_reason": "completed",
                "journal_summary": {"completed": 1, "failed": 0, "pending": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (official / "official_summary.json").write_text(
        json.dumps({"n_generated": 1, "n_scored": 1, "semantic_score": 1.0})
        + "\n",
        encoding="utf-8",
    )
    native_stats = {
        "backend": "sglang_c2kv_native_packed",
        "gist_tokens": 8,
        "workspace_tokens": 4,
        "scope_reused_chunks": 1,
    }
    exact_recovery = {
        "status": "recover",
        "gate": {"type": "risk", "triggered": True},
        "selection": {
            "selector": "risk",
            "score_semantics": "current_turn_failure_risk",
            "available": risk_available,
            "score": 0.9 if risk_available else None,
        },
        "appended_unit_count": 1,
        "appended_units": [{"token_count": 8}],
    }
    if risk_available is None:
        native_stats = {
            "backend": "sglang_c2kv_native_packed",
            "gist_tokens": 0,
            "workspace_tokens": 0,
            "scope_reused_chunks": 0,
        }
        exact_recovery = {
            "status": "no_recovery",
            "gate": {
                "type": "risk",
                "triggered": False,
                "reason": "no_feasible_evidence_sets",
            },
            "selection": {
                "selector": "risk",
                "available": True,
                "reason": "no_feasible_evidence_sets",
                "score": 0.0,
                "score_semantics": None,
            },
        }
    record = {
        "generation_trace": [
            {
                "phase": "regeneration",
                "status": "completed",
                "usage": {"prompt_tokens": 16},
                "controller": {
                    "compression_ratio": {
                        "full_history_bytes": 100,
                        "active_history_bytes": 200,
                    },
                    "kv_bytes_per_token": 10,
                },
                "generation": {"stats": native_stats},
            }
        ],
        "exact_recovery": exact_recovery,
    }
    (server / "steps.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")


def _run_finished_fixture(tmp_path: Path, *, risk_available: bool):
    args = _args(tmp_path)
    task = "multi_turn_base_26"
    task_out = args.out / "task_shards" / task
    processes = [FinishedProcess(), FinishedProcess()]

    def popen(*_args, **_kwargs):
        if len(processes) == 2:
            _write_finished_task(task_out, risk_available=risk_available)
        return processes.pop(0)

    with (
        mock.patch.object(run_c1, "commands_for_task", return_value=(["server"], ["worker"])),
        mock.patch.object(run_c1.subprocess, "Popen", side_effect=popen),
        mock.patch.object(run_c1.runner, "_stop_bfcl"),
        mock.patch.object(run_c1.runner, "_stop_server"),
    ):
        return run_c1.run_task(args, task, tmp_path / "controller.json")


def test_run_task_accepts_completed_low_ratio_recovery(tmp_path):
    receipt, telemetry = _run_finished_fixture(tmp_path, risk_available=True)

    assert receipt["status"] == "completed"
    assert receipt["official_summary"]["semantic_score"] == 1.0
    assert telemetry["compression_ratio"] == 0.5
    checks = run_c1.functional_checks("proposed", "t02_risk", telemetry)
    assert all(checks["required"].values())
    assert checks["observed"]["compression_ratio_gt_one"] is False


def test_run_task_accepts_short_no_feasible_candidates_without_efficiency_signals(tmp_path):
    receipt, telemetry = _run_finished_fixture(tmp_path, risk_available=None)

    assert receipt["status"] == "completed"
    assert telemetry["native_generate_requests"] == 1
    assert telemetry["detector_calls"] == 0
    checks = run_c1.functional_checks("proposed", "t02_risk", telemetry)
    assert all(checks["required"].values())
    assert checks["observed"]["native_packing_present"] is False
    assert checks["observed"]["gist_cache_used"] is False


def test_run_task_still_rejects_unavailable_risk(tmp_path):
    with pytest.raises(RuntimeError, match="'detector_contract': False"):
        _run_finished_fixture(tmp_path, risk_available=False)
