"""Run C1 recovery through an external SGLang engine and official BFCL."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import current
import evidence_sets
import runner

HERE = Path(__file__).resolve().parent
RUNTIME = HERE / "runtime"


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def build_profile(args):
    """Keep the compatibility detector distinct from the newly trained risk head."""
    if args.detector == "t02_risk" and args.selector_artifact is None:
        raise ValueError("t02_risk requires --selector-artifact; no automatic detector substitution")
    if args.detector == "legacy_prefill" and args.selector_artifact is not None:
        raise ValueError("--selector-artifact is only used with --detector t02_risk")
    config, _ = evidence_sets.build_config(
        history="H0", selector="legacy_prefill" if args.detector == "legacy_prefill" else "risk",
        selector_artifact=args.selector_artifact, selector_threshold=args.selector_threshold,
        embedding_model=str(args.embedding_model.resolve()), embedding_device=args.embedding_device,
        semantic_query_overflow_policy="task_head_tail_preserve_draft_v1",
    )
    config["local_models"]["embedding"]["dtype"] = "bfloat16"
    controller = current._configure_controller(evidence_sets._base_controller(), config)
    selected = current.load_config()
    actual = hashlib.sha256((args.checkpoint / "config.json").read_bytes()).hexdigest()
    if actual != selected["checkpoint_selection"]["config_sha256"]:
        raise ValueError("C1 delivery requires the selected C1000 checkpoint config; this detector is model-bound")
    return controller, {
        "schema": "c1-delivery-profile-v1", "detector": args.detector,
        "algorithm": "C1 legacy Prefill compatibility" if args.detector == "legacy_prefill" else "C1 T02 risk",
        "new_c1_training_claimed": args.detector == "t02_risk",
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_config_sha256": actual,
        "selection_protocol": "evidence_sets_v1", "history_variant": "H0", "ratio": selected["ratio"],
        "controller_sha256": hashlib.sha256(json.dumps(controller, sort_keys=True).encode()).hexdigest(),
        "automatic_reruns": 0,
    }


def run_task(args, task, controller_path):
    design = current.load_config()
    design["candidate_id"] = "c1_" + args.detector
    design["run_id_template"] = design["candidate_id"]
    design["runtime"].update(controller=str(controller_path), sglang_backend_url=args.sglang_backend_url)
    server_command = runner.server_command(design, task_id=task, checkpoint=str(args.checkpoint.resolve()),
        output=str(args.out.resolve()), port=args.port, python=sys.executable)
    worker_command = runner.worker_command(design, task_id=task, output=str(args.out.resolve()),
        benchmark_dir=str(args.benchmark_dir.resolve()), port=args.port, python=args.bfcl_python,
        max_wall_seconds=args.task_timeout)
    task_out = args.out / "task_shards" / task
    task_out.mkdir(parents=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(RUNTIME / "python"), str(RUNTIME)))
    worker_env = env.copy()
    worker_env["PYTHONPATH"] = str(RUNTIME)
    process = worker = None
    deadline = time.monotonic() + args.task_timeout
    try:
        with (task_out / "controller.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(server_command, cwd=RUNTIME, env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=os.name == "posix")
            ready_path = task_out / "server" / "ready.json"
            while not ready_path.exists():
                if process.poll() is not None:
                    raise RuntimeError(f"Controller exited {process.returncode}; see {task_out / 'controller.log'}")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Controller readiness timeout")
                time.sleep(1)
            with (task_out / "benchmark.log").open("w", encoding="utf-8") as bench_log:
                worker = subprocess.Popen(worker_command, cwd=RUNTIME, env=worker_env, stdout=bench_log,
                    stderr=subprocess.STDOUT, start_new_session=os.name == "posix")
                result = worker.wait(timeout=max(1, deadline - time.monotonic()))
            if result:
                raise RuntimeError(f"Official BFCL worker exited {result}; see {task_out / 'bfcl'}")
            summary = json.loads((task_out / "bfcl" / "official_summary.json").read_text())
            if summary.get("n_scored") != 1 or summary.get("n_generated") != 1:
                raise RuntimeError("Official BFCL did not generate and score exactly the requested task")
    finally:
        if worker is not None:
            runner._stop_bfcl(worker, task_out / "bfcl" / "running.json")
        if process is not None:
            runner._stop_server(process, task_out / "server.supervisor.json")
    final_path = task_out / "server" / "final.json"
    final = json.loads(final_path.read_text())
    if (final.get("cost_summary_error") or final.get("status") == "failed"
            or final.get("stop_reason") == "runner_failed" or process.returncode != 0):
        raise RuntimeError(f"Controller finalization failed; see {final_path}")
    journal = final.get("journal_summary") or {}
    if journal.get("failed") or journal.get("pending") or not journal.get("completed"):
        raise RuntimeError(f"Model attempts failed, remain pending, or are missing; see {final_path}")
    return {"task_id": task, "status": "completed", "official_summary": summary,
        "qualification": "Functional integration smoke; preliminary, n=1"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sglang-backend-url", required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--bfcl-python", default=sys.executable)
    parser.add_argument("--embedding-model", type=Path, required=True)
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--detector", choices=("legacy_prefill", "t02_risk"), default="legacy_prefill")
    parser.add_argument("--selector-artifact", type=Path)
    parser.add_argument("--selector-threshold", type=float, default=0.5)
    parser.add_argument("--task-id", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=38810)
    parser.add_argument("--task-timeout", type=int, default=3600)
    parser.add_argument("--preview", action="store_true", help="Validate and print the profile without model calls")
    args = parser.parse_args(argv)
    if len(set(args.task_id)) != len(args.task_id) or any(
        not re.fullmatch(r"multi_turn_(?:base|long_context)_[0-9]+", task) for task in args.task_id):
        parser.error("task IDs must be unique supported BFCL base/long-context IDs")
    if args.task_timeout <= 0 or not 1 <= args.port <= 65535:
        parser.error("task timeout and port must be valid positive values")
    if not (args.benchmark_dir / "bfcl_eval").is_dir():
        parser.error("benchmark directory must contain the official bfcl_eval package")
    if not (args.embedding_model / "config.json").is_file():
        parser.error("embedding model must be a local model directory")
    controller, profile = build_profile(args)
    profile.update(task_ids=args.task_id, sglang_backend_url=args.sglang_backend_url,
        benchmark_dir=str(args.benchmark_dir.resolve()))
    if args.preview:
        print(json.dumps(profile | {"model_calls": 0}, indent=2))
        return 0
    args.out.mkdir(parents=True, exist_ok=False)
    controller_path = (args.out / "controller.json").resolve()
    save(controller_path, controller)
    save(args.out / "profile.json", profile)
    receipt = {"schema": "c1-bfcl-delivery-run-v1", "status": "running", "tasks": []}
    try:
        for task in args.task_id:
            print(json.dumps({"task": task, "status": "running"}), flush=True)
            receipt["tasks"].append(run_task(args, task, controller_path))
            save(args.out / "result.json", receipt)
        receipt["status"] = "completed"
    except Exception as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save(args.out / "result.json", receipt)
    print(json.dumps({"status": "completed", "result": str(args.out / 'result.json')}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
