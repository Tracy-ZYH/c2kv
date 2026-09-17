"""Run one frozen hybrid-system candidate over an explicit task manifest."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.parse import urlsplit


HERE = Path(__file__).resolve().parent
ROOT = HERE / "runtime"
DEVELOPMENT_STATUS = "development_not_frozen"
FROZEN_STATUS = "frozen"
SUPPORTED_RATIOS = frozenset({4, 8})
CHECKPOINT_STATUSES = frozenset({"candidate_for_selection", "selected"})
CHECKPOINT_FIELDS = ("status", "path", "config_sha256", "selected_arm", "selected_step")


def task_ids(design: dict) -> tuple[str, ...]:
    tasks = design.get("task_ids")
    if (not isinstance(tasks, list) or not tasks
            or any(not isinstance(t, str) or re.fullmatch(
                r"multi_turn_(?:base|long_context)_[0-9]+", t) is None for t in tasks)
            or len(tasks) != len(set(tasks))):
        raise ValueError("Manifest requires unique explicit supported task IDs")
    return tuple(tasks)

CLEANUP_RESERVE_SECONDS = 60


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def value_after(command: list[str], flag: str) -> str:
    positions = [index for index, value in enumerate(command) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ValueError(f"command requires exactly one value for {flag}")
    return command[positions[0] + 1]


def _bfcl_python(design: dict, args: argparse.Namespace, server_python: str) -> str:
    requested = getattr(args, "bfcl_python", None)
    bound = design.get("runtime", {}).get("bfcl_python")
    if requested and bound and requested != bound:
        raise ValueError("BFCL Python differs from the frozen runtime binding")
    return requested or bound or server_python


def _generation_backend_arguments(design: dict) -> list[str]:
    runtime = design.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("Runtime contract is missing")
    backend = runtime.get("generation_backend", "native")
    if backend not in {"native", "sglang"}:
        raise ValueError("runtime.generation_backend must be native or sglang")
    controller_path = (ROOT / runtime["controller"]).resolve()
    controller = read(controller_path)
    requires_sglang = (
        "post_draft_recovery" in controller or "gp_experiments" in controller
    )
    if requires_sglang and backend != "sglang":
        raise ValueError("D3 post_draft_recovery and G--P require the SGLang backend")
    url = runtime.get("sglang_backend_url")
    if backend == "native":
        if url is not None:
            raise ValueError("runtime.sglang_backend_url requires the SGLang backend")
        return ["--generation-backend", "native"]
    if not isinstance(url, str) or not url.strip():
        raise ValueError("SGLang backend requires runtime.sglang_backend_url")
    url = url.strip().rstrip("/")
    parsed = urlsplit(url)
    if (parsed.scheme != "http" or not parsed.netloc
            or parsed.username is not None or parsed.password is not None
            or parsed.path or parsed.query or parsed.fragment):
        raise ValueError("runtime.sglang_backend_url must be a bare HTTP base URL")
    timeout = runtime.get("sglang_timeout_seconds")
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("runtime.sglang_timeout_seconds must be positive and finite")
    if runtime.get("npu_allocator_metrics") is not False:
        raise ValueError("SGLang runtime must explicitly disable process-local NPU allocator metrics")
    if runtime.get("device") != "cpu":
        raise ValueError("SGLang controller runtime must use device=cpu")
    return [
        "--generation-backend", "sglang",
        "--sglang-backend-url", url,
        "--sglang-timeout-seconds", str(timeout),
    ]


def checkpoint_binding(design: dict) -> dict:
    binding = design.get("checkpoint_selection")
    if not isinstance(binding, dict) or any(key not in binding for key in CHECKPOINT_FIELDS):
        raise ValueError("Checkpoint binding requires status, path, config_sha256, selected_arm and selected_step")
    if binding["status"] not in CHECKPOINT_STATUSES:
        raise ValueError("Checkpoint binding status must be candidate_for_selection or selected")
    if not isinstance(binding["path"], str) or not binding["path"]:
        raise ValueError("Checkpoint binding path must be explicit")
    digest = binding["config_sha256"]
    if (not isinstance(digest, str) or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)):
        raise ValueError("Checkpoint binding config_sha256 must be a lowercase SHA-256 digest")
    if binding["selected_arm"] not in {"B", "C"}:
        raise ValueError("Checkpoint binding selected_arm must be B or C")
    if type(binding["selected_step"]) is not int or binding["selected_step"] < 0:
        raise ValueError("Checkpoint binding selected_step must be a nonnegative integer")
    artifacts = binding.get("artifacts_sha256", {})
    if not isinstance(artifacts, dict):
        raise ValueError("Checkpoint binding artifacts_sha256 must be an object")
    for name, artifact_digest in artifacts.items():
        if not isinstance(name, str):
            raise ValueError("Checkpoint artifact paths must be relative and contained")
        relative = Path(name)
        if not name or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Checkpoint artifact paths must be relative and contained")
        if (not isinstance(artifact_digest, str) or len(artifact_digest) != 64
                or any(char not in "0123456789abcdef" for char in artifact_digest)):
            raise ValueError("Checkpoint artifact hashes must be lowercase SHA-256 digests")
    if artifacts.get("config.json", digest) != digest:
        raise ValueError("Checkpoint config hashes conflict")
    return binding


def validate(design: dict, *, allow_development: bool) -> None:
    if design.get("schema") != "a-history-system-candidate-design-v1":
        raise ValueError("Unexpected system-search design schema")
    allowed = {DEVELOPMENT_STATUS, FROZEN_STATUS} if allow_development else {FROZEN_STATUS}
    if design.get("status") not in allowed:
        raise ValueError("Execution requires a frozen design")
    tasks = task_ids(design)
    if design.get("evaluation_stage") not in {"development_search", "promotion", "release_validation"}:
        raise ValueError("Explicit search stage required")
    if design["evaluation_stage"] == "release_validation" and not design.get("acceptance_parameters_frozen"):
        raise ValueError("Final release claims require a prespecified comparison contract")
    limits = design["limits"]
    if limits != {"tasks": len(tasks), "generation_attempts_per_task": 96,
                  "extraction_calls_per_task": 1152, "stage_wall_seconds": 21600,
                  "server_wall_seconds_per_task": 10800}:
        raise ValueError("Round per-task/stage limits changed")
    if design["retry_contract"] != {"automatic_reruns": 0, "sdk_retries": 0,
        "transport_retries": 0, "cache_miss_retries": 0, "bfcl_workers_per_task": 1}:
        raise ValueError("Zero automatic retry/rerun contract changed")
    if (design["route"] != "ac_native_s0_lexical_raw_reserve_failed_operation"
        or design["compression_policy"] != "always-compress-v1"
        or design["ratio"] not in SUPPORTED_RATIOS or design["prefill_chunk_size"] != 256
        or design["sampling"] != {"mode":"greedy", "temperature":0, "seed":0, "max_completion_tokens":4096}):
        raise ValueError("This search round keeps the current C0 ratio4/8 inference contract")
    expected_cache_policy = (
        "external-sglang-content-addressed-chunks-v1"
        if design["runtime"].get("generation_backend") == "sglang"
        else "last-final-view-memo-only-v1"
    )
    if design.get("session_cache_policy") != expected_cache_policy:
        raise ValueError("Session cache policy differs from the generation backend")
    checkpoint_binding(design)
    for name, expected in design["source_files"].items():
        path = (ROOT / name).resolve()
        if ROOT.resolve() not in path.parents or sha256(path) != expected:
            raise ValueError(f"Frozen runtime file changed: {name}")
    for key in ("controller", "eval_policy", "eval_capacity"):
        if read(ROOT / design["runtime"][key]) != design["resolved_configs"][key]:
            raise ValueError(f"Resolved candidate configuration changed: {key}")
    _generation_backend_arguments(design)
    policy = design["resolved_configs"]["eval_policy"]["policy"]
    cap = min(policy["history_budget_bytes"], policy["workspace_budget_bytes"])
    if not 0 < cap <= design["search_contract"]["B0_history_bytes"]:
        raise ValueError("Unlabelled budget expansion is not allowed in this round")


def _inspect_checkpoint(checkpoint: Path) -> dict:
    source = ROOT / "benchmarks/memory_runtime/event_native.py"
    module_spec = importlib.util.spec_from_file_location(
        "_frozen_history_system_event_native", source)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Cannot load checkpoint inspector from {source}")
    module = importlib.util.module_from_spec(module_spec)
    original_path = list(sys.path)
    sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
    try:
        module_spec.loader.exec_module(module)
        return module.inspect_checkpoint(checkpoint)
    finally:
        sys.path[:] = original_path


def preflight_checkpoint(design: dict, requested_checkpoint: str | Path) -> dict:
    binding = checkpoint_binding(design)
    checkpoint = Path(requested_checkpoint).resolve()
    if checkpoint != Path(binding["path"]).resolve():
        raise ValueError("checkpoint differs from the frozen binding")
    if sha256(checkpoint / "config.json") != binding["config_sha256"]:
        raise ValueError("bound checkpoint config changed")
    profile = _inspect_checkpoint(checkpoint)
    if design["ratio"] not in profile["declared_supported_ratios"]:
        raise ValueError(f"bound checkpoint does not declare ratio{design['ratio']} support")
    if profile["training_arm"] != binding["selected_arm"]:
        raise ValueError("bound checkpoint training arm changed")
    step_match = re.fullmatch(r"checkpoint-([0-9]+)", checkpoint.name)
    if step_match is None or int(step_match.group(1)) != binding["selected_step"]:
        raise ValueError("bound checkpoint step changed")
    verified_artifacts = {}
    for name, expected in binding.get("artifacts_sha256", {}).items():
        artifact = (checkpoint / name).resolve()
        if checkpoint not in artifact.parents or sha256(artifact) != expected:
            raise ValueError(f"bound checkpoint artifact changed: {name}")
        verified_artifacts[name] = expected
    return {
        **profile,
        "binding_status": binding["status"],
        "config_sha256": binding["config_sha256"],
        "verified_artifacts_sha256": verified_artifacts,
    }


def server_command(
    design: dict,
    *,
    task_id: str,
    checkpoint: str,
    output: str,
    port: int,
    python: str,
) -> list[str]:
    command = [
        python,
        "-m",
        design["runtime"]["server_module"],
        "--checkpoint",
        checkpoint,
        "--out",
        str(Path(output) / "task_shards" / task_id / "server"),
        "--run-id",
        f"{design['run_id_template']}__{task_id}",
        "--model-name",
        design["candidate_id"],
        "--benchmark",
        "bfcl",
        "--source-profile",
        "native-v1",
        "--view-mode",
        design["route"],
        "--compression-policy",
        design["compression_policy"],
        "--history-view-protocol",
        design["history_view_protocol"],
        "--ratio",
        str(design["ratio"]),
        "--max-new-tokens",
        str(design["sampling"]["max_completion_tokens"]),
        "--decode-strategy",
        design["decode_strategy"],
        "--prefill-chunk-size",
        str(design["prefill_chunk_size"]),
        "--task-ids",
        task_id,
        "--max-decisions",
        str(design["limits"]["generation_attempts_per_task"]),
        "--max-generation-calls",
        str(design["limits"]["generation_attempts_per_task"]),
        "--max-extraction-calls",
        str(design["limits"]["extraction_calls_per_task"]),
        "--eval-policy",
        str((ROOT / design["runtime"]["eval_policy"]).resolve()),
        "--eval-capacity",
        str((ROOT / design["runtime"]["eval_capacity"]).resolve()),
        "--s0-config",
        str((ROOT / design["runtime"]["controller"]).resolve()),
        "--max-wall-seconds",
        str(design["limits"]["server_wall_seconds_per_task"]),
        "--device",
        design["runtime"]["device"],
        "--dtype",
        design["runtime"]["dtype"],
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--torch-threads",
        "4",
    ]
    if design["runtime"].get("npu_allocator_metrics"):
        command.append("--npu-allocator-metrics")
    if design["runtime"].get("shadow_feature_config"):
        command.extend(["--shadow-feature-config",
                        str((ROOT / design["runtime"]["shadow_feature_config"]).resolve())])
    command.extend(_generation_backend_arguments(design))
    command.append("--no-raw-snapshot")
    return command


def worker_command(
    design: dict,
    *,
    task_id: str,
    output: str,
    benchmark_dir: str,
    port: int,
    python: str,
    max_wall_seconds: float | None = None,
) -> list[str]:
    wall = (
        design["limits"]["server_wall_seconds_per_task"]
        if max_wall_seconds is None
        else max_wall_seconds
    )
    if not math.isfinite(float(wall)) or float(wall) <= 0:
        raise ValueError("official worker wall cap must be positive and finite")
    return [
        python,
        "-m",
        design["runtime"]["official_worker_module"],
        "--server-manifest",
        str(Path(output) / "task_shards" / task_id / "server" / "ready.json"),
        "--base-url",
        f"http://127.0.0.1:{port}/v1",
        "--benchmark-dir",
        benchmark_dir,
        "--out",
        str(Path(output) / "task_shards" / task_id / "bfcl"),
        "--max-wall-seconds",
        str(wall),
    ]


def preview(design: dict, args: argparse.Namespace) -> dict:
    validate(design, allow_development=True)
    checkpoint = args.checkpoint or "<selected-B-or-C-checkpoint>"
    output = args.output or "<unique-native-s0-long20-output>"
    benchmark_dir = args.benchmark_dir or design["task_and_scorer_lineage"]["bfcl_root_default"]
    python = args.python or sys.executable
    bfcl_python = _bfcl_python(design, args, python)
    cells = []
    for index, task_id in enumerate(task_ids(design)):
        port = args.port_base + index
        server = server_command(
            design,
            task_id=task_id,
            checkpoint=checkpoint,
            output=output,
            port=port,
            python=python,
        )
        worker = worker_command(
            design,
            task_id=task_id,
            output=output,
            benchmark_dir=benchmark_dir,
            port=port,
            python=bfcl_python,
            max_wall_seconds=(
                design["limits"]["stage_wall_seconds"] - CLEANUP_RESERVE_SECONDS
            ),
        )
        if value_after(server, "--task-ids") != task_id:
            raise ValueError("server is not isolated to one task")
        for flag, expected in (
            ("--max-decisions", "96"),
            ("--max-generation-calls", "96"),
            ("--max-extraction-calls", "1152"),
            ("--prefill-chunk-size", "256"),
        ):
            if value_after(server, flag) != expected:
                raise ValueError(f"per-task server changed {flag}")
        if server.count("--no-raw-snapshot") != 1:
            raise ValueError("memo-only server requires exactly one --no-raw-snapshot")
        cells.append({"task_id": task_id, "server": server, "official_worker": worker})
    return {
        "schema": "a-history-system-preview-v1",
        "status": "passed_cpu_static_preview_no_model_or_scorer",
        "design_status": design["status"],
        "checkpoint_selection_status": design["checkpoint_selection"]["status"],
        "task_ids": list(task_ids(design)),
        "cells": cells,
        "cap_proof": {
            "fresh_servers": len(task_ids(design)),
            "allowed_task_ids_per_server": 1,
            "generation_calls_per_server": 96,
            "extraction_calls_per_server": 1152,
            "shared_global_generation_pool": False,
        },
        "whole_task_denominator": len(task_ids(design)),
        "automatic_retries": 0,
        "automatic_reruns": 0,
        "session_cache_policy": design["session_cache_policy"],
        "raw_snapshot_retained": False,
        "interpreters": {
            "server": python,
            "official_worker": bfcl_python,
            "separate": python != bfcl_python,
        },
        "model_requests": 0,
        "scorer_calls": 0,
        "network_calls": 0,
        "launches": 0,
    }


def _verify_remote_scorer(design: dict, benchmark_dir: Path) -> dict:
    result = {}
    for relative, binding in design["task_and_scorer_lineage"]["source_bindings"].items():
        path = benchmark_dir / relative
        actual = sha256(path)
        if actual != binding["remote_sha256"]:
            raise ValueError(f"remote BFCL lineage changed: {relative}")
        result[relative] = actual
    return result


def _stop_server(server: subprocess.Popen, supervisor_path: Path) -> int | None:
    child_pid = None
    if supervisor_path.exists():
        child_pid = read(supervisor_path).get("child_pid")
    if server.poll() is not None:
        return server.returncode
    try:
        if os.name == "posix" and isinstance(child_pid, int) and child_pid > 1:
            os.kill(child_pid, signal.SIGTERM)
        else:
            server.terminate()
    except ProcessLookupError:
        pass
    try:
        return server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix" and isinstance(child_pid, int) and child_pid > 1:
                os.kill(child_pid, signal.SIGKILL)
            else:
                server.kill()
        except ProcessLookupError:
            pass
        return server.wait(timeout=10)


def _stop_bfcl(supervisor: subprocess.Popen, running_path: Path) -> int | None:
    child_pid = None
    if running_path.exists():
        child_pid = read(running_path).get("child_pid")
    if os.name == "posix" and isinstance(child_pid, int) and child_pid > 1:
        try:
            os.killpg(child_pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    if supervisor.poll() is None:
        try:
            return supervisor.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                if os.name == "posix":
                    if isinstance(child_pid, int) and child_pid > 1:
                        os.killpg(child_pid, signal.SIGKILL)
                    os.killpg(supervisor.pid, signal.SIGKILL)
                else:
                    supervisor.kill()
            except ProcessLookupError:
                pass
            return supervisor.wait(timeout=10)
    return supervisor.returncode


def terminalize(
    record: dict,
    *,
    status: str,
    started: float,
    error: str | None = None,
    wall_seconds_final: bool = False,
) -> None:
    by_task = {row["task_id"]: row for row in record["task_outcomes"]}
    for task_id in task_ids(record):
        by_task.setdefault(task_id, {
            "task_id": task_id,
            "outcome": "not_started",
            "in_fixed_denominator": True,
            "official_summary": None,
        })
    record["task_outcomes"] = [by_task[task_id] for task_id in task_ids(record)]
    record.update(
        status=status,
        state="completed" if status == "completed_fixed_manifest" else "failed",
        wall_seconds=time.monotonic() - started,
        wall_seconds_final=wall_seconds_final,
        denominator_observed=len(task_ids(record)),
        task_cells_started=sum(row["outcome"] != "not_started" for row in record["task_outcomes"]),
    )
    if error is not None:
        record["terminal_error"] = error


def _worker_failed_without_official_summary(
    worker_returncode: int | None,
    official_summary: Path,
) -> bool:
    """Stop dispatch when a worker failed before producing any score receipt."""
    return worker_returncode not in (None, 0) and not official_summary.exists()


def _server_runtime_failure(shard: Path, server_returncode: int | None) -> bool:
    """A BFCL zero score does not turn a crashed actor server into a valid rollout."""
    final_path = shard / "server" / "final.json"
    if final_path.exists():
        final = json.loads(final_path.read_text(encoding="utf-8"))
        if final.get("stop_reason") == "runner_failed":
            return True
    return server_returncode not in (None, 0)


def run(design: dict, args: argparse.Namespace) -> int:
    validate(design, allow_development=False)
    if "{" in design["run_id_template"] or "}" in design["run_id_template"]:
        raise ValueError("frozen design must bind a concrete run identity")
    bound_checkpoint = checkpoint_binding(design)
    if not design.get("launch_authorized"):
        raise ValueError("frozen launch authorization is required")
    checkpoint = Path(args.checkpoint).resolve()
    checkpoint_profile = preflight_checkpoint(design, checkpoint)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    scorer_hashes = _verify_remote_scorer(design, Path(args.benchmark_dir).resolve())
    output.mkdir(parents=True)
    started = time.monotonic()
    deadline = started + design["limits"]["stage_wall_seconds"]
    record = {
        "schema": "a-history-system-run-v1",
        "status": "running_fixed_manifest",
        "state": "running",
        "candidate_id": design["candidate_id"],
        "evaluation_stage": design["evaluation_stage"],
        "task_ids": list(task_ids(design)),
        "whole_task_denominator": len(task_ids(design)),
        "ratio": design["ratio"],
        "checkpoint_selection": bound_checkpoint,
        "checkpoint_profile": checkpoint_profile,
        "scorer_hashes": scorer_hashes,
        "automatic_retries": 0,
        "automatic_reruns": 0,
        "task_outcomes": [],
    }
    manifest = output / "stage_manifest.json"
    save(manifest, record)
    server_environment = os.environ.copy()
    server_environment["PYTHONPATH"] = os.pathsep.join(
        (str(ROOT / "python"), str(ROOT), server_environment.get("PYTHONPATH", ""))
    ).rstrip(os.pathsep)
    worker_environment = os.environ.copy()
    worker_environment["PYTHONPATH"] = os.pathsep.join(
        (str(ROOT / "python"), str(ROOT))
    )
    python = args.python or sys.executable
    bfcl_python = _bfcl_python(design, args, python)
    record["interpreters"] = {
        "server": python,
        "official_worker": bfcl_python,
        "separate": python != bfcl_python,
    }
    record["official_worker_pythonpath"] = worker_environment["PYTHONPATH"]
    save(manifest, record)
    active: subprocess.Popen | None = None
    active_worker: subprocess.Popen | None = None
    active_task_id: str | None = None
    active_task_launched = False
    active_supervisor: Path | None = None
    active_bfcl_running: Path | None = None
    try:
        for index, task_id in enumerate(task_ids(design)):
            active_task_id = task_id
            active_task_launched = False
            remaining = deadline - time.monotonic()
            if remaining <= CLEANUP_RESERVE_SECONDS:
                terminalize(record, status="stage_wall_exhausted", started=started)
                save(manifest, record)
                return 4
            port = args.port_base + index
            shard = output / "task_shards" / task_id
            shard.mkdir(parents=True)
            server_out = shard / "server"
            supervisor = server_out.parent / f"{server_out.name}.supervisor.json"
            active_supervisor = supervisor
            server_log = shard / "server.log"
            with server_log.open("x", encoding="utf-8", newline="\n") as log:
                active = subprocess.Popen(
                    server_command(design, task_id=task_id, checkpoint=str(checkpoint),
                                   output=str(output), port=port, python=python),
                    cwd=ROOT,
                    env=server_environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=os.name == "posix",
                )
                active_task_launched = True
            ready = server_out / "ready.json"
            while (
                not ready.exists()
                and active.poll() is None
                and time.monotonic() < deadline - CLEANUP_RESERVE_SECONDS
            ):
                time.sleep(0.25)
            if not ready.exists():
                code = _stop_server(active, supervisor)
                active = None
                record["task_outcomes"].append({
                    "task_id": task_id,
                    "outcome": "server_start_failure",
                    "in_fixed_denominator": True,
                    "server_returncode": code,
                })
                terminalize(record, status="stopped_on_server_start_failure", started=started)
                save(manifest, record)
                return 3
            worker_log = shard / "bfcl.log"
            worker = None
            worker_code = None
            active_bfcl_running = shard / "bfcl" / "running.json"
            try:
                with worker_log.open("x", encoding="utf-8", newline="\n") as log:
                    remaining = deadline - time.monotonic()
                    worker_wall = min(
                        design["limits"]["server_wall_seconds_per_task"],
                        remaining - CLEANUP_RESERVE_SECONDS,
                    )
                    if worker_wall <= 0:
                        raise TimeoutError("stage wall budget exhausted before BFCL worker")
                    active_worker = worker = subprocess.Popen(
                        worker_command(design, task_id=task_id, output=str(output),
                                       benchmark_dir=str(Path(args.benchmark_dir).resolve()),
                                       port=port, python=bfcl_python,
                                       max_wall_seconds=worker_wall),
                        cwd=ROOT,
                        env=worker_environment,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=os.name == "posix",
                    )
                    try:
                        worker_code = worker.wait(timeout=max(
                            0.001,
                            deadline - time.monotonic() - CLEANUP_RESERVE_SECONDS / 2,
                        ))
                    except subprocess.TimeoutExpired:
                        worker_code = _stop_bfcl(worker, active_bfcl_running)
                    active_worker = None
            finally:
                if active_worker is not None:
                    worker_code = _stop_bfcl(active_worker, active_bfcl_running)
                    active_worker = None
                server_code = _stop_server(active, supervisor)
                active = None
            official_summary = shard / "bfcl" / "official_summary.json"
            runtime_failed = _server_runtime_failure(shard, server_code)
            outcome = {
                "task_id": task_id,
                "outcome": "runtime_failure_in_denominator" if runtime_failed
                           else "official_completed" if worker_code == 0 and official_summary.exists()
                           else "failed_in_denominator",
                "in_fixed_denominator": True,
                "worker_returncode": worker_code,
                "server_returncode": server_code,
                "official_summary": str(official_summary) if official_summary.exists() else None,
                "runtime_completed": not runtime_failed,
            }
            record["task_outcomes"].append(outcome)
            record["completed_task_cells"] = len(record["task_outcomes"])
            active_task_id = None
            active_task_launched = False
            active_supervisor = None
            active_bfcl_running = None
            save(manifest, record)
            if runtime_failed:
                outcome["dispatch_stop_reason"] = "actor_runtime_failure_even_if_officially_scored"
                terminalize(record, status="stopped_on_actor_runtime_failure", started=started)
                save(manifest, record)
                return 6
            if _worker_failed_without_official_summary(worker_code, official_summary):
                outcome["dispatch_stop_reason"] = (
                    "worker_failure_without_official_summary"
                )
                terminalize(
                    record,
                    status="stopped_on_worker_failure_without_official_summary",
                    started=started,
                )
                save(manifest, record)
                return 5
        terminalize(record, status="completed_fixed_manifest", started=started)
        save(manifest, record)
        return 0
    except BaseException as error:
        error_text = f"{type(error).__name__}: {error}"
        if (
            active_task_id is not None
            and active_task_launched
            and not any(row["task_id"] == active_task_id for row in record["task_outcomes"])
        ):
            official_summary = (
                output / "task_shards" / active_task_id / "bfcl" / "official_summary.json"
            )
            record["task_outcomes"].append({
                "task_id": active_task_id,
                "outcome": "interrupted_in_denominator",
                "in_fixed_denominator": True,
                "worker_returncode": (
                    active_worker.poll() if active_worker is not None else None
                ),
                "server_returncode": active.poll() if active is not None else None,
                "official_summary": (
                    str(official_summary) if official_summary.exists() else None
                ),
                "interruption_error": error_text,
            })
        terminalize(
            record,
            status="stopped_without_rerun",
            started=started,
            error=error_text,
        )
        save(manifest, record)
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        return 2
    finally:
        cleanup_errors = []
        worker_cleanup_code = None
        server_cleanup_code = None
        if active_worker is not None and active_bfcl_running is not None:
            try:
                worker_cleanup_code = _stop_bfcl(active_worker, active_bfcl_running)
            except Exception as cleanup_error:
                cleanup_errors.append(
                    f"bfcl cleanup: {type(cleanup_error).__name__}: {cleanup_error}"
                )
        if active is not None:
            if active_supervisor is not None:
                try:
                    server_cleanup_code = _stop_server(active, active_supervisor)
                except Exception as cleanup_error:
                    cleanup_errors.append(
                        f"server cleanup: {type(cleanup_error).__name__}: {cleanup_error}"
                    )
        if active_task_id is not None and active_task_launched:
            active_rows = [
                row for row in record["task_outcomes"]
                if row["task_id"] == active_task_id
            ]
            if active_rows:
                if worker_cleanup_code is not None:
                    active_rows[0]["worker_returncode"] = worker_cleanup_code
                if server_cleanup_code is not None:
                    active_rows[0]["server_returncode"] = server_cleanup_code
        if record.get("status") == "running_fixed_manifest":
            terminalize(
                record,
                status="stopped_without_rerun",
                started=started,
                error=f"unexpected cleanup state for {active_task_id}",
            )
        if cleanup_errors:
            record["cleanup_errors"] = cleanup_errors
        terminalize(
            record,
            status=record["status"],
            started=started,
            wall_seconds_final=True,
        )
        save(manifest, record)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preview", "run"))
    parser.add_argument("--design", type=Path, default=HERE / "development.json")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--benchmark-dir")
    parser.add_argument("--python")
    parser.add_argument("--bfcl-python")
    parser.add_argument("--port-base", type=int, default=36000)
    parser.add_argument("--preview-output", type=Path)
    parser.add_argument("--runtime-root", type=Path, default=HERE / "runtime")
    args = parser.parse_args(argv)
    global ROOT
    ROOT = args.runtime_root.resolve()
    design = read(args.design)
    if not 1024 <= args.port_base <= 65536 - len(task_ids(design)):
        parser.error("port-base must leave room for every manifest task")
    if args.action == "preview":
        result = preview(design, args)
        if args.preview_output is not None:
            save(args.preview_output, result)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if not all((args.checkpoint, args.output, args.benchmark_dir)):
        parser.error("run requires checkpoint, output and benchmark-dir")
    return run(design, args)


if __name__ == "__main__":
    raise SystemExit(main())
