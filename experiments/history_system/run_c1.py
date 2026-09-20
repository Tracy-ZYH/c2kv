"""Run the selected C1 recovery controller with an official benchmark harness.

The controller/runtime is shared by BFCL, tau2, and ToolSandbox. This file
only owns orchestration: one controller and one official harness worker per
explicit task identity.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import current
import evidence_sets
import runner
from c1_artifact_binding import bind_risk_artifact

HERE = Path(__file__).resolve().parent
RUNTIME = HERE / "runtime"
DEFAULT_RISK_ARTIFACT = HERE / "artifacts/c1_risk.t02_v1.json"
DEFAULT_RISK_ARTIFACT_SHA256 = "18a11f73aa1f7d4b0add86eed66ae9e5e129ea4bdfbe0dfad23faf4f7d2fb4ab"
BENCHMARKS = ("bfcl", "tau2", "toolsandbox")
METHODS = ("proposed", "c2kv_only")
SOURCE_PROFILE = {
    "bfcl": "native-v1",
    "tau2": "openai-single-task-v1",
    "toolsandbox": "openai-single-task-v1",
}
OFFICIAL_SCORER = {
    "bfcl": "official BFCL generation + checker",
    "tau2": "tau2 run -> tau2 evaluate-trajs -> reward_info.reward",
    "toolsandbox": "tool_sandbox official CLI -> result_summary.json",
}
SUMMARY_FIELDS = (
    "benchmark", "task_id", "status", "official_score", "normal_termination",
    "protocol_legal", "generation_calls", "detector_calls",
    "detector_trigger_count", "recovery_count", "successful_recovery_count",
    "evidence_units_appended", "raw_tokens_restored", "full_history_kv",
    "active_history_kv", "kv_retention", "compression_ratio",
    "generation_prefill_tokens", "recovery_prefill_tokens",
    "total_prefill_tokens", "wall_time", "native_generate_requests",
    "prefill_detector_scores", "risk_detector_scores", "risk_detector_unavailable",
    "gist_tokens", "raw_workspace_tokens",
    "gist_cache_hits", "native_packing_present",
)


def save(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def source_revision(path: Path) -> dict:
    """Record source identity without fetching or modifying its worktree."""
    root = path.resolve()
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, text=True, capture_output=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--short"],
        check=True, text=True, capture_output=True,
    ).stdout.splitlines()
    diff = subprocess.run(
        ["git", "-C", str(root), "diff", "HEAD", "--binary"],
        check=True, capture_output=True,
    ).stdout
    dirty_files = {}
    for row in status:
        relative = row[3:].split(" -> ")[-1]
        candidate = root / relative
        if candidate.is_file():
            dirty_files[relative] = hashlib.sha256(candidate.read_bytes()).hexdigest()
    return {
        "path": str(root), "commit": commit, "dirty": bool(status),
        "status_short": status, "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "tracked_diff_bytes": len(diff), "dirty_file_sha256": dirty_files,
    }


def _bare_endpoint(value: str) -> str:
    parsed = urlsplit(value.rstrip("/"))
    if (parsed.scheme != "http" or not parsed.netloc or parsed.username is not None
            or parsed.password is not None or parsed.query or parsed.fragment):
        raise ValueError(f"expected a bare HTTP endpoint, got {value!r}")
    if parsed.path.rstrip("/") not in {"", "/v1"}:
        raise ValueError(f"expected a bare HTTP endpoint, got {value!r}")
    return f"{parsed.scheme}://{parsed.netloc}"


def preflight_sglang_backend(
    args: argparse.Namespace, *, opener=None,
) -> dict[str, Any]:
    """Fail before creating a run directory when the native engine is absent.

    This is capability discovery only; it never submits generation.
    """
    base = _bare_endpoint(args.sglang_backend_url)
    endpoint = f"{base}/model_info"
    opener = opener or urlopen
    try:
        with opener(Request(endpoint, method="GET"), timeout=10) as response:
            info = json.load(response)
    except (HTTPError, URLError, OSError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"C1 SGLang backend is not ready at {endpoint}: "
            f"{type(error).__name__}: {error}. Start the PR #5 SGLang engine "
            "and verify /model_info before running run_c1.py."
        ) from error
    if not isinstance(info, Mapping):
        raise RuntimeError(f"C1 SGLang backend returned invalid /model_info at {endpoint}")
    capability = info.get("c2kv_native_packed")
    if not isinstance(capability, Mapping):
        raise RuntimeError("SGLang /model_info lacks c2kv_native_packed capability")
    binding = capability.get("model_binding")
    if not isinstance(binding, Mapping):
        raise RuntimeError("SGLang native-packed capability lacks model_binding")
    expected_checkpoint = args.checkpoint.resolve()
    observed_paths = [info.get("model_path"), binding.get("model_path")]
    if any(
        not isinstance(path, str) or Path(path).resolve() != expected_checkpoint
        for path in observed_paths
    ):
        raise RuntimeError(
            "SGLang checkpoint binding differs from --checkpoint: "
            f"expected={expected_checkpoint}, observed={observed_paths}"
        )
    checks = {
        "native_generate_enabled": capability.get("enabled") is True,
        "native_generate_endpoint": (
            capability.get("endpoint") == "/v1/c2kv/native_generate"
        ),
        "bfloat16": binding.get("dtype") == "bfloat16",
        "base_query_projection": (
            capability.get("base_query_enforced") is True
            and binding.get("query_projection") == "base"
        ),
        "prefill_feature": (
            isinstance(capability.get("shadow_feature_layer"), int)
            and capability.get("shadow_feature_readout") == "decoder_layer_output"
        ),
        "gist_extraction_packing": (
            binding.get("gist_type") == "dynamic-interleave"
            and binding.get("gist_param") == "qkv"
            and isinstance(capability.get("packing_version"), str)
            and bool(capability.get("packing_version"))
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"SGLang C1 capability check failed: {checks}")
    return {
        "endpoint": endpoint,
        "checkpoint": str(expected_checkpoint),
        "checks": checks,
        "packing_version": capability["packing_version"],
        "shadow_feature_layer": capability["shadow_feature_layer"],
    }


def build_profile(args: argparse.Namespace) -> tuple[dict, dict]:
    """Keep the compatibility detector distinct from the trained risk head."""
    if args.method == "c2kv_only":
        if args.selector_artifact is not None:
            raise ValueError("c2kv_only does not accept --selector-artifact")
        controller = evidence_sets._base_controller()
        controller.pop("post_draft_recovery", None)
        controller.pop("gp_experiments", None)
        selected = current.load_config()
        actual = hashlib.sha256((args.checkpoint / "config.json").read_bytes()).hexdigest()
        if actual != selected["checkpoint_selection"]["config_sha256"]:
            raise ValueError("C1 delivery requires the selected C1000 checkpoint config; this runtime is model-bound")
        return controller, {
            "schema": "c1-delivery-profile-v2",
            "method": "c2kv_only",
            "detector": "disabled",
            "algorithm": "C2KV event-native compression only",
            "new_c1_training_claimed": False,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_config_sha256": actual,
            "selection_protocol": None,
            "history_variant": "H0",
            "ratio": selected["ratio"],
            "controller_sha256": hashlib.sha256(
                json.dumps(controller, sort_keys=True).encode()
            ).hexdigest(),
            "automatic_reruns": 0,
        }
    if args.detector == "legacy_prefill" and args.selector_artifact is not None:
        raise ValueError("--selector-artifact is only used with --detector t02_risk")
    artifact_path = None
    artifact_sha256 = None
    if args.detector == "t02_risk":
        artifact_path = args.selector_artifact or DEFAULT_RISK_ARTIFACT
        if not artifact_path.is_file():
            raise FileNotFoundError(f"selector_artifact does not exist: {artifact_path}")
        artifact_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if args.selector_artifact is None and artifact_sha256 != DEFAULT_RISK_ARTIFACT_SHA256:
            raise ValueError("Bundled T02 C1 artifact differs from the evaluated release")
    config, _ = evidence_sets.build_config(
        history="H0",
        selector="legacy_prefill" if args.detector == "legacy_prefill" else "risk",
        selector_artifact=artifact_path,
        selector_threshold=args.selector_threshold,
        embedding_model=str(args.embedding_model.resolve()),
        embedding_device=args.embedding_device,
        semantic_query_overflow_policy="task_head_tail_preserve_draft_v1",
    )
    config["local_models"]["embedding"]["dtype"] = "bfloat16"
    artifact_binding = None
    if artifact_path is not None:
        config["selector_artifact"], artifact_binding = bind_risk_artifact(
            config["selector_artifact"], args.checkpoint
        )
    controller = current._configure_controller(evidence_sets._base_controller(), config)
    selected = current.load_config()
    actual = hashlib.sha256((args.checkpoint / "config.json").read_bytes()).hexdigest()
    if actual != selected["checkpoint_selection"]["config_sha256"]:
        raise ValueError("C1 delivery requires the selected C1000 checkpoint config; this detector is model-bound")
    return controller, {
        "schema": "c1-delivery-profile-v2",
        "method": "proposed",
        "detector": args.detector,
        "algorithm": "C1 legacy Prefill compatibility" if args.detector == "legacy_prefill" else "C1 T02 risk",
        "new_c1_training_claimed": args.detector == "t02_risk",
        "selector_artifact": str(artifact_path.resolve()) if artifact_path else None,
        "selector_artifact_sha256": artifact_sha256,
        "selector_artifact_binding": artifact_binding,
        "selector_threshold": args.selector_threshold if artifact_path else None,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_config_sha256": actual,
        "selection_protocol": "evidence_sets_v1",
        "history_variant": "H0",
        "ratio": selected["ratio"],
        "controller_sha256": hashlib.sha256(json.dumps(controller, sort_keys=True).encode()).hexdigest(),
        "automatic_reruns": 0,
    }


def _identities(args: argparse.Namespace) -> list[str]:
    if args.benchmark == "bfcl":
        values = args.task_id
        pattern = r"multi_turn_(?:base|long_context)_[0-9]+"
    elif args.benchmark == "tau2":
        values = args.tau2_task_id
        pattern = r"[A-Za-z0-9_.:-]+"
    else:
        values = args.ts_scenario
        pattern = r"[A-Za-z0-9_.:-]+"
    if not values:
        raise ValueError(f"benchmark={args.benchmark} requires at least one explicit task/scenario ID")
    if any(re.fullmatch(pattern, value) is None for value in values) or len(values) != len(set(values)):
        raise ValueError("task/scenario identities must be unique path-safe official IDs")
    return values


def _benchmark_dir(args: argparse.Namespace) -> Path:
    if args.benchmark == "bfcl":
        if args.benchmark_dir is None or not (args.benchmark_dir / "bfcl_eval").is_dir():
            raise ValueError("--benchmark-dir must contain the official bfcl_eval package")
        return args.benchmark_dir
    if args.benchmark == "tau2":
        if args.tau2_dir is None or not (args.tau2_dir / "src" / "tau2").is_dir():
            raise ValueError("--tau2-dir must contain src/tau2")
        return args.tau2_dir
    if args.toolsandbox_dir is None or not (args.toolsandbox_dir / "tool_sandbox").is_dir():
        raise ValueError("--toolsandbox-dir must contain tool_sandbox")
    return args.toolsandbox_dir


def _controller_endpoint(args: argparse.Namespace) -> str:
    return f"http://127.0.0.1:{args.port}"


def _model_name(args: argparse.Namespace) -> str:
    return "c2kv_only" if args.method == "c2kv_only" else f"c1_{args.detector}"


def portable_worker_command(args: argparse.Namespace, task: str, task_out: Path) -> list[str]:
    if args.benchmark == "bfcl":
        raise ValueError("BFCL uses the frozen native official worker")
    python = args.tau2_python if args.benchmark == "tau2" else args.toolsandbox_python
    run_identity = hashlib.sha256(
        str(task_out.resolve()).encode("utf-8")
    ).hexdigest()[:12]
    command = [
        python, "-m", "c2kv_eval.portable.c1_worker",
        "--benchmark", args.benchmark,
        "--task-id", task,
        "--agent-base-url", _controller_endpoint(args),
        "--user-base-url", args.user_base_url,
        "--benchmark-dir", str(_benchmark_dir(args).resolve()),
        "--bench-python", python,
        "--out", str(task_out / args.benchmark),
        "--run-name", f"c1_{args.benchmark}_{task}_{run_identity}",
        "--model", _model_name(args),
        "--task-timeout", str(args.task_timeout),
    ]
    if args.benchmark == "tau2":
        command.extend(["--task-set", args.task_set])
        if args.tau2_max_steps is not None:
            command.extend(["--tau2-max-steps", str(args.tau2_max_steps)])
    else:
        command.extend(["--ts-agent", args.ts_agent, "--ts-user", args.ts_user])
    return command


def commands_for_task(args: argparse.Namespace, task: str, controller_path: Path) -> tuple[list[str], list[str]]:
    design = current.load_config()
    design["candidate_id"] = _model_name(args)
    design["run_id_template"] = design["candidate_id"]
    design["runtime"].update(controller=str(controller_path), sglang_backend_url=args.sglang_backend_url)
    if args.method == "c2kv_only":
        design["runtime"].pop("shadow_feature_config", None)
    temporary_controller = None
    if not controller_path.is_file():
        # Preview validates the exact server argv without creating the requested
        # output directory. The runner reads the controller to validate backend
        # requirements, so give that read a short-lived identical copy.
        controller, _ = build_profile(args)
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        temporary_controller = Path(handle.name)
        try:
            json.dump(controller, handle)
        finally:
            handle.close()
        design["runtime"]["controller"] = str(temporary_controller)
    try:
        server = runner.server_command(
            design, task_id=task, checkpoint=str(args.checkpoint.resolve()),
            output=str(args.out.resolve()), port=args.port, python=sys.executable,
            benchmark=args.benchmark, source_profile=SOURCE_PROFILE[args.benchmark],
        )
    finally:
        if temporary_controller is not None:
            temporary_controller.unlink(missing_ok=True)
    server[server.index("--s0-config") + 1] = str(controller_path)
    task_out = args.out / "task_shards" / task
    if args.benchmark == "bfcl":
        worker = runner.worker_command(
            design, task_id=task, output=str(args.out.resolve()),
            benchmark_dir=str(_benchmark_dir(args).resolve()), port=args.port,
            python=args.bfcl_python, max_wall_seconds=args.task_timeout,
        )
    else:
        worker = portable_worker_command(args, task, task_out)
    return server, worker


def _official_summary_path(args: argparse.Namespace, task_out: Path) -> Path:
    return task_out / args.benchmark / "official_summary.json"


def _single_official_row(benchmark: str, summary: Mapping[str, Any]) -> dict:
    if benchmark == "bfcl":
        if summary.get("n_scored") != 1 or summary.get("n_generated") != 1:
            raise RuntimeError("Official BFCL did not generate and score exactly the requested task")
        score = summary.get("semantic_score")
        if score is None:
            score = summary.get("accuracy")
        return {"semantic_score": score, "normal_termination": True, "protocol_legal": None}
    if summary.get("n") != 1:
        raise RuntimeError(f"Official {benchmark} harness did not score exactly one task")
    rows = summary.get("task_rows") or []
    if len(rows) != 1:
        raise RuntimeError(f"Official {benchmark} summary requires exactly one task row")
    return dict(rows[0])


def _walk(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def summarize_task(benchmark: str, task: str, task_out: Path, official: Mapping[str, Any], wall_time: float) -> dict:
    """Join official scores with C1-native telemetry without redefining either."""
    official_row = _single_official_row(benchmark, official)
    records = []
    steps_path = task_out / "server" / "steps.jsonl"
    if steps_path.exists():
        for line in steps_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    traces = [trace for record in records for trace in record.get("generation_trace", [])]
    nodes = list(_walk(records))
    # One durable step record represents one committed decision.  Inspect its
    # final decision object exactly once: recursively walking recovery_rounds
    # would count the same recovery both there and in exact_recovery.
    decisions = [
        record.get("exact_recovery")
        for record in records
        if isinstance(record.get("exact_recovery"), Mapping)
    ]
    gates = [
        decision["gate"]
        for decision in decisions
        if isinstance(decision.get("gate"), Mapping)
    ]
    risk_selections = [
        decision["selection"] for decision in decisions
        if isinstance(decision.get("selection"), Mapping)
        and decision["selection"].get("selector") == "risk"
        and decision["selection"].get("score_semantics") == "current_turn_failure_risk"
    ]
    legacy_gates = [gate for gate in gates if gate.get("type") != "risk"]
    risk_scores = sum(
        selection.get("available") is True and _number(selection.get("score")) is not None
        for selection in risk_selections
    )
    risk_unavailable = sum(selection.get("available") is False for selection in risk_selections)
    recovery_rows = [
        decision for decision in decisions if decision.get("status") == "recover"
    ]
    generation_calls = sum(
        1 for trace in traces if trace.get("status") in {"started", "pending", "completed", "failed"}
    )
    native_stats = [
        trace.get("generation", {}).get("stats", {})
        for trace in traces
        if isinstance(trace.get("generation"), Mapping)
        and isinstance(trace.get("generation", {}).get("stats"), Mapping)
        and trace.get("generation", {}).get("stats", {}).get("backend")
            == "sglang_c2kv_native_packed"
    ]
    full_bytes = active_bytes = 0.0
    kv_bytes_per_token = None
    generation_prefill = recovery_prefill = 0.0
    for trace in traces:
        if trace.get("status") != "completed":
            continue
        usage = trace.get("usage") or trace.get("generation_usage") or {}
        prompt_tokens = _number(usage.get("prompt_tokens")) or 0.0
        if trace.get("phase") == "regeneration":
            recovery_prefill += prompt_tokens
        else:
            generation_prefill += prompt_tokens
        for node in _walk(trace):
            ratio = node.get("compression_ratio")
            if isinstance(ratio, Mapping):
                full = _number(ratio.get("full_history_bytes"))
                active = _number(ratio.get("active_history_bytes"))
                if full is not None and active is not None:
                    full_bytes += full
                    active_bytes += active
                    unit = _number(node.get("kv_bytes_per_token")) or _number(ratio.get("kv_bytes_per_token"))
                    if unit:
                        kv_bytes_per_token = unit
                    break
    if kv_bytes_per_token is None:
        for node in nodes:
            unit = _number(node.get("kv_bytes_per_token"))
            if unit:
                kv_bytes_per_token = unit
                break
    restored = sum(
        _number(unit.get("token_count")) or 0.0
        for row in recovery_rows for unit in row.get("appended_units") or []
    )
    successful = sum(
        isinstance(record.get("exact_recovery"), Mapping)
        and record["exact_recovery"].get("status") == "recover"
        and any(
            trace.get("phase") == "regeneration"
            and trace.get("status") == "completed"
            for trace in record.get("generation_trace", [])
        )
        for record in records
    )
    score = official_row.get("semantic_score")
    if score is None:
        score = official.get("semantic_score")
    return {
        "benchmark": benchmark, "task_id": task, "status": "completed",
        "official_score": score,
        "normal_termination": official_row.get("normal_termination"),
        "protocol_legal": official_row.get("protocol_legal"),
        "generation_calls": generation_calls,
        "detector_calls": len(legacy_gates) + len(risk_selections),
        "detector_trigger_count": sum(gate.get("triggered") is True for gate in gates),
        "recovery_count": len(recovery_rows),
        "successful_recovery_count": successful,
        "evidence_units_appended": sum(int(row.get("appended_unit_count") or 0) for row in recovery_rows),
        "raw_tokens_restored": int(restored),
        "full_history_kv": full_bytes / kv_bytes_per_token if kv_bytes_per_token else None,
        "active_history_kv": active_bytes / kv_bytes_per_token if kv_bytes_per_token else None,
        "kv_retention": active_bytes / full_bytes if full_bytes else None,
        "compression_ratio": full_bytes / active_bytes if active_bytes else None,
        "generation_prefill_tokens": generation_prefill,
        "recovery_prefill_tokens": recovery_prefill,
        "total_prefill_tokens": generation_prefill + recovery_prefill,
        "wall_time": wall_time,
        "native_generate_requests": len(native_stats),
        "prefill_detector_scores": sum(_number(gate.get("score")) is not None for gate in legacy_gates),
        "risk_detector_scores": risk_scores,
        "risk_detector_unavailable": risk_unavailable,
        "gist_tokens": sum(int(stats.get("gist_tokens") or 0) for stats in native_stats),
        "raw_workspace_tokens": sum(int(stats.get("workspace_tokens") or 0) for stats in native_stats),
        "gist_cache_hits": sum(
            int(stats.get("scope_reused_chunks") or 0)
            + int(stats.get("session_reused_gist_chunks") or 0)
            for stats in native_stats
        ),
        "native_packing_present": bool(native_stats) and any(
            int(stats.get("gist_tokens") or 0) > 0
            and int(stats.get("workspace_tokens") or 0) > 0
            for stats in native_stats
        ),
    }


def write_unified_summary(out: Path, rows: list[dict]) -> None:
    save(out / "unified_summary.json", {"schema": "c1-unified-summary-v1", "rows": rows})
    with (out / "unified_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in SUMMARY_FIELDS} for row in rows)


def functional_checks(method: str, detector: str, telemetry: Mapping[str, Any]) -> dict:
    """Separate required runtime behavior from descriptive efficiency telemetry."""

    return {
        "required": {
            "native_generate_requests": telemetry["native_generate_requests"] > 0,
            "detector_contract": (
                telemetry["detector_calls"] == 0
                if method == "c2kv_only"
                else telemetry["risk_detector_unavailable"] == 0
                if detector == "t02_risk"
                else telemetry["prefill_detector_scores"] > 0
            ),
        },
        "observed": {
            "native_packing_present": telemetry["native_packing_present"] is True,
            "gist_cache_used": telemetry["gist_cache_hits"] > 0,
            "compression_ratio_gt_one": (
                telemetry["compression_ratio"] is not None
                and telemetry["compression_ratio"] > 1.0
            ),
        },
    }


def run_task(args: argparse.Namespace, task: str, controller_path: Path) -> tuple[dict, dict]:
    server_command, worker_command = commands_for_task(args, task, controller_path)
    task_out = args.out / "task_shards" / task
    task_out.mkdir(parents=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(RUNTIME / "python"), str(RUNTIME)))
    worker_env = env.copy()
    if args.benchmark == "bfcl":
        worker_env["PYTHONPATH"] = str(RUNTIME)
    else:
        worker_env["PYTHONPATH"] = os.pathsep.join((str(args.portable_root.resolve()), str(RUNTIME)))
    process = worker = None
    deadline = time.monotonic() + args.task_timeout
    started = time.monotonic()
    summary_path = _official_summary_path(args, task_out)
    try:
        with (task_out / "controller.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                server_command, cwd=RUNTIME, env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=os.name == "posix",
            )
            ready_path = task_out / "server" / "ready.json"
            while not ready_path.exists():
                if process.poll() is not None:
                    raise RuntimeError(f"Controller exited {process.returncode}; see {task_out / 'controller.log'}")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Controller readiness timeout")
                time.sleep(1)
            with (task_out / "benchmark.log").open("w", encoding="utf-8") as bench_log:
                worker = subprocess.Popen(
                    worker_command, cwd=RUNTIME, env=worker_env, stdout=bench_log,
                    stderr=subprocess.STDOUT, start_new_session=os.name == "posix",
                )
                result = worker.wait(timeout=max(1, deadline - time.monotonic()))
            if result:
                raise RuntimeError(f"Official {args.benchmark} worker exited {result}; see {task_out / 'benchmark.log'}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            _single_official_row(args.benchmark, summary)
    finally:
        if worker is not None:
            runner._stop_bfcl(worker, task_out / args.benchmark / "running.json")
        if process is not None:
            runner._stop_server(process, task_out / "server.supervisor.json")
    final_path = task_out / "server" / "final.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    if (final.get("cost_summary_error") or final.get("status") == "failed"
            or final.get("stop_reason") == "runner_failed" or process.returncode != 0):
        raise RuntimeError(f"Controller finalization failed; see {final_path}")
    journal = final.get("journal_summary") or {}
    if journal.get("failed") or journal.get("pending") or not journal.get("completed"):
        raise RuntimeError(f"Model attempts failed, remain pending, or are missing; see {final_path}")
    telemetry = summarize_task(args.benchmark, task, task_out, summary, time.monotonic() - started)
    acceptance = functional_checks(args.method, args.detector, telemetry)
    required = acceptance["required"]
    if not all(required.values()):
        raise RuntimeError(f"C1 functional acceptance failed: {required}; see {task_out / 'server'}")
    return {
        "task_id": task, "status": "completed", "official_summary": summary,
        "unified_metrics": telemetry, "qualification": "official single-task harness result",
    }, telemetry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=BENCHMARKS, default="bfcl")
    parser.add_argument(
        "--method", choices=METHODS, default="proposed",
        help="proposed enables the selected detector/recovery; c2kv_only keeps identical native packing without detector/recovery",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sglang-backend-url", required=True)
    parser.add_argument("--embedding-model", type=Path, required=True)
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--detector", choices=("legacy_prefill", "t02_risk"), default="t02_risk")
    parser.add_argument("--selector-artifact", type=Path,
                        help="Override the bundled, evaluated T02 C1 risk artifact")
    parser.add_argument("--selector-threshold", type=float, default=0.5)
    parser.add_argument("--benchmark-dir", type=Path)
    parser.add_argument("--bfcl-python", default=sys.executable)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--tau2-dir", type=Path)
    parser.add_argument("--tau2-python", default=sys.executable)
    parser.add_argument("--tau2-task-id", action="append", default=[])
    parser.add_argument("--task-set", default="airline")
    parser.add_argument("--tau2-max-steps", type=int)
    parser.add_argument("--toolsandbox-dir", type=Path)
    parser.add_argument("--toolsandbox-python", default=sys.executable)
    parser.add_argument("--ts-scenario", action="append", default=[])
    parser.add_argument("--ts-agent", default="GPT_4_o_2024_05_13")
    parser.add_argument("--ts-user", default="GPT_4_o_2024_05_13")
    parser.add_argument("--user-base-url")
    parser.add_argument("--portable-root", type=Path, default=Path("/home/zhuyuhan/project/bfcl-c2kv"))
    parser.add_argument(
        "--sglang-root", type=Path,
        default=Path("/home/zhuyuhan/project/kvoffload-sglang-c2kv-pr5-runtime"),
        help="SGLang checkout served by --sglang-backend-url; recorded for provenance",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=38810)
    parser.add_argument("--task-timeout", type=int, default=3600)
    parser.add_argument("--preview", action="store_true", help="Print commands without model, harness, or network calls")
    return parser


def validate_args(args: argparse.Namespace) -> list[str]:
    identities = _identities(args)
    _benchmark_dir(args)
    if args.task_timeout <= 0 or not 1 <= args.port <= 65535:
        raise ValueError("task timeout and port must be valid positive values")
    if not (args.embedding_model / "config.json").is_file():
        raise ValueError("embedding model must be a local model directory")
    if not (args.checkpoint / "config.json").is_file():
        raise ValueError("checkpoint must be a local model directory")
    if not (args.sglang_root / ".git").exists():
        raise ValueError("--sglang-root must be the PR #5-compatible SGLang checkout")
    _bare_endpoint(args.sglang_backend_url)
    if args.benchmark != "bfcl":
        if not args.user_base_url:
            raise ValueError("tau2/ToolSandbox require --user-base-url for the raw Full user simulator")
        if _bare_endpoint(args.user_base_url) == _controller_endpoint(args):
            raise ValueError("agent C1 endpoint and raw Full user endpoint must differ")
        if not args.portable_root.is_dir():
            raise ValueError("--portable-root must be the bfcl-c2kv checkout")
    return identities


def create_output_directory(path: Path) -> None:
    """Create a new run root and never resume/overwrite implicitly."""
    path.mkdir(parents=True, exist_ok=False)


def preview(args: argparse.Namespace, identities: list[str], profile: dict) -> dict:
    controller_path = (args.out / "controller.json").resolve()
    cells = []
    for task in identities:
        server, worker = commands_for_task(args, task, controller_path)
        cells.append({
            "task_id": task,
            "C1_server_command": server,
            "benchmark_worker_command": worker,
            "agent_endpoint": _controller_endpoint(args),
            "user_endpoint": args.user_base_url if args.benchmark != "bfcl" else None,
        })
    return profile | {
        "benchmark": args.benchmark,
        "source_profile": SOURCE_PROFILE[args.benchmark],
        "task_or_scenario_ids": identities,
        "official_scorer": OFFICIAL_SCORER[args.benchmark],
        "cells": cells,
        "model_calls": 0,
        "benchmark_calls": 0,
        "network_calls": 0,
    }


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        identities = validate_args(args)
        controller, profile = build_profile(args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    profile.update(
        benchmark=args.benchmark, source_profile=SOURCE_PROFILE[args.benchmark],
        task_ids=identities, sglang_backend_url=args.sglang_backend_url,
        benchmark_dir=str(_benchmark_dir(args).resolve()),
        user_base_url=args.user_base_url if args.benchmark != "bfcl" else None,
        official_scorer=OFFICIAL_SCORER[args.benchmark],
        source_revisions={
            "c2kv": source_revision(HERE.parents[1]),
            "kvoffload-sglang-c2kv": source_revision(args.sglang_root),
            "bfcl-c2kv": source_revision(args.portable_root),
        },
    )
    if args.preview:
        print(json.dumps(preview(args, identities, profile), indent=2, ensure_ascii=False))
        return 0
    try:
        profile["sglang_backend_preflight"] = preflight_sglang_backend(args)
    except RuntimeError as error:
        parser.error(str(error))
    create_output_directory(args.out)
    controller_path = (args.out / "controller.json").resolve()
    save(controller_path, controller)
    save(args.out / "profile.json", profile)
    receipt = {"schema": "c1-multibench-delivery-run-v1", "status": "running", "tasks": []}
    metrics: list[dict] = []
    try:
        for task in identities:
            print(json.dumps({"benchmark": args.benchmark, "task": task, "status": "running"}), flush=True)
            task_receipt, task_metrics = run_task(args, task, controller_path)
            receipt["tasks"].append(task_receipt)
            metrics.append(task_metrics)
            save(args.out / "result.json", receipt)
            write_unified_summary(args.out, metrics)
        receipt["status"] = "completed"
    except Exception as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save(args.out / "result.json", receipt)
        write_unified_summary(args.out, metrics)
    print(json.dumps({"status": "completed", "result": str(args.out / "result.json")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
