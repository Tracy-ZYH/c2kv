"""Preview or serve the selected D3 algorithm from the active source tree."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

HERE = Path(__file__).resolve().parent
RUNTIME = HERE / "runtime"


def load_config():
    return json.loads((HERE / "configs/current_algorithm.json").read_text(encoding="utf-8"))


def _configure_controller(base, switches):
    """Load the active recovery package even if another benchmarks tree is imported."""
    import importlib.util

    package_name = "_c2kv_active_recovery"
    package_path = RUNTIME / "benchmarks/memory_runtime/recovery"
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_path)]
        package.__package__ = package_name
        sys.modules[package_name] = package
    module_name = package_name + ".experiment_config"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(
            module_name, package_path / "experiment_config.py"
        )
        if spec is None or spec.loader is None:
            raise ImportError("Cannot load active G--P experiment_config")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module.configure_controller(base, switches)


def server_command(args):
    # Share the benchmark runner's argument construction and budget contract.
    import importlib.util

    spec = importlib.util.spec_from_file_location("history_runner", HERE / "runner.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    config = load_config()
    runner.ROOT = RUNTIME
    gp_path = getattr(args, "gp_config", None)
    controller_path = getattr(args, "controller_config", None)
    experiment_switches = None
    if controller_path is not None:
        config["runtime"]["controller"] = str(controller_path.resolve())
        experiment_switches = json.loads(controller_path.read_text(encoding="utf-8")).get("gp_experiments")
    if gp_path is not None:
        base_path = RUNTIME / config["runtime"]["controller"]
        controller = _configure_controller(
            json.loads(base_path.read_text(encoding="utf-8")),
            json.loads(gp_path.read_text(encoding="utf-8")))
        experiment_switches = controller["gp_experiments"]
        # Materialize the exact config used by both preview and serve.
        resolved = args.out.resolve() / "gp.controller.json"
        resolved.parent.mkdir(parents=True, exist_ok=True)
        contents = json.dumps(controller, ensure_ascii=False, indent=2) + "\n"
        if resolved.exists() and resolved.read_text(encoding="utf-8") != contents:
            raise FileExistsError(f"Different G--P config already exists: {resolved}")
        resolved.write_text(contents, encoding="utf-8")
        config["runtime"]["controller"] = str(resolved)
    if experiment_switches is not None:
        identity = hashlib.sha256(json.dumps(experiment_switches, sort_keys=True).encode()).hexdigest()[:12]
        config["candidate_id"] = "gp_" + identity
        config["run_id_template"] = "a_history_gp_" + identity
    backend_url = getattr(args, "sglang_backend_url", None)
    backend_timeout = getattr(args, "sglang_timeout_seconds", None)
    if backend_url is not None:
        config["runtime"]["sglang_backend_url"] = backend_url
    if backend_timeout is not None:
        config["runtime"]["sglang_timeout_seconds"] = backend_timeout
    return runner.server_command(
        config, task_id=args.task_id, checkpoint=str(args.checkpoint.resolve()),
        output=str(args.out.resolve()), port=args.port, python=sys.executable,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preview", "serve"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task-id", required=True, help="One BFCL task/session ID.")
    parser.add_argument("--port", type=int, default=28800)
    parser.add_argument(
        "--sglang-backend-url",
        help="Bare URL of the mandatory external SGLang C2KV engine.",
    )
    parser.add_argument(
        "--sglang-timeout-seconds", type=float,
        help="Override the configured external generation request timeout.",
    )
    parser.add_argument("--gp-config", type=Path, help="JSON G--P switch overlay.")
    parser.add_argument("--controller-config", type=Path, help="Explicit full controller config.")
    args = parser.parse_args(argv)
    command = server_command(args)
    if args.action == "preview":
        model_name = command[command.index("--model-name") + 1]
        name = "G–P / " + model_name if model_name.startswith("gp_") else load_config()["name"]
        print(json.dumps({"algorithm": name, "command": command,
                          "model_calls": 0}, indent=2))
        return 0
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(RUNTIME / "python"), str(RUNTIME)))
    return subprocess.call(command, cwd=RUNTIME, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
