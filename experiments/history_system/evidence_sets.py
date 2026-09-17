"""Generate explicit evidence_sets_v1 G--P configs without launching work."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from current import RUNTIME, _configure_controller, load_config


HERE = Path(__file__).resolve().parent

HISTORY_VARIANTS = {"H0": "current", "H1": "record_bound"}
CONTROLLER_VARIANTS = {
    "C0": "candidate_rule",
    "C1": "risk",
    "C2": "reranker",
    "C3": "local_llm",
    "C4": "gain_turn",
    "C5": "parameter_source",
}
SELECTORS = (
    "candidate_rule",
    "legacy_prefill",
    "risk",
    "reranker",
    "local_llm",
    "gain_turn",
    "gain_task",
    "parameter_source",
)
TRAINED_SELECTORS = frozenset({"risk", "gain_turn", "gain_task"})
ARTIFACT_SELECTORS = TRAINED_SELECTORS | {"reranker"}

H1_SOURCE_HASHES = {
    "python/history_memory/encoding_scope.py": (
        "81036aba03bbdc054e7c9189922ff3cd4c42fda6cff16469a302f81a6b3a63ce"
    ),
    "python/history_memory/packing.py": (
        "064e0757d1adfea5ccbb819e41873cd8e41ca5f4d3ab9cdcbb037f473b1faa2f"
    ),
}

MODEL_DEFAULTS = {
    "embedding": {
        "model_name_or_path": "Qwen/Qwen3-Embedding-0.6B",
        "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    },
    "reranker": {
        "model_name_or_path": "Qwen/Qwen3-Reranker-0.6B",
        "revision": "e61197ed45024b0ed8a2d74b80b4d909f1255473",
    },
    "selector": {
        "model_name_or_path": "Qwen/Qwen3-4B-Instruct-2507",
        "revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
    },
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def verify_h1_sources(runtime: Path = RUNTIME) -> dict[str, str]:
    """Verify that H1 still names the frozen G04 implementation."""

    observed: dict[str, str] = {}
    for relative, expected in H1_SOURCE_HASHES.items():
        path = runtime / relative
        if not path.is_file():
            raise FileNotFoundError(f"H1 frozen source is missing: {path}")
        actual = _sha256_bytes(path.read_bytes())
        observed[relative] = actual
        if actual != expected:
            raise RuntimeError(
                f"H1 frozen G04 source changed: {relative}; "
                f"expected sha256={expected}, actual sha256={actual}"
            )
    return observed


def _model_role_config(
    role: str,
    *,
    model_name_or_path: str | None,
    revision: str | None,
    device: str,
    batch_size: int | None = None,
) -> dict[str, Any]:
    defaults = MODEL_DEFAULTS[role]
    model = model_name_or_path or defaults["model_name_or_path"]
    resolved_revision = (
        revision
        if revision is not None
        else defaults["revision"] if model_name_or_path is None else None
    )
    result = {
        "model_name_or_path": model,
        "revision": resolved_revision,
        "device": device,
        "local_files_only": True,
    }
    if batch_size is not None:
        result["batch_size"] = batch_size
    return result


def _base_controller() -> dict[str, Any]:
    current = load_config()
    reference = Path(current["runtime"]["controller"])
    path = reference if reference.is_absolute() else RUNTIME / reference
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Active controller must be a JSON object: {path}")
    return value


def _validate_with_active_runtime(config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize and build through the same code path used by current.py."""

    controller = _configure_controller(_base_controller(), config)
    normalized = controller["gp_experiments"]
    if normalized.get("set_selector") == "legacy_prefill":
        recovery = controller.get("post_draft_recovery")
        if not isinstance(recovery, Mapping) or recovery.get("gate") != "prefill_linear_head":
            raise ValueError(
                "legacy_prefill requires the frozen prefill_linear_head in the base controller"
            )
    module = importlib.import_module("_c2kv_active_recovery.local_selection_models")
    # Constructor validation is lazy: it validates config but imports no model runtime.
    models = module.LocalSelectionModels(normalized["local_models"])
    models.public_config()
    artifact = normalized.get("selector_artifact")
    if artifact:
        runtime_python = str(RUNTIME / "python")
        if runtime_python not in sys.path:
            sys.path.insert(0, runtime_python)
        set_models = importlib.import_module("_c2kv_active_recovery.set_models")
        trained = set_models.load_set_selector(artifact)
        expected = (
            "reranker_calibrator"
            if normalized["set_selector"] == "reranker"
            else normalized["set_selector"]
        )
        if trained.kind != expected:
            raise ValueError("selector artifact target differs from set_selector")
        if expected != "risk":
            set_models.validate_selector_score_models(
                trained,
                models,
                semantic_query_overflow_policy=normalized.get(
                    "semantic_query_overflow_policy", "error"
                ),
            )
    return normalized


def _read_selector_artifact(path: Path | str) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"selector_artifact does not exist: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"selector_artifact is not valid JSON: {source}") from error
    if not isinstance(value, dict):
        raise TypeError("selector_artifact root must be a JSON object")
    return value


def _artifact_receipt(path: Path | str) -> dict[str, str]:
    source = Path(path).resolve()
    return {"source_path": str(source), "source_sha256": _sha256_bytes(source.read_bytes())}


def build_config(
    *,
    history: str,
    selector: str,
    candidate_pool_size: int = 8,
    selected_evidence_max: int = 4,
    fallback_256: bool = False,
    reserve_tokens: int = 0,
    selector_artifact: Path | str | None = None,
    selector_threshold: float = 0.5,
    gain_delta: float = 0.0,
    export_selection_state: bool = False,
    semantic_query_overflow_policy: str = "error",
    embedding_model: str | None = None,
    embedding_revision: str | None = None,
    embedding_device: str = "cpu",
    reranker_model: str | None = None,
    reranker_revision: str | None = None,
    reranker_device: str = "cpu",
    reranker_batch_size: int = 1,
    selector_model: str | None = None,
    selector_revision: str | None = None,
    selector_device: str = "cpu",
    selector_max_input_tokens: int = 32752,
) -> tuple[dict[str, Any], dict[str, str]]:
    if history not in HISTORY_VARIANTS:
        raise ValueError(f"history must be one of {tuple(HISTORY_VARIANTS)}")
    if selector not in SELECTORS:
        raise ValueError(f"selector must be one of {SELECTORS}")
    if type(candidate_pool_size) is not int or not 1 <= candidate_pool_size <= 8:
        raise ValueError("candidate_pool_size must be between 1 and 8")
    if selected_evidence_max not in (1, 2, 4) or isinstance(selected_evidence_max, bool):
        raise ValueError("selected_evidence_max must be one of 1, 2, or 4")
    if reserve_tokens not in (0, 512) or isinstance(reserve_tokens, bool):
        raise ValueError("reserve_tokens must be exactly 0 or 512")
    if (
        type(selector_max_input_tokens) is not int
        or selector_max_input_tokens < 1
        or selector_max_input_tokens > 262128
    ):
        raise ValueError(
            "selector_max_input_tokens must be between 1 and 262128"
        )
    if type(reranker_batch_size) is not int or reranker_batch_size < 1:
        raise ValueError("reranker_batch_size must be a positive integer")

    source_hashes = verify_h1_sources() if history == "H1" else {}
    artifact: dict[str, Any] | None = None
    if selector_artifact is not None:
        if selector not in ARTIFACT_SELECTORS:
            raise ValueError(f"selector_artifact is not used by {selector}")
        artifact = _read_selector_artifact(selector_artifact)
    elif selector in TRAINED_SELECTORS:
        raise ValueError(f"{selector} requires --selector-artifact")

    config: dict[str, Any] = {
        "schema": "a-history-gp-v1",
        "G": HISTORY_VARIANTS[history],
        "U": "tokens_1024",
        "B": "source",
        "Q": "archive_rrf",
        "K": selected_evidence_max,
        "L": "next_decision",
        "R": 1,
        "D": "candidate_rule",
        "P": "quoted",
        "order": "chronological",
        "candidate_limit": candidate_pool_size,
        "selection_protocol": "evidence_sets_v1",
        "set_selector": selector,
        "retrieval_limit": 24,
        "retrieval_route_limit": 16,
        "semantic_query_overflow_policy": semantic_query_overflow_policy,
        "rrf_k": 60,
        "fallback_unit": "tokens_256" if fallback_256 else None,
        "recovery_reserve_tokens": reserve_tokens,
        "selector_threshold": selector_threshold,
        "gain_delta": gain_delta,
        "export_selection_state": export_selection_state,
        "local_models": {
            "embedding": _model_role_config(
                "embedding",
                model_name_or_path=embedding_model,
                revision=embedding_revision,
                device=embedding_device,
            ),
            "reranker": _model_role_config(
                "reranker",
                model_name_or_path=reranker_model,
                revision=reranker_revision,
                device=reranker_device,
                batch_size=reranker_batch_size,
            ),
            "selector": _model_role_config(
                "selector",
                model_name_or_path=selector_model,
                revision=selector_revision,
                device=selector_device,
            )
            | (
                {"max_input_tokens": selector_max_input_tokens}
                if selector_max_input_tokens != 32752
                else {}
            ),
        },
    }
    if artifact is not None:
        config["selector_artifact"] = artifact
    return _validate_with_active_runtime(config), source_hashes


def current_command(
    *,
    config_path: Path,
    checkpoint: Path,
    run_out: Path,
    task_id: str,
    action: str = "preview",
    sglang_backend_url: str | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(HERE / "current.py"),
        action,
        "--checkpoint",
        str(checkpoint.resolve()),
        "--out",
        str(run_out.resolve()),
        "--task-id",
        task_id,
        "--gp-config",
        str(config_path.resolve()),
    ]
    if sglang_backend_url is not None:
        command.extend(("--sglang-backend-url", sglang_backend_url))
    return command


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"Different evidence-set config already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", choices=tuple(HISTORY_VARIANTS), required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--controller", choices=tuple(CONTROLLER_VARIANTS))
    choice.add_argument("--selector", choices=SELECTORS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidate-pool-size", type=int, default=8)
    parser.add_argument("--selected-evidence-max", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--fallback-256", action="store_true")
    parser.add_argument("--reserve-tokens", type=int, choices=(0, 512), default=0)
    parser.add_argument("--selector-artifact", type=Path)
    parser.add_argument("--selector-threshold", type=float, default=0.5)
    parser.add_argument("--gain-delta", type=float, default=0.0)
    parser.add_argument("--export-selection-state", action="store_true")
    parser.add_argument(
        "--semantic-query-overflow-policy",
        choices=("error", "task_head_tail_preserve_draft_v1"),
        default="error",
    )
    parser.add_argument("--selector-max-input-tokens", type=int, default=32752)
    parser.add_argument("--reranker-batch-size", type=int, default=1)
    for role in ("embedding", "reranker", "selector"):
        parser.add_argument(f"--{role}-model")
        parser.add_argument(f"--{role}-revision")
        parser.add_argument(f"--{role}-device", default="cpu")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--run-out", type=Path)
    parser.add_argument("--task-id")
    parser.add_argument("--run-action", choices=("preview", "serve"), default="preview")
    parser.add_argument("--sglang-backend-url")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    selector = CONTROLLER_VARIANTS[args.controller] if args.controller else args.selector
    launch_values = (args.checkpoint, args.run_out, args.task_id)
    if any(value is not None for value in launch_values) and not all(
        value is not None for value in launch_values
    ):
        parser.error("--checkpoint, --run-out, and --task-id must be supplied together")

    config, source_hashes = build_config(
        history=args.history,
        selector=selector,
        candidate_pool_size=args.candidate_pool_size,
        selected_evidence_max=args.selected_evidence_max,
        fallback_256=args.fallback_256,
        reserve_tokens=args.reserve_tokens,
        selector_artifact=args.selector_artifact,
        selector_threshold=args.selector_threshold,
        gain_delta=args.gain_delta,
        export_selection_state=args.export_selection_state,
        semantic_query_overflow_policy=args.semantic_query_overflow_policy,
        embedding_model=args.embedding_model,
        embedding_revision=args.embedding_revision,
        embedding_device=args.embedding_device,
        reranker_model=args.reranker_model,
        reranker_revision=args.reranker_revision,
        reranker_device=args.reranker_device,
        reranker_batch_size=args.reranker_batch_size,
        selector_model=args.selector_model,
        selector_revision=args.selector_revision,
        selector_device=args.selector_device,
        selector_max_input_tokens=args.selector_max_input_tokens,
    )
    _write_once(args.out, config)
    receipt: dict[str, Any] = {
        "out": str(args.out.resolve()),
        "history": args.history,
        "controller": args.controller,
        "set_selector": selector,
        "gp_sha256": _digest(config),
        "h1_source_hashes": source_hashes,
        "model_calls": 0,
    }
    if args.selector_artifact is not None:
        receipt["selector_artifact_source"] = _artifact_receipt(args.selector_artifact)
    if selector == "legacy_prefill":
        recovery = _base_controller()["post_draft_recovery"]
        head = recovery["prefill_head"]
        receipt["legacy_prefill_source"] = {
            "compatibility_mode": True,
            "training_performed": False,
            "gate": recovery["gate"],
            "head_artifact_sha256": head["artifact_sha256"],
            "layer": head["layer"],
            "threshold": head["threshold"],
        }
    if all(value is not None for value in launch_values):
        receipt["current_command"] = current_command(
            config_path=args.out,
            checkpoint=args.checkpoint,
            run_out=args.run_out,
            task_id=args.task_id,
            action=args.run_action,
            sglang_backend_url=args.sglang_backend_url,
        )
        receipt["freeze_controller"] = str(
            (args.run_out.resolve() / "gp.controller.json")
        )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
