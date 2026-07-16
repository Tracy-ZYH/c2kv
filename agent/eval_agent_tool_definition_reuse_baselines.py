from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "inference"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_agent_tool_definition_c2kv import (  # noqa: E402
    _build_tool_chunks,
    _extract_tool_name,
    _generate_one,
    _load_model,
    _normalize_text,
    _setup_device,
    _tool_doc_ids,
)
from eval_agent_tool_definition_hybrid_router import _generate_one_hybrid  # noqa: E402
from reuse_pipeline import LLMInference  # noqa: E402
from train_agent_tool_definition_c2kv import (  # noqa: E402
    AgentLLMTracesSource,
    AgentToolDefinitionDataArgs,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)


REUSE_MODES = {
    "reuse",
    "epic_leading32",
    "cacheblend_vdiff",
    "snapkv_reuse",
    "epic_leading32_snapkv",
    "cacheblend_vdiff_snapkv",
}
AGENT_MODES = {"full", "truncate", "c2kv", "c2kv_untrained", "hybrid"}


def _sync_device(device: Any) -> None:
    device_type = getattr(device, "type", str(device))
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device_type == "npu" and hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()


def _clear_device_cache(device: str) -> None:
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device == "npu" and hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.empty_cache()


def _is_oom_error(error: RuntimeError) -> bool:
    text = str(error).lower()
    return "out of memory" in text or "oom" in text


def _oom_row(example: Any, mode: str, ratio: int) -> Dict[str, Any]:
    return {
        "qid": getattr(example, "qid", None),
        "session_id": getattr(example, "session_id", None),
        "mode": mode,
        "ratio": ratio,
        "skipped": True,
        "skip_reason": "oom",
    }


def _jsonl_write(path: str, rows: List[Dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _render_query_prompt(tokenizer: Any, messages: List[Dict[str, Any]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _reuse_documents(tool_definition: str) -> List[str]:
    return ["Tool definitions:\n" + tool_definition]


def _reuse_mode_settings(mode: str, cacheblend_ratio: float) -> tuple[Optional[str], Optional[str]]:
    if mode == "reuse":
        return None, None
    if mode == "snapkv_reuse":
        return None, "snapkv"
    if mode == "epic_leading32":
        return "leading-32", None
    if mode == "epic_leading32_snapkv":
        return "leading-32", "snapkv"
    if mode == "cacheblend_vdiff":
        return f"vdiff-{cacheblend_ratio}", None
    if mode == "cacheblend_vdiff_snapkv":
        return f"vdiff-{cacheblend_ratio}", "snapkv"
    raise ValueError(f"Unknown reuse mode: {mode}")


@torch.inference_mode()
def _generate_one_reuse(
    evaluator: LLMInference,
    tokenizer: Any,
    example: Any,
    args: argparse.Namespace,
    mode: str,
) -> Dict[str, Any]:
    total_start = time.perf_counter()
    doc_tokens = len(_tool_doc_ids(tokenizer, example.tool_definition))
    if doc_tokens > args.max_tool_definition_tokens:
        return {
            "qid": example.qid,
            "session_id": example.session_id,
            "mode": mode,
            "ratio": args.override_ratio,
            "skipped": True,
            "skip_reason": f"tool_definition_tokens>{args.max_tool_definition_tokens}",
            "doc_tokens": doc_tokens,
        }

    recompute_type, compress_method = _reuse_mode_settings(mode, args.cacheblend_recompute_ratio)

    device = evaluator.device
    _sync_device(device)
    start = time.perf_counter()
    system_cache = evaluator.get_prefill_kv_cache(
        [example.system_prompt],
        keep_bos=True,
        role="system",
    )
    _sync_device(device)
    system_prefill_sec = time.perf_counter() - start

    _sync_device(device)
    start = time.perf_counter()
    context_cache = evaluator.get_prefill_kv_cache(
        _reuse_documents(example.tool_definition),
        keep_bos=False,
        role="user",
        compress_method=compress_method,
    )
    _sync_device(device)
    full_prefill_sec = time.perf_counter() - start
    compressed_doc_tokens = sum(len(ids) for ids in context_cache.input_ids)

    blend_sec = 0.0
    if recompute_type is not None:
        _sync_device(device)
        start = time.perf_counter()
        system_cache = evaluator.selective_recompute(
            system_cache,
            context_cache,
            recompute_type,
            discard_kv=True,
        )
        context_cache = None
        _sync_device(device)
        blend_sec = time.perf_counter() - start

    query_text = _render_query_prompt(tokenizer, example.input_messages)
    prompt_tokens = len(tokenizer.encode(query_text, add_special_tokens=False))
    if args.max_prompt_tokens and prompt_tokens > args.max_prompt_tokens:
        # Keep behavior explicit: raw rendered prompts are hard to truncate safely.
        return {
            "qid": example.qid,
            "session_id": example.session_id,
            "mode": mode,
            "ratio": args.override_ratio,
            "skipped": True,
            "skip_reason": f"prompt_tokens>{args.max_prompt_tokens}",
            "doc_tokens": doc_tokens,
            "prompt_tokens": prompt_tokens,
        }

    _sync_device(device)
    start = time.perf_counter()
    prediction = evaluator.decode_with_past_kv(
        system_prompt_kv=system_cache,
        precomputed_kv=context_cache,
        query_text=query_text,
        max_new_tokens=args.max_new_tokens,
        role=None,
    )
    _sync_device(device)
    generate_sec = time.perf_counter() - start

    target = example.answer.strip()
    target_tool = _extract_tool_name(target)
    pred_tool = _extract_tool_name(prediction)
    generated_tokens = len(tokenizer.encode(prediction, add_special_tokens=False))
    if context_cache is None and mode in {"epic_leading32", "cacheblend_vdiff"}:
        compressed_tokens = doc_tokens
    else:
        compressed_tokens = compressed_doc_tokens
    actual_ratio = doc_tokens / compressed_tokens if compressed_tokens else 1.0
    ttft_sec = system_prefill_sec + full_prefill_sec + blend_sec
    return {
        "qid": example.qid,
        "session_id": example.session_id,
        "mode": mode,
        "ratio": args.override_ratio,
        "skipped": False,
        "doc_tokens": doc_tokens,
        "actual_compression_ratio": round(actual_ratio, 4),
        "prompt_tokens": prompt_tokens,
        "target_tokens": len(tokenizer.encode(target, add_special_tokens=False)),
        "generated_tokens": generated_tokens,
        "latency_sec": round(generate_sec, 4),
        "system_prefill_sec": round(system_prefill_sec, 4),
        "tool_compress_sec": 0.0,
        "full_prefill_sec": round(full_prefill_sec, 4),
        "blend_sec": round(blend_sec, 4),
        "generate_sec": round(generate_sec, 4),
        "ttft_sec": round(ttft_sec, 4),
        "tbt_sec": round(generate_sec / generated_tokens, 6) if generated_tokens else 0.0,
        "total_sec": round(time.perf_counter() - total_start, 4),
        "target_tool_name": target_tool,
        "prediction_tool_name": pred_tool,
        "tool_name_match": target_tool is not None and target_tool == pred_tool,
        "has_tool_call": "<tool_call>" in prediction or "Action:" in prediction,
        "exact_match": _normalize_text(prediction) == _normalize_text(target),
        "prediction": prediction,
        "target": target,
    }


def _summarize_rows(args: argparse.Namespace, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries = []
    keys = sorted({(row.get("mode"), row.get("ratio")) for row in rows})
    for mode, ratio in keys:
        group = [row for row in rows if row.get("mode") == mode and row.get("ratio") == ratio]
        valid_rows = [row for row in group if not row.get("skipped")]
        skip_reasons = Counter(row.get("skip_reason", "unknown") for row in group if row.get("skipped"))
        generated_total = sum(row.get("generated_tokens", 0) for row in valid_rows)
        called = sum(1 for row in valid_rows if row.get("has_tool_call"))
        summaries.append({
            "model": args.model,
            "base_model": args.base_model,
            "dataset_path": args.dataset_path,
            "split": args.split,
            "mode": mode,
            "ratio": ratio,
            "num_examples": len(group),
            "num_valid": len(valid_rows),
            "num_skipped": len(group) - len(valid_rows),
            "skip_reasons": dict(skip_reasons),
            "exact_match": (
                sum(1 for row in valid_rows if row.get("exact_match")) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "tool_name_accuracy": (
                sum(1 for row in valid_rows if row.get("tool_name_match")) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "tool_call_rate": (
                called / len(valid_rows) if valid_rows else 0.0
            ),
            "call_accuracy": (
                sum(1 for row in valid_rows if row.get("tool_name_match")) / called
                if called else 0.0
            ),
            "avg_doc_tokens": (
                sum(row.get("doc_tokens", 0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_prompt_tokens": (
                sum(row.get("prompt_tokens", 0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_generated_tokens": (
                generated_total / len(valid_rows) if valid_rows else 0.0
            ),
            "avg_actual_compression_ratio": (
                sum(row.get("actual_compression_ratio", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_system_prefill_sec": (
                sum(row.get("system_prefill_sec", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_tool_compress_sec": (
                sum(row.get("tool_compress_sec", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_full_prefill_sec": (
                sum(row.get("full_prefill_sec", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_blend_sec": (
                sum(row.get("blend_sec", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_ttft_sec": (
                sum(row.get("ttft_sec", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_generate_sec": (
                sum(row.get("generate_sec", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_tbt_sec": (
                sum(row.get("tbt_sec", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "token_weighted_tbt_sec": (
                sum(row.get("generate_sec", 0.0) for row in valid_rows) / generated_total
                if generated_total else 0.0
            ),
            "avg_total_sec": (
                sum(row.get("total_sec", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
        })
    return summaries


def _select_examples(args: argparse.Namespace, tokenizer: Any) -> tuple[List[Any], Dict[str, int]]:
    data_args = AgentToolDefinitionDataArgs(
        dataset_path=args.dataset_path,
        eval_ratio=args.eval_ratio,
        split_seed=args.split_seed,
        split_manifest_file=args.split_manifest_file,
        split_manifest_name=args.split_manifest_name,
        max_samples_per_session=args.max_samples_per_session,
        max_doc_length=args.max_doc_length,
        max_doc_num=args.max_doc_num,
        max_tool_definition_tokens=args.max_tool_definition_tokens,
        max_length=args.max_length,
        max_system_length=args.max_system_length,
        truncate_tool_definition=args.truncate_tool_definition,
        require_tool_call=args.require_tool_call,
        min_target_tokens=args.min_target_tokens,
    )
    source = AgentLLMTracesSource(data_args)
    source_examples = list(source.iter_examples(args.split))
    if args.max_source_examples is not None:
        source_examples = source_examples[: args.max_source_examples]

    selection_skips: Counter[str] = Counter()
    examples = []
    for example in source_examples:
        if args.selection_filter == "c2kv":
            _, _, _, skip_reason = _build_tool_chunks(
                tokenizer,
                example.tool_definition,
                max_doc_length=args.max_doc_length,
                max_doc_num=args.max_doc_num,
                max_tool_definition_tokens=args.max_tool_definition_tokens,
                truncate_tool_definition=args.truncate_tool_definition,
            )
            if skip_reason is not None:
                selection_skips[skip_reason] += 1
                continue
        examples.append(example)
        if args.max_examples is not None and args.max_examples > 0 and len(examples) >= args.max_examples:
            break
    return examples, dict(selection_skips)


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    device = _setup_device(args.device_type)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model,
        trust_remote_code=True,
        local_files_only=True,
        padding_side="right",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    examples, selection_skips = _select_examples(args, tokenizer)
    logger.info("Selected %d examples; selection_skips=%s", len(examples), selection_skips)

    modes = [item.strip() for item in (args.compare_modes or args.mode).split(",") if item.strip()]
    ratios = [int(item.strip()) for item in (args.ratios or str(args.override_ratio)).split(",") if item.strip()]
    rows: List[Dict[str, Any]] = []

    for mode in modes:
        if mode not in AGENT_MODES and mode not in REUSE_MODES:
            raise ValueError(f"Unknown mode: {mode}")
        if mode in {"full", "reuse", "epic_leading32", "cacheblend_vdiff"}:
            run_ratios = [1]
        elif mode in {"snapkv_reuse", "epic_leading32_snapkv", "cacheblend_vdiff_snapkv"}:
            run_ratios = [4]
        else:
            run_ratios = ratios

        if mode in REUSE_MODES:
            model_path = args.reuse_model or args.base_model or args.model
            logger.info("Loading reuse evaluator for mode=%s model=%s", mode, model_path)
            evaluator = LLMInference(model_path, device=device, attn_impl=args.generate_attn_impl)
            for ratio in run_ratios:
                run_args = copy.copy(args)
                run_args.override_ratio = ratio
                desc = f"{mode}@{ratio}x"
                for example in tqdm(examples, desc=desc):
                    try:
                        row = _generate_one_reuse(evaluator, tokenizer, example, run_args, mode)
                    except RuntimeError as error:
                        if not _is_oom_error(error):
                            raise
                        logger.warning(
                            "Skipping sample after OOM: mode=%s ratio=%s qid=%s",
                            mode,
                            ratio,
                            getattr(example, "qid", None),
                        )
                        row = _oom_row(example, mode, ratio)
                    rows.append(row)
                    _clear_device_cache(device)
            del evaluator
            _clear_device_cache(device)
            continue

        model_args = copy.copy(args)
        model_args.untrained_c2kv = mode == "c2kv_untrained"
        model_args.mode = "c2kv" if mode == "c2kv_untrained" else mode
        model_args.row_mode = mode
        if model_args.max_prompt_tokens is not None and model_args.max_prompt_tokens <= 0:
            model_args.max_prompt_tokens = 0
        if model_args.max_baseline_input_tokens is not None and model_args.max_baseline_input_tokens <= 0:
            model_args.max_baseline_input_tokens = None
        if mode in {"full", "truncate"} and args.base_model:
            model_args.model = args.base_model
        logger.info("Loading agent model for mode=%s", mode)
        model = _load_model(model_args, tokenizer, device)
        for ratio in run_ratios:
            run_args = copy.copy(model_args)
            run_args.override_ratio = ratio
            desc = f"{mode}@{ratio}x" if mode != "full" else "full"
            for example in tqdm(examples, desc=desc):
                try:
                    if mode == "hybrid":
                        row = _generate_one_hybrid(
                            model,
                            tokenizer,
                            example,
                            run_args,
                            top_k=args.hybrid_top_k,
                            ratio=ratio,
                        )
                        row["mode"] = "hybrid"
                        row["ratio"] = ratio
                    else:
                        row = _generate_one(model, tokenizer, example, run_args, device)
                        row["mode"] = mode
                except RuntimeError as error:
                    if not _is_oom_error(error):
                        raise
                    logger.warning(
                        "Skipping sample after OOM: mode=%s ratio=%s qid=%s",
                        mode,
                        ratio,
                        getattr(example, "qid", None),
                    )
                    row = _oom_row(example, mode, ratio)
                    _clear_device_cache(device)
                rows.append(row)
        del model
        _clear_device_cache(device)

    summary = {
        "model": args.model,
        "base_model": args.base_model,
        "reuse_model": args.reuse_model or args.base_model or args.model,
        "dataset_path": args.dataset_path,
        "split": args.split,
        "modes": modes,
        "ratios": ratios,
        "selection_skips": selection_skips,
        "notes": {
            "epic_leading32": "PyTorch selective recompute with recompute_type=leading-32.",
            "cacheblend_vdiff": f"PyTorch value-difference selective recompute with recompute_type=vdiff-{args.cacheblend_recompute_ratio}; not the vLLM+LMCache expr_cacheblend.py path.",
            "snapkv_reuse": "Uses reuse_pipeline SnapKV compression, currently hard-coded to roughly 4x in compress_kv.",
            "epic_leading32_snapkv": "EPIC leading-32 selective recompute on top of SnapKV-compressed document KV.",
            "cacheblend_vdiff_snapkv": f"Value-difference selective recompute on top of SnapKV-compressed document KV with recompute_type=vdiff-{args.cacheblend_recompute_ratio}.",
        },
        "results": _summarize_rows(args, rows),
        "num_rows": len(rows),
    }
    if args.output_file:
        _jsonl_write(args.output_file, rows)
        summary_path = str(Path(args.output_file).with_suffix(".summary.json"))
        Path(summary_path).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Wrote predictions to %s", args.output_file)
        logger.info("Wrote summary to %s", summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate agent tool-definition baselines: Full, reuse, EPIC, CacheBlend-style, SnapKV, C2KV, Hybrid."
    )
    parser.add_argument("--model", required=True, help="C2KV checkpoint path.")
    parser.add_argument("--base_model", help="Base model path for non-C2KV baselines.")
    parser.add_argument("--reuse_model", help="Optional model path for reuse baselines. Defaults to base_model or model.")
    parser.add_argument("--tokenizer", help="Tokenizer path. Defaults to model.")
    parser.add_argument("--dataset_path", default="./datasets/agent-llm-traces")
    parser.add_argument("--output_file", default="./outputs/agent_tooldef_reuse_baselines_eval.jsonl")
    parser.add_argument("--split", choices=["train", "eval"], default="eval")
    parser.add_argument(
        "--mode",
        choices=[
            "full",
            "truncate",
            "reuse",
            "epic_leading32",
            "cacheblend_vdiff",
            "snapkv_reuse",
            "epic_leading32_snapkv",
            "cacheblend_vdiff_snapkv",
            "c2kv",
            "c2kv_untrained",
            "hybrid",
        ],
        default="c2kv",
    )
    parser.add_argument(
        "--compare_modes",
        default="full,snapkv_reuse,epic_leading32_snapkv,cacheblend_vdiff_snapkv,c2kv,hybrid",
        help="Comma-separated modes.",
    )
    parser.add_argument("--ratios", default="4", help="Ratios for truncate/c2kv/snapkv_reuse/hybrid.")
    parser.add_argument("--override_ratio", type=int, default=4)
    parser.add_argument("--hybrid_top_k", type=int, default=3)
    parser.add_argument("--router_scope", choices=["last_user", "all"], default="last_user")
    parser.add_argument("--router_strategy", choices=["lexical", "random"], default="lexical")
    parser.add_argument("--router_seed", type=int, default=42)
    parser.add_argument("--router_hit_filter", choices=["all", "hit", "miss"], default="all")
    parser.add_argument("--cacheblend_recompute_ratio", type=float, default=0.15)
    parser.add_argument("--max_examples", type=int, default=0, help="Maximum examples; <=0 means all selected examples.")
    parser.add_argument("--max_source_examples", type=int)
    parser.add_argument("--selection_filter", choices=["c2kv", "none"], default="c2kv")
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--split_manifest_file")
    parser.add_argument("--split_manifest_name", default="toolset_disjoint")
    parser.add_argument("--max_samples_per_session", type=int, default=4)
    parser.add_argument("--max_doc_length", type=int, default=1024)
    parser.add_argument("--max_doc_num", type=int, default=10)
    parser.add_argument("--max_tool_definition_tokens", type=int, default=10000)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_system_length", type=int, default=256)
    parser.add_argument("--max_prompt_tokens", type=int, default=4096)
    parser.add_argument("--max_baseline_input_tokens", type=int, default=12000)
    parser.add_argument("--min_target_tokens", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--truncate_tool_definition", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--require_tool_call", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--device_type", choices=["auto", "cuda", "npu", "cpu"], default="auto")
    parser.add_argument("--system_attn_impl", default="eager")
    parser.add_argument("--gist_attn_impl", default="eager")
    parser.add_argument("--generate_attn_impl", default="eager")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--baseline_model_class", choices=["gist", "auto"], default="auto")
    parser.add_argument("--untrained_c2kv", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(evaluate(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
