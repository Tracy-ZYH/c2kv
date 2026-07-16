from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_agent_tool_definition_c2kv import (  # noqa: E402
    _build_tool_cache,
    _extract_tool_name,
    _generate_from_input_ids,
    _load_model,
    _normalize_text,
    _prefill_system,
    _prefill_tokens_with_cache,
    _setup_device,
)
from eval_toolathlon_first_tool_c2kv import _arg_f1s, _parse_pred_call  # noqa: E402
from train.train_data_multiturn import _chat_template_ids, _pad  # noqa: E402
from train_unified_next_action_c2kv import (  # noqa: E402
    UnifiedNextActionDataArgs,
    _load_unified_examples,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _jsonl_write(path: str, rows: List[Dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _target_payload(target: str) -> Optional[Dict[str, Any]]:
    parsed = _parse_pred_call(target)
    if not parsed:
        return None
    return {"name": parsed.get("name"), "arguments": parsed.get("arguments", {})}


def _tool_doc_ids(tokenizer: Any, text: str, title: str = "Tool definitions") -> List[int]:
    return _chat_template_ids(tokenizer, [{"role": "user", "content": f"{title}:\n" + text}])


def _context_input_ids_from_example(
    tokenizer: Any,
    example: Any,
    max_doc_length: int,
    max_doc_num: int,
    max_tool_definition_tokens: int,
    truncate_tool_definition: bool,
) -> tuple[Optional[torch.Tensor], int, int, Optional[str]]:
    doc_id_groups = []
    if getattr(example, "tool_documents", None):
        for document in example.tool_documents:
            if document.strip():
                doc_id_groups.append(_tool_doc_ids(tokenizer, document, "Tool definition"))
    else:
        doc_id_groups = [_tool_doc_ids(tokenizer, example.tool_definition)]
    doc_tokens = sum(len(item) for item in doc_id_groups)
    if doc_tokens > max_tool_definition_tokens:
        return None, doc_tokens, 0, f"tool_definition_tokens>{max_tool_definition_tokens}"

    rows = []
    for doc_ids in doc_id_groups:
        if len(doc_ids) <= max_doc_length:
            rows.append(_pad(doc_ids, max_doc_length, -100))
            continue
        if getattr(example, "tool_documents", None) and not truncate_tool_definition:
            return None, doc_tokens, 0, f"tool_document_tokens>{max_doc_length}"
        for start in range(0, len(doc_ids), max_doc_length):
            rows.append(_pad(doc_ids[start : start + max_doc_length], max_doc_length, -100))
    if len(rows) > max_doc_num:
        if not truncate_tool_definition:
            return None, doc_tokens, len(rows), f"tool_definition_docs>{max_doc_num}"
        rows = rows[:max_doc_num]
    if not rows:
        return None, doc_tokens, 0, "empty_tool_definition"
    return torch.tensor(rows, dtype=torch.long), doc_tokens, len(rows), None


@torch.inference_mode()
def _build_prefix(model: Any, tokenizer: Any, example: Any, args: argparse.Namespace) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    system_ids = _chat_template_ids(
        tokenizer,
        [{"role": "system", "content": example.system_prompt}],
        keep_bos=True,
        max_length=args.max_system_length,
    )
    system_input_ids = torch.tensor([system_ids], dtype=torch.long, device=model.device)
    system_cache, system_length, system_prefill_sec = _prefill_system(
        model, system_input_ids, args.system_attn_impl
    )

    if args.mode == "c2kv":
        context_input_ids, doc_tokens, doc_chunks, skip_reason = _context_input_ids_from_example(
            tokenizer,
            example,
            max_doc_length=args.max_doc_length,
            max_doc_num=args.max_doc_num,
            max_tool_definition_tokens=args.max_tool_definition_tokens,
            truncate_tool_definition=args.truncate_tool_definition,
        )
        if context_input_ids is None:
            return None, skip_reason
        tool_cache, tool_length, gist_tokens, actual_ratio, compress_sec, blend_sec = _build_tool_cache(
            model,
            context_input_ids,
            system_cache,
            system_length,
            args.gist_attn_impl,
            args.override_ratio,
        )
        return {
            "cache": tool_cache,
            "system_length": system_length,
            "tool_length": tool_length,
            "cache_length": tool_cache.get_seq_length(),
            "use_gist": True,
            "doc_tokens": doc_tokens,
            "doc_chunks": doc_chunks,
            "gist_tokens": gist_tokens,
            "actual_compression_ratio": actual_ratio,
            "system_prefill_sec": system_prefill_sec,
            "tool_compress_sec": compress_sec,
            "full_prefill_sec": 0.0,
            "blend_sec": blend_sec,
        }, None

    doc_ids = _tool_doc_ids(tokenizer, example.tool_definition)
    doc_tokens = len(doc_ids)
    if doc_tokens > args.max_tool_definition_tokens:
        return None, f"tool_definition_tokens>{args.max_tool_definition_tokens}"
    kept_tool_tokens = doc_tokens
    if args.mode == "truncate":
        kept_tool_tokens = max(1, (doc_tokens + args.override_ratio - 1) // args.override_ratio)
        doc_ids = doc_ids[:kept_tool_tokens]
    tool_input_ids = torch.tensor([doc_ids], dtype=torch.long, device=model.device)
    tool_cache, tool_length, full_prefill_sec = _prefill_tokens_with_cache(
        model,
        tool_input_ids,
        past_key_values=system_cache,
        past_length=system_length,
        attn_impl=args.generate_attn_impl,
    )
    return {
        "cache": tool_cache,
        "system_length": system_length,
        "tool_length": tool_length,
        "cache_length": tool_cache.get_seq_length(),
        "use_gist": False,
        "doc_tokens": doc_tokens,
        "doc_chunks": 1,
        "gist_tokens": 0,
        "actual_compression_ratio": doc_tokens / kept_tool_tokens if kept_tool_tokens else 0.0,
        "system_prefill_sec": system_prefill_sec,
        "tool_compress_sec": 0.0,
        "full_prefill_sec": full_prefill_sec,
        "blend_sec": 0.0,
    }, None


@torch.inference_mode()
def _generate_one(model: Any, tokenizer: Any, example: Any, args: argparse.Namespace) -> Dict[str, Any]:
    prefix, skip_reason = _build_prefix(model, tokenizer, example, args)
    if prefix is None:
        return {
            "qid": example.qid,
            "session_id": example.session_id,
            "mode": args.row_mode,
            "ratio": args.override_ratio,
            "skipped": True,
            "skip_reason": skip_reason,
        }

    prompt_ids = _chat_template_ids(tokenizer, example.input_messages, add_generation_prompt=True)
    if len(prompt_ids) > args.max_prompt_tokens:
        prompt_ids = prompt_ids[-args.max_prompt_tokens :]
    prompt_input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
    mock_cache_ids = prompt_input_ids.new_zeros((1, prefix["cache_length"]))
    input_ids = torch.cat([mock_cache_ids, prompt_input_ids], dim=1)
    position_ids = torch.arange(
        prefix["system_length"] + prefix["tool_length"],
        prefix["system_length"] + prefix["tool_length"] + prompt_input_ids.shape[1],
        dtype=torch.long,
        device=model.device,
    ).unsqueeze(0)
    prediction, generate_sec, generated_tokens, tbt_sec = _generate_from_input_ids(
        model,
        tokenizer,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        attn_impl=args.generate_attn_impl,
        use_gist=prefix["use_gist"],
        position_ids=position_ids,
        past_key_values=prefix["cache"],
    )
    target_payload = _target_payload(example.answer) or {"name": None, "arguments": {}}
    pred_payload = _parse_pred_call(prediction)
    arg_name_f1, arg_value_f1 = _arg_f1s(target_payload, pred_payload)
    ttft = (
        prefix["system_prefill_sec"]
        + prefix["tool_compress_sec"]
        + prefix["full_prefill_sec"]
        + prefix["blend_sec"]
    )
    return {
        "qid": example.qid,
        "session_id": example.session_id,
        "mode": args.row_mode,
        "ratio": args.override_ratio,
        "skipped": False,
        "doc_tokens": prefix["doc_tokens"],
        "doc_chunks": prefix["doc_chunks"],
        "gist_tokens": prefix["gist_tokens"],
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": generated_tokens,
        "actual_compression_ratio": round(prefix["actual_compression_ratio"], 4),
        "system_prefill_sec": round(prefix["system_prefill_sec"], 4),
        "tool_compress_sec": round(prefix["tool_compress_sec"], 4),
        "full_prefill_sec": round(prefix["full_prefill_sec"], 4),
        "blend_sec": round(prefix["blend_sec"], 4),
        "ttft_sec": round(ttft, 4),
        "generate_sec": round(generate_sec, 4),
        "tbt_sec": round(tbt_sec, 6),
        "total_sec": round(ttft + generate_sec, 4),
        "target_tool_name": target_payload.get("name"),
        "prediction_tool_name": pred_payload.get("name") if pred_payload else None,
        "tool_name_match": bool(pred_payload and pred_payload.get("name") == target_payload.get("name")),
        "has_tool_call": pred_payload is not None,
        "argument_name_f1": arg_name_f1,
        "argument_value_f1": arg_value_f1,
        "exact_match": _normalize_text(prediction) == _normalize_text(example.answer),
        "prediction": prediction,
        "target": example.answer,
    }


def _summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries = []
    for mode, ratio in sorted({(row.get("mode"), row.get("ratio")) for row in rows}):
        group = [row for row in rows if row.get("mode") == mode and row.get("ratio") == ratio]
        valid = [row for row in group if not row.get("skipped")]
        skips = Counter(row.get("skip_reason", "unknown") for row in group if row.get("skipped"))
        total_generated = sum(row.get("generated_tokens", 0) for row in valid)
        summaries.append({
            "mode": mode,
            "ratio": ratio,
            "num_examples": len(group),
            "num_valid": len(valid),
            "num_skipped": len(group) - len(valid),
            "skip_reasons": dict(skips),
            "tool_name_accuracy": sum(1 for row in valid if row["tool_name_match"]) / len(valid) if valid else 0.0,
            "tool_call_rate": sum(1 for row in valid if row["has_tool_call"]) / len(valid) if valid else 0.0,
            "argument_name_f1": sum(row["argument_name_f1"] for row in valid) / len(valid) if valid else 0.0,
            "argument_value_f1": sum(row["argument_value_f1"] for row in valid) / len(valid) if valid else 0.0,
            "exact_match": sum(1 for row in valid if row["exact_match"]) / len(valid) if valid else 0.0,
            "avg_doc_tokens": sum(row["doc_tokens"] for row in valid) / len(valid) if valid else 0.0,
            "avg_prompt_tokens": sum(row["prompt_tokens"] for row in valid) / len(valid) if valid else 0.0,
            "avg_generated_tokens": total_generated / len(valid) if valid else 0.0,
            "avg_actual_compression_ratio": (
                sum(row["actual_compression_ratio"] for row in valid) / len(valid) if valid else 0.0
            ),
            "avg_ttft_sec": sum(row["ttft_sec"] for row in valid) / len(valid) if valid else 0.0,
            "avg_generate_sec": sum(row["generate_sec"] for row in valid) / len(valid) if valid else 0.0,
            "avg_tbt_sec": sum(row["tbt_sec"] for row in valid) / len(valid) if valid else 0.0,
            "token_weighted_tbt_sec": (
                sum(row["generate_sec"] for row in valid) / total_generated if total_generated else 0.0
            ),
            "avg_total_sec": sum(row["total_sec"] for row in valid) / len(valid) if valid else 0.0,
        })
    return summaries


def _data_args(args: argparse.Namespace) -> UnifiedNextActionDataArgs:
    return UnifiedNextActionDataArgs(
        agent_dataset_path=args.agent_dataset_path,
        toolathlon_dataset_path=args.toolathlon_dataset_path,
        hermes_dataset_path=args.hermes_dataset_path,
        hermes_configs=args.hermes_configs,
        split_seed=args.split_seed,
        source_mix=args.source_mix,
        max_train_examples=args.max_train_examples,
        max_eval_examples=args.max_eval_examples,
        max_steps_per_trajectory=args.max_steps_per_trajectory,
        max_history_steps=args.max_history_steps,
        max_observation_chars=args.max_observation_chars,
        max_history_chars=args.max_history_chars,
        max_tools_per_sample=args.max_tools_per_sample,
        max_doc_length=args.max_doc_length,
        max_doc_num=args.max_doc_num,
        max_length=args.max_length,
        max_system_length=args.max_system_length,
        max_tool_definition_tokens=args.max_tool_definition_tokens,
        truncate_tool_definition=args.truncate_tool_definition,
        min_target_tokens=args.min_target_tokens,
    )


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

    examples = _load_unified_examples(_data_args(args), args.split)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    logger.info("Loaded %d unified %s examples", len(examples), args.split)

    rows: List[Dict[str, Any]] = []
    modes = [item.strip() for item in (args.compare_modes or args.mode).split(",") if item.strip()]
    ratios = [int(item.strip()) for item in (args.ratios or str(args.override_ratio)).split(",") if item.strip()]
    for mode in modes:
        run_ratios = [1] if mode == "full" else ratios
        model_args = copy.copy(args)
        model_args.untrained_c2kv = mode == "c2kv_untrained"
        model_args.mode = "c2kv" if mode == "c2kv_untrained" else mode
        model_args.row_mode = mode
        logger.info("Loading model for mode=%s", mode)
        model = _load_model(model_args, tokenizer, device)
        for ratio in run_ratios:
            run_args = copy.copy(model_args)
            run_args.override_ratio = ratio
            for example in tqdm(examples, desc=f"{mode}@{ratio}x"):
                rows.append(_generate_one(model, tokenizer, example, run_args))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "npu":
            torch.npu.empty_cache()

    summary = {
        "model": args.model,
        "base_model": args.base_model,
        "split": args.split,
        "num_examples": len(examples),
        "results": _summarize(rows),
    }
    if args.output_file:
        _jsonl_write(args.output_file, rows)
        Path(args.output_file).with_suffix(".summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate unified next-action C2KV training.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base_model")
    parser.add_argument("--tokenizer")
    parser.add_argument("--output_file", default="./outputs/unified_next_action_eval.jsonl")
    parser.add_argument("--split", choices=["train", "eval", "test"], default="eval")
    parser.add_argument("--mode", choices=["c2kv", "c2kv_untrained", "truncate", "full"], default="c2kv")
    parser.add_argument("--compare_modes")
    parser.add_argument("--ratios")
    parser.add_argument("--override_ratio", type=int, default=4)
    parser.add_argument("--max_examples", type=int, default=200)
    parser.add_argument("--agent_dataset_path", default="./datasets/agent-llm-traces")
    parser.add_argument("--toolathlon_dataset_path", default="./datasets/toolathlon")
    parser.add_argument("--hermes_dataset_path", default="./datasets/hermes-agent-reasoning-traces")
    parser.add_argument("--hermes_configs", default="kimi,glm-5.1")
    parser.add_argument("--source_mix", default="agent_llm_traces:0.3,toolathlon:0.4,hermes:0.3")
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--max_train_examples", type=int, default=20000)
    parser.add_argument("--max_eval_examples", type=int, default=2000)
    parser.add_argument("--max_steps_per_trajectory", type=int, default=6)
    parser.add_argument("--max_history_steps", type=int, default=6)
    parser.add_argument("--max_observation_chars", type=int, default=1200)
    parser.add_argument("--max_history_chars", type=int, default=12000)
    parser.add_argument("--max_tools_per_sample", type=int, default=32)
    parser.add_argument("--max_doc_length", type=int, default=1024)
    parser.add_argument("--max_doc_num", type=int, default=32)
    parser.add_argument("--max_tool_definition_tokens", type=int, default=32000)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_system_length", type=int, default=256)
    parser.add_argument("--max_prompt_tokens", type=int, default=1920)
    parser.add_argument("--min_target_tokens", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--truncate_tool_definition", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--device_type", choices=["auto", "cuda", "npu", "cpu"], default="auto")
    parser.add_argument("--system_attn_impl", default="eager")
    parser.add_argument("--gist_attn_impl", default="eager")
    parser.add_argument("--generate_attn_impl", default="eager")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--baseline_model_class", choices=["gist", "auto"], default="gist")
    parser.add_argument("--untrained_c2kv", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(evaluate(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
