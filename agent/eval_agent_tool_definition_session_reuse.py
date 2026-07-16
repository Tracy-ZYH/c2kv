from __future__ import annotations

import argparse
import copy
import json
import logging
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_agent_tool_definition_c2kv import (  # noqa: E402
    _build_tool_cache,
    _build_tool_chunks,
    _extract_tool_name,
    _generate_from_input_ids,
    _generate_one,
    _load_model,
    _normalize_text,
    _prefill_system,
    _prefill_tokens_with_cache,
    _setup_device,
    _tool_doc_ids,
)
from train.train_data_multiturn import _chat_template_ids  # noqa: E402
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


def _jsonl_write(path: str, rows: List[Dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _span_index(qid: str) -> int:
    match = re.search(r":(\d+)$", qid)
    return int(match.group(1)) if match else 0


def _clone_cache(cache: Any, config: Any) -> DynamicCache:
    return DynamicCache(
        [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers],
        config=config,
    )


def _group_examples_by_session(examples: List[Any]) -> OrderedDict[str, List[Any]]:
    sessions: OrderedDict[str, List[Any]] = OrderedDict()
    for example in examples:
        sessions.setdefault(example.session_id, []).append(example)
    for session_id in list(sessions.keys()):
        sessions[session_id] = sorted(sessions[session_id], key=lambda item: _span_index(item.qid))
    return sessions


def _select_sessions(
    tokenizer: Any,
    sessions: OrderedDict[str, List[Any]],
    args: argparse.Namespace,
) -> OrderedDict[str, List[Any]]:
    selected: OrderedDict[str, List[Any]] = OrderedDict()
    skipped = 0
    for session_id, examples in sessions.items():
        if len(examples) < args.min_spans_per_session:
            continue
        if args.selection_filter == "c2kv":
            _, _, _, skip_reason = _build_tool_chunks(
                tokenizer,
                examples[0].tool_definition,
                max_doc_length=args.max_doc_length,
                max_doc_num=args.max_doc_num,
                max_tool_definition_tokens=args.max_tool_definition_tokens,
                truncate_tool_definition=args.truncate_tool_definition,
            )
            if skip_reason is not None:
                skipped += 1
                continue
        if args.max_spans_per_session and len(examples) > args.max_spans_per_session:
            examples = examples[: args.max_spans_per_session]
        selected[session_id] = examples
        if args.max_sessions is not None and len(selected) >= args.max_sessions:
            break
    logger.info("Selected %d sessions; skipped %d sessions by selection filter", len(selected), skipped)
    return selected


@torch.inference_mode()
def _build_reuse_prefix(
    model: Any,
    tokenizer: Any,
    example: Any,
    args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
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
        context_input_ids, doc_tokens, doc_chunks, skip_reason = _build_tool_chunks(
            tokenizer,
            example.tool_definition,
            max_doc_length=args.max_doc_length,
            max_doc_num=args.max_doc_num,
            max_tool_definition_tokens=args.max_tool_definition_tokens,
            truncate_tool_definition=args.truncate_tool_definition,
        )
        if context_input_ids is None:
            return {"skipped": True, "skip_reason": skip_reason, "doc_tokens": doc_tokens}
        (
            prefix_cache,
            tool_length,
            gist_tokens,
            actual_ratio,
            tool_compress_sec,
            blend_sec,
        ) = _build_tool_cache(
            model,
            context_input_ids,
            system_cache,
            system_length,
            args.gist_attn_impl,
            args.override_ratio,
        )
        full_prefill_sec = 0.0
    else:
        doc_ids = _tool_doc_ids(tokenizer, example.tool_definition)
        doc_tokens = len(doc_ids)
        if args.mode == "truncate":
            kept_tool_tokens = max(1, (doc_tokens + args.override_ratio - 1) // args.override_ratio)
            used_doc_ids = doc_ids[:kept_tool_tokens]
        else:
            kept_tool_tokens = doc_tokens
            used_doc_ids = doc_ids
        tool_input_ids = torch.tensor([used_doc_ids], dtype=torch.long, device=model.device)
        prefix_cache, tool_length, full_prefill_sec = _prefill_tokens_with_cache(
            model,
            tool_input_ids,
            past_key_values=system_cache,
            past_length=system_length,
            attn_impl=args.generate_attn_impl,
        )
        gist_tokens = 0
        actual_ratio = float(doc_tokens / kept_tool_tokens) if kept_tool_tokens else 0.0
        doc_chunks = 1
        tool_compress_sec = 0.0
        blend_sec = 0.0

    build_sec = system_prefill_sec + tool_compress_sec + full_prefill_sec + blend_sec
    return {
        "skipped": False,
        "cache": prefix_cache,
        "system_length": system_length,
        "tool_length": tool_length,
        "prefix_original_length": system_length + tool_length,
        "cache_length": prefix_cache.get_seq_length(),
        "doc_tokens": doc_tokens,
        "doc_chunks": doc_chunks,
        "gist_tokens": gist_tokens,
        "actual_compression_ratio": actual_ratio,
        "system_prefill_sec": system_prefill_sec,
        "tool_compress_sec": tool_compress_sec,
        "full_prefill_sec": full_prefill_sec,
        "blend_sec": blend_sec,
        "session_build_sec": build_sec,
    }


@torch.inference_mode()
def _generate_span_with_reuse(
    model: Any,
    tokenizer: Any,
    example: Any,
    prefix: Dict[str, Any],
    args: argparse.Namespace,
    turn_index: int,
) -> Dict[str, Any]:
    prompt_ids = _chat_template_ids(
        tokenizer,
        example.input_messages,
        add_generation_prompt=True,
    )
    if len(prompt_ids) > args.max_prompt_tokens:
        prompt_ids = prompt_ids[-args.max_prompt_tokens :]

    input_total_len = prefix["prefix_original_length"] + len(prompt_ids)
    if args.max_baseline_input_tokens is not None and args.mode in ("full", "truncate"):
        if input_total_len > args.max_baseline_input_tokens:
            return {
                "qid": example.qid,
                "session_id": example.session_id,
                "turn_index": turn_index,
                "mode": args.row_mode,
                "ratio": args.override_ratio,
                "reuse": True,
                "skipped": True,
                "skip_reason": f"baseline_input_tokens>{args.max_baseline_input_tokens}",
                "input_tokens": input_total_len,
            }

    prompt_input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
    mock_cache_ids = prompt_input_ids.new_zeros((1, prefix["cache_length"]))
    input_ids = torch.cat([mock_cache_ids, prompt_input_ids], dim=1)
    position_ids = torch.arange(
        prefix["prefix_original_length"],
        prefix["prefix_original_length"] + prompt_input_ids.shape[1],
        dtype=torch.long,
        device=model.device,
    ).unsqueeze(0)
    cache = _clone_cache(prefix["cache"], model.config)
    prediction, generate_sec, generated_tokens, tbt_sec = _generate_from_input_ids(
        model,
        tokenizer,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        attn_impl=args.generate_attn_impl,
        use_gist=args.mode == "c2kv",
        position_ids=position_ids,
        past_key_values=cache,
    )
    target = example.answer.strip()
    target_tool = _extract_tool_name(target)
    pred_tool = _extract_tool_name(prediction)
    ttft_sec = prefix["session_build_sec"] if turn_index == 1 else 0.0
    return {
        "qid": example.qid,
        "session_id": example.session_id,
        "turn_index": turn_index,
        "mode": args.row_mode,
        "ratio": args.override_ratio,
        "reuse": True,
        "skipped": False,
        "doc_tokens": prefix["doc_tokens"],
        "doc_chunks": prefix["doc_chunks"],
        "gist_tokens": prefix["gist_tokens"],
        "actual_compression_ratio": round(prefix["actual_compression_ratio"], 4),
        "prompt_tokens": len(prompt_ids),
        "input_tokens": input_total_len,
        "target_tokens": len(tokenizer.encode(target, add_special_tokens=False)),
        "generated_tokens": generated_tokens,
        "latency_sec": round(generate_sec, 4),
        "system_prefill_sec": round(prefix["system_prefill_sec"], 4) if turn_index == 1 else 0.0,
        "tool_compress_sec": round(prefix["tool_compress_sec"], 4) if turn_index == 1 else 0.0,
        "full_prefill_sec": round(prefix["full_prefill_sec"], 4) if turn_index == 1 else 0.0,
        "blend_sec": round(prefix["blend_sec"], 4) if turn_index == 1 else 0.0,
        "generate_sec": round(generate_sec, 4),
        "ttft_sec": round(ttft_sec, 4),
        "tbt_sec": round(tbt_sec, 6),
        "session_build_sec": round(prefix["session_build_sec"], 4) if turn_index == 1 else 0.0,
        "total_sec": round((prefix["session_build_sec"] if turn_index == 1 else 0.0) + generate_sec, 4),
        "target_tool_name": target_tool,
        "prediction_tool_name": pred_tool,
        "tool_name_match": target_tool is not None and target_tool == pred_tool,
        "has_tool_call": "<tool_call>" in prediction or "Action:" in prediction,
        "exact_match": _normalize_text(prediction) == _normalize_text(target),
        "prediction": prediction,
        "target": target,
    }


def _summarize_rows(rows: List[Dict[str, Any]], sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid_rows = [row for row in rows if not row.get("skipped")]
    session_ids = sorted({row["session_id"] for row in valid_rows})
    session_accs = []
    session_success = []
    first_error_turns = []
    for session_id in session_ids:
        group = [row for row in valid_rows if row["session_id"] == session_id]
        if not group:
            continue
        correct = [bool(row["tool_name_match"]) for row in group]
        session_accs.append(sum(correct) / len(correct))
        session_success.append(all(correct))
        for row, ok in zip(group, correct):
            if not ok:
                first_error_turns.append(row["turn_index"])
                break

    total_session_sec = sum(item["session_total_sec"] for item in sessions)
    total_build_sec = sum(item["session_build_sec"] for item in sessions)
    total_generate_sec = sum(item["session_generate_sec"] for item in sessions)
    total_spans = sum(item["num_valid_spans"] for item in sessions)
    total_generated_tokens = sum(row.get("generated_tokens", 0) for row in valid_rows)
    return {
        "num_sessions": len(sessions),
        "num_spans": len(rows),
        "num_valid_spans": len(valid_rows),
        "exact_match": (
            sum(1 for row in valid_rows if row["exact_match"]) / len(valid_rows)
            if valid_rows else 0.0
        ),
        "tool_name_accuracy": (
            sum(1 for row in valid_rows if row["tool_name_match"]) / len(valid_rows)
            if valid_rows else 0.0
        ),
        "tool_call_rate": (
            sum(1 for row in valid_rows if row["has_tool_call"]) / len(valid_rows)
            if valid_rows else 0.0
        ),
        "call_accuracy": (
            sum(1 for row in valid_rows if row["tool_name_match"])
            / sum(1 for row in valid_rows if row["has_tool_call"])
            if any(row["has_tool_call"] for row in valid_rows) else 0.0
        ),
        "session_macro_tool_name_accuracy": (
            sum(session_accs) / len(session_accs) if session_accs else 0.0
        ),
        "session_success_rate": (
            sum(1 for item in session_success if item) / len(session_success)
            if session_success else 0.0
        ),
        "avg_first_error_turn": (
            sum(first_error_turns) / len(first_error_turns) if first_error_turns else None
        ),
        "avg_session_build_sec": (
            total_build_sec / len(sessions) if sessions else 0.0
        ),
        "avg_session_total_sec": (
            total_session_sec / len(sessions) if sessions else 0.0
        ),
        "avg_generate_sec_per_span": (
            total_generate_sec / total_spans if total_spans else 0.0
        ),
        "avg_generated_tokens": (
            total_generated_tokens / total_spans if total_spans else 0.0
        ),
        "avg_session_ttft_sec": (
            total_build_sec / len(sessions) if sessions else 0.0
        ),
        "amortized_ttft_sec_per_span": (
            total_build_sec / total_spans if total_spans else 0.0
        ),
        "avg_tbt_sec": (
            sum(row.get("tbt_sec", 0.0) for row in valid_rows) / total_spans
            if total_spans else 0.0
        ),
        "token_weighted_tbt_sec": (
            total_generate_sec / total_generated_tokens
            if total_generated_tokens else 0.0
        ),
        "amortized_total_sec_per_span": (
            total_session_sec / total_spans if total_spans else 0.0
        ),
        "avg_actual_compression_ratio": (
            sum(row.get("actual_compression_ratio", 0.0) for row in valid_rows) / len(valid_rows)
            if valid_rows else 0.0
        ),
    }


def _make_run_args(args: argparse.Namespace, mode: str, ratio: int) -> argparse.Namespace:
    run_args = copy.copy(args)
    run_args.row_mode = mode
    run_args.mode = "c2kv" if mode == "c2kv_untrained" else mode
    run_args.untrained_c2kv = mode == "c2kv_untrained"
    run_args.override_ratio = ratio
    return run_args


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    device = _setup_device(args.device_type)
    tokenizer_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        local_files_only=True,
        padding_side="right",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    data_args = AgentToolDefinitionDataArgs(
        dataset_path=args.dataset_path,
        eval_ratio=args.eval_ratio,
        split_seed=args.split_seed,
        split_manifest_file=args.split_manifest_file,
        split_manifest_name=args.split_manifest_name,
        max_samples_per_session=0,
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
    examples = list(source.iter_examples(args.split))
    sessions = _select_sessions(
        tokenizer,
        _group_examples_by_session(examples),
        args,
    )

    modes = [item.strip() for item in (args.compare_modes or args.mode).split(",") if item.strip()]
    ratios = [int(item.strip()) for item in (args.ratios or str(args.override_ratio)).split(",") if item.strip()]
    all_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    for mode in modes:
        run_ratios = [1] if mode == "full" else ratios
        for ratio in run_ratios:
            for reuse in ([False, True] if args.compare_reuse else [args.reuse]):
                run_args = _make_run_args(args, mode, ratio)
                run_args.reuse = reuse
                logger.info("Loading model for mode=%s ratio=%s reuse=%s", mode, ratio, reuse)
                model = _load_model(run_args, tokenizer, device)
                rows: List[Dict[str, Any]] = []
                session_stats: List[Dict[str, Any]] = []
                desc = f"{mode}@{ratio}x:{'reuse' if reuse else 'no_reuse'}"
                for session_id, session_examples in tqdm(sessions.items(), desc=desc):
                    if reuse:
                        prefix = _build_reuse_prefix(model, tokenizer, session_examples[0], run_args)
                        if prefix is None or prefix.get("skipped"):
                            continue
                        session_rows = []
                        session_generate_sec = 0.0
                        for turn_index, example in enumerate(session_examples, start=1):
                            row = _generate_span_with_reuse(
                                model, tokenizer, example, prefix, run_args, turn_index
                            )
                            session_rows.append(row)
                            if not row.get("skipped"):
                                session_generate_sec += row.get("generate_sec", 0.0)
                        rows.extend(session_rows)
                        valid_count = sum(1 for row in session_rows if not row.get("skipped"))
                        session_stats.append({
                            "session_id": session_id,
                            "reuse": True,
                            "mode": mode,
                            "ratio": ratio,
                            "num_spans": len(session_rows),
                            "num_valid_spans": valid_count,
                            "session_build_sec": prefix["session_build_sec"],
                            "session_generate_sec": session_generate_sec,
                            "session_total_sec": prefix["session_build_sec"] + session_generate_sec,
                        })
                    else:
                        session_rows = []
                        for turn_index, example in enumerate(session_examples, start=1):
                            row = _generate_one(model, tokenizer, example, run_args, device)
                            row["mode"] = mode
                            row["ratio"] = ratio
                            row["reuse"] = False
                            row["turn_index"] = turn_index
                            session_rows.append(row)
                        rows.extend(session_rows)
                        valid_rows = [row for row in session_rows if not row.get("skipped")]
                        session_stats.append({
                            "session_id": session_id,
                            "reuse": False,
                            "mode": mode,
                            "ratio": ratio,
                            "num_spans": len(session_rows),
                            "num_valid_spans": len(valid_rows),
                            "session_build_sec": sum(
                                row.get("system_prefill_sec", 0.0)
                                + row.get("tool_compress_sec", 0.0)
                                + row.get("full_prefill_sec", 0.0)
                                + row.get("blend_sec", 0.0)
                                for row in valid_rows
                            ),
                            "session_generate_sec": sum(row.get("generate_sec", 0.0) for row in valid_rows),
                            "session_total_sec": sum(row.get("total_sec", 0.0) for row in valid_rows),
                        })

                summary = _summarize_rows(rows, session_stats)
                summary.update({"mode": mode, "ratio": ratio, "reuse": reuse})
                summaries.append(summary)
                all_rows.extend(rows)
                del model
                if device == "cuda":
                    torch.cuda.empty_cache()
                elif device == "npu":
                    torch.npu.empty_cache()

    summary_by_key = {
        (item["mode"], item["ratio"], item["reuse"]): item for item in summaries
    }
    for item in summaries:
        if item["reuse"]:
            no_reuse = summary_by_key.get((item["mode"], item["ratio"], False))
            if no_reuse and item["amortized_total_sec_per_span"] > 0:
                item["reuse_speedup"] = (
                    no_reuse["amortized_total_sec_per_span"] / item["amortized_total_sec_per_span"]
                )
            else:
                item["reuse_speedup"] = None

    output_summary = {
        "model": args.model,
        "base_model": args.base_model,
        "dataset_path": args.dataset_path,
        "split": args.split,
        "num_sessions": len(sessions),
        "results": summaries,
    }
    if args.output_file:
        _jsonl_write(args.output_file, all_rows)
        summary_path = str(Path(args.output_file).with_suffix(".summary.json"))
        Path(summary_path).write_text(
            json.dumps(output_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Wrote predictions to %s", args.output_file)
        logger.info("Wrote summary to %s", summary_path)
    return output_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Session-level tool-definition KV reuse evaluation.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base_model")
    parser.add_argument("--tokenizer")
    parser.add_argument("--dataset_path", default="./datasets/agent-llm-traces")
    parser.add_argument("--output_file", default="./outputs/agent_tooldef_session_reuse_eval.jsonl")
    parser.add_argument("--split", choices=["train", "eval"], default="eval")
    parser.add_argument("--mode", choices=["c2kv", "c2kv_untrained", "truncate", "full"], default="c2kv")
    parser.add_argument("--compare_modes")
    parser.add_argument("--ratios")
    parser.add_argument("--compare_reuse", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--reuse", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--max_sessions", type=int, default=20)
    parser.add_argument("--min_spans_per_session", type=int, default=2)
    parser.add_argument("--max_spans_per_session", type=int, default=0)
    parser.add_argument("--selection_filter", choices=["c2kv", "none"], default="c2kv")
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--split_manifest_file")
    parser.add_argument("--split_manifest_name", default="toolset_disjoint")
    parser.add_argument("--max_doc_length", type=int, default=1024)
    parser.add_argument("--max_doc_num", type=int, default=10)
    parser.add_argument("--max_tool_definition_tokens", type=int, default=10000)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_system_length", type=int, default=256)
    parser.add_argument("--max_prompt_tokens", type=int, default=1920)
    parser.add_argument("--max_baseline_input_tokens", type=int, default=12000)
    parser.add_argument("--min_target_tokens", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--truncate_tool_definition", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--require_tool_call", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--device_type", choices=["auto", "cuda", "npu", "cpu"], default="auto")
    parser.add_argument("--system_attn_impl", default="eager")
    parser.add_argument("--gist_attn_impl", default="eager")
    parser.add_argument("--generate_attn_impl", default="eager")
    parser.add_argument("--override_ratio", type=int, default=4)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--baseline_model_class", choices=["gist", "auto"], default="gist")
    parser.add_argument("--untrained_c2kv", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
