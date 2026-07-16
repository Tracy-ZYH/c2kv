from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_agent_tool_definition_c2kv import (  # noqa: E402
    _build_tool_cache,
    _build_tool_chunks,
    _extract_tool_name,
    _generate_from_input_ids,
    _load_model,
    _normalize_text,
    _prefill_system,
    _prefill_tokens_with_cache,
    _setup_device,
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


def _json_loads(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _as_tool_list(tool_definition: str) -> List[Dict[str, Any]]:
    parsed = _json_loads(tool_definition, [])
    if isinstance(parsed, dict):
        if isinstance(parsed.get("tools"), list):
            parsed = parsed["tools"]
        elif isinstance(parsed.get("functions"), list):
            parsed = parsed["functions"]
        else:
            parsed = [parsed]
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _tool_name(tool: Dict[str, Any]) -> str:
    function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
    return str(
        function.get("name")
        or tool.get("name")
        or tool.get("tool_name")
        or tool.get("function_name")
        or ""
    )


def _tool_search_text(tool: Dict[str, Any]) -> str:
    function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
    fields = [
        _tool_name(tool),
        function.get("description", ""),
        tool.get("description", ""),
        function.get("parameters", ""),
        tool.get("parameters", ""),
        tool.get("input_schema", ""),
        tool.get("schema", ""),
    ]
    return " ".join(
        item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
        for item in fields
        if item
    )


def _render_tool_definition(tools: Sequence[Dict[str, Any]]) -> str:
    return json.dumps(list(tools), ensure_ascii=False, separators=(",", ":"))


def _message_text(message: Dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False) if content is not None else ""


def _query_text(messages: Sequence[Dict[str, Any]], router_scope: str) -> str:
    if router_scope == "all":
        return "\n".join(_message_text(message) for message in messages)
    for message in reversed(messages):
        if message.get("role") == "user":
            return _message_text(message)
    return _message_text(messages[-1]) if messages else ""


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _rank_tools(tools: Sequence[Dict[str, Any]], query: str) -> List[int]:
    query_tokens = set(_tokens(query))
    if not query_tokens:
        return list(range(len(tools)))
    scored = []
    for index, tool in enumerate(tools):
        name_tokens = set(_tokens(_tool_name(tool)))
        text_tokens = set(_tokens(_tool_search_text(tool)))
        name_overlap = len(query_tokens & name_tokens)
        text_overlap = len(query_tokens & text_tokens)
        score = 4.0 * name_overlap + float(text_overlap)
        scored.append((-score, index))
    scored.sort()
    return [index for _, index in scored]


def _split_topk_tools(
    tools: Sequence[Dict[str, Any]],
    query: str,
    top_k: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    ranked = _rank_tools(tools, query)
    top_indices = set(ranked[: max(0, top_k)])
    top_tools = [tool for index, tool in enumerate(tools) if index in top_indices]
    rest_tools = [tool for index, tool in enumerate(tools) if index not in top_indices]
    return top_tools, rest_tools, [_tool_name(tools[index]) for index in ranked[: max(0, top_k)]]


def _split_random_topk_tools(
    tools: Sequence[Dict[str, Any]],
    top_k: int,
    seed_text: str,
    seed: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    rng = random.Random(f"{seed}:{seed_text}:{top_k}:{len(tools)}")
    ranked = list(range(len(tools)))
    rng.shuffle(ranked)
    top_indices = set(ranked[: max(0, top_k)])
    top_tools = [tool for index, tool in enumerate(tools) if index in top_indices]
    rest_tools = [tool for index, tool in enumerate(tools) if index not in top_indices]
    return top_tools, rest_tools, [_tool_name(tools[index]) for index in ranked[: max(0, top_k)]]


def _parse_cases(cases: str) -> List[tuple[int, int]]:
    parsed = []
    for item in cases.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            top_k, ratio = item.split(":", 1)
        elif "x" in item.lower():
            top_k, ratio = item.lower().split("x", 1)
        else:
            raise ValueError(f"Invalid case {item!r}; expected TOPK:RATIO, e.g. 3:4")
        parsed.append((int(top_k), int(ratio)))
    return parsed


@torch.inference_mode()
def _generate_one_hybrid(
    model: Any,
    tokenizer: Any,
    example: Any,
    args: argparse.Namespace,
    top_k: int,
    ratio: int,
) -> Dict[str, Any]:
    total_start = time.perf_counter()
    hybrid_mode = getattr(args, "hybrid_mode", "hybrid")
    tools = _as_tool_list(example.tool_definition)
    if not tools:
        return {
            "qid": example.qid,
            "session_id": example.session_id,
            "mode": "hybrid",
            "hybrid_mode": hybrid_mode,
            "router_strategy": args.router_strategy,
            "top_k": top_k,
            "ratio": ratio,
            "skipped": True,
            "skip_reason": "no_parseable_tools",
        }

    query = _query_text(example.input_messages, args.router_scope)
    if args.router_strategy == "random":
        top_tools, rest_tools, top_tool_names = _split_random_topk_tools(
            tools,
            top_k,
            seed_text=example.qid,
            seed=args.router_seed,
        )
    else:
        top_tools, rest_tools, top_tool_names = _split_topk_tools(tools, query, top_k)
    selected_top_tools = top_tools
    selected_rest_tools = rest_tools
    top_definition = _render_tool_definition(top_tools)
    rest_definition = _render_tool_definition(rest_tools)
    target = example.answer.strip()
    target_tool = _extract_tool_name(target)
    router_hit = target_tool in set(top_tool_names) if target_tool else False
    if args.router_hit_filter == "hit" and not router_hit:
        return {
            "qid": example.qid,
            "session_id": example.session_id,
            "mode": "hybrid",
            "hybrid_mode": hybrid_mode,
            "router_strategy": args.router_strategy,
            "top_k": top_k,
            "ratio": ratio,
            "skipped": True,
            "skip_reason": "router_miss_filtered",
            "num_tools": len(tools),
            "top_tool_names": top_tool_names,
            "target_tool_name": target_tool,
            "router_hit": router_hit,
        }
    if args.router_hit_filter == "miss" and router_hit:
        return {
            "qid": example.qid,
            "session_id": example.session_id,
            "mode": "hybrid",
            "hybrid_mode": hybrid_mode,
            "router_strategy": args.router_strategy,
            "top_k": top_k,
            "ratio": ratio,
            "skipped": True,
            "skip_reason": "router_hit_filtered",
            "num_tools": len(tools),
            "top_tool_names": top_tool_names,
            "target_tool_name": target_tool,
            "router_hit": router_hit,
        }

    if hybrid_mode == "drop_selected":
        top_tools = []
        rest_tools = selected_rest_tools
    elif hybrid_mode == "topk_only":
        top_tools = selected_top_tools
        rest_tools = []
    elif hybrid_mode != "hybrid":
        raise ValueError(f"Unknown hybrid_mode: {hybrid_mode}")
    top_definition = _render_tool_definition(top_tools)
    rest_definition = _render_tool_definition(rest_tools)

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

    top_ids = []
    top_prefill_sec = 0.0
    top_length = 0
    prefix_cache = system_cache
    if top_tools:
        top_doc = {"role": "user", "content": "Top-k tool definitions:\n" + top_definition}
        top_ids = _chat_template_ids(tokenizer, [top_doc])
        tool_input_ids = torch.tensor([top_ids], dtype=torch.long, device=model.device)
        prefix_cache, top_length, top_prefill_sec = _prefill_tokens_with_cache(
            model,
            tool_input_ids,
            past_key_values=system_cache,
            past_length=system_length,
            attn_impl=args.generate_attn_impl,
        )

    rest_length = 0
    rest_doc_tokens = 0
    rest_doc_chunks = 0
    rest_gist_tokens = 0
    rest_actual_ratio = 0.0
    rest_compress_sec = 0.0
    blend_sec = 0.0
    has_c2kv_rest = bool(rest_tools)
    if rest_tools:
        context_input_ids, rest_doc_tokens, rest_doc_chunks, skip_reason = _build_tool_chunks(
            tokenizer,
            rest_definition,
            max_doc_length=args.max_doc_length,
            max_doc_num=args.max_doc_num,
            max_tool_definition_tokens=args.max_tool_definition_tokens,
            truncate_tool_definition=args.truncate_tool_definition,
        )
        if context_input_ids is None:
            return {
                "qid": example.qid,
                "session_id": example.session_id,
                "mode": "hybrid",
                "hybrid_mode": hybrid_mode,
                "router_strategy": args.router_strategy,
                "top_k": top_k,
                "ratio": ratio,
                "skipped": True,
                "skip_reason": "rest_" + str(skip_reason),
                "num_tools": len(tools),
                "top_tool_names": top_tool_names,
                "rest_doc_tokens": rest_doc_tokens,
            }
        (
            prefix_cache,
            rest_length,
            rest_gist_tokens,
            rest_actual_ratio,
            rest_compress_sec,
            blend_sec,
        ) = _build_tool_cache(
            model,
            context_input_ids,
            prefix_cache,
            system_length + top_length,
            args.gist_attn_impl,
            ratio,
        )

    prompt_ids = _chat_template_ids(
        tokenizer,
        example.input_messages,
        add_generation_prompt=True,
    )
    if args.max_prompt_tokens and len(prompt_ids) > args.max_prompt_tokens:
        prompt_ids = prompt_ids[-args.max_prompt_tokens :]
    prompt_input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
    cache_length = prefix_cache.get_seq_length()
    mock_cache_ids = prompt_input_ids.new_zeros((1, cache_length))
    input_ids = torch.cat([mock_cache_ids, prompt_input_ids], dim=1)
    original_prefix_length = system_length + top_length + rest_length
    position_ids = torch.arange(
        original_prefix_length,
        original_prefix_length + prompt_input_ids.shape[1],
        dtype=torch.long,
        device=model.device,
    ).unsqueeze(0)

    prediction, generate_sec, generated_tokens, tbt_sec = _generate_from_input_ids(
        model,
        tokenizer,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        attn_impl=args.generate_attn_impl,
        use_gist=has_c2kv_rest,
        position_ids=position_ids,
        past_key_values=prefix_cache,
    )
    pred_tool = _extract_tool_name(prediction)
    full_doc_tokens = len(
        _chat_template_ids(
            tokenizer,
            [{"role": "user", "content": "Tool definitions:\n" + example.tool_definition}],
        )
    )
    online_ttft_sec = system_prefill_sec + top_prefill_sec + rest_compress_sec + blend_sec
    cached_ttft_sec = system_prefill_sec + top_prefill_sec + blend_sec
    tool_only_cached_ttft_sec = top_prefill_sec + blend_sec
    cached_total_sec = cached_ttft_sec + generate_sec
    total_sec = time.perf_counter() - total_start
    compressed_tool_tokens = top_length + rest_gist_tokens
    hybrid_ratio = (
        (top_length + rest_length) / compressed_tool_tokens
        if compressed_tool_tokens else 0.0
    )
    return {
        "qid": example.qid,
        "session_id": example.session_id,
        "mode": "hybrid",
        "hybrid_mode": hybrid_mode,
        "router_strategy": args.router_strategy,
        "top_k": top_k,
        "ratio": ratio,
        "skipped": False,
        "num_tools": len(tools),
        "num_top_tools": len(top_tools),
        "num_rest_tools": len(rest_tools),
        "top_tool_names": top_tool_names,
        "router_scope": args.router_scope,
        "router_strategy": args.router_strategy,
        "router_hit": router_hit,
        "doc_tokens": full_doc_tokens,
        "top_doc_tokens": len(top_ids),
        "rest_doc_tokens": rest_doc_tokens,
        "rest_doc_chunks": rest_doc_chunks,
        "rest_gist_tokens": rest_gist_tokens,
        "actual_compression_ratio": round(hybrid_ratio, 4),
        "rest_actual_compression_ratio": round(rest_actual_ratio, 4),
        "prompt_tokens": len(prompt_ids),
        "target_tokens": len(tokenizer.encode(target, add_special_tokens=False)),
        "generated_tokens": generated_tokens,
        "latency_sec": round(generate_sec, 4),
        "system_prefill_sec": round(system_prefill_sec, 4),
        "top_full_prefill_sec": round(top_prefill_sec, 4),
        "tool_compress_sec": round(rest_compress_sec, 4),
        "full_prefill_sec": round(top_prefill_sec, 4),
        "blend_sec": round(blend_sec, 4),
        "generate_sec": round(generate_sec, 4),
        "ttft_sec": round(online_ttft_sec, 4),
        "online_ttft_sec": round(online_ttft_sec, 4),
        "cached_ttft_sec": round(cached_ttft_sec, 4),
        "tool_only_cached_ttft_sec": round(tool_only_cached_ttft_sec, 4),
        "tbt_sec": round(tbt_sec, 6),
        "total_sec": round(total_sec, 4),
        "cached_total_sec": round(cached_total_sec, 4),
        "target_tool_name": target_tool,
        "prediction_tool_name": pred_tool,
        "tool_name_match": target_tool is not None and target_tool == pred_tool,
        "has_tool_call": "<tool_call>" in prediction or "Action:" in prediction,
        "exact_match": _normalize_text(prediction) == _normalize_text(target),
        "prediction": prediction,
        "target": target,
    }


def _summarize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries = []
    keys = sorted({
        (
            row.get("hybrid_mode", "hybrid"),
            row.get("router_strategy", "lexical"),
            row.get("top_k"),
            row.get("ratio"),
        )
        for row in rows
    })
    for hybrid_mode, router_strategy, top_k, ratio in keys:
        group = [
            row for row in rows
            if row.get("hybrid_mode", "hybrid") == hybrid_mode
            and row.get("router_strategy", "lexical") == router_strategy
            and row.get("top_k") == top_k
            and row.get("ratio") == ratio
        ]
        valid_rows = [row for row in group if not row.get("skipped")]
        skip_reasons = Counter(row.get("skip_reason", "unknown") for row in group if row.get("skipped"))
        total_generated = sum(row.get("generated_tokens", 0) for row in valid_rows)
        summaries.append({
            "mode": "hybrid",
            "hybrid_mode": hybrid_mode,
            "router_strategy": router_strategy,
            "top_k": top_k,
            "ratio": ratio,
            "num_examples": len(group),
            "num_valid": len(valid_rows),
            "num_skipped": len(group) - len(valid_rows),
            "skip_reasons": dict(skip_reasons),
            "router_hit_rate": (
                sum(1 for row in valid_rows if row.get("router_hit")) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "exact_match": (
                sum(1 for row in valid_rows if row.get("exact_match")) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "tool_name_accuracy": (
                sum(1 for row in valid_rows if row.get("tool_name_match")) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "tool_call_rate": (
                sum(1 for row in valid_rows if row.get("has_tool_call")) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "call_accuracy": (
                sum(1 for row in valid_rows if row.get("tool_name_match"))
                / sum(1 for row in valid_rows if row.get("has_tool_call"))
                if any(row.get("has_tool_call") for row in valid_rows) else 0.0
            ),
            "avg_num_tools": (
                sum(row.get("num_tools", 0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_top_doc_tokens": (
                sum(row.get("top_doc_tokens", 0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_rest_doc_tokens": (
                sum(row.get("rest_doc_tokens", 0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_actual_compression_ratio": (
                sum(row.get("actual_compression_ratio", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_online_ttft_sec": (
                sum(row.get("online_ttft_sec", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_cached_ttft_sec": (
                sum(row.get("cached_ttft_sec", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_tool_only_cached_ttft_sec": (
                sum(row.get("tool_only_cached_ttft_sec", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_generate_sec": (
                sum(row.get("generate_sec", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_generated_tokens": (
                total_generated / len(valid_rows) if valid_rows else 0.0
            ),
            "avg_tbt_sec": (
                sum(row.get("tbt_sec", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "token_weighted_tbt_sec": (
                sum(row.get("generate_sec", 0.0) for row in valid_rows) / total_generated
                if total_generated else 0.0
            ),
            "avg_online_total_sec": (
                sum(row.get("total_sec", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_cached_total_sec": (
                sum(row.get("cached_total_sec", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
        })
    return summaries


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

    # Keep the same selection as C2KV eval by default so hybrid is compared on
    # examples whose full tool definition fits the existing C2KV budget.
    examples = []
    selection_skips: Counter[str] = Counter()
    if args.selection_filter == "c2kv":
        for example in source_examples:
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
    else:
        examples = (
            source_examples[: args.max_examples]
            if args.max_examples is not None and args.max_examples > 0
            else source_examples
        )

    cases = _parse_cases(args.hybrid_cases)
    logger.info(
        "Selected %d examples from %d source examples; cases=%s; selection_skips=%s",
        len(examples),
        len(source_examples),
        cases,
        dict(selection_skips),
    )
    model_args = copy.copy(args)
    model_args.mode = "c2kv"
    model_args.untrained_c2kv = False
    model = _load_model(model_args, tokenizer, device)

    rows: List[Dict[str, Any]] = []
    for top_k, ratio in cases:
        desc = f"hybrid_top{top_k}_c2kv{ratio}x"
        for example in tqdm(examples, desc=desc):
            rows.append(_generate_one_hybrid(model, tokenizer, example, args, top_k, ratio))

    summaries = _summarize_rows(rows)
    summary = {
        "model": args.model,
        "dataset_path": args.dataset_path,
        "split": args.split,
        "router_scope": args.router_scope,
        "router_strategy": args.router_strategy,
        "hybrid_mode": args.hybrid_mode,
        "hybrid_cases": args.hybrid_cases,
        "selection_skips": dict(selection_skips),
        "results": summaries,
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
    parser = argparse.ArgumentParser(description="Evaluate top-k full tools + rest C2KV hybrid routing.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base_model")
    parser.add_argument("--tokenizer")
    parser.add_argument("--dataset_path", default="./datasets/agent-llm-traces")
    parser.add_argument("--output_file", default="./outputs/agent_tooldef_hybrid_router_eval.jsonl")
    parser.add_argument("--split", choices=["train", "eval"], default="eval")
    parser.add_argument("--hybrid_cases", default="1:4,3:4,5:8")
    parser.add_argument("--hybrid_mode", choices=["hybrid", "drop_selected", "topk_only"], default="hybrid")
    parser.add_argument("--router_scope", choices=["last_user", "all"], default="last_user")
    parser.add_argument("--router_strategy", choices=["lexical", "random"], default="lexical")
    parser.add_argument("--router_hit_filter", choices=["all", "hit", "miss"], default="all")
    parser.add_argument("--router_seed", type=int, default=42)
    parser.add_argument("--max_examples", type=int, default=50, help="Maximum examples; <=0 means all selected examples.")
    parser.add_argument("--max_source_examples", type=int)
    parser.add_argument("--selection_filter", choices=["c2kv", "none"], default="c2kv")
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--max_samples_per_session", type=int, default=4)
    parser.add_argument("--max_doc_length", type=int, default=1024)
    parser.add_argument("--max_doc_num", type=int, default=10)
    parser.add_argument("--max_tool_definition_tokens", type=int, default=10000)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_system_length", type=int, default=256)
    parser.add_argument("--max_prompt_tokens", type=int, default=1920)
    parser.add_argument("--min_target_tokens", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--truncate_tool_definition", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--require_tool_call", type=lambda x: str(x).lower() == "true", default=True)
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
    summary = evaluate(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
