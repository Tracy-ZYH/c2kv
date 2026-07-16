from __future__ import annotations

import argparse
import copy
import json
import logging
import re
import sys
from collections import Counter, OrderedDict
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
from eval_agent_tool_definition_hybrid_router import (  # noqa: E402
    _as_tool_list,
    _query_text,
    _render_tool_definition,
    _split_topk_tools,
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


TOOL_CALL_INSTRUCTION = (
    "You are continuing an agent task. Use the tool definitions and the relevant "
    "memory if it helps. If the next step requires a tool, respond in this exact format:\n"
    "Action:\n"
    "<tool_call>\n"
    "{\"name\":\"tool_name\",\"arguments\":{}}\n"
    "</tool_call>\n"
    "Do not answer with a plan when a tool call is needed."
)


def _jsonl_write(path: str, rows: List[Dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _span_index(qid: str) -> int:
    match = re.search(r":(\d+)$", qid)
    return int(match.group(1)) if match else 0


def _group_examples_by_session(examples: Sequence[Any]) -> OrderedDict[str, List[Any]]:
    sessions: OrderedDict[str, List[Any]] = OrderedDict()
    for example in examples:
        sessions.setdefault(example.session_id, []).append(example)
    for session_id in list(sessions.keys()):
        sessions[session_id] = sorted(sessions[session_id], key=lambda item: _span_index(item.qid))
    return sessions


def _tool_call_payload(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    blocks = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, flags=re.S)
    candidates = blocks or [text]
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except Exception:
            value = None
        if isinstance(value, dict):
            function = value.get("function") if isinstance(value.get("function"), dict) else {}
            name = (
                value.get("name")
                or value.get("tool_name")
                or value.get("function_name")
                or function.get("name")
            )
            arguments = (
                value.get("arguments")
                or value.get("args")
                or value.get("input")
                or function.get("arguments")
                or {}
            )
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except Exception:
                    pass
            if name:
                return {"name": str(name), "arguments": arguments}
    return None


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _arguments_match(target: str, prediction: str) -> bool:
    target_call = _tool_call_payload(target)
    pred_call = _tool_call_payload(prediction)
    if not target_call or not pred_call:
        return False
    if target_call["name"] != pred_call["name"]:
        return False
    return _canonical_json(target_call.get("arguments", {})) == _canonical_json(pred_call.get("arguments", {}))


def _last_user_query(example: Any) -> str:
    return _query_text(example.input_messages, "last_user")


def _memory_summary(example: Any) -> str:
    target_tool = _extract_tool_name(example.answer) or "none"
    payload = _tool_call_payload(example.answer) or {}
    arguments = payload.get("arguments", {})
    status = "success" if example.has_tool_call else "unknown"
    return (
        f"User query: {_last_user_query(example)}\n"
        f"Tool call: {target_tool}\n"
        f"Arguments: {_canonical_json(arguments)}\n"
        f"Status: {status}"
    )


def _memory_raw(example: Any) -> str:
    payload = _tool_call_payload(example.answer) or {}
    target_tool = payload.get("name") or _extract_tool_name(example.answer) or "none"
    arguments = payload.get("arguments", {})
    return (
        f"User query:\n{_last_user_query(example)}\n\n"
        "Assistant tool call:\n"
        f"{target_tool}\n"
        f"Arguments:\n{_canonical_json(arguments)}"
    )


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n[truncated]"


def _rank_memories(query: str, memories: Sequence[Any]) -> List[int]:
    query_tokens = set(_tokens(query))
    if not query_tokens:
        return list(range(len(memories)))
    scored = []
    for index, memory in enumerate(memories):
        memory_tokens = set(_tokens(_last_user_query(memory) + "\n" + memory.answer))
        score = len(query_tokens & memory_tokens)
        scored.append((-score, -_span_index(memory.qid), index))
    scored.sort()
    return [index for _, _, index in scored]


def _select_memories(
    query: str,
    memories: Sequence[Any],
    top_m: int,
    style: str,
    max_item_chars: int,
    max_total_chars: int,
) -> List[str]:
    ranked = _rank_memories(query, memories)
    selected = [memories[index] for index in ranked[: max(0, top_m)]]
    if style == "summary":
        texts = [_memory_summary(memory) for memory in selected]
    else:
        texts = [_memory_raw(memory) for memory in selected]
    budgeted = []
    used = 0
    for text in texts:
        item = _truncate_text(text, max_item_chars)
        remaining = max_total_chars - used if max_total_chars > 0 else len(item)
        if max_total_chars > 0 and remaining <= 0:
            break
        if max_total_chars > 0 and len(item) > remaining:
            item = _truncate_text(item, remaining)
        budgeted.append(item)
        used += len(item)
    return budgeted


def _build_prompt_messages(
    example: Any,
    method: str,
    memories: Sequence[Any],
    args: argparse.Namespace,
) -> tuple[List[Dict[str, Any]], int, int]:
    if method == "full_history":
        return example.input_messages[-args.max_full_history_messages :], 0, 0

    query = _last_user_query(example)
    memory_style = "summary" if method.endswith("_summary") else "raw"
    selected_texts = []
    if method not in ("no_memory",):
        selected_texts = _select_memories(
            query,
            memories,
            args.top_m,
            memory_style,
            args.max_memory_item_chars,
            args.max_memory_total_chars,
        )

    if selected_texts:
        memory_block = "\n\n---\n\n".join(selected_texts)
        content = (
            f"{TOOL_CALL_INSTRUCTION}\n\n"
            "Relevant memory from previous turns:\n"
            f"{memory_block}\n\n"
            "Current user query:\n"
            f"{query}"
        )
    else:
        content = f"{TOOL_CALL_INSTRUCTION}\n\nCurrent user query:\n{query}"
    return [{"role": "user", "content": content}], len(selected_texts), len(content)


def _full_tool_definition_ids(tokenizer: Any, tool_definition: str) -> List[int]:
    return _chat_template_ids(
        tokenizer,
        [{"role": "user", "content": "Tool definitions:\n" + tool_definition}],
    )


@torch.inference_mode()
def _build_full_tool_prefix(
    model: Any,
    tokenizer: Any,
    example: Any,
    args: argparse.Namespace,
) -> tuple[Optional[tuple[Any, int, int, float, float]], Optional[str]]:
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
    tool_ids = _full_tool_definition_ids(tokenizer, example.tool_definition)
    if len(tool_ids) > args.max_tool_definition_tokens:
        return None, f"tool_definition_tokens>{args.max_tool_definition_tokens}"
    if args.max_baseline_tool_tokens is not None and len(tool_ids) > args.max_baseline_tool_tokens:
        return None, f"baseline_tool_tokens>{args.max_baseline_tool_tokens}"
    tool_input_ids = torch.tensor([tool_ids], dtype=torch.long, device=model.device)
    tool_cache, tool_length, full_prefill_sec = _prefill_tokens_with_cache(
        model,
        tool_input_ids,
        past_key_values=system_cache,
        past_length=system_length,
        attn_impl=args.generate_attn_impl,
    )
    return (tool_cache, system_length, tool_length, system_prefill_sec, full_prefill_sec), None


@torch.inference_mode()
def _build_hybrid_tool_prefix(
    model: Any,
    tokenizer: Any,
    example: Any,
    args: argparse.Namespace,
    query: str,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    tools = _as_tool_list(example.tool_definition)
    if not tools:
        return None, "no_parseable_tools"
    top_tools, rest_tools, top_tool_names = _split_topk_tools(tools, query, args.hybrid_top_k)
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
    prefix_cache = system_cache
    top_length = 0
    top_prefill_sec = 0.0
    if top_tools:
        top_definition = _render_tool_definition(top_tools)
        top_ids = _chat_template_ids(
            tokenizer,
            [{"role": "user", "content": "Top-k tool definitions:\n" + top_definition}],
        )
        top_input_ids = torch.tensor([top_ids], dtype=torch.long, device=model.device)
        prefix_cache, top_length, top_prefill_sec = _prefill_tokens_with_cache(
            model,
            top_input_ids,
            past_key_values=system_cache,
            past_length=system_length,
            attn_impl=args.generate_attn_impl,
        )
    rest_length = 0
    rest_gist_tokens = 0
    rest_compress_sec = 0.0
    blend_sec = 0.0
    has_c2kv_rest = bool(rest_tools)
    if rest_tools:
        rest_definition = _render_tool_definition(rest_tools)
        context_input_ids, _, _, skip_reason = _build_tool_chunks(
            tokenizer,
            rest_definition,
            max_doc_length=args.max_doc_length,
            max_doc_num=args.max_doc_num,
            max_tool_definition_tokens=args.max_tool_definition_tokens,
            truncate_tool_definition=args.truncate_tool_definition,
        )
        if context_input_ids is None:
            return None, "rest_" + str(skip_reason)
        (
            prefix_cache,
            rest_length,
            rest_gist_tokens,
            _,
            rest_compress_sec,
            blend_sec,
        ) = _build_tool_cache(
            model,
            context_input_ids,
            prefix_cache,
            system_length + top_length,
            args.gist_attn_impl,
            args.hybrid_ratio,
        )
    return {
        "cache": prefix_cache,
        "system_length": system_length,
        "tool_length": top_length + rest_length,
        "cache_length": prefix_cache.get_seq_length(),
        "system_prefill_sec": system_prefill_sec,
        "full_prefill_sec": top_prefill_sec,
        "tool_compress_sec": rest_compress_sec,
        "blend_sec": blend_sec,
        "has_c2kv_rest": has_c2kv_rest,
        "top_tool_names": top_tool_names,
        "rest_gist_tokens": rest_gist_tokens,
    }, None


@torch.inference_mode()
def _generate_with_prefix(
    model: Any,
    tokenizer: Any,
    example: Any,
    prompt_messages: Sequence[Dict[str, Any]],
    prefix: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    prompt_ids = _chat_template_ids(tokenizer, list(prompt_messages), add_generation_prompt=True)
    if len(prompt_ids) > args.max_prompt_tokens:
        prompt_ids = prompt_ids[-args.max_prompt_tokens :]
    prompt_input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
    mock_cache_ids = prompt_input_ids.new_zeros((1, prefix["cache_length"]))
    input_ids = torch.cat([mock_cache_ids, prompt_input_ids], dim=1)
    original_prefix_length = prefix["system_length"] + prefix["tool_length"]
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
        use_gist=prefix.get("has_c2kv_rest", False),
        position_ids=position_ids,
        past_key_values=prefix["cache"],
    )
    target = example.answer.strip()
    target_tool = _extract_tool_name(target)
    pred_tool = _extract_tool_name(prediction)
    online_ttft = (
        prefix.get("system_prefill_sec", 0.0)
        + prefix.get("full_prefill_sec", 0.0)
        + prefix.get("tool_compress_sec", 0.0)
        + prefix.get("blend_sec", 0.0)
    )
    cached_ttft = (
        prefix.get("system_prefill_sec", 0.0)
        + prefix.get("full_prefill_sec", 0.0)
        + prefix.get("blend_sec", 0.0)
    )
    return {
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": generated_tokens,
        "latency_sec": round(generate_sec, 4),
        "system_prefill_sec": round(prefix.get("system_prefill_sec", 0.0), 4),
        "full_prefill_sec": round(prefix.get("full_prefill_sec", 0.0), 4),
        "tool_compress_sec": round(prefix.get("tool_compress_sec", 0.0), 4),
        "blend_sec": round(prefix.get("blend_sec", 0.0), 4),
        "generate_sec": round(generate_sec, 4),
        "online_ttft_sec": round(online_ttft, 4),
        "cached_ttft_sec": round(cached_ttft, 4),
        "tbt_sec": round(tbt_sec, 6),
        "online_total_sec": round(online_ttft + generate_sec, 4),
        "cached_total_sec": round(cached_ttft + generate_sec, 4),
        "target_tool_name": target_tool,
        "prediction_tool_name": pred_tool,
        "tool_name_match": target_tool is not None and target_tool == pred_tool,
        "argument_match": _arguments_match(target, prediction),
        "has_tool_call": "<tool_call>" in prediction or "Action:" in prediction,
        "exact_match": _normalize_text(prediction) == _normalize_text(target),
        "prediction": prediction,
        "target": target,
    }


@torch.inference_mode()
def _generate_one(
    model: Any,
    tokenizer: Any,
    example: Any,
    memories: Sequence[Any],
    method: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    query = _last_user_query(example)
    prompt_messages, memory_count, memory_chars = _build_prompt_messages(
        example, method, memories, args
    )
    if method.startswith("hybrid"):
        prefix, skip_reason = _build_hybrid_tool_prefix(model, tokenizer, example, args, query)
        tool_mode = "hybrid"
    else:
        full_prefix, skip_reason = _build_full_tool_prefix(
            model, tokenizer, example, args
        )
        if full_prefix is None:
            prefix = None
            tool_mode = "full"
        else:
            tool_cache, system_length, tool_length, system_prefill_sec, full_prefill_sec = full_prefix
            prefix = {
                "cache": tool_cache,
                "system_length": system_length,
                "tool_length": tool_length,
                "cache_length": tool_cache.get_seq_length(),
                "system_prefill_sec": system_prefill_sec,
                "full_prefill_sec": full_prefill_sec,
                "tool_compress_sec": 0.0,
                "blend_sec": 0.0,
                "has_c2kv_rest": False,
            }
            tool_mode = "full"
    if prefix is None:
        return {
            "qid": example.qid,
            "session_id": example.session_id,
            "method": method,
            "tool_mode": tool_mode,
            "skipped": True,
            "skip_reason": skip_reason,
        }
    row = _generate_with_prefix(model, tokenizer, example, prompt_messages, prefix, args)
    row.update({
        "qid": example.qid,
        "session_id": example.session_id,
        "method": method,
        "tool_mode": tool_mode,
        "memory_count": memory_count,
        "memory_chars": memory_chars,
        "top_m": args.top_m,
    })
    return row


def _summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries = []
    for method in sorted({row.get("method") for row in rows}):
        group = [row for row in rows if row.get("method") == method]
        valid_rows = [row for row in group if not row.get("skipped")]
        skip_reasons = Counter(row.get("skip_reason", "unknown") for row in group if row.get("skipped"))
        total_generated = sum(row.get("generated_tokens", 0) for row in valid_rows)
        summaries.append({
            "method": method,
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
                sum(1 for row in valid_rows if row.get("has_tool_call")) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "call_accuracy": (
                sum(1 for row in valid_rows if row.get("tool_name_match"))
                / sum(1 for row in valid_rows if row.get("has_tool_call"))
                if any(row.get("has_tool_call") for row in valid_rows) else 0.0
            ),
            "argument_accuracy": (
                sum(1 for row in valid_rows if row.get("argument_match")) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_input_tokens": (
                sum(row.get("prompt_tokens", 0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_memory_count": (
                sum(row.get("memory_count", 0) for row in valid_rows) / len(valid_rows)
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
                sum(row.get("online_total_sec", 0.0) for row in valid_rows) / len(valid_rows)
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
    sessions = _group_examples_by_session(list(source.iter_examples(args.split)))
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    model_args = copy.copy(args)
    model_args.mode = "c2kv"
    model_args.untrained_c2kv = False
    model = _load_model(model_args, tokenizer, device)

    rows: List[Dict[str, Any]] = []
    selection_skips: Counter[str] = Counter()
    selected_sessions = 0
    for session_id, examples in sessions.items():
        if args.min_turn_index > 0:
            examples = [example for example in examples if _span_index(example.qid) >= args.min_turn_index]
        if len(examples) < args.min_examples_per_session:
            continue
        selected_sessions += 1
        session_all = sessions[session_id]
        for example in examples:
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
            prior = [item for item in session_all if _span_index(item.qid) < _span_index(example.qid)]
            if not prior and args.require_memory:
                continue
            for method in methods:
                rows.append(_generate_one(model, tokenizer, example, prior, method, args))
                if args.max_examples is not None and len(rows) >= args.max_examples * len(methods):
                    break
            if args.max_examples is not None and len(rows) >= args.max_examples * len(methods):
                break
        if args.max_sessions is not None and selected_sessions >= args.max_sessions:
            break
        if args.max_examples is not None and len(rows) >= args.max_examples * len(methods):
            break

    summary = {
        "model": args.model,
        "dataset_path": args.dataset_path,
        "split": args.split,
        "methods": methods,
        "top_m": args.top_m,
        "hybrid_top_k": args.hybrid_top_k,
        "hybrid_ratio": args.hybrid_ratio,
        "selection_filter": args.selection_filter,
        "selection_skips": dict(selection_skips),
        "num_rows": len(rows),
        "results": _summarize(rows),
    }
    if args.output_file:
        _jsonl_write(args.output_file, rows)
        summary_path = Path(args.output_file).with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Wrote predictions to %s", args.output_file)
        logger.info("Wrote summary to %s", str(summary_path))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate memory retrieval baselines for agent tool calling.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base_model")
    parser.add_argument("--tokenizer")
    parser.add_argument("--dataset_path", default="./datasets/agent-llm-traces")
    parser.add_argument("--output_file", default="./outputs/agent_memory_rag_eval.jsonl")
    parser.add_argument("--split", choices=["train", "eval"], default="eval")
    parser.add_argument(
        "--methods",
        default="no_memory,full_rag,hybrid_rag,hybrid_summary",
        help="Comma-separated: no_memory,full_history,full_rag,hybrid_rag,hybrid_summary.",
    )
    parser.add_argument("--top_m", type=int, default=3)
    parser.add_argument("--max_memory_item_chars", type=int, default=800)
    parser.add_argument("--max_memory_total_chars", type=int, default=2400)
    parser.add_argument("--max_full_history_messages", type=int, default=8)
    parser.add_argument("--selection_filter", choices=["c2kv", "none"], default="c2kv")
    parser.add_argument("--max_baseline_tool_tokens", type=int, default=10000)
    parser.add_argument("--hybrid_top_k", type=int, default=3)
    parser.add_argument("--hybrid_ratio", type=int, default=4)
    parser.add_argument("--max_examples", type=int, default=100)
    parser.add_argument("--max_sessions", type=int)
    parser.add_argument("--min_examples_per_session", type=int, default=2)
    parser.add_argument("--min_turn_index", type=int, default=1)
    parser.add_argument("--require_memory", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)
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
