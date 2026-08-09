from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import re
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
from eval_agent_tool_definition_hybrid_router import (  # noqa: E402
    _as_tool_list,
    _generate_one_hybrid,
    _query_text,
    _render_tool_definition,
    _split_random_topk_tools,
    _split_topk_tools,
)
from reuse_pipeline import BatchedKVInstance, LLMInference  # noqa: E402
from train_agent_tool_definition_c2kv import (  # noqa: E402
    AgentLLMTracesSource,
    AgentToolDefinitionDataArgs,
)
from train.train_data_multiturn import _chat_template_ids  # noqa: E402
from rope_reposition import rotate_k_cache_rope  # noqa: E402


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
    "snapkv_hybrid",
    "epic_leading32_snapkv_hybrid",
    "cacheblend_vdiff_snapkv_hybrid",
    "snapkv_aug_hybrid",
    "epic_leading32_snapkv_aug_hybrid",
    "cacheblend_vdiff_snapkv_aug_hybrid",
}
AGENT_MODES = {"full", "truncate", "c2kv", "c2kv_untrained", "hybrid", "c2kv_aug_hybrid"}
MODE_ALIASES = {
    "c2kv_hybrid": "hybrid",
}
REUSE_HYBRID_MODES = {
    "snapkv_hybrid",
    "epic_leading32_snapkv_hybrid",
    "cacheblend_vdiff_snapkv_hybrid",
}
REUSE_AUG_HYBRID_MODES = {
    "snapkv_aug_hybrid",
    "epic_leading32_snapkv_aug_hybrid",
    "cacheblend_vdiff_snapkv_aug_hybrid",
}


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


def _has_tool_call_text(text: str) -> bool:
    return "<tool_call>" in (text or "") or "Action:" in (text or "")


def _text_tokens(text: str) -> List[str]:
    return re.findall(r"\w+", _normalize_text(text))


def _text_token_f1(target: str, prediction: str) -> float:
    target_tokens = _text_tokens(target)
    prediction_tokens = _text_tokens(prediction)
    if not target_tokens and not prediction_tokens:
        return 1.0
    if not target_tokens or not prediction_tokens:
        return 0.0
    overlap = sum((Counter(target_tokens) & Counter(prediction_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(left: List[str], right: List[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def _rouge_l_f1(target: str, prediction: str) -> float:
    target_tokens = _text_tokens(target)
    prediction_tokens = _text_tokens(prediction)
    if not target_tokens and not prediction_tokens:
        return 1.0
    if not target_tokens or not prediction_tokens:
        return 0.0
    overlap = _lcs_length(target_tokens, prediction_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def _target_has_tool_call(row: Dict[str, Any]) -> bool:
    if "target_has_tool_call" in row:
        return bool(row.get("target_has_tool_call"))
    return bool(row.get("target_tool_name")) or _has_tool_call_text(row.get("target", ""))


def _row_text_token_f1(row: Dict[str, Any]) -> float:
    if "text_token_f1" in row:
        return float(row.get("text_token_f1") or 0.0)
    return _text_token_f1(row.get("target", ""), row.get("prediction", ""))


def _row_rouge_l_f1(row: Dict[str, Any]) -> float:
    if "rouge_l_f1" in row:
        return float(row.get("rouge_l_f1") or 0.0)
    return _rouge_l_f1(row.get("target", ""), row.get("prediction", ""))


def _row_compressed_tool_tokens(row: Dict[str, Any]) -> float:
    if "top_doc_tokens" in row or "rest_gist_tokens" in row:
        return float(row.get("top_doc_tokens", 0) or 0) + float(row.get("rest_gist_tokens", 0) or 0)
    if "gist_tokens" in row:
        return float(row.get("gist_tokens", 0) or 0)
    if "kept_tool_tokens" in row:
        return float(row.get("kept_tool_tokens", 0) or 0)
    ratio = float(row.get("actual_compression_ratio", 0.0) or 0.0)
    doc_tokens = float(row.get("doc_tokens", 0) or 0)
    return doc_tokens / ratio if ratio > 0 and doc_tokens > 0 else 0.0


def _augment_text_overlap_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    if row.get("skipped") or "target" not in row or "prediction" not in row:
        return row
    target = row.get("target", "")
    prediction = row.get("prediction", "")
    target_has_tool_call = _target_has_tool_call(row)
    prediction_has_tool_call = bool(row.get("has_tool_call", _has_tool_call_text(prediction)))
    text_token_f1 = _text_token_f1(target, prediction)
    rouge_l_f1 = _rouge_l_f1(target, prediction)
    row["target_has_tool_call"] = target_has_tool_call
    row["has_tool_call"] = prediction_has_tool_call
    row["response_type_match"] = target_has_tool_call == prediction_has_tool_call
    row["text_token_f1"] = round(text_token_f1, 4)
    row["rouge_l_f1"] = round(rouge_l_f1, 4)
    row["non_tool_exact_match"] = (not target_has_tool_call) and bool(row.get("exact_match"))
    row["non_tool_text_token_f1"] = round(text_token_f1, 4) if not target_has_tool_call else None
    row["non_tool_rouge_l_f1"] = round(rouge_l_f1, 4) if not target_has_tool_call else None
    return row


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
    if mode == "snapkv_hybrid":
        return None, "snapkv"
    if mode == "epic_leading32_snapkv_hybrid":
        return "leading-32", "snapkv"
    if mode == "cacheblend_vdiff_snapkv_hybrid":
        return f"vdiff-{cacheblend_ratio}", "snapkv"
    if mode == "snapkv_aug_hybrid":
        return None, "snapkv"
    if mode == "epic_leading32_snapkv_aug_hybrid":
        return "leading-32", "snapkv"
    if mode == "cacheblend_vdiff_snapkv_aug_hybrid":
        return f"vdiff-{cacheblend_ratio}", "snapkv"
    raise ValueError(f"Unknown reuse mode: {mode}")


def _merge_prefix_caches(
    evaluator: LLMInference,
    *parts: BatchedKVInstance,
    prefix_logical_length: Optional[int] = None,
) -> BatchedKVInstance:
    valid_parts = [part for part in parts if part is not None]
    if len(valid_parts) == 1:
        return valid_parts[0]
    if not valid_parts:
        raise ValueError("At least one cache is required.")
    prefix = valid_parts[0]
    merged_ids = [torch.cat([ids for part in valid_parts for ids in part.input_ids], dim=0)]
    merged_kv = prefix.past_key_values
    cumulative_logical_length = prefix_logical_length
    for part in valid_parts[1:]:
        merged_kv = evaluator._merge_kv_caches(  # noqa: SLF001 - reuse pipeline helper keeps RoPE positions aligned.
            merged_kv,
            part.past_key_values,
            part.original_lengths,
            system_logical_length=cumulative_logical_length,
        )
        merged_kv = tuple((([key[0]], [value[0]])) for key, value in merged_kv)
        if cumulative_logical_length is not None:
            cumulative_logical_length += _kv_original_token_count(part)
    return BatchedKVInstance(
        input_ids=merged_ids,
        past_key_values=merged_kv,
        original_lengths=[cumulative_logical_length or int(merged_ids[0].shape[0])],
    )


def _tool_prefill_text(title: str, definition: str) -> str:
    return f"{title}:\n{definition}"


def _truncate_debug_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...<truncated {len(text) - max_chars} chars>"


def _debug_tool_name(tool: Dict[str, Any]) -> str:
    function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
    return str(
        function.get("name")
        or tool.get("name")
        or tool.get("tool_name")
        or tool.get("function_name")
        or ""
    )


def _debug_tool_items(definition: str) -> List[Dict[str, Any]]:
    try:
        parsed = json.loads(definition)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    return []


def _add_hybrid_debug_fields(
    row: Dict[str, Any],
    args: argparse.Namespace,
    *,
    full_definition: str,
    top_definition: str,
    rest_definition: str,
    numerator_tokens: int,
    denominator_tokens: int,
    top_tokens: int,
    rest_original_tokens: int,
    rest_compressed_tokens: int,
) -> Dict[str, Any]:
    if not (
        getattr(args, "debug_hybrid_tokens", False)
        or getattr(args, "dump_hybrid_definitions", False)
    ):
        return row
    top_debug_tools = _debug_tool_items(top_definition)
    rest_debug_tools = _debug_tool_items(rest_definition)
    row["debug_top_tool_count_from_definition"] = len(top_debug_tools)
    row["debug_rest_tool_count_from_definition"] = len(rest_debug_tools)
    row["debug_top_tool_names_from_definition"] = [_debug_tool_name(tool) for tool in top_debug_tools]
    row["debug_rest_tool_names_from_definition"] = [_debug_tool_name(tool) for tool in rest_debug_tools]
    if getattr(args, "dump_hybrid_definitions", False):
        max_chars = int(getattr(args, "debug_definition_chars", 4000))
        row["debug_full_tool_definition"] = _truncate_debug_text(full_definition, max_chars)
        row["debug_top_tool_definition"] = _truncate_debug_text(top_definition, max_chars)
        row["debug_rest_tool_definition"] = _truncate_debug_text(rest_definition, max_chars)
    row["hybrid_debug_log"] = "\n".join([
        f"num_tools: {row.get('num_tools')}",
        f"num_top_tools: {row.get('num_top_tools')}",
        f"num_rest_tools: {row.get('num_rest_tools')}",
        f"top_tool_names: {row.get('top_tool_names')}",
        f"top tokens: {row.get('top_doc_tokens')}",
        f"rest tokens: {row.get('rest_doc_tokens')}",
        f"debug top chars: {len(row.get('debug_top_tool_definition', ''))}",
        f"debug full chars: {len(row.get('debug_full_tool_definition', ''))}",
    ])
    return row


def _kv_input_token_count(cache: Optional[BatchedKVInstance]) -> int:
    if cache is None:
        return 0
    return sum(int(ids.shape[0]) for ids in cache.input_ids)


def _kv_original_token_count(cache: Optional[BatchedKVInstance]) -> int:
    if cache is None:
        return 0
    original_lengths = cache.original_lengths
    if isinstance(original_lengths, int):
        return int(original_lengths)
    return sum(int(length) for length in original_lengths)


def _model_rope_params(model: Any) -> tuple[float, str]:
    config = getattr(model, "config", None) or getattr(getattr(model, "model", None), "config", None)
    rope_params = getattr(config, "rope_parameters", None) or {}
    rope_theta = rope_params.get("rope_theta", getattr(config, "rope_theta", 10000.0))
    rope_type = rope_params.get("rope_type", "default")
    return float(rope_theta), str(rope_type)


def _append_full_cache_to_prefix(
    model: Any,
    prefix_cache: Any,
    full_cache: Any,
    logical_start: int,
) -> Any:
    rope_theta, rope_type = _model_rope_params(model)
    for prefix_layer, full_layer in zip(prefix_cache.layers, full_cache.layers):
        full_keys = rotate_k_cache_rope(
            full_layer.keys[0],
            logical_start,
            rope_theta,
            rope_type,
        ).unsqueeze(0)
        prefix_layer.keys = torch.cat([prefix_layer.keys, full_keys], dim=-2)
        prefix_layer.values = torch.cat([prefix_layer.values, full_layer.values], dim=-2)
    return prefix_cache


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
    logical_past_length = _kv_original_token_count(system_cache) + _kv_original_token_count(context_cache)

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
        logical_past_length=logical_past_length,
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


@torch.inference_mode()
def _generate_one_reuse_hybrid(
    evaluator: LLMInference,
    tokenizer: Any,
    example: Any,
    args: argparse.Namespace,
    mode: str,
) -> Dict[str, Any]:
    total_start = time.perf_counter()
    ratio = args.override_ratio
    tools = _as_tool_list(example.tool_definition)
    if not tools:
        return {
            "qid": example.qid,
            "session_id": example.session_id,
            "mode": mode,
            "ratio": ratio,
            "top_k": args.hybrid_top_k,
            "skipped": True,
            "skip_reason": "no_parseable_tools",
        }

    query = _query_text(example.input_messages, args.router_scope)
    if args.router_strategy == "random":
        top_tools, rest_tools, top_tool_names = _split_random_topk_tools(
            tools,
            args.hybrid_top_k,
            seed_text=example.qid,
            seed=args.router_seed,
        )
    else:
        top_tools, rest_tools, top_tool_names = _split_topk_tools(tools, query, args.hybrid_top_k)

    target = example.answer.strip()
    target_tool = _extract_tool_name(target)
    router_hit = target_tool in set(top_tool_names) if target_tool else False
    query_text = _render_query_prompt(tokenizer, example.input_messages)
    prompt_tokens = len(tokenizer.encode(query_text, add_special_tokens=False))
    if args.max_prompt_tokens and prompt_tokens > args.max_prompt_tokens:
        return {
            "qid": example.qid,
            "session_id": example.session_id,
            "mode": mode,
            "ratio": ratio,
            "top_k": args.hybrid_top_k,
            "skipped": True,
            "skip_reason": f"prompt_tokens>{args.max_prompt_tokens}",
            "prompt_tokens": prompt_tokens,
            "target_tool_name": target_tool,
            "router_hit": router_hit,
        }
    if args.router_hit_filter == "hit" and not router_hit:
        return {
            "qid": example.qid,
            "session_id": example.session_id,
            "mode": mode,
            "ratio": ratio,
            "top_k": args.hybrid_top_k,
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
            "mode": mode,
            "ratio": ratio,
            "top_k": args.hybrid_top_k,
            "skipped": True,
            "skip_reason": "router_hit_filtered",
            "num_tools": len(tools),
            "top_tool_names": top_tool_names,
            "target_tool_name": target_tool,
            "router_hit": router_hit,
        }

    full_doc_tokens = len(_tool_doc_ids(tokenizer, example.tool_definition))
    if full_doc_tokens > args.max_tool_definition_tokens:
        return {
            "qid": example.qid,
            "session_id": example.session_id,
            "mode": mode,
            "ratio": ratio,
            "top_k": args.hybrid_top_k,
            "skipped": True,
            "skip_reason": f"tool_definition_tokens>{args.max_tool_definition_tokens}",
            "doc_tokens": full_doc_tokens,
        }

    top_definition = _render_tool_definition(top_tools)
    rest_definition = _render_tool_definition(rest_tools)
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

    top_cache = None
    top_prefill_sec = 0.0
    top_doc_tokens = 0
    if top_tools:
        top_text = _tool_prefill_text("Top-k tool definitions", top_definition)
        _sync_device(device)
        start = time.perf_counter()
        top_cache = evaluator.get_prefill_kv_cache(
            [top_text],
            keep_bos=False,
            role="user",
            compress_method=None,
        )
        _sync_device(device)
        top_prefill_sec = time.perf_counter() - start
        top_doc_tokens = sum(len(ids) for ids in top_cache.input_ids)

    prefix_cache = _merge_prefix_caches(evaluator, system_cache, top_cache) if top_cache is not None else system_cache
    logical_past_length = _kv_original_token_count(prefix_cache)
    if top_cache is not None:
        system_cache = None
        top_cache = None
        _clear_device_cache(device)

    rest_cache = None
    rest_prefill_sec = 0.0
    blend_sec = 0.0
    rest_doc_tokens = 0
    rest_compressed_tokens = 0
    if rest_tools:
        rest_text = _tool_prefill_text("Rest tool definitions", rest_definition)
        rest_doc_tokens = len(_tool_doc_ids(tokenizer, rest_definition))
        if args.max_baseline_input_tokens is not None and rest_doc_tokens > args.max_baseline_input_tokens:
            return {
                "qid": example.qid,
                "session_id": example.session_id,
                "mode": mode,
                "ratio": ratio,
                "top_k": args.hybrid_top_k,
                "skipped": True,
                "skip_reason": f"rest_tool_definition_tokens>{args.max_baseline_input_tokens}",
                "doc_tokens": full_doc_tokens,
                "top_doc_tokens": top_doc_tokens,
                "rest_doc_tokens": rest_doc_tokens,
            }
        _sync_device(device)
        start = time.perf_counter()
        rest_cache = evaluator.get_prefill_kv_cache(
            [rest_text],
            keep_bos=False,
            role="user",
            compress_method=compress_method,
        )
        _sync_device(device)
        rest_prefill_sec = time.perf_counter() - start
        rest_doc_tokens = int(sum(rest_cache.original_lengths))
        rest_compressed_tokens = sum(len(ids) for ids in rest_cache.input_ids)
        logical_past_length += rest_doc_tokens

        decode_tokens = _kv_input_token_count(prefix_cache) + rest_compressed_tokens + prompt_tokens
        if args.max_hybrid_decode_tokens and decode_tokens > args.max_hybrid_decode_tokens:
            return {
                "qid": example.qid,
                "session_id": example.session_id,
                "mode": mode,
                "ratio": ratio,
                "top_k": args.hybrid_top_k,
                "skipped": True,
                "skip_reason": f"hybrid_decode_tokens>{args.max_hybrid_decode_tokens}",
                "doc_tokens": full_doc_tokens,
                "top_doc_tokens": top_doc_tokens,
                "rest_doc_tokens": rest_doc_tokens,
                "rest_gist_tokens": rest_compressed_tokens,
                "prompt_tokens": prompt_tokens,
                "decode_tokens": decode_tokens,
                "target_tool_name": target_tool,
                "router_hit": router_hit,
            }

        if recompute_type is not None:
            _sync_device(device)
            start = time.perf_counter()
            prefix_cache = evaluator.selective_recompute(
                prefix_cache,
                rest_cache,
                recompute_type,
                discard_kv=True,
            )
            rest_cache = None
            _sync_device(device)
            blend_sec = time.perf_counter() - start
            _clear_device_cache(device)
        else:
            _sync_device(device)
            start = time.perf_counter()
            prefix_cache = _merge_prefix_caches(evaluator, prefix_cache, rest_cache)
            rest_cache = None
            _sync_device(device)
            blend_sec = time.perf_counter() - start
            _clear_device_cache(device)

    decode_tokens = _kv_input_token_count(prefix_cache) + prompt_tokens
    if args.max_hybrid_decode_tokens and decode_tokens > args.max_hybrid_decode_tokens:
        return {
            "qid": example.qid,
            "session_id": example.session_id,
            "mode": mode,
            "ratio": ratio,
            "top_k": args.hybrid_top_k,
            "skipped": True,
            "skip_reason": f"hybrid_decode_tokens>{args.max_hybrid_decode_tokens}",
            "doc_tokens": full_doc_tokens,
            "top_doc_tokens": top_doc_tokens,
            "rest_doc_tokens": rest_doc_tokens,
            "rest_gist_tokens": rest_compressed_tokens,
            "prompt_tokens": prompt_tokens,
            "decode_tokens": decode_tokens,
            "target_tool_name": target_tool,
            "router_hit": router_hit,
        }

    _sync_device(device)
    start = time.perf_counter()
    prediction = evaluator.decode_with_past_kv(
        system_prompt_kv=prefix_cache,
        precomputed_kv=rest_cache,
        query_text=query_text,
        max_new_tokens=args.max_new_tokens,
        role=None,
        logical_past_length=logical_past_length,
    )
    _sync_device(device)
    generate_sec = time.perf_counter() - start

    pred_tool = _extract_tool_name(prediction)
    generated_tokens = len(tokenizer.encode(prediction, add_special_tokens=False))
    compressed_tool_tokens = top_doc_tokens + rest_compressed_tokens
    actual_ratio = full_doc_tokens / compressed_tool_tokens if compressed_tool_tokens else 1.0
    rest_actual_ratio = rest_doc_tokens / rest_compressed_tokens if rest_compressed_tokens else 0.0
    ttft_sec = system_prefill_sec + top_prefill_sec + rest_prefill_sec + blend_sec
    row = {
        "qid": example.qid,
        "session_id": example.session_id,
        "mode": mode,
        "ratio": ratio,
        "top_k": args.hybrid_top_k,
        "router_strategy": args.router_strategy,
        "router_scope": args.router_scope,
        "skipped": False,
        "num_tools": len(tools),
        "num_top_tools": len(top_tools),
        "num_rest_tools": len(rest_tools),
        "top_tool_names": top_tool_names,
        "router_hit": router_hit,
        "doc_tokens": full_doc_tokens,
        "top_doc_tokens": top_doc_tokens,
        "rest_doc_tokens": rest_doc_tokens,
        "rest_gist_tokens": rest_compressed_tokens,
        "actual_compression_ratio": round(actual_ratio, 4),
        "rest_actual_compression_ratio": round(rest_actual_ratio, 4),
        "prompt_tokens": prompt_tokens,
        "target_tokens": len(tokenizer.encode(target, add_special_tokens=False)),
        "generated_tokens": generated_tokens,
        "latency_sec": round(generate_sec, 4),
        "system_prefill_sec": round(system_prefill_sec, 4),
        "top_full_prefill_sec": round(top_prefill_sec, 4),
        "tool_compress_sec": round(rest_prefill_sec, 4),
        "full_prefill_sec": round(top_prefill_sec, 4),
        "blend_sec": round(blend_sec, 4),
        "generate_sec": round(generate_sec, 4),
        "ttft_sec": round(ttft_sec, 4),
        "online_ttft_sec": round(ttft_sec, 4),
        "cached_ttft_sec": round(system_prefill_sec + top_prefill_sec + blend_sec, 4),
        "tool_only_cached_ttft_sec": round(top_prefill_sec + blend_sec, 4),
        "tbt_sec": round(generate_sec / generated_tokens, 6) if generated_tokens else 0.0,
        "total_sec": round(time.perf_counter() - total_start, 4),
        "cached_total_sec": round(system_prefill_sec + top_prefill_sec + blend_sec + generate_sec, 4),
        "target_tool_name": target_tool,
        "prediction_tool_name": pred_tool,
        "tool_name_match": target_tool is not None and target_tool == pred_tool,
        "has_tool_call": "<tool_call>" in prediction or "Action:" in prediction,
        "exact_match": _normalize_text(prediction) == _normalize_text(target),
        "prediction": prediction,
        "target": target,
    }
    return _add_hybrid_debug_fields(
        row,
        args,
        full_definition=example.tool_definition,
        top_definition=top_definition,
        rest_definition=rest_definition,
        numerator_tokens=full_doc_tokens,
        denominator_tokens=compressed_tool_tokens,
        top_tokens=top_doc_tokens,
        rest_original_tokens=rest_doc_tokens,
        rest_compressed_tokens=rest_compressed_tokens,
    )


@torch.inference_mode()
def _generate_one_reuse_aug_hybrid(
    evaluator: LLMInference,
    tokenizer: Any,
    example: Any,
    args: argparse.Namespace,
    mode: str,
) -> Dict[str, Any]:
    """Compress all tools with the baseline method, then append top-k full tools."""
    total_start = time.perf_counter()
    ratio = args.override_ratio
    tools = _as_tool_list(example.tool_definition)
    if not tools:
        return {
            "qid": example.qid,
            "session_id": example.session_id,
            "mode": mode,
            "ratio": ratio,
            "top_k": args.hybrid_top_k,
            "skipped": True,
            "skip_reason": "no_parseable_tools",
        }

    query = _query_text(example.input_messages, args.router_scope)
    if args.router_strategy == "random":
        top_tools, _, top_tool_names = _split_random_topk_tools(
            tools,
            args.hybrid_top_k,
            seed_text=example.qid,
            seed=args.router_seed,
        )
    else:
        top_tools, _, top_tool_names = _split_topk_tools(tools, query, args.hybrid_top_k)

    target = example.answer.strip()
    target_tool = _extract_tool_name(target)
    router_hit = target_tool in set(top_tool_names) if target_tool else False
    query_text = _render_query_prompt(tokenizer, example.input_messages)
    prompt_tokens = len(tokenizer.encode(query_text, add_special_tokens=False))
    if args.max_prompt_tokens and prompt_tokens > args.max_prompt_tokens:
        return {
            "qid": example.qid,
            "session_id": example.session_id,
            "mode": mode,
            "ratio": ratio,
            "top_k": args.hybrid_top_k,
            "skipped": True,
            "skip_reason": f"prompt_tokens>{args.max_prompt_tokens}",
            "prompt_tokens": prompt_tokens,
            "target_tool_name": target_tool,
            "router_hit": router_hit,
        }
    if args.router_hit_filter == "hit" and not router_hit:
        return {
            "qid": example.qid,
            "session_id": example.session_id,
            "mode": mode,
            "ratio": ratio,
            "top_k": args.hybrid_top_k,
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
            "mode": mode,
            "ratio": ratio,
            "top_k": args.hybrid_top_k,
            "skipped": True,
            "skip_reason": "router_hit_filtered",
            "num_tools": len(tools),
            "top_tool_names": top_tool_names,
            "target_tool_name": target_tool,
            "router_hit": router_hit,
        }

    full_doc_tokens = len(_tool_doc_ids(tokenizer, example.tool_definition))
    if full_doc_tokens > args.max_tool_definition_tokens:
        return {
            "qid": example.qid,
            "session_id": example.session_id,
            "mode": mode,
            "ratio": ratio,
            "top_k": args.hybrid_top_k,
            "skipped": True,
            "skip_reason": f"tool_definition_tokens>{args.max_tool_definition_tokens}",
            "doc_tokens": full_doc_tokens,
        }
    if args.max_baseline_input_tokens is not None and full_doc_tokens > args.max_baseline_input_tokens:
        return {
            "qid": example.qid,
            "session_id": example.session_id,
            "mode": mode,
            "ratio": ratio,
            "top_k": args.hybrid_top_k,
            "skipped": True,
            "skip_reason": f"tool_definition_tokens>{args.max_baseline_input_tokens}",
            "doc_tokens": full_doc_tokens,
            "prompt_tokens": prompt_tokens,
            "target_tool_name": target_tool,
            "router_hit": router_hit,
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
    compressed_cache = evaluator.get_prefill_kv_cache(
        _reuse_documents(example.tool_definition),
        keep_bos=False,
        role="user",
        compress_method=compress_method,
    )
    _sync_device(device)
    compressed_prefill_sec = time.perf_counter() - start
    compressed_doc_tokens = int(sum(compressed_cache.original_lengths))
    compressed_tokens = sum(len(ids) for ids in compressed_cache.input_ids)
    logical_past_length = _kv_original_token_count(system_cache) + compressed_doc_tokens

    blend_sec = 0.0
    if recompute_type is not None:
        _sync_device(device)
        start = time.perf_counter()
        prefix_cache = evaluator.selective_recompute(
            system_cache,
            compressed_cache,
            recompute_type,
            discard_kv=True,
        )
        system_cache = None
        compressed_cache = None
        _sync_device(device)
        blend_sec = time.perf_counter() - start
        _clear_device_cache(device)
    else:
        _sync_device(device)
        start = time.perf_counter()
        prefix_cache = _merge_prefix_caches(evaluator, system_cache, compressed_cache)
        system_cache = None
        compressed_cache = None
        _sync_device(device)
        blend_sec = time.perf_counter() - start
        _clear_device_cache(device)

    top_prefill_sec = 0.0
    top_doc_tokens = 0
    if top_tools:
        top_text = _tool_prefill_text("Top-k tool definitions", _render_tool_definition(top_tools))
        _sync_device(device)
        start = time.perf_counter()
        top_cache = evaluator.get_prefill_kv_cache(
            [top_text],
            keep_bos=False,
            role="user",
            compress_method=None,
        )
        _sync_device(device)
        top_prefill_sec = time.perf_counter() - start
        top_doc_tokens = sum(len(ids) for ids in top_cache.input_ids)
        _sync_device(device)
        start = time.perf_counter()
        prefix_cache = _merge_prefix_caches(
            evaluator,
            prefix_cache,
            top_cache,
            prefix_logical_length=logical_past_length,
        )
        logical_past_length += top_doc_tokens
        top_cache = None
        _sync_device(device)
        blend_sec += time.perf_counter() - start
        _clear_device_cache(device)

    decode_tokens = _kv_input_token_count(prefix_cache) + prompt_tokens
    if args.max_hybrid_decode_tokens and decode_tokens > args.max_hybrid_decode_tokens:
        return {
            "qid": example.qid,
            "session_id": example.session_id,
            "mode": mode,
            "ratio": ratio,
            "top_k": args.hybrid_top_k,
            "skipped": True,
            "skip_reason": f"hybrid_decode_tokens>{args.max_hybrid_decode_tokens}",
            "doc_tokens": full_doc_tokens,
            "top_doc_tokens": top_doc_tokens,
            "rest_doc_tokens": compressed_doc_tokens,
            "rest_gist_tokens": compressed_tokens,
            "prompt_tokens": prompt_tokens,
            "decode_tokens": decode_tokens,
            "target_tool_name": target_tool,
            "router_hit": router_hit,
        }

    _sync_device(device)
    start = time.perf_counter()
    prediction = evaluator.decode_with_past_kv(
        system_prompt_kv=prefix_cache,
        precomputed_kv=None,
        query_text=query_text,
        max_new_tokens=args.max_new_tokens,
        role=None,
        logical_past_length=logical_past_length,
    )
    _sync_device(device)
    generate_sec = time.perf_counter() - start

    pred_tool = _extract_tool_name(prediction)
    generated_tokens = len(tokenizer.encode(prediction, add_special_tokens=False))
    compressed_tool_tokens = compressed_tokens + top_doc_tokens
    actual_ratio = full_doc_tokens / compressed_tool_tokens if compressed_tool_tokens else 1.0
    compressed_actual_ratio = compressed_doc_tokens / compressed_tokens if compressed_tokens else 0.0
    ttft_sec = system_prefill_sec + compressed_prefill_sec + top_prefill_sec + blend_sec
    return {
        "qid": example.qid,
        "session_id": example.session_id,
        "mode": mode,
        "ratio": ratio,
        "top_k": args.hybrid_top_k,
        "router_strategy": args.router_strategy,
        "router_scope": args.router_scope,
        "hybrid_layout": "all_compressed_plus_topk_full",
        "skipped": False,
        "num_tools": len(tools),
        "num_top_tools": len(top_tools),
        "num_rest_tools": len(tools),
        "top_tool_names": top_tool_names,
        "router_hit": router_hit,
        "doc_tokens": full_doc_tokens,
        "top_doc_tokens": top_doc_tokens,
        "rest_doc_tokens": compressed_doc_tokens,
        "rest_gist_tokens": compressed_tokens,
        "actual_compression_ratio": round(actual_ratio, 4),
        "rest_actual_compression_ratio": round(compressed_actual_ratio, 4),
        "prompt_tokens": prompt_tokens,
        "decode_tokens": decode_tokens,
        "target_tokens": len(tokenizer.encode(target, add_special_tokens=False)),
        "generated_tokens": generated_tokens,
        "latency_sec": round(generate_sec, 4),
        "system_prefill_sec": round(system_prefill_sec, 4),
        "top_full_prefill_sec": round(top_prefill_sec, 4),
        "tool_compress_sec": round(compressed_prefill_sec, 4),
        "full_prefill_sec": round(top_prefill_sec, 4),
        "blend_sec": round(blend_sec, 4),
        "generate_sec": round(generate_sec, 4),
        "ttft_sec": round(ttft_sec, 4),
        "online_ttft_sec": round(ttft_sec, 4),
        "cached_ttft_sec": round(system_prefill_sec + top_prefill_sec + blend_sec, 4),
        "tool_only_cached_ttft_sec": round(top_prefill_sec + blend_sec, 4),
        "tbt_sec": round(generate_sec / generated_tokens, 6) if generated_tokens else 0.0,
        "total_sec": round(time.perf_counter() - total_start, 4),
        "cached_total_sec": round(system_prefill_sec + top_prefill_sec + blend_sec + generate_sec, 4),
        "target_tool_name": target_tool,
        "prediction_tool_name": pred_tool,
        "tool_name_match": target_tool is not None and target_tool == pred_tool,
        "has_tool_call": "<tool_call>" in prediction or "Action:" in prediction,
        "exact_match": _normalize_text(prediction) == _normalize_text(target),
        "prediction": prediction,
        "target": target,
    }


@torch.inference_mode()
def _generate_one_c2kv_aug_hybrid(
    model: Any,
    tokenizer: Any,
    example: Any,
    args: argparse.Namespace,
    device: str,
) -> Dict[str, Any]:
    """Compress all tools with C2KV, then append an extra full top-k tool prefix."""
    total_start = time.perf_counter()
    ratio = args.override_ratio
    tools = _as_tool_list(example.tool_definition)
    if not tools:
        return {
            "qid": example.qid,
            "session_id": example.session_id,
            "mode": "c2kv_aug_hybrid",
            "ratio": ratio,
            "top_k": args.hybrid_top_k,
            "skipped": True,
            "skip_reason": "no_parseable_tools",
        }

    query = _query_text(example.input_messages, args.router_scope)
    if args.router_strategy == "random":
        top_tools, _, top_tool_names = _split_random_topk_tools(
            tools,
            args.hybrid_top_k,
            seed_text=example.qid,
            seed=args.router_seed,
        )
    else:
        top_tools, _, top_tool_names = _split_topk_tools(tools, query, args.hybrid_top_k)

    target = example.answer.strip()
    target_tool = _extract_tool_name(target)
    router_hit = target_tool in set(top_tool_names) if target_tool else False
    query_text = _render_query_prompt(tokenizer, example.input_messages)
    prompt_tokens = len(tokenizer.encode(query_text, add_special_tokens=False))
    if args.max_prompt_tokens and prompt_tokens > args.max_prompt_tokens:
        return {
            "qid": example.qid,
            "session_id": example.session_id,
            "mode": "c2kv_aug_hybrid",
            "ratio": ratio,
            "top_k": args.hybrid_top_k,
            "skipped": True,
            "skip_reason": f"prompt_tokens>{args.max_prompt_tokens}",
            "prompt_tokens": prompt_tokens,
            "target_tool_name": target_tool,
            "router_hit": router_hit,
        }
    if args.router_hit_filter == "hit" and not router_hit:
        return {
            "qid": example.qid,
            "session_id": example.session_id,
            "mode": "c2kv_aug_hybrid",
            "ratio": ratio,
            "top_k": args.hybrid_top_k,
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
            "mode": "c2kv_aug_hybrid",
            "ratio": ratio,
            "top_k": args.hybrid_top_k,
            "skipped": True,
            "skip_reason": "router_hit_filtered",
            "num_tools": len(tools),
            "top_tool_names": top_tool_names,
            "target_tool_name": target_tool,
            "router_hit": router_hit,
        }

    full_doc_tokens = len(_tool_doc_ids(tokenizer, example.tool_definition))
    if full_doc_tokens > args.max_tool_definition_tokens:
        return {
            "qid": example.qid,
            "session_id": example.session_id,
            "mode": "c2kv_aug_hybrid",
            "ratio": ratio,
            "top_k": args.hybrid_top_k,
            "skipped": True,
            "skip_reason": f"tool_definition_tokens>{args.max_tool_definition_tokens}",
            "doc_tokens": full_doc_tokens,
        }

    system_ids = _chat_template_ids(
        tokenizer,
        [{"role": "system", "content": example.system_prompt}],
        keep_bos=True,
        max_length=args.max_system_length,
    )
    system_input_ids = torch.tensor([system_ids], dtype=torch.long, device=model.device)
    system_cache, system_length, system_prefill_sec = _prefill_system(
        model,
        system_input_ids,
        args.system_attn_impl,
    )

    context_input_ids, doc_tokens, doc_chunks, skip_reason = _build_tool_chunks(
        tokenizer,
        example.tool_definition,
        max_doc_length=args.max_doc_length,
        max_doc_num=args.max_doc_num,
        max_tool_definition_tokens=args.max_tool_definition_tokens,
        truncate_tool_definition=args.truncate_tool_definition,
        document_mode=args.tool_document_eval_mode,
    )
    if context_input_ids is None:
        return {
            "qid": example.qid,
            "session_id": example.session_id,
            "mode": "c2kv_aug_hybrid",
            "ratio": ratio,
            "top_k": args.hybrid_top_k,
            "skipped": True,
            "skip_reason": skip_reason,
            "doc_tokens": doc_tokens,
        }

    (
        prefix_cache,
        tool_length,
        gist_tokens,
        compressed_actual_ratio,
        tool_compress_sec,
        blend_sec,
    ) = _build_tool_cache(
        model,
        context_input_ids,
        system_cache,
        system_length,
        args.gist_attn_impl,
        ratio,
    )

    top_prefill_sec = 0.0
    top_doc_tokens = 0
    if top_tools:
        top_definition = _render_tool_definition(top_tools)
        top_ids = _chat_template_ids(
            tokenizer,
            [{"role": "user", "content": "Top-k tool definitions:\n" + top_definition}],
        )
        top_doc_tokens = len(top_ids)
        top_input_ids = torch.tensor([top_ids], dtype=torch.long, device=model.device)
        top_cache, _, top_prefill_sec = _prefill_tokens_with_cache(
            model,
            top_input_ids,
            past_key_values=None,
            past_length=0,
            attn_impl=args.generate_attn_impl,
        )
        _sync_device(device)
        append_start = time.perf_counter()
        prefix_cache = _append_full_cache_to_prefix(
            model,
            prefix_cache,
            top_cache,
            logical_start=system_length + tool_length,
        )
        _sync_device(device)
        blend_sec += time.perf_counter() - append_start
        top_cache = None
        _clear_device_cache(device)

    logical_past_length = system_length + tool_length + top_doc_tokens
    compressed_tool_tokens = gist_tokens + top_doc_tokens
    decode_tokens = prefix_cache.get_seq_length() + prompt_tokens
    if args.max_hybrid_decode_tokens and decode_tokens > args.max_hybrid_decode_tokens:
        return {
            "qid": example.qid,
            "session_id": example.session_id,
            "mode": "c2kv_aug_hybrid",
            "ratio": ratio,
            "top_k": args.hybrid_top_k,
            "skipped": True,
            "skip_reason": f"hybrid_decode_tokens>{args.max_hybrid_decode_tokens}",
            "doc_tokens": full_doc_tokens,
            "top_doc_tokens": top_doc_tokens,
            "rest_doc_tokens": doc_tokens,
            "rest_gist_tokens": gist_tokens,
            "prompt_tokens": prompt_tokens,
            "decode_tokens": decode_tokens,
            "target_tool_name": target_tool,
            "router_hit": router_hit,
        }

    prompt_ids = _chat_template_ids(tokenizer, example.input_messages, add_generation_prompt=True)
    if args.max_prompt_tokens and len(prompt_ids) > args.max_prompt_tokens:
        prompt_ids = prompt_ids[-args.max_prompt_tokens :]
    prompt_input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
    cache_length = prefix_cache.get_seq_length()
    mock_cache_ids = prompt_input_ids.new_zeros((1, cache_length))
    input_ids = torch.cat([mock_cache_ids, prompt_input_ids], dim=1)
    position_ids = torch.arange(
        logical_past_length,
        logical_past_length + prompt_input_ids.shape[1],
        dtype=torch.long,
        device=model.device,
    ).unsqueeze(0)

    _sync_device(device)
    start = time.perf_counter()
    prediction, latency, generated_tokens, tbt_sec = _generate_from_input_ids(
        model,
        tokenizer,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        attn_impl=args.generate_attn_impl,
        use_gist=True,
        position_ids=position_ids,
        past_key_values=prefix_cache,
    )
    _sync_device(device)
    generate_sec = time.perf_counter() - start
    pred_tool = _extract_tool_name(prediction)
    actual_ratio = full_doc_tokens / compressed_tool_tokens if compressed_tool_tokens else 0.0
    ttft_sec = system_prefill_sec + tool_compress_sec + top_prefill_sec + blend_sec
    return {
        "qid": example.qid,
        "session_id": example.session_id,
        "mode": "c2kv_aug_hybrid",
        "ratio": ratio,
        "top_k": args.hybrid_top_k,
        "router_strategy": args.router_strategy,
        "router_scope": args.router_scope,
        "hybrid_layout": "all_c2kv_compressed_plus_topk_full",
        "skipped": False,
        "num_tools": len(tools),
        "num_top_tools": len(top_tools),
        "num_rest_tools": len(tools),
        "top_tool_names": top_tool_names,
        "router_hit": router_hit,
        "doc_tokens": full_doc_tokens,
        "doc_chunks": doc_chunks,
        "top_doc_tokens": top_doc_tokens,
        "rest_doc_tokens": doc_tokens,
        "rest_gist_tokens": gist_tokens,
        "gist_tokens": gist_tokens,
        "actual_compression_ratio": round(actual_ratio, 4),
        "rest_actual_compression_ratio": round(compressed_actual_ratio, 4),
        "prompt_tokens": len(prompt_ids),
        "decode_tokens": decode_tokens,
        "target_tokens": len(tokenizer.encode(target, add_special_tokens=False)),
        "generated_tokens": generated_tokens,
        "latency_sec": round(latency, 4),
        "system_prefill_sec": round(system_prefill_sec, 4),
        "top_full_prefill_sec": round(top_prefill_sec, 4),
        "tool_compress_sec": round(tool_compress_sec, 4),
        "full_prefill_sec": round(top_prefill_sec, 4),
        "blend_sec": round(blend_sec, 4),
        "generate_sec": round(generate_sec, 4),
        "ttft_sec": round(ttft_sec, 4),
        "online_ttft_sec": round(ttft_sec, 4),
        "cached_ttft_sec": round(system_prefill_sec + top_prefill_sec + blend_sec, 4),
        "tool_only_cached_ttft_sec": round(top_prefill_sec + blend_sec, 4),
        "tbt_sec": round(tbt_sec, 6),
        "total_sec": round(time.perf_counter() - total_start, 4),
        "cached_total_sec": round(system_prefill_sec + top_prefill_sec + blend_sec + generate_sec, 4),
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
        called_rows = [row for row in valid_rows if row.get("has_tool_call")]
        tool_targets = [row for row in valid_rows if _target_has_tool_call(row)]
        non_tool_targets = [row for row in valid_rows if not _target_has_tool_call(row)]
        compressed_tool_total = sum(_row_compressed_tool_tokens(row) for row in valid_rows)
        no_rest_rows = [row for row in valid_rows if row.get("num_rest_tools") == 0]
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
            "num_tool_targets": len(tool_targets),
            "num_non_tool_targets": len(non_tool_targets),
            "exact_match": (
                sum(1 for row in valid_rows if row.get("exact_match")) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_text_token_f1": (
                sum(_row_text_token_f1(row) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_rouge_l_f1": (
                sum(_row_rouge_l_f1(row) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "response_type_accuracy": (
                sum(
                    1 for row in valid_rows
                    if row.get("response_type_match", _target_has_tool_call(row) == bool(row.get("has_tool_call")))
                ) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "target_tool_call_rate": (
                len(tool_targets) / len(valid_rows) if valid_rows else 0.0
            ),
            "tool_name_accuracy": (
                sum(1 for row in valid_rows if row.get("tool_name_match")) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "tool_name_accuracy_on_tool_targets": (
                sum(1 for row in tool_targets if row.get("tool_name_match")) / len(tool_targets)
                if tool_targets else 0.0
            ),
            "tool_call_rate": (
                len(called_rows) / len(valid_rows) if valid_rows else 0.0
            ),
            "tool_call_rate_on_tool_targets": (
                sum(1 for row in tool_targets if row.get("has_tool_call")) / len(tool_targets)
                if tool_targets else 0.0
            ),
            "call_accuracy": (
                sum(1 for row in called_rows if row.get("tool_name_match")) / len(called_rows)
                if called_rows else 0.0
            ),
            "non_tool_exact_match": (
                sum(1 for row in non_tool_targets if row.get("exact_match")) / len(non_tool_targets)
                if non_tool_targets else 0.0
            ),
            "non_tool_text_token_f1": (
                sum(_row_text_token_f1(row) for row in non_tool_targets) / len(non_tool_targets)
                if non_tool_targets else 0.0
            ),
            "non_tool_rouge_l_f1": (
                sum(_row_rouge_l_f1(row) for row in non_tool_targets) / len(non_tool_targets)
                if non_tool_targets else 0.0
            ),
            "non_tool_false_tool_call_rate": (
                sum(1 for row in non_tool_targets if row.get("has_tool_call")) / len(non_tool_targets)
                if non_tool_targets else 0.0
            ),
            "router_hit_rate": (
                sum(1 for row in valid_rows if row.get("router_hit")) / len(valid_rows)
                if any("router_hit" in row for row in valid_rows) else 0.0
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
            "token_weighted_actual_compression_ratio": (
                sum(row.get("doc_tokens", 0) for row in valid_rows) / compressed_tool_total
                if compressed_tool_total else 0.0
            ),
            "avg_num_tools": (
                sum(row.get("num_tools", 0) for row in valid_rows) / len(valid_rows)
                if valid_rows and any("num_tools" in row for row in valid_rows) else 0.0
            ),
            "num_no_rest_tools": len(no_rest_rows),
            "no_rest_tool_rate": (
                len(no_rest_rows) / len(valid_rows)
                if valid_rows and any("num_rest_tools" in row for row in valid_rows) else 0.0
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
        num_tools = len(_as_tool_list(example.tool_definition))
        if args.min_num_tools > 0 and num_tools < args.min_num_tools:
            selection_skips[f"num_tools<{args.min_num_tools}"] += 1
            continue
        if args.selection_filter == "c2kv":
            _, _, _, skip_reason = _build_tool_chunks(
                tokenizer,
                example.tool_definition,
                max_doc_length=args.max_doc_length,
                max_doc_num=args.max_doc_num,
                max_tool_definition_tokens=args.max_tool_definition_tokens,
                truncate_tool_definition=args.truncate_tool_definition,
                document_mode=args.tool_document_eval_mode,
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
    if args.max_baseline_input_tokens is not None and args.max_baseline_input_tokens <= 0:
        args.max_baseline_input_tokens = None

    requested_modes = [item.strip() for item in (args.compare_modes or args.mode).split(",") if item.strip()]
    modes = [MODE_ALIASES.get(mode, mode) for mode in requested_modes]
    ratios = [int(item.strip()) for item in (args.ratios or str(args.override_ratio)).split(",") if item.strip()]
    rows: List[Dict[str, Any]] = []

    for mode in modes:
        if mode not in AGENT_MODES and mode not in REUSE_MODES:
            raise ValueError(f"Unknown mode: {mode}")
        if mode in {"full", "reuse", "epic_leading32", "cacheblend_vdiff"}:
            run_ratios = [1]
        elif mode in {"snapkv_reuse", "epic_leading32_snapkv", "cacheblend_vdiff_snapkv"} | REUSE_HYBRID_MODES | REUSE_AUG_HYBRID_MODES:
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
                        if mode in REUSE_AUG_HYBRID_MODES:
                            row = _generate_one_reuse_aug_hybrid(evaluator, tokenizer, example, run_args, mode)
                        elif mode in REUSE_HYBRID_MODES:
                            row = _generate_one_reuse_hybrid(evaluator, tokenizer, example, run_args, mode)
                        else:
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
                    row = _augment_text_overlap_metrics(row)
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
                    elif mode == "c2kv_aug_hybrid":
                        row = _generate_one_c2kv_aug_hybrid(
                            model,
                            tokenizer,
                            example,
                            run_args,
                            device,
                        )
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
                row = _augment_text_overlap_metrics(row)
                rows.append(row)
        del model
        _clear_device_cache(device)

    summary = {
        "model": args.model,
        "base_model": args.base_model,
        "reuse_model": args.reuse_model or args.base_model or args.model,
        "dataset_path": args.dataset_path,
        "split": args.split,
        "tool_document_eval_mode": args.tool_document_eval_mode,
        "modes": modes,
        "ratios": ratios,
        "selection_skips": selection_skips,
        "notes": {
            "epic_leading32": "PyTorch selective recompute with recompute_type=leading-32.",
            "cacheblend_vdiff": f"PyTorch value-difference selective recompute with recompute_type=vdiff-{args.cacheblend_recompute_ratio}; not the vLLM+LMCache expr_cacheblend.py path.",
            "snapkv_reuse": "Uses reuse_pipeline SnapKV compression, currently hard-coded to roughly 4x in compress_kv.",
            "epic_leading32_snapkv": "EPIC leading-32 selective recompute on top of SnapKV-compressed document KV.",
            "cacheblend_vdiff_snapkv": f"Value-difference selective recompute on top of SnapKV-compressed document KV with recompute_type=vdiff-{args.cacheblend_recompute_ratio}.",
            "snapkv_hybrid": f"Hybrid top-{args.hybrid_top_k} full tool schemas plus SnapKV-compressed rest tool schemas.",
            "epic_leading32_snapkv_hybrid": f"Hybrid top-{args.hybrid_top_k} full tool schemas plus EPIC leading-32 selective recompute on SnapKV-compressed rest schemas.",
            "cacheblend_vdiff_snapkv_hybrid": f"Hybrid top-{args.hybrid_top_k} full tool schemas plus value-difference selective recompute on SnapKV-compressed rest schemas.",
            "c2kv_aug_hybrid": f"All tool schemas C2KV-compressed plus an extra full top-{args.hybrid_top_k} tool-schema prefix.",
            "snapkv_aug_hybrid": f"All tool schemas SnapKV-compressed plus an extra full top-{args.hybrid_top_k} tool-schema prefix.",
            "epic_leading32_snapkv_aug_hybrid": f"All tool schemas SnapKV-compressed with EPIC leading-32 selective recompute plus an extra full top-{args.hybrid_top_k} tool-schema prefix.",
            "cacheblend_vdiff_snapkv_aug_hybrid": f"All tool schemas SnapKV-compressed with value-difference selective recompute plus an extra full top-{args.hybrid_top_k} tool-schema prefix.",
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
            "snapkv_hybrid",
            "epic_leading32_snapkv_hybrid",
            "cacheblend_vdiff_snapkv_hybrid",
            "snapkv_aug_hybrid",
            "epic_leading32_snapkv_aug_hybrid",
            "cacheblend_vdiff_snapkv_aug_hybrid",
            "c2kv",
            "c2kv_untrained",
            "hybrid",
            "c2kv_aug_hybrid",
        ],
        default="c2kv",
    )
    parser.add_argument(
        "--compare_modes",
        default="full,snapkv_reuse,epic_leading32_snapkv,cacheblend_vdiff_snapkv,c2kv,hybrid,snapkv_hybrid,epic_leading32_snapkv_hybrid,cacheblend_vdiff_snapkv_hybrid",
        help="Comma-separated modes.",
    )
    parser.add_argument("--ratios", default="4", help="Ratios for truncate/c2kv/snapkv_reuse/hybrid.")
    parser.add_argument("--override_ratio", type=int, default=4)
    parser.add_argument("--hybrid_top_k", type=int, default=3)
    parser.add_argument("--router_scope", choices=["last_user", "all"], default="last_user")
    parser.add_argument("--router_strategy", choices=["lexical", "random"], default="lexical")
    parser.add_argument("--router_seed", type=int, default=42)
    parser.add_argument("--router_hit_filter", choices=["all", "hit", "miss"], default="all")
    parser.add_argument(
        "--debug_hybrid_tokens",
        action="store_true",
        help="Write hybrid compression numerator/denominator token breakdowns into each row.",
    )
    parser.add_argument(
        "--dump_hybrid_definitions",
        action="store_true",
        help="Also write selected/full tool definitions into each debug row. Use with small max_examples.",
    )
    parser.add_argument("--debug_definition_chars", type=int, default=4000)
    parser.add_argument("--cacheblend_recompute_ratio", type=float, default=0.15)
    parser.add_argument("--max_examples", type=int, default=0, help="Maximum examples; <=0 means all selected examples.")
    parser.add_argument("--max_source_examples", type=int)
    parser.add_argument("--selection_filter", choices=["c2kv", "none"], default="c2kv")
    parser.add_argument(
        "--tool_document_eval_mode",
        choices=["full", "per_tool"],
        default="full",
        help=(
            "How to build C2KV eval documents from tool schemas. full keeps one "
            "combined tool-definition document; per_tool makes each tool schema an "
            "independent C2KV document."
        ),
    )
    parser.add_argument("--min_num_tools", type=int, default=0)
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
    parser.add_argument(
        "--max_hybrid_decode_tokens",
        type=int,
        default=0,
        help=(
            "Optional safety cap for reuse-hybrid decode length "
            "(merged prefix/cache input ids + query prompt). <=0 disables it."
        ),
    )
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
