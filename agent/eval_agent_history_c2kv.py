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
from typing import Any, Dict, List, Optional, Sequence

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
from train.train_data_multiturn import (  # noqa: E402
    AgentLLMTracesCompressHistorySource,
    CompressHistoryExample,
    _chat_template_ids,
    _fit_reused_history,
    _normal_chat_message,
    _pad,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)


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


def _jsonl_write(path: str, rows: List[Dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _is_oom_error(error: RuntimeError) -> bool:
    text = str(error).lower()
    return "out of memory" in text or "oom" in text


def _oom_row(example: CompressHistoryExample, mode: str, ratio: int) -> Dict[str, Any]:
    return {
        "qid": example.qid,
        "session_id": example.qid.rsplit(":", 1)[0] if ":" in example.qid else None,
        "mode": mode,
        "ratio": ratio,
        "skipped": True,
        "skip_reason": "oom",
    }


def _has_tool_call(text: str) -> bool:
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


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
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


def _history_messages(
    tokenizer: Any,
    example: CompressHistoryExample,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    raw_history = [
        _normal_chat_message(message)
        for message in example.history_messages
        if message.get("content")
    ]
    return _fit_reused_history(
        tokenizer,
        raw_history,
        max_doc_length=args.max_doc_length,
        max_doc_num=args.max_doc_num,
        policy=args.history_selection,
    )


def _current_messages(example: CompressHistoryExample) -> List[Dict[str, Any]]:
    return [
        _normal_chat_message(message)
        for message in example.current_messages
        if message.get("content") or message.get("role") == "assistant"
    ]


def _history_doc_ids(tokenizer: Any, messages: Sequence[Dict[str, Any]]) -> List[List[int]]:
    return [_chat_template_ids(tokenizer, [message]) for message in messages]


def _build_history_chunks(
    tokenizer: Any,
    example: CompressHistoryExample,
    args: argparse.Namespace,
) -> tuple[Optional[torch.Tensor], int, int, List[Dict[str, Any]], Optional[str]]:
    history = _history_messages(tokenizer, example, args)
    if len(history) < args.min_doc_num:
        return None, 0, len(history), history, f"history_docs<{args.min_doc_num}"
    rows = []
    total_tokens = 0
    for message in history:
        doc_ids = _chat_template_ids(tokenizer, [message], max_length=args.max_doc_length)
        total_tokens += len(doc_ids)
        rows.append(_pad(doc_ids, args.max_doc_length, -100))
    if total_tokens > args.max_history_tokens:
        return None, total_tokens, len(history), history, f"history_tokens>{args.max_history_tokens}"
    if len(rows) > args.max_doc_num:
        return None, total_tokens, len(history), history, f"history_docs>{args.max_doc_num}"
    empty_docs = args.max_doc_num - len(rows)
    rows.extend([[-100] * args.max_doc_length for _ in range(empty_docs)])
    return torch.tensor(rows, dtype=torch.long), total_tokens, len(history), history, None


def _truncate_history_ids(
    tokenizer: Any,
    history: Sequence[Dict[str, Any]],
    ratio: int,
    policy: str,
) -> tuple[List[int], int, int]:
    doc_ids = _history_doc_ids(tokenizer, history)
    total_tokens = sum(len(ids) for ids in doc_ids)
    keep_tokens = max(1, (total_tokens + ratio - 1) // ratio) if total_tokens else 0
    if keep_tokens >= total_tokens:
        return [token for ids in doc_ids for token in ids], total_tokens, total_tokens

    selected: List[int] = []
    remaining = keep_tokens
    ordered = list(enumerate(doc_ids))
    if policy == "tail":
        ordered = list(reversed(ordered))
    for _, ids in ordered:
        if remaining <= 0:
            break
        take = min(remaining, len(ids))
        if policy == "tail":
            selected = ids[-take:] + selected
        else:
            selected.extend(ids[:take])
        remaining -= take
    return selected, total_tokens, len(selected)


def _target_metrics(tokenizer: Any, target: str, prediction: str) -> Dict[str, Any]:
    target = target.strip()
    target_tool = _extract_tool_name(target)
    pred_tool = _extract_tool_name(prediction)
    target_has_tool_call = _has_tool_call(target)
    prediction_has_tool_call = _has_tool_call(prediction)
    exact_match = _normalize_text(prediction) == _normalize_text(target)
    text_token_f1 = _text_token_f1(target, prediction)
    rouge_l_f1 = _rouge_l_f1(target, prediction)
    return {
        "target_tokens": len(tokenizer.encode(target, add_special_tokens=False)),
        "generated_tokens": len(tokenizer.encode(prediction, add_special_tokens=False)),
        "target_has_tool_call": target_has_tool_call,
        "target_tool_name": target_tool,
        "prediction_tool_name": pred_tool,
        "tool_name_match": target_tool is not None and target_tool == pred_tool,
        "has_tool_call": prediction_has_tool_call,
        "response_type_match": target_has_tool_call == prediction_has_tool_call,
        "exact_match": exact_match,
        "text_token_f1": round(text_token_f1, 4),
        "rouge_l_f1": round(rouge_l_f1, 4),
        "non_tool_exact_match": (not target_has_tool_call) and exact_match,
        "non_tool_text_token_f1": round(text_token_f1, 4) if not target_has_tool_call else None,
        "non_tool_rouge_l_f1": round(rouge_l_f1, 4) if not target_has_tool_call else None,
        "prediction": prediction,
        "target": target,
    }


@torch.inference_mode()
def _generate_with_prefix(
    model: Any,
    tokenizer: Any,
    example: CompressHistoryExample,
    prefix: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    prompt_ids = _chat_template_ids(tokenizer, _current_messages(example), add_generation_prompt=True)
    if args.max_prompt_tokens and len(prompt_ids) > args.max_prompt_tokens:
        prompt_ids = prompt_ids[-args.max_prompt_tokens :]
    prompt_input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
    cache_length = prefix["cache"].get_seq_length()
    mock_cache_ids = prompt_input_ids.new_zeros((1, cache_length))
    input_ids = torch.cat([mock_cache_ids, prompt_input_ids], dim=1)
    original_prefix_length = prefix["system_length"] + prefix["history_length"]
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
        use_gist=prefix.get("use_gist", False),
        position_ids=position_ids,
        past_key_values=prefix["cache"],
    )
    metrics = _target_metrics(tokenizer, example.answer, prediction)
    metrics["generated_tokens"] = generated_tokens
    metrics.update({
        "prompt_tokens": len(prompt_ids),
        "latency_sec": round(generate_sec, 4),
        "generate_sec": round(generate_sec, 4),
        "tbt_sec": round(tbt_sec, 6),
    })
    return metrics


@torch.inference_mode()
def _build_full_or_truncate_prefix(
    model: Any,
    tokenizer: Any,
    example: CompressHistoryExample,
    args: argparse.Namespace,
    mode: str,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    history = _history_messages(tokenizer, example, args)
    if len(history) < args.min_doc_num:
        return None, f"history_docs<{args.min_doc_num}"

    system_ids = _chat_template_ids(
        tokenizer,
        [{"role": "system", "content": example.system_prompt}],
        tools=example.tools or None,
        keep_bos=True,
        max_length=args.max_system_length,
    )
    if mode == "truncate":
        history_ids, doc_tokens, kept_tokens = _truncate_history_ids(
            tokenizer, history, args.override_ratio, args.truncate_selection
        )
    else:
        history_ids = [token for ids in _history_doc_ids(tokenizer, history) for token in ids]
        doc_tokens = len(history_ids)
        kept_tokens = doc_tokens
    if doc_tokens > args.max_history_tokens:
        return None, f"history_tokens>{args.max_history_tokens}"

    prompt_ids = _chat_template_ids(tokenizer, _current_messages(example), add_generation_prompt=True)
    if args.max_prompt_tokens and len(prompt_ids) > args.max_prompt_tokens:
        prompt_ids = prompt_ids[-args.max_prompt_tokens :]
    total_len = len(system_ids) + len(history_ids) + len(prompt_ids)
    if args.max_baseline_input_tokens and total_len > args.max_baseline_input_tokens:
        return None, f"baseline_input_tokens>{args.max_baseline_input_tokens}"

    system_input_ids = torch.tensor([system_ids], dtype=torch.long, device=model.device)
    system_cache, system_length, system_prefill_sec = _prefill_system(
        model, system_input_ids, args.system_attn_impl
    )
    history_input_ids = torch.tensor([history_ids], dtype=torch.long, device=model.device)
    history_cache, history_length, full_prefill_sec = _prefill_tokens_with_cache(
        model,
        history_input_ids,
        past_key_values=system_cache,
        past_length=system_length,
        attn_impl=args.generate_attn_impl,
    )
    return {
        "cache": history_cache,
        "system_length": system_length,
        "history_length": history_length,
        "cache_length": history_cache.get_seq_length(),
        "doc_tokens": doc_tokens,
        "doc_chunks": len(history),
        "kept_history_tokens": kept_tokens,
        "gist_tokens": 0,
        "actual_compression_ratio": doc_tokens / kept_tokens if kept_tokens else 0.0,
        "system_prefill_sec": system_prefill_sec,
        "full_prefill_sec": full_prefill_sec,
        "tool_compress_sec": 0.0,
        "blend_sec": 0.0,
        "use_gist": False,
    }, None


@torch.inference_mode()
def _build_c2kv_prefix(
    model: Any,
    tokenizer: Any,
    example: CompressHistoryExample,
    args: argparse.Namespace,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    context_input_ids, doc_tokens, doc_chunks, _, skip_reason = _build_history_chunks(
        tokenizer, example, args
    )
    if context_input_ids is None:
        return None, skip_reason

    system_ids = _chat_template_ids(
        tokenizer,
        [{"role": "system", "content": example.system_prompt}],
        tools=example.tools or None,
        keep_bos=True,
        max_length=args.max_system_length,
    )
    system_input_ids = torch.tensor([system_ids], dtype=torch.long, device=model.device)
    system_cache, system_length, system_prefill_sec = _prefill_system(
        model, system_input_ids, args.system_attn_impl
    )
    (
        history_cache,
        history_length,
        gist_tokens,
        actual_ratio,
        compress_sec,
        blend_sec,
    ) = _build_tool_cache(
        model,
        context_input_ids,
        system_cache,
        system_length,
        args.gist_attn_impl,
        args.override_ratio,
    )
    return {
        "cache": history_cache,
        "system_length": system_length,
        "history_length": history_length,
        "cache_length": history_cache.get_seq_length(),
        "doc_tokens": doc_tokens,
        "doc_chunks": doc_chunks,
        "kept_history_tokens": doc_tokens,
        "gist_tokens": gist_tokens,
        "actual_compression_ratio": actual_ratio,
        "system_prefill_sec": system_prefill_sec,
        "full_prefill_sec": 0.0,
        "tool_compress_sec": compress_sec,
        "blend_sec": blend_sec,
        "use_gist": True,
    }, None


@torch.inference_mode()
def _build_hybrid_prefix(
    model: Any,
    tokenizer: Any,
    example: CompressHistoryExample,
    args: argparse.Namespace,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    history = _history_messages(tokenizer, example, args)
    if len(history) < args.min_doc_num:
        return None, f"history_docs<{args.min_doc_num}"
    full_history = history[-args.hybrid_top_k :] if args.history_selection == "tail" else history[: args.hybrid_top_k]
    full_set = set(range(len(history) - len(full_history), len(history))) if args.history_selection == "tail" else set(range(len(full_history)))
    rest_history = [message for index, message in enumerate(history) if index not in full_set]

    system_ids = _chat_template_ids(
        tokenizer,
        [{"role": "system", "content": example.system_prompt}],
        tools=example.tools or None,
        keep_bos=True,
        max_length=args.max_system_length,
    )
    system_input_ids = torch.tensor([system_ids], dtype=torch.long, device=model.device)
    system_cache, system_length, system_prefill_sec = _prefill_system(
        model, system_input_ids, args.system_attn_impl
    )

    full_ids = [token for ids in _history_doc_ids(tokenizer, full_history) for token in ids]
    full_tokens = len(full_ids)
    top_prefill_sec = 0.0
    prefix_cache = system_cache
    if full_ids:
        full_input_ids = torch.tensor([full_ids], dtype=torch.long, device=model.device)
        prefix_cache, full_length, top_prefill_sec = _prefill_tokens_with_cache(
            model,
            full_input_ids,
            past_key_values=prefix_cache,
            past_length=system_length,
            attn_impl=args.generate_attn_impl,
        )
    else:
        full_length = 0

    rest_tokens = 0
    rest_length = 0
    gist_tokens = 0
    compress_sec = 0.0
    blend_sec = 0.0
    if rest_history:
        rows = []
        for message in rest_history:
            ids = _chat_template_ids(tokenizer, [message], max_length=args.max_doc_length)
            rest_tokens += len(ids)
            rows.append(_pad(ids, args.max_doc_length, -100))
        rows.extend([[-100] * args.max_doc_length for _ in range(args.max_doc_num - len(rest_history))])
        context_input_ids = torch.tensor(rows, dtype=torch.long)
        (
            prefix_cache,
            rest_length,
            gist_tokens,
            _,
            compress_sec,
            blend_sec,
        ) = _build_tool_cache(
            model,
            context_input_ids,
            prefix_cache,
            system_length + full_length,
            args.gist_attn_impl,
            args.override_ratio,
        )

    doc_tokens = rest_tokens + full_tokens
    compressed_tokens = gist_tokens + full_tokens
    return {
        "cache": prefix_cache,
        "system_length": system_length,
        "history_length": rest_length + full_length,
        "cache_length": prefix_cache.get_seq_length(),
        "doc_tokens": doc_tokens,
        "doc_chunks": len(history),
        "full_history_docs": len(full_history),
        "rest_history_docs": len(rest_history),
        "top_full_tokens": full_tokens,
        "rest_history_tokens": rest_tokens,
        "kept_history_tokens": full_tokens,
        "gist_tokens": gist_tokens,
        "actual_compression_ratio": doc_tokens / compressed_tokens if compressed_tokens else 0.0,
        "system_prefill_sec": system_prefill_sec,
        "full_prefill_sec": top_prefill_sec,
        "tool_compress_sec": compress_sec,
        "blend_sec": blend_sec,
        "use_gist": bool(rest_history),
    }, None


@torch.inference_mode()
def _generate_one(
    model: Any,
    tokenizer: Any,
    example: CompressHistoryExample,
    args: argparse.Namespace,
    mode: str,
) -> Dict[str, Any]:
    total_start = time.perf_counter()
    if mode in ("full", "truncate"):
        prefix, skip_reason = _build_full_or_truncate_prefix(model, tokenizer, example, args, mode)
    elif mode == "c2kv":
        prefix, skip_reason = _build_c2kv_prefix(model, tokenizer, example, args)
    elif mode == "hybrid":
        prefix, skip_reason = _build_hybrid_prefix(model, tokenizer, example, args)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    if prefix is None:
        return {
            "qid": example.qid,
            "session_id": example.qid.rsplit(":", 1)[0] if ":" in example.qid else None,
            "mode": mode,
            "ratio": args.override_ratio,
            "skipped": True,
            "skip_reason": skip_reason,
        }
    row = _generate_with_prefix(model, tokenizer, example, prefix, args)
    ttft_sec = (
        prefix.get("system_prefill_sec", 0.0)
        + prefix.get("full_prefill_sec", 0.0)
        + prefix.get("tool_compress_sec", 0.0)
        + prefix.get("blend_sec", 0.0)
    )
    row.update({
        "qid": example.qid,
        "session_id": example.qid.rsplit(":", 1)[0] if ":" in example.qid else None,
        "mode": mode,
        "ratio": args.override_ratio,
        "hybrid_top_k": args.hybrid_top_k if mode == "hybrid" else None,
        "history_selection": args.history_selection,
        "skipped": False,
        "doc_tokens": prefix.get("doc_tokens", 0),
        "doc_chunks": prefix.get("doc_chunks", 0),
        "kept_history_tokens": prefix.get("kept_history_tokens", 0),
        "gist_tokens": prefix.get("gist_tokens", 0),
        "actual_compression_ratio": round(prefix.get("actual_compression_ratio", 0.0), 4),
        "system_prefill_sec": round(prefix.get("system_prefill_sec", 0.0), 4),
        "full_prefill_sec": round(prefix.get("full_prefill_sec", 0.0), 4),
        "tool_compress_sec": round(prefix.get("tool_compress_sec", 0.0), 4),
        "blend_sec": round(prefix.get("blend_sec", 0.0), 4),
        "ttft_sec": round(ttft_sec, 4),
        "total_sec": round(time.perf_counter() - total_start, 4),
    })
    for key in ("full_history_docs", "rest_history_docs", "top_full_tokens", "rest_history_tokens"):
        if key in prefix:
            row[key] = prefix[key]
    return row


def _summarize_rows(args: argparse.Namespace, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries = []
    keys = sorted({(row.get("mode"), row.get("ratio")) for row in rows})
    for mode, ratio in keys:
        group = [row for row in rows if row.get("mode") == mode and row.get("ratio") == ratio]
        valid_rows = [row for row in group if not row.get("skipped")]
        skip_reasons = Counter(row.get("skip_reason", "unknown") for row in group if row.get("skipped"))
        generated_total = sum(row.get("generated_tokens", 0) for row in valid_rows)
        called = [row for row in valid_rows if row.get("has_tool_call")]
        tool_targets = [
            row for row in valid_rows
            if row.get("target_has_tool_call") or row.get("target_tool_name") is not None
        ]
        non_tool_targets = [
            row for row in valid_rows
            if not row.get("target_has_tool_call") and row.get("target_tool_name") is None
        ]
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
                sum(row.get("text_token_f1", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_rouge_l_f1": (
                sum(row.get("rouge_l_f1", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "response_type_accuracy": (
                sum(1 for row in valid_rows if row.get("response_type_match")) / len(valid_rows)
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
                len(called) / len(valid_rows) if valid_rows else 0.0
            ),
            "tool_call_rate_on_tool_targets": (
                sum(1 for row in tool_targets if row.get("has_tool_call")) / len(tool_targets)
                if tool_targets else 0.0
            ),
            "call_accuracy": (
                sum(1 for row in called if row.get("tool_name_match")) / len(called)
                if called else 0.0
            ),
            "non_tool_exact_match": (
                sum(1 for row in non_tool_targets if row.get("exact_match")) / len(non_tool_targets)
                if non_tool_targets else 0.0
            ),
            "non_tool_text_token_f1": (
                sum(row.get("text_token_f1", 0.0) for row in non_tool_targets) / len(non_tool_targets)
                if non_tool_targets else 0.0
            ),
            "non_tool_rouge_l_f1": (
                sum(row.get("rouge_l_f1", 0.0) for row in non_tool_targets) / len(non_tool_targets)
                if non_tool_targets else 0.0
            ),
            "non_tool_false_tool_call_rate": (
                sum(1 for row in non_tool_targets if row.get("has_tool_call")) / len(non_tool_targets)
                if non_tool_targets else 0.0
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
            "avg_history_compress_sec": (
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


def _load_examples(args: argparse.Namespace, tokenizer: Any) -> tuple[List[CompressHistoryExample], Dict[str, int]]:
    source = AgentLLMTracesCompressHistorySource(
        args.dataset_path,
        split=args.split,
        eval_ratio=args.eval_ratio,
        split_seed=args.split_seed,
        split_manifest_file=args.split_manifest_file,
        split_manifest_name=args.split_manifest_name,
        max_samples_per_session=args.max_samples_per_session,
        max_records=args.max_source_examples,
        require_tool_call=args.require_tool_call,
        max_input_chars=args.max_input_chars,
        max_answer_chars=args.max_answer_chars,
        include_tools=args.include_tools,
    )
    selection_skips: Counter[str] = Counter()
    examples = []
    for example in source:
        if args.selection_filter == "c2kv":
            _, _, _, _, skip_reason = _build_history_chunks(tokenizer, example, args)
            if skip_reason is not None:
                selection_skips[skip_reason] += 1
                continue
        examples.append(example)
        if args.max_examples and len(examples) >= args.max_examples:
            break
    return examples, dict(selection_skips)


def _load_tokenizer(args: argparse.Namespace) -> Any:
    candidates = []
    for path in (args.tokenizer, args.base_model, args.model):
        if path and path not in candidates:
            candidates.append(path)
    errors = []
    for path in candidates:
        for use_fast in (True, False):
            try:
                logger.info("Loading tokenizer from %s use_fast=%s", path, use_fast)
                return AutoTokenizer.from_pretrained(
                    path,
                    trust_remote_code=True,
                    local_files_only=True,
                    padding_side="right",
                    use_fast=use_fast,
                )
            except Exception as error:
                errors.append(f"{path} use_fast={use_fast}: {type(error).__name__}: {error}")
    raise RuntimeError(
        "Failed to load tokenizer from all candidate paths. "
        "Install sentencepiece/tiktoken if the local tokenizer files require conversion.\n"
        + "\n".join(errors)
    )


def _checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else -1


def _has_loadable_config(path: Path) -> bool:
    config_path = path / "config.json"
    if not config_path.exists():
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(config.get("model_type"))


def _resolve_model_checkpoint(model_path: str) -> str:
    path = Path(model_path)
    if _has_loadable_config(path):
        return model_path
    if not path.is_dir():
        return model_path
    checkpoints = sorted(
        [item for item in path.iterdir() if item.is_dir() and _checkpoint_step(item) >= 0],
        key=_checkpoint_step,
    )
    for checkpoint in reversed(checkpoints):
        if _has_loadable_config(checkpoint):
            logger.info("Resolved model path %s to latest checkpoint %s", model_path, checkpoint)
            return str(checkpoint)
    return model_path


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    device = _setup_device(args.device_type)
    args.model = _resolve_model_checkpoint(args.model)
    tokenizer = _load_tokenizer(args)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    examples, selection_skips = _load_examples(args, tokenizer)
    logger.info("Selected %d examples; selection_skips=%s", len(examples), selection_skips)
    modes = [item.strip() for item in (args.compare_modes or args.mode).split(",") if item.strip()]
    ratios = [int(item.strip()) for item in (args.ratios or str(args.override_ratio)).split(",") if item.strip()]
    rows: List[Dict[str, Any]] = []

    for mode in modes:
        run_ratios = [1] if mode == "full" else ratios
        model_args = copy.copy(args)
        model_args.mode = "c2kv" if mode == "hybrid" else mode
        if mode in {"full", "truncate"} and args.base_model:
            model_args.model = args.base_model
        logger.info("Loading model for mode=%s model=%s", mode, model_args.model)
        model = _load_model(model_args, tokenizer, device)
        for ratio in run_ratios:
            run_args = copy.copy(model_args)
            run_args.override_ratio = ratio
            desc = f"{mode}@{ratio}x" if mode != "full" else "full"
            for example in tqdm(examples, desc=desc):
                try:
                    row = _generate_one(model, tokenizer, example, run_args, mode)
                except RuntimeError as error:
                    if not _is_oom_error(error):
                        raise
                    logger.warning("Skipping sample after OOM: mode=%s ratio=%s qid=%s", mode, ratio, example.qid)
                    row = _oom_row(example, mode, ratio)
                    _clear_device_cache(device)
                rows.append(row)
                _clear_device_cache(device)
        del model
        _clear_device_cache(device)

    summary = {
        "model": args.model,
        "base_model": args.base_model,
        "dataset_path": args.dataset_path,
        "split": args.split,
        "modes": modes,
        "ratios": ratios,
        "history_selection": args.history_selection,
        "truncate_selection": args.truncate_selection,
        "include_tools": args.include_tools,
        "hybrid_top_k": args.hybrid_top_k,
        "max_doc_length": args.max_doc_length,
        "max_doc_num": args.max_doc_num,
        "max_history_tokens": args.max_history_tokens,
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_system_length": args.max_system_length,
        "max_baseline_input_tokens": args.max_baseline_input_tokens,
        "selection_skips": selection_skips,
        "num_rows": len(rows),
        "results": _summarize_rows(args, rows),
    }
    if args.output_file:
        _jsonl_write(args.output_file, rows)
        summary_path = str(Path(args.output_file).with_suffix(".summary.json"))
        Path(summary_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Wrote predictions to %s", args.output_file)
        logger.info("Wrote summary to %s", summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate C2KV compression for multi-turn agent history.")
    parser.add_argument("--model", required=True, help="History C2KV checkpoint path.")
    parser.add_argument("--base_model", help="Base model path for full/truncate baselines.")
    parser.add_argument("--tokenizer", help="Tokenizer path. Defaults to --model.")
    parser.add_argument("--dataset_path", default="./datasets/agent-llm-traces")
    parser.add_argument("--output_file", default="./outputs/agent_history_c2kv_eval.jsonl")
    parser.add_argument("--split", choices=["train", "eval"], default="eval")
    parser.add_argument("--mode", choices=["full", "truncate", "c2kv", "hybrid"], default="c2kv")
    parser.add_argument("--compare_modes", default="full,truncate,c2kv,hybrid")
    parser.add_argument("--ratios", default="4")
    parser.add_argument("--override_ratio", type=int, default=4)
    parser.add_argument("--hybrid_top_k", type=int, default=3)
    parser.add_argument("--history_selection", choices=["head", "tail"], default="tail")
    parser.add_argument("--truncate_selection", choices=["head", "tail"], default="tail")
    parser.add_argument("--max_examples", type=int, default=100)
    parser.add_argument("--max_source_examples", type=int)
    parser.add_argument("--selection_filter", choices=["c2kv", "none"], default="c2kv")
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--split_manifest_file")
    parser.add_argument("--split_manifest_name", default="subset_disjoint")
    parser.add_argument("--max_samples_per_session", type=int, default=4)
    parser.add_argument("--max_doc_length", type=int, default=768)
    parser.add_argument("--min_doc_num", type=int, default=1)
    parser.add_argument("--max_doc_num", type=int, default=16)
    parser.add_argument("--max_history_tokens", type=int, default=12288)
    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--max_system_length", type=int, default=4096)
    parser.add_argument("--max_prompt_tokens", type=int, default=1536)
    parser.add_argument("--max_baseline_input_tokens", type=int, default=16000)
    parser.add_argument("--min_target_tokens", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--require_tool_call", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--max_input_chars", type=int)
    parser.add_argument("--max_answer_chars", type=int)
    parser.add_argument("--include_tools", type=lambda x: str(x).lower() == "true", default=False)
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
