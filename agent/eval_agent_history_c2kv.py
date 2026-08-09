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
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "inference"))
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
    _agent_message_content_to_text,
    _chat_template_ids,
    _fit_reused_history,
    _normal_agent_message,
    _normal_chat_message,
    _pad,
    _render_agent_output_messages,
)
from rope_reposition import rotate_k_cache_rope  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)

FULL_PROMPT_MODES = {
    "original_replay_full",
    "reconstructed_contiguous_full",
    "raw_first15_full",
    "raw_first15_full_same_model",
    "raw_first8_full",
    "raw_first8_full_same_model",
    "raw_prefix8_exact_full",
    "raw_prefix8_exact_full_same_model",
    "raw_prefix_next_full",
    "raw_prefix_next_full_same_model",
    "current_only",
}
SPLIT_FULL_MODES = {"split_full_kv", "sequential_full_kv"}
C2KV_MODES = {"c2kv", "split_c2kv", "contiguous_history_c2kv"}
DECISION_PREFIX_MODES = {
    "history_full",
    "history_all_c2kv4",
    "recent2_full_rest_c2kv4",
    "each_turn_independent_c2kv4",
    "recompress_all_every_turn_c2kv4",
}
HYBRID_MODES = {
    "hybrid",
    "c2kv_hybrid",
    "att_hybrid",
    "att_fullkv_hybrid",
    "hybrid_fullkv_att_rerank",
    "recent1_hybrid",
    "recent2_hybrid",
}
TURN_ABLATION_MODES = {
    "all_full",
    "all_c2kv",
    "recent2_full_rest_c2kv",
    *{f"turn{index}_full_rest_c2kv" for index in range(1, 14)},
}
RAW_FIRST15_MODES = {
    "raw_first15_full",
    "raw_first15_c2kv",
    "raw_first15_hybrid",
}
RAW_FIRST8_MODES = {
    "raw_first8_full",
    "raw_first8_full_same_model",
    "raw_first8_c2kv",
    "raw_first8_hybrid",
}
RAW_PREFIX8_EXACT_MODES = {
    "raw_prefix8_exact_full",
    "raw_prefix8_exact_full_same_model",
    "raw_prefix8_exact_c2kv",
    "raw_prefix8_exact_hybrid",
}
RAW_PREFIX_NEXT_MODES = {
    "raw_prefix_next_full",
    "raw_prefix_next_full_same_model",
    "raw_prefix_next_c2kv",
    "raw_prefix_next_hybrid",
}
TRUNCATE_MODES = {"truncate", "tail_truncate"}


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
        split_oversized_history_docs=getattr(args, "split_oversized_history_docs", True),
    )


def _current_messages(example: CompressHistoryExample) -> List[Dict[str, Any]]:
    return [
        _normal_chat_message(message)
        for message in example.current_messages
        if message.get("content") or message.get("role") == "assistant"
    ]


def _system_message(example: CompressHistoryExample) -> Dict[str, Any]:
    return {"role": "system", "content": example.system_prompt}


def _raw_template_message(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(message, dict):
        return None
    role = message.get("role") or message.get("type") or "user"
    if role not in {"system", "user", "assistant", "tool"}:
        role = "user"
    item = copy.deepcopy(message)
    item["role"] = role
    if item.get("content") is None:
        item["content"] = _agent_message_content_to_text(message) or ""
    return item


def _raw_history_current_pairs(
    example: CompressHistoryExample,
) -> tuple[List[tuple[Dict[str, Any], Dict[str, Any]]], List[tuple[Dict[str, Any], Dict[str, Any]]]]:
    pairs: List[tuple[Dict[str, Any], Dict[str, Any]]] = []
    for message in example.original_messages or []:
        if not isinstance(message, dict) or message.get("role") == "system":
            continue
        normal = _normal_agent_message(message)
        raw = _raw_template_message(message)
        if normal is None or raw is None or normal.get("role") == "system":
            continue
        pairs.append((raw, normal))
    last_user_index = next(
        (index for index in range(len(pairs) - 1, -1, -1) if pairs[index][0].get("role") == "user"),
        None,
    )
    if last_user_index is None:
        return [], []
    return pairs[:last_user_index], pairs[last_user_index:]


def _raw_docs_from_pairs(
    pairs: Sequence[tuple[Dict[str, Any], Dict[str, Any]]],
) -> List[List[Dict[str, Any]]]:
    docs: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    for raw, normal in pairs:
        if raw.get("role") == "user":
            if current:
                docs.append(current)
            current = [raw]
        elif current:
            current.append(raw)
        else:
            current = [raw]
    if current:
        docs.append(current)
    return docs


def _raw_first15_split(
    example: CompressHistoryExample,
    num_turns: int = 15,
    exact: bool = False,
) -> tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]], Optional[str]]:
    history_pairs, current_pairs = _raw_history_current_pairs(example)
    history_docs = _raw_docs_from_pairs(history_pairs)
    current_messages = [raw for raw, _normal in current_pairs]
    if exact and len(history_docs) != num_turns:
        return [], current_messages, f"raw_history_turn_docs!={num_turns}"
    if len(history_docs) < num_turns:
        return [], current_messages, f"raw_history_turn_docs<{num_turns}"
    if not current_messages:
        return [], current_messages, "empty_raw_current"
    return history_docs[:num_turns], current_messages, None


def _raw_prefix_next_split(
    example: CompressHistoryExample,
    num_turns: int,
    max_answer_chars: Optional[int],
    target_scope: str = "turn",
) -> tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]], Optional[str], bool, Optional[str]]:
    history_pairs, _current_pairs = _raw_history_current_pairs(example)
    all_history_docs = _raw_docs_from_pairs(history_pairs)
    if len(all_history_docs) <= num_turns:
        return [], [], None, False, f"raw_history_turn_docs<={num_turns}"
    next_doc = all_history_docs[num_turns]
    if not next_doc:
        return [], [], None, False, "empty_raw_prefix_next_doc"
    current_message = next_doc[0]
    if current_message.get("role") != "user":
        return [], [], None, False, "raw_prefix_next_current_not_user"
    target_messages = next_doc[1:]
    if target_scope == "first_assistant":
        target_messages = [
            message for message in target_messages if message.get("role") == "assistant"
        ][:1]
        if not target_messages:
            return [], [current_message], None, False, "empty_raw_prefix_next_assistant_target"
    if not target_messages:
        return [], [current_message], None, False, "empty_raw_prefix_next_target_messages"
    target, target_has_tool_call = _render_agent_output_messages(target_messages, max_answer_chars)
    if not target.strip():
        return [], [current_message], None, target_has_tool_call, "empty_raw_prefix_next_target"
    return all_history_docs[:num_turns], [current_message], target, target_has_tool_call, None


def _raw_first_n_turns_for_mode(mode: str, args: argparse.Namespace) -> int:
    if mode.startswith("raw_prefix_next_"):
        return int(getattr(args, "raw_prefix_n_turns", 8))
    if mode.startswith("raw_first8_") or mode.startswith("raw_prefix8_"):
        return 8
    return int(getattr(args, "raw_first_n_turns", 15))


def _raw_exact_for_mode(mode: str) -> bool:
    return mode.startswith("raw_prefix8_exact_")


def _raw_prefix_next_for_mode(mode: str) -> bool:
    return mode.startswith("raw_prefix_next_")


def _raw_split_for_mode(
    example: CompressHistoryExample,
    args: argparse.Namespace,
    mode: str,
) -> tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]], Optional[str], Optional[bool], Optional[str]]:
    num_turns = _raw_first_n_turns_for_mode(mode, args)
    if _raw_prefix_next_for_mode(mode):
        history_docs, current_messages, target, target_has_tool_call, skip_reason = _raw_prefix_next_split(
            example,
            num_turns,
            getattr(args, "max_answer_chars", None),
            getattr(args, "raw_prefix_next_target_scope", "turn"),
        )
        return history_docs, current_messages, target, target_has_tool_call, skip_reason
    history_docs, current_messages, skip_reason = _raw_first15_split(
        example,
        num_turns,
        exact=_raw_exact_for_mode(mode),
    )
    return history_docs, current_messages, None, None, skip_reason


def _truncate_debug_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...<truncated {len(text) - max_chars} chars>"


def _raw_doc_debug_payload(
    history_docs: Sequence[Sequence[Dict[str, Any]]],
    current_messages: Sequence[Dict[str, Any]],
    max_chars: int,
) -> Dict[str, Any]:
    doc_texts = []
    doc_messages = []
    for doc in history_docs:
        doc_text = json.dumps(list(doc), ensure_ascii=False, separators=(",", ":"))
        doc_texts.append(_truncate_debug_text(doc_text, max_chars))
        doc_messages.append([
            {
                key: message.get(key)
                for key in ("role", "content", "tool_call_id", "tool_calls", "name")
                if key in message
            }
            for message in doc
        ])
    current_text = json.dumps(list(current_messages), ensure_ascii=False, separators=(",", ":"))
    return {
        "raw_history_doc_texts": doc_texts,
        "raw_history_doc_messages": doc_messages,
        "raw_current_text": _truncate_debug_text(current_text, max_chars),
    }


def _raw_first15_debug(
    args: argparse.Namespace,
    tokenizer: Any,
    history_docs: Sequence[Sequence[Dict[str, Any]]],
    current_messages: Sequence[Dict[str, Any]],
    doc_ids: Sequence[Sequence[int]],
    num_turns: int = 15,
    window_name: Optional[str] = None,
) -> Dict[str, Any]:
    doc_tokens = [len(ids) for ids in doc_ids]
    debug = {
        "raw_history_source": "original_messages",
        "raw_history_window": window_name or f"first{num_turns}",
        "raw_first_n_turns": num_turns,
        "raw_history_docs": len(history_docs),
        "raw_current_messages": len(current_messages),
        "raw_doc_tokens": doc_tokens,
        "avg_turn_original_tokens": sum(doc_tokens) / len(doc_tokens) if doc_tokens else 0.0,
    }
    if getattr(args, "dump_raw_history_docs", False):
        debug.update(_raw_doc_debug_payload(
            history_docs,
            current_messages,
            int(getattr(args, "raw_history_doc_debug_chars", 2000)),
        ))
    return debug


def _original_replay_messages(example: CompressHistoryExample) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = [_system_message(example)]
    for message in example.original_messages or []:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "system":
            continue
        item = _normal_agent_message(message)
        if item is None:
            continue
        if item.get("role") == "assistant" and not str(item.get("content") or "").strip():
            continue
        messages.append(item)
    return messages if len(messages) > 1 else [_system_message(example), *_history_messages_no_fit(example), *_current_messages(example)]


def _history_messages_no_fit(example: CompressHistoryExample) -> List[Dict[str, Any]]:
    return [
        _normal_chat_message(message)
        for message in example.history_messages
        if message.get("content")
    ]


def _turn_ablation_history(example: CompressHistoryExample, num_turns: int = 15) -> List[Dict[str, Any]]:
    history = _history_messages_no_fit(example)
    if len(history) < num_turns:
        return []
    return history[-num_turns:]


def _turn_ablation_full_indices(mode: str) -> List[int]:
    if mode == "all_full":
        return list(range(15))
    if mode == "all_c2kv":
        return []
    if mode == "recent2_full_rest_c2kv":
        return [13, 14]
    match = re.fullmatch(r"turn(\d+)_full_rest_c2kv", mode)
    if match:
        turn_index = int(match.group(1))
        if 1 <= turn_index <= 13:
            return [turn_index - 1, 13, 14]
    raise ValueError(f"Unknown turn ablation mode: {mode}")


def _turn_ablation_position(mode: str) -> Optional[int]:
    match = re.fullmatch(r"turn(\d+)_full_rest_c2kv", mode)
    return int(match.group(1)) if match else None


def _reconstructed_messages(
    tokenizer: Any,
    example: CompressHistoryExample,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    return [_system_message(example), *_history_messages(tokenizer, example, args), *_current_messages(example)]


def _current_only_messages(example: CompressHistoryExample) -> List[Dict[str, Any]]:
    return [_system_message(example), *_current_messages(example)]


def _prompt_ids_for_mode(
    tokenizer: Any,
    example: CompressHistoryExample,
    args: argparse.Namespace,
    mode: str,
) -> tuple[List[int], Dict[str, Any], Optional[str]]:
    if mode in {
        "raw_first15_full",
        "raw_first15_full_same_model",
        "raw_first8_full",
        "raw_first8_full_same_model",
        "raw_prefix8_exact_full",
        "raw_prefix8_exact_full_same_model",
        "raw_prefix_next_full",
        "raw_prefix_next_full_same_model",
    }:
        raw_first_n_turns = _raw_first_n_turns_for_mode(mode, args)
        history_docs, current_messages, target_override, _target_has_tool_call, skip_reason = _raw_split_for_mode(
            example,
            args,
            mode,
        )
        if skip_reason is not None:
            return [], {}, skip_reason
        doc_ids = [_chat_template_ids(tokenizer, doc) for doc in history_docs]
        messages = [_system_message(example), *[message for doc in history_docs for message in doc], *current_messages]
        prompt_ids = _chat_template_ids(
            tokenizer,
            messages,
            tools=example.tools or None,
            add_generation_prompt=True,
            keep_bos=True,
        )
        total_len = len(prompt_ids)
        if args.max_baseline_input_tokens and total_len > args.max_baseline_input_tokens:
            return prompt_ids, {"prompt_tokens": total_len}, f"baseline_input_tokens>{args.max_baseline_input_tokens}"
        debug = {
            "prompt_tokens": total_len,
            "history_docs": len(history_docs),
            "doc_tokens": sum(len(ids) for ids in doc_ids),
            **_raw_first15_debug(
                args,
                tokenizer,
                history_docs,
                current_messages,
                doc_ids,
                raw_first_n_turns,
                window_name=(
                    f"prefix{raw_first_n_turns}_next"
                    if _raw_prefix_next_for_mode(mode)
                    else None
                ),
            ),
        }
        if target_override is not None:
            debug.update({
                "target_override": target_override,
                "raw_target_source": f"raw_prefix{raw_first_n_turns}_next",
                "raw_target_has_tool_call": _target_has_tool_call,
            })
        return prompt_ids, debug, None

    if mode == "original_replay_full":
        messages = _original_replay_messages(example)
    elif mode == "reconstructed_contiguous_full":
        messages = _reconstructed_messages(tokenizer, example, args)
    elif mode == "current_only":
        messages = _current_only_messages(example)
    else:
        raise ValueError(f"Unknown full prompt mode: {mode}")
    if not messages or not _current_messages(example):
        return [], {}, "empty_prompt"
    prompt_ids = _chat_template_ids(
        tokenizer,
        messages,
        tools=example.tools or None,
        add_generation_prompt=True,
        keep_bos=True,
    )
    total_len = len(prompt_ids)
    if args.max_baseline_input_tokens and total_len > args.max_baseline_input_tokens:
        return prompt_ids, {"prompt_tokens": total_len}, f"baseline_input_tokens>{args.max_baseline_input_tokens}"

    debug: Dict[str, Any] = {
        "prompt_tokens": total_len,
        "history_docs": len(_history_messages(tokenizer, example, args)) if mode != "current_only" else 0,
    }
    if mode == "original_replay_full":
        debug.update({
            "original_raw_message_count": len(example.original_messages or []),
            "original_replay_message_count": len(messages),
            "original_replay_normalized": True,
        })
    if mode == "reconstructed_contiguous_full":
        original_ids = _chat_template_ids(
            tokenizer,
            _original_replay_messages(example),
            tools=example.tools or None,
            add_generation_prompt=True,
            keep_bos=True,
        )
        first_diff = _first_token_diff(original_ids, prompt_ids)
        debug.update({
            "original_prompt_tokens": len(original_ids),
            "reconstructed_prompt_tokens": len(prompt_ids),
            "original_reconstructed_ids_equal": original_ids == prompt_ids,
            "first_token_diff_index": first_diff,
        })
    return prompt_ids, debug, None


@torch.inference_mode()
def _generate_full_prompt(
    model: Any,
    tokenizer: Any,
    example: CompressHistoryExample,
    args: argparse.Namespace,
    mode: str,
) -> Dict[str, Any]:
    total_start = time.perf_counter()
    prompt_ids, debug, skip_reason = _prompt_ids_for_mode(tokenizer, example, args, mode)
    if skip_reason is not None:
        return {
            "qid": example.qid,
            "session_id": example.qid.rsplit(":", 1)[0] if ":" in example.qid else None,
            "mode": mode,
            "ratio": 1,
            "skipped": True,
            "skip_reason": skip_reason,
            **debug,
        }
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
    prediction, generate_sec, generated_tokens, tbt_sec = _generate_from_input_ids(
        model,
        tokenizer,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        attn_impl=args.generate_attn_impl,
    )
    target = debug.get("target_override", example.answer)
    row = _target_metrics(tokenizer, target, prediction)
    row.update({
        "qid": example.qid,
        "session_id": example.qid.rsplit(":", 1)[0] if ":" in example.qid else None,
        "mode": mode,
        "ratio": 1,
        "history_selection": args.history_selection,
        "skipped": False,
        "doc_tokens": max(0, debug.get("prompt_tokens", 0) - len(_chat_template_ids(
            tokenizer,
            _current_only_messages(example),
            tools=example.tools or None,
            add_generation_prompt=True,
            keep_bos=True,
        ))),
        "doc_chunks": debug.get("history_docs", 0),
        "kept_history_tokens": debug.get("prompt_tokens", 0),
        "gist_tokens": 0,
        "actual_compression_ratio": 1.0,
        "prompt_tokens": debug.get("prompt_tokens", len(prompt_ids)),
        "input_tokens": debug.get("prompt_tokens", len(prompt_ids)),
        "compressed_history_tokens": max(0, debug.get("prompt_tokens", 0) - len(_chat_template_ids(
            tokenizer,
            _current_only_messages(example),
            tools=example.tools or None,
            add_generation_prompt=True,
            keep_bos=True,
        ))),
        "cache_tokens": 0,
        "generated_tokens": generated_tokens,
        "latency_sec": round(generate_sec, 4),
        "generate_sec": round(generate_sec, 4),
        "tbt_sec": round(tbt_sec, 6),
        "system_prefill_sec": 0.0,
        "full_prefill_sec": 0.0,
        "tool_compress_sec": 0.0,
        "blend_sec": 0.0,
        "ttft_sec": 0.0,
        "total_sec": round(time.perf_counter() - total_start, 4),
        **debug,
    })
    row.pop("target_override", None)
    return row


def _history_doc_ids(tokenizer: Any, messages: Sequence[Dict[str, Any]]) -> List[List[int]]:
    return [_chat_template_ids(tokenizer, [message]) for message in messages]


def _first_token_diff(left: Sequence[int], right: Sequence[int]) -> Optional[int]:
    for index, (left_id, right_id) in enumerate(zip(left, right)):
        if left_id != right_id:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


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


def _build_raw_first15_doc_ids(
    tokenizer: Any,
    example: CompressHistoryExample,
    args: argparse.Namespace,
    mode: str,
) -> tuple[
    List[List[Dict[str, Any]]],
    List[Dict[str, Any]],
    List[List[int]],
    List[List[int]],
    Optional[str],
    Optional[bool],
    Optional[str],
]:
    raw_first_n_turns = _raw_first_n_turns_for_mode(mode, args)
    history_docs, current_messages, target_override, target_has_tool_call, skip_reason = _raw_split_for_mode(
        example,
        args,
        mode,
    )
    if skip_reason is not None:
        return [], current_messages, [], [], target_override, target_has_tool_call, skip_reason
    original_doc_ids = [_chat_template_ids(tokenizer, doc) for doc in history_docs]
    doc_ids = [_chat_template_ids(tokenizer, doc, max_length=args.max_doc_length) for doc in history_docs]
    return history_docs, current_messages, original_doc_ids, doc_ids, target_override, target_has_tool_call, None


def _raw_first15_padded_rows(
    doc_ids: Sequence[Sequence[int]],
    args: argparse.Namespace,
) -> tuple[Optional[torch.Tensor], int, Optional[str]]:
    if len(doc_ids) < args.min_doc_num:
        return None, 0, f"history_docs<{args.min_doc_num}"
    if len(doc_ids) > args.max_doc_num:
        return None, 0, f"history_docs>{args.max_doc_num}"
    total_tokens = sum(len(ids) for ids in doc_ids)
    if total_tokens > args.max_history_tokens:
        return None, total_tokens, f"history_tokens>{args.max_history_tokens}"
    rows = [_pad(list(ids), args.max_doc_length, -100) for ids in doc_ids]
    rows.extend([[-100] * args.max_doc_length for _ in range(args.max_doc_num - len(rows))])
    return torch.tensor(rows, dtype=torch.long), total_tokens, None


@torch.inference_mode()
def _prefill_tokens_with_cache_maybe_gist(
    model: Any,
    input_ids: torch.Tensor,
    past_key_values: Any,
    past_length: int,
    attn_impl: str,
    *,
    use_gist: bool,
) -> tuple[Any, int, float]:
    if input_ids.shape[1] == 0:
        return past_key_values, 0, 0.0
    original_attn_impl = model.model.config._attn_implementation
    model.model.config._attn_implementation = attn_impl
    input_length = input_ids.shape[1]
    cache_length = (
        past_key_values.get_seq_length()
        if past_key_values is not None and hasattr(past_key_values, "get_seq_length")
        else past_length
    )
    attention_mask = torch.ones(
        (input_ids.shape[0], cache_length + input_length),
        dtype=torch.long,
        device=input_ids.device,
    )
    position_ids = torch.arange(
        past_length,
        past_length + input_length,
        dtype=torch.long,
        device=input_ids.device,
    ).unsqueeze(0)
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_values": past_key_values,
        "use_cache": True,
        "logits_to_keep": 1,
    }
    if use_gist:
        kwargs["use_gist"] = True
    _sync_device(input_ids.device)
    start = time.perf_counter()
    outputs = model(**kwargs)
    _sync_device(input_ids.device)
    elapsed = time.perf_counter() - start
    model.model.config._attn_implementation = original_attn_impl
    return outputs.past_key_values, input_length, elapsed


@torch.inference_mode()
def _build_raw_first15_c2kv_prefix(
    model: Any,
    tokenizer: Any,
    example: CompressHistoryExample,
    args: argparse.Namespace,
    mode: str = "raw_first15_c2kv",
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    (
        history_docs,
        current_messages,
        original_doc_ids,
        doc_ids,
        target_override,
        target_has_tool_call,
        skip_reason,
    ) = _build_raw_first15_doc_ids(
        tokenizer, example, args, mode
    )
    if skip_reason is not None:
        return None, skip_reason
    context_input_ids, doc_tokens, skip_reason = _raw_first15_padded_rows(doc_ids, args)
    if context_input_ids is None:
        return None, skip_reason
    current_ids = _chat_template_ids(tokenizer, current_messages, add_generation_prompt=True)
    if not current_ids:
        return None, "empty_raw_current"

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
    prefix = {
        "cache": history_cache,
        "system_length": system_length,
        "history_length": history_length,
        "cache_length": history_cache.get_seq_length(),
        "doc_tokens": doc_tokens,
        "doc_chunks": len(history_docs),
        "kept_history_tokens": doc_tokens,
        "gist_tokens": gist_tokens,
        "compressed_history_tokens": history_cache.get_seq_length() - system_length,
        "actual_compression_ratio": actual_ratio,
        "system_prefill_sec": system_prefill_sec,
        "full_prefill_sec": 0.0,
        "tool_compress_sec": compress_sec,
        "blend_sec": blend_sec,
        "use_gist": True,
        "current_messages": current_messages,
        "raw_current_tokens": len(current_ids),
        **_raw_first15_debug(
            args,
            tokenizer,
            history_docs,
            current_messages,
            original_doc_ids,
            _raw_first_n_turns_for_mode(mode, args),
            window_name=(
                f"prefix{_raw_first_n_turns_for_mode(mode, args)}_next"
                if _raw_prefix_next_for_mode(mode)
                else None
            ),
        ),
    }
    if target_override is not None:
        prefix.update({
            "target_override": target_override,
            "raw_target_source": f"raw_prefix{_raw_first_n_turns_for_mode(mode, args)}_next",
            "raw_target_has_tool_call": target_has_tool_call,
        })
    return prefix, None


@torch.inference_mode()
def _build_raw_first15_hybrid_prefix(
    model: Any,
    tokenizer: Any,
    example: CompressHistoryExample,
    args: argparse.Namespace,
    mode: str = "raw_first15_hybrid",
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    (
        history_docs,
        current_messages,
        original_doc_ids,
        doc_ids,
        target_override,
        target_has_tool_call,
        skip_reason,
    ) = _build_raw_first15_doc_ids(
        tokenizer, example, args, mode
    )
    if skip_reason is not None:
        return None, skip_reason
    if len(history_docs) < args.min_doc_num:
        return None, f"history_docs<{args.min_doc_num}"
    full_count = min(args.hybrid_top_k, len(history_docs))
    full_start = len(history_docs) - full_count
    rest_doc_ids = doc_ids[:full_start]
    full_doc_ids = doc_ids[full_start:]
    rest_tokens = sum(len(ids) for ids in rest_doc_ids)
    full_tokens = sum(len(ids) for ids in full_doc_ids)
    current_ids = _chat_template_ids(tokenizer, current_messages, add_generation_prompt=True)
    if not current_ids:
        return None, "empty_raw_current"
    if rest_tokens + full_tokens > args.max_history_tokens:
        return None, f"history_tokens>{args.max_history_tokens}"

    system_ids = _chat_template_ids(
        tokenizer,
        [{"role": "system", "content": example.system_prompt}],
        tools=example.tools or None,
        keep_bos=True,
        max_length=args.max_system_length,
    )
    system_input_ids = torch.tensor([system_ids], dtype=torch.long, device=model.device)
    prefix_cache, system_length, system_prefill_sec = _prefill_system(
        model, system_input_ids, args.system_attn_impl
    )

    rest_length = 0
    gist_tokens = 0
    compress_sec = 0.0
    blend_sec = 0.0
    if rest_doc_ids:
        context_input_ids, _rest_total, skip_reason = _raw_first15_padded_rows(rest_doc_ids, args)
        if context_input_ids is None:
            return None, skip_reason
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
            system_length,
            args.gist_attn_impl,
            args.override_ratio,
        )

    full_length = 0
    full_prefill_sec = 0.0
    full_ids = [token for ids in full_doc_ids for token in ids]
    if full_ids:
        full_input_ids = torch.tensor([full_ids], dtype=torch.long, device=model.device)
        prefix_cache, full_length, full_prefill_sec = _prefill_tokens_with_cache_maybe_gist(
            model,
            full_input_ids,
            past_key_values=prefix_cache,
            past_length=system_length + rest_length,
            attn_impl=args.generate_attn_impl,
            use_gist=bool(rest_doc_ids),
        )

    doc_tokens = rest_tokens + full_tokens
    compressed_tokens = max(0, prefix_cache.get_seq_length() - system_length)
    prefix = {
        "cache": prefix_cache,
        "system_length": system_length,
        "history_length": rest_length + full_length,
        "cache_length": prefix_cache.get_seq_length(),
        "doc_tokens": doc_tokens,
        "doc_chunks": len(history_docs),
        "full_history_docs": len(full_doc_ids),
        "rest_history_docs": len(rest_doc_ids),
        "top_full_tokens": full_tokens,
        "rest_history_tokens": rest_tokens,
        "kept_history_tokens": full_tokens,
        "gist_tokens": gist_tokens,
        "compressed_history_tokens": compressed_tokens,
        "actual_compression_ratio": doc_tokens / compressed_tokens if compressed_tokens else 0.0,
        "system_prefill_sec": system_prefill_sec,
        "full_prefill_sec": full_prefill_sec,
        "tool_compress_sec": compress_sec,
        "blend_sec": blend_sec,
        "use_gist": bool(rest_doc_ids),
        "current_messages": current_messages,
        "raw_current_tokens": len(current_ids),
        "selected_history_doc_indices": list(range(full_start, len(history_docs))),
        **_raw_first15_debug(
            args,
            tokenizer,
            history_docs,
            current_messages,
            original_doc_ids,
            _raw_first_n_turns_for_mode(mode, args),
            window_name=(
                f"prefix{_raw_first_n_turns_for_mode(mode, args)}_next"
                if _raw_prefix_next_for_mode(mode)
                else None
            ),
        ),
    }
    if target_override is not None:
        prefix.update({
            "target_override": target_override,
            "raw_target_source": f"raw_prefix{_raw_first_n_turns_for_mode(mode, args)}_next",
            "raw_target_has_tool_call": target_has_tool_call,
        })
    return prefix, None


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
    current_messages = prefix.get("current_messages") or _current_messages(example)
    prompt_ids = _chat_template_ids(tokenizer, current_messages, add_generation_prompt=True)
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
    target = prefix.get("target_override", example.answer)
    metrics = _target_metrics(tokenizer, target, prediction)
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
    history_cache = system_cache
    if history_ids:
        history_input_ids = torch.tensor([history_ids], dtype=torch.long, device=model.device)
        history_cache, history_length, full_prefill_sec = _prefill_tokens_with_cache(
            model,
            history_input_ids,
            past_key_values=system_cache,
            past_length=system_length,
            attn_impl=args.generate_attn_impl,
        )
    else:
        history_length = 0
        full_prefill_sec = 0.0
    return {
        "cache": history_cache,
        "system_length": system_length,
        "history_length": history_length,
        "cache_length": history_cache.get_seq_length(),
        "doc_tokens": doc_tokens,
        "doc_chunks": len(history),
        "kept_history_tokens": kept_tokens,
        "gist_tokens": 0,
        "actual_compression_ratio": doc_tokens / kept_tokens if kept_tokens else 1.0,
        "system_prefill_sec": system_prefill_sec,
        "full_prefill_sec": full_prefill_sec,
        "tool_compress_sec": 0.0,
        "blend_sec": 0.0,
        "use_gist": False,
    }, None


@torch.inference_mode()
def _build_system_only_prefix(
    model: Any,
    tokenizer: Any,
    example: CompressHistoryExample,
    args: argparse.Namespace,
) -> tuple[Dict[str, Any], None]:
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
    return {
        "cache": system_cache,
        "system_length": system_length,
        "history_length": 0,
        "cache_length": system_cache.get_seq_length(),
        "doc_tokens": 0,
        "doc_chunks": 0,
        "kept_history_tokens": 0,
        "gist_tokens": 0,
        "compressed_history_tokens": 0,
        "actual_compression_ratio": 1.0,
        "system_prefill_sec": system_prefill_sec,
        "full_prefill_sec": 0.0,
        "tool_compress_sec": 0.0,
        "blend_sec": 0.0,
        "use_gist": False,
    }, None


def _model_rope_params(model: Any) -> tuple[float, str]:
    config = getattr(model, "config", None) or getattr(getattr(model, "model", None), "config", None)
    rope_params = getattr(config, "rope_parameters", None) or {}
    rope_theta = rope_params.get("rope_theta", getattr(config, "rope_theta", 10000.0))
    rope_type = rope_params.get("rope_type", "default")
    return float(rope_theta), str(rope_type)


@torch.inference_mode()
def _prefill_ids_no_past(
    model: Any,
    input_ids: torch.Tensor,
    attn_impl: str,
) -> tuple[Any, int, float]:
    original_attn_impl = model.model.config._attn_implementation
    model.model.config._attn_implementation = attn_impl
    attention_mask = torch.ones_like(input_ids)
    _sync_device(input_ids.device)
    start = time.perf_counter()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        logits_to_keep=1,
    )
    _sync_device(input_ids.device)
    elapsed = time.perf_counter() - start
    model.model.config._attn_implementation = original_attn_impl
    return outputs.past_key_values, int(attention_mask.sum().item()), elapsed


def _append_independent_cache(
    model: Any,
    prefix_cache: Any,
    doc_cache: Any,
    logical_start: int,
) -> Any:
    rope_theta, rope_type = _model_rope_params(model)
    for prefix_layer, doc_layer in zip(prefix_cache.layers, doc_cache.layers):
        doc_keys = rotate_k_cache_rope(
            doc_layer.keys[0],
            logical_start,
            rope_theta,
            rope_type,
        ).unsqueeze(0)
        prefix_layer.keys = torch.cat([prefix_layer.keys, doc_keys], dim=-2)
        prefix_layer.values = torch.cat([prefix_layer.values, doc_layer.values], dim=-2)
    return prefix_cache


@torch.inference_mode()
def _build_split_full_prefix(
    model: Any,
    tokenizer: Any,
    example: CompressHistoryExample,
    args: argparse.Namespace,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    history = _history_messages(tokenizer, example, args)
    if len(history) < args.min_doc_num:
        return None, f"history_docs<{args.min_doc_num}"
    current_ids = _chat_template_ids(tokenizer, _current_messages(example), add_generation_prompt=True)
    if not current_ids:
        return None, "empty_current"

    system_ids = _chat_template_ids(
        tokenizer,
        [{"role": "system", "content": example.system_prompt}],
        tools=example.tools or None,
        keep_bos=True,
        max_length=args.max_system_length,
    )
    doc_ids = _history_doc_ids(tokenizer, history)
    doc_tokens = sum(len(ids) for ids in doc_ids)
    total_len = len(system_ids) + doc_tokens + len(current_ids)
    if args.max_baseline_input_tokens and total_len > args.max_baseline_input_tokens:
        return None, f"baseline_input_tokens>{args.max_baseline_input_tokens}"

    system_input_ids = torch.tensor([system_ids], dtype=torch.long, device=model.device)
    prefix_cache, system_length, system_prefill_sec = _prefill_system(
        model,
        system_input_ids,
        args.system_attn_impl,
    )
    logical_length = system_length
    full_prefill_sec = 0.0
    for ids in doc_ids:
        doc_input_ids = torch.tensor([ids], dtype=torch.long, device=model.device)
        doc_cache, doc_length, doc_prefill_sec = _prefill_ids_no_past(
            model,
            doc_input_ids,
            args.generate_attn_impl,
        )
        prefix_cache = _append_independent_cache(model, prefix_cache, doc_cache, logical_length)
        logical_length += doc_length
        full_prefill_sec += doc_prefill_sec

    return {
        "cache": prefix_cache,
        "system_length": system_length,
        "history_length": doc_tokens,
        "cache_length": prefix_cache.get_seq_length(),
        "split_logical_length": logical_length,
        "split_cache_length": prefix_cache.get_seq_length(),
        "split_cache_length_matches_logical": prefix_cache.get_seq_length() == logical_length,
        "split_total_prompt_tokens": total_len,
        "split_current_tokens": len(current_ids),
        "split_system_tokens": len(system_ids),
        "doc_tokens": doc_tokens,
        "doc_chunks": len(history),
        "kept_history_tokens": doc_tokens,
        "gist_tokens": 0,
        "actual_compression_ratio": 1.0,
        "system_prefill_sec": system_prefill_sec,
        "full_prefill_sec": full_prefill_sec,
        "tool_compress_sec": 0.0,
        "blend_sec": 0.0,
        "use_gist": False,
    }, None


@torch.inference_mode()
def _build_sequential_full_prefix(
    model: Any,
    tokenizer: Any,
    example: CompressHistoryExample,
    args: argparse.Namespace,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    history = _history_messages(tokenizer, example, args)
    if len(history) < args.min_doc_num:
        return None, f"history_docs<{args.min_doc_num}"
    current_ids = _chat_template_ids(tokenizer, _current_messages(example), add_generation_prompt=True)
    if not current_ids:
        return None, "empty_current"

    system_ids = _chat_template_ids(
        tokenizer,
        [{"role": "system", "content": example.system_prompt}],
        tools=example.tools or None,
        keep_bos=True,
        max_length=args.max_system_length,
    )
    doc_ids = _history_doc_ids(tokenizer, history)
    doc_tokens = sum(len(ids) for ids in doc_ids)
    total_len = len(system_ids) + doc_tokens + len(current_ids)
    if args.max_baseline_input_tokens and total_len > args.max_baseline_input_tokens:
        return None, f"baseline_input_tokens>{args.max_baseline_input_tokens}"

    reconstructed_ids = _chat_template_ids(
        tokenizer,
        [_system_message(example), *history, *_current_messages(example)],
        tools=example.tools or None,
        add_generation_prompt=True,
        keep_bos=True,
    )
    sequential_ids = system_ids + [token for ids in doc_ids for token in ids] + current_ids

    system_input_ids = torch.tensor([system_ids], dtype=torch.long, device=model.device)
    prefix_cache, system_length, system_prefill_sec = _prefill_system(
        model,
        system_input_ids,
        args.system_attn_impl,
    )
    logical_length = system_length
    full_prefill_sec = 0.0
    for ids in doc_ids:
        doc_input_ids = torch.tensor([ids], dtype=torch.long, device=model.device)
        prefix_cache, doc_length, doc_prefill_sec = _prefill_tokens_with_cache(
            model,
            doc_input_ids,
            past_key_values=prefix_cache,
            past_length=logical_length,
            attn_impl=args.generate_attn_impl,
        )
        logical_length += doc_length
        full_prefill_sec += doc_prefill_sec

    return {
        "cache": prefix_cache,
        "system_length": system_length,
        "history_length": doc_tokens,
        "cache_length": prefix_cache.get_seq_length(),
        "sequential_logical_length": logical_length,
        "sequential_cache_length": prefix_cache.get_seq_length(),
        "sequential_cache_length_matches_logical": prefix_cache.get_seq_length() == logical_length,
        "sequential_total_prompt_tokens": total_len,
        "sequential_current_tokens": len(current_ids),
        "sequential_system_tokens": len(system_ids),
        "sequential_reconstructed_tokens": len(reconstructed_ids),
        "sequential_ids_equal_reconstructed": sequential_ids == reconstructed_ids,
        "sequential_first_token_diff_index": _first_token_diff(sequential_ids, reconstructed_ids),
        "doc_tokens": doc_tokens,
        "doc_chunks": len(history),
        "kept_history_tokens": doc_tokens,
        "gist_tokens": 0,
        "actual_compression_ratio": 1.0,
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
    if not _history_messages(tokenizer, example, args) and args.min_doc_num <= 0:
        return _build_system_only_prefix(model, tokenizer, example, args)
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
def _build_each_turn_independent_c2kv_prefix(
    model: Any,
    tokenizer: Any,
    example: CompressHistoryExample,
    args: argparse.Namespace,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    history = _history_messages(tokenizer, example, args)
    if len(history) < args.min_doc_num:
        return None, f"history_docs<{args.min_doc_num}"
    if not history:
        return _build_system_only_prefix(model, tokenizer, example, args)
    if len(history) > args.max_doc_num:
        return None, f"history_docs>{args.max_doc_num}"

    doc_ids = [_chat_template_ids(tokenizer, [message], max_length=args.max_doc_length) for message in history]
    doc_tokens = sum(len(ids) for ids in doc_ids)
    if doc_tokens > args.max_history_tokens:
        return None, f"history_tokens>{args.max_history_tokens}"

    system_ids = _chat_template_ids(
        tokenizer,
        [{"role": "system", "content": example.system_prompt}],
        tools=example.tools or None,
        keep_bos=True,
        max_length=args.max_system_length,
    )
    system_input_ids = torch.tensor([system_ids], dtype=torch.long, device=model.device)
    prefix_cache, system_length, system_prefill_sec = _prefill_system(
        model, system_input_ids, args.system_attn_impl
    )

    logical_history_length = 0
    compressed_history_tokens = 0
    compress_sec = 0.0
    blend_sec = 0.0
    for ids in doc_ids:
        before_len = prefix_cache.get_seq_length()
        context_input_ids = torch.tensor([_pad(ids, args.max_doc_length, -100)], dtype=torch.long)
        (
            prefix_cache,
            original_len,
            _gist_delta,
            _actual_ratio,
            doc_compress_sec,
            doc_blend_sec,
        ) = _build_tool_cache(
            model,
            context_input_ids,
            prefix_cache,
            system_length + logical_history_length,
            args.gist_attn_impl,
            args.override_ratio,
        )
        logical_history_length += original_len
        compressed_history_tokens += max(0, prefix_cache.get_seq_length() - before_len)
        compress_sec += doc_compress_sec
        blend_sec += doc_blend_sec

    return {
        "cache": prefix_cache,
        "system_length": system_length,
        "history_length": logical_history_length,
        "cache_length": prefix_cache.get_seq_length(),
        "doc_tokens": doc_tokens,
        "doc_chunks": len(history),
        "kept_history_tokens": doc_tokens,
        "gist_tokens": compressed_history_tokens,
        "compressed_history_tokens": compressed_history_tokens,
        "actual_compression_ratio": doc_tokens / compressed_history_tokens if compressed_history_tokens else 0.0,
        "system_prefill_sec": system_prefill_sec,
        "full_prefill_sec": 0.0,
        "tool_compress_sec": compress_sec,
        "blend_sec": blend_sec,
        "use_gist": True,
        "history_compression_mode": "each_turn_independent_c2kv",
    }, None


@torch.inference_mode()
def _build_recompress_all_every_turn_c2kv_prefix(
    model: Any,
    tokenizer: Any,
    example: CompressHistoryExample,
    args: argparse.Namespace,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    history = _history_messages(tokenizer, example, args)
    if len(history) < args.min_doc_num:
        return None, f"history_docs<{args.min_doc_num}"
    if not history:
        return _build_system_only_prefix(model, tokenizer, example, args)

    history_ids = _chat_template_ids(tokenizer, history)
    doc_tokens = len(history_ids)
    if doc_tokens > args.max_history_tokens:
        return None, f"history_tokens>{args.max_history_tokens}"
    chunks = [
        history_ids[index : index + args.max_doc_length]
        for index in range(0, len(history_ids), args.max_doc_length)
    ]
    if len(chunks) > args.max_doc_num:
        return None, f"history_docs>{args.max_doc_num}"
    rows = [_pad(chunk, args.max_doc_length, -100) for chunk in chunks]
    rows.extend([[-100] * args.max_doc_length for _ in range(args.max_doc_num - len(rows))])
    context_input_ids = torch.tensor(rows, dtype=torch.long)

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
        "doc_chunks": len(history),
        "recompress_chunks": len(chunks),
        "kept_history_tokens": doc_tokens,
        "gist_tokens": gist_tokens,
        "compressed_history_tokens": gist_tokens,
        "actual_compression_ratio": actual_ratio,
        "system_prefill_sec": system_prefill_sec,
        "full_prefill_sec": 0.0,
        "tool_compress_sec": compress_sec,
        "blend_sec": blend_sec,
        "use_gist": True,
        "history_compression_mode": "recompress_all_every_turn_c2kv",
    }, None


@torch.inference_mode()
def _build_contiguous_history_c2kv_prefix(
    model: Any,
    tokenizer: Any,
    example: CompressHistoryExample,
    args: argparse.Namespace,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    history = _history_messages(tokenizer, example, args)
    if len(history) < args.min_doc_num:
        return None, f"history_docs<{args.min_doc_num}"
    current_ids = _chat_template_ids(tokenizer, _current_messages(example), add_generation_prompt=True)
    if not current_ids:
        return None, "empty_current"

    history_ids = _chat_template_ids(tokenizer, history)
    original_history_tokens = len(history_ids)
    if original_history_tokens > args.max_history_tokens:
        return None, f"history_tokens>{args.max_history_tokens}"
    if len(history_ids) > args.max_doc_length:
        if args.history_selection == "tail":
            history_ids = history_ids[-args.max_doc_length :]
        else:
            history_ids = history_ids[: args.max_doc_length]
    doc_tokens = len(history_ids)
    if doc_tokens == 0:
        return None, "empty_history"

    rows = [_pad(history_ids, args.max_doc_length, -100)]
    rows.extend([[-100] * args.max_doc_length for _ in range(max(0, args.max_doc_num - 1))])
    context_input_ids = torch.tensor(rows, dtype=torch.long)

    system_ids = _chat_template_ids(
        tokenizer,
        [{"role": "system", "content": example.system_prompt}],
        tools=example.tools or None,
        keep_bos=True,
        max_length=args.max_system_length,
    )
    total_len = len(system_ids) + doc_tokens + len(current_ids)
    if args.max_baseline_input_tokens and total_len > args.max_baseline_input_tokens:
        return None, f"baseline_input_tokens>{args.max_baseline_input_tokens}"

    reconstructed_ids = _chat_template_ids(
        tokenizer,
        [_system_message(example), *history, *_current_messages(example)],
        tools=example.tools or None,
        add_generation_prompt=True,
        keep_bos=True,
    )
    contiguous_ids = system_ids + history_ids + current_ids

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
        "doc_chunks": 1,
        "contiguous_history_docs_input": len(history),
        "contiguous_original_history_tokens": original_history_tokens,
        "contiguous_history_truncated": original_history_tokens != doc_tokens,
        "contiguous_total_prompt_tokens": total_len,
        "contiguous_current_tokens": len(current_ids),
        "contiguous_system_tokens": len(system_ids),
        "contiguous_reconstructed_tokens": len(reconstructed_ids),
        "contiguous_ids_equal_reconstructed": contiguous_ids == reconstructed_ids,
        "contiguous_first_token_diff_index": _first_token_diff(contiguous_ids, reconstructed_ids),
        "kept_history_tokens": doc_tokens,
        "gist_tokens": gist_tokens,
        "actual_compression_ratio": actual_ratio,
        "system_prefill_sec": system_prefill_sec,
        "full_prefill_sec": 0.0,
        "tool_compress_sec": compress_sec,
        "blend_sec": blend_sec,
        "use_gist": True,
    }, None


def _gist_spans_from_doc_lengths(doc_lengths: Sequence[int], gist_tokens: int) -> List[tuple[int, int]]:
    total = sum(doc_lengths)
    if total <= 0 or gist_tokens <= 0:
        return [(0, 0) for _ in doc_lengths]
    spans = []
    cursor = 0
    for length in doc_lengths:
        start = int(cursor * gist_tokens / total)
        cursor += length
        end = int((cursor * gist_tokens + total - 1) / total)
        if end <= start:
            end = min(gist_tokens, start + 1)
        spans.append((max(0, start), min(gist_tokens, end)))
    return spans


def _history_recency_rank(history: Sequence[Dict[str, Any]], history_selection: str) -> List[int]:
    if history_selection == "head":
        return list(range(len(history)))
    return list(range(len(history) - 1, -1, -1))


def _att_rerank_replacement(
    base_ranked: Sequence[int],
    head_rankings: Sequence[Dict[str, Any]],
    top_k: int,
    pool_size: int,
    min_heads: int,
    min_margin: float,
    min_score_gain: float,
) -> tuple[List[int], Optional[Dict[str, Any]]]:
    if top_k <= 0 or len(base_ranked) <= top_k:
        return list(base_ranked), None
    pool_size = max(top_k + 1, min(pool_size, len(base_ranked)))
    base_top = list(base_ranked[:top_k])
    candidate_indices = set(base_ranked[top_k:pool_size])
    replace_index = base_top[-1]

    votes: Dict[int, Dict[str, Any]] = {}
    for head in head_rankings:
        ranked = head.get("ranked_indices") or []
        scores_by_index = head.get("scores_by_index") or {}
        if not ranked:
            continue
        top_index = ranked[0]
        if top_index not in candidate_indices:
            continue
        top_score = float(scores_by_index.get(top_index, 0.0) or 0.0)
        second_score = float(scores_by_index.get(ranked[1], 0.0) or 0.0) if len(ranked) > 1 else 0.0
        margin = top_score - second_score
        if margin < min_margin:
            continue
        replace_score = float(scores_by_index.get(replace_index, 0.0) or 0.0)
        entry = votes.setdefault(
            top_index,
            {
                "num_heads": 0,
                "margin_sum": 0.0,
                "score_sum": 0.0,
                "replace_score_sum": 0.0,
                "support_heads": [],
            },
        )
        entry["num_heads"] += 1
        entry["margin_sum"] += margin
        entry["score_sum"] += top_score
        entry["replace_score_sum"] += replace_score
        entry["support_heads"].append({
            "layer": head.get("layer"),
            "head": head.get("head"),
            "margin": round(margin, 8),
            "top_score": round(top_score, 8),
            "replace_score": round(replace_score, 8),
        })

    if not votes:
        return list(base_ranked), None
    best_index, best = max(
        votes.items(),
        key=lambda item: (
            item[1]["num_heads"],
            item[1]["margin_sum"],
            item[1]["score_sum"],
            -list(base_ranked).index(item[0]),
        ),
    )
    score_gain = float(best["score_sum"]) - float(best["replace_score_sum"])
    accepted = best["num_heads"] >= min_heads and score_gain >= min_score_gain
    debug = {
        "candidate_doc_index": best_index,
        "candidate_base_rank": list(base_ranked).index(best_index) + 1,
        "replace_doc_index": replace_index,
        "replace_base_rank": top_k,
        "num_support_heads": best["num_heads"],
        "score_gain": round(score_gain, 8),
        "accepted": accepted,
        "support_heads": best["support_heads"][:20],
    }
    if not accepted:
        return list(base_ranked), debug
    final = list(base_ranked)
    final[top_k - 1] = best_index
    selected = set(final[:top_k])
    final = final[:top_k] + [index for index in base_ranked if index not in selected]
    return final, debug


@torch.inference_mode()
def _build_full_history_cache_with_spans(
    model: Any,
    tokenizer: Any,
    history: Sequence[Dict[str, Any]],
    system_cache: Any,
    system_length: int,
    attn_impl: str,
) -> tuple[Any, int, List[tuple[int, int]], List[int], float]:
    prefix_cache = system_cache
    history_length = 0
    spans: List[tuple[int, int]] = []
    doc_lengths: List[int] = []
    prefill_sec = 0.0
    for message in history:
        ids = _chat_template_ids(tokenizer, [message])
        doc_lengths.append(len(ids))
        doc_input_ids = torch.tensor([ids], dtype=torch.long, device=model.device)
        prefix_cache, length, elapsed = _prefill_tokens_with_cache(
            model,
            doc_input_ids,
            past_key_values=prefix_cache,
            past_length=system_length + history_length,
            attn_impl=attn_impl,
        )
        spans.append((history_length, history_length + length))
        history_length += length
        prefill_sec += elapsed
    return prefix_cache, history_length, spans, doc_lengths, prefill_sec


@torch.inference_mode()
def _rank_history_by_attention(
    model: Any,
    tokenizer: Any,
    example: CompressHistoryExample,
    args: argparse.Namespace,
    history: Sequence[Dict[str, Any]],
    cache_mode: str,
) -> tuple[List[int], List[float], float, List[Dict[str, Any]]]:
    system_ids = _chat_template_ids(
        tokenizer,
        [{"role": "system", "content": example.system_prompt}],
        tools=example.tools or None,
        keep_bos=True,
        max_length=args.max_system_length,
    )
    system_input_ids = torch.tensor([system_ids], dtype=torch.long, device=model.device)
    system_cache, system_length, system_prefill_sec = _prefill_system(
        model,
        system_input_ids,
        args.system_attn_impl,
    )

    if cache_mode == "full":
        (
            prefix_cache,
            history_length,
            spans,
            _doc_lengths,
            history_prefill_sec,
        ) = _build_full_history_cache_with_spans(
            model,
            tokenizer,
            history,
            system_cache,
            system_length,
            args.attention_router_attn_impl,
        )
        history_key_tokens = history_length
        router_build_sec = history_prefill_sec
        use_gist_for_query = False
    else:
        rows = []
        doc_lengths = []
        for message in history:
            ids = _chat_template_ids(tokenizer, [message], max_length=args.max_doc_length)
            doc_lengths.append(len(ids))
            rows.append(_pad(ids, args.max_doc_length, -100))
        rows.extend([[-100] * args.max_doc_length for _ in range(args.max_doc_num - len(history))])
        context_input_ids = torch.tensor(rows, dtype=torch.long)
        (
            prefix_cache,
            history_length,
            gist_tokens,
            _,
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
        if gist_tokens <= 0:
            raise RuntimeError("attention_router_empty_gist")
        spans = _gist_spans_from_doc_lengths(doc_lengths, gist_tokens)
        history_key_tokens = gist_tokens
        router_build_sec = compress_sec + blend_sec
        use_gist_for_query = True

    query_ids = _chat_template_ids(tokenizer, _current_messages(example), add_generation_prompt=True)
    if args.attention_router_max_query_tokens and len(query_ids) > args.attention_router_max_query_tokens:
        query_ids = query_ids[-args.attention_router_max_query_tokens :]
    query_input_ids = torch.tensor([query_ids], dtype=torch.long, device=model.device)
    query_len = query_input_ids.shape[1]
    attention_mask = torch.ones(
        (1, prefix_cache.get_seq_length() + query_len),
        dtype=torch.long,
        device=model.device,
    )
    position_ids = torch.arange(
        system_length + history_length,
        system_length + history_length + query_len,
        dtype=torch.long,
        device=model.device,
    ).unsqueeze(0)

    layer_scores: List[List[float]] = []
    head_rankings: List[Dict[str, Any]] = []

    def _score_history_attention(history_attn: torch.Tensor, span_len: int) -> float:
        if args.attention_router_score_mode == "sum":
            score = history_attn.sum(dim=-1).mean()
        elif args.attention_router_score_mode == "sqrt_len":
            score = history_attn.sum(dim=-1).mean() / (span_len ** 0.5)
        else:
            score = history_attn.mean()
        return float(score.item())

    def make_hook(layer_index: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            attn_weights = output[1] if isinstance(output, tuple) and len(output) > 1 else None
            if attn_weights is None:
                return
            cache_attn = attn_weights[0, :, :, system_length : system_length + history_key_tokens].float()
            layer_head_scores = []
            for head_index in range(cache_attn.shape[0]):
                head_attn = cache_attn[head_index]
                scores = []
                for start, end in spans:
                    if end <= start:
                        scores.append(0.0)
                    else:
                        scores.append(_score_history_attention(head_attn[:, start:end], end - start))
                ranked = sorted(range(len(history)), key=lambda index: (-scores[index], index))
                head_rankings.append({
                    "layer": layer_index,
                    "head": head_index,
                    "ranked_indices": ranked,
                    "scores_by_index": {index: scores[index] for index in range(len(history))},
                })
                layer_head_scores.append(scores)
            if layer_head_scores:
                layer_scores.append([
                    sum(head_scores[index] for head_scores in layer_head_scores) / len(layer_head_scores)
                    for index in range(len(history))
                ])
        return hook

    num_layers = len(model.model.layers)
    last_layers = max(1, min(args.attention_router_layers, num_layers))
    handles = [
        model.model.layers[index].self_attn.register_forward_hook(make_hook(index))
        for index in range(num_layers - last_layers, num_layers)
    ]
    original_attn_impl = model.model.config._attn_implementation
    model.model.config._attn_implementation = args.attention_router_attn_impl
    try:
        forward_kwargs = {
            "input_ids": query_input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": prefix_cache,
            "use_cache": True,
            "logits_to_keep": 1,
        }
        if use_gist_for_query:
            forward_kwargs["use_gist"] = True
        model(**forward_kwargs)
    finally:
        model.model.config._attn_implementation = original_attn_impl
        for handle in handles:
            handle.remove()

    if not layer_scores:
        raise RuntimeError("History attention router did not capture attention weights. Use eager attention.")
    scores = [
        sum(layer[index] for layer in layer_scores) / len(layer_scores)
        for index in range(len(history))
    ]
    ranked = sorted(range(len(history)), key=lambda index: (-scores[index], index))
    return ranked, scores, system_prefill_sec + router_build_sec, head_rankings


@torch.inference_mode()
def _build_hybrid_prefix(
    model: Any,
    tokenizer: Any,
    example: CompressHistoryExample,
    args: argparse.Namespace,
    recent_full_docs: Optional[int] = None,
    selected_full_indices: Optional[Sequence[int]] = None,
    router_debug: Optional[Dict[str, Any]] = None,
    history_override: Optional[Sequence[Dict[str, Any]]] = None,
    full_doc_max_length: Optional[int] = None,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    history = list(history_override) if history_override is not None else _history_messages(tokenizer, example, args)
    if len(history) < args.min_doc_num:
        return None, f"history_docs<{args.min_doc_num}"
    if not history:
        return _build_system_only_prefix(model, tokenizer, example, args)
    full_count = args.hybrid_top_k if recent_full_docs is None else recent_full_docs
    if selected_full_indices is not None:
        full_set = set(selected_full_indices)
        full_history = [message for index, message in enumerate(history) if index in full_set]
    elif recent_full_docs is not None:
        full_history = history[-full_count:]
        full_set = set(range(len(history) - len(full_history), len(history)))
    else:
        full_history = history[-full_count:] if args.history_selection == "tail" else history[:full_count]
        full_set = set(range(len(history) - len(full_history), len(history))) if args.history_selection == "tail" else set(range(len(full_history)))
    rest_history = [message for index, message in enumerate(history) if index not in full_set]
    full_after_c2kv = bool(getattr(args, "hybrid_full_after_c2kv", False))

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

    full_ids = [
        token
        for message in full_history
        for token in _chat_template_ids(tokenizer, [message], max_length=full_doc_max_length)
    ]
    full_tokens = len(full_ids)
    top_prefill_sec = 0.0
    prefix_cache = system_cache
    full_length = 0

    def append_full_history(current_past_length: int) -> int:
        nonlocal prefix_cache, top_prefill_sec
        if not full_ids:
            return 0
        full_input_ids = torch.tensor([full_ids], dtype=torch.long, device=model.device)
        prefix_cache, appended_length, prefill_sec = _prefill_tokens_with_cache(
            model,
            full_input_ids,
            past_key_values=prefix_cache,
            past_length=current_past_length,
            attn_impl=args.generate_attn_impl,
        )
        top_prefill_sec += prefill_sec
        return appended_length

    if not full_after_c2kv:
        full_length = append_full_history(system_length)

    rest_tokens = 0
    rest_length = 0
    gist_tokens = 0
    compress_sec = 0.0
    blend_sec = 0.0
    if rest_history:
        if len(rest_history) > args.max_doc_num:
            return None, f"history_docs>{args.max_doc_num}"
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
    if full_after_c2kv:
        full_length = append_full_history(system_length + rest_length)

    doc_tokens = rest_tokens + full_tokens
    compressed_tokens = gist_tokens + full_tokens
    prefix = {
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
        "compressed_history_tokens": compressed_tokens,
        "actual_compression_ratio": doc_tokens / compressed_tokens if compressed_tokens else 0.0,
        "system_prefill_sec": system_prefill_sec,
        "full_prefill_sec": top_prefill_sec,
        "tool_compress_sec": compress_sec,
        "blend_sec": blend_sec,
        "use_gist": bool(rest_history),
    }
    if router_debug:
        prefix.update(router_debug)
    return prefix, None


@torch.inference_mode()
def _build_turn_ablation_prefix(
    model: Any,
    tokenizer: Any,
    example: CompressHistoryExample,
    args: argparse.Namespace,
    mode: str,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    history = _turn_ablation_history(example, 15)
    if len(history) < 15:
        return None, "history_turn_docs<15"
    selected_full_indices = _turn_ablation_full_indices(mode)
    turn_original_tokens = [
        len(_chat_template_ids(tokenizer, [message]))
        for message in history
    ]
    turn_full_kv_tokens = [
        len(_chat_template_ids(tokenizer, [message], max_length=args.max_doc_length))
        for message in history
    ]
    variable_turn_position = _turn_ablation_position(mode)
    if variable_turn_position is None:
        if mode == "all_full":
            variable_original_tokens = sum(turn_original_tokens[:13])
            variable_full_kv_tokens = sum(turn_full_kv_tokens[:13])
        else:
            variable_original_tokens = 0
            variable_full_kv_tokens = 0
    else:
        variable_original_tokens = turn_original_tokens[variable_turn_position - 1]
        variable_full_kv_tokens = turn_full_kv_tokens[variable_turn_position - 1]
    prefix, skip_reason = _build_hybrid_prefix(
        model,
        tokenizer,
        example,
        args,
        selected_full_indices=selected_full_indices,
        history_override=history,
        full_doc_max_length=args.max_doc_length,
    )
    if prefix is None:
        return None, skip_reason
    prefix.update({
        "turn_ablation_mode": mode,
        "turn_position": variable_turn_position,
        "turn_window_size": 15,
        "turn_history_doc_tokens": turn_original_tokens,
        "turn_full_kv_doc_tokens": turn_full_kv_tokens,
        "avg_turn_original_tokens": sum(turn_original_tokens) / len(turn_original_tokens),
        "avg_turn_full_kv_tokens": sum(turn_full_kv_tokens) / len(turn_full_kv_tokens),
        "ablation_turn_original_tokens": variable_original_tokens,
        "full_restore_added_kv_tokens": variable_full_kv_tokens,
        "fixed_recent_full_tokens": sum(turn_full_kv_tokens[13:15]),
        "selected_history_doc_indices": selected_full_indices,
    })
    return prefix, None


@torch.inference_mode()
def _generate_one(
    model: Any,
    tokenizer: Any,
    example: CompressHistoryExample,
    args: argparse.Namespace,
    mode: str,
) -> Dict[str, Any]:
    total_start = time.perf_counter()
    if mode in FULL_PROMPT_MODES:
        return _generate_full_prompt(model, tokenizer, example, args, mode)
    if mode == "history_full":
        prefix, skip_reason = _build_full_or_truncate_prefix(model, tokenizer, example, args, "full")
    elif mode == "history_all_c2kv4":
        prefix, skip_reason = _build_c2kv_prefix(model, tokenizer, example, args)
    elif mode == "recent2_full_rest_c2kv4":
        prefix, skip_reason = _build_hybrid_prefix(model, tokenizer, example, args, recent_full_docs=2)
    elif mode == "each_turn_independent_c2kv4":
        prefix, skip_reason = _build_each_turn_independent_c2kv_prefix(model, tokenizer, example, args)
    elif mode == "recompress_all_every_turn_c2kv4":
        prefix, skip_reason = _build_recompress_all_every_turn_c2kv_prefix(model, tokenizer, example, args)
    elif mode == "full":
        prefix, skip_reason = _build_full_or_truncate_prefix(model, tokenizer, example, args, mode)
    elif mode in TRUNCATE_MODES:
        prefix, skip_reason = _build_full_or_truncate_prefix(model, tokenizer, example, args, "truncate")
    elif mode == "split_full_kv":
        prefix, skip_reason = _build_split_full_prefix(model, tokenizer, example, args)
    elif mode == "sequential_full_kv":
        prefix, skip_reason = _build_sequential_full_prefix(model, tokenizer, example, args)
    elif mode == "contiguous_history_c2kv":
        prefix, skip_reason = _build_contiguous_history_c2kv_prefix(model, tokenizer, example, args)
    elif mode in {
        "raw_first15_c2kv",
        "raw_first8_c2kv",
        "raw_prefix8_exact_c2kv",
        "raw_prefix_next_c2kv",
    }:
        prefix, skip_reason = _build_raw_first15_c2kv_prefix(model, tokenizer, example, args, mode)
    elif mode in {
        "raw_first15_hybrid",
        "raw_first8_hybrid",
        "raw_prefix8_exact_hybrid",
        "raw_prefix_next_hybrid",
    }:
        prefix, skip_reason = _build_raw_first15_hybrid_prefix(model, tokenizer, example, args, mode)
    elif mode in C2KV_MODES:
        prefix, skip_reason = _build_c2kv_prefix(model, tokenizer, example, args)
    elif mode in TURN_ABLATION_MODES:
        prefix, skip_reason = _build_turn_ablation_prefix(model, tokenizer, example, args, mode)
    elif mode in HYBRID_MODES:
        recent_full_docs = None
        if mode == "recent1_hybrid":
            recent_full_docs = 1
        elif mode == "recent2_hybrid":
            recent_full_docs = 2
        history = _history_messages(tokenizer, example, args)
        router_debug = None
        selected_full_indices = None
        if mode in {"att_hybrid", "att_fullkv_hybrid", "hybrid_fullkv_att_rerank"}:
            cache_mode = "full" if mode in {"att_fullkv_hybrid", "hybrid_fullkv_att_rerank"} else "c2kv"
            ranked, scores, router_sec, head_rankings = _rank_history_by_attention(
                model,
                tokenizer,
                example,
                args,
                history,
                cache_mode,
            )
            if mode == "hybrid_fullkv_att_rerank":
                base_ranked = _history_recency_rank(history, args.history_selection)
                ranked, rerank_debug = _att_rerank_replacement(
                    base_ranked,
                    head_rankings,
                    args.hybrid_top_k,
                    args.att_rerank_pool,
                    args.att_rerank_min_heads,
                    args.att_rerank_min_margin,
                    args.att_rerank_min_score_gain,
                )
            else:
                rerank_debug = None
            selected_full_indices = ranked[: args.hybrid_top_k]
            router_debug = {
                "history_router_strategy": mode,
                "history_attention_cache_mode": cache_mode,
                "history_attention_score_mode": args.attention_router_score_mode,
                "history_attention_router_sec": round(router_sec, 4),
                "selected_history_doc_indices": selected_full_indices,
                "history_attention_top_scores": [
                    round(scores[index], 8) for index in selected_full_indices
                ],
                "att_rerank_debug": rerank_debug,
                "att_rerank_replaced": (
                    bool(rerank_debug and rerank_debug.get("accepted"))
                    if mode == "hybrid_fullkv_att_rerank" else None
                ),
            }
        elif mode == "c2kv_hybrid":
            mode = "c2kv_hybrid"
        prefix, skip_reason = _build_hybrid_prefix(
            model,
            tokenizer,
            example,
            args,
            recent_full_docs,
            selected_full_indices=selected_full_indices,
            router_debug=router_debug,
        )
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
        "history_turns": prefix.get("doc_chunks", 0),
        "decision_step": prefix.get("doc_chunks", 0) + 1,
        "kept_history_tokens": prefix.get("kept_history_tokens", 0),
        "gist_tokens": prefix.get("gist_tokens", 0),
        "compressed_history_tokens": prefix.get(
            "compressed_history_tokens",
            max(0, prefix.get("cache_length", 0) - prefix.get("system_length", 0)),
        ),
        "cache_tokens": prefix.get("cache_length", 0),
        "input_tokens": prefix.get("cache_length", 0) + row.get("prompt_tokens", 0),
        "actual_compression_ratio": round(prefix.get("actual_compression_ratio", 0.0), 4),
        "system_prefill_sec": round(prefix.get("system_prefill_sec", 0.0), 4),
        "full_prefill_sec": round(prefix.get("full_prefill_sec", 0.0), 4),
        "tool_compress_sec": round(prefix.get("tool_compress_sec", 0.0), 4),
        "blend_sec": round(prefix.get("blend_sec", 0.0), 4),
        "ttft_sec": round(ttft_sec, 4),
        "total_sec": round(time.perf_counter() - total_start, 4),
    })
    for key in (
        "full_history_docs",
        "rest_history_docs",
        "top_full_tokens",
        "rest_history_tokens",
        "split_logical_length",
        "split_cache_length",
        "split_cache_length_matches_logical",
        "split_total_prompt_tokens",
        "split_current_tokens",
        "split_system_tokens",
        "sequential_logical_length",
        "sequential_cache_length",
        "sequential_cache_length_matches_logical",
        "sequential_total_prompt_tokens",
        "sequential_current_tokens",
        "sequential_system_tokens",
        "sequential_reconstructed_tokens",
        "sequential_ids_equal_reconstructed",
        "sequential_first_token_diff_index",
        "contiguous_history_docs_input",
        "contiguous_original_history_tokens",
        "contiguous_history_truncated",
        "contiguous_total_prompt_tokens",
        "contiguous_current_tokens",
        "contiguous_system_tokens",
        "contiguous_reconstructed_tokens",
        "contiguous_ids_equal_reconstructed",
        "contiguous_first_token_diff_index",
        "history_router_strategy",
        "history_attention_cache_mode",
        "history_attention_score_mode",
        "history_attention_router_sec",
        "selected_history_doc_indices",
        "history_attention_top_scores",
        "att_rerank_debug",
        "att_rerank_replaced",
        "turn_ablation_mode",
        "turn_position",
        "turn_window_size",
        "turn_history_doc_tokens",
        "turn_full_kv_doc_tokens",
        "avg_turn_original_tokens",
        "avg_turn_full_kv_tokens",
        "ablation_turn_original_tokens",
        "full_restore_added_kv_tokens",
        "fixed_recent_full_tokens",
        "raw_history_source",
        "raw_history_window",
        "raw_history_docs",
        "raw_current_messages",
        "raw_current_tokens",
        "raw_doc_tokens",
        "raw_history_doc_texts",
        "raw_history_doc_messages",
        "raw_current_text",
        "raw_target_source",
        "raw_target_has_tool_call",
        "compressed_history_tokens",
        "cache_tokens",
        "input_tokens",
        "history_compression_mode",
        "recompress_chunks",
        "selected_history_doc_indices",
    ):
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
        compressed_history_total = sum(row.get("compressed_history_tokens", 0) for row in valid_rows)
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
            "avg_turn_original_tokens": (
                sum(row.get("ablation_turn_original_tokens", row.get("avg_turn_original_tokens", 0.0)) for row in valid_rows)
                / len(valid_rows)
                if valid_rows and any(
                    "ablation_turn_original_tokens" in row or "avg_turn_original_tokens" in row
                    for row in valid_rows
                ) else 0.0
            ),
            "avg_full_restore_added_kv_tokens": (
                sum(row.get("full_restore_added_kv_tokens", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows and any("full_restore_added_kv_tokens" in row for row in valid_rows) else 0.0
            ),
            "turn_position": next(
                (row.get("turn_position") for row in valid_rows if row.get("turn_position") is not None),
                None,
            ),
            "avg_prompt_tokens": (
                sum(row.get("prompt_tokens", 0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_input_tokens": (
                sum(row.get("input_tokens", row.get("prompt_tokens", 0)) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_cache_tokens": (
                sum(row.get("cache_tokens", 0) for row in valid_rows) / len(valid_rows)
                if valid_rows and any("cache_tokens" in row for row in valid_rows) else 0.0
            ),
            "avg_compressed_history_tokens": (
                compressed_history_total / len(valid_rows)
                if valid_rows and any("compressed_history_tokens" in row for row in valid_rows) else 0.0
            ),
            "avg_generated_tokens": (
                generated_total / len(valid_rows) if valid_rows else 0.0
            ),
            "avg_actual_compression_ratio": (
                sum(row.get("actual_compression_ratio", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "token_weighted_actual_compression_ratio": (
                sum(row.get("doc_tokens", 0) for row in valid_rows) / compressed_history_total
                if compressed_history_total else 0.0
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


def _decision_step_bucket(step: int) -> str:
    if step <= 2:
        return "turn_1_2"
    if step <= 4:
        return "turn_3_4"
    if step <= 8:
        return "turn_5_8"
    return "turn_9_plus"


def _summarize_metric_group(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid_rows = [row for row in rows if not row.get("skipped")]
    called = [row for row in valid_rows if row.get("has_tool_call")]
    compressed_history_total = sum(row.get("compressed_history_tokens", 0) for row in valid_rows)
    return {
        "num_examples": len(rows),
        "num_valid": len(valid_rows),
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
        "tool_name_accuracy": (
            sum(1 for row in valid_rows if row.get("tool_name_match")) / len(valid_rows)
            if valid_rows else 0.0
        ),
        "tool_call_rate": (
            len(called) / len(valid_rows) if valid_rows else 0.0
        ),
        "call_accuracy": (
            sum(1 for row in called if row.get("tool_name_match")) / len(called)
            if called else 0.0
        ),
        "avg_doc_tokens": (
            sum(row.get("doc_tokens", 0) for row in valid_rows) / len(valid_rows)
            if valid_rows else 0.0
        ),
        "avg_compressed_history_tokens": (
            compressed_history_total / len(valid_rows)
            if valid_rows and any("compressed_history_tokens" in row for row in valid_rows) else 0.0
        ),
        "token_weighted_actual_compression_ratio": (
            sum(row.get("doc_tokens", 0) for row in valid_rows) / compressed_history_total
            if compressed_history_total else 0.0
        ),
    }


def _summarize_turn_buckets(args: argparse.Namespace, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries = []
    keys = sorted({(row.get("mode"), row.get("ratio")) for row in rows if row.get("decision_step") is not None})
    for mode, ratio in keys:
        group = [row for row in rows if row.get("mode") == mode and row.get("ratio") == ratio]
        bucket_payload: Dict[str, Any] = {}
        for bucket in ("turn_1_2", "turn_3_4", "turn_5_8", "turn_9_plus"):
            bucket_rows = [
                row for row in group
                if _decision_step_bucket(int(row.get("decision_step") or 1)) == bucket
            ]
            bucket_payload[bucket] = _summarize_metric_group(bucket_rows)
        bucket_payload["overall"] = _summarize_metric_group(group)
        summaries.append({
            "model": args.model,
            "base_model": args.base_model,
            "dataset_path": args.dataset_path,
            "split": args.split,
            "mode": mode,
            "ratio": ratio,
            "buckets": bucket_payload,
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
        prefix_history_doc_num=args.prefix_history_doc_num,
        prefix_history_exact=args.prefix_history_exact,
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
        run_ratios = [1] if mode in {"full", "history_full", "all_full", *FULL_PROMPT_MODES, *SPLIT_FULL_MODES} else ratios
        model_args = copy.copy(args)
        if (
            mode in HYBRID_MODES
            or mode in C2KV_MODES
            or mode in DECISION_PREFIX_MODES
            or mode in {
                "raw_first15_c2kv",
                "raw_first15_hybrid",
                "raw_first15_full_same_model",
                "raw_first8_c2kv",
                "raw_first8_hybrid",
                "raw_first8_full_same_model",
                "raw_prefix8_exact_c2kv",
                "raw_prefix8_exact_hybrid",
                "raw_prefix8_exact_full_same_model",
                "raw_prefix_next_c2kv",
                "raw_prefix_next_hybrid",
                "raw_prefix_next_full_same_model",
            }
            or (mode in TURN_ABLATION_MODES and mode != "all_full")
        ):
            model_args.mode = "c2kv"
        elif mode in TRUNCATE_MODES:
            model_args.mode = "truncate"
        elif mode in FULL_PROMPT_MODES or mode in SPLIT_FULL_MODES or mode == "all_full":
            model_args.mode = "full"
        else:
            model_args.mode = mode
        if (
            mode in {"full", "all_full", *FULL_PROMPT_MODES, *SPLIT_FULL_MODES, *TRUNCATE_MODES}
            and mode not in DECISION_PREFIX_MODES
            and mode not in {
                "raw_first15_full_same_model",
                "raw_first8_full_same_model",
                "raw_prefix8_exact_full_same_model",
                "raw_prefix_next_full_same_model",
            }
            and args.base_model
        ):
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
        "raw_first_n_turns": args.raw_first_n_turns,
        "raw_prefix_n_turns": args.raw_prefix_n_turns,
        "raw_prefix_next_target_scope": args.raw_prefix_next_target_scope,
        "selection_skips": selection_skips,
        "num_rows": len(rows),
        "results": _summarize_rows(args, rows),
        "turn_bucket_results": _summarize_turn_buckets(args, rows),
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
    parser.add_argument(
        "--mode",
        choices=[
            "full",
            "truncate",
            "tail_truncate",
            "c2kv",
            "split_c2kv",
            "contiguous_history_c2kv",
            "hybrid",
            "c2kv_hybrid",
            "att_hybrid",
            "att_fullkv_hybrid",
            "hybrid_fullkv_att_rerank",
            "all_full",
            "all_c2kv",
            "recent2_full_rest_c2kv",
            "turn1_full_rest_c2kv",
            "turn2_full_rest_c2kv",
            "turn3_full_rest_c2kv",
            "turn4_full_rest_c2kv",
            "turn5_full_rest_c2kv",
            "turn6_full_rest_c2kv",
            "turn7_full_rest_c2kv",
            "turn8_full_rest_c2kv",
            "turn9_full_rest_c2kv",
            "turn10_full_rest_c2kv",
            "turn11_full_rest_c2kv",
            "turn12_full_rest_c2kv",
            "turn13_full_rest_c2kv",
            "recent1_hybrid",
            "recent2_hybrid",
            "original_replay_full",
            "reconstructed_contiguous_full",
            "split_full_kv",
            "sequential_full_kv",
            "current_only",
            "history_full",
            "history_all_c2kv4",
            "recent2_full_rest_c2kv4",
            "each_turn_independent_c2kv4",
            "recompress_all_every_turn_c2kv4",
            "raw_first15_full",
            "raw_first15_full_same_model",
            "raw_first15_c2kv",
            "raw_first15_hybrid",
            "raw_first8_full",
            "raw_first8_full_same_model",
            "raw_first8_c2kv",
            "raw_first8_hybrid",
            "raw_prefix8_exact_full",
            "raw_prefix8_exact_full_same_model",
            "raw_prefix8_exact_c2kv",
            "raw_prefix8_exact_hybrid",
            "raw_prefix_next_full",
            "raw_prefix_next_full_same_model",
            "raw_prefix_next_c2kv",
            "raw_prefix_next_hybrid",
        ],
        default="c2kv",
    )
    parser.add_argument("--compare_modes", default="full,truncate,c2kv,hybrid")
    parser.add_argument("--ratios", default="4")
    parser.add_argument("--override_ratio", type=int, default=4)
    parser.add_argument("--hybrid_top_k", type=int, default=3)
    parser.add_argument("--attention_router_layers", type=int, default=32)
    parser.add_argument("--attention_router_attn_impl", default="eager")
    parser.add_argument("--attention_router_max_query_tokens", type=int, default=512)
    parser.add_argument(
        "--attention_router_score_mode",
        choices=["mean", "sqrt_len", "sum"],
        default="mean",
    )
    parser.add_argument("--att_rerank_pool", type=int, default=10)
    parser.add_argument("--att_rerank_min_heads", type=int, default=30)
    parser.add_argument("--att_rerank_min_margin", type=float, default=0.0)
    parser.add_argument("--att_rerank_min_score_gain", type=float, default=0.0)
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
    parser.add_argument(
        "--raw_first_n_turns",
        type=int,
        default=15,
        help=(
            "Number of initial raw turn documents for raw_first15_* modes. "
            "raw_first8_* aliases force this to 8."
        ),
    )
    parser.add_argument(
        "--raw_prefix_n_turns",
        type=int,
        default=8,
        help=(
            "Number of initial raw turn documents for raw_prefix_next_* modes. "
            "The evaluation current/answer are rebuilt from turn n+1."
        ),
    )
    parser.add_argument(
        "--raw_prefix_next_target_scope",
        choices=["turn", "first_assistant"],
        default="turn",
        help=(
            "Target construction for raw_prefix_next_* modes. 'turn' keeps the "
            "entire rebuilt next turn after the user message; 'first_assistant' "
            "uses only the first assistant message after that user message."
        ),
    )
    parser.add_argument(
        "--dump_raw_history_docs",
        action="store_true",
        help="Write raw per-turn document contents and current messages into each JSONL row.",
    )
    parser.add_argument("--raw_history_doc_debug_chars", type=int, default=2000)
    parser.add_argument("--min_target_tokens", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--require_tool_call", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--max_input_chars", type=int)
    parser.add_argument("--max_answer_chars", type=int)
    parser.add_argument("--include_tools", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--prefix_history_doc_num", type=int)
    parser.add_argument("--prefix_history_exact", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--split_oversized_history_docs", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--hybrid_full_after_c2kv", type=lambda x: str(x).lower() == "true", default=False)
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
