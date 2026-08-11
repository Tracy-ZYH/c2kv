from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import types
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from tqdm import tqdm
from transformers import AutoTokenizer

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python" / "inference"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if "jieba" not in sys.modules:
    try:
        __import__("jieba")
    except ModuleNotFoundError:
        jieba_stub = types.ModuleType("jieba")
        jieba_stub.cut = lambda text, *args, **kwargs: list(str(text))
        jieba_stub.lcut = lambda text, *args, **kwargs: list(str(text))
        sys.modules["jieba"] = jieba_stub
if "fuzzywuzzy" not in sys.modules:
    try:
        __import__("fuzzywuzzy")
    except ModuleNotFoundError:
        fuzzywuzzy_stub = types.ModuleType("fuzzywuzzy")
        fuzz_stub = types.SimpleNamespace(ratio=lambda left, right: 0)
        fuzzywuzzy_stub.fuzz = fuzz_stub
        sys.modules["fuzzywuzzy"] = fuzzywuzzy_stub
        sys.modules["fuzzywuzzy.fuzz"] = fuzz_stub
if "rouge" not in sys.modules:
    try:
        __import__("rouge")
    except ModuleNotFoundError:
        rouge_stub = types.ModuleType("rouge")

        class _Rouge:
            def get_scores(self, *args, **kwargs):
                return [{"rouge-l": {"f": 0.0}}]

        rouge_stub.Rouge = _Rouge
        sys.modules["rouge"] = rouge_stub

from train.train_data_multiturn import (  # noqa: E402
    AgentLLMTracesCompressHistorySource,
    CompressHistoryExample,
    _chat_template_ids,
    _fit_reused_history,
    _normal_chat_message,
)


HTTP = requests.Session()
HTTP.trust_env = False


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _text_tokens(text: str) -> List[str]:
    return re.findall(r"\w+", _normalize_text(text).lower())


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


def _extract_tool_name(text: str) -> Optional[str]:
    blocks = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text or "", flags=re.S)
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
            if name:
                return str(name)
    match = re.search(r'"(?:name|tool_name|function_name)"\s*:\s*"([^"]+)"', text or "")
    return match.group(1) if match else None


def _has_tool_call(text: str) -> bool:
    return "<tool_call>" in (text or "") or "Action:" in (text or "")


def _post_json(base_url: str, path: str, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    response = HTTP.post(f"{base_url.rstrip('/')}{path}", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _chat_completion(
    base_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    max_new_tokens: int,
    timeout: int,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> str:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_completion_tokens": max_new_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if tools:
        payload["tools"] = tools
    data = _post_json(base_url, "/v1/chat/completions", payload, timeout)
    content = data["choices"][0]["message"].get("content")
    return content if isinstance(content, str) else ""


def _extract_document(
    base_url: str,
    message: Dict[str, Any],
    ratio: int,
    timeout: int,
) -> Dict[str, Any]:
    return _post_json(
        base_url,
        "/v1/c2kv/extract",
        {
            "text": str(message.get("content") or ""),
            "compression_ratio": ratio,
            "role": message.get("role") or "user",
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout,
    )


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _history_messages(
    tokenizer: Any,
    example: CompressHistoryExample,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], int, Optional[str]]:
    raw_history = [
        _normal_chat_message(message)
        for message in example.history_messages
        if message.get("content")
    ]
    history = _fit_reused_history(
        tokenizer,
        raw_history,
        max_doc_length=args.max_doc_length,
        max_doc_num=args.max_doc_num,
        policy=args.history_selection,
        split_oversized_history_docs=args.split_oversized_history_docs,
    )
    if len(history) < args.min_doc_num:
        return [], 0, f"history_docs<{args.min_doc_num}"
    if len(history) > args.max_doc_num:
        return [], 0, f"history_docs>{args.max_doc_num}"
    token_counts = [
        len(_chat_template_ids(tokenizer, [message], max_length=args.max_doc_length))
        for message in history
    ]
    total_tokens = sum(token_counts)
    if total_tokens > args.max_history_tokens:
        return [], total_tokens, f"history_tokens>{args.max_history_tokens}"
    return history, total_tokens, None


def _current_messages(example: CompressHistoryExample) -> List[Dict[str, Any]]:
    return [
        _normal_chat_message(message)
        for message in example.current_messages
        if message.get("content") or message.get("role") == "assistant"
    ]


def _prompt_token_count(tokenizer: Any, messages: List[Dict[str, Any]]) -> int:
    return len(_chat_template_ids(tokenizer, messages, add_generation_prompt=True))


def _build_messages(
    base_url: str,
    tokenizer: Any,
    example: CompressHistoryExample,
    args: argparse.Namespace,
) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, Any], Optional[str]]:
    history, doc_tokens, skip_reason = _history_messages(tokenizer, example, args)
    if skip_reason is not None:
        return None, {"doc_tokens": doc_tokens}, skip_reason
    current = _current_messages(example)
    if not current:
        return None, {"doc_tokens": doc_tokens}, "empty_current"

    extract_records: List[Dict[str, Any]] = []
    history_messages: List[Dict[str, Any]] = []
    full_history = history
    compressed_history: List[Dict[str, Any]] = []

    if args.mode == "full":
        history_messages = list(full_history)
    elif args.mode == "c2kv":
        compressed_history = list(history)
    elif args.mode == "hybrid":
        full_count = min(args.hybrid_top_k, len(history))
        if args.history_selection == "tail":
            compressed_history = list(history[:-full_count]) if full_count else list(history)
            history_messages.extend(history[-full_count:] if full_count else [])
        else:
            history_messages.extend(history[:full_count])
            compressed_history = list(history[full_count:])
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    for message in compressed_history:
        start = time.perf_counter()
        result = _extract_document(base_url, message, args.ratio, args.timeout)
        result["extract_seconds"] = time.perf_counter() - start
        extract_records.append(result)
        if not result.get("success", True) or not result.get("key_hash"):
            return None, {"doc_tokens": doc_tokens, "extracts": extract_records}, (
                result.get("error") or "extract_failed"
            )
        history_messages.append(
            {
                "role": message.get("role") or "user",
                "content": str(message.get("content") or ""),
                "c2kv_key_hash": result["key_hash"],
            }
        )

    messages = [{"role": "system", "content": example.system_prompt}]
    messages.extend(history_messages)
    messages.extend(current)

    prompt_tokens = _prompt_token_count(tokenizer, current)
    full_input_tokens = (
        len(
            _chat_template_ids(
                tokenizer,
                [{"role": "system", "content": example.system_prompt}, *history, *current],
                tools=example.tools or None,
                add_generation_prompt=True,
            )
        )
        if args.mode == "full"
        else None
    )
    if (
        args.mode == "full"
        and args.max_baseline_input_tokens
        and full_input_tokens is not None
        and full_input_tokens > args.max_baseline_input_tokens
    ):
        return None, {"doc_tokens": doc_tokens, "input_tokens": full_input_tokens}, (
            f"baseline_input_tokens>{args.max_baseline_input_tokens}"
        )

    gist_tokens = sum(
        int(record.get("gist_len") or 0)
        for record in extract_records
        if isinstance(record.get("gist_len"), int)
    )
    original_tokens = sum(
        int(record.get("original_seq_len") or 0)
        for record in extract_records
        if isinstance(record.get("original_seq_len"), int)
    )
    full_history_tokens = doc_tokens - original_tokens if args.mode == "hybrid" else doc_tokens
    compressed_history_tokens = gist_tokens + (
        full_history_tokens if args.mode == "hybrid" else 0
    )

    meta = {
        "doc_tokens": doc_tokens,
        "doc_chunks": len(history),
        "history_turns": len(history),
        "prompt_tokens": prompt_tokens,
        "input_tokens": full_input_tokens,
        "extracts": extract_records,
        "num_extracts": len(extract_records),
        "extract_original_tokens": original_tokens,
        "gist_tokens": gist_tokens,
        "compressed_history_tokens": compressed_history_tokens,
        "full_history_docs": len(history_messages) - len(compressed_history),
        "rest_history_docs": len(compressed_history),
        "actual_compression_ratio": (
            doc_tokens / compressed_history_tokens
            if compressed_history_tokens
            else 1.0
        ),
    }
    return messages, meta, None


def _generate_one(
    tokenizer: Any,
    example: CompressHistoryExample,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    total_start = time.perf_counter()
    messages, meta, skip_reason = _build_messages(args.base_url, tokenizer, example, args)
    if messages is None:
        return {
            "qid": example.qid,
            "session_id": example.qid.rsplit(":", 1)[0] if ":" in example.qid else None,
            "mode": args.mode,
            "ratio": args.ratio,
            "skipped": True,
            "skip_reason": skip_reason,
            **meta,
        }
    chat_start = time.perf_counter()
    try:
        prediction = _chat_completion(
            args.base_url,
            args.model,
            messages,
            args.max_new_tokens,
            args.timeout,
            tools=example.tools if args.include_tools and example.tools else None,
        )
        chat_error = None
    except Exception as error:
        prediction = ""
        chat_error = f"{type(error).__name__}: {error}"
    chat_seconds = time.perf_counter() - chat_start

    target = example.answer.strip()
    target_tool = _extract_tool_name(target)
    pred_tool = _extract_tool_name(prediction)
    return {
        "qid": example.qid,
        "session_id": example.qid.rsplit(":", 1)[0] if ":" in example.qid else None,
        "mode": args.mode,
        "ratio": args.ratio,
        "skipped": bool(chat_error),
        "skip_reason": chat_error,
        **meta,
        "target_tool_name": target_tool,
        "prediction_tool_name": pred_tool,
        "tool_name_match": target_tool is not None and target_tool == pred_tool,
        "has_tool_call": _has_tool_call(prediction),
        "response_type_match": _has_tool_call(target) == _has_tool_call(prediction),
        "exact_match": _normalize_text(prediction) == _normalize_text(target),
        "text_token_f1": _text_token_f1(target, prediction),
        "rouge_l_f1": _rouge_l_f1(target, prediction),
        "prediction": prediction,
        "target": target,
        "chat_seconds": round(chat_seconds, 4),
        "total_seconds": round(time.perf_counter() - total_start, 4),
    }


def _load_examples(args: argparse.Namespace, tokenizer: Any) -> Tuple[List[CompressHistoryExample], Dict[str, int]]:
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
    examples: List[CompressHistoryExample] = []
    for example in source:
        if args.selection_filter == "c2kv":
            _history, _tokens, skip_reason = _history_messages(tokenizer, example, args)
            if skip_reason is not None:
                selection_skips[skip_reason] += 1
                continue
        examples.append(example)
        if args.max_examples and len(examples) >= args.max_examples:
            break
    return examples, dict(selection_skips)


def _summarize(args: argparse.Namespace, rows: List[Dict[str, Any]], selection_skips: Dict[str, int]) -> Dict[str, Any]:
    valid = [row for row in rows if not row.get("skipped")]
    extract_records = [
        record
        for row in valid
        for record in row.get("extracts", [])
        if isinstance(record, dict)
    ]
    return {
        "base_url": args.base_url,
        "model": args.model,
        "dataset_path": args.dataset_path,
        "split": args.split,
        "mode": args.mode,
        "ratio": args.ratio,
        "num_examples": len(rows),
        "num_valid": len(valid),
        "num_skipped": len(rows) - len(valid),
        "selection_skips": selection_skips,
        "skip_reasons": dict(Counter(row.get("skip_reason", "unknown") for row in rows if row.get("skipped"))),
        "tool_name_match": (
            sum(1 for row in valid if row.get("tool_name_match")) / len(valid)
            if valid else 0.0
        ),
        "exact_match": (
            sum(1 for row in valid if row.get("exact_match")) / len(valid)
            if valid else 0.0
        ),
        "response_type_match": (
            sum(1 for row in valid if row.get("response_type_match")) / len(valid)
            if valid else 0.0
        ),
        "text_token_f1": (
            sum(float(row.get("text_token_f1", 0.0) or 0.0) for row in valid) / len(valid)
            if valid else 0.0
        ),
        "rouge_l_f1": (
            sum(float(row.get("rouge_l_f1", 0.0) or 0.0) for row in valid) / len(valid)
            if valid else 0.0
        ),
        "avg_actual_compression_ratio": (
            sum(float(row.get("actual_compression_ratio", 0.0) or 0.0) for row in valid) / len(valid)
            if valid else 0.0
        ),
        "num_extracts": len(extract_records),
        "extract_success_rate": (
            sum(1 for record in extract_records if record.get("success", True)) / len(extract_records)
            if extract_records else None
        ),
        "avg_chat_seconds": (
            sum(float(row.get("chat_seconds", 0.0) or 0.0) for row in valid) / len(valid)
            if valid else 0.0
        ),
        "avg_total_seconds": (
            sum(float(row.get("total_seconds", 0.0) or 0.0) for row in valid) / len(valid)
            if valid else 0.0
        ),
    }


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    tokenizer_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        local_files_only=True,
        padding_side="right",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    examples, selection_skips = _load_examples(args, tokenizer)
    rows = [_generate_one(tokenizer, example, args) for example in tqdm(examples, desc=args.mode)]
    summary = _summarize(args, rows, selection_skips)
    if args.output_file:
        _write_jsonl(args.output_file, rows)
        summary_path = str(Path(args.output_file).with_suffix(".summary.json"))
        Path(summary_path).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multi-turn history C2KV through SGLang HTTP API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="qwen3-history-c2kv")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--dataset-path", default="./datasets/agent-llm-traces")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--split", choices=["train", "eval"], default="eval")
    parser.add_argument("--mode", choices=["full", "c2kv", "hybrid"], required=True)
    parser.add_argument("--ratio", type=int, default=4)
    parser.add_argument("--hybrid-top-k", type=int, default=3)
    parser.add_argument("--history-selection", choices=["head", "tail"], default="tail")
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument("--max-source-examples", type=int)
    parser.add_argument("--selection-filter", choices=["c2kv", "none"], default="c2kv")
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-manifest-file")
    parser.add_argument("--split-manifest-name", default="subset_disjoint")
    parser.add_argument("--max-samples-per-session", type=int, default=4)
    parser.add_argument("--max-doc-length", type=int, default=768)
    parser.add_argument("--min-doc-num", type=int, default=1)
    parser.add_argument("--max-doc-num", type=int, default=16)
    parser.add_argument("--max-history-tokens", type=int, default=12288)
    parser.add_argument("--max-prompt-tokens", type=int, default=1536)
    parser.add_argument("--max-baseline-input-tokens", type=int, default=16000)
    parser.add_argument("--min-target-tokens", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--require-tool-call", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--include-tools", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--max-input-chars", type=int)
    parser.add_argument("--max-answer-chars", type=int)
    parser.add_argument("--prefix-history-doc-num", type=int)
    parser.add_argument("--prefix-history-exact", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--split-oversized-history-docs", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--timeout", type=int, default=900)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(evaluate(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
