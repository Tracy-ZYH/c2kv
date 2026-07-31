from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import pyarrow.parquet as pq


Message = Dict[str, Any]


def _json_loads(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _find_parquet_files(path: Path) -> List[Path]:
    if path.is_file() and path.suffix == ".parquet":
        return [path]
    roots = [path / "data", path]
    files: List[Path] = []
    for root in roots:
        if root.is_dir():
            files = sorted(root.glob("*.parquet"))
            if not files:
                files = sorted(root.rglob("*.parquet"))
        if files:
            break
    return files


def _iter_rows(data_files: Iterable[Path]) -> Iterator[Dict[str, Any]]:
    wanted = ["benchmark", "session_id", "trace_id", "id", "spans"]
    for data_file in data_files:
        pf = pq.ParquetFile(data_file)
        available = set(pf.schema_arrow.names)
        columns = [column for column in wanted if column in available]
        for batch in pf.iter_batches(batch_size=256, columns=columns):
            yield from batch.to_pylist()


def _span_attributes(span: Any) -> Dict[str, Any]:
    span = _json_loads(span, span)
    if not isinstance(span, dict):
        return {}
    attributes = span.get("attributes", span)
    attributes = _json_loads(attributes, attributes)
    return attributes if isinstance(attributes, dict) else {}


def _sort_spans(spans: Sequence[Any]) -> List[Dict[str, Any]]:
    return sorted(
        [span for span in spans if isinstance(span, dict)],
        key=lambda span: (
            span.get("start_time") or "",
            span.get("span_id") or "",
        ),
    )


def _message_parts(message: Message) -> List[Dict[str, Any]]:
    parts = message.get("parts")
    parts = _json_loads(parts, parts)
    if isinstance(parts, dict):
        return [parts]
    if isinstance(parts, list):
        return [part for part in parts if isinstance(part, dict)]
    return []


def _render_tool_calls(tool_calls: Any) -> str:
    tool_calls = _json_loads(tool_calls, [])
    if isinstance(tool_calls, dict):
        tool_calls = [tool_calls]
    if not isinstance(tool_calls, list):
        return ""
    rendered = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        if call.get("type") not in (None, "tool_call", "function_call") and "function" not in call:
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = (
            function.get("name")
            or call.get("name")
            or call.get("tool_name")
            or call.get("function_name")
            or ""
        )
        arguments = (
            function.get("arguments")
            or call.get("arguments")
            or call.get("args")
            or call.get("input")
            or {}
        )
        rendered.append("<tool_call>\n" + _json_dumps({"name": name, "arguments": arguments}) + "\n</tool_call>")
    return "\n".join(rendered)


def _message_content_to_text(message: Message) -> str:
    content = message.get("content", "")
    if not content and _message_parts(message):
        content = _message_parts(message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type in ("tool_call", "function_call"):
                    continue
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(str(item["content"]))
                else:
                    parts.append(_json_dumps(item))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(parts)
    if content is None:
        return ""
    return _json_dumps(content)


def _normal_message(message: Message) -> Optional[Message]:
    if not isinstance(message, dict):
        return None
    role = message.get("role") or message.get("type") or "user"
    if role == "tool":
        role = "user"
    content_parts = []
    content = _message_content_to_text(message)
    if content:
        content_parts.append(content)
    tool_calls_text = _render_tool_calls(
        message.get("tool_calls")
        or message.get("toolCalls")
        or message.get("function_call")
        or _message_parts(message)
    )
    if tool_calls_text:
        content_parts.append("Action:\n" + tool_calls_text)
    if not content_parts and role != "assistant":
        return None
    return {"role": role, "content": "\n\n".join(content_parts)}


def _render_output_messages(value: Any) -> tuple[str, bool]:
    messages = _json_loads(value, [])
    if isinstance(messages, dict):
        messages = [messages]
    if not isinstance(messages, list):
        messages = [{"role": "assistant", "content": str(messages)}]

    rendered_messages: List[str] = []
    has_tool_call = False
    for message in messages:
        if not isinstance(message, dict):
            if message:
                rendered_messages.append(str(message))
            continue
        parts = []
        reasoning = (
            message.get("reasoning_content")
            or message.get("reasoning")
            or message.get("thought")
            or message.get("thinking")
            or message.get("cot")
            or ""
        )
        if reasoning:
            parts.append("Thought:\n" + str(reasoning).strip())
        content = _message_content_to_text(message).strip()
        if content:
            parts.append(content)
        tool_calls_text = _render_tool_calls(
            message.get("tool_calls")
            or message.get("toolCalls")
            or message.get("function_call")
            or _message_parts(message)
        )
        if tool_calls_text:
            parts.append("Action:\n" + tool_calls_text)
        has_tool_call = has_tool_call or bool(tool_calls_text)
        rendered = "\n\n".join(part for part in parts if part).strip()
        if rendered:
            rendered_messages.append(rendered)
    answer = "\n\n".join(rendered_messages).strip()
    marker_text = answer.lower()
    has_tool_call = has_tool_call or any(
        marker in marker_text
        for marker in ("<tool_call>", "action:", "function_call", "tool call")
    )
    return answer, has_tool_call


def _last_user_message(messages: Sequence[Message]) -> Optional[Message]:
    for message in reversed(messages):
        if message.get("role") == "user" and message.get("content"):
            return message
    return None


def _session_id(row: Dict[str, Any], row_index: int) -> str:
    return str(row.get("session_id") or row.get("trace_id") or row.get("id") or f"row-{row_index}")


def _stat(values: List[int]) -> Dict[str, Any]:
    if not values:
        return {"min": 0, "avg": 0.0, "p50": 0, "p90": 0, "p95": 0, "max": 0}
    ordered = sorted(values)

    def pct(p: float) -> int:
        return ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * p)))]

    return {
        "min": ordered[0],
        "avg": round(float(statistics.mean(ordered)), 4),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "max": ordered[-1],
    }


class TokenCounter:
    def __init__(self, tokenizer_path: Optional[str]) -> None:
        self.tokenizer = None
        if tokenizer_path:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True,
                local_files_only=True,
                padding_side="right",
            )

    @property
    def mode(self) -> str:
        return "chat_template_tokenizer" if self.tokenizer is not None else "rough_char_div_4"

    def count_message(self, role: str, content: str) -> int:
        if self.tokenizer is None:
            return max(1, len(content) // 4)
        from train.train_data_multiturn import _chat_template_ids

        return len(_chat_template_ids(self.tokenizer, [{"role": role, "content": content}]))

    def count_text(self, text: str) -> int:
        if self.tokenizer is None:
            return max(1, len(text) // 4)
        return len(self.tokenizer.encode(text, add_special_tokens=False))


def _turn_doc(user_message: Message, assistant_output: str, include_tool_output: bool) -> str:
    output = assistant_output
    if not include_tool_output:
        # Keep the assistant action/answer, but avoid storing huge tool observations if
        # they were serialized into assistant text by a harness.
        marker = "\nObservation:"
        if marker in output:
            output = output.split(marker, 1)[0].rstrip()
    return (
        "Previous turn\n"
        "[User query]\n"
        f"{user_message.get('content', '').strip()}\n\n"
        "[Assistant output]\n"
        f"{output.strip()}"
    ).strip()


def inspect(args: argparse.Namespace) -> Dict[str, Any]:
    data_files = _find_parquet_files(Path(args.dataset_path))
    if not data_files:
        raise FileNotFoundError(f"No parquet files found under {args.dataset_path}")
    counter = TokenCounter(args.tokenizer)

    groups: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "sessions": 0,
            "llm_spans": 0,
            "turn_docs": 0,
            "candidate_samples": 0,
            "candidate_with_tool_call": 0,
            "history_docs_per_sample": [],
            "history_chars_per_sample": [],
            "history_tokens_per_sample": [],
            "turn_doc_chars": [],
            "turn_doc_tokens": [],
            "current_query_chars": [],
            "current_query_tokens": [],
            "answer_chars": [],
            "answer_tokens": [],
            "input_messages_per_span": [],
            "consecutive_prefix_ratio_per_mille": [],
            "consecutive_pairs": 0,
            "prev_input_is_prefix": 0,
            "kept_docs_per_sample": [],
            "kept_history_tokens_per_sample": [],
            "overflow_doc_tokens": 0,
            "skip_reasons": Counter(),
        }
    )

    for row_index, row in enumerate(_iter_rows(data_files)):
        subset = str(row.get("benchmark") or "unknown")
        session_id = _session_id(row, row_index)
        spans = _sort_spans(_json_loads(row.get("spans"), row.get("spans")) or [])
        group = groups[subset]
        group["sessions"] += 1
        prior_turn_docs: List[tuple[str, int]] = []
        previous_input_messages: Optional[List[Message]] = None
        sample_count = 0
        for span_index, span in enumerate(spans):
            attributes = _span_attributes(span)
            input_messages = _json_loads(attributes.get("gen_ai.input.messages"), [])
            output_messages = attributes.get("gen_ai.output.messages")
            if not input_messages or output_messages is None:
                group["skip_reasons"]["missing_input_or_output"] += 1
                continue
            normalized_messages = [
                item
                for item in (_normal_message(message) for message in _json_loads(input_messages, []))
                if item is not None and item.get("role") != "system"
            ]
            current_user = _last_user_message(normalized_messages)
            answer, has_tool_call = _render_output_messages(output_messages)
            if current_user is None or not answer:
                group["skip_reasons"]["missing_current_user_or_answer"] += 1
                continue
            group["llm_spans"] += 1
            group["input_messages_per_span"].append(len(normalized_messages))
            if previous_input_messages is not None:
                prefix_len = 0
                for prev_message, current_message in zip(previous_input_messages, normalized_messages):
                    if prev_message != current_message:
                        break
                    prefix_len += 1
                previous_len = max(1, len(previous_input_messages))
                group["consecutive_pairs"] += 1
                group["prev_input_is_prefix"] += int(prefix_len == len(previous_input_messages))
                group["consecutive_prefix_ratio_per_mille"].append(
                    int(round(1000 * prefix_len / previous_len))
                )

            if len(prior_turn_docs) >= args.min_history_docs:
                if args.max_samples_per_session <= 0 or sample_count < args.max_samples_per_session:
                    sample_count += 1
                    hist = prior_turn_docs
                    if args.history_selection == "tail":
                        hist = hist[-args.max_doc_num :]
                    elif args.history_selection == "head":
                        hist = hist[: args.max_doc_num]
                    elif args.history_selection == "head_tail" and len(hist) > args.max_doc_num:
                        hist = [hist[0]] + hist[-(args.max_doc_num - 1) :]
                    kept_tokens = [tokens for _, tokens in hist]

                    group["candidate_samples"] += 1
                    group["candidate_with_tool_call"] += int(has_tool_call)
                    group["history_docs_per_sample"].append(len(prior_turn_docs))
                    group["history_chars_per_sample"].append(sum(len(doc) for doc, _ in prior_turn_docs))
                    group["history_tokens_per_sample"].append(sum(tokens for _, tokens in prior_turn_docs))
                    group["kept_docs_per_sample"].append(len(hist))
                    group["kept_history_tokens_per_sample"].append(sum(kept_tokens))
                    group["current_query_chars"].append(len(str(current_user.get("content", ""))))
                    group["current_query_tokens"].append(counter.count_message("user", str(current_user.get("content", ""))))
                    group["answer_chars"].append(len(answer))
                    group["answer_tokens"].append(counter.count_text(answer))

            doc = _turn_doc(current_user, answer, include_tool_output=args.include_tool_output)
            doc_tokens = counter.count_message("user", doc)
            group["turn_docs"] += 1
            group["turn_doc_chars"].append(len(doc))
            group["turn_doc_tokens"].append(doc_tokens)
            group["overflow_doc_tokens"] += int(doc_tokens > args.max_doc_length)
            prior_turn_docs.append((doc, doc_tokens))
            previous_input_messages = normalized_messages

    results = []
    for subset, group in sorted(groups.items()):
        candidate_samples = group["candidate_samples"]
        turn_docs = group["turn_docs"]
        consecutive_pairs = group["consecutive_pairs"]
        results.append({
            "subset": subset,
            "sessions": group["sessions"],
            "llm_spans": group["llm_spans"],
            "turn_docs": turn_docs,
            "candidate_samples": candidate_samples,
            "candidate_tool_call_rate": round(group["candidate_with_tool_call"] / candidate_samples, 6) if candidate_samples else 0.0,
            "input_messages_per_span": _stat(group["input_messages_per_span"]),
            "prev_input_prefix_rate": (
                round(group["prev_input_is_prefix"] / consecutive_pairs, 6)
                if consecutive_pairs else 0.0
            ),
            "consecutive_prefix_ratio_per_mille": _stat(group["consecutive_prefix_ratio_per_mille"]),
            "turn_doc_tokens": _stat(group["turn_doc_tokens"]),
            "turn_doc_chars": _stat(group["turn_doc_chars"]),
            "history_docs_per_sample": _stat(group["history_docs_per_sample"]),
            "history_tokens_per_sample": _stat(group["history_tokens_per_sample"]),
            "kept_docs_per_sample": _stat(group["kept_docs_per_sample"]),
            "kept_history_tokens_per_sample": _stat(group["kept_history_tokens_per_sample"]),
            "current_query_tokens": _stat(group["current_query_tokens"]),
            "answer_tokens": _stat(group["answer_tokens"]),
            "overflow_doc_rate": round(group["overflow_doc_tokens"] / turn_docs, 6) if turn_docs else 0.0,
            "skip_reasons": dict(group["skip_reasons"]),
        })

    return {
        "dataset_path": args.dataset_path,
        "token_count_mode": counter.mode,
        "num_parquet_files": len(data_files),
        "history_doc_unit": "one previous user-query plus assistant-output turn per compressed document",
        "settings": {
            "min_history_docs": args.min_history_docs,
            "max_doc_length": args.max_doc_length,
            "max_doc_num": args.max_doc_num,
            "history_selection": args.history_selection,
            "max_samples_per_session": args.max_samples_per_session,
            "include_tool_output": args.include_tool_output,
        },
        "results": results,
    }


def _print_markdown(result: Dict[str, Any]) -> None:
    print("| subset | sessions | llm spans | samples | prefix rate | input msg p50 | tool-call rate | doc tok avg | doc tok p50 | doc tok p95 | hist docs p50 | kept tok p50 | kept tok p95 | overflow doc |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in result["results"]:
        doc = row["turn_doc_tokens"]
        input_messages = row["input_messages_per_span"]
        hist_docs = row["history_docs_per_sample"]
        kept = row["kept_history_tokens_per_sample"]
        print(
            f"| {row['subset']} | {row['sessions']} | {row['llm_spans']} | {row['candidate_samples']} | "
            f"{row['prev_input_prefix_rate']:.4f} | {input_messages['p50']} | "
            f"{row['candidate_tool_call_rate']:.4f} | {doc['avg']} | {doc['p50']} | {doc['p95']} | "
            f"{hist_docs['p50']} | {kept['p50']} | {kept['p95']} | {row['overflow_doc_rate']:.4f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect turn-level history docs for agent-llm-traces.")
    parser.add_argument("--dataset_path", default="./datasets/agent-llm-traces")
    parser.add_argument("--tokenizer", help="Optional local tokenizer path for exact chat-template token counts.")
    parser.add_argument("--output_file", default="./outputs/agent_llm_traces_history_doc_stats.json")
    parser.add_argument("--min_history_docs", type=int, default=1)
    parser.add_argument("--max_doc_length", type=int, default=1024)
    parser.add_argument("--max_doc_num", type=int, default=10)
    parser.add_argument("--history_selection", choices=["head", "tail", "head_tail"], default="head_tail")
    parser.add_argument("--max_samples_per_session", type=int, default=4)
    parser.add_argument("--include_tool_output", action="store_true")
    args = parser.parse_args()

    result = inspect(args)
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print()
    _print_markdown(result)


if __name__ == "__main__":
    main()
