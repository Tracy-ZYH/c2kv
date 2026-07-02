from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Sequence

import datasets
from transformers import AutoTokenizer

from .train_data import DEFAULT_SYSTEM_PROMPT, GistDataset


Message = Dict[str, Any]
HistorySelection = Literal["tail", "head"]


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_loads(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


@dataclass(frozen=True)
class CompressHistoryExample:
    """One training example for compressing conversation history.

    `history_messages` are the only segments intended for C2KV compression.
    `tools` stay with the system prefix and are not turned into compressed
    context documents.
    """

    qid: str
    history_messages: List[Message]
    current_messages: List[Message]
    answer: str
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    tools: List[Dict[str, Any]] = field(default_factory=list)


class CompressHistorySource(ABC):
    """Dataset adapter interface for future multi-turn/agent sources.

    Implement this interface when a concrete dataset is chosen.  The adapter
    should decide how to split each raw conversation into:

    - reusable history messages;
    - non-reused current messages;
    - target answer text.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[CompressHistoryExample]:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError


class JsonlCompressHistorySource(CompressHistorySource):
    """Simple reference source for normalized JSONL experiments.

    Expected fields per line:

    - qid: optional string id;
    - system_prompt: optional string;
    - tools: optional list of tool definitions;
    - history_messages: list of chat messages to compress;
    - current_messages: list of chat messages kept outside compressed history;
    - answer: target assistant text.
    """

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.records = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle):
                if line.strip():
                    record = json.loads(line)
                    record.setdefault("qid", f"{self.path.name}:{line_number}")
                    self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[CompressHistoryExample]:
        for record in self.records:
            yield CompressHistoryExample(
                qid=str(record["qid"]),
                system_prompt=record.get("system_prompt") or DEFAULT_SYSTEM_PROMPT,
                tools=record.get("tools") or [],
                history_messages=list(record.get("history_messages") or []),
                current_messages=list(record.get("current_messages") or []),
                answer=str(record.get("answer") or ""),
            )


def _render_openswe_tool_calls(tool_calls: Sequence[Dict[str, Any]] | None) -> str:
    rendered = []
    for call in tool_calls or []:
        function = call.get("function") or {}
        payload = {
            "name": function.get("name", ""),
            "arguments": function.get("arguments") or "{}",
        }
        rendered.append("<tool_call>\n" + json.dumps(payload, ensure_ascii=False) + "\n</tool_call>")
    return "\n".join(rendered)


def _render_openswe_history_message(message: Message) -> Optional[Message]:
    role = message.get("role", "user")
    if role == "system":
        return None
    parts = []
    content = message.get("content") or ""
    if content:
        parts.append(content)
    if role == "assistant":
        tool_calls = _render_openswe_tool_calls(message.get("tool_calls") or [])
        if tool_calls:
            parts.append(tool_calls)
    if not parts:
        return None
    return {"role": role, "content": "\n\n".join(parts)}


def _render_openswe_assistant_target(message: Message, max_answer_chars: Optional[int]) -> str:
    parts = []
    reasoning = message.get("reasoning_content") or ""
    if reasoning:
        parts.append(reasoning)
    content = message.get("content") or ""
    if content:
        parts.append(content)
    tool_calls = _render_openswe_tool_calls(message.get("tool_calls") or [])
    if tool_calls:
        parts.append(tool_calls)
    answer = "\n\n".join(parts).strip()
    if max_answer_chars is not None and len(answer) > max_answer_chars:
        answer = answer[:max_answer_chars].rstrip()
    return answer


def _openswe_prefix(system_prompt: str, first_user: Message) -> str:
    user_content = first_user.get("content") or ""
    return (
        system_prompt.strip()
        + "\n\n[Initial user request]\n"
        + user_content.strip()
    ).strip()


def _expand_openswe_batch(
    batch: Dict[str, List[Any]],
    resolved_only: bool,
    languages: Optional[List[str]],
    max_total_chars: Optional[int],
    max_answer_chars: Optional[int],
    recent_message_num: int,
    max_samples_per_trace: Optional[int] = None,
) -> Dict[str, List[Any]]:
    allowed_languages = set(languages) if languages else None
    outputs = {
        "qid": [],
        "system_prompt": [],
        "tools": [],
        "history_messages": [],
        "current_messages": [],
        "answer": [],
    }
    for row_index, trajectory in enumerate(batch["trajectory"]):
        if resolved_only and batch.get("resolved", [None])[row_index] != 1:
            continue
        language = batch.get("language", [None])[row_index]
        if allowed_languages is not None and language not in allowed_languages:
            continue
        if not trajectory:
            continue
        system_prompt = next(
            (message.get("content") or "" for message in trajectory if message.get("role") == "system"),
            DEFAULT_SYSTEM_PROMPT,
        )
        first_user_index = next(
            (index for index, message in enumerate(trajectory) if message.get("role") == "user"),
            None,
        )
        if first_user_index is None:
            continue
        prefix = _openswe_prefix(system_prompt or DEFAULT_SYSTEM_PROMPT, trajectory[first_user_index])
        instance_id = batch.get("instance_id", [""])[row_index]
        trajectory_id = batch.get("trajectory_id", [""])[row_index]
        qid_prefix = trajectory_id or instance_id or str(row_index)

        trace_candidates: List[Dict[str, Any]] = []
        for assistant_index, message in enumerate(trajectory):
            if message.get("role") != "assistant":
                continue
            answer = _render_openswe_assistant_target(message, max_answer_chars)
            if not answer:
                continue
            previous = [
                prior
                for prior in trajectory[first_user_index + 1 : assistant_index]
                if prior.get("role") != "system"
            ]
            running_recent_message_num = random.randint(1, recent_message_num)
            if len(previous) <= running_recent_message_num:
                continue
            history_raw = previous[:-running_recent_message_num]
            current_raw = previous[-running_recent_message_num:]
            history = [
                rendered
                for rendered in (_render_openswe_history_message(item) for item in history_raw)
                if rendered is not None
            ]
            current = [
                rendered
                for rendered in (_render_openswe_history_message(item) for item in current_raw)
                if rendered is not None
            ]
            if not history or not current:
                continue
            if max_total_chars is not None:
                sample_chars = len(prefix) + len(answer)
                sample_chars += sum(len(item["content"]) for item in history)
                sample_chars += sum(len(item["content"]) for item in current)
                if sample_chars > max_total_chars:
                    continue
            trace_candidates.append({
                "qid": f"{qid_prefix}:{assistant_index}",
                "system_prompt": prefix,
                "tools": "[]",
                "history_messages": _json_dumps(history),
                "current_messages": _json_dumps(current),
                "answer": answer,
            })
        if max_samples_per_trace is not None and len(trace_candidates) > max_samples_per_trace:
            trace_candidates = random.sample(trace_candidates, max_samples_per_trace)
        for candidate in trace_candidates:
            outputs["qid"].append(candidate["qid"])
            outputs["system_prompt"].append(candidate["system_prompt"])
            outputs["tools"].append(candidate["tools"])
            outputs["history_messages"].append(candidate["history_messages"])
            outputs["current_messages"].append(candidate["current_messages"])
            outputs["answer"].append(candidate["answer"])
    return outputs


class OpenSWETracesCompressHistorySource(CompressHistorySource):
    """Adapter for NVIDIA Open-SWE-Traces.

    Each assistant action becomes one sample.  The prefix is the original
    system prompt plus the first user request.  Reused history excludes
    assistant reasoning; the target output includes assistant reasoning,
    content, and tool calls.
    """

    def __init__(
        self,
        path: str,
        resolved_only: bool = True,
        languages: Optional[str | Sequence[str]] = None,
        max_total_chars: Optional[int] = None,
        max_answer_chars: Optional[int] = None,
        recent_message_num: int = 1,
        num_proc: int = 8,
        max_samples_per_trace: Optional[int] = None,
    ) -> None:
        self.path = Path(path)
        if self.path.is_file():
            data_files = [str(self.path)]
        else:
            data_root = self.path / "data"
            search_root = data_root if data_root.is_dir() else self.path
            data_files = sorted(str(file) for file in search_root.glob("*/*.parquet"))
            if not data_files:
                data_files = sorted(str(file) for file in search_root.glob("*.parquet"))
        if not data_files:
            raise FileNotFoundError(f"No parquet files found under {path}")

        if isinstance(languages, str):
            language_list = [item.strip() for item in languages.split(",") if item.strip()]
        else:
            language_list = list(languages) if languages is not None else None
        raw = datasets.load_dataset(
            "parquet",
            data_files=data_files,
            split="train",
        )
        map_kwargs = {
            "batched": True,
            "remove_columns": raw.column_names,
            "fn_kwargs": {
                "resolved_only": resolved_only,
                "languages": language_list,
                "max_total_chars": max_total_chars,
                "max_answer_chars": max_answer_chars,
                "recent_message_num": recent_message_num,
                "max_samples_per_trace": max_samples_per_trace,
            },
        }
        if num_proc > 1:
            map_kwargs["num_proc"] = num_proc
        self.data = raw.map(_expand_openswe_batch, **map_kwargs)

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[CompressHistoryExample]:
        for record in self.data:
            yield CompressHistoryExample(
                qid=record["qid"],
                system_prompt=record.get("system_prompt") or DEFAULT_SYSTEM_PROMPT,
                tools=_json_loads(record.get("tools"), []),
                history_messages=_json_loads(record.get("history_messages"), []),
                current_messages=_json_loads(record.get("current_messages"), []),
                answer=record.get("answer") or "",
            )


def _chat_template_ids(
    tokenizer: AutoTokenizer,
    messages: Sequence[Message],
    *,
    tools: Optional[List[Dict[str, Any]]] = None,
    add_generation_prompt: bool = False,
    keep_bos: bool = False,
    max_length: Optional[int] = None,
) -> List[int]:
    encoded = tokenizer.apply_chat_template(
        list(messages),
        tools=tools,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
        max_length=max_length + 1 if max_length is not None and not keep_bos else max_length,
        truncation=max_length is not None,
    )
    ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded
    if not keep_bos and ids and ids[0] == tokenizer.bos_token_id:
        ids = ids[1:]
    return ids


def _normal_chat_message(message: Message) -> Message:
    role = message.get("role", "user")
    if role == "tool":
        role = "user"
    content = message.get("content", "")
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    return {"role": role, "content": content}


def _message_token_length(tokenizer: AutoTokenizer, message: Message) -> int:
    return len(_chat_template_ids(tokenizer, [message]))


def _semantic_units(text: str) -> List[str]:
    if not text:
        return []
    if "\ndiff --git " in text:
        pieces = text.split("\ndiff --git ")
        return [pieces[0]] + ["diff --git " + piece for piece in pieces[1:] if piece]
    markers = ["\nTraceback (most recent call last):", "\n@@ ", "\n\n", "\n"]
    for marker in markers:
        if marker in text:
            if marker == "\n\n":
                parts = text.split(marker)
                return [part + marker for part in parts[:-1] if part] + ([parts[-1]] if parts[-1] else [])
            if marker == "\n":
                lines = text.splitlines(keepends=True)
                return [line for line in lines if line]
            parts = text.split(marker)
            return [parts[0]] + [marker.lstrip("\n") + part for part in parts[1:] if part]
    return [text]


def _hard_split_text(text: str, max_chars: int) -> List[str]:
    if max_chars <= 0:
        return [text]
    return [text[start : start + max_chars] for start in range(0, len(text), max_chars)]


def _split_message_to_fit(
    tokenizer: AutoTokenizer,
    message: Message,
    max_doc_length: int,
) -> List[Message]:
    if _message_token_length(tokenizer, message) <= max_doc_length:
        return [message]
    role = message["role"]
    units = _semantic_units(message["content"])
    chunks: List[Message] = []
    current = ""
    approx_chars = max(256, max_doc_length * 3)
    for unit in units:
        candidate = current + unit
        candidate_message = {"role": role, "content": candidate}
        if candidate and _message_token_length(tokenizer, candidate_message) <= max_doc_length:
            current = candidate
            continue
        if current:
            chunks.append({"role": role, "content": current})
            current = ""
        unit_message = {"role": role, "content": unit}
        if _message_token_length(tokenizer, unit_message) <= max_doc_length:
            current = unit
            continue
        for part in _hard_split_text(unit, approx_chars):
            part_message = {"role": role, "content": part}
            if _message_token_length(tokenizer, part_message) <= max_doc_length:
                chunks.append(part_message)
            else:
                token_ids = tokenizer.encode(part, add_special_tokens=False)
                step = max(1, max_doc_length - 16)
                for start in range(0, len(token_ids), step):
                    text = tokenizer.decode(token_ids[start : start + step], skip_special_tokens=True)
                    if text:
                        chunks.append({"role": role, "content": text})
    if current:
        chunks.append({"role": role, "content": current})
    return chunks


def _fit_reused_history(
    tokenizer: AutoTokenizer,
    messages: Sequence[Message],
    max_doc_length: int,
    max_doc_num: int,
    policy: HistorySelection,
) -> List[Message]:
    split_messages: List[Message] = []
    for message in messages:
        split_messages.extend(_split_message_to_fit(tokenizer, message, max_doc_length))
    messages = list(split_messages)
    return _select_history(messages, max_doc_num=max_doc_num, policy=policy)


def _select_history(
    messages: Sequence[Message],
    max_doc_num: int,
    policy: HistorySelection,
) -> List[Message]:
    if len(messages) <= max_doc_num:
        return list(messages)
    if policy == "head":
        return list(messages[:max_doc_num])
    if policy == "tail":
        if max_doc_num <= 1:
            return list(messages[-max_doc_num:])
        return [messages[0]] + list(messages[-(max_doc_num - 1):])
    raise ValueError(f"Unsupported history selection policy: {policy}")


def _pad(values: List[int], length: int, pad_value: int) -> List[int]:
    if len(values) >= length:
        return values[:length]
    return values + [pad_value] * (length - len(values))


_INVALID_SAMPLE_MARKER = {
    "system_input_ids": [],
    "context_input_ids": [],
    "input_ids": [],
    "labels": [],
    "attention_mask": [],
    "dynamic": -1,
}


def _preprocess_record(
    record: Dict[str, Any],
    tokenizer: AutoTokenizer,
    max_length: int,
    max_doc_length: int,
    min_doc_num: int,
    max_doc_num: int,
    max_system_length: int,
    history_selection: HistorySelection,
) -> Dict[str, Any]:
    """Adapter that converts a raw dataset record into a CompressHistoryExample
    and delegates to CompressHistoryDataset.preprocess_example.

    Returns a sentinel dict with ``dynamic == -1`` when the sample is invalid,
    because ``datasets.Dataset.map`` does not support returning ``None``.
    """
    example = CompressHistoryExample(
        qid=str(record.get("qid", "")),
        system_prompt=record.get("system_prompt") or DEFAULT_SYSTEM_PROMPT,
        tools=_json_loads(record.get("tools"), []),
        history_messages=_json_loads(record.get("history_messages"), []),
        current_messages=_json_loads(record.get("current_messages"), []),
        answer=record.get("answer") or "",
    )
    result = CompressHistoryDataset.preprocess_example(
        example,
        tokenizer=tokenizer,
        max_length=max_length,
        max_doc_length=max_doc_length,
        min_doc_num=min_doc_num,
        max_doc_num=max_doc_num,
        max_system_length=max_system_length,
        history_selection=history_selection,
    )
    if result is None:
        return dict(_INVALID_SAMPLE_MARKER)
    return result


class CompressHistoryDataset(GistDataset):
    """Convert CompressHistorySource examples into GistMultiDocTrainer format."""

    def __init__(
        self,
        source: CompressHistorySource,
        tokenizer: AutoTokenizer,
        max_length: int = 1024,
        max_doc_length: int = 1024,
        min_doc_num: int = 2,
        max_doc_num: int = 10,
        max_system_length: int = 2048,
        num_samples: Optional[int] = None,
        shuffle_seed: int = 42,
        history_selection: HistorySelection = "tail",
        num_proc: int = 32,
    ) -> None:
        raw_data = source.data if hasattr(source, "data") else None
        if raw_data is not None and isinstance(raw_data, datasets.Dataset):
            if num_samples is not None:
                raw_data = raw_data.select(range(min(num_samples, len(raw_data))))
            mapped = raw_data.map(
                _preprocess_record,
                fn_kwargs={
                    "tokenizer": tokenizer,
                    "max_length": max_length,
                    "max_doc_length": max_doc_length,
                    "min_doc_num": min_doc_num,
                    "max_doc_num": max_doc_num,
                    "max_system_length": max_system_length,
                    "history_selection": history_selection,
                },
                num_proc=num_proc,
                remove_columns=raw_data.column_names,
            )
            # Filter out invalid samples marked by dynamic == -1
            filtered = mapped.filter(lambda x: x["dynamic"] != -1, num_proc=num_proc)
            data = filtered.shuffle(seed=shuffle_seed) if len(filtered) else filtered
        else:
            # Fallback to sequential iteration for sources without .data attribute
            rows = []
            for index, example in enumerate(source):
                if num_samples is not None and index >= num_samples:
                    break
                row = self.preprocess_example(
                    example,
                    tokenizer=tokenizer,
                    max_length=max_length,
                    max_doc_length=max_doc_length,
                    min_doc_num=min_doc_num,
                    max_doc_num=max_doc_num,
                    max_system_length=max_system_length,
                    history_selection=history_selection,
                )
                if row is not None:
                    rows.append(row)
            data = datasets.Dataset.from_list(rows)
            data = data.shuffle(seed=shuffle_seed) if len(data) else data
        self.data = data
        self.max_doc_length = max_doc_length
        self.min_doc_num = min_doc_num
        self.max_system_length = max_system_length
        self.max_doc_num = max_doc_num
        self.max_length = max_length

    @staticmethod
    def preprocess_example(
        example: CompressHistoryExample,
        tokenizer: AutoTokenizer,
        max_length: int,
        max_doc_length: int,
        min_doc_num: int,
        max_doc_num: int,
        max_system_length: int,
        history_selection: HistorySelection,
    ) -> Optional[Dict[str, Any]]:
        raw_history = [
            _normal_chat_message(message)
            for message in example.history_messages
            if message.get("content")
        ]
        history = _fit_reused_history(
            tokenizer,
            raw_history,
            max_doc_length=max_doc_length,
            max_doc_num=max_doc_num,
            policy=history_selection,
        )
        current = [
            _normal_chat_message(message)
            for message in example.current_messages
            if message.get("content") or message.get("role") == "assistant"
        ]
        if len(history) < min_doc_num or not current or not example.answer:
            return None

        system_ids = _chat_template_ids(
            tokenizer,
            [{"role": "system", "content": example.system_prompt}],
            tools=example.tools or None,
            keep_bos=True,
            max_length=max_system_length,
        )
        system_input_ids = _pad(system_ids, max_system_length, -100)

        context_input_ids: List[int] = []
        for message in history:
            doc_ids = _chat_template_ids(
                tokenizer,
                [message],
                max_length=max_doc_length,
            )
            context_input_ids.extend(_pad(doc_ids, max_doc_length, -100))
        empty_docs = max_doc_num - len(history)
        context_input_ids.extend([-100] * (max_doc_length * empty_docs))

        prompt_ids = _chat_template_ids(
            tokenizer,
            current,
            add_generation_prompt=True,
        )
        answer_ids = tokenizer.encode(example.answer, add_special_tokens=False)
        if not answer_ids:
            return None
        answer_ids.append(tokenizer.eos_token_id)
        if len(prompt_ids) >= max_length:
            prompt_ids = prompt_ids[-(max_length - 1):]
        answer_budget = max_length - len(prompt_ids)
        answer_ids = answer_ids[:answer_budget]
        if not answer_ids:
            return None
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        pad_length = max_length - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * pad_length
        input_ids = input_ids + [tokenizer.pad_token_id] * pad_length
        labels = labels + [-100] * pad_length

        return {
            "system_input_ids": system_input_ids,
            "context_input_ids": context_input_ids,
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "dynamic": 0,
        }


def load_compress_history_source(source_type: str, path: str) -> CompressHistorySource:
    if source_type == "jsonl":
        return JsonlCompressHistorySource(path)
    if source_type == "open_swe":
        return OpenSWETracesCompressHistorySource(path)
    raise NotImplementedError(
        f"Unsupported compress-history source {source_type!r}. "
        "Implement CompressHistorySource for the chosen dataset."
    )


def get_compress_history_dataset(
    path: str,
    tokenizer: AutoTokenizer,
    source_type: str = "jsonl",
    source_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> CompressHistoryDataset:
    if source_type == "jsonl":
        source = JsonlCompressHistorySource(path)
    elif source_type == "open_swe":
        source = OpenSWETracesCompressHistorySource(path, **(source_kwargs or {}))
    else:
        source = load_compress_history_source(source_type, path)
    return CompressHistoryDataset(source, tokenizer=tokenizer, **kwargs)
