from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Sequence

import datasets
from transformers import AutoTokenizer

from .train_data import DEFAULT_SYSTEM_PROMPT, GistDataset


Message = Dict[str, Any]
HistorySelection = Literal["tail", "head"]


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
        return list(messages[-max_doc_num:])
    raise ValueError(f"Unsupported history selection policy: {policy}")


def _pad(values: List[int], length: int, pad_value: int) -> List[int]:
    if len(values) >= length:
        return values[:length]
    return values + [pad_value] * (length - len(values))


class CompressHistoryDataset(GistDataset):
    """Convert CompressHistorySource examples into GistMultiDocTrainer format."""

    def __init__(
        self,
        source: CompressHistorySource,
        tokenizer: AutoTokenizer,
        max_length: int = 1024,
        max_doc_length: int = 1024,
        max_doc_num: int = 10,
        max_system_length: int = 2048,
        num_samples: Optional[int] = None,
        shuffle_seed: int = 42,
        history_selection: HistorySelection = "tail",
    ) -> None:
        rows = []
        for index, example in enumerate(source):
            if num_samples is not None and index >= num_samples:
                break
            row = self.preprocess_example(
                example,
                tokenizer=tokenizer,
                max_length=max_length,
                max_doc_length=max_doc_length,
                max_doc_num=max_doc_num,
                max_system_length=max_system_length,
                history_selection=history_selection,
            )
            if row is not None:
                rows.append(row)
        data = datasets.Dataset.from_list(rows)
        self.data = data.shuffle(seed=shuffle_seed) if len(data) else data
        self.max_doc_length = max_doc_length
        self.max_system_length = max_system_length
        self.max_doc_num = max_doc_num
        self.max_length = max_length

    @staticmethod
    def preprocess_example(
        example: CompressHistoryExample,
        tokenizer: AutoTokenizer,
        max_length: int,
        max_doc_length: int,
        max_doc_num: int,
        max_system_length: int,
        history_selection: HistorySelection,
    ) -> Optional[Dict[str, Any]]:
        history = [
            _normal_chat_message(message)
            for message in _select_history(
                example.history_messages,
                max_doc_num=max_doc_num,
                policy=history_selection,
            )
            if message.get("content")
        ]
        current = [
            _normal_chat_message(message)
            for message in example.current_messages
            if message.get("content") or message.get("role") == "assistant"
        ]
        if not history or not current or not example.answer:
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
        answer_ids.append(tokenizer.eos_token_id)
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
            attention_mask = [1] * max_length
        else:
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
    raise NotImplementedError(
        f"Unsupported compress-history source {source_type!r}. "
        "Implement CompressHistorySource for the chosen dataset."
    )


def get_compress_history_dataset(
    path: str,
    tokenizer: AutoTokenizer,
    source_type: str = "jsonl",
    **kwargs: Any,
) -> CompressHistoryDataset:
    source = load_compress_history_source(source_type, path)
    return CompressHistoryDataset(source, tokenizer=tokenizer, **kwargs)
