from __future__ import annotations

import json
import logging
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, HfArgumentParser

from gist_args import ModelArgs, TrainingArgs
from models import *
from train.train_data import DEFAULT_SYSTEM_PROMPT
from train.train_data_multiturn import _chat_template_ids, _pad
from train.trainer import GistMultiDocTrainer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)


Message = Dict[str, Any]


@dataclass
class AgentToolDefinitionDataArgs:
    dataset_path: str = "./datasets/agent-llm-traces"
    eval_ratio: float = 0.1
    split_seed: int = 42
    split_manifest_file: Optional[str] = None
    split_manifest_name: str = "toolset_disjoint"
    max_sessions: Optional[int] = None
    max_samples_per_session: int = 4
    max_doc_length: int = 4096
    max_doc_num: int = 10
    max_length: int = 2048
    max_system_length: int = 256
    max_tool_definition_chars: Optional[int] = None
    max_tool_definition_tokens: int = 10000
    max_input_chars: Optional[int] = None
    max_target_chars: Optional[int] = None
    min_target_tokens: int = 64
    truncate_tool_definition: bool = True
    require_tool_call: bool = True
    num_proc: int = 8
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    device_type: str = "auto"
    npu_attn_impl: str = "eager"


@dataclass(frozen=True)
class AgentToolDefinitionExample:
    qid: str
    session_id: str
    tool_definition: str
    input_messages: List[Message]
    answer: str
    has_tool_call: bool
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    tool_documents: Optional[List[str]] = None


def _json_loads(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if not isinstance(obj, dict):
        return default
    if key in obj:
        return obj[key]
    current = obj
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _first_value(obj: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        value = _get_value(obj, key, None)
        if value is not None:
            return value
    return default


def _canonical_tool_definition(value: Any) -> str:
    parsed = _json_loads(value, value)
    if isinstance(parsed, str):
        return parsed.strip()
    return _json_dumps(parsed)


def _render_tool_calls(tool_calls: Any) -> tuple[str, bool]:
    tool_calls = _json_loads(tool_calls, [])
    if isinstance(tool_calls, dict):
        tool_calls = [tool_calls]
    if not isinstance(tool_calls, list):
        return "", False
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
        payload = {"name": name, "arguments": arguments}
        rendered.append("<tool_call>\n" + _json_dumps(payload) + "\n</tool_call>")
    return "\n".join(rendered), bool(rendered)


def _message_parts(message: Message) -> List[Dict[str, Any]]:
    parts = message.get("parts")
    parts = _json_loads(parts, parts)
    if isinstance(parts, dict):
        return [parts]
    if isinstance(parts, list):
        return [part for part in parts if isinstance(part, dict)]
    return []


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
    tool_calls_text, _ = _render_tool_calls(
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
        tool_calls_text, found_tool_call = _render_tool_calls(
            message.get("tool_calls")
            or message.get("toolCalls")
            or message.get("function_call")
            or _message_parts(message)
        )
        if tool_calls_text:
            parts.append("Action:\n" + tool_calls_text)
        has_tool_call = has_tool_call or found_tool_call
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


def _iter_sessions(row: Dict[str, Any], row_index: int) -> Iterator[tuple[str, List[Dict[str, Any]]]]:
    trace = _json_loads(_first_value(row, ["trace", "Trace"], row), row)
    sessions = _first_value(row, ["trace.sessions", "sessions"], None)
    if sessions is None and isinstance(trace, dict):
        sessions = _first_value(trace, ["trace.sessions", "sessions"], None)
    sessions = _json_loads(sessions, sessions)

    if sessions is None:
        spans = _first_value(row, ["spans", "trace.spans"], None)
        spans = _json_loads(spans, spans)
        if spans is not None:
            sessions = [{"session_id": _first_value(row, ["session_id", "id"], f"row-{row_index}"), "spans": spans}]

    if isinstance(sessions, dict):
        sessions = [sessions]
    if not isinstance(sessions, list):
        return

    for session_index, session in enumerate(sessions):
        if not isinstance(session, dict):
            continue
        spans = _json_loads(session.get("spans"), session.get("spans"))
        if not isinstance(spans, list):
            continue
        session_id = (
            session.get("session_id")
            or session.get("sessionId")
            or session.get("id")
            or _first_value(row, ["session_id", "trace_id", "id"], None)
            or f"row-{row_index}:session-{session_index}"
        )
        yield str(session_id), spans


def _span_attributes(span: Any) -> Dict[str, Any]:
    span = _json_loads(span, span)
    if not isinstance(span, dict):
        return {}
    attributes = span.get("attributes", span)
    attributes = _json_loads(attributes, attributes)
    return attributes if isinstance(attributes, dict) else {}


class AgentLLMTracesSource:
    def __init__(self, args: AgentToolDefinitionDataArgs) -> None:
        self.args = args
        self.path = self._resolve_dataset_path(Path(args.dataset_path))
        self.source_skips: Counter[str] = Counter()
        data_files = self._find_parquet_files(self.path)
        if not data_files:
            raise FileNotFoundError(self._missing_dataset_message(self.path))
        logger.info("Loading %d parquet shards from %s", len(data_files), self.path)
        self.sessions = self._load_sessions(self._iter_parquet_rows(data_files))
        if args.max_sessions is not None:
            self.sessions = self.sessions[: args.max_sessions]
        logger.info("Loaded %d sessions before train/eval split", len(self.sessions))

    @staticmethod
    def _resolve_dataset_path(path: Path) -> Path:
        if path.exists():
            return path
        fallback = Path("./data/agent-llm-traces")
        if path.as_posix().endswith("/datasets/agent-llm-traces") and fallback.exists():
            logger.warning("Using fallback dataset path %s because %s does not exist", fallback, path)
            return fallback
        return path

    @staticmethod
    def _find_parquet_files(path: Path) -> List[str]:
        if path.is_file() and path.suffix == ".parquet":
            return [str(path)]
        search_roots = [path / "data", path]
        files: List[Path] = []
        for root in search_roots:
            if root.is_dir():
                files.extend(sorted(root.glob("*.parquet")))
                if not files:
                    files.extend(sorted(root.rglob("*.parquet")))
            if files:
                break
        return [str(file) for file in files]

    @staticmethod
    def _missing_dataset_message(path: Path) -> str:
        checked = [path / "data", path]
        lines = [f"No parquet files found under dataset_path={path!s}."]
        for item in checked:
            if item.exists():
                if item.is_dir():
                    preview = sorted(child.name for child in item.iterdir())[:20]
                    lines.append(f"Checked {item!s}: exists, first entries={preview}")
                else:
                    lines.append(f"Checked {item!s}: exists but is not a directory")
            else:
                lines.append(f"Checked {item!s}: missing")
        lines.append(
            "Set DATASET_PATH or --dataset_path to the directory that contains "
            "data/train-00000-of-00039.parquet, or copy the Hugging Face dataset "
            "to ./datasets/agent-llm-traces."
        )
        return "\n".join(lines)

    @staticmethod
    def _iter_parquet_rows(data_files: Sequence[str]) -> Iterator[Dict[str, Any]]:
        """Read parquet rows without Hugging Face metadata decoding.

        Some servers have an older `datasets` package that cannot decode newer
        parquet metadata feature types such as `List`. PyArrow can read the
        actual table data directly, which is all this script needs.
        """
        import pyarrow.parquet as pq

        for data_file in data_files:
            table = pq.read_table(data_file)
            for row in table.to_pylist():
                yield row

    def _load_sessions(self, raw: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sessions_by_id: Dict[str, List[Dict[str, Any]]] = {}
        for row_index, row in enumerate(raw):
            found_nested_session = False
            for session_id, spans in _iter_sessions(row, row_index):
                found_nested_session = True
                sessions_by_id.setdefault(session_id, []).extend(spans)
            if found_nested_session:
                continue

            session_id = (
                row.get("session_id")
                or row.get("trace_id")
                or row.get("TraceId")
                or row.get("traceId")
                or f"row-{row_index}"
            )
            sessions_by_id.setdefault(str(session_id), []).append(dict(row))

        sessions = [
            {"session_id": session_id, "spans": self._sort_spans(spans)}
            for session_id, spans in sessions_by_id.items()
        ]
        session_ids = {item["session_id"] for item in sessions}
        if len(session_ids) != len(sessions):
            logger.info(
                "Found %d session records with %d unique ids; split is still keyed by session_id",
                len(sessions),
                len(session_ids),
            )
        return sessions

    @staticmethod
    def _sort_spans(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(
            spans,
            key=lambda span: (
                span.get("start_time") or "",
                span.get("span_id") or "",
            ),
        )

    def split_session_ids(self) -> tuple[set[str], set[str]]:
        if self.args.split_manifest_file:
            return self._split_session_ids_from_manifest()
        session_ids = sorted({item["session_id"] for item in self.sessions})
        rng = random.Random(self.args.split_seed)
        rng.shuffle(session_ids)
        eval_count = max(1, int(round(len(session_ids) * self.args.eval_ratio))) if session_ids else 0
        eval_ids = set(session_ids[:eval_count])
        train_ids = set(session_ids[eval_count:])
        return train_ids, eval_ids

    def _split_session_ids_from_manifest(self) -> tuple[set[str], set[str]]:
        manifest_path = Path(self.args.split_manifest_file or "")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "train_session_ids" in manifest and "eval_session_ids" in manifest:
            selected = manifest
            split_name = "root"
        else:
            split_name = self.args.split_manifest_name
            if split_name not in manifest:
                raise KeyError(
                    f"Split {split_name!r} not found in {manifest_path}. "
                    f"Available splits: {sorted(manifest)}"
                )
            selected = manifest[split_name]
        train_ids = set(str(item) for item in selected.get("train_session_ids", []))
        eval_ids = set(str(item) for item in selected.get("eval_session_ids", []))
        overlap = train_ids & eval_ids
        if overlap:
            raise RuntimeError(
                f"Split manifest {manifest_path}::{split_name} has train/eval overlap: "
                f"{sorted(overlap)[:5]}"
            )
        available_ids = {item["session_id"] for item in self.sessions}
        missing_train = train_ids - available_ids
        missing_eval = eval_ids - available_ids
        if missing_train or missing_eval:
            logger.warning(
                "Split manifest %s::%s references missing sessions; missing_train=%d missing_eval=%d",
                manifest_path,
                split_name,
                len(missing_train),
                len(missing_eval),
            )
        train_ids &= available_ids
        eval_ids &= available_ids
        logger.info(
            "Using split manifest %s::%s with %d train sessions and %d eval sessions",
            manifest_path,
            split_name,
            len(train_ids),
            len(eval_ids),
        )
        return train_ids, eval_ids

    def iter_examples(self, split: str) -> Iterator[AgentToolDefinitionExample]:
        train_ids, eval_ids = self.split_session_ids()
        keep_ids = train_ids if split == "train" else eval_ids
        rng = random.Random(self.args.split_seed + (0 if split == "train" else 1))
        for session in self.sessions:
            session_id = session["session_id"]
            if session_id not in keep_ids:
                continue
            candidates = self._session_examples(session_id, session["spans"])
            if self.args.max_samples_per_session and len(candidates) > self.args.max_samples_per_session:
                candidates = rng.sample(candidates, self.args.max_samples_per_session)
            yield from candidates

    def _session_examples(self, session_id: str, spans: Sequence[Any]) -> List[AgentToolDefinitionExample]:
        tool_definition = ""
        candidates = []
        for span_index, span in enumerate(spans):
            attributes = _span_attributes(span)
            tool_value = attributes.get("gen_ai.tool.definitions")
            if tool_value and not tool_definition:
                tool_definition = _canonical_tool_definition(tool_value)
            input_messages = _json_loads(attributes.get("gen_ai.input.messages"), [])
            output_messages = attributes.get("gen_ai.output.messages")
            if not tool_definition or not input_messages or output_messages is None:
                self.source_skips["missing_tool_input_or_output"] += 1
                continue
            if self.args.max_tool_definition_chars is not None and len(tool_definition) > self.args.max_tool_definition_chars:
                self.source_skips["tool_definition_too_many_chars"] += 1
                return []
            if self.args.max_input_chars is not None and len(str(input_messages)) > self.args.max_input_chars:
                self.source_skips["input_too_many_chars"] += 1
                continue
            answer, has_tool_call = _render_output_messages(output_messages)
            if self.args.max_target_chars is not None and len(answer) > self.args.max_target_chars:
                answer = answer[: self.args.max_target_chars].rstrip()
            if not answer:
                self.source_skips["empty_answer"] += 1
                continue
            if self.args.require_tool_call and not has_tool_call:
                self.source_skips["no_tool_call"] += 1
                continue
            normalized_messages = [
                item
                for item in (_normal_message(message) for message in _json_loads(input_messages, []))
                if item is not None and item.get("role") != "system"
            ]
            if not normalized_messages:
                self.source_skips["empty_prompt"] += 1
                continue
            candidates.append(
                AgentToolDefinitionExample(
                    qid=f"{session_id}:{span_index}",
                    session_id=session_id,
                    tool_definition=tool_definition,
                    input_messages=normalized_messages,
                    answer=answer,
                    has_tool_call=has_tool_call,
                    system_prompt=self.args.system_prompt,
                )
            )
        return candidates


def _resolve_device_type(requested: str) -> str:
    requested = (requested or "auto").lower()
    if requested != "auto":
        return requested
    try:
        import torch_npu  # noqa: F401

        if hasattr(torch, "npu") and torch.npu.is_available():
            return "npu"
    except Exception:
        pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _visible_npu_count() -> Optional[int]:
    visible_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES") or os.environ.get("ASCEND_VISIBLE_DEVICES")
    if visible_devices:
        devices = [item.strip() for item in visible_devices.split(",") if item.strip()]
        return len(devices)
    try:
        return int(torch.npu.device_count())
    except Exception:
        return None


def _setup_device(model_args: ModelArgs, data_args: AgentToolDefinitionDataArgs) -> str:
    device_type = _resolve_device_type(data_args.device_type)
    if device_type == "npu":
        import torch_npu  # noqa: F401

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        visible_count = _visible_npu_count()
        if visible_count is not None and local_rank >= visible_count:
            visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES") or os.environ.get("ASCEND_VISIBLE_DEVICES")
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only {visible_count} NPU device(s) are visible "
                f"({visible}). Set NPROC_PER_NODE={visible_count}, or expose more NPUs."
            )
        torch.npu.set_device(local_rank)
        if model_args.attn_impl in (None, "flex_attention", "flash_attention_2"):
            logger.info(
                "Overriding attention backend from %s to %s for Ascend NPU",
                model_args.attn_impl,
                data_args.npu_attn_impl,
            )
            model_args.attn_impl = data_args.npu_attn_impl
        if model_args.device_map == "auto":
            logger.info("Disabling device_map=auto for NPU distributed/deepspeed training")
            model_args.device_map = None
    return device_type


class AgentToolDefinitionDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[AgentToolDefinitionExample],
        tokenizer,
        max_doc_length: int,
        max_doc_num: int,
        max_length: int,
        max_system_length: int,
        truncate_tool_definition: bool,
        min_target_tokens: int,
        max_tool_definition_tokens: int,
    ) -> None:
        self.max_doc_length = max_doc_length
        self.max_doc_num = max_doc_num
        self.max_length = max_length
        self.max_system_length = max_system_length
        self.data = []
        skipped_by_reason: Counter[str] = Counter()
        for example in examples:
            row, reason = self._preprocess_example(
                example,
                tokenizer=tokenizer,
                max_doc_length=max_doc_length,
                max_doc_num=max_doc_num,
                max_length=max_length,
                max_system_length=max_system_length,
                truncate_tool_definition=truncate_tool_definition,
                min_target_tokens=min_target_tokens,
                max_tool_definition_tokens=max_tool_definition_tokens,
            )
            if row is None:
                skipped_by_reason[reason] += 1
            else:
                self.data.append(row)
        logger.info(
            "Built %d samples; skipped %d samples by reason=%s",
            len(self.data),
            sum(skipped_by_reason.values()),
            dict(skipped_by_reason),
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.data[index]

    @staticmethod
    def _preprocess_example(
        example: AgentToolDefinitionExample,
        tokenizer,
        max_doc_length: int,
        max_doc_num: int,
        max_length: int,
        max_system_length: int,
        truncate_tool_definition: bool,
        min_target_tokens: int,
        max_tool_definition_tokens: int,
    ) -> tuple[Optional[Dict[str, Any]], str]:
        system_ids = _chat_template_ids(
            tokenizer,
            [{"role": "system", "content": example.system_prompt}],
            keep_bos=True,
            max_length=max_system_length,
        )
        system_input_ids = _pad(system_ids, max_system_length, -100)

        if example.tool_documents:
            doc_id_groups = [
                _chat_template_ids(
                    tokenizer,
                    [{"role": "user", "content": "Tool definition:\n" + document}],
                )
                for document in example.tool_documents
                if document.strip()
            ]
            doc_tokens = sum(len(item) for item in doc_id_groups)
        else:
            tool_doc = {
                "role": "user",
                "content": "Tool definitions:\n" + example.tool_definition,
            }
            doc_id_groups = [_chat_template_ids(tokenizer, [tool_doc])]
            doc_tokens = len(doc_id_groups[0])
        if doc_tokens > max_tool_definition_tokens:
            return None, f"tool_definition_tokens>{max_tool_definition_tokens}"
        max_context_tokens = max_doc_length * max_doc_num
        if doc_tokens > max_context_tokens and not example.tool_documents:
            if not truncate_tool_definition:
                return None, f"tool_definition_tokens>{max_context_tokens}"
            doc_id_groups = [doc_id_groups[0][:max_context_tokens]]
        doc_chunks = []
        for doc_ids in doc_id_groups:
            if len(doc_ids) <= max_doc_length:
                doc_chunks.append(doc_ids)
                continue
            if example.tool_documents and not truncate_tool_definition:
                return None, f"tool_document_tokens>{max_doc_length}"
            doc_chunks.extend(
                doc_ids[start : start + max_doc_length]
                for start in range(0, len(doc_ids), max_doc_length)
            )
        if len(doc_chunks) > max_doc_num:
            if not truncate_tool_definition:
                return None, f"tool_definition_docs>{max_doc_num}"
            doc_chunks = doc_chunks[:max_doc_num]
        context_input_ids: List[int] = []
        for chunk in doc_chunks:
            context_input_ids.extend(_pad(chunk, max_doc_length, -100))
        context_input_ids.extend([-100] * (max_doc_length * (max_doc_num - len(doc_chunks))))

        prompt_ids = _chat_template_ids(
            tokenizer,
            example.input_messages,
            add_generation_prompt=True,
        )
        answer_ids = tokenizer.encode(example.answer, add_special_tokens=False)
        if not prompt_ids or not answer_ids:
            return None, "empty_prompt_or_answer"
        answer_ids.append(tokenizer.eos_token_id)
        reserved_target_tokens = min(len(answer_ids), max(1, min_target_tokens))
        if reserved_target_tokens >= max_length:
            reserved_target_tokens = max(1, max_length // 2)
        prompt_budget = max_length - reserved_target_tokens
        if len(prompt_ids) > prompt_budget:
            prompt_ids = prompt_ids[-prompt_budget:]
        answer_budget = max_length - len(prompt_ids)
        answer_ids = answer_ids[:answer_budget]
        if len(answer_ids) < reserved_target_tokens:
            return None, f"target_tokens<{reserved_target_tokens}"
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
        }, "ok"


def main() -> None:
    parser = HfArgumentParser([ModelArgs, TrainingArgs, AgentToolDefinitionDataArgs])
    model_args, training_args, data_args = parser.parse_args_into_dataclasses()
    device_type = _setup_device(model_args, data_args)

    if model_args.gist_gradient_checkpointing:
        import models.gist_utils as _gist_utils

        _gist_utils.GIST_GRADIENT_CHECKPOINTING = True

    model, tokenizer = get_model_and_tokenizer(
        model_args,
        device=device_type,
        evaluation_mode=not training_args.do_train,
    )

    if model_args.enable_gist and training_args.only_train_gist:
        for name, param in model.named_parameters():
            param.requires_grad_("gist" in name)

    logger.info(f"Total Model params: {format_numel_str(sum(p.numel() for p in model.parameters()))}")
    logger.info(
        "Trainable Model params: "
        f"{format_numel_str(sum(p.numel() for p in model.parameters() if p.requires_grad))}"
    )

    with training_args.main_process_first(desc="Build agent tool-definition dataset"):
        source = AgentLLMTracesSource(data_args)
        train_examples = list(source.iter_examples("train"))
        eval_examples = list(source.iter_examples("eval"))
        logger.info("Source-stage skipped spans by reason=%s", dict(source.source_skips))
        train_session_ids = {example.session_id for example in train_examples}
        eval_session_ids = {example.session_id for example in eval_examples}
        overlap = train_session_ids & eval_session_ids
        if overlap:
            raise RuntimeError(f"Session split leakage detected: {sorted(overlap)[:5]}")
        logger.info(
            "Expanded %d train samples from %d sessions; %d eval samples from %d sessions",
            len(train_examples),
            len(train_session_ids),
            len(eval_examples),
            len(eval_session_ids),
        )
        train_dataset = AgentToolDefinitionDataset(
            train_examples,
            tokenizer=tokenizer,
            max_doc_length=data_args.max_doc_length,
            max_doc_num=data_args.max_doc_num,
            max_length=data_args.max_length,
            max_system_length=data_args.max_system_length,
            truncate_tool_definition=data_args.truncate_tool_definition,
            min_target_tokens=data_args.min_target_tokens,
            max_tool_definition_tokens=data_args.max_tool_definition_tokens,
        )
        eval_dataset = AgentToolDefinitionDataset(
            eval_examples,
            tokenizer=tokenizer,
            max_doc_length=data_args.max_doc_length,
            max_doc_num=data_args.max_doc_num,
            max_length=data_args.max_length,
            max_system_length=data_args.max_system_length,
            truncate_tool_definition=data_args.truncate_tool_definition,
            min_target_tokens=data_args.min_target_tokens,
            max_tool_definition_tokens=data_args.max_tool_definition_tokens,
        )

    if len(train_dataset) == 0:
        raise RuntimeError("No train samples remained after filtering")
    if len(eval_dataset) == 0:
        logger.warning("No eval samples remained after filtering")

    trainer = GistMultiDocTrainer(
        model=model,
        args=training_args,
        max_doc_length=train_dataset.max_doc_length,
        model_args=model_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if len(eval_dataset) else None,
        data_collator=DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=True,
            return_tensors="pt",
        ),
    )

    if training_args.do_train:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
    else:
        if len(eval_dataset) == 0:
            raise ValueError("Evaluation requested but eval dataset is empty")
        eval_result = trainer.evaluate()
        with training_args.main_process_first(desc="Evaluate model"):
            logger.info(f"Evaluation result: {eval_result}")


if __name__ == "__main__":
    main()
