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


def _tool_list_from_agent_value(value: Any) -> List[Dict[str, Any]]:
    parsed = _json_loads(value, [])
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
    original_messages: List[Message] = field(default_factory=list)


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
                original_messages=list(record.get("original_messages") or []),
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


def _first_value(obj: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in obj and obj[key] is not None:
            return obj[key]
    return default


def _find_agent_parquet_files(path: Path) -> List[Path]:
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


def _find_agent_jsonl_files(path: Path) -> List[Path]:
    if path.is_file() and path.suffix == ".jsonl":
        return [path]
    roots = [path / "data", path]
    files: List[Path] = []
    for root in roots:
        if root.is_dir():
            files = sorted(root.glob("*.jsonl"))
            if not files:
                files = sorted(root.rglob("*.jsonl"))
        if files:
            break
    return files


def _iter_agent_rows(data_files: Sequence[Path]) -> Iterator[Dict[str, Any]]:
    import pyarrow.parquet as pq

    wanted = ["benchmark", "subset", "dataset", "task", "session_id", "trace_id", "id", "spans"]
    for data_file in data_files:
        pf = pq.ParquetFile(data_file)
        available = set(pf.schema_arrow.names)
        columns = [column for column in wanted if column in available]
        for batch in pf.iter_batches(batch_size=256, columns=columns):
            yield from batch.to_pylist()


def _iter_agent_jsonl_rows(data_files: Sequence[Path]) -> Iterator[Dict[str, Any]]:
    for data_file in data_files:
        with data_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = _json_loads(line, None)
                if isinstance(row, dict):
                    yield row


def _span_attributes(span: Any) -> Dict[str, Any]:
    span = _json_loads(span, span)
    if not isinstance(span, dict):
        return {}
    attributes = span.get("attributes", span)
    attributes = _json_loads(attributes, attributes)
    return attributes if isinstance(attributes, dict) else {}


def _sort_agent_spans(spans: Sequence[Any]) -> List[Dict[str, Any]]:
    return sorted(
        [span for span in spans if isinstance(span, dict)],
        key=lambda span: (
            span.get("start_time") or "",
            span.get("span_id") or "",
        ),
    )


def _toolathlon_row_to_agent_session(row: Dict[str, Any], row_index: int) -> Optional[Dict[str, Any]]:
    tools = _tool_list_from_agent_value(row.get("tool_calls"))
    messages = _json_loads(row.get("messages"), [])
    if not tools or not isinstance(messages, list):
        return None
    config = _json_loads(row.get("config"), {})
    system_prompt = ""
    if isinstance(config, dict):
        system_prompts = config.get("system_prompts") if isinstance(config.get("system_prompts"), dict) else {}
        system_prompt = str(system_prompts.get("agent") or "").strip()
    system_message = {"role": "system", "content": system_prompt} if system_prompt else None
    spans = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        has_tool_call = bool(
            message.get("tool_calls")
            or message.get("toolCalls")
            or message.get("function_call")
            or _agent_message_parts(message)
        )
        if not has_tool_call:
            continue
        input_messages = [item for item in messages[:message_index] if isinstance(item, dict)]
        if system_message and not any(item.get("role") == "system" for item in input_messages):
            input_messages = [system_message, *input_messages]
        if not any(item.get("role") == "user" for item in input_messages):
            continue
        spans.append({
            "start_time": f"{message_index:06d}",
            "span_id": f"toolathlon-{message_index}",
            "attributes": {
                "gen_ai.tool.definitions": _json_dumps(tools),
                "gen_ai.input.messages": _json_dumps(input_messages),
                "gen_ai.output.messages": _json_dumps([message]),
            },
        })
    if not spans:
        return None
    return {
        "session_id": str(row.get("request_id") or row.get("task_name") or f"toolathlon-row-{row_index}"),
        "subset": str(row.get("task_name") or row.get("modelname_run") or "toolathlon"),
        "spans": spans,
    }


def _agent_message_parts(message: Message) -> List[Dict[str, Any]]:
    parts = message.get("parts")
    parts = _json_loads(parts, parts)
    if isinstance(parts, dict):
        return [parts]
    if isinstance(parts, list):
        return [part for part in parts if isinstance(part, dict)]
    return []


def _render_agent_tool_calls(tool_calls: Any) -> str:
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


def _agent_message_content_to_text(message: Message) -> str:
    content = message.get("content", "")
    if not content and _agent_message_parts(message):
        content = _agent_message_parts(message)
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


def _normal_agent_message(message: Message) -> Optional[Message]:
    if not isinstance(message, dict):
        return None
    role = message.get("role") or message.get("type") or "user"
    if role == "tool":
        role = "user"
    content_parts = []
    content = _agent_message_content_to_text(message)
    if content:
        content_parts.append(content)
    tool_calls_text = _render_agent_tool_calls(
        message.get("tool_calls")
        or message.get("toolCalls")
        or message.get("function_call")
        or _agent_message_parts(message)
    )
    if tool_calls_text:
        content_parts.append("Action:\n" + tool_calls_text)
    if not content_parts and role != "assistant":
        return None
    return {"role": role, "content": "\n\n".join(content_parts)}


def _render_agent_output_messages(value: Any, max_answer_chars: Optional[int]) -> tuple[str, bool]:
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
        content = _agent_message_content_to_text(message).strip()
        if content:
            parts.append(content)
        tool_calls_text = _render_agent_tool_calls(
            message.get("tool_calls")
            or message.get("toolCalls")
            or message.get("function_call")
            or _agent_message_parts(message)
        )
        if tool_calls_text:
            parts.append("Action:\n" + tool_calls_text)
        has_tool_call = has_tool_call or bool(tool_calls_text)
        rendered = "\n\n".join(part for part in parts if part).strip()
        if rendered:
            rendered_messages.append(rendered)
    answer = "\n\n".join(rendered_messages).strip()
    if max_answer_chars is not None and len(answer) > max_answer_chars:
        answer = answer[:max_answer_chars].rstrip()
    marker_text = answer.lower()
    has_tool_call = has_tool_call or any(
        marker in marker_text
        for marker in ("<tool_call>", "action:", "function_call", "tool call")
    )
    return answer, has_tool_call


def _agent_system_prompt(messages: Sequence[Message]) -> str:
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "system":
            content = _agent_message_content_to_text(message).strip()
            if content:
                return content
    return DEFAULT_SYSTEM_PROMPT


def _agent_history_turn_docs(messages: Sequence[Message]) -> List[Message]:
    docs: List[Message] = []
    current_user: Optional[str] = None
    outputs: List[str] = []

    def flush() -> None:
        nonlocal current_user, outputs
        if current_user is None and not outputs:
            return
        parts = ["Previous turn"]
        if current_user:
            parts.extend(["[User query]", current_user.strip()])
        if outputs:
            parts.extend(["[Assistant output]", "\n\n".join(item.strip() for item in outputs if item.strip())])
        docs.append({"role": "user", "content": "\n".join(parts).strip()})
        current_user = None
        outputs = []

    for message in messages:
        role = message.get("role", "user")
        content = str(message.get("content") or "").strip()
        if not content and role != "assistant":
            continue
        if role == "user":
            flush()
            current_user = content
        elif role == "assistant":
            outputs.append(content)
        else:
            outputs.append(f"[{role}]\n{content}")
    flush()
    return docs


class AgentLLMTracesCompressHistorySource(CompressHistorySource):
    """Turn-level history compression source for agent-llm-traces.

    For every LLM span, the current prompt is the last user message in that
    span's input.  Earlier input messages are grouped into turn-level documents,
    each containing a previous user query and its assistant/tool output.
    """

    def __init__(
        self,
        path: str,
        split: str = "train",
        eval_ratio: float = 0.1,
        split_seed: int = 42,
        split_manifest_file: Optional[str] = None,
        split_manifest_name: str = "subset_disjoint",
        max_samples_per_session: Optional[int] = 4,
        max_records: Optional[int] = None,
        require_tool_call: bool = False,
        max_input_chars: Optional[int] = None,
        max_answer_chars: Optional[int] = None,
        include_tools: bool = False,
        prefix_history_doc_num: Optional[int] = None,
        prefix_history_exact: bool = False,
    ) -> None:
        self.path = Path(path)
        self.split = split
        self.eval_ratio = eval_ratio
        self.split_seed = split_seed
        self.split_manifest_file = split_manifest_file
        self.split_manifest_name = split_manifest_name
        self.max_samples_per_session = max_samples_per_session
        self.max_records = max_records
        self.require_tool_call = require_tool_call
        self.max_input_chars = max_input_chars
        self.max_answer_chars = max_answer_chars
        self.include_tools = include_tools
        self.prefix_history_doc_num = prefix_history_doc_num
        self.prefix_history_exact = prefix_history_exact
        self.records = self._load_records()

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[CompressHistoryExample]:
        yield from self.records

    def _load_records(self) -> List[CompressHistoryExample]:
        data_files = _find_agent_parquet_files(self.path)
        jsonl_files: List[Path] = []
        if not data_files:
            jsonl_files = _find_agent_jsonl_files(self.path)
        if not data_files and not jsonl_files:
            raise FileNotFoundError(f"No parquet/jsonl files found under {self.path}")
        if self.split_manifest_file:
            return self._load_records_from_manifest(data_files or jsonl_files)
        sessions = []
        if data_files:
            for row_index, row in enumerate(_iter_agent_rows(data_files)):
                session_id = str(
                    row.get("session_id")
                    or row.get("trace_id")
                    or row.get("id")
                    or f"row-{row_index}"
                )
                subset = str(row.get("benchmark") or row.get("subset") or row.get("dataset") or row.get("task") or "unknown")
                spans = _sort_agent_spans(_json_loads(row.get("spans"), row.get("spans")) or [])
                sessions.append({"session_id": session_id, "subset": subset, "spans": spans})
        else:
            for row_index, row in enumerate(_iter_agent_jsonl_rows(jsonl_files)):
                session = _toolathlon_row_to_agent_session(row, row_index)
                if session is not None:
                    sessions.append(session)

        train_ids, eval_ids = self._split_session_ids(sessions)
        keep_ids = train_ids if self.split == "train" else eval_ids
        rng = random.Random(self.split_seed + (0 if self.split == "train" else 1))
        records: List[CompressHistoryExample] = []
        for session in sessions:
            if session["session_id"] not in keep_ids:
                continue
            examples = self._session_examples(session["session_id"], session["spans"])
            if self.max_samples_per_session and len(examples) > self.max_samples_per_session:
                examples = rng.sample(examples, self.max_samples_per_session)
            records.extend(examples)
            if self.max_records is not None and len(records) >= self.max_records:
                return records[: self.max_records]
        return records

    def _load_records_from_manifest(self, data_files: Sequence[Path]) -> List[CompressHistoryExample]:
        manifest = json.loads(Path(self.split_manifest_file).read_text(encoding="utf-8"))
        if "train_session_ids" in manifest and "eval_session_ids" in manifest:
            selected = manifest
        else:
            selected = manifest[self.split_manifest_name]
        keep_ids = {
            str(item)
            for item in selected.get(
                "train_session_ids" if self.split == "train" else "eval_session_ids",
                [],
            )
        }
        rng = random.Random(self.split_seed + (0 if self.split == "train" else 1))
        records: List[CompressHistoryExample] = []
        row_iter = _iter_agent_rows(data_files) if data_files and data_files[0].suffix == ".parquet" else _iter_agent_jsonl_rows(data_files)
        for row_index, row in enumerate(row_iter):
            session_id = str(
                row.get("session_id")
                or row.get("trace_id")
                or row.get("id")
                or f"row-{row_index}"
            )
            if data_files and data_files[0].suffix == ".jsonl":
                session = _toolathlon_row_to_agent_session(row, row_index)
                if session is None:
                    continue
                session_id = session["session_id"]
                spans = session["spans"]
            else:
                spans = _sort_agent_spans(_json_loads(row.get("spans"), row.get("spans")) or [])
            if session_id not in keep_ids:
                continue
            examples = self._session_examples(session_id, spans)
            if self.max_samples_per_session and len(examples) > self.max_samples_per_session:
                examples = rng.sample(examples, self.max_samples_per_session)
            records.extend(examples)
            if self.max_records is not None and len(records) >= self.max_records:
                return records[: self.max_records]
        return records

    def _split_session_ids(self, sessions: Sequence[Dict[str, Any]]) -> tuple[set[str], set[str]]:
        if self.split_manifest_file:
            manifest = json.loads(Path(self.split_manifest_file).read_text(encoding="utf-8"))
            if "train_session_ids" in manifest and "eval_session_ids" in manifest:
                selected = manifest
            else:
                selected = manifest[self.split_manifest_name]
            available = {session["session_id"] for session in sessions}
            train_ids = {str(item) for item in selected.get("train_session_ids", [])} & available
            eval_ids = {str(item) for item in selected.get("eval_session_ids", [])} & available
            return train_ids, eval_ids

        session_ids = sorted(session["session_id"] for session in sessions)
        rng = random.Random(self.split_seed)
        rng.shuffle(session_ids)
        eval_count = max(1, int(round(len(session_ids) * self.eval_ratio))) if session_ids else 0
        eval_ids = set(session_ids[:eval_count])
        train_ids = set(session_ids[eval_count:])
        return train_ids, eval_ids

    def _session_examples(self, session_id: str, spans: Sequence[Any]) -> List[CompressHistoryExample]:
        examples = []
        tools: List[Dict[str, Any]] = []
        for span_index, span in enumerate(spans):
            attributes = _span_attributes(span)
            if self.include_tools and not tools:
                tools = _tool_list_from_agent_value(attributes.get("gen_ai.tool.definitions"))
            raw_input_messages = _json_loads(attributes.get("gen_ai.input.messages"), [])
            output_messages = attributes.get("gen_ai.output.messages")
            if not raw_input_messages or output_messages is None:
                continue
            if self.max_input_chars is not None and len(str(raw_input_messages)) > self.max_input_chars:
                continue
            system_prompt = _agent_system_prompt(_json_loads(raw_input_messages, []))
            messages = [
                item
                for item in (_normal_agent_message(message) for message in _json_loads(raw_input_messages, []))
                if item is not None and item.get("role") != "system"
            ]
            last_user_index = next(
                (index for index in range(len(messages) - 1, -1, -1) if messages[index].get("role") == "user"),
                None,
            )
            if last_user_index is None:
                continue
            history_docs = _agent_history_turn_docs(messages[:last_user_index])
            if self.prefix_history_doc_num is not None:
                if len(history_docs) < self.prefix_history_doc_num:
                    continue
                if self.prefix_history_exact and len(history_docs) != self.prefix_history_doc_num:
                    continue
                history_docs = history_docs[-self.prefix_history_doc_num :]
            current_messages = messages[last_user_index:]
            answer, has_tool_call = _render_agent_output_messages(output_messages, self.max_answer_chars)
            if self.require_tool_call and not has_tool_call:
                continue
            if not history_docs or not current_messages or not answer:
                continue
            examples.append(
                CompressHistoryExample(
                    qid=f"{session_id}:{span_index}",
                    history_messages=history_docs,
                    current_messages=current_messages,
                    answer=answer,
                    system_prompt=system_prompt,
                    tools=list(tools) if self.include_tools else [],
                    original_messages=[
                        message
                        for message in _json_loads(raw_input_messages, [])
                        if isinstance(message, dict)
                    ],
                )
            )
        return examples


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
    split_oversized_history_docs: bool = True,
) -> List[Message]:
    if split_oversized_history_docs:
        split_messages: List[Message] = []
        for message in messages:
            split_messages.extend(_split_message_to_fit(tokenizer, message, max_doc_length))
        messages = list(split_messages)
    else:
        messages = [
            message
            for message in messages
            if _message_token_length(tokenizer, message) <= max_doc_length
        ]
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
    full_history_doc_num: int = 0,
    split_oversized_history_docs: bool = True,
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
        full_history_doc_num=full_history_doc_num,
        split_oversized_history_docs=split_oversized_history_docs,
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
        full_history_doc_num: int = 0,
        split_oversized_history_docs: bool = True,
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
                    "full_history_doc_num": full_history_doc_num,
                    "split_oversized_history_docs": split_oversized_history_docs,
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
                    full_history_doc_num=full_history_doc_num,
                    split_oversized_history_docs=split_oversized_history_docs,
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
        full_history_doc_num: int = 0,
        split_oversized_history_docs: bool = True,
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
            split_oversized_history_docs=split_oversized_history_docs,
        )
        current = [
            _normal_chat_message(message)
            for message in example.current_messages
            if message.get("content") or message.get("role") == "assistant"
        ]
        if len(history) < min_doc_num or not current or not example.answer:
            return None
        if full_history_doc_num < 0:
            raise ValueError(f"full_history_doc_num must be non-negative, got {full_history_doc_num}")
        if full_history_doc_num:
            if full_history_doc_num >= len(history):
                compressed_history: List[Message] = []
                full_history = list(history)
            else:
                compressed_history = list(history[:-full_history_doc_num])
                full_history = list(history[-full_history_doc_num:])
        else:
            compressed_history = list(history)
            full_history = []

        system_ids = _chat_template_ids(
            tokenizer,
            [{"role": "system", "content": example.system_prompt}],
            tools=example.tools or None,
            keep_bos=True,
            max_length=max_system_length,
        )
        system_input_ids = _pad(system_ids, max_system_length, -100)

        context_input_ids: List[int] = []
        for message in compressed_history:
            doc_ids = _chat_template_ids(
                tokenizer,
                [message],
                max_length=max_doc_length,
            )
            context_input_ids.extend(_pad(doc_ids, max_doc_length, -100))
        empty_docs = max_doc_num - len(compressed_history)
        context_input_ids.extend([-100] * (max_doc_length * empty_docs))

        full_history_ids: List[int] = []
        for message in full_history:
            full_history_ids.extend(
                _chat_template_ids(
                    tokenizer,
                    [message],
                    max_length=max_doc_length,
                )
            )
        prompt_ids = _chat_template_ids(
            tokenizer,
            current,
            add_generation_prompt=True,
        )
        prompt_ids = full_history_ids + prompt_ids
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
    if source_type == "agent_llm_traces":
        return AgentLLMTracesCompressHistorySource(path)
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
    elif source_type == "agent_llm_traces":
        source = AgentLLMTracesCompressHistorySource(path, **(source_kwargs or {}))
    else:
        source = load_compress_history_source(source_type, path)
    return CompressHistoryDataset(source, tokenizer=tokenizer, **kwargs)
