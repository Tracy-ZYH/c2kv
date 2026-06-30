from __future__ import annotations

import json
import re
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


@dataclass(frozen=True)
class AgentExample:
    qid: str
    system_prompt: str
    tools: List[Dict[str, Any]]
    messages: List[Dict[str, Any]]
    expected_tool_calls: List[Dict[str, Any]]
    max_new_tokens: Optional[int] = None


class AgentDataset(ABC):
    """Base interface for agent evaluation datasets.

    Every example exposes the three inputs needed by an agent chat template:
    a system prompt, tool definitions, and conversation messages. Tool
    definitions are deliberately not encoded as ordinary message content.
    """

    default_system_prompt = DEFAULT_SYSTEM_PROMPT
    max_new_tokens = 128

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> AgentExample:
        raise NotImplementedError

    def score(self, prediction: str, example: AgentExample) -> float:
        predicted = extract_tool_calls(prediction)
        return float(
            canonical_tool_calls(predicted)
            == canonical_tool_calls(example.expected_tool_calls)
        )


def _json_loads(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _message_parts(messages: Sequence[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    for message in messages:
        yield from message.get("parts") or []


def normalize_tools(tools: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for tool in tools:
        if isinstance(tool.get("function"), dict):
            normalized.append(dict(tool))
            continue
        normalized.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters")
                    or {"type": "object", "properties": {}},
                },
            }
        )
    return normalized


def normalize_messages(
    messages: Sequence[Dict[str, Any]],
) -> tuple[str, List[Dict[str, Any]]]:
    system_parts: List[str] = []
    normalized: List[Dict[str, Any]] = []
    for message in messages:
        role = message.get("role", "user")
        parts = message.get("parts")
        if parts is None:
            content = message.get("content", "")
        else:
            content = "\n".join(
                str(part.get("content", ""))
                for part in parts
                if part.get("type") == "text"
            )
        if role == "system":
            if content:
                system_parts.append(content)
            continue
        if content or role in {"assistant", "tool"}:
            normalized.append({"role": role, "content": content})
    system_prompt = "\n".join(system_parts).strip() or DEFAULT_SYSTEM_PROMPT
    return system_prompt, normalized


def _normalize_call(call: Dict[str, Any]) -> Dict[str, Any]:
    function = call.get("function") if isinstance(call.get("function"), dict) else call
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            pass
    return {
        "name": function.get("name", ""),
        "arguments": arguments,
    }


def canonical_tool_calls(calls: Sequence[Dict[str, Any]]) -> str:
    return json.dumps(
        [_normalize_call(call) for call in calls],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def extract_tool_calls(text: str) -> List[Dict[str, Any]]:
    payloads = _TOOL_CALL_RE.findall(text)
    if not payloads:
        payloads = [text.strip()]
    calls = []
    for payload in payloads:
        try:
            call = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(call, dict):
            calls.append(call)
        elif isinstance(call, list):
            calls.extend(item for item in call if isinstance(item, dict))
    return calls


def _agent_example_to_record(example: AgentExample) -> Dict[str, Any]:
    return {
        "qid": example.qid,
        "system_prompt": example.system_prompt,
        "tools": json.dumps(example.tools, ensure_ascii=False),
        "messages": json.dumps(example.messages, ensure_ascii=False),
        "expected_tool_calls": json.dumps(
            example.expected_tool_calls,
            ensure_ascii=False,
        ),
        "max_new_tokens": example.max_new_tokens,
    }


def _record_to_agent_example(record: Dict[str, Any]) -> AgentExample:
    return AgentExample(
        qid=record["qid"],
        system_prompt=record["system_prompt"],
        tools=_json_loads(record["tools"], []),
        messages=_json_loads(record["messages"], []),
        expected_tool_calls=_json_loads(record["expected_tool_calls"], []),
        max_new_tokens=record.get("max_new_tokens"),
    )


def _normalize_benchmarks(
    benchmark: Optional[str | Sequence[str]],
) -> Optional[List[str]]:
    if benchmark is None:
        return None
    if isinstance(benchmark, str):
        values = [item.strip() for item in benchmark.split(",")]
    else:
        values = [str(item).strip() for item in benchmark]
    values = [item for item in values if item]
    return values or None


def _dataset_fingerprint(
    files: Sequence[Path],
    max_tools: Optional[int],
    max_input_tokens: Optional[int],
    max_new_tokens: int,
    benchmarks: Optional[Sequence[str]],
) -> str:
    payload = {
        "version": 3,
        "max_tools": max_tools,
        "max_input_tokens": max_input_tokens,
        "max_new_tokens": max_new_tokens,
        "benchmarks": sorted(benchmarks) if benchmarks is not None else None,
        "files": [
            {
                "path": str(file.resolve()),
                "size": file.stat().st_size,
                "mtime_ns": file.stat().st_mtime_ns,
            }
            for file in files
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def _iter_agent_llm_trace_records(
    files: Sequence[str],
    max_tools: Optional[int],
    max_input_tokens: Optional[int],
    max_new_tokens: int,
    benchmarks: Optional[Sequence[str]],
) -> Iterator[Dict[str, Any]]:
    import pyarrow.parquet as pq

    benchmark_set = set(benchmarks) if benchmarks is not None else None
    for file_name in files:
        file = Path(file_name)
        table = pq.read_table(file)
        for row_index, row in enumerate(table.to_pylist()):
            row_benchmark = row.get("benchmark") or ""
            for span_index, span in enumerate(row.get("spans") or []):
                benchmark = span.get("benchmark") or row_benchmark
                if benchmark_set is not None and benchmark not in benchmark_set:
                    continue
                example = AgentLLMTracesDataset._extract_example(
                    row,
                    span,
                    f"{file.name}:{row_index}:{span_index}",
                    max_tools=max_tools,
                    max_input_tokens=max_input_tokens,
                    max_new_tokens=max_new_tokens,
                )
                if example is not None:
                    yield _agent_example_to_record(example)


class AgentLLMTracesDataset(AgentDataset):
    """Simple tool-call subset of Exgentic/agent-llm-traces."""

    def __init__(
        self,
        data_path: str,
        max_samples: Optional[int] = None,
        max_tools: Optional[int] = None,
        max_input_tokens: Optional[int] = None,
        max_new_tokens: int = 128,
        benchmark: Optional[str | Sequence[str]] = None,
        use_hf_cache: bool = True,
        dataset_cache_dir: Optional[str] = None,
    ) -> None:
        self.data_path = Path(data_path).expanduser()
        data_dir = (
            self.data_path / "data"
            if (self.data_path / "data").is_dir()
            else self.data_path
        )
        files = sorted(data_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found under {data_dir}")

        self.max_new_tokens = max_new_tokens
        benchmarks = _normalize_benchmarks(benchmark)
        if use_hf_cache:
            self.dataset = self._load_hf_dataset(
                files,
                max_samples=max_samples,
                max_tools=max_tools,
                max_input_tokens=max_input_tokens,
                max_new_tokens=max_new_tokens,
                benchmarks=benchmarks,
                dataset_cache_dir=dataset_cache_dir,
            )
            self.examples = None
        else:
            self.dataset = None
            self.examples: Optional[List[AgentExample]] = []
            for record in _iter_agent_llm_trace_records(
                [str(file) for file in files],
                max_tools=max_tools,
                max_input_tokens=max_input_tokens,
                max_new_tokens=max_new_tokens,
                benchmarks=benchmarks,
            ):
                self.examples.append(_record_to_agent_example(record))
                if max_samples is not None and len(self.examples) >= max_samples:
                    return

    @staticmethod
    def _load_hf_dataset(
        files: Sequence[Path],
        max_samples: Optional[int],
        max_tools: Optional[int],
        max_input_tokens: Optional[int],
        max_new_tokens: int,
        benchmarks: Optional[Sequence[str]],
        dataset_cache_dir: Optional[str],
    ) -> Any:
        from datasets import Dataset, Features, Value

        features = Features(
            {
                "qid": Value("string"),
                "system_prompt": Value("string"),
                "tools": Value("string"),
                "messages": Value("string"),
                "expected_tool_calls": Value("string"),
                "max_new_tokens": Value("int64"),
            }
        )
        dataset = Dataset.from_generator(
            _iter_agent_llm_trace_records,
            features=features,
            cache_dir=dataset_cache_dir,
            gen_kwargs={
                "files": [str(file) for file in files],
                "max_tools": max_tools,
                "max_input_tokens": max_input_tokens,
                "max_new_tokens": max_new_tokens,
                "benchmarks": tuple(benchmarks) if benchmarks is not None else None,
            },
            fingerprint=_dataset_fingerprint(
                files,
                max_tools=max_tools,
                max_input_tokens=max_input_tokens,
                max_new_tokens=max_new_tokens,
                benchmarks=benchmarks,
            ),
        )
        if max_samples is not None and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
        return dataset

    @staticmethod
    def _extract_example(
        row: Dict[str, Any],
        span: Dict[str, Any],
        qid: str,
        max_tools: Optional[int],
        max_input_tokens: Optional[int],
        max_new_tokens: int = 128,
    ) -> Optional[AgentExample]:
        attrs = span.get("attributes") or {}
        raw_messages = _json_loads(attrs.get("gen_ai.input.messages"), [])
        raw_outputs = _json_loads(attrs.get("gen_ai.output.messages"), [])
        raw_tools = _json_loads(attrs.get("gen_ai.tool.definitions"), [])
        expected_calls = [
            part
            for part in _message_parts(raw_outputs)
            if part.get("type") == "tool_call"
        ]
        prior_tool_use = any(
            part.get("type") in {"tool_call", "tool_call_response", "tool_result"}
            for part in _message_parts(raw_messages)
        )
        input_tokens = attrs.get("gen_ai.usage.input_tokens") or 0
        if not raw_messages or not raw_tools or prior_tool_use or not expected_calls:
            return None
        if max_tools is not None and len(raw_tools) > max_tools:
            return None
        if max_input_tokens is not None and input_tokens > max_input_tokens:
            return None

        system_prompt, messages = normalize_messages(raw_messages)
        if not messages:
            return None
        return AgentExample(
            qid=qid,
            system_prompt=system_prompt,
            tools=normalize_tools(raw_tools),
            messages=messages,
            expected_tool_calls=[_normalize_call(call) for call in expected_calls],
            max_new_tokens=max_new_tokens,
        )

    def __len__(self) -> int:
        if self.dataset is not None:
            return len(self.dataset)
        assert self.examples is not None
        return len(self.examples)

    def __getitem__(self, index: int) -> AgentExample:
        if self.dataset is not None:
            return _record_to_agent_example(self.dataset[index])
        assert self.examples is not None
        return self.examples[index]


def load_agent_dataset(
    name: str,
    path: str,
    **kwargs: Any,
) -> AgentDataset:
    if name == "agent_llm_traces":
        return AgentLLMTracesDataset(path, **kwargs)
    raise ValueError(f"Unsupported agent dataset: {name}")
