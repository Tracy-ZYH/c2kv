from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import requests
from tqdm import tqdm
from transformers import AutoTokenizer

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))


HTTP = requests.Session()
HTTP.trust_env = False
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


@dataclass
class AgentToolDefinitionDataArgs:
    dataset_path: str = "./datasets/agent-llm-traces"
    eval_ratio: float = 0.1
    split_seed: int = 42
    split_manifest_file: Optional[str] = None
    split_manifest_name: str = "subset_disjoint"
    max_sessions: Optional[int] = None
    max_samples_per_session: int = 4
    max_doc_length: int = 2048
    max_doc_num: int = 16
    max_length: int = 2048
    max_system_length: int = 256
    max_tool_definition_tokens: int = 131072
    min_target_tokens: int = 128
    truncate_tool_definition: bool = False
    require_tool_call: bool = True
    tool_document_mode: str = "full"
    hard_negative_num: int = 15
    hard_negative_router_scope: str = "last_user"
    shuffle_tool_documents: bool = True
    balance_subsets: bool = True
    max_samples_per_subset: Optional[int] = None
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


@dataclass(frozen=True)
class AgentToolDefinitionExample:
    qid: str
    session_id: str
    tool_definition: str
    input_messages: List[Dict[str, Any]]
    answer: str
    has_tool_call: bool
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    tool_documents: Optional[List[str]] = None
    subset: str = "unknown"


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


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
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _as_tool_list(tool_definition: Any) -> List[Dict[str, Any]]:
    parsed = _json_loads(tool_definition, [])
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


def _tool_name(tool: Dict[str, Any]) -> str:
    function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
    return str(
        function.get("name")
        or tool.get("name")
        or tool.get("tool_name")
        or tool.get("function_name")
        or ""
    )


def _tool_search_text(tool: Dict[str, Any]) -> str:
    function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
    fields = [
        _tool_name(tool),
        function.get("description", ""),
        tool.get("description", ""),
        function.get("parameters", ""),
        tool.get("parameters", ""),
        tool.get("input_schema", ""),
        tool.get("schema", ""),
    ]
    return " ".join(
        item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
        for item in fields
        if item
    )


def _message_text(message: Dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False) if content is not None else ""


def _query_text(messages: Sequence[Dict[str, Any]], scope: str) -> str:
    if scope == "all":
        return "\n".join(_message_text(message) for message in messages)
    for message in reversed(messages):
        if message.get("role") == "user":
            return _message_text(message)
    return _message_text(messages[-1]) if messages else ""


def _rank_tokens(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _rank_tools(tools: Sequence[Dict[str, Any]], query: str) -> List[int]:
    query_tokens = set(_rank_tokens(query))
    if not query_tokens:
        return list(range(len(tools)))
    scored = []
    for index, tool in enumerate(tools):
        name_tokens = set(_rank_tokens(_tool_name(tool)))
        text_tokens = set(_rank_tokens(_tool_search_text(tool)))
        score = 4.0 * len(query_tokens & name_tokens) + float(
            len(query_tokens & text_tokens)
        )
        scored.append((-score, index))
    scored.sort()
    return [index for _, index in scored]


def _split_topk_tools(
    tools: Sequence[Dict[str, Any]],
    query: str,
    top_k: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    ranked = _rank_tools(tools, query)
    top_indices = set(ranked[: max(0, top_k)])
    top_tools = [tool for index, tool in enumerate(tools) if index in top_indices]
    rest_tools = [tool for index, tool in enumerate(tools) if index not in top_indices]
    return top_tools, rest_tools, [_tool_name(tool) for tool in top_tools]


def _render_tool_definition(tools: Sequence[Dict[str, Any]]) -> str:
    return json.dumps(list(tools), ensure_ascii=False, separators=(",", ":"))


def _extract_tool_name(text: str) -> Optional[str]:
    if not text:
        return None
    patterns = [
        r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
        r"Action:\s*(?:<tool_call>)?\s*(\{.*?\})(?:\s*</tool_call>)?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.S)
        if not match:
            continue
        payload = _json_loads(match.group(1), None)
        if not isinstance(payload, dict):
            continue
        function = payload.get("function") if isinstance(payload.get("function"), dict) else {}
        name = (
            function.get("name")
            or payload.get("name")
            or payload.get("tool_name")
            or payload.get("function_name")
        )
        if name:
            return str(name)
    match = re.search(r'"(?:name|tool_name|function_name)"\s*:\s*"([^"]+)"', text)
    return match.group(1) if match else None


def _message_parts(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    parts = message.get("parts")
    parts = _json_loads(parts, parts)
    if isinstance(parts, dict):
        return [parts]
    if isinstance(parts, list):
        return [part for part in parts if isinstance(part, dict)]
    return []


def _message_content_to_text(message: Dict[str, Any]) -> str:
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


def _render_tool_calls(tool_calls: Any) -> Tuple[str, bool]:
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
        rendered.append("<tool_call>\n" + _json_dumps({"name": name, "arguments": arguments}) + "\n</tool_call>")
    return "\n".join(rendered), bool(rendered)


def _normal_message(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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


def _render_output_messages(value: Any) -> Tuple[str, bool]:
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


def _iter_sessions(row: Dict[str, Any], row_index: int) -> Iterator[Tuple[str, List[Dict[str, Any]]]]:
    trace = _json_loads(_first_value(row, ["trace", "Trace"], row), row)
    sessions = _first_value(row, ["trace.sessions", "sessions"], None)
    if sessions is None and isinstance(trace, dict):
        sessions = _first_value(trace, ["trace.sessions", "sessions"], None)
    sessions = _json_loads(sessions, sessions)

    if sessions is None:
        spans = _first_value(row, ["spans", "trace.spans"], None)
        spans = _json_loads(spans, spans)
        if spans is not None:
            sessions = [
                {
                    "session_id": _first_value(row, ["session_id", "id"], f"row-{row_index}"),
                    "spans": spans,
                }
            ]

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
        self.path = Path(args.dataset_path)
        self.source_skips: Counter[str] = Counter()
        parquet_files = self._find_parquet_files(self.path)
        if parquet_files:
            self.sessions = self._load_sessions(self._iter_parquet_rows(parquet_files))
        else:
            jsonl_files = self._find_jsonl_files(self.path)
            if not jsonl_files:
                raise FileNotFoundError(f"No parquet/jsonl files found under {self.path}")
            sessions = []
            for row_index, row in enumerate(self._iter_jsonl_rows(jsonl_files)):
                session = self._toolathlon_row_to_session(row, row_index)
                if session is not None:
                    sessions.append(session)
            self.sessions = sessions
        if args.max_sessions is not None:
            self.sessions = self.sessions[: args.max_sessions]

    @staticmethod
    def _find_parquet_files(path: Path) -> List[str]:
        if path.is_file() and path.suffix == ".parquet":
            return [str(path)]
        files: List[Path] = []
        for root in (path / "data", path):
            if root.is_dir():
                files.extend(sorted(root.glob("*.parquet")))
                if not files:
                    files.extend(sorted(root.rglob("*.parquet")))
            if files:
                break
        return [str(file) for file in files]

    @staticmethod
    def _find_jsonl_files(path: Path) -> List[str]:
        if path.is_file() and path.suffix == ".jsonl":
            return [str(path)]
        files: List[Path] = []
        for root in (path / "data", path):
            if root.is_dir():
                files.extend(sorted(root.glob("*.jsonl")))
                if not files:
                    files.extend(sorted(root.rglob("*.jsonl")))
            if files:
                break
        return [str(file) for file in files]

    @staticmethod
    def _iter_parquet_rows(data_files: Sequence[str]) -> Iterator[Dict[str, Any]]:
        import pyarrow.parquet as pq

        for data_file in data_files:
            table = pq.read_table(data_file)
            for row in table.to_pylist():
                yield row

    @staticmethod
    def _iter_jsonl_rows(data_files: Sequence[str]) -> Iterator[Dict[str, Any]]:
        for data_file in data_files:
            with Path(data_file).open("r", encoding="utf-8") as f:
                for line in f:
                    row = _json_loads(line, None)
                    if isinstance(row, dict):
                        yield row

    @staticmethod
    def _toolathlon_row_to_session(row: Dict[str, Any], row_index: int) -> Optional[Dict[str, Any]]:
        tools = _as_tool_list(_json_loads(row.get("tool_calls"), {}))
        messages = _json_loads(row.get("messages"), [])
        if not tools or not isinstance(messages, list):
            return None
        spans = []
        for message_index, message in enumerate(messages):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            has_tool_call = bool(
                message.get("tool_calls")
                or message.get("toolCalls")
                or message.get("function_call")
                or _message_parts(message)
            )
            if not has_tool_call:
                continue
            input_messages = [item for item in messages[:message_index] if isinstance(item, dict)]
            if not any(item.get("role") == "user" for item in input_messages):
                continue
            spans.append(
                {
                    "start_time": f"{message_index:06d}",
                    "span_id": f"toolathlon-{message_index}",
                    "attributes": {
                        "gen_ai.tool.definitions": _json_dumps(tools),
                        "gen_ai.input.messages": _json_dumps(input_messages),
                        "gen_ai.output.messages": _json_dumps([message]),
                    },
                }
            )
        if not spans:
            return None
        return {
            "session_id": str(row.get("request_id") or row.get("task_name") or f"toolathlon-row-{row_index}"),
            "subset": str(row.get("task_name") or row.get("modelname_run") or "toolathlon"),
            "spans": spans,
        }

    def _load_sessions(self, raw: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sessions_by_id: Dict[str, List[Dict[str, Any]]] = {}
        subset_by_id: Dict[str, str] = {}
        for row_index, row in enumerate(raw):
            row_subset = str(
                row.get("benchmark")
                or row.get("subset")
                or row.get("dataset")
                or row.get("task")
                or "unknown"
            )
            found_nested_session = False
            for session_id, spans in _iter_sessions(row, row_index):
                found_nested_session = True
                sessions_by_id.setdefault(session_id, []).extend(spans)
                subset_by_id.setdefault(session_id, row_subset)
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
            subset_by_id.setdefault(str(session_id), row_subset)
        return [
            {
                "session_id": session_id,
                "subset": subset_by_id.get(session_id, "unknown"),
                "spans": sorted(
                    spans,
                    key=lambda span: (
                        span.get("start_time") or "",
                        span.get("span_id") or "",
                    ),
                ),
            }
            for session_id, spans in sessions_by_id.items()
        ]

    def split_session_ids(self) -> Tuple[set[str], set[str]]:
        if self.args.split_manifest_file:
            manifest = json.loads(Path(self.args.split_manifest_file).read_text(encoding="utf-8"))
            selected = (
                manifest
                if "train_session_ids" in manifest and "eval_session_ids" in manifest
                else manifest[self.args.split_manifest_name]
            )
            train_ids = set(str(item) for item in selected.get("train_session_ids", []))
            eval_ids = set(str(item) for item in selected.get("eval_session_ids", []))
            available = {item["session_id"] for item in self.sessions}
            return train_ids & available, eval_ids & available
        session_ids = sorted({item["session_id"] for item in self.sessions})
        rng = random.Random(self.args.split_seed)
        rng.shuffle(session_ids)
        eval_count = max(1, int(round(len(session_ids) * self.args.eval_ratio))) if session_ids else 0
        return set(session_ids[eval_count:]), set(session_ids[:eval_count])

    def iter_examples(self, split: str) -> Iterator[AgentToolDefinitionExample]:
        train_ids, eval_ids = self.split_session_ids()
        keep_ids = train_ids if split == "train" else eval_ids
        rng = random.Random(self.args.split_seed + (0 if split == "train" else 1))
        for session in self.sessions:
            session_id = session["session_id"]
            if session_id not in keep_ids:
                continue
            candidates = self._session_examples(session_id, session["spans"], str(session.get("subset") or "unknown"))
            if self.args.max_samples_per_session and len(candidates) > self.args.max_samples_per_session:
                candidates = rng.sample(candidates, self.args.max_samples_per_session)
            yield from candidates

    def _build_tool_documents(
        self,
        tool_definition: str,
        messages: Sequence[Dict[str, Any]],
        answer: str,
        qid: str,
    ) -> Optional[List[str]]:
        mode = (self.args.tool_document_mode or "full").lower()
        if mode in {"full", "none"}:
            return None
        if mode != "target_hard_negatives":
            raise ValueError(f"Unknown tool_document_mode={self.args.tool_document_mode!r}")
        tools = _as_tool_list(tool_definition)
        target_tool = _extract_tool_name(answer)
        if not tools or not target_tool:
            return []
        target_indices = [index for index, tool in enumerate(tools) if _tool_name(tool) == target_tool]
        if not target_indices:
            return []
        target_index = target_indices[0]
        query = _query_text(messages, self.args.hard_negative_router_scope)
        ranked = _rank_tools(tools, query)
        negative_indices = [index for index in ranked if index != target_index][: self.args.hard_negative_num]
        selected = [tools[target_index]] + [tools[index] for index in negative_indices]
        if self.args.shuffle_tool_documents:
            rng = random.Random(f"{self.args.split_seed}:{qid}:tool_documents")
            rng.shuffle(selected)
        return [_render_tool_definition([tool]) for tool in selected]

    def _session_examples(
        self,
        session_id: str,
        spans: Sequence[Any],
        subset: str = "unknown",
    ) -> List[AgentToolDefinitionExample]:
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
                continue
            answer, has_tool_call = _render_output_messages(output_messages)
            if not answer:
                continue
            if self.args.require_tool_call and not has_tool_call:
                continue
            normalized_messages = [
                item
                for item in (_normal_message(message) for message in _json_loads(input_messages, []))
                if item is not None and item.get("role") != "system"
            ]
            if not normalized_messages:
                continue
            qid = f"{session_id}:{span_index}"
            tool_documents = self._build_tool_documents(
                tool_definition,
                normalized_messages,
                answer,
                qid,
            )
            if tool_documents == []:
                continue
            candidates.append(
                AgentToolDefinitionExample(
                    qid=qid,
                    session_id=session_id,
                    tool_definition=tool_definition,
                    input_messages=normalized_messages,
                    answer=answer,
                    has_tool_call=has_tool_call,
                    system_prompt=self.args.system_prompt,
                    tool_documents=tool_documents,
                    subset=subset,
                )
            )
        return candidates


def _chat_template_ids(
    tokenizer: Any,
    messages: Sequence[Dict[str, Any]],
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
    ids = list(ids)
    if not keep_bos and ids and ids[0] == tokenizer.bos_token_id:
        ids = ids[1:]
    return ids


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _post_json(base_url: str, path: str, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    response = HTTP.post(
        f"{base_url.rstrip('/')}{path}",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def _chat_completion(
    base_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    max_new_tokens: int,
    timeout: int,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_completion_tokens": max_new_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = _post_json(base_url, "/v1/chat/completions", payload, timeout)
    content = data["choices"][0]["message"].get("content")
    return content if isinstance(content, str) else ""


def _extract_document(
    base_url: str,
    text: str,
    ratio: int,
    timeout: int,
) -> Dict[str, Any]:
    return _post_json(
        base_url,
        "/v1/c2kv/extract",
        {
            "text": text,
            "compression_ratio": ratio,
            "role": "user",
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout,
    )


def _has_tool_call_text(text: str) -> bool:
    return "<tool_call>" in (text or "") or "Action:" in (text or "")


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


def _tool_documents(example: Any, document_mode: str) -> List[str]:
    if document_mode == "loader" and getattr(example, "tool_documents", None):
        return [
            "Tool definition:\n" + document
            for document in example.tool_documents
            if str(document).strip()
        ]
    if document_mode == "per_tool":
        tools = _as_tool_list(example.tool_definition)
        return [
            "Tool definition:\n" + _render_tool_definition([tool])
            for tool in tools
        ]
    if document_mode != "full":
        raise ValueError(f"Unknown tool document eval mode: {document_mode}")
    return ["Tool definitions:\n" + example.tool_definition]


def _doc_token_count(tokenizer: Any, docs: Iterable[str]) -> int:
    total = 0
    for doc in docs:
        total += len(_chat_template_ids(tokenizer, [{"role": "user", "content": doc}]))
    return total


def _build_eval_documents(
    example: Any,
    tokenizer: Any,
    document_mode: str,
    args: argparse.Namespace,
    *,
    skip_prefix: str = "",
) -> Tuple[Optional[List[str]], int, Optional[str]]:
    docs = _tool_documents(example, document_mode)
    return _validate_eval_documents(
        docs,
        tokenizer,
        document_mode,
        args,
        skip_prefix=skip_prefix,
    )


def _validate_eval_documents(
    docs: List[str],
    tokenizer: Any,
    document_mode: str,
    args: argparse.Namespace,
    *,
    skip_prefix: str = "",
) -> Tuple[Optional[List[str]], int, Optional[str]]:
    if not docs:
        return None, 0, skip_prefix + "empty_tool_definition"

    token_lengths = [
        len(_chat_template_ids(tokenizer, [{"role": "user", "content": doc}]))
        for doc in docs
    ]
    doc_tokens = sum(token_lengths)
    if doc_tokens > args.max_tool_definition_tokens:
        return (
            None,
            doc_tokens,
            skip_prefix + f"tool_definition_tokens>{args.max_tool_definition_tokens}",
        )

    if document_mode == "full":
        max_context_tokens = args.max_doc_length * args.max_doc_num
        if doc_tokens > max_context_tokens and not args.truncate_tool_definition:
            return (
                None,
                doc_tokens,
                skip_prefix + f"tool_definition_tokens>{max_context_tokens}",
            )
    elif document_mode == "loader":
        too_long = next(
            (length for length in token_lengths if length > args.max_doc_length),
            None,
        )
        if too_long is not None and not args.truncate_tool_definition:
            return (
                None,
                doc_tokens,
                skip_prefix + f"tool_document_tokens>{args.max_doc_length}",
            )
        if len(docs) > args.max_doc_num:
            if not args.truncate_tool_definition:
                return (
                    None,
                    doc_tokens,
                    skip_prefix + f"tool_definition_docs>{args.max_doc_num}",
                )
            docs = docs[: args.max_doc_num]
            doc_tokens = sum(token_lengths[: args.max_doc_num])
    elif document_mode == "per_tool":
        too_long = next(
            (length for length in token_lengths if length > args.max_doc_length),
            None,
        )
        if too_long is not None and not args.truncate_tool_definition:
            return (
                None,
                doc_tokens,
                skip_prefix + f"tool_document_tokens>{args.max_doc_length}",
            )
    else:
        raise ValueError(f"Unknown tool document eval mode: {document_mode}")

    return docs, doc_tokens, None


def _prompt_token_count(tokenizer: Any, messages: List[Dict[str, Any]]) -> int:
    return len(_chat_template_ids(tokenizer, messages, add_generation_prompt=True))


def _split_hybrid_docs(
    example: Any,
    top_k: int,
    router_scope: str,
    document_mode: str,
) -> Tuple[List[str], List[str], List[str], int]:
    query = _query_text(example.input_messages, router_scope)
    if document_mode == "loader" and getattr(example, "tool_documents", None):
        doc_tools = []
        for document in example.tool_documents:
            tools = _as_tool_list(document)
            if tools:
                doc_tools.append(tools[0])
        if doc_tools:
            top_tools, rest_tools, top_tool_names = _split_topk_tools(
                doc_tools,
                query,
                top_k,
            )
            top_docs = [
                "Top-k tool definitions:\n" + _render_tool_definition([tool])
                for tool in top_tools
            ]
            rest_docs = [
                "Tool definition:\n" + _render_tool_definition([tool])
                for tool in rest_tools
            ]
            return top_docs, rest_docs, top_tool_names, len(doc_tools)
        return [], _tool_documents(example, document_mode), [], 0

    tools = _as_tool_list(example.tool_definition)
    if not tools:
        return [], _tool_documents(example, document_mode), [], 0
    top_tools, rest_tools, top_tool_names = _split_topk_tools(tools, query, top_k)
    top_docs = ["Top-k tool definitions:\n" + _render_tool_definition(top_tools)]
    if document_mode == "per_tool":
        rest_docs = [
            "Tool definition:\n" + _render_tool_definition([tool])
            for tool in rest_tools
        ]
    elif document_mode == "full":
        rest_docs = [
            "Rest tool definitions:\n" + _render_tool_definition(rest_tools)
        ] if rest_tools else []
    else:
        raise ValueError(f"Unknown tool document eval mode: {document_mode}")
    return top_docs, rest_docs, top_tool_names, len(tools)


def _extract_docs_as_messages(
    base_url: str,
    docs: List[str],
    ratio: int,
    timeout: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, int]:
    messages: List[Dict[str, Any]] = []
    records: List[Dict[str, Any]] = []
    original_tokens = 0
    gist_tokens = 0
    for doc_idx, doc in enumerate(docs):
        try:
            result = _extract_document(base_url, doc, ratio, timeout)
            success = bool(result.get("success") and result.get("key_hash"))
        except Exception as exc:
            result = {"success": False, "error": str(exc)}
            success = False
        record = {
            "doc_idx": doc_idx,
            "success": success,
            "key_hash": result.get("key_hash"),
            "gist_len": result.get("gist_len"),
            "original_seq_len": result.get("original_seq_len"),
            "error": result.get("error"),
        }
        records.append(record)
        if isinstance(record["original_seq_len"], int):
            original_tokens += record["original_seq_len"]
        if isinstance(record["gist_len"], int):
            gist_tokens += record["gist_len"]
        if success:
            messages.append(
                {
                    "role": "user",
                    "content": doc,
                    "c2kv_key_hash": result["key_hash"],
                }
            )
        else:
            warnings.warn(f"extract failed for doc {doc_idx}: {record}")
            messages.append({"role": "user", "content": doc})
    return messages, records, original_tokens, gist_tokens


def evaluate_one(
    example: Any,
    tokenizer: Any,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    total_start = time.perf_counter()
    target = example.answer.strip()
    target_tool = _extract_tool_name(target)
    doc_messages: List[Dict[str, Any]]
    extract_records: List[Dict[str, Any]] = []
    top_tool_names: List[str] = []
    num_tools = len(_as_tool_list(example.tool_definition))
    doc_tokens = 0
    gist_tokens = 0
    full_doc_tokens = 0
    top_doc_tokens = 0

    if args.mode == "full":
        docs = _tool_documents(example, "full")
        full_doc_tokens = _doc_token_count(tokenizer, docs)
        doc_tokens = full_doc_tokens
        doc_messages = [{"role": "user", "content": doc} for doc in docs]
    elif args.mode == "c2kv":
        docs, full_doc_tokens, skip_reason = _build_eval_documents(
            example,
            tokenizer,
            args.tool_document_eval_mode,
            args,
        )
        if docs is None:
            return {
                "qid": example.qid,
                "session_id": example.session_id,
                "subset": getattr(example, "subset", "unknown"),
                "mode": args.mode,
                "ratio": args.ratio,
                "skipped": True,
                "skip_reason": skip_reason,
                "doc_tokens": full_doc_tokens,
            }
        doc_messages, extract_records, doc_tokens, gist_tokens = _extract_docs_as_messages(
            args.base_url,
            docs,
            args.ratio,
            args.timeout,
        )
    elif args.mode == "hybrid":
        top_docs, rest_docs, top_tool_names, num_tools = _split_hybrid_docs(
            example,
            args.hybrid_top_k,
            args.router_scope,
            args.tool_document_eval_mode,
        )
        full_docs = top_docs + rest_docs
        full_doc_tokens = _doc_token_count(tokenizer, full_docs)
        top_doc_tokens = _doc_token_count(tokenizer, top_docs)
        if rest_docs:
            limited_rest_docs, rest_limit_tokens, skip_reason = _validate_eval_documents(
                rest_docs,
                tokenizer,
                args.tool_document_eval_mode,
                args,
                skip_prefix="rest_",
            )
            if limited_rest_docs is None:
                return {
                    "qid": example.qid,
                    "session_id": example.session_id,
                    "subset": getattr(example, "subset", "unknown"),
                    "mode": args.mode,
                    "ratio": args.ratio,
                    "skipped": True,
                    "skip_reason": skip_reason,
                    "doc_tokens": full_doc_tokens,
                    "rest_doc_tokens": rest_limit_tokens,
                }
            rest_docs = limited_rest_docs
        rest_messages, extract_records, rest_original, gist_tokens = _extract_docs_as_messages(
            args.base_url,
            rest_docs,
            args.ratio,
            args.timeout,
        )
        doc_tokens = top_doc_tokens + rest_original
        doc_messages = [{"role": "user", "content": doc} for doc in top_docs]
        doc_messages.extend(rest_messages)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    messages = [{"role": "system", "content": example.system_prompt}]
    messages.extend(doc_messages)
    messages.extend(example.input_messages)
    prompt_tokens = _prompt_token_count(tokenizer, example.input_messages)

    chat_start = time.perf_counter()
    try:
        prediction = _chat_completion(
            args.base_url,
            args.model,
            messages,
            args.max_new_tokens,
            args.timeout,
        )
        chat_error = None
    except Exception as exc:
        warnings.warn(f"[{example.qid}] chat error: {exc}")
        prediction = ""
        chat_error = str(exc)
    chat_sec = time.perf_counter() - chat_start

    pred_tool = _extract_tool_name(prediction)
    generated_tokens = len(tokenizer.encode(prediction, add_special_tokens=False))
    if args.mode == "full":
        compressed_tool_tokens = doc_tokens
    else:
        compressed_tool_tokens = sum(
            int(item.get("gist_len") or item.get("original_seq_len") or 0)
            for item in extract_records
        )
    if args.mode == "hybrid":
        compressed_tool_tokens += top_doc_tokens
    actual_ratio = (
        full_doc_tokens / compressed_tool_tokens
        if compressed_tool_tokens > 0 else 0.0
    )
    row = {
        "qid": example.qid,
        "session_id": example.session_id,
        "subset": getattr(example, "subset", "unknown"),
        "mode": args.mode,
        "ratio": args.ratio,
        "skipped": False,
        "doc_tokens": doc_tokens,
        "full_doc_tokens": full_doc_tokens,
        "gist_tokens": gist_tokens,
        "compressed_tool_tokens": compressed_tool_tokens,
        "actual_compression_ratio": round(actual_ratio, 4),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "num_tools": num_tools,
        "num_tool_documents": len(doc_messages),
        "num_c2kv_documents": sum(1 for msg in doc_messages if "c2kv_key_hash" in msg),
        "top_k": args.hybrid_top_k if args.mode == "hybrid" else None,
        "top_tool_names": top_tool_names if args.mode == "hybrid" else None,
        "target_tool_name": target_tool,
        "prediction_tool_name": pred_tool,
        "tool_name_match": target_tool is not None and target_tool == pred_tool,
        "target_has_tool_call": bool(target_tool) or _has_tool_call_text(target),
        "has_tool_call": _has_tool_call_text(prediction),
        "response_type_match": (
            (bool(target_tool) or _has_tool_call_text(target))
            == _has_tool_call_text(prediction)
        ),
        "exact_match": _normalize_text(prediction) == _normalize_text(target),
        "text_token_f1": round(_text_token_f1(target, prediction), 4),
        "rouge_l_f1": round(_rouge_l_f1(target, prediction), 4),
        "prediction": prediction,
        "target": target,
        "extracts": extract_records,
        "timing": {
            "chat_seconds": round(chat_sec, 4),
            "total_seconds": round(time.perf_counter() - total_start, 4),
        },
    }
    if chat_error is not None:
        row["chat_error"] = chat_error
    return row


def _select_examples(args: argparse.Namespace, tokenizer: Any) -> Tuple[List[Any], Dict[str, int]]:
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
        tool_document_mode=args.dataset_tool_document_mode,
        hard_negative_num=args.hard_negative_num,
        hard_negative_router_scope=args.hard_negative_router_scope,
        shuffle_tool_documents=args.shuffle_tool_documents,
        balance_subsets=args.balance_subsets,
        max_samples_per_subset=args.max_samples_per_subset,
    )
    source = AgentLLMTracesSource(data_args)
    source_examples = list(source.iter_examples(args.split))
    if args.max_source_examples is not None:
        source_examples = source_examples[: args.max_source_examples]

    selection_skips: Counter[str] = Counter()
    selected = []
    for example in source_examples:
        docs, _, skip_reason = _build_eval_documents(
            example,
            tokenizer,
            args.tool_document_eval_mode,
            args,
        )
        if docs is None:
            selection_skips[str(skip_reason)] += 1
            continue
        if args.min_num_tools > 0 and len(docs) < args.min_num_tools:
            selection_skips[f"num_tools<{args.min_num_tools}"] += 1
            continue
        selected.append(example)
        if args.max_examples is not None and args.max_examples > 0 and len(selected) >= args.max_examples:
            break
    return selected, dict(selection_skips)


def summarize(args: argparse.Namespace, rows: List[Dict[str, Any]], selection_skips: Dict[str, int]) -> Dict[str, Any]:
    valid_rows = [row for row in rows if not row.get("skipped")]
    total_extracts = [item for row in rows for item in row.get("extracts", [])]
    success_extracts = [item for item in total_extracts if item.get("success")]
    prediction_counts = Counter(row.get("prediction", "") for row in valid_rows)

    def avg(key: str) -> float:
        return (
            sum(float(row.get(key, 0.0) or 0.0) for row in valid_rows) / len(valid_rows)
            if valid_rows else 0.0
        )

    return {
        "base_url": args.base_url,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "dataset_path": args.dataset_path,
        "split": args.split,
        "mode": args.mode,
        "ratio": args.ratio,
        "dataset_tool_document_mode": args.dataset_tool_document_mode,
        "tool_document_eval_mode": args.tool_document_eval_mode,
        "hard_negative_num": args.hard_negative_num,
        "num_rows": len(rows),
        "num_valid": len(valid_rows),
        "selection_skips": selection_skips,
        "tool_name_match": avg("tool_name_match"),
        "exact_match": avg("exact_match"),
        "response_type_match": avg("response_type_match"),
        "text_token_f1": avg("text_token_f1"),
        "rouge_l_f1": avg("rouge_l_f1"),
        "avg_actual_compression_ratio": avg("actual_compression_ratio"),
        "num_extracts": len(total_extracts),
        "extract_success_rate": (
            len(success_extracts) / len(total_extracts) if total_extracts else None
        ),
        "avg_chat_seconds": (
            sum(row.get("timing", {}).get("chat_seconds", 0.0) for row in valid_rows)
            / len(valid_rows)
            if valid_rows else 0.0
        ),
        "avg_total_seconds": (
            sum(row.get("timing", {}).get("total_seconds", 0.0) for row in valid_rows)
            / len(valid_rows)
            if valid_rows else 0.0
        ),
        "top_predictions": prediction_counts.most_common(20),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate agent tool-definition modes through SGLang HTTP API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="qwen3-agent-tooldef")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--dataset-path", default="./datasets/agent-llm-traces")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--split", choices=["train", "eval"], default="eval")
    parser.add_argument("--mode", choices=["full", "c2kv", "hybrid"], required=True)
    parser.add_argument("--ratio", type=int, default=4)
    parser.add_argument("--hybrid-top-k", type=int, default=3)
    parser.add_argument("--router-scope", choices=["last_user", "all"], default="last_user")
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--max-source-examples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-manifest-file")
    parser.add_argument("--split-manifest-name", default="subset_disjoint")
    parser.add_argument("--max-samples-per-session", type=int, default=4)
    parser.add_argument("--max-doc-length", type=int, default=2048)
    parser.add_argument("--max-doc-num", type=int, default=16)
    parser.add_argument("--max-tool-definition-tokens", type=int, default=131072)
    parser.add_argument("--min-num-tools", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-system-length", type=int, default=256)
    parser.add_argument("--min-target-tokens", type=int, default=128)
    parser.add_argument("--truncate-tool-definition", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--require-tool-call", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument(
        "--dataset-tool-document-mode",
        "--tool-document-mode",
        dest="dataset_tool_document_mode",
        default="full",
        help="Tool document construction used while reading the source dataset.",
    )
    parser.add_argument(
        "--tool-document-eval-mode",
        choices=["full", "per_tool", "loader"],
        default="per_tool",
        help=(
            "Tool document layout used by this API evaluator. "
            "per_tool matches the previous ToolDoc per-tool eval; loader uses "
            "tool_documents materialized by the dataset loader."
        ),
    )
    parser.add_argument("--hard-negative-num", type=int, default=15)
    parser.add_argument("--hard-negative-router-scope", choices=["last_user", "all"], default="last_user")
    parser.add_argument("--shuffle-tool-documents", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--balance-subsets", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--max-samples-per-subset", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        local_files_only=True,
        padding_side="right",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    examples, selection_skips = _select_examples(args, tokenizer)
    rows = [
        evaluate_one(example, tokenizer, args)
        for example in tqdm(examples, desc=f"{args.mode}@{args.ratio}x")
    ]
    _write_jsonl(args.output_file, rows)
    summary = summarize(args, rows, selection_skips)
    summary_file = str(Path(args.output_file).with_suffix(".summary.json"))
    Path(summary_file).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
