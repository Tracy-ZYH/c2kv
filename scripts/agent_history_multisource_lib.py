
#!/usr/bin/env python
from __future__ import annotations

import ast
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
BENCHMARK_BLOCKLIST = {"tau2_airline", "tau2_retail", "tau2_telecom"}
EVAL_BENCHMARK_NAMES = {"bfcl", "berkeley_function_calling_leaderboard", "toolsandbox", "tool_sandbox"}

Message = Dict[str, Any]


def json_loads(value: Any, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        try:
            return ast.literal_eval(value)
        except Exception:
            return default


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def parquet_files(path: str | Path) -> List[Path]:
    path = Path(path)
    if path.is_file() and path.suffix == ".parquet":
        return [path]
    return sorted(path.rglob("*.parquet"))


def read_parquet_rows(path: str | Path, columns: Optional[Sequence[str]] = None) -> Iterator[Dict[str, Any]]:
    import pyarrow.parquet as pq

    for file in parquet_files(path):
        table = pq.read_table(file, columns=list(columns) if columns else None)
        for row in table.to_pylist():
            row["__file__"] = str(file)
            yield row


def parquet_file_summary(path: str | Path) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq

    rows = []
    for file in parquet_files(path):
        pf = pq.ParquetFile(file)
        rows.append({"path": str(file), "rows": pf.metadata.num_rows, "schema": str(pf.schema_arrow)})
    return rows


def tool_list(value: Any) -> List[Dict[str, Any]]:
    parsed = json_loads(value, value)
    if isinstance(parsed, dict):
        if isinstance(parsed.get("tools"), list):
            parsed = parsed["tools"]
        elif isinstance(parsed.get("functions"), list):
            parsed = parsed["functions"]
        else:
            parsed = [parsed]
    if not isinstance(parsed, list):
        return []
    out = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        if "function" in item or item.get("type") == "function":
            out.append(item)
        else:
            out.append({"type": "function", "function": item})
    return out


def strip_reasoning_text(text: str) -> tuple[str, bool]:
    if not text:
        return text, False
    new = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    changed = new != text
    new = re.sub(r"(?is)^\s*(thought|reasoning)\s*:\s*.*?(?=\n\s*(action|<tool_call>|final|assistant|answer)\b|$)", "", new).strip()
    return new, changed or new != text


def _parts_text(parts: Any, skip_tool_calls: bool = True) -> str:
    parts = json_loads(parts, parts)
    if isinstance(parts, dict):
        parts = [parts]
    if not isinstance(parts, list):
        return ""
    texts = []
    for part in parts:
        if isinstance(part, dict):
            if skip_tool_calls and part.get("type") in {"tool_call", "function_call"}:
                continue
            if part.get("content") is not None:
                texts.append(str(part.get("content")))
            elif part.get("text") is not None:
                texts.append(str(part.get("text")))
            else:
                texts.append(json_dumps(part))
        elif part is not None:
            texts.append(str(part))
    return "\n".join(t for t in texts if t)


def render_tool_call(call: Any) -> str:
    call = json_loads(call, call)
    if not isinstance(call, dict):
        return ""
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    name = function.get("name") or call.get("name") or call.get("tool_name") or call.get("function_name") or ""
    arguments = function.get("arguments") or call.get("arguments") or call.get("args") or call.get("input") or {}
    payload = {"name": name, "arguments": arguments}
    return "<tool_call>\n" + json_dumps(payload) + "\n</tool_call>"


def render_tool_calls(value: Any) -> str:
    value = json_loads(value, value)
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return ""
    rendered = [render_tool_call(item) for item in value]
    return "\n".join(item for item in rendered if item)


def normalize_message(message: Message, *, strip_reasoning: bool = False) -> tuple[Optional[Message], bool]:
    if not isinstance(message, dict):
        return None, False
    raw_role = message.get("role") or message.get("from") or message.get("type") or "user"
    role_map = {
        "human": "user",
        "gpt": "assistant",
        "tool_response": "tool_response",
        "tool_result": "tool_response",
        "tool": "tool_response",
        "function": "tool_response",
        "tool_call": "tool_call",
        "function_call": "tool_call",
    }
    role = role_map.get(str(raw_role), str(raw_role))
    content = message.get("content")
    if content is None:
        content = message.get("value")
    if content is None:
        content = _parts_text(message.get("parts"))
    if not isinstance(content, str):
        content = json_dumps(content) if content is not None else ""
    reasoning_stripped = False
    if strip_reasoning and role == "assistant":
        content, reasoning_stripped = strip_reasoning_text(content)
    tool_calls = render_tool_calls(
        message.get("tool_calls") or message.get("toolCalls") or message.get("function_call") or message.get("parts")
    )
    if role == "tool_call":
        content = render_tool_call(content) or content
        role = "assistant"
    elif role == "tool_response":
        content = "[Tool result]\n" + content.strip()
        role = "user"
    elif tool_calls:
        content = (content.strip() + "\n\nAction:\n" + tool_calls).strip()
    if not content and role != "assistant":
        return None, reasoning_stripped
    if role not in {"system", "user", "assistant"}:
        content = f"[{role}]\n{content}".strip()
        role = "user"
    return {"role": role, "content": content.strip()}, reasoning_stripped


def normalize_messages(messages: Any, *, strip_reasoning: bool = False) -> tuple[List[Message], int]:
    messages = json_loads(messages, messages)
    if isinstance(messages, dict):
        messages = [messages]
    out: List[Message] = []
    stripped = 0
    for message in messages or []:
        item, did_strip = normalize_message(message, strip_reasoning=strip_reasoning)
        stripped += int(did_strip)
        if item is not None:
            out.append(item)
    return out, stripped


def strip_tools_xml(system: str) -> str:
    return re.sub(r"(?is)<tools>.*?</tools>", "", system or "").strip() or DEFAULT_SYSTEM_PROMPT


def split_turns(messages: Sequence[Message]) -> List[Dict[str, Any]]:
    turns: List[Dict[str, Any]] = []
    current: List[Message] = []
    turn_id = 0
    for msg in messages:
        if msg.get("role") == "system":
            continue
        if msg.get("role") == "user" and current and any(m.get("role") == "assistant" for m in current):
            turns.append({"turn_id": turn_id, "messages": current})
            turn_id += 1
            current = [msg]
        else:
            current.append(msg)
    if current:
        turns.append({"turn_id": turn_id, "messages": current})
    return turns


def render_history_doc(turn: Dict[str, Any]) -> Dict[str, Any]:
    parts = [f"Completed user turn {turn.get('turn_id', 0)}"]
    for msg in turn.get("messages", []):
        role = msg.get("role", "user")
        label = "User" if role == "user" else "Assistant"
        parts.append(f"[{label}]\n{msg.get('content', '').strip()}")
    return {"turn_id": turn.get("turn_id", 0), "chunk_id": 0, "content": "\n\n".join(parts).strip()}


def target_from_message(message: Message) -> str:
    content = str(message.get("content") or "").strip()
    return content


def has_tool_call_text(text: str) -> bool:
    low = (text or "").lower()
    return "<tool_call>" in low or "action:" in low or "function_call" in low or "tool_call" in low


def extract_tool_call(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    patterns = [r"<tool_call>\s*(\{.*?\})\s*</tool_call>", r"Action:\s*(?:<tool_call>)?\s*(\{.*?\})(?:\s*</tool_call>)?"]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if not m:
            continue
        payload = json_loads(m.group(1), None)
        if not isinstance(payload, dict):
            continue
        function = payload.get("function") if isinstance(payload.get("function"), dict) else {}
        name = function.get("name") or payload.get("name") or payload.get("tool_name") or payload.get("function_name")
        args = function.get("arguments") or payload.get("arguments") or payload.get("args") or payload.get("input") or {}
        if isinstance(args, str):
            args = json_loads(args, args)
        return {"name": str(name or ""), "arguments": args}
    return None


@dataclass
class UnifiedTrajectory:
    source: str
    task_id: str
    trajectory_id: str
    group_id: str
    system: List[Message]
    tools: List[Dict[str, Any]]
    turns: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


def build_decision_points(
    traj: UnifiedTrajectory,
    *,
    min_completed_history_turns: int = 2,
    max_history_docs: int = 16,
    require_previous_history: bool = True,
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    completed: List[Dict[str, Any]] = []
    for turn in traj.turns:
        history_docs = [render_history_doc(item) for item in completed]
        if len(history_docs) >= min_completed_history_turns or (history_docs and not require_previous_history):
            messages = list(turn.get("messages", []))
            for idx, msg in enumerate(messages):
                if msg.get("role") != "assistant":
                    continue
                target = target_from_message(msg)
                if not target:
                    continue
                current = messages[:idx]
                if not current:
                    continue
                selected_history = history_docs[-max_history_docs:]
                sample_id = f"{traj.source}:{traj.trajectory_id}:{turn.get('turn_id', 0)}:{idx}"
                samples.append({
                    "sample_id": sample_id,
                    "source": traj.source,
                    "task_id": traj.task_id,
                    "trajectory_id": traj.trajectory_id,
                    "group_id": traj.group_id,
                    "system": traj.system,
                    "tools": traj.tools,
                    "history_docs": selected_history,
                    "current_messages": current,
                    "target": target,
                    "metadata": {
                        **traj.metadata,
                        "turn_id": turn.get("turn_id", 0),
                        "target_message_index": idx,
                        "target_has_tool_call": has_tool_call_text(target),
                        "history_turn_count": len(history_docs),
                    },
                })
        if any(m.get("role") == "assistant" for m in turn.get("messages", [])):
            completed.append(turn)
    return samples


def agent_contamination_reason(row: Dict[str, Any]) -> Optional[str]:
    benchmark = str(row.get("benchmark") or "").lower()
    subset = str(row.get("benchmark_subset") or "").lower()
    config_path = str(row.get("config_path") or "").lower()
    if benchmark in BENCHMARK_BLOCKLIST or benchmark.startswith("tau2") or subset in {"airline", "retail", "telecom"} and "tau2" in benchmark:
        return "tau2"
    meta = " ".join([benchmark, subset, config_path])
    if any(name in meta for name in EVAL_BENCHMARK_NAMES):
        return "heldout_benchmark"
    return None


def iter_agent_llm(path: str | Path, stats: Counter, *, strip_reasoning: bool = False) -> Iterator[UnifiedTrajectory]:
    columns = ["session_id", "run_id", "harness", "benchmark", "benchmark_subset", "config_path", "spans", "success", "status"]
    for row_index, row in enumerate(read_parquet_rows(path, columns=columns)):
        stats["agent_raw_trajectories"] += 1
        reason = agent_contamination_reason(row)
        if reason:
            stats[f"agent_filtered_{reason}"] += 1
            continue
        spans = row.get("spans") or []
        spans = sorted([sp for sp in spans if isinstance(sp, dict)], key=lambda sp: (sp.get("start_time") or "", sp.get("span_id") or ""))
        span_turns: List[Dict[str, Any]] = []
        tools: List[Dict[str, Any]] = []
        system_prompt = ""
        for span_i, span in enumerate(spans):
            attrs = span.get("attributes") or {}
            raw_in = attrs.get("gen_ai.input.messages")
            raw_out = attrs.get("gen_ai.output.messages")
            if raw_in is None or raw_out is None:
                continue
            input_messages, stripped_in = normalize_messages(raw_in, strip_reasoning=strip_reasoning)
            output_messages, stripped_out = normalize_messages(raw_out, strip_reasoning=strip_reasoning)
            stats["agent_reasoning_stripped_messages"] += stripped_in + stripped_out
            if not input_messages or not output_messages:
                continue
            if not tools:
                tools = tool_list(attrs.get("gen_ai.tool.definitions"))
            if not system_prompt:
                system_prompt = next((m.get("content", "") for m in input_messages if m.get("role") == "system"), "")
            non_system = [m for m in input_messages if m.get("role") != "system"]
            last_user = next((i for i in range(len(non_system) - 1, -1, -1) if non_system[i].get("role") == "user"), None)
            if last_user is None:
                continue
            history_turns = split_turns(non_system[:last_user])
            current_prefix = non_system[last_user:]
            synthetic_turn = {"turn_id": len(history_turns), "messages": current_prefix + output_messages}
            turns = history_turns + [synthetic_turn]
            traj = UnifiedTrajectory(
                source="agent_llm",
                task_id=str(row.get("session_id") or row.get("run_id") or row_index),
                trajectory_id=str(row.get("session_id") or f"agent-row-{row_index}"),
                group_id="agent_llm:" + str(row.get("session_id") or f"agent-row-{row_index}"),
                system=[{"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT}],
                tools=tools,
                turns=turns,
                metadata={
                    "benchmark": row.get("benchmark"),
                    "benchmark_subset": row.get("benchmark_subset"),
                    "harness": row.get("harness"),
                    "status": row.get("status"),
                    "source_file": row.get("__file__"),
                },
            )
            yield traj
            stats["agent_kept_span_trajectories"] += 1


def iter_hermes(path: str | Path, stats: Counter, *, strip_reasoning: bool = True) -> Iterator[UnifiedTrajectory]:
    seen: set[str] = set()
    for row_index, row in enumerate(read_parquet_rows(path)):
        stats["hermes_raw_rows"] += 1
        row_id = str(row.get("id") or f"hermes-row-{row_index}")
        if row_id in seen:
            stats["hermes_duplicate_ids"] += 1
            continue
        seen.add(row_id)
        tools = tool_list(row.get("tools"))
        messages, stripped = normalize_messages(row.get("conversations") or [], strip_reasoning=strip_reasoning)
        stats["hermes_reasoning_stripped_messages"] += stripped
        if not tools or not messages:
            stats["hermes_skipped_unusable"] += 1
            continue
        text = "\n".join(m.get("content", "") for m in messages)
        if not has_tool_call_text(text) or "[Tool result]" not in text:
            stats["hermes_skipped_no_tool_flow"] += 1
            continue
        system_prompt = next((m.get("content", "") for m in messages if m.get("role") == "system"), DEFAULT_SYSTEM_PROMPT)
        turns = split_turns([m for m in messages if m.get("role") != "system"])
        yield UnifiedTrajectory(
            source="hermes",
            task_id=str(row.get("task") or row_id),
            trajectory_id=row_id,
            group_id="hermes:" + row_id,
            system=[{"role": "system", "content": strip_tools_xml(system_prompt)}],
            tools=tools,
            turns=turns,
            metadata={"category": row.get("category"), "subcategory": row.get("subcategory"), "task": row.get("task"), "source_file": row.get("__file__")},
        )
        stats["hermes_kept_trajectories"] += 1


def iter_toucan(path: str | Path, stats: Counter, *, allow_irrelevant: bool = True) -> Iterator[UnifiedTrajectory]:
    for row_index, row in enumerate(read_parquet_rows(path)):
        stats["toucan_raw_rows"] += 1
        subset = str(row.get("subset_name") or "")
        if subset == "multi-turn":
            stats["toucan_multi_turn_rows"] += 1
        elif subset == "irrelevant":
            stats["toucan_irrelevant_rows"] += 1
            if not allow_irrelevant:
                stats["toucan_skipped_irrelevant"] += 1
                continue
        else:
            stats[f"toucan_skipped_{subset or 'unknown'}"] += 1
            continue
        tools = tool_list(row.get("tools"))
        raw_messages = json_loads(row.get("messages"), [])
        messages, _ = normalize_messages(raw_messages, strip_reasoning=False)
        if not tools or not messages:
            stats["toucan_skipped_unusable"] += 1
            continue
        turns = split_turns(messages)
        row_id = str(row.get("uuid") or f"toucan-row-{row_index}")
        yield UnifiedTrajectory(
            source="toucan",
            task_id=row_id,
            trajectory_id=row_id,
            group_id="toucan:" + row_id,
            system=[{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}],
            tools=tools,
            turns=turns,
            metadata={"subset_name": subset, "target_tools": row.get("target_tools"), "question": row.get("question"), "source_file": row.get("__file__")},
        )
        stats["toucan_kept_trajectories"] += 1


def iter_all_sources(agent_path: str, hermes_path: str, toucan_path: str, stats: Counter, *, strip_reasoning: bool = True) -> Iterator[UnifiedTrajectory]:
    yield from iter_agent_llm(agent_path, stats, strip_reasoning=False)
    yield from iter_toucan(toucan_path, stats, allow_irrelevant=True)
    yield from iter_hermes(hermes_path, stats, strip_reasoning=strip_reasoning)


def percentile(values: Sequence[int | float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * p))))
    return float(values[idx])


def summarize_values(values: Sequence[int | float]) -> Dict[str, float]:
    return {"p50": percentile(values, 0.50), "p90": percentile(values, 0.90), "p95": percentile(values, 0.95), "max": float(max(values) if values else 0)}


def group_split(samples: Sequence[Dict[str, Any]], val_ratio: float = 0.05, seed: int = 42) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_group[str(sample["group_id"])].append(sample)
    groups = sorted(by_group)
    rng = random.Random(seed)
    rng.shuffle(groups)
    val_group_count = max(1, round(len(groups) * val_ratio)) if groups else 0
    val_groups = set(groups[:val_group_count])
    train_groups = set(groups[val_group_count:])
    assert train_groups.isdisjoint(val_groups)
    train = [sample for g in groups if g in train_groups for sample in by_group[g]]
    val = [sample for g in groups if g in val_groups for sample in by_group[g]]
    return train, val, {"train_groups": len(train_groups), "val_groups": len(val_groups), "overlap": 0}


def source_quotas(max_samples: int, mix: Dict[str, float]) -> Dict[str, int]:
    quotas = {source: int(max_samples * prob) for source, prob in mix.items()}
    while sum(quotas.values()) < max_samples:
        source = max(mix, key=mix.get)
        quotas[source] += 1
    return quotas


def select_mixed_samples(pools: Dict[str, List[Dict[str, Any]]], max_samples: int, seed: int = 42) -> List[Dict[str, Any]]:
    mix = {"agent_llm": 0.35, "toucan": 0.35, "hermes": 0.30}
    quotas = source_quotas(max_samples, mix)
    rng = random.Random(seed)
    selected: List[Dict[str, Any]] = []
    for source, quota in quotas.items():
        items = list(pools.get(source, []))
        rng.shuffle(items)
        if source == "toucan":
            multi = [s for s in items if s.get("metadata", {}).get("subset_name") == "multi-turn"]
            irr = [s for s in items if s.get("metadata", {}).get("subset_name") == "irrelevant"]
            irr_quota = min(len(irr), int(round(quota * 0.10)))
            source_items = multi[: max(0, quota - irr_quota)] + irr[:irr_quota]
            if len(source_items) < quota:
                source_items += [s for s in items if s not in source_items][: quota - len(source_items)]
        else:
            source_items = items[:quota]
        selected.extend(source_items[:quota])
    rng.shuffle(selected)
    return selected[:max_samples]


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count
