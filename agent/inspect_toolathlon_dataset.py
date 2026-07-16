from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _json_loads(value: Any, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default if default is not None else value


def _iter_samples(data_dir: Path, limit: Optional[int] = None) -> Iterable[tuple[Path, Dict[str, Any]]]:
    seen = 0
    for path in sorted(data_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                yield path, json.loads(line)
                seen += 1
                if limit is not None and seen >= limit:
                    return


def _tool_definition(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    tool_calls = _json_loads(sample.get("tool_calls"), {}) or {}
    if isinstance(tool_calls, dict) and isinstance(tool_calls.get("tools"), list):
        return tool_calls["tools"]
    if isinstance(tool_calls, list):
        return tool_calls
    return []


def _messages(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    messages = _json_loads(sample.get("messages"), []) or []
    return messages if isinstance(messages, list) else []


def _config(sample: Dict[str, Any]) -> Dict[str, Any]:
    config = _json_loads(sample.get("config"), {}) or {}
    return config if isinstance(config, dict) else {}


def _status(sample: Dict[str, Any]) -> Dict[str, Any]:
    status = _json_loads(sample.get("task_status"), {}) or {}
    return status if isinstance(status, dict) else {}


def _actual_tool_calls(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    calls = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if isinstance(call, dict):
                calls.append(call)
    return calls


def _observations(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [m for m in messages if isinstance(m, dict) and m.get("role") == "tool"]


def _token_counter(tokenizer_path: Optional[str]):
    if not tokenizer_path:
        return None
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return count


def _count_text_tokens(counter, value: Any) -> int:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if counter is None:
        # Fallback rough estimate used only when no tokenizer is available.
        return max(1, len(text) // 4)
    return counter(text)


def _summary(values: List[int | float]) -> Dict[str, float]:
    if not values:
        return {"min": 0, "avg": 0, "p50": 0, "p95": 0, "max": 0}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "avg": round(sum(ordered) / len(ordered), 4),
        "p50": ordered[len(ordered) // 2],
        "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "max": ordered[-1],
    }


def inspect(args: argparse.Namespace) -> Dict[str, Any]:
    data_dir = Path(args.data_dir)
    counter = _token_counter(args.tokenizer)
    token_note = "tokenizer" if counter is not None else "rough_char_div_4"

    file_counts = Counter()
    task_counts = Counter()
    task_models: Dict[str, set[str]] = defaultdict(set)
    task_runs: Dict[str, set[str]] = defaultdict(set)
    status_counts = Counter()
    eval_counts = Counter()
    running_counts = Counter()
    server_task_counts = Counter()

    message_counts = []
    actual_call_counts = []
    available_tool_counts = []
    tool_def_tokens = []
    trajectory_tokens = []
    observation_tokens = []
    success_lengths = defaultdict(list)
    failure_lengths = defaultdict(list)

    for path, sample in _iter_samples(data_dir, args.limit):
        file_counts[path.name] += 1
        task_name = sample.get("task_name") or "unknown"
        task_counts[task_name] += 1
        model_run = sample.get("modelname_run") or path.stem
        task_runs[task_name].add(model_run)
        task_models[task_name].add(str(model_run).rsplit("_", 1)[0])

        config = _config(sample)
        status = _status(sample)
        messages = _messages(sample)
        tools = _tool_definition(sample)
        actual_calls = _actual_tool_calls(messages)
        observations = _observations(messages)

        status_counts[json.dumps(status, ensure_ascii=False, sort_keys=True)] += 1
        eval_value = status.get("evaluation")
        eval_counts[eval_value] += 1
        running_counts[status.get("running")] += 1
        for server in config.get("needed_mcp_servers") or []:
            server_task_counts[server] += 1

        message_counts.append(len(messages))
        actual_call_counts.append(len(actual_calls))
        available_tool_counts.append(len(tools))
        tool_tokens = _count_text_tokens(counter, tools)
        traj_tokens = _count_text_tokens(counter, messages)
        obs_tokens = _count_text_tokens(counter, observations)
        tool_def_tokens.append(tool_tokens)
        trajectory_tokens.append(traj_tokens)
        observation_tokens.append(obs_tokens)

        bucket = success_lengths if eval_value is True else failure_lengths
        bucket["messages"].append(len(messages))
        bucket["actual_tool_calls"].append(len(actual_calls))
        bucket["tool_def_tokens"].append(tool_tokens)
        bucket["trajectory_tokens"].append(traj_tokens)
        bucket["observation_tokens"].append(obs_tokens)

    per_task_model_run = {
        task: {
            "num_records": count,
            "num_models": len(task_models[task]),
            "num_model_runs": len(task_runs[task]),
        }
        for task, count in sorted(task_counts.items())
    }

    return {
        "data_dir": str(data_dir),
        "num_jsonl_files": len(file_counts),
        "num_trajectories": sum(file_counts.values()),
        "num_tasks": len(task_counts),
        "token_count_mode": token_note,
        "success_failure_counts": {
            "evaluation_true": eval_counts.get(True, 0),
            "evaluation_false": eval_counts.get(False, 0),
            "evaluation_none": eval_counts.get(None, 0),
        },
        "running_counts": dict(running_counts),
        "messages_per_trajectory": _summary(message_counts),
        "actual_tool_calls_per_trajectory": _summary(actual_call_counts),
        "available_tools_per_trajectory": _summary(available_tool_counts),
        "tool_definition_tokens": _summary(tool_def_tokens),
        "full_trajectory_tokens": _summary(trajectory_tokens),
        "observation_tokens": _summary(observation_tokens),
        "success_lengths": {key: _summary(values) for key, values in success_lengths.items()},
        "failure_or_unsuccessful_lengths": {key: _summary(values) for key, values in failure_lengths.items()},
        "mcp_server_record_counts": dict(server_task_counts.most_common()),
        "per_task_model_run_summary": {
            "num_models": _summary([item["num_models"] for item in per_task_model_run.values()]),
            "num_model_runs": _summary([item["num_model_runs"] for item in per_task_model_run.values()]),
            "num_records": _summary([item["num_records"] for item in per_task_model_run.values()]),
        },
        "per_task_model_run": per_task_model_run if args.include_tasks else None,
        "top_statuses": dict(status_counts.most_common(10)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Toolathlon trajectory dataset format and statistics.")
    parser.add_argument("--data_dir", default="./datasets/toolathlon")
    parser.add_argument("--tokenizer", help="Optional tokenizer path for exact token counts.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--include_tasks", action="store_true")
    parser.add_argument("--output_file")
    args = parser.parse_args()

    result = inspect(args)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
