from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pyarrow.parquet as pq


def _json_loads(value: Any, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default


def _as_tool_list(value: Any) -> List[Dict[str, Any]]:
    parsed = _json_loads(value, value)
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


def _iter_rows(data_files: Iterable[Path]) -> Iterable[Dict[str, Any]]:
    wanted = ["benchmark", "session_id", "spans"]
    for data_file in data_files:
        pf = pq.ParquetFile(data_file)
        columns = [column for column in wanted if column in pf.schema_arrow.names]
        for batch in pf.iter_batches(batch_size=256, columns=columns):
            yield from batch.to_pylist()


def _first_tool_definition(row: Dict[str, Any]) -> Optional[Any]:
    for span in row.get("spans") or []:
        if not isinstance(span, dict):
            continue
        attributes = span.get("attributes") if isinstance(span.get("attributes"), dict) else {}
        value = attributes.get("gen_ai.tool.definitions")
        if value:
            return value
    return None


def _tool_signature(tool: Dict[str, Any]) -> str:
    text = json.dumps(tool, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _tool_name(tool: Dict[str, Any]) -> str:
    function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
    return str(
        function.get("name")
        or tool.get("name")
        or tool.get("tool_name")
        or tool.get("function_name")
        or ""
    )


def _render_tool_definition(tools: List[Dict[str, Any]]) -> str:
    return json.dumps(tools, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stat(values: List[int]) -> Dict[str, Any]:
    if not values:
        return {"min": 0, "avg": 0.0, "p50": 0, "p95": 0, "max": 0}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))
    return {
        "min": ordered[0],
        "avg": round(float(statistics.mean(ordered)), 4),
        "p50": ordered[len(ordered) // 2],
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


def _ratio(count: int, total: int) -> float:
    return round(count / total, 6) if total else 0.0


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

    def count_tool(self, tool: Dict[str, Any]) -> int:
        text = "Tool definition:\n" + _render_tool_definition([tool])
        if self.tokenizer is None:
            return max(1, len(text) // 4)
        from train.train_data_multiturn import _chat_template_ids

        return len(_chat_template_ids(self.tokenizer, [{"role": "user", "content": text}]))


def inspect(args: argparse.Namespace) -> Dict[str, Any]:
    data_files = _find_parquet_files(Path(args.dataset_path))
    if not data_files:
        raise FileNotFoundError(f"No parquet files found under {args.dataset_path}")
    counter = TokenCounter(args.tokenizer)
    groups: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "sessions": 0,
            "sessions_with_tools": 0,
            "tool_counts": [],
            "tool_instance_tokens": [],
            "unique_tool_tokens": {},
            "unique_tool_names": set(),
        }
    )

    for row in _iter_rows(data_files):
        subset = str(row.get("benchmark") or "unknown")
        group = groups[subset]
        group["sessions"] += 1
        tools = _as_tool_list(_first_tool_definition(row))
        if not tools:
            continue
        group["sessions_with_tools"] += 1
        group["tool_counts"].append(len(tools))
        for tool in tools:
            tokens = counter.count_tool(tool)
            group["tool_instance_tokens"].append(tokens)
            signature = _tool_signature(tool)
            group["unique_tool_tokens"].setdefault(signature, tokens)
            name = _tool_name(tool)
            if name:
                group["unique_tool_names"].add(name)

    results = []
    thresholds = [64, 128, 256, 512, 1024]
    for subset, group in sorted(groups.items()):
        instance_tokens = group["tool_instance_tokens"]
        unique_tokens = list(group["unique_tool_tokens"].values())
        item = {
            "subset": subset,
            "sessions": group["sessions"],
            "sessions_with_tools": group["sessions_with_tools"],
            "unique_tool_names": len(group["unique_tool_names"]),
            "unique_tool_definitions": len(unique_tokens),
            "tools_per_session": _stat(group["tool_counts"]),
            "single_tool_instance_tokens": _stat(instance_tokens),
            "single_unique_tool_tokens": _stat(unique_tokens),
            "instance_token_threshold_rates": {
                f">={threshold}": _ratio(sum(1 for value in instance_tokens if value >= threshold), len(instance_tokens))
                for threshold in thresholds
            },
            "unique_token_threshold_rates": {
                f">={threshold}": _ratio(sum(1 for value in unique_tokens if value >= threshold), len(unique_tokens))
                for threshold in thresholds
            },
        }
        results.append(item)
    return {
        "dataset_path": args.dataset_path,
        "token_count_mode": counter.mode,
        "num_parquet_files": len(data_files),
        "results": results,
    }


def _print_markdown(result: Dict[str, Any]) -> None:
    print("| subset | sessions w/tools | tools/session avg | inst avg | inst p50 | inst p95 | inst max | unique avg | unique p50 | unique p95 | >=256 inst |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in result["results"]:
        inst = row["single_tool_instance_tokens"]
        uniq = row["single_unique_tool_tokens"]
        print(
            f"| {row['subset']} | {row['sessions_with_tools']} | {row['tools_per_session']['avg']} | "
            f"{inst['avg']} | {inst['p50']} | {inst['p95']} | {inst['max']} | "
            f"{uniq['avg']} | {uniq['p50']} | {uniq['p95']} | "
            f"{row['instance_token_threshold_rates']['>=256']:.4f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect single-tool document token lengths for agent-llm-traces.")
    parser.add_argument("--dataset_path", default="./datasets/agent-llm-traces")
    parser.add_argument("--tokenizer")
    parser.add_argument("--output_file", default="./outputs/agent_llm_traces_single_tool_token_stats.json")
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
