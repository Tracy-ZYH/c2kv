from __future__ import annotations

import argparse
import hashlib
import json
import statistics
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


def _tool_name(tool: Dict[str, Any]) -> str:
    function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
    return str(
        function.get("name")
        or tool.get("name")
        or tool.get("tool_name")
        or tool.get("function_name")
        or ""
    )


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


def _toolset_hash(tools: List[Dict[str, Any]]) -> str:
    names = sorted(name for name in (_tool_name(tool) for tool in tools) if name)
    text = json.dumps(names, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


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
    columns = ["benchmark", "harness", "session_id", "spans"]
    for data_file in data_files:
        pf = pq.ParquetFile(data_file)
        for batch in pf.iter_batches(batch_size=128, columns=columns):
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


def inspect(args: argparse.Namespace) -> Dict[str, Any]:
    data_files = _find_parquet_files(Path(args.dataset_path))
    if not data_files:
        raise FileNotFoundError(f"No parquet files found under {args.dataset_path}")

    groups: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "sessions": 0,
            "sessions_with_tools": 0,
            "tool_counts": [],
            "unique_tool_names": set(),
            "toolset_hashes": set(),
            "harnesses": set(),
        }
    )
    total_rows = 0
    missing_tool_defs = 0
    for row in _iter_rows(data_files):
        total_rows += 1
        subset = row.get("benchmark") or "unknown"
        group = groups[str(subset)]
        group["sessions"] += 1
        if row.get("harness"):
            group["harnesses"].add(str(row.get("harness")))
        tool_definition = _first_tool_definition(row)
        tools = _as_tool_list(tool_definition)
        if not tools:
            missing_tool_defs += 1
            continue
        tool_names = {name for name in (_tool_name(tool) for tool in tools) if name}
        group["sessions_with_tools"] += 1
        group["tool_counts"].append(len(tools))
        group["unique_tool_names"].update(tool_names)
        group["toolset_hashes"].add(_toolset_hash(tools))

    subsets = []
    for subset, group in sorted(groups.items()):
        subsets.append(
            {
                "subset": subset,
                "sessions": group["sessions"],
                "sessions_with_tools": group["sessions_with_tools"],
                "harnesses": sorted(group["harnesses"]),
                "tool_count_per_session": _stat(group["tool_counts"]),
                "unique_tool_names": len(group["unique_tool_names"]),
                "unique_toolsets": len(group["toolset_hashes"]),
            }
        )
    result = {
        "dataset_path": args.dataset_path,
        "num_parquet_files": len(data_files),
        "total_sessions": total_rows,
        "missing_tool_definition_sessions": missing_tool_defs,
        "subsets": subsets,
    }
    return result


def _print_markdown(result: Dict[str, Any]) -> None:
    print("| subset | sessions | with tools | unique tools | unique toolsets | min | avg | p50 | p95 | max |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in result["subsets"]:
        stat = row["tool_count_per_session"]
        print(
            f"| {row['subset']} | {row['sessions']} | {row['sessions_with_tools']} | "
            f"{row['unique_tool_names']} | {row['unique_toolsets']} | "
            f"{stat['min']} | {stat['avg']} | {stat['p50']} | {stat['p95']} | {stat['max']} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Count callable tools by agent-llm-traces benchmark/subset.")
    parser.add_argument("--dataset_path", default="./datasets/agent-llm-traces")
    parser.add_argument("--output_file", default="./outputs/agent_llm_traces_tools_by_subset.json")
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
