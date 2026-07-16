from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow.parquet as pq


def _stats(values: List[int]) -> Dict[str, Any]:
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


def _load_tokenizer(path: Optional[str]):
    if not path:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)


def _count_tokens(tokenizer: Any, text: str) -> int:
    if tokenizer is None:
        return max(1, len(text or "") // 4)
    return len(tokenizer.encode(text or "", add_special_tokens=False))


def _count_tools(tools_text: str) -> int:
    try:
        value = json.loads(tools_text or "[]")
    except Exception:
        return 0
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict) and isinstance(value.get("tools"), list):
        return len(value["tools"])
    return 1 if isinstance(value, dict) else 0


def _count_tool_calls(conversations: Any) -> int:
    if not isinstance(conversations, list):
        return 0
    text = "\n".join(str(item.get("value", "")) for item in conversations if isinstance(item, dict))
    return len(re.findall(r"<tool_call>", text))


def inspect(data_dir: Path, tokenizer_path: Optional[str]) -> Dict[str, Any]:
    tokenizer = _load_tokenizer(tokenizer_path)
    files = [
        data_dir / "data" / "kimi" / "train.parquet",
        data_dir / "data" / "glm-5.1" / "train.parquet",
    ]
    result = {
        "data_dir": str(data_dir),
        "token_count_mode": "tokenizer" if tokenizer else "rough_char_div_4",
        "configs": {},
    }
    all_tool_tokens: List[int] = []
    all_tools_per_sample: List[int] = []
    all_turns: List[int] = []
    all_tool_calls: List[int] = []
    for file in files:
        if not file.exists():
            continue
        pf = pq.ParquetFile(file)
        tool_tokens: List[int] = []
        tools_per_sample: List[int] = []
        turns: List[int] = []
        tool_calls: List[int] = []
        categories: Dict[str, int] = {}
        subcategories: Dict[str, int] = {}
        rows = 0
        for batch in pf.iter_batches(columns=["tools", "conversations", "category", "subcategory"], batch_size=256):
            data = batch.to_pylist()
            for row in data:
                rows += 1
                tools = row.get("tools") or ""
                conv = row.get("conversations") or []
                tool_tokens.append(_count_tokens(tokenizer, tools))
                tools_per_sample.append(_count_tools(tools))
                turns.append(len(conv) if isinstance(conv, list) else 0)
                tool_calls.append(_count_tool_calls(conv))
                categories[str(row.get("category") or "")] = categories.get(str(row.get("category") or ""), 0) + 1
                subcategories[str(row.get("subcategory") or "")] = subcategories.get(str(row.get("subcategory") or ""), 0) + 1
        config_name = file.parent.name
        result["configs"][config_name] = {
            "file": str(file),
            "rows": rows,
            "tool_definition_tokens": _stats(tool_tokens),
            "tools_per_sample": _stats(tools_per_sample),
            "turns_per_sample": _stats(turns),
            "tool_calls_per_sample": _stats(tool_calls),
            "categories": dict(sorted(categories.items(), key=lambda item: (-item[1], item[0]))),
            "top_subcategories": dict(sorted(subcategories.items(), key=lambda item: (-item[1], item[0]))[:20]),
        }
        all_tool_tokens.extend(tool_tokens)
        all_tools_per_sample.extend(tools_per_sample)
        all_turns.extend(turns)
        all_tool_calls.extend(tool_calls)
    result["combined"] = {
        "rows": sum(item["rows"] for item in result["configs"].values()),
        "tool_definition_tokens": _stats(all_tool_tokens),
        "tools_per_sample": _stats(all_tools_per_sample),
        "turns_per_sample": _stats(all_turns),
        "tool_calls_per_sample": _stats(all_tool_calls),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Hermes agent reasoning traces.")
    parser.add_argument("--data_dir", default="./datasets/hermes-agent-reasoning-traces")
    parser.add_argument("--tokenizer")
    parser.add_argument("--output_file", default="./outputs/hermes_agent_reasoning_stats.json")
    args = parser.parse_args()
    result = inspect(Path(args.data_dir), args.tokenizer)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
