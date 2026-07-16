from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries = []
    keys = sorted({(row.get("mode"), row.get("ratio")) for row in rows})
    for mode, ratio in keys:
        group = [row for row in rows if row.get("mode") == mode and row.get("ratio") == ratio]
        valid_rows = [row for row in group if not row.get("skipped")]
        skip_reasons = Counter(row.get("skip_reason", "unknown") for row in group if row.get("skipped"))
        summaries.append({
            "mode": mode,
            "ratio": ratio,
            "num_examples": len(group),
            "num_valid": len(valid_rows),
            "num_skipped": len(group) - len(valid_rows),
            "skip_reasons": dict(skip_reasons),
            "exact_match": (
                sum(1 for row in valid_rows if row.get("exact_match")) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "tool_name_accuracy": (
                sum(1 for row in valid_rows if row.get("tool_name_match")) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "tool_call_rate": (
                sum(1 for row in valid_rows if row.get("has_tool_call")) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_latency_sec": (
                sum(float(row.get("latency_sec", 0.0)) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_system_prefill_sec": (
                sum(float(row.get("system_prefill_sec", 0.0)) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_tool_compress_sec": (
                sum(float(row.get("tool_compress_sec", 0.0)) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_full_prefill_sec": (
                sum(float(row.get("full_prefill_sec", 0.0)) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_blend_sec": (
                sum(float(row.get("blend_sec", 0.0)) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_generate_sec": (
                sum(float(row.get("generate_sec", 0.0)) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_generated_tokens": (
                sum(int(row.get("generated_tokens", 0)) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_ttft_sec": (
                sum(float(row.get("ttft_sec", 0.0)) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_tbt_sec": (
                sum(float(row.get("tbt_sec", 0.0)) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "token_weighted_tbt_sec": (
                sum(float(row.get("generate_sec", 0.0)) for row in valid_rows)
                / sum(int(row.get("generated_tokens", 0)) for row in valid_rows)
                if sum(int(row.get("generated_tokens", 0)) for row in valid_rows) else 0.0
            ),
            "avg_total_sec": (
                sum(float(row.get("total_sec", 0.0)) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_actual_compression_ratio": (
                sum(float(row.get("actual_compression_ratio", 0.0)) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
        })
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge parallel agent C2KV eval shards.")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--input_files", nargs="+", required=True)
    parser.add_argument("--model")
    parser.add_argument("--base_model")
    parser.add_argument("--dataset_path")
    parser.add_argument("--split", default="eval")
    args = parser.parse_args()

    rows = []
    for input_file in args.input_files:
        path = Path(input_file)
        if path.exists():
            rows.extend(_read_jsonl(path))

    output_path = Path(args.output_file)
    _write_jsonl(output_path, rows)
    summary = {
        "model": args.model,
        "base_model": args.base_model,
        "dataset_path": args.dataset_path,
        "split": args.split,
        "num_rows": len(rows),
        "results": _summarize(rows),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
