from __future__ import annotations

import argparse
import json
from collections import defaultdict
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


def _summarize_group(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid_rows = [row for row in rows if not row.get("skipped")]
    session_ids = sorted({row["session_id"] for row in valid_rows})
    session_accs = []
    session_success = []
    first_error_turns = []
    session_totals = []
    session_builds = []
    session_generates = []

    for session_id in session_ids:
        group = sorted(
            [row for row in valid_rows if row["session_id"] == session_id],
            key=lambda item: item.get("turn_index", 0),
        )
        if not group:
            continue
        correct = [bool(row.get("tool_name_match")) for row in group]
        session_accs.append(sum(correct) / len(correct))
        session_success.append(all(correct))
        for row, ok in zip(group, correct):
            if not ok:
                first_error_turns.append(row.get("turn_index", 0))
                break

        if group[0].get("reuse"):
            session_build = sum(
                row.get("system_prefill_sec", 0.0)
                + row.get("tool_compress_sec", 0.0)
                + row.get("full_prefill_sec", 0.0)
                + row.get("blend_sec", 0.0)
                for row in group
            )
            # Reuse rows store build cost outside per-span total in the per-case script.
            # If merged from rows only, recover by taking the max nonzero build cost.
            if session_build == 0:
                session_build = max((row.get("session_build_sec", 0.0) for row in group), default=0.0)
            session_generate = sum(row.get("generate_sec", 0.0) for row in group)
            session_total = session_build + session_generate
        else:
            session_build = sum(
                row.get("system_prefill_sec", 0.0)
                + row.get("tool_compress_sec", 0.0)
                + row.get("full_prefill_sec", 0.0)
                + row.get("blend_sec", 0.0)
                for row in group
            )
            session_generate = sum(row.get("generate_sec", 0.0) for row in group)
            session_total = sum(row.get("total_sec", 0.0) for row in group)
        session_builds.append(session_build)
        session_generates.append(session_generate)
        session_totals.append(session_total)

    total_spans = len(valid_rows)
    total_session_sec = sum(session_totals)
    total_build_sec = sum(session_builds)
    total_generate_sec = sum(session_generates)
    total_generated_tokens = sum(int(row.get("generated_tokens", 0)) for row in valid_rows)
    return {
        "mode": rows[0].get("mode") if rows else None,
        "ratio": rows[0].get("ratio") if rows else None,
        "reuse": rows[0].get("reuse") if rows else None,
        "num_sessions": len(session_ids),
        "num_spans": len(rows),
        "num_valid_spans": total_spans,
        "exact_match": (
            sum(1 for row in valid_rows if row.get("exact_match")) / total_spans
            if total_spans else 0.0
        ),
        "tool_name_accuracy": (
            sum(1 for row in valid_rows if row.get("tool_name_match")) / total_spans
            if total_spans else 0.0
        ),
        "tool_call_rate": (
            sum(1 for row in valid_rows if row.get("has_tool_call")) / total_spans
            if total_spans else 0.0
        ),
        "call_accuracy": (
            sum(1 for row in valid_rows if row.get("tool_name_match"))
            / sum(1 for row in valid_rows if row.get("has_tool_call"))
            if any(row.get("has_tool_call") for row in valid_rows) else 0.0
        ),
        "session_macro_tool_name_accuracy": (
            sum(session_accs) / len(session_accs) if session_accs else 0.0
        ),
        "session_success_rate": (
            sum(1 for item in session_success if item) / len(session_success)
            if session_success else 0.0
        ),
        "avg_first_error_turn": (
            sum(first_error_turns) / len(first_error_turns) if first_error_turns else None
        ),
        "avg_session_build_sec": (
            sum(session_builds) / len(session_builds) if session_builds else 0.0
        ),
        "avg_session_total_sec": (
            sum(session_totals) / len(session_totals) if session_totals else 0.0
        ),
        "avg_generate_sec_per_span": (
            total_generate_sec / total_spans if total_spans else 0.0
        ),
        "avg_generated_tokens": (
            total_generated_tokens / total_spans if total_spans else 0.0
        ),
        "avg_session_ttft_sec": (
            total_build_sec / len(session_builds) if session_builds else 0.0
        ),
        "amortized_ttft_sec_per_span": (
            total_build_sec / total_spans if total_spans else 0.0
        ),
        "avg_tbt_sec": (
            sum(float(row.get("tbt_sec", 0.0)) for row in valid_rows) / total_spans
            if total_spans else 0.0
        ),
        "token_weighted_tbt_sec": (
            total_generate_sec / total_generated_tokens if total_generated_tokens else 0.0
        ),
        "amortized_total_sec_per_span": (
            total_session_sec / total_spans if total_spans else 0.0
        ),
        "avg_actual_compression_ratio": (
            sum(row.get("actual_compression_ratio", 0.0) for row in valid_rows) / total_spans
            if total_spans else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge parallel session-reuse eval shards.")
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

    groups: Dict[tuple[Any, Any, Any], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row.get("mode"), row.get("ratio"), row.get("reuse"))].append(row)
    summaries = [_summarize_group(group) for _, group in sorted(groups.items())]
    by_key = {
        (item["mode"], item["ratio"], item["reuse"]): item for item in summaries
    }
    for item in summaries:
        if item["reuse"]:
            no_reuse = by_key.get((item["mode"], item["ratio"], False))
            if no_reuse and item["amortized_total_sec_per_span"] > 0:
                item["reuse_speedup"] = (
                    no_reuse["amortized_total_sec_per_span"] / item["amortized_total_sec_per_span"]
                )
            else:
                item["reuse_speedup"] = None

    summary = {
        "model": args.model,
        "base_model": args.base_model,
        "dataset_path": args.dataset_path,
        "split": args.split,
        "num_rows": len(rows),
        "results": summaries,
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
