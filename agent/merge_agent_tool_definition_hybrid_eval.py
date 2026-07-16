from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


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


def _stat(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "avg": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))
    return {
        "min": float(ordered[0]),
        "avg": sum(ordered) / len(ordered),
        "p50": float(ordered[len(ordered) // 2]),
        "p95": float(ordered[p95_index]),
        "max": float(ordered[-1]),
    }


def _sample_key(row: Dict[str, Any]) -> Optional[str]:
    qid = row.get("qid")
    if qid is None:
        return None
    session_id = row.get("session_id")
    return f"{session_id}\t{qid}" if session_id is not None else str(qid)


def _metrics(valid_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_generated = sum(int(row.get("generated_tokens", 0)) for row in valid_rows)
    called = sum(1 for row in valid_rows if row.get("has_tool_call"))
    return {
        "num_valid": len(valid_rows),
        "router_hit_rate": (
            sum(1 for row in valid_rows if row.get("router_hit")) / len(valid_rows)
            if valid_rows else 0.0
        ),
        "random_expected_hit_rate": (
            sum(
                min(
                    1.0,
                    float(row.get("top_k") or 0) / max(1.0, float(row.get("num_tools", 0))),
                )
                for row in valid_rows
            )
            / len(valid_rows)
            if valid_rows else 0.0
        ),
        "exact_match": (
            sum(1 for row in valid_rows if row.get("exact_match")) / len(valid_rows)
            if valid_rows else 0.0
        ),
        "tool_name_accuracy": (
            sum(1 for row in valid_rows if row.get("tool_name_match")) / len(valid_rows)
            if valid_rows else 0.0
        ),
        "tool_call_rate": called / len(valid_rows) if valid_rows else 0.0,
        "call_accuracy": (
            sum(1 for row in valid_rows if row.get("tool_name_match")) / called
            if called else 0.0
        ),
        "avg_num_tools": (
            sum(row.get("num_tools", 0) for row in valid_rows) / len(valid_rows)
            if valid_rows else 0.0
        ),
        "num_tools_stats": _stat([float(row.get("num_tools", 0)) for row in valid_rows]),
        "avg_top_doc_tokens": (
            sum(row.get("top_doc_tokens", 0) for row in valid_rows) / len(valid_rows)
            if valid_rows else 0.0
        ),
        "avg_rest_doc_tokens": (
            sum(row.get("rest_doc_tokens", 0) for row in valid_rows) / len(valid_rows)
            if valid_rows else 0.0
        ),
        "avg_actual_compression_ratio": (
            sum(float(row.get("actual_compression_ratio", 0.0)) for row in valid_rows) / len(valid_rows)
            if valid_rows else 0.0
        ),
        "avg_online_ttft_sec": (
            sum(float(row.get("online_ttft_sec", 0.0)) for row in valid_rows) / len(valid_rows)
            if valid_rows else 0.0
        ),
        "avg_cached_ttft_sec": (
            sum(float(row.get("cached_ttft_sec", 0.0)) for row in valid_rows) / len(valid_rows)
            if valid_rows else 0.0
        ),
        "avg_tool_only_cached_ttft_sec": (
            sum(float(row.get("tool_only_cached_ttft_sec", 0.0)) for row in valid_rows) / len(valid_rows)
            if valid_rows else 0.0
        ),
        "avg_generate_sec": (
            sum(float(row.get("generate_sec", 0.0)) for row in valid_rows) / len(valid_rows)
            if valid_rows else 0.0
        ),
        "avg_generated_tokens": total_generated / len(valid_rows) if valid_rows else 0.0,
        "avg_tbt_sec": (
            sum(float(row.get("tbt_sec", 0.0)) for row in valid_rows) / len(valid_rows)
            if valid_rows else 0.0
        ),
        "token_weighted_tbt_sec": (
            sum(float(row.get("generate_sec", 0.0)) for row in valid_rows) / total_generated
            if total_generated else 0.0
        ),
        "avg_online_total_sec": (
            sum(float(row.get("total_sec", 0.0)) for row in valid_rows) / len(valid_rows)
            if valid_rows else 0.0
        ),
        "avg_cached_total_sec": (
            sum(float(row.get("cached_total_sec", 0.0)) for row in valid_rows) / len(valid_rows)
            if valid_rows else 0.0
        ),
    }


def _summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries = []
    keys = sorted({
        (
            row.get("hybrid_mode", "hybrid"),
            row.get("router_strategy", "lexical"),
            row.get("top_k"),
            row.get("ratio"),
        )
        for row in rows
    })
    for hybrid_mode, router_strategy, top_k, ratio in keys:
        group = [
            row for row in rows
            if row.get("hybrid_mode", "hybrid") == hybrid_mode
            and row.get("router_strategy", "lexical") == router_strategy
            and row.get("top_k") == top_k
            and row.get("ratio") == ratio
        ]
        valid_rows = [row for row in group if not row.get("skipped")]
        skip_reasons = Counter(row.get("skip_reason", "unknown") for row in group if row.get("skipped"))
        summaries.append({
            "mode": "hybrid",
            "hybrid_mode": hybrid_mode,
            "router_strategy": router_strategy,
            "top_k": top_k,
            "ratio": ratio,
            "num_examples": len(group),
            "num_skipped": len(group) - len(valid_rows),
            "skip_reasons": dict(skip_reasons),
            **_metrics(valid_rows),
        })
    return summaries


def _summarize_by_hit(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries = []
    keys = sorted({
        (
            row.get("hybrid_mode", "hybrid"),
            row.get("router_strategy", "lexical"),
            row.get("top_k"),
            row.get("ratio"),
            bool(row.get("router_hit")),
        )
        for row in rows
        if not row.get("skipped")
    })
    for hybrid_mode, router_strategy, top_k, ratio, router_hit in keys:
        group = [
            row for row in rows
            if not row.get("skipped")
            and row.get("hybrid_mode", "hybrid") == hybrid_mode
            and row.get("router_strategy", "lexical") == router_strategy
            and row.get("top_k") == top_k
            and row.get("ratio") == ratio
            and bool(row.get("router_hit")) == router_hit
        ]
        total_generated = sum(int(row.get("generated_tokens", 0)) for row in group)
        summaries.append({
            "mode": "hybrid",
            "hybrid_mode": hybrid_mode,
            "router_strategy": router_strategy,
            "top_k": top_k,
            "ratio": ratio,
            "router_hit": router_hit,
            "num_valid": len(group),
            "tool_name_accuracy": (
                sum(1 for row in group if row.get("tool_name_match")) / len(group)
                if group else 0.0
            ),
            "tool_call_rate": (
                sum(1 for row in group if row.get("has_tool_call")) / len(group)
                if group else 0.0
            ),
            "call_accuracy": (
                sum(1 for row in group if row.get("tool_name_match"))
                / sum(1 for row in group if row.get("has_tool_call"))
                if any(row.get("has_tool_call") for row in group) else 0.0
            ),
            "exact_match": (
                sum(1 for row in group if row.get("exact_match")) / len(group)
                if group else 0.0
            ),
            "avg_cached_total_sec": (
                sum(float(row.get("cached_total_sec", 0.0)) for row in group) / len(group)
                if group else 0.0
            ),
            "token_weighted_tbt_sec": (
                sum(float(row.get("generate_sec", 0.0)) for row in group) / total_generated
                if total_generated else 0.0
            ),
        })
    return summaries


def _valid_rows_by_strategy(
    rows: List[Dict[str, Any]],
    hybrid_mode: Any,
    top_k: Any,
    ratio: Any,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        if (
            row.get("skipped")
            or row.get("hybrid_mode", "hybrid") != hybrid_mode
            or row.get("top_k") != top_k
            or row.get("ratio") != ratio
        ):
            continue
        sample_key = _sample_key(row)
        if sample_key is None:
            continue
        strategy = row.get("router_strategy", "lexical")
        out.setdefault(strategy, {})
        if sample_key not in out[strategy]:
            out[strategy][sample_key] = row
    return out


def _common_sample_keys(by_strategy: Dict[str, Dict[str, Dict[str, Any]]]) -> List[str]:
    if not by_strategy:
        return []
    return sorted(set.intersection(*(set(rows) for rows in by_strategy.values())))


def _summarize_common_strategy(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries = []
    cases = sorted({(row.get("hybrid_mode", "hybrid"), row.get("top_k"), row.get("ratio")) for row in rows})
    for hybrid_mode, top_k, ratio in cases:
        by_strategy = _valid_rows_by_strategy(rows, hybrid_mode, top_k, ratio)
        common_keys = _common_sample_keys(by_strategy)
        if len(by_strategy) < 2 or not common_keys:
            continue
        for strategy, keyed_rows in sorted(by_strategy.items()):
            valid_rows = [keyed_rows[key] for key in common_keys]
            summaries.append({
                "mode": "hybrid",
                "hybrid_mode": hybrid_mode,
                "comparison": "common_strategy",
                "router_strategy": strategy,
                "top_k": top_k,
                "ratio": ratio,
                "num_common_samples": len(common_keys),
                **_metrics(valid_rows),
            })
    return summaries


def _summarize_common_hit_breakdown(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries = []
    cases = sorted({(row.get("hybrid_mode", "hybrid"), row.get("top_k"), row.get("ratio")) for row in rows})
    for hybrid_mode, top_k, ratio in cases:
        by_strategy = _valid_rows_by_strategy(rows, hybrid_mode, top_k, ratio)
        common_keys = _common_sample_keys(by_strategy)
        if len(by_strategy) < 2 or not common_keys:
            continue
        for strategy, keyed_rows in sorted(by_strategy.items()):
            common_rows = [keyed_rows[key] for key in common_keys]
            for router_hit in (True, False):
                bucket = [row for row in common_rows if bool(row.get("router_hit")) == router_hit]
                summaries.append({
                    "mode": "hybrid",
                    "hybrid_mode": hybrid_mode,
                    "comparison": "common_hit_breakdown",
                    "router_strategy": strategy,
                    "top_k": top_k,
                    "ratio": ratio,
                    "router_hit": router_hit,
                    "num_common_base_samples": len(common_keys),
                    **_metrics(bucket),
                })
    return summaries


def _paired_hit_outcomes(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries = []
    cases = sorted({(row.get("hybrid_mode", "hybrid"), row.get("top_k"), row.get("ratio")) for row in rows})
    for hybrid_mode, top_k, ratio in cases:
        by_strategy = _valid_rows_by_strategy(rows, hybrid_mode, top_k, ratio)
        if "lexical" not in by_strategy or "random" not in by_strategy:
            continue
        common_keys = sorted(set(by_strategy["lexical"]) & set(by_strategy["random"]))
        if not common_keys:
            continue
        buckets: Dict[str, List[str]] = {
            "both_hit": [],
            "lexical_hit_random_miss": [],
            "lexical_miss_random_hit": [],
            "both_miss": [],
        }
        for key in common_keys:
            lexical_hit = bool(by_strategy["lexical"][key].get("router_hit"))
            random_hit = bool(by_strategy["random"][key].get("router_hit"))
            if lexical_hit and random_hit:
                bucket = "both_hit"
            elif lexical_hit:
                bucket = "lexical_hit_random_miss"
            elif random_hit:
                bucket = "lexical_miss_random_hit"
            else:
                bucket = "both_miss"
            buckets[bucket].append(key)
        for bucket, keys in buckets.items():
            lexical_rows = [by_strategy["lexical"][key] for key in keys]
            random_rows = [by_strategy["random"][key] for key in keys]
            summaries.append({
                "mode": "hybrid",
                "hybrid_mode": hybrid_mode,
                "comparison": "paired_hit_outcome",
                "top_k": top_k,
                "ratio": ratio,
                "hit_outcome": bucket,
                "num_samples": len(keys),
                "lexical": _metrics(lexical_rows),
                "random": _metrics(random_rows),
            })
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge hybrid top-k full + C2KV eval shards.")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--input_files", nargs="+", required=True)
    parser.add_argument("--model")
    parser.add_argument("--dataset_path")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--router_scope", default="last_user")
    parser.add_argument("--router_strategy")
    parser.add_argument("--hybrid_modes")
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
        "dataset_path": args.dataset_path,
        "split": args.split,
        "router_scope": args.router_scope,
        "router_strategy": args.router_strategy,
        "hybrid_modes": [item.strip() for item in (args.hybrid_modes or "").split(",") if item.strip()],
        "num_rows": len(rows),
        "results": _summarize(rows),
        "hit_breakdown_results": _summarize_by_hit(rows),
        "common_strategy_results": _summarize_common_strategy(rows),
        "common_hit_breakdown_results": _summarize_common_hit_breakdown(rows),
        "paired_hit_outcome_results": _paired_hit_outcomes(rows),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
