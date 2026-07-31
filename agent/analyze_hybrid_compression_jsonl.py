from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _display_mode(row: Dict[str, Any]) -> str:
    mode = str(row.get("mode"))
    hybrid_mode = row.get("hybrid_mode")
    router_strategy = row.get("router_strategy")
    top_k = row.get("top_k")
    top_schema_mode = row.get("top_schema_mode", "full")
    if mode == "hybrid" and hybrid_mode:
        if hybrid_mode == "hybrid" and router_strategy == "attention":
            attention_score_mode = row.get("attention_score_mode")
            attention_cache_mode = row.get("attention_cache_mode", "c2kv")
            prefix = "att_fullkv_hybrid" if attention_cache_mode == "full" else "att_hybrid"
            if attention_score_mode:
                return f"{prefix}_{attention_score_mode}"
            return prefix
        if hybrid_mode == "hybrid" and router_strategy == "lex_attention":
            attention_score_mode = row.get("attention_score_mode")
            attention_cache_mode = row.get("attention_cache_mode", "c2kv")
            prefix = "lex_att_fullkv_hybrid" if attention_cache_mode == "full" else "lex_att_hybrid"
            if attention_score_mode:
                return f"{prefix}_{attention_score_mode}"
            return prefix
        if hybrid_mode == "hybrid" and router_strategy == "att_rerank":
            attention_score_mode = row.get("attention_score_mode")
            attention_cache_mode = row.get("attention_cache_mode", "c2kv")
            prefix = "hybrid_fullkv_att_rerank" if attention_cache_mode == "full" else "hybrid_att_rerank"
            if attention_score_mode:
                return f"{prefix}_{attention_score_mode}"
            return prefix
        if hybrid_mode == "hybrid" and router_strategy == "random":
            return "random_hybrid"
        if hybrid_mode == "hybrid":
            if top_schema_mode == "compact":
                return "c2kv_hybrid_compact"
            if top_k is not None and top_k != 3:
                return f"c2kv_hybrid_top{top_k}"
            return "c2kv_hybrid"
        return str(hybrid_mode)
    return mode


def _num(row: Dict[str, Any], key: str) -> float:
    value = row.get(key, 0)
    return float(value or 0)


def _compressed_tool_tokens(row: Dict[str, Any]) -> float:
    if "top_doc_tokens" in row or "rest_gist_tokens" in row:
        return _num(row, "top_doc_tokens") + _num(row, "rest_gist_tokens")
    if "gist_tokens" in row:
        return _num(row, "gist_tokens")
    if "kept_tool_tokens" in row:
        return _num(row, "kept_tool_tokens")
    ratio = _num(row, "actual_compression_ratio")
    doc_tokens = _num(row, "doc_tokens")
    return doc_tokens / ratio if ratio > 0 else 0.0


def _quantiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    last = len(ordered) - 1

    def pick(p: float) -> float:
        return ordered[int(round(last * p))]

    return {
        "min": ordered[0],
        "p10": pick(0.10),
        "p25": pick(0.25),
        "p50": pick(0.50),
        "p75": pick(0.75),
        "p90": pick(0.90),
        "max": ordered[-1],
    }


def _bucket_num_tools(num_tools: int) -> str:
    if num_tools <= 3:
        return "<=3"
    if num_tools <= 5:
        return "4-5"
    if num_tools <= 10:
        return "6-10"
    if num_tools <= 20:
        return "11-20"
    if num_tools <= 50:
        return "21-50"
    return ">50"


def _summarize(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [row for row in rows if not row.get("skipped")]
    skipped = [row for row in rows if row.get("skipped")]
    compressed_total = sum(_compressed_tool_tokens(row) for row in valid)
    doc_total = sum(_num(row, "doc_tokens") for row in valid)
    ratios = [_num(row, "actual_compression_ratio") for row in valid if _num(row, "actual_compression_ratio")]
    top_shares = [
        _num(row, "top_doc_tokens") / _num(row, "doc_tokens")
        for row in valid
        if _num(row, "doc_tokens") and "top_doc_tokens" in row
    ]
    hybrid_cache_top_shares = [
        _num(row, "top_doc_tokens") / _compressed_tool_tokens(row)
        for row in valid
        if _compressed_tool_tokens(row) and "top_doc_tokens" in row
    ]
    num_tools_values = [int(row.get("num_tools") or 0) for row in valid if "num_tools" in row]
    no_rest = [row for row in valid if row.get("num_rest_tools") == 0]
    low_ratio = [row for row in valid if _num(row, "actual_compression_ratio") < 3.0]
    return {
        "num_rows": len(valid) + len(skipped),
        "num_valid": len(valid),
        "num_skipped": len(skipped),
        "skip_reasons": dict(Counter(str(row.get("skip_reason", "unknown")) for row in skipped)),
        "token_weighted_actual_compression_ratio": doc_total / compressed_total if compressed_total else 0.0,
        "avg_actual_compression_ratio": mean(ratios) if ratios else 0.0,
        "ratio_quantiles": _quantiles(ratios),
        "avg_doc_tokens": mean([_num(row, "doc_tokens") for row in valid]) if valid else 0.0,
        "avg_top_doc_tokens": mean([_num(row, "top_doc_tokens") for row in valid]) if top_shares else 0.0,
        "avg_rest_doc_tokens": mean([_num(row, "rest_doc_tokens") for row in valid]) if top_shares else 0.0,
        "avg_rest_gist_tokens": mean([_num(row, "rest_gist_tokens") for row in valid]) if top_shares else 0.0,
        "avg_top_share_of_full_doc": mean(top_shares) if top_shares else 0.0,
        "top_share_of_full_doc_quantiles": _quantiles(top_shares),
        "avg_top_share_of_hybrid_cache": mean(hybrid_cache_top_shares) if hybrid_cache_top_shares else 0.0,
        "avg_num_tools": mean(num_tools_values) if num_tools_values else 0.0,
        "num_tools_buckets": dict(Counter(_bucket_num_tools(value) for value in num_tools_values)),
        "num_no_rest_tools": len(no_rest),
        "no_rest_tool_rate": len(no_rest) / len(valid) if valid else 0.0,
        "num_ratio_lt_3": len(low_ratio),
        "ratio_lt_3_rate": len(low_ratio) / len(valid) if valid else 0.0,
    }


def _interesting_rows(rows: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    valid = [row for row in rows if not row.get("skipped") and _num(row, "actual_compression_ratio") > 0]
    valid.sort(key=lambda row: _num(row, "actual_compression_ratio"))
    output = []
    for row in valid[:limit]:
        compressed = _compressed_tool_tokens(row)
        doc_tokens = _num(row, "doc_tokens")
        output.append({
            "qid": row.get("qid"),
            "session_id": row.get("session_id"),
            "mode": _display_mode(row),
            "ratio": row.get("ratio"),
            "num_tools": row.get("num_tools"),
            "num_top_tools": row.get("num_top_tools"),
            "num_rest_tools": row.get("num_rest_tools"),
            "doc_tokens": row.get("doc_tokens"),
            "top_doc_tokens": row.get("top_doc_tokens"),
            "rest_doc_tokens": row.get("rest_doc_tokens"),
            "rest_gist_tokens": row.get("rest_gist_tokens"),
            "compressed_tool_tokens": compressed,
            "actual_compression_ratio": row.get("actual_compression_ratio"),
            "top_share_of_full_doc": doc_tokens and _num(row, "top_doc_tokens") / doc_tokens,
            "top_tool_names": row.get("top_tool_names"),
            "top_tool_token_lengths": row.get("top_tool_token_lengths"),
            "target_tool_name": row.get("target_tool_name"),
            "target_lexical_rank": row.get("target_lexical_rank"),
            "target_attention_rank": row.get("target_attention_rank"),
            "target_final_rank": row.get("target_final_rank"),
            "att_rerank_replaced": row.get("att_rerank_replaced"),
            "att_rerank_debug": row.get("att_rerank_debug"),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze hybrid compression token accounting in an eval JSONL.")
    parser.add_argument("jsonl_file")
    parser.add_argument(
        "--mode",
        default="c2kv_hybrid",
        help="Display mode to analyze, comma-separated modes, or all.",
    )
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    rows = _read_jsonl(Path(args.jsonl_file))
    all_modes = dict(Counter(_display_mode(row) for row in rows))
    if args.mode == "all":
        modes = sorted(all_modes)
    else:
        modes = [item.strip() for item in args.mode.split(",") if item.strip()]
    per_mode = {}
    for mode in modes:
        selected = [row for row in rows if _display_mode(row) == mode]
        per_mode[mode] = {
            "summary": _summarize(selected),
            "lowest_ratio_rows": _interesting_rows(selected, args.limit),
        }
    payload = {
        "file": args.jsonl_file,
        "modes": modes,
        "all_modes": all_modes,
        "per_mode": per_mode,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
