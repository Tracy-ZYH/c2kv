from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
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


def _summarize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries = []
    keys = sorted({(row.get("mode"), row.get("ratio")) for row in rows})
    for mode, ratio in keys:
        group = [row for row in rows if row.get("mode") == mode and row.get("ratio") == ratio]
        valid_rows = [row for row in group if not row.get("skipped")]
        skip_reasons = Counter(row.get("skip_reason", "unknown") for row in group if row.get("skipped"))
        generated_total = sum(row.get("generated_tokens", 0) for row in valid_rows)
        called = sum(1 for row in valid_rows if row.get("has_tool_call"))
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
                called / len(valid_rows) if valid_rows else 0.0
            ),
            "call_accuracy": (
                sum(1 for row in valid_rows if row.get("tool_name_match")) / called
                if called else 0.0
            ),
            "avg_doc_tokens": (
                sum(row.get("doc_tokens", 0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_prompt_tokens": (
                sum(row.get("prompt_tokens", 0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_generated_tokens": (
                generated_total / len(valid_rows) if valid_rows else 0.0
            ),
            "avg_actual_compression_ratio": (
                sum(row.get("actual_compression_ratio", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_system_prefill_sec": (
                sum(row.get("system_prefill_sec", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_tool_compress_sec": (
                sum(row.get("tool_compress_sec", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_full_prefill_sec": (
                sum(row.get("full_prefill_sec", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_blend_sec": (
                sum(row.get("blend_sec", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_ttft_sec": (
                sum(row.get("ttft_sec", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_generate_sec": (
                sum(row.get("generate_sec", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_tbt_sec": (
                sum(row.get("tbt_sec", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "token_weighted_tbt_sec": (
                sum(row.get("generate_sec", 0.0) for row in valid_rows) / generated_total
                if generated_total else 0.0
            ),
            "avg_total_sec": (
                sum(row.get("total_sec", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
        })
    return summaries


def _group_key(row: Dict[str, Any]) -> tuple[Any, Any]:
    return row.get("mode"), row.get("ratio")


def _sample_key(row: Dict[str, Any]) -> Optional[str]:
    qid = row.get("qid")
    if qid is None:
        return None
    session_id = row.get("session_id")
    return f"{session_id}\t{qid}" if session_id is not None else str(qid)


def _common_subset(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = sorted({_group_key(row) for row in rows})
    valid_samples_by_key: Dict[str, set[str]] = {}
    duplicate_valid_rows_by_key: Dict[str, int] = {}
    for mode, ratio in keys:
        group = [row for row in rows if _group_key(row) == (mode, ratio)]
        valid_sample_counts = Counter(
            key
            for row in group
            if not row.get("skipped")
            for key in [_sample_key(row)]
            if key is not None
        )
        sample_keys = {
            key for key, count in valid_sample_counts.items() if count > 0
        }
        group_name = f"{mode}@{ratio}"
        valid_samples_by_key[group_name] = sample_keys
        duplicate_valid_rows_by_key[group_name] = sum(
            count - 1 for count in valid_sample_counts.values() if count > 1
        )
    if not valid_samples_by_key:
        common_samples: set[str] = set()
    else:
        common_samples = set.intersection(*valid_samples_by_key.values())
    return {
        "num_groups": len(valid_samples_by_key),
        "num_common_samples": len(common_samples),
        "valid_samples_by_group": {
            key: len(value) for key, value in valid_samples_by_key.items()
        },
        "duplicate_valid_rows_by_group": duplicate_valid_rows_by_key,
        "common_sample_keys": sorted(common_samples),
    }


def _dedupe_common_rows(rows: List[Dict[str, Any]], common_samples: set[str]) -> List[Dict[str, Any]]:
    selected: Dict[tuple[Any, Any, str], Dict[str, Any]] = {}
    for row in rows:
        if row.get("skipped"):
            continue
        sample_key = _sample_key(row)
        if sample_key not in common_samples:
            continue
        key = (*_group_key(row), sample_key)
        if key not in selected:
            selected[key] = row
    return list(selected.values())


def _summarize_common_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    common = _common_subset(rows)
    common_samples = set(common["common_sample_keys"])
    if not common_samples:
        return []
    common_rows = _dedupe_common_rows(rows, common_samples)
    return _summarize_rows(common_rows)


def _common_fairness_check(common_results: List[Dict[str, Any]], common_subset: Dict[str, Any]) -> Dict[str, Any]:
    expected = common_subset.get("num_common_samples", 0)
    counts = {
        f"{item.get('mode')}@{item.get('ratio')}": item.get("num_valid", 0)
        for item in common_results
    }
    return {
        "expected_num_valid_per_group": expected,
        "num_valid_by_group": counts,
        "is_fair_common_subset": bool(counts) and all(value == expected for value in counts.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge agent tool-definition reuse baseline eval shards.")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--input_files", nargs="+", required=True)
    parser.add_argument("--model")
    parser.add_argument("--base_model")
    parser.add_argument("--reuse_model")
    parser.add_argument("--dataset_path")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--modes")
    parser.add_argument("--ratios")
    parser.add_argument("--cacheblend_recompute_ratio", type=float, default=0.15)
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    missing_files = []
    for input_file in args.input_files:
        path = Path(input_file)
        if not path.exists():
            missing_files.append(str(path))
            continue
        rows.extend(_read_jsonl(path))

    output_path = Path(args.output_file)
    _write_jsonl(output_path, rows)
    common_subset = _common_subset(rows)
    common_subset_results = _summarize_common_rows(rows)
    summary = {
        "model": args.model,
        "base_model": args.base_model,
        "reuse_model": args.reuse_model,
        "dataset_path": args.dataset_path,
        "split": args.split,
        "modes": [item.strip() for item in (args.modes or "").split(",") if item.strip()],
        "ratios": [item.strip() for item in (args.ratios or "").split(",") if item.strip()],
        "num_rows": len(rows),
        "missing_files": missing_files,
        "notes": {
            "epic_leading32": "PyTorch selective recompute with recompute_type=leading-32.",
            "cacheblend_vdiff": f"PyTorch value-difference selective recompute with recompute_type=vdiff-{args.cacheblend_recompute_ratio}; not the vLLM+LMCache expr_cacheblend.py path.",
            "snapkv_reuse": "Uses reuse_pipeline SnapKV compression, currently hard-coded to roughly 4x in compress_kv.",
            "epic_leading32_snapkv": "EPIC leading-32 selective recompute on top of SnapKV-compressed document KV.",
            "cacheblend_vdiff_snapkv": f"Value-difference selective recompute on top of SnapKV-compressed document KV with recompute_type=vdiff-{args.cacheblend_recompute_ratio}.",
            "common_subset_results": "Metrics recomputed only on qids that are valid for every present mode/ratio group.",
        },
        "results": _summarize_rows(rows),
        "common_subset": common_subset,
        "common_subset_results": common_subset_results,
        "common_subset_fairness_check": _common_fairness_check(common_subset_results, common_subset),
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
