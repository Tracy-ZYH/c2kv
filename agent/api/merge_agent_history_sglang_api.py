from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _dataset_name(summary: Dict[str, Any]) -> str:
    explicit = summary.get("dataset_name")
    if explicit:
        return str(explicit)
    path = str(summary.get("dataset_path") or "")
    return Path(path).name or "dataset"


def _part_key(path: str) -> Tuple[str, str]:
    stem = Path(path).stem
    for mode in ("hybrid", "c2kv", "full"):
        suffix = f"_{mode}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)], mode
    return "dataset", stem


def _valid_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in rows if not row.get("skipped")]


def _summarize_rows(
    dataset_name: str,
    mode: str,
    rows: List[Dict[str, Any]],
    common_qids: List[str],
) -> Dict[str, Any]:
    valid = _valid_rows(rows)
    extract_records = [
        record
        for row in valid
        for record in row.get("extracts", [])
        if isinstance(record, dict)
    ]
    return {
        "dataset_name": dataset_name,
        "mode": mode,
        "ratio": valid[0].get("ratio") if valid else None,
        "num_examples": len(rows),
        "num_valid": len(valid),
        "num_skipped": len(rows) - len(valid),
        "common_valid_qids": common_qids,
        "skip_reasons": dict(
            Counter(row.get("skip_reason", "unknown") for row in rows if row.get("skipped"))
        ),
        "tool_name_match": (
            sum(1 for row in valid if row.get("tool_name_match")) / len(valid)
            if valid else 0.0
        ),
        "exact_match": (
            sum(1 for row in valid if row.get("exact_match")) / len(valid)
            if valid else 0.0
        ),
        "response_type_match": (
            sum(1 for row in valid if row.get("response_type_match")) / len(valid)
            if valid else 0.0
        ),
        "text_token_f1": (
            sum(float(row.get("text_token_f1", 0.0) or 0.0) for row in valid) / len(valid)
            if valid else 0.0
        ),
        "rouge_l_f1": (
            sum(float(row.get("rouge_l_f1", 0.0) or 0.0) for row in valid) / len(valid)
            if valid else 0.0
        ),
        "avg_actual_compression_ratio": (
            sum(float(row.get("actual_compression_ratio", 0.0) or 0.0) for row in valid) / len(valid)
            if valid else 0.0
        ),
        "num_extracts": len(extract_records),
        "extract_success_rate": (
            sum(1 for record in extract_records if record.get("success", True)) / len(extract_records)
            if extract_records else None
        ),
        "avg_chat_seconds": (
            sum(float(row.get("chat_seconds", 0.0) or 0.0) for row in valid) / len(valid)
            if valid else 0.0
        ),
        "avg_total_seconds": (
            sum(float(row.get("total_seconds", 0.0) or 0.0) for row in valid) / len(valid)
            if valid else 0.0
        ),
    }


def _write_report(
    path: str,
    summaries: List[Dict[str, Any]],
    common_valid_subset: bool,
) -> None:
    lines = [
        "# Agent History SGLang API Results",
        "",
        "| dataset | mode | valid | tool_name_match | exact_match | response_type_match | text_f1 | rouge_l | ratio | extracts | extract_success | avg_chat_s | avg_total_s |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if common_valid_subset:
        lines.insert(2, "Metrics are recomputed on the per-dataset common valid `qid` subset.")
        lines.insert(3, "")
    for item in sorted(summaries, key=lambda x: (_dataset_name(x), str(x.get("mode")))):
        extract_success = item.get("extract_success_rate")
        lines.append(
            "| {dataset} | {mode} | {valid} | {tool_name:.4f} | {exact:.4f} | "
            "{type_match:.4f} | {text_f1:.4f} | {rouge_l:.4f} | {ratio:.4f} | "
            "{extracts} | {extract_success} | {chat:.4f} | {total:.4f} |".format(
                dataset=_dataset_name(item),
                mode=item.get("mode"),
                valid=int(item.get("num_valid", 0) or 0),
                tool_name=float(item.get("tool_name_match", 0.0) or 0.0),
                exact=float(item.get("exact_match", 0.0) or 0.0),
                type_match=float(item.get("response_type_match", 0.0) or 0.0),
                text_f1=float(item.get("text_token_f1", 0.0) or 0.0),
                rouge_l=float(item.get("rouge_l_f1", 0.0) or 0.0),
                ratio=float(item.get("avg_actual_compression_ratio", 0.0) or 0.0),
                extracts=int(item.get("num_extracts", 0) or 0),
                extract_success=(
                    "-"
                    if extract_success is None
                    else f"{float(extract_success):.4f}"
                ),
                chat=float(item.get("avg_chat_seconds", 0.0) or 0.0),
                total=float(item.get("avg_total_seconds", 0.0) or 0.0),
            )
        )
    lines.append("")
    lines.append(
        "Merged row files are under `merged_common/`."
        if common_valid_subset
        else "Merged row files are under `merged/`."
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge agent history SGLang API evaluation outputs.")
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--common-valid-subset",
        action="store_true",
        help="Recompute summaries from each dataset's common valid qid set across modes.",
    )
    args = parser.parse_args()

    part_paths = sorted(glob.glob(args.input_glob))
    summaries = []
    all_rows: List[Dict[str, Any]] = []
    merged_dir = Path(args.output_dir) / (
        "merged_common" if args.common_valid_subset else "merged"
    )
    merged_dir.mkdir(parents=True, exist_ok=True)
    rows_by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    for part_path in part_paths:
        rows = _read_jsonl(part_path)
        dataset_name, mode = _part_key(part_path)
        rows_by_key[(dataset_name, mode)] = rows

    common_qids_by_dataset: Dict[str, List[str]] = {}
    if args.common_valid_subset:
        datasets = sorted({dataset_name for dataset_name, _mode in rows_by_key})
        for dataset_name in datasets:
            qid_sets = [
                {str(row.get("qid")) for row in _valid_rows(rows)}
                for (cur_dataset, _mode), rows in rows_by_key.items()
                if cur_dataset == dataset_name
            ]
            common = sorted(set.intersection(*qid_sets)) if qid_sets else []
            common_qids_by_dataset[dataset_name] = common

    for part_path in part_paths:
        name = Path(part_path).stem
        dataset_name, mode = _part_key(part_path)
        rows = rows_by_key[(dataset_name, mode)]
        if args.common_valid_subset:
            common_qids = set(common_qids_by_dataset.get(dataset_name, []))
            rows = [
                row
                for row in rows
                if not row.get("skipped") and str(row.get("qid")) in common_qids
            ]
            summaries.append(
                _summarize_rows(
                    dataset_name,
                    mode,
                    rows,
                    common_qids_by_dataset.get(dataset_name, []),
                )
            )
        else:
            summary_path = str(Path(part_path).with_suffix(".summary.json"))
            if os.path.exists(summary_path):
                summaries.append(_read_json(summary_path))
        all_rows.extend(rows)
        _write_jsonl(str(merged_dir / f"{name}.jsonl"), rows)

    _write_jsonl(str(merged_dir / "all.jsonl"), all_rows)
    summary_payload = {
        "results": summaries,
        "num_rows": len(all_rows),
        "common_valid_subset": args.common_valid_subset,
        "common_valid_counts": {
            dataset_name: len(qids)
            for dataset_name, qids in common_qids_by_dataset.items()
        },
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_report(
        str(Path(args.output_dir) / "report.md"),
        summaries,
        args.common_valid_subset,
    )


if __name__ == "__main__":
    main()
