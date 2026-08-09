import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize_mode(mode: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    extracts = [item for record in records for item in record.get("extracts", [])]
    successful_extracts = [item for item in extracts if item.get("success")]
    original_tokens = [
        item["original_seq_len"]
        for item in successful_extracts
        if isinstance(item.get("original_seq_len"), int)
    ]
    gist_tokens = [
        item["gist_len"]
        for item in successful_extracts
        if isinstance(item.get("gist_len"), int)
    ]
    total_original = sum(original_tokens)
    total_gist = sum(gist_tokens)
    prediction_counts = Counter(record.get("prediction", "") for record in records)

    return {
        "mode": mode,
        "num_examples": len(records),
        "avg_f1": (
            sum(float(record.get("f1", 0.0)) for record in records) / len(records)
            if records
            else 0.0
        ),
        "exact_match": (
            sum(float(record.get("exact_match", 0.0)) for record in records)
            / len(records)
            if records
            else 0.0
        ),
        "num_extracts": len(extracts),
        "extract_success_rate": (
            len(successful_extracts) / len(extracts) if extracts else None
        ),
        "total_original_seq_len": total_original,
        "total_gist_len": total_gist,
        "actual_compression_ratio": (
            total_original / total_gist if total_gist > 0 else None
        ),
        "avg_extract_seconds": (
            sum(record.get("timing", {}).get("extract_seconds", 0.0) for record in records)
            / len(records)
            if records
            else 0.0
        ),
        "avg_chat_seconds": (
            sum(record.get("timing", {}).get("chat_seconds", 0.0) for record in records)
            / len(records)
            if records
            else 0.0
        ),
        "top_predictions": prediction_counts.most_common(20),
    }


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(path: str, summaries: List[Dict[str, Any]]) -> None:
    lines = [
        "# SGLang C2KV HotpotQA Results",
        "",
        "| mode | examples | avg_f1 | exact_match | extracts | extract_success | actual_ratio | avg_extract_s | avg_chat_s | top_predictions |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for summary in summaries:
        top_predictions = ", ".join(
            f"{repr(pred)}:{count}" for pred, count in summary["top_predictions"][:5]
        )
        lines.append(
            "| {mode} | {num_examples} | {avg_f1} | {exact_match} | "
            "{num_extracts} | {extract_success_rate} | {actual_compression_ratio} | "
            "{avg_extract_seconds} | {avg_chat_seconds} | {top_predictions} |".format(
                mode=summary["mode"],
                num_examples=summary["num_examples"],
                avg_f1=fmt(summary["avg_f1"]),
                exact_match=fmt(summary["exact_match"]),
                num_extracts=summary["num_extracts"],
                extract_success_rate=fmt(summary["extract_success_rate"]),
                actual_compression_ratio=fmt(summary["actual_compression_ratio"]),
                avg_extract_seconds=fmt(summary["avg_extract_seconds"]),
                avg_chat_seconds=fmt(summary["avg_chat_seconds"]),
                top_predictions=top_predictions.replace("|", "\\|"),
            )
        )

    lines.append("")
    lines.append("Merged JSONL files are written under `merged/`.")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge SGLang C2KV shard results.")
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--merged-dir", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--report-md", required=True)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.input_glob))
    if not paths:
        raise FileNotFoundError(f"No JSONL files match {args.input_glob!r}")

    by_mode: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    duplicate_indices: Dict[str, List[int]] = defaultdict(list)
    for path in paths:
        for record in read_jsonl(path):
            mode = record.get("mode") or Path(path).name.split("_shard", 1)[0]
            idx = int(record["idx"])
            if idx in by_mode[mode]:
                duplicate_indices[mode].append(idx)
            by_mode[mode][idx] = record

    summaries = []
    for mode in sorted(by_mode):
        records = [by_mode[mode][idx] for idx in sorted(by_mode[mode])]
        write_jsonl(os.path.join(args.merged_dir, f"{mode}.jsonl"), records)
        summary = summarize_mode(mode, records)
        if duplicate_indices.get(mode):
            summary["duplicate_indices"] = sorted(set(duplicate_indices[mode]))
        summaries.append(summary)

    output = {
        "input_files": paths,
        "merged_dir": args.merged_dir,
        "summaries": summaries,
    }
    os.makedirs(os.path.dirname(args.summary_json), exist_ok=True)
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    write_markdown(args.report_md, summaries)

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
