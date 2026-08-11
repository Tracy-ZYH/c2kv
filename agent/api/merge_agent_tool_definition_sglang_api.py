from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(path: str, summaries: List[Dict[str, Any]]) -> None:
    lines = [
        "# Agent Tool-Definition SGLang API Results",
        "",
        "| mode | valid | tool_name_match | exact_match | response_type_match | text_f1 | rouge_l | ratio | extracts | extract_success | avg_chat_s | avg_total_s | top_predictions |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in summaries:
        top = ", ".join(f"{repr(pred)}:{count}" for pred, count in item.get("top_predictions", [])[:5])
        lines.append(
            "| {mode} | {num_valid} | {tool_name_match:.4f} | {exact_match:.4f} | "
            "{response_type_match:.4f} | {text_token_f1:.4f} | {rouge_l_f1:.4f} | "
            "{avg_actual_compression_ratio:.4f} | {num_extracts} | {extract_success} | "
            "{avg_chat_seconds:.4f} | {avg_total_seconds:.4f} | {top} |".format(
                mode=item.get("mode"),
                num_valid=item.get("num_valid", 0),
                tool_name_match=float(item.get("tool_name_match", 0.0) or 0.0),
                exact_match=float(item.get("exact_match", 0.0) or 0.0),
                response_type_match=float(item.get("response_type_match", 0.0) or 0.0),
                text_token_f1=float(item.get("text_token_f1", 0.0) or 0.0),
                rouge_l_f1=float(item.get("rouge_l_f1", 0.0) or 0.0),
                avg_actual_compression_ratio=float(item.get("avg_actual_compression_ratio", 0.0) or 0.0),
                num_extracts=item.get("num_extracts", 0),
                extract_success=(
                    "-"
                    if item.get("extract_success_rate") is None
                    else f"{float(item.get('extract_success_rate')):.4f}"
                ),
                avg_chat_seconds=float(item.get("avg_chat_seconds", 0.0) or 0.0),
                avg_total_seconds=float(item.get("avg_total_seconds", 0.0) or 0.0),
                top=top.replace("|", "\\|"),
            )
        )
    lines.append("")
    lines.append("Merged row files are under `merged/`.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge agent SGLang API evaluation outputs.")
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.input_glob))
    if not paths:
        raise FileNotFoundError(f"No files match {args.input_glob!r}")

    summaries = []
    merged_dir = os.path.join(args.output_dir, "merged")
    for path in paths:
        if path.endswith(".summary.json"):
            continue
        rows = read_jsonl(path)
        mode = rows[0].get("mode") if rows else Path(path).stem.split("_", 1)[0]
        write_jsonl(os.path.join(merged_dir, f"{mode}.jsonl"), rows)
        summary_path = str(Path(path).with_suffix(".summary.json"))
        if os.path.exists(summary_path):
            summaries.append(read_json(summary_path))

    combined = {
        "input_files": paths,
        "summaries": sorted(summaries, key=lambda item: str(item.get("mode"))),
    }
    os.makedirs(args.output_dir, exist_ok=True)
    summary_file = os.path.join(args.output_dir, "summary.json")
    report_file = os.path.join(args.output_dir, "report.md")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    write_report(report_file, combined["summaries"])
    print(json.dumps(combined, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
