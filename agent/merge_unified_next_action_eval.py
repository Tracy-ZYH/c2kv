from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from eval_unified_next_action_c2kv import _summarize


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


def _common_valid_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keys = sorted({(row.get("mode"), row.get("ratio")) for row in rows})
    if not keys:
        return []
    valid_ids_by_key = []
    for mode, ratio in keys:
        group_ids = {
            row.get("qid")
            for row in rows
            if row.get("mode") == mode and row.get("ratio") == ratio and not row.get("skipped")
        }
        valid_ids_by_key.append(group_ids)
    common_ids = set.intersection(*valid_ids_by_key) if valid_ids_by_key else set()
    return [row for row in rows if row.get("qid") in common_ids]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge unified next-action eval shards.")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--input_files", nargs="+", required=True)
    parser.add_argument("--model")
    parser.add_argument("--base_model")
    parser.add_argument("--split", default="eval")
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for input_file in args.input_files:
        rows.extend(_read_jsonl(Path(input_file)))

    output_path = Path(args.output_file)
    _write_jsonl(output_path, rows)
    summary = {
        "model": args.model,
        "base_model": args.base_model,
        "split": args.split,
        "num_rows": len(rows),
        "results": _summarize(rows),
        "common_num_qids": len({row.get("qid") for row in _common_valid_rows(rows)}),
        "common_results": _summarize(_common_valid_rows(rows)),
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
