from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pyarrow.parquet as pq


def _find_parquet_files(path: Path) -> List[Path]:
    if path.is_file() and path.suffix == ".parquet":
        return [path]
    roots = [path / "data", path]
    files: List[Path] = []
    for root in roots:
        if root.is_dir():
            files = sorted(root.glob("*.parquet"))
            if not files:
                files = sorted(root.rglob("*.parquet"))
        if files:
            break
    return files


def _iter_rows(data_files: Iterable[Path]) -> Iterable[Dict[str, Any]]:
    columns = ["benchmark", "session_id"]
    for data_file in data_files:
        pf = pq.ParquetFile(data_file)
        available = set(pf.schema_arrow.names)
        read_columns = [column for column in columns if column in available]
        for batch in pf.iter_batches(batch_size=1024, columns=read_columns):
            yield from batch.to_pylist()


def build_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    data_files = _find_parquet_files(Path(args.dataset_path))
    if not data_files:
        raise FileNotFoundError(f"No parquet files found under {args.dataset_path}")

    eval_subsets = {item.strip() for item in args.eval_subsets.split(",") if item.strip()}
    if not eval_subsets:
        raise ValueError("--eval_subsets must contain at least one subset name")

    train_ids: List[str] = []
    eval_ids: List[str] = []
    subsets: Dict[str, set[str]] = defaultdict(set)
    missing_session_ids = 0
    duplicate_session_ids = 0
    seen_session_ids: set[str] = set()

    for row_index, row in enumerate(_iter_rows(data_files)):
        session_id = row.get("session_id")
        if session_id is None:
            missing_session_ids += 1
            session_id = f"row-{row_index}"
        session_id = str(session_id)
        if session_id in seen_session_ids:
            duplicate_session_ids += 1
        seen_session_ids.add(session_id)

        subset = str(row.get("benchmark") or "unknown")
        subsets[subset].add(session_id)
        if subset in eval_subsets:
            eval_ids.append(session_id)
        else:
            train_ids.append(session_id)

    train_set = set(train_ids)
    eval_set = set(eval_ids)
    overlap = train_set & eval_set
    if overlap:
        raise RuntimeError(f"Train/eval session overlap detected: {sorted(overlap)[:5]}")
    missing_eval_subsets = sorted(eval_subsets - set(subsets))
    if missing_eval_subsets:
        raise ValueError(
            f"Requested eval subsets not found: {missing_eval_subsets}. "
            f"Available subsets: {sorted(subsets)}"
        )

    subset_counts = {
        subset: {
            "sessions": len(session_ids),
            "split": "eval" if subset in eval_subsets else "train",
        }
        for subset, session_ids in sorted(subsets.items())
    }
    split_name = args.split_name
    return {
        split_name: {
            "train_session_ids": sorted(train_set),
            "eval_session_ids": sorted(eval_set),
            "eval_subsets": sorted(eval_subsets),
            "subset_counts": subset_counts,
        },
        "metadata": {
            "dataset_path": args.dataset_path,
            "num_parquet_files": len(data_files),
            "split_name": split_name,
            "eval_subsets": sorted(eval_subsets),
            "num_train_sessions": len(train_set),
            "num_eval_sessions": len(eval_set),
            "num_total_sessions": len(train_set) + len(eval_set),
            "missing_session_ids": missing_session_ids,
            "duplicate_session_ids": duplicate_session_ids,
            "subsets": sorted(subsets),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a subset-disjoint split manifest for agent-llm-traces.")
    parser.add_argument("--dataset_path", default="./datasets/agent-llm-traces")
    parser.add_argument("--output_file", default="./outputs/agent_tooldef_subset_split_manifest.json")
    parser.add_argument("--split_name", default="subset_disjoint")
    parser.add_argument("--eval_subsets", default="swebench")
    args = parser.parse_args()

    manifest = build_manifest(args)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = manifest["metadata"]
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print()
    print("| subset | split | sessions |")
    print("|---|---|---:|")
    for subset, item in manifest[args.split_name]["subset_counts"].items():
        print(f"| {subset} | {item['split']} | {item['sessions']} |")


if __name__ == "__main__":
    main()
