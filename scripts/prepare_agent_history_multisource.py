
#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm
from transformers import AutoTokenizer

from agent_history_multisource_lib import (
    build_decision_points,
    group_split,
    iter_agent_llm,
    iter_hermes,
    iter_toucan,
    select_mixed_samples,
    summarize_values,
    write_jsonl,
)


def chat_ids(tokenizer, messages, *, tools=None, add_generation_prompt=False, max_length=None):
    encoded = tokenizer.apply_chat_template(
        list(messages),
        tools=tools,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
        max_length=max_length,
        truncation=max_length is not None,
    )
    return encoded.input_ids if hasattr(encoded, "input_ids") else encoded


def split_history_docs(sample: Dict[str, Any], tokenizer, max_doc_length: int) -> tuple[Dict[str, Any], int]:
    out_docs = []
    split_count = 0
    for doc in sample.get("history_docs", []):
        content = str(doc.get("content") or "")
        token_count = len(chat_ids(tokenizer, [{"role": "user", "content": content}]))
        if token_count <= max_doc_length:
            new_doc = dict(doc)
            new_doc["token_count"] = token_count
            out_docs.append(new_doc)
            continue
        token_ids = tokenizer.encode(content, add_special_tokens=False)
        step = max(1, max_doc_length - 32)
        for chunk_id, start in enumerate(range(0, len(token_ids), step)):
            text = tokenizer.decode(token_ids[start : start + step], skip_special_tokens=True)
            if not text:
                continue
            out_docs.append({
                "turn_id": doc.get("turn_id"),
                "chunk_id": chunk_id,
                "content": text,
                "token_count": len(chat_ids(tokenizer, [{"role": "user", "content": text}])),
                "split_from_oversized_doc": True,
            })
        split_count += 1
    sample = dict(sample)
    sample["history_docs"] = out_docs
    sample.setdefault("metadata", {})["oversized_history_docs_split"] = split_count
    return sample, split_count


def add_token_stats(sample: Dict[str, Any], tokenizer, max_doc_length: int, max_history_docs: int) -> tuple[Dict[str, Any], Dict[str, int]]:
    sample, split_count = split_history_docs(sample, tokenizer, max_doc_length)
    if len(sample["history_docs"]) > max_history_docs:
        sample["history_docs"] = sample["history_docs"][-max_history_docs:]
        sample.setdefault("metadata", {})["history_docs_truncated_to_max"] = True
    system_tokens = len(chat_ids(tokenizer, sample.get("system") or [], tools=sample.get("tools") or None))
    history_lens = [int(doc.get("token_count") or 0) for doc in sample.get("history_docs", [])]
    current_tokens = len(chat_ids(tokenizer, sample.get("current_messages") or [], add_generation_prompt=True))
    target_tokens = len(tokenizer.encode(sample.get("target") or "", add_special_tokens=False))
    tool_tokens = len(chat_ids(tokenizer, [{"role": "system", "content": ""}], tools=sample.get("tools") or None)) if sample.get("tools") else 0
    total = system_tokens + sum(history_lens) + current_tokens + target_tokens
    sample.setdefault("metadata", {}).update({
        "system_token_count": system_tokens,
        "tools_token_count": tool_tokens,
        "history_doc_token_counts": history_lens,
        "current_token_count": current_tokens,
        "target_token_count": target_tokens,
        "total_full_context_token_count": total,
    })
    return sample, {
        "split_count": split_count,
        "history_turn_count": len(sample.get("history_docs", [])),
        "current_tokens": current_tokens,
        "target_tokens": target_tokens,
        "tools_tokens": tool_tokens,
        "total_tokens": total,
    }



def group_split_name(group_id: str, seed: int, val_ratio: float) -> str:
    payload = f"{seed}:{group_id}".encode("utf-8")
    value = int(hashlib.md5(payload).hexdigest()[:12], 16) / float(16**12)
    return "val" if value < val_ratio else "train"


def take_quota(items: List[Dict[str, Any]], quota: int, seed: int) -> List[Dict[str, Any]]:
    import random

    rng_items = list(items)
    random.Random(seed).shuffle(rng_items)
    if len(rng_items) < quota:
        raise RuntimeError(f"Requested {quota} samples but only built {len(rng_items)}")
    return rng_items[:quota]


def append_progress(output_dir: str | Path, event: str, payload: Dict[str, Any]) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event, **payload}
    with (out / "prepare_progress.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_for_stats(rows: List[Dict[str, Any]], sample_size: int, seed: int, label: str) -> List[Dict[str, Any]]:
    if sample_size <= 0 or len(rows) <= sample_size:
        return list(rows)
    import random

    sampled = list(rows)
    random.Random(f"{seed}:token-stats:{label}").shuffle(sampled)
    return sampled[:sample_size]


def build_strict_quota_samples(args, tokenizer) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    import random

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    append_progress(args.output_dir, "start", {
        "pid": os.getpid(),
        "strict_train_samples": args.strict_train_samples,
        "val_samples": args.val_samples,
        "max_samples_per_trajectory": args.max_samples_per_trajectory,
    })
    stats = Counter()
    train_pools: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    val_pools: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    train_quotas = {
        "agent_llm": args.train_agent_samples,
        "toucan": args.train_toucan_samples,
        "hermes": args.train_hermes_samples,
    }
    val_quotas = {
        "agent_llm": round(args.val_samples * args.train_agent_samples / args.strict_train_samples),
        "toucan": round(args.val_samples * args.train_toucan_samples / args.strict_train_samples),
        "hermes": 0,
    }
    val_quotas["hermes"] = args.val_samples - val_quotas["agent_llm"] - val_quotas["toucan"]

    iterators = [
        ("agent_llm", iter_agent_llm(args.agent_path, stats)),
        ("toucan", iter_toucan(args.toucan_path, stats, allow_irrelevant=False)),
        ("hermes", iter_hermes(args.hermes_path, stats, strip_reasoning=args.strip_reasoning)),
    ]
    for source, iterator in iterators:
        pbar = tqdm(iterator, desc=f"strict build {source}")
        seen_trajectories = 0
        last_progress = time.monotonic()
        for traj in pbar:
            seen_trajectories += 1
            split = group_split_name(traj.group_id, args.seed, args.val_ratio)
            samples = build_decision_points(
                traj,
                min_completed_history_turns=args.min_completed_history_turns,
                max_history_docs=args.max_history_docs,
            )
            if not samples:
                continue
            random.Random(f"{args.seed}:{traj.group_id}").shuffle(samples)
            samples = samples[: args.max_samples_per_trajectory]
            stats[f"{source}_{split}_decision_points"] += len(samples)
            pool = val_pools if split == "val" else train_pools
            pool[source].extend(samples)
            pbar.set_postfix(train=len(train_pools[source]), val=len(val_pools[source]))
            now = time.monotonic()
            if now - last_progress >= args.progress_interval_seconds:
                append_progress(args.output_dir, "strict_build", {
                    "source": source,
                    "seen_trajectories": seen_trajectories,
                    "train_pool": len(train_pools[source]),
                    "val_pool": len(val_pools[source]),
                    "train_quota": train_quotas[source],
                    "val_quota": val_quotas[source],
                })
                last_progress = now
            if len(train_pools[source]) >= train_quotas[source] and len(val_pools[source]) >= val_quotas[source]:
                break
        append_progress(args.output_dir, "strict_build_done", {
            "source": source,
            "seen_trajectories": seen_trajectories,
            "train_pool": len(train_pools[source]),
            "val_pool": len(val_pools[source]),
            "train_quota": train_quotas[source],
            "val_quota": val_quotas[source],
        })

    train_selected = []
    val_selected = []
    for source in ["agent_llm", "toucan", "hermes"]:
        train_selected.extend(take_quota(train_pools[source], train_quotas[source], args.seed + len(source)))
        val_selected.extend(take_quota(val_pools[source], val_quotas[source], args.seed + 100 + len(source)))

    processed_train = list(train_selected)
    processed_val = list(val_selected)
    length_stats = defaultdict(list)
    split_docs = 0
    stats_sample_sizes = {
        "train": args.stats_sample_size,
        "val": args.val_stats_sample_size if args.val_stats_sample_size is not None else args.stats_sample_size,
    }
    stats_actual_counts = {}
    for split_name, selected in [
        ("train", processed_train),
        ("val", processed_val),
    ]:
        stat_rows = sample_for_stats(selected, stats_sample_sizes[split_name], args.seed, split_name)
        stats_actual_counts[split_name] = len(stat_rows)
        append_progress(args.output_dir, "token_stats_start", {
            "split": split_name,
            "sampled": len(stat_rows),
            "total_rows": len(selected),
        })
        for sample_index, sample in enumerate(tqdm(stat_rows, desc=f"{split_name} token stats"), start=1):
            stat_sample, row_stats = add_token_stats(dict(sample), tokenizer, args.max_doc_length, args.max_history_docs)
            split_docs += row_stats["split_count"]
            for key in ["history_turn_count", "current_tokens", "target_tokens", "tools_tokens", "total_tokens"]:
                length_stats[f"{split_name}_{key}"].append(row_stats[key])
            for count in stat_sample.get("metadata", {}).get("history_doc_token_counts", []):
                length_stats[f"{split_name}_history_doc_tokens"].append(count)
            if sample_index % args.progress_interval_samples == 0:
                append_progress(args.output_dir, "token_stats", {
                    "split": split_name,
                    "processed": sample_index,
                    "sampled": len(stat_rows),
                    "total_rows": len(selected),
                })
        append_progress(args.output_dir, "token_stats_done", {
            "split": split_name,
            "processed": len(stat_rows),
            "sampled": len(stat_rows),
            "total_rows": len(selected),
        })

    train_groups = {s["group_id"] for s in processed_train}
    val_groups = {s["group_id"] for s in processed_val}
    overlap = train_groups & val_groups
    assert not overlap, f"train/val group overlap detected: {list(overlap)[:5]}"

    stats_payload = {
        "raw_and_filter_stats": dict(stats),
        "pool_sizes": {
            "train": {k: len(v) for k, v in train_pools.items()},
            "val": {k: len(v) for k, v in val_pools.items()},
        },
        "train_samples": len(processed_train),
        "val_samples": len(processed_val),
        "strict_train_samples": args.strict_train_samples,
        "requested_train_source_distribution": train_quotas,
        "requested_val_source_distribution": val_quotas,
        "train_source_distribution": Counter(s["source"] for s in processed_train),
        "val_source_distribution": Counter(s["source"] for s in processed_val),
        "train_val_group_overlap": len(overlap),
        "train_group_count": len(train_groups),
        "val_group_count": len(val_groups),
        "train_history_length_distribution": summarize_values(length_stats["train_history_turn_count"]),
        "train_history_document_token_length": summarize_values(length_stats["train_history_doc_tokens"]),
        "train_current_turn_token_length": summarize_values(length_stats["train_current_tokens"]),
        "train_tools_token_length": summarize_values(length_stats["train_tools_tokens"]),
        "train_total_context_token_length": summarize_values(length_stats["train_total_tokens"]),
        "val_history_length_distribution": summarize_values(length_stats["val_history_turn_count"]),
        "val_history_document_token_length": summarize_values(length_stats["val_history_doc_tokens"]),
        "val_current_turn_token_length": summarize_values(length_stats["val_current_tokens"]),
        "val_tools_token_length": summarize_values(length_stats["val_tools_tokens"]),
        "val_total_context_token_length": summarize_values(length_stats["val_total_tokens"]),
        "oversized_history_docs_split_in_stats_sample": split_docs,
        "token_stats_sampled": True,
        "token_stats_sample_size_requested": {"train": stats_sample_sizes["train"], "val": stats_sample_sizes["val"]},
        "token_stats_sample_size_actual": stats_actual_counts,
        "max_samples_per_trajectory": args.max_samples_per_trajectory,
        "ratio_ready_statistics": {"supported_ratios": [2, 4, 8], "homogeneous_sample_level_ratio": True},
        "examples": processed_train[:3],
    }
    return processed_train, processed_val, stats_payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_path", default="datasets/agent-llm-traces-v2")
    parser.add_argument("--hermes_path", default="datasets/hermes-agent-reasoning-traces")
    parser.add_argument("--toucan_path", default="datasets/toucan-1.5m")
    parser.add_argument("--model_name_or_path", default="models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--output_dir", default="datasets/processed_agent_history_c2kv")
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--strict_train_samples", type=int, default=0)
    parser.add_argument("--train_agent_samples", type=int, default=21000)
    parser.add_argument("--train_toucan_samples", type=int, default=21000)
    parser.add_argument("--train_hermes_samples", type=int, default=18000)
    parser.add_argument("--val_samples", type=int, default=3000)
    parser.add_argument("--max_samples_per_trajectory", type=int, default=8)
    parser.add_argument("--max_history_docs", type=int, default=16)
    parser.add_argument("--max_doc_length", type=int, default=1024)
    parser.add_argument("--min_completed_history_turns", type=int, default=2)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strip_reasoning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress_interval_seconds", type=float, default=30.0)
    parser.add_argument("--progress_interval_samples", type=int, default=1000)
    parser.add_argument("--stats_sample_size", type=int, default=5000)
    parser.add_argument("--val_stats_sample_size", type=int, default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, local_files_only=True, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id


    if args.strict_train_samples:
        expected = args.train_agent_samples + args.train_toucan_samples + args.train_hermes_samples
        if expected != args.strict_train_samples:
            raise ValueError(f"strict_train_samples={args.strict_train_samples} but source quotas sum to {expected}")
        train, val, final_stats = build_strict_quota_samples(args, tokenizer)
        out = Path(args.output_dir)
        train_count = write_jsonl(out / "train.jsonl", train)
        val_count = write_jsonl(out / "val.jsonl", val)
        final_stats["train_samples"] = train_count
        final_stats["val_samples"] = val_count
        serializable = json.loads(json.dumps(final_stats, ensure_ascii=False, default=lambda x: dict(x)))
        (out / "stats.json").write_text(json.dumps(serializable, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(serializable, ensure_ascii=False, indent=2)[:12000])
        return

    stats = Counter()
    pools: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    iterators = [
        ("agent_llm", iter_agent_llm(args.agent_path, stats)),
        ("toucan", iter_toucan(args.toucan_path, stats, allow_irrelevant=True)),
        ("hermes", iter_hermes(args.hermes_path, stats, strip_reasoning=args.strip_reasoning)),
    ]
    wanted_per_source = {"agent_llm": int(args.max_samples * 0.35 * 3), "toucan": int(args.max_samples * 0.35 * 3), "hermes": int(args.max_samples * 0.30 * 3)}
    for source, iterator in iterators:
        for traj in tqdm(iterator, desc=f"build {source}"):
            samples = build_decision_points(
                traj,
                min_completed_history_turns=args.min_completed_history_turns,
                max_history_docs=args.max_history_docs,
            )
            if samples:
                pools[source].extend(samples)
                stats[f"{source}_decision_points"] += len(samples)
            if len(pools[source]) >= wanted_per_source[source]:
                break

    selected = select_mixed_samples(pools, args.max_samples, seed=args.seed)
    processed = []
    length_stats = defaultdict(list)
    split_docs = 0
    for sample in tqdm(selected, desc="token stats"):
        sample, row_stats = add_token_stats(sample, tokenizer, args.max_doc_length, args.max_history_docs)
        processed.append(sample)
        split_docs += row_stats["split_count"]
        for key in ["history_turn_count", "current_tokens", "target_tokens", "tools_tokens", "total_tokens"]:
            length_stats[key].append(row_stats[key])
        for count in sample.get("metadata", {}).get("history_doc_token_counts", []):
            length_stats["history_doc_tokens"].append(count)

    train, val, split_info = group_split(processed, val_ratio=args.val_ratio, seed=args.seed)
    train_groups = {s["group_id"] for s in train}
    val_groups = {s["group_id"] for s in val}
    assert train_groups.isdisjoint(val_groups), "train/val group overlap detected"

    out = Path(args.output_dir)
    train_count = write_jsonl(out / "train.jsonl", train)
    val_count = write_jsonl(out / "val.jsonl", val)
    final_stats = {
        "raw_and_filter_stats": dict(stats),
        "pool_sizes": {k: len(v) for k, v in pools.items()},
        "train_samples": train_count,
        "val_samples": val_count,
        "split": split_info,
        "source_distribution": Counter(s["source"] for s in processed),
        "train_source_distribution": Counter(s["source"] for s in train),
        "val_source_distribution": Counter(s["source"] for s in val),
        "history_length_distribution": summarize_values(length_stats["history_turn_count"]),
        "history_document_token_length": summarize_values(length_stats["history_doc_tokens"]),
        "current_turn_token_length": summarize_values(length_stats["current_tokens"]),
        "tools_token_length": summarize_values(length_stats["tools_tokens"]),
        "total_context_token_length": summarize_values(length_stats["total_tokens"]),
        "oversized_history_docs_split": split_docs,
        "ratio_ready_statistics": {"supported_ratios": [2, 4, 8], "ratios_sampled_at_training_time": True},
        "example": processed[0] if processed else None,
    }
    serializable = json.loads(json.dumps(final_stats, ensure_ascii=False, default=lambda x: dict(x)))
    (out / "stats.json").write_text(json.dumps(serializable, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(serializable, ensure_ascii=False, indent=2)[:12000])


if __name__ == "__main__":
    main()
