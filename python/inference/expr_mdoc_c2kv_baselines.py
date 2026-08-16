from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "agent"))

from eval_agent_tool_definition_c2kv import (  # noqa: E402
    _build_tool_cache,
    _generate_from_input_ids,
    _load_model,
    _prefill_system,
    _prefill_tokens_with_cache,
    _setup_device,
)
from mdocdataset import (  # noqa: E402
    AbstractMDQADataset,
    load_mdoc_dataset,
    max_f1_score,
    max_rouge_score,
)
from reuse_pipeline import tokenize_for_reuse  # noqa: E402


def _safe_metric(metric_fn, pred: str, ground_truth: Any) -> float:
    try:
        return float(metric_fn(pred or "", ground_truth))
    except Exception:
        return 0.0


def _sync_device(device: Any) -> None:
    device_type = getattr(device, "type", str(device))
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device_type == "npu" and hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()


def _cut_documents(documents: Sequence[str], max_chars: Optional[int]) -> List[str]:
    if max_chars is None:
        return [str(doc) for doc in documents]
    chunks: List[str] = []
    for document in documents:
        current = ""
        for passage in str(document).split("\n\n"):
            if not passage.strip():
                continue
            candidate = current + passage + "\n\n"
            if current and len(candidate) > max_chars:
                chunks.append(current)
                current = passage + "\n\n"
            else:
                current = candidate
        if current:
            chunks.append(current)
    return chunks


def _tokenize_one_doc(tokenizer: Any, text: str, max_doc_length: int, doc_selection: str) -> List[int]:
    inputs = tokenize_for_reuse(tokenizer, [text], keep_bos=False, role="user")
    length = int(inputs.attention_mask[0].sum().item())
    ids = inputs.input_ids[0, :length].tolist()
    if len(ids) > max_doc_length:
        if doc_selection == "head":
            ids = ids[:max_doc_length]
        elif doc_selection == "tail":
            ids = ids[-max_doc_length:]
        else:
            raise ValueError(f"Unsupported doc_selection={doc_selection!r}")
    return ids


def _pad(ids: Sequence[int], length: int, pad_value: int = -100) -> List[int]:
    if len(ids) >= length:
        return list(ids[:length])
    return list(ids) + [pad_value] * (length - len(ids))


def _text_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", (text or "").lower())


def _lexical_ranking(question: str, documents: Sequence[str]) -> List[int]:
    query_counts: Dict[str, int] = {}
    for token in _text_tokens(question):
        query_counts[token] = query_counts.get(token, 0) + 1
    scored = []
    for index, document in enumerate(documents):
        doc_counts: Dict[str, int] = {}
        for token in _text_tokens(document):
            doc_counts[token] = doc_counts.get(token, 0) + 1
        overlap = sum(min(count, doc_counts.get(token, 0)) for token, count in query_counts.items())
        scored.append((overlap, -index, index))
    scored.sort(reverse=True)
    return [index for _, _, index in scored]


def _lexical_topk(question: str, documents: Sequence[str], top_k: int) -> set[int]:
    if top_k <= 0:
        return set()
    ranking = _lexical_ranking(question, documents)
    return set(ranking[: min(top_k, len(ranking))])


def _parse_rank_span(span: str) -> tuple[int, Optional[int]]:
    span = span.strip().lower()
    if span in {"*", "rest", "default"}:
        return 1, None
    if "-" in span:
        start_text, end_text = span.split("-", 1)
        start = int(start_text)
        end = int(end_text) if end_text else None
        return start, end
    rank = int(span)
    return rank, rank


def _parse_precision(value: str, default_ratio: int) -> tuple[str, int]:
    value = value.strip().lower().replace("_", "")
    if value == "full":
        return "full", 1
    if value == "c2kv":
        return "c2kv", default_ratio
    if value.startswith("c2kv"):
        ratio_text = value[len("c2kv") :]
        return "c2kv", int(ratio_text) if ratio_text else default_ratio
    raise ValueError(f"Unsupported rank plan precision={value!r}")


def _rank_plan_states(
    question: str,
    documents: Sequence[str],
    rank_plan: str,
    default_ratio: int,
) -> tuple[Dict[int, tuple[str, int]], List[int]]:
    ranking = _lexical_ranking(question, documents)
    rank_by_index = {index: rank for rank, index in enumerate(ranking, start=1)}
    rules: List[tuple[int, Optional[int], str, int]] = []
    for item in rank_plan.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                f"Invalid rank_plan item={item!r}. Expected syntax like '1:full,2-3:c2kv4,4-:full'."
            )
        span_text, precision_text = item.split(":", 1)
        start, end = _parse_rank_span(span_text)
        kind, ratio = _parse_precision(precision_text, default_ratio)
        rules.append((start, end, kind, ratio))
    if not rules:
        raise ValueError("rank_plan must not be empty when mode='rank_plan'")

    states: Dict[int, tuple[str, int]] = {}
    for index in range(len(documents)):
        rank = rank_by_index[index]
        state = ("c2kv", default_ratio)
        for start, end, kind, ratio in rules:
            if rank >= start and (end is None or rank <= end):
                state = (kind, ratio)
        states[index] = state
    return states, ranking


def _parse_int_list(text: str) -> List[int]:
    values: List[int] = []
    for item in (text or "").split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values


def _score_token_ids(query_counts: Dict[str, int], tokenizer: Any, token_ids: Sequence[int]) -> float:
    text = tokenizer.decode(list(token_ids), skip_special_tokens=True)
    doc_counts: Dict[str, int] = {}
    for token in _text_tokens(text):
        doc_counts[token] = doc_counts.get(token, 0) + 1
    overlap = sum(min(count, doc_counts.get(token, 0)) for token, count in query_counts.items())
    if overlap == 0:
        return 0.0
    return overlap / max(1.0, len(token_ids) ** 0.5)


def _evidence_span_candidates(
    question: str,
    doc_ids: Sequence[Sequence[int]],
    documents: Sequence[str],
    tokenizer: Any,
    candidate_docs: int,
    span_sizes: Sequence[int],
) -> List[Dict[str, Any]]:
    ranking = _lexical_ranking(question, documents)
    selected_docs = ranking[: min(candidate_docs, len(ranking))]
    query_counts: Dict[str, int] = {}
    for token in _text_tokens(question):
        query_counts[token] = query_counts.get(token, 0) + 1

    candidates: List[Dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    for doc_rank, doc_index in enumerate(selected_docs, start=1):
        ids = list(doc_ids[doc_index])
        if not ids:
            continue
        for size in sorted(set(span_sizes), reverse=True):
            if size <= 0:
                continue
            window = min(size, len(ids))
            stride = max(1, window // 2)
            for start in range(0, len(ids), stride):
                end = min(len(ids), start + window)
                key = (doc_index, start, end)
                if key in seen or end <= start:
                    continue
                seen.add(key)
                span_ids = ids[start:end]
                score = _score_token_ids(query_counts, tokenizer, span_ids)
                if score <= 0:
                    continue
                candidates.append({
                    "doc_index": doc_index,
                    "doc_rank": doc_rank,
                    "start_token": start,
                    "end_token": end,
                    "token_count": len(span_ids),
                    "score": score,
                    "token_ids": span_ids,
                })
                if end == len(ids):
                    break
    candidates.sort(
        key=lambda item: (
            item["score"],
            -item["doc_rank"],
            -item["token_count"],
            -item["start_token"],
        ),
        reverse=True,
    )
    return candidates


def _input_ids_from_tokens(tokens: Sequence[int], device: Any) -> torch.Tensor:
    return torch.tensor([list(tokens)], dtype=torch.long, device=device)


@torch.inference_mode()
def _build_prefix(
    model: Any,
    tokenizer: Any,
    example: Dict[str, Any],
    args: argparse.Namespace,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    documents = _cut_documents(example["documents"], args.cut_length)
    if args.max_doc_num and len(documents) > args.max_doc_num:
        documents = list(documents[: args.max_doc_num])
    if not documents:
        return None, "empty_documents"

    doc_ids = [
        _tokenize_one_doc(tokenizer, document, args.max_doc_length, args.doc_selection)
        for document in documents
    ]
    doc_tokens = sum(len(ids) for ids in doc_ids)
    if args.max_context_tokens and doc_tokens > args.max_context_tokens:
        return None, f"context_tokens>{args.max_context_tokens}"

    full_indices: set[int]
    doc_states: Dict[int, tuple[str, int]]
    lexical_ranking: List[int] = []
    if args.mode == "full":
        full_indices = set(range(len(doc_ids)))
        doc_states = {index: ("full", 1) for index in range(len(doc_ids))}
    elif args.mode == "hybrid":
        full_indices = _lexical_topk(example["question"], documents, args.hybrid_top_k)
        doc_states = {
            index: ("full", 1) if index in full_indices else ("c2kv", args.override_ratio)
            for index in range(len(doc_ids))
        }
    elif args.mode == "c2kv":
        full_indices = set()
        doc_states = {index: ("c2kv", args.override_ratio) for index in range(len(doc_ids))}
    elif args.mode == "budget_recovery":
        full_indices = set()
        doc_states = {index: ("c2kv", args.override_ratio) for index in range(len(doc_ids))}
    elif args.mode == "rank_plan":
        doc_states, lexical_ranking = _rank_plan_states(
            example["question"],
            documents,
            args.rank_plan,
            args.override_ratio,
        )
        full_indices = {index for index, (kind, _ratio) in doc_states.items() if kind == "full"}
    else:
        raise ValueError(f"Unsupported mode={args.mode!r}")

    system_prompt = "You are a helpful assistant." if args.override_system_prompt else getattr(args, "system_prompt", None)
    if not system_prompt:
        system_prompt = "You are a helpful assistant."
    system_inputs = tokenize_for_reuse(tokenizer, [system_prompt], keep_bos=True, role="system")
    system_ids = system_inputs.input_ids.to(model.device)
    prefix_cache, system_length, system_prefill_sec = _prefill_system(
        model,
        system_ids,
        args.system_attn_impl,
    )

    logical_length = system_length
    full_doc_tokens = 0
    c2kv_doc_tokens = 0
    gist_tokens = 0
    full_prefill_sec = 0.0
    compress_sec = 0.0
    blend_sec = 0.0
    c2kv_doc_count = 0
    c2kv_doc_counts_by_ratio: Dict[str, int] = {}
    c2kv_doc_tokens_by_ratio: Dict[str, int] = {}
    gist_tokens_by_ratio: Dict[str, int] = {}
    recovery_span_tokens = 0
    recovery_span_count = 0
    recovery_spans: List[Dict[str, Any]] = []

    for index, ids in enumerate(doc_ids):
        doc_kind, doc_ratio = doc_states[index]
        if doc_kind == "full":
            input_ids = _input_ids_from_tokens(ids, model.device)
            prefix_cache, added_length, elapsed = _prefill_tokens_with_cache(
                model,
                input_ids,
                prefix_cache,
                logical_length,
                args.generate_attn_impl,
            )
            logical_length += added_length
            full_doc_tokens += added_length
            full_prefill_sec += elapsed
            continue

        context_input_ids = torch.tensor([_pad(ids, args.max_doc_length)], dtype=torch.long)
        before_cache_len = prefix_cache.get_seq_length()
        (
            prefix_cache,
            original_length,
            doc_gist_tokens,
            _actual_ratio,
            doc_compress_sec,
            doc_blend_sec,
        ) = _build_tool_cache(
            model,
            context_input_ids,
            prefix_cache,
            logical_length,
            args.gist_attn_impl,
            doc_ratio,
        )
        logical_length += original_length
        c2kv_doc_tokens += original_length
        added_gist_tokens = max(doc_gist_tokens, max(0, prefix_cache.get_seq_length() - before_cache_len))
        gist_tokens += added_gist_tokens
        ratio_key = str(doc_ratio)
        c2kv_doc_counts_by_ratio[ratio_key] = c2kv_doc_counts_by_ratio.get(ratio_key, 0) + 1
        c2kv_doc_tokens_by_ratio[ratio_key] = c2kv_doc_tokens_by_ratio.get(ratio_key, 0) + original_length
        gist_tokens_by_ratio[ratio_key] = gist_tokens_by_ratio.get(ratio_key, 0) + added_gist_tokens
        compress_sec += doc_compress_sec
        blend_sec += doc_blend_sec
        c2kv_doc_count += 1

    if args.mode == "budget_recovery":
        budget_tokens = max(1, doc_tokens // max(1, args.target_compression_ratio))
        current_compressed_tokens = full_doc_tokens + gist_tokens
        extra_budget = budget_tokens - current_compressed_tokens
        if extra_budget > 0:
            candidates = _evidence_span_candidates(
                example["question"],
                doc_ids,
                documents,
                tokenizer,
                args.recovery_candidate_docs,
                _parse_int_list(args.recovery_span_tokens),
            )
            used_docs: set[int] = set()
            used_ranges: Dict[int, List[tuple[int, int]]] = {}
            for candidate in candidates:
                token_count = int(candidate["token_count"])
                if token_count > extra_budget:
                    continue
                doc_index = int(candidate["doc_index"])
                if args.recovery_distinct_docs and doc_index in used_docs:
                    continue
                existing = used_ranges.setdefault(doc_index, [])
                start = int(candidate["start_token"])
                end = int(candidate["end_token"])
                if any(not (end <= old_start or start >= old_end) for old_start, old_end in existing):
                    continue

                input_ids = _input_ids_from_tokens(candidate["token_ids"], model.device)
                prefix_cache, added_length, elapsed = _prefill_tokens_with_cache(
                    model,
                    input_ids,
                    prefix_cache,
                    logical_length,
                    args.generate_attn_impl,
                )
                if added_length > extra_budget:
                    return None, "recovery_budget_overflow"
                logical_length += added_length
                full_doc_tokens += added_length
                recovery_span_tokens += added_length
                recovery_span_count += 1
                full_prefill_sec += elapsed
                extra_budget -= added_length
                used_docs.add(doc_index)
                existing.append((start, end))
                recovery_spans.append({
                    key: value
                    for key, value in candidate.items()
                    if key != "token_ids"
                })
                if recovery_span_count >= args.recovery_max_spans:
                    break

    compressed_context_tokens = full_doc_tokens + gist_tokens
    actual_ratio = doc_tokens / compressed_context_tokens if compressed_context_tokens else 0.0
    if args.mode == "budget_recovery" and compressed_context_tokens > doc_tokens / max(1, args.target_compression_ratio):
        return None, "budget_constraint_violation"
    return {
        "cache": prefix_cache,
        "logical_length": logical_length,
        "cache_length": prefix_cache.get_seq_length(),
        "doc_tokens": doc_tokens,
        "doc_chunks": len(doc_ids),
        "full_doc_count": len(full_indices),
        "c2kv_doc_count": c2kv_doc_count,
        "c2kv_doc_counts_by_ratio": c2kv_doc_counts_by_ratio,
        "full_doc_tokens": full_doc_tokens,
        "recovery_span_tokens": recovery_span_tokens,
        "recovery_span_count": recovery_span_count,
        "recovery_spans": recovery_spans,
        "c2kv_doc_tokens": c2kv_doc_tokens,
        "c2kv_doc_tokens_by_ratio": c2kv_doc_tokens_by_ratio,
        "gist_tokens": gist_tokens,
        "gist_tokens_by_ratio": gist_tokens_by_ratio,
        "compressed_context_tokens": compressed_context_tokens,
        "actual_compression_ratio": actual_ratio,
        "rank_plan": args.rank_plan if args.mode == "rank_plan" else None,
        "target_compression_ratio": args.target_compression_ratio if args.mode == "budget_recovery" else None,
        "lexical_ranked_doc_indices": lexical_ranking[:10] if lexical_ranking else None,
        "system_prefill_sec": system_prefill_sec,
        "full_prefill_sec": full_prefill_sec,
        "tool_compress_sec": compress_sec,
        "blend_sec": blend_sec,
        "use_gist": bool(c2kv_doc_count),
    }, None


@torch.inference_mode()
def _generate_one(
    model: Any,
    tokenizer: Any,
    dataset: AbstractMDQADataset,
    example: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    total_start = time.perf_counter()
    prefix, skip_reason = _build_prefix(model, tokenizer, example, args)
    if prefix is None:
        return {
            "qid": example.get("qid"),
            "mode": args.mode,
            "ratio": args.override_ratio if args.mode != "full" else 1,
            "skipped": True,
            "skip_reason": skip_reason,
        }

    query_inputs = tokenize_for_reuse(
        tokenizer,
        [example["question"]],
        keep_bos=False,
        role="user",
        add_generation_prompt=True,
    )
    query_ids = query_inputs.input_ids.to(model.device)
    if query_ids.shape[1] > args.max_query_tokens:
        query_ids = query_ids[:, -args.max_query_tokens :]

    query_length = query_ids.shape[1]
    cache_length = prefix["cache"].get_seq_length()
    mock_prefix_ids = torch.zeros((1, cache_length), dtype=torch.long, device=model.device)
    input_ids = torch.cat([mock_prefix_ids, query_ids], dim=1)
    position_ids = torch.arange(
        prefix["logical_length"],
        prefix["logical_length"] + query_length,
        dtype=torch.long,
        device=model.device,
    ).unsqueeze(0)
    prediction, generate_sec, generated_tokens, tbt_sec = _generate_from_input_ids(
        model,
        tokenizer,
        input_ids,
        max_new_tokens=example.get("max_new_tokens", dataset.max_new_tokens),
        attn_impl=args.generate_attn_impl,
        use_gist=prefix["use_gist"],
        position_ids=position_ids,
        past_key_values=prefix["cache"],
    )
    target = example["answer"]
    primary_score = _safe_metric(dataset.metric, prediction, target)
    token_f1 = _safe_metric(max_f1_score, prediction, target)
    rouge_l = _safe_metric(max_rouge_score, prediction, target)
    total_sec = time.perf_counter() - total_start
    return {
        "qid": example.get("qid"),
        "mode": args.mode,
        "ratio": args.override_ratio if args.mode != "full" else 1,
        "prediction": prediction,
        "ground_truth": target,
        "em_score": primary_score,
        "token_f1": token_f1,
        "rouge_l": rouge_l,
        "skipped": False,
        "prompt_tokens": query_length,
        "generated_tokens": generated_tokens,
        "ttft_sec": prefix["system_prefill_sec"] + prefix["full_prefill_sec"] + prefix["tool_compress_sec"] + prefix["blend_sec"],
        "generate_sec": generate_sec,
        "tbt_sec": tbt_sec,
        "total_sec": total_sec,
        **{key: value for key, value in prefix.items() if key != "cache"},
    }


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [row for row in rows if not row.get("skipped")]
    skipped = [row for row in rows if row.get("skipped")]
    generated_total = sum(row.get("generated_tokens", 0) for row in valid)
    compressed_total = sum(row.get("compressed_context_tokens", 0) for row in valid)
    doc_total = sum(row.get("doc_tokens", 0) for row in valid)
    return {
        "num_examples": len(rows),
        "num_valid": len(valid),
        "num_skipped": len(skipped),
        "skip_reasons": {
            reason: sum(1 for row in skipped if row.get("skip_reason") == reason)
            for reason in sorted({row.get("skip_reason", "unknown") for row in skipped})
        },
        "primary_score": sum(row.get("em_score", 0.0) for row in valid) / len(valid) if valid else 0.0,
        "avg_token_f1": sum(row.get("token_f1", 0.0) for row in valid) / len(valid) if valid else 0.0,
        "avg_rouge_l": sum(row.get("rouge_l", 0.0) for row in valid) / len(valid) if valid else 0.0,
        "avg_doc_tokens": doc_total / len(valid) if valid else 0.0,
        "avg_full_doc_tokens": sum(row.get("full_doc_tokens", 0) for row in valid) / len(valid) if valid else 0.0,
        "avg_recovery_span_tokens": sum(row.get("recovery_span_tokens", 0) for row in valid) / len(valid) if valid else 0.0,
        "avg_recovery_span_count": sum(row.get("recovery_span_count", 0) for row in valid) / len(valid) if valid else 0.0,
        "avg_gist_tokens": sum(row.get("gist_tokens", 0) for row in valid) / len(valid) if valid else 0.0,
        "avg_compressed_context_tokens": compressed_total / len(valid) if valid else 0.0,
        "avg_actual_compression_ratio": sum(row.get("actual_compression_ratio", 0.0) for row in valid) / len(valid) if valid else 0.0,
        "min_actual_compression_ratio": min((row.get("actual_compression_ratio", 0.0) for row in valid), default=0.0),
        "token_weighted_actual_compression_ratio": doc_total / compressed_total if compressed_total else 0.0,
        "avg_full_doc_count": sum(row.get("full_doc_count", 0) for row in valid) / len(valid) if valid else 0.0,
        "avg_c2kv_doc_count": sum(row.get("c2kv_doc_count", 0) for row in valid) / len(valid) if valid else 0.0,
        "avg_ttft_sec": sum(row.get("ttft_sec", 0.0) for row in valid) / len(valid) if valid else 0.0,
        "avg_generate_sec": sum(row.get("generate_sec", 0.0) for row in valid) / len(valid) if valid else 0.0,
        "avg_tbt_sec": sum(row.get("tbt_sec", 0.0) for row in valid) / len(valid) if valid else 0.0,
        "token_weighted_tbt_sec": sum(row.get("generate_sec", 0.0) for row in valid) / generated_total if generated_total else 0.0,
        "avg_total_sec": sum(row.get("total_sec", 0.0) for row in valid) / len(valid) if valid else 0.0,
    }


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    device = _setup_device(args.device_type)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model,
        trust_remote_code=True,
        local_files_only=True,
        padding_side="right",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model_args = argparse.Namespace(
        model=args.model,
        base_model=args.base_model,
        mode="c2kv",
        untrained_c2kv=False,
        baseline_model_class="gist",
        generate_attn_impl=args.generate_attn_impl,
        dtype=args.dtype,
    )
    model = _load_model(model_args, tokenizer, device)
    dataset = load_mdoc_dataset(
        args.dataset,
        args.dataset_path,
        only_supporting=args.only_supporting,
        enable_cot=args.cot,
    )
    rows: List[Dict[str, Any]] = []
    num_examples = (
        len(dataset)
        if args.max_examples is None or args.max_examples <= 0
        else min(args.max_examples, len(dataset))
    )
    for index in tqdm(range(num_examples), desc=f"{args.mode}@{args.override_ratio if args.mode != 'full' else 1}"):
        example = dataset[index]
        try:
            rows.append(_generate_one(model, tokenizer, dataset, example, args))
        except RuntimeError as error:
            message = str(error).lower()
            if "out of memory" not in message and "oom" not in message:
                raise
            if hasattr(torch, "npu") and torch.npu.is_available():
                torch.npu.empty_cache()
            rows.append({
                "qid": example.get("qid"),
                "mode": args.mode,
                "ratio": args.override_ratio if args.mode != "full" else 1,
                "skipped": True,
                "skip_reason": "oom",
            })

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary = {
            "model": args.model,
            "base_model": args.base_model,
            "dataset": dataset.__class__.__name__,
            "dataset_path": args.dataset_path,
            "mode": args.mode,
            "ratio": args.override_ratio if args.mode != "full" else 1,
            "hybrid_top_k": args.hybrid_top_k,
            **_summarize(rows),
        }
        output_path.with_suffix(".summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return _summarize(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct MDOC full/C2KV/hybrid eval without SGLang.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base_model")
    parser.add_argument("--tokenizer")
    parser.add_argument("--dataset", default="wikimqa")
    parser.add_argument("--dataset_path")
    parser.add_argument("--mode", choices=["full", "c2kv", "hybrid", "rank_plan", "budget_recovery"], required=True)
    parser.add_argument("--override_ratio", type=int, default=16)
    parser.add_argument("--hybrid_top_k", type=int, default=3)
    parser.add_argument(
        "--rank_plan",
        default="1:full,2-:c2kv",
        help=(
            "Lexical-rank precision plan for mode=rank_plan. Examples: "
            "'1:full,2-:c2kv16' or '1:full,2-3:c2kv4,4-:full'. "
            "Later rules override earlier matching rules."
        ),
    )
    parser.add_argument("--target_compression_ratio", type=int, default=8)
    parser.add_argument("--recovery_candidate_docs", type=int, default=4)
    parser.add_argument("--recovery_span_tokens", default="256,128,64")
    parser.add_argument("--recovery_max_spans", type=int, default=2)
    parser.add_argument("--recovery_distinct_docs", action="store_true", default=True)
    parser.add_argument("--no_recovery_distinct_docs", dest="recovery_distinct_docs", action="store_false")
    parser.add_argument("--max_examples", type=int)
    parser.add_argument("--output_file")
    parser.add_argument("--only_supporting", action="store_true")
    parser.add_argument("--cot", action="store_true", default=False)
    parser.add_argument("--cut_length", type=int)
    parser.add_argument("--max_doc_num", type=int, default=0)
    parser.add_argument("--max_doc_length", type=int, default=2048)
    parser.add_argument("--max_context_tokens", type=int, default=0)
    parser.add_argument("--max_query_tokens", type=int, default=1024)
    parser.add_argument("--doc_selection", choices=["head", "tail"], default="head")
    parser.add_argument("--device_type", choices=["auto", "cuda", "npu", "cpu"], default="auto")
    parser.add_argument("--system_attn_impl", default="eager")
    parser.add_argument("--gist_attn_impl", default="npu_fusion_attention")
    parser.add_argument("--generate_attn_impl", default="npu_fusion_attention")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--override_system_prompt", action="store_true", default=True)
    parser.add_argument("--no_override_system_prompt", dest="override_system_prompt", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(evaluate(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
