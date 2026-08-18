from __future__ import annotations

import argparse
import json
import math
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


def _bm25_scores(query: str, texts: Sequence[str]) -> List[float]:
    tokenized_docs = [_text_tokens(text) for text in texts]
    query_terms = _text_tokens(query)
    if not tokenized_docs or not query_terms:
        return [0.0 for _ in texts]

    doc_freq: Dict[str, int] = {}
    for tokens in tokenized_docs:
        for token in set(tokens):
            doc_freq[token] = doc_freq.get(token, 0) + 1

    num_docs = len(tokenized_docs)
    avg_len = sum(len(tokens) for tokens in tokenized_docs) / max(1, num_docs)
    k1 = 1.5
    b = 0.75
    scores: List[float] = []
    for tokens in tokenized_docs:
        if not tokens:
            scores.append(0.0)
            continue
        counts: Dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        doc_len = len(tokens)
        score = 0.0
        for term in query_terms:
            tf = counts.get(term, 0)
            if tf <= 0:
                continue
            df = doc_freq.get(term, 0)
            idf = math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))
            denom = tf + k1 * (1.0 - b + b * doc_len / max(avg_len, 1e-6))
            score += idf * (tf * (k1 + 1.0)) / max(denom, 1e-6)
        scores.append(score)
    return scores


def _clear_device_cache(device: Any) -> None:
    device_type = getattr(device, "type", str(device)).split(":")[0]
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device_type == "npu" and hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.empty_cache()


def _gist_spans_from_lengths(lengths: Sequence[int], gist_tokens: int) -> List[tuple[int, int]]:
    total = sum(max(0, int(length)) for length in lengths)
    if total <= 0 or gist_tokens <= 0:
        return [(0, 0) for _ in lengths]
    spans: List[tuple[int, int]] = []
    cursor = 0
    for length in lengths:
        start = int(cursor * gist_tokens / total)
        cursor += max(0, int(length))
        end = int((cursor * gist_tokens + total - 1) / total)
        if end <= start and length > 0:
            end = min(gist_tokens, start + 1)
        spans.append((max(0, start), min(gist_tokens, end)))
    return spans


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


def _build_chunk_candidates(
    question: str,
    doc_ids: Sequence[Sequence[int]],
    documents: Sequence[str],
    tokenizer: Any,
    chunk_tokens: int,
    chunk_overlap: int,
) -> tuple[List[Dict[str, Any]], List[str], Dict[str, int]]:
    ranking = _lexical_ranking(question, documents)
    rank_by_doc = {doc_index: rank for rank, doc_index in enumerate(ranking, start=1)}

    chunk_tokens = max(1, int(chunk_tokens))
    chunk_overlap = max(0, min(int(chunk_overlap), chunk_tokens - 1))
    stride = max(1, chunk_tokens - chunk_overlap)

    candidates: List[Dict[str, Any]] = []
    chunk_texts: List[str] = []
    for doc_index, ids in enumerate(doc_ids):
        ids = list(ids)
        if not ids:
            continue
        chunk_id = 0
        for start in range(0, len(ids), stride):
            end = min(len(ids), start + chunk_tokens)
            if end <= start:
                continue
            token_ids = ids[start:end]
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            chunk_texts.append(text)
            candidates.append({
                "doc_index": doc_index,
                "doc_rank": rank_by_doc.get(doc_index, len(doc_ids)),
                "chunk_id": chunk_id,
                "start_token": start,
                "end_token": end,
                "token_count": len(token_ids),
                "score": 0.0,
                "token_ids": token_ids,
            })
            chunk_id += 1
            if end == len(ids):
                break
    return candidates, chunk_texts, {"chunk_tokens": chunk_tokens, "chunk_overlap": chunk_overlap}


@torch.inference_mode()
def _rank_chunks_by_attention(
    model: Any,
    tokenizer: Any,
    question: str,
    candidates: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    *,
    cache_mode: str,
) -> tuple[List[Dict[str, Any]], float]:
    if not candidates:
        return [], 0.0

    system_prompt = "You are a helpful assistant."
    system_inputs = tokenize_for_reuse(tokenizer, [system_prompt], keep_bos=True, role="system")
    system_ids = system_inputs.input_ids.to(model.device)
    system_cache, system_length, system_prefill_sec = _prefill_system(
        model,
        system_ids,
        args.system_attn_impl,
    )

    start_time = time.perf_counter()
    prefix_cache = system_cache
    logical_length = system_length
    chunk_key_tokens = 0
    spans: List[tuple[int, int]] = []
    prefill_sec = 0.0
    blend_sec = 0.0
    use_gist_for_query = False

    normalized_cache_mode = cache_mode.lower()
    if normalized_cache_mode == "fullkv":
        for candidate in candidates:
            token_ids = candidate["token_ids"]
            input_ids = _input_ids_from_tokens(token_ids, model.device)
            before_length = prefix_cache.get_seq_length()
            prefix_cache, added_length, elapsed = _prefill_tokens_with_cache(
                model,
                input_ids,
                prefix_cache,
                logical_length,
                args.attention_router_attn_impl,
            )
            logical_length += added_length
            spans.append((before_length - system_length, before_length - system_length + added_length))
            chunk_key_tokens += added_length
            prefill_sec += elapsed
    elif normalized_cache_mode == "c2kv":
        max_chunk_len = max(len(candidate["token_ids"]) for candidate in candidates)
        pad_len = max(1, max_chunk_len)
        context_input_ids = torch.tensor(
            [_pad(candidate["token_ids"], pad_len) for candidate in candidates],
            dtype=torch.long,
        )
        (
            prefix_cache,
            original_length,
            gist_tokens,
            _actual_ratio,
            compress_sec,
            doc_blend_sec,
        ) = _build_tool_cache(
            model,
            context_input_ids,
            prefix_cache,
            logical_length,
            args.gist_attn_impl,
            args.override_ratio,
        )
        if gist_tokens <= 0:
            raise RuntimeError("attention_chunk_router_empty_gist")
        logical_length += original_length
        spans = _gist_spans_from_lengths(
            [len(candidate["token_ids"]) for candidate in candidates],
            gist_tokens,
        )
        chunk_key_tokens = gist_tokens
        prefill_sec += compress_sec
        blend_sec += doc_blend_sec
        use_gist_for_query = True
    else:
        raise ValueError(f"Unsupported attention chunk cache mode={cache_mode!r}")

    query_inputs = tokenize_for_reuse(
        tokenizer,
        [question],
        keep_bos=False,
        role="user",
        add_generation_prompt=True,
    )
    query_ids = query_inputs.input_ids.to(model.device)
    if args.attention_router_max_query_tokens and query_ids.shape[1] > args.attention_router_max_query_tokens:
        query_ids = query_ids[:, -args.attention_router_max_query_tokens :]
    query_len = query_ids.shape[1]
    attention_mask = torch.ones(
        (1, prefix_cache.get_seq_length() + query_len),
        dtype=torch.long,
        device=model.device,
    )
    position_ids = torch.arange(
        logical_length,
        logical_length + query_len,
        dtype=torch.long,
        device=model.device,
    ).unsqueeze(0)

    layer_scores: List[List[float]] = []

    def _score_chunk_attention(chunk_attn: torch.Tensor, span_len: int) -> float:
        if args.attention_router_score_mode == "sum":
            score = chunk_attn.sum(dim=-1).mean()
        elif args.attention_router_score_mode == "sqrt_len":
            score = chunk_attn.sum(dim=-1).mean() / (span_len ** 0.5)
        elif args.attention_router_score_mode == "top4_mean":
            flat = chunk_attn.reshape(-1)
            top_n = min(max(1, args.attention_router_span_top_tokens), flat.numel())
            score = torch.topk(flat, top_n).values.mean()
        else:
            score = chunk_attn.mean()
        return float(score.item())

    def make_hook(_layer_index: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            attn_weights = output[1] if isinstance(output, tuple) and len(output) > 1 else None
            if attn_weights is None:
                return
            cache_attn = attn_weights[0, :, :, system_length : system_length + chunk_key_tokens].float()
            head_scores: List[List[float]] = []
            for head_index in range(cache_attn.shape[0]):
                head_attn = cache_attn[head_index]
                scores: List[float] = []
                for span_start, span_end in spans:
                    if span_end <= span_start:
                        scores.append(0.0)
                    else:
                        scores.append(_score_chunk_attention(head_attn[:, span_start:span_end], span_end - span_start))
                head_scores.append(scores)
            if head_scores:
                layer_scores.append([
                    sum(scores[index] for scores in head_scores) / len(head_scores)
                    for index in range(len(candidates))
                ])
        return hook

    num_layers = len(model.model.layers)
    last_layers = max(1, min(args.attention_router_layers, num_layers))
    handles = [
        model.model.layers[index].self_attn.register_forward_hook(make_hook(index))
        for index in range(num_layers - last_layers, num_layers)
    ]
    original_attn_impl = model.model.config._attn_implementation
    model.model.config._attn_implementation = args.attention_router_attn_impl
    try:
        forward_kwargs = {
            "input_ids": query_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": prefix_cache,
            "use_cache": True,
            "output_attentions": True,
            "logits_to_keep": 1,
        }
        if use_gist_for_query:
            forward_kwargs["use_gist"] = True
        _sync_device(model.device)
        model(**forward_kwargs)
        _sync_device(model.device)
    finally:
        model.model.config._attn_implementation = original_attn_impl
        for handle in handles:
            handle.remove()
        prefix_cache = None
        system_cache = None
        _clear_device_cache(model.device)

    if not layer_scores:
        raise RuntimeError(
            "Attention chunk router did not capture attention weights. Try --attention_router_attn_impl eager."
        )

    scores = [
        sum(layer[index] for layer in layer_scores) / len(layer_scores)
        for index in range(len(candidates))
    ]
    ranked_candidates: List[Dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        item = dict(candidate)
        item["score"] = scores[index]
        item["ranker"] = f"attention_{normalized_cache_mode}"
        item["attention_router_layers"] = last_layers
        item["attention_score_mode"] = args.attention_router_score_mode
        ranked_candidates.append(item)
    ranked_candidates.sort(
        key=lambda item: (
            item["score"],
            -item["doc_rank"],
            -item["token_count"],
            -item["start_token"],
        ),
        reverse=True,
    )
    return ranked_candidates, system_prefill_sec + (time.perf_counter() - start_time)


def _chunk_recovery_candidates(
    question: str,
    doc_ids: Sequence[Sequence[int]],
    documents: Sequence[str],
    tokenizer: Any,
    chunk_tokens: int,
    chunk_overlap: int,
    ranker: str,
    model: Any | None = None,
    args: argparse.Namespace | None = None,
) -> tuple[List[Dict[str, Any]], float]:
    candidates, chunk_texts, _ = _build_chunk_candidates(
        question,
        doc_ids,
        documents,
        tokenizer,
        chunk_tokens,
        chunk_overlap,
    )
    query_counts: Dict[str, int] = {}
    for token in _text_tokens(question):
        query_counts[token] = query_counts.get(token, 0) + 1
    ranker = ranker.lower()
    if ranker == "bm25":
        scores = _bm25_scores(question, chunk_texts)
    elif ranker in {"lexical", "overlap"}:
        scores = [
            _score_token_ids(query_counts, tokenizer, candidate["token_ids"])
            for candidate in candidates
        ]
    elif ranker in {"attention_fullkv", "att_fullkv"}:
        if model is None or args is None:
            raise ValueError("attention_fullkv ranker requires model and args")
        return _rank_chunks_by_attention(
            model,
            tokenizer,
            question,
            candidates,
            args,
            cache_mode="fullkv",
        )
    elif ranker in {"attention_c2kv", "att_c2kv", "attention_compressedkv", "att_compressedkv"}:
        if model is None or args is None:
            raise ValueError("attention_c2kv ranker requires model and args")
        return _rank_chunks_by_attention(
            model,
            tokenizer,
            question,
            candidates,
            args,
            cache_mode="c2kv",
        )
    else:
        raise ValueError(f"Unsupported hybrid_chunk_ranker={ranker!r}")
    for candidate, score in zip(candidates, scores):
        candidate["score"] = score
        candidate["ranker"] = ranker
    candidates.sort(
        key=lambda item: (
            item["score"],
            -item["doc_rank"],
            -item["token_count"],
            -item["start_token"],
        ),
        reverse=True,
    )
    return candidates, 0.0


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
    elif args.mode == "chunk_hybrid":
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
    attention_router_sec = 0.0

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

    if args.mode == "chunk_hybrid":
        candidates, attention_router_sec = _chunk_recovery_candidates(
            example["question"],
            doc_ids,
            documents,
            tokenizer,
            args.hybrid_chunk_tokens,
            args.hybrid_chunk_overlap,
            args.hybrid_chunk_ranker,
            model,
            args,
        )
        used_ranges: Dict[int, List[tuple[int, int]]] = {}
        for candidate in candidates:
            doc_index = int(candidate["doc_index"])
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
            logical_length += added_length
            full_doc_tokens += added_length
            recovery_span_tokens += added_length
            recovery_span_count += 1
            full_prefill_sec += elapsed
            existing.append((start, end))
            recovery_spans.append({
                key: value
                for key, value in candidate.items()
                if key != "token_ids"
            })
            if recovery_span_count >= args.hybrid_chunk_top_k:
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
        "hybrid_chunk_top_k": args.hybrid_chunk_top_k if args.mode == "chunk_hybrid" else None,
        "hybrid_chunk_tokens": args.hybrid_chunk_tokens if args.mode == "chunk_hybrid" else None,
        "hybrid_chunk_overlap": args.hybrid_chunk_overlap if args.mode == "chunk_hybrid" else None,
        "hybrid_chunk_ranker": args.hybrid_chunk_ranker if args.mode == "chunk_hybrid" else None,
        "attention_router_sec": attention_router_sec if args.mode == "chunk_hybrid" else 0.0,
        "attention_router_layers": args.attention_router_layers if args.mode == "chunk_hybrid" else None,
        "attention_router_attn_impl": args.attention_router_attn_impl if args.mode == "chunk_hybrid" else None,
        "attention_router_score_mode": args.attention_router_score_mode if args.mode == "chunk_hybrid" else None,
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
        "avg_attention_router_sec": sum(row.get("attention_router_sec", 0.0) for row in valid) / len(valid) if valid else 0.0,
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
    parser.add_argument("--mode", choices=["full", "c2kv", "hybrid", "rank_plan", "budget_recovery", "chunk_hybrid"], required=True)
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
    parser.add_argument("--hybrid_chunk_top_k", type=int, default=8)
    parser.add_argument("--hybrid_chunk_tokens", type=int, default=256)
    parser.add_argument("--hybrid_chunk_overlap", type=int, default=64)
    parser.add_argument(
        "--hybrid_chunk_ranker",
        choices=[
            "lexical",
            "bm25",
            "attention_fullkv",
            "att_fullkv",
            "attention_c2kv",
            "att_c2kv",
            "attention_compressedkv",
            "att_compressedkv",
        ],
        default="lexical",
    )
    parser.add_argument("--attention_router_layers", type=int, default=4)
    parser.add_argument("--attention_router_attn_impl", default="eager")
    parser.add_argument("--attention_router_max_query_tokens", type=int, default=512)
    parser.add_argument(
        "--attention_router_score_mode",
        choices=["mean", "sum", "sqrt_len", "top4_mean"],
        default="top4_mean",
    )
    parser.add_argument("--attention_router_span_top_tokens", type=int, default=4)
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
