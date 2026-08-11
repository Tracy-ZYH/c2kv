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


def _lexical_topk(question: str, documents: Sequence[str], top_k: int) -> set[int]:
    if top_k <= 0:
        return set()
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
    return {index for _, _, index in scored[: min(top_k, len(scored))]}


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
    if args.mode == "full":
        full_indices = set(range(len(doc_ids)))
    elif args.mode == "hybrid":
        full_indices = _lexical_topk(example["question"], documents, args.hybrid_top_k)
    elif args.mode == "c2kv":
        full_indices = set()
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

    for index, ids in enumerate(doc_ids):
        if index in full_indices:
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
            args.override_ratio,
        )
        logical_length += original_length
        c2kv_doc_tokens += original_length
        gist_tokens += max(doc_gist_tokens, max(0, prefix_cache.get_seq_length() - before_cache_len))
        compress_sec += doc_compress_sec
        blend_sec += doc_blend_sec
        c2kv_doc_count += 1

    compressed_context_tokens = full_doc_tokens + gist_tokens
    actual_ratio = doc_tokens / compressed_context_tokens if compressed_context_tokens else 0.0
    return {
        "cache": prefix_cache,
        "logical_length": logical_length,
        "cache_length": prefix_cache.get_seq_length(),
        "doc_tokens": doc_tokens,
        "doc_chunks": len(doc_ids),
        "full_doc_count": len(full_indices),
        "c2kv_doc_count": c2kv_doc_count,
        "full_doc_tokens": full_doc_tokens,
        "c2kv_doc_tokens": c2kv_doc_tokens,
        "gist_tokens": gist_tokens,
        "compressed_context_tokens": compressed_context_tokens,
        "actual_compression_ratio": actual_ratio,
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
        "avg_gist_tokens": sum(row.get("gist_tokens", 0) for row in valid) / len(valid) if valid else 0.0,
        "avg_compressed_context_tokens": compressed_total / len(valid) if valid else 0.0,
        "avg_actual_compression_ratio": sum(row.get("actual_compression_ratio", 0.0) for row in valid) / len(valid) if valid else 0.0,
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
    num_examples = len(dataset) if args.max_examples is None else min(args.max_examples, len(dataset))
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
    parser.add_argument("--mode", choices=["full", "c2kv", "hybrid"], required=True)
    parser.add_argument("--override_ratio", type=int, default=16)
    parser.add_argument("--hybrid_top_k", type=int, default=3)
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
