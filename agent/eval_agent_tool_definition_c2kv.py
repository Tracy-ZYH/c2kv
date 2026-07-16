from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from models import blend_gist_key_values, get_model_class  # noqa: E402
from train.train_data_multiturn import _chat_template_ids, _pad  # noqa: E402
from train_agent_tool_definition_c2kv import (  # noqa: E402
    AgentLLMTracesSource,
    AgentToolDefinitionDataArgs,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _sync_device(device: Any) -> None:
    device_type = getattr(device, "type", str(device))
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device_type == "npu" and hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()


def _setup_device(device_type: str) -> str:
    if device_type == "auto":
        if hasattr(torch, "npu") and torch.npu.is_available():
            device_type = "npu"
        elif torch.cuda.is_available():
            device_type = "cuda"
        else:
            device_type = "cpu"
    if device_type == "npu":
        import torch_npu  # noqa: F401

        torch.npu.set_device(0)
    elif device_type == "cuda":
        torch.cuda.set_device(0)
    return device_type


def _jsonl_write(path: str, rows: List[Dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _extract_tool_name(text: str) -> Optional[str]:
    if not text:
        return None
    blocks = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, flags=re.S)
    candidates = blocks or [text]
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except Exception:
            value = None
        if isinstance(value, dict):
            function = value.get("function") if isinstance(value.get("function"), dict) else {}
            name = (
                value.get("name")
                or value.get("tool_name")
                or value.get("function_name")
                or function.get("name")
            )
            if name:
                return str(name)
    match = re.search(r'"(?:name|tool_name|function_name)"\s*:\s*"([^"]+)"', text)
    if match:
        return match.group(1)
    match = re.search(r"<tool_call>.*?([A-Za-z0-9_.:-]+).*?</tool_call>", text, flags=re.S)
    if match:
        return match.group(1)
    return None


def _build_tool_chunks(
    tokenizer: Any,
    tool_definition: str,
    max_doc_length: int,
    max_doc_num: int,
    max_tool_definition_tokens: int,
    truncate_tool_definition: bool,
) -> tuple[Optional[torch.Tensor], int, int, Optional[str]]:
    tool_doc = {"role": "user", "content": "Tool definitions:\n" + tool_definition}
    doc_ids = _chat_template_ids(tokenizer, [tool_doc])
    if len(doc_ids) > max_tool_definition_tokens:
        return None, len(doc_ids), 0, f"tool_definition_tokens>{max_tool_definition_tokens}"
    max_context_tokens = max_doc_length * max_doc_num
    if len(doc_ids) > max_context_tokens:
        if not truncate_tool_definition:
            return None, len(doc_ids), 0, f"tool_definition_tokens>{max_context_tokens}"
        doc_ids = doc_ids[:max_context_tokens]
    doc_chunks = [
        doc_ids[start : start + max_doc_length]
        for start in range(0, len(doc_ids), max_doc_length)
    ]
    if len(doc_chunks) > max_doc_num:
        return None, len(doc_ids), 0, f"tool_definition_docs>{max_doc_num}"
    rows = []
    for chunk in doc_chunks:
        rows.append(_pad(chunk, max_doc_length, -100))
    if not rows:
        return None, len(doc_ids), 0, "empty_tool_definition"
    return torch.tensor(rows, dtype=torch.long), len(doc_ids), len(rows), None


def _tool_doc_ids(tokenizer: Any, tool_definition: str) -> List[int]:
    tool_doc = {"role": "user", "content": "Tool definitions:\n" + tool_definition}
    return _chat_template_ids(tokenizer, [tool_doc])


@torch.inference_mode()
def _prefill_system(model: Any, input_ids: torch.Tensor, attn_impl: str) -> tuple[Any, int, float]:
    original_attn_impl = model.model.config._attn_implementation
    model.model.config._attn_implementation = attn_impl
    attention_mask = torch.ones_like(input_ids)
    _sync_device(input_ids.device)
    start = time.perf_counter()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        logits_to_keep=1,
    )
    _sync_device(input_ids.device)
    elapsed = time.perf_counter() - start
    model.model.config._attn_implementation = original_attn_impl
    return outputs.past_key_values, int(attention_mask.sum().item()), elapsed


@torch.inference_mode()
def _prefill_tokens_with_cache(
    model: Any,
    input_ids: torch.Tensor,
    past_key_values: Any,
    past_length: int,
    attn_impl: str,
) -> tuple[Any, int, float]:
    if input_ids.shape[1] == 0:
        return past_key_values, 0, 0.0
    original_attn_impl = model.model.config._attn_implementation
    model.model.config._attn_implementation = attn_impl
    input_length = input_ids.shape[1]
    attention_mask = torch.ones(
        (input_ids.shape[0], past_length + input_length),
        dtype=torch.long,
        device=input_ids.device,
    )
    position_ids = torch.arange(
        past_length,
        past_length + input_length,
        dtype=torch.long,
        device=input_ids.device,
    ).unsqueeze(0)
    _sync_device(input_ids.device)
    start = time.perf_counter()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=True,
        logits_to_keep=1,
    )
    _sync_device(input_ids.device)
    elapsed = time.perf_counter() - start
    model.model.config._attn_implementation = original_attn_impl
    return outputs.past_key_values, input_length, elapsed


@torch.inference_mode()
def _build_tool_cache(
    model: Any,
    context_input_ids: torch.Tensor,
    system_cache: Any,
    system_length: int,
    attn_impl: str,
    override_ratio: int,
) -> tuple[Any, int, int, float, float, float]:
    device = model.device
    context_input_ids = context_input_ids.to(device)
    valid_mask = context_input_ids != -100
    tool_length = int(valid_mask.sum().item())
    input_ids = context_input_ids.clone()
    input_ids[~valid_mask] = model.model.gist_token_id

    original_attn_impl = model.model.config._attn_implementation
    model.model.config._attn_implementation = attn_impl
    gist_kwargs = {}
    if getattr(model.config, "gist_type", None) == "dynamic-interleave":
        gist_kwargs["ratio"] = override_ratio
    _sync_device(input_ids.device)
    compress_start = time.perf_counter()
    outputs, gist_mask, pos_ids = model.model.generate_gist(
        input_ids=input_ids,
        attention_mask=valid_mask,
        **gist_kwargs,
    )
    _sync_device(input_ids.device)
    tool_compress_sec = time.perf_counter() - compress_start
    model.model.config._attn_implementation = original_attn_impl

    _sync_device(input_ids.device)
    blend_start = time.perf_counter()
    tool_cache, _ = blend_gist_key_values(
        model.config,
        [outputs.past_key_values],
        [gist_mask],
        [pos_ids],
        model.model.rotary_emb,
        system_length,
    )
    for system_layer, tool_layer in zip(system_cache.layers, tool_cache.layers):
        tool_layer.keys = torch.cat([system_layer.keys, tool_layer.keys], dim=-2)
        tool_layer.values = torch.cat([system_layer.values, tool_layer.values], dim=-2)
    _sync_device(input_ids.device)
    blend_sec = time.perf_counter() - blend_start
    gist_tokens = max(0, tool_cache.get_seq_length() - system_length)
    actual_ratio = float(tool_length / gist_tokens) if gist_tokens else 0.0
    return tool_cache, tool_length, gist_tokens, actual_ratio, tool_compress_sec, blend_sec


@torch.inference_mode()
def _generate_from_input_ids(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    attn_impl: str,
    use_gist: bool = False,
    position_ids: Optional[torch.Tensor] = None,
    past_key_values: Any = None,
) -> tuple[str, float, int, float]:
    original_attn_impl = model.model.config._attn_implementation if hasattr(model, "model") else None
    if original_attn_impl is not None:
        model.model.config._attn_implementation = attn_impl
    attention_mask = torch.ones_like(input_ids)
    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if position_ids is not None:
        generate_kwargs["position_ids"] = position_ids
    if past_key_values is not None:
        generate_kwargs["past_key_values"] = past_key_values
    if use_gist:
        generate_kwargs["use_gist"] = True
    _sync_device(input_ids.device)
    start = time.perf_counter()
    generated = model.generate(**generate_kwargs)
    _sync_device(input_ids.device)
    latency = time.perf_counter() - start
    if original_attn_impl is not None:
        model.model.config._attn_implementation = original_attn_impl
    prediction = tokenizer.decode(
        generated[0, input_ids.shape[1] :],
        skip_special_tokens=True,
    ).strip()
    generated_tokens = int(generated.shape[1] - input_ids.shape[1])
    tbt_sec = latency / generated_tokens if generated_tokens > 0 else 0.0
    return prediction, latency, generated_tokens, tbt_sec


@torch.inference_mode()
def _generate_one(
    model: Any,
    tokenizer: Any,
    example: Any,
    args: argparse.Namespace,
    device: str,
) -> Dict[str, Any]:
    total_start = time.perf_counter()
    if args.mode in ("truncate", "full"):
        return _generate_one_baseline(model, tokenizer, example, args)

    system_ids = _chat_template_ids(
        tokenizer,
        [{"role": "system", "content": example.system_prompt}],
        keep_bos=True,
        max_length=args.max_system_length,
    )
    system_input_ids = torch.tensor([system_ids], dtype=torch.long, device=model.device)
    system_cache, system_length, system_prefill_sec = _prefill_system(
        model, system_input_ids, args.system_attn_impl
    )

    context_input_ids, doc_tokens, doc_chunks, skip_reason = _build_tool_chunks(
        tokenizer,
        example.tool_definition,
        max_doc_length=args.max_doc_length,
        max_doc_num=args.max_doc_num,
        max_tool_definition_tokens=args.max_tool_definition_tokens,
        truncate_tool_definition=args.truncate_tool_definition,
    )
    if context_input_ids is None:
        return {
            "qid": example.qid,
            "session_id": example.session_id,
            "skipped": True,
            "skip_reason": skip_reason,
            "doc_tokens": doc_tokens,
        }

    (
        tool_cache,
        tool_length,
        gist_tokens,
        actual_ratio,
        tool_compress_sec,
        blend_sec,
    ) = _build_tool_cache(
        model,
        context_input_ids,
        system_cache,
        system_length,
        args.gist_attn_impl,
        args.override_ratio,
    )

    prompt_ids = _chat_template_ids(
        tokenizer,
        example.input_messages,
        add_generation_prompt=True,
    )
    if args.max_prompt_tokens and len(prompt_ids) > args.max_prompt_tokens:
        prompt_ids = prompt_ids[-args.max_prompt_tokens :]
    prompt_input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
    prompt_length = prompt_input_ids.shape[1]
    cache_length = tool_cache.get_seq_length()
    mock_cache_ids = prompt_input_ids.new_zeros((1, cache_length))
    input_ids = torch.cat([mock_cache_ids, prompt_input_ids], dim=1)
    attention_mask = torch.ones_like(input_ids)
    original_prefix_length = system_length + tool_length
    position_ids = torch.arange(
        original_prefix_length,
        original_prefix_length + prompt_length,
        dtype=torch.long,
        device=model.device,
    ).unsqueeze(0)

    prediction, latency, generated_tokens, tbt_sec = _generate_from_input_ids(
        model,
        tokenizer,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        attn_impl=args.generate_attn_impl,
        use_gist=True,
        position_ids=position_ids,
        past_key_values=tool_cache,
    )
    target = example.answer.strip()
    target_tool = _extract_tool_name(target)
    pred_tool = _extract_tool_name(prediction)
    total_sec = time.perf_counter() - total_start
    ttft_sec = system_prefill_sec + tool_compress_sec + blend_sec
    return {
        "qid": example.qid,
        "session_id": example.session_id,
        "mode": args.mode,
        "ratio": args.override_ratio,
        "skipped": False,
        "doc_tokens": doc_tokens,
        "doc_chunks": doc_chunks,
        "gist_tokens": gist_tokens,
        "actual_compression_ratio": round(actual_ratio, 4),
        "prompt_tokens": prompt_length,
        "target_tokens": len(tokenizer.encode(target, add_special_tokens=False)),
        "generated_tokens": generated_tokens,
        "latency_sec": round(latency, 4),
        "system_prefill_sec": round(system_prefill_sec, 4),
        "tool_compress_sec": round(tool_compress_sec, 4),
        "full_prefill_sec": 0.0,
        "blend_sec": round(blend_sec, 4),
        "generate_sec": round(latency, 4),
        "ttft_sec": round(ttft_sec, 4),
        "tbt_sec": round(tbt_sec, 6),
        "total_sec": round(total_sec, 4),
        "target_tool_name": target_tool,
        "prediction_tool_name": pred_tool,
        "tool_name_match": target_tool is not None and target_tool == pred_tool,
        "has_tool_call": "<tool_call>" in prediction or "Action:" in prediction,
        "exact_match": _normalize_text(prediction) == _normalize_text(target),
        "prediction": prediction,
        "target": target,
    }


@torch.inference_mode()
def _generate_one_baseline(
    model: Any,
    tokenizer: Any,
    example: Any,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    total_start = time.perf_counter()
    system_ids = _chat_template_ids(
        tokenizer,
        [{"role": "system", "content": example.system_prompt}],
        keep_bos=True,
        max_length=args.max_system_length,
    )
    doc_ids = _tool_doc_ids(tokenizer, example.tool_definition)
    doc_tokens = len(doc_ids)
    if args.mode == "truncate":
        kept_tool_tokens = max(1, (doc_tokens + args.override_ratio - 1) // args.override_ratio)
        used_doc_ids = doc_ids[:kept_tool_tokens]
    else:
        kept_tool_tokens = doc_tokens
        used_doc_ids = doc_ids

    prompt_ids = _chat_template_ids(
        tokenizer,
        example.input_messages,
        add_generation_prompt=True,
    )
    if args.max_prompt_tokens and len(prompt_ids) > args.max_prompt_tokens:
        prompt_ids = prompt_ids[-args.max_prompt_tokens :]
    total_len = len(system_ids) + len(used_doc_ids) + len(prompt_ids)
    if args.max_baseline_input_tokens is not None and args.max_baseline_input_tokens > 0 and total_len > args.max_baseline_input_tokens:
        return {
            "qid": example.qid,
            "session_id": example.session_id,
            "mode": args.mode,
            "ratio": args.override_ratio,
            "skipped": True,
            "skip_reason": f"baseline_input_tokens>{args.max_baseline_input_tokens}",
            "doc_tokens": doc_tokens,
            "kept_tool_tokens": kept_tool_tokens,
            "input_tokens": total_len,
        }

    system_input_ids = torch.tensor([system_ids], dtype=torch.long, device=model.device)
    system_cache, system_length, system_prefill_sec = _prefill_system(
        model, system_input_ids, args.system_attn_impl
    )
    tool_input_ids = torch.tensor([used_doc_ids], dtype=torch.long, device=model.device)
    tool_cache, tool_length, full_prefill_sec = _prefill_tokens_with_cache(
        model,
        tool_input_ids,
        past_key_values=system_cache,
        past_length=system_length,
        attn_impl=args.generate_attn_impl,
    )

    prompt_input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
    cache_length = tool_cache.get_seq_length()
    mock_cache_ids = prompt_input_ids.new_zeros((1, cache_length))
    input_ids = torch.cat([mock_cache_ids, prompt_input_ids], dim=1)
    position_ids = torch.arange(
        system_length + tool_length,
        system_length + tool_length + prompt_input_ids.shape[1],
        dtype=torch.long,
        device=model.device,
    ).unsqueeze(0)
    prediction, latency, generated_tokens, tbt_sec = _generate_from_input_ids(
        model,
        tokenizer,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        attn_impl=args.generate_attn_impl,
        use_gist=False,
        position_ids=position_ids,
        past_key_values=tool_cache,
    )
    target = example.answer.strip()
    target_tool = _extract_tool_name(target)
    pred_tool = _extract_tool_name(prediction)
    total_sec = time.perf_counter() - total_start
    ttft_sec = system_prefill_sec + full_prefill_sec
    return {
        "qid": example.qid,
        "session_id": example.session_id,
        "mode": args.mode,
        "ratio": args.override_ratio,
        "skipped": False,
        "doc_tokens": doc_tokens,
        "kept_tool_tokens": kept_tool_tokens,
        "actual_compression_ratio": round(doc_tokens / kept_tool_tokens, 4) if kept_tool_tokens else 0.0,
        "prompt_tokens": len(prompt_ids),
        "input_tokens": total_len,
        "target_tokens": len(tokenizer.encode(target, add_special_tokens=False)),
        "generated_tokens": generated_tokens,
        "latency_sec": round(latency, 4),
        "system_prefill_sec": round(system_prefill_sec, 4),
        "tool_compress_sec": 0.0,
        "full_prefill_sec": round(full_prefill_sec, 4),
        "blend_sec": 0.0,
        "generate_sec": round(latency, 4),
        "ttft_sec": round(ttft_sec, 4),
        "tbt_sec": round(tbt_sec, 6),
        "total_sec": round(total_sec, 4),
        "target_tool_name": target_tool,
        "prediction_tool_name": pred_tool,
        "tool_name_match": target_tool is not None and target_tool == pred_tool,
        "has_tool_call": "<tool_call>" in prediction or "Action:" in prediction,
        "exact_match": _normalize_text(prediction) == _normalize_text(target),
        "prediction": prediction,
        "target": target,
    }


def _summarize_rows(args: argparse.Namespace, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries = []
    keys = sorted({(row.get("mode"), row.get("ratio")) for row in rows})
    for mode, ratio in keys:
        group = [row for row in rows if row.get("mode") == mode and row.get("ratio") == ratio]
        valid_rows = [row for row in group if not row.get("skipped")]
        skip_reasons = Counter(row.get("skip_reason", "unknown") for row in group if row.get("skipped"))
        summaries.append({
            "model": args.model,
            "dataset_path": args.dataset_path,
            "split": args.split,
            "mode": mode,
            "ratio": ratio,
            "num_examples": len(group),
            "num_valid": len(valid_rows),
            "num_skipped": len(group) - len(valid_rows),
            "skip_reasons": dict(skip_reasons),
            "exact_match": (
                sum(1 for row in valid_rows if row["exact_match"]) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "tool_name_accuracy": (
                sum(1 for row in valid_rows if row["tool_name_match"]) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "tool_call_rate": (
                sum(1 for row in valid_rows if row["has_tool_call"]) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_latency_sec": (
                sum(row["latency_sec"] for row in valid_rows) / len(valid_rows)
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
            "avg_generate_sec": (
                sum(row.get("generate_sec", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_generated_tokens": (
                sum(row.get("generated_tokens", 0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_ttft_sec": (
                sum(row.get("ttft_sec", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_tbt_sec": (
                sum(row.get("tbt_sec", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "token_weighted_tbt_sec": (
                sum(row.get("generate_sec", 0.0) for row in valid_rows)
                / sum(row.get("generated_tokens", 0) for row in valid_rows)
                if sum(row.get("generated_tokens", 0) for row in valid_rows) else 0.0
            ),
            "avg_total_sec": (
                sum(row.get("total_sec", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_actual_compression_ratio": (
                sum(row.get("actual_compression_ratio", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
        })
    return summaries


def _load_model(args: argparse.Namespace, tokenizer: Any, device: str) -> Any:
    if args.untrained_c2kv and not args.base_model:
        raise ValueError("--base_model is required for c2kv_untrained baseline")
    model_path = args.base_model if args.untrained_c2kv else args.model
    if args.mode in ("full", "truncate") and args.baseline_model_class == "auto":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            device_map={"": device} if device != "cpu" else None,
            dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float16 if args.dtype == "fp16" else torch.float32,
            attn_implementation=args.generate_attn_impl,
        )
        model.eval()
        return model

    config_class, model_class = get_model_class(model_path, "qkv")
    if args.untrained_c2kv:
        config = config_class.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            gist_type="dynamic-interleave",
            gist_param="qkv",
            gist_residual_type="embed-mean",
            gist_overlap=64,
            gist_token_id=tokenizer.eos_token_id,
            pad_token_id=None,
        )
        model = model_class.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            local_files_only=True,
            device_map={"": device} if device != "cpu" else None,
            dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float16 if args.dtype == "fp16" else torch.float32,
            attn_implementation=args.generate_attn_impl,
        )
    else:
        model = model_class.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            device_map={"": device} if device != "cpu" else None,
            dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float16 if args.dtype == "fp16" else torch.float32,
            attn_implementation=args.generate_attn_impl,
        )
    model.eval()
    return model


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    device = _setup_device(args.device_type)
    tokenizer_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        local_files_only=True,
        padding_side="right",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    data_args = AgentToolDefinitionDataArgs(
        dataset_path=args.dataset_path,
        eval_ratio=args.eval_ratio,
        split_seed=args.split_seed,
        split_manifest_file=args.split_manifest_file,
        split_manifest_name=args.split_manifest_name,
        max_samples_per_session=args.max_samples_per_session,
        max_doc_length=args.max_doc_length,
        max_doc_num=args.max_doc_num,
        max_tool_definition_tokens=args.max_tool_definition_tokens,
        max_length=args.max_length,
        max_system_length=args.max_system_length,
        truncate_tool_definition=args.truncate_tool_definition,
        require_tool_call=args.require_tool_call,
        min_target_tokens=args.min_target_tokens,
    )
    source = AgentLLMTracesSource(data_args)
    source_examples = list(source.iter_examples(args.split))
    if args.max_source_examples is not None:
        source_examples = source_examples[: args.max_source_examples]

    modes = [item.strip() for item in (args.compare_modes or args.mode).split(",") if item.strip()]
    ratios = [int(item.strip()) for item in (args.ratios or str(args.override_ratio)).split(",") if item.strip()]
    selection_skips: Counter[str] = Counter()
    examples = []
    if args.selection_filter == "c2kv":
        for example in source_examples:
            _, _, _, skip_reason = _build_tool_chunks(
                tokenizer,
                example.tool_definition,
                max_doc_length=args.max_doc_length,
                max_doc_num=args.max_doc_num,
                max_tool_definition_tokens=args.max_tool_definition_tokens,
                truncate_tool_definition=args.truncate_tool_definition,
            )
            if skip_reason is not None:
                selection_skips[skip_reason] += 1
                continue
            examples.append(example)
            if args.max_examples is not None and args.max_examples > 0 and len(examples) >= args.max_examples:
                break
    else:
        examples = (
            source_examples[: args.max_examples]
            if args.max_examples is not None and args.max_examples > 0
            else source_examples
        )

    logger.info(
        "Selected %d %s examples from %d source examples; selection_skips=%s",
        len(examples),
        args.split,
        len(source_examples),
        dict(selection_skips),
    )

    rows: List[Dict[str, Any]] = []
    for mode in modes:
        run_ratios = [1] if mode == "full" else ratios
        model_args = copy.copy(args)
        model_args.untrained_c2kv = mode == "c2kv_untrained"
        model_args.mode = "c2kv" if mode == "c2kv_untrained" else mode
        model_args.row_mode = mode
        logger.info("Loading model for mode=%s", mode)
        model = _load_model(model_args, tokenizer, device)
        for ratio in run_ratios:
            run_args = copy.copy(model_args)
            run_args.override_ratio = ratio
            desc = f"{mode}@{ratio}x" if mode != "full" else mode
            for example in tqdm(examples, desc=desc):
                row = _generate_one(model, tokenizer, example, run_args, device)
                row["mode"] = mode
                rows.append(row)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "npu":
            torch.npu.empty_cache()

    summaries = _summarize_rows(args, rows)
    summary = {
        "model": args.model,
        "base_model": args.base_model,
        "dataset_path": args.dataset_path,
        "split": args.split,
        "modes": modes,
        "ratios": ratios,
        "selection_skips": dict(selection_skips),
        "results": summaries,
        "num_examples": len(rows),
    }
    if args.output_file:
        _jsonl_write(args.output_file, rows)
        summary_path = str(Path(args.output_file).with_suffix(".summary.json"))
        Path(summary_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Wrote predictions to %s", args.output_file)
        logger.info("Wrote summary to %s", summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate C2KV tool-definition compression on agent traces.")
    parser.add_argument("--model", required=True, help="Saved C2KV model/checkpoint directory.")
    parser.add_argument("--base_model", help="Base model directory for untrained C2KV baseline.")
    parser.add_argument("--tokenizer", help="Tokenizer directory. Defaults to --model.")
    parser.add_argument("--dataset_path", default="./datasets/agent-llm-traces")
    parser.add_argument("--output_file", default="./outputs/agent_tooldef_c2kv_eval.jsonl")
    parser.add_argument("--split", choices=["train", "eval"], default="eval")
    parser.add_argument("--mode", choices=["c2kv", "c2kv_untrained", "truncate", "full"], default="c2kv")
    parser.add_argument(
        "--compare_modes",
        help="Comma-separated modes to run, e.g. c2kv,c2kv_untrained,truncate.",
    )
    parser.add_argument("--ratios", help="Comma-separated compression ratios for c2kv/truncate, e.g. 2,4,8.")
    parser.add_argument("--max_examples", type=int, default=50, help="Maximum examples; <=0 means all selected examples.")
    parser.add_argument("--max_source_examples", type=int)
    parser.add_argument("--selection_filter", choices=["c2kv", "none"], default="c2kv")
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--split_manifest_file")
    parser.add_argument("--split_manifest_name", default="toolset_disjoint")
    parser.add_argument("--max_samples_per_session", type=int, default=4)
    parser.add_argument("--max_doc_length", type=int, default=1024)
    parser.add_argument("--max_doc_num", type=int, default=10)
    parser.add_argument("--max_tool_definition_tokens", type=int, default=10000)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_system_length", type=int, default=256)
    parser.add_argument("--max_prompt_tokens", type=int, default=1920)
    parser.add_argument("--max_baseline_input_tokens", type=int, default=12000)
    parser.add_argument("--min_target_tokens", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--truncate_tool_definition", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--require_tool_call", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--device_type", choices=["auto", "cuda", "npu", "cpu"], default="auto")
    parser.add_argument("--system_attn_impl", default="eager")
    parser.add_argument("--gist_attn_impl", default="eager")
    parser.add_argument("--generate_attn_impl", default="eager")
    parser.add_argument("--override_ratio", type=int, default=4)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--baseline_model_class", choices=["gist", "auto"], default="gist")
    parser.add_argument("--untrained_c2kv", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
