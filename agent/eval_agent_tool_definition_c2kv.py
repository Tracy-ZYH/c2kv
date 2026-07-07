from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

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


@torch.inference_mode()
def _prefill_system(model: Any, input_ids: torch.Tensor, attn_impl: str) -> tuple[Any, int]:
    original_attn_impl = model.model.config._attn_implementation
    model.model.config._attn_implementation = attn_impl
    attention_mask = torch.ones_like(input_ids)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        logits_to_keep=1,
    )
    model.model.config._attn_implementation = original_attn_impl
    return outputs.past_key_values, int(attention_mask.sum().item())


@torch.inference_mode()
def _build_tool_cache(
    model: Any,
    context_input_ids: torch.Tensor,
    system_cache: Any,
    system_length: int,
    attn_impl: str,
    override_ratio: int,
) -> tuple[Any, int]:
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
    outputs, gist_mask, pos_ids = model.model.generate_gist(
        input_ids=input_ids,
        attention_mask=valid_mask,
        **gist_kwargs,
    )
    model.model.config._attn_implementation = original_attn_impl

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
    return tool_cache, tool_length


@torch.inference_mode()
def _generate_one(
    model: Any,
    tokenizer: Any,
    example: Any,
    args: argparse.Namespace,
    device: str,
) -> Dict[str, Any]:
    system_ids = _chat_template_ids(
        tokenizer,
        [{"role": "system", "content": example.system_prompt}],
        keep_bos=True,
        max_length=args.max_system_length,
    )
    system_input_ids = torch.tensor([system_ids], dtype=torch.long, device=model.device)
    system_cache, system_length = _prefill_system(model, system_input_ids, args.system_attn_impl)

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

    tool_cache, tool_length = _build_tool_cache(
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
    if len(prompt_ids) > args.max_prompt_tokens:
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

    original_attn_impl = model.model.config._attn_implementation
    model.model.config._attn_implementation = args.generate_attn_impl
    start = time.perf_counter()
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=tool_cache,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        use_gist=True,
    )
    latency = time.perf_counter() - start
    model.model.config._attn_implementation = original_attn_impl
    prediction = tokenizer.decode(
        generated[0, input_ids.shape[1] :],
        skip_special_tokens=True,
    ).strip()
    target = example.answer.strip()
    target_tool = _extract_tool_name(target)
    pred_tool = _extract_tool_name(prediction)
    return {
        "qid": example.qid,
        "session_id": example.session_id,
        "skipped": False,
        "doc_tokens": doc_tokens,
        "doc_chunks": doc_chunks,
        "prompt_tokens": prompt_length,
        "target_tokens": len(tokenizer.encode(target, add_special_tokens=False)),
        "latency_sec": round(latency, 4),
        "target_tool_name": target_tool,
        "prediction_tool_name": pred_tool,
        "tool_name_match": target_tool is not None and target_tool == pred_tool,
        "has_tool_call": "<tool_call>" in prediction or "Action:" in prediction,
        "exact_match": _normalize_text(prediction) == _normalize_text(target),
        "prediction": prediction,
        "target": target,
    }


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

    _, model_class = get_model_class(args.model, "qkv")
    model = model_class.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
        device_map={"": device} if device != "cpu" else None,
        dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float16 if args.dtype == "fp16" else torch.float32,
        attn_implementation=args.generate_attn_impl,
    )
    model.eval()

    data_args = AgentToolDefinitionDataArgs(
        dataset_path=args.dataset_path,
        eval_ratio=args.eval_ratio,
        split_seed=args.split_seed,
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
    examples = list(source.iter_examples(args.split))
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    logger.info("Evaluating %d %s examples", len(examples), args.split)

    rows: List[Dict[str, Any]] = []
    for example in tqdm(examples, desc="agent-tooldef-c2kv-eval"):
        rows.append(_generate_one(model, tokenizer, example, args, device))

    valid_rows = [row for row in rows if not row.get("skipped")]
    summary = {
        "model": args.model,
        "dataset_path": args.dataset_path,
        "split": args.split,
        "num_examples": len(rows),
        "num_valid": len(valid_rows),
        "num_skipped": len(rows) - len(valid_rows),
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
    parser.add_argument("--tokenizer", help="Tokenizer directory. Defaults to --model.")
    parser.add_argument("--dataset_path", default="./datasets/agent-llm-traces")
    parser.add_argument("--output_file", default="./outputs/agent_tooldef_c2kv_eval.jsonl")
    parser.add_argument("--split", choices=["train", "eval"], default="eval")
    parser.add_argument("--max_examples", type=int, default=50)
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--max_samples_per_session", type=int, default=4)
    parser.add_argument("--max_doc_length", type=int, default=1024)
    parser.add_argument("--max_doc_num", type=int, default=10)
    parser.add_argument("--max_tool_definition_tokens", type=int, default=10000)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_system_length", type=int, default=256)
    parser.add_argument("--max_prompt_tokens", type=int, default=1920)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
