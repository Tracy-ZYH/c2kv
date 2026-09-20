#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import models.gist_utils as gist_utils  # noqa: E402
from gist_args import ModelArgs  # noqa: E402
from models import format_numel_str, get_model_and_tokenizer  # noqa: E402
from quick_test_agent_c2kv import (  # noqa: E402
    QuickAgentDataset,
    build_system_kv,
    chat_ids,
    c2kv_forward,
    collate,
    extract_tool_call,
    full_forward,
    move_batch,
    pad,
    restore_attn_impl,
    set_attn_impl,
    shifted_target_logits,
    target_ce_loss,
    write_json,
    write_jsonl,
)


def is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def maybe_init_distributed(device: str) -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if device.startswith("cuda") else "gloo"
        dist.init_process_group(backend=backend)
    return rank, local_rank, world_size


def load_fixed_samples(path: str, seed: int) -> List[Dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    rng = random.Random(seed)
    selected: List[Dict[str, Any]] = []
    for source, quota in [("agent_llm", 12), ("toucan", 10), ("hermes", 10)]:
        items = [row for row in rows if row.get("source") == source]
        rng.shuffle(items)
        if len(items) < quota:
            raise RuntimeError(f"Need {quota} {source} samples, found {len(items)} in {path}")
        selected.extend(items[:quota])
    rng.shuffle(selected)
    for i, sample in enumerate(selected):
        sample["final_verify_id"] = i
    return selected


def pair_metrics(model, loader, tokenizer, args) -> Dict[str, float]:
    model.eval()
    total_c2kv_ce = 0.0
    total_full_ce = 0.0
    total_agree = 0.0
    total_kl = 0.0
    total_tokens = 0
    total_batches = 0
    for batch in loader:
        batch = move_batch(batch, args.device)
        with torch.no_grad():
            full_loss, full_logits, full_labels = full_forward(model, batch, args)
            c2kv_loss, c2kv_logits, c2kv_labels = c2kv_forward(model, batch, args.max_doc_length, args)
        common_len = min(full_logits.shape[1], c2kv_logits.shape[1], full_labels.shape[1], c2kv_labels.shape[1])
        full_logits = full_logits[:, :common_len]
        c2kv_logits = c2kv_logits[:, :common_len]
        full_labels = full_labels[:, :common_len]
        c2kv_labels = c2kv_labels[:, :common_len]
        mask = (full_labels != -100) & (c2kv_labels != -100) & (full_labels == c2kv_labels)
        token_count = int(mask.sum().item())
        total_full_ce += float(full_loss.detach().cpu())
        total_c2kv_ce += float(c2kv_loss.detach().cpu())
        total_batches += 1
        if token_count:
            full_pred = full_logits.argmax(dim=-1)
            c2kv_pred = c2kv_logits.argmax(dim=-1)
            total_agree += float((full_pred[mask] == c2kv_pred[mask]).float().sum().detach().cpu())
            kl = F.kl_div(
                F.log_softmax(c2kv_logits[mask].float(), dim=-1),
                F.softmax(full_logits[mask].float(), dim=-1),
                reduction="sum",
            )
            total_kl += float(kl.detach().cpu())
            total_tokens += token_count
    return {
        "full_ce": total_full_ce / max(1, total_batches),
        "ce": total_c2kv_ce / max(1, total_batches),
        "token_agreement": total_agree / max(1, total_tokens),
        "kl_full_c2kv": total_kl / max(1, total_tokens),
        "target_tokens": float(total_tokens),
    }


def gist_param_snapshot(model) -> Dict[str, torch.Tensor]:
    raw_model = unwrap_model(model)
    return {
        name: param.detach().float().cpu().clone()
        for name, param in raw_model.named_parameters()
        if param.requires_grad and "gist" in name
    }


def param_change(before: Dict[str, torch.Tensor], model) -> Dict[str, float]:
    raw_model = unwrap_model(model)
    sq_delta = 0.0
    sq_before = 0.0
    abs_sum = 0.0
    count = 0
    max_abs = 0.0
    for name, param in raw_model.named_parameters():
        if name not in before:
            continue
        old = before[name]
        new = param.detach().float().cpu()
        delta = new - old
        sq_delta += float(delta.pow(2).sum())
        sq_before += float(old.pow(2).sum())
        abs_sum += float(delta.abs().sum())
        count += delta.numel()
        max_abs = max(max_abs, float(delta.abs().max()))
    return {
        "abs_mean": abs_sum / max(1, count),
        "abs_max": max_abs,
        "delta_l2": sq_delta ** 0.5,
        "relative_l2": (sq_delta ** 0.5) / max(1e-12, sq_before ** 0.5),
    }


def train_steps(model, loader, args) -> Tuple[List[Dict[str, float]], bool]:
    model.train()
    raw_model = unwrap_model(model)
    optimizer = torch.optim.AdamW([p for p in raw_model.parameters() if p.requires_grad], lr=args.learning_rate)
    logs: List[Dict[str, float]] = []
    grad_ok = False
    iterator = iter(loader)
    for step in range(1, args.train_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = move_batch(batch, args.device)
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = c2kv_forward(model, batch, args.max_doc_length, args)
        loss.backward()
        grad_norm_sq = 0.0
        for param in raw_model.parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm_sq += float(param.grad.detach().float().pow(2).sum().cpu())
        grad_norm = grad_norm_sq ** 0.5
        grad_ok = grad_ok or grad_norm > 0
        optimizer.step()
        if is_rank0() and (step == 1 or step % args.log_every == 0 or step == args.train_steps):
            log = {"step": float(step), "ce": float(loss.detach().cpu()), "grad_norm": grad_norm}
            logs.append(log)
            print(f"step={step} ce={log['ce']:.6f} gist_grad_norm={grad_norm:.6e}", flush=True)
    return logs, grad_ok


def build_generation_inputs(sample: Dict[str, Any], tokenizer, args, *, c2kv: bool) -> Dict[str, Any]:
    system = sample.get("system") or [{"role": "system", "content": "You are a helpful assistant."}]
    tools = sample.get("tools") or None
    history_docs = list(sample.get("history_docs") or [])[-args.max_history_docs :]
    history_messages = [{"role": "user", "content": str(doc.get("content") or "")} for doc in history_docs]
    current_messages = list(sample.get("current_messages") or [])
    if not c2kv:
        ids = chat_ids(
            tokenizer,
            system + history_messages + current_messages,
            tools=tools,
            add_generation_prompt=True,
            keep_bos=True,
            max_length=args.max_full_prompt_length,
        )
        return {"input_ids": torch.tensor([ids], dtype=torch.long, device=args.device)}
    system_ids = chat_ids(tokenizer, system, tools=tools, keep_bos=True, max_length=args.max_system_length)
    current_ids = chat_ids(tokenizer, current_messages, add_generation_prompt=True, max_length=args.max_current_length)
    context_ids: List[int] = []
    for doc in history_docs:
        doc_ids = chat_ids(tokenizer, [{"role": "user", "content": str(doc.get("content") or "")}], max_length=args.max_doc_length)
        context_ids.extend(pad(doc_ids, args.max_doc_length, -100))
    context_ids.extend([-100] * (args.max_doc_length * (args.max_history_docs - len(history_docs))))
    return {
        "system_input_ids": torch.tensor([system_ids], dtype=torch.long, device=args.device),
        "context_input_ids": torch.tensor([context_ids], dtype=torch.long, device=args.device),
        "context_ratios": torch.tensor([[args.compression_ratio] * args.max_history_docs], dtype=torch.long, device=args.device),
        "input_ids": torch.tensor([current_ids], dtype=torch.long, device=args.device),
    }


@torch.no_grad()
def generate_full(model, tokenizer, sample: Dict[str, Any], args) -> str:
    model.eval()
    payload = build_generation_inputs(sample, tokenizer, args, c2kv=False)
    generated = payload["input_ids"]
    old_outer, old_inner = set_attn_impl(model, args.decode_attn_impl)
    try:
        prompt_len = generated.shape[1]
        for _ in range(args.gen_max_new_tokens):
            outputs = model(input_ids=generated, attention_mask=torch.ones_like(generated), use_cache=False, logits_to_keep=1)
            next_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_id], dim=1)
            if int(next_id.item()) == tokenizer.eos_token_id:
                break
    finally:
        restore_attn_impl(model, old_outer, old_inner)
    return tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True)


@torch.no_grad()
def generate_c2kv(model, tokenizer, sample: Dict[str, Any], args) -> str:
    model.eval()
    payload = build_generation_inputs(sample, tokenizer, args, c2kv=True)
    generated = payload["input_ids"]
    context = payload["context_input_ids"].reshape(1, -1, args.max_doc_length)
    ratios = payload["context_ratios"]
    doc_lengths = (context != -100).sum(dim=2)
    max_doc_active = int(doc_lengths.max().item()) if doc_lengths.numel() else 0
    if 0 < max_doc_active < args.max_doc_length:
        context = context[:, :, :max_doc_active]
    system_kv, system_mask, past_length = build_system_kv(model, payload["system_input_ids"])
    context_token_len = int((payload["context_input_ids"] != -100).sum().item())
    old_outer, old_inner = set_attn_impl(model, args.decode_attn_impl)
    try:
        prompt_len = generated.shape[1]
        for _ in range(args.gen_max_new_tokens):
            position_ids = torch.arange(generated.shape[1], dtype=torch.long, device=args.device).unsqueeze(0)
            position_ids += past_length + context_token_len
            outputs = model(
                input_ids=generated,
                attention_mask=torch.ones_like(generated),
                context_input_ids=context,
                context_ratios=ratios,
                past_key_values=system_kv,
                past_attention_mask=system_mask,
                position_ids=position_ids,
                use_cache=False,
                logits_to_keep=1,
            )
            next_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_id], dim=1)
            if int(next_id.item()) == tokenizer.eos_token_id:
                break
    finally:
        restore_attn_impl(model, old_outer, old_inner)
    return tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True)


def generation_metrics(model, tokenizer, samples: List[Dict[str, Any]], args, stage: str) -> Dict[str, float]:
    valid = 0
    agree = 0
    parser_errors = 0
    outputs = []
    for sample in samples[: args.generate_samples]:
        try:
            full_text = generate_full(model, tokenizer, sample, args)
            c2kv_text = generate_c2kv(model, tokenizer, sample, args)
            full_call = extract_tool_call(full_text)
            c2kv_call = extract_tool_call(c2kv_text)
        except Exception as exc:
            parser_errors += 1
            outputs.append({"sample_id": sample.get("sample_id"), "error": repr(exc)})
            continue
        outputs.append({"sample_id": sample.get("sample_id"), "full": full_text, "c2kv": c2kv_text})
        if full_call is not None and c2kv_call is not None:
            valid += 1
            agree += int(full_call.get("name") == c2kv_call.get("name"))
    write_jsonl(Path(args.output_dir) / f"generation_outputs_{stage}.jsonl", outputs)
    return {
        "generated_valid_pair_parse_rate": valid / max(1, min(args.generate_samples, len(samples))),
        "generated_tool_name_agreement": agree / max(1, valid),
        "generation_parser_errors": float(parser_errors),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_file", default="outputs/quick_agent_c2kv_sanity/quick_samples.jsonl")
    parser.add_argument("--model_name_or_path", default="models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--output_dir", default="outputs/final_verify_agent_c2kv_32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_history_docs", type=int, default=4)
    parser.add_argument("--max_doc_length", type=int, default=512)
    parser.add_argument("--max_system_length", type=int, default=2048)
    parser.add_argument("--max_current_length", type=int, default=1024)
    parser.add_argument("--max_c2kv_length", type=int, default=2048)
    parser.add_argument("--max_full_prompt_length", type=int, default=4096)
    parser.add_argument("--max_full_length", type=int, default=6144)
    parser.add_argument("--compression_ratio", type=int, default=2)
    parser.add_argument("--train_steps", type=int, default=500)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-6)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--attn_impl", default="flex_attention")
    parser.add_argument("--decode_attn_impl", default="sdpa")
    parser.add_argument("--gen_max_new_tokens", type=int, default=128)
    parser.add_argument("--generate_samples", type=int, default=20)
    parser.add_argument("--no_ddp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.environ["C2KV_GIST_TRAIN_RATIOS"] = str(args.compression_ratio)
    gist_utils.GIST_GRADIENT_CHECKPOINTING = False
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"
    else:
        args.device = "cpu"
    rank = int(os.environ.get("RANK", "0"))
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    world_size = 1 if args.no_ddp else env_world_size
    output_dir = Path(args.output_dir)
    if is_rank0():
        output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_fixed_samples(args.samples_file, args.seed)
    if is_rank0():
        write_jsonl(output_dir / "fixed_32_samples.jsonl", samples)
        print("Fixed samples:", dict(Counter(s.get("source") for s in samples)), flush=True)

    model_args = ModelArgs(
        model_name_or_path=args.model_name_or_path,
        padding_side="right",
        attn_impl=args.attn_impl,
        dtype=args.dtype,
        enable_gist=True,
        gist_param="qkv",
        gist_type="dynamic-interleave",
        gist_overlap=64,
        gist_residual_type="embed-mean",
        gist_gradient_checkpointing=False,
    )
    model, tokenizer = get_model_and_tokenizer(model_args, device=args.device, evaluation_mode=False)
    model.requires_grad_(False)
    for name, param in model.named_parameters():
        param.requires_grad_("gist" in name)
    trainable = [(name, p.numel()) for name, p in model.named_parameters() if p.requires_grad]
    if is_rank0():
        print("Trainable parameters:")
        for name, numel in trainable:
            print(f"  {name}: {numel}")
        print(f"Total trainable: {format_numel_str(sum(n for _, n in trainable))}")

    dataset = QuickAgentDataset(samples, tokenizer, args)
    eval_loader = DataLoader(dataset, batch_size=args.per_device_batch_size, shuffle=False, collate_fn=collate)
    marker = output_dir / f"before_eval_done.{os.environ.get('MASTER_PORT', 'none')}.marker"
    if is_rank0() and marker.exists():
        marker.unlink()

    if is_rank0():
        before = pair_metrics(model, eval_loader, tokenizer, args)
        before_gen = generation_metrics(model, tokenizer, samples, args, "before")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        marker.write_text("done\n", encoding="utf-8")
    else:
        before = before_gen = {}
        while not marker.exists():
            time.sleep(5)

    param_before = gist_param_snapshot(model)
    if world_size > 1:
        rank, local_rank, world_size = maybe_init_distributed(args.device)
        model = DDP(model, device_ids=[local_rank] if args.device.startswith("cuda") else None, find_unused_parameters=False, static_graph=True)
    train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed) if world_size > 1 else None
    train_loader = DataLoader(dataset, batch_size=args.per_device_batch_size, shuffle=train_sampler is None, sampler=train_sampler, collate_fn=collate)
    step_logs, grad_ok = train_steps(model, train_loader, args)
    raw_model = unwrap_model(model)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    model = raw_model

    if not is_rank0():
        return
    change = param_change(param_before, model)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    after = pair_metrics(model, eval_loader, tokenizer, args)
    after_gen = generation_metrics(model, tokenizer, samples, args, "after")
    ce_drop = (before["ce"] - after["ce"]) / max(1e-12, before["ce"])
    pass_flag = (
        change["relative_l2"] > 0
        and grad_ok
        and (
            ce_drop >= 0.20
            or after["token_agreement"] > before["token_agreement"]
            or after["kl_full_c2kv"] < before["kl_full_c2kv"]
        )
    )
    print("\n| Metric | Before | After |")
    print("|---|---:|---:|")
    print(f"| CE | {before['ce']:.6f} | {after['ce']:.6f} |")
    print(f"| Token Agreement | {before['token_agreement']:.6f} | {after['token_agreement']:.6f} |")
    print(f"| KL | {before['kl_full_c2kv']:.6f} | {after['kl_full_c2kv']:.6f} |")
    print(f"| Generated Tool Name Agreement | {before_gen['generated_tool_name_agreement']:.6f} | {after_gen['generated_tool_name_agreement']:.6f} |")
    print(f"Param change: abs_mean={change['abs_mean']:.6e} abs_max={change['abs_max']:.6e} relative_l2={change['relative_l2']:.6e}")
    print(f"Gradient nonzero: {grad_ok}")
    print("PASS" if pass_flag else "FAIL")
    metrics = {
        "before": before,
        "after": after,
        "before_generation": before_gen,
        "after_generation": after_gen,
        "param_change": change,
        "step_logs": step_logs,
        "gradient_nonzero": grad_ok,
        "ce_relative_drop": ce_drop,
        "pass": pass_flag,
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "metrics.json", metrics)
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
