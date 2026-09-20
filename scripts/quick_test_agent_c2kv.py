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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import set_seed
from transformers.cache_utils import DynamicCache

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from agent_history_multisource_lib import (  # noqa: E402
    build_decision_points,
    extract_tool_call,
    iter_agent_llm,
    iter_hermes,
    iter_toucan,
)
from gist_args import ModelArgs  # noqa: E402
from models import format_numel_str, get_model_and_tokenizer  # noqa: E402


def pad(values: List[int], length: int, pad_value: int) -> List[int]:
    if len(values) >= length:
        return values[:length]
    return values + [pad_value] * (length - len(values))


def truncate_prompt_for_target(prompt_ids: List[int], target_ids: List[int], max_length: int) -> Tuple[List[int], List[int]]:
    if max_length < 2:
        raise ValueError(f"max_length must be >= 2, got {max_length}")
    prompt_budget = max_length - 1
    if len(prompt_ids) > prompt_budget:
        prompt_ids = prompt_ids[-prompt_budget:]
    target_budget = max(1, max_length - len(prompt_ids))
    return prompt_ids, target_ids[:target_budget]


def _messages_with_tools_in_system(messages: Sequence[Dict[str, Any]], tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = [dict(m) for m in messages]
    tool_text = "\n\nAvailable tools:\n" + json.dumps(tools, ensure_ascii=False, indent=2)
    if out and out[0].get("role") == "system":
        out[0]["content"] = str(out[0].get("content") or "") + tool_text
    else:
        out.insert(0, {"role": "system", "content": tool_text.strip()})
    return out


def chat_ids(
    tokenizer,
    messages: Sequence[Dict[str, Any]],
    *,
    tools: Optional[List[Dict[str, Any]]] = None,
    add_generation_prompt: bool = False,
    keep_bos: bool = False,
    max_length: Optional[int] = None,
) -> List[int]:
    template_kwargs = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": False,
        "max_length": max_length + 1 if max_length is not None and not keep_bos else max_length,
        "truncation": max_length is not None,
    }
    try:
        encoded = tokenizer.apply_chat_template(list(messages), tools=tools, **template_kwargs)
    except Exception as exc:
        if not tools:
            raise
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"WARNING: tokenizer tools= serialization failed, falling back to system-embedded tools: {type(exc).__name__}: {exc}", flush=True)
        encoded = tokenizer.apply_chat_template(_messages_with_tools_in_system(messages, tools), tools=None, **template_kwargs)
    ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded
    if not keep_bos and ids and ids[0] == tokenizer.bos_token_id:
        ids = ids[1:]
    return ids


def collect_source_samples(
    name: str,
    iterator: Iterable[Any],
    quota: int,
    rng: random.Random,
    *,
    min_completed_history_turns: int,
    max_history_docs: int,
    reservoir_factor: int = 8,
) -> List[Dict[str, Any]]:
    pool: List[Dict[str, Any]] = []
    limit = max(quota * reservoir_factor, quota + 16)
    for traj in iterator:
        points = build_decision_points(
            traj,
            min_completed_history_turns=min_completed_history_turns,
            max_history_docs=max_history_docs,
            require_previous_history=True,
        )
        if name == "toucan":
            points = [p for p in points if p.get("metadata", {}).get("subset_name") == "multi-turn"]
        for point in points:
            pool.append(point)
            if len(pool) >= limit:
                break
        if len(pool) >= limit:
            break
    rng.shuffle(pool)
    return pool[:quota]


def collect_samples(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Counter]:
    rng = random.Random(args.seed)
    stats: Counter = Counter()
    quotas = {
        "agent_llm": args.agent_samples,
        "toucan": args.toucan_samples,
        "hermes": args.hermes_samples,
    }
    samples: List[Dict[str, Any]] = []
    samples.extend(
        collect_source_samples(
            "agent_llm",
            iter_agent_llm(args.agent_llm_path, stats, strip_reasoning=False),
            quotas["agent_llm"],
            rng,
            min_completed_history_turns=args.min_completed_history_turns,
            max_history_docs=args.max_history_docs,
        )
    )
    samples.extend(
        collect_source_samples(
            "toucan",
            iter_toucan(args.toucan_path, stats, allow_irrelevant=False),
            quotas["toucan"],
            rng,
            min_completed_history_turns=args.min_completed_history_turns,
            max_history_docs=args.max_history_docs,
        )
    )
    samples.extend(
        collect_source_samples(
            "hermes",
            iter_hermes(args.hermes_path, stats, strip_reasoning=True),
            quotas["hermes"],
            rng,
            min_completed_history_turns=args.min_completed_history_turns,
            max_history_docs=args.max_history_docs,
        )
    )
    rng.shuffle(samples)
    samples = samples[: args.max_samples]
    for i, sample in enumerate(samples):
        sample["quick_test_id"] = i
    return samples, stats




def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def print_sample(sample: Dict[str, Any], index: int) -> None:
    print(f"\n===== SAMPLE {index} | {sample.get('source')} | {sample.get('sample_id')} =====")
    print("\n[SYSTEM]")
    print(json.dumps(sample.get("system", []), ensure_ascii=False, indent=2))
    print("\n[TOOLS]")
    print(json.dumps(sample.get("tools", []), ensure_ascii=False, indent=2))
    print("\n[HISTORY DOCS]")
    for j, doc in enumerate(sample.get("history_docs", [])):
        print(f"\n--- history_doc[{j}] turn_id={doc.get('turn_id')} chunk_id={doc.get('chunk_id')} ---")
        print(doc.get("content", ""))
    print("\n[CURRENT TURN]")
    print(json.dumps(sample.get("current_messages", []), ensure_ascii=False, indent=2))
    print("\n[TARGET]")
    print(sample.get("target", ""))


class QuickAgentDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]], tokenizer, args: argparse.Namespace) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.args = args

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.samples[index]
        system = row.get("system") or [{"role": "system", "content": "You are a helpful assistant."}]
        tools = row.get("tools") or None
        history_docs = list(row.get("history_docs") or [])[-self.args.max_history_docs :]
        history_messages = [{"role": "user", "content": str(doc.get("content") or "")} for doc in history_docs]
        current_messages = list(row.get("current_messages") or [])
        target_ids = self.tokenizer.encode(str(row.get("target") or ""), add_special_tokens=False)
        target_ids = (target_ids or [self.tokenizer.eos_token_id]) + [self.tokenizer.eos_token_id]

        full_prompt = chat_ids(
            self.tokenizer,
            system + history_messages + current_messages,
            tools=tools,
            add_generation_prompt=True,
            keep_bos=True,
            max_length=self.args.max_full_prompt_length,
        )
        full_prompt, full_target = truncate_prompt_for_target(full_prompt, target_ids, self.args.max_full_length)
        full_input_ids = full_prompt + full_target
        full_labels = [-100] * len(full_prompt) + full_target
        full_attention_mask = [1] * len(full_input_ids)

        system_ids = chat_ids(
            self.tokenizer,
            system,
            tools=tools,
            keep_bos=True,
            max_length=self.args.max_system_length,
        )
        context_input_ids: List[int] = []
        for doc in history_docs:
            doc_ids = chat_ids(
                self.tokenizer,
                [{"role": "user", "content": str(doc.get("content") or "")}],
                max_length=self.args.max_doc_length,
            )
            context_input_ids.extend(pad(doc_ids, self.args.max_doc_length, -100))
        empty_docs = self.args.max_history_docs - len(history_docs)
        context_input_ids.extend([-100] * (self.args.max_doc_length * empty_docs))

        current_prompt = chat_ids(
            self.tokenizer,
            current_messages,
            add_generation_prompt=True,
            max_length=self.args.max_current_length,
        )
        current_prompt, c2kv_target = truncate_prompt_for_target(current_prompt, target_ids, self.args.max_c2kv_length)
        c2kv_input_ids = current_prompt + c2kv_target
        c2kv_labels = [-100] * len(current_prompt) + c2kv_target
        c2kv_attention_mask = [1] * len(c2kv_input_ids)

        return {
            "full_input_ids": pad(full_input_ids, self.args.max_full_length, self.tokenizer.pad_token_id),
            "full_attention_mask": pad(full_attention_mask, self.args.max_full_length, 0),
            "full_labels": pad(full_labels, self.args.max_full_length, -100),
            "system_input_ids": pad(system_ids, self.args.max_system_length, -100),
            "context_input_ids": context_input_ids,
            "context_ratios": [self.args.compression_ratio] * self.args.max_history_docs,
            "input_ids": pad(c2kv_input_ids, self.args.max_c2kv_length, self.tokenizer.pad_token_id),
            "attention_mask": pad(c2kv_attention_mask, self.args.max_c2kv_length, 0),
            "labels": pad(c2kv_labels, self.args.max_c2kv_length, -100),
            "metadata_json": json.dumps(
                {
                    "quick_test_id": row.get("quick_test_id", index),
                    "source": row.get("source"),
                    "sample_id": row.get("sample_id"),
                    "target": row.get("target", ""),
                },
                ensure_ascii=False,
            ),
        }


def collate(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch: Dict[str, Any] = {}
    for key in features[0]:
        if key == "metadata_json":
            batch[key] = [f[key] for f in features]
        else:
            batch[key] = torch.tensor([f[key] for f in features], dtype=torch.long)
    return batch


def trim_batch(input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    active = attention_mask.sum(dim=1)
    max_len = int(active.max().item())
    return input_ids[:, :max_len], attention_mask[:, :max_len], labels[:, :max_len]


def shifted_target_logits(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    mask = shift_labels != -100
    return shift_logits, shift_labels, mask




def target_logits_window(labels: torch.Tensor) -> Tuple[int, torch.Tensor]:
    label_mask = labels != -100
    if not label_mask.any():
        return 0, labels
    first_positions = label_mask.float().argmax(dim=1)
    first_label = int(first_positions[label_mask.any(dim=1)].min().item())
    logits_start = max(0, first_label - 1)
    return logits_start, labels[:, logits_start:]

def target_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits, shift_labels, mask = shifted_target_logits(logits, labels)
    if not mask.any():
        return shift_logits.new_zeros(())
    return F.cross_entropy(shift_logits[mask].float(), shift_labels[mask], reduction="mean")


@torch.no_grad()
def build_system_kv(model, system_input_ids: torch.Tensor) -> Tuple[DynamicCache, torch.Tensor, int]:
    raw_model = unwrap_model(model)
    inner = raw_model.model if hasattr(raw_model, "model") else raw_model
    device = next(raw_model.parameters()).device
    system_input_ids = system_input_ids.to(device)
    real_mask = system_input_ids != -100
    real_lens = real_mask.sum(dim=1)
    batch_size = system_input_ids.shape[0]
    max_len = int(real_lens.max().item())
    pad_id = inner.config.pad_token_id or 0
    left_ids = system_input_ids.new_full((batch_size, max_len), pad_id)
    system_mask = system_input_ids.new_zeros((batch_size, max_len))
    for i in range(batch_size):
        n = int(real_lens[i].item())
        if n:
            left_ids[i, max_len - n :] = system_input_ids[i][real_mask[i]]
            system_mask[i, max_len - n :] = 1
    was_training = model.training
    model.eval()
    old_outer, old_inner = set_attn_impl(raw_model, "sdpa")
    try:
        outputs = raw_model(left_ids, attention_mask=system_mask, use_cache=True, logits_to_keep=1)
    finally:
        restore_attn_impl(raw_model, old_outer, old_inner)
    if was_training:
        model.train()
    return outputs.past_key_values, system_mask, max_len


def full_forward(model, batch: Dict[str, Any], args: Optional[argparse.Namespace] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids, attention_mask, labels = trim_batch(
        batch["full_input_ids"],
        batch["full_attention_mask"],
        batch["full_labels"],
    )
    logits_start, labels_for_logits = target_logits_window(labels)
    old_outer, old_inner = set_attn_impl(model, args.decode_attn_impl if hasattr(args, "decode_attn_impl") else "sdpa")
    try:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, logits_to_keep=input_ids.shape[1] - logits_start)
    finally:
        restore_attn_impl(model, old_outer, old_inner)
    loss = target_ce_loss(outputs.logits, labels_for_logits)
    return loss, outputs.logits, labels_for_logits


def c2kv_forward(model, batch: Dict[str, Any], max_doc_length: int, args: Optional[argparse.Namespace] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = batch["input_ids"].shape[0]
    input_ids, attention_mask, labels = trim_batch(batch["input_ids"], batch["attention_mask"], batch["labels"])
    context_input_ids = batch["context_input_ids"].reshape(batch_size, -1, max_doc_length)
    context_ratios = batch["context_ratios"].reshape(batch_size, -1)
    doc_lengths = (context_input_ids != -100).sum(dim=2)
    max_doc_active_length = int(doc_lengths.max().item()) if doc_lengths.numel() else 0
    if 0 < max_doc_active_length < max_doc_length:
        context_input_ids = context_input_ids[:, :, :max_doc_active_length]

    system_kv, system_mask, past_length = build_system_kv(model, batch["system_input_ids"])
    context_token_lens = (batch["context_input_ids"] != -100).sum(dim=1)
    position_ids = torch.arange(input_ids.shape[1], dtype=torch.long, device=input_ids.device).unsqueeze(0).repeat(batch_size, 1)
    for i, seqlen in enumerate(context_token_lens.tolist()):
        position_ids[i] += past_length + int(seqlen)

    logits_start, labels_for_logits = target_logits_window(labels)
    old_outer, old_inner = set_attn_impl(model, args.decode_attn_impl if hasattr(args, "decode_attn_impl") else "sdpa")
    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            context_input_ids=context_input_ids,
            context_ratios=context_ratios,
            past_key_values=system_kv,
            past_attention_mask=system_mask,
            position_ids=position_ids,
            use_cache=False,
            logits_to_keep=input_ids.shape[1] - logits_start,
        )
    finally:
        restore_attn_impl(model, old_outer, old_inner)
    return target_ce_loss(outputs.logits, labels_for_logits), outputs.logits, labels_for_logits


def decode_target_prediction(tokenizer, logits: torch.Tensor, labels: torch.Tensor) -> List[Dict[str, Any]]:
    _, shifted_labels, mask = shifted_target_logits(logits, labels)
    pred = logits[:, :-1].argmax(dim=-1)
    rows: List[Dict[str, Any]] = []
    for i in range(labels.shape[0]):
        pos = mask[i].nonzero(as_tuple=False).squeeze(1)
        if pos.numel() == 0:
            rows.append({"target_text": "", "pred_text": "", "target_call": None, "pred_call": None})
            continue
        target_ids = shifted_labels[i, pos].tolist()
        pred_ids = pred[i, pos].tolist()
        target_text = tokenizer.decode(target_ids, skip_special_tokens=True)
        pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
        rows.append(
            {
                "target_text": target_text,
                "pred_text": pred_text,
                "target_call": extract_tool_call(target_text),
                "pred_call": extract_tool_call(pred_text),
            }
        )
    return rows


@torch.no_grad()
def evaluate(model, loader: DataLoader, tokenizer, args: argparse.Namespace, mode: str) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    total_tool_targets = 0
    parse_success = 0
    tool_name_correct = 0
    all_decoded: List[Dict[str, Any]] = []
    for batch in loader:
        batch = move_batch(batch, args.device)
        if mode == "full":
            loss, logits, labels = full_forward(model, batch, args)
        else:
            loss, logits, labels = c2kv_forward(model, batch, args.max_doc_length, args)
        total_loss += float(loss.detach().cpu())
        total_batches += 1
        decoded = decode_target_prediction(tokenizer, logits.detach().cpu(), labels.detach().cpu())
        all_decoded.extend(decoded)
        for row in decoded:
            target_call = row["target_call"]
            if target_call is None:
                continue
            total_tool_targets += 1
            pred_call = row["pred_call"]
            if pred_call is not None:
                parse_success += 1
                tool_name_correct += int(pred_call.get("name") == target_call.get("name"))
    out = {
        "target_ce_loss": total_loss / max(1, total_batches),
        "tool_call_parse_success": parse_success / max(1, total_tool_targets),
        "tool_name_accuracy": tool_name_correct / max(1, total_tool_targets),
        "tool_targets": float(total_tool_targets),
    }
    out["_decoded"] = all_decoded  # type: ignore[assignment]
    return out


def prediction_agreement(before_or_after: Dict[str, Any], full: Dict[str, Any]) -> float:
    c2kv_rows = before_or_after.get("_decoded", [])
    full_rows = full.get("_decoded", [])
    if not c2kv_rows or not full_rows:
        return 0.0
    matches = 0
    count = min(len(c2kv_rows), len(full_rows))
    for a, b in zip(c2kv_rows[:count], full_rows[:count]):
        matches += int(a.get("pred_text") == b.get("pred_text"))
    return matches / max(1, count)


def move_batch(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def set_attn_impl(model, attn_impl: str):
    raw_model = unwrap_model(model)
    old_outer = getattr(getattr(raw_model, "config", None), "_attn_implementation", None)
    inner = getattr(raw_model, "model", None)
    old_inner = getattr(getattr(inner, "config", None), "_attn_implementation", None)
    if hasattr(raw_model, "config"):
        raw_model.config._attn_implementation = attn_impl
    if inner is not None and hasattr(inner, "config"):
        inner.config._attn_implementation = attn_impl
    return old_outer, old_inner


def restore_attn_impl(model, old_outer, old_inner) -> None:
    raw_model = unwrap_model(model)
    inner = getattr(raw_model, "model", None)
    if old_outer is not None and hasattr(raw_model, "config"):
        raw_model.config._attn_implementation = old_outer
    if old_inner is not None and inner is not None and hasattr(inner, "config"):
        inner.config._attn_implementation = old_inner


def maybe_init_distributed(device: str) -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if device.startswith("cuda") else "gloo"
        dist.init_process_group(backend=backend)
    return rank, local_rank, world_size


def train_steps(model, loader: DataLoader, args: argparse.Namespace) -> Tuple[List[float], bool]:
    model.train()
    raw_model = unwrap_model(model)
    params = [p for p in raw_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    losses: List[float] = []
    grad_nonzero_seen = False
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
        for name, param in raw_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm_sq += float(param.grad.detach().float().pow(2).sum().cpu())
        grad_norm = grad_norm_sq ** 0.5
        grad_nonzero_seen = grad_nonzero_seen or grad_norm > 0
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if is_rank0() and (step == 1 or step % args.log_every == 0 or step == args.train_steps):
            print(f"step={step} loss={losses[-1]:.6f} gist_grad_norm={grad_norm:.6e}", flush=True)
    return losses, grad_nonzero_seen


def print_table(full: Dict[str, Any], before: Dict[str, Any], after: Dict[str, Any]) -> None:
    rows = [
        ("Full", full, 1.0),
        ("C2KV Before", before, prediction_agreement(before, full)),
        ("C2KV After", after, prediction_agreement(after, full)),
    ]
    print("\n===== METRICS =====")
    print("| mode | target CE loss | tool parse success | tool name acc | Full-C2KV pred agreement | tool targets |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, metrics, agree in rows:
        print(
            f"| {name} | {metrics['target_ce_loss']:.6f} | "
            f"{metrics['tool_call_parse_success']:.4f} | {metrics['tool_name_accuracy']:.4f} | "
            f"{agree:.4f} | {int(metrics['tool_targets'])} |"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal sanity test for multi-source Agent History C2KV.")
    parser.add_argument("--agent_llm_path", default="datasets/agent-llm-traces-v2")
    parser.add_argument("--hermes_path", default="datasets/hermes-agent-reasoning-traces")
    parser.add_argument("--toucan_path", default="datasets/toucan-1.5m")
    parser.add_argument("--model_name_or_path", default="models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--output_dir", default="outputs/quick_agent_c2kv_sanity")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--agent_samples", type=int, default=35)
    parser.add_argument("--toucan_samples", type=int, default=35)
    parser.add_argument("--hermes_samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_completed_history_turns", type=int, default=2)
    parser.add_argument("--max_history_docs", type=int, default=4)
    parser.add_argument("--max_doc_length", type=int, default=512)
    parser.add_argument("--max_system_length", type=int, default=2048)
    parser.add_argument("--max_current_length", type=int, default=1024)
    parser.add_argument("--max_c2kv_length", type=int, default=2048)
    parser.add_argument("--max_full_prompt_length", type=int, default=6144)
    parser.add_argument("--max_full_length", type=int, default=8192)
    parser.add_argument("--compression_ratio", type=int, default=2)
    parser.add_argument("--train_steps", type=int, default=200)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--attn_impl", default="flex_attention")
    parser.add_argument("--decode_attn_impl", default="sdpa")
    parser.add_argument("--no_ddp", action="store_true")
    parser.add_argument("--gist_overlap", type=int, default=64)
    parser.add_argument("--gist_gradient_checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--print_samples", type=int, default=5)
    parser.add_argument("--skip_training", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.environ["C2KV_GIST_TRAIN_RATIOS"] = str(args.compression_ratio)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"
    elif hasattr(torch, "npu") and torch.npu.is_available():
        args.device = f"npu:{local_rank}"
    else:
        args.device = "cpu"

    rank = int(os.environ.get("RANK", "0"))
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    world_size = 1 if args.no_ddp else env_world_size
    samples, stats = collect_samples(args)
    output_dir = Path(args.output_dir)
    if is_rank0():
        output_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(output_dir / "quick_samples.jsonl", samples)
        write_json(output_dir / "construction_stats.json", {"stats": dict(stats), "source_distribution": dict(Counter(s.get("source") for s in samples)), "sample_count": len(samples)})
        print(f"Collected {len(samples)} samples")
        print("Construction stats:")
        for key, value in sorted(stats.items()):
            print(f"  {key}: {value}")
        print("Source distribution:", dict(Counter(s.get("source") for s in samples)))
    if len(samples) < args.max_samples:
        raise RuntimeError(f"Only collected {len(samples)} samples, expected {args.max_samples}")

    if local_rank == 0:
        for i, sample in enumerate(samples[: args.print_samples]):
            print_sample(sample, i)

    import models.gist_utils as gist_utils
    gist_utils.GIST_GRADIENT_CHECKPOINTING = bool(args.gist_gradient_checkpointing)
    if args.gist_gradient_checkpointing:
        os.environ.setdefault("C2KV_GIST_CHECKPOINT_USE_REENTRANT", "false")

    model_args = ModelArgs(
        model_name_or_path=args.model_name_or_path,
        padding_side="right",
        attn_impl=args.attn_impl,
        dtype=args.dtype,
        enable_gist=True,
        gist_param="qkv",
        gist_type="dynamic-interleave",
        gist_overlap=args.gist_overlap,
        gist_residual_type="embed-mean",
        gist_gradient_checkpointing=args.gist_gradient_checkpointing,
    )
    model, tokenizer = get_model_and_tokenizer(model_args, device=args.device, evaluation_mode=False)
    model.requires_grad_(False)
    for name, param in model.named_parameters():
        param.requires_grad_("gist" in name)
    non_gist_trainable = [name for name, param in model.named_parameters() if param.requires_grad and "gist" not in name]
    assert not non_gist_trainable, f"Base model params unexpectedly trainable: {non_gist_trainable[:8]}"
    trainable = [(name, p.numel()) for name, p in model.named_parameters() if p.requires_grad]
    if is_rank0():
        print("\n===== TRAINABLE PARAMETERS =====")
        for name, numel in trainable:
            print(f"{name}: {numel}")
        print(f"Total trainable: {format_numel_str(sum(n for _, n in trainable))}")
    dataset = QuickAgentDataset(samples, tokenizer, args)
    loader = DataLoader(dataset, batch_size=args.per_device_batch_size, shuffle=False, collate_fn=collate)
    run_marker_id = os.environ.get("TORCHELASTIC_RUN_ID") or os.environ.get("MASTER_PORT") or str(os.getppid())
    pre_eval_marker = output_dir / f"pre_eval_done.{run_marker_id}.marker"
    if is_rank0() and pre_eval_marker.exists():
        pre_eval_marker.unlink()

    # Rank0 does the expensive full-history eval before NCCL is initialized.
    # Other ranks wait on a filesystem marker, so no collective can timeout.
    if is_rank0():
        full_metrics = evaluate(model, loader, tokenizer, args, mode="full")
        before_metrics = evaluate(model, loader, tokenizer, args, mode="c2kv")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        pre_eval_marker.write_text("done\n", encoding="utf-8")
    else:
        full_metrics = before_metrics = {}
        while not pre_eval_marker.exists():
            time.sleep(5)

    grad_ok = False
    losses: List[float] = []
    if not args.skip_training:
        if world_size > 1:
            rank, local_rank, world_size = maybe_init_distributed(args.device)
            model = DDP(
                model,
                device_ids=[local_rank] if args.device.startswith("cuda") else None,
                find_unused_parameters=False,
                static_graph=True,
            )
        train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed) if world_size > 1 else None
        train_loader = DataLoader(
            dataset,
            batch_size=args.per_device_batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            collate_fn=collate,
        )
        losses, grad_ok = train_steps(model, train_loader, args)
        raw_model = unwrap_model(model)
        if world_size > 1:
            dist.barrier()
            dist.destroy_process_group()
        model = raw_model

    if not is_rank0():
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    after_metrics = evaluate(model, loader, tokenizer, args, mode="c2kv")
    print_table(full_metrics, before_metrics, after_metrics)
    if losses:
        print(f"\nTraining loss: first={losses[0]:.6f} last={losses[-1]:.6f} min={min(losses):.6f}")
    print(f"Non-zero C2KV/gist gradient observed: {grad_ok}")
    pass_conditions = [
        full_metrics["target_ce_loss"] > 0,
        before_metrics["target_ce_loss"] >= full_metrics["target_ce_loss"] * 0.5,
        (not losses) or losses[-1] < losses[0],
        grad_ok or args.skip_training,
    ]
    print("\nPASS" if all(pass_conditions) else "\nFAIL")
    metrics_payload = {
        "full": {k: v for k, v in full_metrics.items() if not k.startswith("_")},
        "c2kv_before": {k: v for k, v in before_metrics.items() if not k.startswith("_")},
        "c2kv_after": {k: v for k, v in after_metrics.items() if not k.startswith("_")},
        "full_c2kv_before_prediction_agreement": prediction_agreement(before_metrics, full_metrics),
        "full_c2kv_after_prediction_agreement": prediction_agreement(after_metrics, full_metrics),
        "train_loss_first": losses[0] if losses else None,
        "train_loss_last": losses[-1] if losses else None,
        "train_loss_min": min(losses) if losses else None,
        "nonzero_gist_gradient_observed": grad_ok,
        "pass": all(pass_conditions),
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "metrics.json", metrics_payload)
    print(f"Output directory: {output_dir}")
    print("Output files: quick_samples.jsonl, construction_stats.json, metrics.json")
    if not all(pass_conditions):
        print("Most likely places to inspect: sample serialization/chat template, target mask alignment, optimizer trainable gist params, or context_ratios/process_context_input_ids.")


if __name__ == "__main__":
    main()
