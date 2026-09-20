
#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, set_seed

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "python"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from gist_args import ModelArgs  # noqa: E402
from models import format_numel_str, get_model_and_tokenizer  # noqa: E402
from train.trainer import GistMultiDocTrainer, _as_scalar_loss  # noqa: E402
from agent_history_multisource_lib import extract_tool_call  # noqa: E402


def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml

        data = yaml.safe_load(text)
        return data or {}
    except Exception:
        return json.loads(text)


def parse_prob_map(value: str | Dict[str, float]) -> Dict[str, float]:
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items()}
    out = {}
    for item in str(value).split(","):
        if not item.strip():
            continue
        key, prob = item.split(":", 1)
        out[key.strip()] = float(prob)
    total = sum(out.values())
    if total <= 0:
        raise ValueError(f"Invalid probability map: {value}")
    return {k: v / total for k, v in out.items()}


def sample_from_probs(probs: Dict[str, float], rng: random.Random) -> str:
    x = rng.random()
    acc = 0.0
    last = next(iter(probs))
    for key, prob in probs.items():
        acc += prob
        last = key
        if x <= acc:
            return key
    return last


def pad(values: List[int], length: int, pad_value: int) -> List[int]:
    if len(values) >= length:
        return values[:length]
    return values + [pad_value] * (length - len(values))


def chat_ids(tokenizer, messages, *, tools=None, add_generation_prompt=False, keep_bos=False, max_length=None) -> List[int]:
    encoded = tokenizer.apply_chat_template(
        list(messages),
        tools=tools,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
        max_length=max_length + 1 if max_length is not None and not keep_bos else max_length,
        truncation=max_length is not None,
    )
    ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded
    if not keep_bos and ids and ids[0] == tokenizer.bos_token_id:
        ids = ids[1:]
    return ids


def choose_full_indices(n: int, policy: str, rng: random.Random) -> set[int]:
    if n <= 0 or policy == "all_compressed":
        return set()
    if policy == "recent1_full":
        return {n - 1}
    if policy == "recent2_full":
        return set(range(max(0, n - 2), n))
    if policy == "random1_full":
        return {rng.randrange(n)}
    if policy == "random2_full":
        return set(rng.sample(range(n), k=min(2, n)))
    raise ValueError(f"Unknown history policy: {policy}")


def choose_ratios(n: int, ratio_probs: Dict[str, float], hetero_prob: float, rng: random.Random) -> List[int]:
    if n <= 0:
        return []
    if rng.random() >= hetero_prob:
        ratio = int(sample_from_probs(ratio_probs, rng))
        return [ratio] * n
    ratios = []
    for i in range(n):
        frac = i / max(1, n - 1)
        if frac < 1 / 3:
            ratios.append(8)
        elif frac < 2 / 3:
            ratios.append(4)
        else:
            ratios.append(2)
    return ratios


def flatten_arg_items(args: Any) -> tuple[set[str], set[str]]:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            return set(), {args}
    names, values = set(), set()
    if isinstance(args, dict):
        for key, value in args.items():
            names.add(str(key))
            if isinstance(value, (dict, list)):
                values.add(json.dumps(value, ensure_ascii=False, sort_keys=True))
            else:
                values.add(str(value))
    return names, values


def f1(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return 2 * inter / (len(a) + len(b))


class AgentHistoryJsonlDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer,
        *,
        max_history_docs: int = 16,
        max_doc_length: int = 1024,
        max_length: int = 2048,
        max_system_length: int = 4096,
        max_teacher_length: int = 24576,
        max_target_length: int = 256,
        history_policy_probs: str | Dict[str, float] = "all_compressed:0.45,recent1_full:0.20,recent2_full:0.15,random1_full:0.10,random2_full:0.10",
        ratio_probs: str | Dict[str, float] = "2:0.333333,4:0.333333,8:0.333334",
        heterogeneous_ratio_prob: float = 0.20,
        seed: int = 42,
        max_precomputed_total_tokens: Optional[int] = None,
        max_precomputed_tools_tokens: Optional[int] = None,
        max_precomputed_current_tokens: Optional[int] = None,
    ) -> None:
        self.path = Path(path)
        self.filter_stats = Counter()
        self.offsets: List[int] = []
        with self.path.open("rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)
        # The 60k train JSONL is around 10GB. Loading every row as Python dicts in
        # every DDP rank costs tens of GB of host RAM per rank, and DataLoader
        # workers multiply that again. Keep only byte offsets and parse one row on
        # demand in __getitem__. Precomputed token filters are therefore only used
        # when a future compact manifest is provided.
        self.filter_stats["kept"] = len(self.offsets)
        self.filter_stats["raw"] = len(self.offsets)
        self._fh = None
        self.tokenizer = tokenizer
        self.max_history_docs = max_history_docs
        self.max_doc_length = max_doc_length
        self.max_length = max_length
        self.max_system_length = max_system_length
        self.max_teacher_length = max_teacher_length
        self.max_target_length = max_target_length
        self.history_policy_probs = parse_prob_map(history_policy_probs)
        self.ratio_probs = parse_prob_map(ratio_probs)
        self.heterogeneous_ratio_prob = heterogeneous_ratio_prob
        self.seed = seed

    def __len__(self) -> int:
        return len(self.offsets)

    def _read_row(self, index: int) -> Dict[str, Any]:
        if self._fh is None:
            self._fh = self.path.open("rb")
        self._fh.seek(self.offsets[index])
        return json.loads(self._fh.readline().decode("utf-8"))

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self._read_row(index)
        rng = random.Random(self.seed + index + random.randrange(1_000_000_000))
        history_docs = list(row.get("history_docs") or [])[-self.max_history_docs :]
        policy = sample_from_probs(self.history_policy_probs, rng)
        full_indices = choose_full_indices(len(history_docs), policy, rng)
        compressed_docs = [doc for i, doc in enumerate(history_docs) if i not in full_indices]
        full_docs = [doc for i, doc in enumerate(history_docs) if i in full_indices]
        ratios = choose_ratios(len(compressed_docs), self.ratio_probs, self.heterogeneous_ratio_prob, rng)

        system = row.get("system") or [{"role": "system", "content": "You are a helpful assistant."}]
        tools = row.get("tools") or None
        system_ids = chat_ids(self.tokenizer, system, tools=tools, keep_bos=True, max_length=self.max_system_length)
        system_input_ids = pad(system_ids, self.max_system_length, -100)

        context_input_ids: List[int] = []
        context_ratios: List[int] = []
        for doc, ratio in zip(compressed_docs, ratios):
            doc_ids = chat_ids(self.tokenizer, [{"role": "user", "content": str(doc.get("content") or "")}], max_length=self.max_doc_length)
            context_input_ids.extend(pad(doc_ids, self.max_doc_length, -100))
            context_ratios.append(int(ratio))
        empty_docs = self.max_history_docs - len(compressed_docs)
        context_input_ids.extend([-100] * (self.max_doc_length * empty_docs))
        context_ratios.extend([2] * empty_docs)

        full_history_ids: List[int] = []
        for doc in full_docs:
            full_history_ids.extend(chat_ids(self.tokenizer, [{"role": "user", "content": str(doc.get("content") or "")}], max_length=self.max_doc_length))
        current_ids = chat_ids(self.tokenizer, row.get("current_messages") or [], add_generation_prompt=True)
        student_prompt_ids = full_history_ids + current_ids

        teacher_history_messages = [{"role": "user", "content": str(doc.get("content") or "")} for doc in history_docs]
        teacher_prompt_ids = chat_ids(self.tokenizer, system + teacher_history_messages + list(row.get("current_messages") or []), tools=tools, add_generation_prompt=True, keep_bos=True)

        target_ids = self.tokenizer.encode(str(row.get("target") or ""), add_special_tokens=False)
        if not target_ids:
            target_ids = [self.tokenizer.eos_token_id]
        target_ids = (target_ids + [self.tokenizer.eos_token_id])[: max(1, self.max_target_length)]
        student_prompt_budget = max(0, self.max_length - len(target_ids))
        teacher_prompt_budget = max(0, self.max_teacher_length - len(target_ids))
        if len(student_prompt_ids) > student_prompt_budget:
            student_prompt_ids = student_prompt_ids[-student_prompt_budget:] if student_prompt_budget > 0 else []
        if len(teacher_prompt_ids) > teacher_prompt_budget:
            teacher_prompt_ids = teacher_prompt_ids[-teacher_prompt_budget:] if teacher_prompt_budget > 0 else []
        target_budget = min(self.max_length - len(student_prompt_ids), self.max_teacher_length - len(teacher_prompt_ids))
        target_ids = target_ids[: max(1, target_budget)]

        input_ids = student_prompt_ids + target_ids
        labels = [-100] * len(student_prompt_ids) + target_ids
        attention_mask = [1] * len(input_ids)
        input_ids = pad(input_ids, self.max_length, self.tokenizer.pad_token_id)
        labels = pad(labels, self.max_length, -100)
        attention_mask = pad(attention_mask, self.max_length, 0)

        teacher_input_ids = teacher_prompt_ids + target_ids
        teacher_labels = [-100] * len(teacher_prompt_ids) + target_ids
        teacher_attention_mask = [1] * len(teacher_input_ids)
        teacher_input_ids = pad(teacher_input_ids, self.max_teacher_length, self.tokenizer.pad_token_id)
        teacher_labels = pad(teacher_labels, self.max_teacher_length, -100)
        teacher_attention_mask = pad(teacher_attention_mask, self.max_teacher_length, 0)

        metadata = {
            "sample_id": row.get("sample_id"),
            "source": row.get("source"),
            "history_policy": policy,
            "compressed_history_doc_count": len(compressed_docs),
            "full_history_doc_indices": sorted(full_indices),
            "context_ratios": ratios,
            "target": row.get("target", ""),
        }
        return {
            "system_input_ids": system_input_ids,
            "context_input_ids": context_input_ids,
            "context_ratios": context_ratios[: self.max_history_docs],
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "teacher_input_ids": teacher_input_ids,
            "teacher_labels": teacher_labels,
            "teacher_attention_mask": teacher_attention_mask,
            "metadata_json": json.dumps(metadata, ensure_ascii=False),
        }


def collate(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch: Dict[str, Any] = {}
    keys = [k for k in features[0] if k != "metadata_json"]
    for key in keys:
        batch[key] = torch.tensor([f[key] for f in features], dtype=torch.long)
    batch["metadata_json"] = [f["metadata_json"] for f in features]
    return batch


class AgentHistoryKDTrainer(GistMultiDocTrainer):
    def __init__(self, *args, tokenizer=None, kd_coef=0.7, ce_coef=0.3, kd_temperature=2.0, teacher_attn_impl="sdpa", **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.kd_coef = float(kd_coef)
        self.ce_coef = float(ce_coef)
        self.kd_temperature = float(kd_temperature)
        self.teacher_attn_impl = teacher_attn_impl
        self.best_val_loss = math.inf
        self.best_kd = math.inf
        self._eval_accum: Optional[Dict[str, list[float]]] = None
        self.log_data.update({"kd_loss": [], "ce_loss": [], "target_token_agreement": [], "first_token_agreement": [], "next_token_kl": []})

    def _adapter_state_dict(self) -> Dict[str, torch.Tensor]:
        model = self._unwrap_model(self.model)
        return {name: param.detach().cpu() for name, param in model.named_parameters() if "gist" in name}

    def _save_adapter(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self._adapter_state_dict(), output_dir / "c2kv_adapter.bin")
        (output_dir / "agent_history_training_config.json").write_text(json.dumps({"kd_coef": self.kd_coef, "ce_coef": self.ce_coef, "kd_temperature": self.kd_temperature}, indent=2) + "\n", encoding="utf-8")
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

    def _save(self, output_dir: Optional[str] = None, state_dict=None) -> None:
        self._save_adapter(output_dir or self.args.output_dir)

    def evaluate(self, *args, **kwargs):
        self._eval_accum = defaultdict(list)
        metrics = super().evaluate(*args, **kwargs)
        if self._eval_accum:
            for key, values in self._eval_accum.items():
                if values:
                    metrics[f"eval_{key}"] = float(sum(values) / len(values))
        self._eval_accum = None
        if metrics.get("eval_loss", math.inf) < self.best_val_loss:
            self.best_val_loss = metrics["eval_loss"]
            self._save_adapter(Path(self.args.output_dir) / "best_val_loss")
        if metrics.get("eval_kd_loss", math.inf) < self.best_kd:
            self.best_kd = metrics["eval_kd_loss"]
            self._save_adapter(Path(self.args.output_dir) / "best_kd")
        return metrics

    def _record_metric(self, name: str, value: torch.Tensor | float) -> None:
        if isinstance(value, torch.Tensor):
            tensor = value.detach()
            self.log_data[name].append(tensor if tensor.dim() == 0 else tensor.mean())
            scalar = float((tensor if tensor.dim() == 0 else tensor.mean()).float().cpu().item())
        else:
            scalar = float(value)
            self.log_data[name].append(torch.tensor(scalar, device=self.args.device))
        if self._eval_accum is not None:
            self._eval_accum[name].append(scalar)

    def _tool_metrics(self, labels: torch.Tensor, teacher_logits: torch.Tensor, student_logits: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
        if self.tokenizer is None or labels.shape[0] == 0:
            return {}
        counts = Counter()
        sums = Counter()
        pred_teacher = teacher_logits.argmax(dim=-1)
        pred_student = student_logits.argmax(dim=-1)
        for i in range(labels.shape[0]):
            pos = mask[i].nonzero(as_tuple=False).squeeze(1)
            if pos.numel() == 0:
                continue
            target_text = self.tokenizer.decode(labels[i, pos].tolist(), skip_special_tokens=True)
            if extract_tool_call(target_text) is None:
                continue
            counts["tool_targets"] += 1
            try:
                teacher_call = extract_tool_call(self.tokenizer.decode(pred_teacher[i, pos].tolist(), skip_special_tokens=True))
                student_call = extract_tool_call(self.tokenizer.decode(pred_student[i, pos].tolist(), skip_special_tokens=True))
            except Exception:
                counts["parser_errors"] += 1
                continue
            if not teacher_call or not student_call:
                counts["parser_errors"] += 1
                continue
            sums["tool_name_agreement"] += float(teacher_call.get("name") == student_call.get("name"))
            sums["tool_call_exact_match"] += float(teacher_call == student_call)
            t_names, t_values = flatten_arg_items(teacher_call.get("arguments"))
            s_names, s_values = flatten_arg_items(student_call.get("arguments"))
            sums["argument_name_f1"] += f1(t_names, s_names)
            sums["argument_value_f1"] += f1(t_values, s_values)
        if not counts["tool_targets"]:
            return {"tool_parser_errors": float(counts["parser_errors"])}
        denom = counts["tool_targets"]
        return {k: float(v / denom) for k, v in sums.items()} | {"tool_parser_errors": float(counts["parser_errors"])}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        metadata_json = inputs.pop("metadata_json", None)
        teacher_input_ids = inputs.pop("teacher_input_ids")
        teacher_attention_mask = inputs.pop("teacher_attention_mask")
        teacher_labels = inputs.pop("teacher_labels")
        context_ratios = inputs.pop("context_ratios")
        batch_size = inputs["input_ids"].shape[0]

        teacher_active = teacher_attention_mask.sum(dim=1)
        teacher_max_len = int(teacher_active.max().item())
        teacher_input_ids = teacher_input_ids[:, :teacher_max_len]
        teacher_attention_mask = teacher_attention_mask[:, :teacher_max_len]
        teacher_labels = teacher_labels[:, :teacher_max_len]
        teacher_label_mask = teacher_labels != -100
        teacher_first = teacher_label_mask.float().argmax(dim=1)
        teacher_logits_start = max(0, int(teacher_first[teacher_label_mask.any(dim=1)].min().item()) - 1)
        inner_model = self._inner_model(model)
        original_attn_impl = inner_model.config._attn_implementation
        with torch.no_grad():
            was_training = model.training
            model.eval()
            if self.teacher_attn_impl:
                inner_model.config._attn_implementation = self.teacher_attn_impl
            teacher_outputs = model(
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask,
                use_cache=False,
                logits_to_keep=teacher_max_len - teacher_logits_start,
            )
            inner_model.config._attn_implementation = original_attn_impl
            if was_training:
                model.train()
        teacher_logits = teacher_outputs.logits.detach()
        del teacher_outputs
        teacher_labels = teacher_labels[:, teacher_logits_start:]

        context_masks = inputs["context_input_ids"] != -100
        system_input_ids = inputs.pop("system_input_ids")
        system_kv, system_mask, past_length = self._build_system_kv(model, system_input_ids)
        context_input_ids = inputs["context_input_ids"].reshape((batch_size, -1, self.max_doc_length))
        context_ratios = context_ratios.reshape((batch_size, -1))
        doc_lengths = (context_input_ids != -100).sum(dim=2)
        max_doc_active_length = int(doc_lengths.max().item()) if doc_lengths.numel() else 0
        if 0 < max_doc_active_length < self.max_doc_length:
            context_input_ids = context_input_ids[:, :, :max_doc_active_length]
        inputs["context_input_ids"] = context_input_ids
        inputs["context_ratios"] = context_ratios
        inputs["past_key_values"] = system_kv
        inputs["past_attention_mask"] = system_mask

        active_lengths = inputs["attention_mask"].sum(dim=1)
        input_length = int(active_lengths.max().item())
        inputs["input_ids"] = inputs["input_ids"][:, :input_length]
        inputs["attention_mask"] = inputs["attention_mask"][:, :input_length]
        inputs["labels"] = inputs["labels"][:, :input_length]
        position_ids = torch.arange(input_length, dtype=torch.long, device=inputs["input_ids"].device).unsqueeze(0).repeat(batch_size, 1)
        for i, seqlen in enumerate(context_masks.sum(dim=1).tolist()):
            position_ids[i] += past_length + seqlen
        inputs["position_ids"] = position_ids

        label_mask = inputs["labels"] != -100
        if not label_mask.any():
            raise ValueError("Batch has no supervised target tokens")
        first_label_positions = label_mask.float().argmax(dim=1)
        logits_start = max(0, int(first_label_positions[label_mask.any(dim=1)].min().item()) - 1)
        if logits_start > 0:
            inputs["labels"] = inputs["labels"][:, logits_start:]
            inputs["logits_to_keep"] = input_length - logits_start

        self._inner_model(model).config._attn_implementation = self._gist_attn_impl()
        outputs = model(**inputs)
        ce_loss = _as_scalar_loss(outputs.loss)

        student_logits = outputs.logits
        student_labels = inputs["labels"]
        common_len = min(student_logits.shape[1], teacher_logits.shape[1], student_labels.shape[1], teacher_labels.shape[1])
        student_logits = student_logits[:, :common_len]
        teacher_logits = teacher_logits[:, :common_len]
        student_labels = student_labels[:, :common_len]
        teacher_labels = teacher_labels[:, :common_len]
        kd_mask = (student_labels != -100) & (teacher_labels != -100)
        if not kd_mask.any():
            kd_loss = ce_loss.new_zeros(())
            agreement = ce_loss.new_zeros(())
            first_agreement = ce_loss.new_zeros(())
            next_kl = ce_loss.new_zeros(())
        else:
            t = self.kd_temperature
            kd_loss = F.kl_div(
                F.log_softmax(student_logits[kd_mask] / t, dim=-1),
                F.softmax(teacher_logits[kd_mask] / t, dim=-1),
                reduction="batchmean",
            ) * (t ** 2)
            teacher_argmax = teacher_logits.argmax(dim=-1)
            student_argmax = student_logits.argmax(dim=-1)
            agreement = (teacher_argmax[kd_mask] == student_argmax[kd_mask]).float().mean()
            first_values = []
            next_kls = []
            for i in range(kd_mask.shape[0]):
                pos = kd_mask[i].nonzero(as_tuple=False).squeeze(1)
                if pos.numel() == 0:
                    continue
                p = int(pos[0].item())
                first_values.append(float(teacher_argmax[i, p] == student_argmax[i, p]))
                next_kls.append(F.kl_div(F.log_softmax(student_logits[i, p] / t, dim=-1), F.softmax(teacher_logits[i, p] / t, dim=-1), reduction="sum") * (t ** 2))
            first_agreement = agreement.new_tensor(sum(first_values) / max(1, len(first_values)))
            next_kl = torch.stack(next_kls).mean() if next_kls else kd_loss.detach()

        loss = self.kd_coef * kd_loss + self.ce_coef * ce_loss
        self._record_metric("kd_loss", kd_loss)
        self._record_metric("ce_loss", ce_loss)
        self._record_metric("target_token_agreement", agreement)
        self._record_metric("first_token_agreement", first_agreement)
        self._record_metric("next_token_kl", next_kl)
        if self._eval_accum is not None:
            for key, value in self._tool_metrics(student_labels.detach().cpu(), teacher_logits.detach().cpu(), student_logits.detach().cpu(), kd_mask.detach().cpu()).items():
                self._eval_accum[key].append(value)
        return (loss, outputs) if return_outputs else loss




def parse_unknown_overrides(items: Sequence[str]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    i = 0
    while i < len(items):
        item = items[i]
        if not item.startswith("--"):
            i += 1
            continue
        key = item[2:].replace("-", "_")
        if i + 1 >= len(items) or items[i + 1].startswith("--"):
            overrides[key] = True
            i += 1
            continue
        raw = items[i + 1]
        low = raw.lower()
        if low in {"true", "false"}:
            value: Any = low == "true"
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
        overrides[key] = value
        i += 2
    return overrides


def load_adapter_if_needed(model, checkpoint: Optional[str]) -> None:
    if not checkpoint:
        return
    ckpt = Path(checkpoint)
    adapter_file = ckpt / "c2kv_adapter.bin" if ckpt.is_dir() else ckpt
    if not adapter_file.exists():
        return
    state = torch.load(adapter_file, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded adapter from {adapter_file}; missing={len(missing)} unexpected={len(unexpected)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--train_data", default=None)
    parser.add_argument("--eval_data", default=None)
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--dry_run_one_batch", action="store_true")
    args, unknown = parser.parse_known_args()
    cfg = load_config(args.config)
    cfg.update(parse_unknown_overrides(unknown))
    for key, value in vars(args).items():
        if value is not None and value is not False:
            cfg[key] = value
    cfg.setdefault("train_data", "datasets/processed_agent_history_c2kv/train.jsonl")
    cfg.setdefault("eval_data", "datasets/processed_agent_history_c2kv/val.jsonl")
    cfg.setdefault("model_name_or_path", "models/Qwen3-4B-Instruct-2507")
    cfg.setdefault("output_dir", "checkpoints/qwen3-4b-agent-history-c2kv-248")
    cfg.setdefault("seed", 42)
    resume_from_checkpoint = cfg.get("resume_from_checkpoint")
    hf_resume_from_checkpoint = resume_from_checkpoint
    if resume_from_checkpoint:
        ckpt = Path(resume_from_checkpoint)
        has_adapter = (ckpt / "c2kv_adapter.bin").exists() if ckpt.is_dir() else ckpt.exists()
        has_hf_model = any((ckpt / name).exists() for name in ["pytorch_model.bin", "model.safetensors", "adapter_model.bin"] ) if ckpt.is_dir() else False
        if has_adapter and not has_hf_model:
            hf_resume_from_checkpoint = None
            print(f"Adapter-style checkpoint detected at {ckpt}; loading C2KV weights manually and starting a fresh Trainer loop.")
    set_seed(int(cfg["seed"]))
    os.environ.setdefault("C2KV_GIST_TRAIN_RATIOS", "2,4,8")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    model_args = ModelArgs(
        model_name_or_path=cfg["model_name_or_path"],
        padding_side="right",
        attn_impl=cfg.get("attn_impl", "flex_attention"),
        dtype=cfg.get("dtype", "bf16"),
        enable_gist=True,
        gist_param="qkv",
        gist_type="dynamic-interleave",
        gist_overlap=int(cfg.get("gist_overlap", 64)),
        gist_residual_type=cfg.get("gist_residual_type", "embed-mean"),
        gist_gradient_checkpointing=bool(cfg.get("gist_gradient_checkpointing", False)),
    )
    if model_args.gist_gradient_checkpointing:
        import models.gist_utils as gist_utils

        gist_utils.GIST_GRADIENT_CHECKPOINTING = True
    model, tokenizer = get_model_and_tokenizer(model_args, device=device, evaluation_mode=False)
    model.requires_grad_(False)
    for name, param in model.named_parameters():
        param.requires_grad_("gist" in name)
    non_gist_trainable = [name for name, param in model.named_parameters() if param.requires_grad and "gist" not in name]
    assert not non_gist_trainable, f"Base Qwen params unexpectedly trainable: {non_gist_trainable[:10]}"
    trainable = [(name, p.numel()) for name, p in model.named_parameters() if p.requires_grad]
    if local_rank == 0:
        print("Trainable parameters:")
        for name, numel in trainable:
            print(f"  {name}: {numel}")
        print(f"Total trainable: {format_numel_str(sum(n for _, n in trainable))}")

    load_adapter_if_needed(model, resume_from_checkpoint)
    train_dataset = AgentHistoryJsonlDataset(
        cfg["train_data"], tokenizer,
        max_history_docs=int(cfg.get("max_history_docs", 16)),
        max_doc_length=int(cfg.get("max_doc_length", 1024)),
        max_length=int(cfg.get("max_length", 2048)),
        max_system_length=int(cfg.get("max_system_length", 4096)),
        max_teacher_length=int(cfg.get("max_teacher_length", 24576)),
        max_target_length=int(cfg.get("max_target_length", 256)),
        history_policy_probs=cfg.get("history_policy_probs", "all_compressed:0.45,recent1_full:0.20,recent2_full:0.15,random1_full:0.10,random2_full:0.10"),
        ratio_probs=cfg.get("ratio_probs", "2:0.333333,4:0.333333,8:0.333334"),
        heterogeneous_ratio_prob=float(cfg.get("heterogeneous_ratio_prob", 0.20)),
        seed=int(cfg.get("seed", 42)),
        max_precomputed_total_tokens=cfg.get("max_precomputed_total_tokens"),
        max_precomputed_tools_tokens=cfg.get("max_precomputed_tools_tokens"),
        max_precomputed_current_tokens=cfg.get("max_precomputed_current_tokens"),
    )
    if local_rank == 0:
        print(f"Train dataset filter stats: {dict(train_dataset.filter_stats)}")
    eval_dataset = AgentHistoryJsonlDataset(
        cfg["eval_data"], tokenizer,
        max_history_docs=int(cfg.get("max_history_docs", 16)),
        max_doc_length=int(cfg.get("max_doc_length", 1024)),
        max_length=int(cfg.get("max_length", 2048)),
        max_system_length=int(cfg.get("max_system_length", 4096)),
        max_teacher_length=int(cfg.get("max_teacher_length", 24576)),
        max_target_length=int(cfg.get("max_target_length", 256)),
        history_policy_probs=cfg.get("history_policy_probs", "all_compressed:0.45,recent1_full:0.20,recent2_full:0.15,random1_full:0.10,random2_full:0.10"),
        ratio_probs=cfg.get("ratio_probs", "2:0.333333,4:0.333333,8:0.333334"),
        heterogeneous_ratio_prob=float(cfg.get("heterogeneous_ratio_prob", 0.20)),
        seed=int(cfg.get("seed", 42)) + 999,
        max_precomputed_total_tokens=cfg.get("max_precomputed_total_tokens"),
        max_precomputed_tools_tokens=cfg.get("max_precomputed_tools_tokens"),
        max_precomputed_current_tokens=cfg.get("max_precomputed_current_tokens"),
    )
    if local_rank == 0:
        print(f"Eval dataset filter stats: {dict(eval_dataset.filter_stats)}")
    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=float(cfg.get("num_train_epochs", 1)),
        max_steps=int(cfg.get("max_steps", -1)),
        per_device_train_batch_size=int(cfg.get("per_device_batch_size", 1)),
        per_device_eval_batch_size=int(cfg.get("per_device_eval_batch_size", cfg.get("per_device_batch_size", 1))),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(cfg.get("learning_rate", 5e-7)),
        weight_decay=float(cfg.get("weight_decay", 0.1)),
        warmup_steps=int(cfg.get("warmup_steps", 20)),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        logging_steps=int(cfg.get("logging_steps", 1)),
        eval_strategy=cfg.get("eval_strategy", "steps"),
        eval_steps=int(cfg.get("eval_steps", 50)),
        save_strategy=cfg.get("save_strategy", "steps"),
        save_steps=int(cfg.get("save_steps", 100)),
        save_total_limit=int(cfg.get("save_total_limit", 3)),
        bf16=bool(cfg.get("bf16", True)),
        remove_unused_columns=False,
        dataloader_num_workers=int(cfg.get("dataloader_num_workers", 2)),
        logging_nan_inf_filter=False,
        ddp_timeout=int(cfg.get("ddp_timeout", 7200)),
        report_to=cfg.get("report_to", "tensorboard"),
    )
    # GistMultiDocTrainer inherits TrainerDistillMixin, which expects these
    # custom attributes on args. Keep legacy self-distill disabled; this script
    # computes Full-teacher KD explicitly in AgentHistoryKDTrainer.compute_loss.
    training_args.gist_self_distill_coef = None
    training_args.gist_self_distill_temperature = float(cfg.get("gist_self_distill_temperature", 2.0))

    trainer = AgentHistoryKDTrainer(
        model=model,
        args=training_args,
        max_doc_length=int(cfg.get("max_doc_length", 1024)),
        model_args=model_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate,
        kd_coef=float(cfg.get("kd_coef", 0.7)),
        ce_coef=float(cfg.get("ce_coef", 0.3)),
        kd_temperature=float(cfg.get("kd_temperature", 2.0)),
        teacher_attn_impl=cfg.get("teacher_attn_impl", "sdpa"),
    )
    if cfg.get("dry_run_one_batch", False):
        batch = collate([train_dataset[0]])
        batch = {k: (v.to(training_args.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        model.train()
        loss = trainer.compute_loss(model, batch)
        loss.backward()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(cfg.get("learning_rate", 5e-7)))
        opt.step()
        opt.zero_grad(set_to_none=True)
        if local_rank == 0:
            print(f"dry_run_one_batch_loss={float(loss.detach().cpu())}")
        return
    trainer.train(resume_from_checkpoint=hf_resume_from_checkpoint)
    trainer._save_adapter(Path(cfg["output_dir"]) / "last")


if __name__ == "__main__":
    main()
