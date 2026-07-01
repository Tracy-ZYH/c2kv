import torch
import itertools
from gist_args import ModelArgs
from torch.utils.data import Sampler
from transformers import DataCollatorWithPadding
from transformers.trainer import Trainer
from transformers.cache_utils import DynamicCache
from typing import Any, Dict, List, Optional, Union, Iterator


class TrainerDistillMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distill_coef: float | None = kwargs["args"].gist_self_distill_coef
        self.log_data: dict[str, list[torch.Tensor]] = {}
        if self.distill_coef is not None:
            assert 0. < self.distill_coef <= 1., "The self_distill_coef should be in (0, 1]!"
            self.kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
            self.distill_temperature = kwargs["args"].gist_self_distill_temperature
            self.log_data.update({"distill_loss": [], "label_loss": []})

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        for loss_name, losses in self.log_data.items():
            if len(losses) > 0:
                loss = self._nested_gather(torch.stack(losses)).mean().item()
                logs[loss_name] = round(loss, 6)
                self.log_data[loss_name] = []
        super().log(logs, start_time)
    
    def apply_distill_loss(self, labels: torch.Tensor, label_loss: torch.Tensor,
        teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
        label_mask = labels != -100
        distill_loss: torch.Tensor = self.kl_loss(
            torch.log_softmax(student_logits[label_mask] / self.distill_temperature, dim=-1), 
            torch.softmax(teacher_logits[label_mask] / self.distill_temperature, dim=-1)
        ) * (self.distill_temperature ** 2)
        self.log_data["distill_loss"].append(distill_loss.detach())
        self.log_data["label_loss"].append(label_loss.detach())
        loss = (1 - self.distill_coef) * label_loss + self.distill_coef * distill_loss
        return loss


class GistPretrainTrainer(TrainerDistillMixin, Trainer):
    def __init__(self, *args, model_args: ModelArgs, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_args = model_args
        gist_mode_args = model_args.gist_mode.split("-")
        self.gist_chunk_size = list(map(int, gist_mode_args[0].split(",")))
        self.gist_max_chunk_num = int(gist_mode_args[1])
        self.gist_max_chunk_size = max(self.gist_chunk_size)
        if self.model_args.gist_reconstruct_loss_coef is not None:
            self.log_data.update({"compress_loss": []})

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None
    ):
        """
        Override the default compute_loss to process inputs 
        """
        batch_size, max_seq_len = inputs["input_ids"].shape
        # sample a gist chunk size
        min_seq_len = inputs["attention_mask"].sum(dim=1).min().item()
        chunk_sizes = [size for size in self.gist_chunk_size if size < min_seq_len]
        assert len(chunk_sizes) > 0, "The minimum sequence length is less than the gist chunk size!"
        gist_chunk_size = chunk_sizes[min_seq_len % len(chunk_sizes)] # pseudo-random
        # gist_chunk_size = max(chunk_sizes)
        num_chunk = min((min_seq_len - 1) // gist_chunk_size, self.gist_max_chunk_num)
        context_len = gist_chunk_size * num_chunk
        if not self.model_args.enable_gist:
            labels = inputs["input_ids"].clone()
            labels[~inputs["attention_mask"].bool()] = -100
            labels[:, :context_len] = -100
            inputs["labels"] = labels
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        if model.training and self.distill_coef is not None:
            with torch.no_grad():
                inputs_len = max_seq_len - context_len
                self_distill_logits = model(logits_to_keep=inputs_len, use_cache=False, **inputs).logits
        # split inputs_ids into context and input_ids
        context_input_ids = inputs["input_ids"][:, :context_len].reshape((batch_size, num_chunk, gist_chunk_size))
        inputs["context_input_ids"] = context_input_ids
        inputs["input_ids"] = inputs["input_ids"][:, context_len:]
        inputs["attention_mask"] = inputs["attention_mask"][:, context_len:].bool()
        # prepare position_ids
        position_ids = torch.arange(context_len, max_seq_len, dtype=torch.long, device=inputs["input_ids"].device)
        inputs["position_ids"] = position_ids.unsqueeze(0).expand(batch_size, -1)
        # prepare labels
        labels = inputs["input_ids"].clone()
        labels[~inputs["attention_mask"]] = -100
        inputs["labels"] = labels
        inputs["reconstruct_loss_coef"] = self.model_args.gist_reconstruct_loss_coef
        loss, outputs = super().compute_loss(model, inputs, True, num_items_in_batch)
        if self.model_args.gist_reconstruct_loss_coef is not None:
            self.log_data["compress_loss"].append(outputs["reconstruct_loss"].detach())
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        attn_impl = model.model.config._attn_implementation
        model.model.config._attn_implementation = "sdpa"
        pred = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        model.model.config._attn_implementation = attn_impl
        return pred


class GistSFTTrainer(Trainer):
    def __init__(self, *args, model_args: ModelArgs, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_args = model_args
        gist_mode_args = model_args.gist_mode.split("-")
        self.gist_chunk_size = list(map(int, gist_mode_args[0].split(",")))
        self.gist_max_chunk_num = int(gist_mode_args[1])
        self.gist_max_chunk_size = max(self.gist_chunk_size)

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None
    ):
        """
        Override the default compute_loss to process inputs 
        """
        if not self.model_args.enable_gist:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        batch_size, max_seq_len = inputs["input_ids"].shape
        # sample a gist chunk size
        seq_lens = inputs["attention_mask"].sum(dim=1)
        response_lens = (inputs["labels"] != -100).sum(dim=1)
        min_prompt_len = (seq_lens - response_lens).min().item()
        chunk_sizes = [size for size in self.gist_chunk_size if size < min_prompt_len]
        assert len(chunk_sizes) > 0, "The minimum sequence length is less than the gist chunk size!"
        gist_chunk_size = chunk_sizes[min_prompt_len % len(chunk_sizes)] # pseudo-random
        # gist_chunk_size = max(chunk_sizes)
        num_chunk = min((min_prompt_len - 1) // gist_chunk_size, self.gist_max_chunk_num)
        context_len = gist_chunk_size * num_chunk
        # split inputs_ids into context and input_ids
        context_input_ids = inputs["input_ids"][:, :context_len].reshape((batch_size, num_chunk, gist_chunk_size))
        inputs["context_input_ids"] = context_input_ids
        inputs["input_ids"] = inputs["input_ids"][:, context_len:]
        inputs["labels"] = inputs["labels"][:, context_len:]
        inputs["attention_mask"] = inputs["attention_mask"][:, context_len:].bool()
        # prepare position_ids
        position_ids = torch.arange(context_len, max_seq_len, dtype=torch.long, device=inputs["input_ids"].device)
        inputs["position_ids"] = position_ids.unsqueeze(0).expand(batch_size, -1)
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
    
    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        attn_impl = model.model.config._attn_implementation
        model.model.config._attn_implementation = "sdpa"
        pred = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        model.model.config._attn_implementation = attn_impl
        return pred


class GistMultiDocTrainer(TrainerDistillMixin, Trainer):
    def __init__(
        self, *args,
        max_doc_length: int,
        model_args: ModelArgs,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model_args = model_args
        self.max_doc_length = max_doc_length
        if self.model_args.gist_reconstruct_loss_coef is not None:
            self.log_data.update({"compress_loss": []})

    @torch.no_grad()
    def _build_system_kv(
        self, model, system_input_ids: torch.Tensor
    ) -> tuple[DynamicCache, torch.Tensor, int]:
        """Build a per-sample system KV cache from variable-length system prompts.

        `system_input_ids` is right-padded with -100. We left-pad the real tokens to a
        uniform width `L_sys` (= batch max real length) so the padded past length is the
        same for every sample, keeping downstream position arithmetic scalar. Padded slots
        are masked out via the returned 2-D `system_mask`.
        """
        device = model.device
        system_input_ids = system_input_ids.to(device)
        real_mask = system_input_ids != -100
        real_lens = real_mask.sum(dim=1)
        batch_size = system_input_ids.shape[0]
        L_sys = int(real_lens.max().item())
        pad_id = model.model.config.pad_token_id
        if pad_id is None:
            pad_id = 0
        left_ids = system_input_ids.new_full((batch_size, L_sys), pad_id)
        system_mask = system_input_ids.new_zeros((batch_size, L_sys))
        for i in range(batch_size):
            n = int(real_lens[i].item())
            if n == 0:
                continue
            left_ids[i, L_sys - n:] = system_input_ids[i][real_mask[i]]
            system_mask[i, L_sys - n:] = 1
        attn_impl = model.model.config._attn_implementation
        model.model.config._attn_implementation = "sdpa"
        was_training = model.training
        model.eval()
        outputs = model(left_ids, attention_mask=system_mask, use_cache=True, logits_to_keep=1)
        model.model.config._attn_implementation = attn_impl
        if was_training:
            model.train()
        return outputs.past_key_values, system_mask, L_sys

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None
    ):
        """
        Override the default compute_loss to process inputs 
        """
        batch_size, doc_total_len = inputs['context_input_ids'].shape
        is_dynamic = bool(inputs.pop('dynamic')[0].item())
        context_masks = inputs['context_input_ids'] != -100
        # build a per-sample system KV cache from variable-length system prompts
        system_input_ids = inputs.pop('system_input_ids')
        system_kv, system_mask, past_length = self._build_system_kv(model, system_input_ids)
        # prepare inputs for gist inference
        if is_dynamic:
            assert batch_size == 1, "dynamic context requires per_device_train_batch_size=1"
        inputs['context_input_ids'] = inputs['context_input_ids'].reshape((batch_size, -1, self.max_doc_length))
        inputs['past_key_values'] = system_kv
        inputs['past_attention_mask'] = system_mask
        input_length = inputs['input_ids'].shape[1]
        position_ids = torch.arange(input_length, dtype=torch.long, device=inputs["input_ids"].device)
        position_ids = position_ids.unsqueeze(0).repeat(batch_size, 1)
        for i, seqlen in enumerate(context_masks.sum(dim=1).tolist()):
            position_ids[i] += past_length + seqlen
        inputs["position_ids"] = position_ids
        inputs["reconstruct_loss_coef"] = self.model_args.gist_reconstruct_loss_coef
        loss, outputs = super().compute_loss(model, inputs, True, num_items_in_batch)
        if self.model_args.gist_reconstruct_loss_coef is not None and model.training:
            self.log_data["compress_loss"].append(outputs["reconstruct_loss"].detach())
        if model.training and self.distill_coef is not None:
            loss = self.apply_distill_loss(inputs["labels"], loss, self_distill_logits, outputs["logits"])
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        attn_impl = model.model.config._attn_implementation
        model.model.config._attn_implementation = "sdpa"
        pred = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        model.model.config._attn_implementation = attn_impl
        return pred
