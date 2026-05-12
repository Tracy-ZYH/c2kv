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
        if model.training and self.distill_coef is not None:
            loss = self.apply_distill_loss(labels, loss, self_distill_logits, outputs["logits"])
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
        system_ids: List[int],
        max_doc_length: int,
        model_args: ModelArgs, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model_args = model_args
        self.system_ids = torch.tensor(system_ids, dtype=torch.long).unsqueeze(0)
        self.system_kv: Optional[DynamicCache] = None
        self.max_doc_length = max_doc_length
        if self.model_args.gist_reconstruct_loss_coef is not None:
            self.log_data.update({"compress_loss": []})
    
    @torch.no_grad()
    def _get_system_kv(self, model, batch_size: int) -> DynamicCache | None:
        if self.system_kv == None: # if not initialized
            attn_impl = model.model.config._attn_implementation
            model.model.config._attn_implementation = "sdpa"
            model.eval()
            outputs = model(self.system_ids.to(model.device), use_cache=True)
            model.model.config._attn_implementation = attn_impl
            model.train()
            self.system_kv = outputs.past_key_values
            self.past_length = self.system_kv.get_seq_length()
        # make a copy of the system_kv, repeated to batch_size
        system_kv = []
        for layer in self.system_kv.layers:
            keys, values = layer.keys, layer.values
            system_kv.append((keys.expand(batch_size, -1, -1, -1), values.expand(batch_size, -1, -1, -1)))
        return DynamicCache(ddp_cache_data=system_kv, config=model.model.config)

    def prepare_vanilla_inputs(
        self,
        inputs: dict[str, torch.Tensor],
        context_masks: torch.Tensor,
        past_key_values: DynamicCache | None,
    ) -> dict[str, torch.Tensor]:
        context_input_ids, query_ids = inputs["context_input_ids"], inputs["input_ids"]
        batch_size, context_length = context_input_ids.shape
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        context_position_ids = torch.arange(
            past_length, past_length + context_length, dtype=torch.long, device=context_input_ids.device
        ).unsqueeze(0).expand(batch_size, -1)
        query_position_ids = torch.arange(
            query_ids.shape[1], dtype=torch.long, device=context_input_ids.device
        ).unsqueeze(0).repeat(batch_size, 1)
        context_ids = context_input_ids.new_zeros(context_input_ids.shape)
        context_attention_mask = inputs["attention_mask"].new_zeros(context_input_ids.shape)
        past_attention_mask = inputs["attention_mask"].new_ones((1, 1)).expand(batch_size, past_length)
        for i, mask in enumerate(context_masks):
            seq_len = mask.sum().item()
            context_ids[i, :seq_len] = context_input_ids[i, mask]
            context_attention_mask[i, :seq_len] = 1
            query_position_ids[i] += seq_len + past_length
        return {
            "input_ids": torch.cat([context_ids, query_ids], dim=1),
            "attention_mask": torch.cat([past_attention_mask, context_attention_mask, inputs["attention_mask"]], dim=1),
            "position_ids": torch.cat([context_position_ids, query_position_ids], dim=1),
            "past_key_values": past_key_values,
            "logits_to_keep": query_ids.shape[1],
            "use_cache": past_key_values is not None,
        }

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
        context_masks = inputs['context_input_ids'] != -100
        # trim the context_input_ids if the last half is empty
        # half_doc_total_len = doc_total_len // 2
        # if not context_masks[:, half_doc_total_len:].any():
        #     inputs['context_input_ids'] = inputs['context_input_ids'][:, :half_doc_total_len]
        #     context_masks = context_masks[:, :half_doc_total_len]
        # special cases: no gist inference
        system_kv: DynamicCache = self._get_system_kv(model, batch_size)
        if not self.model_args.enable_gist:
            vanilla_inputs = self.prepare_vanilla_inputs(inputs, context_masks, system_kv)
            vanilla_inputs['labels'] = inputs['labels']
            return super().compute_loss(model, vanilla_inputs, return_outputs, num_items_in_batch)
        if model.training and self.distill_coef is not None:
            vanilla_inputs = self.prepare_vanilla_inputs(inputs, context_masks, system_kv)
            with torch.no_grad():
                self_distill_logits = model(**vanilla_inputs).logits
        # prepare inputs for gist inference
        inputs['context_input_ids'] = inputs['context_input_ids'].reshape((batch_size, -1, self.max_doc_length))
        inputs['past_key_values'] = system_kv
        input_length = inputs['input_ids'].shape[1]
        position_ids = torch.arange(input_length, dtype=torch.long, device=inputs["input_ids"].device)
        position_ids = position_ids.unsqueeze(0).repeat(batch_size, 1)
        for i, seqlen in enumerate(context_masks.sum(dim=1).tolist()):
            position_ids[i] += self.past_length + seqlen
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
