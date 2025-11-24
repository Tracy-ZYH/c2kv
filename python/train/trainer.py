import torch
import itertools
from gist_args import ModelArgs
from torch.utils.data import Sampler
from transformers import DataCollatorWithPadding
from transformers.trainer import Trainer
from typing import Any, Dict, List, Optional, Union, Iterator


class GistPretrainTrainer(Trainer):
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
        # split inputs_ids into context and input_ids
        context_input_ids = inputs["input_ids"][:, :context_len].reshape((batch_size, num_chunk, gist_chunk_size))
        inputs["context_input_ids"] = context_input_ids
        inputs["input_ids"] = inputs["input_ids"][:, context_len:]
        inputs["attention_mask"] = inputs["attention_mask"][:, context_len:].bool()
        # prepare position_ids
        position_ids = torch.arange(context_len, max_seq_len, dtype=torch.long, device=inputs["input_ids"].device)
        inputs["position_ids"] = position_ids.unsqueeze(0).repeat(batch_size, 1)
        # prepare labels
        labels = inputs["input_ids"].clone()
        labels[~inputs["attention_mask"]] = -100
        inputs["labels"] = labels
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
    
    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs["labels"] = inputs["input_ids"].clone()
        attn_impl = model.model.config._attn_implementation
        model.model.config._attn_implementation = "sdpa"
        pred = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        model.model.config._attn_implementation = attn_impl
        return pred
    
    def _get_train_sampler(self, data_source) -> Sampler:
        return InifiniteSampler()


class InifiniteSampler(Sampler):
    def __iter__(self) -> Iterator[int]:
        return itertools.count(start=0)


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
        inputs["attention_mask"] = inputs["attention_mask"][:, context_len:].bool()
        # prepare position_ids
        position_ids = torch.arange(context_len, max_seq_len, dtype=torch.long, device=inputs["input_ids"].device)
        inputs["position_ids"] = position_ids.unsqueeze(0).repeat(batch_size, 1)
        # prepare labels
        dst_labels = inputs["input_ids"].clone()
        src_labels = inputs["labels"][:, context_len:]
        dst_labels[:, :src_labels.shape[1]] = src_labels
        inputs["labels"] = dst_labels
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
