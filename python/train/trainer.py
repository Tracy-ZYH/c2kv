import torch
import itertools
from random import choice as rand_choice
from gist_args import ModelArgs
from torch.utils.data import Sampler
from transformers import DataCollatorWithPadding
from transformers.trainer import Trainer
from typing import Any, Dict, List, Optional, Union, Iterator


class GistTrainer(Trainer):
    def __init__(self, *args, model_args: ModelArgs, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_args = model_args
        gist_mode_args = model_args.gist_mode.split("-")
        self.gist_chunk_size = list(map(int, gist_mode_args[0].split(",")))
        self.gist_max_chunk_num = int(gist_mode_args[1])

    def compute_loss(
        self, model, inputs, num_items_in_batch, return_outputs=False):
        """
        Override the default compute_loss to process inputs 
        """
        batch_size, max_seq_len = inputs["input_ids"].shape
        # sample a gist chunk size
        min_seq_len = inputs["attention_mask"].sum(dim=1).min().item()
        chunk_sizes = [size for size in self.gist_chunk_size if size < min_seq_len]
        assert len(chunk_sizes) > 0, "The minimum sequence length is less than the gist chunk size!"
        gist_chunk_size = rand_choice(chunk_sizes)
        # gist_chunk_size = ((min_seq_len - 1) // self.gist_chunk_size) * self.gist_chunk_size # !! stage 1
        num_chunk = min((min_seq_len - 1) // gist_chunk_size, self.gist_max_chunk_num)
        context_len = gist_chunk_size * num_chunk
        # split inputs_ids into context and input_ids
        inputs["context_input_ids"] = inputs["input_ids"][:, :context_len].reshape((batch_size, num_chunk, gist_chunk_size))
        # inputs["context_input_ids"] = [context for context in inputs["context_input_ids"]]
        inputs["input_ids"] = inputs["input_ids"][:, context_len:]
        inputs["attention_mask"] = inputs["attention_mask"][:, context_len:].bool()
        # prepare position_ids
        position_ids = torch.arange(context_len, max_seq_len, dtype=torch.long, device=inputs["input_ids"].device)
        inputs["position_ids"] = position_ids.unsqueeze(0).repeat(batch_size, 1)
        # prepare labels
        labels = inputs["input_ids"].clone()
        labels[~inputs["attention_mask"]] = -100
        inputs["labels"] = labels
        outputs = super().compute_loss(model, inputs, return_outputs)
        return outputs
    
    def _get_train_sampler(self, data_source) -> Sampler:
        return InifiniteSampler()


class InifiniteSampler(Sampler):
    def __iter__(self) -> Iterator[int]:
        return itertools.count(start=0)
