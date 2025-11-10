import os
import datasets
import glob
from itertools import repeat
from functools import partial
from transformers import AutoTokenizer
from typing import Dict, List, Any, Mapping
from logging import getLogger

logger = getLogger(__name__)


def add_eos(inputs: Mapping, eos_token_id: int):
    """Add eos for BatchEncoding object."""
    assert isinstance(inputs["input_ids"], list), f"Make sure the return_tensors are set to list!"
    if inputs["input_ids"][-1] != eos_token_id:
        for k, v in inputs.items():
            if k in ["input_ids", "labels"]:
                v = v + [eos_token_id]
            elif k == "attention_mask":
                v = v + [1]
            elif k == "position_ids":
                v = v + [v[-1] + 1]
            elif k == "token_type_ids":
                v = v + v[-1:]
            else:
                raise NotImplementedError(f"Inputs key {k} not implemented!")
            inputs[k] = v
    return inputs

def _preprocess_pretrain_data(
    data: Dict[str, List[Any]], 
    tokenizer: AutoTokenizer,
    min_length: int,
    max_length: int,
) -> Dict[str, List[Any]]:
    outputs = {'input_ids': [], "length": []}
    for encoded in map(tokenizer, data['text']):  # ignore max model input length warning here
        seq_len = len(encoded["input_ids"])
        for start in range(0, seq_len - min_length, max_length):
            chunk_len = min(max_length, seq_len - start)
            for k, v in encoded.items():
                if k in outputs:
                    outputs[k].append(v[start:start + chunk_len])
            outputs["length"].append(chunk_len)
    return outputs


class PretrainDataset:
    def __init__(self, 
        path: str, 
        tokenizer: AutoTokenizer = None, 
        split: str = 'train', 
        shuffle_seed: int = 42,
        min_length: int = 1024,
        max_length: int = 4096,
        **kwargs: Any,
    ):
        # dataset = datasets.load_dataset(path, split=split, streaming=True)
        # NOTE: package datasets is modified to avoid globbing on nas, which costs hours
        data_files = [ 
            file for file in glob.iglob(os.path.join(path, split, '**'), recursive=True)
            if '.' in os.path.basename(file)
        ]
        dataset = datasets.load_dataset(path, data_files=data_files, streaming=True)['train']
        self.dataset = dataset.map(
            _preprocess_pretrain_data,
            fn_kwargs={
                'tokenizer': tokenizer,
                'min_length': min_length,
                'max_length': max_length,
            },
            batched=True, batch_size=32,
            remove_columns=['text', 'meta'],
        ).shuffle(seed=shuffle_seed)
        self.iterator = iter(self.dataset)
    
    def __iter__(self):
        assert False, "PretrainDataset is not iterable"
        return iter(self.dataset)

    def __getitem__(self, index):
        return next(self.iterator)


def get_dataset(type: str, path: str, tokenizer: AutoTokenizer, **kwargs):
    for k, v in kwargs.items():
        if v is None:
            kwargs.pop(k)
    if type == "pretrain":
        return PretrainDataset(path, tokenizer, **kwargs)
    return None
