import datasets
from random import randint
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
    indices: List[int],
    tokenizer: AutoTokenizer,
    min_length: int,
    max_length: int,
) -> Dict[str, List[Any]]:
    outputs = {'input_ids': [], "length": [], "index": []}
    for index, text in zip(indices, data['text']):
        # truncate text for faster processing
        encoded = tokenizer(text)
        if len(encoded["input_ids"]) < min_length:
            continue
        if len(encoded['input_ids']) < max_length:
            encoded = add_eos(encoded, tokenizer.eos_token_id)
        else: # sample a substring from the text
            start = randint(0, len(encoded['input_ids']) - max_length)
            encoded = {k: v[start:start+max_length] for k, v in encoded.items()}
        # encoded["labels"] = encoded["input_ids"]
        for k, v in encoded.items():
            if k in outputs:
                outputs[k].append(v)
        # length is required for grouping
        outputs["length"].append(len(encoded['input_ids']))
        outputs["index"].append(index)
    return outputs


class PretrainDataset:
    def __init__(self, path: str, tokenizer: AutoTokenizer = None, shuffle_seed: int = 42):
        dataset = datasets.load_dataset(path, split='train', streaming=True)
        dataset.shuffle(seed=shuffle_seed)
        self.dataset = dataset.map(
            _preprocess_pretrain_data,
            fn_kwargs={
                'tokenizer': tokenizer,
                'min_length': 1024,
                'max_length': 4096,
            },
            batched=True, batch_size=32, with_indices=True,
            remove_columns=['text', 'meta'],
        )
        self.iterator = iter(self.dataset)
    
    def __iter__(self):
        assert False, "PretrainDataset is not iterable"
        return iter(self.dataset)

    def __getitem__(self, index):
        return next(self.iterator)


def get_training_dataset(type: str, path: str, tokenizer: AutoTokenizer):
    if type == "pretrain":
        return PretrainDataset(path, tokenizer)
    return None
