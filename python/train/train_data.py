import os
import datasets
import glob
from transformers import AutoTokenizer
from typing import Dict, List, Any, Mapping, Optional, Callable
from logging import getLogger

from inference.mdocdataset import load_mdoc_dataset, QA_SYSTEM_PROMPT

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


class GistDataset:
    def __init__(self, data: datasets.Dataset):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __iter__(self):
        return iter(self.data)


class PretrainDataset(GistDataset):
    def __init__(self, 
        path: str, 
        tokenizer: AutoTokenizer = None, 
        split: str = 'train', 
        shuffle_seed: int = 42,
        min_length: int = 1024,
        max_length: int = 4096,
        num_samples: int = 32768,
        cut_long_seq: bool = False,
    ):
        shuffle_seed += int(os.environ.get("LOCAL_RANK", 0))
        # dataset = datasets.load_dataset(path, split=split, streaming=True)
        # NOTE: package datasets is modified to avoid globbing on nas, which costs hours
        data_files = [ 
            file for file in glob.iglob(os.path.join(path, split, '**'), recursive=True)
            if '.' in os.path.basename(file)
        ]
        self.data = datasets.load_dataset(path, data_files=data_files, streaming=True)['train'].map(
            self._preprocess_pretrain_data,
            fn_kwargs={
                'tokenizer': tokenizer,
                'min_length': min_length,
                'max_length': max_length,
                'cut_long_seq': cut_long_seq
            },
            batched=True, batch_size=32,
            remove_columns=['text', 'meta'],
        ).shuffle(seed=shuffle_seed)
        self.iterator = iter(self.data)
        self.num_samples = num_samples
        self.cached_data: Dict[int, Dict[str, List[Any]]] = {}

    @staticmethod
    def _preprocess_pretrain_data(
        data: Dict[str, List[Any]], 
        tokenizer: AutoTokenizer,
        min_length: int,
        max_length: int,
        cut_long_seq: bool,
    ) -> Dict[str, List[Any]]:
        outputs = {'input_ids': [], "length": []}
        for encoded in map(tokenizer, data['text']):  # ignore max model input length warning here
            seq_len = len(encoded["input_ids"])
            for start in range(0, seq_len - min_length, max_length):
                chunk_len = min(max_length, seq_len - start)
                if chunk_len < max_length:
                    chunk_len += 1
                    encoded = add_eos(encoded, tokenizer.eos_token_id)
                for k, v in encoded.items():
                    if k in outputs:
                        outputs[k].append(v[start:start + chunk_len])
                outputs["length"].append(chunk_len)
                if cut_long_seq:
                    break
        return outputs
    
    def __iter__(self):
        return iter(self.data)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        if index in self.cached_data:
            return self.cached_data[index]
        sample = next(self.iterator)
        self.cached_data[index] = sample
        return sample


class SFTDataset(GistDataset):
    def __init__(
        self,
        path: str,
        tokenizer: AutoTokenizer,
        split: str = 'train',
        num_samples: Optional[int] = None,
        max_length: int = 4096,
        min_length: int = 1024,
        shuffle_seed: int = 42,
    ):
        self.path = path
        data = datasets.load_dataset(path, split=split)
        if num_samples is not None:
            data = data.select(range(num_samples))
        else:
            data = data.select(range(256, len(data)))
        self.data = data.map(
            self._preprocess_sft_data,
            fn_kwargs={
                'tokenizer': tokenizer,
                'min_length': min_length,
                'max_length': max_length,
            },
            batched=True, batch_size=32, num_proc=32,
            remove_columns=data.column_names,
        ).shuffle(seed=shuffle_seed)

    @staticmethod
    def _preprocess_sft_data(
        data: Dict[str, List[Any]], 
        tokenizer: AutoTokenizer,
        min_length: int,
        max_length: int,
    ) -> Dict[str, List[Any]]:
        outputs = {'input_ids': [], 'labels': []}
        for prompt, response in zip(data['instruction'], data['output']):
            prompt_input_ids = tokenizer(prompt)["input_ids"]
            response_input_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
            response_input_ids.append(tokenizer.eos_token_id)
            req_length = len(prompt_input_ids) + len(response_input_ids)
            if req_length < min_length or req_length > max_length:
                continue
            outputs['input_ids'].append(prompt_input_ids + response_input_ids)
            label = [-100] * len(prompt_input_ids) + response_input_ids
            label.extend([-100] * (max_length - req_length)) # pad here with -100
            outputs['labels'].append(label)
        return outputs


class MultiDocDataset(GistDataset):
    def __init__(
        self,
        path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 256,
        max_doc_length: int = 512,
        num_samples: Optional[int] = None,
        shuffle_seed: int = 42,
    ):
        self.max_doc_length = max_doc_length
        self.system_prompt_ids = QA_SYSTEM_PROMPT
        dataset = load_mdoc_dataset("musique", path)
        if num_samples is None:
            data = dataset.data.select(range(512, len(dataset.data)))
        else:
            data = dataset.data.select(range(num_samples))
        self.data = data.map(
            self._preprocess_mdoc_sample,
            fn_kwargs={
                'tokenizer': tokenizer,
                'max_length': max_length,
                'max_doc_length': max_doc_length,
                'extract_docs': dataset.extract_documents
            },
            batched=False, 
            num_proc=32,
            remove_columns=data.column_names
        ).shuffle(seed=shuffle_seed)

    @staticmethod
    def _preprocess_mdoc_sample(
        sample: Dict[str, Any], 
        tokenizer: AutoTokenizer,
        max_length: int,
        max_doc_length: int,
        extract_docs: Callable
    ) -> Dict[str, Any]:
        sample = extract_docs(sample)
        documents_ids = tokenizer(sample['documents'], add_special_tokens=False)["input_ids"]
        concat_doc_ids = []
        for doc_ids in documents_ids:
            if len(doc_ids) > max_doc_length:
                doc_ids = doc_ids[:max_doc_length]
            pad_length = max_doc_length - len(doc_ids)
            concat_doc_ids.extend(doc_ids)
            concat_doc_ids.extend([-100] * pad_length)
        concat_doc_ids.extend([-100] * (max_doc_length * (20 - len(documents_ids))))
        question_ids = tokenizer(sample['question'], add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(sample['answer'][0], add_special_tokens=False)["input_ids"]
        answer_ids.append(tokenizer.eos_token_id)
        input_ids = question_ids + answer_ids
        labels = [-100] * len(question_ids) + answer_ids
        pad_length = max_length - len(input_ids)
        assert pad_length >= 0, f"pad_length {pad_length} < 0"
        attention_mask = [1] * len(input_ids) + [0] * pad_length
        input_ids.extend([tokenizer.pad_token_id] * pad_length)
        labels.extend([-100] * pad_length)
        return {
            'context_input_ids': concat_doc_ids,
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }


def get_dataset(dataset_type: str, path: str, tokenizer: AutoTokenizer, **kwargs):
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if '_eval' in dataset_type:
        kwargs['shuffle_seed'] = 42 # fix seed for evaluation
    if dataset_type == "pretrain":
        return PretrainDataset(path, tokenizer, **kwargs)
    elif dataset_type == "pretrain_eval":
        kwargs['num_samples'] = kwargs.pop('num_samples', 512)
        return PretrainDataset(path, tokenizer, **kwargs)
    elif dataset_type == "sft":
        return SFTDataset(path, tokenizer, **kwargs)
    elif dataset_type == "sft_eval":
        kwargs['num_samples'] = kwargs.pop('num_samples', 256)
        return SFTDataset(path, tokenizer, **kwargs)
    elif dataset_type == "mdoc":
        return MultiDocDataset(path, tokenizer, **kwargs)
    elif dataset_type == "mdoc_eval":
        kwargs['num_samples'] = kwargs.pop('num_samples', 512)
        return MultiDocDataset(path, tokenizer, **kwargs)
    return None
