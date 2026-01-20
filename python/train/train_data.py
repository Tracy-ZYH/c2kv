import os
import datasets
import glob
from transformers import AutoTokenizer
from typing import Dict, List, Any, Mapping, Optional, Callable, Iterator
from logging import getLogger

from .configs import QA_QUERY_PROMPTS
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

def get_data_files(path: str, split: str) -> List[str]:
    filename = os.path.basename(path) + '_cached_data_files.txt'
    cached_data_files = os.path.join('/tmp', filename)
    if os.path.exists(cached_data_files):
        with open(cached_data_files, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    data_files = []
    with open(cached_data_files, 'w') as f:
        for file in glob.iglob(os.path.join(path, split, '**'), recursive=True):
            if '.' in file:
                data_files.append(file)
                f.write(file + '\n')
    return data_files

def tokenize(
    tokenizer: AutoTokenizer, 
    text: str, 
    role: str, 
    max_length: int | None = None,
    keep_bos: bool = False,
    add_generation_prompt: bool = False,
):
    if not keep_bos and tokenizer.bos_token is not None and max_length is not None:
        # the bos token is not counted in max_length
        max_length = max_length + 1
    input_ids = tokenizer.apply_chat_template(
        [{"role": role, "content": text}], tokenize=True, 
        max_length=max_length, truncation=max_length is not None,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )
    if not keep_bos and input_ids[0] == tokenizer.bos_token_id:
        input_ids = input_ids[1:]
    return input_ids


class GistDataset:
    def __init__(self, data: datasets.Dataset):
        self.data = data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index) -> Dict[str, Any]:
        return self.data[index]
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self.data)
    
    def merge(self, others: List["GistDataset"], method: str = 'interleave') -> None:
        data_list: List[datasets.Dataset] = [self.data] + [other.data for other in others]
        if method == 'interleave':
            weights = [len(data) for data in data_list]
            probabilities = [weight / sum(weights) for weight in weights]
            self.data = datasets.interleave_datasets(
                data_list, probabilities=probabilities,
                stopping_strategy="all_exhausted_without_replacement",
            )
        elif method == 'concat':
            self.data = datasets.concatenate_datasets(data_list)
        else:
            raise NotImplementedError(f"Method {method} not implemented!")


class PretrainDataset(GistDataset):
    def __init__(self, 
        path: str, 
        tokenizer: AutoTokenizer = None, 
        split: str = 'train', 
        shuffle_seed: int = 42,
        min_length: int = 1024,
        max_length: int = 4096,
        num_samples: int = 2**15,
        cut_long_seq: bool = False,
        streaming: bool = True,
    ):
        self.streaming = streaming
        shuffle_seed += int(os.environ.get("LOCAL_RANK", 0))
        # dataset = datasets.load_dataset(path, split=split, streaming=True)
        # NOTE: package datasets is modified to avoid globbing on nas, which costs hours
        map_args = {
            "fn_kwargs": {
                'tokenizer': tokenizer,
                'min_length': min_length,
                'max_length': max_length,
                'cut_long_seq': cut_long_seq    
            },
            "batched": True, "batch_size": 32,
            "remove_columns": ["text", "meta"],
        }
        if streaming: # pretty hugh to load all data
            data_files = get_data_files(path, split)
            data = datasets.load_dataset(path, data_files=data_files, streaming=True)['train']
        else: # load a preprocessed subset
            data = datasets.load_from_disk(path)
            map_args['num_proc'] = 32
        self.data = data.map(self._preprocess_pretrain_data, **map_args).shuffle(seed=shuffle_seed)
        self.iterator = iter(self.data)
        self.num_samples = num_samples
        self.cached_data: Dict[int, Dict[str, List[Any]]] = {}
        self.min_length, self.max_length = min_length, max_length

    @staticmethod
    def _preprocess_pretrain_data(
        data: Dict[str, List[Any]], 
        tokenizer: AutoTokenizer,
        min_length: int,
        max_length: int,
        cut_long_seq: bool,
    ) -> Dict[str, List[Any]]:
        outputs = {'input_ids': [], "length": []}
        for text in data['text']:
            if cut_long_seq:
                encoded = tokenizer(text, max_length=max_length, truncation=True)
            else:
                encoded = tokenizer(text)
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
    
    def to_mdoc_format(self, tokenizer: AutoTokenizer, mdoc_dataset: "MultiDocDataset") -> GistDataset:
        assert not self.streaming, "Streaming is not supported when to_mdoc_format!"
        doc_length = mdoc_dataset.max_doc_length
        if self.min_length <= doc_length:
            raise ValueError(f"min_length {self.min_length} <= doc_length {doc_length}")
        def _mdoc_formatter(sample: Dict[str, Any]) -> Dict[str, Any]:
            input_ids: List[int] = sample['input_ids']
            original_length = len(input_ids)
            context_length = ((original_length - 1) // doc_length) * doc_length
            context_input_ids = input_ids[:context_length]
            context_pad_length = doc_length * mdoc_dataset.max_doc_num - context_length
            context_input_ids.extend([-100] * context_pad_length)
            input_ids = input_ids[context_length:context_length + mdoc_dataset.max_length]
            input_length = len(input_ids)
            input_pad_length = mdoc_dataset.max_length - input_length
            labels = input_ids + [-100] * input_pad_length
            input_ids.extend([tokenizer.pad_token_id] * input_pad_length)
            attention_mask = [1] * input_length + [0] * input_pad_length
            return {
                'context_input_ids': context_input_ids,
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': attention_mask
            }
        return GistDataset(self.data.map(_mdoc_formatter, batched=False, num_proc=32).select(range(self.num_samples)))

    def to_mdoc_format2(self, tokenizer: AutoTokenizer, mdoc_dataset: "MultiDocDataset") -> GistDataset:
        assert not self.streaming, "Streaming is not supported when to_mdoc_format!"
        doc_length = mdoc_dataset.max_doc_length
        if self.min_length <= doc_length:
            raise ValueError(f"min_length {self.min_length} <= doc_length {doc_length}")
        # 多样化的提示问题列表
        continuation_prompts = [
            "Continue this passage", "Please continue", "What comes next?", "Continue the text",
            "Keep going", "Please proceed with the continuation", "What follows?",
            "Continue this passage", "Please continue the following text",
            "Continue writing from where it left off", "What comes next in this passage?",
            "Continue this document", "Extend this content further",
        ]
        # 获取换行符的token id
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        newline_id = newline_ids[0] if newline_ids else None
        def split_by_sentences(input_ids: List[int], max_length: int) -> List[List[int]]:
            """按照换行符分割，并将句子尽可能多地放入文档中"""
            if newline_id is None:
                # 如果没有换行符，直接按长度分割
                return [input_ids[i:i+max_length] for i in range(0, len(input_ids), max_length)]
            documents = []
            current_doc = []
            current_length = 0
            # 找到所有换行符位置
            sentence_boundaries = [0]
            for i, token_id in enumerate(input_ids):
                if token_id == newline_id:
                    sentence_boundaries.append(i + 1)
            sentence_boundaries.append(len(input_ids))
            # 按句子分组到文档中
            for i in range(len(sentence_boundaries) - 1):
                sentence = input_ids[sentence_boundaries[i]:sentence_boundaries[i+1]]
                sentence_length = len(sentence)
                # 如果单个句子超过最大长度，需要分割
                if sentence_length > max_length:
                    if current_doc:
                        documents.append(current_doc)
                        current_doc = []
                        current_length = 0
                    # 分割长句子
                    for j in range(0, sentence_length, max_length):
                        documents.append(sentence[j:j+max_length])
                elif current_length + sentence_length <= max_length:
                    # 可以放入当前文档
                    current_doc.extend(sentence)
                    current_length += sentence_length
                else:
                    # 当前文档已满，开始新文档
                    if current_doc:
                        documents.append(current_doc)
                    current_doc = sentence
                    current_length = sentence_length
            return documents
        
        def _mdoc_formatter(sample: Dict[str, Any]) -> Dict[str, Any]:
            input_ids: List[int] = sample['input_ids']
            # 使用伪随机数选择提示（基于样本内容确保可复现）
            seed = sum(input_ids[:min(10, len(input_ids))]) % len(continuation_prompts)
            prompt_text = continuation_prompts[seed]
            # 将文本分割成文档
            documents = split_by_sentences(input_ids, doc_length)
            if len(documents) <= 1:
                # 文本太短，无法分割
                return None
            # 限制文档数量
            if len(documents) > mdoc_dataset.max_doc_num + 1:
                documents = documents[:mdoc_dataset.max_doc_num + 1]
            # 准备context documents (所有文档除了最后一个用于回答)
            context_docs = documents[:-1]
            answer_doc = documents[-1]
            # 将每个context document转换为user message
            context_input_ids = []
            for doc in context_docs:
                # 解码文档内容
                doc_text = tokenizer.decode(doc, skip_special_tokens=True)
                # 使用tokenize函数转换为user message
                doc_ids = tokenize(tokenizer, doc_text, "user", max_length=doc_length)
                context_input_ids.extend(doc_ids)
                assert len(doc_ids) <= doc_length, \
                    f"Context document length {len(context_input_ids)} > doc_length {doc_length}"
                context_input_ids.extend([-100] * (doc_length - len(doc_ids)))
            empty_doc_num = mdoc_dataset.max_doc_num - len(context_docs)
            if empty_doc_num > 0:
                context_input_ids.extend([-100] * (doc_length * empty_doc_num))

            prompt_ids = tokenize(tokenizer, prompt_text, "user", add_generation_prompt=True)
            answer_ids = answer_doc + [tokenizer.eos_token_id]
            
            # 拼接 prompt 和 answer
            input_ids = prompt_ids + answer_ids
            labels = ([-100] * len(prompt_ids)) + answer_ids
            attention_mask = [1] * len(input_ids)
            # 填充
            input_pad_length = mdoc_dataset.max_length - len(input_ids)
            if input_pad_length > 0:
                labels.extend([-100] * input_pad_length)
                attention_mask.extend([0] * input_pad_length)
                input_ids.extend([tokenizer.pad_token_id] * input_pad_length)
            else:
                labels = labels[:mdoc_dataset.max_length]
                attention_mask = attention_mask[:mdoc_dataset.max_length]
                input_ids = input_ids[:mdoc_dataset.max_length]
        
            return {
                'context_input_ids': context_input_ids,
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': attention_mask
            }
        
        # 过滤掉None结果
        mapped_data = self.data.map(
            _mdoc_formatter, 
            batched=False, 
            num_proc=32,
            remove_columns=self.data.column_names
        ).filter(lambda x: x is not None, num_proc=32)
        
        return GistDataset(mapped_data.select(range(min(self.num_samples, len(mapped_data)))))

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
        data = data.shuffle(seed=shuffle_seed)
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
        )

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
        if "musique" in path:
            dataset = load_mdoc_dataset("musique", path)
            extract_documents = dataset.extract_documents
            data = dataset.data
            max_doc_num = 20
        elif "hotpotqa" in path:
            data = datasets.load_dataset("jsonl", data_files=path, split="train")
            extract_documents = lambda sample: sample
            # max_doc_num = 10
            max_doc_num = 20
        elif "longmagpie" in path or "longalpaca" in path:
            data = datasets.load_from_disk(path)
            extract_documents = None
            max_doc_num = 20
        else:
            raise NotImplementedError(f"Unsupported dataset {path}")
        if num_samples is None:
            data = data.select(range(512, len(data)))
        else:
            data = data.select(range(num_samples))
        self.data = data.shuffle(seed=shuffle_seed).map(
            self._preprocess_mdoc_sample,
            fn_kwargs={
                'tokenizer': tokenizer,
                'max_length': max_length,
                'max_doc_length': max_doc_length,
                'max_doc_num': max_doc_num,
                'extract_docs': extract_documents
            },
            batched=False, 
            num_proc=64,
            remove_columns=data.column_names
        )
        self.max_doc_length = max_doc_length
        self.system_prompt_ids = tokenize(
            tokenizer, "You are a helpful assistant.", "system", keep_bos=True,
        )
        self.max_doc_num = max_doc_num
        self.max_length = max_length

    @staticmethod
    def _preprocess_mdoc_sample(
        sample: Dict[str, Any], 
        tokenizer: AutoTokenizer,
        max_length: int,
        max_doc_length: int,
        max_doc_num: int,
        extract_docs: Callable | None,
    ) -> Dict[str, Any]:
        if extract_docs is not None:
            sample = extract_docs(sample)
            query_prompt_seed = sum(map(len, sample['documents'])) % len(QA_QUERY_PROMPTS)
            sample['question'] = QA_QUERY_PROMPTS[query_prompt_seed] + '\n' + sample['question']
        concat_doc_ids = []
        for doc in sample['documents']:
            doc_ids = tokenize(tokenizer, doc, "user", max_doc_length)
            pad_length = max_doc_length - len(doc_ids)
            assert pad_length >= 0, f"pad_length {pad_length} < 0"
            concat_doc_ids.extend(doc_ids)
            concat_doc_ids.extend([-100] * pad_length)
        concat_doc_ids.extend([-100] * (max_doc_length * (max_doc_num - len(sample['documents']))))
        question_ids = tokenize(tokenizer, sample['question'], "user", add_generation_prompt=True)
        answer_ids = tokenizer.encode(sample['answer'][0], add_special_tokens=False)
        answer_ids.append(tokenizer.eos_token_id)
        input_ids = question_ids + answer_ids
        labels = [-100] * len(question_ids) + answer_ids
        pad_length = max_length - len(input_ids)
        if pad_length > 0:
            attention_mask = [1] * len(input_ids) + [0] * pad_length
            input_ids.extend([tokenizer.pad_token_id] * pad_length)
            labels.extend([-100] * pad_length)
        else:
            attention_mask = [1] * max_length
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
        return {
            'context_input_ids': concat_doc_ids,
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }


def get_dataset(dataset_type: str, path: str, tokenizer: AutoTokenizer, **kwargs) -> GistDataset:
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if dataset_type == "pretrain":
        return PretrainDataset(path, tokenizer, **kwargs)
    elif dataset_type == "pretrain_eval":
        kwargs['shuffle_seed'] = 42 # fix seed for evaluation
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
    raise NotImplementedError(f"Unsupported dataset type {dataset_type}")
