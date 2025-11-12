from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import datasets
import string
import regex

from longbench_metrics import qa_f1_score, rouge_score, qa_f1_zh_score, rouge_zh_score


def max_f1_score(pred: str, gt_list: List[str]) -> float:
    pred = pred.split('. ')[0]
    return max([qa_f1_score(pred, gt) for gt in gt_list])

def max_rouge_score(pred: str, gt_list: List[str]) -> float:
    pred = pred.split('\n')[0]
    return max([rouge_score(pred, gt) for gt in gt_list])
    

QA_SYSTEM_PROMPT: str = ("You will be asked a question after reading several passages. "
    "Please directly answer the question based on the given passages. Do NOT repeat the question. "
    "The answer should be within 5 words.\n\n")

QA_QUERY_PROMPT: str = ("Answer the question directly based on the given passages. "
    "Do NOT repeat the question. The answer should be within 5 words.\n\nQuestion: ")

QA_MAX_NEW_TOKENS: int = 8


class AbstractMDQADataset(ABC):
    """
    An abstract base class for multi-document question answering datasets.
    Subclasses must implement methods to load and access multi-document examples,
    supporting both extractive and abstractive QA tasks.
    """
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of examples in the dataset."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Return a single example at index `idx` as a dictionary with:
          - 'qid': Unique question identifier (str)
          - 'question': The question text (str)
          - 'documents': List of context documents (List[str])
          - 'answer': list of answers (List[str])
        """
        pass


class WikiMQADataset(AbstractMDQADataset):
    def __init__(self, data_path: str) -> None:
        self.data = datasets.load_dataset("json", data_files=data_path)['train']
        self.system_prompt: str = QA_SYSTEM_PROMPT
        self.max_new_tokens: int = QA_MAX_NEW_TOKENS
        self.query_prompt: str = QA_QUERY_PROMPT
        print(f"Loading dataset from {data_path}...")
        self.context = self.data['context']
        self.qid = self.data['_id']
        self.question = self.data['question']
        self.answer = self.data['answer']
        self.metric = max_f1_score
        print(f"Done loading {data_path}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        context_list = []
        for i, item in enumerate(eval(self.context[idx])):
            context_str = f"Document {i+1} (title: {item[0]}) " + " ".join(item[1]) + '\n\n'
            context_list.append(context_str)
        return {
            'qid': self.qid[idx],
            'question': self.query_prompt + self.question[idx] + "\n\nAnswer: ",
            'documents': context_list,
            'answer': [self.answer[idx]],
        }


class MusiqueDataset(AbstractMDQADataset):
    def __init__(self, data_path: str, only_supporting: bool=False) -> None:
        self.data = datasets.load_dataset("json", data_files=data_path)['train']
        # self.data = datasets.load_dataset(data_path)['train']
        print(f"Loading dataset from {data_path}...")
        self.system_prompt: str = QA_SYSTEM_PROMPT
        self.max_new_tokens: int = QA_MAX_NEW_TOKENS
        self.query_prompt: str = QA_QUERY_PROMPT
        self.only_supporting = only_supporting
        self.paragraphs = self.data['paragraphs']
        self.qid = self.data['id']
        self.question = self.data['question']
        self.answer = [[answer] + answer_aliases for answer, answer_aliases in zip(self.data['answer'], self.data['answer_aliases'])]
        self.metric = max_f1_score
        print(f"Done loading {data_path}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        context_list = []
        for item in self.paragraphs[idx]:
            if self.only_supporting and not item['is_supporting']:
                continue
            context_list.append(f"Document {item['idx']} (title: {item['title']}) " + item['paragraph_text'] + '\n\n')
        return {
            'qid': self.qid[idx],
            'question': self.query_prompt + self.question[idx] + "\n\nAnswer: ",
            'documents': context_list,
            'answer': self.answer[idx],
        }


class HotpotQADataset(AbstractMDQADataset):
    def __init__(self) -> None:
        print(f"Loading dataset from zai-org/LongBench hotpotqa...")
        self.data = datasets.load_dataset('zai-org/LongBench', 'hotpotqa')['test']
        self.system_prompt: str = QA_SYSTEM_PROMPT
        self.max_new_tokens: int = QA_MAX_NEW_TOKENS
        self.query_prompt: str = QA_QUERY_PROMPT
        self.paragraphs = self.data['context']
        self.qid = self.data['_id']
        self.answer = self.data['answers']
        self.question = self.data['input']
        self.metric = max_f1_score
        print(f"Done loading zai-org/LongBench hotpotq")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        context_list = []
        for item in self.paragraphs[idx].split('Passage'):
            if len(item) > 10:
                context_list.append('Passage' + item + '\n\n')
        return {
            'qid': self.qid[idx],
            'question': self.query_prompt + self.question[idx] + "\n\nAnswer: ",
            'documents': context_list,
            'answer': self.answer[idx],
        }


class MultiNewsDataset(AbstractMDQADataset):
    def __init__(self) -> None:
        print(f"Loading dataset from zai-org/LongBench multi_news...")
        self.data = datasets.load_dataset('zai-org/LongBench', 'multi_news')['test']
        self.system_prompt: str = "You are given several news passages. Write a one-page summary of all news. \n\nNews:"
        self.max_new_tokens: int = 512
        self.paragraphs = self.data['context']
        self.qid = self.data['_id']
        self.answer = self.data['answers']
        self.question = '\n\nNow, write a one-page summary of all the news.\n\nSummary: '
        self.metric = max_rouge_score
        print(f"Done loading zai-org/LongBench multi_news")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        context_list = []
        for item in self.paragraphs[idx].split('Passage'):
            if len(item) > 10:
                context_list.append('\n\nPassage' + item.replace('NEWLINE_CHAR', '\n'))
        return {
            'qid': self.qid[idx],
            'question': self.question,
            'documents': context_list,
            'answer': self.answer[idx],
        }


class SAMSumDataset(AbstractMDQADataset):
    def __init__(self) -> None:
        print(f"Loading dataset from zai-org/LongBench samsum...")
        self.data = datasets.load_dataset('zai-org/LongBench', 'samsum')['test']
        self.system_prompt: str = "Summarize the dialogue into a few short sentences. The following are some examples.\n\n"
        self.max_new_tokens: int = 128
        self.paragraphs = self.data['context']
        self.qid = self.data['_id']
        self.answer = self.data['answers']
        self.question = self.data['input']
        self.metric = max_rouge_score
        print(f"Done loading zai-org/LongBench multi_news")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        context_list = []
        for item in self.paragraphs[idx].split('Dialogue'):
            if len(item) > 10:
                context_list.append('Dialogue' + item + '\n')
        return {
            'qid': self.qid[idx],
            'question': self.question[idx],
            'documents': context_list,
            'answer': self.answer[idx],
        }


class AmapDataset(AbstractMDQADataset):
    CONTEXT_BEGIN: str = "\n<召回的通用搜索内容>\n"
    CONTEXT_END: str = "</召回的通用搜索内容>"

    @staticmethod
    def metric(pred: str, gt_list: List[str]) -> float:
        return max([qa_f1_zh_score(pred, gt) for gt in gt_list])

    def _preprocess_amap_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        system_prompt, prompt = sample['prompt'].split(self.CONTEXT_BEGIN, 1)
        context_str, question = prompt.split(self.CONTEXT_END, 1)
        contexts = [
            '[1]' + context for context in context_str.split('[1]')
            if len(context.strip()) > 2
        ]
        return {
            'qid': sample['traceId'],
            'system_prompt': system_prompt + self.CONTEXT_BEGIN,
            'question': self.CONTEXT_END + question,
            'documents': contexts,
            'answer': [sample['response']],
        }

    def __init__(self, csv_path: str, load_full: bool=False) -> None:
        print(f"Loading inhouse Amap dataset from {csv_path}...")
        data = datasets.load_dataset("csv", data_files=csv_path)['train'].filter(
            lambda sample: (
                (load_full or 16e3 < len(sample['prompt']) < 24e3) and 
                self.CONTEXT_BEGIN in sample['prompt']
            ), num_proc=32
        )
        data = data.map(self._preprocess_amap_sample, num_proc=32, remove_columns=data.column_names)
        data = data.filter(lambda sample: len(sample['documents']) > 0, num_proc=32)
        if not load_full:
            data = data.shuffle(seed=42).select(range(1000))
        self.data = data
        self.system_prompt: Optional[str] = None
        self.max_new_tokens: int = 768
        print(f"Done loading Amap dataset from {csv_path}")

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


def load_mdoc_dataset(name: str, path: Optional[str]=None, **kwargs) -> AbstractMDQADataset:
    if name == "musique":
        if path is None:
            print('Defaulting musique dataset path to "../datasets/musique.jsonl"')
            path = "../datasets/musique.jsonl"
        if 'only_supporting' not in kwargs:
            kwargs['only_supporting'] = False
        return MusiqueDataset(path, kwargs['only_supporting'])
    elif name == "wikimqa":
        if path is None:
            print('Defaulting wikimqa dataset path to "../datasets/wikimqa.json"')
            path = "../datasets/wikimqa.json"
        return WikiMQADataset(path)
    elif name == "hotpotqa":
        return HotpotQADataset()
    elif name == "multinews":
        return MultiNewsDataset()
    elif name == "samsum":
        return SAMSumDataset()
    elif name == "amap":
        if path is None:
            print('Defaulting amap dataset path to "../datasets/AmapData.csv"')
            path = "../datasets/AmapData.csv"
        return AmapDataset(path, load_full=kwargs.get('load_full', False))
    else:
        raise ValueError(f"Unsupported dataset name: {name}")
