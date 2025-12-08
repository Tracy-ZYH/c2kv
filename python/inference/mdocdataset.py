from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import datasets
import string
import regex

try:
    from .longbench_metrics import qa_f1_score, rouge_score, qa_f1_zh_score, rouge_zh_score
except ImportError:
    from longbench_metrics import qa_f1_score, rouge_score, qa_f1_zh_score, rouge_zh_score


def max_f1_score(pred: str, gt_list: List[str]) -> float:
    pred = pred.split('. ')[0]
    return max([qa_f1_score(pred, gt) for gt in gt_list])

def max_f1_score_with_reasoning(pred: str, gt_list: List[str]) -> float:
    """
    从带有推理过程的预测字符串中提取最终答案，并计算与基准答案列表的最大F1分数。
    
    Args:
        pred (str): LLM的完整输出，应包含 "[Reasoning]... [Answer]..." 格式。
        gt_list (List[str]): 基准答案（Ground Truth）的列表。

    Returns:
        float: 计算出的最大F1分数。
    """
    parts = pred.split('[Answer]')
    if len(parts) > 1: # 1. 如果成功分割，答案在第二部分
        extracted_answer = parts[-1].split('\n\n')[0].split('.')[0].strip()
    else: # 2. 如果模型没有遵循格式（例如没有输出 [Answer] 标签）
        return 0.0
    return max([qa_f1_score(extracted_answer, gt) for gt in gt_list])


def max_rouge_score(pred: str, gt_list: List[str]) -> float:
    pred = pred.split('\n')[0]
    return max([rouge_score(pred, gt) for gt in gt_list])
    

QA_SYSTEM_PROMPT: str = ("You will be asked a question after reading several passages. "
    "Please directly answer the question based on the given passages. Do NOT repeat the question. "
    "The answer should be within 5 words.\n\n")

QA_QUERY_PROMPT: str = ("Answer the question directly based on the given passages. "
    "Do NOT repeat the question. The answer should be within 5 words.\n\nQuestion: ")

QA_MAX_NEW_TOKENS: int = 8

QA_SYSTEM_PROMPT_COT: str = (
    "You are a helpful QA assistant. After reading the provided passages, you will be asked a question. "
    "Your task is to first provide a step-by-step reasoning process on how to answer the question based on the passages. "
    "After your reasoning, provide the final, concise answer in a specific format."
)

QA_QUERY_PROMPT_COT: str = (
    "Based on the given passages, answer the question. Please follow the format below:\n\n"
    "[Reasoning] (Your step-by-step reasoning on how to arrive at the answer based on the provided text)\n\n"
    "[Answer] (The final, concise answer, typically within 5 words)\n\n"
    "Question: "
)

QA_MAX_NEW_TOKENS_COT: int = 512


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
    def __init__(self, data_path: str, enable_cot: bool) -> None:
        self.data = datasets.load_dataset("json", data_files=data_path)['train']
        self.system_prompt: str = QA_SYSTEM_PROMPT_COT if enable_cot else QA_SYSTEM_PROMPT
        self.max_new_tokens: int = QA_MAX_NEW_TOKENS_COT if enable_cot else QA_MAX_NEW_TOKENS
        self.query_prompt: str = QA_QUERY_PROMPT_COT if enable_cot else QA_QUERY_PROMPT
        print(f"Loading dataset from {data_path}...")
        self.context = self.data['context']
        self.qid = self.data['_id']
        self.question = self.data['question']
        self.answer = self.data['answer']
        self.metric = max_f1_score_with_reasoning if enable_cot else max_f1_score
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
    def __init__(self, data_path: str, only_supporting: bool=False, enable_cot: bool=False) -> None:
        self.data = datasets.load_dataset("json", data_files=data_path)['train']
        # self.data = datasets.load_dataset(data_path)['train']
        print(f"Loading dataset from {data_path}...")
        self.system_prompt: str = QA_SYSTEM_PROMPT_COT if enable_cot else QA_SYSTEM_PROMPT
        self.max_new_tokens: int = QA_MAX_NEW_TOKENS_COT if enable_cot else QA_MAX_NEW_TOKENS
        self.query_prompt: str = QA_QUERY_PROMPT_COT if enable_cot else QA_QUERY_PROMPT
        self.only_supporting = only_supporting
        self.metric = max_f1_score_with_reasoning if enable_cot else max_f1_score
        print(f"Done loading {data_path}")
    
    @staticmethod
    def extract_documents(sample: Dict[str, Any]) -> Dict[str, Any]:
        context_list = []
        for item in sample['paragraphs']:
            context_list.append(f"Document {item['idx']} (title: {item['title']}) " + item['paragraph_text'] + '\n\n')
        return {
            'qid': sample['id'],
            'question': QA_QUERY_PROMPT + sample['question'] + '\n\nAnswer: ',
            'documents': context_list,
            'answer': [sample['answer']] + sample['answer_aliases'],
        }
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.extract_documents(self.data[idx])


class HotpotQADataset(AbstractMDQADataset):
    def __init__(self, enable_cot: bool=False) -> None:
        print(f"Loading dataset from zai-org/LongBench hotpotqa...")
        self.data = datasets.load_dataset('zai-org/LongBench', 'hotpotqa')['test']
        self.system_prompt: str = QA_SYSTEM_PROMPT_COT if enable_cot else QA_SYSTEM_PROMPT
        self.max_new_tokens: int = QA_MAX_NEW_TOKENS_COT if enable_cot else QA_MAX_NEW_TOKENS
        self.query_prompt: str = QA_QUERY_PROMPT_COT if enable_cot else QA_QUERY_PROMPT
        self.paragraphs = self.data['context']
        self.qid = self.data['_id']
        self.answer = self.data['answers']
        self.question = self.data['input']
        self.metric = max_f1_score_with_reasoning if enable_cot else max_f1_score
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
        print(f"Loaded {len(self.data)} Amap dataset from {csv_path}")

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]
    

class LongAlpacaDataset(AbstractMDQADataset):
    CONTEXT_BEGIN: str = "The paper begins. "
    CONTEXT_END: str = "Now the paper ends."

    def __init__(self, prompts_path: str) -> None:
        self.system_prompt = None
        with open(prompts_path, 'r') as f:
            self.system_prompts = f.readlines()
        data = datasets.load_dataset("Yukang/LongAlpaca-16k-length")['train']
        data = data.filter(
            lambda sample: (
                # len(sample['instruction']) < 5e4 and 
                self.CONTEXT_BEGIN in sample['instruction'] and
                self.CONTEXT_END in sample['instruction']
            ), num_proc=32
        )
        data = data.map(
            self._preprocess_sample, 
            remove_columns=data.column_names,
            num_proc=32, with_indices=True
        )
        self.data = data.filter(lambda sample: len(sample['documents']) > 0, num_proc=32)
        self.max_new_tokens: int = 768
        self.metric = max_rouge_score

    def _preprocess_sample(self, sample: Dict[str, Any], idx: int) -> Dict[str, Any]:
        documents = []
        last_document = []
        context = sample['instruction'].split(self.CONTEXT_BEGIN, 1)[1]
        context, question = context.split(self.CONTEXT_END, 1)
        for line in context.split('\n'):
            if len(line) > 0 and line[0] in string.digits:
                if len(last_document) > 0:
                    documents.append('\n'.join(last_document))
                last_document = []
            last_document.append(line)
        if len(last_document) > 0:
            documents.append('\n'.join(last_document))
        system_prompt = self.system_prompts[idx % len(self.system_prompts)] + self.CONTEXT_BEGIN
        return {
            'qid': str(idx),
            'system_prompt': system_prompt,
            'question': question,
            'documents': documents,
            'answer': [self.CONTEXT_END + sample['output']],
        }

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


def load_mdoc_dataset(name: str, path: Optional[str]=None, **kwargs) -> AbstractMDQADataset:
    enable_cot = kwargs.get('enable_cot', False)
    if name == "musique":
        if path is None:
            print('Defaulting musique dataset path to "../datasets/musique.jsonl"')
            path = "../datasets/musique.jsonl"
        return MusiqueDataset(path, only_supporting=kwargs.get('only_supporting', False), enable_cot=enable_cot)
    elif name == "wikimqa":
        if path is None:
            print('Defaulting wikimqa dataset path to "../datasets/wikimqa.json"')
            path = "../datasets/wikimqa.json"
        return WikiMQADataset(path, enable_cot=enable_cot)
    elif name == "hotpotqa":
        return HotpotQADataset(enable_cot=enable_cot)
    elif name == "multinews":
        return MultiNewsDataset()
    elif name == "samsum":
        return SAMSumDataset()
    elif name == "amap":
        if path is None:
            print('Defaulting amap dataset path to "../datasets/AmapData.csv"')
            path = "../datasets/AmapData.csv"
        return AmapDataset(path, load_full=kwargs.get('load_full', False))
    elif name == "longalpaca":
        if path is None:
            print('Defaulting longalpaca dataset path to "../datasets/longalpaca_prompts.txt"')
            path = "../datasets/longalpaca_prompts.txt"
        return LongAlpacaDataset(path)
    else:
        raise ValueError(f"Unsupported dataset name: {name}")
