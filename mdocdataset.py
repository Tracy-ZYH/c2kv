from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import datasets
import string
import regex

from longbench_metrics import qa_f1_score, rouge_score


def max_f1_score(pred: str, gt_list: List[str]) -> float:
    pred = pred.split('. ')[0]
    return max([qa_f1_score(pred, gt) for gt in gt_list])

QA_SYSTEM_PROMPT: str = ("You will be asked a question after reading several passages. "
    "Please directly answer the question based on the given passages. Do NOT repeat the question. "
    "The answer should be within 5 words.\n\n")

QA_QUERY_PROMPT: str = ("Answer the question directly based on the given passages. "
    "Do NOT repeat the question. The answer should be within 5 words.\n\nQuestion: ")

QA_MAX_NEW_TOKENS: int = 8
SUMMARY_MAX_NEW_TOKENS: int = 512


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
            'question': QA_QUERY_PROMPT + self.question[idx] + "\n\nAnswer: ",
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
            'question': QA_QUERY_PROMPT + self.question[idx] + "\n\nAnswer: ",
            'documents': context_list,
            'answer': self.answer[idx],
        }


class HotpotQADataset(AbstractMDQADataset):
    def __init__(self) -> None:
        print(f"Loading dataset from zai-org/LongBench hotpotqa...")
        self.data = datasets.load_dataset('zai-org/LongBench', 'hotpotqa')['test']
        self.system_prompt: str = QA_SYSTEM_PROMPT
        self.max_new_tokens: int = QA_MAX_NEW_TOKENS
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
                context_list.append('\n\nPassage' + item)
        return {
            'qid': self.qid[idx],
            'question': QA_QUERY_PROMPT + self.question[idx] + "\n\nAnswer: ",
            'documents': context_list,
            'answer': self.answer[idx],
        }


class MultiNewsDataset(AbstractMDQADataset):
    def __init__(self) -> None:
        print(f"Loading dataset from zai-org/LongBench multi_news...")
        self.data = datasets.load_dataset('zai-org/LongBench', 'multi_news')['test']
        self.system_prompt: str = "You are given several news passages. Write a one-page summary of all news. \n\nNews:"
        self.max_new_tokens: int = SUMMARY_MAX_NEW_TOKENS
        self.paragraphs = self.data['context']
        self.qid = self.data['_id']
        self.answer = self.data['answers']
        self.question = '\n\nNow, write a one-page summary of all the news.\n\nSummary: '
        self.metric = rouge_score
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


def load_mdoc_dataset(name: str, path: Optional[str], **kwargs) -> AbstractMDQADataset:
    if name == "musique":
        assert path is not None, "path should be specified when using Musique dataset"
        return MusiqueDataset(path, kwargs['only_supporting'])
    elif name == "wikimqa":
        assert path is not None, "path should be specified when using 2WikiMQA dataset"
        return WikiMQADataset(path)
    elif name == "hotpotqa":
        return HotpotQADataset()
    else:
        raise ValueError(f"Unsupported dataset name: {name}")


if __name__ == "__main__":
    wikimqa = WikiMQADataset("../datasets/wikimqa.json")
    print(wikimqa[100])
    dataset = datasets.load_dataset("jsonl", data_files="../datasets/musique_ans_v1.0_dev.jsonl")['train']
    print(dataset)
    print(dataset['answer'][:10])
    musique = MusiqueDataset("../musique_ans_v1.0_dev.jsonl")
    print(musique[100])
