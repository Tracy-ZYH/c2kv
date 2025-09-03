from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import datasets
import string
import regex


SYSTEM_PROMPT: str = ("You will be asked a question after reading several passages. "
    "Please directly answer the question based on the given passages. Do NOT repeat the question. "
    "The answer should be within 5 words.\n\n")

QUERY_PROMPT: str = ("Answer the question directly based on the given passages. "
    "Do NOT repeat the question. The answer should be within 5 words.\n\nQuestion: ")


def normalize_answer(s: str) -> str:
    """Normalization from the SQuAD evaluation script.

    See https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
    """

    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def best_subspan_em(prediction: str, ground_truths: List[str]) -> float:
    normalized_prediction = normalize_answer(prediction).lower()

    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_ground_truth.lower() in normalized_prediction:
            return 1.0
    return 0.0


class AbstractMDQADataset(ABC):
    """
    An abstract base class for multi-document question answering datasets.
    Subclasses must implement methods to load and access multi-document examples,
    supporting both extractive and abstractive QA tasks.
    """

    @abstractmethod
    def get_system_prompt(self) -> str:
        pass
    
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
        self.system_prompt: str = SYSTEM_PROMPT
        print(f"Loading dataset from {data_path}...")
        self.context = self.data['context']
        self.qid = self.data['_id']
        self.question = self.data['question']
        self.answer = self.data['answer']
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
            'question': QUERY_PROMPT + self.question[idx] + "\n\nAnswer: ",
            'documents': context_list,
            'answer': [self.answer[idx]],
        }

    def get_system_prompt(self) -> str:
        return self.system_prompt


class MusiqueDataset(AbstractMDQADataset):
    def __init__(self, data_path: str, only_supporting: bool=False) -> None:
        self.data = datasets.load_dataset("json", data_files=data_path)['train']
        # self.data = datasets.load_dataset(data_path)['train']
        print(f"Loading dataset from {data_path}...")
        self.system_prompt: str = SYSTEM_PROMPT
        self.only_supporting = only_supporting
        self.paragraphs = self.data['paragraphs']
        self.qid = self.data['id']
        self.question = self.data['question']
        self.answer = [[answer] + answer_aliases for answer, answer_aliases in zip(self.data['answer'], self.data['answer_aliases'])]
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
            'question': QUERY_PROMPT + self.question[idx] + "\n\nAnswer: ",
            'documents': context_list,
            'answer': self.answer[idx],
        }

    def get_system_prompt(self) -> str:
        return self.system_prompt

if __name__ == "__main__":
    wikimqa = WikiMQADataset("/home/admin/workspace/aop_lab/app_source/duchuheng/2WikiMultihopQA/dev.json")
    print(wikimqa[100])
    dataset = datasets.load_dataset("jsonl", data_files="../musique_ans_v1.0_dev.jsonl")['train']
    print(dataset)
    print(dataset['answer'][:10])
    musique = MusiqueDataset("../musique_ans_v1.0_dev.jsonl")
    print(musique[100])
