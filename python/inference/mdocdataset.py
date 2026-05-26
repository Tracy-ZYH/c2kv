from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import itertools
import datasets
import string
from numpy import isin
import regex

try:
    from .longbench_metrics import qa_f1_score, rouge_score, qa_f1_zh_score, rouge_zh_score
except ImportError:
    from longbench_metrics import qa_f1_score, rouge_score, qa_f1_zh_score, rouge_zh_score


def max_f1_score(pred: str, gt_list: List[str]) -> float:
    return max([qa_f1_score(pred, gt) for gt in gt_list])

def max_f1_zh_score(pred: str, gt_list: List[str]) -> float:
    """Chinese F1 score using qa_f1_zh_score."""
    return max([qa_f1_zh_score(pred, gt) for gt in gt_list])

def max_f1_score_with_reasoning(pred: str, gt_list: List[str]) -> float:
    """
    从带有推理过程的预测字符串中提取最终答案，并计算与基准答案列表的最大F1分数。

    Args:
        pred (str): LLM的完整输出，应包含 "[Reasoning]... [Answer]..." 格式。
        gt_list (List[str]): 基准答案（Ground Truth）的列表。

    Returns:
        float: 计算出的最大F1分数。
    """
    if "[Answer]" in pred:
        parts = pred.split('[Answer]')
    elif "Answer:" in pred:
        parts = pred.split('Answer:')
    else:
        return 0.0
    if len(parts) > 1: # 1. 如果成功分割，答案在第二部分
        extracted_answer = parts[-1].split('\n\n')[0].split('.')[0].split('[')[0].strip()
    else: # 2. 如果模型没有遵循格式（例如没有输出 [Answer] 标签）
        return 0.0
    return max([qa_f1_score(extracted_answer, gt) for gt in gt_list])


def max_rouge_score(pred: str, gt_list: List[str]) -> float:
    return max([rouge_score(pred, gt) for gt in gt_list])

def max_rouge_zh_score(pred: str, gt_list: List[str]) -> float:
    """Chinese ROUGE score using rouge_zh_score."""
    return max([rouge_zh_score(pred, gt) for gt in gt_list])
    

QA_SYSTEM_PROMPT: str = ("You will be asked a question after reading several passages. "
    "Please directly answer the question based on the given passages.\n\n")

QA_QUERY_PROMPT: str = (
    "Answer the question directly based on the given passages. "
    "Output exactly one phrase as the final answer. No explanation. No extra text.\n"
    "Example: Question: What is the capital of France? Paris.\n\n"
    "Question: "
)

QA_MAX_NEW_TOKENS: int = 16

QA_SYSTEM_PROMPT_COT: str = (
    "You are a helpful QA assistant. After reading the provided passages, you will be asked a question. "
    "Your task is to first provide a step-by-step reasoning process on how to answer the question based on the passages. "
    "After your reasoning, provide the final, concise answer in a specific format."
)

QA_QUERY_PROMPT_COT: str = (
    "Based on the given passages, answer the question. Please follow the format below:\n\n"
    "[Reasoning] (Your step-by-step reasoning on how to arrive at the answer based on the provided text)\n\n"
    "[Answer] (The final, concise answer, typically within 5 words).\n\n"
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
    def __init__(self, data_path: str | None, enable_cot: bool = False, split: str = 'test') -> None:
        self.data_path = data_path
        if data_path is None:
            self.data = datasets.load_dataset('zai-org/LongBench', '2wikimqa', split=split)
        else:
            self.data = datasets.load_dataset(data_path, split=split)
        self.system_prompt: str = QA_SYSTEM_PROMPT_COT if enable_cot else QA_SYSTEM_PROMPT
        self.max_new_tokens: int = QA_MAX_NEW_TOKENS_COT if enable_cot else QA_MAX_NEW_TOKENS
        self.query_prompt: str = QA_QUERY_PROMPT_COT if enable_cot else QA_QUERY_PROMPT
        print(f"Loading dataset from {data_path}...")
        self.metric = max_f1_score_with_reasoning if enable_cot else max_f1_score
        print(f"Done loading {data_path}")
    
    @staticmethod
    def extract_documents(
        sample: Dict[str, Any], 
        query_prompt: str | None=None, 
        data_path: str | None=None) -> Dict[str, Any]:
        context_list = []
        if data_path is None:
            for item in sample['context'].split('Passage'):
                if len(item) > 10:
                    context_list.append('Passage' + item + '\n\n')
        elif isinstance(sample['context'], dict):
            for i, (title, lines) in enumerate(zip(sample['context']['title'], sample['context']['content'])):
                context_str = f"Document {i+1} (title: {title}) " + " ".join(lines) + '\n\n'
                context_list.append(context_str)
        else:
            for i, item in enumerate(eval(sample['context'])):
                context_str = f"Document {i+1} (title: {item[0]}) " + " ".join(item[1]) + '\n\n'
                context_list.append(context_str)
        answer = sample['answers' if data_path is None else 'answer']
        query = sample['input' if data_path is None else 'question']
        query_prompt = query if query_prompt is None else query_prompt + query
        return {
            'qid': sample['_id'],
            'question': query_prompt,
            'documents': context_list,
            'answer': answer if isinstance(answer, list) else [answer],
        }
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.extract_documents(self.data[idx], query_prompt=self.query_prompt, data_path=self.data_path)


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
    def extract_documents(sample: Dict[str, Any], query_prompt: str | None=None, only_supporting: bool=False) -> Dict[str, Any]:
        context_list = []
        query_prompt = sample['question'] if query_prompt is None else query_prompt + sample['question']
        for item in sample['paragraphs']:
            if not item['is_supporting'] and only_supporting:
                continue
            context_list.append(f"Document {item['idx']} (title: {item['title']}) " + item['paragraph_text'] + '\n\n')
        return {
            'qid': sample['id'],
            'question': query_prompt,
            'documents': context_list,
            'answer': [sample['answer']] + sample['answer_aliases'],
        }
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.extract_documents(self.data[idx], query_prompt=self.query_prompt, only_supporting=self.only_supporting)


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
            'question': self.query_prompt + self.question[idx],
            'documents': context_list,
            'answer': self.answer[idx],
        }


class MultiNewsDataset(AbstractMDQADataset):
    def __init__(self) -> None:
        print(f"Loading dataset from zai-org/LongBench multi_news...")
        self.data = datasets.load_dataset('zai-org/LongBench', 'multi_news')['test']
        self.system_prompt: str = "You are given several news passages. Write a one-page summary of all news."
        self.max_new_tokens: int = 512
        self.paragraphs = self.data['context']
        self.qid = self.data['_id']
        self.answer = self.data['answers']
        self.question = '\n\nNow, write a one-page summary of all the news.'
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
        self.question_prompt: str = "As the above examples, please summarize the dialogue into a few short sentences.\n\n"
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
            'question': self.question_prompt + self.question[idx],
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
        self.system_prompt = "You are a helpful assistant."
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
        return {
            'qid': str(idx),
            'question': question,
            'documents': documents,
            'answer': [self.CONTEXT_END + sample['output']],
        }

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


def gsm8k_normalize_answer(answer: str) -> str:
    """
    标准化答案：去除逗号、空格、百分号等，统一格式。
    """
    # 去除千位分隔符逗号
    answer = answer.replace(",", "")
    # 去除美元符号等常见前缀
    answer = answer.replace("$", "").strip()
    # 处理百分比: "50%" -> "0.5" 或保持原样视情况而定
    if answer.endswith("%"):
        try:
            val = float(answer[:-1]) / 100
            # 如果结果是整数，转为整数字符串
            if val == int(val):
                return str(int(val))
            return str(val)
        except ValueError:
            pass
    # 去除多余空格
    answer = answer.strip()
    return answer

_GSM8K_MATCHER1 = regex.compile(r"#### (-?[0-9.,]+)")
_GSM8K_MATCHER2 = regex.compile(r"-?\d+\.?\d*")
class GSM8KDataset(AbstractMDQADataset):
    def __init__(self, shot_num: int = 4) -> None:
        data = datasets.load_dataset('openai/gsm8k', 'main', split='test')
        self.example_documents = []
        for i in range(shot_num):
            self.example_documents.append(data[i]['question'] + "\n\n" + data[i]['answer'])
        self.data = data.select(range(shot_num, len(data))).map(self.precess_sample, 
            fn_kwargs={'documents': self.example_documents},
            with_indices=True, batched=False, num_proc=32, remove_columns=data.column_names
        )
        self.system_prompt = ("You are a helpful assistant to answer math questions. "
            "You are given several examples, and you are asked to answer the question. "
            "Please solve the question step-by-step and write the final answer after '#### '.\n\n")
        self.max_new_tokens: int = 1024
    
    @staticmethod
    def metric(pred: str, gt_list: List[str]) -> float:
        gt_m = _GSM8K_MATCHER1.search(gt_list[0])
        pred_m = _GSM8K_MATCHER1.search(pred)
        if pred_m is None:
            pred_m = _GSM8K_MATCHER2.findall(pred)
            pred_m = pred_m[-1] if pred_m else None
        else:
            pred_m = pred_m.group(1)
        assert gt_m is not None, f"No match found in {gt_list[0]}"
        gt_m = gsm8k_normalize_answer(gt_m.group(1))
        if pred_m and gsm8k_normalize_answer(pred_m) == gt_m:
            return 1.0
        return 0.0
    
    @staticmethod
    def precess_sample(sample: Dict[str, Any], indice: int, documents: List[str]) -> Dict[str, Any]:
        query_prompt = ("Following the above examples. First solve the following question step-by-step,"
                        "Write the final answer after '#### <number>', only output the final numeric answer in the format of '#### <number>.\n\n")
        return {
            'qid': str(indice),
            'question': query_prompt + sample['question'],
            'documents': documents,
            'answer': [sample['answer']],
        }
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


class NeedleDataset(AbstractMDQADataset):
    def __init__(self, path: str):
        self.data = datasets.load_dataset("jsonl", data_files=path)['train']
        self.system_prompt = "You are a helpful assistant. Use only the information in the context to answer the question."
        self.max_new_tokens: int = 128
        self.data = self.data.map(self._process_sample, num_proc=64, remove_columns=self.data.column_names, batched=False)
    
    @staticmethod
    def _process_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'qid': sample['id'],
            'question': QA_QUERY_PROMPT + sample['question'],
            'documents': sample['context'],
            'answer': [sample['answer']],
        }

    @staticmethod
    def metric(pred: str, gt_list: List[str]) -> float:
        pred = pred.strip().lower()
        if any(gt.strip().lower() in pred for gt in gt_list):
            return 1.0
        return max_f1_score(pred, gt_list)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


class RULERDataset(AbstractMDQADataset):
    """
    NVIDIA RULER benchmark dataset adapter.
    Supports NIAH, variable tracking, common words extraction, and QA tasks.
    Splits input into system prompt, context chunks, and question for KV concatenation.
    """

    def __init__(self, path: str, chunk_size: int = 8192):
        """
        Args:
            path: Path to RULER JSONL file
            chunk_size: Maximum characters per context chunk for KV concatenation
        """
        self.data = datasets.load_dataset("json", data_files=path)['train']
        self.chunk_size = chunk_size
        self.system_prompt = None  # Will be set per-example based on task type
        self.max_new_tokens = 128  # Default, can be overridden per task
        print(f"Loading RULER dataset from {path}...")
        self.data = self.data.map(
            self._process_sample,
            num_proc=8,
            remove_columns=self.data.column_names,
            batched=False
        )
        print(f"Done loading RULER dataset with {len(self.data)} examples")

    @staticmethod
    def _detect_task_type(input_text: str) -> str:
        """Detect RULER task type from input text."""
        if "special magic" in input_text.lower() or "needle" in input_text.lower():
            return "niah"
        elif "variable assignment" in input_text.lower() or "track the chain" in input_text.lower():
            return "variable_tracking"
        elif "most often" in input_text.lower() or "common words" in input_text.lower():
            return "common_words"
        elif "frequency" in input_text.lower() and "ignore the dots" in input_text.lower():
            return "frequency_words"
        elif "answer the question based on" in input_text.lower():
            return "qa"
        else:
            return "unknown"

    @staticmethod
    def _split_ruler_input(input_text: str, task_type: str, chunk_size: int) -> Dict[str, Any]:
        """
        Split RULER input into system prompt, context chunks, and question.

        Returns:
            Dict with 'system_prompt', 'documents' (list of chunks), 'question'
        """
        # Common patterns to identify sections
        context_markers = [
            "\n\n",  # Double newline often separates sections
            "The following is",
            "Below is",
            "Here is",
        ]

        question_markers = [
            "What is the",
            "Which",
            "How many",
            "Answer:",
            "Question:",
            "Find the",
            "Extract the",
        ]

        # Strategy: Find the last question marker to separate question from context
        question_start = -1
        question_marker_found = None

        for marker in question_markers:
            pos = input_text.rfind(marker)
            if pos > question_start:
                question_start = pos
                question_marker_found = marker

        if question_start == -1:
            # No clear question found, treat last 500 chars as question
            question_start = max(0, len(input_text) - 500)

        # Split into instruction + context and question
        context_part = input_text[:question_start].strip()
        question_part = input_text[question_start:].strip()

        # Find where actual context starts (after initial instruction)
        instruction_end = 0
        for i, line in enumerate(context_part.split('\n')):
            if i < 5 and (len(line) < 200 or any(marker in line for marker in ["following", "Below", "Memorize"])):
                instruction_end += len(line) + 1
            else:
                break

        system_prompt = context_part[:instruction_end].strip()
        context_text = context_part[instruction_end:].strip()

        # Split context into chunks for KV concatenation
        documents = []
        if len(context_text) <= chunk_size:
            documents = [context_text] if context_text else [""]
        else:
            # Split by paragraphs first
            paragraphs = context_text.split('. ')
            current_chunk = ""

            for para in paragraphs:
                if len(current_chunk) + len(para) + 2 <= chunk_size:
                    current_chunk += para + "\n"
                else:
                    if current_chunk:
                        documents.append(current_chunk.strip())
                    current_chunk = para + "\n"

            if current_chunk:
                documents.append(current_chunk.strip())

        # Ensure documents is never empty to maintain type consistency
        if not documents:
            documents = [""]

        return {
            'system_prompt': system_prompt if system_prompt else "You are a helpful assistant.",
            'documents': documents,
            'question': question_part
        }

    def _process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single RULER sample."""
        input_text = sample['input']
        task_type = self._detect_task_type(input_text)

        # Split input into components
        split_result = self._split_ruler_input(input_text, task_type, self.chunk_size)

        # Determine max_new_tokens based on task
        if task_type in ["niah", "variable_tracking"]:
            max_new_tokens = 64
        elif task_type in ["common_words", "frequency_words"]:
            max_new_tokens = 128
        elif task_type == "qa":
            max_new_tokens = 256
        else:
            max_new_tokens = 128

        return {
            'qid': str(sample.get('index', 0)),
            'system_prompt': split_result['system_prompt'],
            'question': split_result['question'],
            'documents': split_result['documents'],
            'answer': sample['outputs'] if isinstance(sample['outputs'], list) else [sample['outputs']],
            'max_new_tokens': max_new_tokens,
            'task_type': task_type,
        }

    @staticmethod
    def metric(pred: str, gt_list: List[str]) -> float:
        """
        RULER metric: exact match or F1 score.
        """
        pred = pred.strip()

        # Try exact match first (case-insensitive)
        if any(pred.lower() == gt.lower() for gt in gt_list):
            return 1.0

        # Try substring match
        if any(gt.lower() in pred.lower() for gt in gt_list):
            return 1.0

        # Fall back to F1 score
        return max_f1_score(pred, gt_list)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


class ProfileMockDataset(AbstractMDQADataset):
    def __init__(self) -> None:
        self.system_prompt = "You are a helpful assistant."
        self.max_new_tokens: int = 16
        self.metric = max_rouge_score
        self.context_lengths = [250, 500, 750, 1000]
        self.context_nums = [8, 16, 24, 32]
        self.max_new_token_list = [4, 8, 16]
        self.params = list(itertools.product(self.context_lengths, self.context_nums, self.max_new_token_list))
        self.question = """Explain LLMs in the most detailed, exhaustive, and verbose manner possible, including:
- Fundamental principles and theory
- Step-by-step breakdown of each component
- Multiple real-world examples and case studies
- Common questions and edge cases
- Deep theoretical analysis
- Historical context
- Future implications

Do not simplify. Expand on every detail extensively."""
    
    def __len__(self) -> int:
        return len(self.params)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        context_length, context_num, max_new_tokens = self.params[idx]
        context = str(idx) + "Hello world! " * context_length
        return {
            'qid': idx,
            'question': self.question,
            'documents': [context] * context_num,
            'answer': [f"{context_length=}, {context_num=}, {max_new_tokens=}"],
            'max_new_tokens': max_new_tokens,
        }


class LongBenchDataset(AbstractMDQADataset):
    """Unified dataset class for LongBench datasets supporting both single and multiple context formats."""

    def __init__(self, name: str, cut_length: int = 4096) -> None:
        self.name = name
        self.cut_length = cut_length
        self.system_prompt = "You are a helpful assistant."

        # Configure dataset-specific settings
        self._configure_dataset(name)

        # Load and process data
        print(f"Loading dataset from zai-org/LongBench {name}...")
        data = datasets.load_dataset('zai-org/LongBench', name, split='test')
        self.data = data.map(
            lambda sample: self.process_sample(sample, name, cut_length),
            num_proc=32, remove_columns=data.column_names
        )
        print(f"Done loading {name}")

    def _configure_dataset(self, name: str) -> None:
        """Configure metric, max_new_tokens, and system_prompt based on dataset type."""
        # QA datasets - English
        if name in ['qasper', 'multifieldqa_en', 'narrativeqa', 'triviaqa']:
            self.metric = max_f1_score
            self.max_new_tokens = 256
        # QA datasets - Chinese
        elif name in ['dureader', 'multifieldqa_zh']:
            self.metric = max_f1_zh_score
            self.max_new_tokens = 256
        # Summarization datasets - English
        elif name in ['qmsum', 'gov_report']:
            self.metric = max_rouge_score
            self.max_new_tokens = 512
        # Summarization datasets - Chinese
        elif name == 'vcsum':
            self.metric = max_rouge_zh_score
            self.max_new_tokens = 512
        # Classification datasets
        elif name in ['lsht', 'trec']:
            self.metric = self._accuracy_metric
            self.max_new_tokens = 32
        # Code understanding datasets
        elif name in ['lcc', 'repobench-p']:
            self.metric = max_f1_score
            self.max_new_tokens = 256
        # Passage retrieval datasets
        elif name in ['passage_count', 'passage_retrieval_en', 'passage_retrieval_zh']:
            self.metric = self._exact_match_metric
            self.max_new_tokens = 32
        else:
            raise ValueError(f"Unsupported dataset: {name}")

    @staticmethod
    def _accuracy_metric(pred: str, gt_list: List[str]) -> float:
        """Exact match accuracy for classification tasks."""
        pred = pred.strip()
        return 1.0 if any(pred == gt.strip() for gt in gt_list) else 0.0

    @staticmethod
    def _exact_match_metric(pred: str, gt_list: List[str]) -> float:
        """Exact match for passage retrieval tasks."""
        pred = pred.strip()
        for gt in gt_list:
            if gt.strip() in pred or pred in gt.strip():
                return 1.0
        return 0.0

    @staticmethod
    def process_sample(sample: Dict[str, Any], name: str, cut_length: int) -> Dict[str, Any]:
        """Process a sample based on dataset type."""
        dataset_type = sample['dataset']

        QA_QUERY_PROMPT_ZH = "请直接回答问题，只输出答案，不要有任何解释或额外文字。\n例子：问题：巴黎是哪个国家的首都？法国。\n\n问题："

        # QA datasets
        if dataset_type == 'qasper':
            question = """Answer the question directly based on the given passages.
Output only the answer. No explanation. No extra text.\n\nQuestion: """ + sample['input']
            documents = LongBenchDataset._split_context(sample['context'], cut_length)

        elif dataset_type == 'dureader':
            question = QA_QUERY_PROMPT_ZH + sample['input']
            # Split by article markers
            documents = LongBenchDataset._split_by_markers(sample['context'], ['文章'], cut_length)

        elif dataset_type == 'multifieldqa_en':
            question = QA_QUERY_PROMPT + sample['input']
            documents = LongBenchDataset._split_context(sample['context'], cut_length)

        elif dataset_type == 'multifieldqa_zh':
            question = QA_QUERY_PROMPT_ZH + sample['input']
            documents = LongBenchDataset._split_context(sample['context'], cut_length)

        elif dataset_type == 'narrativeqa':
            question = QA_QUERY_PROMPT + sample['input']
            documents = LongBenchDataset._split_context(sample['context'], cut_length)

        elif dataset_type == 'triviaqa':
            question = QA_QUERY_PROMPT + sample['input']
            # Context has "Passage:" markers
            documents = LongBenchDataset._split_by_markers(sample['context'], ['Passage:'], cut_length)

        # Summarization datasets
        elif dataset_type == 'gov_report':
            question = "Summarize the given government report. Output only the summary with no extra text or preamble."
            documents = LongBenchDataset._split_context(sample['context'], cut_length)

        elif dataset_type == 'qmsum':
            question = """Answer the question based on the given passages.
Output only the answer in one paragraph. No markdown format. No explanation or extra text.\n\nQuestion: """ + sample['input']
            documents = LongBenchDataset._split_context(sample['context'], cut_length)

        elif dataset_type == 'vcsum':
            question = "请总结以上对话，只输出摘要内容，不要有任何额外文字或前言。"
            # Split by speaker markers
            documents = LongBenchDataset._split_by_markers(sample['context'], ['讲者'], cut_length)

        # Classification datasets
        elif dataset_type == 'lsht':
            question = sample['input'] + "\n\n请从以下类别中选择一个最合适的，只输出类别名称，不要有任何解释或额外文字：\n" + "、".join(sample['all_classes'])
            # Context contains example news articles
            documents = LongBenchDataset._split_by_markers(sample['context'], ['新闻内容：'], cut_length)

        elif dataset_type == 'trec':
            question = sample['input'] + "\n\nCategories: " + ", ".join(sample['all_classes']) + "\n\nChoose the most appropriate category. Output only the category name with no explanation or extra text."
            # Context contains example questions
            documents = LongBenchDataset._split_by_markers(sample['context'], ['Question:'], cut_length)

        # Code understanding datasets
        elif dataset_type == 'lcc':
            question = ("Complete the following code by finding the next line. Output only the code line with no explanation or extra text.\n\n" + sample['input']) if sample['input'] else "Find the missing line in the code. Output only the code line."
            documents = LongBenchDataset._split_context(sample['context'], cut_length)

        elif dataset_type == 'repobench-p':
            question = "Based on the code context, complete the following code snippet. Output only the code with no explanation or extra text.\n\n" + sample['input']
            documents = LongBenchDataset._split_context(sample['context'], cut_length)

        # Passage retrieval datasets
        elif dataset_type == 'passage_count':
            question = "How many paragraphs are there in the following text? Output only the number with no explanation or extra text."
            documents = LongBenchDataset._split_by_markers(sample['context'], ['Paragraph'], cut_length, keep_marker=True)

        elif dataset_type == 'passage_retrieval_en':
            question = "Which paragraph contains information about: " + sample['input'] + "\n\nOutput only the paragraph number (e.g., 'Paragraph 15') with no explanation or extra text."
            documents = LongBenchDataset._split_by_markers(sample['context'], ['Paragraph'], cut_length, keep_marker=True)

        elif dataset_type == 'passage_retrieval_zh':
            question = "哪一段包含以下信息：" + sample['input'] + "\n\n只输出段落编号（例如：'段落15'），不要有任何解释或额外文字。"
            documents = LongBenchDataset._split_by_markers(sample['context'], ['Paragraph'], cut_length, keep_marker=True)

        else:
            raise ValueError(f"Unsupported dataset: {dataset_type}")

        return {
            'qid': sample['_id'],
            'question': question,
            'documents': documents,
            'answer': sample['answers'] if isinstance(sample['answers'], list) else [sample['answers']],
        }

    @staticmethod
    def _split_context(context: str, cut_length: int) -> List[str]:
        """Split context into documents by cut_length."""
        documents = []
        last_document = ''
        sep = '\n' if '\n' in context else '. '

        for line in context.split(sep):
            if not line.strip():
                continue
            if len(last_document) + len(line) > cut_length:
                if last_document:
                    documents.append(last_document)
                last_document = line + sep
            else:
                last_document += line + sep

        if last_document:
            documents.append(last_document)

        return documents if documents else [context]

    @staticmethod
    def _split_by_markers(context: str, markers: List[str], cut_length: int, keep_marker: bool = False) -> List[str]:
        """Split context by specific markers (e.g., 'Paragraph', '文章')."""
        documents = []

        # Try to split by markers
        for marker in markers:
            if marker in context:
                parts = context.split(marker)
                for i, part in enumerate(parts):
                    if i == 0 and not part.strip():
                        continue
                    doc = (marker + part) if keep_marker and i > 0 else part
                    if doc.strip():
                        # Further split if too long
                        if len(doc) > cut_length * 2:
                            documents.extend(LongBenchDataset._split_context(doc, cut_length))
                        else:
                            documents.append(doc)
                return documents

        # Fallback to regular splitting if no markers found
        return LongBenchDataset._split_context(context, cut_length)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


def load_mdoc_dataset(name: str, path: Optional[str]=None, **kwargs) -> AbstractMDQADataset:
    if name == "musique":
        if path is None:
            print('Defaulting musique dataset path to "../datasets/musique.jsonl"')
            path = "../datasets/musique.jsonl"
        return MusiqueDataset(path, **kwargs)
    kwargs.pop('only_supporting', None)
    if name == "wikimqa":
        return WikiMQADataset(path, **kwargs)
    elif name == "hotpotqa":
        return HotpotQADataset(**kwargs)
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
    elif name == "gsm8k":
        return GSM8KDataset()
    elif name == "needle":
        if path is None:
            print('Defaulting needle dataset path to "../datasets/needle_haystack_testset.jsonl"')
            path = "../datasets/needle_haystack_testset.jsonl"
        return NeedleDataset(path)
    elif name == "profile":
        return ProfileMockDataset()
    elif name == "ruler":
        if path is None:
            raise ValueError("RULER dataset requires a path to the JSONL file")
        chunk_size = kwargs.get('chunk_size', 8192)
        return RULERDataset(path, chunk_size=chunk_size)
    # LongBench datasets (original 3 + new 13)
    elif name in [
        'qmsum', 'gov_report', 'qasper',  # Original 3
        'dureader', 'multifieldqa_en', 'multifieldqa_zh', 'narrativeqa', 'triviaqa',  # QA (5)
        'lsht', 'trec',  # Classification (2)
        'lcc', 'repobench-p',  # Code understanding (2)
        'passage_count', 'passage_retrieval_en', 'passage_retrieval_zh',  # Passage retrieval (3)
        'vcsum'  # Summarization (1)
    ]:
        return LongBenchDataset(name, kwargs.get('cut_length', 4096))
    else:
        raise ValueError(f"Unsupported dataset name: {name}")
