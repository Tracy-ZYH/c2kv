import argparse
import json
import os
import re
import string
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from datasets import load_dataset, load_from_disk

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


QA_SYSTEM_PROMPT = (
    "You will be asked a question after reading several passages. "
    "Please directly answer the question based on the given passages.\n\n"
)

QA_QUERY_PROMPT = (
    "Answer the question directly based on the given passages. "
    "Output exactly one phrase as the final answer. No explanation. No extra text.\n"
    "Example: Question: What is the capital of France? Paris.\n\n"
    "Question: "
)

HTTP = requests.Session()
HTTP.trust_env = False


def normalize_answer(text: str) -> str:
    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_punc(s: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in s if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def qa_f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return float(pred_tokens == gt_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def max_f1_score(prediction: str, answers: List[str]) -> float:
    return max((qa_f1_score(prediction, answer) for answer in answers), default=0.0)


def max_exact_match(prediction: str, answers: List[str]) -> float:
    normalized_prediction = normalize_answer(prediction)
    return float(
        any(normalized_prediction == normalize_answer(answer) for answer in answers)
    )


def load_hotpotqa(path: str):
    path_obj = Path(path)
    if path_obj.is_dir():
        return load_from_disk(str(path_obj))
    if path_obj.is_file():
        return load_dataset("json", data_files=str(path_obj))["train"]
    return load_dataset("zai-org/LongBench", "hotpotqa")["test"]


def split_hotpotqa_context(context: str) -> List[str]:
    documents = []
    for item in context.split("Passage"):
        if len(item) > 10:
            documents.append("Passage" + item + "\n\n")
    return documents


def build_example(sample: Dict[str, Any]) -> Dict[str, Any]:
    if {"qid", "question", "documents", "answer"}.issubset(sample):
        return {
            "qid": sample["qid"],
            "question": sample["question"],
            "documents": list(sample["documents"]),
            "answer": list(sample["answer"]),
        }

    return {
        "qid": sample["_id"],
        "question": QA_QUERY_PROMPT + sample["input"],
        "documents": split_hotpotqa_context(sample["context"]),
        "answer": list(sample["answers"]),
    }


def cut_documents(documents: List[str], max_length: Optional[int]) -> List[str]:
    if max_length is None:
        return documents

    docs = []
    for document in documents:
        last_document = ""
        for passage in document.split("\n\n"):
            if not passage.strip():
                continue
            if len(last_document) + len(passage) > max_length:
                if last_document:
                    docs.append(last_document)
                last_document = passage
            else:
                last_document += passage + "\n\n"
        if last_document:
            docs.append(last_document)
    return docs


def post_json(base_url: str, path: str, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    resp = HTTP.post(
        f"{base_url.rstrip('/')}{path}",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def extract_document(
    base_url: str,
    text: str,
    compression_ratio: int,
    timeout: int,
) -> Dict[str, Any]:
    return post_json(
        base_url,
        "/v1/c2kv/extract",
        {
            "text": text,
            "compression_ratio": compression_ratio,
            "role": "user",
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout,
    )


def chat_completion(
    base_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = post_json(base_url, "/v1/chat/completions", payload, timeout)
    content = data["choices"][0]["message"].get("content")
    return content if isinstance(content, str) else ""


def evaluate_one(
    idx: int,
    sample: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[int, Dict[str, Any]]:
    example = build_example(sample)
    documents = cut_documents(example["documents"], args.cut_length)

    extract_records = []
    doc_messages = []
    t0 = time.perf_counter()
    if args.mode == "full":
        doc_messages = [{"role": "user", "content": doc} for doc in documents]
    else:
        extract_docs = documents
        if args.mode == "combined-c2kv":
            extract_docs = ["".join(documents)] if documents else []

        for doc_idx, doc in enumerate(extract_docs):
            try:
                result = extract_document(
                    args.base_url,
                    doc,
                    args.compression_ratio,
                    args.timeout,
                )
                success = bool(result.get("success") and result.get("key_hash"))
                extract_records.append(
                    {
                        "doc_idx": doc_idx,
                        "success": success,
                        "key_hash": result.get("key_hash"),
                        "gist_len": result.get("gist_len"),
                        "original_seq_len": result.get("original_seq_len"),
                        "error": result.get("error"),
                    }
                )
                if success:
                    doc_messages.append(
                        {
                            "role": "user",
                            "content": doc,
                            "c2kv_key_hash": result["key_hash"],
                        }
                    )
                else:
                    warnings.warn(f"[{example['qid']}] extract failed: {result}")
                    doc_messages.append({"role": "user", "content": doc})
            except Exception as exc:
                warnings.warn(f"[{example['qid']}] extract error: {exc}")
                extract_records.append(
                    {"doc_idx": doc_idx, "success": False, "error": str(exc)}
                )
                doc_messages.append({"role": "user", "content": doc})

        if args.mode == "combined-c2kv" and not doc_messages:
            doc_messages = [
                {"role": "user", "content": "".join(documents)}
            ]
    extract_seconds = time.perf_counter() - t0

    messages = [{"role": "system", "content": args.system_prompt}]
    messages.extend(doc_messages)
    messages.append({"role": "user", "content": example["question"]})

    t1 = time.perf_counter()
    try:
        prediction = chat_completion(
            args.base_url,
            args.model,
            messages,
            args.max_tokens,
            args.temperature,
            args.timeout,
        )
        chat_error = None
    except Exception as exc:
        warnings.warn(f"[{example['qid']}] chat error: {exc}")
        prediction = ""
        chat_error = str(exc)
    chat_seconds = time.perf_counter() - t1

    answers = example["answer"]
    record = {
        "idx": idx,
        "qid": example["qid"],
        "mode": args.mode,
        "prediction": prediction,
        "ground_truth": answers,
        "f1": max_f1_score(prediction, answers),
        "exact_match": max_exact_match(prediction, answers),
        "num_documents": len(documents),
        "num_c2kv_messages": sum(
            1 for msg in doc_messages if "c2kv_key_hash" in msg
        ),
        "extracts": extract_records,
        "timing": {
            "extract_seconds": round(extract_seconds, 4),
            "chat_seconds": round(chat_seconds, 4),
        },
    }
    if chat_error is not None:
        record["chat_error"] = chat_error
    return idx, record


def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_summary(path: str, summary: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def summarize(args: argparse.Namespace, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_extracts = [item for record in records for item in record["extracts"]]
    successful_extracts = [item for item in all_extracts if item.get("success")]
    original_tokens = [
        item["original_seq_len"]
        for item in successful_extracts
        if isinstance(item.get("original_seq_len"), int)
    ]
    gist_tokens = [
        item["gist_len"]
        for item in successful_extracts
        if isinstance(item.get("gist_len"), int)
    ]
    total_original = sum(original_tokens)
    total_gist = sum(gist_tokens)
    prediction_counts = Counter(record["prediction"] for record in records)

    return {
        "base_url": args.base_url,
        "model": args.model,
        "dataset_path": args.dataset_path,
        "compression_ratio": args.compression_ratio,
        "mode": args.mode,
        "num_examples": len(records),
        "avg_f1": sum(record["f1"] for record in records) / len(records)
        if records
        else 0.0,
        "exact_match": sum(record["exact_match"] for record in records) / len(records)
        if records
        else 0.0,
        "num_extracts": len(all_extracts),
        "extract_success_rate": len(successful_extracts) / len(all_extracts)
        if all_extracts
        else 0.0,
        "total_original_seq_len": total_original,
        "total_gist_len": total_gist,
        "actual_compression_ratio": total_original / total_gist
        if total_gist > 0
        else None,
        "avg_extract_seconds": sum(
            record["timing"]["extract_seconds"] for record in records
        )
        / len(records)
        if records
        else 0.0,
        "avg_chat_seconds": sum(record["timing"]["chat_seconds"] for record in records)
        / len(records)
        if records
        else 0.0,
        "top_predictions": prediction_counts.most_common(20),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SGLang C2KV on LongBench HotpotQA via HTTP API."
    )
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:30000", help="SGLang base URL"
    )
    parser.add_argument("--model", default="qwen3-4b", help="served model name")
    parser.add_argument(
        "--dataset-path",
        default="/home/zhuyuhan/project/c2kv/datasets/longbench_hotpotqa_test",
    )
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--compression-ratio", type=int, default=16)
    parser.add_argument(
        "--mode",
        choices=["c2kv", "full", "combined-c2kv"],
        default="c2kv",
        help=(
            "full: send uncompressed documents; c2kv: extract each passage; "
            "combined-c2kv: extract all passages as one segment for diagnosis"
        ),
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful assistant.",
        help="System prompt. Defaults to the original C2KV API evaluator setting.",
    )
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cut-length", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split selected examples into this many shards.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Evaluate examples whose index modulo num-shards equals this value.",
    )
    args = parser.parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must satisfy 0 <= index < num-shards.")

    # Keep this script standalone: importing mdocdataset pulls in jieba even for
    # English HotpotQA, while this evaluator only needs the raw LongBench fields.
    dataset = load_hotpotqa(args.dataset_path)
    num_examples = len(dataset)
    if args.max_examples is not None:
        num_examples = min(num_examples, args.max_examples)

    selected_indices = [
        i for i in range(num_examples) if i % args.num_shards == args.shard_index
    ]
    samples = [(i, dataset[i]) for i in selected_indices]
    records: List[Dict[str, Any]] = []

    iterator = None
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(evaluate_one, i, sample, args): i for i, sample in samples
        }
        completed = as_completed(futures)
        if tqdm is not None:
            iterator = tqdm(completed, total=len(futures))
        else:
            iterator = completed
        for future in iterator:
            idx, record = future.result()
            records.append(record)

    final_records = sorted(records, key=lambda record: record["idx"])
    write_jsonl(args.output_file, final_records)

    summary = summarize(args, final_records)
    summary["num_shards"] = args.num_shards
    summary["shard_index"] = args.shard_index
    summary["selected_examples"] = len(selected_indices)
    summary_file = args.output_file.replace(".jsonl", ".summary.json")
    write_summary(summary_file, summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
