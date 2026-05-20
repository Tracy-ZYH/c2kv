import argparse
import json
import os
import time
import warnings
from typing import List, Dict, Any, Optional

import numpy as np
import requests
from tqdm import tqdm

from mdocdataset import AbstractMDQADataset, load_mdoc_dataset


def cut_documents(documents: List[str], max_length: int | None) -> List[str]:
    if max_length is None:
        return documents
    docs = []
    for document in documents:
        last_document = ''
        for passage in document.split('\n\n'):
            if not passage.strip():
                continue
            if len(last_document) + len(passage) > max_length:
                docs.append(last_document)
                last_document = passage
            else:
                last_document += passage + '\n\n'
        if last_document:
            docs.append(last_document)
    return docs


def extract_document(base_url: str, text: str, compression_ratio: int = 4, role: str = "user") -> dict:
    url = f"{base_url}/v1/c2kv/extract"
    payload = {"text": text, "compression_ratio": compression_ratio, "role": role}
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def chat_completion(base_url: str, model: str, messages: list, max_new_tokens: int, temperature: float = 0.0) -> str:
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_new_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def evaluate_via_api(args: argparse.Namespace, dataset: AbstractMDQADataset) -> Dict[str, float]:
    base_url = args.base_url.rstrip("/")
    model = args.model
    max_examples = args.max_examples
    output_file = args.output_file
    cut_length = args.cut_length
    profile = args.profile
    compression_ratio = args.compression_ratio

    em_scores = []
    results = []
    extract_times = []
    chat_times = []

    num_examples = len(dataset) if max_examples is None else min(max_examples, len(dataset))
    max_new_tokens = dataset.max_new_tokens

    dataset.system_prompt = "You are a helpful assistant."

    for i in tqdm(range(num_examples)):
        example = dataset[i]

        system_prompt = example.get('system_prompt', dataset.system_prompt)
        if 'max_new_tokens' in example:
            max_new_tokens = example['max_new_tokens']

        documents = cut_documents(example['documents'], max_length=cut_length)

        # Extract each document into C2KV cache
        doc_messages = []
        t_extract_start = time.perf_counter()
        for doc in documents:
            try:
                result = extract_document(base_url, doc, compression_ratio, role="user")
                if result.get("success") and result.get("key_hash"):
                    doc_messages.append({
                        "role": "user",
                        "content": doc,
                        "c2kv_key_hash": result["key_hash"],
                    })
                else:
                    warnings.warn(f"Extraction failed for qid={example['qid']}: {result.get('error')}")
                    doc_messages.append({"role": "user", "content": doc})
            except requests.RequestException as e:
                warnings.warn(f"Extract request failed for qid={example['qid']}: {e}")
                doc_messages.append({"role": "user", "content": doc})
        t_extract = time.perf_counter() - t_extract_start

        # Build full message list
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(doc_messages)
        messages.append({"role": "user", "content": example['question']})

        # Generate
        t_chat_start = time.perf_counter()
        pred = chat_completion(base_url, model, messages, max_new_tokens)
        t_chat = time.perf_counter() - t_chat_start

        print(pred)

        em_score = dataset.metric(pred, example['answer'])
        em_scores.append(em_score)

        record: Dict[str, Any] = {
            'qid': example['qid'],
            'prediction': pred,
            'ground_truth': example['answer'],
            'em_score': em_score,
        }
        if profile:
            record['timer'] = {'extract': round(t_extract, 4), 'chat': round(t_chat, 4)}
            extract_times.append(t_extract)
            chat_times.append(t_chat)

        results.append(record)

    exact_match = sum(em_scores) / len(em_scores) if em_scores else 0.0

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        summary: Dict[str, Any] = {
            'base_url': base_url,
            'model': model,
            'dataset': dataset.__class__.__name__,
            'num_examples': len(results),
            'exact_match': exact_match,
            'compression_ratio': compression_ratio,
        }
        if profile and extract_times:
            summary['extract_mean'] = round(float(np.mean(extract_times)), 4)
            summary['extract_std'] = round(float(np.std(extract_times)), 4)
            summary['chat_mean'] = round(float(np.mean(chat_times)), 4)
            summary['chat_std'] = round(float(np.std(chat_times)), 4)

        summary_file = output_file.replace('.jsonl', '_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

    return {'exact_match': exact_match, 'num_examples': len(results)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate C2KV inference engine on MDQA datasets via HTTP API")

    parser.add_argument("--base-url", type=str, default="http://localhost:30000",
                        help="SGLang server base URL (e.g. http://localhost:30000)")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Type of dataset (musique, wikimqa, hotpotqa, …)")
    parser.add_argument("--model", type=str, default="default",
                        help="Model name passed in chat completions payload")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Dataset file path")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Maximum number of examples to evaluate")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output JSONL file path for results")
    parser.add_argument("--only_supporting", action="store_true",
                        help="For Musique dataset, use only supporting paragraphs")
    parser.add_argument("--cot", action="store_true", default=False,
                        help="Use chain-of-thought prompt")
    parser.add_argument("--cut_length", type=int, default=None,
                        help="Cut documents to at most this many characters")
    parser.add_argument("--compression-ratio", type=int, default=4, dest="compression_ratio",
                        help="Gist compression ratio passed to /v1/c2kv/extract (default: 4)")
    parser.add_argument("--profile", action="store_true", default=False,
                        help="Record extract and chat latencies per example")

    args = parser.parse_args()

    if args.output_file and os.path.exists(args.output_file):
        print(f"Output file {args.output_file} already exists, skipping")
        return

    dataset = load_mdoc_dataset(
        args.dataset,
        args.dataset_path,
        only_supporting=args.only_supporting,
        enable_cot=args.cot,
    )

    print(f"Loaded {len(dataset)} examples from {args.dataset} dataset")

    results = evaluate_via_api(args, dataset)

    print(f"\nEvaluation Results:")
    print(f"Base URL: {args.base_url}")
    print(f"Dataset: {args.dataset}")
    print(f"Number of examples evaluated: {results['num_examples']}")
    print(f"Score: {results['exact_match']:.4f}")


if __name__ == "__main__":
    main()
