"""
Data cleaning script that uses a large LLM (via API) to answer questions
in the training set. Samples where the LLM cannot produce a correct answer
are considered low-quality and filtered out.

Usage:
    python -m train.clean_data \
        --input_path /path/to/dataset \
        --output_path /path/to/cleaned_dataset \
        --dataset_type musique \
        --api_base https://your-api-endpoint/v1 \
        --api_key YOUR_API_KEY \
        --model gpt-4o \
        --f1_threshold 0.5 \
        --max_concurrent 8
"""

import os
import json
import argparse
import asyncio
import time
from typing import Dict, List, Any, Optional
from logging import getLogger, basicConfig, INFO
from dataclasses import dataclass, field

import datasets
from tqdm import tqdm

try:
    from inference.mdocdataset import (
        load_mdoc_dataset, QA_SYSTEM_PROMPT, qa_f1_score,
    )
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from inference.mdocdataset import (
        load_mdoc_dataset, QA_SYSTEM_PROMPT, qa_f1_score,
    )

logger = getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading (mirrors train_data.py logic)
# ---------------------------------------------------------------------------

def load_raw_dataset(dataset_type: str, input_path: str) -> datasets.Dataset:
    """Load the raw HuggingFace dataset according to its type."""
    if dataset_type == "musique":
        mdoc = load_mdoc_dataset("musique", input_path)
        return mdoc.data
    elif dataset_type == "hotpotqa":
        return datasets.load_dataset("jsonl", data_files=input_path, split="train")
    elif dataset_type == "wikimqa":
        mdoc = load_mdoc_dataset("wikimqa", "xanhho/2WikiMultihopQA", split="train")
        return mdoc.data
    elif dataset_type == "longmagpie" or dataset_type == "longalpaca":
        return datasets.load_from_disk(input_path)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def extract_qa_sample(sample: Dict[str, Any], dataset_type: str, input_path: str) -> Dict[str, Any]:
    """
    Extract a unified QA dict from a raw sample.
    Returns dict with keys: question, documents (List[str]), answer (List[str]).
    """
    if dataset_type == "musique":
        documents = []
        for item in sample["paragraphs"]:
            documents.append(
                f"Document {item['idx']} (title: {item['title']}) "
                + item["paragraph_text"]
            )
        answer = [sample["answer"]] + sample.get("answer_aliases", [])
        return {
            "question": sample["question"],
            "documents": documents,
            "answer": answer,
        }

    elif dataset_type == "hotpotqa":
        documents = []
        if isinstance(sample.get("context"), str):
            for item in sample["context"].split("Passage"):
                if len(item) > 10:
                    documents.append("Passage" + item)
        else:
            documents = sample.get("documents", [])
        answer = sample.get("answers", sample.get("answer", []))
        if isinstance(answer, str):
            answer = [answer]
        question = sample.get("input", sample.get("question", ""))
        return {"question": question, "documents": documents, "answer": answer}

    elif dataset_type == "wikimqa":
        documents = []
        context = sample.get("context", {})
        if isinstance(context, dict):
            for i, (title, lines) in enumerate(
                zip(context["title"], context["content"])
            ):
                documents.append(
                    f"Document {i+1} (title: {title}) " + " ".join(lines)
                )
        else:
            for i, item in enumerate(eval(context)):
                documents.append(
                    f"Document {i+1} (title: {item[0]}) " + " ".join(item[1])
                )
        answer = sample.get("answer", sample.get("answers", []))
        if isinstance(answer, str):
            answer = [answer]
        question = sample.get("question", sample.get("input", ""))
        return {"question": question, "documents": documents, "answer": answer}

    elif dataset_type in ("longmagpie", "longalpaca"):
        documents = sample.get("documents", [])
        answer = sample.get("answer", [])
        if isinstance(answer, str):
            answer = [answer]
        question = sample.get("question", "")
        return {"question": question, "documents": documents, "answer": answer}

    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


# ---------------------------------------------------------------------------
# LLM API caller (OpenAI-compatible)
# ---------------------------------------------------------------------------

@dataclass
class APIConfig:
    api_base: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 64
    max_retries: int = 3
    retry_delay: float = 2.0
    max_concurrent: int = 8


def _build_prompt(question: str, documents: List[str], dataset_type: str) -> List[Dict[str, str]]:
    """Build the chat messages for the LLM."""
    context_text = "\n\n".join(documents)
    if dataset_type == "longmagpie":
        user_content = (
            f"{context_text}\n\n"
            f"Answer the question directly based on the given passages. "
            f"Question: {question}"
        )
    else:
        user_content = (
            f"{context_text}\n\n"
            f"Answer the question directly based on the given passages. "
            f"Output exactly one phrase as the final answer. No explanation. No extra text.\n\n"
            f"Question: {question}"
        )
    return [
        {"role": "system", "content": QA_SYSTEM_PROMPT.strip()},
        {"role": "user", "content": user_content},
    ]


async def call_llm_async(
    session,
    messages: List[Dict[str, str]],
    config: APIConfig,
    semaphore: asyncio.Semaphore,
) -> str:
    """Call the LLM API with retries (async, OpenAI-compatible)."""
    import aiohttp

    url = f"{config.api_base.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}",
    }
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }

    for attempt in range(config.max_retries):
        try:
            async with semaphore:
                async with session.post(url, json=payload, headers=headers, timeout=60) as resp:
                    if resp.status == 429:
                        wait_time = config.retry_delay * (2 ** attempt)
                        logger.warning(f"Rate limited, waiting {wait_time:.1f}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    resp.raise_for_status()
                    result = await resp.json()
                    return result["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            logger.warning(f"API call failed (attempt {attempt+1}): {exc}")
            if attempt < config.max_retries - 1:
                await asyncio.sleep(config.retry_delay * (2 ** attempt))
    return ""


def call_llm_sync(
    messages: List[Dict[str, str]],
    config: APIConfig,
) -> str:
    """Synchronous fallback using requests."""
    import requests

    url = f"{config.api_base.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}",
    }
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }

    for attempt in range(config.max_retries):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=60)
            if resp.status_code == 429:
                wait_time = config.retry_delay * (2 ** attempt)
                logger.warning(f"Rate limited, waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue
            resp.raise_for_status()
            result = resp.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            logger.warning(f"API call failed (attempt {attempt+1}): {exc}")
            if attempt < config.max_retries - 1:
                time.sleep(config.retry_delay * (2 ** attempt))
    return ""


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def compute_max_f1(prediction: str, ground_truths: List[str]) -> float:
    """Compute the max token-level F1 between prediction and any ground truth."""
    if not prediction or not ground_truths:
        return 0.0
    return max(qa_f1_score(prediction, gt) for gt in ground_truths)


# ---------------------------------------------------------------------------
# Main cleaning pipeline
# ---------------------------------------------------------------------------

async def clean_dataset_async(
    raw_data: datasets.Dataset,
    dataset_type: str,
    input_path: str,
    config: APIConfig,
    f1_threshold: float,
    checkpoint_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Process all samples asynchronously and return kept indices + metadata."""
    import aiohttp

    # Load checkpoint if exists
    processed: Dict[int, Dict[str, Any]] = {}
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            for line in f:
                record = json.loads(line)
                processed[record["index"]] = record
        logger.info(f"Resumed from checkpoint with {len(processed)} records")

    semaphore = asyncio.Semaphore(config.max_concurrent)
    results: List[Dict[str, Any]] = list(processed.values())

    async with aiohttp.ClientSession() as session:
        tasks = []
        indices = []
        for idx in range(len(raw_data)):
            if idx in processed:
                continue
            sample = raw_data[idx]
            qa = extract_qa_sample(sample, dataset_type, input_path)
            if not qa["documents"] or not qa["question"]:
                continue
            messages = _build_prompt(qa["question"], qa["documents"], dataset_type)
            tasks.append(call_llm_async(session, messages, config, semaphore))
            indices.append((idx, qa))

        checkpoint_file = None
        if checkpoint_path:
            checkpoint_file = open(checkpoint_path, "a")

        batch_size = 100
        for batch_start in tqdm(
            range(0, len(tasks), batch_size),
            desc="Cleaning batches",
            total=(len(tasks) + batch_size - 1) // batch_size,
        ):
            batch_tasks = tasks[batch_start : batch_start + batch_size]
            batch_indices = indices[batch_start : batch_start + batch_size]
            predictions = await asyncio.gather(*batch_tasks)

            for (idx, qa), prediction in zip(batch_indices, predictions):
                f1_score = compute_max_f1(prediction, qa["answer"])
                record = {
                    "index": idx,
                    "question": qa["question"],
                    "prediction": prediction,
                    "ground_truth": qa["answer"],
                    "f1_score": f1_score,
                    "keep": f1_score >= f1_threshold,
                }
                results.append(record)
                if checkpoint_file:
                    checkpoint_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    checkpoint_file.flush()

        if checkpoint_file:
            checkpoint_file.close()

    return results


def clean_dataset_sync(
    raw_data: datasets.Dataset,
    dataset_type: str,
    input_path: str,
    config: APIConfig,
    f1_threshold: float,
    checkpoint_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Synchronous fallback for environments without asyncio/aiohttp."""
    processed: Dict[int, Dict[str, Any]] = {}
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            for line in f:
                record = json.loads(line)
                processed[record["index"]] = record
        logger.info(f"Resumed from checkpoint with {len(processed)} records")

    results: List[Dict[str, Any]] = list(processed.values())
    checkpoint_file = open(checkpoint_path, "a") if checkpoint_path else None

    for idx in tqdm(range(len(raw_data)), desc="Cleaning samples"):
        if idx in processed:
            continue
        sample = raw_data[idx]
        qa = extract_qa_sample(sample, dataset_type, input_path)
        if not qa["documents"] or not qa["question"]:
            continue

        messages = _build_prompt(qa["question"], qa["documents"])
        prediction = call_llm_sync(messages, config)
        f1_score = compute_max_f1(prediction, qa["answer"])
        record = {
            "index": idx,
            "question": qa["question"],
            "prediction": prediction,
            "ground_truth": qa["answer"],
            "f1_score": f1_score,
            "keep": f1_score >= f1_threshold,
        }
        results.append(record)
        if checkpoint_file:
            checkpoint_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            checkpoint_file.flush()

    if checkpoint_file:
        checkpoint_file.close()
    return results


def save_cleaned_dataset(
    raw_data: datasets.Dataset,
    cleaning_results: List[Dict[str, Any]],
    output_path: str,
):
    """Filter the raw dataset and save the cleaned version to disk.

    The saved dataset retains both the original answer and the LLM prediction
    as additional columns (``llm_prediction``, ``llm_f1_score``).
    """
    kept_indices = sorted(r["index"] for r in cleaning_results if r["keep"])
    total = len(cleaning_results)
    kept = len(kept_indices)
    removed = total - kept

    logger.info(f"Total processed: {total}, Kept: {kept}, Removed: {removed}")
    logger.info(f"Keep rate: {kept / total * 100:.1f}%")

    # Build index -> result mapping for fast lookup
    index_to_result = {r["index"]: r for r in cleaning_results}

    cleaned_data = raw_data.select(kept_indices)

    # Add LLM prediction and F1 score columns so that both the original
    # answer and the LLM-generated answer are preserved in the output.
    llm_predictions = [index_to_result[idx]["prediction"] for idx in kept_indices]
    llm_f1_scores = [index_to_result[idx]["f1_score"] for idx in kept_indices]
    llm_ground_truths = [index_to_result[idx]["ground_truth"] for idx in kept_indices]

    cleaned_data = cleaned_data.add_column("llm_prediction", llm_predictions)
    cleaned_data = cleaned_data.add_column("llm_f1_score", llm_f1_scores)
    cleaned_data = cleaned_data.add_column("original_answer", llm_ground_truths)

    cleaned_data.save_to_disk(output_path)
    logger.info(f"Cleaned dataset saved to {output_path}")

    # Save cleaning report alongside the dataset
    report_path = os.path.join(output_path, "cleaning_report.json")
    avg_f1 = sum(r["f1_score"] for r in cleaning_results) / max(len(cleaning_results), 1)
    report = {
        "total_samples": total,
        "kept_samples": kept,
        "removed_samples": removed,
        "keep_rate": kept / max(total, 1),
        "average_f1": avg_f1,
        "f1_threshold": cleaning_results[0].get("f1_score", 0.0) if not cleaning_results else None,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Cleaning report saved to {report_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean QA training data using a large LLM via API."
    )
    parser.add_argument(
        "--input_path", type=str, required=True,
        help="Path to the input dataset (same format as train_data.py expects).",
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Path to save the cleaned dataset.",
    )
    parser.add_argument(
        "--dataset_type", type=str, required=True,
        choices=["musique", "hotpotqa", "wikimqa", "longmagpie", "longalpaca"],
        help="Type of the dataset to clean.",
    )
    parser.add_argument(
        "--api_base", type=str,
        default=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
        help="Base URL for the OpenAI-compatible API.",
    )
    parser.add_argument(
        "--api_key", type=str,
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="API key for authentication.",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o",
        help="Model name to use for answering.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature for the LLM.",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=64,
        help="Max tokens for the LLM response.",
    )
    parser.add_argument(
        "--f1_threshold", type=float, default=0.5,
        help="Minimum F1 score to keep a sample. Samples below this are removed.",
    )
    parser.add_argument(
        "--max_concurrent", type=int, default=8,
        help="Max concurrent API requests (async mode).",
    )
    parser.add_argument(
        "--sync", action="store_true",
        help="Use synchronous requests instead of async.",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default=None,
        help="Path for JSONL checkpoint file to enable resumable processing.",
    )
    parser.add_argument(
        "--num_samples", type=int, default=None,
        help="Only process the first N samples (for debugging).",
    )
    return parser.parse_args()


def main():
    basicConfig(level=INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()

    logger.info(f"Loading dataset from {args.input_path} (type={args.dataset_type})")
    raw_data = load_raw_dataset(args.dataset_type, args.input_path)

    if args.num_samples is not None:
        raw_data = raw_data.select(range(min(args.num_samples, len(raw_data))))
    logger.info(f"Loaded {len(raw_data)} samples")

    api_config = APIConfig(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_concurrent=args.max_concurrent,
    )

    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            os.path.dirname(args.output_path),
            f"clean_checkpoint_{args.dataset_type}.jsonl",
        )

    if args.sync:
        results = clean_dataset_sync(
            raw_data, args.dataset_type, args.input_path,
            api_config, args.f1_threshold, checkpoint_path,
        )
    else:
        results = asyncio.run(clean_dataset_async(
            raw_data, args.dataset_type, args.input_path,
            api_config, args.f1_threshold, checkpoint_path,
        ))

    os.makedirs(args.output_path, exist_ok=True)
    save_cleaned_dataset(raw_data, results, args.output_path)
    logger.info("Done!")


if __name__ == "__main__":
    main()
