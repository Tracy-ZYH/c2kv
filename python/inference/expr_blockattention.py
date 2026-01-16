import os
import argparse
import json
import time
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import subprocess
import signal
import requests
from transformers import AutoTokenizer

from mdocdataset import AbstractMDQADataset, load_mdoc_dataset


def evaluate_model_on_dataset(
    port: int,
    model_name: str,
    dataset: AbstractMDQADataset,
    max_examples: Optional[int] = None,
    output_file: Optional[str] = None,
) -> Dict[str, float]:
    """Evaluate a model on a MDQA dataset"""
    # Calculate metrics
    em_scores = []
    results = []
    
    num_examples = len(dataset) if max_examples is None else min(max_examples, len(dataset))
    system_prompt = dataset.system_prompt
    url = f"http://localhost:{port}/generate"

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    
    for i in tqdm(range(num_examples)):
        example = dataset[i]

        if 'system_prompt' in example:
            system_prompt = example['system_prompt']
        assert system_prompt is not None, "System prompt is not pre-computed"

        blocks = [tokenizer.apply_chat_template([{"role": "system", "content": system_prompt}], tokenize=False, add_generation_prompt=False)]
        blocks.extend([
            tokenizer.apply_chat_template([{"role": "user", "content": doc}], tokenize=False, add_generation_prompt=False)
            for doc in example['documents']
        ])
        blocks.append(
            tokenizer.apply_chat_template([{"role": "user", "content": example['question']}], tokenize=False, add_generation_prompt=True)
        )

        r = requests.post(
            url=url,
            data=json.dumps({'blocks': blocks}),
            headers={"Content-Type": "application/json"}
        )

        pred = r.json()["generated"]

        em_score = dataset.metric(pred, example['answer'])
        em_scores.append(em_score)
        
        results.append({
            'qid': example['qid'],
            'prediction': pred,
            'ground_truth': example['answer'],
            'em_score': em_score
        })
    
    # Calculate overall metrics
    exact_match = sum(em_scores) / len(em_scores) if em_scores else 0
    
    # Save results if output file is specified
    if output_file:
        with open(output_file, 'w') as f:
            for result in results:
                if result:  # 只写入非空结果
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        # Also save a summary
        summary = {
            'model': model_name,
            'dataset': dataset.__class__.__name__,
            'num_examples': len(results),
            'exact_match': exact_match,
        }
        
        summary_file = output_file.replace('.jsonl', '_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    return {
        'exact_match': exact_match,
        'num_examples': len(results)
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate autoregressive models on MDQA datasets")
    
    # Required arguments
    parser.add_argument("--model", type=str, required=True, 
                       help="Model name or path (e.g., 'Qwen/Qwen2-7B-Instruct', 'mistralai/Mistral-7B-v0.1')")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Type of dataset")
    parser.add_argument("--port", type=int, default=None,
                       help="Port to use for the model")
    
    # Optional arguments
    parser.add_argument("--dataset_path", type=str, default=None,
                       help="Dataset path")
    parser.add_argument("--max_examples", type=int, default=None,
                       help="Maximum number of examples to evaluate")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output file to save results")
    parser.add_argument("--only_supporting", action="store_true",
                       help="For Musique dataset, use only supporting paragraphs")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda, cpu, mps)")
    parser.add_argument("--cot", action="store_true", default=False,
                       help="Use cot prompt")

    args = parser.parse_args()

    program_path = os.path.join(os.path.dirname(__file__), "block_generate_server.py")
    model_process_args = [
        "python", program_path,
        "--model", args.model,
        "--port", str(args.port)
    ]
    print("Starting model process:", ' '.join(model_process_args))
    model_process = subprocess.Popen(
        model_process_args, 
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    wait_secs = 120
    print(f"Sleeping for {wait_secs} seconds to wait for model to start")
    time.sleep(wait_secs)

    # Load dataset
    dataset = load_mdoc_dataset(
        args.dataset, 
        args.dataset_path, 
        only_supporting=args.only_supporting,
        enable_cot=args.cot,
    )
    
    print(f"Loaded {len(dataset)} examples from {args.dataset} dataset")
    
    # Evaluate model
    try:
        results = evaluate_model_on_dataset(
            port=args.port,
            model_name=args.model,
            dataset=dataset,
            max_examples=args.max_examples,
            output_file=args.output_file,
        )
    except Exception as e:
        model_process.send_signal(signal.SIGINT)
        raise e
    else:
        model_process.send_signal(signal.SIGINT)
    
    print(f"\nEvaluation Results:")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Number of examples evaluated: {results['num_examples']}")
    print(f"Score: {results['exact_match']:.4f}")


if __name__ == "__main__":
    main()
