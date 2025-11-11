# python python/inference/expr_blockattention.py --model /home/admin/workspace/aop_lab/app_data/models/ldsjmdy--Tulu3-Block-FT --dataset musique --program_path /home/admin/workspace/aop_lab/app_source/duchuheng/Block-Attention/server/block_generate_server.py --output_file results/musique/llama3.1-8b_blockattn.json --port PORT
import os
import argparse
import json
import time
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import subprocess
import signal
import requests

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
    
    for i in tqdm(range(num_examples)):
        example = dataset[i]

        if 'system_prompt' in example:
            system_prompt = example['system_prompt']
        assert system_prompt is not None, "System prompt is not pre-computed"

        blocks = [system_prompt]
        blocks.extend(example['documents'])
        blocks.append(example['question'])

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
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Also save a summary
        summary = {
            'model': model_name,
            'dataset': dataset.__class__.__name__,
            'num_examples': len(results),
            'exact_match': exact_match,
        }
        
        summary_file = output_file.replace('.json', '_summary.json')
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
    wait_secs = 15
    print(f"Sleeping for {wait_secs} seconds to wait for model to start")
    time.sleep(wait_secs)

    # Load dataset
    dataset = load_mdoc_dataset(
        args.dataset, 
        args.dataset_path, 
        only_supporting=args.only_supporting,
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
