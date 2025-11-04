import argparse
import json
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm
import numpy as np

from mdocdataset import AbstractMDQADataset, load_mdoc_dataset
from memorag import MemoRAG


def evaluate_model_on_dataset(
    model_name: str,
    dataset: AbstractMDQADataset,
    max_examples: Optional[int] = None,
    output_file: Optional[str] = None,
) -> Dict[str, float]:
    """Evaluate a model on a MDQA dataset"""
    
    # Initialize evaluator
    evaluator = MemoRAG(
        mem_model_name_or_path="TommyChien/memorag-qwen2-7b-inst",
        ret_model_name_or_path="BAAI/bge-m3",
        gen_model_name_or_path=model_name,
        beacon_ratio=4,
    )
    
    # Calculate metrics
    em_scores = []
    results = []
    
    num_examples = len(dataset) if max_examples is None else min(max_examples, len(dataset))
    
    for i in tqdm(range(num_examples)):
        example = dataset[i]
        context = dataset.system_prompt + "".join(example['documents'])

        pred = evaluator(
            query=example["question"],
            context=context,
            task_type="memorag",
            max_new_tokens=dataset.max_new_tokens,
            reset_each_call=True,
            use_memory_answer=False,
        )

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
            json.dump(results, f, indent=2)
        
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

    # Load dataset
    dataset = load_mdoc_dataset(
        args.dataset, 
        args.dataset_path, 
        only_supporting=args.only_supporting,
    )
    
    print(f"Loaded {len(dataset)} examples from {args.dataset} dataset")
    
    # Evaluate model
    results = evaluate_model_on_dataset(
        model_name=args.model,
        dataset=dataset,
        max_examples=args.max_examples,
        output_file=args.output_file,
    )
    
    print(f"\nEvaluation Results:")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Number of examples evaluated: {results['num_examples']}")
    print(f"Score: {results['exact_match']:.4f}")


if __name__ == "__main__":
    main()
