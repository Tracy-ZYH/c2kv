import argparse
import json
import string
import regex
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm
import datasets

from mdocdataset import AbstractMDQADataset, WikiMQADataset, MusiqueDataset, best_subspan_em
from reuse_pipeline import LLMInference


def evaluate_model_on_dataset(
    model_name: str,
    dataset: AbstractMDQADataset,
    max_examples: Optional[int] = None,
    output_file: Optional[str] = None
) -> Dict[str, float]:
    """Evaluate a model on a MDQA dataset"""
    
    # Initialize evaluator
    evaluator = LLMInference(model_name)
    
    # Prepare generation config
    generation_config = GenerationConfig(
        max_new_tokens=256,
        do_sample=False,
        temperature=0.1,
        top_p=0.9,
        pad_token_id=evaluator.tokenizer.pad_token_id
    )
    
    # Calculate metrics
    em_scores = []
    results = []
    
    num_examples = len(dataset) if max_examples is None else min(max_examples, len(dataset))

    sys_cache = evaluator.get_prefill_kv_cache(dataset.get_system_prompt())
    
    for i in tqdm(range(num_examples)):
        example = dataset[i]
        prompt = dataset.get_system_prompt() + "".join(example['documents']) + example['question']

        context_cache = evaluator.get_prefill_kv_cache(example['documents'], False)

        pred = evaluator.decode_with_past_kv(
            system_prompt_kv=sys_cache,
            precomputed_kv=context_cache,
            query_text=prompt,
            max_new_tokens=256
        )

        em_score = best_subspan_em(pred, example['answer'])
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
                       help="Dataset path")
    parser.add_argument("--dataset_type", type=str, required=True, choices=["wikimqa", "musique"],
                       help="Type of dataset")
    
    # Optional arguments
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
    if args.dataset_type == "wikimqa":
        dataset = WikiMQADataset(args.dataset)
    elif args.dataset_type == "musique":
        dataset = MusiqueDataset(args.dataset, args.only_supporting)
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")
    
    print(f"Loaded {len(dataset)} examples from {args.dataset_type} dataset")
    
    # Evaluate model
    results = evaluate_model_on_dataset(
        model_name=args.model,
        dataset=dataset,
        max_examples=args.max_examples,
        output_file=args.output_file
    )
    
    print(f"\nEvaluation Results:")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset_type}")
    print(f"Number of examples evaluated: {results['num_examples']}")
    print(f"Exact Match Score: {results['exact_match']:.4f}")


if __name__ == "__main__":
    main()