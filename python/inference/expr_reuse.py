import argparse
import json
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm
import numpy as np
from itertools import batched

from mdocdataset import AbstractMDQADataset, load_mdoc_dataset
from reuse_pipeline import LLMInference, BatchedKVInstance, gen_recompute_mask


@torch.inference_mode()
def evaluate_model_on_dataset(
    model_name: str,
    dataset: AbstractMDQADataset,
    max_examples: Optional[int] = None,
    output_file: Optional[str] = None,
    recompute_type: Optional[str] = None,
) -> Dict[str, float]:
    """Evaluate a model on a MDQA dataset"""
    
    # Initialize evaluator
    evaluator = LLMInference(model_name)
    
    # Calculate metrics
    em_scores = []
    results = []
    
    num_examples = len(dataset) if max_examples is None else min(max_examples, len(dataset))

    if dataset.system_prompt is None:
        sys_cache = None
    else:
        sys_cache = evaluator.get_prefill_kv_cache(dataset.system_prompt, True)
    
    for i in tqdm(range(num_examples)):
        example = dataset[i]

        system_cache = sys_cache
        if 'system_prompt' in example:
            system_cache = evaluator.get_prefill_kv_cache(example['system_prompt'], True)
        assert system_cache is not None, "System prompt is not pre-computed"
        context_cache = evaluator.get_prefill_kv_cache(example['documents'], False)

        if recompute_type is not None:
            system_cache = evaluator.selective_recompute(
                system_cache, context_cache, gen_recompute_mask(evaluator.tokenizer, context_cache, recompute_type),
                discard_kv='system_prompt' in example
            )
            context_cache = None

        pred = evaluator.decode_with_past_kv(
            system_prompt_kv=system_cache,
            precomputed_kv=context_cache,
            query_text=example['question'],
            max_new_tokens=dataset.max_new_tokens,
        )

        del context_cache
        del system_cache
        torch.cuda.empty_cache()

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
    parser.add_argument("--recompute_type", type=str, default=None,
                       help="Type of mask for selective recompute (e.g. \"leading-5\")")
    
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
        recompute_type=args.recompute_type,
    )
    
    print(f"\nEvaluation Results:")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Number of examples evaluated: {results['num_examples']}")
    print(f"Score: {results['exact_match']:.4f}")


if __name__ == "__main__":
    main()
