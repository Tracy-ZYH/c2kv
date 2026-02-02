import argparse
import json
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm

from mdocdataset import AbstractMDQADataset, load_mdoc_dataset
from reuse_pipeline import LLMInference, BatchedKVInstance, gen_recompute_mask
from expr_timer import DataRecorder, ExprTimer


@torch.inference_mode()
def evaluate_model_on_dataset(
    dataset: AbstractMDQADataset,
    args: argparse.Namespace,
) -> Dict[str, float]:
    """Evaluate a model on a MDQA dataset"""

    model_name = args.model
    max_examples = args.max_examples
    output_file = args.output_file

    
    # Initialize evaluator
    evaluator = LLMInference(model_name)
    
    # Calculate metrics
    em_scores = []
    results = []
    
    num_examples = len(dataset) if max_examples is None else min(max_examples, len(dataset))

    timer = ExprTimer(f"reuse-{str(args.recompute_type)}", args.profile)

    if dataset.system_prompt is None:
        sys_cache = None
    else:
        sys_cache = evaluator.get_prefill_kv_cache([dataset.system_prompt], True, role='system')
    
    for i in tqdm(range(num_examples)):
        example = dataset[i]
        record = timer.record(example['qid'])

        system_cache = sys_cache
        if 'system_prompt' in example:
            system_cache = evaluator.get_prefill_kv_cache([example['system_prompt']], True, role='system')
        assert system_cache is not None, "System prompt is not pre-computed"

        if not args.profile:
            def append_cache(batch: List[str], cache: BatchedKVInstance | None) -> BatchedKVInstance:
                if not batch:
                    return cache
                new_cache = evaluator.get_prefill_kv_cache(batch, False, 'user', args.compress)
                return cache.stack(new_cache) if cache is not None else new_cache
            context_cache = None
            batched_documents = []
            for doc in example['documents']:
                if len(doc) > 4096:
                    context_cache = append_cache(batched_documents, context_cache)
                    context_cache = append_cache([doc], context_cache)
                    batched_documents = []
                else:
                    batched_documents.append(doc)
            context_cache = append_cache(batched_documents, context_cache)
        else:
            with record.record("extract"):
                context_cache = evaluator.get_prefill_kv_cache(example['documents'], False, 'user', args.compress)

        with record.record("blend"):
            if args.recompute_type is not None:
                system_cache = evaluator.selective_recompute(
                    system_cache, context_cache, args.recompute_type,
                    discard_kv='system_prompt' in example
                )
                context_cache = None

        with record.record("generate"):
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
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                if result:  # Only write non-empty results
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        # Also save a summary
        summary = {
            'model': model_name,
            'dataset': dataset.__class__.__name__,
            'num_examples': len(results),
            'exact_match': exact_match,
            **timer.statistics(),
        }
        
        summary_file = output_file.replace('.jsonl', '_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
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
    parser.add_argument("--cot", action="store_true", default=False,
                       help="Use cot prompt")
    parser.add_argument("--compress", type=str, default=None,
                       help="KV Cache compression type (e.g. \"snapkv\")")
    parser.add_argument("--profile", action="store_true", default=True,
                       help="Profile model")
    
    args = parser.parse_args()

    # Load dataset
    dataset = load_mdoc_dataset(
        args.dataset, 
        args.dataset_path, 
        only_supporting=args.only_supporting,
        enable_cot=args.cot,
    )
    
    print(f"Loaded {len(dataset)} examples from {args.dataset} dataset")
    
    # Evaluate model
    results = evaluate_model_on_dataset(dataset=dataset, args=args)
    
    print(f"\nEvaluation Results:")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Number of examples evaluated: {results['num_examples']}")
    print(f"Score: {results['exact_match']:.4f}")


if __name__ == "__main__":
    main()
