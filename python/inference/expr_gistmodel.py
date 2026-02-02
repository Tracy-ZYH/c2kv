import argparse
import json
import os
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers.cache_utils import DynamicCache
from tqdm import tqdm
import numpy as np

from mdocdataset import AbstractMDQADataset, load_mdoc_dataset
from models import get_model_class, blend_gist_key_values
from reuse_pipeline import tokenize_for_reuse, prefill_kv_cache
from expr_timer import DataRecorder, ExprTimer

MODEL_GENERATE_API_WARNING_STRING = """==== PLEASE READ ====
With transformers==4.57.1 (which is required by this project), model.generate() API is buggy:
It is not compatible with custom position_ids, and it will cause incorrect results.
See https://github.com/huggingface/transformers/issues/36510 for how to fix it.
==== PLEASE READ ====
"""

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


@torch.inference_mode()
def evaluate_model_on_dataset(
    model_name: str,
    dataset: AbstractMDQADataset,
    max_examples: Optional[int] = None,
    output_file: Optional[str] = None,
    cut_length: Optional[int] = None,
    profile: bool = False,
) -> Dict[str, float]:
    """Evaluate a model on a MDQA dataset"""
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    _, model_class = get_model_class(model_name, "qkv")
    model = model_class.from_pretrained(
        model_name, 
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa"
    )
    device = model.device
    
    # Calculate metrics
    em_scores = []
    results = []
    
    num_examples = len(dataset) if max_examples is None else min(max_examples, len(dataset))
    max_new_tokens = dataset.max_new_tokens

    dataset.system_prompt = "You are a helpful assistant."
    if dataset.system_prompt is None:
        sys_cache = None
    else:
        system_inputs = tokenize_for_reuse(tokenizer, [dataset.system_prompt], keep_bos=True, role='system').to(device)
        sys_cache = prefill_kv_cache(model, system_inputs)

    timer = ExprTimer("gist", enable=profile)
    
    for i in tqdm(range(num_examples)):
        example = dataset[i]
        record = timer.record(example['qid'])

        # Pre-compute system prompt
        system_cache = sys_cache
        if 'system_prompt' in example:
            system_inputs = tokenize_for_reuse(tokenizer, [example['system_prompt']], keep_bos=True, role='system').to(device)
            system_cache = prefill_kv_cache(model, system_inputs)
        assert system_cache is not None, "System prompt has not been pre-computed"
        system_length = system_cache.get_seq_length()

        # Pre-compute context
        documents = cut_documents(example['documents'], max_length=cut_length)

        context_inputs = tokenize_for_reuse(tokenizer, documents, keep_bos=False, role='user').to(device)
        model.model.config._attn_implementation = "sdpa"
        with record.record("extract"):
            outputs, gist_mask, pos_ids = model.model.generate_gist(**context_inputs)
        pos_ids = pos_ids[:, -gist_mask.shape[1]:]
        with record.record("blend"):
            context_cache, _ = blend_gist_key_values(
                model.config, [outputs.past_key_values], [gist_mask], [pos_ids],
                model.model.rotary_emb, system_length
            )
        context_length = context_inputs.attention_mask.sum().item()
        precompute_length = pos_ids.max().item() + 1
        assert precompute_length == system_length + context_length, \
            f"Precompute position id mismatch: {precompute_length} != {system_length} + {context_length}"

        # Concatenate system prompt and context
        for system_layer, context_layer in zip(system_cache.layers, context_cache.layers):
            context_layer.keys = torch.cat([system_layer.keys, context_layer.keys], dim=-2)
            context_layer.values = torch.cat([system_layer.values, context_layer.values], dim=-2)
        cache_length = context_cache.get_seq_length()
        del system_cache

        input_ids = tokenize_for_reuse(
            tokenizer, [example['question']], keep_bos=False, role='user', add_generation_prompt=True
        ).input_ids.to(device)
        query_length = input_ids.shape[1]
        original_length = query_length + precompute_length
        position_ids = torch.arange(precompute_length, original_length, dtype=torch.long, device=device)
        mock_gist_ids = torch.full((1, cache_length), 0, dtype=torch.long, device=device)
        input_ids = torch.cat([mock_gist_ids, input_ids], dim=1)
        attention_mask = torch.ones_like(input_ids)

        # Generate text
        model.model.config._attn_implementation = "flash_attention_2"
        with record.record("generate"):
            generated_outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids.unsqueeze(0),
                past_key_values=context_cache,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                use_gist=True,
            )
        del context_cache
        
        # Decode generated text (skip query part)
        generated_tokens = generated_outputs[0][input_ids.shape[1]:]
        generated_tokens = generated_tokens[:max_new_tokens] # force limited output
        pred = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(pred)

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
        # create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

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
    parser.add_argument("--cot", action="store_true", default=False,
                       help="Use cot prompt")
    parser.add_argument("--cut_length", type=int, default=None,
                       help="Cut documents to specified length")
    parser.add_argument("--profile", action="store_true", default=True,
                       help="Profile model")
    
    args = parser.parse_args()

    print(MODEL_GENERATE_API_WARNING_STRING)

    # Load dataset
    dataset = load_mdoc_dataset(
        args.dataset, 
        args.dataset_path, 
        only_supporting=args.only_supporting,
        enable_cot=args.cot,
    )
    
    print(f"Loaded {len(dataset)} examples from {args.dataset} dataset")
    
    # Evaluate model
    results = evaluate_model_on_dataset(
        model_name=args.model,
        dataset=dataset,
        max_examples=args.max_examples,
        output_file=args.output_file,
        cut_length=args.cut_length,
        profile=args.profile,
    )
    
    print(f"\nEvaluation Results:")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Number of examples evaluated: {results['num_examples']}")
    print(f"Score: {results['exact_match']:.4f}")


if __name__ == "__main__":
    main()
