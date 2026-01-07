import torch
import numpy as np
from typing import Tuple, List
import argparse
import json
from tqdm import tqdm

from reuse_pipeline import LLMInference, gen_recompute_mask
from mdocdataset import load_mdoc_dataset


def calculate_layerwise_differences(kv_cache1, kv_cache2, selected_head_idx, max_new_tokens):
    """
    Calculate L2 norm differences between two KV caches for a specific attention head.
    
    Args:
        kv_cache1: First KV cache (DynamicCache object)
        kv_cache2: Second KV cache (DynamicCache object) 
        selected_head_idx: Index of the attention head to compare
    
    Returns:
        List of mean L2 differences per layer
    """
    key_differences = []
    value_differences = []
    
    # Assuming DynamicCache has keys and values attributes that are lists
    # Each element in the list corresponds to a layer
    for layer_idx in range(len(kv_cache1.layers)):
        # Get keys and values for the current layer
        keys1, values1 = kv_cache1[layer_idx]
        keys2, values2 = kv_cache2[layer_idx]

        seq_len = min(keys1.shape[2], keys2.shape[2]) - max_new_tokens
        
        # Extract specific attention head
        keys1_head = keys1[:, selected_head_idx, :seq_len, :]  # [batch_size, seq_len, head_dim]
        values1_head = values1[:, selected_head_idx, :seq_len, :]
        keys2_head = keys2[:, selected_head_idx, :seq_len, :]
        values2_head = values2[:, selected_head_idx, :seq_len, :]
        
        # Calculate L2 norm differences for keys and values
        key_diff = torch.norm(keys1_head - keys2_head, p=2, dim=-1)  # L2 norm over head_dim
        value_diff = torch.norm(values1_head - values2_head, p=2, dim=-1)

        key_diff = key_diff / torch.norm(keys1_head, p=2, dim=-1)
        value_diff = value_diff / torch.norm(values1_head, p=2, dim=-1)
        
        # Mean over sequence length and batch dimensions
        mean_key_diff = torch.mean(key_diff).item()
        mean_value_diff = torch.mean(value_diff).item()
        
        key_differences.append(mean_key_diff)
        value_differences.append(mean_value_diff)
    
    return key_differences, value_differences


def process_sample(inference, sample, musique, max_new_tokens, recompute_mask_method, selected_head_idx):
    """
    Process a single sample and return the layer-wise differences.
    
    Args:
        inference: LLMInference instance
        sample: Dataset sample
        musique: Dataset object
        max_new_tokens: Maximum new tokens to generate
        recompute_mask_method: Method for generating recompute masks
        selected_head_idx: Index of the attention head to analyze
    
    Returns:
        Dictionary with sample id and layer-wise differences
    """
    # Prepare full query
    full_query = musique.system_prompt + "".join(sample['documents']) + sample['question']
    
    # Get system prompt KV cache
    sys_instance = inference.get_prefill_kv_cache(musique.system_prompt, True)
    
    # Get document KV cache
    context_instance = inference.get_prefill_kv_cache(sample['documents'], False)
    
    # Full recompute
    _, recompute_kv_cache = inference.decode_with_past_kv(
        query_text=full_query,
        max_new_tokens=max_new_tokens,
        return_kv=True,
    )
    
    if recompute_mask_method != 'none':
        # Generate recompute masks
        recompute_masks = gen_recompute_mask(
            inference.tokenizer,
            context_instance,
            recompute_mask_method,
        )
        
        # Selective recompute
        updated_cache = inference.selective_recompute(sys_instance, context_instance, recompute_masks)
    
        _, selective_kv_cache = inference.decode_with_past_kv(
            system_prompt_kv=updated_cache,
            query_text=sample['question'],
            max_new_tokens=max_new_tokens,
            return_kv=True,
        )
    
    else:
        _, selective_kv_cache = inference.decode_with_past_kv(
            system_prompt_kv=sys_instance,
            precomputed_kv=context_instance,
            query_text=sample['question'],
            max_new_tokens=max_new_tokens,
            return_kv=True,
        )
    
    # Calculate layer-wise differences
    key_differences, value_differences = calculate_layerwise_differences(
        recompute_kv_cache, 
        selective_kv_cache, 
        selected_head_idx,
        max_new_tokens,
    )
    
    return {
        "sample_id": sample.get('qid', 'unknown'),
        "key_differences": key_differences,
        "value_differences": value_differences,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare KV cache differences between selective and full recompute")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., wikimqa, hotpotqa, samsum, multinews)")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., meta-llama/Meta-Llama-3.1-8B-Instruct)")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file path")
    parser.add_argument("--attention_head", type=int, required=True, help="Selected attention head index")
    parser.add_argument("--recompute_method", type=str, default="leading-8", help="Method for generating recompute masks")
    
    args = parser.parse_args()
    
    # Load dataset
    dataset = load_mdoc_dataset(args.dataset)
    max_new_tokens = dataset.max_new_tokens
    
    # Initialize inference model
    inference = LLMInference(args.model)
    
    results = []
    
    # Process first 100 samples
    for i in tqdm(range(min(100, len(dataset))), desc="Processing samples"):
        try:
            sample = dataset[i]
            result = process_sample(
                inference, 
                sample, 
                dataset, 
                max_new_tokens, 
                args.recompute_method, 
                args.attention_head
            )
            results.append(result)
        except Exception as e:
            print(f"Error processing sample {i}: {str(e)}")
            continue
    
    # Write results to JSONL file
    with open(args.output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')


if __name__ == "__main__":
    main()
