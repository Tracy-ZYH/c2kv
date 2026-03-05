import argparse
import json
import os
from typing import List, Dict, Any, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from tqdm import tqdm
import numpy as np

from mdocdataset import AbstractMDQADataset, load_mdoc_dataset
from models import get_model_class, blend_gist_key_values
from reuse_pipeline import tokenize_for_reuse, prefill_kv_cache
from expr_timer import DataRecorder, ExprTimer
from expr_gistmodel import cut_documents, MODEL_GENERATE_API_WARNING_STRING


def extract_and_blend_single(
    model,
    tokenizer,
    device: torch.device,
    example: Dict[str, Any],
    system_cache: DynamicCache,
    cut_length: Optional[int],
    record: Optional[DataRecorder] = None,
) -> Tuple[DynamicCache, int, int]:
    """
    For a single example, run extract and blend to produce a merged KV cache.

    Returns:
        merged_cache: DynamicCache with system + gist context KV
        precompute_length: position offset for query tokens (system_len + context_token_count)
        cache_seq_length: actual KV sequence length in the cache
    """
    system_length = system_cache.get_seq_length()

    documents = cut_documents(example['documents'], max_length=cut_length)
    context_inputs = tokenize_for_reuse(
        tokenizer, documents, keep_bos=False, role='user'
    ).to(device)

    context_token_count = context_inputs.attention_mask.sum().item()

    model.model.config._attn_implementation = "flex_attention"

    if record is not None:
        with record.record("extract"):
            outputs, gist_mask, pos_ids = model.model.generate_gist(**context_inputs)
    else:
        outputs, gist_mask, pos_ids = model.model.generate_gist(**context_inputs)

    pos_ids = pos_ids[:, -gist_mask.shape[1]:]

    if record is not None:
        with record.record("blend"):
            context_cache, _ = blend_gist_key_values(
                model.config, [outputs.past_key_values], [gist_mask], [pos_ids],
                model.model.rotary_emb, system_length
            )
    else:
        context_cache, _ = blend_gist_key_values(
            model.config, [outputs.past_key_values], [gist_mask], [pos_ids],
            model.model.rotary_emb, system_length
        )

    # Free intermediate tensors immediately
    del outputs, gist_mask, pos_ids, context_inputs
    torch.cuda.empty_cache()

    # Merge system cache into context cache
    for system_layer, context_layer in zip(system_cache.layers, context_cache.layers):
        context_layer.keys = torch.cat([system_layer.keys, context_layer.keys], dim=-2)
        context_layer.values = torch.cat([system_layer.values, context_layer.values], dim=-2)

    cache_seq_length = context_cache.get_seq_length()
    precompute_length = system_length + context_token_count

    return context_cache, precompute_length, cache_seq_length


def pad_kv_cache_to_length(
    cache: DynamicCache,
    target_length: int,
) -> None:
    """Pad a single-batch DynamicCache in-place so its seq dim equals target_length."""
    current_length = cache.get_seq_length()
    if current_length >= target_length:
        return
    pad_size = target_length - current_length
    for layer in cache.layers:
        # layer.keys shape: (batch=1, num_heads, seq_len, head_dim)
        key_pad = torch.zeros(
            (*layer.keys.shape[:-2], pad_size, layer.keys.shape[-1]),
            dtype=layer.keys.dtype, device=layer.keys.device,
        )
        value_pad = torch.zeros(
            (*layer.values.shape[:-2], pad_size, layer.values.shape[-1]),
            dtype=layer.values.dtype, device=layer.values.device,
        )
        layer.keys = torch.cat([layer.keys, key_pad], dim=-2)
        layer.values = torch.cat([layer.values, value_pad], dim=-2)


def batch_kv_caches(
    caches: List[DynamicCache],
    model_config,
) -> Tuple[DynamicCache, List[int]]:
    """
    Stack multiple single-sample DynamicCaches into one batched DynamicCache.
    All caches are padded to the max sequence length before stacking.

    Returns:
        batched_cache: DynamicCache with batch_size > 1
        original_cache_lengths: per-sample cache lengths before padding
    """
    original_lengths = [c.get_seq_length() for c in caches]
    max_length = max(original_lengths)

    # Pad all caches to max_length
    for cache in caches:
        pad_kv_cache_to_length(cache, max_length)

    num_layers = len(caches[0].layers)
    batched_kv = []
    for layer_idx in range(num_layers):
        batched_keys = torch.cat(
            [c.layers[layer_idx].keys for c in caches], dim=0
        )
        batched_values = torch.cat(
            [c.layers[layer_idx].values for c in caches], dim=0
        )
        batched_kv.append((batched_keys, batched_values))

    batched_cache = DynamicCache(batched_kv, config=model_config)
    return batched_cache, original_lengths


@torch.inference_mode()
def evaluate_model_on_dataset_batched(
    model_name: str,
    dataset: AbstractMDQADataset,
    batch_size: int = 4,
    max_examples: Optional[int] = None,
    output_file: Optional[str] = None,
    cut_length: Optional[int] = None,
    profile: bool = False,
) -> Dict[str, float]:
    """Evaluate a gist model on a MDQA dataset with batched decoding."""

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
        attn_implementation="flash_attention_2",
    )
    device = model.device

    num_examples = len(dataset) if max_examples is None else min(max_examples, len(dataset))
    max_new_tokens = dataset.max_new_tokens

    # Pre-compute system prompt cache
    dataset.system_prompt = "You are a helpful assistant."
    if dataset.system_prompt is None:
        sys_cache = None
    else:
        system_inputs = tokenize_for_reuse(
            tokenizer, [dataset.system_prompt], keep_bos=True, role='system'
        ).to(device)
        sys_cache = prefill_kv_cache(model, system_inputs)

    timer = ExprTimer("gist_batch", enable=profile)
    em_scores = []
    results = []

    # Iterate over dataset in batches
    for batch_start in tqdm(range(0, num_examples, batch_size), desc="Batches"):
        batch_end = min(batch_start + batch_size, num_examples)
        batch_examples = [dataset[i] for i in range(batch_start, batch_end)]
        current_batch_size = len(batch_examples)

        # ---- Phase 1: Extract & Blend (per-sample) ----
        per_sample_caches: List[DynamicCache] = []
        per_sample_precompute_lengths: List[int] = []
        per_sample_cache_lengths: List[int] = []
        records: List[DataRecorder] = []

        for sample_idx, example in enumerate(batch_examples):
            record = timer.record(example['qid'])
            records.append(record)

            system_cache = sys_cache
            if 'system_prompt' in example:
                system_inputs = tokenize_for_reuse(
                    tokenizer, [example['system_prompt']], keep_bos=True, role='system'
                ).to(device)
                system_cache = prefill_kv_cache(model, system_inputs)
            assert system_cache is not None, "System prompt has not been pre-computed"

            sample_cache, precompute_length, cache_seq_length = extract_and_blend_single(
                model, tokenizer, device, example, system_cache,
                cut_length=cut_length, record=record,
            )

            per_sample_caches.append(sample_cache)
            per_sample_precompute_lengths.append(precompute_length)
            per_sample_cache_lengths.append(cache_seq_length)

        # ---- Phase 2: Prepare batched inputs ----
        # Tokenize all queries
        query_input_ids_list = []
        query_lengths = []
        for example in batch_examples:
            qids = tokenize_for_reuse(
                tokenizer, [example['question']], keep_bos=False,
                role='user', add_generation_prompt=True,
            ).input_ids.to(device)
            query_input_ids_list.append(qids.squeeze(0))  # (query_len,)
            query_lengths.append(qids.shape[1])

        max_query_length = max(query_lengths)

        # Pad query input_ids to the same length (left-pad with pad_token_id)
        padded_query_ids = []
        for qids, qlen in zip(query_input_ids_list, query_lengths):
            pad_len = max_query_length - qlen
            if pad_len > 0:
                pad_tensor = torch.full(
                    (pad_len,), tokenizer.pad_token_id,
                    dtype=qids.dtype, device=device,
                )
                qids = torch.cat([pad_tensor, qids], dim=0)
            padded_query_ids.append(qids)

        # Build batched KV cache
        max_cache_length = max(per_sample_cache_lengths)
        batched_cache, original_cache_lengths = batch_kv_caches(
            per_sample_caches, model.config
        )

        # Free individual caches
        del per_sample_caches
        torch.cuda.empty_cache()

        # Build input_ids: [mock_cache_ids | padded_query]
        # mock_cache_ids fill the cache positions with 0 (won't be used for embedding)
        batch_input_ids = []
        batch_position_ids = []
        batch_attention_masks = []

        for sample_idx in range(current_batch_size):
            cache_len = max_cache_length
            precompute_len = per_sample_precompute_lengths[sample_idx]
            real_cache_len = original_cache_lengths[sample_idx]
            qlen = query_lengths[sample_idx]
            query_pad_len = max_query_length - qlen

            # Mock ids for cache positions
            mock_ids = torch.full(
                (cache_len,), 0, dtype=torch.long, device=device,
            )
            # Concatenate mock + padded query
            input_ids = torch.cat([mock_ids, padded_query_ids[sample_idx]], dim=0)
            batch_input_ids.append(input_ids)

            # Position ids: only the query tokens get real positions
            # Cache tokens don't need position ids (already embedded in KV)
            # For the padded query tokens, we use dummy positions
            # Real query positions: [precompute_len, precompute_len + qlen)
            query_position_ids = torch.arange(
                precompute_len, precompute_len + qlen,
                dtype=torch.long, device=device,
            )
            if query_pad_len > 0:
                # Left-padded query: pad positions with 0
                pad_positions = torch.zeros(
                    query_pad_len, dtype=torch.long, device=device,
                )
                query_position_ids = torch.cat([pad_positions, query_position_ids], dim=0)
            batch_position_ids.append(query_position_ids)

            # Attention mask: 1 for real cache + real query, 0 for padding
            attn_mask = torch.zeros(
                cache_len + max_query_length, dtype=torch.long, device=device,
            )
            # Real cache positions
            attn_mask[:real_cache_len] = 1
            # Real query positions (right-aligned due to left-padding)
            attn_mask[cache_len + query_pad_len:] = 1
            batch_attention_masks.append(attn_mask)

        batch_input_ids = torch.stack(batch_input_ids, dim=0)
        batch_position_ids = torch.stack(batch_position_ids, dim=0)
        batch_attention_masks = torch.stack(batch_attention_masks, dim=0)

        # ---- Phase 3: Batched generation ----
        model.model.config._attn_implementation = "flash_attention_2"

        # Use the first record for timing the generate phase (approximate)
        with records[0].record("generate"):
            generated_outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_masks,
                position_ids=batch_position_ids,
                past_key_values=batched_cache,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                use_gist=True,
            )

        # Free batched cache immediately
        del batched_cache, batch_attention_masks, batch_position_ids
        torch.cuda.empty_cache()

        # ---- Phase 4: Decode and evaluate per sample ----
        input_total_length = batch_input_ids.shape[1]
        for sample_idx in range(current_batch_size):
            example = batch_examples[sample_idx]
            generated_tokens = generated_outputs[sample_idx][input_total_length:]
            generated_tokens = generated_tokens[:max_new_tokens]
            pred = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            print(pred)

            answer = example['answer']
            em_score = dataset.metric(pred, answer)
            em_scores.append(em_score)

            result_entry = {
                'qid': example['qid'],
                'prediction': pred,
                'ground_truth': answer,
                'em_score': em_score,
            }
            if timer.enable:
                record = records[sample_idx]
                record.phases['generate'] = record.phases.get('generate', 0.0) / max(len(tokenizer.encode(pred)), 1)
                result_entry['timer'] = record.summary()
                del result_entry['prediction']
            results.append(result_entry)

        del generated_outputs, batch_input_ids
        torch.cuda.empty_cache()

    # Calculate overall metrics
    exact_match = sum(em_scores) / len(em_scores) if em_scores else 0

    # Save results
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                if result:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')

        summary = {
            'model': model_name,
            'dataset': dataset.__class__.__name__,
            'num_examples': len(results),
            'batch_size': batch_size,
            'exact_match': exact_match,
            **timer.statistics(),
        }

        summary_file = output_file.replace('.jsonl', '_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

    return {
        'exact_match': exact_match,
        'num_examples': len(results),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate gist models on MDQA datasets with batched decoding"
    )

    parser.add_argument("--model", type=str, required=True,
                        help="Model name or path")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Type of dataset")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Dataset path")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for decoding")
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
    parser.add_argument("--profile", action="store_true", default=False,
                        help="Profile model")

    args = parser.parse_args()

    if args.output_file and os.path.exists(args.output_file):
        print(f"Output file {args.output_file} already exists, skipping")
        return

    print(MODEL_GENERATE_API_WARNING_STRING)

    dataset = load_mdoc_dataset(
        args.dataset,
        args.dataset_path,
        only_supporting=args.only_supporting,
        enable_cot=args.cot,
    )

    print(f"Loaded {len(dataset)} examples from {args.dataset} dataset")
    print(f"Using batch_size={args.batch_size}")

    results = evaluate_model_on_dataset_batched(
        model_name=args.model,
        dataset=dataset,
        batch_size=args.batch_size,
        max_examples=args.max_examples,
        output_file=args.output_file,
        cut_length=args.cut_length,
        profile=args.profile,
    )

    print(f"\nEvaluation Results:")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of examples evaluated: {results['num_examples']}")
    print(f"Score: {results['exact_match']:.4f}")


if __name__ == "__main__":
    main()
