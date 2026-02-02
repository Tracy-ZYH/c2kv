import os
import argparse
import json
import torch
import math
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaConfig, LlamaForCausalLM

from rope_reposition import rotate_k_cache_rope
from compress_kv import compress_kv, QueryStorage
from expr_timer import DataRecorder, ExprTimer

from transformers import (
    AutoTokenizer, PreTrainedTokenizer, AutoModelForCausalLM, GenerationConfig, AutoConfig
)

from mdocdataset import AbstractMDQADataset, load_mdoc_dataset


def merge_and_rotary_past_key_values(pkvs: List[DynamicCache], block_lengths: List[int], rope_theta: float) -> DynamicCache:
    for b_idx in range(1, len(pkvs)):
        prefix_length = sum(block_lengths[:b_idx])
        for layer in pkvs[b_idx].layers:
            layer.keys = rotate_k_cache_rope(layer.keys[0], prefix_length, rope_theta, "llama3").unsqueeze(0)
    cache = pkvs[0]
    for l_idx in range(0, len(cache)):
        cache.layers[l_idx].keys = torch.cat(
            tensors=[cache.layers[l_idx].keys] + [pkvs[b_idx].layers[l_idx].keys for b_idx in range(1, len(pkvs))],
            dim=-2
        )
        cache.layers[l_idx].values = torch.cat(
            tensors=[cache.layers[l_idx].values] + [pkvs[b_idx].layers[l_idx].values for b_idx in range(1, len(pkvs))],
            dim=-2
        )
    return cache


@torch.no_grad()
def build_block_past_key_values(
        blocks: List[str], instruction: str, tokenizer: PreTrainedTokenizer, model: LlamaForCausalLM,
        num_local_attention_blocks: int, compression_method: Optional[str] = None,
) -> tuple:
    if len(blocks) > num_local_attention_blocks:
        instruction = "".join(blocks[num_local_attention_blocks:]) + instruction
        blocks = blocks[:num_local_attention_blocks]

    if num_local_attention_blocks == 0:
        instruction = "".join(blocks) + instruction
        blocks = []

    # print(f"Prompt | num local attention blocks: {num_local_attention_blocks}\n")
    # print(json.dumps({
    #     "blocks": blocks,
    #     "instruction_ans_response": instruction,
    # }, ensure_ascii=False, indent=4))

    caches: List[DynamicCache] = []
    block_lengths = []
    for b_idx, block in enumerate(blocks):
        block_input_ids = torch.tensor(
            data=[tokenizer.encode(block, add_special_tokens=False)],
            dtype=torch.int64,
            device=model.device
        )
        block_lengths.append(block_input_ids.size(-1))

        enable_compress = compression_method is not None
        with QueryStorage(model, enable_compress) as query_storage:
            past_key_values: DynamicCache = model(
                input_ids=block_input_ids, use_cache=True, past_key_values=DynamicCache(), return_dict=True
            ).past_key_values
            queries = query_storage.get_all_queries()
        if enable_compress:
            compressed_kv = []
            capacity = math.ceil(past_key_values.get_seq_length() / 4.0)
            for query, (keys, values) in zip(queries, past_key_values):
                keys, values, _ = compress_kv(compression_method, capacity, query[0], keys[0], values[0])
                compressed_kv.append((keys.unsqueeze(0), values.unsqueeze(0)))
            past_key_values = DynamicCache(compressed_kv)
        caches.append(past_key_values)

    response_input_ids = torch.tensor(
        data=[tokenizer.encode(instruction, add_special_tokens=False)],
        dtype=torch.int64,
        device=model.device
    )

    cache_length = sum(cache.get_seq_length() for cache in caches)
    input_ids = response_input_ids.new_zeros((1, 1)).expand((-1, cache_length))
    input_ids = torch.cat(tensors=[input_ids, response_input_ids], dim=-1)
    return caches, input_ids, block_lengths


@torch.inference_mode()
def block_generate(
        blocks: List[str], instruction: str, generation_config: GenerationConfig, model: LlamaForCausalLM,
        emb: LlamaRotaryEmbedding, tokenizer: PreTrainedTokenizer, num_local_attention_blocks: int, record: DataRecorder,
        compression_method: Optional[str] = None) -> str:
    with record.record("extract"):
        past_key_values, input_ids, block_lengths = build_block_past_key_values(
            blocks=blocks, instruction=instruction, tokenizer=tokenizer, model=model,
            num_local_attention_blocks=num_local_attention_blocks, compression_method=compression_method,
        )
    with record.record("blend"):
        if past_key_values is not None:
            past_key_values = merge_and_rotary_past_key_values(pkvs=past_key_values, block_lengths=block_lengths, rope_theta=model.config.rope_theta)
    input_length = input_ids.size(-1)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    with record.record("generate"):
        outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            generation_config=generation_config, past_key_values=past_key_values,
            use_cache=True,
        )
    return tokenizer.decode(token_ids=outputs[0][input_length:].tolist(), skip_special_tokens=True)


def evaluate_model_on_dataset(
    model_name: str,
    dataset: AbstractMDQADataset,
    generation_config: GenerationConfig,
    model: LlamaForCausalLM,
    emb: LlamaRotaryEmbedding,
    tokenizer: PreTrainedTokenizer,
    timer: ExprTimer,
    max_examples: Optional[int] = None,
    output_file: Optional[str] = None,
    compression_method: Optional[str] = None,
) -> Dict[str, float]:
    """Evaluate a model on a MDQA dataset"""
    em_scores = []
    results = []
    
    num_examples = len(dataset) if max_examples is None else min(max_examples, len(dataset))
    system_prompt = dataset.system_prompt

    for i in tqdm(range(num_examples)):
        example = dataset[i]
        record = timer.record(example['qid'])

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

        pred = block_generate(
            blocks=blocks[:-1],
            instruction=blocks[-1],
            generation_config=generation_config,
            model=model,
            emb=emb,
            tokenizer=tokenizer,
            num_local_attention_blocks=10000,
            record=record,
            compression_method=compression_method,
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
            for result in results:
                if result:
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
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    return {
        'exact_match': exact_match,
        'num_examples': len(results)
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate autoregressive models on MDQA datasets")
    
    parser.add_argument("--model", type=str, required=True, 
                       help="Model name or path")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Type of dataset")
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
    parser.add_argument("--compress", type=str, default=None,
                       help="Compress method")
    parser.add_argument("--profile", action="store_true", default=True,
                       help="Profile model")

    args = parser.parse_args()

    # Load model and tokenizer
    print(f"Loading model from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model,
        use_fast=False, local_files_only=True
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2",
        local_files_only=True,
    )
    model.eval()
    
    config: LlamaConfig = AutoConfig.from_pretrained(pretrained_model_name_or_path=args.model, local_files_only=True)
    emb: LlamaRotaryEmbedding = LlamaRotaryEmbedding(config=config).to(device=model.device, dtype=torch.float32)
    emb.eval()

    # Load dataset
    dataset = load_mdoc_dataset(
        args.dataset, 
        args.dataset_path, 
        only_supporting=args.only_supporting,
        enable_cot=args.cot,
    )

    generation_config = GenerationConfig(
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=dataset.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    print(f"Loaded {len(dataset)} examples from {args.dataset} dataset")
    
    # Evaluate model
    results = evaluate_model_on_dataset(
        model_name=args.model,
        dataset=dataset,
        generation_config=generation_config,
        model=model,
        emb=emb,
        tokenizer=tokenizer,
        timer=ExprTimer("block-attn", args.profile),
        max_examples=args.max_examples,
        output_file=args.output_file,
        compression_method=args.compress,
    )
    
    print(f"\nEvaluation Results:")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Number of examples evaluated: {results['num_examples']}")
    print(f"Score: {results['exact_match']:.4f}")


if __name__ == "__main__":
    main()
