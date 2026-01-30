# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import asdict
import argparse
import contextlib
import os
import sys
import time
import json
from tqdm import tqdm
from typing import Tuple

# Third Party
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs

# First Party
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder


from mdocdataset import load_mdoc_dataset


def setup_environment_variables(
    use_disk: bool = False,
    blend_special_str: str = "# #",
    enable_sparse: bool = False,
):
    # LMCache-related environment variables
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"

    # Blending related config# Enable blending in LMCache
    os.environ["LMCACHE_ENABLE_BLENDING"] = "True"
    # Separator string between different chunks
    os.environ["LMCACHE_BLEND_SPECIAL_STR"] = blend_special_str
    # Layerwise must be turned on when blending is enabled
    os.environ["LMCACHE_USE_LAYERWISE"] = "True"
    # Determining which tokens to recompute at layer 1
    os.environ["LMCACHE_BLEND_CHECK_LAYERS"] = "1"
    # Ratio of tokens to recompute
    os.environ["LMCACHE_BLEND_RECOMPUTE_RATIOS"] = "0.15"

    # Optionally, we can use sparse attention to improve generation quality
    # by using more accurate attention mask
    if enable_sparse:
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
        os.environ["LMCACHE_EXTRA_CONFIG"] = '{"enable_sparse": true}'
    
    # Enable local CPU backend in LMCache
    os.environ["LMCACHE_LOCAL_CPU"] = "True"

    # Set the maximum size of the local CPU size to 32GB
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "32"


@contextlib.contextmanager
def build_llm_with_lmcache(lmcache_connector: str, model: str):
    ktc = KVTransferConfig(
        kv_connector=lmcache_connector,
        kv_role="kv_both",
    )

    llm_args = EngineArgs(
        model=model,
        kv_transfer_config=ktc,
        gpu_memory_utilization=0.8,
        enable_prefix_caching=False,
        enforce_eager=True,
    )

    llm = LLM(**asdict(llm_args))
    try:
        yield llm
    finally:
        # Clean up lmcache backend
        LMCacheEngineBuilder.destroy(ENGINE_NAME)


def print_output(
    llm: LLM,
    prompt: list[int],
    sampling_params: SamplingParams,
    req_str: str,
) -> str:
    start = time.time()
    outputs = llm.generate(
        prompts={"prompt_token_ids": prompt}, sampling_params=sampling_params
    )
    print("-" * 50)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
    print(f"Generation took {time.time() - start:.2f} seconds, {req_str} request done.")
    print("-" * 50)


def tokenize_example(
    example: dict, dataset, tokenizer: AutoTokenizer, blend_special_ids: list[int], tokenizer_has_bos: bool
) -> Tuple[list[int], list[list[int]], list[int]]:
    # Add system prompt if exists
    if 'system_prompt' in example:
        system_prompt = example['system_prompt']
    else:
        system_prompt = dataset.system_prompt
    system_input_ids = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}], tokenize=True, 
        add_generation_prompt=False, enable_thinking=False
    ) + blend_special_ids

    # Add documents and question with blend special string
    document_input_ids = []
    for doc in example['documents']:
        document_input_ids.append(tokenizer.apply_chat_template(
            [{"role": "user", "content": doc}], tokenize=True, 
            add_generation_prompt=False, enable_thinking=False
        ) + blend_special_ids)
        if tokenizer_has_bos:
            document_input_ids[-1] = document_input_ids[-1][1:]
    
    query_input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": example['question']}], tokenize=True, 
        add_generation_prompt=True, enable_thinking=False
    )
    if tokenizer_has_bos:
        query_input_ids = query_input_ids[1:]

    return system_input_ids, document_input_ids, query_input_ids


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--use-disk",
        action="store_true",
        help="Specify whether to use disk as backend (default: False)",
    )
    parser.add_argument(
        "-b",
        "--blend-special-str",
        default="# #",
        help="Specify the special separators to separate chunks (default: '# #')",
    )
    parser.add_argument(
        "--only_supporting", 
        action="store_true",
        help="For Musique dataset, use only supporting paragraphs"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
    )
    parser.add_argument(
        "--enable-sparse",
        action="store_true",
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
    )
    parser.add_argument(
        "--output_file", type=str, required=True,
    )
    parser.add_argument(
        "--cot", action="store_true", default=False, help="Use cot prompt"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    lmcache_connector = "LMCacheConnectorV1"
    model = args.model
    output_file = args.output_file

    setup_environment_variables(
        args.use_disk, args.blend_special_str, args.enable_sparse
    )

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)

    dataset = load_mdoc_dataset(args.dataset, only_supporting=args.only_supporting, enable_cot=args.cot)

    scores = []
    results = []
    num_examples = len(dataset)

    with build_llm_with_lmcache(lmcache_connector, model) as llm:
        # Define the shared prompt and specific prompts
        warmup_prompt = tokenizer.encode("Nice to meet you" * 500)
        tokenizer_has_bos = (warmup_prompt[0] == tokenizer.bos_token_id)
        print(f"Tokenizer has begin_of_sentence special token: {tokenizer_has_bos}.")

        blend_special_ids = tokenizer.encode(os.getenv("LMCACHE_BLEND_SPECIAL_STR"))
        if tokenizer_has_bos:
            blend_special_ids = blend_special_ids[1:]

        sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=dataset.max_new_tokens)

        print_output(llm, warmup_prompt, sampling_params, "warmup")
        warmup_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Summarize this document."}], tokenize=True, 
            add_generation_prompt=True, enable_thinking=False
        )

        for i in tqdm(range(num_examples), file=sys.stdout):
            example = dataset[i]
            system_ids, doc_ids_list, query_ids = tokenize_example(
                example, dataset, tokenizer, blend_special_ids, tokenizer_has_bos,
            )

            # Cache the documents separately
            cache_prompts = [{"prompt_token_ids": system_ids + doc_ids + warmup_prompt} for doc_ids in doc_ids_list]
            _ = llm.generate(
                prompts=cache_prompts, 
                sampling_params=SamplingParams(max_tokens=1),
            )

            # Generate the final output
            full_prompt = system_ids + [ids for doc_ids in doc_ids_list for ids in doc_ids] + query_ids
            output = llm.generate(
                prompts={"prompt_token_ids": full_prompt}, 
                sampling_params=sampling_params
            )
            pred = output[0].outputs[0].text

            score = dataset.metric(pred, example['answer'])
            scores.append(score)
            results.append({
                'qid': example['qid'],
                'prediction': pred,
                'ground_truth': example['answer'],
                'em_score': score,
            })
    
    avg_score = sum(scores) / len(scores) if scores else 0

    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            if result:  # Only write non-empty results
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # Also save a summary
    summary = {
        'model': model,
        'dataset': dataset.__class__.__name__,
        'num_examples': len(results),
        'exact_match': avg_score,
    }
    
    summary_file = output_file.replace('.jsonl', '_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nEvaluation Results:")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Number of examples evaluated: {len(results)}")
    print(f"Score: {avg_score:.4f}")


if __name__ == "__main__":
    main()
