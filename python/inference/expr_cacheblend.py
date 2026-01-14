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
    blend_special_str: str = " # # ",
    enable_sparse: bool = False,
):
    # LMCache-related environment variables

    # Blending related config# Enable blending in LMCache
    os.environ["LMCACHE_ENABLE_BLENDING"] = "True"

    # Separator string between different chunks
    os.environ["LMCACHE_BLEND_SPECIAL_STR"] = " # # "

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

    # Set the maximum size of the local CPU size to 128GB
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "128"


@contextlib.contextmanager
def build_llm_with_lmcache(lmcache_connector: str, model: str):
    ktc = KVTransferConfig(
        kv_connector=lmcache_connector,
        kv_role="kv_both",
    )

    llm_args = EngineArgs(
        model=model,
        kv_transfer_config=ktc,
        max_model_len=32648,
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

    dataset = load_mdoc_dataset(args.dataset, enable_cot=args.cot)

    scores = []
    results = []
    num_examples = len(dataset)

    with build_llm_with_lmcache(lmcache_connector, model) as llm:
        # Define the shared prompt and specific prompts
        warmup_prompt = tokenizer.encode("Nice to meet you" * 500)
        tokenizer_has_bos = (warmup_prompt[0] == tokenizer.bos_token_id)
        print(f"Tokenizer has begin_of_sentence special token: {tokenizer_has_bos}.")

        # Prepare system prompt for chat template
        if dataset.system_prompt is None:
            sys_prompt = None
        else:
            # Use chat template for system prompt
            sys_messages = [{"role": "system", "content": dataset.system_prompt}]
            sys_prompt = tokenizer.apply_chat_template(sys_messages, tokenize=False)
            sys_prompt = tokenizer.encode(sys_prompt)
        
        blend_special_str = tokenizer.encode(os.getenv("LMCACHE_BLEND_SPECIAL_STR"))
        if tokenizer_has_bos:
            blend_special_str = blend_special_str[1:]

        sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=dataset.max_new_tokens)

        print_output(llm, warmup_prompt, sampling_params, "warmup")

        for i in tqdm(range(num_examples), file=sys.stdout):
            example = dataset[i]

            # Build chat messages
            messages = []
            
            # Add system prompt if exists
            if 'system_prompt' in example:
                messages.append({"role": "system", "content": example['system_prompt']})
            elif sys_prompt is not None:
                # If example doesn't have system prompt but dataset has one, add it
                messages.append({"role": "system", "content": dataset.system_prompt})

            # Add each document as a separate user message
            for doc in example['documents']:
                try:
                    doc_content = doc
                except TypeError as e:
                    doc_content = doc.encode("utf-8", errors="ignore").decode("utf-8")
                messages.append({"role": "user", "content": doc_content})

            # Add the question as a separate user message
            messages.append({"role": "user", "content": example['question']})

            # Apply chat template to get the full prompt
            try:
                example_input = tokenizer.apply_chat_template(messages, tokenize=False)
            except Exception as e:
                # Handle any encoding issues
                processed_messages = []
                for msg in messages:
                    processed_content = msg['content'].encode("utf-8", errors="ignore").decode("utf-8")
                    processed_messages.append({"role": msg['role'], "content": processed_content})
                example_input = tokenizer.apply_chat_template(processed_messages, tokenize=False)
            
            example_input = tokenizer.encode(example_input)

            _ = llm.generate(
                prompts={"prompt_token_ids": example_input}, sampling_params=sampling_params
            ) # warmup
            output = llm.generate(
                prompts={"prompt_token_ids": example_input}, sampling_params=sampling_params
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
