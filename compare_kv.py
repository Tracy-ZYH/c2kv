import torch
import numpy as np
from typing import Tuple, List
from itertools import batched

from reuse_pipeline import LLMInference, gen_recompute_mask
from mdocdataset import load_mdoc_dataset


def save_kv_cache(
    kv_cache: Tuple[Tuple[torch.Tensor, ...], ...],
    filepath: str
):
    layer_num = len(kv_cache)
    batch_size, head_num, seq_len, head_size = kv_cache[0][0].shape
    # cache_np = np.zeros((layer_num, 2, batch_size, head_num, seq_len, head_size))
    cache_np = np.zeros((layer_num, 2, batch_size, 1, seq_len, head_size))
    for layer_i, layer_kv in enumerate(kv_cache):
        for kv_i, korv in enumerate(layer_kv):
            # cache_np[layer_i][kv_i] = korv.cpu().float().numpy()
            cache_np[layer_i][kv_i] = korv[:, 0].view((batch_size, 1, seq_len, head_size)).cpu().float().numpy()
    np.save(filepath, cache_np)

# ============ 1. Load Dataset and Model ==============

musique = load_mdoc_dataset('musique', only_supporting=False)
# musique = load_mdoc_dataset('hotpotqa')
# musique = load_mdoc_dataset('samsum')
# musique = load_mdoc_dataset('multinews')
sample = musique[179]
    
# max_new_tokens = musique.max_new_tokens
max_new_tokens = 1

print(len(sample["documents"]))
# sample["documents"] = sample["documents"][:10]
# sample["documents"] = ["".join(batch_doc) for batch_doc in batched(sample["documents"], 3)]
full_query = musique.system_prompt + "".join(sample['documents']) + sample['question']

print("============ Request ============")
print(full_query)
print("Answer:", sample['answer'])
print("=================================")

inference = LLMInference("meta-llama/Meta-Llama-3.1-8B-Instruct")
query_input = inference.tokenizer(full_query).input_ids
np.save("./saved_kv/input_ids.npy", np.array(query_input))

# ============ 2. Prefill System Prompt and Documents ==============

sys_instance = inference.get_prefill_kv_cache(musique.system_prompt, True)
sys_ids, sys_cache = sys_instance.unpack()
print("System prompt KV Cache Shape")
print(len(sys_cache), len(sys_cache[0]), len(sys_cache[0][0]), sys_cache[0][0][0].shape)

context_instance = inference.get_prefill_kv_cache(sample['documents'], False)
context_ids, context_cache = context_instance.unpack()
print("Context KV Cache Shape")
print(len(context_cache), len(context_cache[0]))
for cache in context_cache[0][0]:
    print(cache.shape, end=' ')
print("\n===================\n")

# ============ 3. Full Reuse ==============

# inference._prefill_with_past_kv(
#     system_prompt_kv=sys_instance,
#     precomputed_kv=context_instance,
#     query_text=sample['question'],
#     max_new_tokens=max_new_tokens,
# )

output_text, kv_cache = inference.decode_with_past_kv(
    system_prompt_kv=sys_instance,
    precomputed_kv=context_instance,
    query_text=sample['question'],
    max_new_tokens=max_new_tokens,
    return_kv=True,
)

print("Reuse output:", output_text)
print(len(kv_cache), len(kv_cache[0]), kv_cache[0][0].shape)

save_kv_cache(kv_cache, './saved_kv/reuse.npy')
del kv_cache

# ============ 4. Full Recompute ==============

output_text, kv_cache = inference.decode_with_past_kv(
    query_text=full_query,
    max_new_tokens=max_new_tokens,
    return_kv=True,
)

print("\nRecompute output:", output_text)
print(len(kv_cache), len(kv_cache[0]), kv_cache[0][0].shape)

save_kv_cache(kv_cache, './saved_kv/recompute.npy')
del kv_cache

# ============ 5. Selective Recompute ==============

recompute_masks = gen_recompute_mask(
    inference.tokenizer,
    context_instance,
    "punc-4",
)

updated_cache = inference.selective_recompute(sys_instance, context_instance, recompute_masks)

output_text, kv_cache = inference.decode_with_past_kv(
    system_prompt_kv=updated_cache,
    query_text=sample['question'],
    max_new_tokens=max_new_tokens,
    return_kv=True,
)

print("\nSelective output:", output_text)
print(len(kv_cache), len(kv_cache[0]), kv_cache[0][0].shape)

save_kv_cache(kv_cache, './saved_kv/selective.npy')
del kv_cache