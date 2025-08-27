import torch
import numpy as np
from typing import Tuple, List

from reuse_pipeline import LLMInference
from mdocdataset import MusiqueDataset


musique = MusiqueDataset('../musique_ans_v1.0_dev.jsonl', True)
sample = musique[193]

full_query = musique.get_system_prompt() + "".join(sample['documents']) + sample['question']
print(full_query)
print(sample['answer'])

inference = LLMInference("meta-llama/Meta-Llama-3-8B-Instruct")

sys_cache, sys_ids = inference.get_prefill_kv_cache(musique.get_system_prompt())
print(len(sys_cache), len(sys_cache[0]), len(sys_cache[0][0]), sys_cache[0][0][0].shape)

context_cache, context_ids = inference.get_prefill_kv_cache(sample['documents'], False)
print(len(context_cache), len(context_cache[0]))
for cache in context_cache[0][0]:
    print(cache.shape, end=' ')
print()

text_reuse, kv_cache_reuse = inference.decode_with_past_kv(
    system_prompt_ids=sys_ids,
    system_prompt_kv=sys_cache,
    precomputed_ids=context_ids,
    precomputed_kv=context_cache,
    query_text=sample['question'],
    max_new_tokens=1,
    return_kv=True,
)

print(text_reuse)
print(len(kv_cache_reuse), len(kv_cache_reuse[0]), kv_cache_reuse[0][0].shape)

_, kv_cache_recomp = inference.decode_with_past_kv(
    query_text=full_query,
    max_new_tokens=1,
    return_kv=True,
)

print(len(kv_cache_recomp), len(kv_cache_recomp[0]), kv_cache_recomp[0][0].shape)
print(context_cache[0][0][0][0, :10, 0])
print(kv_cache_reuse[0][0][0][0, :40, 0])
print(kv_cache_recomp[0][0][0][0, :40, 0])

def save_kv_cache(
    kv_cache: Tuple[Tuple[torch.Tensor, ...], ...],
    filepath: str
):
    layer_num = len(kv_cache)
    batch_size, head_num, seq_len, head_size = kv_cache[0][0].shape
    cache_np = np.zeros((layer_num, 2, batch_size, head_num, seq_len, head_size))
    for layer_i, layer_kv in enumerate(kv_cache):
        for kv_i, korv in enumerate(layer_kv):
            cache_np[layer_i][kv_i] = korv.cpu().numpy()
    np.save(filepath, cache_np)

save_kv_cache(kv_cache_recomp, './saved_kv/recompute.npy')
save_kv_cache(kv_cache_reuse, './saved_kv/reuse.npy')