import torch
import math
from typing import Tuple


def prepare_gist_input(
    gist_id: int,
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor,
    gist_type: str
) -> Tuple[torch.LongTensor, torch.Tensor, torch.BoolTensor, torch.LongTensor]:
    """
    Insert gist tokens to the input embeddings.
    Return the attention mask, gist mask and position ids.
    """
    if gist_type == "":
        return (
            attention_mask, torch.zeros_like(attention_mask, dtype=torch.bool),
            torch.arange(input_ids.shape[1], dtype=torch.long, device=input_ids.device).unsqueeze(0)
        )
    assert attention_mask[:, 0].all(), "attention_mask must be left-aligned"
    if gist_type.startswith("interleave-"):
        ratio = int(gist_type.split("-")[1])
        max_seqlen = input_ids.shape[1]
        max_gist_num = math.ceil(max_seqlen / ratio)
        new_attn_mask = torch.zeros( # (batch_size, query_len, kv_length)
            (input_ids.shape[0], max_seqlen + max_gist_num, max_seqlen + max_gist_num), 
            dtype=torch.bool, device=input_ids.device
        )
        new_attn_mask[:, :max_seqlen, :max_seqlen] = torch.tril(
            torch.ones((max_seqlen, max_seqlen), dtype=torch.bool, device=input_ids.device)
        )
        position_ids = torch.cat([
            torch.arange(max_seqlen, device=input_ids.device),
            torch.arange(max_gist_num, device=input_ids.device)
        ], dim=0).unsqueeze(0)
        gist_mask = torch.zeros((input_ids.shape[0], max_gist_num), dtype=torch.bool, device=input_ids.device)
        for i in range(input_ids.shape[0]):
            seqlen = attention_mask[i].sum().item()
            gist_num = math.ceil(seqlen / ratio)
            gist_mask[i, :gist_num] = 1
            for j in range(gist_num):
                begin = j * ratio
                end = min(begin + ratio, seqlen)
                new_attn_masl[i, max_seqlen + j, begin:end] = 1
                new_attn_mask[i, max_seqlen + j, max_seqlen:max_seqlen + j + 1] = 1
        new_attn_mask = new_attn_mask.unsqueeze(1) # (batch_size, head_size, query_len, kv_length)
        return new_attn_mask, gist_mask, position_ids
    else:
        raise NotImplementedError(f"gist_type {gist_type} not implemented")

def prepare_decode_with_gist():
    pass # TODO
