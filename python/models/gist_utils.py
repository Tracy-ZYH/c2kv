import torch
import math
import torch
from dataclasses import dataclass
from typing import Tuple, Optional, Callable, List, Union
from transformers import PretrainedConfig
from transformers.cache_utils import DynamicCache


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed

def prepare_gist_input(
    gist_id: int,
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor,
    gist_type: str
) -> Tuple[torch.BoolTensor, torch.BoolTensor, torch.LongTensor]:
    """
    Insert gist tokens to the input embeddings.
    Return the attention mask, gist mask and position ids.
    """
    if gist_type == "":
        return (
            attention_mask, None,
            torch.arange(input_ids.shape[1], dtype=torch.long, device=input_ids.device).unsqueeze(0)
        )
    assert attention_mask[:, 0].all(), "tokenizer must be right-padded"
    if gist_type.startswith("interleave-"):
        ratio = int(gist_type.split("-")[1])
        batch_size = input_ids.shape[0]
        max_seqlen = input_ids.shape[1]
        max_gist_num = math.ceil(max_seqlen / ratio)
        new_attn_mask = torch.zeros( # (batch_size, query_len, kv_length)
            (batch_size, max_seqlen + max_gist_num, max_seqlen + max_gist_num), 
            dtype=torch.bool, device=input_ids.device
        )
        position_ids = torch.arange(max_seqlen, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).repeat(batch_size, 1)
        gist_position_ids = torch.zeros((batch_size, max_gist_num), dtype=torch.long, device=input_ids.device)
        gist_mask = torch.zeros((batch_size, max_gist_num), dtype=torch.bool, device=input_ids.device)
        for i in range(batch_size):
            seqlen = attention_mask[i].sum().item()
            padlen = max_seqlen - seqlen
            new_attn_mask[i, padlen:max_seqlen, padlen:max_seqlen] = torch.tril(
                torch.ones(seqlen, seqlen, dtype=torch.bool, device=input_ids.device)
            )
            gist_num = math.ceil(seqlen / ratio)
            gist_mask[i, :gist_num] = 1
            for j in range(gist_num):
                begin = j * ratio
                end = min(begin + ratio, seqlen)
                gist_position_ids[i, j] = end - 1
                begin += padlen
                end += padlen
                new_attn_mask[i, max_seqlen + j, begin:end] = 1
                new_attn_mask[i, max_seqlen + j, max_seqlen:max_seqlen + j + 1] = 1
        new_attn_mask = new_attn_mask.unsqueeze(1) # (batch_size, head_size, query_len, kv_length)
        position_ids = torch.cat([position_ids, gist_position_ids], dim=1)
        return new_attn_mask, gist_mask, position_ids
    else:
        raise NotImplementedError(f"gist_type {gist_type} not implemented")

def _concat_gist_key_values(
    model_config: PretrainedConfig,
    gist_key_values: Tuple[Tuple[torch.Tensor, ...], ...],
    gist_mask: torch.Tensor,
    gist_position_ids: torch.Tensor,
    rotary_emb: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    prefix_length: int,
    pad_length: int,
):
    seq_lens = gist_mask.sum(dim=1).tolist()
    # first accumulate the positional embeddings
    for i, seq_len in enumerate(seq_lens):
        original_seq_len = gist_position_ids[i, seq_len-1].item() + 1
        gist_position_ids[i, :seq_len] += prefix_length
        prefix_length += original_seq_len
    cos, sin = rotary_emb(gist_key_values[0][0], gist_position_ids)
    pad_length = pad_length - sum(seq_lens)
    key_values = []
    def _pad(tensor, mask, pad_length):
        tensor = tensor.transpose(1, 2)[mask]
        if pad_length > 0:
            tensor = torch.cat([tensor, tensor.new_zeros(pad_length, *tensor.shape[1:])], dim=0)
        return tensor.transpose(0, 1) # (num_heads, seq_len, hidden_size)
    for key, values in gist_key_values:
        key = apply_rotary_pos_emb(key, cos, sin)
        key_values.append((_pad(key, gist_mask, pad_length), _pad(values, gist_mask, pad_length)))
    return tuple(key_values)

def blend_gist_key_values(
    model_config: PretrainedConfig,
    gist_key_values: List[Tuple[Tuple[torch.Tensor, ...], ...]],
    gist_mask: List[torch.Tensor],
    gist_position_ids: List[torch.Tensor],
    rotary_emb: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    prefix_length: Union[int, List[int]] = 0,
) -> Tuple[DynamicCache, torch.Tensor]:
    merged_gist_length = [mask.sum().item() for mask in gist_mask]
    batch_length = max(merged_gist_length)
    batch_size = len(gist_key_values)
    num_layer = len(gist_key_values[0])
    if isinstance(prefix_length, int):
        prefix_length = [prefix_length] * batch_size
    key_values = []
    merged_gist_kv = []
    attention_mask = torch.zeros((batch_size, batch_length), dtype=torch.bool, device=gist_mask[0].device)
    for batch_i in range(batch_size):
        key_values.append(_concat_gist_key_values(
            model_config, 
            gist_key_values[batch_i], gist_mask[batch_i], gist_position_ids[batch_i], 
            rotary_emb, prefix_length[batch_i], batch_length
        ))
        attention_mask[batch_i, :merged_gist_length[batch_i]] = 1
    for layer_i in range(num_layer):
        merged_gist_kv.append((
            torch.stack([kv[layer_i][0] for kv in key_values], dim=0), 
            torch.stack([kv[layer_i][1] for kv in key_values], dim=0)
        ))
    return DynamicCache(merged_gist_kv, config=model_config), attention_mask
