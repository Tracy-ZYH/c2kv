import torch
import math
import random
from dataclasses import dataclass
from typing import Tuple, Optional, Callable, List, Union
from transformers import PretrainedConfig
from transformers.cache_utils import DynamicCache
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

GIST_GRADIENT_CHECKPOINTING = False


@dataclass
class GistModelOutputWithPast(CausalLMOutputWithPast):
    reconstruct_loss: Optional[torch.Tensor] = None


@dataclass
class GistConfigMixin:
    gist_type: str = "interleave-4"
    gist_param: str = "qkv"
    gist_extra_embed_num: int = 1
    gist_token_id: int | None = None
    gist_residual_type: str = "none"
    gist_overlap: int = 0


def rotate_half(x) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1) -> torch.Tensor:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed

def _build_interleave_mask_vectorized(
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor,
    ratio: int,
    padding_check_idx: int,
    padding_side: str,
    gist_residual_type: str = "none",
    gist_overlap: int = 0,
) -> Tuple[torch.BoolTensor, torch.BoolTensor, torch.LongTensor]:
    device = input_ids.device
    batch_size, max_seqlen = input_ids.shape
    max_gist_num = math.ceil(max_seqlen / ratio)
    total_len = max_seqlen + max_gist_num

    # --- per-sample sequence lengths (vectorized) ---
    original_seqlens = attention_mask.sum(dim=1)  # (batch_size,)
    seqlens = original_seqlens.clone()

    if gist_residual_type in ("mean", "embed-mean"):
        residual = seqlens % ratio
        needs_pad = (residual != 0) & (seqlens > 0)
        seqlens = torch.where(needs_pad, torch.clamp(seqlens + ratio - residual, max=max_seqlen), seqlens)

    # padlen: left-padding offset per sample
    if padding_check_idx == 0:  # right-padded
        padlens = torch.zeros(batch_size, dtype=torch.long, device=device)
    else:  # left-padded
        padlens = max_seqlen - seqlens

    gist_nums = ((seqlens + ratio - 1) // ratio).long()  # ceil division, (batch_size,)
    if padding_check_idx == 0:
        gist_pads = torch.zeros(batch_size, dtype=torch.long, device=device)
    else:
        gist_pads = max_gist_num - gist_nums

    # --- gist_mask: (batch_size, max_gist_num) ---
    gist_idx = torch.arange(max_gist_num, device=device).unsqueeze(0)  # (1, max_gist_num)
    gist_mask = (gist_idx >= gist_pads.unsqueeze(1)) & (gist_idx < (gist_pads + gist_nums).unsqueeze(1))

    # --- token-token causal mask (tril) ---
    # Build a shared tril matrix and mask per-sample valid region
    row_idx = torch.arange(max_seqlen, device=device).unsqueeze(1)  # (max_seqlen, 1)
    col_idx = torch.arange(max_seqlen, device=device).unsqueeze(0)  # (1, max_seqlen)
    causal_base = row_idx >= col_idx  # (max_seqlen, max_seqlen) lower-triangular

    # Per-sample valid token range: [padlen, padlen + seqlen)
    token_pos = torch.arange(max_seqlen, device=device).unsqueeze(0)  # (1, max_seqlen)
    valid_row = (token_pos >= padlens.unsqueeze(1)) & (token_pos < (padlens + seqlens).unsqueeze(1))  # (B, max_seqlen)
    valid_region = valid_row.unsqueeze(2) & valid_row.unsqueeze(1)  # (B, max_seqlen, max_seqlen)
    token_token_mask = causal_base.unsqueeze(0) & valid_region  # (B, max_seqlen, max_seqlen)

    # --- gist-related masks ---
    # j_idx: local gist index (0-based) for each position in max_gist_num
    # padded_j = j + gist_pad, so local j = gist_idx - gist_pad
    local_j = gist_idx - gist_pads.unsqueeze(1)  # (B, max_gist_num), local chunk index

    # chunk begin/end for each gist token
    chunk_begin = local_j * ratio  # (B, max_gist_num)
    chunk_end = torch.min((local_j + 1) * ratio, original_seqlens.unsqueeze(1))  # (B, max_gist_num)

    # gist_position_ids: end - 1 for valid gist tokens, 0 otherwise
    gist_position_ids = torch.where(gist_mask, (chunk_end - 1).long(), torch.zeros_like(local_j))

    # sink_end per sample: min(seqlen, ratio)
    sink_ends = torch.clamp(seqlens, max=ratio)  # (B,)

    # --- gist-to-token attention mask: (B, max_gist_num, max_seqlen) ---
    # For each gist token (row in gist dim), it attends to:
    #   1) sink region: token positions in [padlen, padlen + sink_end)
    #   2) chunk region: token positions in [padlen + chunk_begin, padlen + chunk_end)
    # token_pos: (1, 1, max_seqlen), padlens: (B, 1, 1)
    token_pos_3d = token_pos.unsqueeze(1)  # (1, 1, max_seqlen)
    padlens_3d = padlens.unsqueeze(1).unsqueeze(2)  # (B, 1, 1)

    # sink mask: gist j attends to [padlen, padlen + sink_end) — same for all j
    sink_mask = (token_pos_3d >= padlens_3d) & (token_pos_3d < (padlens + sink_ends).unsqueeze(1).unsqueeze(2))
    # (B, 1, max_seqlen) -> broadcast to (B, max_gist_num, max_seqlen)

    # chunk mask: gist j attends to [padlen + overlap_begin_j, padlen + chunk_end_j)
    overlap_begin = torch.clamp(chunk_begin - gist_overlap, min=0)
    chunk_begin_abs = padlens.unsqueeze(1) + overlap_begin
    chunk_end_abs = padlens.unsqueeze(1) + chunk_end  # (B, max_gist_num)
    chunk_mask = (token_pos_3d >= chunk_begin_abs.unsqueeze(2)) & (token_pos_3d < chunk_end_abs.unsqueeze(2))
    # (B, max_gist_num, max_seqlen)

    gist_to_token_mask = (sink_mask | chunk_mask) & gist_mask.unsqueeze(2)  # mask out invalid gist rows

    # --- gist-to-gist causal mask: (B, max_gist_num, max_gist_num) ---
    # gist j attends to gist [gist_pad, gist_pad + j + 1), i.e., all previous valid gists
    gist_col_idx = torch.arange(max_gist_num, device=device).unsqueeze(0).unsqueeze(0)  # (1, 1, max_gist_num)
    gist_row_idx = gist_idx.unsqueeze(2)  # (1, max_gist_num, 1)
    gist_gist_causal = (gist_col_idx >= gist_pads.unsqueeze(1).unsqueeze(2)) & (gist_col_idx <= gist_row_idx)
    gist_gist_mask = gist_gist_causal & gist_mask.unsqueeze(2) & gist_mask.unsqueeze(1)

    # --- assemble full attention mask: (B, total_len, total_len) ---
    new_attn_mask = torch.zeros((batch_size, total_len, total_len), dtype=torch.bool, device=device)
    # token-token block (top-left)
    new_attn_mask[:, :max_seqlen, :max_seqlen] = token_token_mask
    # gist-to-token block (bottom-left)
    new_attn_mask[:, max_seqlen:, :max_seqlen] = gist_to_token_mask
    # gist-to-gist block (bottom-right)
    new_attn_mask[:, max_seqlen:, max_seqlen:] = gist_gist_mask

    new_attn_mask = new_attn_mask.unsqueeze(1)  # (B, 1, total_len, total_len)

    # --- position_ids ---
    position_ids = torch.arange(max_seqlen, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, max_seqlen)
    position_ids = torch.cat([position_ids, gist_position_ids], dim=1)

    return new_attn_mask, gist_mask, position_ids


def _build_pattern_mask_vectorized(
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor,
    pattern: torch.BoolTensor,
    padding_check_idx: int,
) -> Tuple[torch.BoolTensor, torch.BoolTensor, torch.LongTensor]:
    """Build attention mask for pattern-based gist insertion (vectorized).

    Each True in `pattern` means a gist token is inserted after that position.
    Attention rules (interleave-style):
      - token-to-token: standard causal (lower-triangular)
      - gist-to-token: each gist sees the "window" of original tokens between
        the previous gist position (exclusive) and the current gist position (inclusive)
      - gist-to-gist: causal (each gist sees itself and all previous gists)
      - token-to-gist: none (original tokens do not attend to gist tokens)
    """
    device = input_ids.device
    batch_size, max_seqlen = input_ids.shape

    # --- per-sample valid region ---
    original_seqlens = attention_mask.sum(dim=1)  # (B,)
    if padding_check_idx == 0:  # right-padded
        padlens = torch.zeros(batch_size, dtype=torch.long, device=device)
    else:  # left-padded
        padlens = max_seqlen - original_seqlens

    # --- determine max_gist_num across the batch ---
    # mask out pattern positions that fall outside valid tokens
    token_pos = torch.arange(max_seqlen, device=device).unsqueeze(0)  # (1, S)
    valid_token = (token_pos >= padlens.unsqueeze(1)) & (token_pos < (padlens + original_seqlens).unsqueeze(1))
    pattern = pattern & valid_token  # (B, S)

    gist_nums = pattern.sum(dim=1)  # (B,)
    max_gist_num = gist_nums.max().item()
    if max_gist_num == 0:
        # no gist tokens needed — return trivial masks
        total_len = max_seqlen
        causal = torch.tril(torch.ones(max_seqlen, max_seqlen, dtype=torch.bool, device=device))
        valid_row = valid_token  # (B, S)
        valid_region = valid_row.unsqueeze(2) & valid_row.unsqueeze(1)
        token_mask = causal.unsqueeze(0) & valid_region
        token_mask = token_mask.unsqueeze(1)  # (B, 1, S, S)
        gist_mask = torch.zeros((batch_size, 0), dtype=torch.bool, device=device)
        position_ids = torch.arange(max_seqlen, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        return token_mask, gist_mask, position_ids

    total_len = max_seqlen + max_gist_num

    # --- build gist_mask and per-gist original-token positions ---
    # For each sample, scatter the True positions into a dense (B, max_gist_num) layout.
    # gist_token_pos[b, j] = the original-sequence index that gist j corresponds to.
    # We use left-aligned packing (right-padded gist dim) for right-padded input,
    # and right-aligned packing (left-padded gist dim) for left-padded input.

    # Compute prefix-sum of pattern to get local gist indices (0-based within each sample)
    gist_local_idx = pattern.long().cumsum(dim=1) - 1  # (B, S), -1 where pattern=False
    # For left-padded: shift gist indices so they are right-aligned
    if padding_check_idx != 0:
        gist_pads = max_gist_num - gist_nums  # (B,)
        gist_local_idx = gist_local_idx + gist_pads.unsqueeze(1)
    else:
        gist_pads = torch.zeros(batch_size, dtype=torch.long, device=device)

    # Build gist_mask: (B, max_gist_num)
    gist_idx_range = torch.arange(max_gist_num, device=device).unsqueeze(0)  # (1, G)
    gist_mask = (gist_idx_range >= gist_pads.unsqueeze(1)) & (gist_idx_range < (gist_pads + gist_nums).unsqueeze(1))

    # Build gist_token_pos: (B, max_gist_num) — the original token position each gist corresponds to
    # Scatter pattern positions into dense gist layout
    gist_token_pos = torch.zeros((batch_size, max_gist_num), dtype=torch.long, device=device)
    # For each (b, s) where pattern[b, s] = True, write s into gist_token_pos[b, gist_local_idx[b, s]]
    pattern_positions = token_pos.expand(batch_size, -1)  # (B, S)
    # Use scatter — only write where pattern is True
    scatter_idx = gist_local_idx.clone()
    scatter_idx[~pattern] = 0  # avoid out-of-bounds; these won't be used
    gist_token_pos.scatter_(1, scatter_idx, pattern_positions)
    # Zero out invalid gist positions
    gist_token_pos = gist_token_pos * gist_mask.long()

    # --- gist_position_ids: same as the original token position ---
    gist_position_ids = gist_token_pos.clone()

    # --- token-token causal mask ---
    row_idx = torch.arange(max_seqlen, device=device).unsqueeze(1)  # (S, 1)
    col_idx = torch.arange(max_seqlen, device=device).unsqueeze(0)  # (1, S)
    causal_base = row_idx >= col_idx  # (S, S) lower-triangular

    valid_row = valid_token  # (B, S)
    valid_region = valid_row.unsqueeze(2) & valid_row.unsqueeze(1)  # (B, S, S)
    token_token_mask = causal_base.unsqueeze(0) & valid_region  # (B, S, S)

    # --- gist-to-token mask: (B, max_gist_num, max_seqlen) ---
    # Each gist j sees original tokens in the window (prev_gist_pos, current_gist_pos].
    # prev_gist_pos for j=first_gist is the padlen - 1 (i.e., window starts at padlen).
    # We compute window_begin[b, j] = gist_token_pos[b, j-1] + 1 for j > first,
    #                                 = padlen[b]               for j = first valid gist.

    # Shift gist_token_pos to get previous gist positions
    # prev_pos[b, j] = gist_token_pos[b, j-1] if j > gist_pad, else padlen[b] - 1
    prev_pos = torch.zeros_like(gist_token_pos)
    prev_pos[:, 1:] = gist_token_pos[:, :-1]
    # For the first valid gist in each sample, set prev_pos to padlen - 1
    # so that window_begin = padlen - 1 + 1 = padlen
    first_gist_col = gist_pads  # (B,) — index of first valid gist
    # Use scatter to set prev_pos at the first valid gist position
    first_gist_col_expanded = first_gist_col.unsqueeze(1)  # (B, 1)
    sentinel_val = (padlens - 1).unsqueeze(1)  # (B, 1) — so window_begin = padlen
    prev_pos.scatter_(1, first_gist_col_expanded, sentinel_val)

    window_begin = prev_pos + 1  # (B, G) — inclusive start of window
    window_end = gist_token_pos + 1  # (B, G) — exclusive end of window

    # token_pos_3d: (1, 1, S)
    token_pos_3d = token_pos.unsqueeze(1)  # (1, 1, S)
    gist_to_token_mask = (
        (token_pos_3d >= window_begin.unsqueeze(2)) &
        (token_pos_3d < window_end.unsqueeze(2)) &
        gist_mask.unsqueeze(2)
    )  # (B, G, S)

    # --- gist-to-gist causal mask: (B, max_gist_num, max_gist_num) ---
    gist_col_range = gist_idx_range.unsqueeze(1)  # (1, 1, G)
    gist_row_range = gist_idx_range.unsqueeze(2)  # (1, G, 1)
    gist_gist_causal = gist_col_range <= gist_row_range  # (1, G, G)
    gist_gist_mask = gist_gist_causal & gist_mask.unsqueeze(2) & gist_mask.unsqueeze(1)  # (B, G, G)

    # --- assemble full attention mask: (B, total_len, total_len) ---
    new_attn_mask = torch.zeros((batch_size, total_len, total_len), dtype=torch.bool, device=device)
    # token-token block (top-left)
    new_attn_mask[:, :max_seqlen, :max_seqlen] = token_token_mask
    # gist-to-token block (bottom-left)
    new_attn_mask[:, max_seqlen:, :max_seqlen] = gist_to_token_mask
    # gist-to-gist block (bottom-right)
    new_attn_mask[:, max_seqlen:, max_seqlen:] = gist_gist_mask

    new_attn_mask = new_attn_mask.unsqueeze(1)  # (B, 1, total_len, total_len)

    # --- position_ids ---
    position_ids = torch.arange(max_seqlen, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, max_seqlen)
    position_ids = torch.cat([position_ids, gist_position_ids], dim=1)

    return new_attn_mask, gist_mask, position_ids


def get_prepare_gist_input_func(config: GistConfigMixin, padding_side: str = "right") -> Callable:
    gist_type: str = config.gist_type
    gist_residual_type: str = config.gist_residual_type
    assert gist_type, "gist_type must be specified"
    padding_check_idx = 0 if padding_side == "right" else -1
    gist_overlap = getattr(config, 'gist_overlap', 0)
    if gist_type.startswith("interleave-"):
        ratio = int(gist_type.split("-")[1])
        def _prepare_gist_input_interleave(
            input_ids: torch.LongTensor, attention_mask: torch.Tensor, **kwargs
        ) -> Tuple[torch.BoolTensor, torch.BoolTensor, torch.LongTensor]:
            for mask in attention_mask:
                if mask.any():
                    assert mask[padding_check_idx].all(), f"tokenizer is not {config.padding_side}-padded"
            return _build_interleave_mask_vectorized(
                input_ids, attention_mask, ratio, padding_check_idx, padding_side, gist_residual_type, gist_overlap,
            )
        return _prepare_gist_input_interleave
    elif gist_type.startswith('anchor-'):
        ratio = int(gist_type.split("-")[1])
        def _prepare_gist_input_anchor(
            input_ids: torch.LongTensor, attention_mask: torch.Tensor, **kwargs
        ) -> Tuple[torch.BoolTensor, torch.BoolTensor, torch.LongTensor]:
            for mask in attention_mask:
                if mask.any(): # only check non-empty sequences
                    assert mask[padding_check_idx].all(), f"tokenizer is not {config.padding_side}-padded"
            batch_size, max_seqlen = input_ids.shape
            max_gist_num = math.ceil(max_seqlen / ratio)
            new_attn_mask = torch.zeros( # (batch_size, query_len, kv_length)
                (batch_size, max_seqlen + max_gist_num, max_seqlen + max_gist_num), 
                dtype=torch.bool, device=input_ids.device
            )
            position_ids = torch.arange(max_seqlen, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, max_seqlen)
            gist_position_ids = torch.zeros((batch_size, max_gist_num), dtype=torch.long, device=input_ids.device)
            gist_mask = torch.zeros((batch_size, max_gist_num), dtype=torch.bool, device=input_ids.device)
            for i, seqlen in enumerate(attention_mask.sum(dim=1).tolist()):
                if seqlen == 0:
                    continue
                original_seqlen = seqlen
                if gist_residual_type in ("mean", "embed-mean"): # need to pad input_ids to multiple of ratio
                    residual_padlen = seqlen % ratio
                    if residual_padlen != 0:
                        seqlen = min(max_seqlen, seqlen + ratio - residual_padlen)
                padlen = 0 if padding_check_idx == 0 else max_seqlen - seqlen
                # new_attn_mask[i, padlen:seqlen + padlen, padlen:seqlen + padlen] = torch.tril(
                #     torch.ones(seqlen, seqlen, dtype=torch.bool, device=input_ids.device)
                # )
                gist_num = math.ceil(seqlen / ratio)
                gist_pad = 0 if padding_check_idx == 0 else max_gist_num - gist_num
                gist_mask[i, gist_pad:gist_pad + gist_num] = 1
                for j in range(gist_num):
                    # attention sink at beginning of chunk
                    sink_end = min(seqlen, ratio)
                    new_attn_mask[i, max_seqlen + j, padlen:sink_end + padlen] = 1
                    # attention sink at end of chunk
                    begin = j * ratio
                    end = min((j + 1) * ratio, original_seqlen)
                    group_length = end - begin
                    padded_j = j + gist_pad
                    gist_position_ids[i, padded_j] = end - 1
                    new_attn_mask[i, padlen + begin:padlen + end, padlen + begin:padlen + end] = torch.tril(
                        torch.ones((1, 1), dtype=torch.bool, device=input_ids.device).expand(group_length, group_length)
                    )
                    new_attn_mask[i, padlen + begin:padlen + end, max_seqlen + gist_pad:max_seqlen + padded_j] = 1
                    new_attn_mask[i, max_seqlen + padded_j, begin + padlen:end + padlen] = 1
                    new_attn_mask[i, max_seqlen + padded_j, max_seqlen + gist_pad:max_seqlen + padded_j + 1] = 1
            new_attn_mask = new_attn_mask.unsqueeze(1) # (batch_size, head_size, query_len, kv_length)
            position_ids = torch.cat([position_ids, gist_position_ids], dim=1)
            return new_attn_mask, gist_mask, position_ids
        return _prepare_gist_input_anchor
    elif gist_type == 'dynamic-interleave':
        def _prepare_gist_input_dynamic_interleave(
            input_ids: torch.LongTensor, attention_mask: torch.Tensor, **kwargs
        ) -> Tuple[torch.BoolTensor, torch.BoolTensor, torch.LongTensor]:
            for mask in attention_mask:
                if mask.any():
                    assert mask[padding_check_idx].all(), f"tokenizer is not {config.padding_side}-padded"
            return _build_interleave_mask_vectorized(
                input_ids, attention_mask, kwargs["ratio"], padding_check_idx, padding_side, "none", gist_overlap,
            )
        return _prepare_gist_input_dynamic_interleave
    elif gist_type == 'pattern':
        def _prepare_gist_input_pattern(
            input_ids: torch.LongTensor, attention_mask: torch.Tensor, **kwargs
        ) -> Tuple[torch.BoolTensor, torch.BoolTensor, torch.LongTensor]:
            assert "pattern" in kwargs, "pattern must be provided in kwargs for gist_type='pattern'"
            pattern = kwargs["pattern"]
            assert pattern.shape == input_ids.shape, \
                f"pattern shape {pattern.shape} must match input_ids shape {input_ids.shape}"
            for mask in attention_mask:
                if mask.any():
                    assert mask[padding_check_idx].all(), f"tokenizer is not {padding_side}-padded"
            return _build_pattern_mask_vectorized(
                input_ids, attention_mask, pattern, padding_check_idx, padding_side,
            )
        return _prepare_gist_input_pattern
    else:
        raise NotImplementedError(f"gist_type {gist_type} not implemented")

def get_apply_gist_residual_func(config: GistConfigMixin, layer_idx: int = 0) -> Callable:
    def _apply_gist_residual_interleave(
        tokens_tensor: torch.Tensor, gist_tensor: torch.Tensor, ratio: int = 4
    ) -> torch.Tensor:
        batch_size, seq_length, hidden_size = tokens_tensor.shape
        pad_length = seq_length % ratio
        nopad_length = seq_length - pad_length
        mean_tensor = tokens_tensor[:, :nopad_length].reshape(batch_size, -1, ratio, hidden_size).mean(dim=2)
        if pad_length != 0:
            pad_mean = tokens_tensor[:, nopad_length:].mean(dim=1, keepdim=True)
            mean_tensor = torch.cat([mean_tensor, pad_mean], dim=1)
        return mean_tensor + gist_tensor
    if config.gist_residual_type == "embed-mean":
        if layer_idx == 0:
            if config.gist_type.startswith("interleave-"):
                ratio = int(config.gist_type.split('-')[1])
                return lambda tokens_tensor, gist_tensor, **kwargs: _apply_gist_residual_interleave(tokens_tensor, gist_tensor, ratio=ratio)
            elif config.gist_type == "dynamic-interleave":
                return lambda tokens_tensor, gist_tensor, **kwargs: _apply_gist_residual_interleave(tokens_tensor, gist_tensor, ratio=kwargs["ratio"])
        return lambda tokens_tensor, gist_tensor, **kwargs: gist_tensor
    if config.gist_residual_type == "mean" and config.gist_type.startswith("interleave-"):
        ratio = int(config.gist_type.split('-')[1])
        return lambda tokens_tensor, gist_tensor, **kwargs: _apply_gist_residual_interleave(tokens_tensor, gist_tensor, ratio=ratio)
    elif config.gist_residual_type == "mean" and config.gist_type == "dynamic-interleave":
        return lambda tokens_tensor, gist_tensor, **kwargs: _apply_gist_residual_interleave(tokens_tensor, gist_tensor, ratio=kwargs["ratio"])
    return lambda tokens_tensor, gist_tensor, **kwargs: gist_tensor

def process_context_input_ids(
    model: PreTrainedModel,
    context_input_ids: torch.LongTensor,
    past_key_values: DynamicCache | None,
    attention_mask: torch.Tensor,
    position_ids: torch.LongTensor,
    reconstruct_kwargs: dict[str, ...] | None = None,
) -> Tuple[DynamicCache, torch.Tensor, Optional[torch.Tensor]]:
    assert position_ids is not None, "position_ids is required when context_input_ids is given"
    if past_key_values is None:
        past_key_values = DynamicCache(config=model.config)
    past_length = past_key_values.get_seq_length()
    # reshape context_input_ids and generate gist
    batch_size, chunk_num, seq_len = context_input_ids.shape
    context_input_ids = context_input_ids.reshape(batch_size * chunk_num, seq_len)
    input_ids = context_input_ids.clone()
    gist_attn_mask = input_ids != -100
    input_ids[~gist_attn_mask] = model.gist_token_id
    generate_gist_kwargs = {}
    if model.config.gist_type == "dynamic-interleave":
        generate_gist_kwargs["ratio"] = random.choice([2, 4, 8])
    outputs, gist_mask, pos_ids = model.generate_gist(input_ids, gist_attn_mask, **generate_gist_kwargs)
    # do reconstruction if reconstruct_kwargs is given
    reconstruct_loss = None
    if reconstruct_kwargs is not None and model.training:
        reconstruct_loss = _get_reconstruction_loss(
            model=model, input_ids=input_ids, labels=context_input_ids,
            attention_mask=gist_attn_mask, gist_mask=gist_mask, position_ids=pos_ids,
            past_key_values=outputs.past_key_values, **reconstruct_kwargs
        )
    # prepare pos_ids and generate positional embeddings
    max_gist_len = gist_mask.shape[1]
    pos_ids = pos_ids.reshape(batch_size, chunk_num, max_gist_len)
    if gist_mask.all(): # context input_ids is full
        for j in range(chunk_num):
            pos_ids[:, j] += j * seq_len
    else:
        assert attention_mask is not None, "attention_mask is required when context_input_ids is given"
        gist_lens = gist_mask.reshape((batch_size, chunk_num, max_gist_len)).sum(dim=2)
        for i in range(batch_size):
            prefix_length = past_length
            for j in range(chunk_num):
                gist_len = gist_lens[i, j]
                original_len = pos_ids[i, j, gist_len-1].item() + 1
                pos_ids[i, j, :gist_len] += prefix_length
                prefix_length += original_len
    # print(f"{past_length=}, {pos_ids.amax(dim=2)=}, {position_ids=}")
    pos_ids = pos_ids.reshape(batch_size * chunk_num, max_gist_len)
    cos, sin = model.rotary_emb(outputs.last_hidden_state, pos_ids)
    # apply rotary pos emb to gist key/value and store in past_key_values
    head_num = outputs.past_key_values[0][0].shape[1]
    for layer_idx, (key, value) in enumerate(outputs.past_key_values):
        key = apply_rotary_pos_emb(key, cos, sin)
        key = key.reshape(batch_size, chunk_num, head_num, max_gist_len, -1).transpose(1, 2)
        value = value.reshape(batch_size, chunk_num, head_num, max_gist_len, -1).transpose(1, 2)
        past_key_values.update(
            key.reshape(batch_size, head_num, chunk_num * max_gist_len, -1), 
            value.reshape(batch_size, head_num, chunk_num * max_gist_len, -1), 
            layer_idx
        )
    gist_mask = gist_mask.reshape(batch_size, chunk_num * max_gist_len)
    past_mask = gist_mask.new_ones((batch_size, past_length))
    # concat gist_attn_mask to attention_mask
    if attention_mask is not None:
        attention_mask = torch.cat([past_mask, gist_mask, attention_mask], dim=1)
    return past_key_values, attention_mask, reconstruct_loss

def _get_reconstruction_loss(
    model: PreTrainedModel,
    lm_head: torch.nn.Linear, loss_function: Callable[..., torch.Tensor],
    labels: torch.LongTensor, input_ids: torch.LongTensor,
    position_ids: torch.LongTensor,
    gist_mask: torch.Tensor, attention_mask: torch.Tensor,
    past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    **kwargs
) -> torch.Tensor:
    assert model.gist_embed_tokens.num_embeddings == 2, "Make sure gist_embed_tokens.num_embeddings is 2"
    sampled_length = 128
    batch_size = input_ids.shape[0]
    input_ids = input_ids[:, :sampled_length - 1]
    labels = labels[:, :sampled_length - 1]
    attention_mask = torch.cat([gist_mask, gist_mask.new_ones(batch_size, 1), attention_mask[:, :sampled_length - 1]], dim=1)
    # prepare embeddings
    inputs_embeds = model.embed_tokens(input_ids)
    reconstruct_embeds = model.gist_embed_tokens(input_ids.new_ones(batch_size, 1))
    inputs_embeds = torch.cat([reconstruct_embeds, inputs_embeds], dim=1)
    # prepare position ids
    pos_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0).repeat(batch_size, 1)
    pos_ids += position_ids.max(dim=1, keepdim=True).values + 1
    # apply position embeddings to gist KVs
    cos, sin = model.rotary_emb(inputs_embeds, position_ids)
    past_key_values = [(apply_rotary_pos_emb(key, cos, sin), value) for (key, value) in past_key_values]
    # generate reconstruction
    reconstruct_outputs: BaseModelOutputWithPast = model(
        inputs_embeds=inputs_embeds,
        position_ids=pos_ids,
        attention_mask=attention_mask,
        past_key_values=DynamicCache(past_key_values, config=model.config),
        **kwargs,
    )
    reconstruct_logits = lm_head(reconstruct_outputs.last_hidden_state[:, 1:, :]) # remove the first token
    reconstruct_loss = loss_function(
        logits=reconstruct_logits, labels=labels, vocab_size=model.config.vocab_size, **kwargs
    )
    return reconstruct_loss

def _concat_gist_key_values(
    model_config: PretrainedConfig,
    gist_key_values: Tuple[Tuple[torch.Tensor, ...], ...],
    gist_mask: torch.Tensor,
    gist_position_ids: torch.Tensor,
    rotary_emb: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    prefix_length: int,
    pad_length: int,
):
    assert gist_mask.shape == gist_position_ids.shape, \
        f"gist_mask {gist_mask.shape} != gist_position_ids {gist_position_ids.shape}"
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
            gist_key_values[batch_i], gist_mask[batch_i].bool(), gist_position_ids[batch_i], 
            rotary_emb, prefix_length[batch_i], batch_length
        ))
        attention_mask[batch_i, :merged_gist_length[batch_i]] = 1
    for layer_i in range(num_layer):
        merged_gist_kv.append((
            torch.stack([kv[layer_i][0] for kv in key_values], dim=0), 
            torch.stack([kv[layer_i][1] for kv in key_values], dim=0)
        ))
    return DynamicCache(merged_gist_kv, config=model_config), attention_mask

def gen_gist_proj(attn_hidden_size: int, config: PretrainedConfig) -> torch.nn.Linear:
    bias = config.attention_bias if hasattr(config, "attention_bias") else True # Qwen2.5 has attention_bias
    proj = torch.nn.Linear(config.hidden_size, attn_hidden_size, bias=bias)
    proj.weight.data.zero_()
    proj._is_hf_initialized = True
    return proj

def init_gist_proj(model, missing_keys):
    if is_deepspeed_zero3_enabled():
        import deepspeed
        def init_proj_deepspeed(gist_proj, proj, gist_name):
            if gist_name not in model.config.gist_param.lower():
                return
            params = [gist_proj.weight, proj.weight]
            if proj.bias is not None:
                params.extend([gist_proj.bias, proj.bias])
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                if (gist_proj.weight.sum(-1) == 0).any() or (gist_proj.weight > 1e29).any():
                    gist_proj.weight.data.copy_(proj.weight.data)
                    if proj.bias is not None:
                        gist_proj.bias.data.copy_(proj.bias.data)
        init_proj_deepspeed(model.gist_q_proj, model.q_proj, 'q')
        init_proj_deepspeed(model.gist_k_proj, model.k_proj, 'k')
        init_proj_deepspeed(model.gist_v_proj, model.v_proj, 'v')
        return
    def init_proj(gist_proj, proj, gist_name):
        module_name = f'gist_{gist_name}_proj'
        if gist_name not in model.config.gist_param.lower():
            return
        if not any(module_name in missing_key for missing_key in missing_keys):
            return
        gist_proj.weight.data.copy_(proj.weight.data)
        if proj.bias is not None:
            gist_proj.bias.data.copy_(proj.bias.data)
    init_proj(model.gist_q_proj, model.q_proj, 'q')
    init_proj(model.gist_k_proj, model.k_proj, 'k')
    init_proj(model.gist_v_proj, model.v_proj, 'v')
    
def init_gist_embed(model, missing_keys):
    if is_deepspeed_zero3_enabled():
        import deepspeed
        params = [model.gist_embed_tokens.weight, model.embed_tokens.weight]
        with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
            # deepspeed will initialize the parameters to zero
            # NOTE: with Llama3.1, change the following line to `if True` in order to initialize the parameters
            # if (model.gist_embed_tokens.weight == 0).all():
            if True:
                if model.config.gist_residual_type in ("mean", "embed-mean"):
                    model.gist_embed_tokens.weight.data.zero_()
                else:
                    model.gist_embed_tokens.weight.data[:] = model.embed_tokens.weight.data[
                        model.gist_token_id: model.gist_token_id + 1
                    ]
        return
    if "model.gist_embed_tokens.weight" in missing_keys:
        model.gist_embed_tokens.weight.data[:] = model.embed_tokens.weight.data[
            model.gist_token_id: model.gist_token_id + 1
        ]
