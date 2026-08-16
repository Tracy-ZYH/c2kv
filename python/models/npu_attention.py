from __future__ import annotations

import os
from typing import Any

import torch


def _to_npu_attention_mask(attention_mask: torch.Tensor | None) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    if attention_mask.dtype == torch.bool:
        return attention_mask
    return attention_mask < 0


def _attention_output(output: Any, expected_shape: torch.Size) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if not isinstance(output, tuple) or not output:
        raise RuntimeError(f"Unexpected npu_fusion_attention output type: {type(output)!r}")
    # torch_npu versions commonly return the attention output as the first item.
    # Some Ascend docs describe a longer tuple with attention_out in slot 3.
    candidates = [item for item in output if isinstance(item, torch.Tensor) and item.dim() == 4]
    if not candidates:
        raise RuntimeError("npu_fusion_attention returned no 4-D attention output tensor.")
    for tensor in candidates:
        if tensor.shape == expected_shape:
            return tensor
    return candidates[0]


def _slice_attn_mask(
    mask: torch.Tensor | None,
    start: int,
    end: int,
    full_q_len: int,
) -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.dim() >= 2 and mask.shape[-2] == full_q_len:
        return mask[..., start:end, :].contiguous()
    return mask


def _run_npu_fa(
    torch_npu: Any,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    atten_mask: torch.Tensor | None,
    scaling: float,
    dropout: float,
) -> torch.Tensor:
    output = torch_npu.npu_fusion_attention(
        query_states,
        key_states,
        value_states,
        query_states.shape[1],
        input_layout="BNSD",
        atten_mask=atten_mask,
        scale=scaling,
        keep_prob=1.0 - dropout,
        sparse_mode=0,
    )
    return _attention_output(output, query_states.shape)


def npu_fusion_attention_forward(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    if not getattr(query, "is_npu", False):
        raise RuntimeError("npu_fusion_attention requires NPU tensors.")
    try:
        import torch_npu
    except ImportError as error:
        raise RuntimeError("torch_npu is required for npu_fusion_attention.") from error

    if kwargs.get("output_attentions", False):
        raise RuntimeError("npu_fusion_attention does not return attention weights; use eager for debug.")
    if getattr(module, "sliding_window", None) is not None:
        raise RuntimeError("npu_fusion_attention path does not support sliding-window attention yet.")

    query_states = query.contiguous()
    key_states = key.contiguous()
    value_states = value.contiguous()
    atten_mask = _to_npu_attention_mask(attention_mask)
    if atten_mask is not None:
        atten_mask = atten_mask.contiguous()

    q_chunk_size = int(os.environ.get("C2KV_NPU_ATTN_Q_CHUNK_SIZE", "512"))
    full_q_len = query_states.shape[2]

    if q_chunk_size > 0 and full_q_len > q_chunk_size:
        output_chunks = []
        for start in range(0, full_q_len, q_chunk_size):
            end = min(start + q_chunk_size, full_q_len)
            q_chunk = query_states[:, :, start:end, :].contiguous()
            mask_chunk = _slice_attn_mask(atten_mask, start, end, full_q_len)
            output_chunks.append(
                _run_npu_fa(
                    torch_npu,
                    q_chunk,
                    key_states,
                    value_states,
                    mask_chunk,
                    scaling,
                    dropout,
                )
            )
        attn_output = torch.cat(output_chunks, dim=2)
    else:
        attn_output = _run_npu_fa(
            torch_npu,
            query_states,
            key_states,
            value_states,
            atten_mask,
            scaling,
            dropout,
        )
    return attn_output.transpose(1, 2).contiguous(), None
