from __future__ import annotations

from typing import Any

import torch


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch,
        num_key_value_heads,
        n_rep,
        seq_len,
        head_dim,
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


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
    key_states = _repeat_kv(key, module.num_key_value_groups).contiguous()
    value_states = _repeat_kv(value, module.num_key_value_groups).contiguous()
    atten_mask = _to_npu_attention_mask(attention_mask)
    if atten_mask is not None:
        atten_mask = atten_mask.contiguous()

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
    attn_output = _attention_output(output, query_states.shape)
    return attn_output.transpose(1, 2).contiguous(), None
