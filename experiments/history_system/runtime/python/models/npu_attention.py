from __future__ import annotations

from typing import Any

import torch

# ---------------------------------------------------------------------------
# transformers>=5 mask registry fix (CRITICAL).
#
# transformers 5.x builds the causal mask via ALL_MASK_ATTENTION_FUNCTIONS.
# An unregistered attention implementation makes `create_causal_mask` return
# None, which silently turns the teacher-forced training/eval forward into
# FULL BIDIRECTIONAL attention (answer positions attend to future answer
# tokens -> label leakage: train/eval losses collapse to ~0 while causal
# ability is unchanged).  Every c2kv checkpoint trained through
# GistMultiDocTrainer with --attn_impl npu_fusion_attention was affected.
# Registering the eager mask factory restores a proper additive causal mask;
# `_to_npu_attention_mask` below converts it to torch_npu's bool-drop
# convention.  Verified empirically: pre-registration -> None, post ->
# [B,1,Q,KV] upper-triangular -inf float mask.
# ---------------------------------------------------------------------------
try:
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask

    if "npu_fusion_attention" not in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping:
        ALL_MASK_ATTENTION_FUNCTIONS.register("npu_fusion_attention", eager_mask)
except Exception:  # pragma: no cover - transformers without the mask registry
    pass


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
