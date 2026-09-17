"""Runtime key-bias / LESS-fold registry for the eager attention path.

This is the decode channel the D4/D5/D7 arms inject through (review D:
without it the logit bias never reaches the model).  Design:

- ``ACTIVE_LAYER_BIAS: {layer_idx: LayerBiasEntry}`` — module-level,
  default EMPTY.  The eager path calls ``get_entry(layer_idx)`` at two
  guarded sites; when the registry is empty nothing is touched and the
  output is bit-identical to the vanilla path (the running v2 arms never
  populate it).
- ``LayerBiasEntry.key_bias`` — (H_kv, P) additive bias over the cache's
  PHYSICAL key positions (ResKV log-counts, KeepKV votes, SelKV alpha·log
  R).  Positions beyond P (new tokens) get 0.
- ``LayerBiasEntry.less_H / less_z`` — the D5 ledger fold: with phi(q) =
  elu(q)+1 of the CURRENT query (per q head via its kv head),
  o = (sum e^{qk}v + phi(q)H) / (sum e^{qk} + phi(q)z), carried in
  max-shifted units exactly as d_attn_ext.attention_with_bias does.

Phase state: phase 1 (pre-softmax) stashes the row shift/denominator,
phase 2 (post-matmul) consumes it — strictly within one forward call.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch


class LayerBiasEntry:
    def __init__(self, key_bias: Optional[torch.Tensor] = None,
                 less_H: Optional[torch.Tensor] = None,
                 less_z: Optional[torch.Tensor] = None):
        if key_bias is None and less_H is None:
            raise ValueError("empty LayerBiasEntry")
        self.key_bias = key_bias        # (H_kv, P) float32
        self.less_H = less_H            # (H_kv, D, Dv) float32
        self.less_z = less_z            # (H_kv, D) float32
        self._den0 = None               # (B, H_q, Lq, 1) stashed by phase 1
        self._e_neg_m = None            # (B, H_q, Lq, 1)

    # ---- phase 1: pre-softmax, on (B, H_q, Lq, Lk) logits ---------------
    def pre_softmax(self, attn_weights: torch.Tensor, n_rep: int) -> torch.Tensor:
        B, Hq, Lq, Lk = attn_weights.shape
        w = attn_weights.float()
        if self.key_bias is not None:
            Hkv, P = self.key_bias.shape
            if P > Lk:
                raise ValueError(f"bias spans {P} positions > cache+tokens {Lk}")
            pad = torch.zeros(Hkv, Lk - P, dtype=self.key_bias.dtype, device=self.key_bias.device)
            bias = torch.cat([self.key_bias.to(w.device), pad], dim=1)          # (Hkv, Lk)
            bias = bias.repeat_interleave(n_rep, dim=0).view(1, Hq, 1, Lk)      # GQA expand
            w = w + bias.float()
        if self.less_H is not None:
            m = w.max(dim=-1, keepdim=True).values
            m = m.where(torch.isfinite(m), torch.zeros_like(m))
            self._e_neg_m = torch.exp(-m)
            self._den0 = torch.exp(w - m).sum(dim=-1, keepdim=True)
        return w.to(attn_weights.dtype) if self.key_bias is not None else attn_weights

    # ---- phase 2: post-matmul correction on (B, H_q, Lq, Dv) ------------
    def post_output(self, attn_output: torch.Tensor, query: torch.Tensor,
                    n_rep: int) -> torch.Tensor:
        if self.less_H is None or self._den0 is None:
            return attn_output
        B, Hq, Lq, Dv = attn_output.shape
        phi = torch.nn.functional.elu(query.float()) + 1.0        # (B, Hq, Lq, D)
        Hq_kv = self.less_H.shape[0]
        n_rep_actual = Hq // Hq_kv
        # each q head uses its kv head's ledger
        H_rep = self.less_H.to(phi.device).repeat_interleave(n_rep_actual, dim=0)  # (Hq, D, Dv)
        z_rep = self.less_z.to(phi.device).repeat_interleave(n_rep_actual, dim=0)  # (Hq, D)
        extra_num = torch.einsum("bhqd,hde->bhqe", phi, H_rep) * self._e_neg_m.to(phi.device)
        extra_den = torch.einsum("bhqd,hd->bhq", phi, z_rep)[..., None] * self._e_neg_m.to(phi.device)
        den = self._den0.to(phi.device) + extra_den
        out = (attn_output.float() * self._den0.to(phi.device) + extra_num) / den.clamp_min(1e-20)
        self._den0 = None
        self._e_neg_m = None
        return out.to(attn_output.dtype)


ACTIVE_LAYER_BIAS: Dict[int, LayerBiasEntry] = {}


def get_entry(layer_idx: int) -> Optional[LayerBiasEntry]:
    return ACTIVE_LAYER_BIAS.get(layer_idx)


def set_entries(entries: Dict[int, LayerBiasEntry]) -> None:
    ACTIVE_LAYER_BIAS.clear()
    ACTIVE_LAYER_BIAS.update(entries)


def clear() -> None:
    ACTIVE_LAYER_BIAS.clear()
