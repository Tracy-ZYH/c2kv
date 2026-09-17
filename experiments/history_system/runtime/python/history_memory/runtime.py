"""Differentiable tensor runtime for event-native history-memory training.

The packing layer owns text, event, and position planning.  This module owns
only tensor execution: it extracts pre-RoPE gist KV once per exact chunk input
within a forward batch, places those KVs at the packer's source-span positions,
and supervises the complete assistant continuation with a per-decision mean.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Real
from typing import Any, Iterator, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import DynamicCache

from models.gist_utils import apply_rotary_pos_emb

from .packing import EncoderChunk, GistPlacement, PackedMemory


@dataclass(frozen=True)
class PreparedDecision:
    """One already-tokenized decision consumed by :class:`HistoryMemoryModel`."""

    memory: PackedMemory
    target_ids: tuple[int, ...]
    ratio: int
    weight: float = 1.0
    decision_id: str = ""


@dataclass(frozen=True)
class _EncodedChunk:
    """Valid pre-RoPE gist KV rows for one exact encoder input."""

    key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    local_position_ids: tuple[int, ...]

    @property
    def gist_tokens(self) -> int:
        return len(self.local_position_ids)


@dataclass(frozen=True)
class _LayerState:
    keys: torch.Tensor
    values: torch.Tensor


class _OneLayerPast:
    """Minimal training cache view consumed by Qwen3Attention.forward."""

    def __init__(
        self,
        num_layers: int,
        layer_index: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        self.layers: list[_LayerState | None] = [None] * num_layers
        self.layers[layer_index] = _LayerState(key, value)

    def update(self, key: torch.Tensor, value: torch.Tensor, layer_index: int):
        """Support eval-mode CE without mutating the reusable event prefix."""
        prefix = self.layers[layer_index]
        if prefix is None:
            raise ValueError("Only the current decoder layer is present in this cache")
        return torch.cat((prefix.keys, key), dim=-2), torch.cat((prefix.values, value), dim=-2)


class HistoryMemoryModel(nn.Module):
    """Train gist embedding/QKV parameters against packed assistant decisions.

    ``base_model`` must be the repository's ``Qwen3ForCausalLM``-compatible
    model.  Construction freezes the native model and enables gradients only
    for ``gist_embed_tokens`` and ``gist_{q,k,v}_proj``.  Chunk reuse is scoped
    to one :meth:`forward` call so gradient accumulation never retains a graph
    after its loss has been backpropagated.
    """

    _GIST_PROJECTIONS = (".gist_q_proj.", ".gist_k_proj.", ".gist_v_proj.")

    def __init__(self, base_model: nn.Module):
        super().__init__()
        if not hasattr(base_model, "model") or not hasattr(base_model, "lm_head"):
            raise TypeError("base_model must expose Qwen causal-LM 'model' and 'lm_head' modules")
        inner = base_model.model
        for attribute in ("generate_gist", "rotary_emb", "embed_tokens", "layers"):
            if not hasattr(inner, attribute):
                raise TypeError(f"base_model.model must expose {attribute!r}")

        self.base_model = base_model
        self._parameter_version = 0
        for name, parameter in self.base_model.named_parameters():
            trainable = (
                name.startswith("model.gist_embed_tokens.")
                or any(marker in name for marker in self._GIST_PROJECTIONS)
            )
            parameter.requires_grad_(trainable)
            # AdamW must update trainable weights in FP32.  Keeping a BF16
            # parameter plus BF16 optimizer state makes lr=5e-5 updates vanish
            # below the parameter's quantization step.  Preserve FP32 and
            # higher precision tiny/reference models unchanged.
            if trainable and parameter.dtype in (torch.float16, torch.bfloat16):
                parameter.data = parameter.data.float()

        trainable_names = [
            name for name, parameter in self.base_model.named_parameters()
            if parameter.requires_grad
        ]
        if not trainable_names:
            raise ValueError("base_model exposes no trainable gist embedding/QKV parameters")

    @property
    def parameter_version(self) -> int:
        """Version included in exact-input keys for the current gist weights."""

        return self._parameter_version

    def advance_parameter_version(self) -> int:
        """Advance the cache namespace after one successful optimizer update."""

        self._parameter_version += 1
        return self._parameter_version

    def restore_parameter_version(self, version: int) -> None:
        """Restore the optimizer-step namespace from a training checkpoint."""

        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            raise ValueError("parameter version must be a nonnegative integer")
        self._parameter_version = version

    def forward(self, decisions: Sequence[PreparedDecision]) -> dict[str, Any]:
        """Return a weighted decision-mean loss and auditable token counters.

        Items are intentionally accepted by attribute (duck typing), allowing
        a corpus record to carry extra metadata without an adapter allocation.
        ``loss`` is ``sum(weight_i * mean_token_ce_i) / sum(weight_i)``.
        """

        if not isinstance(decisions, Sequence) or isinstance(
            decisions, (str, bytes, bytearray)
        ):
            raise TypeError("decisions must be a sequence of prepared records")
        if not decisions:
            raise ValueError("A forward batch must contain at least one decision")

        graph_cache: dict[tuple[Any, ...], _EncodedChunk] = {}
        system_cache: dict[tuple[int, ...], tuple[tuple[torch.Tensor, torch.Tensor], ...]] = {}
        losses: list[torch.Tensor] = []
        weights: list[float] = []
        stats = {
            "source_tokens": 0,
            "presented_encoder_tokens": 0,
            "gist_tokens": 0,
            "resident_kv_tokens": 0,
            "extracted_chunks": 0,
            "reused_chunks": 0,
            "supervised_tokens": 0,
            "decision_count": len(decisions),
        }

        for decision in decisions:
            memory, target_ids, ratio, weight = self._validate_decision(decision)
            placements = memory.gist_layout(ratio)
            encoded: list[_EncodedChunk] = []
            for placement in placements:
                key = placement.chunk.encoding_key(
                    parameter_version=self._parameter_version,
                    ratio=ratio,
                )
                chunk_output = graph_cache.get(key)
                if chunk_output is None:
                    chunk_output = self._encode_chunk(placement.chunk, ratio)
                    graph_cache[key] = chunk_output
                    stats["extracted_chunks"] += 1
                else:
                    stats["reused_chunks"] += 1
                self._validate_placement(placement, chunk_output)
                encoded.append(chunk_output)

            prefix_key_values = system_cache.get(memory.system_input_ids)
            if prefix_key_values is None and memory.system_input_ids:
                prefix_key_values = self._encode_system(memory.system_input_ids)
                system_cache[memory.system_input_ids] = prefix_key_values
            elif prefix_key_values is None:
                prefix_key_values = ()

            past_key_values, physical_past = self._assemble_cache(
                prefix_key_values,
                placements,
                encoded,
            )
            decision_loss = self._decision_loss(
                memory,
                target_ids,
                past_key_values,
                physical_past,
            )
            losses.append(decision_loss)
            weights.append(weight)

            costs = memory.costs(ratio)
            stats["source_tokens"] += self._source_tokens(memory)
            stats["presented_encoder_tokens"] += costs["presented_encoder_tokens"]
            stats["gist_tokens"] += costs["gist_tokens"]
            stats["resident_kv_tokens"] += costs["resident_kv_tokens"]
            stats["supervised_tokens"] += len(target_ids)

        total_weight = sum(weights)
        if total_weight <= 0.0:
            raise ValueError("A forward batch must have positive total decision weight")
        weighted_loss = torch.stack(
            [loss * weight for loss, weight in zip(losses, weights)]
        ).sum() / total_weight
        # Touch every trainable tensor on every rank.  In particular, an
        # all-no-gist batch remains backwardable and yields explicit zero gist
        # gradients without unfreezing any native Qwen parameter.
        weighted_loss = weighted_loss + self._trainable_zero_anchor()

        return {
            "loss": weighted_loss,
            "decision_losses": torch.stack(losses),
            "stats": stats,
        }

    def _validate_decision(
        self, decision: Any
    ) -> tuple[PackedMemory, tuple[int, ...], int, float]:
        try:
            memory = decision.memory
            raw_target_ids = decision.target_ids
            ratio = decision.ratio
            raw_weight = decision.weight
        except AttributeError as exc:
            raise TypeError(
                "Every decision must expose memory, target_ids, ratio, and weight"
            ) from exc
        if not isinstance(memory, PackedMemory):
            raise TypeError("decision.memory must be a PackedMemory")
        if isinstance(ratio, bool) or not isinstance(ratio, int) or ratio <= 0:
            raise ValueError("decision.ratio must be a positive integer")
        if not isinstance(raw_target_ids, Sequence) or isinstance(
            raw_target_ids, (str, bytes, bytearray)
        ):
            raise TypeError("decision.target_ids must be a token sequence")
        target_ids = tuple(raw_target_ids)
        if not target_ids:
            raise ValueError("decision.target_ids must contain the complete assistant target")
        if any(isinstance(token, bool) or not isinstance(token, int) for token in target_ids):
            raise TypeError("decision.target_ids must contain integers")
        vocab_size = int(self.base_model.config.vocab_size)
        if any(token < 0 or token >= vocab_size for token in target_ids):
            raise ValueError("decision.target_ids contains a token outside the model vocabulary")
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, Real):
            raise TypeError("decision.weight must be a real number")
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("decision.weight must be finite and nonnegative")
        return memory, target_ids, ratio, weight

    def _encode_chunk(self, chunk: EncoderChunk, ratio: int) -> _EncodedChunk:
        device = self.base_model.model.embed_tokens.weight.device
        input_ids = torch.tensor(chunk.token_ids, dtype=torch.long, device=device).unsqueeze(0)
        if input_ids.shape[1] == 0:
            raise ValueError("Encoder chunks must contain at least one token")
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        with self._gist_embedding_in_base_dtype():
            outputs, gist_mask, position_ids = self.base_model.model.generate_gist(
                input_ids=input_ids,
                attention_mask=attention_mask,
                ratio=ratio,
            )
        valid = gist_mask[0].bool()
        local_positions = tuple(int(value) for value in position_ids[0, valid].tolist())
        key_values = tuple(
            (
                key[:, :, valid, :],
                value[:, :, valid, :],
            )
            for key, value in outputs.past_key_values
        )
        if not local_positions or any(
            key.shape[-2] != len(local_positions) or value.shape[-2] != len(local_positions)
            for key, value in key_values
        ):
            raise RuntimeError("generate_gist returned inconsistent valid KV rows")
        return _EncodedChunk(key_values, local_positions)

    @staticmethod
    def _validate_placement(
        placement: GistPlacement, encoded: _EncodedChunk
    ) -> None:
        expected_local = tuple(
            position - placement.source_position_start
            for position in placement.position_ids
        )
        if encoded.local_position_ids != expected_local:
            raise RuntimeError(
                "generate_gist positions disagree with PackedMemory.gist_layout: "
                f"generated={encoded.local_position_ids}, packed={expected_local}"
            )

    def _encode_system(
        self, system_input_ids: tuple[int, ...], *, prefill_chunk_size: int | None = None
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        if prefill_chunk_size is not None and (
            type(prefill_chunk_size) is not int or prefill_chunk_size <= 0
        ):
            raise ValueError("prefill_chunk_size must be a positive integer or None")
        device = self.base_model.model.embed_tokens.weight.device
        input_ids = torch.tensor(system_input_ids, dtype=torch.long, device=device).unsqueeze(0)
        # Preserve the existing one-call training/default path.  Opt-in serving
        # chunking appends the same causal prefix at its original RoPE positions.
        with self._temporary_eval(), torch.no_grad():
            if prefill_chunk_size is None:
                outputs = self.base_model.model(
                    input_ids=input_ids, attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
                    use_cache=True, use_gist=False,
                )
                cache = outputs.past_key_values
            else:
                cache = None
                for start in range(0, len(system_input_ids), prefill_chunk_size):
                    end = min(start + prefill_chunk_size, len(system_input_ids))
                    outputs = self.base_model.model(
                        input_ids=input_ids[:, start:end],
                        attention_mask=torch.ones((1, end), dtype=torch.bool, device=device),
                        position_ids=torch.arange(start, end, device=device).unsqueeze(0),
                        past_key_values=cache, use_cache=True, use_gist=False,
                    )
                    cache = outputs.past_key_values
                    if cache.get_seq_length() != end:
                        raise RuntimeError("Chunked system prefix cache length mismatch")
        key_values = tuple((layer.keys, layer.values) for layer in cache.layers)
        if any(key.shape[-2] != len(system_input_ids) for key, _ in key_values):
            raise RuntimeError("Native system prefix cache has an unexpected sequence length")
        return key_values

    def _assemble_cache(
        self,
        prefix_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        placements: tuple[GistPlacement, ...],
        encoded_chunks: Sequence[_EncodedChunk],
    ) -> tuple[DynamicCache | None, int]:
        gist_tokens = sum(encoded.gist_tokens for encoded in encoded_chunks)
        prefix_tokens = prefix_key_values[0][0].shape[-2] if prefix_key_values else 0
        physical_past = int(prefix_tokens) + gist_tokens
        if physical_past == 0:
            return None, 0

        num_layers = len(self.base_model.model.layers)
        if prefix_key_values and len(prefix_key_values) != num_layers:
            raise RuntimeError("System cache layer count does not match the model")
        if any(len(encoded.key_values) != num_layers for encoded in encoded_chunks):
            raise RuntimeError("Gist cache layer count does not match the model")

        rotated_chunks: list[tuple[tuple[torch.Tensor, torch.Tensor], ...]] = []
        for placement, encoded in zip(placements, encoded_chunks):
            first_key = encoded.key_values[0][0]
            positions = torch.tensor(
                placement.position_ids,
                dtype=torch.long,
                device=first_key.device,
            ).unsqueeze(0)
            cos, sin = self.base_model.model.rotary_emb(first_key, positions)
            rotated_layers = []
            for key, value in encoded.key_values:
                if key.device != first_key.device:
                    raise RuntimeError("A gist chunk spans devices; layer-sharded execution is unsupported")
                rotated_layers.append((apply_rotary_pos_emb(key, cos, sin), value))
            rotated_chunks.append(tuple(rotated_layers))

        merged_layers = []
        for layer_index in range(num_layers):
            keys = []
            values = []
            if prefix_key_values:
                prefix_key, prefix_value = prefix_key_values[layer_index]
                keys.append(prefix_key)
                values.append(prefix_value)
            for chunk_layers in rotated_chunks:
                key, value = chunk_layers[layer_index]
                keys.append(key)
                values.append(value)
            merged_layers.append((torch.cat(keys, dim=-2), torch.cat(values, dim=-2)))
        return DynamicCache(merged_layers, config=self.base_model.config), physical_past

    def _decision_loss(
        self,
        memory: PackedMemory,
        target_ids: tuple[int, ...],
        past_key_values: DynamicCache | None,
        physical_past: int,
    ) -> torch.Tensor:
        device = self.base_model.model.embed_tokens.weight.device
        current_ids = memory.workspace_input_ids + target_ids
        if len(memory.workspace_input_ids) == 0:
            raise ValueError("A packed decision needs a native workspace/generation prefix")
        input_ids = torch.tensor(current_ids, dtype=torch.long, device=device).unsqueeze(0)
        attention_mask = torch.ones(
            (1, physical_past + len(current_ids)),
            dtype=torch.bool,
            device=device,
        )
        position_ids = torch.arange(
            memory.workspace_position_start,
            memory.workspace_position_start + len(current_ids),
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        supervised_indices = torch.arange(
            len(memory.workspace_input_ids) - 1,
            len(memory.workspace_input_ids) - 1 + len(target_ids),
            dtype=torch.long,
            device=device,
        )
        prediction_logits = self._target_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            supervised_indices=supervised_indices,
        )
        labels = torch.tensor(target_ids, dtype=torch.long, device=prediction_logits.device)
        return F.cross_entropy(prediction_logits.float(), labels, reduction="mean")

    def _target_logits(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: DynamicCache | None,
        supervised_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Run native-QKV Qwen layers while retaining a differentiable past.

        Transformers' generic ``GradientCheckpointingLayer`` deliberately
        drops ``past_key_values`` before invoking a decoder layer.  Gist
        training needs that past and needs gradients through it, so the
        runtime checkpoints ``layer.forward`` directly and passes the current
        layer's past K/V as tensor inputs.  This also makes the physical causal
        mask independent of the source-span values used only by RoPE.
        """

        inner = self.base_model.model
        hidden_states = inner.embed_tokens(input_ids)
        query_length = input_ids.shape[1]
        physical_past = attention_mask.shape[1] - query_length
        key_indices = torch.arange(
            physical_past + query_length, device=hidden_states.device
        )
        query_indices = torch.arange(query_length, device=hidden_states.device)
        allowed = key_indices.unsqueeze(0) <= (
            physical_past + query_indices.unsqueeze(1)
        )
        causal_mask = torch.zeros(
            (1, 1, query_length, physical_past + query_length),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        causal_mask.masked_fill_(~allowed.unsqueeze(0).unsqueeze(0), torch.finfo(hidden_states.dtype).min)
        causal_mask.masked_fill_(
            ~attention_mask[:, None, None, :].bool(),
            torch.finfo(hidden_states.dtype).min,
        )
        position_embeddings = inner.rotary_emb(hidden_states, position_ids)
        num_layers = len(inner.layers)

        for layer_index, decoder_layer in enumerate(inner.layers[: inner.config.num_hidden_layers]):
            if past_key_values is None:
                layer_key = layer_value = None
            else:
                layer_key = past_key_values.layers[layer_index].keys
                layer_value = past_key_values.layers[layer_index].values

            def layer_forward(
                states: torch.Tensor,
                key: torch.Tensor | None = layer_key,
                value: torch.Tensor | None = layer_value,
                *,
                layer: nn.Module = decoder_layer,
                index: int = layer_index,
            ) -> torch.Tensor:
                layer_past = None
                if key is not None and value is not None:
                    layer_past = _OneLayerPast(num_layers, index, key, value)
                return layer.forward(
                    states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=layer_past,
                    use_cache=False,
                    position_embeddings=position_embeddings,
                    use_gist=False,
                )

            if getattr(decoder_layer, "gradient_checkpointing", False) and decoder_layer.training:
                if layer_key is None:
                    hidden_states = decoder_layer._gradient_checkpointing_func(
                        layer_forward,
                        hidden_states,
                    )
                else:
                    hidden_states = decoder_layer._gradient_checkpointing_func(
                        layer_forward,
                        hidden_states,
                        layer_key,
                        layer_value,
                    )
            else:
                hidden_states = layer_forward(hidden_states, layer_key, layer_value)

        hidden_states = inner.norm(hidden_states)
        selected = hidden_states.index_select(1, supervised_indices)
        return self.base_model.lm_head(selected)[0]

    def _trainable_zero_anchor(self) -> torch.Tensor:
        anchor = None
        for parameter in self.base_model.parameters():
            if not parameter.requires_grad:
                continue
            term = parameter.reshape(-1)[0] * 0.0
            anchor = term if anchor is None else anchor + term
        if anchor is None:  # guarded in __init__, retained for post-init mutation
            raise RuntimeError("HistoryMemoryModel has no trainable gist parameters")
        return anchor

    @staticmethod
    def _source_tokens(memory: PackedMemory) -> int:
        return sum(
            max(
                chunk.source_token_end
                for chunk in memory.chunks
                if chunk.event_id == event_id
            )
            for event_id in {chunk.event_id for chunk in memory.chunks}
        )

    @contextmanager
    def _temporary_eval(self) -> Iterator[None]:
        states = tuple((module, module.training) for module in self.base_model.modules())
        self.base_model.eval()
        try:
            yield
        finally:
            for module, training in states:
                module.training = training

    @contextmanager
    def _gist_embedding_in_base_dtype(self) -> Iterator[None]:
        """Keep compressor hidden states in the frozen base compute dtype."""

        target_dtype = self.base_model.model.embed_tokens.weight.dtype

        def cast_output(_module: nn.Module, _args: tuple[Any, ...], output: torch.Tensor):
            return output.to(dtype=target_dtype)

        handle = self.base_model.model.gist_embed_tokens.register_forward_hook(cast_output)
        try:
            yield
        finally:
            handle.remove()


__all__ = ["HistoryMemoryModel", "PreparedDecision"]
