"""Shadow-only online features for event-native tool-call generation."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch


SHADOW_FEATURE_SCHEMA = "event-native-shadow-features-v1"
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
_NAME_RE = re.compile(r'"name"\s*:\s*"((?:[^"\\]|\\.)*)"')


@dataclass(frozen=True)
class ShadowFeatureConfig:
    """Explicit opt-in contract for read-only generation features.

    A non-None layer requests that decoder-layer output. Negative indices use
    normal Python indexing and are normalized in the emitted trace. The decode
    callback must preserve protocol special tokens and disable text cleanup.
    """

    enabled: bool = False
    decode_token_ids: Callable[[Sequence[int]], str] | None = None
    prefill_layer: int | None = None
    memgen_layer: int | None = None
    model_binding: str | None = None
    tokenizer_binding: str | None = None

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be bool")
        if self.enabled and not callable(self.decode_token_ids):
            raise ValueError(
                "decode_token_ids must be callable when shadow features are enabled"
            )
        for field_name in ("prefill_layer", "memgen_layer"):
            value = getattr(self, field_name)
            if value is not None and type(value) is not int:
                raise TypeError(f"{field_name} must be int or None")
        for field_name in ("model_binding", "tokenizer_binding"):
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"{field_name} must be a nonempty string or None")


def _first_json_object_span(text: str, start: int) -> tuple[int, int] | None:
    object_start = text.find("{", start)
    if object_start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(object_start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return object_start, index + 1
    return None


def _token_char_offsets(
    decode_token_ids: Callable[[Sequence[int]], str],
    generated_ids: Sequence[int],
) -> tuple[tuple[int, int], ...]:
    offsets = []
    previous_end = 0
    for index in range(len(generated_ids)):
        start = len(decode_token_ids(generated_ids[:index]))
        end = len(decode_token_ids(generated_ids[: index + 1]))
        if start < previous_end or end < start:
            raise ValueError("decoded token-prefix lengths are not monotonic")
        offsets.append((start, end))
        previous_end = end
    return tuple(offsets)


def _char_span_to_token_span(
    offsets: Sequence[tuple[int, int]],
    start: int,
    end: int,
) -> tuple[int, int] | None:
    hits = [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > start and token_start < end
    ]
    if not hits:
        return None
    return hits[0], hits[-1]


def locate_tool_name_tokens(
    decode_token_ids: Callable[[Sequence[int]], str],
    generated_ids: Sequence[int],
) -> dict[str, Any]:
    """Locate the strict first tool-name value over continuation token indices."""

    try:
        text = decode_token_ids(generated_ids)
        if not isinstance(text, str):
            raise TypeError("decode_token_ids must return str")
        open_at = text.find(TOOL_CALL_OPEN)
        if open_at < 0:
            return {"status": "not_present", "reason": "no_tool_call"}
        payload_start = open_at + len(TOOL_CALL_OPEN)
        close_at = text.find(TOOL_CALL_CLOSE, payload_start)
        if close_at < 0:
            return {"status": "unavailable", "reason": "unclosed_tool_call"}
        json_span = _first_json_object_span(text, payload_start)
        if json_span is None or json_span[1] > close_at:
            return {"status": "unavailable", "reason": "malformed_tool_call_json"}
        try:
            payload = json.loads(text[json_span[0] : json_span[1]])
        except json.JSONDecodeError:
            return {"status": "unavailable", "reason": "malformed_tool_call_json"}
        if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
            return {"status": "unavailable", "reason": "invalid_tool_name"}

        match = _NAME_RE.search(text, json_span[0], json_span[1])
        if match is None:
            return {"status": "unavailable", "reason": "tool_name_span_missing"}
        offsets = _token_char_offsets(decode_token_ids, generated_ids)
        token_span = _char_span_to_token_span(offsets, match.start(1), match.end(1))
        if token_span is None:
            return {"status": "unavailable", "reason": "tool_name_token_unlocatable"}
        return {
            "status": "located",
            "reason": None,
            "first_generated_token_index": token_span[0],
            "last_generated_token_index": token_span[1],
        }
    except Exception as error:  # Shadow decoding must never change generation.
        return {
            "status": "unavailable",
            "reason": "tool_name_locator_error",
            "error_type": type(error).__name__,
        }


def distribution_features(
    log_probs: torch.Tensor,
    *,
    selected_token_id: int,
) -> dict[str, Any]:
    """Read top-two margin and full-vocabulary entropy without changing logits."""

    if log_probs.ndim != 1 or log_probs.numel() < 2:
        raise ValueError("log_probs must be a one-dimensional vocabulary distribution")
    if not 0 <= selected_token_id < log_probs.numel():
        raise ValueError("selected_token_id is outside the vocabulary")
    values, indices = torch.topk(log_probs, k=2)
    entropy = -(log_probs.exp() * log_probs).sum()
    numbers = (*values, entropy, log_probs[selected_token_id])
    if not all(math.isfinite(float(value.item())) for value in numbers):
        raise ValueError("shadow distribution features must be finite")
    return {
        "selected_token_id": int(selected_token_id),
        "selected_token_logprob": float(log_probs[selected_token_id].item()),
        "top1_token_id": int(indices[0].item()),
        "top2_token_id": int(indices[1].item()),
        "top1_logprob": float(values[0].item()),
        "top2_logprob": float(values[1].item()),
        "top2_logprob_margin": float((values[0] - values[1]).item()),
        "full_vocab_entropy_nats": float(entropy.item()),
        "vocab_size": int(log_probs.numel()),
    }


class ShadowFeatureRecorder:
    """Collect opt-in logits and layer outputs from the existing forward calls."""

    def __init__(
        self,
        config: ShadowFeatureConfig,
        *,
        base_model: torch.nn.Module,
        workspace_position_start: int,
        workspace_tokens: int,
    ) -> None:
        if not config.enabled:
            raise ValueError("ShadowFeatureRecorder requires an enabled config")
        self.config = config
        self.base_model = base_model
        self.workspace_position_start = int(workspace_position_start)
        self.workspace_tokens = int(workspace_tokens)
        self.steps: list[dict[str, Any]] = []
        self.prefill_hidden: list[float] | None = None
        self.generated_hidden: dict[int, list[float]] = {}
        self.capture_errors: list[dict[str, str]] = []
        self._handles: list[Any] = []
        self._forward_generated_count: int | None = None
        self._prefill_layer = self._normalize_layer(config.prefill_layer)
        self._memgen_layer = self._normalize_layer(config.memgen_layer)

    def _normalize_layer(self, layer: int | None) -> int | None:
        if layer is None:
            return None
        layers = self.base_model.model.layers
        normalized = layer if layer >= 0 else len(layers) + layer
        if not 0 <= normalized < len(layers):
            raise ValueError(
                f"shadow feature layer {layer} is outside {len(layers)} decoder layers"
            )
        return normalized

    def install(self) -> None:
        """Install read-only hooks on requested decoder layers."""

        requested = {self._prefill_layer, self._memgen_layer} - {None}
        for layer_index in sorted(requested):
            layer = self.base_model.model.layers[layer_index]
            self._handles.append(
                layer.register_forward_hook(self._make_layer_hook(layer_index))
            )

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def begin_forward(self, generated_count: int) -> None:
        self._forward_generated_count = int(generated_count)

    def _make_layer_hook(self, layer_index: int):
        def hook(_module, _inputs, output) -> None:
            try:
                hidden = output[0] if isinstance(output, tuple) else output
                if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                    raise ValueError("decoder layer output must have shape [batch, seq, hidden]")
                vector = (
                    hidden[0, -1]
                    .detach()
                    .to(device="cpu", dtype=torch.float16)
                    .tolist()
                )
                generated_count = self._forward_generated_count
                if generated_count is None:
                    return
                if generated_count == 0 and layer_index == self._prefill_layer:
                    # Chunked prefill overwrites this until the true prompt-last chunk.
                    self.prefill_hidden = vector
                elif generated_count > 0 and layer_index == self._memgen_layer:
                    # This forward consumes generated token k-1 and exposes its state.
                    self.generated_hidden[generated_count - 1] = vector
            except Exception as error:  # Hooks are observational and fail open.
                self.capture_errors.append(
                    {"component": "hidden_hook", "error_type": type(error).__name__}
                )

        return hook

    def observe_distribution(
        self,
        log_probs: torch.Tensor,
        *,
        selected_token_id: int,
    ) -> None:
        generated_token_index = len(self.steps)
        try:
            row = distribution_features(
                log_probs,
                selected_token_id=selected_token_id,
            )
            logical_position = (
                self.workspace_position_start
                + self.workspace_tokens
                + generated_token_index
            )
            row.update(
                {
                    "generated_token_index": generated_token_index,
                    "logical_position": logical_position,
                    "produced_by_logical_position": logical_position - 1,
                }
            )
            self.steps.append(row)
        except Exception as error:  # Distribution capture cannot block generation.
            self.capture_errors.append(
                {"component": "distribution", "error_type": type(error).__name__}
            )
            self.steps.append(
                {
                    "generated_token_index": generated_token_index,
                    "status": "unavailable",
                }
            )

    def finalize(self, generated_ids: Sequence[int]) -> dict[str, Any]:
        locator = locate_tool_name_tokens(
            self.config.decode_token_ids,
            generated_ids,
        )
        first_index = locator.get("first_generated_token_index")
        last_index = locator.get("last_generated_token_index")
        name_rows: list[Mapping[str, Any]] = []
        if locator["status"] == "located":
            if (
                type(first_index) is not int
                or type(last_index) is not int
                or not 0 <= first_index <= last_index < len(self.steps)
            ):
                locator = {
                    "status": "unavailable",
                    "reason": "tool_name_distribution_unavailable",
                }
                first_index = last_index = None
            else:
                name_rows = self.steps[first_index : last_index + 1]

        first_row = name_rows[0] if name_rows else None
        signals = {
            "first_name_top2_logprob_margin": (
                first_row.get("top2_logprob_margin") if first_row else None
            ),
            "first_name_full_vocab_entropy_nats": (
                first_row.get("full_vocab_entropy_nats") if first_row else None
            ),
        }
        return {
            "schema": SHADOW_FEATURE_SCHEMA,
            "status": locator["status"],
            "gold_used": False,
            "source": "existing_greedy_generation_forward_calls",
            "bindings": {
                "model": self.config.model_binding,
                "tokenizer": self.config.tokenizer_binding,
                "layer_indexing": "zero_based_decoder_layer_output",
                "generated_token_indexing": "zero_based_continuation",
                "logit_position": "distribution_that_selected_generated_token",
                "tool_name_locator": "strict_native_tool_call_name_span_v1",
                "decode_contract": "protocol_special_tokens_preserved_no_cleanup",
            },
            "signals": signals,
            "tool_name": {
                **locator,
                "first_token": dict(first_row) if first_row else None,
                "tokens": [dict(row) for row in name_rows],
            },
            "prefill": self._prefill_trace(),
            "memgen": self._memgen_trace(first_index),
            "capture_errors": list(self.capture_errors),
        }

    def _prefill_trace(self) -> dict[str, Any]:
        if self._prefill_layer is None:
            return {"status": "disabled", "layer": None, "hidden": None}
        position = self.workspace_position_start + self.workspace_tokens - 1
        return {
            "status": "captured" if self.prefill_hidden is not None else "unavailable",
            "reason": None if self.prefill_hidden is not None else "layer_hook_not_observed",
            "layer": self._prefill_layer,
            "position": {
                "kind": "prompt_last",
                "logical_position": position,
            },
            "readout": "decoder_layer_output",
            "stored_dtype": "float16",
            "hidden": self.prefill_hidden,
        }

    def _memgen_trace(self, first_index: int | None) -> dict[str, Any]:
        if self._memgen_layer is None:
            return {"status": "disabled", "layer": None, "hidden": None}
        if first_index is None:
            return {
                "status": "unavailable",
                "reason": "tool_name_unavailable",
                "layer": self._memgen_layer,
                "hidden": None,
            }
        hidden = self.generated_hidden.get(first_index)
        logical_position = (
            self.workspace_position_start + self.workspace_tokens + first_index
        )
        return {
            "status": "captured" if hidden is not None else "unavailable",
            "reason": None if hidden is not None else "post_token_hidden_not_observed",
            "layer": self._memgen_layer,
            "position": {
                "kind": "tool_name_first_token",
                "generated_token_index": first_index,
                "logical_position": logical_position,
            },
            "readout": "decoder_layer_output_after_emitted_token",
            "stored_dtype": "float16",
            "hidden": hidden,
        }
