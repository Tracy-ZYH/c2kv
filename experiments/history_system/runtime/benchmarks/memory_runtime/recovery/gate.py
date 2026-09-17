"""Observable gate evaluation for post-draft recovery."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any, Mapping

from .config import E1_RECOVERY_VERSION, FIRST_NAME_MARGIN_FEATURE


def evaluate_gate(
    config: Mapping[str, Any],
    *,
    session_id: str,
    decision_key: str,
    shadow_features: Any,
) -> dict[str, Any]:
    gate = config["gate"]
    if gate == "seeded_random":
        digest = hashlib.sha256(
            canonical(
                [
                    E1_RECOVERY_VERSION,
                    config["random_seed"],
                    session_id,
                    decision_key,
                    "gate",
                ]
            ).encode("utf-8")
        ).hexdigest()
        numerator = int(digest, 16)
        probability = config["random_probability"]
        triggered = (
            numerator * probability["denominator"]
            < probability["numerator"] * (1 << 256)
        )
        return {
            "type": gate,
            "triggered": triggered,
            "reason": (
                "seeded_random_trigger" if triggered else "seeded_random_abstain"
            ),
            "seed": config["random_seed"],
            "draw_sha256": digest,
            "probability": copy.deepcopy(probability),
            "online_only": True,
        }

    if gate == "first_name_margin":
        calibration = config["margin_calibration"]
        margin, reason = _read_margin(shadow_features)
        if reason is not None:
            return {
                "type": gate,
                "triggered": False,
                "reason": reason,
                "feature": FIRST_NAME_MARGIN_FEATURE,
                "value": None,
                "calibration_artifact_sha256": calibration["artifact_sha256"],
            }
        triggered = margin <= calibration["threshold"]
        return {
            "type": gate,
            "triggered": triggered,
            "reason": (
                "margin_at_or_below_threshold"
                if triggered
                else "margin_above_threshold"
            ),
            "feature": FIRST_NAME_MARGIN_FEATURE,
            "value": margin,
            "direction": "at_or_below",
            "threshold": calibration["threshold"],
            "calibration_artifact_sha256": calibration["artifact_sha256"],
        }

    head = config["prefill_head"]
    score, reason = _read_prefill_score(shadow_features, head)
    if reason is not None:
        return {
            "type": gate,
            "triggered": False,
            "reason": reason,
            "feature": head["feature"],
            "score": None,
            "head_artifact_sha256": head["artifact_sha256"],
        }
    triggered = score >= head["threshold"]
    return {
        "type": gate,
        "triggered": triggered,
        "reason": (
            "prefill_score_at_or_above_threshold"
            if triggered
            else "prefill_score_below_threshold"
        ),
        "feature": head["feature"],
        "score": score,
        "direction": "at_or_above",
        "threshold": head["threshold"],
        "layer": head["layer"],
        "head_artifact_sha256": head["artifact_sha256"],
    }


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _read_margin(value: Any):
    if not isinstance(value, Mapping):
        return None, "shadow_features_unavailable"
    if value.get("schema") != "event-native-shadow-features-v1":
        return None, "shadow_features_schema_unavailable"
    tool_name = value.get("tool_name")
    if not isinstance(tool_name, Mapping) or tool_name.get("status") != "located":
        return None, "tool_name_feature_unavailable"
    signals = value.get("signals")
    if not isinstance(signals, Mapping):
        return None, "margin_signal_unavailable"
    margin = signals.get(FIRST_NAME_MARGIN_FEATURE)
    if isinstance(margin, bool) or not isinstance(margin, (int, float)):
        return None, "margin_signal_unavailable"
    margin = float(margin)
    if not math.isfinite(margin) or margin < 0:
        return None, "margin_signal_invalid"
    return margin, None


def _read_prefill_score(value: Any, head: Mapping[str, Any]):
    if (
        not isinstance(value, Mapping)
        or value.get("schema") != "event-native-shadow-features-v1"
    ):
        return None, "shadow_features_schema_unavailable"
    prefill = value.get("prefill")
    if not isinstance(prefill, Mapping) or prefill.get("status") != "captured":
        return None, "prefill_hidden_unavailable"
    if prefill.get("layer") != head["layer"]:
        return None, "prefill_layer_mismatch"
    position = prefill.get("position")
    if not isinstance(position, Mapping) or position.get("kind") != "prompt_last":
        return None, "prefill_position_mismatch"
    if prefill.get("readout") != "decoder_layer_output":
        return None, "prefill_readout_mismatch"
    hidden = prefill.get("hidden")
    dimension = len(head["weights"])
    if not isinstance(hidden, list) or len(hidden) != dimension:
        return None, "prefill_feature_dimension_mismatch"
    values = []
    for item in hidden:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None, "prefill_hidden_invalid"
        item = float(item)
        if not math.isfinite(item):
            return None, "prefill_hidden_invalid"
        values.append(item)
    normalized = [
        (item - mean) / scale
        for item, mean, scale in zip(
            values, head["input_mean"], head["input_scale"], strict=True
        )
    ]
    logit = math.fsum(
        weight * item
        for weight, item in zip(head["weights"], normalized, strict=True)
    ) + head["bias"]
    if logit >= 0:
        score = 1.0 / (1.0 + math.exp(-logit))
    else:
        exponent = math.exp(logit)
        score = exponent / (1.0 + exponent)
    return score, None
