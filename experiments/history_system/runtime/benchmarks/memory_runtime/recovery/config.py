"""Configuration contract for post-draft event recovery."""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping


E1_RECOVERY_CONFIG_KEY = "post_draft_recovery"
E1_RECOVERY_VERSION = "a-e1-post-draft-event-recovery-v1"
FIRST_NAME_MARGIN_FEATURE = "first_name_top2_logprob_margin"
TASK_GENERATION_LIMIT = 96

_GATES = frozenset(
    {"disabled", "seeded_random", "first_name_margin", "prefill_linear_head"}
)
_ONE_FIFTH = {"numerator": 1, "denominator": 5}


def parse_recovery_config(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("post_draft_recovery must be a mapping")
    allowed = {
        "schema",
        "gate",
        "random_seed",
        "random_probability",
        "quota",
        "task_generation_limit",
        "margin_calibration",
        "prefill_head",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"Unknown post-draft recovery fields: {unknown!r}")
    if value.get("schema") != E1_RECOVERY_VERSION:
        raise ValueError(
            f"post_draft_recovery.schema must be {E1_RECOVERY_VERSION!r}"
        )
    gate = value.get("gate")
    if gate not in _GATES:
        raise ValueError(
            f"post_draft_recovery.gate must be one of {sorted(_GATES)!r}"
        )
    seed = value.get("random_seed", 0)
    if type(seed) is not int or seed < 0:
        raise ValueError(
            "post_draft_recovery.random_seed must be a nonnegative integer"
        )
    probability = _fraction(
        value.get("random_probability", _ONE_FIFTH), "random_probability"
    )
    quota = _fraction(value.get("quota", _ONE_FIFTH), "quota")
    if probability != _ONE_FIFTH or quota != _ONE_FIFTH:
        raise ValueError("E1 fixes random_probability and quota to exactly one fifth")
    task_limit = value.get("task_generation_limit", TASK_GENERATION_LIMIT)
    if task_limit != TASK_GENERATION_LIMIT or type(task_limit) is not int:
        raise ValueError("E1 fixes task_generation_limit to 96")

    calibration = value.get("margin_calibration")
    prefill_head = value.get("prefill_head")
    if gate == "first_name_margin":
        if not isinstance(calibration, Mapping):
            raise ValueError(
                "first_name_margin requires explicit margin_calibration"
            )
        if set(calibration) != {
            "feature",
            "direction",
            "threshold",
            "artifact_sha256",
        }:
            raise ValueError(
                "margin_calibration requires feature, direction, threshold, and "
                "artifact_sha256"
            )
        if calibration["feature"] != FIRST_NAME_MARGIN_FEATURE:
            raise ValueError("margin_calibration uses the wrong feature")
        if calibration["direction"] != "at_or_below":
            raise ValueError("margin_calibration.direction must be at_or_below")
        threshold = calibration["threshold"]
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or threshold < 0
        ):
            raise ValueError(
                "margin_calibration.threshold must be finite and nonnegative"
            )
        _validate_sha256(
            calibration["artifact_sha256"], "margin_calibration.artifact_sha256"
        )
        calibration = {**dict(calibration), "threshold": float(threshold)}
    elif calibration is not None:
        raise ValueError(
            "margin_calibration is only valid for first_name_margin"
        )

    if gate == "prefill_linear_head":
        prefill_head = _parse_prefill_head(prefill_head)
    elif prefill_head is not None:
        raise ValueError("prefill_head is only valid for prefill_linear_head")

    return {
        "schema": E1_RECOVERY_VERSION,
        "gate": gate,
        "random_seed": seed,
        "random_probability": probability,
        "quota": quota,
        "task_generation_limit": task_limit,
        "margin_calibration": calibration,
        "prefill_head": prefill_head,
    }


def public_recovery_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    head = result.get("prefill_head")
    if head is not None:
        result["prefill_head"] = {
            key: copy.deepcopy(head[key])
            for key in (
                "schema",
                "feature",
                "layer",
                "score_transform",
                "direction",
                "threshold",
                "artifact_sha256",
            )
        }
        result["prefill_head"]["dimension"] = len(head["weights"])
        result["prefill_head"]["parameters_embedded_in_runtime_config"] = True
    return result


def ceil_fraction(value: int, fraction: Mapping[str, int]) -> int:
    return (
        value * fraction["numerator"] + fraction["denominator"] - 1
    ) // fraction["denominator"]


def _parse_prefill_head(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(
            "prefill_linear_head requires an explicit exported prefill_head"
        )
    required = {
        "schema",
        "feature",
        "layer",
        "weights",
        "bias",
        "input_mean",
        "input_scale",
        "score_transform",
        "direction",
        "threshold",
        "artifact_sha256",
    }
    if set(value) != required:
        raise ValueError(
            "prefill_head fields do not match the exported linear-head schema"
        )
    if value["schema"] != "event-native-prefill-linear-head-v1":
        raise ValueError("prefill_head.schema is unsupported")
    if value["feature"] != "prefill.prompt_last.decoder_layer_output":
        raise ValueError("prefill_head.feature is unsupported")
    if type(value["layer"]) is not int or value["layer"] < 0:
        raise ValueError(
            "prefill_head.layer must be a zero-based nonnegative integer"
        )
    if (
        value["score_transform"] != "sigmoid"
        or value["direction"] != "at_or_above"
    ):
        raise ValueError(
            "prefill_head requires sigmoid scoring with at_or_above calibration"
        )
    weights = _finite_vector(value["weights"], "weights")
    means = _finite_vector(value["input_mean"], "input_mean")
    scales = _finite_vector(value["input_scale"], "input_scale")
    if not weights or len(weights) != len(means) or len(weights) != len(scales):
        raise ValueError(
            "prefill_head vectors must have one equal nonzero dimension"
        )
    if any(scale <= 0 for scale in scales):
        raise ValueError("prefill_head.input_scale values must be positive")
    bias = _finite_scalar(value["bias"], "bias")
    threshold = _finite_scalar(value["threshold"], "threshold")
    if not 0 <= threshold <= 1:
        raise ValueError(
            "prefill_head.threshold must be a probability from zero to one"
        )
    _validate_sha256(value["artifact_sha256"], "prefill_head.artifact_sha256")
    return {
        **dict(value),
        "weights": weights,
        "input_mean": means,
        "input_scale": scales,
        "bias": bias,
        "threshold": threshold,
    }


def _finite_vector(value: Any, name: str) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"prefill_head.{name} must be a list")
    return [
        _finite_scalar(item, f"{name}[{index}]")
        for index, item in enumerate(value)
    ]


def _finite_scalar(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"prefill_head.{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"prefill_head.{name} must be finite")
    return result


def _fraction(value: Any, name: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != {
        "numerator",
        "denominator",
    }:
        raise ValueError(f"{name} requires numerator and denominator")
    numerator, denominator = value["numerator"], value["denominator"]
    if (
        type(numerator) is not int
        or type(denominator) is not int
        or numerator < 0
        or denominator <= 0
        or numerator > denominator
    ):
        raise ValueError(f"{name} must be a probability fraction")
    return {"numerator": numerator, "denominator": denominator}


def _validate_sha256(value: Any, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be hexadecimal") from error
