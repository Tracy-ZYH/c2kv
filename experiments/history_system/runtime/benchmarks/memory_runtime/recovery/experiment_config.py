"""Independent G--P switches for the current history runtime."""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping


GP_CONFIG_KEY = "gp_experiments"
GP_SCHEMA = "a-history-gp-v1"
GP_DEFAULTS = {
    "schema": GP_SCHEMA,
    "G": "current", "U": "event", "B": "source", "Q": "lexical",
    "K": 1, "L": "next_decision", "R": 1, "D": "detector",
    "P": "quoted", "order": "chronological", "candidate_limit": 8,
    "selection_threshold": 0.0, "backend": None, "candidate_scorer": None,
}
CHOICES = {
    "G": ("current", "event", "record", "adjacent_pair", "record_bound", "record_bound_structural"),
    "U": ("event", "tokens_256", "tokens_512", "tokens_1024", "record", "field",
          "tokens_1024_aligned", "tokens_1024_shifted"),
    "B": ("source", "adjacent", "predecessor_1", "predecessor_2"),
    "Q": ("lexical", "dual", "hybrid", "llm_rewrite", "archive_rrf"),
    "K": (1, 2, 4, "threshold", "llm"),
    "L": ("next_decision", "two_decisions", "user_turn", "task"),
    "R": (1, 2, 3, 4),
    "D": ("detector", "candidate_rule", "detector_llm", "joint_llm", "supervised", "candidate_or_detector"),
    "P": ("quoted", "structured"),
    "order": ("chronological", "relevance"),
}

# Optional extensions are deliberately not inserted into legacy configurations:
# old serialized configurations and their candidate identities remain stable.
EXTENSION_DEFAULTS = {
    "selector": None, "selector_min_units": 1, "selector_max_units": 1,
    "selector_catalog": "units", "field_candidate_limit": 32,
    "hybrid_fusion": "weighted", "rrf_k": 60, "detector_threshold": None,
    "detector_calibration_telemetry": False,
    "selection_protocol": "legacy", "set_selector": "candidate_rule",
    "retrieval_limit": 24, "retrieval_route_limit": 16, "fallback_unit": None,
    "semantic_query_overflow_policy": "error",
    "recovery_reserve_tokens": 0, "local_models": None, "selector_artifact": None,
    "selector_threshold": 0.5, "gain_delta": 0.0, "export_selection_state": False,
}
EXTENSION_CHOICES = {
    "selector": ("fixed", "llm", "supervised"),
    "selector_catalog": ("units", "retrieved_fields"),
    "hybrid_fusion": ("weighted", "rrf"),
    "selection_protocol": ("legacy", "evidence_sets_v1"),
    "set_selector": ("candidate_rule", "legacy_prefill", "risk", "reranker", "local_llm",
                     "gain_turn", "gain_task", "parameter_source"),
    "semantic_query_overflow_policy": (
        "error", "task_head_tail_preserve_draft_v1"
    ),
}


def parse_gp_config(value: Mapping) -> dict:
    if not isinstance(value, Mapping):
        raise TypeError("gp_experiments must be an object")
    unknown = set(value) - set(GP_DEFAULTS) - set(EXTENSION_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown G--P fields: {sorted(unknown)}")
    config = {**copy.deepcopy(GP_DEFAULTS), **copy.deepcopy(dict(value))}
    if config["schema"] != GP_SCHEMA:
        raise ValueError(f"gp_experiments.schema must be {GP_SCHEMA}")
    for key, choices in CHOICES.items():
        if isinstance(config[key], bool) or config[key] not in choices:
            raise ValueError(f"{key} must be one of {choices!r}")
    if type(config["candidate_limit"]) is not int or config["candidate_limit"] < 1:
        raise ValueError("candidate_limit must be a positive integer")
    threshold = config["selection_threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
        raise ValueError("selection_threshold must be finite")
    for key, choices in EXTENSION_CHOICES.items():
        if key in config and config[key] not in choices:
            raise ValueError(f"{key} must be one of {choices!r}")
    if "selector" in config and config["D"] in {"detector_llm", "joint_llm", "supervised"}:
        raise ValueError("Explicit selector requires an independent D: detector, candidate_rule, or candidate_or_detector")
    for key in ("selector_max_units", "field_candidate_limit", "rrf_k"):
        if key in config and (type(config[key]) is not int or config[key] < 1):
            raise ValueError(f"{key} must be a positive integer")
    minimum = config.get("selector_min_units", 1)
    maximum = config.get("selector_max_units", config["K"] if type(config["K"]) is int else 1)
    if type(minimum) is not int or not 0 <= minimum <= maximum <= 4:
        raise ValueError("selector bounds must satisfy 0 <= min <= max <= 4")
    if config.get("field_candidate_limit", 32) > 32:
        raise ValueError("field_candidate_limit must be <= 32")
    if config.get("selector_catalog") == "retrieved_fields" and config["candidate_limit"] > 8:
        raise ValueError("retrieved_fields requires candidate_limit <= 8 source candidates")
    if config.get("selector_catalog") == "retrieved_fields" and config["U"] != "field":
        raise ValueError("retrieved_fields requires U=field before starting the actor")
    protocol = config.get("selection_protocol", "legacy")
    if protocol == "evidence_sets_v1":
        if config["D"] != "candidate_rule" or "selector" in config:
            raise ValueError("evidence_sets_v1 uses one set_selector; D must be candidate_rule without legacy selector")
        if config["B"] != "source":
            raise ValueError("evidence_sets_v1 currently requires B=source; each action names all appended units")
        if type(config["K"]) is not int:
            raise ValueError("evidence_sets_v1 requires integer K as the final selection limit")
        if config["Q"] not in {"lexical", "archive_rrf"}:
            raise ValueError("evidence_sets_v1 Q must be lexical or archive_rrf")
        if config["candidate_limit"] > 8:
            raise ValueError("evidence_sets_v1 supports at most eight candidates")
        for key in ("retrieval_limit", "retrieval_route_limit"):
            if type(config.get(key, EXTENSION_DEFAULTS[key])) is not int or not 1 <= config.get(key, EXTENSION_DEFAULTS[key]) <= 24:
                raise ValueError(f"{key} must be between 1 and 24")
        if config.get("fallback_unit") not in {None, "tokens_256"}:
            raise ValueError("fallback_unit must be null or tokens_256")
        if config.get("fallback_unit") and config["U"] != "tokens_1024":
            raise ValueError("static tokens_256 fallback requires U=tokens_1024")
        for key in ("selector_threshold", "gain_delta"):
            value = config.get(key, EXTENSION_DEFAULTS[key])
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{key} must be finite and between zero and one")
        if config.get("local_models") is not None and not isinstance(config["local_models"], Mapping):
            raise ValueError("local_models must be an object")
        if config.get("set_selector") in {"risk", "gain_turn", "gain_task"} and not config.get("selector_artifact"):
            raise ValueError("trained set_selector requires a selector_artifact")
        if config.get("selector_artifact") and config.get("set_selector") not in {"risk", "gain_turn", "gain_task", "reranker"}:
            raise ValueError("selector_artifact is only used by a trained selector or T01 reranker calibration")
        if config.get("set_selector") == "legacy_prefill" and config.get("detector_threshold") is not None:
            raise ValueError("legacy_prefill uses the frozen base detector threshold")
    elif config["Q"] == "archive_rrf" or config.get("recovery_reserve_tokens", 0):
        raise ValueError("archive_rrf and recovery reserve require evidence_sets_v1")
    reserve = config.get("recovery_reserve_tokens", 0)
    if type(reserve) is not int or not 0 <= reserve <= 512:
        raise ValueError("recovery_reserve_tokens must be between zero and 512")
    if type(config.get("export_selection_state", False)) is not bool:
        raise ValueError("export_selection_state must be boolean")
    detector_threshold = config.get("detector_threshold")
    if "detector_calibration_telemetry" in config and type(config["detector_calibration_telemetry"]) is not bool:
        raise ValueError("detector_calibration_telemetry must be a boolean")
    if detector_threshold is not None and (
        isinstance(detector_threshold, bool) or not isinstance(detector_threshold, (int, float))
        or not math.isfinite(detector_threshold)
    ):
        raise ValueError("detector_threshold must be finite or null")
    if config["backend"] is not None:
        from .selection_backends import OpenAICompatibleSelectionBackend

        if not isinstance(config["backend"], Mapping):
            raise TypeError("backend must be an object or null")
        OpenAICompatibleSelectionBackend(config["backend"])
    return config


def configure_controller(controller_config: Mapping, switches: Mapping) -> dict:
    """Return a D3 controller config with an explicit experiment overlay."""
    result = copy.deepcopy(dict(controller_config))
    if "post_draft_recovery" not in result:
        raise ValueError("G--P requires the current post_draft_recovery detector config")
    result[GP_CONFIG_KEY] = parse_gp_config(switches)
    return result
