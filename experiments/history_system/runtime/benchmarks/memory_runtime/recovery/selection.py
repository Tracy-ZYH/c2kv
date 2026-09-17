"""Configurable, bounded evidence selection for post-draft recovery."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..policy import (
    _anchors as _current_anchors,
    _argument_strings as _current_argument_strings,
    _tokens as _current_tokens,
)

from .selection_backends import (
    SelectionBackendError,
    call_chat,
    call_embed,
    make_backends,
    public_backend_config,
    required_backend_capabilities,
)


SELECTION_SCHEMA = "recovery-evidence-selection-v1"
SCORER_SCHEMA = "recovery-candidate-linear-scorer-v1"
SCORER_LABELS = frozenset({"candidate_relevance", "intervention_value"})
SCORER_FEATURES = (
    "goal_jaccard",
    "draft_jaccard",
    "combined_jaccard",
    "overlap_log1p",
    "exact_anchor_fraction",
    "tool_name_match",
    "is_tool_or_record",
)


def prepare_selection_dependencies(
    config: Mapping[str, Any], backends: Any = None
) -> Any:
    """Validate dependencies once when constructing a runtime controller."""

    _validate_selection_config(config)
    resolved = make_backends(config, backends)
    if config.get("selector") == "supervised" or (
        config.get("selector") is None and config.get("D") == "supervised"
    ):
        load_candidate_scorer(config.get("candidate_scorer"))
    return resolved


def select_candidates(
    catalog: Sequence[Any],
    *,
    goal: str,
    draft_text: str,
    draft_tool_calls: Any,
    config: Mapping[str, Any],
    backends: Any = None,
) -> tuple[list[Any], dict[str, Any]]:
    """Return original catalog objects and an auditable content-free receipt.

    LLMs may rewrite a bounded query or choose IDs from a bounded catalog.  No
    backend response can create or modify an evidence unit.
    """

    _validate_selection_config(config)
    if not isinstance(goal, str):
        raise TypeError("goal must be a string")
    if not isinstance(draft_text, str):
        raise TypeError("draft_text must be a string")
    units = list(catalog)
    records = _catalog_records(units)
    resolved = make_backends(config, backends)
    backend_config = _backend_options(config)
    backend_calls: list[dict[str, Any]] = []
    query = _query_context(goal, draft_text, draft_tool_calls)
    selector_mode = config.get("selector")
    selector_catalog = config.get("selector_catalog", "units")
    selector_min = config.get("selector_min_units", 1)
    default_selector_max = (
        config["K"]
        if isinstance(config["K"], int) and not isinstance(config["K"], bool)
        else 1
    )
    selector_max = config.get("selector_max_units", default_selector_max)

    receipt: dict[str, Any] = {
        "schema": SELECTION_SCHEMA,
        "query_mode": config["Q"],
        "count_policy": config["K"],
        "controller_mode": config["D"],
        "selector_mode": selector_mode,
        "selector_catalog": selector_catalog,
        "selector_min_units": selector_min,
        "selector_max_units": selector_max,
        "catalog_count": len(records),
        "candidate_limit": config["candidate_limit"],
        "bounded_catalog_ids": [],
        "ranked_candidates": [],
        "selected_ids": [],
        "selection_threshold": float(config["selection_threshold"]),
        "effective_threshold": None,
        "backend": {
            "required_capabilities": sorted(required_backend_capabilities(config)),
            "config": public_backend_config(resolved, config),
            "calls": backend_calls,
        },
        "candidate_scorer": None,
        "goal_serialized_in_receipt": False,
        "draft_serialized_in_receipt": False,
        "candidate_text_serialized_in_receipt": False,
        "selected_ids_validated_against_catalog": True,
        "returns_original_catalog_objects": True,
        "uses_gold_future_or_tool_result": False,
    }
    if not records:
        receipt["reason"] = "empty_catalog"
        return [], receipt

    if config["Q"] == "llm_rewrite":
        query = _rewrite_query(
            query, resolved, backend_config, backend_calls
        )

    supervised = selector_mode == "supervised" or (
        selector_mode is None and config["D"] == "supervised"
    )
    if supervised:
        scorer = load_candidate_scorer(config.get("candidate_scorer"))
        ranked = _rank_supervised(records, query, scorer)
        receipt["candidate_scorer"] = {
            "schema": scorer["schema"],
            "label_kind": scorer["label_kind"],
            "artifact_sha256": scorer["artifact_sha256"],
            "feature_schema": list(scorer["feature_schema"]),
            "dataset_provenance": scorer["dataset_provenance"],
            "unknown_labels_excluded": scorer["training_summary"][
                "unknown_labels_excluded"
            ],
        }
    else:
        ranked = _rank_query(records, query, config, resolved, backend_config, backend_calls)

    if selector_catalog == "retrieved_fields":
        bounded, source_ids = _retrieved_field_catalog(ranked, config)
        receipt["retrieved_source_ids"] = source_ids
        receipt["retrieved_source_count"] = len(source_ids)
        receipt["field_candidate_limit"] = config.get("field_candidate_limit", 32)
    else:
        bounded = ranked[: config["candidate_limit"]]
    receipt["bounded_catalog_ids"] = [row["unit_id"] for row in bounded]
    receipt["ranked_candidates"] = [
        {
            "unit_id": row["unit_id"],
            "score": row["score"],
            "components": dict(row["components"]),
        }
        for row in bounded
    ]
    receipt["ranked_candidate_count"] = len(ranked)
    receipt["ranked_candidates_omitted"] = max(0, len(ranked) - len(bounded))

    if selector_mode == "llm":
        try:
            chosen_ids = _llm_choose_ids(
                bounded,
                query,
                resolved,
                backend_config,
                backend_calls,
                allow_empty=selector_min == 0,
                minimum=selector_min,
                maximum=selector_max,
                purpose="recovery_evidence_selection",
            )
            receipt["selector_fallback"] = {"applied": False, "reason": None}
        except SelectionBackendError:
            fallback_id = _lexical_top_id(bounded, query)
            chosen_ids = [fallback_id] if fallback_id is not None else []
            receipt["selector_fallback"] = {
                "applied": fallback_id is not None,
                "reason": "invalid_or_malformed_selector_output",
                "policy": "lexical_top_1",
            }
        effective_threshold = None
    elif selector_mode == "fixed":
        chosen_ids = [row["unit_id"] for row in bounded[:selector_max]]
        effective_threshold = None
    elif selector_mode == "supervised":
        label_kind = scorer["label_kind"]
        if label_kind == "candidate_relevance":
            chosen_ids = [row["unit_id"] for row in bounded[:selector_max]]
            effective_threshold = None
            receipt["candidate_scorer"]["selection_semantics"] = "ranking_only"
            receipt["candidate_scorer"]["empty_set_allowed"] = False
        else:
            effective_threshold = max(0.0, float(config["selection_threshold"]))
            chosen_ids = [
                row["unit_id"]
                for row in bounded
                if row["score"] > effective_threshold
            ][:selector_max]
            receipt["candidate_scorer"]["selection_semantics"] = (
                "positive_intervention_value"
            )
            receipt["candidate_scorer"]["empty_set_allowed"] = True
    else:
        llm_mode = config["K"] == "llm" or config["D"] in {
            "detector_llm",
            "joint_llm",
        }
        if llm_mode:
            chosen_ids = _llm_choose_ids(
                bounded,
                query,
                resolved,
                backend_config,
                backend_calls,
                allow_empty=config["D"] == "joint_llm",
                purpose=(
                    "joint_recovery_decision"
                    if config["D"] == "joint_llm"
                    else "recovery_candidate_selection"
                ),
            )
            effective_threshold = None
        else:
            eligible = bounded
            effective_threshold: float | None = None
            if config["D"] == "supervised" or config["K"] == "threshold":
                effective_threshold = float(config["selection_threshold"])
                eligible = [
                    row for row in eligible if row["score"] >= effective_threshold
                ]
            if isinstance(config["K"], int) and not isinstance(config["K"], bool):
                eligible = eligible[: config["K"]]
            chosen_ids = [row["unit_id"] for row in eligible]

    by_id = {record["unit_id"]: record for record in records}
    _validate_selected_ids(chosen_ids, set(by_id), allow_empty=True)
    selected_records = [by_id[unit_id] for unit_id in chosen_ids]
    if config.get("order") == "chronological":
        selected_records.sort(
            key=lambda record: (record["source_order"], record["unit_id"])
        )
    else:
        relevance_position = {
            row["unit_id"]: index for index, row in enumerate(ranked)
        }
        selected_records.sort(
            key=lambda record: relevance_position[record["unit_id"]]
        )
    receipt["selected_ids"] = [record["unit_id"] for record in selected_records]
    receipt["effective_threshold"] = effective_threshold
    receipt["backend"]["call_count"] = len(backend_calls)
    receipt["reason"] = (
        "selected_catalog_units" if selected_records else "selector_abstained"
    )
    return [record["unit"] for record in selected_records], receipt


def train_candidate_scorer(
    examples: Sequence[Mapping[str, Any]],
    *,
    label_kind: str,
    dataset_provenance: Mapping[str, Any],
    l2: float = 1.0,
) -> dict[str, Any]:
    """Fit a small deterministic linear scorer from allowed candidate labels.

    Each known row must contain ``goal``, ``draft_text``, ``candidate``,
    ``label``, and nonempty ``provenance``.  Rows whose label is absent or whose
    ``label_status`` is ``unknown`` are explicitly excluded.
    """

    if label_kind not in SCORER_LABELS:
        raise ValueError(
            "candidate scorer label_kind must be candidate_relevance or "
            "intervention_value; risk labels are forbidden"
        )
    if not isinstance(dataset_provenance, Mapping) or not dataset_provenance:
        raise ValueError("dataset_provenance must be a nonempty mapping")
    if (
        isinstance(l2, bool)
        or not isinstance(l2, (int, float))
        or not math.isfinite(float(l2))
        or l2 < 0
    ):
        raise ValueError("l2 must be finite and nonnegative")
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - declared runtime dependency
        raise RuntimeError("numpy is required to train a candidate scorer") from error

    feature_rows: list[list[float]] = []
    labels: list[float] = []
    excluded_unknown = 0
    for index, example in enumerate(examples):
        if not isinstance(example, Mapping):
            raise TypeError(f"training example {index} must be a mapping")
        row_kind = example.get("label_kind", label_kind)
        if row_kind not in SCORER_LABELS:
            raise ValueError(f"training example {index} uses a forbidden label kind")
        if row_kind != label_kind:
            raise ValueError(f"training example {index} label_kind differs from training target")
        if _is_unknown_label(example):
            excluded_unknown += 1
            continue
        if not _has_provenance(example.get("provenance")):
            raise ValueError(f"known training example {index} lacks provenance")
        goal = example.get("goal", "")
        draft_text = example.get("draft_text", "")
        if not isinstance(goal, str) or not isinstance(draft_text, str):
            raise TypeError(f"training example {index} goal and draft_text must be strings")
        candidate = example.get("candidate")
        record = _candidate_for_training(candidate, index)
        query = _query_context(
            goal, draft_text, example.get("draft_tool_calls", [])
        )
        label = example["label"]
        if (
            isinstance(label, bool)
            or not isinstance(label, (int, float))
            or not math.isfinite(float(label))
        ):
            raise ValueError(f"training example {index} label must be finite")
        if label_kind == "candidate_relevance" and not 0 <= float(label) <= 1:
            raise ValueError("candidate_relevance labels must be from zero to one")
        feature_rows.append(_scorer_features(query, record))
        labels.append(float(label))
    if len(feature_rows) < 2:
        raise ValueError("candidate scorer requires at least two known labeled rows")

    matrix = np.asarray(feature_rows, dtype=np.float64)
    target = np.asarray(labels, dtype=np.float64)
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[scales == 0] = 1.0
    normalized = (matrix - means) / scales
    design = np.column_stack([normalized, np.ones(len(normalized))])
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(l2)
    penalty[-1, -1] = 0.0
    parameters = np.linalg.pinv(design.T @ design + penalty) @ design.T @ target
    artifact: dict[str, Any] = {
        "schema": SCORER_SCHEMA,
        "label_kind": label_kind,
        "feature_schema": list(SCORER_FEATURES),
        "weights": [float(value) for value in parameters[:-1]],
        "bias": float(parameters[-1]),
        "input_mean": [float(value) for value in means],
        "input_scale": [float(value) for value in scales],
        "score_transform": "sigmoid" if label_kind == "candidate_relevance" else "identity",
        "dataset_provenance": json.loads(
            json.dumps(dataset_provenance, ensure_ascii=False, allow_nan=False)
        ),
        "training_summary": {
            "known_labels_used": len(feature_rows),
            "unknown_labels_excluded": excluded_unknown,
            "label_source": label_kind,
            "risk_labels_used": False,
            "l2": float(l2),
        },
    }
    artifact["artifact_sha256"] = _artifact_hash(artifact)
    return load_candidate_scorer(artifact)


def export_candidate_scorer(
    artifact: Mapping[str, Any], path: str | Path
) -> Path:
    """Validate and export a scorer as a self-contained JSON artifact."""

    checked = load_candidate_scorer(artifact)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(checked, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def load_candidate_scorer(value: Any) -> dict[str, Any]:
    """Load and validate an embedded scorer config or exported JSON path."""

    if isinstance(value, (str, Path)):
        try:
            value = json.loads(Path(value).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("candidate_scorer path is unreadable or invalid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError("supervised selection requires candidate_scorer config or path")
    required = {
        "schema",
        "label_kind",
        "feature_schema",
        "weights",
        "bias",
        "input_mean",
        "input_scale",
        "score_transform",
        "dataset_provenance",
        "training_summary",
        "artifact_sha256",
    }
    if set(value) != required:
        raise ValueError("candidate_scorer fields do not match the frozen scorer schema")
    artifact = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    if artifact["schema"] != SCORER_SCHEMA:
        raise ValueError("candidate_scorer.schema is unsupported")
    if artifact["label_kind"] not in SCORER_LABELS:
        raise ValueError("candidate_scorer label source is forbidden")
    if tuple(artifact["feature_schema"]) != SCORER_FEATURES:
        raise ValueError("candidate_scorer feature schema is unsupported")
    dimensions = len(SCORER_FEATURES)
    for name in ("weights", "input_mean", "input_scale"):
        vector = _finite_vector(artifact[name], f"candidate_scorer.{name}")
        if len(vector) != dimensions:
            raise ValueError(f"candidate_scorer.{name} has the wrong dimension")
        artifact[name] = vector
    if any(value <= 0 for value in artifact["input_scale"]):
        raise ValueError("candidate_scorer.input_scale must be positive")
    artifact["bias"] = _finite_number(artifact["bias"], "candidate_scorer.bias")
    expected_transform = (
        "sigmoid"
        if artifact["label_kind"] == "candidate_relevance"
        else "identity"
    )
    if artifact["score_transform"] != expected_transform:
        raise ValueError("candidate_scorer score_transform differs from label kind")
    if not isinstance(artifact["dataset_provenance"], Mapping) or not artifact[
        "dataset_provenance"
    ]:
        raise ValueError("candidate_scorer dataset_provenance is missing")
    summary = artifact["training_summary"]
    if not isinstance(summary, Mapping) or summary.get("risk_labels_used") is not False:
        raise ValueError("candidate_scorer training summary must exclude risk labels")
    if summary.get("label_source") != artifact["label_kind"]:
        raise ValueError("candidate_scorer training label provenance is inconsistent")
    for name, minimum in (("known_labels_used", 2), ("unknown_labels_excluded", 0)):
        count = summary.get(name)
        if type(count) is not int or count < minimum:
            raise ValueError(f"candidate_scorer training_summary.{name} is invalid")
    l2 = summary.get("l2")
    if _finite_number(l2, "candidate_scorer.training_summary.l2") < 0:
        raise ValueError("candidate_scorer training_summary.l2 must be nonnegative")
    digest = artifact["artifact_sha256"]
    if not isinstance(digest, str) or digest != _artifact_hash(
        {key: val for key, val in artifact.items() if key != "artifact_sha256"}
    ):
        raise ValueError("candidate_scorer artifact_sha256 mismatch")
    return artifact


def _rank_query(
    records: list[dict[str, Any]],
    query: Mapping[str, Any],
    config: Mapping[str, Any],
    backends: Any,
    backend_config: Mapping[str, Any],
    backend_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    mode = config["Q"]
    lexical_rows = []
    for record in records:
        if mode == "dual":
            goal_score, goal_parts = _lexical_score(query["goal"], record)
            draft_score, draft_parts = _lexical_score(
                query["draft"], record, exact_anchors=query["exact_anchors"]
            )
            score = goal_score + draft_score
            parts = {
                "goal_lexical": goal_score,
                "draft_lexical": draft_score,
                "goal_exact_hits": goal_parts["exact_hits"],
                "draft_exact_hits": draft_parts["exact_hits"],
            }
        else:
            score, parts = _lexical_score(
                query["combined"], record, exact_anchors=query["exact_anchors"]
            )
        lexical_rows.append({**record, "score": score, "components": parts})

    if mode == "hybrid":
        inputs = [query["combined"]] + [record["search_text"] for record in records]
        vectors = call_embed(
            backends,
            texts=inputs,
            purpose="hybrid_query_and_catalog",
            config=backend_config,
        )
        vectors = _validate_embeddings(vectors, len(inputs))
        _record_backend_call(
            backends,
            backend_calls,
            capability="embed",
            purpose="hybrid_query_and_catalog",
            input_count=len(inputs),
        )
        semantic_scores = {
            row["unit_id"]: (_cosine(vectors[0], vector) + 1.0) / 2.0
            for row, vector in zip(lexical_rows, vectors[1:])
        }
        fusion = config.get("hybrid_fusion", "weighted")
        if fusion == "rrf":
            rrf_k = config.get("rrf_k", 60)
            lexical_order = _sort_ranked(list(lexical_rows))
            semantic_order = sorted(
                lexical_rows,
                key=lambda row: (
                    -semantic_scores[row["unit_id"]],
                    -row["source_recency"],
                    row["input_order"],
                    row["unit_id"],
                ),
            )
            lexical_rank = {
                row["unit_id"]: index
                for index, row in enumerate(lexical_order, start=1)
            }
            semantic_rank = {
                row["unit_id"]: index
                for index, row in enumerate(semantic_order, start=1)
            }
            for row in lexical_rows:
                lr = lexical_rank[row["unit_id"]]
                sr = semantic_rank[row["unit_id"]]
                row["components"] = {
                    **row["components"],
                    "semantic_cosine_normalized": semantic_scores[row["unit_id"]],
                    "hybrid_fusion": "rrf",
                    "rrf_k": rrf_k,
                    "lexical_rank": lr,
                    "semantic_rank": sr,
                }
                row["score"] = 1.0 / (rrf_k + lr) + 1.0 / (rrf_k + sr)
        else:
            lexical_weight = float(backend_config.get("hybrid_lexical_weight", 0.5))
            for row in lexical_rows:
                semantic = semantic_scores[row["unit_id"]]
                lexical_normalized = row["score"] / (row["score"] + 20.0)
                row["components"] = {
                    **row["components"],
                    "lexical_normalized": lexical_normalized,
                    "semantic_cosine_normalized": semantic,
                    "hybrid_lexical_weight": lexical_weight,
                    "hybrid_fusion": "weighted",
                }
                row["score"] = lexical_weight * lexical_normalized + (
                    1.0 - lexical_weight
                ) * semantic

    include_zero = mode == "hybrid" or config.get("selector") == "llm" or (
        config.get("selector") is None and (config["K"] == "llm" or config["D"] in {
            "detector_llm", "joint_llm"}))
    if not include_zero:
        lexical_rows = [row for row in lexical_rows if row["score"] > 0]
    return _sort_ranked(lexical_rows)


def _rank_supervised(
    records: list[dict[str, Any]], query: Mapping[str, Any], scorer: Mapping[str, Any]
) -> list[dict[str, Any]]:
    ranked = []
    for record in records:
        features = _scorer_features(query, record)
        normalized = [
            (value - mean) / scale
            for value, mean, scale in zip(
                features, scorer["input_mean"], scorer["input_scale"]
            )
        ]
        raw = scorer["bias"] + sum(
            weight * value for weight, value in zip(scorer["weights"], normalized)
        )
        score = _sigmoid(raw) if scorer["score_transform"] == "sigmoid" else raw
        ranked.append(
            {
                **record,
                "score": score,
                "components": {
                    "frozen_scorer_raw": raw,
                    "frozen_scorer_output": score,
                },
            }
        )
    return _sort_ranked(ranked)


def _rewrite_query(
    query: Mapping[str, Any],
    backends: Any,
    backend_config: Mapping[str, Any],
    backend_calls: list[dict[str, Any]],
) -> dict[str, str]:
    maximum = int(backend_config.get("query_max_chars", 4096))
    prompt = {
        "goal": query["goal"][:maximum],
        "held_draft": query["draft"][:maximum],
        "instruction": (
            "Return JSON only as {\"query\": \"...\"}. Rewrite the observable "
            "goal and held draft into one retrieval query. Do not invent facts."
        ),
    }
    response = call_chat(
        backends,
        messages=[
            {"role": "system", "content": "You produce bounded retrieval queries."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        purpose="query_rewrite",
        config=backend_config,
    )
    _record_backend_call(
        backends,
        backend_calls,
        capability="chat",
        purpose="query_rewrite",
        catalog_count=0,
    )
    value = _json_object(response, "query rewrite")
    rewritten = value.get("query")
    if not isinstance(rewritten, str) or not rewritten.strip():
        raise SelectionBackendError("query rewrite must return a nonempty query string")
    rewritten = rewritten.strip()[:maximum]
    return {**query, "combined": rewritten, "exact_anchors": tuple(_anchors(rewritten))}


def _llm_choose_ids(
    bounded: list[dict[str, Any]],
    query: Mapping[str, Any],
    backends: Any,
    backend_config: Mapping[str, Any],
    backend_calls: list[dict[str, Any]],
    *,
    allow_empty: bool,
    purpose: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> list[str]:
    if not bounded:
        return []
    text_limit = int(backend_config.get("candidate_text_max_chars", 2048))
    query_limit = int(backend_config.get("query_max_chars", 4096))
    catalog = [
        {
            "unit_id": row["unit_id"],
            "source_type": row["source_type"],
            "text": row["search_text"][:text_limit],
            "context": _bounded_context(row["context"], text_limit),
        }
        for row in bounded
    ]
    payload = {
        "goal": query["goal"][:query_limit],
        "held_draft": query["draft"][:query_limit],
        "catalog": catalog,
        "instruction": (
            "Return JSON only as {\"selected_ids\": [...]}. Use each listed "
            "unit_id at most once and never return an ID outside catalog."
            + (" Return an empty list when recovery is not useful." if allow_empty else "")
        ),
    }
    if minimum is not None:
        payload["minimum_selected_ids"] = minimum
    if maximum is not None:
        payload["maximum_selected_ids"] = maximum
    response = call_chat(
        backends,
        messages=[
            {"role": "system", "content": "You select evidence only by catalog ID."},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        purpose=purpose,
        config=backend_config,
    )
    _record_backend_call(
        backends,
        backend_calls,
        capability="chat",
        purpose=purpose,
        catalog_count=len(catalog),
    )
    value = _json_object(response, "candidate selection")
    selected = value.get("selected_ids")
    if not isinstance(selected, list) or any(not isinstance(item, str) for item in selected):
        raise SelectionBackendError("candidate selection must return selected_ids strings")
    _validate_selected_ids(
        selected, {row["unit_id"] for row in bounded}, allow_empty=True
    )
    if minimum is not None and len(selected) < minimum:
        raise SelectionBackendError("candidate selection returned too few IDs")
    if maximum is not None and len(selected) > maximum:
        raise SelectionBackendError("candidate selection returned too many IDs")
    return selected


def _retrieved_field_catalog(
    ranked: list[dict[str, Any]], config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Bound a field directory to already-retrieved observable sources."""

    non_fields = [row["unit_id"] for row in ranked if row["source_type"] != "field"]
    if non_fields:
        raise ValueError("selector_catalog='retrieved_fields' requires field evidence units")
    source_limit = config["candidate_limit"]
    source_ids: list[str] = []
    for row in ranked:
        event_id = row.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("retrieved field candidate lacks event_id")
        if event_id not in source_ids:
            source_ids.append(event_id)
            if len(source_ids) == source_limit:
                break
    allowed = set(source_ids)
    fields = [row for row in ranked if row["event_id"] in allowed]
    return fields[: config.get("field_candidate_limit", 32)], source_ids


def _lexical_top_id(
    bounded: Sequence[Mapping[str, Any]], query: Mapping[str, Any]
) -> str | None:
    """Return the frozen lexical top-1 fallback from the bounded legal catalog."""

    rows: list[dict[str, Any]] = []
    for record in bounded:
        score, components = _lexical_score(
            query["combined"], record, exact_anchors=query["exact_anchors"]
        )
        rows.append({**record, "score": score, "components": components})
    ranked = _sort_ranked(rows)
    return ranked[0]["unit_id"] if ranked else None


def _catalog_records(units: list[Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, unit in enumerate(units):
        unit_id = _field(unit, "unit_id")
        text = _field(unit, "text")
        source_type = _field(unit, "source_type", default="unknown")
        source_indices = _field(unit, "source_indices", default=())
        if not isinstance(unit_id, str) or not unit_id:
            raise ValueError(f"catalog unit {index} has no nonempty unit_id")
        if unit_id in seen:
            raise ValueError(f"catalog contains duplicate unit_id {unit_id!r}")
        if not isinstance(text, str):
            raise TypeError(f"catalog unit {unit_id!r} text must be a string")
        if not isinstance(source_type, str):
            raise TypeError(f"catalog unit {unit_id!r} source_type must be a string")
        if not isinstance(source_indices, Sequence) or isinstance(
            source_indices, (str, bytes)
        ):
            raise TypeError(f"catalog unit {unit_id!r} source_indices must be a sequence")
        indices = tuple(source_indices)
        if any(type(value) is not int or value < 0 for value in indices):
            raise ValueError(f"catalog unit {unit_id!r} source_indices are invalid")
        seen.add(unit_id)
        records.append(
            {
                "unit": unit,
                "unit_id": unit_id,
                "text": text,
                "source_type": source_type,
                "source_order": min(indices) if indices else index,
                "source_recency": max(indices) if indices else index,
                "input_order": index,
                "event_id": _field(unit, "event_id", default=None),
                "association_id": _field(unit, "association_id", default=None),
                "metadata": _field(unit, "metadata", default={}),
            }
        )
        records[-1]["context"] = _readable_context(records[-1])
        search_source = (
            _event_unit_search_text(records[-1]["text"])
            if source_type == "event"
            else records[-1]["text"]
        )
        records[-1]["search_text"] = _search_text(
            search_source, records[-1]["context"]
        )
    return records


def _candidate_for_training(value: Any, index: int) -> dict[str, Any]:
    if isinstance(value, str):
        return {
            "unit_id": f"training-{index}",
            "text": value,
            "source_type": "unknown",
            "metadata": {},
        }
    if value is None:
        raise ValueError(f"training example {index} lacks candidate")
    unit_id = _field(value, "unit_id", default=f"training-{index}")
    text = _field(value, "text")
    source_type = _field(value, "source_type", default="unknown")
    if not isinstance(text, str):
        raise TypeError(f"training example {index} candidate text must be a string")
    record = {
        "unit_id": str(unit_id),
        "text": text,
        "source_type": str(source_type),
        "event_id": _field(value, "event_id", default=None),
        "association_id": _field(value, "association_id", default=None),
        "metadata": _field(value, "metadata", default={}),
    }
    record["context"] = _readable_context(record)
    search_source = (
        _event_unit_search_text(record["text"])
        if record["source_type"] == "event"
        else record["text"]
    )
    record["search_text"] = _search_text(search_source, record["context"])
    return record


def _query_context(goal: str, draft_text: str, draft_tool_calls: Any) -> dict[str, Any]:
    tool_parts: list[str] = []
    exact_parts: list[str] = []
    for call in _valid_tool_calls(draft_tool_calls):
        function = call["function"]
        tool_parts.append(function["name"])
        arguments = _argument_strings(function["arguments"])
        tool_parts.extend(arguments)
        exact_parts.extend(arguments)
    draft = "\n".join(part for part in [draft_text, *tool_parts] if part)
    combined = "\n".join(part for part in [goal, draft] if part)
    exact_anchors = {
        normalized
        for part in exact_parts
        if len(normalized := " ".join(part.casefold().split())) >= 2
    }
    exact_anchors.update(_anchors(combined))
    return {
        "goal": goal,
        "draft": draft,
        "combined": combined,
        "exact_anchors": tuple(sorted(exact_anchors)),
    }


def _lexical_score(
    query: str,
    record: Mapping[str, Any],
    *,
    exact_anchors: Sequence[str] = (),
) -> tuple[float, dict[str, Any]]:
    query_tokens = _tokens(query)
    candidate_text = record.get("search_text", record["text"])
    candidate_tokens = _tokens(candidate_text)
    overlap = query_tokens & candidate_tokens
    union = query_tokens | candidate_tokens
    anchors = _anchors(query) | set(exact_anchors)
    lower = candidate_text.casefold()
    exact_hits = sum(1 for anchor in anchors if anchor in lower)
    jaccard = len(overlap) / len(union) if union else 0.0
    kind_bonus = 8.0 if _is_tool_source(record) else 0.0
    if not overlap and not exact_hits:
        kind_bonus = 0.0
    score = exact_hits * 100.0 + len(overlap) * 4.0 + jaccard + kind_bonus
    return score, {
        "exact_hits": exact_hits,
        "overlap_count": len(overlap),
        "jaccard": jaccard,
        "source_type_bonus": kind_bonus,
    }


def _scorer_features(query: Mapping[str, Any], record: Mapping[str, Any]) -> list[float]:
    goal_tokens = _tokens(query["goal"])
    draft_tokens = _tokens(query["draft"])
    combined_tokens = goal_tokens | draft_tokens
    candidate_text = record.get("search_text", record["text"])
    candidate_tokens = _tokens(candidate_text)

    def jaccard(left: set[str], right: set[str]) -> float:
        union = left | right
        return len(left & right) / len(union) if union else 0.0

    overlap = combined_tokens & candidate_tokens
    anchors = _anchors(query["combined"]) | set(query.get("exact_anchors", ()))
    exact = sum(1 for anchor in anchors if anchor in candidate_text.casefold())
    tool_names = {
        token for token in _tokens(query["draft"]) if "." in token or "_" in token
    }
    return [
        jaccard(goal_tokens, candidate_tokens),
        jaccard(draft_tokens, candidate_tokens),
        jaccard(combined_tokens, candidate_tokens),
        math.log1p(len(overlap)),
        exact / len(anchors) if anchors else 0.0,
        1.0 if tool_names & candidate_tokens else 0.0,
        1.0 if _is_tool_source(record) or record["source_type"] == "record" else 0.0,
    ]


def _sort_ranked(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -row["score"], -row["source_recency"], row["unit_id"]
        ),
    )


def _validate_selected_ids(
    selected: Sequence[str], catalog_ids: set[str], *, allow_empty: bool
) -> None:
    if not selected and not allow_empty:
        raise SelectionBackendError("candidate selection returned an empty ID set")
    if len(selected) != len(set(selected)):
        raise SelectionBackendError("candidate selection returned duplicate IDs")
    unknown = sorted(set(selected) - catalog_ids)
    if unknown:
        raise SelectionBackendError(
            f"candidate selection returned IDs outside bounded catalog: {unknown!r}"
        )


def _validate_selection_config(config: Mapping[str, Any]) -> None:
    if not isinstance(config, Mapping):
        raise TypeError("selection config must be a mapping")
    for name in ("Q", "K", "D", "candidate_limit", "selection_threshold"):
        if name not in config:
            raise ValueError(f"selection config lacks {name}")
    if config["Q"] not in {"lexical", "dual", "hybrid", "llm_rewrite"}:
        raise ValueError("unsupported Q selection mode")
    if isinstance(config["K"], bool) or config["K"] not in {1, 2, 4, "threshold", "llm"}:
        raise ValueError("unsupported K selection policy")
    if config["D"] not in {
        "detector",
        "candidate_rule",
        "candidate_or_detector",
        "detector_llm",
        "joint_llm",
        "supervised",
    }:
        raise ValueError("unsupported D controller mode")
    if type(config["candidate_limit"]) is not int or config["candidate_limit"] < 1:
        raise ValueError("candidate_limit must be a positive integer")
    _finite_number(config["selection_threshold"], "selection_threshold")
    selector = config.get("selector")
    if selector is not None and selector not in {"fixed", "llm", "supervised"}:
        raise ValueError("unsupported selector mode")
    selector_catalog = config.get("selector_catalog", "units")
    if selector_catalog not in {"units", "retrieved_fields"}:
        raise ValueError("unsupported selector catalog")
    if selector_catalog == "retrieved_fields" and config["candidate_limit"] > 8:
        raise ValueError("retrieved_fields supports at most eight source candidates")
    minimum = config.get("selector_min_units", 1)
    default_maximum = (
        config["K"]
        if isinstance(config["K"], int) and not isinstance(config["K"], bool)
        else 1
    )
    maximum = config.get("selector_max_units", default_maximum)
    if type(minimum) is not int or type(maximum) is not int:
        raise ValueError("selector unit bounds must be integers")
    if not 0 <= minimum <= maximum <= 4:
        raise ValueError("selector unit bounds must satisfy 0 <= min <= max <= 4")
    field_limit = config.get("field_candidate_limit", 32)
    if type(field_limit) is not int or not 1 <= field_limit <= 32:
        raise ValueError("field_candidate_limit must be an integer from 1 to 32")
    fusion = config.get("hybrid_fusion", "weighted")
    if fusion not in {"weighted", "rrf"}:
        raise ValueError("hybrid_fusion must be weighted or rrf")
    rrf_k = config.get("rrf_k", 60)
    if type(rrf_k) is not int or rrf_k <= 0:
        raise ValueError("rrf_k must be a positive integer")


def _backend_options(config: Mapping[str, Any]) -> dict[str, Any]:
    backend = config.get("backend")
    result = dict(backend) if isinstance(backend, Mapping) else {}
    result.setdefault("query_max_chars", 4096)
    result.setdefault("candidate_text_max_chars", 2048)
    result.setdefault("hybrid_lexical_weight", 0.5)
    return result


def _record_backend_call(
    backends: Any,
    receipts: list[dict[str, Any]],
    *,
    capability: str,
    purpose: str,
    **counts: Any,
) -> None:
    drain = getattr(backends, "drain_receipts", None)
    if callable(drain):
        observed = drain()
        if observed:
            for row in observed:
                receipts.append({**dict(row), **counts})
            return
    receipts.append(
        {
            "capability": capability,
            "purpose": purpose,
            **counts,
            "latency_seconds": None,
            "usage": {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            },
        }
    )


def _field(value: Any, name: str, *, default: Any = ...) -> Any:
    if isinstance(value, Mapping):
        if name in value:
            return value[name]
    elif hasattr(value, name):
        return getattr(value, name)
    if default is not ...:
        return default
    raise ValueError(f"candidate lacks {name}")


def _readable_context(record: Mapping[str, Any]) -> dict[str, Any]:
    """Expose source labels needed to retrieve opaque scalar field values."""

    if record.get("source_type") not in {"record", "field"}:
        return {}
    metadata = record.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    result: dict[str, Any] = {}
    for name in ("field_name", "tool_name", "container_kind", "role"):
        value = metadata.get(name)
        if isinstance(value, str) and value:
            result[name] = value
    for name in ("field_path", "object_path", "record_path"):
        value = metadata.get(name)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            parts = [str(part) for part in value]
            if parts:
                result[name] = parts
    return result


def _search_text(text: str, context: Mapping[str, Any]) -> str:
    labels: list[str] = []
    for value in context.values():
        if isinstance(value, str):
            labels.append(value)
        elif isinstance(value, Sequence):
            labels.append(".".join(str(part) for part in value))
            labels.extend(str(part) for part in value)
    return "\n".join([text, *labels])


def _event_unit_search_text(text: str) -> str:
    """Match policy._event_text while keeping the EvidenceUnit source immutable."""

    decoder = json.JSONDecoder()
    cursor = 0
    messages: list[Mapping[str, Any]] = []
    while cursor < len(text):
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor == len(text):
            break
        try:
            value, cursor = decoder.raw_decode(text, cursor)
        except json.JSONDecodeError:
            return text
        if not isinstance(value, Mapping):
            return text
        messages.append(value)
    if not messages:
        return text
    parts: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            function = call.get("function") if isinstance(call, Mapping) else None
            if not isinstance(function, Mapping):
                continue
            name = function.get("name")
            if isinstance(name, str):
                parts.append(name)
            parts.extend(_current_argument_strings(function.get("arguments")))
    return "\n".join(parts)


def _is_tool_source(record: Mapping[str, Any]) -> bool:
    metadata = record.get("metadata")
    return record.get("source_type") == "tool_event" or (
        isinstance(metadata, Mapping) and metadata.get("event_kind") == "tool_event"
    )


def _bounded_context(value: Mapping[str, Any], maximum: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    remaining = maximum
    for key, item in value.items():
        if remaining <= 0:
            break
        if isinstance(item, str):
            bounded: Any = item[:remaining]
            remaining -= len(bounded)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            bounded = []
            for part in item:
                text = str(part)[:remaining]
                if not text:
                    break
                bounded.append(text)
                remaining -= len(text)
        else:
            continue
        result[key] = bounded
    return result


def _valid_tool_calls(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    calls: list[dict[str, Any]] = []
    for call in value:
        if not isinstance(call, Mapping):
            return []
        function = call.get("function")
        if not isinstance(function, Mapping) or not isinstance(function.get("name"), str):
            return []
        arguments = function.get("arguments")
        if isinstance(arguments, Mapping):
            arguments = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
        if not isinstance(arguments, str):
            return []
        try:
            json.loads(arguments)
        except json.JSONDecodeError:
            return []
        calls.append({"function": {"name": function["name"], "arguments": arguments}})
    return calls


def _argument_strings(value: str) -> list[str]:
    return _current_argument_strings(value)


def _tokens(value: str) -> set[str]:
    return _current_tokens(value)


def _anchors(value: str) -> set[str]:
    return _current_anchors(value)


def _json_object(value: Any, context: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str):
        raise SelectionBackendError(f"{context} response must be JSON")
    text = value.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1])
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise SelectionBackendError(f"{context} response must be valid JSON") from error
    if not isinstance(parsed, Mapping):
        raise SelectionBackendError(f"{context} response must be a JSON object")
    return parsed


def _validate_embeddings(value: Any, count: int) -> list[list[float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise SelectionBackendError("embedding backend must return a vector list")
    if len(value) != count:
        raise SelectionBackendError("embedding response count differs from input")
    vectors = [_finite_vector(row, "embedding") for row in value]
    dimension = len(vectors[0]) if vectors else 0
    if not dimension or any(len(row) != dimension for row in vectors):
        raise SelectionBackendError("embedding vectors must share one nonzero dimension")
    return vectors


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


def _finite_vector(value: Any, name: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a numeric vector")
    return [_finite_number(item, name) for item in value]


def _finite_number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _has_provenance(value: Any) -> bool:
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, str):
        return bool(value.strip())
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and bool(value)


def _is_unknown_label(example: Mapping[str, Any]) -> bool:
    label = example.get("label")
    return (
        label is None
        or (isinstance(label, str) and label.casefold() == "unknown")
        or example.get("label_status") == "unknown"
        or example.get("label_known") is False
        or example.get("known") is False
    )


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _artifact_hash(value: Mapping[str, Any]) -> str:
    payload = {key: val for key, val in value.items() if key != "artifact_sha256"}
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "SCORER_FEATURES",
    "SCORER_LABELS",
    "SCORER_SCHEMA",
    "SELECTION_SCHEMA",
    "export_candidate_scorer",
    "load_candidate_scorer",
    "prepare_selection_dependencies",
    "select_candidates",
    "train_candidate_scorer",
]
