"""Offline grouped-CV training for portable recovery-set models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .set_models import (
    ARTIFACT_SCHEMA,
    C1_FEATURE_NAMES,
    C1_FEATURE_SCHEMA,
    C4_FEATURE_NAMES,
    C4_FEATURE_SCHEMA,
    T01_FEATURE_NAMES,
    T01_FEATURE_SCHEMA,
    artifact_sha256,
    extract_c1_raw_features,
    extract_c4_features,
    extract_t01_features,
    tokenizer_contract,
)


C4_ALPHAS = (1.0, 10.0, 100.0)
C1_CS = (0.01, 0.1, 1.0)
T01_CS = (0.01, 0.1, 1.0)


def train_c4_models(
    rows: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    provenance: Mapping[str, Any],
    output_dir: str | Path | None = None,
    models: Any = None,
    score_model_contract: Mapping[str, Any] | None = None,
    semantic_query_overflow_policy: str = "error",
) -> dict[str, dict[str, Any]]:
    """Fit independent turn/task Ridge models from T02 labeled states.

    A ``t02-labeled-state-v1`` state is expanded into its A1/A2 tested actions.
    Unknown labels are excluded target-by-target and counted; they are never
    converted to zero.  Grouped folds always use ``task_group_id``.
    """

    _validate_training_inputs(rows, provenance)
    training_rows, calibration_count = _select_training_split(rows)
    if models is not None:
        training_rows, enrichment = _enrich_c4_training_rows(
            training_rows,
            models=models,
            calibration_count=calibration_count,
            semantic_query_overflow_policy=semantic_query_overflow_policy,
        )
        bound_score_models = _local_score_model_contract(models)
    else:
        bound_score_models = _json_mapping(
            score_model_contract, "score_model_contract"
        )
        enrichment = {
            "mode": "precomputed_input_features",
            "state_count": len(training_rows),
            "calibration_state_count_excluded_before_enrichment": calibration_count,
            "backend_receipts": [],
        }
    examples = _c4_examples(training_rows, tokenizer)
    bound_tokenizer = tokenizer_contract(tokenizer)
    artifacts: dict[str, dict[str, Any]] = {}
    for target, model_kind in (
        ("delta_turn", "c4_gain_turn"),
        ("delta_task", "c4_gain_task"),
    ):
        known = [example for example in examples if example[target] is not None]
        if not known:
            raise ValueError(f"no known {target} labels")
        x = np.asarray([example["features"] for example in known], dtype=float)
        y = np.asarray([example[target] for example in known], dtype=float)
        groups = np.asarray([example["task_group_id"] for example in known], dtype=object)
        _require_three_groups(groups, target)
        selected, cv = _select_ridge_alpha(x, y, groups)
        scaler = StandardScaler().fit(x)
        model = Ridge(alpha=selected).fit(scaler.transform(x), y)
        excluded = len(examples) - len(known)
        artifact = {
            "schema": ARTIFACT_SCHEMA,
            "model_kind": model_kind,
            "target": target,
            "feature_contract": {
                "schema": C4_FEATURE_SCHEMA,
                "feature_names": list(C4_FEATURE_NAMES),
                "dimension": len(C4_FEATURE_NAMES),
                "candidate_required_fields": [
                    "unit_id",
                    "event_id",
                    "text",
                    "token_count",
                    "provenance",
                    "reranker_score",
                    "task_similarity",
                    "draft_similarity",
                    "tool_name_match",
                ],
                "tokenizer_required_at_runtime": True,
                "tokenizer_contract": bound_tokenizer,
                "score_model_contract": bound_score_models,
                "semantic_query_overflow_policy": semantic_query_overflow_policy,
                "token_count_must_match_tokenizer": True,
                "raw_novelty": (
                    "selected original-text token positions whose ID is absent "
                    "from raw-visible original-message token IDs"
                ),
                "typed_parameter_matching": (
                    "recursive JSON or bounded literal scalar equality with string "
                    "and number types distinct; bool and null ignored"
                ),
                "missing_feature_policy": "unavailable_and_abstain",
            },
            "components": {
                "scaler": _scaler_components(scaler),
                "weights": _float_list(model.coef_),
                "intercept": float(model.intercept_),
            },
            "fit": {
                "estimator": "sklearn.linear_model.Ridge",
                "selected_alpha": selected,
                "alpha_candidates": list(C4_ALPHAS),
                "selection_metric": "mean_grouped_3fold_mse",
                "fold_preprocessing": "StandardScaler fitted on training fold only",
                "cv": cv,
                "known_example_count": len(known),
                "unknown_label_count": excluded,
                "calibration_state_count_excluded": calibration_count,
                "state_count": len({example["state_id"] for example in examples}),
                "group_count": len(set(groups.tolist())),
                "feature_enrichment": enrichment,
            },
            "provenance": _json_mapping(provenance, "provenance"),
        }
        artifact["artifact_sha256"] = artifact_sha256(artifact)
        artifacts[target] = artifact

    if output_dir is not None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        _write_artifact(directory / "c4_gain_turn.json", artifacts["delta_turn"])
        _write_artifact(directory / "c4_gain_task.json", artifacts["delta_task"])
    return artifacts


def prepare_c4_features(
    rows: Sequence[Mapping[str, Any]],
    *,
    models: Any,
    semantic_query_overflow_policy: str = "error",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Enrich only the preassigned train split with frozen local scores.

    This function sends only observable ``q``/``draft`` context and candidate
    text to the frozen embedding/reranker backends. Labels and official outcomes
    are never passed to a model call. Calibration rows are excluded first.
    """

    if not _sequence(rows) or not rows:
        raise ValueError("C4 feature preparation requires nonempty rows")
    training_rows, calibration_count = _select_training_split(rows)
    return _enrich_c4_training_rows(
        training_rows,
        models=models,
        calibration_count=calibration_count,
        semantic_query_overflow_policy=semantic_query_overflow_policy,
    )


def _enrich_c4_training_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    models: Any,
    calibration_count: int,
    semantic_query_overflow_policy: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if models is None:
        raise ValueError("models is required for C4 feature enrichment")
    from .set_retrieval import enrich_candidates

    enriched: list[dict[str, Any]] = []
    backend_receipts: list[dict[str, Any]] = []
    per_state: list[dict[str, Any]] = []
    drain = getattr(models, "drain_receipts", None)
    if callable(drain):
        drain()
    for index, row in enumerate(rows):
        context = _state_context(row)
        candidates = _state_candidates(row, index)
        enrich_candidates(
            context,
            candidates,
            models,
            rerank=True,
            semantic=True,
            overflow_policy=semantic_query_overflow_policy,
        )
        copied = json.loads(_canonical_json(row))
        copied["candidates"] = candidates
        enriched.append(copied)
        receipts = drain() if callable(drain) else []
        if receipts:
            backend_receipts.extend(receipts)
        per_state.append(
            {
                "state_id": row.get("state_id"),
                "candidate_count": len(candidates),
                "reranker_scores_present": sum(
                    candidate.get("reranker_score") is not None
                    for candidate in candidates
                ),
                "task_similarities_present": sum(
                    candidate.get("task_similarity") is not None
                    for candidate in candidates
                ),
                "draft_similarities_present": sum(
                    candidate.get("draft_similarity") is not None
                    for candidate in candidates
                ),
            }
        )
    public_config = getattr(models, "public_config", None)
    config = public_config() if callable(public_config) else None
    receipt = {
        "mode": "frozen_local_reranker_and_embeddings",
        "state_count": len(enriched),
        "calibration_state_count_excluded_before_enrichment": calibration_count,
        "labels_or_outcomes_passed_to_models": False,
        "local_model_config": config,
        "per_state": per_state,
        "backend_receipts": backend_receipts,
        "semantic_query_overflow_policy": semantic_query_overflow_policy,
    }
    _canonical_json(receipt)
    return enriched, receipt


def _local_score_model_contract(models: Any) -> dict[str, Any]:
    public_config = getattr(models, "public_config", None)
    if not callable(public_config):
        raise ValueError("models must expose public_config for artifact binding")
    config = public_config()
    if not isinstance(config, Mapping):
        raise ValueError("models.public_config() must return a mapping")
    result = {
        role: config[role]
        for role in ("embedding", "reranker")
        if role in config
    }
    if set(result) != {"embedding", "reranker"}:
        raise ValueError("local model config must bind embedding and reranker roles")
    return _json_mapping(result, "local score model contract")


def train_c1_model(
    rows: Sequence[Mapping[str, Any]],
    *,
    provenance: Mapping[str, Any],
    prefill_contract: Mapping[str, Any],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Fit the C1 PCA8 + NLL/STOP/parse L2 logistic model."""

    _validate_training_inputs(rows, provenance)
    rows, calibration_count = _select_training_split(rows)
    required_prefill = _prefill_contract(prefill_contract)
    parsed: list[dict[str, Any]] = []
    null_count = 0
    for index, row in enumerate(rows):
        if "c1_risk_label" not in row:
            raise ValueError(f"row {index} lacks c1_risk_label")
        label = row["c1_risk_label"]
        status = row.get("c1_label_status")
        if (status == "known" and label is None) or (status == "unknown" and label is not None):
            raise ValueError(f"row {index} c1 label/status mismatch")
        if label is None or row.get("c1_label_status") == "unknown":
            null_count += 1
            continue
        if type(label) is not int or label not in {0, 1}:
            raise ValueError(f"row {index} c1_risk_label must be 0, 1, or null")
        context = _state_context(row)
        feature = extract_c1_raw_features(context)
        if not feature.available or feature.vector is None:
            raise ValueError(f"row {index} C1 feature unavailable: {feature.reason}")
        observed = context.get("prefill_contract")
        if not isinstance(observed, Mapping) or _canonical_json(observed) != _canonical_json(
            required_prefill
        ):
            raise ValueError(f"row {index} prefill_contract mismatch")
        parsed.append(
            {
                "raw": np.asarray(feature.vector, dtype=float),
                "label": label,
                "group": _group_id(row, index),
            }
        )
    if not parsed:
        raise ValueError("no known C1 risk labels")
    dimensions = {len(example["raw"]) - 3 for example in parsed}
    if len(dimensions) != 1:
        raise ValueError("C1 rows mix prefill hidden dimensions")
    hidden_dimension = dimensions.pop()
    if hidden_dimension < 8:
        raise ValueError("C1 PCA8 requires prefill hidden dimension >= 8")
    raw = np.stack([example["raw"] for example in parsed])
    y = np.asarray([example["label"] for example in parsed], dtype=int)
    if set(y.tolist()) != {0, 1}:
        raise ValueError("C1 training requires both risk classes")
    groups = np.asarray([example["group"] for example in parsed], dtype=object)
    _require_three_groups(groups, "c1_risk_label")
    selected, cv = _select_c1_c(raw, y, groups, hidden_dimension)

    pca = PCA(n_components=8, svd_solver="full").fit(raw[:, :hidden_dimension])
    transformed = np.concatenate(
        [pca.transform(raw[:, :hidden_dimension]), raw[:, hidden_dimension:]], axis=1
    )
    scaler = StandardScaler().fit(transformed)
    model = LogisticRegression(
        C=selected, solver="lbfgs", max_iter=2000, random_state=0
    ).fit(scaler.transform(transformed), y)
    artifact = {
        "schema": ARTIFACT_SCHEMA,
        "model_kind": "c1_risk_logistic",
        "target": "c1_risk_label",
        "feature_contract": {
            "schema": C1_FEATURE_SCHEMA,
            "feature_names": list(C1_FEATURE_NAMES),
            "dimension": len(C1_FEATURE_NAMES),
            "prefill_contract": required_prefill,
            "draft_logprobs_required": True,
            "missing_feature_policy": "unavailable_and_abstain",
        },
        "components": {
            "hidden_dimension": hidden_dimension,
            "pca": {
                "n_components": 8,
                "mean": _float_list(pca.mean_),
                "components": [_float_list(row) for row in pca.components_],
            },
            "scaler": _scaler_components(scaler),
            "weights": _float_list(model.coef_[0]),
            "intercept": float(model.intercept_[0]),
        },
        "fit": {
            "estimator": "sklearn.linear_model.LogisticRegression",
            "penalty": "l2",
            "selected_c": selected,
            "c_candidates": list(C1_CS),
            "selection_metric": "mean_grouped_3fold_log_loss",
            "fold_preprocessing": "PCA8 then StandardScaler fitted on training fold only",
            "known_example_count": len(parsed),
            "unknown_label_count": null_count,
            "calibration_state_count_excluded": calibration_count,
            "group_count": len(set(groups.tolist())),
            "cv": cv,
        },
        "provenance": _json_mapping(provenance, "provenance"),
    }
    artifact["artifact_sha256"] = artifact_sha256(artifact)
    if output_path is not None:
        _write_artifact(Path(output_path), artifact)
    return artifact


def train_t01_calibrator(
    rows: Sequence[Mapping[str, Any]],
    *,
    provenance: Mapping[str, Any],
    score_model_contract: Mapping[str, Any],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Fit the optional weak-supervised head over frozen candidate scores."""

    _validate_training_inputs(rows, provenance)
    bound_score_models = _json_mapping(
        score_model_contract, "score_model_contract"
    )
    rows, calibration_count = _select_training_split(rows)
    parsed: list[dict[str, Any]] = []
    unknown_count = 0
    for index, row in enumerate(rows):
        if "relevance_label" not in row:
            raise ValueError(f"row {index} lacks relevance_label")
        label = row["relevance_label"]
        if label is None or label == "unknown":
            unknown_count += 1
            continue
        if type(label) is not int or label not in {0, 1}:
            raise ValueError(f"row {index} relevance_label must be 0, 1, or unknown")
        candidate = row.get("candidate")
        if not isinstance(candidate, Mapping):
            raise ValueError(f"row {index} lacks candidate")
        feature = extract_t01_features({}, candidate)
        if not feature.available or feature.vector is None:
            raise ValueError(f"row {index} T01 feature unavailable: {feature.reason}")
        parsed.append(
            {"features": feature.vector, "label": label, "group": _group_id(row, index)}
        )
    if not parsed:
        raise ValueError("no known T01 labels")
    x = np.asarray([row["features"] for row in parsed], dtype=float)
    y = np.asarray([row["label"] for row in parsed], dtype=int)
    if set(y.tolist()) != {0, 1}:
        raise ValueError("T01 training requires both relevance classes")
    groups = np.asarray([row["group"] for row in parsed], dtype=object)
    _require_three_groups(groups, "relevance_label")
    selected, cv = _select_logistic_c(x, y, groups, T01_CS)
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(
        C=selected, solver="lbfgs", max_iter=2000, random_state=0
    ).fit(scaler.transform(x), y)
    artifact = {
        "schema": ARTIFACT_SCHEMA,
        "model_kind": "t01_relevance_logistic",
        "target": "relevance_label",
        "feature_contract": {
            "schema": T01_FEATURE_SCHEMA,
            "feature_names": list(T01_FEATURE_NAMES),
            "dimension": len(T01_FEATURE_NAMES),
            "frozen_reranker_required": True,
            "score_model_contract": bound_score_models,
            "unknown_label_policy": "exclude_without_imputation",
            "missing_feature_policy": "unavailable_and_abstain",
        },
        "components": {
            "scaler": _scaler_components(scaler),
            "weights": _float_list(model.coef_[0]),
            "intercept": float(model.intercept_[0]),
        },
        "fit": {
            "estimator": "sklearn.linear_model.LogisticRegression",
            "penalty": "l2",
            "selected_c": selected,
            "c_candidates": list(T01_CS),
            "selection_metric": "mean_grouped_3fold_log_loss",
            "known_example_count": len(parsed),
            "unknown_label_count": unknown_count,
            "calibration_state_count_excluded": calibration_count,
            "group_count": len(set(groups.tolist())),
            "cv": cv,
        },
        "provenance": _json_mapping(provenance, "provenance"),
    }
    artifact["artifact_sha256"] = artifact_sha256(artifact)
    if output_path is not None:
        _write_artifact(Path(output_path), artifact)
    return artifact


def _c4_examples(
    rows: Sequence[Mapping[str, Any]], tokenizer: Any
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"row {row_index} must be a mapping")
        context = _state_context(row)
        candidates = _state_candidates(row, row_index)
        state_id = row.get("state_id")
        if not isinstance(state_id, str) or not state_id:
            raise ValueError(f"row {row_index} lacks state_id")
        group = _group_id(row, row_index)
        if row.get("schema") == "t02-labeled-state-v1" or "labels" in row:
            labels = row.get("labels")
            if not isinstance(labels, Mapping):
                raise ValueError(f"row {row_index} lacks T02 labels mapping")
            for branch_id in ("A1", "A2"):
                label = labels.get(branch_id)
                if not isinstance(label, Mapping):
                    raise ValueError(f"row {row_index} lacks label record {branch_id}")
                chosen_ids = _branch_candidate_ids(row, branch_id, row_index)
                example = _one_c4_example(
                    context, candidates, chosen_ids, tokenizer, row_index, branch_id
                )
                example.update(
                    {
                        "state_id": state_id,
                        "branch_id": branch_id,
                        "task_group_id": group,
                        "delta_turn": _delta_label(label, "delta_turn", row_index, branch_id),
                        "delta_task": _delta_label(label, "delta_task", row_index, branch_id),
                    }
                )
                examples.append(example)
        else:
            chosen = row.get("chosen_ids")
            if not _sequence(chosen):
                raise ValueError(f"row {row_index} lacks chosen_ids")
            example = _one_c4_example(
                context, candidates, tuple(chosen), tokenizer, row_index, "explicit"
            )
            example.update(
                {
                    "state_id": state_id,
                    "branch_id": str(row.get("branch_id", "explicit")),
                    "task_group_id": group,
                    "delta_turn": _delta_label(row, "delta_turn", row_index, "explicit"),
                    "delta_task": _delta_label(row, "delta_task", row_index, "explicit"),
                }
            )
            examples.append(example)
    if not examples:
        raise ValueError("C4 training rows are empty")
    return examples


def _one_c4_example(
    context: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    chosen_ids: Sequence[str],
    tokenizer: Any,
    row_index: int,
    branch_id: str,
) -> dict[str, Any]:
    feature = extract_c4_features(
        context, candidates, chosen_ids, tokenizer=tokenizer
    )
    if not feature.available or feature.vector is None:
        raise ValueError(
            f"row {row_index} branch {branch_id} C4 feature unavailable: {feature.reason}"
        )
    return {"features": feature.vector}


def _select_ridge_alpha(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> tuple[float, list[dict[str, Any]]]:
    splitter = GroupKFold(n_splits=3)
    rows: list[dict[str, Any]] = []
    for alpha in C4_ALPHAS:
        losses: list[float] = []
        folds: list[dict[str, Any]] = []
        for train, validation in splitter.split(x, y, groups):
            scaler = StandardScaler().fit(x[train])
            model = Ridge(alpha=alpha).fit(scaler.transform(x[train]), y[train])
            prediction = model.predict(scaler.transform(x[validation]))
            loss = float(mean_squared_error(y[validation], prediction))
            losses.append(loss)
            folds.append(_fold_receipt(groups, train, validation, loss, "mse"))
        rows.append({
            "alpha": alpha,
            "fold_mse": losses,
            "mean_mse": float(np.mean(losses)),
            "folds": folds,
        })
    selected = min(rows, key=lambda row: (row["mean_mse"], row["alpha"]))["alpha"]
    return float(selected), rows


def _select_c1_c(
    raw: np.ndarray, y: np.ndarray, groups: np.ndarray, hidden_dimension: int
) -> tuple[float, list[dict[str, Any]]]:
    splitter = GroupKFold(n_splits=3)
    rows: list[dict[str, Any]] = []
    for c_value in C1_CS:
        losses: list[float] = []
        folds: list[dict[str, Any]] = []
        for train, validation in splitter.split(raw, y, groups):
            if len(train) < 8:
                raise ValueError("each C1 training fold needs at least 8 labeled rows for PCA8")
            if set(y[train].tolist()) != {0, 1}:
                raise ValueError("each C1 training fold must contain both classes")
            pca = PCA(n_components=8, svd_solver="full").fit(raw[train, :hidden_dimension])
            train_x = np.concatenate(
                [pca.transform(raw[train, :hidden_dimension]), raw[train, hidden_dimension:]],
                axis=1,
            )
            validation_x = np.concatenate(
                [
                    pca.transform(raw[validation, :hidden_dimension]),
                    raw[validation, hidden_dimension:],
                ],
                axis=1,
            )
            scaler = StandardScaler().fit(train_x)
            model = LogisticRegression(
                C=c_value,
                solver="lbfgs",
                max_iter=2000,
                random_state=0,
            ).fit(scaler.transform(train_x), y[train])
            prediction = model.predict_proba(scaler.transform(validation_x))[:, 1]
            loss = float(log_loss(y[validation], prediction, labels=[0, 1]))
            losses.append(loss)
            folds.append(_fold_receipt(groups, train, validation, loss, "log_loss"))
        rows.append({
            "c": c_value,
            "fold_log_loss": losses,
            "mean_log_loss": float(np.mean(losses)),
            "folds": folds,
        })
    selected = min(rows, key=lambda row: (row["mean_log_loss"], row["c"]))["c"]
    return float(selected), rows


def _select_logistic_c(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, candidates: Sequence[float]
) -> tuple[float, list[dict[str, Any]]]:
    splitter = GroupKFold(n_splits=3)
    rows: list[dict[str, Any]] = []
    for c_value in candidates:
        losses: list[float] = []
        folds: list[dict[str, Any]] = []
        for train, validation in splitter.split(x, y, groups):
            if set(y[train].tolist()) != {0, 1}:
                raise ValueError("each logistic training fold must contain both classes")
            scaler = StandardScaler().fit(x[train])
            model = LogisticRegression(
                C=c_value,
                solver="lbfgs",
                max_iter=2000,
                random_state=0,
            ).fit(scaler.transform(x[train]), y[train])
            prediction = model.predict_proba(scaler.transform(x[validation]))[:, 1]
            loss = float(log_loss(y[validation], prediction, labels=[0, 1]))
            losses.append(loss)
            folds.append(_fold_receipt(groups, train, validation, loss, "log_loss"))
        rows.append({
            "c": c_value,
            "fold_log_loss": losses,
            "mean_log_loss": float(np.mean(losses)),
            "folds": folds,
        })
    selected = min(rows, key=lambda row: (row["mean_log_loss"], row["c"]))["c"]
    return float(selected), rows


def _state_context(row: Mapping[str, Any]) -> dict[str, Any]:
    context = row.get("context")
    if isinstance(context, Mapping):
        return dict(context)
    q = row.get("q")
    draft = row.get("draft")
    if isinstance(q, Mapping) and isinstance(draft, Mapping):
        result = {**dict(q), **dict(draft)}
        aliases = {
            "task": "goal",
            "token_logprobs": "draft_logprobs",
            "text": "draft_text",
            "tool_calls": "draft_tool_calls",
        }
        for source, target in aliases.items():
            if target not in result and source in result:
                result[target] = result[source]
        if "is_stop" not in result and draft.get("kind") in {"call", "stop"}:
            result["is_stop"] = draft["kind"] == "stop"
        return result
    raise ValueError("training row lacks context or q/draft mappings")


def _state_candidates(row: Mapping[str, Any], index: int) -> list[dict[str, Any]]:
    values = row.get("candidates")
    if not _sequence(values):
        raise ValueError(f"row {index} lacks candidates")
    result: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(values):
        if not isinstance(candidate, Mapping):
            raise ValueError(f"row {index} candidate {candidate_index} must be a mapping")
        normalized = dict(candidate)
        if "unit_id" not in normalized and "candidate_id" in normalized:
            normalized["unit_id"] = normalized["candidate_id"]
        if "event_id" not in normalized and "source_id" in normalized:
            normalized["event_id"] = normalized["source_id"]
        result.append(normalized)
    return result


def _branch_candidate_ids(
    row: Mapping[str, Any], branch_id: str, row_index: int
) -> tuple[str, ...]:
    labels = row.get("labels")
    label = labels.get(branch_id) if isinstance(labels, Mapping) else None
    if isinstance(label, Mapping):
        direct = label.get("candidate_ids", label.get("selected_ids"))
        if _sequence(direct):
            selected = tuple(direct)
            audited = _branch_spec_ids(row, branch_id)
            if audited is not None and audited != selected:
                raise ValueError(
                    f"row {row_index} label/branch candidate mismatch for {branch_id}"
                )
            return selected
        if label.get("action_id") is not None:
            return _action_candidate_ids(
                row, label["action_id"], row_index, branch_id
            )
    for container_name in ("branches", "branch_specs"):
        container = row.get(container_name)
        if not isinstance(container, Mapping):
            continue
        branch = container.get(branch_id)
        if not isinstance(branch, Mapping):
            continue
        direct = branch.get("candidate_ids", branch.get("selected_ids"))
        if _sequence(direct):
            return tuple(direct)
        action_id = branch.get("action_id")
        if action_id is not None:
            return _action_candidate_ids(row, action_id, row_index, branch_id)
    raise ValueError(f"row {row_index} lacks chosen candidate IDs for {branch_id}")


def _branch_spec_ids(row: Mapping[str, Any], branch_id: str) -> tuple[str, ...] | None:
    branches = row.get("branches")
    if isinstance(branches, Mapping):
        value = branches.get(branch_id)
        if isinstance(value, Mapping) and _sequence(value.get("candidate_ids")):
            return tuple(value["candidate_ids"])
    if _sequence(branches):
        matches = [
            value
            for value in branches
            if isinstance(value, Mapping) and value.get("branch_id") == branch_id
        ]
        if len(matches) == 1 and _sequence(matches[0].get("candidate_ids")):
            return tuple(matches[0]["candidate_ids"])
    return None


def _action_candidate_ids(
    row: Mapping[str, Any], action_id: Any, row_index: int, branch_id: str
) -> tuple[str, ...]:
    actions = row.get("allowed_actions")
    if not _sequence(actions):
        raise ValueError(f"row {row_index} lacks allowed_actions for {branch_id}")
    matches = [
        action
        for action in actions
        if isinstance(action, Mapping) and action.get("action_id") == action_id
    ]
    if len(matches) != 1 or not _sequence(matches[0].get("candidate_ids")):
        raise ValueError(f"row {row_index} cannot resolve action_id for {branch_id}")
    return tuple(matches[0]["candidate_ids"])


def _delta_label(
    record: Mapping[str, Any], name: str, row_index: int, branch_id: str
) -> float | None:
    if name not in record:
        raise ValueError(f"row {row_index} branch {branch_id} lacks {name}")
    value = record[name]
    status_names = (
        "status",
        f"{name}_label_status",
        "turn_label_status" if name == "delta_turn" else "task_label_status",
    )
    unknown = value is None or value == "unknown"
    if any(record.get(status_name) == ("known" if unknown else "unknown") for status_name in status_names):
        raise ValueError(f"row {row_index} branch {branch_id} {name} label/status mismatch")
    if unknown:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"row {row_index} branch {branch_id} {name} is invalid")
    result = float(value)
    if result not in {-1.0, 0.0, 1.0}:
        raise ValueError(f"row {row_index} branch {branch_id} {name} must be -1, 0, 1, or null")
    return result


def _group_id(row: Mapping[str, Any], index: int) -> str:
    value = row.get("task_group_id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"row {index} lacks task_group_id")
    return value


def _prefill_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    contract = _json_mapping(value, "prefill_contract")
    required = ("layer", "readout")
    missing = [name for name in required if name not in contract or contract[name] is None]
    if missing:
        raise ValueError(f"prefill_contract lacks {missing!r}")
    position = contract.get("position")
    nested_kind = position.get("kind") if isinstance(position, Mapping) else None
    if not isinstance(nested_kind, str) and not isinstance(
        contract.get("position_kind"), str
    ):
        raise ValueError("prefill_contract position.kind or position_kind is required")
    bindings = contract.get("bindings")
    has_direct_bindings = contract.get("model") is not None and contract.get(
        "tokenizer"
    ) is not None
    if not has_direct_bindings and not isinstance(bindings, Mapping):
        raise ValueError(
            "prefill_contract requires model/tokenizer or a nonempty bindings mapping"
        )
    return contract


def _require_three_groups(groups: np.ndarray, target: str) -> None:
    count = len(set(groups.tolist()))
    if count < 3:
        raise ValueError(f"{target} grouped 3-fold CV requires at least three task groups")


def _fold_receipt(
    groups: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    loss: float,
    loss_name: str,
) -> dict[str, Any]:
    train_groups = sorted({str(groups[index]) for index in train})
    validation_groups = sorted({str(groups[index]) for index in validation})
    if set(train_groups) & set(validation_groups):
        raise RuntimeError("GroupKFold leaked a task group across train and validation")
    return {
        "train_groups": train_groups,
        "validation_groups": validation_groups,
        "train_example_count": int(len(train)),
        "validation_example_count": int(len(validation)),
        loss_name: loss,
    }


def _validate_training_inputs(
    rows: Sequence[Mapping[str, Any]], provenance: Mapping[str, Any]
) -> None:
    if not _sequence(rows) or not rows:
        raise ValueError("training rows must be a nonempty sequence")
    _json_mapping(provenance, "provenance")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"row {index} must be a mapping")
        row_provenance = row.get("provenance")
        is_t02_labeled = row.get("schema") == "t02-labeled-state-v1"
        has_t02_provenance = (
            is_t02_labeled
            and isinstance(row.get("state_id"), str)
            and isinstance(row.get("decision_key"), str)
            and isinstance(row.get("outcomes"), Mapping)
        )
        if not has_t02_provenance and (
            not isinstance(row_provenance, Mapping) or not row_provenance
        ):
            raise ValueError(f"row {index} lacks nonempty provenance")


def _select_training_split(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], int]:
    has_split = ["split" in row for row in rows]
    if not any(has_split):
        return list(rows), 0
    if not all(has_split):
        raise ValueError("training rows mix explicit and missing split values")
    unsupported = sorted(
        {row.get("split") for row in rows if row.get("split") not in {"train", "calibration"}},
        key=str,
    )
    if unsupported:
        raise ValueError(f"unsupported training split values: {unsupported!r}")
    group_splits = {}
    for index, row in enumerate(rows):
        group = _group_id(row, index)
        previous = group_splits.setdefault(group, row["split"])
        if previous != row["split"]:
            raise ValueError(f"task group {group!r} crosses train/calibration splits")
    selected = [row for row in rows if row["split"] == "train"]
    if not selected:
        raise ValueError("training input contains no train split rows")
    return selected, len(rows) - len(selected)


def _scaler_components(scaler: StandardScaler) -> dict[str, list[float]]:
    return {"mean": _float_list(scaler.mean_), "scale": _float_list(scaler.scale_)}


def _float_list(values: Any) -> list[float]:
    result = [float(value) for value in np.asarray(values).reshape(-1)]
    if any(not math.isfinite(value) for value in result):
        raise ValueError("fitted model contains non-finite values")
    return result


def _json_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a nonempty mapping")
    try:
        return json.loads(_canonical_json(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be JSON serializable") from error


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_dataset(path: Path) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    payload = path.read_bytes()
    source = {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("training input must be UTF-8 JSON or JSONL") from error
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = [json.loads(line) for line in text.splitlines() if line.strip()]
        source["format"] = "jsonl"
    else:
        source["format"] = "json"
    if isinstance(value, Mapping) and isinstance(value.get("rows"), list):
        source["dataset_schema"] = value.get("schema")
        rows = value["rows"]
    elif isinstance(value, list):
        rows = value
    else:
        raise ValueError("training input must be a row list or an object with rows")
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("training input rows must be JSON objects")
    return list(rows), source


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Fit portable C1/C4/T01 recovery-set JSON artifacts."
    )
    parser.add_argument("model", choices=("c1", "c4", "t01"))
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokenizer", help="Local path or Hugging Face tokenizer ID for C4")
    parser.add_argument(
        "--local-models",
        type=Path,
        help="C4 LocalSelectionModels JSON for offline score enrichment",
    )
    parser.add_argument(
        "--score-model-contract",
        type=Path,
        help="Embedding/reranker contract JSON when scores are already present",
    )
    parser.add_argument(
        "--semantic-query-overflow-policy",
        choices=("error", "task_head_tail_preserve_draft_v1"),
        default="error",
        help=(
            "C4 embedding/reranker query overflow policy; stored in the fitted "
            "feature contract and required to match inference"
        ),
    )
    parser.add_argument("--prefill-contract", type=Path, help="C1 prefill contract JSON")
    parser.add_argument(
        "--provenance-json",
        type=Path,
        help="Optional JSON object merged into the source-file provenance",
    )
    args = parser.parse_args()
    rows, provenance = _read_dataset(args.input)
    if args.provenance_json is not None:
        extra = json.loads(args.provenance_json.read_text(encoding="utf-8"))
        provenance["user"] = _json_mapping(extra, "provenance JSON")
    provenance["trainer"] = "memory_runtime.recovery.set_training"
    supplied_score_contract = None
    if args.score_model_contract is not None:
        supplied_score_contract = _json_mapping(
            json.loads(args.score_model_contract.read_text(encoding="utf-8")),
            "score model contract",
        )

    if args.model == "c4":
        if not args.tokenizer:
            parser.error("c4 requires --tokenizer")
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise RuntimeError("C4 CLI requires transformers for exact tokenization") from error
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, trust_remote_code=False, local_files_only=True
        )
        models = None
        if args.local_models is not None:
            from .local_selection_models import LocalSelectionModels

            model_config = json.loads(args.local_models.read_text(encoding="utf-8"))
            models = LocalSelectionModels(
                _json_mapping(model_config, "local models config")
            )
        train_c4_models(
            rows,
            tokenizer=tokenizer,
            provenance=provenance,
            output_dir=args.output,
            models=models,
            score_model_contract=supplied_score_contract,
            semantic_query_overflow_policy=args.semantic_query_overflow_policy,
        )
    elif args.model == "c1":
        if args.prefill_contract is None:
            parser.error("c1 requires --prefill-contract")
        contract = json.loads(args.prefill_contract.read_text(encoding="utf-8"))
        train_c1_model(
            rows,
            provenance=provenance,
            prefill_contract=_json_mapping(contract, "prefill contract"),
            output_path=args.output,
        )
    else:
        if args.local_models is not None:
            from .local_selection_models import LocalSelectionModels

            model_config = json.loads(args.local_models.read_text(encoding="utf-8"))
            supplied_score_contract = _local_score_model_contract(
                LocalSelectionModels(_json_mapping(model_config, "local models config"))
            )
        train_t01_calibrator(
            rows,
            provenance=provenance,
            score_model_contract=supplied_score_contract,
            output_path=args.output,
        )
    return 0


def _sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


__all__ = [
    "C1_CS",
    "C4_ALPHAS",
    "T01_CS",
    "train_c1_model",
    "train_c4_models",
    "train_t01_calibrator",
    "prepare_c4_features",
]


if __name__ == "__main__":
    raise SystemExit(_cli())
