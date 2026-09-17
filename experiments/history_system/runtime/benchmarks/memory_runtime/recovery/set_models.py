"""Portable runtime models for recovery-set selection.

Training lives in :mod:`set_training`.  This module deliberately contains no
sklearn dependency so a fitted JSON artifact can be used by the benchmark
runtime without importing the training stack.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .set_protocol import supported_parameters as _protocol_supported_parameters
from .set_protocol import typed_parameters as _protocol_typed_parameters


ARTIFACT_SCHEMA = "c2kv-recovery-set-model-v1"
C4_FEATURE_SCHEMA = "c2kv-c4-fixed18-v1"
C1_FEATURE_SCHEMA = "c2kv-c1-pca8-nll-stop-parse-v1"
T01_FEATURE_SCHEMA = "c2kv-t01-frozen-candidate4-v1"

C4_FEATURE_NAMES = (
    "draft_mean_nll",
    "draft_nll_p95",
    "is_stop",
    "parse_ok",
    "reranker_score_max",
    "reranker_score_mean",
    "reranker_score_min",
    "task_similarity_max",
    "task_similarity_mean",
    "draft_similarity_max",
    "draft_similarity_mean",
    "tool_name_match_fraction",
    "draft_parameter_coverage_fraction",
    "raw_token_novelty_fraction",
    "distinct_event_count",
    "selected_token_count_log1p",
    "draft_mean_nll_x_reranker_mean",
    "parameter_coverage_x_raw_token_novelty",
)
C1_FEATURE_NAMES = tuple(f"prefill_pca_{index}" for index in range(8)) + (
    "draft_mean_nll",
    "is_stop",
    "parse_ok",
)
T01_FEATURE_NAMES = (
    "reranker_score",
    "task_similarity",
    "draft_similarity",
    "tool_name_match",
)


@dataclass(frozen=True)
class FeatureResult:
    available: bool
    vector: tuple[float, ...] | None
    missing_fields: tuple[str, ...] = ()
    reason: str | None = None
    details: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class PredictionResult:
    available: bool
    score: float | None
    reason: str | None = None
    missing_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class SelectionResult:
    selected_ids: tuple[str, ...]
    score: float | None
    available: bool
    reason: str
    scores: tuple[tuple[tuple[str, ...], float], ...] = ()
    missing_fields: tuple[str, ...] = ()


class ArtifactContractError(ValueError):
    """Raised when a serialized model does not satisfy its runtime contract."""


def extract_c4_features(
    context: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    chosen_ids: Sequence[str],
    *,
    tokenizer: Any,
) -> FeatureResult:
    """Extract the fixed C4 feature vector from original candidate text.

    ``tokenizer`` is mandatory.  It must expose ``encode`` or be callable and
    return token IDs.  Token novelty counts candidate token *positions* whose
    IDs do not occur in the currently raw-visible original message text.  No
    lower-casing, lexical fallback, or guessed token count is used.
    """

    missing: list[str] = []
    if not isinstance(context, Mapping):
        return _unavailable("context must be a mapping", ("context",))
    for name in (
        "goal",
        "raw_visible",
        "raw_source_ids",
        "draft_logprobs",
        "draft_text",
        "draft_tool_calls",
        "parse_ok",
        "is_stop",
    ):
        if name not in context or context[name] is None:
            missing.append(f"context.{name}")
    if tokenizer is None:
        missing.append("tokenizer")
    if missing:
        return _unavailable("required C4 input is missing", missing)

    try:
        mean_nll, p95_nll = _draft_nll_features(context["draft_logprobs"])
        is_stop = _strict_bool(context["is_stop"], "context.is_stop")
        parse_ok = _strict_bool(context["parse_ok"], "context.parse_ok")
        if not isinstance(context["goal"], str):
            raise ValueError("context.goal must be a string")
        if not isinstance(context["draft_text"], str):
            raise ValueError("context.draft_text must be a string")
        raw_source_ids = context["raw_source_ids"]
        if not _sequence(raw_source_ids) or any(
            not isinstance(value, str) or not value for value in raw_source_ids
        ):
            raise ValueError("context.raw_source_ids must be nonempty string IDs")
        raw_visible = context["raw_visible"]
        if not _sequence(raw_visible):
            raise ValueError("context.raw_visible must be a sequence")
        visible_tokens: set[int | str] = set()
        for index, message in enumerate(raw_visible):
            visible_tokens.update(
                _token_ids(_original_message_text(message, index), tokenizer)
            )

        records = _candidate_map(candidates)
        selected_ids = tuple(chosen_ids)
        if not selected_ids:
            raise ValueError("chosen_ids must be nonempty for C4 features")
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("chosen_ids contains duplicates")
        unknown = sorted(set(selected_ids) - set(records))
        if unknown:
            raise ValueError(f"chosen_ids contains unknown IDs: {unknown!r}")
        selected = [records[unit_id] for unit_id in selected_ids]

        reranker = [_candidate_number(row, "reranker_score") for row in selected]
        task_similarity = [
            _candidate_number(row, "task_similarity") for row in selected
        ]
        draft_similarity = [
            _candidate_number(row, "draft_similarity") for row in selected
        ]
        tool_matches = [
            _candidate_fraction(row, "tool_name_match") for row in selected
        ]

        selected_token_ids: list[int | str] = []
        candidate_texts: list[str] = []
        event_ids: set[str] = set()
        for row in selected:
            text = row["text"]
            tokens = _token_ids(text, tokenizer)
            declared = row["token_count"]
            if type(declared) is not int or declared < 0:
                raise ValueError(
                    f"candidate {row['unit_id']!r} token_count must be nonnegative int"
                )
            if declared != len(tokens):
                raise ValueError(
                    f"candidate {row['unit_id']!r} token_count differs from tokenizer"
                )
            selected_token_ids.extend(tokens)
            candidate_texts.append(text)
            event_ids.add(row["event_id"])

        parameters = (
            _draft_typed_parameters(context["draft_tool_calls"])
            if parse_ok
            else set()
        )
        supported_parameters = set().union(
            *(_supported_parameters(parameters, text) for text in candidate_texts)
        ) if candidate_texts else set()
        parameter_coverage = (
            len(supported_parameters) / len(parameters) if parameters else 0.0
        )
        novelty = (
            sum(token not in visible_tokens for token in selected_token_ids)
            / len(selected_token_ids)
            if selected_token_ids
            else 0.0
        )
        reranker_mean = float(np.mean(reranker))
        vector = (
            mean_nll,
            p95_nll,
            float(is_stop),
            float(parse_ok),
            max(reranker),
            reranker_mean,
            min(reranker),
            max(task_similarity),
            float(np.mean(task_similarity)),
            max(draft_similarity),
            float(np.mean(draft_similarity)),
            float(np.mean(tool_matches)),
            parameter_coverage,
            novelty,
            float(len(event_ids)),
            math.log1p(len(selected_token_ids)),
            mean_nll * reranker_mean,
            parameter_coverage * novelty,
        )
        _finite_vector(vector, "C4 features", expected=len(C4_FEATURE_NAMES))
        return FeatureResult(
            available=True,
            vector=tuple(float(value) for value in vector),
            details={
                "feature_schema": C4_FEATURE_SCHEMA,
                "token_count": len(selected_token_ids),
                "parameter_count": len(parameters),
                "matched_parameter_count": len(supported_parameters),
                "raw_visible_token_type_count": len(visible_tokens),
                "novelty_definition": (
                    "fraction of selected original-text token positions whose token ID "
                    "is absent from all raw-visible original message token IDs"
                ),
            },
        )
    except (TypeError, ValueError) as error:
        return _unavailable(str(error), ())


def extract_c1_raw_features(context: Mapping[str, Any]) -> FeatureResult:
    """Return hidden state and three scalar C1 inputs before PCA/scaling."""

    if not isinstance(context, Mapping):
        return _unavailable("context must be a mapping", ("context",))
    missing = [
        f"context.{name}"
        for name in ("prefill_hidden", "draft_logprobs", "is_stop", "parse_ok")
        if name not in context or context[name] is None
    ]
    if missing:
        return _unavailable("required C1 input is missing", missing)
    try:
        hidden = _finite_vector(context["prefill_hidden"], "prefill_hidden")
        if not hidden:
            raise ValueError("prefill_hidden must be nonempty")
        mean_nll, _ = _draft_nll_features(context["draft_logprobs"])
        is_stop = _strict_bool(context["is_stop"], "context.is_stop")
        parse_ok = _strict_bool(context["parse_ok"], "context.parse_ok")
        return FeatureResult(
            True,
            tuple(hidden + [mean_nll, float(is_stop), float(parse_ok)]),
            details={"hidden_dimension": len(hidden)},
        )
    except (TypeError, ValueError) as error:
        return _unavailable(str(error), ())


def extract_t01_features(
    context: Mapping[str, Any], candidate: Mapping[str, Any]
) -> FeatureResult:
    """Extract frozen-score matching features for the optional T01 calibrator."""

    del context  # Reserved so later contracts can bind state without changing the API.
    try:
        vector = (
            _candidate_number(candidate, "reranker_score"),
            _candidate_number(candidate, "task_similarity"),
            _candidate_number(candidate, "draft_similarity"),
            _candidate_fraction(candidate, "tool_name_match"),
        )
        return FeatureResult(True, vector)
    except (TypeError, ValueError) as error:
        return _unavailable(str(error), ())


class C4GainArtifact:
    def __init__(self, artifact: Mapping[str, Any] | str | Path):
        self.artifact = _load_and_validate(artifact, {"c4_gain_turn", "c4_gain_task"})
        contract = self.artifact["feature_contract"]
        if contract.get("schema") != C4_FEATURE_SCHEMA or tuple(
            contract.get("feature_names", ())
        ) != C4_FEATURE_NAMES:
            raise ArtifactContractError("C4 feature contract does not match fixed18")
        if not isinstance(contract.get("tokenizer_contract"), Mapping):
            raise ArtifactContractError("C4 feature contract lacks tokenizer binding")
        if not isinstance(contract.get("score_model_contract"), Mapping):
            raise ArtifactContractError("C4 feature contract lacks score-model binding")
        if contract.get("semantic_query_overflow_policy") not in {
            "error",
            "task_head_tail_preserve_draft_v1",
        }:
            raise ArtifactContractError(
                "C4 feature contract lacks a valid semantic query overflow policy"
            )

    @property
    def target(self) -> str:
        return str(self.artifact["target"])

    def predict(
        self,
        context: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
        chosen_ids: Sequence[str],
        *,
        tokenizer: Any,
    ) -> PredictionResult:
        expected_tokenizer = self.artifact["feature_contract"]["tokenizer_contract"]
        try:
            observed_tokenizer = tokenizer_contract(tokenizer)
        except (TypeError, ValueError) as error:
            return PredictionResult(False, None, str(error), ("tokenizer",))
        if _canonical_json(observed_tokenizer) != _canonical_json(expected_tokenizer):
            return PredictionResult(False, None, "tokenizer contract mismatch")
        features = extract_c4_features(
            context, candidates, chosen_ids, tokenizer=tokenizer
        )
        if not features.available or features.vector is None:
            return PredictionResult(
                False, None, features.reason, features.missing_fields
            )
        try:
            score = _linear_prediction(self.artifact, features.vector)
        except (TypeError, ValueError) as error:
            return PredictionResult(False, None, str(error))
        return PredictionResult(True, score)


class C4GainSelector:
    def __init__(self, artifact: Mapping[str, Any] | str | Path, *, delta: float = 0.0):
        self.model = C4GainArtifact(artifact)
        self.delta = _finite_number(delta, "delta")

    @property
    def kind(self) -> str:
        return "gain_turn" if self.model.target == "delta_turn" else "gain_task"

    @property
    def artifact(self) -> Mapping[str, Any]:
        return self.model.artifact

    @property
    def score_model_contract(self) -> Mapping[str, Any]:
        return self.model.artifact["feature_contract"]["score_model_contract"]

    @property
    def semantic_query_overflow_policy(self) -> str:
        return str(
            self.model.artifact["feature_contract"][
                "semantic_query_overflow_policy"
            ]
        )

    def select(
        self,
        context: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
        legal_actions: Sequence[Sequence[str]],
        *,
        tokenizer: Any,
        delta: float | None = None,
    ) -> SelectionResult:
        effective_delta = self.delta if delta is None else _finite_number(delta, "delta")
        actions = _validated_legal_actions(legal_actions, candidates)
        scored: list[tuple[tuple[str, ...], float]] = []
        for action in actions:
            if not action:
                continue
            prediction = self.model.predict(
                context, candidates, action, tokenizer=tokenizer
            )
            if not prediction.available or prediction.score is None:
                return SelectionResult(
                    (), None, False, prediction.reason or "model unavailable", (),
                    prediction.missing_fields,
                )
            scored.append((action, prediction.score))
        if not scored:
            return SelectionResult((), 0.0, True, "no_nonempty_legal_action")
        best_action, best_score = min(
            scored, key=lambda item: (-item[1], len(item[0]), item[0])
        )
        if best_score <= effective_delta:
            return SelectionResult(
                (), best_score, True, "score_not_above_delta", tuple(scored)
            )
        return SelectionResult(
            best_action, best_score, True, "selected_positive_gain", tuple(scored)
        )


class C1RiskArtifact:
    def __init__(self, artifact: Mapping[str, Any] | str | Path):
        self.artifact = _load_and_validate(artifact, {"c1_risk_logistic"})
        contract = self.artifact["feature_contract"]
        if contract.get("schema") != C1_FEATURE_SCHEMA or tuple(
            contract.get("feature_names", ())
        ) != C1_FEATURE_NAMES:
            raise ArtifactContractError("C1 feature contract does not match PCA8")

    def predict_risk(self, context: Mapping[str, Any]) -> PredictionResult:
        features = extract_c1_raw_features(context)
        if not features.available or features.vector is None:
            return PredictionResult(False, None, features.reason, features.missing_fields)
        expected_prefill = self.artifact["feature_contract"].get("prefill_contract")
        observed_prefill = context.get("prefill_contract")
        if not isinstance(observed_prefill, Mapping):
            return PredictionResult(
                False, None, "context.prefill_contract is required", ("context.prefill_contract",)
            )
        if _canonical_json(observed_prefill) != _canonical_json(expected_prefill):
            return PredictionResult(False, None, "prefill contract mismatch")
        components = self.artifact["components"]
        hidden_dimension = int(components["hidden_dimension"])
        raw = np.asarray(features.vector, dtype=float)
        if raw.size != hidden_dimension + 3:
            return PredictionResult(False, None, "prefill hidden dimension mismatch")
        hidden = raw[:hidden_dimension]
        pca = components["pca"]
        transformed = (hidden - np.asarray(pca["mean"], dtype=float)) @ np.asarray(
            pca["components"], dtype=float
        ).T
        combined = np.concatenate([transformed, raw[hidden_dimension:]])
        try:
            logit = _linear_prediction(self.artifact, combined)
        except (TypeError, ValueError) as error:
            return PredictionResult(False, None, str(error))
        return PredictionResult(True, _sigmoid(logit))


class C1RiskSelector:
    def __init__(
        self, artifact: Mapping[str, Any] | str | Path, *, threshold: float = 0.5
    ):
        self.model = C1RiskArtifact(artifact)
        self.threshold = _finite_number(threshold, "threshold")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")

    kind = "risk"

    @property
    def artifact(self) -> Mapping[str, Any]:
        return self.model.artifact

    def select(
        self,
        context: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
        legal_actions: Sequence[Sequence[str]],
        *,
        threshold: float | None = None,
    ) -> SelectionResult:
        effective_threshold = (
            self.threshold
            if threshold is None
            else _finite_number(threshold, "threshold")
        )
        if not 0.0 <= effective_threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        actions = _validated_legal_actions(legal_actions, candidates)
        prediction = self.model.predict_risk(context)
        if not prediction.available or prediction.score is None:
            return SelectionResult(
                (), None, False, prediction.reason or "model unavailable", (),
                prediction.missing_fields,
            )
        if prediction.score <= effective_threshold:
            return SelectionResult((), prediction.score, True, "risk_not_above_threshold")
        first_id = _candidate_map(candidates, validate_model_fields=False)
        first = next(iter(first_id), None)
        action = (first,) if first is not None else ()
        if action not in actions:
            return SelectionResult(
                (), prediction.score, True, "fixed_top_candidate_not_legal"
            )
        return SelectionResult(action, prediction.score, True, "risk_triggered")


class T01RelevanceArtifact:
    def __init__(self, artifact: Mapping[str, Any] | str | Path):
        self.artifact = _load_and_validate(artifact, {"t01_relevance_logistic"})
        contract = self.artifact["feature_contract"]
        if contract.get("schema") != T01_FEATURE_SCHEMA or tuple(
            contract.get("feature_names", ())
        ) != T01_FEATURE_NAMES:
            raise ArtifactContractError("T01 feature contract mismatch")
        if not isinstance(contract.get("score_model_contract"), Mapping):
            raise ArtifactContractError("T01 feature contract lacks score-model binding")

    def predict_candidate(
        self, context: Mapping[str, Any], candidate: Mapping[str, Any]
    ) -> PredictionResult:
        features = extract_t01_features(context, candidate)
        if not features.available or features.vector is None:
            return PredictionResult(False, None, features.reason, features.missing_fields)
        return PredictionResult(
            True, _sigmoid(_linear_prediction(self.artifact, features.vector))
        )


class T01Selector:
    def __init__(
        self, artifact: Mapping[str, Any] | str | Path, *, threshold: float = 0.5
    ):
        self.model = T01RelevanceArtifact(artifact)
        self.threshold = _finite_number(threshold, "threshold")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")

    kind = "reranker_calibrator"

    @property
    def artifact(self) -> Mapping[str, Any]:
        return self.model.artifact

    @property
    def score_model_contract(self) -> Mapping[str, Any]:
        return self.model.artifact["feature_contract"]["score_model_contract"]

    def select(
        self,
        context: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
        legal_actions: Sequence[Sequence[str]],
    ) -> SelectionResult:
        actions = _validated_legal_actions(legal_actions, candidates)
        records = _candidate_map(candidates)
        probabilities: dict[str, float] = {}
        for unit_id, candidate in records.items():
            prediction = self.model.predict_candidate(context, candidate)
            if not prediction.available or prediction.score is None:
                return SelectionResult(
                    (), None, False, prediction.reason or "model unavailable", (),
                    prediction.missing_fields,
                )
            probabilities[unit_id] = prediction.score
        scored = [
            (action, sum(probabilities[unit_id] - self.threshold for unit_id in action))
            for action in actions
            if action
        ]
        if not scored:
            return SelectionResult((), 0.0, True, "no_nonempty_legal_action")
        best_action, best_score = min(
            scored, key=lambda item: (-item[1], len(item[0]), item[0])
        )
        if best_score <= 0.0:
            return SelectionResult((), best_score, True, "score_not_positive", tuple(scored))
        return SelectionResult(best_action, best_score, True, "selected_relevant", tuple(scored))


def load_set_model(path: str | Path | Mapping[str, Any]) -> Any:
    """Load one portable artifact and return its runtime predictor."""

    artifact = _artifact_mapping(path)
    kind = artifact.get("model_kind")
    if kind in {"c4_gain_turn", "c4_gain_task"}:
        return C4GainArtifact(artifact)
    if kind == "c1_risk_logistic":
        return C1RiskArtifact(artifact)
    if kind == "t01_relevance_logistic":
        return T01RelevanceArtifact(artifact)
    raise ArtifactContractError(f"unsupported model_kind {kind!r}")


def load_set_selector(path: str | Path | Mapping[str, Any]) -> Any:
    """Load one artifact as the online selector used by root dispatch."""

    artifact = _artifact_mapping(path)
    kind = artifact.get("model_kind")
    if kind in {"c4_gain_turn", "c4_gain_task"}:
        return C4GainSelector(artifact)
    if kind == "c1_risk_logistic":
        return C1RiskSelector(artifact)
    if kind == "t01_relevance_logistic":
        return T01Selector(artifact)
    raise ArtifactContractError(f"unsupported model_kind {kind!r}")


def validate_selector_score_models(
    selector: Any,
    models: Any,
    *,
    semantic_query_overflow_policy: str | None = None,
) -> None:
    """Reject online score-model or payload-policy drift from fitted features."""

    expected = getattr(selector, "score_model_contract", None)
    public_config = getattr(models, "public_config", None)
    if not isinstance(expected, Mapping) or not callable(public_config):
        raise ArtifactContractError(
            "selector/model pair does not expose a score-model contract"
        )
    config = public_config()
    if not isinstance(config, Mapping):
        raise ArtifactContractError("online local model config is unavailable")
    observed = {
        role: config[role]
        for role in ("embedding", "reranker")
        if role in config
    }
    def feature_config(value):
        # Placement and cache policy are execution choices, not feature models.
        # An artifact trained on one free NPU must be usable on another.
        return {role: {key: item for key, item in settings.items()
                      if key not in {"device", "cache_size", "local_files_only"}}
                for role, settings in value.items()}
    if _canonical_json(feature_config(observed)) != _canonical_json(feature_config(expected)):
        raise ArtifactContractError(
            "online embedding/reranker config differs from fitted artifact"
        )
    expected_policy = getattr(selector, "semantic_query_overflow_policy", None)
    if expected_policy is not None:
        if semantic_query_overflow_policy is None:
            raise ArtifactContractError(
                "online semantic query overflow policy is unavailable"
            )
        if semantic_query_overflow_policy != expected_policy:
            raise ArtifactContractError(
                "online semantic query overflow policy differs from fitted artifact"
            )


def artifact_sha256(artifact: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def tokenizer_contract(tokenizer: Any) -> dict[str, Any]:
    """Bind C4 training and inference to identical tokenization behavior."""

    if tokenizer is None:
        raise ValueError("tokenizer is required")
    probes = (
        "plain ASCII tokens",
        'JSON {"id":7,"flag":true}',
        "Unicode 中文 evidence",
        "newline\nseparated",
    )
    encoded = [_token_ids(probe, tokenizer) for probe in probes]
    identity: dict[str, Any] = {
        "class": f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__qualname__}",
        "probe_sha256": hashlib.sha256(
            _canonical_json(encoded).encode("utf-8")
        ).hexdigest(),
    }
    name = getattr(tokenizer, "name_or_path", None)
    if isinstance(name, str) and name:
        identity["name_or_path"] = name
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if type(vocab_size) is int and vocab_size > 0:
        identity["vocab_size"] = vocab_size
    return identity


def _load_and_validate(
    source: Mapping[str, Any] | str | Path, kinds: set[str]
) -> dict[str, Any]:
    artifact = _artifact_mapping(source)
    if artifact.get("schema") != ARTIFACT_SCHEMA:
        raise ArtifactContractError("unsupported artifact schema")
    if artifact.get("model_kind") not in kinds:
        raise ArtifactContractError("artifact model kind is incompatible")
    provenance = artifact.get("provenance")
    if not isinstance(provenance, Mapping) or not provenance:
        raise ArtifactContractError("artifact provenance must be nonempty")
    contract = artifact.get("feature_contract")
    if not isinstance(contract, Mapping):
        raise ArtifactContractError("artifact lacks feature_contract")
    expected_hash = artifact.get("artifact_sha256")
    if not isinstance(expected_hash, str) or expected_hash != artifact_sha256(artifact):
        raise ArtifactContractError("artifact SHA-256 mismatch")
    components = artifact.get("components")
    if not isinstance(components, Mapping):
        raise ArtifactContractError("artifact lacks model components")
    return artifact


def _linear_prediction(artifact: Mapping[str, Any], vector: Sequence[float]) -> float:
    components = artifact["components"]
    scaler = components["scaler"]
    raw = np.asarray(_finite_vector(vector, "model input"), dtype=float)
    mean = np.asarray(_finite_vector(scaler["mean"], "scaler mean"), dtype=float)
    scale = np.asarray(_finite_vector(scaler["scale"], "scaler scale"), dtype=float)
    weights = np.asarray(_finite_vector(components["weights"], "weights"), dtype=float)
    if raw.shape != mean.shape or raw.shape != scale.shape or raw.shape != weights.shape:
        raise ValueError("model component dimensions do not match input")
    if np.any(scale <= 0):
        raise ValueError("scaler scale must be positive")
    intercept = _finite_number(components["intercept"], "intercept")
    return float(((raw - mean) / scale) @ weights + intercept)


def _validated_legal_actions(
    legal_actions: Sequence[Sequence[str]], candidates: Sequence[Mapping[str, Any]]
) -> tuple[tuple[str, ...], ...]:
    if not _sequence(legal_actions):
        raise ValueError("legal_actions must be a nonempty sequence from the real packer")
    candidate_ids = set(_candidate_map(candidates, validate_model_fields=False))
    actions: list[tuple[str, ...]] = []
    for index, value in enumerate(legal_actions):
        if not _sequence(value):
            if isinstance(value, tuple) and not value:
                action = ()
            elif isinstance(value, list) and not value:
                action = ()
            else:
                raise ValueError(f"legal action {index} must be an ID sequence")
        else:
            action = tuple(value)
        if any(not isinstance(unit_id, str) or not unit_id for unit_id in action):
            raise ValueError(f"legal action {index} contains invalid IDs")
        if len(action) != len(set(action)):
            raise ValueError(f"legal action {index} contains duplicate IDs")
        unknown = sorted(set(action) - candidate_ids)
        if unknown:
            raise ValueError(f"legal action {index} contains unknown IDs: {unknown!r}")
        actions.append(action)
    if actions.count(()) != 1:
        raise ValueError("legal_actions must contain the empty action exactly once")
    if len(actions) != len(set(actions)):
        raise ValueError("legal_actions contains duplicate actions")
    return tuple(actions)


def _candidate_map(
    candidates: Sequence[Mapping[str, Any]], *, validate_model_fields: bool = True
) -> dict[str, Mapping[str, Any]]:
    if not _sequence(candidates):
        raise ValueError("candidates must be a sequence")
    result: dict[str, Mapping[str, Any]] = {}
    required = (
        "unit_id",
        "event_id",
        "text",
        "token_count",
        "provenance",
        "reranker_score",
        "task_similarity",
        "draft_similarity",
        "tool_name_match",
    )
    for index, row in enumerate(candidates):
        if not isinstance(row, Mapping):
            raise ValueError(f"candidate {index} must be a mapping")
        names = required if validate_model_fields else ("unit_id",)
        missing = [name for name in names if name not in row or row[name] is None]
        if missing:
            raise ValueError(f"candidate {index} lacks required fields {missing!r}")
        unit_id = row["unit_id"]
        if not isinstance(unit_id, str) or not unit_id:
            raise ValueError(f"candidate {index} has invalid unit_id")
        if unit_id in result:
            raise ValueError(f"duplicate candidate unit_id {unit_id!r}")
        if validate_model_fields:
            if not isinstance(row["event_id"], str) or not row["event_id"]:
                raise ValueError(f"candidate {unit_id!r} has invalid event_id")
            if not isinstance(row["text"], str):
                raise ValueError(f"candidate {unit_id!r} text must be a string")
            provenance = row["provenance"]
            if not (
                (isinstance(provenance, Mapping) and provenance)
                or (isinstance(provenance, str) and provenance)
                or (_sequence(provenance) and len(provenance) > 0)
            ):
                raise ValueError(f"candidate {unit_id!r} lacks provenance")
        result[unit_id] = row
    return result


def _candidate_number(row: Mapping[str, Any], name: str) -> float:
    if name not in row or row[name] is None:
        raise ValueError(f"candidate {row.get('unit_id')!r} lacks {name}")
    return _finite_number(row[name], f"candidate {row.get('unit_id')!r} {name}")


def _candidate_fraction(row: Mapping[str, Any], name: str) -> float:
    raw = row.get(name)
    value = float(raw) if type(raw) is bool else _candidate_number(row, name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"candidate {row.get('unit_id')!r} {name} must be in [0, 1]")
    return value


def _draft_nll_features(logprobs: Any) -> tuple[float, float]:
    values = _finite_vector(logprobs, "draft_logprobs")
    if not values:
        raise ValueError("draft_logprobs must be nonempty")
    nll = -np.asarray(values, dtype=float)
    return float(np.mean(nll)), float(np.percentile(nll, 95, method="linear"))


def _token_ids(text: str, tokenizer: Any) -> list[int | str]:
    if not isinstance(text, str):
        raise ValueError("tokenizer input must be a string")
    if hasattr(tokenizer, "encode") and callable(tokenizer.encode):
        try:
            value = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            value = tokenizer.encode(text)
    elif callable(tokenizer):
        value = tokenizer(text)
    else:
        raise ValueError("tokenizer must be callable or expose encode")
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not _sequence(value):
        if value == [] or value == ():
            return []
        raise ValueError("tokenizer must return a token ID sequence")
    result: list[int | str] = []
    for token in value:
        if isinstance(token, bool) or not isinstance(token, (int, str)):
            raise ValueError("token IDs must be integers or strings")
        result.append(token)
    return result


def _original_message_text(message: Any, index: int) -> str:
    if isinstance(message, str):
        return message
    if not isinstance(message, Mapping):
        raise ValueError(f"raw_visible[{index}] must be a string or message mapping")
    original = message.get("original_text")
    if isinstance(original, str):
        return original
    parts: list[str] = []
    content = message.get("content")
    if isinstance(content, str):
        parts.append(content)
    elif _sequence(content):
        for part in content:
            if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                parts.append(part["text"])
    calls = message.get("tool_calls")
    if _sequence(calls):
        for call in calls:
            if not isinstance(call, Mapping):
                continue
            function = call.get("function")
            if not isinstance(function, Mapping):
                continue
            if isinstance(function.get("name"), str):
                parts.append(function["name"])
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                parts.append(arguments)
            elif isinstance(arguments, Mapping):
                parts.append(json.dumps(arguments, ensure_ascii=False, sort_keys=True))
    if not parts:
        raise ValueError(
            f"raw_visible[{index}] lacks original_text/content/tool-call text"
        )
    return "\n".join(parts)


def _draft_typed_parameters(tool_calls: Any) -> set[tuple[str, Any]]:
    if not _sequence(tool_calls):
        return set()
    for index, call in enumerate(tool_calls):
        if not isinstance(call, Mapping):
            raise ValueError(f"draft_tool_calls[{index}] must be a mapping")
        function = call.get("function", call)
        if not isinstance(function, Mapping) or "arguments" not in function:
            raise ValueError(f"draft_tool_calls[{index}] lacks function arguments")
        arguments = function["arguments"]
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as error:
                raise ValueError("draft tool arguments are not valid JSON") from error
        if not isinstance(arguments, Mapping):
            raise ValueError("draft tool arguments must decode to an object")
    return set(_protocol_typed_parameters(tool_calls))


def _supported_parameters(
    parameters: set[tuple[str, Any]], text: str
) -> set[tuple[str, Any]]:
    """Find typed parameters in serialized JSON or bounded ordinary text."""

    return set(_protocol_supported_parameters(parameters, text))


def _artifact_mapping(source: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactContractError(f"cannot load artifact {path}") from error
    if not isinstance(value, Mapping):
        raise ArtifactContractError("artifact root must be a JSON object")
    return dict(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _finite_vector(value: Any, name: str, expected: int | None = None) -> list[float]:
    if isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    if not _sequence(value):
        raise ValueError(f"{name} must be a numeric sequence")
    result = [_finite_number(item, name) for item in value]
    if expected is not None and len(result) != expected:
        raise ValueError(f"{name} must have {expected} values")
    return result


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _strict_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be boolean")
    return value


def _sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _unavailable(reason: str, missing: Sequence[str]) -> FeatureResult:
    return FeatureResult(False, None, tuple(sorted(set(missing))), reason)


__all__ = [
    "ARTIFACT_SCHEMA",
    "C1_FEATURE_NAMES",
    "C1_FEATURE_SCHEMA",
    "C4_FEATURE_NAMES",
    "C4_FEATURE_SCHEMA",
    "T01_FEATURE_NAMES",
    "T01_FEATURE_SCHEMA",
    "ArtifactContractError",
    "C1RiskArtifact",
    "C1RiskSelector",
    "C4GainArtifact",
    "C4GainSelector",
    "FeatureResult",
    "PredictionResult",
    "SelectionResult",
    "T01RelevanceArtifact",
    "T01Selector",
    "artifact_sha256",
    "extract_c1_raw_features",
    "extract_c4_features",
    "extract_t01_features",
    "load_set_model",
    "load_set_selector",
    "tokenizer_contract",
    "validate_selector_score_models",
]
