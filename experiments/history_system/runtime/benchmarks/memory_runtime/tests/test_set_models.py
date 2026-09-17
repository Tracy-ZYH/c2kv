"""CPU-only contracts for C1/C4/T01 recovery-set models."""

from __future__ import annotations

import sys
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "benchmarks"), str(ROOT / "python")]

from memory_runtime.recovery.set_models import (  # noqa: E402
    C1RiskSelector,
    C4_FEATURE_NAMES,
    C4GainSelector,
    artifact_sha256,
    extract_c4_features,
    load_set_selector,
    tokenizer_contract,
    validate_selector_score_models,
)
from memory_runtime.recovery.set_training import (  # noqa: E402
    prepare_c4_features,
    train_c1_model,
    train_c4_models,
    train_t01_calibrator,
)


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return list(text)


TOKENIZER = CharacterTokenizer()
PREFILL_CONTRACT = {
    "layer": 20,
    "readout": "last_visible_token",
    "position_kind": "prefill_end",
    "bindings": {"model": "fixture-model", "tokenizer": "character-v1"},
}
SCORE_MODEL_CONTRACT = {
    "embedding": {"model": "fixture-embedding", "revision": "sha-embed"},
    "reranker": {"model": "fixture-reranker", "revision": "sha-rerank"},
}


def test_model_binding_allows_device_placement_but_rejects_revision_drift():
    expected = copy.deepcopy(SCORE_MODEL_CONTRACT)
    observed = copy.deepcopy(SCORE_MODEL_CONTRACT)
    for role in expected:
        expected[role]["device"] = "npu:0"
        observed[role]["device"] = "npu:3"
    selector = SimpleNamespace(score_model_contract=expected)
    models = SimpleNamespace(public_config=lambda: observed)
    validate_selector_score_models(selector, models)
    observed["embedding"]["revision"] = "different"
    with pytest.raises(ValueError, match="differs from fitted artifact"):
        validate_selector_score_models(selector, models)


class FrozenScoreModels:
    def __init__(self):
        self.receipts = []

    def rerank(self, query, documents):
        self.receipts.append({"capability": "rerank", "input_count": len(documents)})
        return [0.8 - index * 0.1 for index in range(len(documents))]

    def rerank_retrieval_candidates(
        self, *, task, draft, documents, overflow_policy
    ):
        self.receipts.append(
            {"capability": "rerank_input_limit", "overflow_policy": overflow_policy}
        )
        return self.rerank(task + "\n" + draft, documents)

    def embed(self, *, texts, purpose, config):
        del config
        self.receipts.append(
            {"capability": "embed", "purpose": purpose, "input_count": len(texts)}
        )
        return [[float(len(text) + 1), 1.0] for text in texts]

    def drain_receipts(self):
        result, self.receipts = self.receipts, []
        return result

    def public_config(self):
        return {
            "embedding": {"model": "fixture-embedding", "revision": "sha-embed"},
            "reranker": {"model": "fixture-reranker", "revision": "sha-rerank"},
            "selector": {"model": "unused"},
        }


def context(index=0, *, hidden=True):
    value = {
        "goal": "use exact record",
        "last_action_observation": [],
        "raw_visible": [{"content": "already seen"}],
        "raw_source_ids": ["current"],
        "draft_logprobs": [-0.1 - index * 0.01, -0.3],
        "draft_text": "call tool",
        "draft_tool_calls": [
            {"function": {"name": "lookup", "arguments": '{"id":7,"flag":true}'}}
        ],
        "parse_ok": True,
        "is_stop": False,
    }
    if hidden:
        value["prefill_hidden"] = [float(index + offset) for offset in range(8)]
        value["prefill_contract"] = PREFILL_CONTRACT
    return value


def candidate(unit_id, score, value=7, *, event_id=None):
    text = '{"id":%d,"name":"novel"}' % value
    return {
        "unit_id": unit_id,
        "event_id": event_id or f"event-{unit_id}",
        "text": text,
        "token_count": len(TOKENIZER.encode(text)),
        "provenance": {"source": f"fixture-{unit_id}"},
        "reranker_score": score,
        "task_similarity": score / 2,
        "draft_similarity": score / 3,
        "tool_name_match": score > 0.5,
    }


def test_c4_fixed18_uses_typed_parameters_and_exact_original_tokens():
    result = extract_c4_features(
        context(hidden=False),
        [candidate("u1", 0.8)],
        ("u1",),
        tokenizer=TOKENIZER,
    )

    assert result.available is True
    assert result.vector is not None and len(result.vector) == len(C4_FEATURE_NAMES) == 18
    assert result.vector[C4_FEATURE_NAMES.index("draft_parameter_coverage_fraction")] == 1.0
    assert 0.0 < result.vector[C4_FEATURE_NAMES.index("raw_token_novelty_fraction")] <= 1.0
    assert result.details["matched_parameter_count"] == 1  # bool is ignored


def test_c4_missing_model_feature_is_unavailable_not_zero_imputed():
    bad = candidate("u1", 0.8)
    bad["reranker_score"] = None
    result = extract_c4_features(
        context(hidden=False), [bad], ("u1",), tokenizer=TOKENIZER
    )
    assert result.available is False
    assert result.vector is None
    assert "reranker_score" in result.reason


def _manual_c4_artifact():
    weights = [0.0] * len(C4_FEATURE_NAMES)
    weights[C4_FEATURE_NAMES.index("reranker_score_mean")] = 1.0
    artifact = {
        "schema": "c2kv-recovery-set-model-v1",
        "model_kind": "c4_gain_turn",
        "target": "delta_turn",
        "feature_contract": {
            "schema": "c2kv-c4-fixed18-v1",
            "feature_names": list(C4_FEATURE_NAMES),
            "tokenizer_contract": tokenizer_contract(TOKENIZER),
            "score_model_contract": SCORE_MODEL_CONTRACT,
            "semantic_query_overflow_policy": "error",
        },
        "components": {
            "scaler": {
                "mean": [0.0] * len(weights),
                "scale": [1.0] * len(weights),
            },
            "weights": weights,
            "intercept": 0.0,
        },
        "provenance": {"fixture": "manual"},
    }
    artifact["artifact_sha256"] = artifact_sha256(artifact)
    return artifact


def test_c4_selector_only_scores_root_legal_actions_and_ties_choose_fewer_ids():
    candidates = [candidate("high", 0.9), candidate("low", 0.1)]
    selector = C4GainSelector(_manual_c4_artifact())
    result = selector.select(
        context(hidden=False),
        candidates,
        legal_actions=[(), ("high",), ("low",), ("high", "low")],
        tokenizer=TOKENIZER,
        delta=0.0,
    )
    assert result.available is True
    assert result.selected_ids == ("high",)
    assert selector.kind == "gain_turn"
    validate_selector_score_models(
        selector, FrozenScoreModels(), semantic_query_overflow_policy="error"
    )

    abstained = selector.select(
        context(hidden=False),
        candidates,
        legal_actions=[(), ("high",)],
        tokenizer=TOKENIZER,
        delta=1.0,
    )
    assert abstained.selected_ids == ()
    assert abstained.reason == "score_not_above_delta"

    class DifferentTokenizer:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return text.split()

    mismatch = selector.select(
        context(hidden=False),
        candidates,
        legal_actions=[(), ("high",)],
        tokenizer=DifferentTokenizer(),
    )
    assert mismatch.available is False
    assert mismatch.reason == "tokenizer contract mismatch"


def _c4_training_rows():
    rows = []
    for index in range(12):
        group = f"group-{index % 3}"
        rows.append(
            {
                "state_id": f"state-{index}",
                "task_group_id": group,
                "context": context(index, hidden=False),
                "candidates": [candidate(f"u-{index}", 0.1 + index / 20)],
                "chosen_ids": [f"u-{index}"],
                "delta_turn": (-1, 0, 1)[index % 3],
                "delta_task": None if index == 0 else (-1, 0, 1)[(index + 1) % 3],
                "provenance": {"state": index},
            }
        )
    return rows


def test_c4_grouped_training_exports_two_portable_artifacts(tmp_path):
    artifacts = train_c4_models(
        _c4_training_rows(),
        tokenizer=TOKENIZER,
        provenance={"dataset": "synthetic-cpu"},
        score_model_contract=SCORE_MODEL_CONTRACT,
        output_dir=tmp_path,
    )

    assert set(artifacts) == {"delta_turn", "delta_task"}
    assert artifacts["delta_turn"]["fit"]["selected_alpha"] in {1.0, 10.0, 100.0}
    assert artifacts["delta_task"]["fit"]["unknown_label_count"] == 1
    selector = load_set_selector(tmp_path / "c4_gain_turn.json")
    assert selector.semantic_query_overflow_policy == "error"


def test_c4_feature_contract_rejects_inference_overflow_policy_drift(tmp_path):
    artifacts = train_c4_models(
        _c4_training_rows(),
        tokenizer=TOKENIZER,
        provenance={"dataset": "synthetic-cpu"},
        score_model_contract=SCORE_MODEL_CONTRACT,
        semantic_query_overflow_policy="task_head_tail_preserve_draft_v1",
        output_dir=tmp_path,
    )
    assert artifacts["delta_turn"]["feature_contract"][
        "semantic_query_overflow_policy"
    ] == "task_head_tail_preserve_draft_v1"
    selector = load_set_selector(tmp_path / "c4_gain_turn.json")
    validate_selector_score_models(
        selector,
        FrozenScoreModels(),
        semantic_query_overflow_policy="task_head_tail_preserve_draft_v1",
    )
    with pytest.raises(ValueError, match="differs from fitted artifact"):
        validate_selector_score_models(
            selector,
            FrozenScoreModels(),
            semantic_query_overflow_policy="error",
        )
    assert selector.kind == "gain_turn"
    prediction = selector.select(
        context(hidden=False),
        [candidate("online", 0.7)],
        legal_actions=[(), ("online",)],
        tokenizer=TOKENIZER,
    )
    assert prediction.available is True


def _c1_rows():
    rows = []
    for index in range(18):
        rows.append(
            {
                "state_id": f"risk-{index}",
                "task_group_id": f"group-{index % 3}",
                "context": context(index),
                "c1_risk_label": index % 2,
                "provenance": {"state": index},
            }
        )
    return rows


def _t02_rows():
    rows = []
    for index in range(19):
        base = context(index)
        first = candidate(f"t02-{index}-a", 0.2 + index / 100)
        second = candidate(f"t02-{index}-b", 0.7 - index / 100, value=8)
        candidates = []
        for rank, value in enumerate((first, second)):
            normalized = dict(value)
            normalized["candidate_id"] = normalized.pop("unit_id")
            normalized["source_id"] = normalized.pop("event_id")
            normalized.update({"rank": rank, "feasible": True})
            candidates.append(normalized)
        q = {
            key: value
            for key, value in base.items()
            if key not in {"draft_logprobs", "draft_text", "draft_tool_calls", "parse_ok", "is_stop"}
        }
        draft = {
            "token_logprobs": base["draft_logprobs"],
            "text": base["draft_text"],
            "tool_calls": base["draft_tool_calls"],
            "parse_ok": base["parse_ok"],
            "kind": "call",
        }
        first_id = candidates[0]["candidate_id"]
        second_id = candidates[1]["candidate_id"]
        labels = {
            "A1": {
                "candidate_ids": [first_id],
                "delta_turn": (-1, 0, 1)[index % 3],
                "delta_task": None if index == 0 else (-1, 0, 1)[(index + 1) % 3],
                "turn_label_status": "known",
                "task_label_status": "unknown" if index == 0 else "known",
            },
            "A2": {
                "candidate_ids": [second_id],
                "delta_turn": (-1, 0, 1)[(index + 1) % 3],
                "delta_task": (-1, 0, 1)[index % 3],
                "turn_label_status": "known",
                "task_label_status": "known",
            },
        }
        rows.append(
            {
                "schema": "t02-labeled-state-v1",
                "state_id": f"t02-state-{index}",
                "task_group_id": f"group-{index % 3}" if index < 18 else "calibration-group",
                "decision_key": f"turn-{index}/step-0",
                "split": "train" if index < 18 else "calibration",
                "q": q,
                "draft": draft,
                "candidates": candidates,
                "allowed_actions": [
                    {"action_id": "none", "candidate_ids": []},
                    {"action_id": "a1", "candidate_ids": [first_id]},
                    {"action_id": "a2", "candidate_ids": [second_id]},
                ],
                "branches": [
                    {"branch_id": "A0", "candidate_ids": []},
                    {"branch_id": "A1", "candidate_ids": [first_id]},
                    {"branch_id": "A2", "candidate_ids": [second_id]},
                ],
                "outcomes": {"A0": {}, "A1": {}, "A2": {}},
                "labels": labels,
                "c1_risk_label": index % 2,
                "c1_label_status": "known",
            }
        )
    return rows


def test_t02_labeled_rows_import_without_zero_filling_or_group_leakage():
    rows = _t02_rows()
    c4 = train_c4_models(
        rows,
        tokenizer=TOKENIZER,
        provenance={"dataset": "t02-synthetic"},
        score_model_contract=SCORE_MODEL_CONTRACT,
    )
    task_fit = c4["delta_task"]["fit"]
    assert task_fit["known_example_count"] == 35
    assert task_fit["unknown_label_count"] == 1
    assert task_fit["calibration_state_count_excluded"] == 1
    for candidate in task_fit["cv"]:
        for fold in candidate["folds"]:
            assert set(fold["train_groups"]).isdisjoint(fold["validation_groups"])

    c1 = train_c1_model(
        rows,
        provenance={"dataset": "t02-synthetic"},
        prefill_contract=PREFILL_CONTRACT,
    )
    assert c1["fit"]["known_example_count"] == 18
    assert c1["fit"]["calibration_state_count_excluded"] == 1
    for candidate in c1["fit"]["cv"]:
        for fold in candidate["folds"]:
            assert set(fold["train_groups"]).isdisjoint(fold["validation_groups"])


def test_t02_missing_label_field_is_rejected_instead_of_imputed():
    rows = _t02_rows()
    del rows[0]["labels"]["A1"]["delta_turn"]
    with pytest.raises(ValueError, match="lacks delta_turn"):
        train_c4_models(
            rows,
            tokenizer=TOKENIZER,
            provenance={"dataset": "t02-synthetic"},
            score_model_contract=SCORE_MODEL_CONTRACT,
        )


def test_training_rejects_task_group_overlap_before_feature_enrichment():
    rows = _t02_rows()
    rows[-1]["task_group_id"] = rows[0]["task_group_id"]
    models = FrozenScoreModels()
    with pytest.raises(ValueError, match="crosses train/calibration"):
        prepare_c4_features(rows, models=models)
    assert models.receipts == []


@pytest.mark.parametrize("label,status", [(None, "known"), (1, "unknown")])
def test_training_rejects_contradictory_label_status(label, status):
    rows = _t02_rows()
    rows[0]["labels"]["A1"].update(delta_turn=label, turn_label_status=status)
    with pytest.raises(ValueError, match="label/status mismatch"):
        train_c4_models(rows, tokenizer=TOKENIZER, provenance={"dataset": "synthetic"},
            score_model_contract=SCORE_MODEL_CONTRACT)
    rows = _t02_rows()
    rows[0].update(c1_risk_label=label, c1_label_status=status)
    with pytest.raises(ValueError, match="label/status mismatch"):
        train_c1_model(rows, provenance={"dataset": "synthetic"}, prefill_contract=PREFILL_CONTRACT)


def test_offline_enrichment_excludes_calibration_before_frozen_model_calls():
    rows = _t02_rows()
    for row in rows:
        for item in row["candidates"]:
            item.pop("reranker_score")
            item.pop("task_similarity")
            item.pop("draft_similarity")
    enriched, receipt = prepare_c4_features(
        rows,
        models=FrozenScoreModels(),
        semantic_query_overflow_policy="task_head_tail_preserve_draft_v1",
    )
    assert len(enriched) == 18
    assert receipt["calibration_state_count_excluded_before_enrichment"] == 1
    assert receipt["labels_or_outcomes_passed_to_models"] is False
    assert len(receipt["backend_receipts"]) == 18 * 4
    assert sum(
        row["capability"] == "rerank_input_limit"
        for row in receipt["backend_receipts"]
    ) == 18
    assert receipt["semantic_query_overflow_policy"] == "task_head_tail_preserve_draft_v1"
    assert all(
        row.get("overflow_policy") == "task_head_tail_preserve_draft_v1"
        for row in receipt["backend_receipts"]
        if row["capability"] == "rerank_input_limit"
    )
    for row in enriched:
        for item in row["candidates"]:
            assert 0.0 <= item["reranker_score"] <= 1.0
            assert "task_similarity" in item and "draft_similarity" in item


def test_c1_pca8_training_and_prefill_contract_abstention(tmp_path):
    path = tmp_path / "c1.json"
    artifact = train_c1_model(
        _c1_rows(),
        provenance={"dataset": "synthetic-cpu"},
        prefill_contract=PREFILL_CONTRACT,
        output_path=path,
    )
    assert artifact["components"]["pca"]["n_components"] == 8
    assert artifact["fit"]["selected_c"] in {0.01, 0.1, 1.0}

    selector = load_set_selector(path)
    candidates = [candidate("first", 0.5)]
    selected = selector.select(
        context(20), candidates, legal_actions=[(), ("first",)], threshold=0.0
    )
    assert selected.available is True
    assert selected.selected_ids == ("first",)
    assert selector.kind == "risk"

    mismatched = context(20)
    mismatched["prefill_contract"] = {**PREFILL_CONTRACT, "layer": 21}
    abstained = selector.select(
        mismatched, candidates, legal_actions=[(), ("first",)], threshold=0.0
    )
    assert abstained.available is False
    assert abstained.selected_ids == ()
    assert abstained.reason == "prefill contract mismatch"


def test_t01_unknown_labels_are_excluded_and_model_is_portable(tmp_path):
    rows = []
    for index in range(13):
        rows.append(
            {
                "task_group_id": f"group-{index % 3}",
                "candidate": candidate(f"t01-{index}", 0.05 + index / 20),
                "relevance_label": None if index == 12 else index % 2,
                "provenance": {"row": index},
            }
        )
    path = tmp_path / "t01.json"
    artifact = train_t01_calibrator(
        rows,
        provenance={"dataset": "weak-supervision-cpu"},
        score_model_contract=SCORE_MODEL_CONTRACT,
        output_path=path,
    )
    assert artifact["fit"]["unknown_label_count"] == 1
    assert load_set_selector(path).kind == "reranker_calibrator"


def test_c1_constructor_rejects_tampered_artifact():
    artifact = _manual_c4_artifact()
    artifact["components"]["intercept"] = 99.0
    with pytest.raises(ValueError, match="SHA-256"):
        C4GainSelector(artifact)
