"""CPU tests for bounded recovery query, count, and decision selectors."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "benchmarks"), str(ROOT / "python")]

from memory_runtime.recovery.experiment_config import parse_gp_config
from memory_runtime.recovery.evidence_units import build_catalog
from memory_runtime.recovery.gate import evaluate_gate
from memory_runtime.recovery.selection import (
    export_candidate_scorer,
    load_candidate_scorer,
    prepare_selection_dependencies,
    select_candidates,
    train_candidate_scorer,
)
from memory_runtime.recovery.source import rank_visible_source_events
from memory_runtime.recovery.set_selectors import select_evidence_set
from memory_runtime.recovery.selection_backends import (
    OpenAICompatibleSelectionBackend,
    SelectionBackendError,
    make_backends,
)
from memory_runtime.policy import _event_text
from history_memory.events import EventStore


@dataclass(frozen=True)
class Unit:
    unit_id: str
    text: str
    source_indices: tuple[int, ...]
    source_type: str = "event"
    metadata: dict = field(default_factory=dict)


def unit(unit_id, text, index, source_type="event"):
    return Unit(unit_id, text, (index,), source_type)


class Backend:
    def __init__(self, replies=None, vectors=None):
        self.replies = replies or {}
        self.vectors = vectors
        self.calls = []

    def chat(self, *, messages, purpose, config):
        self.calls.append(("chat", purpose, messages))
        return self.replies[purpose]

    def embed(self, *, texts, purpose, config):
        self.calls.append(("embed", purpose, list(texts)))
        return self.vectors


def config(**updates):
    return parse_gp_config(updates)


def select(catalog, *, gp=None, backend=None, goal="Find invoice ZX", draft="Use lookup"):
    return select_candidates(
        catalog,
        goal=goal,
        draft_text=draft,
        draft_tool_calls=[
            {"id": "d1", "type": "function", "function": {
                "name": "lookup_invoice", "arguments": '{"invoice":"ZX"}'}}
        ],
        config=gp or config(),
        backends=backend,
    )


def test_lexical_keeps_current_exact_overlap_ranking_and_original_objects():
    catalog = [
        unit("weak", "A note about invoice records.", 1),
        unit("best", "lookup_invoice returned invoice ZX", 2, "tool_event"),
        unit("none", "unrelated weather", 3),
    ]
    selected, receipt = select(catalog)
    assert selected == [catalog[1]] and selected[0] is catalog[1]
    assert receipt["selected_ids"] == ["best"]
    assert receipt["ranked_candidates"][0]["components"]["source_type_bonus"] == 8.0
    assert {row["unit_id"] for row in receipt["ranked_candidates"]} == {"weak", "best"}
    assert receipt["candidate_text_serialized_in_receipt"] is False


def test_event_units_match_current_event_selector_ranking_and_tool_bonus():
    messages = [
        {"role": "system", "content": "Use observed results."},
        {"role": "user", "content": "Prepare alpha."},
        {"role": "assistant", "tool_calls": [{
            "id": "target-wrapper",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"key":"alpha"}'},
        }]},
        {"role": "tool", "tool_call_id": "target-wrapper", "content": '{"value":"blue"}'},
        {"role": "user", "content": "target-wrapper is prose evidence."},
        {"role": "assistant", "content": "Acknowledged."},
        {"role": "user", "content": "Use target-wrapper and blue."},
    ]

    class Tokenizer:
        @staticmethod
        def encode(text, **kwargs):
            return list(map(ord, text))

    store = EventStore.from_messages("differential", messages)
    current_user = [event for event in store.events if event.kind == "user"][-1]
    eligible = {event.event_id for event in store.events}
    expected = rank_visible_source_events(
        store,
        current_user=current_user,
        draft_text="",
        draft_tool_calls=[],
        eligible=eligible,
        excluded=set(),
    )
    catalog = [
        candidate
        for candidate in build_catalog(store, Tokenizer(), "event")
        if candidate.event_id in eligible and candidate.event_id != current_user.event_id
    ]
    selected, receipt = select_candidates(
        catalog,
        goal=_event_text(store, current_user),
        draft_text="",
        draft_tool_calls=[],
        config=config(K=4, order="relevance"),
    )
    assert tuple(candidate.event_id for candidate in selected) == expected
    by_unit_id = {candidate.unit_id: candidate for candidate in catalog}
    tool_rows = [
        row for row in receipt["ranked_candidates"]
        if by_unit_id[row["unit_id"]].metadata["event_kind"] == "tool_event"
    ]
    assert tool_rows and tool_rows[0]["components"]["source_type_bonus"] == 8.0


def test_event_unit_llm_text_omits_transport_wrappers_and_ids():
    messages = [
        {"role": "assistant", "tool_calls": [{
            "id": "secret-transport-id",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"key":"alpha"}'},
        }]},
        {"role": "tool", "tool_call_id": "secret-transport-id", "content": '{"value":"blue"}'},
    ]

    class Tokenizer:
        @staticmethod
        def encode(text, **kwargs):
            return list(map(ord, text))

    store = EventStore.from_messages("llm-event", messages)
    catalog = build_catalog(store, Tokenizer(), "event")
    assert len(catalog) == 1
    backend = Backend(replies={
        "recovery_candidate_selection": json.dumps(
            {"selected_ids": [catalog[0].unit_id]}
        )
    })
    select_candidates(
        catalog,
        goal="Use blue.",
        draft_text="",
        draft_tool_calls=[],
        config=config(K="llm"),
        backends=backend,
    )
    payload = json.loads(backend.calls[0][2][1]["content"])
    candidate_text = payload["catalog"][0]["text"]
    assert "lookup" in candidate_text and "alpha" in candidate_text and "blue" in candidate_text
    assert "secret-transport-id" not in candidate_text
    assert "tool_call_id" not in candidate_text and '"role"' not in candidate_text


def test_dual_reports_task_and_held_draft_scores_separately():
    catalog = [
        unit("goal", "Find invoice ZX", 1),
        unit("draft", "lookup_invoice invoice ZX", 2),
    ]
    selected, receipt = select(catalog, gp=config(Q="dual", K=2))
    assert {item.unit_id for item in selected} == {"goal", "draft"}
    assert all(
        {"goal_lexical", "draft_lexical"} <= set(row["components"])
        for row in receipt["ranked_candidates"]
    )


def test_field_labels_and_tool_association_are_searchable_without_receipt_text():
    catalog = [
        Unit("file", '"F7"', (1,), "field", {
            "field_name": "file_id",
            "field_path": ["content", "records", 0, "file_id"],
            "tool_name": "lookup_file",
        }),
        Unit("title", '"F7"', (2,), "field", {"field_name": "title"}),
    ]
    selected, receipt = select_candidates(
        catalog,
        goal="Use file_id returned by lookup_file",
        draft_text="",
        draft_tool_calls=[],
        config=config(K=1),
    )
    assert [item.unit_id for item in selected] == ["file"]
    assert "F7" not in json.dumps(receipt)


def test_hybrid_requires_embeddings_and_can_reorder_semantic_candidate():
    catalog = [
        unit("lexical", "Find invoice ZX", 1),
        unit("semantic", "billing record", 2),
    ]
    gp = config(Q="hybrid", K=1)
    with pytest.raises(ValueError, match="embed"):
        prepare_selection_dependencies(gp)
    backend = Backend(vectors=[[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]])
    selected, receipt = select(catalog, gp=gp, backend=backend)
    assert [item.unit_id for item in selected] == ["semantic"]
    call = receipt["backend"]["calls"][0]
    assert call["capability"] == "embed"
    assert call["purpose"] == "hybrid_query_and_catalog"
    assert call["input_count"] == 3
    assert call["usage"]["total_tokens"] is None


def test_llm_rewrite_is_bounded_then_retrieves_by_lexical_score():
    catalog = [
        unit("match", "ticket ABC-9 owner", 1),
        unit("miss", "invoice ZX", 2),
    ]
    backend = Backend(replies={"query_rewrite": '{"query":"ticket ABC-9"}'})
    selected, receipt = select(
        catalog,
        gp=config(Q="llm_rewrite", K=1),
        backend=backend,
        goal="x" * 5000,
        draft="held",
    )
    assert [item.unit_id for item in selected] == ["match"]
    sent = json.loads(backend.calls[0][2][1]["content"])
    assert len(sent["goal"]) == 4096
    assert receipt["backend"]["calls"][0]["purpose"] == "query_rewrite"


@pytest.mark.parametrize("count", [1, 2, 4])
def test_fixed_k_selects_supported_counts(count):
    catalog = [unit(f"u{i}", f"invoice ZX {i}", i) for i in range(6)]
    selected, _ = select(catalog, gp=config(K=count))
    assert len(selected) == count


def test_threshold_selection_discloses_effective_threshold():
    catalog = [
        unit("strong", 'invoice "ZX" lookup_invoice', 1),
        unit("weak", "invoice", 2),
    ]
    selected, receipt = select(
        catalog, gp=config(K="threshold", selection_threshold=150.0)
    )
    assert [item.unit_id for item in selected] == ["strong"]
    assert receipt["effective_threshold"] == receipt["selection_threshold"] == 150.0


def test_llm_sees_only_bounded_catalog_and_selected_ids_are_validated():
    catalog = [unit(f"u{i}", "invoice ZX", i) for i in range(10)]
    backend = Backend(replies={
        "recovery_candidate_selection": '{"selected_ids":["u9","u8"]}'
    })
    selected, receipt = select(
        catalog, gp=config(K="llm", candidate_limit=8), backend=backend
    )
    assert [item.unit_id for item in selected] == ["u8", "u9"]
    prompt = json.loads(backend.calls[0][2][1]["content"])
    assert len(prompt["catalog"]) == 8
    assert set(receipt["selected_ids"]) <= set(receipt["bounded_catalog_ids"])

    bad = Backend(replies={
        "recovery_candidate_selection": '{"selected_ids":["fabricated"]}'
    })
    with pytest.raises(SelectionBackendError, match="outside bounded catalog"):
        select(catalog, gp=config(K="llm"), backend=bad)


def test_relevance_order_follows_rank_instead_of_llm_response_order():
    catalog = [unit("weak", "invoice", 1), unit("strong", "invoice ZX lookup_invoice", 2)]
    backend = Backend(replies={
        "recovery_candidate_selection": '{"selected_ids":["weak","strong"]}'
    })
    selected, _ = select(
        catalog, gp=config(K="llm", order="relevance"), backend=backend
    )
    assert [item.unit_id for item in selected] == ["strong", "weak"]


def test_llm_duplicate_ids_are_rejected():
    backend = Backend(replies={
        "recovery_candidate_selection": '{"selected_ids":["u1","u1"]}'
    })
    with pytest.raises(SelectionBackendError, match="duplicate"):
        select([unit("u1", "invoice ZX", 1)], gp=config(K="llm"), backend=backend)


def test_detector_llm_uses_llm_even_with_fixed_k_and_joint_llm_can_abstain():
    catalog = [unit("u1", "invoice ZX", 1)]
    detector_backend = Backend(replies={
        "recovery_candidate_selection": '{"selected_ids":["u1"]}'
    })
    selected, receipt = select(
        catalog, gp=config(D="detector_llm", K=1), backend=detector_backend
    )
    assert [item.unit_id for item in selected] == ["u1"]
    assert receipt["controller_mode"] == "detector_llm"

    joint_backend = Backend(replies={
        "joint_recovery_decision": '{"selected_ids":[]}'
    })
    selected, receipt = select(
        catalog, gp=config(D="joint_llm", K=4), backend=joint_backend
    )
    assert selected == [] and receipt["reason"] == "selector_abstained"


@pytest.mark.parametrize(
    ("shadow_features", "expected_ids", "expected_reason"),
    (
        (
            {
                "schema": "event-native-shadow-features-v1",
                "prefill": {
                    "status": "captured",
                    "layer": 34,
                    "position": {"kind": "prompt_last"},
                    "readout": "decoder_layer_output",
                    "hidden": [2.0],
                },
            },
            ("first",),
            "prefill_score_at_or_above_threshold",
        ),
        (
            {
                "schema": "event-native-shadow-features-v1",
                "prefill": {
                    "status": "captured",
                    "layer": 34,
                    "position": {"kind": "prompt_last"},
                    "readout": "decoder_layer_output",
                    "hidden": [-2.0],
                },
            },
            (),
            "prefill_score_below_threshold",
        ),
        (None, (), "shadow_features_schema_unavailable"),
        (
            {
                "schema": "event-native-shadow-features-v1",
                "prefill": {
                    "status": "captured",
                    "layer": 33,
                    "position": {"kind": "prompt_last"},
                    "readout": "decoder_layer_output",
                    "hidden": [2.0],
                },
            },
            (),
            "prefill_layer_mismatch",
        ),
    ),
)
def test_legacy_prefill_gate_never_falls_back_to_candidate_rule(
    shadow_features, expected_ids, expected_reason
):
    detector = {
        "gate": "prefill_linear_head",
        "prefill_head": {
            "feature": "prefill.prompt_last.decoder_layer_output",
            "layer": 34,
            "weights": [1.0],
            "bias": 0.0,
            "input_mean": [0.0],
            "input_scale": [1.0],
            "threshold": 0.5,
            "artifact_sha256": "a" * 64,
        },
    }
    gate = evaluate_gate(
        detector,
        session_id="task-1",
        decision_key="turn-0/step-0",
        shadow_features=shadow_features,
    )
    gp = parse_gp_config(
        {
            "selection_protocol": "evidence_sets_v1",
            "D": "candidate_rule",
            "Q": "lexical",
            "U": "tokens_1024",
            "set_selector": "legacy_prefill",
        }
    )
    chosen, receipt = select_evidence_set(
        {},
        [{"unit_id": "first"}, {"unit_id": "second"}],
        [(), ("first",), ("second",)],
        gp,
        models=object(),
        tokenizer=object(),
        legacy_gate=gate,
    )

    assert gate["reason"] == expected_reason
    assert chosen == expected_ids
    assert receipt["score_semantics"] == "frozen_legacy_prefill_failure_risk"
    if not expected_ids:
        assert receipt["action_id"] == 0


def test_legacy_prefill_rejects_threshold_override():
    with pytest.raises(ValueError, match="frozen base detector threshold"):
        parse_gp_config(
            {
                "selection_protocol": "evidence_sets_v1",
                "D": "candidate_rule",
                "Q": "lexical",
                "U": "tokens_1024",
                "set_selector": "legacy_prefill",
                "detector_threshold": 0.25,
            }
        )


def training_rows():
    return [
        {"goal": "find invoice ZX", "draft_text": "lookup invoice", "candidate": "invoice ZX result",
         "label": 1.0, "provenance": {"task_id": "t1"}},
        {"goal": "find invoice ZX", "draft_text": "lookup invoice", "candidate": "weather report",
         "label": 0.0, "provenance": {"task_id": "t1"}},
        {"goal": "find ticket AB", "draft_text": "lookup ticket", "candidate": "ticket AB owner",
         "label": 1.0, "provenance": {"task_id": "t2"}},
        {"goal": "find ticket AB", "draft_text": "lookup ticket", "candidate": "calendar events",
         "label": 0.0, "provenance": {"task_id": "t2"}},
        {"goal": "unknown", "draft_text": "", "candidate": "unknown", "label": "unknown",
         "label_status": "unknown", "provenance": {"task_id": "t3"}},
    ]


def test_supervised_scorer_trains_exports_loads_and_selects_without_risk_labels(tmp_path):
    artifact = train_candidate_scorer(
        training_rows(),
        label_kind="candidate_relevance",
        dataset_provenance={"dataset": "cpu-fixture", "split": "train"},
    )
    assert artifact["training_summary"] == {
        "known_labels_used": 4,
        "unknown_labels_excluded": 1,
        "label_source": "candidate_relevance",
        "risk_labels_used": False,
        "l2": 1.0,
    }
    path = export_candidate_scorer(artifact, tmp_path / "scorer.json")
    loaded = load_candidate_scorer(path)
    gp = config(D="supervised", K=1, candidate_scorer=str(path))
    catalog = [unit("good", "invoice ZX result", 1), unit("bad", "weather report", 2)]
    selected, receipt = select(catalog, gp=gp)
    assert [item.unit_id for item in selected] == ["good"]
    assert receipt["candidate_scorer"]["artifact_sha256"] == loaded["artifact_sha256"]
    assert receipt["candidate_scorer"]["dataset_provenance"]["dataset"] == "cpu-fixture"


def test_scorer_rejects_risk_labels_and_known_rows_without_provenance():
    with pytest.raises(ValueError, match="risk labels are forbidden"):
        train_candidate_scorer(
            training_rows(), label_kind="risk", dataset_provenance={"dataset": "x"}
        )
    rows = training_rows()
    rows[0] = {**rows[0], "provenance": None}
    with pytest.raises(ValueError, match="lacks provenance"):
        train_candidate_scorer(
            rows, label_kind="candidate_relevance", dataset_provenance={"dataset": "x"}
        )


def test_backend_config_forbids_inline_secret_and_reports_public_dependency_config(monkeypatch):
    gp = config(Q="hybrid", backend={
        "type": "openai_compatible",
        "base_url": "https://example.invalid/v1",
        "embedding_model": "embedding-model",
        "api_key_env": "SELECTION_TEST_KEY",
    })
    backend = make_backends(gp)
    assert isinstance(backend, OpenAICompatibleSelectionBackend)
    assert backend.public_config()["api_key_env"] == "SELECTION_TEST_KEY"
    assert "api_key" not in backend.public_config()
    with pytest.raises(ValueError, match="secrets must be supplied"):
        make_backends(config(Q="hybrid", backend={
            "type": "openai_compatible",
            "base_url": "https://example.invalid/v1",
            "embedding_model": "embedding-model",
            "api_key": "must-not-be-accepted",
        }))


def test_openai_compatible_backend_records_actual_usage_and_latency(monkeypatch):
    monkeypatch.setenv("SELECTION_TEST_KEY", "private-test-value")
    observed = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({
                "choices": [{"message": {"content": '{"selected_ids":[]}'}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
            }).encode()

    def urlopen(request, timeout):
        observed["authorization"] = request.get_header("Authorization")
        observed["timeout"] = timeout
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    backend = OpenAICompatibleSelectionBackend({
        "type": "openai_compatible",
        "base_url": "https://example.invalid/v1",
        "chat_model": "selector-model",
        "api_key_env": "SELECTION_TEST_KEY",
    })
    assert backend.chat(
        messages=[{"role": "user", "content": "select"}],
        purpose="test",
        config={},
    ) == '{"selected_ids":[]}'
    receipt = backend.drain_receipts()[0]
    assert receipt["usage"] == {
        "prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14
    }
    assert receipt["latency_seconds"] >= 0
    assert observed == {"authorization": "Bearer private-test-value", "timeout": 30.0}
    assert "private-test-value" not in json.dumps(backend.public_config())


def test_configured_backend_requires_models_for_used_capabilities():
    gp = config(K="llm", backend={
        "type": "openai_compatible", "base_url": "http://localhost:9999/v1"
    })
    with pytest.raises(ValueError, match="chat_model"):
        prepare_selection_dependencies(gp)


def test_candidate_limit_restricts_ranked_receipt_without_dropping_catalog_count():
    catalog = [unit(f"u{i}", "invoice ZX", i) for i in range(12)]
    _, receipt = select(catalog, gp=config(K=4, candidate_limit=3))
    assert receipt["catalog_count"] == 12
    assert len(receipt["bounded_catalog_ids"]) == 3
    assert len(receipt["selected_ids"]) == 3
