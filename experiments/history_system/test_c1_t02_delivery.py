from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
for runtime_path in (HERE / "runtime" / "python", HERE / "runtime" / "benchmarks"):
    if str(runtime_path) not in sys.path:
        sys.path.insert(0, str(runtime_path))

import run_c1
from memory_runtime.recovery.set_models import (
    PredictionResult,
    artifact_sha256,
    load_set_selector,
)


BUNDLED_ARTIFACT = HERE / "artifacts" / "c1_risk.t02_v1.json"
BUNDLED_ARTIFACT_FILE_SHA256 = (
    "18a11f73aa1f7d4b0add86eed66ae9e5e129ea4bdfbe0dfad23faf4f7d2fb4ab"
)
LEGACY_HEAD_SHA256 = (
    "4bd01ef68206cbc3902da49dd0ba288b41f9c92aa8f5a2473d5ffdfdb9d97c57"
)
LEGACY_HEAD_THRESHOLD = 0.997483851175923


class C1T02DeliveryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temporary.name)
        self.checkpoint = self.tmp_path / "checkpoint"
        self.checkpoint.mkdir()
        self.checkpoint_config = self.checkpoint / "config.json"
        self.checkpoint_config.write_text("{}\n", encoding="utf-8")
        self.embedding = self.tmp_path / "embedding"
        self.embedding.mkdir()
        self.binding_calls: list[tuple[dict, Path]] = []

    def tearDown(self):
        self.temporary.cleanup()

    def args(
        self,
        *,
        detector: str = "t02_risk",
        selector_artifact: Path | None = None,
    ) -> argparse.Namespace:
        return argparse.Namespace(
            method="proposed",
            detector=detector,
            selector_artifact=selector_artifact,
            selector_threshold=0.5,
            checkpoint=self.checkpoint,
            embedding_model=self.embedding,
            embedding_device="cpu",
        )

    def selected_config(self) -> dict:
        selected = copy.deepcopy(run_c1.current.load_config())
        selected["checkpoint_selection"]["config_sha256"] = hashlib.sha256(
            self.checkpoint_config.read_bytes()
        ).hexdigest()
        return selected

    def build_profile(self, args: argparse.Namespace) -> tuple[dict, dict]:
        selected = self.selected_config()

        def bind_fixture(artifact: dict, checkpoint: Path) -> tuple[dict, dict]:
            self.binding_calls.append((copy.deepcopy(artifact), checkpoint))
            digest = artifact["artifact_sha256"]
            return copy.deepcopy(artifact), {
                "schema": "c2kv-c1-artifact-binding-receipt-v1",
                "status": "fixture_identity",
                "checkpoint": str(checkpoint.resolve()),
                "source_artifact_sha256": digest,
                "derived_artifact_sha256": digest,
            }

        with (
            mock.patch.object(run_c1.current, "load_config", return_value=selected),
            mock.patch.object(run_c1.evidence_sets, "load_config", return_value=selected),
            mock.patch.object(
                run_c1,
                "bind_risk_artifact",
                side_effect=bind_fixture,
            ),
        ):
            return run_c1.build_profile(args)

    def write_artifact(self, name: str, mutate) -> Path:
        artifact = json.loads(BUNDLED_ARTIFACT.read_text(encoding="utf-8"))
        mutate(artifact)
        artifact["artifact_sha256"] = artifact_sha256(artifact)
        path = self.tmp_path / name
        path.write_text(
            json.dumps(artifact, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return path

    def test_default_loads_bundled_real_t02_artifact(self):
        parsed = run_c1.build_parser().parse_args(
            [
                "--checkpoint", str(self.checkpoint),
                "--sglang-backend-url", "http://127.0.0.1:38800",
                "--embedding-model", str(self.embedding),
                "--out", str(self.tmp_path / "out"),
            ]
        )
        self.assertEqual(parsed.detector, "t02_risk")
        self.assertEqual(
            hashlib.sha256(BUNDLED_ARTIFACT.read_bytes()).hexdigest(),
            BUNDLED_ARTIFACT_FILE_SHA256,
        )

        controller, profile = self.build_profile(parsed)

        artifact = controller["gp_experiments"]["selector_artifact"]
        self.assertEqual(controller["gp_experiments"]["set_selector"], "risk")
        self.assertEqual(artifact["model_kind"], "c1_risk_logistic")
        self.assertEqual(load_set_selector(artifact).kind, "risk")
        self.assertEqual(profile["detector"], "t02_risk")
        self.assertEqual(Path(profile["selector_artifact"]), BUNDLED_ARTIFACT)
        self.assertEqual(
            profile["selector_artifact_sha256"], BUNDLED_ARTIFACT_FILE_SHA256
        )
        self.assertEqual(profile["selector_threshold"], 0.5)
        self.assertEqual(profile["selector_artifact_binding"]["status"], "fixture_identity")
        self.assertEqual(len(self.binding_calls), 1)

    def test_explicit_artifact_override_remains_supported(self):
        override = self.tmp_path / "override.json"
        override.write_bytes(BUNDLED_ARTIFACT.read_bytes())

        controller, profile = self.build_profile(
            self.args(selector_artifact=override)
        )

        self.assertEqual(
            controller["gp_experiments"]["selector_artifact"]["model_kind"],
            "c1_risk_logistic",
        )
        self.assertEqual(Path(profile["selector_artifact"]), override)
        self.assertEqual(
            profile["selector_artifact_sha256"], BUNDLED_ARTIFACT_FILE_SHA256
        )
        self.assertEqual(profile["selector_artifact_binding"]["status"], "fixture_identity")
        self.assertEqual(len(self.binding_calls), 1)

    def test_legacy_prefill_is_explicit_and_keeps_frozen_head(self):
        controller, profile = self.build_profile(self.args(detector="legacy_prefill"))

        gp = controller["gp_experiments"]
        head = controller["post_draft_recovery"]["prefill_head"]
        self.assertEqual(gp["set_selector"], "legacy_prefill")
        self.assertNotIn("selector_artifact", gp)
        self.assertEqual(controller["post_draft_recovery"]["gate"], "prefill_linear_head")
        self.assertEqual(head["artifact_sha256"], LEGACY_HEAD_SHA256)
        self.assertEqual(head["threshold"], LEGACY_HEAD_THRESHOLD)
        self.assertEqual(profile["detector"], "legacy_prefill")

        with self.assertRaisesRegex(ValueError, "only used with --detector t02_risk"):
            self.build_profile(
                self.args(
                    detector="legacy_prefill",
                    selector_artifact=BUNDLED_ARTIFACT,
                )
            )

    def test_explicit_missing_artifact_fails(self):
        missing = self.tmp_path / "missing.json"
        with self.assertRaises(FileNotFoundError):
            self.build_profile(self.args(selector_artifact=missing))

    def test_wrong_artifact_kind_fails(self):
        wrong = self.write_artifact(
            "wrong-kind.json",
            lambda artifact: artifact.update(model_kind="not_a_c1_model"),
        )
        with self.assertRaisesRegex(ValueError, "model_kind|model kind|incompatible"):
            self.build_profile(self.args(selector_artifact=wrong))
        self.assertEqual(self.binding_calls, [])

    def test_wrong_feature_contract_fails(self):
        def mutate(artifact):
            artifact["feature_contract"]["schema"] = "not-c1-pca8"

        wrong = self.write_artifact("wrong-feature.json", mutate)
        with self.assertRaisesRegex(ValueError, "feature contract"):
            self.build_profile(self.args(selector_artifact=wrong))
        self.assertEqual(self.binding_calls, [])

    def test_runtime_rejects_wrong_prefill_model_binding(self):
        bundled = json.loads(BUNDLED_ARTIFACT.read_text(encoding="utf-8"))

        def mutate(artifact):
            model = artifact["feature_contract"]["prefill_contract"]["bindings"]["model"]
            model["model_path"] = "/wrong/checkpoint"

        wrong = self.write_artifact("wrong-binding.json", mutate)
        selector = load_set_selector(wrong)
        context = {
            "prefill_hidden": [0.0] * selector.artifact["components"]["hidden_dimension"],
            "prefill_contract": bundled["feature_contract"]["prefill_contract"],
            "draft_logprobs": [-1.0],
            "is_stop": False,
            "parse_ok": True,
        }

        result = selector.select(
            context,
            [{"unit_id": "first"}],
            [(), ("first",)],
            threshold=0.5,
        )

        self.assertFalse(result.available)
        self.assertEqual(result.selected_ids, ())
        self.assertEqual(result.reason, "prefill contract mismatch")

    def test_risk_threshold_is_strict_and_empty_selection_does_not_mutate_inputs(self):
        selector = load_set_selector(BUNDLED_ARTIFACT)
        context: dict = {}
        candidates = [{"unit_id": "first"}]
        actions = [(), ("first",)]
        original_context = copy.deepcopy(context)
        original_candidates = copy.deepcopy(candidates)
        original_actions = copy.deepcopy(actions)

        with mock.patch.object(
            selector.model,
            "predict_risk",
            return_value=PredictionResult(True, 0.5),
        ):
            at_threshold = selector.select(context, candidates, actions, threshold=0.5)
        self.assertEqual(at_threshold.selected_ids, ())
        self.assertEqual(at_threshold.reason, "risk_not_above_threshold")

        with mock.patch.object(
            selector.model,
            "predict_risk",
            return_value=PredictionResult(True, math.nextafter(0.5, 1.0)),
        ):
            above_threshold = selector.select(context, candidates, actions, threshold=0.5)
        self.assertEqual(above_threshold.selected_ids, ("first",))
        self.assertEqual(context, original_context)
        self.assertEqual(candidates, original_candidates)
        self.assertEqual(actions, original_actions)

        empty_candidates: list[dict] = []
        empty_actions = [()]
        with mock.patch.object(
            selector.model,
            "predict_risk",
            return_value=PredictionResult(True, 0.9),
        ):
            empty = selector.select({}, empty_candidates, empty_actions, threshold=0.5)
        self.assertEqual(empty.selected_ids, ())
        self.assertEqual(empty_candidates, [])
        self.assertEqual(empty_actions, [()])

    def test_summary_counts_only_actual_risk_scores_and_tracks_unavailable(self):
        task_out = self.tmp_path / "task"
        server = task_out / "server"
        server.mkdir(parents=True)
        scored = {
            "status": "recover",
            "appended_unit_count": 1,
            "appended_units": [{"token_count": 7}],
            "gate": {"type": "risk", "triggered": True, "reason": "risk_triggered"},
            "selection": {
                "schema": "recovery-set-selector-v1",
                "selector": "risk",
                "available": True,
                "reason": "selected_legal_evidence_set",
                "score": 0.75,
                "score_semantics": "current_turn_failure_risk",
            },
        }
        no_feasible = {
            "status": "no_recovery",
            "gate": {
                "type": "risk",
                "triggered": False,
                "reason": "no_feasible_evidence_sets",
            },
            "selection": {
                "schema": "recovery-set-selector-v1",
                "selector": "risk",
                "available": True,
                "reason": "no_feasible_evidence_sets",
                "score": 0.0,
                "score_semantics": None,
            },
        }
        unavailable = {
            "status": "no_recovery",
            "gate": {"type": "risk", "triggered": False, "reason": "selector_unavailable"},
            "selection": {
                "schema": "recovery-set-selector-v1",
                "selector": "risk",
                "available": False,
                "reason": "prefill contract mismatch",
                "score": None,
                "score_semantics": "current_turn_failure_risk",
            },
        }
        records = [
            {"generation_trace": [], "exact_recovery": row}
            for row in (scored, no_feasible, unavailable)
        ]
        (server / "steps.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
        )
        official = {
            "n": 1,
            "task_rows": [{
                "semantic_score": 1.0,
                "normal_termination": True,
                "protocol_legal": True,
            }],
        }

        summary = run_c1.summarize_task("tau2", "0", task_out, official, 1.0)

        self.assertEqual(summary["risk_detector_scores"], 1)
        self.assertEqual(summary["risk_detector_unavailable"], 1)
        self.assertEqual(summary["detector_calls"], 2)
        self.assertEqual(summary["detector_trigger_count"], 1)
        self.assertEqual(summary["prefill_detector_scores"], 0)


if __name__ == "__main__":
    unittest.main()
