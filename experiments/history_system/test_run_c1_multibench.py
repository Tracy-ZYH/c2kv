from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import run_c1
import runner


CHECKPOINT = Path("/home/zhuyuhan/project/model/arm-C/checkpoint-1000")
EMBEDDING = Path("/home/liuyancheng/c2kv-evidence-sets-20260916/models/Qwen3-Embedding-0.6B")
BFCL = Path("/home/zhuyuhan/project/bfcl-c2kv")
TAU2 = Path("/home/zhuyuhan/benchmarks/tau2")
TOOLSANDBOX = Path("/home/zhuyuhan/benchmarks/ToolSandbox")


def args(tmp_path: Path, benchmark: str) -> argparse.Namespace:
    return argparse.Namespace(
        benchmark=benchmark,
        method="proposed",
        checkpoint=CHECKPOINT,
        sglang_backend_url="http://127.0.0.1:38800",
        embedding_model=EMBEDDING,
        embedding_device="cpu",
        detector="legacy_prefill",
        selector_artifact=None,
        selector_threshold=0.5,
        benchmark_dir=BFCL,
        bfcl_python="/home/liuyancheng/envs/bench/bin/python",
        task_id=["multi_turn_base_0"] if benchmark == "bfcl" else [],
        tau2_dir=TAU2,
        tau2_python="/home/liuyancheng/envs/bench312/bin/python",
        tau2_task_id=["0"] if benchmark == "tau2" else [],
        task_set="airline",
        tau2_max_steps=None,
        toolsandbox_dir=TOOLSANDBOX,
        toolsandbox_python="/home/liuyancheng/envs/benchts/bin/python",
        ts_scenario=["get_wifi"] if benchmark == "toolsandbox" else [],
        ts_agent="GPT_4_o_2024_05_13",
        ts_user="GPT_4_o_2024_05_13",
        user_base_url=None if benchmark == "bfcl" else "http://127.0.0.1:38800",
        portable_root=BFCL,
        sglang_root=Path("/home/zhuyuhan/project/kvoffload-sglang-c2kv-pr5-runtime"),
        out=tmp_path / benchmark,
        port=38810,
        task_timeout=3600,
        preview=True,
    )


def _value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


class C1MultiBenchmarkTest(unittest.TestCase):
 def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.tmp_path = Path(self.temporary.name)

 def tearDown(self):
    self.temporary.cleanup()

 def test_bfcl_default_server_command_is_unchanged(self):
    tmp_path = self.tmp_path
    design = run_c1.current.load_config()
    controller, _ = run_c1.build_profile(args(tmp_path, "bfcl"))
    path = tmp_path / "controller.json"
    run_c1.save(path, controller)
    design["runtime"].update(controller=str(path), sglang_backend_url="http://127.0.0.1:38800")
    old = runner.server_command(
        design, task_id="multi_turn_base_0", checkpoint=str(CHECKPOINT),
        output=str(tmp_path / "out"), port=38810, python="python",
    )
    explicit = runner.server_command(
        design, task_id="multi_turn_base_0", checkpoint=str(CHECKPOINT),
        output=str(tmp_path / "out"), port=38810, python="python",
        benchmark="bfcl", source_profile="native-v1",
    )
    assert old == explicit


 def test_single_task_server_binding(self):
    for benchmark, task in (("tau2", "0"), ("toolsandbox", "get_wifi")):
        with self.subTest(benchmark=benchmark):
            parsed = args(self.tmp_path, benchmark)
            controller, _ = run_c1.build_profile(parsed)
            path = self.tmp_path / f"{benchmark}.controller.json"
            run_c1.save(path, controller)
            server, _ = run_c1.commands_for_task(parsed, task, path)
            self.assertEqual(_value(server, "--benchmark"), benchmark)
            self.assertEqual(_value(server, "--source-profile"), "openai-single-task-v1")
            self.assertEqual(_value(server, "--task-ids"), task)

 def test_user_simulator_bypasses_c1_controller(self):
    for benchmark, task in (("tau2", "0"), ("toolsandbox", "get_wifi")):
        with self.subTest(benchmark=benchmark):
            parsed = args(self.tmp_path, benchmark)
            command = run_c1.portable_worker_command(
                parsed, task, parsed.out / "task_shards" / task)
            agent = _value(command, "--agent-base-url")
            user = _value(command, "--user-base-url")
            self.assertEqual(agent, "http://127.0.0.1:38810")
            self.assertEqual(user, "http://127.0.0.1:38800")
            self.assertNotEqual(agent, user)
            self.assertEqual(_value(command, "--model"), "c1_legacy_prefill")

 def test_portable_run_name_is_bound_to_absolute_task_output(self):
    parsed = args(self.tmp_path, "tau2")
    first = parsed.out / "task_shards" / "0"
    second = self.tmp_path / "another-parent" / parsed.out.name / "task_shards" / "0"
    command_a = run_c1.portable_worker_command(parsed, "0", first)
    command_b = run_c1.portable_worker_command(parsed, "0", second)
    self.assertNotEqual(_value(command_a, "--run-name"), _value(command_b, "--run-name"))

 def test_existing_output_is_rejected(self):
    output = self.tmp_path / "existing"
    output.mkdir()
    with self.assertRaises(FileExistsError):
        run_c1.create_output_directory(output)

 def test_backend_preflight_requires_native_c1_capabilities(self):
    parsed = args(self.tmp_path, "bfcl")
    payload = {
        "model_path": str(CHECKPOINT),
        "c2kv_native_packed": {
            "enabled": True,
            "endpoint": "/v1/c2kv/native_generate",
            "packing_version": "history-event-v1",
            "base_query_enforced": True,
            "shadow_feature_layer": 34,
            "shadow_feature_readout": "decoder_layer_output",
            "model_binding": {
                "model_path": str(CHECKPOINT),
                "dtype": "bfloat16",
                "query_projection": "base",
                "gist_type": "dynamic-interleave",
                "gist_param": "qkv",
            },
        },
    }

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *unused):
            self.close()

    receipt = run_c1.preflight_sglang_backend(
        parsed,
        opener=lambda request, timeout: Response(json.dumps(payload).encode()),
    )
    self.assertTrue(all(receipt["checks"].values()))

 def test_preview_does_not_preflight_or_connect(self):
    parsed = args(self.tmp_path, "bfcl")
    identities = run_c1.validate_args(parsed)
    _, profile = run_c1.build_profile(parsed)
    with mock.patch.object(
        run_c1, "preflight_sglang_backend",
        side_effect=AssertionError("preview attempted backend access"),
    ):
        result = run_c1.preview(parsed, identities, profile)
    self.assertEqual(result["network_calls"], 0)

 def test_one_single_task_controller_rejects_multiple_ids(self):
    tmp_path = self.tmp_path
    design = run_c1.current.load_config()
    controller, _ = run_c1.build_profile(args(tmp_path, "tau2"))
    path = tmp_path / "controller.json"
    run_c1.save(path, controller)
    design["runtime"].update(controller=str(path), sglang_backend_url="http://127.0.0.1:38800")
    with self.assertRaisesRegex(ValueError, "exactly one"):
        runner.server_command(
            design, task_id="0,1", checkpoint=str(CHECKPOINT),
            output=str(tmp_path / "out"), port=38810, python="python",
            benchmark="tau2", source_profile="openai-single-task-v1",
        )


 def test_controller_config_is_identical_across_benchmarks(self):
    tmp_path = self.tmp_path
    built = [run_c1.build_profile(args(tmp_path, benchmark)) for benchmark in run_c1.BENCHMARKS]
    controllers = [controller for controller, _ in built]
    hashes = [profile["controller_sha256"] for _, profile in built]
    self.assertEqual(controllers[0], controllers[1])
    self.assertEqual(controllers[1], controllers[2])
    self.assertEqual(len(set(hashes)), 1)

 def test_c2kv_only_removes_detector_and_recovery_but_keeps_runtime_identity(self):
    proposed = args(self.tmp_path, "tau2")
    baseline = args(self.tmp_path, "tau2")
    baseline.method = "c2kv_only"
    proposed_controller, proposed_profile = run_c1.build_profile(proposed)
    baseline_controller, baseline_profile = run_c1.build_profile(baseline)
    self.assertIn("post_draft_recovery", proposed_controller)
    self.assertNotIn("post_draft_recovery", baseline_controller)
    self.assertNotIn("gp_experiments", baseline_controller)
    self.assertEqual(proposed_profile["checkpoint"], baseline_profile["checkpoint"])
    self.assertEqual(proposed_profile["ratio"], baseline_profile["ratio"])
    self.assertEqual(baseline_profile["detector"], "disabled")

 def test_c2kv_only_command_does_not_request_shadow_detector_features(self):
    parsed = args(self.tmp_path, "tau2")
    parsed.method = "c2kv_only"
    controller, _ = run_c1.build_profile(parsed)
    path = self.tmp_path / "c2kv-only.controller.json"
    run_c1.save(path, controller)
    server, worker = run_c1.commands_for_task(parsed, "0", path)
    self.assertNotIn("--shadow-feature-config", server)
    self.assertEqual(_value(worker, "--model"), "c2kv_only")

 def test_unified_summary_reads_native_packing_and_recovery(self):
    task_out = self.tmp_path / "task"
    server = task_out / "server"
    server.mkdir(parents=True)
    record = {
        "generation_trace": [
            {
                "phase": "draft", "status": "completed",
                "usage": {"prompt_tokens": 12},
                "controller": {
                    "compression_ratio": {
                        "full_history_bytes": 800, "active_history_bytes": 100,
                    },
                    "kv_bytes_per_token": 10,
                },
                "generation": {"stats": {
                    "backend": "sglang_c2kv_native_packed",
                    "gist_tokens": 10, "workspace_tokens": 2,
                    "scope_reused_chunks": 1,
                }},
            },
            {
                "phase": "regeneration", "status": "completed",
                "usage": {"prompt_tokens": 18},
                "generation": {"stats": {
                    "backend": "sglang_c2kv_native_packed",
                    "gist_tokens": 8, "workspace_tokens": 5,
                    "session_reused_gist_chunks": 1,
                }},
            },
        ],
        "exact_recovery": {
            "status": "recover", "appended_unit_count": 1,
            "appended_units": [{"token_count": 7}],
            "gate": {"score": 0.99, "triggered": True},
        },
        # The runtime records the same final decision in its round history.
        # Unified counters must still count one recovery for this step.
        "recovery_rounds": [{
            "status": "recover", "appended_unit_count": 1,
            "appended_units": [{"token_count": 7}],
            "gate": {"score": 0.99, "triggered": True},
        }],
    }
    (server / "steps.jsonl").write_text(json.dumps(record) + "\n")
    official = {"n": 1, "task_rows": [{
        "semantic_score": 1.0, "normal_termination": True,
        "protocol_legal": True,
    }]}
    row = run_c1.summarize_task("tau2", "0", task_out, official, 2.0)
    self.assertEqual(row["official_score"], 1.0)
    self.assertEqual(row["native_generate_requests"], 2)
    self.assertEqual(row["prefill_detector_scores"], 1)
    self.assertEqual(row["recovery_count"], 1)
    self.assertEqual(row["successful_recovery_count"], 1)
    self.assertEqual(row["raw_tokens_restored"], 7)
    self.assertEqual(row["compression_ratio"], 8.0)
    self.assertTrue(row["native_packing_present"])


if __name__ == "__main__":
    unittest.main()
