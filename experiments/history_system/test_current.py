import copy
import hashlib
import json

import current
import pytest


def value_after(command, flag):
    index = command.index(flag)
    return command[index + 1]


def test_selected_config_is_the_frozen_d3_runtime_contract():
    config = current.load_config()
    runtime = config["runtime"]

    assert config["schema"] == "a-current-algorithm-v1"
    assert config["name"] == "Prefill-guided event recovery"
    assert config["candidate_id"] == "d3_prefill_event"
    assert config["route"] == "ac_native_s0_lexical_raw_reserve_failed_operation"
    assert config["ratio"] == 8
    assert config["compression_policy"] == "always-compress-v1"
    assert config["history_view_protocol"] == "fixed-budget-main"
    assert config["session_cache_policy"] == "external-sglang-content-addressed-chunks-v1"
    assert runtime == {
        "controller": "configs/controller.json",
        "eval_policy": "configs/eval_policy.json",
        "eval_capacity": "configs/eval_capacity.json",
        "shadow_feature_config": "configs/shadow_features.json",
        "server_module": "benchmarks.memory_runtime.event_native_server",
        "official_worker_module": "benchmarks.memory_runtime.event_native_bfcl",
        "generation_backend": "sglang",
        "sglang_backend_url": "http://127.0.0.1:36100",
        "sglang_timeout_seconds": 10800,
        "device": "cpu",
        "dtype": "bfloat16",
        "bfcl_python": "/home/liuyancheng/envs/bench/bin/python",
        "npu_allocator_metrics": False,
    }

    controller_path = current.RUNTIME / runtime["controller"]
    controller = json.loads(controller_path.read_text(encoding="utf-8"))
    recovery = controller["post_draft_recovery"]
    head = recovery["prefill_head"]
    assert hashlib.sha256(controller_path.read_bytes()).hexdigest() == config["source"][
        "controller_sha256"
    ]
    assert set(recovery) == {
        "gate",
        "prefill_head",
        "quota",
        "random_probability",
        "random_seed",
        "schema",
        "task_generation_limit",
    }
    assert recovery["schema"] == "a-e1-post-draft-event-recovery-v1"
    assert recovery["gate"] == "prefill_linear_head"
    assert recovery["quota"] == {"numerator": 1, "denominator": 5}
    assert recovery["task_generation_limit"] == 96
    assert head["feature"] == "prefill.prompt_last.decoder_layer_output"
    assert head["direction"] == "at_or_above"
    assert head["threshold"] == 0.997483851175923
    assert head["artifact_sha256"] == (
        "4bd01ef68206cbc3902da49dd0ba288b41f9c92aa8f5a2473d5ffdfdb9d97c57"
    )
    assert len(head["input_mean"]) == len(head["weights"]) == 2560

    eval_policy = json.loads(
        (current.RUNTIME / runtime["eval_policy"]).read_text(encoding="utf-8")
    )
    assert eval_policy["policy"] == {
        "history_budget_bytes": 113246208,
        "workspace_budget_bytes": 113246208,
        "lease_decisions": 0,
        "max_retrieved_events": 2,
    }
    assert json.loads(
        (current.RUNTIME / runtime["eval_capacity"]).read_text(encoding="utf-8")
    )["capacity"] == {
        "max_workspace_tokens": 36864,
        "max_sequence_tokens": 40960,
    }
    assert json.loads(
        (current.RUNTIME / runtime["shadow_feature_config"]).read_text(
            encoding="utf-8"
        )
    ) == {"enabled": True, "prefill_layer": -2, "memgen_layer": None}


def test_server_command_uses_active_runtime_configs_without_later_extensions(tmp_path):
    checkpoint = tmp_path / "checkpoint-1000"
    output = tmp_path / "preview"
    args = type(
        "Args",
        (),
        {
            "device": None,
            "task_id": "multi_turn_base_20",
            "checkpoint": checkpoint,
            "out": output,
            "port": 28800,
        },
    )()

    command = current.server_command(args)
    assert command[:3] == [
        current.sys.executable,
        "-m",
        "benchmarks.memory_runtime.event_native_server",
    ]
    assert value_after(command, "--checkpoint") == str(checkpoint.resolve())
    assert value_after(command, "--view-mode") == (
        "ac_native_s0_lexical_raw_reserve_failed_operation"
    )
    assert value_after(command, "--ratio") == "8"
    assert value_after(command, "--max-generation-calls") == "96"
    assert value_after(command, "--eval-policy") == str(
        (current.RUNTIME / "configs/eval_policy.json").resolve()
    )
    assert value_after(command, "--eval-capacity") == str(
        (current.RUNTIME / "configs/eval_capacity.json").resolve()
    )
    assert value_after(command, "--s0-config") == str(
        (current.RUNTIME / "configs/controller.json").resolve()
    )
    assert value_after(command, "--shadow-feature-config") == str(
        (current.RUNTIME / "configs/shadow_features.json").resolve()
    )
    assert value_after(command, "--generation-backend") == "sglang"
    assert value_after(command, "--sglang-backend-url") == "http://127.0.0.1:36100"
    assert value_after(command, "--sglang-timeout-seconds") == "10800"
    assert value_after(command, "--device") == "cpu"
    assert "--npu-allocator-metrics" not in command
    assert "--extraction-policy" not in command
    assert "--recovery-request-budget-seconds" not in command


def test_preview_prints_static_command_without_launch(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        current.subprocess,
        "call",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("preview launched a process")
        ),
    )
    result = current.main(
        [
            "preview",
            "--checkpoint",
            str(tmp_path / "checkpoint-1000"),
            "--out",
            str(tmp_path / "preview"),
            "--task-id",
            "multi_turn_base_20",
        ]
    )

    preview = json.loads(capsys.readouterr().out)
    assert result == 0
    assert preview["algorithm"] == "Prefill-guided event recovery"
    assert preview["model_calls"] == 0
    assert preview["command"][1:3] == [
        "-m",
        "benchmarks.memory_runtime.event_native_server",
    ]


def test_preview_backend_url_override_is_forwarded(tmp_path, capsys):
    current.main(
        [
            "preview",
            "--checkpoint",
            str(tmp_path / "checkpoint-1000"),
            "--out",
            str(tmp_path / "preview"),
            "--task-id",
            "multi_turn_base_20",
            "--sglang-backend-url",
            "http://127.0.0.1:46100",
            "--sglang-timeout-seconds",
            "120",
        ]
    )
    command = json.loads(capsys.readouterr().out)["command"]
    assert value_after(command, "--sglang-backend-url") == "http://127.0.0.1:46100"
    assert value_after(command, "--sglang-timeout-seconds") == "120.0"


def test_current_d3_cannot_be_reconfigured_to_native_generation(tmp_path, monkeypatch):
    config = copy.deepcopy(current.load_config())
    config["runtime"]["generation_backend"] = "native"
    config["runtime"].pop("sglang_backend_url")
    config["runtime"].pop("sglang_timeout_seconds")
    monkeypatch.setattr(current, "load_config", lambda: config)
    args = type(
        "Args",
        (),
        {
            "task_id": "multi_turn_base_20",
            "checkpoint": tmp_path / "checkpoint-1000",
            "out": tmp_path / "preview",
            "port": 28800,
        },
    )()

    with pytest.raises(ValueError, match="D3 post_draft_recovery.*SGLang"):
        current.server_command(args)


def test_gp_preview_materializes_all_switches_without_touching_d3(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "options.json"
    config_path.write_text(json.dumps({"G": "record", "U": "field", "K": 4, "R": 3}), encoding="utf-8")
    original_path = current.RUNTIME / "configs/controller.json"
    original = original_path.read_bytes()
    monkeypatch.setattr(current.subprocess, "call", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("started model")))
    current.main(["preview", "--checkpoint", str(tmp_path / "checkpoint"),
                  "--out", str(tmp_path / "output"), "--task-id", "multi_turn_base_20",
                  "--gp-config", str(config_path)])
    command = json.loads(capsys.readouterr().out)["command"]
    path = tmp_path / "output/gp.controller.json"
    resolved = json.loads(path.read_text(encoding="utf-8"))
    assert value_after(command, "--s0-config") == str(path.resolve())
    assert resolved["gp_experiments"]["G"] == "record"
    assert resolved["gp_experiments"]["K"] == 4
    assert resolved["gp_experiments"]["R"] == 3
    assert original_path.read_bytes() == original
