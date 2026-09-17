"""Contracts for the bounded official BFCL event-native wrapper."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest

from benchmarks.adapters import bfcl_adapter
from benchmarks.memory_runtime import event_native_bfcl as wrapper


def _ready() -> dict:
    return {
        "schema": "a-event-native-server-v1",
        "status": "ready",
        "run_id": "event-native-run-1",
        "model_name": "c2kv-event-native",
        "view_mode": "capacity_exact_once",
        "route_contract": {
            "view_mode": "capacity_exact_once", "baseline_identity": "C2KV-recover-once",
            "recovery_enabled": True, "max_generations_per_decision": 2,
            "legacy_1088_equivalent": False,
        },
        "runtime_policy_contract": {
            "schema": "a-event-native-runtime-policy-v1",
            "source": "checkpoint_training_policy",
            "policy_id": None, "policy_sha256": None, "eval_policy": None, "source_path": None,
            "effective_policy": {"history_budget_bytes": 1000, "workspace_budget_bytes": 500,
                                 "lease_decisions": 3, "max_retrieved_events": 1},
        },
        "decode_strategy": "incremental",
        "session_cache_policy": "last-final-view-v1",
        "allowed_task_ids": ["multi_turn_base_1", "multi_turn_base_30"],
        "max_new_tokens": 768,
        "max_decisions": 20,
        "max_generation_calls": 40,
        "sampling": {"mode": "greedy", "temperature": 0, "seed": 0},
        "checkpoint": {"source": "local-checkpoint"},
    }


def _health(*, terminal: bool = False) -> dict:
    ready = _ready()
    return {
        "schema": "a-event-native-api-health-v1",
        **{
            key: copy.deepcopy(ready[key])
            for key in (
                "run_id",
                "model_name",
                "view_mode",
                "route_contract",
                "runtime_policy_contract",
                "decode_strategy",
                "session_cache_policy",
                "allowed_task_ids",
                "max_new_tokens",
                "max_decisions",
                "max_generation_calls",
            )
        },
        "terminal": terminal,
        "decisions_reserved": 0 if not terminal else ready["max_decisions"],
        "generation_calls_reserved": 0 if not terminal else 1,
    }


def _write_manifest(path: Path, value: dict | None = None) -> Path:
    path.write_text(json.dumps(value or _ready()), encoding="utf-8")
    return path


def _benchmark_dir(path: Path) -> Path:
    (path / "bfcl_eval").mkdir(parents=True)
    return path


@pytest.mark.parametrize('policy', sorted(wrapper.SUPPORTED_SESSION_CACHE_POLICIES))
def test_validate_server_identity_accepts_matching_fresh_endpoint(policy) -> None:
    ready, health = _ready(), _health()
    ready['session_cache_policy'] = health['session_cache_policy'] = policy
    wrapper.validate_server_identity(ready, health)


@pytest.mark.parametrize('source', ['ready', 'health', 'both'])
def test_bfcl_worker_rejects_other_benchmark_namespace(source):
    ready, health = _ready(), _health()
    if source in ('ready', 'both'):
        ready['benchmark'] = 'acebench'
    if source in ('health', 'both'):
        health['benchmark'] = 'acebench'
    with pytest.raises(ValueError, match='benchmark namespace'):
        wrapper.validate_server_identity(ready, health)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "wrong-health-schema"),
        ("run_id", "other-run"),
        ("model_name", "other-model"),
        ("view_mode", "capacity_exact_no_gist"),
        ("route_contract", {"view_mode": "static", "baseline_identity": "legacy"}),
        ("runtime_policy_contract", {"schema": "different-runtime-policy"}),
        ("decode_strategy", "full_recompute"),
        ("session_cache_policy", "other-policy"),
        ("max_new_tokens", 1024),
        ("max_decisions", 21),
        ("max_generation_calls", 41),
        ("allowed_task_ids", ["multi_turn_base_1"]),
        ("terminal", True),
        ("decisions_reserved", 1),
        ("generation_calls_reserved", 1),
    ],
)
def test_validate_server_identity_rejects_wrong_or_consumed_endpoint(
    field: str, value,
) -> None:
    health = _health()
    health[field] = value

    with pytest.raises(ValueError):
        wrapper.validate_server_identity(_ready(), health)


@pytest.mark.parametrize('field', ['policy_id','policy_sha256','effective_policy','field_roles'])
def test_runtime_policy_nested_mismatch_rejects_before_official_worker(field):
    ready, health = _ready(), _health()
    health['runtime_policy_contract'][field] = {'changed':True}
    with pytest.raises(ValueError, match='runtime_policy_contract'):
        wrapper.validate_server_identity(ready, health)


@pytest.mark.parametrize('category', ['multi_turn_base', 'multi_turn_long_context'])
def test_worker_calls_existing_official_runner_with_frozen_contract(
    tmp_path, monkeypatch, category,
) -> None:
    root = tmp_path / "official-output"
    summary_path = tmp_path / "official-summary.json"
    benchmark_dir = tmp_path / "official-bfcl"
    benchmark_dir.mkdir()
    contract = {
        "server_manifest": _ready(),
        "bfcl_project_root": str(root),
        "benchmark_dir": str(benchmark_dir),
        "base_url": "http://127.0.0.1:31000/v1",
        "handler_name": "c2kv-event-native-capacity-exact-once",
        "summary_path": str(summary_path),
    }
    contract_path = tmp_path / "contract.json"
    contract['server_manifest']['allowed_task_ids'] = [f'{category}_1', f'{category}_30']
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    seen = {}

    def run_bfcl(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return {"benchmark": "bfcl", "scored": True}

    monkeypatch.setattr(bfcl_adapter, "run_bfcl", run_bfcl)
    monkeypatch.setattr(wrapper, 'bind_benchmark', lambda path: {'fixture': 'scripted-package'})
    # The production worker is a disposable process. Isolate its environment
    # mutation when invoking it directly inside this shared pytest process.
    monkeypatch.setenv("BFCL_PROJECT_ROOT", str(tmp_path / "previous-root"))
    monkeypatch.setattr(wrapper.os, "chdir", lambda path: seen.setdefault("cwd", path))

    wrapper.worker(contract_path)

    assert seen["args"] == (contract["base_url"],)
    assert seen["cwd"] == contract["benchmark_dir"]
    assert seen["kwargs"] == {
        "categories": category,
        "mode": "both",
        "run_ids": contract['server_manifest']['allowed_task_ids'],
        "model": _ready()["model_name"],
        "handler_name": contract["handler_name"],
        "project_root": root,
        "gold_recovery": None,
        "task_audit_path": root / "task_audit" / "tasks.jsonl",
        "num_threads": 1,
        "no_upstream_retries": True,
        "generation_temperature": 0,
        "generation_seed": 0,
        "generation_max_tokens": _ready()["max_new_tokens"],
    }
    assert json.loads(summary_path.read_text(encoding="utf-8")) == {
        "benchmark": "bfcl",
        "scored": True,
    }


def test_wrong_endpoint_identity_fails_before_any_child_spawn(
    tmp_path, monkeypatch,
) -> None:
    ready_path = _write_manifest(tmp_path / "ready.json")
    benchmark_dir = _benchmark_dir(tmp_path / "official-bfcl")
    health = _health()
    health["run_id"] = "wrong-run"
    monkeypatch.setattr(wrapper, "read_health", lambda base_url: health)

    def unexpected_popen(*args, **kwargs):
        raise AssertionError("identity failure must precede Popen")

    monkeypatch.setattr(wrapper.subprocess, "Popen", unexpected_popen)
    out = tmp_path / "run-output"

    with pytest.raises(ValueError, match="endpoint identity differs for run_id"):
        wrapper.main([
            "--server-manifest", str(ready_path),
            "--base-url", "http://127.0.0.1:31000/v1",
            "--benchmark-dir", str(benchmark_dir),
            "--out", str(out),
            "--max-wall-seconds", "10",
        ])

    assert not out.exists()


def test_wall_cap_terminates_owned_child_and_writes_final_wall_receipt(
    tmp_path, monkeypatch,
) -> None:
    ready_path = _write_manifest(tmp_path / "ready.json")
    benchmark_dir = _benchmark_dir(tmp_path / "official-bfcl")
    before = _health()
    after = _health(terminal=True)
    health_responses = iter((before, after))
    monkeypatch.setattr(wrapper, "read_health", lambda base_url: next(health_responses))
    clock = iter((100.0, 100.5, 101.0, 105.0))
    monkeypatch.setattr(wrapper.time, "monotonic", lambda: next(clock))
    processes = []

    class Process:
        pid = 4321

        def __init__(self, *args, **kwargs):
            self.command = args[0]
            self.kwargs = kwargs
            self.returncode = None
            self.terminated = False
            self.killed = False
            self.wait_calls = []
            processes.append(self)

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.wait_calls.append(timeout)
            if len(self.wait_calls) == 1:
                raise subprocess.TimeoutExpired(self.command, timeout)
            self.returncode = -15
            return self.returncode

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True
            self.returncode = -9

    monkeypatch.setattr(wrapper.subprocess, "Popen", Process)
    out = tmp_path / "run-output"

    with pytest.raises(SystemExit) as stopped:
        wrapper.main([
            "--server-manifest", str(ready_path),
            "--base-url", "http://127.0.0.1:31000/v1",
            "--benchmark-dir", str(benchmark_dir),
            "--out", str(out),
            "--max-wall-seconds", "2",
        ])

    assert stopped.value.code == 1
    assert len(processes) == 1
    process = processes[0]
    assert process.kwargs["env"]["PYTHONPATH"] == str(Path(wrapper.__file__).resolve().parents[2])
    assert process.terminated is True
    assert process.killed is False
    assert process.wait_calls == [pytest.approx(1.0), 5]
    final = json.loads((out / "final.json").read_text(encoding="utf-8"))
    assert final["status"] == "wall_cap_reached"
    assert final["worker_returncode"] == -15
    assert final["wall_seconds"] == pytest.approx(5.0)
    assert final["wall_seconds_final"] is True
    assert final["server_health_before"] == before
    assert final["server_health_after"] == after
    assert (out / "worker.log").is_file()


def test_wall_exhaustion_before_admission_writes_terminal_receipt_without_spawn(
    tmp_path, monkeypatch,
) -> None:
    ready_path = _write_manifest(tmp_path / "ready.json")
    benchmark_dir = _benchmark_dir(tmp_path / "official-bfcl")
    health = _health()
    monkeypatch.setattr(wrapper, "read_health", lambda base_url: health)
    clock = iter((100.0, 102.0, 103.0))
    monkeypatch.setattr(wrapper.time, "monotonic", lambda: next(clock))

    def unexpected_popen(*args, **kwargs):
        raise AssertionError("expired wall budget must precede Popen")

    monkeypatch.setattr(wrapper.subprocess, "Popen", unexpected_popen)
    out = tmp_path / "run-output"

    with pytest.raises(TimeoutError, match="before worker admission"):
        wrapper.main([
            "--server-manifest", str(ready_path),
            "--base-url", "http://127.0.0.1:31000/v1",
            "--benchmark-dir", str(benchmark_dir),
            "--out", str(out),
            "--max-wall-seconds", "1",
        ])

    final = json.loads((out / "final.json").read_text(encoding="utf-8"))
    assert final["status"] == "wall_cap_reached"
    assert final["wall_seconds"] == pytest.approx(3.0)
    assert final["wall_seconds_final"] is True
    assert "child_pid" not in final
    assert final["server_health_before"] == health
    assert final["server_health_after"] == health


def _overlap_wrapper_inputs(tmp_path):
    from benchmarks.memory_runtime.audit_b_training_overlap import audit_formal_corpus
    from benchmarks.memory_runtime.tests.test_audit_b_training_overlap import (
        _write_corpus, _write_checkpoint, _write_json, _write_jsonl,
    )

    corpus, corpus_identity = _write_corpus(tmp_path)
    checkpoint = _write_checkpoint(tmp_path, corpus_identity)
    benchmark = _benchmark_dir(tmp_path / 'official-bfcl')
    source = benchmark / 'bfcl_eval' / 'data' / wrapper.QUESTION_FILENAME
    _write_jsonl(source, [
        {'id': task, 'question': [[{'role': 'user', 'content': f'Synthetic request {task}'}]]}
        for task in ('multi_turn_base_1', 'multi_turn_base_30', 'multi_turn_base_2')
    ])
    audit_path = tmp_path / 'overlap.json'
    _write_json(audit_path, audit_formal_corpus(
        corpus, bfcl_tasks_path=source,
        candidate_ids=['multi_turn_base_1', 'multi_turn_base_30', 'multi_turn_base_2'],
        dev_task_ids=['multi_turn_base_2'], checkpoint_dirs=[checkpoint]))
    ready = _ready()
    ready['checkpoint'] = {'checkpoint': str(checkpoint), 'corpus_identity': corpus_identity,
                           'training_arm': 'B'}
    ready['checkpoint_path'] = str(checkpoint)
    ready_path = _write_manifest(tmp_path / 'ready.json', ready)
    out = tmp_path / 'run'
    argv = ['--server-manifest', str(ready_path), '--base-url', 'http://127.0.0.1:31000/v1',
            '--benchmark-dir', str(benchmark), '--out', str(out), '--max-wall-seconds', '30',
            '--overlap-audit', str(audit_path)]
    return ready, source, out, argv


@pytest.mark.parametrize('installed_source_changed', [False, True])
def test_overlap_main_to_worker_checks_actual_imported_source_before_generation(
    tmp_path, monkeypatch, installed_source_changed,
):
    ready, source, out, argv = _overlap_wrapper_inputs(tmp_path)
    # The installed package can differ from --benchmark-dir. The worker must
    # validate its imported PROMPT_PATH rather than reuse the parent's path.
    installed_source = tmp_path / 'installed-prompts' / source.name
    installed_source.parent.mkdir()
    installed_source.write_bytes(source.read_bytes())
    if installed_source_changed:
        installed_source.write_bytes(source.read_bytes().replace(b'Synthetic', b'Changed'))
    monkeypatch.setattr(wrapper, 'official_question_path', lambda: installed_source)
    monkeypatch.setattr(wrapper, 'bind_benchmark', lambda path: {'fixture': 'scripted-package'})
    monkeypatch.setattr(wrapper, 'read_health', lambda base_url: _health())
    monkeypatch.setattr(wrapper.os, 'chdir', lambda path: None)
    monkeypatch.setenv('BFCL_PROJECT_ROOT', str(tmp_path / 'previous-project'))
    calls = []

    def run_bfcl(*args, **kwargs):
        calls.append(kwargs)
        return {'fixture': 'scripted-official-harness', 'n_scored': len(kwargs['run_ids'])}

    monkeypatch.setattr(bfcl_adapter, 'run_bfcl', run_bfcl)

    class Process:
        pid = 4321

        def __init__(self, command, **kwargs):
            wrapper.worker(Path(command[command.index('--worker-contract') + 1]))

        def wait(self, timeout):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(wrapper.subprocess, 'Popen', Process)
    if installed_source_changed:
        with pytest.raises(ValueError, match='question source differs'):
            wrapper.main(argv)
        assert calls == []
        assert not (out / 'official_summary.json').exists()
        assert not (out / 'overlap_admission.json').exists()
    else:
        wrapper.main(argv)
        assert len(calls) == 1
        assert calls[0]['run_ids'] == ready['allowed_task_ids']
        admission = json.loads((out / 'overlap_admission.json').read_text(encoding='utf-8'))
        assert admission['status'] == 'passed' and admission['formal_split_frozen'] is False
        assert admission['bfcl_source']['path'] == str(installed_source.resolve())
        contract = json.loads((out / 'contract.json').read_text(encoding='utf-8'))
        assert admission['audit'] == contract['overlap_admission']['audit']


def test_excluded_task_is_rejected_by_parent_before_spawn_or_output(tmp_path, monkeypatch):
    ready, source, out, argv = _overlap_wrapper_inputs(tmp_path)
    ready['allowed_task_ids'].append('multi_turn_base_2')
    _write_manifest(tmp_path / 'ready.json', ready)
    health = _health()
    health['allowed_task_ids'] = ready['allowed_task_ids']
    monkeypatch.setattr(wrapper, 'read_health', lambda base_url: health)

    def unexpected(*args, **kwargs):
        raise AssertionError('excluded task must not reach the official worker')

    monkeypatch.setattr(wrapper.subprocess, 'Popen', unexpected)
    with pytest.raises(ValueError, match='excluded or unaudited task'):
        wrapper.main(argv)
    assert not out.exists()
