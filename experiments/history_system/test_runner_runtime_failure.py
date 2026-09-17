import json

import runner


def test_official_zero_does_not_hide_terminal_actor_failure(tmp_path):
    (tmp_path / "bfcl").mkdir()
    (tmp_path / "server").mkdir()
    (tmp_path / "bfcl/official_summary.json").write_text(json.dumps({"n_scored": 1, "correct_count": 0}))
    (tmp_path / "server/final.json").write_text(json.dumps({"stop_reason": "runner_failed"}))
    assert runner._server_runtime_failure(tmp_path, 1)
    assert runner._server_runtime_failure(tmp_path, 0)


def test_normal_incorrect_response_is_still_a_completed_rollout(tmp_path):
    (tmp_path / "server").mkdir()
    (tmp_path / "server/final.json").write_text(json.dumps({"stop_reason": "shutdown_requested"}))
    assert not runner._server_runtime_failure(tmp_path, 0)
    assert runner._server_runtime_failure(tmp_path, 1)
