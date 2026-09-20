from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import c1_artifact_binding as binding


HERE = Path(__file__).resolve().parent
FROZEN_ARTIFACT = HERE / "artifacts/c1_risk.t02_v1.json"
FILES = {
    "chat_template.jinja": b"template",
    "config.json": b"config",
    "generation_config.json": b"generation",
    "model.safetensors": b"weights",
    "model.safetensors.reconstructing.json": b"reconstructing",
    "tokenizer_config.json": b"tokenizer-config",
    "tokenizer.json": b"tokenizer",
}


def _artifact(source: Path) -> dict:
    artifact = json.loads(FROZEN_ARTIFACT.read_text(encoding="utf-8"))
    model = artifact["feature_contract"]["prefill_contract"]["bindings"]["model"]
    model["model_path"] = str(source.resolve())
    model["tokenizer_path"] = str(source.resolve())
    artifact["artifact_sha256"] = binding.artifact_sha256(artifact)
    return artifact


def _identity(tmp_path: Path, monkeypatch, source: str | Path) -> Path:
    checkpoint = tmp_path / "checkpoint-copy"
    checkpoint.mkdir()
    records = {}
    for relative, payload in FILES.items():
        (checkpoint / relative).write_bytes(payload)
        records[relative] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    manifest = tmp_path / "identity.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "c2kv-c1000-identity-v1",
                "identity_id": "fixture-c1000",
                "known_binding_paths": [
                    str(source.resolve()) if isinstance(source, Path) else source
                ],
                "files": records,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(binding, "IDENTITY_MANIFEST", manifest)
    return checkpoint


def test_exact_resolved_binding_is_unchanged(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    artifact = _artifact(checkpoint)

    bound, receipt = binding.bind_risk_artifact(artifact, checkpoint)

    assert bound == artifact
    assert bound is not artifact
    assert receipt["status"] == "exact_binding"
    assert receipt["source_artifact_sha256"] == artifact["artifact_sha256"]
    assert receipt["derived_artifact_sha256"] == artifact["artifact_sha256"]
    assert receipt["changed_fields"] == []


def test_identical_checkpoint_copy_rebinds_only_paths_and_digest(tmp_path, monkeypatch):
    source = tmp_path / "known-source"
    source.mkdir()
    artifact = _artifact(source)
    original = copy.deepcopy(artifact)
    checkpoint = _identity(tmp_path, monkeypatch, source)

    bound, receipt = binding.bind_risk_artifact(artifact, checkpoint)

    assert artifact == original
    expected = copy.deepcopy(original)
    model = expected["feature_contract"]["prefill_contract"]["bindings"]["model"]
    model["model_path"] = str(checkpoint.resolve())
    model["tokenizer_path"] = str(checkpoint.resolve())
    expected["artifact_sha256"] = binding.artifact_sha256(expected)
    assert bound == expected
    assert receipt["status"] == "rebound_verified_identity"
    assert receipt["source_artifact_sha256"] == original["artifact_sha256"]
    assert receipt["derived_artifact_sha256"] == bound["artifact_sha256"]
    assert set(receipt["identity_evidence"]["verified_files"]) == set(FILES)


def test_real_artifact_posix_binding_parses_and_rebinds_on_windows(tmp_path, monkeypatch):
    artifact = json.loads(FROZEN_ARTIFACT.read_text(encoding="utf-8"))
    source = artifact["feature_contract"]["prefill_contract"]["bindings"]["model"][
        "model_path"
    ]
    checkpoint = _identity(tmp_path, monkeypatch, source)

    bound, receipt = binding.bind_risk_artifact(artifact, checkpoint)

    assert receipt["status"] == "rebound_verified_identity"
    assert receipt["source_model_path"] == source
    assert (
        bound["feature_contract"]["prefill_contract"]["bindings"]["model"][
            "model_path"
        ]
        == str(checkpoint.resolve())
    )


def test_rebind_rejects_unknown_artifact_source(tmp_path, monkeypatch):
    known = tmp_path / "known-source"
    known.mkdir()
    unknown = tmp_path / "unknown-source"
    unknown.mkdir()
    checkpoint = _identity(tmp_path, monkeypatch, known)

    with pytest.raises(ValueError, match="known frozen C1000 source"):
        binding.bind_risk_artifact(_artifact(unknown), checkpoint)


def test_rebind_rejects_checkpoint_content_mismatch(tmp_path, monkeypatch):
    source = tmp_path / "known-source"
    source.mkdir()
    checkpoint = _identity(tmp_path, monkeypatch, source)
    (checkpoint / "model.safetensors").write_bytes(b"different")

    with pytest.raises(ValueError, match="model.safetensors"):
        binding.bind_risk_artifact(_artifact(source), checkpoint)


def test_rebind_rejects_incomplete_identity_manifest(tmp_path, monkeypatch):
    source = tmp_path / "known-source"
    source.mkdir()
    checkpoint = _identity(tmp_path, monkeypatch, source)
    manifest = json.loads(binding.IDENTITY_MANIFEST.read_text(encoding="utf-8"))
    del manifest["files"]["tokenizer.json"]
    binding.IDENTITY_MANIFEST.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="complete inference file set"):
        binding.bind_risk_artifact(_artifact(source), checkpoint)
