"""Safely bind the frozen T02 C1 risk artifact to an identical C1000 copy."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


HERE = Path(__file__).resolve().parent
RUNTIME = HERE / "runtime"
RUNTIME_PYTHON = RUNTIME / "python"
IDENTITY_MANIFEST = HERE / "artifacts/c1000_identity.json"
REQUIRED_IDENTITY_FILES = frozenset(
    {
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "model.safetensors.reconstructing.json",
        "tokenizer_config.json",
        "tokenizer.json",
    }
)

for search_path in (RUNTIME_PYTHON, RUNTIME):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from benchmarks.memory_runtime.recovery.set_models import (  # noqa: E402
    C1RiskArtifact,
    artifact_sha256,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_absolute_path(value: str) -> bool:
    return PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()


def _path_identity(value: str) -> tuple[str, str]:
    """Compare stored absolute paths without interpreting alien OS syntax."""

    if PureWindowsPath(value).is_absolute():
        return "windows", str(PureWindowsPath(value)).casefold()
    if PurePosixPath(value).is_absolute():
        return "posix", PurePosixPath(value).as_posix()
    raise ValueError(f"path must be absolute: {value}")


def _model_binding(artifact: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        binding = artifact["feature_contract"]["prefill_contract"]["bindings"]["model"]
    except (KeyError, TypeError) as error:
        raise ValueError("C1 artifact lacks the prefill model binding") from error
    if not isinstance(binding, Mapping):
        raise ValueError("C1 artifact prefill model binding must be an object")
    for key in ("model_path", "tokenizer_path"):
        if not isinstance(binding.get(key), str) or not binding[key]:
            raise ValueError(f"C1 artifact model binding lacks {key}")
        if not _is_absolute_path(binding[key]):
            raise ValueError(f"C1 artifact {key} must be absolute")
    return binding


def _load_identity_manifest() -> tuple[dict[str, Any], str]:
    try:
        raw = IDENTITY_MANIFEST.read_bytes()
        manifest = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load C1000 identity manifest: {IDENTITY_MANIFEST}") from error
    if not isinstance(manifest, dict) or manifest.get("schema") != "c2kv-c1000-identity-v1":
        raise ValueError("unsupported C1000 identity manifest")
    if not isinstance(manifest.get("identity_id"), str) or not manifest["identity_id"]:
        raise ValueError("C1000 identity manifest lacks identity_id")
    known_paths = manifest.get("known_binding_paths")
    if not isinstance(known_paths, list) or not known_paths or not all(
        isinstance(path, str) and _is_absolute_path(path) for path in known_paths
    ):
        raise ValueError("C1000 identity manifest lacks absolute known binding paths")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != REQUIRED_IDENTITY_FILES:
        raise ValueError("C1000 identity manifest does not cover the complete inference file set")
    for relative, expected in files.items():
        if not isinstance(expected, dict):
            raise ValueError(f"invalid C1000 identity record for {relative}")
        digest = expected.get("sha256")
        size = expected.get("bytes")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or type(size) is not int
            or size < 0
        ):
            raise ValueError(f"invalid C1000 identity record for {relative}")
    return manifest, hashlib.sha256(raw).hexdigest()


def _verify_checkpoint(
    checkpoint: Path, manifest: Mapping[str, Any], manifest_sha256: str
) -> dict[str, Any]:
    verified: dict[str, dict[str, Any]] = {}
    for relative, expected in sorted(manifest["files"].items()):
        path = checkpoint / relative
        if not path.is_file():
            raise ValueError(f"C1000 checkpoint is missing required file: {relative}")
        actual_size = path.stat().st_size
        if actual_size != expected["bytes"]:
            raise ValueError(
                f"C1000 checkpoint size mismatch for {relative}: "
                f"expected={expected['bytes']}, actual={actual_size}"
            )
        actual_sha256 = _sha256(path)
        if actual_sha256 != expected["sha256"]:
            raise ValueError(
                f"C1000 checkpoint SHA-256 mismatch for {relative}: "
                f"expected={expected['sha256']}, actual={actual_sha256}"
            )
        verified[relative] = {"bytes": actual_size, "sha256": actual_sha256}
    return {
        "identity_id": manifest["identity_id"],
        "identity_manifest": str(IDENTITY_MANIFEST.resolve()),
        "identity_manifest_sha256": manifest_sha256,
        "verified_files": verified,
    }


def bind_risk_artifact(
    artifact: dict[str, Any], checkpoint: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a validated artifact whose path binding matches ``checkpoint``.

    A byte-identical checkpoint copy may live at a different absolute path.  In
    that case only the two path fields and the artifact's derived digest change.
    """

    if not isinstance(artifact, dict):
        raise TypeError("artifact must be a dict")
    source = copy.deepcopy(artifact)
    C1RiskArtifact(source)
    binding = _model_binding(source)
    try:
        resolved_checkpoint = checkpoint.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"checkpoint does not resolve: {checkpoint}") from error
    if not resolved_checkpoint.is_dir():
        raise ValueError(f"checkpoint is not a directory: {resolved_checkpoint}")

    source_digest = source["artifact_sha256"]
    source_model_path = binding["model_path"]
    source_tokenizer_path = binding["tokenizer_path"]
    resolved_checkpoint_text = str(resolved_checkpoint)
    exact = (
        source_model_path == resolved_checkpoint_text
        and source_tokenizer_path == resolved_checkpoint_text
    )
    if exact:
        return source, {
            "schema": "c2kv-c1-artifact-binding-receipt-v1",
            "status": "exact_binding",
            "checkpoint": str(resolved_checkpoint),
            "source_model_path": source_model_path,
            "source_tokenizer_path": source_tokenizer_path,
            "bound_model_path": source_model_path,
            "bound_tokenizer_path": source_tokenizer_path,
            "source_artifact_sha256": source_digest,
            "derived_artifact_sha256": source_digest,
            "identity_evidence": {"method": "resolved_path_identity"},
            "changed_fields": [],
        }

    manifest, manifest_sha256 = _load_identity_manifest()
    known_paths = {_path_identity(path) for path in manifest["known_binding_paths"]}
    if (
        _path_identity(source_model_path) not in known_paths
        or _path_identity(source_tokenizer_path) not in known_paths
    ):
        raise ValueError("C1 artifact is not bound to a known frozen C1000 source")
    identity_evidence = _verify_checkpoint(
        resolved_checkpoint, manifest, manifest_sha256
    )

    bound = copy.deepcopy(source)
    bound_binding = _model_binding(bound)
    bound_binding["model_path"] = str(resolved_checkpoint)
    bound_binding["tokenizer_path"] = str(resolved_checkpoint)
    bound["artifact_sha256"] = artifact_sha256(bound)
    C1RiskArtifact(bound)
    return bound, {
        "schema": "c2kv-c1-artifact-binding-receipt-v1",
        "status": "rebound_verified_identity",
        "checkpoint": str(resolved_checkpoint),
        "source_model_path": source_model_path,
        "source_tokenizer_path": source_tokenizer_path,
        "bound_model_path": str(resolved_checkpoint),
        "bound_tokenizer_path": str(resolved_checkpoint),
        "source_artifact_sha256": source_digest,
        "derived_artifact_sha256": bound["artifact_sha256"],
        "identity_evidence": identity_evidence,
        "changed_fields": [
            "feature_contract.prefill_contract.bindings.model.model_path",
            "feature_contract.prefill_contract.bindings.model.tokenizer_path",
            "artifact_sha256",
        ],
    }
