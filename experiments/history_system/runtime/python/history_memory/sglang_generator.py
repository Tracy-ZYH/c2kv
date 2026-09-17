"""SGLang transport for event-native packed history generation.

The adapter owns no model or KV tensors.  It sends the exact ``PackedMemory``
layout to the C2KV SGLang engine, validates the returned layout/accounting, and
exposes the small generator interface consumed by ``EventNativeDecisionRunner``.
There is deliberately no alternate generation path when the upstream request
fails.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Sequence
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from .encoding_scope import validate_encoding_scope
from .packing import PACKING_VERSION, EncoderChunk, PackedMemory

if TYPE_CHECKING:
    from .shadow_features import ShadowFeatureConfig


REQUEST_SCHEMA = "c2kv-native-packed-generation-v1"
RESPONSE_SCHEMA = "c2kv-native-packed-generation-response-v1"
HANDLE_SCHEMA = "c2kv-native-chunk-handle-v1"
GENERATION_SCHEMA = "event-native-sglang-generation-v1"
HTTP_JOURNAL_SCHEMA = "event-native-sglang-http-v1"
SESSION_CACHE_POLICY = "external-sglang-content-addressed-chunks-v1"
SHADOW_FEATURE_SCHEMA = "event-native-shadow-features-v1"
EXTRACTION_FAILURE_SCHEMA = "c2kv-native-extraction-failure-v1"
EXTRACTION_BUDGET_ERROR_CODE = "C2KV_EXTRACTION_BUDGET_EXHAUSTED"


@dataclass(frozen=True)
class SGLangEventNativeGenerationResult:
    """The observable ``EventNativeGenerationResult`` interface."""

    token_ids: tuple[int, ...]
    finish_reason: str
    token_logprobs: tuple[float, ...]
    stats: dict[str, Any]


class SGLangEventNativeError(RuntimeError):
    """One terminal upstream or response-contract failure."""


class SGLangExtractionBudgetExhausted(SGLangEventNativeError):
    """A verified upstream stop after consuming the remaining extraction budget."""

    def __init__(self, message: str, receipt: Mapping[str, Any]):
        super().__init__(message)
        self.receipt = copy.deepcopy(dict(receipt))


@dataclass
class _Session:
    session_id: str
    generation: int = 0
    handles: set[str] = field(default_factory=set)


@dataclass
class _DecisionScope:
    session_id: str | None
    session_handles: set[str]
    retained_handles: set[str] = field(default_factory=set)
    generate_calls: int = 0
    pending_stats: dict[str, Any] | None = None
    reset_reason: str | None = None


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return parsed


def _json_snapshot(value: Any, name: str) -> Any:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} must contain finite JSON data") from error


def _json_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    copied = _json_snapshot(value, name)
    if not isinstance(copied, dict):
        raise TypeError(f"{name} must be a JSON object")
    return copied


def _base_url(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("upstream must be a nonempty http:// base URL")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/")
    ):
        raise ValueError(
            "upstream must be a bare http:// host:port URL without credentials, path, query, or fragment"
        )
    return value.rstrip("/")


def _token_ids(value: Any, name: str, *, nonempty: bool = False) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence of token IDs")
    result = tuple(value)
    if nonempty and not result:
        raise ValueError(f"{name} must not be empty")
    if any(type(token) is not int or token < 0 for token in result):
        raise ValueError(f"{name} must contain nonnegative integer token IDs")
    return result


def _eos_ids(value: Any) -> tuple[int, ...]:
    if isinstance(value, bool):
        raise TypeError("eos_token_ids must be an integer or sequence of integers")
    values = (value,) if isinstance(value, int) else _token_ids(value, "eos_token_ids")
    if not values:
        raise ValueError("eos_token_ids must not be empty")
    return tuple(sorted(set(values)))


def _nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise SGLangEventNativeError(f"{name} must be a nonnegative integer")
    return value


def _optional_nonnegative_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_int(value, name)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


class _HTTPJournal:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.name:
            raise ValueError("journal_path must include a file name")
        self._lock = threading.Lock()

    def append(self, record: Mapping[str, Any]) -> None:
        encoded = _canonical_bytes(record) + b"\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_BINARY", 0)
            descriptor = os.open(self.path, flags, 0o600)
            try:
                remaining = memoryview(encoded)
                while remaining:
                    written = os.write(descriptor, remaining)
                    if written <= 0:
                        raise OSError("HTTP journal append made no progress")
                    remaining = remaining[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


class SGLangEventNativeGenerator:
    """Finite, no-fallback adapter for SGLang's native packed C2KV endpoint."""

    cache_trace_schema = "event-native-cache-trace-v1"
    decode_strategy = "incremental"
    session_cache_policy = SESSION_CACHE_POLICY

    def __init__(
        self,
        upstream: str,
        *,
        expected_model_path: str | Path,
        model_context: int,
        max_new_tokens: int,
        max_generation_calls: int,
        max_extraction_calls: int,
        timeout_seconds: float,
        eos_token_ids: int | Sequence[int],
        eos_source: str,
        encoding_scope: str = "current",
        journal_path: str | Path | None = None,
        sampling_params: Mapping[str, Any] | None = None,
        shadow_feature_config: "ShadowFeatureConfig | None" = None,
        max_response_bytes: int = 16 * 1024 * 1024,
        opener: Any | None = None,
    ) -> None:
        self.upstream = _base_url(upstream)
        if not isinstance(expected_model_path, (str, Path)):
            raise TypeError("expected_model_path must be a path string")
        self.expected_model_path = str(expected_model_path).rstrip("/")
        if not self.expected_model_path:
            raise ValueError("expected_model_path must be nonempty")
        self.model_context = _positive_int(model_context, "model_context")
        self.max_new_tokens = _positive_int(max_new_tokens, "max_new_tokens")
        self.max_generation_calls = _positive_int(
            max_generation_calls, "max_generation_calls"
        )
        self.timeout_seconds = _positive_finite(timeout_seconds, "timeout_seconds")
        self.eos_token_ids = _eos_ids(eos_token_ids)
        if not isinstance(eos_source, str) or not eos_source:
            raise ValueError("eos_source must be a nonempty string")
        self.eos_source = eos_source
        self.encoding_scope = validate_encoding_scope(encoding_scope)
        self.max_extraction_calls = _positive_int(
            max_extraction_calls, "max_extraction_calls"
        )
        self.max_response_bytes = _positive_int(max_response_bytes, "max_response_bytes")

        sampling = _json_object(
            {"temperature": 0.0, "seed": 0}
            if sampling_params is None
            else sampling_params,
            "sampling_params",
        )
        forbidden = {"max_new_tokens", "max_tokens", "stream", "n"} & set(sampling)
        if forbidden:
            raise ValueError(
                "sampling_params must omit per-call fields: " + ", ".join(sorted(forbidden))
            )
        temperature = sampling.get("temperature", 0)
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise ValueError("sampling_params.temperature must be numeric")
        if float(temperature) != 0.0:
            raise ValueError("event-native D3/GP serving requires greedy temperature=0")
        self.sampling_params = sampling
        self.shadow_feature_config = shadow_feature_config

        self._http_journal = _HTTPJournal(journal_path) if journal_path is not None else None
        self._opener = opener or build_opener(ProxyHandler({}))
        if not callable(getattr(self._opener, "open", None)):
            raise TypeError("opener must expose open(request, timeout=...)")

        self._requests_submitted = 0
        self.extraction_calls_reserved = 0
        self._model_binding: dict[str, Any] | None = None
        self._kv_bytes_per_token: int | None = None
        self._active_decision_scope: _DecisionScope | None = None
        self._session_cache: _Session | None = None
        self.last_generation_trace: dict[str, Any] | None = None
        self.last_cache_lifecycle_trace: dict[str, Any] | None = None
        self._exact_snapshots: dict[str, dict[str, Any]] = {}

    def _exact_local_state(self) -> dict[str, Any]:
        session = self._session_cache
        scope = self._active_decision_scope
        return {
            "session": None if session is None else {
                "session_id": session.session_id, "generation": session.generation,
                "handles": sorted(session.handles),
            },
            "scope": None if scope is None else {
                "session_id": scope.session_id,
                "session_handles": sorted(scope.session_handles),
                "retained_handles": sorted(scope.retained_handles),
                "generate_calls": scope.generate_calls,
                "pending_stats": copy.deepcopy(scope.pending_stats),
                "reset_reason": scope.reset_reason,
            },
            "requests_submitted": self._requests_submitted,
            "extraction_calls_reserved": self.extraction_calls_reserved,
            "sampling_params": copy.deepcopy(self.sampling_params),
            "last_generation_trace": copy.deepcopy(self.last_generation_trace),
            "last_cache_lifecycle_trace": copy.deepcopy(self.last_cache_lifecycle_trace),
            "model_binding": copy.deepcopy(self._model_binding),
            "kv_bytes_per_token": self._kv_bytes_per_token,
        }

    def _exact_request(self, operation: str, snapshot_id: str | None = None) -> dict[str, Any]:
        response, status = self._read_json(Request(
            self.upstream + "/v1/c2kv/exact_state",
            data=_canonical_bytes({"operation": operation, "snapshot_id": snapshot_id}),
            headers={"Content-Type": "application/json"}, method="POST",
        ), label="SGLang exact_state")
        if status != 200 or not isinstance(response, dict):
            error = response.get("error") if isinstance(response, dict) else response
            raise SGLangEventNativeError(f"Exact state {operation} failed: HTTP {status}: {error}")
        if response.get("schema") != "c2kv-exact-backend-state-v1" or response.get("operation") != operation:
            raise SGLangEventNativeError("Exact state response schema/operation mismatch")
        if snapshot_id is not None and response.get("snapshot_id") != snapshot_id:
            raise SGLangEventNativeError("Exact state snapshot ID mismatch")
        if operation != "release":
            digests = response.get("component_digests")
            if response.get("exact") is not True or not isinstance(digests, dict) or set(digests) != {"actor_kv", "actor_positions", "backend_stats", "rng"}:
                raise SGLangEventNativeError("Exact backend state components are incomplete")
            for value in digests.values():
                if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                    raise SGLangEventNativeError("Exact backend digest is malformed")
        return response

    @staticmethod
    def _exact_receipt(remote: Mapping[str, Any], local: Mapping[str, Any]) -> dict[str, Any]:
        digests = dict(remote["component_digests"])
        for name, payload in {
            "actor_kv": {"session": local["session"], "scope": local["scope"]},
            "actor_positions": {"last_generation_trace": local["last_generation_trace"]},
            "backend_stats": local,
            "rng": {"sampling_params": local["sampling_params"]},
        }.items():
            digests[name] = hashlib.sha256(_canonical_bytes({
                "backend": digests[name], "adapter": payload,
            })).hexdigest()
        return {"schema": "event-native-sglang-exact-state-v1",
                "snapshot_id": remote["snapshot_id"], "component_digests": digests,
                "exact": True, "backend": copy.deepcopy(dict(remote))}

    def capture_exact_state(self) -> dict[str, Any]:
        """Copy live backend tensors and current adapter decision/session state."""
        self._ensure_model_info()
        local = self._exact_local_state()
        remote = self._exact_request("capture")
        receipt = self._exact_receipt(remote, local)
        self._exact_snapshots[remote["snapshot_id"]] = {
            "local": local, "remote": remote, "receipt": receipt,
        }
        return copy.deepcopy(receipt)

    def restore_exact_state(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        """Restore tensors plus adapter state; preserve an active context object."""
        snapshot_id = snapshot.get("snapshot_id")
        saved = self._exact_snapshots.get(snapshot_id)
        if saved is None or snapshot.get("component_digests") != saved["receipt"]["component_digests"]:
            raise SGLangEventNativeError("Unknown or changed exact adapter snapshot")
        local = copy.deepcopy(saved["local"])
        scope = self._active_decision_scope
        if (scope is None) != (local["scope"] is None):
            raise SGLangEventNativeError("Restore must use the captured decision-scope boundary")
        remote = self._exact_request("restore", snapshot_id)
        if remote.get("verified_from_live_state") is not True or remote["component_digests"] != saved["remote"]["component_digests"]:
            raise SGLangEventNativeError("Backend failed live exact-state verification")
        session = local["session"]
        self._session_cache = None if session is None else _Session(
            session_id=session["session_id"], generation=session["generation"],
            handles=set(session["handles"]),
        )
        if scope is not None:
            for name, value in local["scope"].items():
                setattr(scope, name, set(value) if name in {"session_handles", "retained_handles"} else value)
        self._requests_submitted = local["requests_submitted"]
        self.extraction_calls_reserved = local["extraction_calls_reserved"]
        self.sampling_params = local["sampling_params"]
        self.last_generation_trace = local["last_generation_trace"]
        self.last_cache_lifecycle_trace = local["last_cache_lifecycle_trace"]
        self._model_binding = local["model_binding"]
        self._kv_bytes_per_token = local["kv_bytes_per_token"]
        receipt = self._exact_receipt(remote, self._exact_local_state())
        if receipt["component_digests"] != snapshot["component_digests"]:
            raise SGLangEventNativeError("Adapter state differs after exact restoration")
        receipt["verified_from_live_state"] = True
        return receipt

    def release_exact_state(self, snapshot: Mapping[str, Any]) -> None:
        snapshot_id = snapshot.get("snapshot_id")
        if snapshot_id not in self._exact_snapshots:
            raise SGLangEventNativeError("Unknown exact adapter snapshot")
        response = self._exact_request("release", snapshot_id)
        if response.get("released") is not True:
            raise SGLangEventNativeError("Backend did not release the exact snapshot")
        del self._exact_snapshots[snapshot_id]

    @property
    def shadow_feature_config(self) -> "ShadowFeatureConfig | None":
        return self._shadow_feature_config

    @shadow_feature_config.setter
    def shadow_feature_config(self, config: "ShadowFeatureConfig | None") -> None:
        if config is not None:
            if type(getattr(config, "enabled", None)) is not bool:
                raise TypeError(
                    "shadow_feature_config must expose the ShadowFeatureConfig contract"
                )
            prefill_layer = getattr(config, "prefill_layer", None)
            if prefill_layer is not None and type(prefill_layer) is not int:
                raise TypeError("shadow_feature_config.prefill_layer must be int or None")
            if getattr(config, "memgen_layer", None) is not None:
                raise ValueError(
                    "SGLang native packed generation supports prefill shadow features only"
                )
        self._shadow_feature_config = config

    def configure_shadow_features(self, config: "ShadowFeatureConfig | None") -> None:
        self.shadow_feature_config = config

    def kv_bytes_per_token(self) -> int:
        """Read the serving model's exact KV unit once, without loading weights."""

        self._ensure_model_info()
        assert self._kv_bytes_per_token is not None
        return self._kv_bytes_per_token

    @contextmanager
    def decision_scope(self, *, session_id: str | None = None) -> Iterator[None]:
        if self._active_decision_scope is not None:
            raise RuntimeError("decision_scope is not reentrant")
        if session_id is not None and (
            not isinstance(session_id, str) or not session_id
        ):
            raise ValueError("session_id must be None or a nonempty string")

        reset_reason = None
        if session_id is None:
            self._session_cache = None
            session_handles: set[str] = set()
        else:
            if self._session_cache is not None and self._session_cache.session_id != session_id:
                self._session_cache = None
                reset_reason = "session_id_changed"
            session_handles = (
                set(self._session_cache.handles)
                if self._session_cache is not None
                else set()
            )
        scope = _DecisionScope(
            session_id=session_id,
            session_handles=session_handles,
            reset_reason=reset_reason,
        )
        self._active_decision_scope = scope
        try:
            yield
        except BaseException:
            if scope.pending_stats is not None:
                scope.pending_stats["session_cache_commit_status"] = "cleared_on_failure"
            if session_id is not None:
                self._session_cache = None
            raise
        else:
            if session_id is not None and scope.pending_stats is not None:
                generation = (
                    self._session_cache.generation + 1
                    if self._session_cache is not None
                    else 1
                )
                self._session_cache = _Session(
                    session_id=session_id,
                    generation=generation,
                    handles=set(scope.retained_handles),
                )
                scope.pending_stats["session_cache_commit_status"] = "committed"
        finally:
            self._active_decision_scope = None

    def close_session(self) -> None:
        if (
            self._active_decision_scope is not None
            and self._active_decision_scope.session_id is not None
        ):
            raise RuntimeError("cannot close a session inside its active decision_scope")
        self._session_cache = None

    def session_cache_info(self) -> dict[str, Any]:
        session = self._session_cache
        return {
            "policy": self.session_cache_policy,
            "session_id": session.session_id if session is not None else None,
            "generation": session.generation if session is not None else 0,
            "cpu_memo_present": False,
            "cpu_memo_logical_bytes": 0,
            "device_raw_snapshot_present": False,
            "device_raw_snapshot_logical_bytes": 0,
            "device_raw_snapshot_backing_bytes": 0,
            "device_raw_snapshot_prefix_tokens": 0,
            "device_raw_snapshot_raw_tokens": 0,
            "last_transfer_bytes_in": 0,
            "last_transfer_bytes_out": 0,
            "last_transfer_bytes_total": 0,
            "chunk_handles_present": bool(session and session.handles),
            "chunk_handle_count": len(session.handles) if session is not None else 0,
            "external_cache_state": "content_addressed_lru_unowned",
            "remote_pins_held": False,
            "last_lifecycle_trace": self.last_cache_lifecycle_trace,
        }

    def generate(
        self,
        memory: PackedMemory,
        *,
        ratio: int,
        max_new_tokens: int,
        eos_token_id: int | Sequence[int] | None = None,
        trace_context: Mapping[str, Any] | None = None,
        compression_chunks: Sequence[EncoderChunk] | None = None,
    ) -> SGLangEventNativeGenerationResult:
        """Submit one exact packed decision.  A failure is terminal and unretried."""

        self.last_generation_trace = None
        if not isinstance(memory, PackedMemory):
            raise TypeError("memory must be PackedMemory")
        ratio = _positive_int(ratio, "ratio")
        requested_tokens = _positive_int(max_new_tokens, "max_new_tokens")
        if requested_tokens > self.max_new_tokens:
            raise ValueError("max_new_tokens exceeds this adapter's finite cap")
        if trace_context is not None and not isinstance(trace_context, Mapping):
            raise TypeError("trace_context must be a mapping or None")
        if self._requests_submitted >= self.max_generation_calls:
            raise RuntimeError("SGLang generation cap exhausted; automatic retry is disabled")

        scope = self._active_decision_scope
        if self._session_cache is not None and scope is None:
            self._session_cache = None
        if scope is not None and scope.pending_stats is not None:
            scope.pending_stats["session_cache_commit_status"] = "discarded_by_regeneration"
            scope.pending_stats = None

        self._ensure_model_info()
        selected, extras = self._prepare_chunks(memory, ratio, compression_chunks)
        logical_tokens = (
            len(memory.system_input_ids)
            + sum(len(item["token_ids"]) for item in selected)
            + len(memory.workspace_input_ids)
        )
        if logical_tokens + requested_tokens > self.model_context:
            raise ValueError("logical packed prompt plus completion exceeds model_context")

        generation_index = (scope.generate_calls + 1) if scope is not None else 1
        context = self._trace_context(trace_context)
        session_id = scope.session_id if scope is not None else context.get("session_id")
        generation_id = self._generation_id(context, generation_index)
        remaining_extractions = (
            self.max_extraction_calls - self.extraction_calls_reserved
        )
        if remaining_extractions < 0:
            raise RuntimeError("finite extraction-call cap is already exhausted")

        shadow_request = None
        config = self.shadow_feature_config
        if config is not None and config.enabled:
            shadow_request = {"prefill_layer": config.prefill_layer}
        effective_eos_ids = (
            self.eos_token_ids if eos_token_id is None else _eos_ids(eos_token_id)
        )
        engine_sampling = copy.deepcopy(self.sampling_params)
        if "seed" in engine_sampling:
            seed = engine_sampling.pop("seed")
            if "sampling_seed" in engine_sampling and engine_sampling["sampling_seed"] != seed:
                raise ValueError("seed and sampling_seed must agree")
            engine_sampling["sampling_seed"] = seed
        payload = {
            "schema": REQUEST_SCHEMA,
            "rid": context.get("attempt_uid") or generation_id,
            "session_id": session_id,
            "generation_id": generation_id,
            "packing_version": PACKING_VERSION,
            "raw_layout_profile": memory.raw_layout_profile,
            "encoding_scope": self.encoding_scope,
            "system_input_ids": list(memory.system_input_ids),
            "workspace_input_ids": list(memory.workspace_input_ids),
            "encoder_chunks": selected,
            "compression_chunks": extras,
            "compression_ratio": ratio,
            "max_extraction_calls": remaining_extractions,
            "sampling_params": {
                **engine_sampling,
                "max_new_tokens": requested_tokens,
                "stop_token_ids": list(effective_eos_ids),
            },
            "shadow_features": shadow_request,
        }

        request_index = self._requests_submitted + 1
        self._requests_submitted += 1
        if scope is not None:
            scope.generate_calls += 1
        started = time.perf_counter()
        try:
            response, http_status = self._post_native_generate(payload, request_index)
            self._charge_extractions(response)
            result = self._result_from_response(
                response,
                payload=payload,
                memory=memory,
                selected=selected,
                extras=extras,
                ratio=ratio,
                requested_tokens=requested_tokens,
                request_index=request_index,
                http_status=http_status,
                wall_seconds=time.perf_counter() - started,
                scope=scope,
                effective_eos_ids=effective_eos_ids,
            )
        except BaseException:
            if scope is not None and scope.session_id is not None:
                self._session_cache = None
            raise

        if scope is not None:
            scope.retained_handles = {item["handle"] for item in selected}
            scope.pending_stats = result.stats
        return result

    def _ensure_model_info(self) -> None:
        if self._model_binding is not None:
            return
        request = Request(
            self.upstream + "/model_info",
            headers={"Accept": "application/json"},
            method="GET",
        )
        response, status = self._read_json(request, label="SGLang /model_info")
        if status != 200:
            raise SGLangEventNativeError(f"SGLang /model_info returned HTTP {status}")
        if not isinstance(response, Mapping):
            raise SGLangEventNativeError("SGLang /model_info must return a JSON object")
        native = response.get("c2kv_native_packed")
        if not isinstance(native, Mapping):
            raise SGLangEventNativeError(
                "SGLang /model_info lacks c2kv_native_packed admission data"
            )
        binding = _json_object(native.get("model_binding"), "model_binding")
        if not binding:
            raise SGLangEventNativeError("model_binding must not be empty")
        if binding.get("model_path") != self.expected_model_path:
            raise SGLangEventNativeError(
                "SGLang model_path does not match expected_model_path"
            )
        if response.get("model_path") != self.expected_model_path:
            raise SGLangEventNativeError(
                "SGLang top-level model_path does not match expected_model_path"
            )
        kv_bytes = _nonnegative_int(
            native.get("kv_bytes_per_token"), "kv_bytes_per_token"
        )
        if kv_bytes <= 0:
            raise SGLangEventNativeError("kv_bytes_per_token must be positive")
        self._model_binding = binding
        self._kv_bytes_per_token = kv_bytes

    def _prepare_chunks(
        self,
        memory: PackedMemory,
        ratio: int,
        compression_chunks: Sequence[EncoderChunk] | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        placements = memory.gist_layout(ratio)
        selected = [
            self._chunk_payload(
                placement.chunk,
                ratio,
                source_position_start=placement.source_position_start,
                gist_position_ids=placement.position_ids,
            )
            for placement in placements
        ]
        selected_handles = {item["handle"] for item in selected}
        if len(selected_handles) != len(selected):
            raise ValueError("memory contains duplicate selected encoder chunks")

        extras: list[dict[str, Any]] = []
        if compression_chunks is not None:
            if not isinstance(compression_chunks, Sequence) or isinstance(
                compression_chunks, (str, bytes, bytearray)
            ):
                raise TypeError("compression_chunks must be a sequence or None")
            all_eligible: dict[str, dict[str, Any]] = {}
            for chunk in compression_chunks:
                item = self._chunk_payload(chunk, ratio)
                previous = all_eligible.setdefault(item["handle"], item)
                if previous != item:
                    raise ValueError("stable chunk handle collision")
            if not selected_handles <= set(all_eligible):
                raise ValueError(
                    "eligible compression chunks must include every retained gist chunk"
                )
            extras = [
                item
                for handle, item in all_eligible.items()
                if handle not in selected_handles
            ]
        return selected, extras

    def _chunk_payload(
        self,
        chunk: EncoderChunk,
        ratio: int,
        *,
        source_position_start: int | None = None,
        gist_position_ids: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(chunk, EncoderChunk):
            raise TypeError("encoder chunks must be EncoderChunk instances")
        token_ids = _token_ids(chunk.token_ids, "chunk.token_ids", nonempty=True)
        descriptor = {
            "event_id": chunk.event_id,
            "part_index": chunk.part_index,
            "source_indices": list(chunk.source_indices),
            "source_token_start": chunk.source_token_start,
            "source_token_end": chunk.source_token_end,
            "token_ids": list(token_ids),
        }
        if not isinstance(chunk.event_id, str) or not chunk.event_id:
            raise ValueError("chunk.event_id must be a nonempty string")
        for name in ("part_index", "source_token_start", "source_token_end"):
            if type(descriptor[name]) is not int or descriptor[name] < 0:
                raise ValueError(f"chunk.{name} must be a nonnegative integer")
        _token_ids(chunk.source_indices, "chunk.source_indices", nonempty=True)
        if chunk.source_token_end <= chunk.source_token_start:
            raise ValueError("chunk source token range must be nonempty")

        chunk_id = hashlib.sha256(_canonical_bytes(descriptor)).hexdigest()
        canonical_chunk = {"chunk_id": chunk_id, **descriptor}
        assert self._model_binding is not None
        handle_input = {
            "schema": HANDLE_SCHEMA,
            "model_binding": self._model_binding,
            "packing_version": PACKING_VERSION,
            "encoding_scope": self.encoding_scope,
            "compression_ratio": ratio,
            "chunk": canonical_chunk,
        }
        handle = hashlib.sha256(_canonical_bytes(handle_input)).hexdigest()
        result = {**canonical_chunk, "handle": handle}
        if source_position_start is not None:
            if type(source_position_start) is not int or source_position_start < 0:
                raise ValueError("source_position_start must be nonnegative")
            positions = _token_ids(
                gist_position_ids, "gist_position_ids", nonempty=True
            )
            result.update(
                source_position_start=source_position_start,
                gist_position_ids=list(positions),
            )
        return result

    def _post_native_generate(
        self, payload: Mapping[str, Any], request_index: int
    ) -> tuple[Any, int]:
        body = _canonical_bytes(payload)
        request = Request(
            self.upstream + "/v1/c2kv/native_generate",
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        if self._http_journal is not None:
            self._http_journal.append(
                {
                    "schema": HTTP_JOURNAL_SCHEMA,
                    "event": "request",
                    "status": "started",
                    "request_index": request_index,
                    "timestamp_unix_ns": time.time_ns(),
                    "request": payload,
                    "retries": 0,
                }
            )
        response: Any = None
        status: int | None = None
        try:
            response, status = self._read_json(
                request, label="SGLang /v1/c2kv/native_generate"
            )
            if status != 200:
                error = response.get("error") if isinstance(response, Mapping) else None
                if (
                    status == 400
                    and isinstance(error, Mapping)
                    and error.get("code") == EXTRACTION_BUDGET_ERROR_CODE
                ):
                    receipt = self._validate_extraction_budget_failure(response, payload)
                    model_calls = receipt["model_calls"]
                    if self.extraction_calls_reserved + model_calls > self.max_extraction_calls:
                        raise SGLangEventNativeError(
                            "extraction-budget failure receipt exceeds the task cap"
                        )
                    self.extraction_calls_reserved += model_calls
                    raise SGLangExtractionBudgetExhausted(
                        error["message"], receipt
                    )
                message = (
                    error.get("message") if isinstance(error, Mapping) else error
                )
                suffix = f": {message}" if isinstance(message, str) and message else ""
                raise SGLangEventNativeError(
                    f"SGLang native_generate returned HTTP {status}{suffix}"
                )
            return response, status
        finally:
            if self._http_journal is not None:
                self._http_journal.append(
                    {
                        "schema": HTTP_JOURNAL_SCHEMA,
                        "event": "response",
                        "status": "completed" if status == 200 else "failed",
                        "request_index": request_index,
                        "timestamp_unix_ns": time.time_ns(),
                        "http_status": status,
                        "response": response,
                        "retries": 0,
                    }
                )

    def _read_json(self, request: Request, *, label: str) -> tuple[Any, int]:
        try:
            with self._opener.open(request, timeout=self.timeout_seconds) as opened:
                status_value = getattr(opened, "status", None)
                status = int(opened.getcode() if status_value is None else status_value)
                raw = opened.read(self.max_response_bytes + 1)
        except HTTPError as error:
            status = int(error.code)
            try:
                raw = error.read(self.max_response_bytes + 1)
            except OSError:
                raw = b""
        except (OSError, TimeoutError) as error:
            raise SGLangEventNativeError(f"{label} transport failed without retry") from error
        if len(raw) > self.max_response_bytes:
            raise SGLangEventNativeError(f"{label} response exceeds the byte cap")
        try:
            return json.loads(raw.decode("utf-8")), status
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SGLangEventNativeError(f"{label} did not return valid UTF-8 JSON") from error

    def _charge_extractions(self, response: Any) -> None:
        if not isinstance(response, Mapping):
            return
        extraction = response.get("extraction")
        if not isinstance(extraction, Mapping):
            return
        value = extraction.get("model_calls")
        if type(value) is int and value >= 0:
            self.extraction_calls_reserved += value

    def _validate_extraction_budget_failure(
        self, response: Any, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        if not isinstance(response, Mapping):
            raise SGLangEventNativeError(
                "extraction-budget failure response must be an object"
            )
        error = response.get("error")
        if not isinstance(error, Mapping):
            raise SGLangEventNativeError(
                "extraction-budget failure error must be an object"
            )
        if error.get("code") != EXTRACTION_BUDGET_ERROR_CODE:
            raise SGLangEventNativeError("extraction-budget failure code mismatch")
        message = error.get("message")
        if not isinstance(message, str) or not message.startswith(
            EXTRACTION_BUDGET_ERROR_CODE + ":"
        ):
            raise SGLangEventNativeError(
                "extraction-budget failure message mismatch"
            )
        value = response.get("extraction")
        if not isinstance(value, Mapping):
            raise SGLangEventNativeError(
                "extraction-budget failure receipt must be an object"
            )
        if value.get("schema") != EXTRACTION_FAILURE_SCHEMA:
            raise SGLangEventNativeError(
                "extraction-budget failure receipt schema mismatch"
            )
        names = (
            "requested_chunks",
            "unique_chunks",
            "processed_chunks",
            "cache_hits",
            "cache_misses",
            "model_calls",
            "max_extraction_calls",
            "failed_chunk_index",
        )
        result = {
            name: _nonnegative_int(
                value.get(name), f"response.extraction.{name}"
            )
            for name in names
        }
        rows = [*payload["encoder_chunks"], *payload["compression_chunks"]]
        handles = {row["handle"] for row in rows}
        if result["requested_chunks"] != len(rows):
            raise SGLangEventNativeError(
                "extraction-budget failure requested_chunks mismatch"
            )
        if result["unique_chunks"] != len(handles):
            raise SGLangEventNativeError(
                "extraction-budget failure unique_chunks mismatch"
            )
        if result["processed_chunks"] != (
            result["cache_hits"] + result["cache_misses"]
        ):
            raise SGLangEventNativeError(
                "extraction-budget failure processed count mismatch"
            )
        if result["failed_chunk_index"] != result["processed_chunks"]:
            raise SGLangEventNativeError(
                "extraction-budget failure index mismatch"
            )
        if result["processed_chunks"] >= result["unique_chunks"]:
            raise SGLangEventNativeError(
                "extraction-budget failure does not identify an unprocessed miss"
            )
        if result["model_calls"] != result["cache_misses"]:
            raise SGLangEventNativeError(
                "extraction-budget failure model_calls/cache_misses disagree"
            )
        if result["max_extraction_calls"] != payload["max_extraction_calls"]:
            raise SGLangEventNativeError(
                "extraction-budget failure max_extraction_calls mismatch"
            )
        if result["model_calls"] != result["max_extraction_calls"]:
            raise SGLangEventNativeError(
                "extraction-budget failure did not consume the request budget"
            )
        return {"schema": EXTRACTION_FAILURE_SCHEMA, **result}

    def _result_from_response(
        self,
        response: Any,
        *,
        payload: Mapping[str, Any],
        memory: PackedMemory,
        selected: Sequence[Mapping[str, Any]],
        extras: Sequence[Mapping[str, Any]],
        ratio: int,
        requested_tokens: int,
        request_index: int,
        http_status: int,
        wall_seconds: float,
        scope: _DecisionScope | None,
        effective_eos_ids: tuple[int, ...],
    ) -> SGLangEventNativeGenerationResult:
        if not isinstance(response, Mapping):
            raise SGLangEventNativeError("native_generate response must be a JSON object")
        for name, expected in (
            ("schema", RESPONSE_SCHEMA),
            ("rid", payload["rid"]),
            ("session_id", payload["session_id"]),
            ("generation_id", payload["generation_id"]),
        ):
            if response.get(name) != expected:
                raise SGLangEventNativeError(f"native_generate response {name} mismatch")
        if not isinstance(response.get("text"), str):
            raise SGLangEventNativeError("native_generate response lacks text")
        output_ids = _token_ids(response.get("output_ids"), "response.output_ids")
        raw_logprobs = response.get("token_logprobs")
        if not isinstance(raw_logprobs, Sequence) or isinstance(
            raw_logprobs, (str, bytes, bytearray)
        ):
            raise SGLangEventNativeError("response.token_logprobs must be a sequence")
        if len(raw_logprobs) != len(output_ids):
            raise SGLangEventNativeError("response token/logprob counts disagree")
        logprobs = []
        for value in raw_logprobs:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SGLangEventNativeError("response token_logprobs must be numeric")
            value = float(value)
            if not math.isfinite(value):
                raise SGLangEventNativeError("response token_logprobs must be finite")
            logprobs.append(value)
        finish_reason = response.get("finish_reason")
        if isinstance(finish_reason, Mapping):
            finish_reason = finish_reason.get("type")
        if not isinstance(finish_reason, str) or not finish_reason:
            raise SGLangEventNativeError("response.finish_reason must be nonempty")
        if len(output_ids) > requested_tokens:
            raise SGLangEventNativeError("response exceeds requested max_new_tokens")

        selected_rows = self._validate_chunk_results(
            response.get("encoder_chunks"), selected, "encoder_chunks", ratio
        )
        extra_rows = self._validate_chunk_results(
            response.get("compression_chunks"), extras, "compression_chunks", ratio
        )
        extraction = self._validate_extraction(
            response.get("extraction"),
            selected_rows + extra_rows,
            expected_max_extraction_calls=payload["max_extraction_calls"],
        )
        costs = self._validate_costs(response.get("costs"), memory, ratio)
        shadow = self._validate_shadow_features(
            response.get("shadow_features"), memory
        )
        allocator = response.get("allocator")
        if allocator is not None:
            allocator = _json_object(allocator, "response.allocator")

        before_scope = set(scope.retained_handles) if scope is not None else set()
        before_session = set(scope.session_handles) if scope is not None else set()
        selected_by_handle = {row["handle"]: row for row in selected_rows}
        extras_by_handle = {row["handle"]: row for row in extra_rows}
        scope_reused = {
            handle for handle, row in selected_by_handle.items()
            if row["cache_hit"] and handle in before_scope
        }
        session_reused = {
            handle for handle, row in selected_by_handle.items()
            if row["cache_hit"] and handle in before_session and handle not in before_scope
        }
        unretained_reused = sum(row["cache_hit"] for row in extras_by_handle.values())
        kv_bytes = self.kv_bytes_per_token()
        prefix_tokens = costs["system_tokens"] + costs["gist_tokens"]
        prompt_resident_tokens = costs["resident_kv_tokens"]
        final_resident_tokens = prompt_resident_tokens + max(0, len(output_ids) - 1)
        generation_index = scope.generate_calls if scope is not None else 1

        stats = {
            "schema": GENERATION_SCHEMA,
            "backend": "sglang_c2kv_native_packed",
            "raw_layout_profile": memory.raw_layout_profile,
            "packing_version": PACKING_VERSION,
            "encoding_scope": self.encoding_scope,
            "ratio": ratio,
            "requested_max_new_tokens": requested_tokens,
            "generated_tokens": len(output_ids),
            "workspace_tokens": len(memory.workspace_input_ids),
            "source_tokens": (
                len(memory.system_input_ids)
                + sum(len(chunk.token_ids) for chunk in memory.chunks)
                + len(memory.workspace_input_ids)
            ),
            "packed_encoder_tokens": costs["presented_encoder_tokens"],
            "materialized_encoder_tokens": extraction["materialized_encoder_tokens"],
            "scope_reused_encoder_tokens": sum(
                len(selected_by_handle[handle]["token_ids"])
                for handle in scope_reused
            ),
            "gist_tokens": costs["gist_tokens"],
            "system_prefix_kv_tokens": costs["system_tokens"],
            "gist_prefix_kv_tokens": costs["gist_tokens"],
            "system_prefix_kv_logical_bytes": costs["system_prefix_kv_logical_bytes"],
            "gist_prefix_kv_logical_bytes": costs["gist_prefix_kv_logical_bytes"],
            "resident_prefix_kv_tokens": prefix_tokens,
            "resident_prefix_kv_bytes": prefix_tokens * kv_bytes,
            "kv_bytes_per_token": kv_bytes,
            "unique_chunks": len(selected_by_handle),
            "chunk_placements": len(selected_rows),
            "max_extraction_calls": self.max_extraction_calls,
            "extraction_calls_reserved": self.extraction_calls_reserved,
            "extracted_chunks": extraction["model_calls"],
            "unretained_extracted_chunks": sum(
                not row["cache_hit"] for row in extras_by_handle.values()
            ),
            "unretained_reused_chunks": unretained_reused,
            "scope_reused_chunks": len(scope_reused),
            "reused_chunk_placements": len(selected_rows) - len(selected_by_handle),
            "scope_reused_chunk_placements": len(scope_reused),
            # The engine contract does not expose exact radix-forward work.
            "system_prefill_calls": None,
            "system_prefill_tokens": None,
            "scope_reused_system_prefill_calls": None,
            "scope_reused_system_prefill_tokens": None,
            "decision_scope_active": scope is not None,
            "decision_scope_generation_index": generation_index,
            "scope_system_kv_logical_bytes_before": 0,
            "scope_system_kv_logical_bytes_after": 0,
            "scope_gist_kv_logical_bytes_before": 0,
            "scope_gist_kv_logical_bytes_after": 0,
            "decode_strategy": "external_sglang_incremental",
            "prefill_chunk_size": None,
            "target_execution_strategy": "external_sglang_incremental",
            "target_forward_calls": None,
            "target_input_tokens": None,
            "recomputed_raw_tokens": None,
            "raw_prefill_forward_calls": None,
            "raw_prefill_input_tokens": None,
            "one_token_decode_forward_calls": None,
            "one_token_decode_input_tokens": None,
            "raw_recompute_forward_calls": 0,
            "raw_recompute_input_tokens": 0,
            "suffix_recompute_tokens": 0,
            "resident_kv_tokens_after_raw_prefill": prompt_resident_tokens,
            "resident_kv_logical_bytes_after_raw_prefill": costs[
                "resident_kv_logical_bytes"
            ],
            "resident_kv_tokens_final": final_resident_tokens,
            "resident_kv_logical_bytes_final": final_resident_tokens * kv_bytes,
            "torch_allocator_peak_allocated_bytes": None,
            "server_kv_memory_report": allocator,
            "session_cache_policy": self.session_cache_policy,
            "session_scope_active": scope is not None and scope.session_id is not None,
            "session_reused_raw_tokens": 0,
            "session_reused_gist_chunks": len(session_reused),
            "session_reused_encoder_tokens": sum(
                len(selected_by_handle[handle]["token_ids"])
                for handle in session_reused
            ),
            "session_reused_system_tokens": 0,
            "session_cpu_memo_gist_hits": len(session_reused),
            "session_cpu_memo_system_hit": False,
            "session_cache_cpu_logical_bytes_before": 0,
            "session_pending_device_resident_logical_bytes": 0,
            "session_pending_device_backing_bytes": 0,
            "session_cache_transfer_bytes_in": 0,
            "raw_cache_reuse_eligibility_reason": "external_server_session_disabled",
            "session_cache_reset_reason": scope.reset_reason if scope is not None else None,
            "session_cache_eviction_reason": (
                "same_decision_regeneration"
                if scope is not None and generation_index > 1
                else None
            ),
            "session_cache_commit_status": (
                "pending"
                if scope is not None and scope.session_id is not None
                else "not_applicable"
            ),
            "raw_recompute_strategy": "unknown_external_sglang_radix_execution",
            "parameter_version": copy.deepcopy(self._model_binding),
            "eos_token_ids": list(
                effective_eos_ids
            ),
            "eos_source": self.eos_source,
            "elapsed_sec": wall_seconds,
            "sglang_transport": {
                "schema": HTTP_JOURNAL_SCHEMA,
                "url": self.upstream + "/v1/c2kv/native_generate",
                "http_status": http_status,
                "request_index": request_index,
                "wall_seconds": wall_seconds,
                "timeout_seconds": self.timeout_seconds,
                "retries": 0,
                "rid": payload["rid"],
                "generation_id": payload["generation_id"],
            },
            "sglang_extraction": extraction,
            "sglang_chunk_handles": {
                "selected": [row["handle"] for row in selected_rows],
                "compression_only": [row["handle"] for row in extra_rows],
                "server_cache_keys": [
                    row["cache_key"] for row in selected_rows + extra_rows
                ],
            },
        }
        if shadow is not None:
            stats["shadow_features"] = shadow
        return SGLangEventNativeGenerationResult(
            token_ids=output_ids,
            finish_reason=finish_reason,
            token_logprobs=tuple(logprobs),
            stats=stats,
        )

    def _validate_chunk_results(
        self,
        value: Any,
        requested: Sequence[Mapping[str, Any]],
        name: str,
        ratio: int,
    ) -> list[dict[str, Any]]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise SGLangEventNativeError(f"response.{name} must be a sequence")
        rows = list(value)
        if len(rows) != len(requested):
            raise SGLangEventNativeError(f"response.{name} count mismatch")
        validated = []
        for index, (raw, expected) in enumerate(zip(rows, requested, strict=True)):
            if not isinstance(raw, Mapping):
                raise SGLangEventNativeError(f"response.{name}[{index}] must be an object")
            for field_name in ("chunk_id", "handle"):
                if raw.get(field_name) != expected[field_name]:
                    raise SGLangEventNativeError(
                        f"response.{name}[{index}].{field_name} mismatch"
                    )
            cache_key = raw.get("cache_key")
            if not isinstance(cache_key, str) or not cache_key:
                raise SGLangEventNativeError(
                    f"response.{name}[{index}].cache_key must be nonempty"
                )
            if type(raw.get("cache_hit")) is not bool:
                raise SGLangEventNativeError(
                    f"response.{name}[{index}].cache_hit must be bool"
                )
            original = _nonnegative_int(
                raw.get("original_seq_len"),
                f"response.{name}[{index}].original_seq_len",
            )
            gist_len = _nonnegative_int(
                raw.get("gist_len"), f"response.{name}[{index}].gist_len"
            )
            if original != len(expected["token_ids"]):
                raise SGLangEventNativeError(
                    f"response.{name}[{index}] original_seq_len mismatch"
                )
            if gist_len != (original + ratio - 1) // ratio:
                raise SGLangEventNativeError(
                    f"response.{name}[{index}] gist_len mismatch"
                )
            validated.append(
                {
                    **dict(raw),
                    "token_ids": tuple(expected["token_ids"]),
                    "original_seq_len": original,
                    "gist_len": gist_len,
                }
            )
        return validated

    def _validate_extraction(
        self,
        value: Any,
        rows: Sequence[Mapping[str, Any]],
        *,
        expected_max_extraction_calls: int,
    ) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise SGLangEventNativeError("response.extraction must be an object")
        result = {
            name: _nonnegative_int(value.get(name), f"response.extraction.{name}")
            for name in (
                "requested_chunks",
                "unique_chunks",
                "cache_hits",
                "cache_misses",
                "model_calls",
            )
        }
        result["max_extraction_calls"] = _optional_nonnegative_int(
            value.get("max_extraction_calls"),
            "response.extraction.max_extraction_calls",
        )
        if result["max_extraction_calls"] != expected_max_extraction_calls:
            raise SGLangEventNativeError(
                "response.extraction.max_extraction_calls mismatch"
            )
        handles = {row["handle"] for row in rows}
        if result["requested_chunks"] != len(rows):
            raise SGLangEventNativeError("response.extraction.requested_chunks mismatch")
        if result["unique_chunks"] != len(handles):
            raise SGLangEventNativeError("response.extraction.unique_chunks mismatch")
        if result["cache_hits"] + result["cache_misses"] != len(handles):
            raise SGLangEventNativeError("response extraction hit/miss counts disagree")
        if result["model_calls"] != result["cache_misses"]:
            raise SGLangEventNativeError("response extraction model_calls/cache_misses disagree")
        row_hits: dict[str, bool] = {}
        for row in rows:
            previous = row_hits.setdefault(row["handle"], row["cache_hit"])
            if previous != row["cache_hit"]:
                raise SGLangEventNativeError("duplicate handle has inconsistent cache_hit")
        if sum(row_hits.values()) != result["cache_hits"]:
            raise SGLangEventNativeError("response extraction cache_hits mismatch")
        if self.extraction_calls_reserved > self.max_extraction_calls:
            raise SGLangEventNativeError("server exceeded finite extraction-call budget")
        result["materialized_encoder_tokens"] = sum(
            len(row["token_ids"])
            for row in {row["handle"]: row for row in rows}.values()
            if not row["cache_hit"]
        )
        return result

    def _validate_costs(
        self, value: Any, memory: PackedMemory, ratio: int
    ) -> dict[str, int]:
        if not isinstance(value, Mapping):
            raise SGLangEventNativeError("response.costs must be an object")
        names = (
            "system_tokens",
            "raw_tokens",
            "presented_encoder_tokens",
            "gist_tokens",
            "resident_kv_tokens",
            "system_prefix_kv_logical_bytes",
            "gist_prefix_kv_logical_bytes",
            "raw_workspace_kv_logical_bytes",
            "resident_kv_logical_bytes",
        )
        result = {
            name: _nonnegative_int(value.get(name), f"response.costs.{name}")
            for name in names
        }
        expected = memory.costs(ratio)
        for name in (
            "system_tokens",
            "raw_tokens",
            "presented_encoder_tokens",
            "gist_tokens",
            "resident_kv_tokens",
        ):
            if result[name] != expected[name]:
                raise SGLangEventNativeError(f"response.costs.{name} mismatch")
        kv_bytes = self.kv_bytes_per_token()
        byte_expectations = {
            "system_prefix_kv_logical_bytes": expected["system_tokens"] * kv_bytes,
            "gist_prefix_kv_logical_bytes": expected["gist_tokens"] * kv_bytes,
            "raw_workspace_kv_logical_bytes": expected["raw_tokens"] * kv_bytes,
            "resident_kv_logical_bytes": expected["resident_kv_tokens"] * kv_bytes,
        }
        for name, expected_value in byte_expectations.items():
            if result[name] != expected_value:
                raise SGLangEventNativeError(f"response.costs.{name} mismatch")
        return result

    def _validate_shadow_features(
        self, value: Any, memory: PackedMemory
    ) -> dict[str, Any] | None:
        config = self.shadow_feature_config
        enabled = config is not None and config.enabled
        if not enabled:
            if value is not None:
                raise SGLangEventNativeError(
                    "server returned shadow_features when they were disabled"
                )
            return None
        if not isinstance(value, Mapping):
            raise SGLangEventNativeError("response.shadow_features must be an object")
        result = _json_object(value, "response.shadow_features")
        if result.get("schema") != SHADOW_FEATURE_SCHEMA:
            raise SGLangEventNativeError("response shadow feature schema mismatch")
        prefill = result.get("prefill")
        if not isinstance(prefill, Mapping):
            raise SGLangEventNativeError("response shadow_features.prefill is missing")
        if prefill.get("status") == "captured":
            if type(prefill.get("layer")) is not int or prefill["layer"] < 0:
                raise SGLangEventNativeError("captured prefill layer must be normalized")
            position = prefill.get("position")
            expected_position = (
                len(memory.system_input_ids)
                + sum(len(chunk.token_ids) for chunk in memory.chunks)
                + len(memory.workspace_input_ids)
                - 1
            )
            if (
                not isinstance(position, Mapping)
                or position.get("kind") != "prompt_last"
                or position.get("logical_position") != expected_position
            ):
                raise SGLangEventNativeError("captured prompt_last position mismatch")
            if prefill.get("readout") != "decoder_layer_output":
                raise SGLangEventNativeError("captured prefill readout mismatch")
            hidden = prefill.get("hidden")
            if not isinstance(hidden, list) or not hidden:
                raise SGLangEventNativeError("captured prefill hidden vector is empty")
            if any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in hidden
            ):
                raise SGLangEventNativeError("captured prefill hidden vector is invalid")
        elif prefill.get("status") != "unavailable":
            raise SGLangEventNativeError("prefill status must be captured or unavailable")
        return result

    @staticmethod
    def _trace_context(value: Mapping[str, Any] | None) -> dict[str, Any]:
        if value is None:
            return {}
        allowed = {"attempt_uid", "session_id", "decision_key", "phase"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                "trace_context has unsupported fields: " + ", ".join(sorted(unknown))
            )
        result = dict(value)
        for name, item in result.items():
            if not isinstance(item, str) or not item:
                raise ValueError(f"trace_context.{name} must be a nonempty string")
        return result

    @staticmethod
    def _generation_id(context: Mapping[str, Any], generation_index: int) -> str:
        attempt_uid = context.get("attempt_uid")
        if isinstance(attempt_uid, str):
            return attempt_uid
        material = {
            "session_id": context.get("session_id"),
            "decision_key": context.get("decision_key"),
            "phase": context.get("phase"),
            "generation_index": generation_index,
            "nonce": time.time_ns(),
        }
        return hashlib.sha256(_canonical_bytes(material)).hexdigest()


__all__ = [
    "EXTRACTION_BUDGET_ERROR_CODE",
    "EXTRACTION_FAILURE_SCHEMA",
    "GENERATION_SCHEMA",
    "HANDLE_SCHEMA",
    "HTTP_JOURNAL_SCHEMA",
    "REQUEST_SCHEMA",
    "RESPONSE_SCHEMA",
    "SESSION_CACHE_POLICY",
    "SGLangEventNativeError",
    "SGLangExtractionBudgetExhausted",
    "SGLangEventNativeGenerationResult",
    "SGLangEventNativeGenerator",
]
