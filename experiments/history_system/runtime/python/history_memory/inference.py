"""Greedy event-native generation over a packed history-memory decision."""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import torch
from transformers import DynamicCache

from .evidence import EVIDENCE_VERSION
from .cache_trace import (
    CACHE_MEMORY_OPERATION_KINDS,
    CACHE_MEMORY_RANGE_PREFIX,
    CacheTrace,
)
from .packing import PACKING_VERSION, RAW_LAYOUT_PROFILE, PackedMemory, EncoderChunk
from .runtime import HistoryMemoryModel
from .shadow_features import ShadowFeatureConfig, ShadowFeatureRecorder


TRAINING_PROFILE = "history-event-base-query-v1"
NORMAL_QUERY_PROFILE = "base"
GIST_TYPE = "dynamic-interleave"
GIST_PARAM = "qkv"
GIST_RESIDUAL_TYPE = "embed-mean"


@dataclass(frozen=True)
class EventNativeGenerationResult:
    """One greedy continuation and its reference-runtime accounting."""

    token_ids: tuple[int, ...]
    finish_reason: str
    token_logprobs: tuple[float, ...]
    stats: dict[str, Any]


@dataclass
class _RawCacheSnapshot:
    """One owned assembled-prefix plus forwarded-raw DynamicCache."""

    cache: DynamicCache | None
    prefix_signature: tuple[Any, ...]
    logical_start: int
    prefix_tokens: int
    raw_token_ids: tuple[int, ...]
    resident_logical_bytes: int
    backing_bytes: int
    provenance: dict[str, Any] = field(default_factory=dict)

    def clear(self) -> None:
        self.cache = None


@dataclass
class _SessionCache:
    """The single last-final view retained across explicit decision scopes."""

    session_id: str
    execution_signature: tuple[Any, ...]
    ratio: int | None = None
    generation: int = 0
    cpu_encoded_by_key: dict[tuple[Any, ...], Any] = field(default_factory=dict)
    encoded_provenance: dict[tuple[Any, ...], dict[str, Any]] = field(default_factory=dict)
    system_provenance: dict[str, Any] | None = None
    cpu_system_tokens: tuple[int, ...] = ()
    cpu_system_key_values: Any = ()
    device_snapshot: _RawCacheSnapshot | None = None
    last_transfer_bytes_in: int = 0
    last_transfer_bytes_out: int = 0

    def clear(self) -> None:
        self.cpu_encoded_by_key.clear()
        self.encoded_provenance.clear()
        self.system_provenance = None
        self.cpu_system_tokens = ()
        self.cpu_system_key_values = ()
        if self.device_snapshot is not None:
            self.device_snapshot.clear()
        self.device_snapshot = None
        self.ratio = None


@dataclass
class _DecisionScopeCache:
    """Per-decision reuse state plus an optional explicit-session handoff."""

    parameter_version: int
    model_id: int
    compute_device: torch.device
    compute_dtype: torch.dtype
    parameter_signature: tuple[tuple[int, int, torch.device, torch.dtype], ...]
    execution_signature: tuple[Any, ...]
    session_id: str | None = None
    session_source_snapshot: _RawCacheSnapshot | None = None
    pending_snapshot: _RawCacheSnapshot | None = None
    pending_encoding_keys: tuple[tuple[Any, ...], ...] = ()
    pending_system_tokens: tuple[int, ...] = ()
    pending_stats: dict[str, Any] | None = None
    session_reset_reason: str | None = None
    session_eviction_reason: str | None = None
    session_transfer_bytes_in: int = 0
    session_invalidated: bool = False
    ratio: int | None = None
    generate_calls: int = 0
    encoded_by_key: dict[tuple[Any, ...], Any] = field(default_factory=dict)
    system_by_tokens: dict[tuple[int, ...], Any] = field(default_factory=dict)
    encoded_provenance: dict[tuple[Any, ...], dict[str, Any]] = field(default_factory=dict)
    system_provenance: dict[tuple[int, ...], dict[str, Any]] = field(default_factory=dict)
    pending_trace: CacheTrace | None = None

    def clear(self) -> None:
        self.encoded_by_key.clear()
        self.system_by_tokens.clear()
        self.encoded_provenance.clear()
        self.system_provenance.clear()
        self.pending_trace = None
        if self.session_source_snapshot is not None:
            self.session_source_snapshot.clear()
        if self.pending_snapshot is not None:
            self.pending_snapshot.clear()
        self.session_source_snapshot = None
        self.pending_snapshot = None
        self.pending_encoding_keys = ()
        self.pending_system_tokens = ()
        self.pending_stats = None


class EventNativeGenerator:
    """Generate without targets from reusable event and system prefix KV.

    Incremental decode is the default for models with default RoPE and only
    full-attention layers. ``full_recompute`` preserves the reference path.
    A decision scope reuses exact event/system encodings across an initial
    draft and at most one regeneration. An explicit session id additionally
    commits the last successful view for the next decision: CPU event/system
    memo tensors for either strategy and one device raw snapshot for incremental
    decode.
    """

    session_cache_policy = "last-final-view-v1"
    cache_trace_schema = "event-native-cache-trace-v1"

    def __init__(
        self,
        runtime: HistoryMemoryModel,
        *,
        decode_strategy: str = "incremental",
        prefill_chunk_size: int | None = None,
        max_extraction_calls: int | None = None,
        retain_raw_snapshot: bool = True,
        shadow_feature_config: ShadowFeatureConfig | None = None,
    ) -> None:
        if not isinstance(runtime, HistoryMemoryModel):
            raise TypeError("runtime must be a HistoryMemoryModel")
        if decode_strategy not in ("incremental", "full_recompute"):
            raise ValueError(
                "decode_strategy must be 'incremental' or 'full_recompute'"
            )
        if prefill_chunk_size is not None and (
            type(prefill_chunk_size) is not int or prefill_chunk_size <= 0
        ):
            raise ValueError("prefill_chunk_size must be a positive integer or None")
        if decode_strategy == "full_recompute" and prefill_chunk_size is not None:
            raise ValueError(
                "prefill_chunk_size is supported only with incremental decode"
            )
        if max_extraction_calls is not None and (
            type(max_extraction_calls) is not int or max_extraction_calls <= 0
        ):
            raise ValueError("max_extraction_calls must be a positive integer or None")
        if type(retain_raw_snapshot) is not bool:
            raise TypeError("retain_raw_snapshot must be bool")
        self.retain_raw_snapshot = retain_raw_snapshot
        if not retain_raw_snapshot:
            self.session_cache_policy = "last-final-view-memo-only-v1"
        self.max_extraction_calls = max_extraction_calls
        self.extraction_calls_reserved = 0
        self.runtime = runtime
        self.decode_strategy = decode_strategy
        self.prefill_chunk_size = prefill_chunk_size
        self.shadow_feature_config = shadow_feature_config
        self._active_decision_scope: _DecisionScopeCache | None = None
        self._session_cache: _SessionCache | None = None
        self._cache_trace: CacheTrace | None = None
        self._cache_memory_annotations_active = False
        self.last_generation_trace: dict[str, Any] | None = None
        self.last_cache_lifecycle_trace: dict[str, Any] | None = None

    @property
    def shadow_feature_config(self) -> ShadowFeatureConfig | None:
        return self._shadow_feature_config

    @shadow_feature_config.setter
    def shadow_feature_config(self, config: ShadowFeatureConfig | None) -> None:
        if config is not None and not isinstance(config, ShadowFeatureConfig):
            raise TypeError("shadow_feature_config must be ShadowFeatureConfig or None")
        self._shadow_feature_config = config

    def configure_shadow_features(
        self,
        config: ShadowFeatureConfig | None,
    ) -> None:
        """Configure optional shadow reads between server load and generation."""

        self.shadow_feature_config = config

    def _encode_chunk_with_budget(self, chunk, ratio, operation):
        """Reserve actual encoder work; cache hits never enter this seam."""
        operation["max_extraction_calls"] = self.max_extraction_calls
        operation["extraction_call_reserved"] = False
        operation["extraction_calls_reserved_before"] = self.extraction_calls_reserved
        if (self.max_extraction_calls is not None
                and self.extraction_calls_reserved >= self.max_extraction_calls):
            raise RuntimeError("Finite extraction-call cap exhausted before encoding")
        self.extraction_calls_reserved += 1
        operation["extraction_call_reserved"] = True
        operation["extraction_call_index"] = self.extraction_calls_reserved
        # Failed encoder invocations remain charged; close_session cannot reset it.
        return self.runtime._encode_chunk(chunk, ratio)

    @contextmanager
    def decision_scope(self, *, session_id: str | None = None) -> Iterator[None]:
        """Bound one two-pass decision and optionally retain its final view.

        The scope is deliberately non-reentrant and permits at most two
        :meth:`generate` calls: an initial draft and one upgraded-memory
        regeneration. ``session_id=None`` releases all scope tensors on exit.
        A nonempty id retains only the final successful generation for reuse by
        the next scope with that id. Any generation or caller failure clears
        that session.
        """

        if self._active_decision_scope is not None:
            raise RuntimeError("decision_scope is not reentrant")
        if session_id is not None and (
            not isinstance(session_id, str) or not session_id
        ):
            raise ValueError("session_id must be None or a nonempty string")
        weight = self.runtime.base_model.model.embed_tokens.weight
        execution_signature = self._execution_signature()
        scope = _DecisionScopeCache(
            parameter_version=self.runtime.parameter_version,
            model_id=id(self.runtime.base_model),
            compute_device=weight.device,
            compute_dtype=weight.dtype,
            parameter_signature=self._model_parameter_signature(),
            execution_signature=execution_signature,
            session_id=session_id,
        )
        if session_id is None and self._session_cache is not None:
            self._discard_session_cache()
        if session_id is not None:
            reset_reason = None
            if (
                self._session_cache is not None
                and self._session_cache.session_id != session_id
            ):
                self._discard_session_cache()
                reset_reason = "session_id_changed"
            elif (
                self._session_cache is not None
                and self._session_cache.execution_signature != execution_signature
            ):
                self._discard_session_cache()
                raise RuntimeError(
                    "model execution configuration changed between session decisions"
                )
            if self._session_cache is None:
                self._session_cache = _SessionCache(
                    session_id=session_id,
                    execution_signature=execution_signature,
                )
            scope.session_source_snapshot = self._session_cache.device_snapshot
            self._session_cache.device_snapshot = None
            scope.session_reset_reason = reset_reason
        self._active_decision_scope = scope
        try:
            yield
        except BaseException:
            if session_id is not None:
                self._invalidate_session_scope(scope)
            raise
        else:
            if session_id is not None and not scope.session_invalidated:
                try:
                    self._commit_session_scope(scope)
                except BaseException:
                    if scope.pending_trace is not None:
                        for entry in scope.pending_trace.data["entries"]:
                            if entry["kind"] in ("encoded_cpu", "system_cpu"):
                                self._trace_release(entry, "unpublished_commit_failure", trace=scope.pending_trace)
                    self._invalidate_session_scope(scope)
                    raise
        finally:
            if self._cache_trace is not None:
                for entry in (*scope.encoded_provenance.values(), *scope.system_provenance.values()):
                    self._trace_release(entry, "decision_scope_wrapper_release")
            scope.clear()
            if self._active_decision_scope is scope:
                self._active_decision_scope = None
            # Returned traces are final after scope exit; later resets/close
            # receive an independent lifecycle record instead of rewriting them.
            self._cache_trace = None

    def close_session(self) -> None:
        """Release every cross-decision CPU memo and device cache tensor."""

        if (
            self._active_decision_scope is not None
            and self._active_decision_scope.session_id is not None
        ):
            raise RuntimeError("cannot close a session inside its active decision_scope")
        self._discard_session_cache(reason="explicit_session_close")

    def session_cache_info(self) -> dict[str, Any]:
        """Return bounded accounting for the currently retained session view."""

        session = self._session_cache
        if session is None:
            return {
                "policy": self.session_cache_policy,
                "session_id": None,
                "generation": 0,
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
                "last_lifecycle_trace": self.last_cache_lifecycle_trace,
            }
        snapshot = session.device_snapshot
        cpu_bytes = self._encoded_cache_logical_bytes(session.cpu_encoded_by_key)
        cpu_bytes += self._system_key_values_logical_bytes(
            session.cpu_system_key_values
        )
        return {
            "policy": self.session_cache_policy,
            "session_id": session.session_id,
            "generation": session.generation,
            "cpu_memo_present": bool(
                session.cpu_encoded_by_key or session.cpu_system_key_values
            ),
            "cpu_memo_logical_bytes": cpu_bytes,
            "device_raw_snapshot_present": snapshot is not None,
            "device_raw_snapshot_logical_bytes": (
                snapshot.resident_logical_bytes if snapshot is not None else 0
            ),
            "device_raw_snapshot_backing_bytes": (
                snapshot.backing_bytes if snapshot is not None else 0
            ),
            "device_raw_snapshot_prefix_tokens": (
                snapshot.prefix_tokens if snapshot is not None else 0
            ),
            "device_raw_snapshot_raw_tokens": (
                len(snapshot.raw_token_ids) if snapshot is not None else 0
            ),
            "last_transfer_bytes_in": session.last_transfer_bytes_in,
            "last_transfer_bytes_out": session.last_transfer_bytes_out,
            "last_transfer_bytes_total": (
                session.last_transfer_bytes_in + session.last_transfer_bytes_out
            ),
            "last_lifecycle_trace": self.last_cache_lifecycle_trace,
        }

    def generate(
        self,
        memory: PackedMemory,
        *,
        ratio: int,
        max_new_tokens: int,
        eos_token_id: int | Sequence[int] | None = None,
        trace_context: dict[str, Any] | None = None,
        compression_chunks: Sequence[EncoderChunk] | None = None,
    ) -> EventNativeGenerationResult:
        """Return a deterministic greedy continuation for one packed decision."""

        self.last_generation_trace = None
        trace = CacheTrace(trace_context)
        self._cache_trace = trace
        self.last_generation_trace = trace.data
        if self._active_decision_scope is None and self._session_cache is not None:
            self._discard_session_cache()
        try:
            extra = {} if compression_chunks is None else {'compression_chunks': compression_chunks}
            result = self._generate(
                memory,
                ratio=ratio,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                **extra,
            )
            trace.data["status"] = "completed"
            return result
        except BaseException as error:
            trace.data["status"] = "failed"
            trace.data["error_type"] = type(error).__name__
            scope = self._active_decision_scope
            if scope is not None and scope.session_id is not None:
                self._invalidate_session_scope(scope)
            raise
        finally:
            if self._active_decision_scope is None:
                self._cache_trace = None

    @contextmanager
    def cache_memory_annotations(self) -> Iterator[None]:
        """Opt in to profiler ranges around physical cache-memory operations."""

        if getattr(self, "_cache_memory_annotations_active", False):
            raise RuntimeError("cache_memory_annotations is not reentrant")
        self._cache_memory_annotations_active = True
        try:
            yield
        finally:
            self._cache_memory_annotations_active = False

    @contextmanager
    def _cache_op(self, kind: str, *, trace=None, **fields):
        """Record actual operation return, independently of later validation."""
        trace = trace or self._cache_trace
        fields.setdefault("input_tokens_requested", 0)
        fields.setdefault("transfer_bytes", 0)
        fields.setdefault("source_entry_id", None)
        fields.setdefault("result_entry_id", None)
        fields.setdefault("consumer_placement_ids", [])
        op = trace.start_op(kind, **fields)
        annotation_name = None
        if (
            getattr(self, "_cache_memory_annotations_active", False)
            and kind in CACHE_MEMORY_OPERATION_KINDS
        ):
            annotation_name = CACHE_MEMORY_RANGE_PREFIX + op["op_id"]
            op["memory_profile_annotation"] = annotation_name
        try:
            if annotation_name is None:
                yield op
            else:
                with torch.profiler.record_function(annotation_name):
                    yield op
        except BaseException as error:
            # A failed copy/forward may have performed partial work; it is unknown.
            trace.fail_op(op, error)
            op["input_tokens_completed"] = None
            op["transfer_bytes"] = None
            raise
        else:
            trace.finish_op(op, input_tokens_completed=op["input_tokens_requested"])

    def _trace_release(self, entry, reason, *, trace=None):
        if entry:
            with self._cache_op("release", trace=trace,
                                source_entry_id=entry["entry_id"], reason=reason):
                pass

    def _trace_placement(self, placement, op, entry, *, origin=None):
        chunk = placement.chunk
        row = self._cache_trace.add_placement(
            event_id=chunk.event_id, part_index=chunk.part_index,
            source_indices=list(chunk.source_indices),
            source_token_start=chunk.source_token_start,
            source_token_end=chunk.source_token_end,
            source_position_start=placement.source_position_start,
            position_ids=list(placement.position_ids), access_op_id=op["op_id"],
            accessed_entry_id=entry["entry_id"],
            origin_extraction_op_id=(origin or entry).get("origin_extraction_op_id"),
        )
        op.setdefault("consumer_placement_ids", []).append(row["placement_id"])
        return row

    def _generate(
        self,
        memory: PackedMemory,
        *,
        ratio: int,
        max_new_tokens: int,
        eos_token_id: int | Sequence[int] | None = None,
        compression_chunks: Sequence[EncoderChunk] | None = None,
    ) -> EventNativeGenerationResult:
        """Implement :meth:`generate` under its session failure boundary."""

        started = time.perf_counter()
        shadow_recorder = None
        self._validate_scalar_inputs(memory, ratio, max_new_tokens)
        self._validate_model_contract(ratio)
        if self.decode_strategy == "incremental":
            self._validate_incremental_model_contract()
        vocab_size = int(self.runtime.base_model.config.vocab_size)
        self._validate_memory_token_ids(memory, vocab_size)
        eos_ids, eos_source = self._resolve_eos_ids(eos_token_id, vocab_size)
        if (
            self.shadow_feature_config is not None
            and self.shadow_feature_config.enabled
        ):
            shadow_recorder = ShadowFeatureRecorder(
                self.shadow_feature_config,
                base_model=self.runtime.base_model,
                workspace_position_start=memory.workspace_position_start,
                workspace_tokens=len(memory.workspace_input_ids),
            )

        placements = memory.gist_layout(ratio)
        trace = self._cache_trace
        trace.data["workspace_source_group"] = list(memory.raw_source_indices)
        planned_gist_tokens = sum(len(placement.position_ids) for placement in placements)
        max_context = self._model_context_length()
        logical_end = (
            memory.workspace_position_start
            + len(memory.workspace_input_ids)
            + max_new_tokens
        )
        physical_end = (
            len(memory.system_input_ids)
            + planned_gist_tokens
            + len(memory.workspace_input_ids)
            + max_new_tokens
        )
        if logical_end > max_context:
            raise ValueError(
                "Requested generation exceeds the logical model context: "
                f"end_exclusive={logical_end}, context={max_context}"
            )
        if physical_end > max_context:
            raise ValueError(
                "Requested generation exceeds the physical model context: "
                f"end_exclusive={physical_end}, context={max_context}"
            )

        parameter_version = self.runtime.parameter_version
        scope = self._acquire_decision_scope(ratio)
        scope_generation_index = scope.generate_calls if scope is not None else 0
        session_scope_active = scope is not None and scope.session_id is not None
        session = self._session_cache if session_scope_active else None
        if session_scope_active and (
            session is None or session.session_id != scope.session_id
        ):
            raise RuntimeError("active decision scope lost its session cache")
        if session is not None:
            if session.ratio is None:
                session.ratio = ratio
            elif session.ratio != ratio:
                self._invalidate_session_scope(scope)
                raise RuntimeError("ratio changed between session decisions")
        if session_scope_active and scope.pending_stats is not None:
            if scope.pending_snapshot is not None:
                self._trace_release(scope.pending_snapshot.provenance,
                                    "discarded_by_regeneration", trace=scope.pending_trace)
                scope.pending_snapshot.clear()
            scope.pending_snapshot = None
            scope.pending_stats["session_cache_commit_status"] = (
                "discarded_by_regeneration"
            )
            scope.pending_trace.data["commit_status"] = "discarded_by_regeneration"
            scope.pending_trace = None
            scope.pending_stats = None
            scope.pending_encoding_keys = ()
            scope.pending_system_tokens = ()
            scope.session_eviction_reason = "same_decision_regeneration"

        placement_keys = tuple(
            placement.chunk.encoding_key(
                parameter_version=parameter_version,
                ratio=ratio,
            )
            for placement in placements
        )
        unique_encoding_keys = tuple(dict.fromkeys(placement_keys))
        extra_compression = {}
        if compression_chunks is not None:
            for chunk in compression_chunks:
                if (not isinstance(chunk, EncoderChunk) or not chunk.token_ids
                        or any(type(token) is not int or not 0 <= token < vocab_size for token in chunk.token_ids)):
                    raise ValueError('Invalid always-compress encoder chunk')
                key = chunk.encoding_key(parameter_version=parameter_version, ratio=ratio)
                extra_compression.setdefault(key, chunk)
            if not set(unique_encoding_keys) <= set(extra_compression):
                raise ValueError('Eligible compression chunks must include every retained gist chunk')
            for key in unique_encoding_keys:
                extra_compression.pop(key, None)
        prefix_signature = self._prefix_signature(
            memory,
            placements,
            placement_keys,
            ratio,
            scope.execution_signature if scope is not None else self._execution_signature(),
        )
        encoded_cache = scope.encoded_by_key if scope is not None else {}
        system_cache = scope.system_by_tokens if scope is not None else {}
        encoded_provenance = scope.encoded_provenance if scope is not None else {}
        system_provenance = scope.system_provenance if scope is not None else {}
        prefix_bindings = []
        system_binding = None
        parent_snapshot = None
        scope_system_bytes_before = (
            self._system_cache_logical_bytes(system_cache) if scope is not None else 0
        )
        scope_gist_bytes_before = (
            self._encoded_cache_logical_bytes(encoded_cache) if scope is not None else 0
        )
        encoded_for_call: dict[tuple[Any, ...], Any] = {}
        scope_keys_before_call = frozenset(encoded_cache) if scope is not None else frozenset()
        encoded = []
        prefix_key_values: Any = ()
        past_key_values: Any = None
        incremental_cache: Any = None
        outputs: Any = None
        generated: list[int] = []
        token_logprobs: list[float] = []
        target_input_tokens = 0
        raw_prefill_forward_calls = 0
        raw_prefill_input_tokens = 0
        one_token_decode_forward_calls = 0
        one_token_decode_input_tokens = 0
        raw_recompute_forward_calls = 0
        raw_recompute_input_tokens = 0
        resident_kv_tokens_after_raw_prefill: int | None = None
        resident_kv_logical_bytes_after_raw_prefill: int | None = None
        resident_kv_tokens_final: int | None = None
        resident_kv_logical_bytes_final: int | None = None
        materialized_encoder_tokens = 0
        scope_reused_encoder_tokens = 0
        extracted_chunks = 0
        unretained_extracted_chunks = 0
        unretained_reused_chunks = 0
        scope_reused_chunks = 0
        scope_reused_chunk_placements = 0
        system_prefill_calls = 0
        system_prefill_tokens = 0
        scope_reused_system_prefill_calls = 0
        scope_reused_system_prefill_tokens = 0
        target_forward_calls = 0
        session_reused_raw_tokens = 0
        session_reused_gist_chunks = 0
        session_reused_encoder_tokens = 0
        session_reused_system_tokens = 0
        session_cpu_memo_gist_hits = 0
        session_cpu_memo_system_hit = False
        session_cache_transfer_bytes_in = 0
        session_cache_cpu_logical_bytes_before = (
            self._session_cpu_memo_logical_bytes(session) if session is not None else 0
        )
        session_pending_device_resident_logical_bytes = 0
        session_pending_device_backing_bytes = 0
        raw_cache_reuse_eligibility_reason = (
            "session_disabled"
            if not session_scope_active
            else "same_decision_regeneration_cold"
            if scope_generation_index > 1
            else "raw_snapshot_disabled"
            if not self.retain_raw_snapshot
            else "no_committed_raw_snapshot"
        )
        raw_prefill_ids = memory.workspace_input_ids

        try:
            with self._temporary_eval(), torch.inference_mode(), self._base_autocast():
                # Always-compress also processes eligible history excluded by
                # the active-view budget. Keep the existing last-final-view
                # cache policy: unretained products are released immediately.
                for key, chunk in extra_compression.items():
                    if key in encoded_cache or (session is not None and key in session.cpu_encoded_by_key):
                        unretained_reused_chunks += 1
                        continue
                    with self._cache_op('extract', input_tokens_requested=len(chunk.token_ids),
                            event_id=chunk.event_id, part_index=chunk.part_index,
                            source_indices=list(chunk.source_indices),
                            source_token_start=chunk.source_token_start,
                            source_token_end=chunk.source_token_end,
                            retained_in_active_view=False) as access_op:
                        product = self._encode_chunk_with_budget(chunk, ratio, access_op)
                    entry = trace.new_entry('encoded_device', access_op,
                        origin_extraction_op_id=access_op['op_id'])
                    access_op['logical_bytes'] = self._encoded_cache_logical_bytes({key:product})
                    access_op['result_entry_id'] = entry['entry_id']
                    extracted_chunks += 1
                    unretained_extracted_chunks += 1
                    materialized_encoder_tokens += len(chunk.token_ids)
                    self._trace_release(entry, 'unretained_always_compress_product')
                    del product
                expected_physical_past = (
                    len(memory.system_input_ids) + planned_gist_tokens
                )
                kv_bytes_per_token = self.kv_bytes_per_token()
                direct_snapshot_reuse = False
                source_snapshot = (
                    scope.session_source_snapshot if session_scope_active else None
                )
                if source_snapshot is not None and scope_generation_index == 1:
                    if source_snapshot.prefix_signature != prefix_signature:
                        self._trace_release(source_snapshot.provenance, "prefix_signature_changed")
                        source_snapshot.clear()
                        scope.session_source_snapshot = None
                        scope.session_eviction_reason = "prefix_signature_changed"
                        raw_cache_reuse_eligibility_reason = "prefix_signature_changed"
                    elif source_snapshot.logical_start != memory.workspace_position_start:
                        self._trace_release(source_snapshot.provenance, "logical_start_changed")
                        source_snapshot.clear()
                        scope.session_source_snapshot = None
                        scope.session_eviction_reason = "logical_start_changed"
                        raw_cache_reuse_eligibility_reason = "logical_start_changed"
                    elif source_snapshot.cache is None:
                        scope.session_source_snapshot = None
                        raw_cache_reuse_eligibility_reason = "source_snapshot_empty"
                    else:
                        if source_snapshot.prefix_tokens != expected_physical_past:
                            raise RuntimeError(
                                "Session snapshot prefix length disagrees with its signature"
                            )
                        lcp = self._token_lcp(
                            source_snapshot.raw_token_ids,
                            memory.workspace_input_ids,
                        )
                        session_reused_raw_tokens = min(
                            lcp,
                            len(memory.workspace_input_ids) - 1,
                        )
                        keep_tokens = (
                            expected_physical_past + session_reused_raw_tokens
                        )
                        parent_snapshot = source_snapshot.provenance
                        total_snapshot_tokens = source_snapshot.cache.get_seq_length()
                        take_kind = (
                            "snapshot_take_whole" if keep_tokens == total_snapshot_tokens
                            else "snapshot_empty_prefix" if keep_tokens == 0
                            else "snapshot_clone_prefix"
                        )
                        with self._cache_op(
                            take_kind, source_entry_id=parent_snapshot["entry_id"],
                            retained_prefix_tokens=expected_physical_past,
                            retained_raw_tokens=session_reused_raw_tokens,
                            dropped_raw_tail_tokens=len(source_snapshot.raw_token_ids) - session_reused_raw_tokens,
                            logical_bytes=keep_tokens * kv_bytes_per_token,
                        ) as snapshot_op:
                            incremental_cache = self._take_snapshot_prefix(source_snapshot, keep_tokens)
                            snapshot_op.update(consumes_source_entry=True,
                                               source_disposition="ownership_handoff" if take_kind == "snapshot_take_whole" else "released_after_prefix_copy")
                        prefix_bindings = parent_snapshot["prefix_bindings"]
                        system_binding = parent_snapshot.get("system_binding")
                        if len(prefix_bindings) != len(placements):
                            raise RuntimeError("Snapshot provenance disagrees with its prefix slots")
                        for placement, binding in zip(placements, prefix_bindings):
                            self._trace_placement(placement, snapshot_op, parent_snapshot, origin=binding)
                        past_key_values = incremental_cache
                        raw_prefill_ids = memory.workspace_input_ids[
                            session_reused_raw_tokens:
                        ]
                        direct_snapshot_reuse = True
                        raw_cache_reuse_eligibility_reason = (
                            "eligible_lcp_reuse"
                            if session_reused_raw_tokens
                            else "eligible_prefix_only"
                        )
                        if session_reused_raw_tokens < len(
                            source_snapshot.raw_token_ids
                        ):
                            scope.session_eviction_reason = "raw_tail_rewrite"
                        scope.session_source_snapshot = None
                        session_reused_gist_chunks = len(unique_encoding_keys)
                        session_reused_encoder_tokens = (
                            self._unique_encoder_tokens(
                                placements,
                                placement_keys,
                            )
                        )
                        session_reused_system_tokens = len(
                            memory.system_input_ids
                        )

                if not direct_snapshot_reuse:
                    if source_snapshot is not None and scope.session_source_snapshot is not None:
                        self._trace_release(source_snapshot.provenance, "unused_source_snapshot")
                        source_snapshot.clear()
                        scope.session_source_snapshot = None
                    for placement, key in zip(placements, placement_keys):
                        chunk_output = encoded_for_call.get(key)
                        reused_in_call = chunk_output is not None
                        access_op = None
                        if chunk_output is None:
                            chunk_output = encoded_cache.get(key)
                            if chunk_output is None:
                                cpu_encoded = (
                                    session.cpu_encoded_by_key.get(key)
                                    if session is not None
                                    else None
                                )
                                if cpu_encoded is not None:
                                    parent_entry = session.encoded_provenance[key]
                                    with self._cache_op("cpu_memo_hydrate", source_entry_id=parent_entry["entry_id"]) as access_op:
                                        chunk_output, transfer_bytes = self._hydrate_encoded_chunk(cpu_encoded)
                                        access_op.update(transfer_bytes=transfer_bytes,
                                                         logical_bytes=self._encoded_cache_logical_bytes({key: chunk_output}))
                                    entry = trace.new_entry("encoded_device", access_op, parent=parent_entry)
                                    encoded_provenance[key] = entry
                                    encoded_cache[key] = chunk_output
                                    session_reused_gist_chunks += 1
                                    session_cpu_memo_gist_hits += 1
                                    session_reused_encoder_tokens += len(
                                        placement.chunk.token_ids
                                    )
                                    session_cache_transfer_bytes_in += transfer_bytes
                                else:
                                    with self._cache_op(
                                        "extract", input_tokens_requested=len(placement.chunk.token_ids),
                                        event_id=placement.chunk.event_id,
                                        part_index=placement.chunk.part_index,
                                        source_indices=list(placement.chunk.source_indices),
                                        source_token_start=placement.chunk.source_token_start,
                                        source_token_end=placement.chunk.source_token_end,
                                        temporary_allocator_peak_bytes=None, device_elapsed_seconds=None,
                                    ) as access_op:
                                        chunk_output = self._encode_chunk_with_budget(placement.chunk, ratio, access_op)
                                    entry = trace.new_entry("encoded_device", access_op,
                                                            origin_extraction_op_id=access_op["op_id"])
                                    access_op["logical_bytes"] = self._encoded_cache_logical_bytes({key: chunk_output})
                                    encoded_provenance[key] = entry
                                    encoded_cache[key] = chunk_output
                                    extracted_chunks += 1
                                    materialized_encoder_tokens += len(
                                        placement.chunk.token_ids
                                    )
                            elif key in scope_keys_before_call:
                                scope_reused_chunks += 1
                                scope_reused_encoder_tokens += len(
                                    placement.chunk.token_ids
                                )
                            encoded_for_call[key] = chunk_output
                        if access_op is None:
                            entry = encoded_provenance[key]
                            kind = "call_memo_reuse" if reused_in_call else "scope_memo_reuse"
                            with self._cache_op(kind, source_entry_id=entry["entry_id"],
                                                result_entry_id=entry["entry_id"]) as access_op:
                                pass
                        access_op["result_entry_id"] = entry["entry_id"]
                        self._trace_placement(placement, access_op, entry)
                        prefix_bindings.append(dict(entry))
                        if key in scope_keys_before_call:
                            scope_reused_chunk_placements += 1
                        self.runtime._validate_placement(placement, chunk_output)
                        encoded.append(chunk_output)

                    if memory.system_input_ids:
                        prefix_key_values = system_cache.get(
                            memory.system_input_ids
                        )
                        if prefix_key_values is None:
                            if (
                                session is not None
                                and session.cpu_system_tokens
                                == memory.system_input_ids
                                and session.cpu_system_key_values
                            ):
                                with self._cache_op("system_cpu_memo_hydrate", source_entry_id=session.system_provenance["entry_id"]) as system_op:
                                    prefix_key_values, transfer_bytes = self._hydrate_system_key_values(session.cpu_system_key_values)
                                    system_op["transfer_bytes"] = transfer_bytes
                                system_binding = trace.new_entry("system_device", system_op, parent=session.system_provenance)
                                system_provenance[memory.system_input_ids] = system_binding
                                system_op["result_entry_id"] = system_binding["entry_id"]
                                system_cache[memory.system_input_ids] = (
                                    prefix_key_values
                                )
                                session_reused_system_tokens = len(
                                    memory.system_input_ids
                                )
                                session_cpu_memo_system_hit = True
                                session_cache_transfer_bytes_in += transfer_bytes
                            else:
                                system_prefill_calls = self._prefill_forward_count(
                                    len(memory.system_input_ids)
                                )
                                with self._cache_op(
                                    "system_prefill",
                                    input_tokens_requested=len(memory.system_input_ids),
                                    planned_model_forward_calls=system_prefill_calls,
                                    model_forward_calls=None,
                                    prefill_chunk_size=self.prefill_chunk_size,
                                ) as system_op:
                                    if self.prefill_chunk_size is None:
                                        prefix_key_values = self.runtime._encode_system(
                                            memory.system_input_ids
                                        )
                                    else:
                                        prefix_key_values = self.runtime._encode_system(
                                            memory.system_input_ids,
                                            prefill_chunk_size=self.prefill_chunk_size,
                                        )
                                    system_op["model_forward_calls"] = (
                                        system_prefill_calls
                                    )
                                system_binding = trace.new_entry("system_device", system_op)
                                system_provenance[memory.system_input_ids] = system_binding
                                system_op["result_entry_id"] = system_binding["entry_id"]
                                system_cache[memory.system_input_ids] = (
                                    prefix_key_values
                                )
                                system_prefill_tokens = len(
                                    memory.system_input_ids
                                )
                        elif scope is not None:
                            system_binding = system_provenance[memory.system_input_ids]
                            with self._cache_op("system_scope_memo_reuse", source_entry_id=system_binding["entry_id"],
                                                result_entry_id=system_binding["entry_id"]) as system_op:
                                pass
                            scope_reused_system_prefill_calls = (
                                self._prefill_forward_count(
                                    len(memory.system_input_ids)
                                )
                            )
                            scope_reused_system_prefill_tokens = len(
                                memory.system_input_ids
                            )

                        system_op["logical_bytes"] = self._system_key_values_logical_bytes(prefix_key_values)

                    with self._cache_op("assemble_prefix", source_entry_ids=[entry["entry_id"] for entry in prefix_bindings],
                                        system_entry_id=system_binding["entry_id"] if system_binding else None,
                                        logical_bytes=expected_physical_past * kv_bytes_per_token):
                        past_key_values, physical_past = self.runtime._assemble_cache(prefix_key_values, placements, encoded)
                else:
                    physical_past = expected_physical_past
                if physical_past != expected_physical_past:
                    raise RuntimeError(
                        "Assembled prefix length disagrees with PackedMemory: "
                        f"assembled={physical_past}, planned={expected_physical_past}"
                    )

                prefix_kv_bytes = physical_past * kv_bytes_per_token
                retained_before_prefill = (
                    past_key_values.get_seq_length()
                    if past_key_values is not None
                    else 0
                )
                retained_before_prefill_bytes = (
                    self._cache_logical_bytes(past_key_values)
                    if past_key_values is not None
                    else 0
                )
                if retained_before_prefill_bytes != (
                    retained_before_prefill * kv_bytes_per_token
                ):
                    raise RuntimeError(
                        "Assembled KV dtype/shape differs from the base-model byte geometry"
                    )

                finish_reason = "length"
                device = self.runtime.base_model.model.embed_tokens.weight.device
                if self.decode_strategy == "incremental":
                    incremental_cache = past_key_values
                    if incremental_cache is None:
                        incremental_cache = DynamicCache(
                            config=self.runtime.base_model.config
                        )
                if shadow_recorder is not None:
                    shadow_recorder.install()
                while len(generated) < max_new_tokens:
                    if self.decode_strategy == "full_recompute":
                        target_forward_calls += 1
                        current_ids = memory.workspace_input_ids + tuple(generated)
                        target_input_tokens += len(current_ids)
                        if generated:
                            raw_recompute_forward_calls += 1
                            raw_recompute_input_tokens += len(current_ids)
                        else:
                            raw_prefill_forward_calls = 1
                            raw_prefill_input_tokens = len(current_ids)
                        input_ids = torch.tensor(
                            current_ids,
                            dtype=torch.long,
                            device=device,
                        ).unsqueeze(0)
                        attention_mask = torch.ones(
                            (1, physical_past + len(current_ids)),
                            dtype=torch.bool,
                            device=device,
                        )
                        position_ids = torch.arange(
                            memory.workspace_position_start,
                            memory.workspace_position_start + len(current_ids),
                            dtype=torch.long,
                            device=device,
                        ).unsqueeze(0)
                        final_index = torch.tensor(
                            [len(current_ids) - 1],
                            dtype=torch.long,
                            device=device,
                        )
                        with self._cache_op(
                            "full_recompute" if generated else "raw_prefill", input_tokens_requested=len(current_ids),
                            decode_strategy="full_recompute",
                            workspace_source_group=list(memory.raw_source_indices),
                            workspace_token_range=[0, len(memory.workspace_input_ids)],
                            generated_prefix_tokens=len(generated),
                        ):
                            if shadow_recorder is not None:
                                shadow_recorder.begin_forward(len(generated))
                            logits = self.runtime._target_logits(
                                input_ids=input_ids, attention_mask=attention_mask,
                                position_ids=position_ids, past_key_values=past_key_values,
                                supervised_indices=final_index,
                            )
                    else:
                        if generated:
                            forward_chunks = (
                                (
                                    (generated[-1],),
                                    memory.workspace_position_start
                                    + len(memory.workspace_input_ids)
                                    + len(generated)
                                    - 1,
                                    None,
                                ),
                            )
                        else:
                            chunk_size = (
                                self.prefill_chunk_size or len(raw_prefill_ids)
                            )
                            forward_chunks = tuple(
                                (
                                    raw_prefill_ids[start:end],
                                    memory.workspace_position_start
                                    + session_reused_raw_tokens
                                    + start,
                                    (
                                        session_reused_raw_tokens + start,
                                        session_reused_raw_tokens + end,
                                    ),
                                )
                                for start in range(0, len(raw_prefill_ids), chunk_size)
                                for end in (
                                    min(start + chunk_size, len(raw_prefill_ids)),
                                )
                            )
                        for chunk_index, (
                            step_ids,
                            position_start,
                            workspace_range,
                        ) in enumerate(forward_chunks):
                            target_forward_calls += 1
                            target_input_tokens += len(step_ids)
                            input_ids = torch.tensor(
                                step_ids,
                                dtype=torch.long,
                                device=device,
                            ).unsqueeze(0)
                            position_ids = torch.arange(
                                position_start,
                                position_start + len(step_ids),
                                dtype=torch.long,
                                device=device,
                            ).unsqueeze(0)
                            cache_before = incremental_cache.get_seq_length()
                            expected_before = (
                                physical_past
                                + (
                                    workspace_range[0]
                                    if workspace_range is not None
                                    else len(memory.workspace_input_ids)
                                    + len(generated)
                                    - 1
                                )
                            )
                            if cache_before != expected_before:
                                raise RuntimeError(
                                    "Incremental cache length disagrees with the call-local decode state: "
                                    f"cached={cache_before}, expected={expected_before}"
                                )
                            causal_mask = self._physical_causal_mask(
                                query_length=len(step_ids),
                                physical_past=cache_before,
                                device=device,
                            )
                            with self._cache_op(
                                "incremental_decode" if generated else "raw_prefill",
                                decode_strategy="incremental",
                                input_tokens_requested=len(step_ids),
                                workspace_source_group=list(memory.raw_source_indices),
                                workspace_token_range=(
                                    None
                                    if workspace_range is None
                                    else list(workspace_range)
                                ),
                                generated_prefix_tokens=1 if generated else 0,
                                generated_token_range=(
                                    [len(generated) - 1, len(generated)]
                                    if generated
                                    else None
                                ),
                                prefill_chunk_index=(
                                    chunk_index if not generated else None
                                ),
                                prefill_chunk_count=(
                                    len(forward_chunks) if not generated else None
                                ),
                                prefill_chunk_size=self.prefill_chunk_size,
                            ):
                                if shadow_recorder is not None:
                                    shadow_recorder.begin_forward(len(generated))
                                outputs = self.runtime.base_model(
                                    input_ids=input_ids,
                                    attention_mask=self._full_attention_mask_mapping(causal_mask),
                                    position_ids=position_ids,
                                    past_key_values=incremental_cache,
                                    use_cache=True,
                                    use_gist=False,
                                    logits_to_keep=1,
                                )
                            incremental_cache = outputs.past_key_values
                            expected_after = cache_before + len(step_ids)
                            if (
                                incremental_cache is None
                                or incremental_cache.get_seq_length()
                                != expected_after
                            ):
                                observed = (
                                    None
                                    if incremental_cache is None
                                    else incremental_cache.get_seq_length()
                                )
                                raise RuntimeError(
                                    "Incremental cache append produced an unexpected length: "
                                    f"cached={observed}, expected={expected_after}"
                                )
                            if generated:
                                one_token_decode_forward_calls += 1
                                one_token_decode_input_tokens += len(step_ids)
                            else:
                                raw_prefill_forward_calls += 1
                                raw_prefill_input_tokens += len(step_ids)
                                resident_kv_tokens_after_raw_prefill = expected_after
                                resident_kv_logical_bytes_after_raw_prefill = (
                                    self._cache_logical_bytes(incremental_cache)
                                )
                            logits = outputs.logits[:, -1, :]
                    if logits.ndim != 2 or logits.shape != (1, vocab_size):
                        raise RuntimeError(
                            "HistoryMemoryModel returned unexpected final-position logits: "
                            f"shape={tuple(logits.shape)}, expected={(1, vocab_size)}"
                        )
                    log_probs = torch.log_softmax(logits[0].float(), dim=-1)
                    next_token = int(torch.argmax(log_probs).item())
                    next_logprob = float(log_probs[next_token].item())
                    if not math.isfinite(next_logprob):
                        raise RuntimeError("Greedy token has a non-finite log probability")
                    if shadow_recorder is not None:
                        shadow_recorder.observe_distribution(
                            log_probs,
                            selected_token_id=next_token,
                        )
                    generated.append(next_token)
                    token_logprobs.append(next_logprob)
                    if next_token in eos_ids:
                        finish_reason = "stop"
                        break

                if shadow_recorder is not None:
                    shadow_recorder.close()

                if self.runtime.parameter_version != parameter_version:
                    raise RuntimeError("Generation changed the runtime parameter version")
                if self.decode_strategy == "incremental":
                    resident_kv_tokens_final = incremental_cache.get_seq_length()
                    expected_final = (
                        physical_past
                        + len(memory.workspace_input_ids)
                        + len(generated)
                        - 1
                    )
                    if resident_kv_tokens_final != expected_final:
                        raise RuntimeError(
                            "Final incremental cache contains an unconsumed output token: "
                            f"cached={resident_kv_tokens_final}, expected={expected_final}"
                        )
                    resident_kv_logical_bytes_final = self._cache_logical_bytes(
                        incremental_cache
                    )
                    if session_scope_active and self.retain_raw_snapshot:
                        session_pending_device_resident_logical_bytes = (
                            resident_kv_logical_bytes_final
                        )
                        session_pending_device_backing_bytes = (
                            self._cache_backing_bytes(incremental_cache)
                        )
                        with self._cache_op("snapshot_publish_pending", logical_bytes=resident_kv_logical_bytes_final,
                                            source_entry_id=parent_snapshot["entry_id"] if parent_snapshot else None) as snapshot_op:
                            snapshot_entry = trace.new_entry(
                                "raw_snapshot", snapshot_op, parent=parent_snapshot,
                                prefix_bindings=prefix_bindings, system_binding=system_binding,
                                raw_source_group=list(memory.raw_source_indices),
                                forwarded_raw_tokens=len(memory.workspace_input_ids) + len(generated) - 1,
                            )
                            snapshot_op["result_entry_id"] = snapshot_entry["entry_id"]
                        scope.pending_snapshot = _RawCacheSnapshot(
                            cache=incremental_cache,
                            prefix_signature=prefix_signature,
                            logical_start=memory.workspace_position_start,
                            prefix_tokens=physical_past,
                            raw_token_ids=(
                                memory.workspace_input_ids
                                + tuple(generated[:-1])
                            ),
                            resident_logical_bytes=(
                                session_pending_device_resident_logical_bytes
                            ),
                            backing_bytes=session_pending_device_backing_bytes,
                            provenance=snapshot_entry,
                        )
                if session_scope_active:
                    scope.pending_encoding_keys = unique_encoding_keys
                    scope.pending_system_tokens = memory.system_input_ids
                    scope.session_transfer_bytes_in += session_cache_transfer_bytes_in

                costs = memory.costs(ratio)
                unique_raw_tokens = len(memory.workspace_input_ids) + len(generated) - 1
                scope_system_bytes_after = (
                    self._system_cache_logical_bytes(system_cache)
                    if scope is not None
                    else 0
                )
                scope_gist_bytes_after = (
                    self._encoded_cache_logical_bytes(encoded_cache)
                    if scope is not None
                    else 0
                )
                stats = {
                    "cache_trace": trace.data,
                    "raw_layout_profile": memory.raw_layout_profile,
                    "ratio": ratio,
                    "requested_max_new_tokens": max_new_tokens,
                    "generated_tokens": len(generated),
                    "workspace_tokens": len(memory.workspace_input_ids),
                    "source_tokens": self.runtime._source_tokens(memory),
                    "packed_encoder_tokens": costs["presented_encoder_tokens"],
                    "materialized_encoder_tokens": materialized_encoder_tokens,
                    "scope_reused_encoder_tokens": scope_reused_encoder_tokens,
                    "gist_tokens": planned_gist_tokens,
                    "system_prefix_kv_tokens": len(memory.system_input_ids),
                    "gist_prefix_kv_tokens": planned_gist_tokens,
                    "system_prefix_kv_logical_bytes": (
                        len(memory.system_input_ids) * kv_bytes_per_token
                    ),
                    "gist_prefix_kv_logical_bytes": (
                        planned_gist_tokens * kv_bytes_per_token
                    ),
                    "resident_prefix_kv_tokens": physical_past,
                    "resident_prefix_kv_bytes": prefix_kv_bytes,
                    "kv_bytes_per_token": kv_bytes_per_token,
                    # unique_chunks is per-generate regardless of whether its
                    # tensor was extracted now or reused from this scope.
                    "unique_chunks": len(unique_encoding_keys),
                    "chunk_placements": len(placements),
                    "max_extraction_calls": self.max_extraction_calls,
                    "extraction_calls_reserved": self.extraction_calls_reserved,
                    "extracted_chunks": extracted_chunks,
                    "unretained_extracted_chunks": unretained_extracted_chunks,
                    "unretained_reused_chunks": unretained_reused_chunks,
                    "scope_reused_chunks": scope_reused_chunks,
                    "reused_chunk_placements": len(placements) - len(unique_encoding_keys),
                    "scope_reused_chunk_placements": scope_reused_chunk_placements,
                    "system_prefill_calls": system_prefill_calls,
                    "system_prefill_tokens": system_prefill_tokens,
                    "scope_reused_system_prefill_calls": scope_reused_system_prefill_calls,
                    "scope_reused_system_prefill_tokens": scope_reused_system_prefill_tokens,
                    "decision_scope_active": scope is not None,
                    "decision_scope_generation_index": scope_generation_index,
                    "scope_system_kv_logical_bytes_before": scope_system_bytes_before,
                    "scope_system_kv_logical_bytes_after": scope_system_bytes_after,
                    "scope_gist_kv_logical_bytes_before": scope_gist_bytes_before,
                    "scope_gist_kv_logical_bytes_after": scope_gist_bytes_after,
                    "decode_strategy": self.decode_strategy,
                    "prefill_chunk_size": self.prefill_chunk_size,
                    "target_execution_strategy": self.decode_strategy,
                    "target_forward_calls": target_forward_calls,
                    "target_input_tokens": target_input_tokens,
                    # Legacy alias retained for consumers of the first schema.
                    "recomputed_raw_tokens": target_input_tokens,
                    "raw_prefill_forward_calls": raw_prefill_forward_calls,
                    "raw_prefill_input_tokens": raw_prefill_input_tokens,
                    "one_token_decode_forward_calls": one_token_decode_forward_calls,
                    "one_token_decode_input_tokens": one_token_decode_input_tokens,
                    "raw_recompute_forward_calls": raw_recompute_forward_calls,
                    "raw_recompute_input_tokens": raw_recompute_input_tokens,
                    "suffix_recompute_tokens": (
                        target_input_tokens - unique_raw_tokens
                        if self.decode_strategy == "full_recompute"
                        else 0
                    ),
                    "resident_kv_tokens_after_raw_prefill": (
                        resident_kv_tokens_after_raw_prefill
                    ),
                    "resident_kv_logical_bytes_after_raw_prefill": (
                        resident_kv_logical_bytes_after_raw_prefill
                    ),
                    "resident_kv_tokens_final": resident_kv_tokens_final,
                    "resident_kv_logical_bytes_final": (
                        resident_kv_logical_bytes_final
                    ),
                    "torch_allocator_peak_allocated_bytes": None,
                    "session_cache_policy": self.session_cache_policy,
                    "session_scope_active": session_scope_active,
                    "session_reused_raw_tokens": session_reused_raw_tokens,
                    "session_reused_gist_chunks": session_reused_gist_chunks,
                    "session_reused_encoder_tokens": session_reused_encoder_tokens,
                    "session_reused_system_tokens": session_reused_system_tokens,
                    "session_cpu_memo_gist_hits": session_cpu_memo_gist_hits,
                    "session_cpu_memo_system_hit": session_cpu_memo_system_hit,
                    "session_cache_cpu_logical_bytes_before": (
                        session_cache_cpu_logical_bytes_before
                    ),
                    "session_pending_device_resident_logical_bytes": (
                        session_pending_device_resident_logical_bytes
                    ),
                    "session_pending_device_backing_bytes": (
                        session_pending_device_backing_bytes
                    ),
                    "session_cache_transfer_bytes_in": (
                        session_cache_transfer_bytes_in
                    ),
                    "raw_cache_reuse_eligibility_reason": (
                        raw_cache_reuse_eligibility_reason
                    ),
                    "session_cache_reset_reason": scope.session_reset_reason if scope else None,
                    "session_cache_eviction_reason": (
                        scope.session_eviction_reason if scope else None
                    ),
                    "session_cache_commit_status": (
                        "pending" if session_scope_active else "not_applicable"
                    ),
                    "raw_recompute_strategy": (
                        "full_workspace_plus_generated_each_step"
                        if self.decode_strategy == "full_recompute"
                        else "none_incremental_raw_kv"
                    ),
                    "parameter_version": parameter_version,
                    "eos_token_ids": sorted(eos_ids),
                    "eos_source": eos_source,
                    "elapsed_sec": time.perf_counter() - started,
                }
                if shadow_recorder is not None:
                    stats["shadow_features"] = shadow_recorder.finalize(generated)
                result = EventNativeGenerationResult(
                    token_ids=tuple(generated),
                    finish_reason=finish_reason,
                    token_logprobs=tuple(token_logprobs),
                    stats=stats,
                )
                if session_scope_active:
                    scope.pending_stats = result.stats
                    scope.pending_trace = trace
                    trace.data["commit_status"] = "pending"
                return result
        except BaseException:
            if session_scope_active:
                if scope.pending_snapshot is not None:
                    self._trace_release(scope.pending_snapshot.provenance, "generation_failure")
                    scope.pending_snapshot.clear()
                    scope.pending_snapshot = None
                if scope.session_source_snapshot is not None:
                    self._trace_release(scope.session_source_snapshot.provenance, "generation_failure")
                    scope.session_source_snapshot.clear()
                    scope.session_source_snapshot = None
                self._discard_session_cache(reason="generation_failure")
            raise
        finally:
            if shadow_recorder is not None:
                shadow_recorder.close()
            encoded.clear()
            encoded_for_call.clear()
            if scope is None:
                for entry in (*encoded_provenance.values(), *system_provenance.values()):
                    self._trace_release(entry, "call_wrapper_release")
                encoded_cache.clear()
                system_cache.clear()
            prefix_key_values = ()
            past_key_values = None
            incremental_cache = None
            outputs = None

    def _acquire_decision_scope(self, ratio: int) -> _DecisionScopeCache | None:
        scope = self._active_decision_scope
        if scope is None:
            return None
        if self.runtime.parameter_version != scope.parameter_version:
            self._invalidate_session_scope(scope)
            raise RuntimeError("runtime parameter_version changed within decision_scope")
        if id(self.runtime.base_model) != scope.model_id:
            self._invalidate_session_scope(scope)
            raise RuntimeError("runtime base_model changed within decision_scope")
        weight = self.runtime.base_model.model.embed_tokens.weight
        if weight.device != scope.compute_device or weight.dtype != scope.compute_dtype:
            self._invalidate_session_scope(scope)
            raise RuntimeError("model device or dtype changed within decision_scope")
        if self._model_parameter_signature() != scope.parameter_signature:
            self._invalidate_session_scope(scope)
            raise RuntimeError("model parameters changed within decision_scope")
        if self._execution_signature() != scope.execution_signature:
            self._invalidate_session_scope(scope)
            raise RuntimeError("model execution configuration changed within decision_scope")
        if scope.ratio is None:
            scope.ratio = ratio
        elif ratio != scope.ratio:
            self._invalidate_session_scope(scope)
            raise RuntimeError("ratio changed within decision_scope")
        if scope.generate_calls >= 2:
            raise RuntimeError("decision_scope permits at most two generate calls")
        scope.generate_calls += 1
        return scope

    def _model_parameter_signature(
        self,
    ) -> tuple[tuple[int, int, torch.device, torch.dtype], ...]:
        return tuple(
            (id(parameter), parameter._version, parameter.device, parameter.dtype)
            for parameter in self.runtime.base_model.parameters()
        )

    def _execution_signature(self) -> tuple[Any, ...]:
        model = self.runtime.base_model
        config = model.config
        rotary = model.model.rotary_emb
        config_names = (
            "_attn_implementation",
            "layer_types",
            "sliding_window",
            "max_position_embeddings",
            "rope_parameters",
            "rope_scaling",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "hidden_size",
        )
        config_signature = tuple(
            (name, self._freeze_signature_value(getattr(config, name, None)))
            for name in config_names
        )
        attention_backends = tuple(
            getattr(layer.self_attn.config, "_attn_implementation", None)
            for layer in model.model.layers
        )
        rotary_buffers = tuple(
            (
                name,
                id(tensor),
                tensor._version,
                tensor.device,
                tensor.dtype,
                tuple(tensor.shape),
            )
            for name, tensor in rotary.named_buffers(recurse=False)
        )
        return (
            self.decode_strategy,
            self.prefill_chunk_size,
            self.runtime.parameter_version,
            id(model),
            id(config),
            self._model_parameter_signature(),
            config_signature,
            attention_backends,
            id(rotary),
            getattr(rotary, "rope_type", None),
            id(getattr(rotary, "config", None)),
            rotary_buffers,
        )

    @classmethod
    def _freeze_signature_value(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return tuple(
                (key, cls._freeze_signature_value(item))
                for key, item in sorted(value.items())
            )
        if isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            return tuple(cls._freeze_signature_value(item) for item in value)
        if isinstance(value, (str, bytes, int, float, bool, type(None))):
            return value
        return repr(value)

    def _prefill_forward_count(self, token_count: int) -> int:
        """Return the actual model forwards for one opt-in prefix prefill."""

        if token_count <= 0:
            return 0
        if self.prefill_chunk_size is None:
            return 1
        return math.ceil(token_count / self.prefill_chunk_size)

    def _prefix_signature(
        self,
        memory: PackedMemory,
        placements: Sequence[Any],
        placement_keys: Sequence[tuple[Any, ...]],
        ratio: int,
        execution_signature: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        return (
            "event-native-prefix-signature-v1",
            memory.raw_layout_profile,
            memory.system_input_ids,
            tuple(
                (key, tuple(placement.position_ids))
                for key, placement in zip(placement_keys, placements)
            ),
            ratio,
            execution_signature,
        )

    @staticmethod
    def _token_lcp(left: Sequence[int], right: Sequence[int]) -> int:
        length = 0
        for left_token, right_token in zip(left, right):
            if left_token != right_token:
                break
            length += 1
        return length

    @staticmethod
    def _unique_encoder_tokens(
        placements: Sequence[Any],
        placement_keys: Sequence[tuple[Any, ...]],
    ) -> int:
        seen = set()
        total = 0
        for placement, key in zip(placements, placement_keys):
            if key in seen:
                continue
            seen.add(key)
            total += len(placement.chunk.token_ids)
        return total

    def _take_snapshot_prefix(
        self,
        snapshot: _RawCacheSnapshot,
        keep_tokens: int,
    ) -> DynamicCache:
        cache = snapshot.cache
        if cache is None:
            raise RuntimeError("cannot reuse a released session snapshot")
        total_tokens = cache.get_seq_length()
        if not 0 <= keep_tokens <= total_tokens:
            raise RuntimeError(
                "requested session cache prefix exceeds the retained snapshot"
            )
        if keep_tokens == total_tokens:
            snapshot.cache = None
            return cache
        if keep_tokens == 0:
            result = DynamicCache(config=self.runtime.base_model.config)
        else:
            layers = []
            for layer in cache.layers:
                if layer.keys is None or layer.values is None:
                    raise RuntimeError("session cache has an uninitialized layer")
                layers.append(
                    (
                        layer.keys[..., :keep_tokens, :].clone(),
                        layer.values[..., :keep_tokens, :].clone(),
                    )
                )
            result = DynamicCache(layers, config=self.runtime.base_model.config)
        snapshot.clear()
        if result.get_seq_length() != keep_tokens:
            raise RuntimeError("cloned session prefix has an unexpected length")
        return result

    def _hydrate_encoded_chunk(self, encoded: Any) -> tuple[Any, int]:
        device = self.runtime.base_model.model.embed_tokens.weight.device
        transfer_bytes = 0
        key_values = []
        for layer in encoded.key_values:
            moved_layer = []
            for tensor in layer:
                moved = tensor.to(device=device)
                if tensor.device != moved.device:
                    transfer_bytes += moved.numel() * moved.element_size()
                moved_layer.append(moved)
            key_values.append(tuple(moved_layer))
        return type(encoded)(tuple(key_values), encoded.local_position_ids), transfer_bytes

    def _hydrate_system_key_values(self, key_values: Any) -> tuple[Any, int]:
        device = self.runtime.base_model.model.embed_tokens.weight.device
        transfer_bytes = 0
        moved_layers = []
        for layer in key_values:
            moved_layer = []
            for tensor in layer:
                moved = tensor.to(device=device)
                if tensor.device != moved.device:
                    transfer_bytes += moved.numel() * moved.element_size()
                moved_layer.append(moved)
            moved_layers.append(tuple(moved_layer))
        return tuple(moved_layers), transfer_bytes

    @staticmethod
    def _copy_tensor_to_cpu(tensor: torch.Tensor) -> tuple[torch.Tensor, int]:
        detached = tensor.detach()
        transfer_bytes = (
            detached.numel() * detached.element_size()
            if detached.device.type != "cpu"
            else 0
        )
        return detached.to(device="cpu").clone(), transfer_bytes

    def _copy_encoded_to_cpu(self, encoded: Any) -> tuple[Any, int]:
        transfer_bytes = 0
        layers = []
        for layer in encoded.key_values:
            cpu_layer = []
            for tensor in layer:
                copied, moved = self._copy_tensor_to_cpu(tensor)
                transfer_bytes += moved
                cpu_layer.append(copied)
            layers.append(tuple(cpu_layer))
        return type(encoded)(tuple(layers), encoded.local_position_ids), transfer_bytes

    def _copy_system_to_cpu(self, key_values: Any) -> tuple[Any, int]:
        transfer_bytes = 0
        layers = []
        for layer in key_values:
            cpu_layer = []
            for tensor in layer:
                copied, moved = self._copy_tensor_to_cpu(tensor)
                transfer_bytes += moved
                cpu_layer.append(copied)
            layers.append(tuple(cpu_layer))
        return tuple(layers), transfer_bytes

    def _commit_session_scope(self, scope: _DecisionScopeCache) -> None:
        session = self._session_cache
        if (
            session is None
            or scope.session_id is None
            or session.session_id != scope.session_id
        ):
            raise RuntimeError("session disappeared before decision commit")
        if self._execution_signature() != scope.execution_signature:
            raise RuntimeError("model execution configuration changed before commit")
        if scope.pending_stats is None:
            if scope.session_source_snapshot is not None:
                session.device_snapshot = scope.session_source_snapshot
                scope.session_source_snapshot = None
            return

        cpu_encoded: dict[tuple[Any, ...], Any] = {}
        cpu_provenance = {}
        trace = scope.pending_trace
        transfer_bytes_out = 0
        for key in scope.pending_encoding_keys:
            existing_cpu = session.cpu_encoded_by_key.get(key)
            if existing_cpu is not None:
                entry = session.encoded_provenance[key]
                with self._cache_op("cpu_memo_retain", trace=trace, source_entry_id=entry["entry_id"],
                                    result_entry_id=entry["entry_id"]):
                    pass
                cpu_encoded[key] = existing_cpu
                cpu_provenance[key] = entry
                continue
            encoded = scope.encoded_by_key.get(key)
            if encoded is None:
                raise RuntimeError("final session view lost a selected gist encoding")
            with self._cache_op("cpu_memo_copy", trace=trace,
                                source_entry_id=scope.encoded_provenance[key]["entry_id"]) as op:
                copied, moved = self._copy_encoded_to_cpu(encoded)
                op.update(transfer_bytes=moved, logical_bytes=self._encoded_cache_logical_bytes({key: copied}))
            entry = trace.new_entry("encoded_cpu", op, parent=scope.encoded_provenance[key])
            op["result_entry_id"] = entry["entry_id"]
            cpu_provenance[key] = entry
            cpu_encoded[key] = copied
            transfer_bytes_out += moved

        system_key_values: Any = ()
        cpu_system_provenance = None
        if scope.pending_system_tokens:
            if (
                session.cpu_system_tokens == scope.pending_system_tokens
                and session.cpu_system_key_values
            ):
                system_key_values = session.cpu_system_key_values
                cpu_system_provenance = session.system_provenance
                with self._cache_op("system_cpu_memo_retain", trace=trace,
                                    source_entry_id=cpu_system_provenance["entry_id"],
                                    result_entry_id=cpu_system_provenance["entry_id"]):
                    pass
            else:
                device_system = scope.system_by_tokens.get(
                    scope.pending_system_tokens
                )
                if device_system is None:
                    raise RuntimeError("final session view lost its system encoding")
                with self._cache_op("system_cpu_memo_copy", trace=trace,
                                    source_entry_id=scope.system_provenance[scope.pending_system_tokens]["entry_id"]) as op:
                    system_key_values, moved = self._copy_system_to_cpu(device_system)
                    op.update(transfer_bytes=moved, logical_bytes=self._system_key_values_logical_bytes(system_key_values))
                cpu_system_provenance = trace.new_entry("system_cpu", op, parent=scope.system_provenance[scope.pending_system_tokens])
                op["result_entry_id"] = cpu_system_provenance["entry_id"]
                transfer_bytes_out += moved

        if scope.session_source_snapshot is not None:
            self._trace_release(scope.session_source_snapshot.provenance, "final_view_replacement", trace=trace)
            scope.session_source_snapshot.clear()
            scope.session_source_snapshot = None
        for key, old_entry in session.encoded_provenance.items():
            if key not in cpu_provenance:
                self._trace_release(old_entry, "final_view_replacement", trace=trace)
        if session.system_provenance != cpu_system_provenance:
            self._trace_release(session.system_provenance, "final_view_replacement", trace=trace)
        session.cpu_encoded_by_key = cpu_encoded
        session.encoded_provenance = cpu_provenance
        session.system_provenance = cpu_system_provenance
        session.cpu_system_tokens = scope.pending_system_tokens
        session.cpu_system_key_values = system_key_values
        session.device_snapshot = scope.pending_snapshot
        scope.pending_snapshot = None
        session.generation += 1
        session.ratio = scope.ratio
        session.execution_signature = scope.execution_signature
        session.last_transfer_bytes_in = scope.session_transfer_bytes_in
        session.last_transfer_bytes_out = transfer_bytes_out
        if scope.pending_stats is not None:
            trace.data["commit_status"] = "committed"
            with self._cache_op("session_commit", trace=trace,
                                retained_entry_ids=[entry["entry_id"] for entry in cpu_provenance.values()],
                                retained_system_entry_id=cpu_system_provenance["entry_id"] if cpu_system_provenance else None,
                                retained_snapshot_entry_id=session.device_snapshot.provenance["entry_id"] if session.device_snapshot else None):
                pass
            scope.pending_stats["session_cache_commit_status"] = "committed"
            scope.pending_stats["session_cache_transfer_bytes_out"] = (
                transfer_bytes_out
            )
            scope.pending_stats["session_cache_transfer_bytes"] = (
                scope.session_transfer_bytes_in + transfer_bytes_out
            )
        scope.pending_stats = None
        scope.pending_trace = None
        scope.pending_encoding_keys = ()
        scope.pending_system_tokens = ()

    def _invalidate_session_scope(self, scope: _DecisionScopeCache) -> None:
        if scope.session_id is None:
            return
        standalone = self._cache_trace is None
        if standalone:
            self._cache_trace = self._new_lifecycle_trace(scope.session_id, "scope_failure_before_generation")
        scope.session_invalidated = True
        if scope.session_source_snapshot is not None:
            self._trace_release(scope.session_source_snapshot.provenance, "session_failure")
            scope.session_source_snapshot.clear()
            scope.session_source_snapshot = None
        if scope.pending_snapshot is not None:
            self._trace_release(scope.pending_snapshot.provenance, "session_failure")
            scope.pending_snapshot.clear()
            scope.pending_snapshot = None
        if scope.pending_stats is not None:
            scope.pending_stats["session_cache_commit_status"] = "cleared_on_failure"
        if scope.pending_trace is not None:
            scope.pending_trace.data["commit_status"] = "cleared_on_failure"
        self._discard_session_cache(reason="session_failure")
        if standalone:
            self._cache_trace.data["status"] = "completed"
            self.last_cache_lifecycle_trace = self._cache_trace.data
            self._cache_trace = None

    def _new_lifecycle_trace(self, session_id, reason):
        trace = CacheTrace({"session_id": session_id, "phase": "cache_lifecycle"})
        trace.data["schema"] = "event-native-cache-lifecycle-v1"
        trace.data["lifecycle_id"] = trace.data.pop("attempt_uid")
        trace.data["associated_attempt_uid"] = (self.last_generation_trace or {}).get("attempt_uid")
        trace.data["reason"] = reason
        return trace

    def _discard_session_cache(self, *, reason="session_reset") -> None:
        if self._session_cache is not None:
            trace = self._cache_trace
            standalone = trace is None
            if standalone:
                trace = self._new_lifecycle_trace(self._session_cache.session_id, reason)
            for entry in self._session_cache.encoded_provenance.values():
                self._trace_release(entry, reason, trace=trace)
            self._trace_release(self._session_cache.system_provenance, reason, trace=trace)
            if self._session_cache.device_snapshot is not None:
                self._trace_release(self._session_cache.device_snapshot.provenance, reason, trace=trace)
            self._session_cache.clear()
            if standalone:
                trace.data["status"] = "completed"
                self.last_cache_lifecycle_trace = trace.data
        self._session_cache = None

    def _session_cpu_memo_logical_bytes(
        self,
        session: _SessionCache | None,
    ) -> int:
        if session is None:
            return 0
        return self._encoded_cache_logical_bytes(session.cpu_encoded_by_key) + (
            self._system_key_values_logical_bytes(session.cpu_system_key_values)
        )

    @staticmethod
    def _validate_scalar_inputs(
        memory: PackedMemory,
        ratio: int,
        max_new_tokens: int,
    ) -> None:
        if not isinstance(memory, PackedMemory):
            raise TypeError("memory must be a PackedMemory")
        if memory.raw_layout_profile != RAW_LAYOUT_PROFILE:
            raise ValueError(
                f"memory must use raw_layout_profile={RAW_LAYOUT_PROFILE!r}"
            )
        if not memory.workspace_input_ids:
            raise ValueError("memory.workspace_input_ids must not be empty")
        if isinstance(ratio, bool) or not isinstance(ratio, int) or ratio <= 0:
            raise ValueError("ratio must be a positive integer")
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens <= 0
        ):
            raise ValueError("max_new_tokens must be a positive integer")

    def _validate_memory_token_ids(
        self,
        memory: PackedMemory,
        vocab_size: int,
    ) -> None:
        if vocab_size <= 0:
            raise ValueError("model vocabulary size must be positive")
        sequences = [
            ("system_input_ids", memory.system_input_ids),
            ("workspace_input_ids", memory.workspace_input_ids),
        ]
        sequences.extend(
            (f"chunks[{index}].token_ids", chunk.token_ids)
            for index, chunk in enumerate(memory.chunks)
        )
        for name, token_ids in sequences:
            if any(
                isinstance(token, bool)
                or not isinstance(token, int)
                or token < 0
                or token >= vocab_size
                for token in token_ids
            ):
                raise ValueError(
                    f"{name} contains a non-integer or out-of-vocabulary token"
                )

    def _validate_model_contract(self, ratio: int) -> None:
        config = self.runtime.base_model.config
        expected = {
            "gist_type": GIST_TYPE,
            "gist_param": GIST_PARAM,
            "gist_residual_type": GIST_RESIDUAL_TYPE,
            "history_memory_training_profile": TRAINING_PROFILE,
            "history_memory_normal_query": NORMAL_QUERY_PROFILE,
            "history_memory_raw_layout": RAW_LAYOUT_PROFILE,
            "history_memory_packing_version": PACKING_VERSION,
            "history_memory_evidence_version": EVIDENCE_VERSION,
        }
        for name, value in expected.items():
            observed = getattr(config, name, None)
            if observed != value:
                raise ValueError(
                    f"model config {name}={observed!r}; expected {value!r}"
                )
        supported = getattr(config, "history_memory_supported_ratios", None)
        if (
            not isinstance(supported, Sequence)
            or isinstance(supported, (str, bytes, bytearray))
            or not supported
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in supported
            )
        ):
            raise ValueError(
                "model config must expose positive integer "
                "history_memory_supported_ratios"
            )
        if ratio not in supported:
            raise ValueError(
                f"ratio {ratio} is unsupported; checkpoint supports "
                f"{list(supported)!r}"
            )

    def _validate_incremental_model_contract(self) -> None:
        config = self.runtime.base_model.config
        rope_sources = (
            getattr(self.runtime.base_model.model.rotary_emb, "rope_type", None),
            getattr(config, "rope_parameters", None),
            getattr(config, "rope_scaling", None),
        )
        rope_types = []
        for source in rope_sources:
            if isinstance(source, dict):
                rope_types.append(source.get("rope_type", source.get("type")))
            elif source is not None:
                rope_types.append(source)
        if not rope_types or any(rope_type != "default" for rope_type in rope_types):
            raise ValueError(
                "incremental decode supports only default RoPE; use "
                "decode_strategy='full_recompute' for scaled or dynamic RoPE"
            )
        layer_types = getattr(config, "layer_types", None)
        if (
            not isinstance(layer_types, Sequence)
            or isinstance(layer_types, (str, bytes, bytearray))
            or not layer_types
            or any(layer_type != "full_attention" for layer_type in layer_types)
        ):
            raise ValueError(
                "incremental decode supports only full_attention layers; use "
                "decode_strategy='full_recompute' for sliding attention"
            )

    def _physical_causal_mask(
        self,
        *,
        query_length: int,
        physical_past: int,
        device: torch.device,
    ) -> torch.Tensor:
        dtype = self.runtime.base_model.model.embed_tokens.weight.dtype
        key_indices = torch.arange(physical_past + query_length, device=device)
        query_indices = torch.arange(query_length, device=device)
        allowed = key_indices.unsqueeze(0) <= (
            physical_past + query_indices.unsqueeze(1)
        )
        mask = torch.zeros(
            (1, 1, query_length, physical_past + query_length),
            dtype=dtype,
            device=device,
        )
        return mask.masked_fill(
            ~allowed.unsqueeze(0).unsqueeze(0),
            torch.finfo(dtype).min,
        )

    def _full_attention_mask_mapping(
        self,
        causal_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            layer_type: causal_mask
            for layer_type in set(self.runtime.base_model.config.layer_types)
        }

    @staticmethod
    def _cache_logical_bytes(cache: Any) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for layer in cache.layers
            for tensor in (layer.keys, layer.values)
            if isinstance(tensor, torch.Tensor)
        )

    @staticmethod
    def _cache_backing_bytes(cache: Any) -> int:
        storages = {}
        for layer in cache.layers:
            for tensor in (layer.keys, layer.values):
                if not isinstance(tensor, torch.Tensor):
                    continue
                storage = tensor.untyped_storage()
                key = (
                    tensor.device.type,
                    tensor.device.index,
                    storage.data_ptr(),
                    storage.nbytes(),
                )
                storages[key] = storage.nbytes()
        return sum(storages.values())

    @staticmethod
    def _system_key_values_logical_bytes(key_values: Any) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for layer in key_values
            for tensor in layer
            if isinstance(tensor, torch.Tensor)
        )

    @classmethod
    def _system_cache_logical_bytes(cls, cache: dict[tuple[int, ...], Any]) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for key_values in cache.values()
            for layer in key_values
            for tensor in layer
            if isinstance(tensor, torch.Tensor)
        )

    @classmethod
    def _encoded_cache_logical_bytes(cls, cache: dict[tuple[Any, ...], Any]) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for encoded in cache.values()
            for layer in encoded.key_values
            for tensor in layer
            if isinstance(tensor, torch.Tensor)
        )

    def _resolve_eos_ids(
        self,
        eos_token_id: int | Sequence[int] | None,
        vocab_size: int,
    ) -> tuple[frozenset[int], str]:
        source = "explicit"
        raw_ids: Any = eos_token_id
        if raw_ids is None:
            source = "generation_config"
            generation_config = getattr(
                self.runtime.base_model,
                "generation_config",
                None,
            )
            raw_ids = getattr(generation_config, "eos_token_id", None)
            if raw_ids is None:
                source = "model_config"
                raw_ids = getattr(
                    self.runtime.base_model.config,
                    "eos_token_id",
                    None,
                )
        if raw_ids is None:
            return frozenset(), "none"
        if isinstance(raw_ids, bool):
            raise TypeError("eos_token_id must be an integer or a sequence of integers")
        if isinstance(raw_ids, int):
            ids = (raw_ids,)
        elif isinstance(raw_ids, Sequence) and not isinstance(
            raw_ids,
            (str, bytes, bytearray),
        ):
            ids = tuple(raw_ids)
            if not ids:
                raise ValueError("eos_token_id sequence must not be empty")
        else:
            raise TypeError("eos_token_id must be an integer or a sequence of integers")
        if any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or token < 0
            or token >= vocab_size
            for token in ids
        ):
            raise ValueError("eos_token_id contains an out-of-vocabulary token")
        return frozenset(ids), source

    def _model_context_length(self) -> int:
        value = getattr(
            self.runtime.base_model.config,
            "max_position_embeddings",
            None,
        )
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                "model config must expose a positive max_position_embeddings"
            )
        return value

    @contextmanager
    def _base_autocast(self) -> Iterator[None]:
        """Match training's compute dtype while preserving FP32 gist weights."""
        weight = self.runtime.base_model.model.embed_tokens.weight
        if weight.dtype in (torch.float16, torch.bfloat16):
            with torch.autocast(device_type=weight.device.type, dtype=weight.dtype):
                yield
        else:
            yield

    def kv_bytes_per_token(self) -> int:
        config = self.runtime.base_model.config
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        return (
            2 * config.num_hidden_layers * config.num_key_value_heads * head_dim
            * self.runtime.base_model.model.embed_tokens.weight.element_size()
        )

    @contextmanager
    def _temporary_eval(self) -> Iterator[None]:
        states = tuple(
            (module, module.training) for module in self.runtime.modules()
        )
        self.runtime.eval()
        try:
            yield
        finally:
            for module, training in states:
                module.training = training


__all__ = ["EventNativeGenerationResult", "EventNativeGenerator"]
