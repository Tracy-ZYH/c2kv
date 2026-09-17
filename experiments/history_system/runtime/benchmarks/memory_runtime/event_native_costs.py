"""Strict task-level cost summaries for event-native step records."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .attempt_journal import read_attempt_journal, summarize_attempt_journal


SCHEMA = "a-event-native-task-cost-summary-v1"
_TRACE_PHASES = ("draft", "regeneration")
_SUMMARY_PHASES = (*_TRACE_PHASES, "unrecorded")
_USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
_WORK_FIELDS = (
    "materialized_encoder_tokens",
    "system_prefill_tokens",
    "target_input_tokens",
    "raw_prefill_input_tokens",
)
_REUSE_FIELDS = (
    "session_reused_raw_tokens",
    "session_reused_gist_chunks",
    "session_reused_encoder_tokens",
    "session_reused_system_tokens",
)
_PEAK_FIELDS = (
    "system_prefix_kv_logical_bytes",
    "gist_prefix_kv_logical_bytes",
    "resident_prefix_kv_logical_bytes",
    "resident_raw_kv_logical_bytes",
    "resident_total_kv_logical_bytes",
    "scope_system_kv_logical_bytes",
    "scope_gist_kv_logical_bytes",
    "scope_total_kv_logical_bytes",
    "torch_allocator_peak_allocated_bytes",
)
_SESSION_CACHE_PEAK_FIELDS = (
    "cpu_memo_logical_bytes",
    "device_raw_snapshot_logical_bytes",
    "device_raw_snapshot_backing_bytes",
)
_SESSION_TRANSFER_FIELDS = (
    "last_transfer_bytes_in",
    "last_transfer_bytes_out",
    "last_transfer_bytes_total",
)
_CACHE_TRACE_SCHEMA = "event-native-cache-trace-v1"
_CACHE_LIFECYCLE_TRACE_SCHEMA = "event-native-cache-lifecycle-v1"
_CACHE_OPERATION_STATUSES = ("started", "completed", "failed")
_MISSING = object()


def _canonical(value: Mapping[str, Any], *, label: str) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite JSON data") from error


def _nonnegative_int(value: Any, *, label: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer or null")
    return value


def _nonnegative_number(value: Any, *, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a nonnegative finite number or null")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{label} must be a nonnegative finite number or null")
    return parsed


def _stats(trace: Mapping[str, Any], *, label: str) -> Mapping[str, Any] | None:
    generation = trace.get("generation")
    if generation is None:
        return None
    if not isinstance(generation, Mapping):
        raise ValueError(f"{label}.generation must be an object or null")
    stats = generation.get("stats")
    if stats is None:
        return None
    if not isinstance(stats, Mapping):
        raise ValueError(f"{label}.generation.stats must be an object or null")
    return stats


def _usage(trace: Mapping[str, Any], *, label: str) -> dict[str, int | None]:
    value = trace.get("usage")
    if value is None:
        return {name: None for name in _USAGE_FIELDS}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}.usage must be an object or null")
    result = {
        name: _nonnegative_int(value.get(name), label=f"{label}.usage.{name}")
        for name in _USAGE_FIELDS
    }
    if all(result[name] is not None for name in _USAGE_FIELDS):
        if result["total_tokens"] != result["prompt_tokens"] + result["completion_tokens"]:
            raise ValueError(f"{label}.usage total does not equal prompt plus completion")
    planned = _nonnegative_int(
        trace.get("planned_resident_prompt_tokens"),
        label=f"{label}.planned_resident_prompt_tokens",
    )
    if planned is not None and result["prompt_tokens"] is not None:
        if planned != result["prompt_tokens"]:
            raise ValueError(f"{label} planned and recorded prompt tokens differ")
    return result


def _stat_int(
    stats: Mapping[str, Any] | None,
    name: str,
    *,
    label: str,
) -> int | None:
    if stats is None or name not in stats:
        return None
    return _nonnegative_int(stats[name], label=f"{label}.generation.stats.{name}")


def _target_input_tokens(
    stats: Mapping[str, Any] | None,
    *,
    label: str,
) -> int | None:
    current = _stat_int(stats, "target_input_tokens", label=label)
    legacy = _stat_int(stats, "recomputed_raw_tokens", label=label)
    if current is not None and legacy is not None and current != legacy:
        raise ValueError(f"{label} target_input_tokens differs from its legacy alias")
    return current if current is not None else legacy


def _scope_peak(
    stats: Mapping[str, Any] | None,
    prefix: str,
    *,
    label: str,
) -> int | None:
    before = _stat_int(stats, f"scope_{prefix}_kv_logical_bytes_before", label=label)
    after = _stat_int(stats, f"scope_{prefix}_kv_logical_bytes_after", label=label)
    if before is None or after is None:
        return None
    return max(before, after)


def _attempt_values(trace: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    usage = _usage(trace, label=label)
    stats = _stats(trace, label=label)
    work = {
        "materialized_encoder_tokens": _stat_int(
            stats, "materialized_encoder_tokens", label=label
        ),
        "system_prefill_tokens": _stat_int(stats, "system_prefill_tokens", label=label),
        "target_input_tokens": _target_input_tokens(stats, label=label),
        "raw_prefill_input_tokens": _stat_int(
            stats, "raw_prefill_input_tokens", label=label
        ),
    }
    reuse = {
        name: _stat_int(stats, name, label=label)
        for name in _REUSE_FIELDS
    }

    system_prefix = _stat_int(stats, "system_prefix_kv_logical_bytes", label=label)
    gist_prefix = _stat_int(stats, "gist_prefix_kv_logical_bytes", label=label)
    prefix = _stat_int(stats, "resident_prefix_kv_bytes", label=label)
    if system_prefix is not None and gist_prefix is not None and prefix is not None:
        if system_prefix + gist_prefix != prefix:
            raise ValueError(f"{label} system and gist prefix bytes do not equal prefix bytes")

    after_prefill = _stat_int(
        stats, "resident_kv_logical_bytes_after_raw_prefill", label=label
    )
    final_cache = _stat_int(stats, "resident_kv_logical_bytes_final", label=label)
    if after_prefill is not None and final_cache is not None and final_cache < after_prefill:
        raise ValueError(f"{label} final cache is smaller than its raw-prefill cache")
    total_cache = final_cache
    raw_cache = None
    if total_cache is not None and prefix is not None:
        if total_cache < prefix:
            raise ValueError(f"{label} final cache is smaller than its prefix")
        raw_cache = total_cache - prefix

    scope_system = _scope_peak(stats, "system", label=label)
    scope_gist = _scope_peak(stats, "gist", label=label)
    scope_total = None
    if stats is not None:
        system_before = _stat_int(
            stats, "scope_system_kv_logical_bytes_before", label=label
        )
        system_after = _stat_int(
            stats, "scope_system_kv_logical_bytes_after", label=label
        )
        gist_before = _stat_int(stats, "scope_gist_kv_logical_bytes_before", label=label)
        gist_after = _stat_int(stats, "scope_gist_kv_logical_bytes_after", label=label)
        if None not in (system_before, system_after, gist_before, gist_after):
            scope_total = max(
                system_before + gist_before,
                system_after + gist_after,
            )

    peaks = {
        "system_prefix_kv_logical_bytes": system_prefix,
        "gist_prefix_kv_logical_bytes": gist_prefix,
        "resident_prefix_kv_logical_bytes": prefix,
        "resident_raw_kv_logical_bytes": raw_cache,
        "resident_total_kv_logical_bytes": total_cache,
        "scope_system_kv_logical_bytes": scope_system,
        "scope_gist_kv_logical_bytes": scope_gist,
        "scope_total_kv_logical_bytes": scope_total,
        "torch_allocator_peak_allocated_bytes": _stat_int(
            stats, "torch_allocator_peak_allocated_bytes", label=label
        ),
    }
    return {"usage": usage, "work": work, "reuse": reuse, "peaks": peaks}


def _sum_metric(values: list[int | float | None]) -> dict[str, Any]:
    if not values:
        return {
            "strict_total": 0,
            "known_total": 0,
            "known_calls": 0,
            "unknown_calls": 0,
        }
    known = [value for value in values if value is not None]
    unknown = len(values) - len(known)
    known_total = sum(known) if known else None
    return {
        "strict_total": known_total if unknown == 0 else None,
        "known_total": known_total,
        "known_calls": len(known),
        "unknown_calls": unknown,
    }


def _peak_metric(
    values: list[int | None],
    *,
    aggregation: str = "max_across_generation_attempts",
) -> dict[str, Any]:
    known = [value for value in values if value is not None]
    unknown = len(values) - len(known)
    known_peak = max(known) if known else None
    return {
        "strict_peak": known_peak if unknown == 0 else None,
        "known_peak": known_peak,
        "known_calls": len(known),
        "unknown_calls": unknown,
        "aggregation": aggregation,
    }


def _generation_costs(
    attempts: list[dict[str, Any]],
    *,
    strict_totals_scope: str = "input_steps_only",
    framed_input_complete: bool = True,
) -> dict[str, Any]:
    result = {
        "strict_totals_scope": strict_totals_scope,
        "framed_input_complete": framed_input_complete,
        "openai_resident_usage": {
            name: _sum_metric([attempt["values"]["usage"][name] for attempt in attempts])
            for name in _USAGE_FIELDS
        },
        "actual_model_work": {
            name: _sum_metric([attempt["values"]["work"][name] for attempt in attempts])
            for name in _WORK_FIELDS
        },
        "session_reuse": {
            name: _sum_metric([attempt["values"]["reuse"][name] for attempt in attempts])
            for name in _REUSE_FIELDS
        },
        "logical_kv_peaks": {
            name: _peak_metric([attempt["values"]["peaks"][name] for attempt in attempts])
            for name in _PEAK_FIELDS
        },
    }
    if not framed_input_complete:
        for name in _USAGE_FIELDS:
            result["openai_resident_usage"][name]["strict_total"] = None
        for group, names in (
            ("actual_model_work", _WORK_FIELDS),
            ("session_reuse", _REUSE_FIELDS),
        ):
            for name in names:
                result[group][name]["strict_total"] = None
        for name in _PEAK_FIELDS:
            result["logical_kv_peaks"][name]["strict_peak"] = None
    return result


def _cache_identifier(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _cache_optional_identifier(value: Any, *, label: str) -> str | None:
    if value is None:
        return None
    return _cache_identifier(value, label=label)


def _cache_id_list(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    result = [_cache_identifier(item, label=f"{label}[{index}]") for index, item in enumerate(value)]
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must not repeat identifiers")
    return result


def _cache_int_list(value: Any, *, label: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return [
        _nonnegative_int(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    ]


def _cache_field(
    value: Mapping[str, Any],
    name: str,
    *,
    label: str,
) -> Any:
    if name not in value:
        raise ValueError(f"{label}.{name} is required")
    return value[name]


def _cache_json_value(value: Any, *, label: str) -> Any:
    try:
        json.dumps(value, ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite JSON data") from error
    return value


def _parse_cache_trace(
    raw_trace: Any,
    *,
    attempt: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    if not isinstance(raw_trace, Mapping):
        raise ValueError(f"{label}.cache_trace must be an object or null")
    trace = dict(raw_trace)
    if trace.get("schema") != _CACHE_TRACE_SCHEMA:
        raise ValueError(f"{label}.cache_trace has an unsupported schema")

    for name in ("attempt_uid", "session_id", "decision_key", "phase", "status"):
        value = _cache_field(trace, name, label=f"{label}.cache_trace")
        if value != attempt[name]:
            raise ValueError(f"{label}.cache_trace.{name} disagrees with the attempt")
    if type(_cache_field(trace, "context_complete", label=f"{label}.cache_trace")) is not bool:
        raise ValueError(f"{label}.cache_trace.context_complete must be a bool")

    workspace_source_group = _cache_json_value(
        _cache_field(trace, "workspace_source_group", label=f"{label}.cache_trace"),
        label=f"{label}.cache_trace.workspace_source_group",
    )
    commit_status = _cache_json_value(
        _cache_field(trace, "commit_status", label=f"{label}.cache_trace"),
        label=f"{label}.cache_trace.commit_status",
    )
    raw_ops = _cache_field(trace, "ops", label=f"{label}.cache_trace")
    raw_entries = _cache_field(trace, "entries", label=f"{label}.cache_trace")
    raw_placements = _cache_field(trace, "placements", label=f"{label}.cache_trace")
    if not isinstance(raw_ops, list):
        raise ValueError(f"{label}.cache_trace.ops must be a list")
    if not isinstance(raw_entries, list):
        raise ValueError(f"{label}.cache_trace.entries must be a list")
    if not isinstance(raw_placements, list):
        raise ValueError(f"{label}.cache_trace.placements must be a list")

    ops = []
    op_by_id: dict[str, dict[str, Any]] = {}
    for index, raw_op in enumerate(raw_ops):
        op_label = f"{label}.cache_trace.ops[{index}]"
        if not isinstance(raw_op, Mapping):
            raise ValueError(f"{op_label} must be an object")
        op = dict(raw_op)
        _cache_json_value(op, label=op_label)
        op_id = _cache_identifier(_cache_field(op, "op_id", label=op_label), label=f"{op_label}.op_id")
        if op_id in op_by_id:
            raise ValueError(f"{label}.cache_trace repeats op_id {op_id!r}")
        kind = _cache_identifier(_cache_field(op, "kind", label=op_label), label=f"{op_label}.kind")
        status = _cache_field(op, "status", label=op_label)
        if status not in _CACHE_OPERATION_STATUSES:
            raise ValueError(f"{op_label}.status is unsupported")
        requested = _nonnegative_int(
            _cache_field(op, "input_tokens_requested", label=op_label),
            label=f"{op_label}.input_tokens_requested",
        )
        completed = _nonnegative_int(
            _cache_field(op, "input_tokens_completed", label=op_label),
            label=f"{op_label}.input_tokens_completed",
        )
        if requested is not None and completed is not None and completed > requested:
            raise ValueError(f"{op_label} completed more input tokens than requested")
        parsed = {
            **op,
            "op_id": op_id,
            "kind": kind,
            "status": status,
            "input_tokens_requested": requested,
            "input_tokens_completed": completed,
            "transfer_bytes": _nonnegative_int(
                _cache_field(op, "transfer_bytes", label=op_label),
                label=f"{op_label}.transfer_bytes",
            ),
            "logical_bytes": _nonnegative_int(
                _cache_field(op, "logical_bytes", label=op_label),
                label=f"{op_label}.logical_bytes",
            ),
            "source_entry_id": _cache_optional_identifier(
                op.get("source_entry_id"),
                label=f"{op_label}.source_entry_id",
            ),
            "result_entry_id": _cache_optional_identifier(
                op.get("result_entry_id"),
                label=f"{op_label}.result_entry_id",
            ),
            "consumer_placement_ids": _cache_id_list(
                op.get("consumer_placement_ids", []),
                label=f"{op_label}.consumer_placement_ids",
            ),
        }
        op_by_id[op_id] = parsed
        ops.append(parsed)

    entries = []
    entry_ids: set[str] = set()
    for index, raw_entry in enumerate(raw_entries):
        entry_label = f"{label}.cache_trace.entries[{index}]"
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"{entry_label} must be an object")
        entry = dict(raw_entry)
        _cache_json_value(entry, label=entry_label)
        entry_id = _cache_identifier(
            _cache_field(entry, "entry_id", label=entry_label),
            label=f"{entry_label}.entry_id",
        )
        if entry_id in entry_ids:
            raise ValueError(f"{label}.cache_trace repeats entry_id {entry_id!r}")
        entry_ids.add(entry_id)
        entries.append(
            {
                **entry,
                "entry_id": entry_id,
                "producer_attempt_uid": _cache_identifier(
                    _cache_field(entry, "producer_attempt_uid", label=entry_label),
                    label=f"{entry_label}.producer_attempt_uid",
                ),
                "created_by_op_id": _cache_optional_identifier(
                    _cache_field(entry, "created_by_op_id", label=entry_label),
                    label=f"{entry_label}.created_by_op_id",
                ),
                "origin_extraction_op_id": _cache_optional_identifier(
                    _cache_field(entry, "origin_extraction_op_id", label=entry_label),
                    label=f"{entry_label}.origin_extraction_op_id",
                ),
                "parent_entry_id": _cache_optional_identifier(
                    _cache_field(entry, "parent_entry_id", label=entry_label),
                    label=f"{entry_label}.parent_entry_id",
                ),
                "kind": _cache_identifier(
                    _cache_field(entry, "kind", label=entry_label),
                    label=f"{entry_label}.kind",
                ),
            }
        )

    placements = []
    placement_by_id: dict[str, dict[str, Any]] = {}
    for index, raw_placement in enumerate(raw_placements):
        placement_label = f"{label}.cache_trace.placements[{index}]"
        if not isinstance(raw_placement, Mapping):
            raise ValueError(f"{placement_label} must be an object")
        placement = dict(raw_placement)
        _cache_json_value(placement, label=placement_label)
        placement_id = _cache_identifier(
            _cache_field(placement, "placement_id", label=placement_label),
            label=f"{placement_label}.placement_id",
        )
        if placement_id in placement_by_id:
            raise ValueError(f"{label}.cache_trace repeats placement_id {placement_id!r}")
        source_token_start = _nonnegative_int(
            _cache_field(placement, "source_token_start", label=placement_label),
            label=f"{placement_label}.source_token_start",
        )
        source_token_end = _nonnegative_int(
            _cache_field(placement, "source_token_end", label=placement_label),
            label=f"{placement_label}.source_token_end",
        )
        if source_token_start is None or source_token_end is None:
            raise ValueError(f"{placement_label} source token bounds must be known")
        if source_token_end < source_token_start:
            raise ValueError(f"{placement_label} source token bounds are reversed")
        parsed = {
            **placement,
            "placement_id": placement_id,
            "event_id": _cache_identifier(
                _cache_field(placement, "event_id", label=placement_label),
                label=f"{placement_label}.event_id",
            ),
            "part_index": _nonnegative_int(
                _cache_field(placement, "part_index", label=placement_label),
                label=f"{placement_label}.part_index",
            ),
            "source_indices": _cache_int_list(
                _cache_field(placement, "source_indices", label=placement_label),
                label=f"{placement_label}.source_indices",
            ),
            "source_token_start": source_token_start,
            "source_token_end": source_token_end,
            "source_position_start": _nonnegative_int(
                _cache_field(placement, "source_position_start", label=placement_label),
                label=f"{placement_label}.source_position_start",
            ),
            "position_ids": _cache_int_list(
                _cache_field(placement, "position_ids", label=placement_label),
                label=f"{placement_label}.position_ids",
            ),
            "access_op_id": _cache_identifier(
                _cache_field(placement, "access_op_id", label=placement_label),
                label=f"{placement_label}.access_op_id",
            ),
            "accessed_entry_id": _cache_optional_identifier(
                _cache_field(placement, "accessed_entry_id", label=placement_label),
                label=f"{placement_label}.accessed_entry_id",
            ),
            "origin_extraction_op_id": _cache_optional_identifier(
                _cache_field(placement, "origin_extraction_op_id", label=placement_label),
                label=f"{placement_label}.origin_extraction_op_id",
            ),
        }
        placement_by_id[placement_id] = parsed
        placements.append(parsed)

    for placement in placements:
        access_op = op_by_id.get(placement["access_op_id"])
        if access_op is None:
            raise ValueError(
                f"{label}.cache_trace placement {placement['placement_id']!r} "
                "references an access op outside this trace"
            )
        if access_op["status"] != "completed":
            raise ValueError(
                f"{label}.cache_trace placement {placement['placement_id']!r} "
                "references an access op that did not complete"
            )
    for op in ops:
        for placement_id in op["consumer_placement_ids"]:
            if placement_id not in placement_by_id:
                raise ValueError(
                    f"{label}.cache_trace op {op['op_id']!r} "
                    f"references an unknown consumer placement {placement_id!r}"
                )

    return {
        "attempt_uid": attempt["attempt_uid"],
        "session_id": attempt["session_id"],
        "decision_key": attempt["decision_key"],
        "phase": attempt["phase"],
        "context_complete": trace["context_complete"],
        "status": attempt["status"],
        "workspace_source_group": workspace_source_group,
        "commit_status": commit_status,
        "ops": ops,
        "entries": entries,
        "placements": placements,
    }


def _cache_trace_for_attempt(
    trace: Mapping[str, Any],
    *,
    attempt: Mapping[str, Any],
    label: str,
) -> dict[str, Any] | None:
    stats = _stats(trace, label=label)
    top_level = trace.get("cache_trace", _MISSING)
    nested = stats.get("cache_trace", _MISSING) if stats is not None else _MISSING
    if attempt["status"] == "completed":
        if top_level is not _MISSING and top_level is not None:
            raise ValueError(f"{label}.cache_trace must be inside generation.stats on success")
        raw = nested
        location = "generation.stats.cache_trace"
    else:
        if nested is not _MISSING and nested is not None:
            raise ValueError(f"{label}.generation.stats.cache_trace is only for success")
        raw = top_level
        location = "generation_trace.cache_trace"
    if raw is _MISSING or raw is None:
        return None
    return {
        "location": location,
        "trace": _parse_cache_trace(raw, attempt=attempt, label=label),
    }


def _operation_metric(operations: list[dict[str, Any]], field: str) -> dict[str, Any]:
    known = [operation[field] for operation in operations if operation[field] is not None]
    return {
        "known_total": sum(known) if known else None,
        "known_operations": len(known),
        "unknown_operations": len(operations) - len(known),
    }


def _cache_lifecycle_traces(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    unique: dict[str, dict[str, Any]] = {}
    records_with_trace = 0
    for item in records:
        record = item["record"]
        label = f"record {item['session_id']!r}/{item['decision_key']!r}"
        for cache_field in ("session_cache_after", "session_cache_after_close"):
            cache = record.get(cache_field)
            if cache is None:
                continue
            cache_label = f"{label}.{cache_field}"
            if not isinstance(cache, Mapping):
                raise ValueError(f"{cache_label} must be an object or null")
            raw = cache.get("last_lifecycle_trace")
            if raw is None:
                continue
            records_with_trace += 1
            trace_label = f"{cache_label}.last_lifecycle_trace"
            if not isinstance(raw, Mapping):
                raise ValueError(f"{trace_label} must be an object or null")
            trace = dict(raw)
            if trace.get("schema") != _CACHE_LIFECYCLE_TRACE_SCHEMA:
                raise ValueError(f"{trace_label} has an unsupported schema")
            lifecycle_id = _cache_identifier(
                _cache_field(trace, "lifecycle_id", label=trace_label),
                label=f"{trace_label}.lifecycle_id",
            )
            associated_attempt_uid = _cache_optional_identifier(
                _cache_field(trace, "associated_attempt_uid", label=trace_label),
                label=f"{trace_label}.associated_attempt_uid",
            )
            raw_ops = _cache_field(trace, "ops", label=trace_label)
            if not isinstance(raw_ops, list):
                raise ValueError(f"{trace_label}.ops must be a list")
            ops = []
            op_ids: set[str] = set()
            for index, raw_op in enumerate(raw_ops):
                op_label = f"{trace_label}.ops[{index}]"
                if not isinstance(raw_op, Mapping):
                    raise ValueError(f"{op_label} must be an object")
                op = dict(raw_op)
                op_id = _cache_identifier(
                    _cache_field(op, "op_id", label=op_label), label=f"{op_label}.op_id"
                )
                if op_id in op_ids:
                    raise ValueError(f"{trace_label} repeats op_id {op_id!r}")
                op_ids.add(op_id)
                _cache_json_value(op, label=op_label)
                ops.append(op)
            parsed = {
                "lifecycle_id": lifecycle_id,
                "associated_attempt_uid": associated_attempt_uid,
                "session_id": _cache_identifier(
                    _cache_field(trace, "session_id", label=trace_label),
                    label=f"{trace_label}.session_id",
                ),
                "ops": ops,
                "observing_session_id": item["session_id"],
                "observing_decision_key": item["decision_key"],
            }
            previous = unique.get(lifecycle_id)
            if previous is not None:
                observer_fields = {"observing_session_id", "observing_decision_key"}
                comparable = {
                    key: value for key, value in parsed.items() if key not in observer_fields
                }
                prior_comparable = {
                    key: value for key, value in previous.items() if key not in observer_fields
                }
                if _canonical(prior_comparable, label=f"cache lifecycle {lifecycle_id!r}") != _canonical(
                    comparable, label=f"cache lifecycle {lifecycle_id!r}"
                ):
                    raise ValueError(f"cache lifecycle {lifecycle_id!r} has conflicting repeated data")
                continue
            unique[lifecycle_id] = parsed
    return {
        "trace_schema": _CACHE_LIFECYCLE_TRACE_SCHEMA,
        "records_with_trace": records_with_trace,
        "unique_lifecycle_traces": len(unique),
        "traces": list(unique.values()),
    }


def _cache_provenance(
    attempts: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    links = []
    known_attempts = 0
    context_complete_attempts = 0
    unique_ops: dict[str, dict[str, Any]] = {}
    for attempt in attempts:
        attached = attempt.get("cache_trace")
        if attached is None:
            links.append(
                {
                    "session_id": attempt["session_id"],
                    "decision_key": attempt["decision_key"],
                    "attempt_uid": attempt["attempt_uid"],
                    "attempt_index": attempt["attempt_index"],
                    "phase": attempt["phase"],
                    "status": attempt["status"],
                    "source": attempt["source"],
                    "trace_coverage": "unknown",
                    "op_ids": [],
                    "entry_ids": [],
                    "placement_ids": [],
                }
            )
            continue
        known_attempts += 1
        trace = attached["trace"]
        if trace["context_complete"]:
            context_complete_attempts += 1
        links.append(
            {
                "session_id": attempt["session_id"],
                "decision_key": attempt["decision_key"],
                "attempt_uid": attempt["attempt_uid"],
                "attempt_index": attempt["attempt_index"],
                "phase": attempt["phase"],
                "status": attempt["status"],
                "source": attempt["source"],
                "trace_coverage": "known",
                "trace_location": attached["location"],
                "context_complete": trace["context_complete"],
                "workspace_source_group": trace["workspace_source_group"],
                "commit_status": trace["commit_status"],
                "ops": trace["ops"],
                "entries": trace["entries"],
                "placements": trace["placements"],
                "op_ids": [op["op_id"] for op in trace["ops"]],
                "entry_ids": [entry["entry_id"] for entry in trace["entries"]],
                "placement_ids": [placement["placement_id"] for placement in trace["placements"]],
            }
        )
        for operation in trace["ops"]:
            op_id = operation["op_id"]
            previous = unique_ops.get(op_id)
            if previous is not None:
                if _canonical(previous, label=f"cache operation {op_id!r}") != _canonical(
                    operation, label=f"cache operation {op_id!r}"
                ):
                    raise ValueError(f"cache operation {op_id!r} has conflicting repeated data")
                continue
            unique_ops[op_id] = operation

    by_kind: dict[str, list[dict[str, Any]]] = {}
    for operation in unique_ops.values():
        by_kind.setdefault(operation["kind"], []).append(operation)
    return {
        "trace_schema": _CACHE_TRACE_SCHEMA,
        "trace_coverage": {
            "known_attempts": known_attempts,
            "unknown_attempts": len(attempts) - known_attempts,
            "context_complete_attempts": context_complete_attempts,
            "context_incomplete_attempts": known_attempts - context_complete_attempts,
        },
        "attempts": links,
        "unique_operations": len(unique_ops),
        "by_operation_kind": {
            kind: {
                "unique_operations": len(operations),
                "completed_operations": sum(
                    operation["status"] == "completed" for operation in operations
                ),
                "not_completed_operations": sum(
                    operation["status"] != "completed" for operation in operations
                ),
                "input_tokens_completed": _operation_metric(
                    operations, "input_tokens_completed"
                ),
                "transfer_bytes": _operation_metric(operations, "transfer_bytes"),
                "logical_bytes": _operation_metric(operations, "logical_bytes"),
            }
            for kind, operations in by_kind.items()
        },
        "lifecycle_traces": _cache_lifecycle_traces(records),
    }


def _successful_session_cache(
    records: list[dict[str, Any]],
    *,
    unrecorded_decisions: int = 0,
    framed_input_complete: bool = True,
) -> dict[str, Any]:
    peak_values: dict[str, list[int | None]] = {
        name: [] for name in _SESSION_CACHE_PEAK_FIELDS
    }
    transfer_values: dict[str, list[int | None]] = {
        name: [] for name in _SESSION_TRANSFER_FIELDS
    }
    successful = [item for item in records if item["record"].get("status") == "ok"]
    for item in successful:
        record = item["record"]
        label = f"record {item['session_id']!r}/{item['decision_key']!r}"
        cache = record.get("session_cache_after")
        if cache is not None and not isinstance(cache, Mapping):
            raise ValueError(f"{label}.session_cache_after must be an object or null")
        if isinstance(cache, Mapping):
            cache_session = cache.get("session_id")
            if cache_session is not None and cache_session != item["session_id"]:
                raise ValueError(f"{label}.session_cache_after has a different session_id")
        for name in peak_values:
            raw = cache.get(name) if isinstance(cache, Mapping) else None
            peak_values[name].append(
                _nonnegative_int(raw, label=f"{label}.session_cache_after.{name}")
            )
        for name in transfer_values:
            raw = cache.get(name) if isinstance(cache, Mapping) else None
            transfer_values[name].append(
                _nonnegative_int(raw, label=f"{label}.session_cache_after.{name}")
            )
        transfer_in = transfer_values["last_transfer_bytes_in"][-1]
        transfer_out = transfer_values["last_transfer_bytes_out"][-1]
        transfer_total = transfer_values["last_transfer_bytes_total"][-1]
        if None not in (transfer_in, transfer_out, transfer_total):
            if transfer_total != transfer_in + transfer_out:
                raise ValueError(f"{label}.session_cache_after transfer total is inconsistent")
    for _ in range(unrecorded_decisions):
        for values in peak_values.values():
            values.append(None)
        for values in transfer_values.values():
            values.append(None)
    result = {
        "successful_decisions": len(successful),
        "unrecorded_decisions": unrecorded_decisions,
        "host_device_byte_peaks": {
            name: _peak_metric(
                values, aggregation="max_across_successful_decisions"
            )
            for name, values in peak_values.items()
        },
        "transfer_bytes": {
            name: _sum_metric(values) for name, values in transfer_values.items()
        },
    }
    if not framed_input_complete:
        for name in _SESSION_CACHE_PEAK_FIELDS:
            result["host_device_byte_peaks"][name]["strict_peak"] = None
        for name in _SESSION_TRANSFER_FIELDS:
            result["transfer_bytes"][name]["strict_total"] = None
    return result


def _decision_runtime(
    records: list[dict[str, Any]],
    *,
    unrecorded_decisions: int = 0,
    framed_input_complete: bool = True,
) -> dict[str, Any]:
    values = [
        _nonnegative_number(
            item["record"].get("decision_runtime_seconds"),
            label=(
                f"record {item['session_id']!r}/{item['decision_key']!r} "
                "decision_runtime_seconds"
            ),
        )
        for item in records
    ]
    values.extend([None] * unrecorded_decisions)
    metric = _sum_metric(values)
    if not framed_input_complete:
        metric["strict_total"] = None
    metric["known_decisions"] = metric.pop("known_calls")
    metric["unknown_decisions"] = metric.pop("unknown_calls")
    return {
        "decision_runtime_seconds": metric,
        "includes_controller_timing": True,
    }


def _controller_timing(
    records: list[dict[str, Any]],
    *,
    unrecorded_decisions: int = 0,
    framed_input_complete: bool = True,
) -> dict[str, Any]:
    values: dict[str, list[float | None]] = {
        "prepare_seconds": [],
        "reconsider_seconds": [],
    }
    for item in records:
        record = item["record"]
        timing = record.get("controller_timing")
        if timing is not None and not isinstance(timing, Mapping):
            raise ValueError(
                f"record {item['session_id']!r}/{item['decision_key']!r} "
                "controller_timing must be an object or null"
            )
        for name in values:
            raw = timing.get(name) if isinstance(timing, Mapping) else None
            values[name].append(
                _nonnegative_number(
                    raw,
                    label=(
                        f"record {item['session_id']!r}/{item['decision_key']!r} "
                        f"controller_timing.{name}"
                    ),
                )
            )
    for metric_values in values.values():
        metric_values.extend([None] * unrecorded_decisions)
    result = {}
    for name, metric_values in values.items():
        metric = _sum_metric(metric_values)
        if not framed_input_complete:
            metric["strict_total"] = None
        metric["known_decisions"] = metric.pop("known_calls")
        metric["unknown_decisions"] = metric.pop("unknown_calls")
        result[name] = metric
    return result


def _source_attempt(attempt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source": attempt["source"],
        "decision_key": attempt["decision_key"],
        "phase": attempt["phase"],
        "attempt_uid": attempt["attempt_uid"],
        "attempt_index": attempt["attempt_index"],
        "status": attempt["status"],
        "discarded": attempt["discarded"],
    }


def _journal_identity(started: Mapping[str, Any], *, label: str) -> tuple[str, str]:
    context = started.get("eval_context")
    if not isinstance(context, Mapping):
        raise ValueError(f"{label}.eval_context must be an object")
    session_id = context.get("task_id")
    decision_key = context.get("decision_id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError(f"{label}.eval_context.task_id must be a nonempty string")
    if not isinstance(decision_key, str) or not decision_key:
        raise ValueError(f"{label}.eval_context.decision_id must be a nonempty string")
    try:
        request_identity = json.loads(started.get("request_id"))
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label}.request_id must encode [session_id, decision_key]") from error
    if request_identity != [session_id, decision_key]:
        raise ValueError(f"{label}.request_id disagrees with eval_context")
    return session_id, decision_key


def _read_generation_journal(path: str | Path) -> dict[str, Any]:
    journal = read_attempt_journal(path)
    validated = summarize_attempt_journal(path)
    if (
        len([row for row in journal["records"] if row.get("event") == "started"])
        != validated["started"]
        or len([row for row in journal["records"] if row.get("event") == "finished"])
        != validated["finished"]
        or journal["truncated_tail"] != validated["truncated_tail"]
    ):
        raise RuntimeError("attempt journal changed while it was being summarized")

    started = {
        row["attempt_uid"]: row
        for row in journal["records"]
        if row["event"] == "started"
    }
    finished = {
        row["attempt_uid"]: row
        for row in journal["records"]
        if row["event"] == "finished"
    }
    generation = []
    for uid, start in started.items():
        if start["kind"] != "generation":
            continue
        label = f"attempt journal generation {uid!r}"
        session_id, decision_key = _journal_identity(start, label=label)
        finish = finished.get(uid)
        generation.append(
            {
                "attempt_uid": uid,
                "attempt_index": start["attempt_index"],
                "session_id": session_id,
                "decision_key": decision_key,
                "status": finish["status"] if finish is not None else "pending",
                "usage": finish.get("usage") if finish is not None else None,
            }
        )
    return {
        "path": str(Path(path)),
        "truncated_tail": journal["truncated_tail"],
        "generation_attempts": generation,
    }


def _reconcile_attempt_journal(
    attempts: list[dict[str, Any]],
    unique_records: Mapping[tuple[str, str], dict[str, Any]],
    path: str | Path,
) -> dict[str, Any]:
    journal = _read_generation_journal(path)
    by_uid = {
        attempt["attempt_uid"]: attempt
        for attempt in journal["generation_attempts"]
    }
    matched: set[str] = set()
    for attempt in attempts:
        uid = attempt["attempt_uid"]
        ledger = by_uid.get(uid)
        if ledger is None:
            raise ValueError(f"step trace attempt {uid!r} has no generation journal start")
        if uid in matched:
            raise ValueError(f"step traces reuse generation journal attempt {uid!r}")
        matched.add(uid)
        identity = (attempt["session_id"], attempt["decision_key"])
        if identity != (ledger["session_id"], ledger["decision_key"]):
            raise ValueError(f"step trace attempt {uid!r} disagrees with journal identity")
        if attempt["attempt_index"] != ledger["attempt_index"]:
            raise ValueError(f"step trace attempt {uid!r} disagrees with journal index")
        trace_status = (
            "pending"
            if attempt["status"] in {"started", "pending"}
            else attempt["status"]
        )
        if trace_status != ledger["status"]:
            raise ValueError(f"step trace attempt {uid!r} disagrees with journal status")
        journal_usage = _usage(
            {"usage": ledger["usage"]}, label=f"attempt journal generation {uid!r}"
        )
        if attempt["values"]["usage"] != journal_usage:
            raise ValueError(f"step trace attempt {uid!r} disagrees with journal usage")

    journal_only = []
    for ledger in journal["generation_attempts"]:
        if ledger["attempt_uid"] in matched:
            continue
        identity = (ledger["session_id"], ledger["decision_key"])
        if identity in unique_records:
            raise ValueError(
                "complete step record omits generation journal attempt "
                f"{ledger['attempt_uid']!r}"
            )
        trace = {"usage": ledger["usage"], "generation": None}
        journal_only.append(
            {
                "session_id": ledger["session_id"],
                "decision_key": ledger["decision_key"],
                "phase": "unrecorded",
                "status": ledger["status"],
                "discarded": None,
                "attempt_uid": ledger["attempt_uid"],
                "attempt_index": ledger["attempt_index"],
                "source": "attempt_journal_only",
                "values": _attempt_values(
                    trace,
                    label=f"attempt journal generation {ledger['attempt_uid']!r}",
                ),
                "cache_trace": None,
            }
        )
    attempts.extend(journal_only)
    journal["matched_step_attempts"] = len(matched)
    journal["journal_only_generation_attempts"] = len(journal_only)
    return journal


def _summarize_session(
    session_id: str,
    records: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    *,
    strict_totals_scope: str,
    framed_input_complete: bool,
) -> dict[str, Any]:
    recorded_by_decision = {item["decision_key"]: item for item in records}
    decision_order = list(recorded_by_decision)
    for attempt in attempts:
        if attempt["decision_key"] not in decision_order:
            decision_order.append(attempt["decision_key"])
    unrecorded_decisions = [
        decision for decision in decision_order if decision not in recorded_by_decision
    ]
    phases = []
    for phase in _SUMMARY_PHASES:
        selected = [attempt for attempt in attempts if attempt["phase"] == phase]
        if not selected:
            continue
        phases.append(
            {
                "phase": phase,
                "generation_attempts": len(selected),
                "completed_attempts": sum(
                    attempt["status"] == "completed" for attempt in selected
                ),
                "failed_or_pending_attempts": sum(
                    attempt["status"] != "completed" for attempt in selected
                ),
                "discarded_attempts": sum(
                    attempt["discarded"] is True for attempt in selected
                ),
                "discarded_status_unknown_attempts": sum(
                    attempt["discarded"] is None for attempt in selected
                ),
                "source_attempts": [_source_attempt(attempt) for attempt in selected],
                "costs": _generation_costs(
                    selected,
                    strict_totals_scope=strict_totals_scope,
                    framed_input_complete=framed_input_complete,
                ),
            }
        )

    return {
        "session_id": session_id,
        "decision_count": len(decision_order),
        "recorded_decisions": len(records),
        "unrecorded_decisions": len(unrecorded_decisions),
        "source_decisions": [
            {
                "decision_key": decision_key,
                "record_status": (
                    recorded_by_decision[decision_key]["record"].get("status")
                    if decision_key in recorded_by_decision
                    else "unrecorded"
                ),
                "attempt_uids": [
                    attempt["attempt_uid"]
                    for attempt in attempts
                    if attempt["decision_key"] == decision_key
                ],
            }
            for decision_key in decision_order
        ],
        "generation_attempts": len(attempts),
        "completed_attempts": sum(
            attempt["status"] == "completed" for attempt in attempts
        ),
        "failed_or_pending_attempts": sum(
            attempt["status"] != "completed" for attempt in attempts
        ),
        "discarded_attempts": sum(attempt["discarded"] is True for attempt in attempts),
        "discarded_status_unknown_attempts": sum(
            attempt["discarded"] is None for attempt in attempts
        ),
        "source_attempts": [_source_attempt(attempt) for attempt in attempts],
        "decision_runtime": _decision_runtime(
            records,
            unrecorded_decisions=len(unrecorded_decisions),
            framed_input_complete=framed_input_complete,
        ),
        "controller_timing": _controller_timing(
            records,
            unrecorded_decisions=len(unrecorded_decisions),
            framed_input_complete=framed_input_complete,
        ),
        "session_cache_after": _successful_session_cache(
            records,
            unrecorded_decisions=len(unrecorded_decisions),
            framed_input_complete=framed_input_complete,
        ),
        "costs": _generation_costs(
            attempts,
            strict_totals_scope=strict_totals_scope,
            framed_input_complete=framed_input_complete,
        ),
        "phases": phases,
    }


def summarize_event_native_steps(
    records: Iterable[Mapping[str, Any]] | Mapping[str, Any],
    *,
    attempt_journal: str | Path | None = None,
    steps_truncated_tail: bool = False,
) -> dict[str, Any]:
    """Summarize actual generation attempts without inventing missing costs.

    Exact duplicate decision records are charged once. A second record for the
    same session and decision must be canonically equivalent as finite JSON;
    otherwise its input or trace provenance is ambiguous and is rejected.
    """

    if type(steps_truncated_tail) is not bool:
        raise TypeError("steps_truncated_tail must be a bool")
    source = [records] if isinstance(records, Mapping) else list(records)
    if not source and attempt_journal is None and not steps_truncated_tail:
        raise ValueError("at least one event-native step record is required")

    unique: dict[tuple[str, str], dict[str, Any]] = {}
    session_order: list[str] = []
    duplicate_records = 0
    for index, raw_record in enumerate(source):
        if not isinstance(raw_record, Mapping):
            raise TypeError(f"records[{index}] must be an object")
        record = dict(raw_record)
        session_id = record.get("session_id")
        decision_key = record.get("decision_key")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError(f"records[{index}].session_id must be a nonempty string")
        if not isinstance(decision_key, str) or not decision_key:
            raise ValueError(f"records[{index}].decision_key must be a nonempty string")
        trace = record.get("generation_trace")
        if not isinstance(trace, list):
            raise ValueError(f"records[{index}].generation_trace must be a list")
        canonical = _canonical(record, label=f"records[{index}]")
        key = (session_id, decision_key)
        previous = unique.get(key)
        if previous is not None:
            if previous["canonical"] != canonical:
                raise ValueError(
                    "conflicting duplicate decision record for "
                    f"session={session_id!r} decision={decision_key!r}"
                )
            duplicate_records += 1
            continue
        if session_id not in session_order:
            session_order.append(session_id)
        unique[key] = {
            "session_id": session_id,
            "decision_key": decision_key,
            "record": record,
            "canonical": canonical,
        }

    parsed_attempts: list[dict[str, Any]] = []
    attempt_identity: dict[tuple[str, str, str], str] = {}
    for item in unique.values():
        record = item["record"]
        for trace_index, raw_trace in enumerate(record["generation_trace"]):
            label = (
                f"record {item['session_id']!r}/{item['decision_key']!r} "
                f"generation_trace[{trace_index}]"
            )
            if not isinstance(raw_trace, Mapping):
                raise ValueError(f"{label} must be an object")
            trace = dict(raw_trace)
            phase = trace.get("phase")
            if phase not in _TRACE_PHASES:
                raise ValueError(f"{label}.phase must be draft or regeneration")
            status = trace.get("status")
            if status not in {"started", "pending", "completed", "failed"}:
                raise ValueError(f"{label}.status is unsupported")
            discarded = trace.get("discarded")
            if type(discarded) is not bool:
                raise ValueError(f"{label}.discarded must be a bool")
            attempt_uid = trace.get("attempt_uid")
            if not isinstance(attempt_uid, str) or not attempt_uid:
                raise ValueError(f"{label}.attempt_uid must be a nonempty string")
            attempt_index = trace.get("attempt_index")
            if type(attempt_index) is not int or attempt_index <= 0:
                raise ValueError(f"{label}.attempt_index must be a positive integer")
            identity = (item["session_id"], item["decision_key"], attempt_uid)
            trace_canonical = _canonical(trace, label=label)
            previous = attempt_identity.get(identity)
            if previous is not None:
                if previous != trace_canonical:
                    raise ValueError(
                        "conflicting generation attempt for "
                        f"session={identity[0]!r} decision={identity[1]!r} "
                        f"attempt_uid={identity[2]!r}"
                    )
                raise ValueError(f"{label} repeats an attempt within one decision record")
            attempt_identity[identity] = trace_canonical
            attempt = {
                "session_id": item["session_id"],
                "decision_key": item["decision_key"],
                "phase": phase,
                "status": status,
                "discarded": discarded,
                "attempt_uid": attempt_uid,
                "attempt_index": attempt_index,
                "source": "step_trace",
                "values": _attempt_values(trace, label=label),
            }
            attempt["cache_trace"] = _cache_trace_for_attempt(
                trace, attempt=attempt, label=label
            )
            parsed_attempts.append(attempt)

    journal_info = None
    if attempt_journal is not None:
        journal_info = _reconcile_attempt_journal(
            parsed_attempts, unique, attempt_journal
        )
    journal_truncated_tail = (
        journal_info["truncated_tail"] if journal_info is not None else None
    )
    framed_input_complete = not steps_truncated_tail and not bool(journal_truncated_tail)
    strict_totals_scope = (
        "attempt_journal_generation_attempts"
        if journal_info is not None
        else "input_steps_only"
    )

    for attempt in parsed_attempts:
        if attempt["session_id"] not in session_order:
            session_order.append(attempt["session_id"])

    sessions = []
    for session_id in session_order:
        session_records = [
            item for item in unique.values() if item["session_id"] == session_id
        ]
        session_attempts = [
            attempt for attempt in parsed_attempts if attempt["session_id"] == session_id
        ]
        sessions.append(
            _summarize_session(
                session_id,
                session_records,
                session_attempts,
                strict_totals_scope=strict_totals_scope,
                framed_input_complete=framed_input_complete,
            )
        )

    observed_decisions = {
        (item["session_id"], item["decision_key"]) for item in unique.values()
    }
    observed_decisions.update(
        (attempt["session_id"], attempt["decision_key"])
        for attempt in parsed_attempts
    )
    journal_only_attempts = sum(
        attempt["source"] == "attempt_journal_only" for attempt in parsed_attempts
    )

    return {
        "schema": SCHEMA,
        "accounting_scope": (
            "Every unique generation attempt, including discarded drafts and "
            "regenerations. OpenAI resident prompt usage and actual model work are "
            "separate. KV tensor bytes are logical; PyTorch allocator peak is a "
            "separate optional measurement. Decision runtime already includes "
            "controller timing. Session transfers are summed once per successful "
            "decision from session_cache_after, never from generation stats. "
            "Without an attempt journal, strict totals cover only the supplied "
            "complete step records and are not a claim about a whole run."
        ),
        "input_records": len(source),
        "recorded_decisions": len(unique),
        "unique_decisions": len(observed_decisions),
        "duplicate_records_ignored": duplicate_records,
        "steps_truncated_tail": steps_truncated_tail,
        "attempt_journal_truncated_tail": journal_truncated_tail,
        "attempt_inventory": {
            "scope": strict_totals_scope,
            "run_inventory_verified": journal_info is not None and framed_input_complete,
            "attempt_journal_supplied": journal_info is not None,
            "journal_generation_attempts": (
                len(journal_info["generation_attempts"])
                if journal_info is not None
                else None
            ),
            "matched_step_attempts": (
                journal_info["matched_step_attempts"]
                if journal_info is not None
                else None
            ),
            "journal_only_generation_attempts": journal_only_attempts,
        },
        "generation_attempts": len(parsed_attempts),
        "source_attempts": [
            {"session_id": attempt["session_id"], **_source_attempt(attempt)}
            for attempt in parsed_attempts
        ],
        "costs": _generation_costs(
            parsed_attempts,
            strict_totals_scope=strict_totals_scope,
            framed_input_complete=framed_input_complete,
        ),
        "cache_provenance": _cache_provenance(
            parsed_attempts, list(unique.values())
        ),
        "session_count": len(sessions),
        "sessions": sessions,
    }


def read_event_native_steps(path: str | Path) -> dict[str, Any]:
    """Read complete JSONL frames and report, but never parse, a partial tail."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    raw = source.read_bytes()
    truncated_tail = bool(raw) and not raw.endswith(b"\n")
    complete = raw.rpartition(b"\n")[0] if truncated_tail else raw
    records = []
    for line_number, line in enumerate(complete.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid JSON in {source} line {line_number}") from error
        if not isinstance(value, dict):
            raise ValueError(f"{source} line {line_number} must contain an object")
        records.append(value)
    return {"records": records, "truncated_tail": truncated_tail}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=Path, required=True)
    parser.add_argument("--attempt-journal", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    loaded = read_event_native_steps(args.steps)
    summary = summarize_event_native_steps(
        loaded["records"],
        attempt_journal=args.attempt_journal,
        steps_truncated_tail=loaded["truncated_tail"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


if __name__ == "__main__":
    main()
