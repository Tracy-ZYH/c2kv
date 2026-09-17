"""Durable, content-free journal for costly generation and extraction attempts."""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA = "a-runtime-attempt-journal-v1"
KINDS = frozenset({"generation", "extraction"})
FINISH_STATUSES = frozenset({"completed", "failed"})
EVAL_CONTEXT_FIELDS = (
    "benchmark", "run_id", "task_id", "attempt_id", "decision_id",
    "user_turn", "step", "attempt", "user_turn_index", "tool_step_index",
)
USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
_SCALAR_TYPES = (str, int, float, bool, type(None))


@dataclass(frozen=True)
class AttemptHandle:
    """Opaque identity returned only after the durable started record exists."""

    attempt_uid: str
    kind: str
    attempt_index: int
    request_id: str
    eval_context: dict[str, Any]


def _filtered_eval_context(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("eval_context must be a mapping")
    filtered: dict[str, Any] = {}
    for key in EVAL_CONTEXT_FIELDS:
        if key not in value:
            continue
        item = value[key]
        if not isinstance(item, _SCALAR_TYPES):
            raise ValueError(f"eval_context.{key} must be a JSON scalar")
        filtered[key] = item
    return filtered


def _validate_identity(kind: str, attempt_index: int, request_id: str) -> None:
    if kind not in KINDS:
        raise ValueError(f"attempt kind must be one of {sorted(KINDS)}")
    if type(attempt_index) is not int or attempt_index <= 0:
        raise ValueError("attempt_index must be a positive integer")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("request_id must be a non-empty string")


def _filtered_usage(value: Mapping[str, Any] | None) -> dict[str, int] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("usage must be a mapping or None")
    filtered: dict[str, int] = {}
    for key in USAGE_FIELDS:
        if key not in value:
            continue
        item = value[key]
        if type(item) is not int or item < 0:
            raise ValueError(f"usage.{key} must be a nonnegative integer")
        filtered[key] = item
    return filtered


class AttemptJournal:
    """Append records under a process-local lock and fsync before returning."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def _append(self, record: Mapping[str, Any]) -> None:
        encoded = (json.dumps(record, ensure_ascii=True, separators=(",", ":"))
                   + "\n").encode("utf-8")
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_BINARY", 0)
            descriptor = os.open(self.path, flags, 0o600)
            try:
                remaining = memoryview(encoded)
                while remaining:
                    written = os.write(descriptor, remaining)
                    if written <= 0:
                        raise OSError("attempt journal append made no progress")
                    remaining = remaining[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    def start(
        self,
        kind: str,
        attempt_index: int,
        request_id: str,
        eval_context: Mapping[str, Any],
    ) -> AttemptHandle:
        """Durably record a reserved attempt immediately before costly work."""
        _validate_identity(kind, attempt_index, request_id)
        filtered = _filtered_eval_context(eval_context)
        handle = AttemptHandle(
            attempt_uid=uuid.uuid4().hex,
            kind=kind,
            attempt_index=attempt_index,
            request_id=request_id,
            eval_context=filtered,
        )
        self._append({
            "schema": SCHEMA,
            "event": "started",
            "status": "started",
            "attempt_uid": handle.attempt_uid,
            "kind": handle.kind,
            "attempt_index": handle.attempt_index,
            "request_id": handle.request_id,
            "eval_context": handle.eval_context,
            "timestamp_unix_ns": time.time_ns(),
            "scope": "reserved and about to submit; backend receipt is not established",
        })
        return handle

    def finish(
        self,
        handle: AttemptHandle,
        status: str,
        *,
        usage: Mapping[str, Any] | None = None,
    ) -> None:
        """Append a local return state, independent of semantic/scorer success."""
        if not isinstance(handle, AttemptHandle):
            raise TypeError("handle must be an AttemptHandle")
        if status not in FINISH_STATUSES:
            raise ValueError(f"finish status must be one of {sorted(FINISH_STATUSES)}")
        filtered_usage = _filtered_usage(usage)
        self._append({
            "schema": SCHEMA,
            "event": "finished",
            "status": status,
            "attempt_uid": handle.attempt_uid,
            "kind": handle.kind,
            "attempt_index": handle.attempt_index,
            "request_id": handle.request_id,
            "eval_context": handle.eval_context,
            "timestamp_unix_ns": time.time_ns(),
            "usage": filtered_usage,
            "status_scope": (
                "completed means HTTP JSON or producer returned; it does not mean "
                "backend semantic success or scorer success"),
        })


def read_attempt_journal(path: str | Path) -> dict[str, Any]:
    """Read complete JSONL records; never interpret a trailing partial line."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    raw = source.read_bytes()
    truncated_tail = bool(raw) and not raw.endswith(b"\n")
    complete = raw.rpartition(b"\n")[0] if truncated_tail else raw
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(complete.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"attempt journal line {line_number} is invalid JSON") from error
        if not isinstance(record, dict) or record.get("schema") != SCHEMA:
            raise ValueError(f"attempt journal line {line_number} has invalid schema")
        records.append(record)
    return {"schema": SCHEMA, "records": records, "truncated_tail": truncated_tail}


def attempt_journal_path(request_log: str | Path) -> Path:
    """Place the journal beside a request log without matching proxy_*.jsonl."""
    value = str(request_log)
    if not value:
        raise ValueError("request_log must be a non-empty path")
    path = Path(request_log)
    if not path.name:
        raise ValueError("request_log must include a file name")
    return path.with_name("attempts_" + path.name)


def summarize_attempt_journal(path: str | Path) -> dict[str, Any]:
    """Validate pairs and report unfinished reservations as pending/unknown."""
    journal = read_attempt_journal(path)
    started: dict[str, dict[str, Any]] = {}
    finished: dict[str, dict[str, Any]] = {}
    for line_number, record in enumerate(journal["records"], 1):
        uid = record.get("attempt_uid")
        if not isinstance(uid, str) or not uid:
            raise ValueError(f"attempt journal line {line_number} lacks attempt_uid")
        event = record.get("event")
        if event == "started":
            if uid in started:
                raise ValueError(f"duplicate started record for {uid}")
            _validate_identity(
                record.get("kind"), record.get("attempt_index"), record.get("request_id"))
            _filtered_eval_context(record.get("eval_context"))
            if record.get("status") != "started":
                raise ValueError(f"started record {uid} has invalid status")
            started[uid] = record
        elif event == "finished":
            if uid not in started:
                raise ValueError(f"finished record {uid} has no durable start")
            if uid in finished:
                raise ValueError(f"duplicate finished record for {uid}")
            if record.get("status") not in FINISH_STATUSES:
                raise ValueError(f"finished record {uid} has invalid status")
            _filtered_usage(record.get("usage"))
            identity = ("kind", "attempt_index", "request_id", "eval_context")
            if any(record.get(key) != started[uid].get(key) for key in identity):
                raise ValueError(f"finished record {uid} changes attempt identity")
            finished[uid] = record
        else:
            raise ValueError(f"attempt journal line {line_number} has invalid event")

    pending_uids = [uid for uid in started if uid not in finished]
    usage_records = [record["usage"] for record in finished.values()
                     if record.get("usage") is not None]
    usage_observed = {key: [usage[key] for usage in usage_records if key in usage]
                      for key in USAGE_FIELDS}
    usage_totals = {key: sum(values) if values else None
                    for key, values in usage_observed.items()}
    by_kind = {}
    for kind in sorted(KINDS):
        kind_started = [uid for uid, record in started.items() if record["kind"] == kind]
        kind_finished = [uid for uid in kind_started if uid in finished]
        by_kind[kind] = {
            "started": len(kind_started),
            "finished": len(kind_finished),
            "completed": sum(finished[uid]["status"] == "completed" for uid in kind_finished),
            "failed": sum(finished[uid]["status"] == "failed" for uid in kind_finished),
            "pending": sum(uid not in finished for uid in kind_started),
        }
    return {
        "schema": SCHEMA,
        "started": len(started),
        "finished": len(finished),
        "completed": sum(record["status"] == "completed" for record in finished.values()),
        "failed": sum(record["status"] == "failed" for record in finished.values()),
        "pending": len(pending_uids),
        "pending_attempts": [
            {key: started[uid][key] for key in (
                "attempt_uid", "kind", "attempt_index", "request_id", "eval_context")}
            for uid in pending_uids
        ],
        "pending_token_accounting": "unknown",
        "attempt_count_scope": {
            "generation": "reserved chat transport attempts; backend receipt is not established",
            "extraction": "reserved client producer invocations, not general HTTP attempts",
        },
        "finished_with_usage": len(usage_records),
        "finished_usage_totals": usage_totals,
        "finished_usage_observed_attempts": {
            key: len(values) for key, values in usage_observed.items()},
        "usage_totals_scope": (
            "sums of observed usage fields only, not total run cost; absent fields "
            "and pending attempt usage are unknown"),
        "finished_status_scope": (
            "completed means HTTP JSON or producer returned; it does not mean "
            "backend semantic success or scorer success"),
        "truncated_tail": journal["truncated_tail"],
        "by_kind": by_kind,
    }
