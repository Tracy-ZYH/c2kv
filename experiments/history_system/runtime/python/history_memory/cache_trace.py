"""JSON-safe provenance records for event-native cache operations."""

from __future__ import annotations

import math
import uuid
from collections.abc import Mapping
from typing import Any


CACHE_MEMORY_RANGE_PREFIX = "c2kv_cache_memory:"
CACHE_MEMORY_OPERATION_KINDS = frozenset(
    {
        "extract",
        "system_prefill",
        "assemble_prefix",
        "raw_prefill",
        "incremental_decode",
        "full_recompute",
        "cpu_memo_hydrate",
        "cpu_memo_copy",
        "system_cpu_memo_hydrate",
        "system_cpu_memo_copy",
        "snapshot_take_whole",
        "snapshot_clone_prefix",
        "snapshot_empty_prefix",
    }
)


_CONTEXT_FIELDS = ("attempt_uid", "session_id", "decision_key", "phase")
_OP_RESERVED_FIELDS = {"op_id", "kind", "status", "error_type"}
_ENTRY_RESERVED_FIELDS = {
    "entry_id",
    "kind",
    "created_by_op_id",
    "producer_attempt_uid",
    "parent_entry_id",
    "origin_extraction_op_id",
}
_PLACEMENT_RESERVED_FIELDS = {"placement_id"}
_OP_DEFAULTS = {
    "input_tokens_requested": None,
    "input_tokens_completed": None,
    "transfer_bytes": None,
    "logical_bytes": None,
}


def _json_snapshot(value: Any, *, field: str) -> Any:
    """Copy one finite, JSON-native value without coercing foreign objects."""

    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{field} must not contain a non-finite float")
        return value
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{field} contains a non-string mapping key")
            copied[key] = _json_snapshot(item, field=f"{field}.{key}")
        return copied
    if type(value) is list:
        return [
            _json_snapshot(item, field=f"{field}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{field} must contain only JSON-native values; got {type(value).__name__}"
    )


def _nonempty_string(value: Any, *, field: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{field} must be a nonempty string")
    return value


class CacheTrace:
    """Own one attempted cache trace with flat provenance records.

    The trace deliberately stores only JSON-native snapshots. It cannot retain
    tensors, exceptions, or arbitrary model objects by accident.
    """

    schema = "event-native-cache-trace-v1"

    def __init__(self, context: Mapping[str, Any] | None = None) -> None:
        if context is None:
            source: Mapping[str, Any] = {}
        elif isinstance(context, Mapping):
            source = context
        else:
            raise TypeError("context must be a mapping or None")

        unknown = set(source) - set(_CONTEXT_FIELDS)
        if unknown:
            names = ", ".join(sorted(str(name) for name in unknown))
            raise ValueError(f"context has unsupported fields: {names}")

        supplied: dict[str, str | None] = {}
        for field in _CONTEXT_FIELDS:
            value = source.get(field)
            if value is None:
                supplied[field] = None
            else:
                supplied[field] = _nonempty_string(value, field=f"context.{field}")

        attempt_uid = supplied["attempt_uid"] or str(uuid.uuid4())
        self.attempt_uid = attempt_uid
        self._op_sequence = 0
        self._entry_sequence = 0
        self._placement_sequence = 0
        self.data: dict[str, Any] = {
            "schema": self.schema,
            "attempt_uid": attempt_uid,
            "session_id": supplied["session_id"],
            "decision_key": supplied["decision_key"],
            "phase": supplied["phase"],
            "context_complete": all(supplied[field] is not None for field in _CONTEXT_FIELDS),
            "status": "started",
            "ops": [],
            "entries": [],
            "placements": [],
            "workspace_source_group": [],
            "commit_status": "not_requested",
        }

    def start_op(self, kind: str, **fields: Any) -> dict[str, Any]:
        """Append and return a started operation record."""

        _nonempty_string(kind, field="kind")
        values = self._fields_snapshot(fields, reserved=_OP_RESERVED_FIELDS, field="op")
        self._op_sequence += 1
        operation = {
            "op_id": f"{self.attempt_uid}:op:{self._op_sequence}",
            "kind": kind,
            "status": "started",
            **_OP_DEFAULTS,
            **values,
        }
        self.data["ops"].append(operation)
        return operation

    def finish_op(self, op: Mapping[str, Any], **fields: Any) -> dict[str, Any]:
        """Mark a locally-owned started operation as completed."""

        operation = self._owned_started_op(op)
        values = self._fields_snapshot(fields, reserved=_OP_RESERVED_FIELDS, field="op")
        operation.update(values)
        operation["status"] = "completed"
        return operation

    def fail_op(self, op: Mapping[str, Any], error: BaseException) -> dict[str, Any]:
        """Mark an operation failed while retaining only the exception type."""

        operation = self._owned_started_op(op)
        if not isinstance(error, BaseException):
            raise TypeError("error must be an exception instance")
        operation["status"] = "failed"
        operation["error_type"] = type(error).__name__
        return operation

    def new_entry(
        self,
        kind: str,
        op: Mapping[str, Any],
        *,
        parent: Mapping[str, Any] | None = None,
        origin_extraction_op_id: str | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        """Record an entry created by one operation without nesting ancestry."""

        _nonempty_string(kind, field="kind")
        operation = self._owned_op(op)
        values = self._fields_snapshot(fields, reserved=_ENTRY_RESERVED_FIELDS, field="entry")
        parent_entry_id: str | None = None
        inherited_origin: str | None = None
        if parent is not None:
            if not isinstance(parent, Mapping):
                raise TypeError("parent must be a mapping or None")
            parent_entry_id = _nonempty_string(
                parent.get("entry_id"), field="parent.entry_id"
            )
            parent_origin = parent.get("origin_extraction_op_id")
            if parent_origin is not None:
                inherited_origin = _nonempty_string(
                    parent_origin, field="parent.origin_extraction_op_id"
                )
        if origin_extraction_op_id is None:
            origin = inherited_origin
        else:
            origin = _nonempty_string(
                origin_extraction_op_id, field="origin_extraction_op_id"
            )

        self._entry_sequence += 1
        entry = {
            "entry_id": f"{self.attempt_uid}:entry:{self._entry_sequence}",
            "kind": kind,
            "created_by_op_id": operation["op_id"],
            "producer_attempt_uid": self.attempt_uid,
            "parent_entry_id": parent_entry_id,
            "origin_extraction_op_id": origin,
            **values,
        }
        self.data["entries"].append(entry)
        return entry

    def add_placement(self, **fields: Any) -> dict[str, Any]:
        """Append one JSON-safe logical access/placement record."""

        values = self._fields_snapshot(
            fields, reserved=_PLACEMENT_RESERVED_FIELDS, field="placement"
        )
        self._placement_sequence += 1
        placement = {
            "placement_id": f"{self.attempt_uid}:placement:{self._placement_sequence}",
            **values,
        }
        self.data["placements"].append(placement)
        return placement

    def _owned_op(self, op: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(op, Mapping):
            raise TypeError("op must be an operation mapping returned by start_op")
        op_id = op.get("op_id")
        if not isinstance(op_id, str):
            raise ValueError("op is missing op_id")
        for candidate in self.data["ops"]:
            if candidate is op or candidate.get("op_id") == op_id:
                return candidate
        raise ValueError("op does not belong to this trace")

    def _owned_started_op(self, op: Mapping[str, Any]) -> dict[str, Any]:
        operation = self._owned_op(op)
        if operation.get("status") != "started":
            raise ValueError("op is not started")
        return operation

    @staticmethod
    def _fields_snapshot(
        fields: Mapping[str, Any],
        *,
        reserved: set[str],
        field: str,
    ) -> dict[str, Any]:
        overlap = set(fields) & reserved
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(f"{field} fields cannot override: {names}")
        return {
            _nonempty_string(name, field=f"{field} field name"):
            _json_snapshot(value, field=f"{field}.{name}")
            for name, value in fields.items()
        }
