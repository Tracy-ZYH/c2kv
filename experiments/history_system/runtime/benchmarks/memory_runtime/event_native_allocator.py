"""Per-generation NPU caching-allocator peak measurements."""

from __future__ import annotations

import threading
from collections.abc import Mapping, MutableMapping
from typing import Any


MEASUREMENT_SCHEMA = "c2kv-npu-allocator-measurement-v1"
MEASUREMENT_BACKEND = "torch_npu.npu"
MEASUREMENT_WINDOW = (
    "generator.generate only; excludes model load, controller and decision_scope exit"
)
MEASUREMENT_CONTRACT: dict[str, Any] = {
    "schema": MEASUREMENT_SCHEMA,
    "backend": MEASUREMENT_BACKEND,
    "window": MEASUREMENT_WINDOW,
    "counter_scope": "this process's NPU caching allocator only",
    "peak_semantics": (
        "absolute allocator peak; includes resident model weights and cache allocations"
    ),
    "execution_requirement": (
        "one serial generator with no competing allocator peak-stat resets"
    ),
}


def _default_npu() -> Any:
    import torch_npu

    return torch_npu.npu


def _attempt_uid(kwargs: Mapping[str, Any]) -> str:
    trace_context = kwargs.get("trace_context")
    if not isinstance(trace_context, Mapping):
        raise ValueError("trace_context with a nonempty attempt_uid is required")
    attempt_uid = trace_context.get("attempt_uid")
    if not isinstance(attempt_uid, str) or not attempt_uid:
        raise ValueError("trace_context.attempt_uid must be a nonempty string")
    return attempt_uid


def _generator_device(generator: Any) -> Any:
    try:
        device = generator.runtime.base_model.model.embed_tokens.weight.device
    except AttributeError as error:
        raise TypeError(
            "generator must expose runtime.base_model.model.embed_tokens.weight.device"
        ) from error
    device_type = getattr(device, "type", None)
    if device_type is None and isinstance(device, str):
        device_type = device.partition(":")[0]
    if device_type != "npu":
        raise ValueError(f"allocator measurement requires an NPU model device, got {device!s}")
    return device


def _nonnegative_int(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _result_stats(result: Any) -> MutableMapping[str, Any]:
    stats = getattr(result, "stats", None)
    if not isinstance(stats, MutableMapping):
        raise TypeError("generator result.stats must be a mutable mapping")
    return stats


class NpuAllocatorMeasuredGenerator:
    """Add one process-local NPU allocator measurement to each generation."""

    measurement_contract = MEASUREMENT_CONTRACT

    def __init__(self, generator: Any, *, npu: Any = None) -> None:
        self.generator = generator
        self._npu = _default_npu() if npu is None else npu
        self._generate_lock = threading.Lock()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.generator, name)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        attempt_uid = _attempt_uid(kwargs)
        device = _generator_device(self.generator)
        if not self._generate_lock.acquire(blocking=False):
            raise RuntimeError(
                "NPU allocator measurement does not permit reentrant or concurrent generate"
            )
        try:
            values = {
                "allocated_bytes_before": None,
                "reserved_bytes_before": None,
                "allocated_bytes_after": None,
                "reserved_bytes_after": None,
                "peak_allocated_bytes": None,
                "peak_reserved_bytes": None,
            }
            start_error: Exception | None = None
            try:
                self._npu.synchronize(device)
                self._npu.reset_peak_memory_stats(device)
                values["allocated_bytes_before"] = _nonnegative_int(
                    self._npu.memory_allocated(device), name="memory_allocated before"
                )
                values["reserved_bytes_before"] = _nonnegative_int(
                    self._npu.memory_reserved(device), name="memory_reserved before"
                )
            except Exception as error:
                start_error = error

            # Do not catch model failures: the generator owns its exception and trace.
            result = self.generator.generate(*args, **kwargs)

            if start_error is not None:
                self._record_failure(
                    result,
                    device=device,
                    attempt_uid=attempt_uid,
                    values=values,
                    error=start_error,
                )
                return result

            try:
                self._npu.synchronize(device)
                values["allocated_bytes_after"] = _nonnegative_int(
                    self._npu.memory_allocated(device), name="memory_allocated after"
                )
                values["reserved_bytes_after"] = _nonnegative_int(
                    self._npu.memory_reserved(device), name="memory_reserved after"
                )
                values["peak_allocated_bytes"] = _nonnegative_int(
                    self._npu.max_memory_allocated(device), name="max_memory_allocated"
                )
                values["peak_reserved_bytes"] = _nonnegative_int(
                    self._npu.max_memory_reserved(device), name="max_memory_reserved"
                )
            except Exception as error:
                values["peak_allocated_bytes"] = None
                values["peak_reserved_bytes"] = None
                self._record_failure(
                    result,
                    device=device,
                    attempt_uid=attempt_uid,
                    values=values,
                    error=error,
                )
                return result

            stats = _result_stats(result)
            stats["torch_allocator_peak_allocated_bytes"] = values[
                "peak_allocated_bytes"
            ]
            stats["allocator_measurement"] = self._measurement(
                device=device,
                attempt_uid=attempt_uid,
                values=values,
                status="ok",
            )
            return result
        finally:
            self._generate_lock.release()

    @staticmethod
    def _measurement(
        *,
        device: Any,
        attempt_uid: str,
        values: Mapping[str, int | None],
        status: str,
    ) -> dict[str, Any]:
        return {
            "schema": MEASUREMENT_SCHEMA,
            "backend": MEASUREMENT_BACKEND,
            "device": str(device),
            "attempt_uid": attempt_uid,
            "window": MEASUREMENT_WINDOW,
            **values,
            "status": status,
        }

    @classmethod
    def _record_failure(
        cls,
        result: Any,
        *,
        device: Any,
        attempt_uid: str,
        values: Mapping[str, int | None],
        error: Exception,
    ) -> None:
        stats = _result_stats(result)
        stats["torch_allocator_peak_allocated_bytes"] = None
        measurement = cls._measurement(
            device=device,
            attempt_uid=attempt_uid,
            values=values,
            status="failed",
        )
        measurement["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        stats["allocator_measurement"] = measurement


__all__ = ["MEASUREMENT_CONTRACT", "NpuAllocatorMeasuredGenerator"]
