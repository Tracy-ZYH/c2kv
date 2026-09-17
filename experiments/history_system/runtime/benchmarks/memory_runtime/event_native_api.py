"""Finite loopback OpenAI transport for the event-native decision runner."""
from __future__ import annotations

import copy
import ipaddress
import json
import math
import os
import socket
import time
import uuid
from collections.abc import Mapping, Sequence
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .event_native import EventStore
from .event_native_controls import describe_event_native_route


_REQUEST_FIELDS = frozenset({
    "messages",
    "model",
    "temperature",
    "store",
    "max_completion_tokens",
    "tools",
    "seed",
    "stream",
    "c2kv_eval_context",
})
_CONTEXT_FIELDS = frozenset({"benchmark", "task_id", "user_turn", "step", "attempt"})
_MESSAGE_FIELDS = frozenset({
    "role", "content", "name", "tool_call_id", "tool_calls", "reasoning_content"
})
_PRIVILEGED_MARKERS = ("gold", "oracle", "target", "answer", "label")


class EventNativeAPIError(ValueError):
    """A request error with an explicit HTTP status and stable code."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class EventNativeAPI:
    """Expose one finite benchmark run without leaking draft traces."""

    request_fields = _REQUEST_FIELDS

    def __init__(
        self,
        runner: Any,
        *,
        run_id: str,
        model_name: str,
        view_mode: str,
        max_new_tokens: int,
        allowed_task_ids: Sequence[str] | set[str] | frozenset[str],
        max_decisions: int,
        deadline_monotonic: float,
        steps_path: str | Path,
        runtime_policy_contract: Mapping[str, Any] | None = None,
        benchmark: str = "bfcl",
        compression_policy: str | None = None,
        history_view_protocol: str = "fixed-budget-main",
    ) -> None:
        if not callable(getattr(runner, "run", None)):
            raise TypeError("runner must expose run(payload)")
        for name, value in (
            ("run_id", run_id),
            ("model_name", model_name),
            ("view_mode", view_mode),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty string")
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        if benchmark not in ("bfcl", "acebench", "tau2", "toolsandbox", "acon_appworld"):
            raise ValueError("Unsupported frozen benchmark namespace")
        if isinstance(allowed_task_ids, (str, bytes, bytearray)):
            raise TypeError("allowed_task_ids must be a finite collection of strings")
        allowed = frozenset(allowed_task_ids)
        if not allowed or any(not isinstance(value, str) or not value for value in allowed):
            raise ValueError("allowed_task_ids must contain nonempty strings")
        if type(max_decisions) is not int or max_decisions <= 0:
            raise ValueError("max_decisions must be a positive integer")
        if (
            isinstance(deadline_monotonic, bool)
            or not isinstance(deadline_monotonic, (int, float))
            or not math.isfinite(float(deadline_monotonic))
        ):
            raise ValueError("deadline_monotonic must be finite")
        path = Path(steps_path)
        if not path.name:
            raise ValueError("steps_path must include a file name")

        self.runner = runner
        self.run_id = run_id
        self.model_name = model_name
        self.benchmark = benchmark
        self.view_mode = view_mode
        self.route_contract = describe_event_native_route(
            view_mode,
            compression_policy=compression_policy,
            history_view_protocol=history_view_protocol,
        )
        if runtime_policy_contract is not None and not isinstance(runtime_policy_contract, Mapping):
            raise TypeError('runtime_policy_contract must be a mapping or None')
        self.runtime_policy_contract = (
            copy.deepcopy(dict(runtime_policy_contract)) if runtime_policy_contract is not None else None
        )
        self.max_new_tokens = max_new_tokens
        self.allowed_task_ids = allowed
        self.max_decisions = max_decisions
        self.deadline_monotonic = float(deadline_monotonic)
        self.steps_path = path
        self.decisions_reserved = 0
        self._completed: dict[
            tuple[str, int, int], tuple[str, dict[str, Any]]
        ] = {}
        self._terminal_failure: dict[str, str] | None = None

    def health(self) -> dict[str, Any]:
        """Return only bounded operational state, without prompts or errors."""

        deadline_exceeded = time.monotonic() >= self.deadline_monotonic
        decision_cap_reached = self.decisions_reserved >= self.max_decisions
        terminal = (
            self._terminal_failure is not None
            or deadline_exceeded
            or decision_cap_reached
        )
        if self._terminal_failure is not None:
            terminal_reason = self._terminal_failure["code"]
        elif deadline_exceeded:
            terminal_reason = "deadline_exceeded"
        elif decision_cap_reached:
            terminal_reason = "decision_cap_reached"
        else:
            terminal_reason = None
        generation_calls = getattr(self.runner, "generation_calls", 0)
        if type(generation_calls) is not int or generation_calls < 0:
            generation_calls = 0
        max_generation_calls = getattr(self.runner, "max_generation_calls", None)
        if type(max_generation_calls) is not int or max_generation_calls <= 0:
            max_generation_calls = None
        return {
            "schema": "a-event-native-api-health-v1",
            "status": "terminal" if terminal else "ok",
            "terminal": terminal,
            "terminal_reason": terminal_reason,
            "accepting_new_decisions": not terminal,
            "run_id": self.run_id,
            "model_name": self.model_name,
            "benchmark": self.benchmark,
            "view_mode": self.view_mode,
            "route_contract": copy.deepcopy(self.route_contract),
            "runtime_policy_contract": copy.deepcopy(self.runtime_policy_contract),
            "decode_strategy": getattr(getattr(self.runner, "generator", None), "decode_strategy", None),
            "session_cache_policy": getattr(getattr(self.runner, "generator", None), "session_cache_policy", None),
            "max_new_tokens": self.max_new_tokens,
            "allowed_task_ids": sorted(self.allowed_task_ids),
            "decisions_reserved": self.decisions_reserved,
            "max_decisions": self.max_decisions,
            "generation_calls_reserved": generation_calls,
            "max_generation_calls": max_generation_calls,
            "deadline_exceeded": deadline_exceeded,
        }

    def handle_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Validate one identified request and return only its final response."""

        runner_payload, identity, signature = self._validate_request(payload)
        if self._terminal_failure is not None:
            raise EventNativeAPIError(
                503,
                "terminal_failure",
                "Event-native generation stopped after a terminal runner failure",
            )

        cached = self._completed.get(identity)
        if cached is not None:
            if cached[0] != signature:
                raise EventNativeAPIError(
                    409,
                    "decision_conflict",
                    "Decision identity was reused with different visible input",
                )
            return copy.deepcopy(cached[1])

        if time.monotonic() >= self.deadline_monotonic:
            raise EventNativeAPIError(
                408,
                "deadline_exceeded",
                "The finite event-native deadline has expired",
            )
        if self.decisions_reserved >= self.max_decisions:
            raise EventNativeAPIError(
                429,
                "decision_cap_reached",
                "The finite event-native decision cap is exhausted",
            )

        self.decisions_reserved += 1
        try:
            record = self.runner.run(copy.deepcopy(runner_payload))
        except Exception as error:
            record = getattr(error, "record", None)
            if not isinstance(record, Mapping):
                record = self._failure_record(runner_payload, error)
            self._terminal_failure = {
                "code": "runner_failed",
                "type": type(error).__name__,
            }
            self._release_runner_cache()
            try:
                self._append_step(record)
            except Exception as journal_error:
                self._terminal_failure = {
                    "code": "steps_write_failed",
                    "type": type(journal_error).__name__,
                }
                raise EventNativeAPIError(
                    500,
                    "steps_write_failed",
                    "Failed to durably record the terminal runner trace",
                ) from journal_error
            raise EventNativeAPIError(
                500,
                "runner_failed",
                "Event-native decision generation failed terminally",
            ) from error

        if not isinstance(record, Mapping):
            error = TypeError("runner returned a non-object record")
            failure_record = self._failure_record(runner_payload, error)
            self._terminal_failure = {
                "code": "invalid_runner_record",
                "type": type(record).__name__,
            }
            self._release_runner_cache()
            try:
                self._append_step(failure_record)
            except Exception as journal_error:
                self._terminal_failure = {
                    "code": "steps_write_failed",
                    "type": type(journal_error).__name__,
                }
                raise EventNativeAPIError(
                    500,
                    "steps_write_failed",
                    "Failed to durably record the invalid runner return",
                ) from journal_error
            raise EventNativeAPIError(
                500,
                "invalid_runner_record",
                "Runner returned a non-object record",
            )

        try:
            self._append_step(record)
        except Exception as error:
            self._terminal_failure = {
                "code": "steps_write_failed",
                "type": type(error).__name__,
            }
            self._release_runner_cache()
            raise EventNativeAPIError(
                500,
                "steps_write_failed",
                "Failed to durably record the completed runner trace",
            ) from error

        try:
            response = self._openai_response(record)
        except Exception as error:
            self._terminal_failure = {
                "code": "invalid_runner_record",
                "type": type(error).__name__,
            }
            self._release_runner_cache()
            raise EventNativeAPIError(
                500,
                "invalid_runner_record",
                "Runner record cannot be converted to an OpenAI response",
            ) from error
        self._completed[identity] = (signature, copy.deepcopy(response))
        return response

    def _release_runner_cache(self) -> None:
        close = getattr(self.runner, "close", None)
        if callable(close):
            close()

    def _validate_request(
        self, payload: Any
    ) -> tuple[dict[str, Any], tuple[str, int, int], str]:
        if not isinstance(payload, Mapping):
            raise EventNativeAPIError(400, "invalid_request", "Request body must be an object")
        unknown = sorted(set(payload) - self.request_fields)
        if unknown:
            privileged = any(
                any(marker in str(field).lower() for marker in _PRIVILEGED_MARKERS)
                for field in unknown
            )
            raise EventNativeAPIError(
                403 if privileged else 400,
                "privileged_field" if privileged else "unknown_field",
                f"Unsupported request fields: {unknown!r}",
            )
        if "stream" in payload:
            raise EventNativeAPIError(400, "stream_unsupported", "Streaming is disabled")
        if payload.get("model") != self.model_name:
            raise EventNativeAPIError(400, "model_mismatch", "Request model does not match")
        temperature = payload.get("temperature")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature))
            or float(temperature) != 0.0
        ):
            raise EventNativeAPIError(400, "non_greedy", "temperature must be exactly zero")
        if payload.get("store") is not False:
            raise EventNativeAPIError(400, "store_unsupported", "store must be false")
        if (
            type(payload.get("max_completion_tokens")) is not int
            or payload.get("max_completion_tokens") != self.max_new_tokens
        ):
            raise EventNativeAPIError(
                400,
                "max_tokens_mismatch",
                "max_completion_tokens must match the configured finite cap",
            )
        if "seed" in payload and (
            type(payload["seed"]) is not int or payload["seed"] != 0
        ):
            raise EventNativeAPIError(400, "seed_mismatch", "seed must be zero when present")

        context = payload.get("c2kv_eval_context")
        if not isinstance(context, Mapping) or set(context) != _CONTEXT_FIELDS:
            raise EventNativeAPIError(
                400,
                "invalid_eval_context",
                "c2kv_eval_context must contain exactly the benchmark identity fields",
            )
        if context.get("benchmark") != self.benchmark:
            raise EventNativeAPIError(
                400, "invalid_benchmark", "benchmark does not match the frozen server namespace"
            )
        task_id = context.get("task_id")
        if not isinstance(task_id, str) or task_id not in self.allowed_task_ids:
            raise EventNativeAPIError(403, "unknown_task", "task_id is not allowed")
        user_turn = context.get("user_turn")
        step = context.get("step")
        if type(user_turn) is not int or user_turn < 0:
            raise EventNativeAPIError(400, "invalid_user_turn", "user_turn must be nonnegative")
        if type(step) is not int or step < 0:
            raise EventNativeAPIError(400, "invalid_step", "step must be nonnegative")
        if type(context.get("attempt")) is not int or context.get("attempt") != 0:
            raise EventNativeAPIError(403, "attempt_forbidden", "Only attempt zero is allowed")

        messages = payload.get("messages")
        if (
            not isinstance(messages, Sequence)
            or isinstance(messages, (str, bytes, bytearray))
            or not messages
        ):
            raise EventNativeAPIError(400, "invalid_messages", "messages must be nonempty")
        message_snapshot: list[dict[str, Any]] = []
        for index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                raise EventNativeAPIError(
                    400, "invalid_messages", f"messages[{index}] must be an object"
                )
            call_id = message.get("tool_call_id")
            if (isinstance(call_id, str) and call_id.startswith("acebench-execution-")
                    and self.benchmark != "acebench"):
                raise EventNativeAPIError(400, "invalid_messages", "ACEBench text execution IDs require the ACEBench source protocol")
            message_unknown = sorted(set(message) - _MESSAGE_FIELDS)
            if message_unknown:
                raise EventNativeAPIError(
                    400,
                    "unknown_message_field",
                    f"messages[{index}] has unsupported fields: {message_unknown!r}",
                )
            if message.get("role") == "developer":
                raise EventNativeAPIError(
                    400,
                    "developer_role_unsupported",
                    "developer messages require an explicit source adapter",
                )
            content = message.get("content")
            if content is not None and not isinstance(content, str):
                raise EventNativeAPIError(
                    400, "invalid_messages", f"messages[{index}].content must be text or null"
                )
            reasoning = message.get("reasoning_content")
            if reasoning is not None and not isinstance(reasoning, str):
                raise EventNativeAPIError(
                    400,
                    "invalid_messages",
                    f"messages[{index}].reasoning_content must be text or null",
                )
            message_snapshot.append(copy.deepcopy(dict(message)))
        source_fields = self._validate_source(payload, message_snapshot)

        tools = payload.get("tools", [])
        if tools is None:
            tools = []
        if (
            not isinstance(tools, Sequence)
            or isinstance(tools, (str, bytes, bytearray))
            or any(not isinstance(tool, Mapping) for tool in tools)
        ):
            raise EventNativeAPIError(400, "invalid_tools", "tools must be a list of objects")
        tool_snapshot = [copy.deepcopy(dict(tool)) for tool in tools]

        # Event IDs are rendered into exact evidence, so their session prefix
        # must be stable across independently hosted arm/run processes.  The
        # server-owned runner and cache still provide arm/run state isolation.
        session_id = f"{self.benchmark}/{task_id}/attempt-0"
        decision_key = f"turn-{user_turn}/step-{step}"
        runner_payload = {
            "session_id": session_id,
            "decision_key": decision_key,
            "messages": message_snapshot,
            "tools": tool_snapshot,
            **source_fields,
        }
        try:
            signature = json.dumps(
                runner_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as error:
            raise EventNativeAPIError(
                400, "non_json_request", "Visible request fields must be finite JSON"
            ) from error
        return runner_payload, (task_id, user_turn, step), signature

    def _validate_source(self, payload, messages):
        """Validate the default native source; adapters opt in by subclassing."""
        try:
            EventStore.from_messages("request-validation", messages)
        except (TypeError, ValueError) as error:
            raise EventNativeAPIError(400, "invalid_messages", str(error)) from error
        return {}

    def _openai_response(self, record: Mapping[str, Any]) -> dict[str, Any]:
        if record.get("status") != "ok":
            raise ValueError("runner record is not successful")
        final = record.get("response")
        if not isinstance(final, Mapping) or final.get("role") != "assistant":
            raise ValueError("runner record has no final assistant response")
        content = final.get("content")
        if content is not None and not isinstance(content, str):
            raise ValueError("runner response content is invalid")
        tool_calls = final.get("tool_calls", [])
        if not isinstance(tool_calls, list) or any(not isinstance(call, Mapping) for call in tool_calls):
            raise ValueError("runner response tool_calls are invalid")
        finish_reason = "tool_calls" if tool_calls else final.get("finish_reason")
        if finish_reason not in {"stop", "length", "tool_calls"}:
            raise ValueError("runner response finish_reason is invalid")
        usage = record.get("generation_usage_total")
        if not isinstance(usage, Mapping):
            raise ValueError("runner record has no total usage")
        usage_snapshot: dict[str, int] = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if type(value) is not int or value < 0:
                raise ValueError(f"runner usage {key} is unavailable")
            usage_snapshot[key] = value
        message = {
            "role": "assistant",
            "content": content,
            "tool_calls": copy.deepcopy(tool_calls) if tool_calls else None,
            "reasoning_content": final.get("reasoning_content"),
        }
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }],
            "usage": usage_snapshot,
        }

    def _append_step(self, record: Mapping[str, Any]) -> None:
        encoded = (
            json.dumps(record, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
        self.steps_path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_BINARY", 0)
        descriptor = os.open(self.steps_path, flags, 0o600)
        try:
            remaining = memoryview(encoded)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("steps append made no progress")
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _failure_record(runner_payload: Mapping[str, Any], error: Exception) -> dict[str, Any]:
        return {
            "schema": "a-event-native-api-runner-failure-v1",
            "status": "failed",
            "session_id": runner_payload["session_id"],
            "decision_key": runner_payload["decision_key"],
            "generation_trace": [],
            "response": None,
            "generation_usage_total": {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            },
            "error": {"type": type(error).__name__, "message": str(error)},
        }


def _is_loopback(host: str) -> bool:
    if not isinstance(host, str) or not host:
        return False
    try:
        addresses = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        return bool(addresses) and all(
            ipaddress.ip_address(address[4][0]).is_loopback for address in addresses
        )
    except (OSError, ValueError):
        return False


def make_server(
    api: EventNativeAPI,
    host: str = "127.0.0.1",
    port: int = 0,
) -> HTTPServer:
    """Construct a serial loopback HTTP server; the caller owns its lifecycle."""

    if not isinstance(api, EventNativeAPI):
        raise TypeError("api must be an EventNativeAPI")
    if not _is_loopback(host):
        raise ValueError("event-native API host must resolve only to loopback addresses")
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("port must be an integer in [0, 65535]")

    class Handler(BaseHTTPRequestHandler):
        server_version = "EventNativeAPI/1"

        def do_GET(self) -> None:
            if urlsplit(self.path).path != "/health":
                self._send_json(404, {"error": {"code": "not_found", "message": "Not found"}})
                return
            self._send_json(200, api.health())

        def do_POST(self) -> None:
            if urlsplit(self.path).path != "/v1/chat/completions":
                self._send_json(404, {"error": {"code": "not_found", "message": "Not found"}})
                return
            content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type != "application/json":
                self._send_json(
                    415,
                    {"error": {"code": "invalid_content_type", "message": "Use application/json"}},
                )
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
                if length < 0:
                    raise ValueError
            except ValueError:
                self._send_json(
                    411,
                    {"error": {"code": "invalid_content_length", "message": "Content-Length required"}},
                )
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send_json(
                    400,
                    {"error": {"code": "invalid_json", "message": "Body is not valid JSON"}},
                )
                return
            try:
                response = api.handle_chat(payload)
            except EventNativeAPIError as error:
                self._send_json(
                    error.status_code,
                    {"error": {
                        "message": str(error),
                        "type": "invalid_request_error" if error.status_code < 500 else "server_error",
                        "code": error.code,
                    }},
                )
                return
            self._send_json(200, response)

        def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
            body = json.dumps(
                payload, ensure_ascii=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *args: Any) -> None:
            return

    class LoopbackServer(HTTPServer):
        address_family = socket.AF_INET6 if ':' in host else socket.AF_INET

    server = LoopbackServer((host, port), Handler)
    server.event_native_api = api  # type: ignore[attr-defined]
    return server


__all__ = ["EventNativeAPI", "EventNativeAPIError", "make_server"]
