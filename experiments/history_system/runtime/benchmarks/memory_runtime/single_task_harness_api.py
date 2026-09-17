"""Bind an ordinary OpenAI harness to one explicit, isolated official task."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping

from .event_native_api import EventNativeAPI, EventNativeAPIError


SINGLE_TASK_SOURCE_PROFILE = "openai-single-task-v1"
BENCHMARKS = frozenset({"tau2", "toolsandbox", "acon_appworld", "acebench"})


class SingleTaskHarnessAPI(EventNativeAPI):
    """Preserve visible messages while supplying server-owned task identity.

    Each server belongs to exactly one official task. No task identity is
    guessed from prompt text, and no gold or executor state enters this API.
    Protocol special tokens and plain assistant code remain in the underlying
    native decoder. Text-only harnesses receive that original assistant text.
    """

    def __init__(self, *args, **kwargs):
        if kwargs.get("benchmark") not in BENCHMARKS:
            raise ValueError("Unsupported single-task harness namespace")
        if len(kwargs.get("allowed_task_ids", ())) != 1:
            raise ValueError("Ordinary harness transport requires exactly one frozen task")
        super().__init__(*args, **kwargs)
        self._wire_identities = {}
        self._wire_mode = None
        self._transport_receipts = {}

    def health(self):
        result = super().health()
        result["source_protocol_contract"] = {
            "profile": SINGLE_TASK_SOURCE_PROFILE,
            "task_identity": "one frozen official task per server",
            "history": "original role-preserving messages; no fabricated tool receipts",
            "sampling": "greedy with the configured finite completion cap",
            "harness_sampling_fields": (
                "recognized source defaults are validated, recorded, and normalized "
                "to the frozen server policy"
            ),
            "text_only_output": "original generated assistant content",
        }
        return result

    def handle_chat(self, payload):
        if not isinstance(payload, Mapping):
            raise EventNativeAPIError(400, "invalid_request", "Request body must be an object")
        known = {"messages", "model", "tools", "temperature", "max_tokens",
                 "max_completion_tokens", "stream", "seed", "store", "top_p",
                 "presence_penalty", "frequency_penalty", "chat_template_kwargs",
                 "c2kv_eval_context", "tool_choice", "parallel_tool_calls", "n"}
        unknown = set(payload) - known
        if unknown:
            raise EventNativeAPIError(400, "unknown_field", f"Unsupported harness fields: {sorted(unknown)!r}")
        if payload.get("stream", False) is not False:
            raise EventNativeAPIError(400, "stream_unsupported", "Streaming is disabled")
        if payload.get("n", 1) != 1:
            raise EventNativeAPIError(400, "multiple_candidates_unsupported", "Only one final answer is generated")
        if payload.get("tool_choice", "auto") not in (None, "auto"):
            raise EventNativeAPIError(400, "tool_choice_unsupported", "Forced tool selection is not configured")
        if payload.get("parallel_tool_calls", True) is not True:
            raise EventNativeAPIError(400, "parallel_calls_unsupported", "A single-call restriction requires its own profile")
        temperature = payload.get("temperature", 0)
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or float(temperature) != 0.0:
            raise EventNativeAPIError(400, "non_greedy", "temperature must be exactly zero")
        top_p = payload.get("top_p", 1)
        if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or float(top_p) != 1.0:
            raise EventNativeAPIError(400, "non_greedy", "top_p must be exactly one")
        if payload.get("frequency_penalty", 0) not in (0, 0.0):
            raise EventNativeAPIError(400, "penalty_unsupported", "frequency_penalty must be zero")
        presence = payload.get("presence_penalty", 0)
        if presence not in (0, 0.0) and not (
            self.benchmark == "acon_appworld" and presence in (0.5,)
        ):
            raise EventNativeAPIError(400, "penalty_unsupported", "presence_penalty is not a recognized harness default")
        template = payload.get("chat_template_kwargs")
        if template is not None and not (
            self.benchmark == "acon_appworld" and template == {"enable_thinking": False}
        ):
            raise EventNativeAPIError(400, "chat_template_unsupported", "chat_template_kwargs differs from the frozen source")
        client_context = payload.get("c2kv_eval_context")
        if client_context is not None and (
            self.benchmark != "acebench" or not isinstance(client_context, Mapping)
        ):
            raise EventNativeAPIError(400, "client_context_unsupported", "Task identity is server-owned")
        seed = payload.get("seed", 0)
        # tau2's orchestrator supplies a task seed through LLMConfig.set_seed.
        # Greedy generation stays bound to the server seed; retain the client value.
        recognized_seed = type(seed) is int and (
            (self.benchmark == "tau2" and seed >= 0)
            or seed in ((0, 42) if self.benchmark == "acon_appworld" else (0,))
        )
        if not recognized_seed:
            raise EventNativeAPIError(400, "seed_mismatch", "seed is not a recognized harness default")
        caps = [payload[name] for name in ("max_tokens", "max_completion_tokens")
                if payload.get(name) is not None]
        if any(type(cap) is not int or cap != self.max_new_tokens for cap in caps):
            raise EventNativeAPIError(400, "max_tokens_mismatch", "Harness completion cap differs from the frozen server")
        messages = copy.deepcopy(payload.get("messages"))
        if not isinstance(messages, list) or not messages:
            raise EventNativeAPIError(400, "invalid_messages", "messages must be a nonempty list")
        tools = copy.deepcopy(payload.get("tools") or [])
        mode = "tools" if tools else "text"
        if self._wire_mode is not None and mode != self._wire_mode:
            raise EventNativeAPIError(409, "task_protocol_changed", "Task changed between text and tool transport")
        signature = json.dumps({"messages": messages, "tools": tools},
                               ensure_ascii=False, sort_keys=True, allow_nan=False)
        identity = self._wire_identities.get(signature)
        if identity is None:
            user_turn = max(0, sum(message.get("role") == "user"
                                   for message in messages if isinstance(message, Mapping)) - 1)
            if (self.benchmark == "acon_appworld"
                    and hasattr(getattr(self.runner, "controller", None), "gp")):
                # AppWorld user-role messages after the task prompt are execution
                # observations, not new user requests. L=user_turn follows this
                # source contract while step still advances on every decision.
                user_turn = 0
            identity = {"benchmark": self.benchmark,
                        "task_id": next(iter(self.allowed_task_ids)),
                        "user_turn": user_turn, "step": len(self._wire_identities), "attempt": 0}
        normalized = {"messages": messages, "tools": tools,
                      "model": payload.get("model"),
                      "temperature": 0,
                      "store": payload.get("store", False),
                      "seed": 0,
                      "max_completion_tokens": self.max_new_tokens,
                      "c2kv_eval_context": identity}
        # Validate before reserving an identity or changing the transport mode.
        self._validate_request(normalized)
        self._wire_identities.setdefault(signature, identity)
        self._wire_mode = mode
        decision = (f"{self.benchmark}/{identity['task_id']}/attempt-0",
                    f"turn-{identity['user_turn']}/step-{identity['step']}")
        client_sampling = {
            key: copy.deepcopy(payload[key])
            for key in ("temperature", "top_p", "seed", "presence_penalty",
                        "frequency_penalty", "max_tokens", "max_completion_tokens",
                        "chat_template_kwargs")
            if key in payload
        }
        self._transport_receipts[decision] = {
            "schema": "openai-single-task-normalization-v1",
            "client_sampling_fields": client_sampling,
            "normalized_server_fields": {
                "temperature": 0, "seed": 0,
                "max_completion_tokens": self.max_new_tokens,
            },
            "normalized_away_fields": sorted(
                key for key in client_sampling
                if key not in ("temperature", "max_tokens", "max_completion_tokens")
            ),
            "client_context_ignored": client_context is not None,
            "task_identity_source": "server",
        }
        try:
            return super().handle_chat(normalized)
        finally:
            # Completed and failed records are already durable at this point.
            self._transport_receipts.pop(decision, None)

    def _append_step(self, record):
        snapshot = copy.deepcopy(record)
        decision = (snapshot.get("session_id"), snapshot.get("decision_key"))
        receipt = self._transport_receipts.get(decision)
        if receipt is not None:
            snapshot["harness_transport_normalization"] = copy.deepcopy(receipt)
        return super()._append_step(snapshot)

    def _openai_response(self, record):
        if self._wire_mode != "text":
            return super()._openai_response(record)
        snapshot = copy.deepcopy(record)
        if isinstance(snapshot.get("response"), dict):
            # The generator's raw text is the action for code/text harnesses.
            trace = snapshot.get("generation_trace", [])
            draft = trace[-1].get("native_draft", {}) if trace else {}
            if isinstance(draft.get("text"), str):
                snapshot["response"]["content"] = draft["text"]
            snapshot["response"]["tool_calls"] = []
        return super()._openai_response(snapshot)
