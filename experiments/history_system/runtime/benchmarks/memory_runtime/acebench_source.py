"""ACEBench textual actions and receipt-backed event extraction.

The official harness executes Python-like action text and returns one aggregate
execution observation.  This adapter parses that dialect only for recovery
queries and event boundaries; it never turns historical text into executable
OpenAI tool calls.
"""

from __future__ import annotations

import ast
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from history_memory.events import EventRecord, EventStore, Message

from .event_native_draft import NativeDraft


ACE_SOURCE_VERSION = "acebench-text-actions-v1"
ACE_RECEIPT_VERSION = "acebench-execution-receipt-v1"
ACE_OPAQUE_EVENT_KIND = "acebench_execution_opaque"

_RECEIPT_FIELDS = frozenset(
    {
        "execution_message_index",
        "version",
        "agent_history_index",
        "decode_status",
        "decoded_calls",
        "executor_status",
        "executor_return_shape",
        "executor_return_count",
    }
)
_CALL_LIKE = re.compile(r"\b[A-Za-z_]\w*\s*\(")
_MESSAGE_FIELDS = frozenset({"role", "content", "tool_call_id"})


class _UnsupportedAceGrammar(ValueError):
    pass


def _unsupported(code: str) -> _UnsupportedAceGrammar:
    return _UnsupportedAceGrammar(f"unsupported_ace_grammar:{code}")


def _decode_value(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        value = node.value
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise _unsupported("non_finite_number")
            return value
        if isinstance(value, bytes):
            raise _unsupported("bytes_value")
        if value is Ellipsis:
            raise _unsupported("ellipsis_value")
        raise _unsupported("constant_value")
    if isinstance(node, ast.List):
        return [_decode_value(item) for item in node.elts]
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, ast.USub):
            raise _unsupported("unary_operator")
        operand = node.operand
        if (
            not isinstance(operand, ast.Constant)
            or isinstance(operand.value, bool)
            or not isinstance(operand.value, (int, float))
        ):
            raise _unsupported("unary_operand")
        value = -operand.value
        if isinstance(value, float) and not math.isfinite(value):
            raise _unsupported("non_finite_number")
        return value
    if isinstance(node, ast.Dict):
        raise _unsupported("dict_value")
    if isinstance(node, ast.Tuple):
        raise _unsupported("tuple_value")
    if isinstance(node, ast.Name):
        raise _unsupported("name_value")
    if isinstance(node, ast.Call):
        raise _unsupported("nested_call")
    if isinstance(node, ast.BinOp):
        raise _unsupported("binary_operator")
    if isinstance(node, ast.Lambda):
        raise _unsupported("lambda_value")
    if isinstance(node, ast.Subscript):
        raise _unsupported("subscript_value")
    if isinstance(node, ast.Set):
        raise _unsupported("set_value")
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        raise _unsupported("comprehension_value")
    raise _unsupported(f"{type(node).__name__.lower()}_value")


def _parse_call(node: ast.AST, *, call_id: str) -> dict[str, Any]:
    if not isinstance(node, ast.Call):
        raise _unsupported("list_item_not_call")
    if not isinstance(node.func, ast.Name):
        raise _unsupported("callee_not_name")
    if any(isinstance(argument, ast.Starred) for argument in node.args):
        raise _unsupported("starred_argument")
    if node.args:
        raise _unsupported("positional_argument")
    arguments: dict[str, Any] = {}
    for keyword in node.keywords:
        if keyword.arg is None:
            raise _unsupported("keyword_unpack")
        if keyword.arg in arguments:
            raise _unsupported("duplicate_keyword")
        arguments[keyword.arg] = _decode_value(keyword.value)
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": node.func.id,
            "arguments": json.dumps(
                arguments,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ),
        },
    }


def _looks_like_broken_action(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith("[") or bool(_CALL_LIKE.search(text))


def parse_acebench_draft(text: str, *, call_id_prefix: str) -> NativeDraft:
    """Parse one complete ACEBench action list without evaluating Python.

    Only ``[Name(keyword=value), ...]`` is admitted.  Values are recursively
    limited to JSON-compatible primitive/list literals.  Any unsupported AST
    form fails the entire draft, so no partial executable-looking calls escape.
    """

    if not isinstance(text, str):
        raise TypeError("ACEBench draft must be decoded text")
    if not isinstance(call_id_prefix, str) or not call_id_prefix:
        raise ValueError("ACEBench draft requires an explicit call ID prefix")
    stripped = text.strip()
    if not stripped or stripped.casefold() == "finish conversation":
        return NativeDraft(
            text=text,
            content=text,
            tool_calls=(),
            status="text",
            reason="no_ace_actions",
            reasoning_content=None,
        )
    # The shared parser intentionally accepts only the common subset of the
    # official step/turn routers.  In particular, STEP uses a start-anchored
    # bracket regex without whitespace normalization or DOTALL.
    if not text.startswith("["):
        return NativeDraft(text, text, (), "text", "no_ace_actions", None)
    first_line = text.splitlines()[0]
    if "]" not in first_line:
        return NativeDraft(text, text, (), "malformed", "invalid_ace_syntax", None)
    try:
        expression = ast.parse(text, mode="eval").body
    except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
        if _looks_like_broken_action(text):
            return NativeDraft(
                text, text, (), "malformed", "invalid_ace_syntax", None
            )
        return NativeDraft(text, text, (), "text", "no_ace_actions", None)

    try:
        if not isinstance(expression, ast.List):
            raise _unsupported("root_not_list")
        if not expression.elts:
            raise _unsupported("empty_call_list")
        calls = tuple(
            _parse_call(item, call_id=f"{call_id_prefix}_{index}")
            for index, item in enumerate(expression.elts)
        )
    except (_UnsupportedAceGrammar, OverflowError, RecursionError) as error:
        reason = str(error)
        if not reason.startswith("unsupported_ace_grammar:"):
            reason = "unsupported_ace_grammar:value_overflow"
        return NativeDraft(text, text, (), "malformed", reason, None)
    return NativeDraft(
        text=text,
        content=text,
        tool_calls=calls,
        status="tool_calls",
        reason="complete_ace_actions",
        reasoning_content=None,
    )


# The runtime uses the shorter name; retain the descriptive public name for
# direct source tests and callers.
parse_ace_draft = parse_acebench_draft


def _nonnegative_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _validated_receipts(
    messages: Sequence[Mapping[str, Any]], source: Any
) -> dict[int, dict[str, Any]]:
    if not isinstance(source, Mapping) or set(source) != {"version", "receipts"}:
        raise ValueError(
            "c2kv_ace_source must contain exactly version and receipts"
        )
    if source.get("version") != ACE_SOURCE_VERSION:
        raise ValueError("ACEBench source version mismatch")
    receipts = source.get("receipts")
    if not isinstance(receipts, list):
        raise ValueError("ACEBench source receipts must be a list")

    by_execution: dict[int, dict[str, Any]] = {}
    for position, raw in enumerate(receipts):
        if not isinstance(raw, Mapping) or set(raw) != _RECEIPT_FIELDS:
            raise ValueError(
                f"ACEBench receipt {position} must contain exactly the receipt fields"
            )
        receipt = dict(raw)
        execution_index = receipt["execution_message_index"]
        agent_history_index = receipt["agent_history_index"]
        if not _nonnegative_int(execution_index):
            raise ValueError("ACEBench execution_message_index must be an integer")
        if execution_index in by_execution:
            raise ValueError(
                f"Duplicate ACEBench receipt for source index {execution_index}"
            )
        if execution_index >= len(messages):
            raise ValueError("ACEBench receipt points outside the observable prefix")
        if messages[execution_index].get("role") != "tool":
            raise ValueError("ACEBench receipt must target a tool source row")
        assistant_index = execution_index - 1
        if assistant_index < 0 or messages[assistant_index].get("role") != "assistant":
            raise ValueError(
                "ACEBench execution observation must immediately follow an assistant row"
            )
        if (
            not _nonnegative_int(agent_history_index)
            or agent_history_index + 1 != assistant_index
        ):
            raise ValueError("ACEBench receipt agent_history_index mismatch")
        if receipt["version"] != ACE_RECEIPT_VERSION:
            raise ValueError("ACEBench execution receipt version mismatch")
        if receipt["decode_status"] not in {"ok", "error"}:
            raise ValueError("ACEBench receipt decode_status is invalid")
        decoded_calls = receipt["decoded_calls"]
        if decoded_calls is not None and (
            not isinstance(decoded_calls, list)
            or any(not isinstance(call, str) for call in decoded_calls)
        ):
            raise ValueError("ACEBench decoded_calls must be a list of strings or null")
        if receipt["executor_status"] not in {"not_called", "returned"}:
            raise ValueError("ACEBench receipt executor_status is invalid")
        if receipt["executor_return_shape"] not in {None, "list", "non_list"}:
            raise ValueError("ACEBench receipt executor_return_shape is invalid")
        count = receipt["executor_return_count"]
        if count is not None and not _nonnegative_int(count):
            raise ValueError(
                "ACEBench executor_return_count must be a nonnegative integer or null"
            )
        if receipt["executor_status"] == "not_called" and (
            receipt["executor_return_shape"] is not None or count is not None
        ):
            raise ValueError("A not-called ACEBench executor cannot have a return")
        if receipt["executor_status"] == "returned":
            if receipt["executor_return_shape"] not in {"list", "non_list"}:
                raise ValueError("A returned ACEBench executor requires a return shape")
            if (receipt["executor_return_shape"] == "list") != (count is not None):
                raise ValueError(
                    "ACEBench list return shape and return count are inconsistent"
                )
        by_execution[execution_index] = receipt

    tool_indices = {
        index for index, message in enumerate(messages) if message.get("role") == "tool"
    }
    if set(by_execution) != tool_indices:
        missing = sorted(tool_indices - set(by_execution))
        raise ValueError(
            f"ACEBench tool source rows require execution receipts: {missing!r}"
        )
    return by_execution


def _json_no_duplicates(text: str) -> Any:
    def pairs(items):
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"non-JSON constant: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def _call_identity(call: Mapping[str, Any]) -> tuple[str, str]:
    function = call["function"]
    arguments = _json_no_duplicates(function["arguments"])
    if not isinstance(arguments, dict):
        raise ValueError("ACEBench call arguments must be an object")
    # Keep keyword order and JSON scalar types.  Python container equality
    # would incorrectly treat True, 1, and 1.0 as interchangeable.
    return function["name"], json.dumps(
        arguments,
        ensure_ascii=False,
        sort_keys=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _eligible_execution(
    assistant_text: Any,
    execution_content: Any,
    receipt: Mapping[str, Any],
    *,
    call_id_prefix: str,
) -> tuple[dict[str, Any], ...] | None:
    if not isinstance(assistant_text, str):
        return None
    submitted = parse_acebench_draft(
        assistant_text, call_id_prefix=call_id_prefix
    )
    if submitted.status != "tool_calls":
        return None
    if (
        receipt["decode_status"] != "ok"
        or not isinstance(receipt["decoded_calls"], list)
        or receipt["executor_status"] != "returned"
        or receipt["executor_return_shape"] != "list"
        or receipt["executor_return_count"] != len(submitted.tool_calls)
    ):
        return None
    decoded = parse_acebench_draft(
        "[" + ",".join(receipt["decoded_calls"]) + "]",
        call_id_prefix=f"{call_id_prefix}_receipt",
    )
    if decoded.status != "tool_calls" or len(decoded.tool_calls) != len(
        submitted.tool_calls
    ):
        return None
    try:
        if tuple(_call_identity(call) for call in decoded.tool_calls) != tuple(
            _call_identity(call) for call in submitted.tool_calls
        ):
            return None
        if not isinstance(execution_content, str):
            return None
        result = _json_no_duplicates(execution_content)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if not isinstance(result, list) or len(result) != len(submitted.tool_calls):
        return None
    return submitted.tool_calls


def build_ace_event_store(
    session_id: str,
    messages: Sequence[Mapping[str, Any]],
    source: Any,
) -> EventStore:
    """Build receipt-backed ACE events while preserving every source message."""

    if not isinstance(session_id, str) or not session_id:
        raise ValueError("An explicit nonempty session_id is required")
    if (
        not isinstance(messages, Sequence)
        or isinstance(messages, (str, bytes, bytearray))
        or not messages
        or any(not isinstance(message, Mapping) for message in messages)
    ):
        raise ValueError("messages must be a nonempty sequence of mappings")
    raw_messages = [dict(message) for message in messages]
    for index, message in enumerate(raw_messages):
        unknown = set(message) - _MESSAGE_FIELDS
        if unknown:
            raise ValueError(
                f"ACEBench source message {index} has unsupported fields: "
                f"{sorted(unknown)!r}"
            )
        role = message.get("role")
        expected_roles = {"system"} if index == 0 else {"user", "assistant", "tool"}
        if role not in expected_roles:
            raise ValueError(
                f"ACEBench source message {index} has invalid role {role!r}"
            )
        if not isinstance(message.get("content"), str):
            raise ValueError(
                f"ACEBench source message {index} content must be text"
            )
        if "tool_call_id" in message and (
            role != "tool"
            or not isinstance(message["tool_call_id"], str)
            or not message["tool_call_id"]
        ):
            raise ValueError(
                "ACEBench tool_call_id is allowed only as nonempty tool metadata"
            )
    snapshots = tuple(Message.from_dict(message) for message in raw_messages)
    receipts = _validated_receipts(raw_messages, source)

    events: list[EventRecord] = []
    index = 0
    while index < len(raw_messages):
        message = raw_messages[index]
        role = message["role"]
        receipt = receipts.get(index + 1) if role == "assistant" else None
        if receipt is not None:
            execution_index = index + 1
            calls = _eligible_execution(
                message.get("content"),
                raw_messages[execution_index].get("content"),
                receipt,
                call_id_prefix=f"{session_id}:m{index}",
            )
            if calls is None:
                events.append(
                    EventRecord(
                        event_id=f"{session_id}:m{index}",
                        kind=ACE_OPAQUE_EVENT_KIND,
                        source_indices=(index, execution_index),
                        complete=False,
                    )
                )
            else:
                call_ids = tuple(call["id"] for call in calls)
                events.append(
                    EventRecord(
                        event_id=f"{session_id}:m{index}",
                        kind="tool_event",
                        source_indices=(index, execution_index),
                        complete=True,
                        tool_call_ids=call_ids,
                    )
                )
            index += 2
            continue

        if role == "tool":
            # Receipt validation guarantees that every tool row is paired with
            # the immediately preceding assistant and consumed above.
            raise RuntimeError("validated ACEBench tool row was not paired")
        if role == "assistant" and isinstance(message.get("content"), str):
            draft = parse_acebench_draft(
                message["content"], call_id_prefix=f"{session_id}:m{index}"
            )
            if draft.status != "text":
                events.append(
                    EventRecord(
                        event_id=f"{session_id}:m{index}",
                        kind=ACE_OPAQUE_EVENT_KIND,
                        source_indices=(index,),
                        complete=False,
                    )
                )
                index += 1
                continue
        kind = "instruction" if role in {"system", "developer"} else role
        events.append(
            EventRecord(
                event_id=f"{session_id}:m{index}",
                kind=kind,
                source_indices=(index,),
                complete=True,
            )
        )
        index += 1
    return EventStore(session_id=session_id, messages=snapshots, events=tuple(events))


def ace_source_prefix_signature(
    messages: Sequence[Mapping[str, Any]], source: Any
) -> tuple[str, ...]:
    """Canonical per-row message/receipt identity for session prefix checks."""

    raw_messages = [dict(message) for message in messages]
    receipts = _validated_receipts(raw_messages, source)
    return tuple(
        json.dumps(
            {"message": message, "receipt": receipts.get(index)},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        for index, message in enumerate(raw_messages)
    )


__all__ = [
    "ACE_OPAQUE_EVENT_KIND",
    "ACE_RECEIPT_VERSION",
    "ACE_SOURCE_VERSION",
    "ace_source_prefix_signature",
    "build_ace_event_store",
    "parse_ace_draft",
    "parse_acebench_draft",
]
