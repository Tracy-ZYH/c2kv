"""Pure-CPU capacity accounting for an already rendered Full prompt."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _count(token_counter, messages, tools):
    value = token_counter(messages, tools)
    if type(value) is not int or value < 0:
        raise ValueError("token_counter must return a nonnegative integer")
    return value


def measure_full_history(
    full_view: list[dict[str, Any]],
    full_counts: dict[str, Any],
    token_counter: Callable[[list[dict[str, Any]], Any], int],
    tools: Any,
    bytes_per_kv_token: int,
) -> dict[str, int]:
    """Measure history bytes outside Full's system-plus-current boundary."""
    if not isinstance(full_view, list) or any(
        not isinstance(message, dict) for message in full_view
    ):
        raise TypeError("full_view must be a list of messages")
    if any("c2kv_key_hash" in message for message in full_view):
        raise ValueError("full_view must not contain gist carriers")
    if not isinstance(full_counts, dict):
        raise TypeError("full_counts must be a dict")
    cutoff = full_counts.get("current_start_out_index")
    if type(cutoff) is not int or not 0 <= cutoff <= len(full_view):
        raise ValueError("current_start_out_index must be an integer within full_view")
    if not callable(token_counter):
        raise TypeError("token_counter must be callable")
    if type(bytes_per_kv_token) is not int or bytes_per_kv_token <= 0:
        raise ValueError("bytes_per_kv_token must be a positive integer")

    view = [dict(message) for message in full_view]
    common = [
        message for message in view[:cutoff] if message.get("role") == "system"
    ] + view[cutoff:]
    total_tokens = _count(token_counter, view, tools)
    common_tokens = _count(token_counter, common, tools)
    history_tokens = total_tokens - common_tokens
    if history_tokens < 0:
        raise ValueError("Full raw history token delta became negative")
    return {
        "total_raw_prompt_tokens": total_tokens,
        "common_raw_prompt_tokens": common_tokens,
        "raw_history_tokens": history_tokens,
        "active_history_bytes": history_tokens * bytes_per_kv_token,
    }
