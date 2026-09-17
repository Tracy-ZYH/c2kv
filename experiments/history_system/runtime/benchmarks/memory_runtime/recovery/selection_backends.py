"""Bounded chat and embedding backends for recovery selection.

The runtime accepts injectable callbacks for tests and an OpenAI-compatible
HTTP backend for deployed use.  Credentials are named by environment variable;
secret values are never accepted in configuration or copied into receipts.
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any


class SelectionBackendError(RuntimeError):
    """A configured selection dependency is unavailable or malformed."""


_ALLOWED_BACKEND_FIELDS = frozenset(
    {
        "type",
        "base_url",
        "chat_model",
        "embedding_model",
        "api_key_env",
        "timeout_seconds",
        "max_tokens",
        "query_max_chars",
        "candidate_text_max_chars",
        "hybrid_lexical_weight",
    }
)
_SECRET_FIELDS = frozenset(
    {"api_key", "token", "access_token", "authorization", "password"}
)


def required_backend_capabilities(config: Mapping[str, Any]) -> frozenset[str]:
    """Return external capabilities required by one parsed G--P config."""

    required: set[str] = set()
    if config.get("Q") == "hybrid":
        required.add("embed")
    if (
        config.get("Q") == "llm_rewrite"
        or (config.get("selector") is None and (
            config.get("K") == "llm" or config.get("D") in {"detector_llm", "joint_llm"}))
        or config.get("selector") == "llm"
    ):
        required.add("chat")
    return frozenset(required)


def make_backends(
    config: Mapping[str, Any], backends: Any = None
) -> Any:
    """Resolve and validate selection dependencies before generation starts."""

    required = required_backend_capabilities(config)
    if backends is None:
        backend_config = config.get("backend")
        if not required:
            return None
        if not isinstance(backend_config, Mapping):
            names = ", ".join(sorted(required))
            raise ValueError(
                f"selection requires backend capabilities {names}; configure backend"
            )
        backends = OpenAICompatibleSelectionBackend(backend_config)
        if "chat" in required and not backends.config.get("chat_model"):
            raise ValueError("backend.chat_model is required by selection config")
        if "embed" in required and not backends.config.get("embedding_model"):
            raise ValueError("backend.embedding_model is required by selection config")
    for capability in sorted(required):
        callback = _callback(backends, capability)
        if not callable(callback):
            raise TypeError(
                f"selection backend must provide a callable {capability} method"
            )
    return backends


def public_backend_config(backends: Any, config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return dependency settings without resolving or exposing a secret."""

    if backends is not None and callable(getattr(backends, "public_config", None)):
        value = backends.public_config()
        return dict(value) if isinstance(value, Mapping) else None
    backend_config = config.get("backend")
    if not isinstance(backend_config, Mapping):
        return {"type": "injected_callbacks"} if backends is not None else None
    return _public_config(_validate_backend_config(backend_config))


def call_chat(
    backends: Any,
    *,
    messages: Sequence[Mapping[str, str]],
    purpose: str,
    config: Mapping[str, Any],
) -> Any:
    callback = _callback(backends, "chat")
    if not callable(callback):
        raise SelectionBackendError("chat backend is unavailable")
    return callback(messages=list(messages), purpose=purpose, config=dict(config))


def call_embed(
    backends: Any,
    *,
    texts: Sequence[str],
    purpose: str,
    config: Mapping[str, Any],
) -> Any:
    callback = _callback(backends, "embed")
    if not callable(callback):
        raise SelectionBackendError("embedding backend is unavailable")
    return callback(texts=list(texts), purpose=purpose, config=dict(config))


class OpenAICompatibleSelectionBackend:
    """Minimal stdlib client for OpenAI-compatible chat and embeddings APIs."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = _validate_backend_config(config)
        self._receipts: list[dict[str, Any]] = []

    def public_config(self) -> dict[str, Any]:
        return _public_config(self.config)

    def chat(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        purpose: str,
        config: Mapping[str, Any],
    ) -> str:
        model = self.config.get("chat_model")
        if not model:
            raise SelectionBackendError("backend.chat_model is required")
        payload = {
            "model": model,
            "messages": [dict(message) for message in messages],
            "temperature": 0,
            "max_tokens": self.config["max_tokens"],
        }
        response, latency = self._post("chat/completions", payload)
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise SelectionBackendError(
                "chat response lacks choices[0].message.content"
            ) from error
        if not isinstance(content, str):
            raise SelectionBackendError("chat response content must be a string")
        self._receipts.append(
            {
                "capability": "chat",
                "purpose": purpose,
                "model": model,
                "latency_seconds": latency,
                "usage": _public_usage(response.get("usage")),
            }
        )
        return content

    def embed(
        self,
        *,
        texts: Sequence[str],
        purpose: str,
        config: Mapping[str, Any],
    ) -> list[list[float]]:
        model = self.config.get("embedding_model")
        if not model:
            raise SelectionBackendError("backend.embedding_model is required")
        response, latency = self._post(
            "embeddings", {"model": model, "input": list(texts)}
        )
        try:
            rows = response["data"]
            ordered = sorted(rows, key=lambda row: row.get("index", 0))
            vectors = [row["embedding"] for row in ordered]
        except (KeyError, TypeError) as error:
            raise SelectionBackendError("embedding response has invalid data") from error
        if len(vectors) != len(texts):
            raise SelectionBackendError("embedding response count differs from input")
        self._receipts.append(
            {
                "capability": "embed",
                "purpose": purpose,
                "model": model,
                "input_count": len(texts),
                "latency_seconds": latency,
                "usage": _public_usage(response.get("usage")),
            }
        )
        return [_finite_vector(vector) for vector in vectors]

    def drain_receipts(self) -> list[dict[str, Any]]:
        result = self._receipts
        self._receipts = []
        return result

    def _post(
        self, suffix: str, payload: Mapping[str, Any]
    ) -> tuple[Mapping[str, Any], float]:
        url = self.config["base_url"].rstrip("/") + "/" + suffix
        headers = {"Content-Type": "application/json"}
        env_name = self.config.get("api_key_env")
        if env_name:
            secret = os.environ.get(env_name)
            if not secret:
                raise SelectionBackendError(
                    f"backend credential environment variable {env_name!r} is unset"
                )
            headers["Authorization"] = "Bearer " + secret
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(
                request, timeout=self.config["timeout_seconds"]
            ) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            raise SelectionBackendError(
                f"selection backend returned HTTP {error.code}"
            ) from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise SelectionBackendError("selection backend request failed") from error
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SelectionBackendError("selection backend returned invalid JSON") from error
        if not isinstance(value, Mapping):
            raise SelectionBackendError("selection backend response must be an object")
        return value, time.monotonic() - started


def _callback(backends: Any, capability: str) -> Any:
    if isinstance(backends, Mapping):
        value = backends.get(capability)
        if value is None and capability == "embed":
            value = backends.get("embeddings")
        return value
    return getattr(backends, capability, None)


def _validate_backend_config(value: Mapping[str, Any]) -> dict[str, Any]:
    forbidden = sorted(set(value) & _SECRET_FIELDS)
    if forbidden:
        raise ValueError(
            "backend secrets must be supplied through api_key_env, not config: "
            f"{forbidden!r}"
        )
    unknown = sorted(set(value) - _ALLOWED_BACKEND_FIELDS)
    if unknown:
        raise ValueError(f"Unknown selection backend fields: {unknown!r}")
    if value.get("type") != "openai_compatible":
        raise ValueError("backend.type must be 'openai_compatible'")
    base_url = value.get("base_url")
    if not isinstance(base_url, str) or not base_url:
        raise ValueError("backend.base_url must be a nonempty HTTP(S) URL")
    parsed = urllib.parse.urlsplit(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "backend.base_url must be an HTTP(S) URL without credentials, query, or fragment"
        )
    result = dict(value)
    for field in ("chat_model", "embedding_model", "api_key_env"):
        item = result.get(field)
        if item is not None and (not isinstance(item, str) or not item):
            raise ValueError(f"backend.{field} must be a nonempty string")
    env_name = result.get("api_key_env")
    if env_name and not env_name.replace("_", "A").isalnum():
        raise ValueError("backend.api_key_env must name an environment variable")
    result["timeout_seconds"] = _positive_number(
        result.get("timeout_seconds", 30.0), "timeout_seconds"
    )
    result["max_tokens"] = _positive_integer(
        result.get("max_tokens", 128), "max_tokens"
    )
    result["query_max_chars"] = _positive_integer(
        result.get("query_max_chars", 4096), "query_max_chars"
    )
    result["candidate_text_max_chars"] = _positive_integer(
        result.get("candidate_text_max_chars", 2048), "candidate_text_max_chars"
    )
    weight = result.get("hybrid_lexical_weight", 0.5)
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(float(weight))
        or not 0 <= float(weight) <= 1
    ):
        raise ValueError("backend.hybrid_lexical_weight must be from zero to one")
    result["hybrid_lexical_weight"] = float(weight)
    return result


def _public_config(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in (
            "type",
            "base_url",
            "chat_model",
            "embedding_model",
            "api_key_env",
            "timeout_seconds",
            "max_tokens",
            "query_max_chars",
            "candidate_text_max_chars",
            "hybrid_lexical_weight",
        )
        if key in value
    }


def _positive_integer(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"backend.{name} must be a positive integer")
    return value


def _positive_number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(f"backend.{name} must be finite and positive")
    return float(value)


def _finite_vector(value: Any) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise SelectionBackendError("embedding must be a numeric vector")
    result: list[float] = []
    for item in value:
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
        ):
            raise SelectionBackendError("embedding contains a non-finite value")
        result.append(float(item))
    if not result:
        raise SelectionBackendError("embedding vector must be nonempty")
    return result


def _public_usage(value: Any) -> dict[str, int | None]:
    fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    if not isinstance(value, Mapping):
        return {field: None for field in fields}
    return {
        field: item if type(item := value.get(field)) is int and item >= 0 else None
        for field in fields
    }


__all__ = [
    "OpenAICompatibleSelectionBackend",
    "SelectionBackendError",
    "call_chat",
    "call_embed",
    "make_backends",
    "public_backend_config",
    "required_backend_capabilities",
]
