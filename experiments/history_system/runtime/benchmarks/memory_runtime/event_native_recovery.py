"""Public entry point for deployable post-draft event recovery."""

from __future__ import annotations

from typing import Any, Mapping

from .recovery.config import (
    E1_RECOVERY_CONFIG_KEY,
    E1_RECOVERY_VERSION,
    FIRST_NAME_MARGIN_FEATURE,
    parse_recovery_config as _parse_config,
)
from .recovery.orchestrator import (
    EventNativeRecoveryController,
    PreparedEventNativeRecovery,
)
from .recovery.gate import _read_prefill_score
from .recovery.source import rank_visible_source_events as _rank_visible_source_events


def wrap_with_event_native_recovery(base: Any, config: Mapping[str, Any]):
    return EventNativeRecoveryController(base, config)


__all__ = [
    "E1_RECOVERY_CONFIG_KEY",
    "E1_RECOVERY_VERSION",
    "FIRST_NAME_MARGIN_FEATURE",
    "EventNativeRecoveryController",
    "PreparedEventNativeRecovery",
    "wrap_with_event_native_recovery",
]
