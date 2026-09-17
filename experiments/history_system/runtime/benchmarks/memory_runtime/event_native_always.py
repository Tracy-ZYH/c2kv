"""Native-only route catalog for the bounded pre-B always-compress interface."""

from __future__ import annotations


NATIVE_ALWAYS_IMPLEMENTATION_PROFILE = "event-native-always-compress-v1"
NATIVE_S0_MODE = "ac_native_s0_lexical_raw_reserve_failed_operation"
NATIVE_ALWAYS_CANONICAL_MODES = {
    "ac_gist_static": "capacity_protect",
    "ac_protect": "capacity_protect",
    "ac_exact_once": "capacity_exact_once",
    "ac_exact_persistent": "capacity_exact_persistent",
    "ac_full_shared": "full_exact_shared",
    "raw_exact_shared": "capacity_exact_no_gist",
    NATIVE_S0_MODE: NATIVE_S0_MODE,
}
NATIVE_ALWAYS_ROUTE_MODES = frozenset(NATIVE_ALWAYS_CANONICAL_MODES)


__all__ = [
    "NATIVE_ALWAYS_CANONICAL_MODES",
    "NATIVE_ALWAYS_IMPLEMENTATION_PROFILE",
    "NATIVE_ALWAYS_ROUTE_MODES",
]
