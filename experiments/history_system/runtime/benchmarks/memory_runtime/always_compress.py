"""Explicit pre-B routes and source accounting for always-compressed history."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


ALWAYS_COMPRESSION_POLICY = "always-compress-v1"
ALWAYS_ROUTE_MODES = {
    "ac_gist_static": "legacy",
    "ac_protect": "capacity_protect",
    "ac_exact_once": "capacity_exact_once",
    "ac_exact_persistent": "capacity_exact_persistent",
    "ac_acquire_for_next": "capacity_exact_persistent",
    "ac_full_shared": "full_exact_shared",
    "raw_exact_shared": "capacity_exact_no_gist",
    "ac_packet_workspace": "capacity_protect",
    "ac_native_workspace": "capacity_protect",
}
WORKSPACE_ROUTE_RENDERERS = {
    "ac_packet_workspace": "evidence_packet",
    "ac_native_workspace": "native_workspace",
}
HISTORY_VIEW_PROTOCOLS = frozenset({
    "fixed-budget-main", "coverage-preserving-diagnostic",
})


class CapacityInfeasible(ValueError):
    """The declared method cannot represent this prefix within its budget."""

    kind = "capacity_infeasible"


class NativeCoverageUnsupported(ValueError):
    """The native packer omitted sources required by a full-coverage view."""

    kind = "unsupported_by_native_packing"


def eligible_history_sources(store: Any, source_cutoff: int) -> frozenset[int]:
    """Select completed source occurrences before the legacy live boundary."""
    return frozenset(
        index for event in store.events
        if event.complete and event.kind != "instruction"
        for index in event.source_indices if index < source_cutoff
    )


def coverage_accounting(
    *, eligible_sources: frozenset[int], raw_sources: set[int],
    retained_blocks: Sequence[Mapping[str, Any]],
    packing_fragments: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Keep occurrence coverage distinct from fitted-document token coverage.

    A source shared by several native fragments is fully represented only
    when every such fragment survives, or the complete source is raw-visible.
    Encoder template tokens are not presented as original conversation tokens.
    """
    gist_sources = set().union(*(set(b["source_indices"]) for b in retained_blocks)) if retained_blocks else set()
    raw = set(raw_sources) & eligible_sources
    touched = gist_sources & eligible_sources
    ledger_known = packing_fragments is not None
    fully_gist: set[int] = set()
    native_missing: set[int] = set()
    packing_tokens = None
    retained_tokens = None
    fragment_count = None
    retained_fragment_count = None
    if ledger_known:
        fragments = list(packing_fragments)
        kept_ids = {b.get("packing_fragment_id") for b in retained_blocks}
        kept_ids.discard(None)
        surviving = [f for f in fragments if f["fragment_id"] in kept_ids]
        candidates = set().union(*(set(f["source_indices"]) for f in fragments)) if fragments else set()
        for source in eligible_sources:
            relevant = [f for f in fragments if source in f["source_indices"]]
            if relevant and all(f["fragment_id"] in kept_ids for f in relevant):
                fully_gist.add(source)
        native_missing = set(eligible_sources) - candidates
        packing_tokens = sum(int(f["encoder_input_tokens"]) for f in fragments)
        retained_tokens = sum(int(f["encoder_input_tokens"]) for f in surviving)
        fragment_count = len(fragments)
        retained_fragment_count = len(surviving)
    complete = raw | fully_gist
    missing = sorted(set(eligible_sources) - complete) if ledger_known else None
    return {
        "schema": "a-source-occurrence-coverage-v1",
        "packing_ledger_available": ledger_known,
        "eligible_source_indices": sorted(eligible_sources),
        "raw_source_indices": sorted(raw),
        "gist_touched_source_indices": sorted(touched),
        "gist_fully_represented_source_indices": sorted(fully_gist) if ledger_known else None,
        "raw_gist_overlap_source_indices": sorted(raw & touched),
        "unrepresented_source_indices": missing,
        "native_unencoded_source_indices": sorted(native_missing) if ledger_known else None,
        "fully_represented_source_fraction": len(complete) / len(eligible_sources)
        if ledger_known and eligible_sources else None,
        "complete_history_coverage": not missing if ledger_known else None,
        "fitted_fragment_count": fragment_count,
        "retained_fragment_count": retained_fragment_count,
        "fitted_encoder_input_tokens": packing_tokens,
        "retained_encoder_input_tokens": retained_tokens,
        "fitted_fragment_retained_token_fraction": retained_tokens / packing_tokens
        if packing_tokens else None,
        "token_weight_scope": "fitted native encoder inputs including their templates; not raw history tokens",
    }


def ratio_accounting(full: Mapping[str, Any], runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Use the same observable prefix, never another arm's generated history."""
    geometry = int(runtime["bytes_per_kv_token"])
    history = int(full["active_history_bytes"])
    common = int(full["common_raw_prompt_tokens"]) * geometry
    active = int(runtime["active_history_bytes"])
    actual_common = int(runtime["common_raw_prompt_tokens"]) * geometry
    if actual_common != common:
        raise ValueError("always-compress common-input boundary differs from its Full reference")
    return {
        "schema": "a-same-prefix-compression-ratio-v1",
        "denominator_source": "Full renderer on this arm's observable prefix; no Full generation",
        "full_history_bytes": history,
        "common_live_bytes": common,
        "active_history_bytes": active,
        "active_gist_bytes": int(runtime["gist_tokens"]) * geometry,
        "active_raw_history_bytes": int(runtime["raw_history_tokens"]) * geometry,
        "n_history": history / active if history and active else None,
        "n_total": (common + history) / (common + active) if common + active else None,
        "includes_coverage_loss": True,
    }
