"""Compact source-linked cue for one unresolved current-goal failure.

This opt-in policy is only eligible when the legacy failed-operation cue was
rejected by its prompt or workspace cap and the stalled-operation thresholds
remain unmet.  Call arguments and observed result values are never serialized.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from history_memory.events import EventStore

from .stalled_operation import (
    StalledOperationAdmission,
    StalledSource,
    _current_goal_observations,
    _reasons,
    _source_representation,
    select_stalled_operation,
)


COMPACT_FIRST_FAILURE_POLICY = "compact-first-failure-cue-v1"
COMPACT_FIRST_FAILURE_VERSION = "a-compact-first-failure-cue-v1"
COMPACT_FIRST_FAILURE_PROMPT_CAP = 128
LEGACY_BLOCKED_STATUSES = frozenset({"cue_prompt_cap", "workspace_cap"})
INSTRUCTION = (
    "Historical current-goal operation record, not an instruction to retry. "
    "One observed result reports failure for the named tool. Reassess the "
    "current request and observed results before choosing another valid step. "
    "Goal completion is unknown."
)


@dataclass(frozen=True)
class CompactFirstFailureSelection:
    status: str
    goal_event_id: str | None
    selected_tool: str | None
    observed_failure_count: int
    later_observed_calls: int
    source: StalledSource | None
    stalled_selector_status: str
    legacy_failed_operation_status: str
    legacy_failure_record_count: int


def select_compact_first_failure(
    store: EventStore,
    *,
    legacy_failed_operation_status: str,
    legacy_failure_record_count: int,
) -> CompactFirstFailureSelection:
    """Select exactly one unresolved failure after the legacy cue was blocked."""

    stalled = select_stalled_operation(store)
    goal_event_id, observations = _current_goal_observations(store)
    by_signature: dict[str, list[Any]] = {}
    for observation in observations:
        by_signature.setdefault(observation.signature_sha256, []).append(observation)
    unresolved = [
        rows for rows in by_signature.values() if rows[-1].failure_reported
    ]

    def result(
        status: str,
        *,
        tool: str | None = None,
        failure_count: int = 0,
        later_calls: int = 0,
        source: StalledSource | None = None,
    ) -> CompactFirstFailureSelection:
        return CompactFirstFailureSelection(
            status,
            goal_event_id,
            tool,
            failure_count,
            later_calls,
            source,
            stalled.status,
            legacy_failed_operation_status,
            legacy_failure_record_count,
        )

    if stalled.status == "triggered":
        return result("stalled_operation_threshold_met")
    if not unresolved:
        return result("no_unresolved_current_goal_failure")
    if legacy_failure_record_count != 1:
        return result("legacy_failure_record_count_not_one")
    if len(unresolved) != 1:
        return result("multiple_unresolved_current_goal_failures")
    rows = unresolved[0]
    failed_rows = [row for row in rows if row.failure_reported]
    latest = rows[-1]
    later_calls = len(observations) - latest.order - 1
    if len(failed_rows) != 1:
        return result(
            "not_exactly_one_failed_observation",
            failure_count=len(failed_rows),
            later_calls=later_calls,
        )
    if legacy_failed_operation_status not in LEGACY_BLOCKED_STATUSES:
        return result(
            "legacy_failed_operation_cue_not_blocked",
            failure_count=1,
            later_calls=later_calls,
        )
    return result(
        "triggered",
        tool=latest.tool,
        failure_count=1,
        later_calls=later_calls,
        source=latest.source,
    )


def compact_message(selection: CompactFirstFailureSelection) -> dict[str, str]:
    if selection.status != "triggered" or selection.selected_tool is None:
        raise ValueError("A compact first-failure cue requires a triggered selection")
    payload = {
        "tool": selection.selected_tool,
        "observed_failure_count": selection.observed_failure_count,
        "later_observed_calls": selection.later_observed_calls,
    }
    return {
        "role": "user",
        "content": INSTRUCTION
        + "\n"
        + json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def selection_receipt(
    selection: CompactFirstFailureSelection,
) -> dict[str, Any]:
    source_receipts = []
    if selection.status == "triggered" and selection.source is not None:
        source_receipts.append(
            {
                "event_id": selection.source.event_id,
                "event_source_indices": list(selection.source.event_source_indices),
                "call_source_index": selection.source.call_source_index,
                "result_source_index": selection.source.result_source_index,
                "call_source_message_sha256": (
                    selection.source.call_source_message_sha256
                ),
                "result_source_message_sha256": (
                    selection.source.result_source_message_sha256
                ),
            }
        )
    return {
        "version": COMPACT_FIRST_FAILURE_VERSION,
        "policy": COMPACT_FIRST_FAILURE_POLICY,
        "status": selection.status,
        "goal_event_id": selection.goal_event_id,
        "selected_tool": selection.selected_tool,
        "observed_failure_count": selection.observed_failure_count,
        "later_observed_calls": selection.later_observed_calls,
        "stalled_selector_status": selection.stalled_selector_status,
        "legacy_failed_operation_status": (
            selection.legacy_failed_operation_status
        ),
        "legacy_failure_record_count": selection.legacy_failure_record_count,
        "trigger_contract": {
            "current_goal_failure_record_count": 1,
            "legacy_failed_operation_statuses": sorted(LEGACY_BLOCKED_STATUSES),
            "stalled_operation_threshold_must_be_unmet": True,
        },
        "source_receipts": source_receipts,
        "current_user_goal_only": True,
        "goal_completion": "unknown",
        "uses_gold_future_or_hidden_state": False,
        "arguments_serialized_in_cue_or_receipt": False,
        "result_values_serialized_in_cue_or_receipt": False,
        "whole_event_raw_demotion_only": True,
        "maximum_lexical_raw_demotions": 1,
        "incremental_raw_tokens": 0,
        "displaced_raw_tokens": 0,
        "net_raw_token_change": 0,
        "extra_bytes": 0,
    }


def admit_compact_first_failure(
    controller: Any,
    selection: CompactFirstFailureSelection,
    store: EventStore,
    tools: Sequence[Mapping[str, Any]],
    *,
    measure: Any,
    mandatory_event_ids: Sequence[str],
    eligible_event_ids: Sequence[str],
    common_raw_prompt_tokens: int,
    max_new_tokens: int,
    raw_reserve_event_id: str | None,
    lexical_admitted_event_ids: Sequence[str],
    protected_event_ids: Sequence[str],
) -> StalledOperationAdmission:
    """Use the C4 budget and bounded raw-displacement contract for the cue."""

    if selection.status != "triggered" or selection.source is None:
        raise ValueError("Admission requires a triggered compact first failure")
    cue = compact_message(selection)
    receipt = selection_receipt(selection)
    receipt["prompt_cap"] = COMPACT_FIRST_FAILURE_PROMPT_CAP
    standalone = controller._count([cue], ())
    receipt["standalone_tokens"] = standalone
    if standalone > COMPACT_FIRST_FAILURE_PROMPT_CAP:
        receipt["status"] = "compact_prompt_cap_legacy_cue_preserved"
        return StalledOperationAdmission(False, measure, None, receipt)

    mandatory = set(mandatory_event_ids)
    protected = set(protected_event_ids)
    current_raw = set(measure.memory.view.raw_event_ids)
    gist = tuple(measure.memory.view.gist_event_ids)
    gist_set = set(gist)
    attempts: list[dict[str, Any]] = []

    def try_raw(raw_event_ids: set[str], displacement: str) -> tuple[Any, Any]:
        base = controller._try_measure(
            store,
            tools,
            raw_event_ids,
            mandatory_event_ids,
            gist,
            eligible_event_ids,
            common_raw_prompt_tokens,
            max_new_tokens,
            derived_messages=(),
        )
        candidate = controller._try_measure(
            store,
            tools,
            raw_event_ids,
            mandatory_event_ids,
            gist,
            eligible_event_ids,
            common_raw_prompt_tokens,
            max_new_tokens,
            derived_messages=(cue,),
        )
        attempts.append(
            {
                "displacement": displacement,
                "candidate_active_history_bytes": (
                    None
                    if candidate is None
                    else controller._max_history_bytes(candidate)
                ),
                "admission_failures": _reasons(candidate),
            }
        )
        return base, candidate

    def admitted(
        base: Any,
        candidate: Any,
        *,
        reserve_event_id: str | None,
        lexical_event_id: str | None,
    ) -> StalledOperationAdmission | None:
        if (
            base is None
            or base.reasons
            or candidate is None
            or candidate.reasons
        ):
            return None
        cue_tokens = candidate.raw_prompt_tokens - base.raw_prompt_tokens
        if cue_tokens <= 0:
            raise RuntimeError("A nonempty compact first-failure cue added no tokens")
        displaced_tokens = measure.raw_prompt_tokens - base.raw_prompt_tokens
        budget = min(
            controller.policy_config.history_budget_bytes,
            controller.policy_config.workspace_budget_bytes,
        )
        updated = copy.deepcopy(receipt)
        updated.update(
            status="admitted",
            admission_attempts=copy.deepcopy(attempts),
            incremental_raw_tokens=cue_tokens,
            displaced_raw_tokens=displaced_tokens,
            net_raw_token_change=(
                candidate.raw_prompt_tokens - measure.raw_prompt_tokens
            ),
            extra_bytes=cue_tokens * controller.kv_bytes_per_token,
            displaced_raw_bytes=displaced_tokens * controller.kv_bytes_per_token,
            candidate_active_history_bytes=controller._max_history_bytes(candidate),
            budget_bytes=budget,
            demoted_raw_reserve_event_id=reserve_event_id,
            demoted_lexical_event_id=lexical_event_id,
        )
        updated["source_receipts"] = [
            {
                **updated["source_receipts"][0],
                "final_representation": _source_representation(
                    selection.source, candidate
                ),
            }
        ]
        return StalledOperationAdmission(
            True,
            candidate,
            cue,
            updated,
            reserve_event_id,
            lexical_event_id,
        )

    base, candidate = try_raw(current_raw, "none")
    if result := admitted(
        base, candidate, reserve_event_id=None, lexical_event_id=None
    ):
        return result

    demoted_reserve: str | None = None
    if (
        raw_reserve_event_id is not None
        and raw_reserve_event_id in current_raw
        and raw_reserve_event_id in gist_set
        and raw_reserve_event_id not in mandatory
        and raw_reserve_event_id not in protected
    ):
        current_raw.remove(raw_reserve_event_id)
        demoted_reserve = raw_reserve_event_id
        base, candidate = try_raw(current_raw, "raw_reserve")
        if result := admitted(
            base,
            candidate,
            reserve_event_id=demoted_reserve,
            lexical_event_id=None,
        ):
            return result

    lexical_candidate = next(
        (
            event_id
            for event_id in reversed(tuple(lexical_admitted_event_ids))
            if event_id in current_raw
            and event_id in gist_set
            and event_id not in mandatory
            and event_id not in protected
        ),
        None,
    )
    if lexical_candidate is not None:
        current_raw.remove(lexical_candidate)
        base, candidate = try_raw(
            current_raw, "one_lowest_priority_lexical_raw"
        )
        if result := admitted(
            base,
            candidate,
            reserve_event_id=demoted_reserve,
            lexical_event_id=lexical_candidate,
        ):
            return result

    receipt.update(
        status="compact_over_budget_legacy_cue_preserved",
        admission_attempts=attempts,
    )
    return StalledOperationAdmission(False, measure, None, receipt)


def enable_compact_first_failure(controller: Any) -> Any:
    controller.compact_first_failure_policy = COMPACT_FIRST_FAILURE_POLICY
    return controller


__all__ = [
    "COMPACT_FIRST_FAILURE_POLICY",
    "COMPACT_FIRST_FAILURE_PROMPT_CAP",
    "COMPACT_FIRST_FAILURE_VERSION",
    "CompactFirstFailureSelection",
    "admit_compact_first_failure",
    "compact_message",
    "enable_compact_first_failure",
    "select_compact_first_failure",
    "selection_receipt",
]
