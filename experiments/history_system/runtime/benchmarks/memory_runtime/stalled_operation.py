"""Compact current-goal ledger for repeated unresolved tool failures.

The selector uses only the observable message prefix.  It groups exact call
signatures internally but never serializes arguments or observed result values
into the cue or receipt.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from history_memory.events import EventStore

from .failed_operation import failed as result_reports_failure
from .observed_entity_slot import _source_message_sha256


STALLED_OPERATION_POLICY = "current-goal-stalled-operation-ledger-v1"
STALLED_OPERATION_VERSION = "a-current-goal-stalled-operation-ledger-v1"
EXACT_FAILURE_REPEAT_THRESHOLD = 2
TOOL_FAILED_VARIANT_THRESHOLD = 3
LATER_OBSERVED_CALL_THRESHOLD = 3
LEDGER_PROMPT_CAP = 128
INSTRUCTION = (
    "Historical current-goal operation ledger, not an instruction to retry. "
    "Observed results report a stalled pattern around an unresolved failure. Avoid "
    "repeating that operation without a new reason; reassess the current request "
    "and choose another valid step when appropriate. Goal completion is unknown."
)


@dataclass(frozen=True)
class StalledSource:
    event_id: str
    event_source_indices: tuple[int, ...]
    call_source_index: int
    result_source_index: int
    call_source_message_sha256: str
    result_source_message_sha256: str


@dataclass(frozen=True)
class StalledOperationSelection:
    status: str
    goal_event_id: str | None
    selected_tool: str | None
    trigger_reasons: tuple[str, ...]
    unresolved_failed_signatures: int
    max_failed_observations_for_one_signature: int
    max_later_observed_calls: int
    sources: tuple[StalledSource, ...]


@dataclass(frozen=True)
class StalledOperationAdmission:
    admitted: bool
    memory_measure: Any
    cue: dict[str, str] | None
    receipt: dict[str, Any]
    demoted_raw_reserve_event_id: str | None = None
    demoted_lexical_event_id: str | None = None


@dataclass(frozen=True)
class _Observation:
    tool: str
    signature_sha256: str
    failure_reported: bool
    order: int
    source: StalledSource


def _parse(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _signature_sha256(tool: str, arguments: Any) -> str:
    payload = json.dumps(
        {"tool": tool, "arguments": arguments},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _current_goal_observations(
    store: EventStore,
) -> tuple[str | None, list[_Observation]]:
    goal_event_id: str | None = None
    observations: list[_Observation] = []
    for event in store.events:
        if event.kind == "user":
            goal_event_id = event.event_id
            observations = []
            continue
        if event.kind != "tool_event" or not event.complete:
            continue
        indexed = [
            (index, message.to_dict())
            for index, message in zip(
                event.source_indices, store.event_messages(event.event_id)
            )
        ]
        results = {
            message["tool_call_id"]: (index, _parse(message.get("content")))
            for index, message in indexed
            if message["role"] == "tool"
        }
        for call_source_index, message in indexed:
            for call in message.get("tool_calls") or ():
                function = call["function"]
                tool = function["name"]
                arguments = _parse(function.get("arguments"))
                result_source_index, result = results[call["id"]]
                source = StalledSource(
                    event_id=event.event_id,
                    event_source_indices=tuple(event.source_indices),
                    call_source_index=call_source_index,
                    result_source_index=result_source_index,
                    call_source_message_sha256=_source_message_sha256(
                        store, call_source_index
                    ),
                    result_source_message_sha256=_source_message_sha256(
                        store, result_source_index
                    ),
                )
                observations.append(
                    _Observation(
                        tool=tool,
                        signature_sha256=_signature_sha256(tool, arguments),
                        failure_reported=result_reports_failure(result),
                        order=len(observations),
                        source=source,
                    )
                )
    return goal_event_id, observations


def select_stalled_operation(store: EventStore) -> StalledOperationSelection:
    """Select the newest tool group that crosses one fixed stall threshold."""

    goal_event_id, observations = _current_goal_observations(store)
    by_signature: dict[str, list[_Observation]] = {}
    for observation in observations:
        by_signature.setdefault(observation.signature_sha256, []).append(observation)
    unresolved = {
        signature: rows
        for signature, rows in by_signature.items()
        if rows[-1].failure_reported
    }
    if not unresolved:
        return StalledOperationSelection(
            "no_unresolved_current_goal_failure",
            goal_event_id,
            None,
            (),
            0,
            0,
            0,
            (),
        )

    grouped: dict[str, list[tuple[str, list[_Observation]]]] = {}
    for signature, rows in unresolved.items():
        grouped.setdefault(rows[-1].tool, []).append((signature, rows))
    candidates: list[
        tuple[int, str, tuple[str, ...], int, int, int, tuple[StalledSource, ...]]
    ] = []
    total_observations = len(observations)
    for tool, signatures in grouped.items():
        failed_counts = [
            sum(row.failure_reported for row in rows) for _, rows in signatures
        ]
        exact_repeat = max(failed_counts)
        variant_count = len(signatures)
        max_later = max(
            total_observations - rows[-1].order - 1 for _, rows in signatures
        )
        reasons: list[str] = []
        if exact_repeat >= EXACT_FAILURE_REPEAT_THRESHOLD:
            reasons.append("exact_signature_repeated_failure")
        if variant_count >= TOOL_FAILED_VARIANT_THRESHOLD:
            reasons.append("same_tool_failed_signature_variants")
        if max_later >= LATER_OBSERVED_CALL_THRESHOLD:
            reasons.append("unresolved_failure_followed_by_later_calls")
        if not reasons:
            continue
        sources = tuple(
            row.source
            for _, rows in signatures
            for row in rows
            if row.failure_reported
        )
        newest_failure_order = max(rows[-1].order for _, rows in signatures)
        candidates.append(
            (
                newest_failure_order,
                tool,
                tuple(reasons),
                variant_count,
                exact_repeat,
                max_later,
                sources,
            )
        )

    if not candidates:
        return StalledOperationSelection(
            "no_stall_threshold_met",
            goal_event_id,
            None,
            (),
            len(unresolved),
            max(
                sum(row.failure_reported for row in rows)
                for rows in unresolved.values()
            ),
            max(
                len(observations) - rows[-1].order - 1
                for rows in unresolved.values()
            ),
            (),
        )

    newest = sorted(candidates, key=lambda row: (-row[0], row[1]))[0]
    _, tool, reasons, variant_count, exact_repeat, max_later, sources = newest
    return StalledOperationSelection(
        "triggered",
        goal_event_id,
        tool,
        reasons,
        variant_count,
        exact_repeat,
        max_later,
        sources,
    )


def ledger_message(selection: StalledOperationSelection) -> dict[str, str]:
    if selection.status != "triggered" or selection.selected_tool is None:
        raise ValueError("A stalled-operation ledger requires a triggered selection")
    ledger = {
        "tool": selection.selected_tool,
        "trigger_reasons": list(selection.trigger_reasons),
        "unresolved_failed_signatures": selection.unresolved_failed_signatures,
        "max_failed_observations_for_one_signature": (
            selection.max_failed_observations_for_one_signature
        ),
        "max_later_observed_calls": selection.max_later_observed_calls,
    }
    return {
        "role": "user",
        "content": INSTRUCTION
        + "\n"
        + json.dumps(
            ledger,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def selection_receipt(selection: StalledOperationSelection) -> dict[str, Any]:
    return {
        "version": STALLED_OPERATION_VERSION,
        "policy": STALLED_OPERATION_POLICY,
        "status": selection.status,
        "goal_event_id": selection.goal_event_id,
        "selected_tool": selection.selected_tool,
        "trigger_reasons": list(selection.trigger_reasons),
        "thresholds": {
            "exact_signature_failed_observations": EXACT_FAILURE_REPEAT_THRESHOLD,
            "same_tool_failed_signature_variants": TOOL_FAILED_VARIANT_THRESHOLD,
            "later_observed_calls_after_unresolved_failure": (
                LATER_OBSERVED_CALL_THRESHOLD
            ),
        },
        "unresolved_failed_signatures": selection.unresolved_failed_signatures,
        "max_failed_observations_for_one_signature": (
            selection.max_failed_observations_for_one_signature
        ),
        "max_later_observed_calls": selection.max_later_observed_calls,
        "source_receipts": [
            {
                "event_id": source.event_id,
                "event_source_indices": list(source.event_source_indices),
                "call_source_index": source.call_source_index,
                "result_source_index": source.result_source_index,
                "call_source_message_sha256": source.call_source_message_sha256,
                "result_source_message_sha256": source.result_source_message_sha256,
            }
            for source in selection.sources
        ],
        "current_user_goal_only": True,
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


def _reasons(candidate: Any) -> list[str]:
    return (
        ["packing_budget_exceeded"]
        if candidate is None
        else list(candidate.reasons)
    )


def _source_representation(source: StalledSource, measure: Any) -> str:
    raw = source.event_id in set(measure.memory.view.raw_event_ids)
    gist = source.event_id in set(measure.memory.view.gist_event_ids)
    if raw and gist:
        return "raw_and_gist"
    if raw:
        return "raw"
    if gist:
        return "gist"
    return "omitted"


def admit_stalled_operation(
    controller: Any,
    selection: StalledOperationSelection,
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
    """Admit the ledger, displacing only bounded gist-backed raw events."""

    if selection.status != "triggered":
        raise ValueError("Admission requires a triggered stalled-operation selection")
    cue = ledger_message(selection)
    receipt = selection_receipt(selection)
    receipt["prompt_cap"] = LEDGER_PROMPT_CAP
    standalone = controller._count([cue], ())
    receipt["standalone_tokens"] = standalone
    if standalone > LEDGER_PROMPT_CAP:
        receipt["status"] = "ledger_prompt_cap_legacy_cue_preserved"
        return StalledOperationAdmission(False, measure, None, receipt)

    mandatory = set(mandatory_event_ids)
    protected = set(protected_event_ids)
    original_raw = set(measure.memory.view.raw_event_ids)
    gist = tuple(measure.memory.view.gist_event_ids)
    gist_set = set(gist)
    attempts: list[dict[str, Any]] = []

    def try_raw(
        raw_event_ids: set[str], displacement: str
    ) -> tuple[Any, Any]:
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
        ledger_tokens = candidate.raw_prompt_tokens - base.raw_prompt_tokens
        if ledger_tokens <= 0:
            raise RuntimeError("A nonempty stalled-operation ledger added no tokens")
        displaced_tokens = measure.raw_prompt_tokens - base.raw_prompt_tokens
        budget = min(
            controller.policy_config.history_budget_bytes,
            controller.policy_config.workspace_budget_bytes,
        )
        updated = copy.deepcopy(receipt)
        updated.update(
            status="admitted",
            admission_attempts=copy.deepcopy(attempts),
            incremental_raw_tokens=ledger_tokens,
            displaced_raw_tokens=displaced_tokens,
            net_raw_token_change=candidate.raw_prompt_tokens
            - measure.raw_prompt_tokens,
            extra_bytes=ledger_tokens * controller.kv_bytes_per_token,
            displaced_raw_bytes=displaced_tokens * controller.kv_bytes_per_token,
            candidate_active_history_bytes=controller._max_history_bytes(candidate),
            budget_bytes=budget,
            demoted_raw_reserve_event_id=reserve_event_id,
            demoted_lexical_event_id=lexical_event_id,
        )
        updated["source_receipts"] = [
            {
                **source_receipt,
                "final_representation": _source_representation(source, candidate),
            }
            for source_receipt, source in zip(updated["source_receipts"], selection.sources)
        ]
        return StalledOperationAdmission(
            True,
            candidate,
            cue,
            updated,
            reserve_event_id,
            lexical_event_id,
        )

    current_raw = set(original_raw)
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
        base, candidate = try_raw(current_raw, "one_lowest_priority_lexical_raw")
        if result := admitted(
            base,
            candidate,
            reserve_event_id=demoted_reserve,
            lexical_event_id=lexical_candidate,
        ):
            return result

    receipt.update(
        status="ledger_over_budget_legacy_cue_preserved",
        admission_attempts=attempts,
    )
    return StalledOperationAdmission(False, measure, None, receipt)


def enable_stalled_operation(controller: Any) -> Any:
    controller.stalled_operation_policy = STALLED_OPERATION_POLICY
    return controller


__all__ = [
    "EXACT_FAILURE_REPEAT_THRESHOLD",
    "LATER_OBSERVED_CALL_THRESHOLD",
    "LEDGER_PROMPT_CAP",
    "STALLED_OPERATION_POLICY",
    "STALLED_OPERATION_VERSION",
    "TOOL_FAILED_VARIANT_THRESHOLD",
    "StalledOperationAdmission",
    "StalledOperationSelection",
    "admit_stalled_operation",
    "enable_stalled_operation",
    "ledger_message",
    "select_stalled_operation",
    "selection_receipt",
]
