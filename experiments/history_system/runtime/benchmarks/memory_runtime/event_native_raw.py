"""Raw event-native inference controls with explicit omission semantics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from history_memory.events import EventStore
from history_memory.evidence import evidence_message
from history_memory.packing import (
    MemoryView,
    PackedMemory,
    PackingBudgetError,
    native_ids,
    pack_memory,
    select_view,
    visible_message,
)

from .event_native_policy import (
    CURRENT_INPUT_BASELINE,
    HISTORY_BUDGET_DEFINITION,
    POLICY_SOURCE_COMMIT,
    WORKSPACE_BUDGET_DEFINITION,
    _PackingConfig,
    _current_input_baseline_indices,
)


RAW_CONTROL_VERSION = "a-event-native-raw-control-v1"
_LAYOUTS = {
    "no_gist": "no-gist-native-v1",
    "full_original": "full-original-native-v1",
    "full_shared": "full-shared-duplicate-evidence-v1",
}
_BUDGETED_GIST_LAYOUT = "event-native-budgeted-gist-v1"
_ALWAYS_COMPRESS_GIST_LAYOUT = "event-native-always-compress-gist-v1"
_POLICY_FIELDS = (
    "mode",
    "history_budget_bytes",
    "workspace_budget_bytes",
    "lease_decisions",
    "max_retrieved_events",
    "kv_bytes_per_token",
    "source_commit",
    "history_budget_definition",
    "workspace_budget_definition",
    "current_input_baseline",
)


@dataclass(frozen=True)
class RuntimeMemoryView(MemoryView):
    """A runtime-only view where omitted events are neither raw nor encoded."""

    omitted_event_ids: tuple[str, ...] = ()
    mandatory_raw_event_ids: tuple[str, ...] = ()
    raw_control_layout: str = ""

    def validate(self, store: EventStore) -> None:
        if not isinstance(store, EventStore):
            raise TypeError("store must be an EventStore")
        if self.raw_control_layout not in {
            *_LAYOUTS.values(),
            _BUDGETED_GIST_LAYOUT,
            _ALWAYS_COMPRESS_GIST_LAYOUT,
        }:
            raise ValueError(f"Unsupported raw control layout: {self.raw_control_layout!r}")

        components = (
            self.gist_event_ids,
            self.raw_event_ids,
            self.evidence_event_ids,
            self.omitted_event_ids,
            self.mandatory_raw_event_ids,
        )
        if any(len(ids) != len(set(ids)) for ids in components):
            raise ValueError("Duplicate event within a runtime memory view component")

        raw = set(self.raw_event_ids)
        gist = set(self.gist_event_ids)
        omitted = set(self.omitted_event_ids)
        raw_gist_overlap = raw & gist
        if (
            (raw_gist_overlap and self.raw_control_layout != _ALWAYS_COMPRESS_GIST_LAYOUT)
            or raw & omitted
            or gist & omitted
        ):
            raise ValueError("Raw, gist, and omitted event sets must be disjoint")
        known = {event.event_id for event in store.events}
        if raw | gist | omitted != known:
            raise ValueError(
                "Runtime view must cover every visible event exactly once; "
                f"missing={known - (raw | gist | omitted)}, "
                f"unknown={(raw | gist | omitted) - known}"
            )
        evidence = set(self.evidence_event_ids)
        mandatory = set(self.mandatory_raw_event_ids)
        if not evidence <= raw:
            raise ValueError("Evidence events must be a subset of raw events")
        if not mandatory <= raw or mandatory & omitted:
            raise ValueError("Mandatory raw events must remain raw")
        for event_id in self.evidence_event_ids:
            event = store.event(event_id)
            if not event.complete or event.kind == "instruction":
                raise ValueError("Evidence events must be complete non-instruction events")

        # Reuse B's coverage and event-eligibility validation without exposing
        # omitted events to B's encoder path.  The always-compress layout may
        # deliberately keep the reserved gist for an event that is also
        # materialized as exact raw evidence.  Validate that event's gist
        # eligibility directly, then remove the overlap only from the
        # disjoint base view used for the remaining checks.
        if self.raw_control_layout == _ALWAYS_COMPRESS_GIST_LAYOUT:
            for event_id in raw_gist_overlap:
                event = store.event(event_id)
                if not event.complete or event.kind == "instruction":
                    raise ValueError(
                        "Only complete non-instruction events may overlap raw and gist"
                    )
            validation_gist = tuple(
                event_id for event_id in self.gist_event_ids if event_id not in raw
            )
        else:
            validation_gist = self.gist_event_ids
        base_validation = MemoryView(
            gist_event_ids=validation_gist + self.omitted_event_ids,
            raw_event_ids=self.raw_event_ids,
            evidence_event_ids=self.evidence_event_ids,
        )
        base_validation.validate(store)

        if self.gist_event_ids and self.raw_control_layout not in {
            _BUDGETED_GIST_LAYOUT,
            _ALWAYS_COMPRESS_GIST_LAYOUT,
        }:
            raise ValueError("Raw-only controls cannot contain gist events")
        if self.raw_control_layout == _LAYOUTS["full_original"]:
            if self.evidence_event_ids or self.omitted_event_ids:
                raise ValueError("full_original must retain every source without evidence")
        if self.raw_control_layout == _LAYOUTS["full_shared"] and self.omitted_event_ids:
            raise ValueError("full_shared must retain every original source")


@dataclass(frozen=True)
class PreparedRawControl:
    memory: PackedMemory
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _RawPolicy:
    history_budget_bytes: int
    workspace_budget_bytes: int
    kv_bytes_per_token: int


@dataclass(frozen=True)
class _Measurement:
    memory: PackedMemory
    full_tokens: int
    host_tokens: int
    baseline_tokens: int
    history_tokens: int
    history_bytes: int
    evidence_tokens: int
    evidence_bytes: int
    sequence_tokens: int


def render_raw_control_messages(
    store: EventStore,
    view: RuntimeMemoryView,
    *,
    include_evidence: bool = True,
) -> tuple[dict[str, Any], ...]:
    """Render the exact native messages for an A raw-control view.

    ``include_evidence=False`` removes only the shared packet.  For no-gist,
    evidence-owned originals remain excluded from both hosts.  For
    full-shared, every original source remains present in both hosts.
    """

    if not isinstance(include_evidence, bool):
        raise TypeError("include_evidence must be a boolean")
    view.validate(store)
    if view.raw_control_layout == _LAYOUTS["full_shared"]:
        native_events = set(view.raw_event_ids)
    else:
        native_events = set(view.raw_event_ids) - set(view.evidence_event_ids)
    native_indices = sorted(
        {
            index
            for event in store.events
            if event.event_id in native_events
            for index in event.source_indices
        }
    )
    messages = [visible_message(store.messages[index]) for index in native_indices]
    packet = (
        evidence_message(store, view.evidence_event_ids)
        if include_evidence
        else None
    )
    if packet is not None:
        prefix_length = 0
        while prefix_length < len(messages) and messages[prefix_length]["role"] == "system":
            prefix_length += 1
        messages.insert(prefix_length, packet)
    return tuple(messages)


def build_raw_control(
    store: EventStore,
    tokenizer: Any,
    *,
    packing: Mapping[str, Any],
    policy: Mapping[str, Any],
    mode: str,
    evidence_event_ids: Sequence[str] = (),
    max_new_tokens: int,
    tools: Sequence[Mapping[str, Any]] | None = None,
) -> PreparedRawControl:
    """Build one all-native A control without policy or lease state."""

    if not isinstance(store, EventStore):
        raise TypeError("store must be an EventStore")
    if not callable(getattr(tokenizer, "apply_chat_template", None)):
        raise TypeError("tokenizer must expose apply_chat_template")
    if mode not in _LAYOUTS:
        raise ValueError(f"mode must be one of {sorted(_LAYOUTS)!r}")
    packing_config = _PackingConfig.from_mapping(packing)
    policy_config = _parse_policy(policy)
    if not _positive_int(max_new_tokens):
        raise ValueError("max_new_tokens must be a positive integer")
    if max_new_tokens > packing_config.max_target_tokens:
        raise PackingBudgetError(
            f"Generation needs up to {max_new_tokens} tokens; budget is "
            f"{packing_config.max_target_tokens}"
        )
    tools_value = _validate_tools(tools)
    evidence_ids = _validate_evidence_ids(store, evidence_event_ids)
    if mode == "full_original" and evidence_ids:
        raise ValueError("full_original does not accept evidence_event_ids")
    static_raw = set(
        select_view(
            store, recent_tool_events=packing_config.recent_tool_events
        ).raw_event_ids
    )
    overlap = static_raw & set(evidence_ids)
    if overlap:
        raise ValueError(
            "Evidence events must be disjoint from the static mandatory raw view: "
            f"{sorted(overlap)!r}"
        )

    if mode == "no_gist":
        return _build_no_gist(
            store,
            tokenizer,
            packing_config,
            policy_config,
            evidence_ids,
            max_new_tokens,
            tools_value,
        )
    return _build_full(
        store,
        tokenizer,
        packing_config,
        policy_config,
        mode,
        evidence_ids,
        max_new_tokens,
        tools_value,
    )


def _build_no_gist(
    store: EventStore,
    tokenizer: Any,
    packing: _PackingConfig,
    policy: _RawPolicy,
    evidence_ids: tuple[str, ...],
    max_new_tokens: int,
    tools: Sequence[Mapping[str, Any]] | None,
) -> PreparedRawControl:
    static = select_view(store, recent_tool_events=packing.recent_tool_events)
    mandatory_set = set(static.raw_event_ids) | set(evidence_ids)
    mandatory_ids = _ordered_event_ids(store, mandatory_set)
    all_ids = tuple(event.event_id for event in store.events)
    view = RuntimeMemoryView(
        gist_event_ids=(),
        raw_event_ids=mandatory_ids,
        evidence_event_ids=evidence_ids,
        omitted_event_ids=tuple(
            event_id for event_id in all_ids if event_id not in mandatory_set
        ),
        mandatory_raw_event_ids=mandatory_ids,
        raw_control_layout=_LAYOUTS["no_gist"],
    )
    measurement = _measure(
        store, view, tokenizer, policy, max_new_tokens, tools
    )
    mandatory_workspace_tokens = len(measurement.memory.workspace_input_ids)
    _require_system_limit(measurement, packing)
    if mandatory_workspace_tokens > packing.max_workspace_tokens:
        raise PackingBudgetError(
            "Mandatory raw workspace needs "
            f"{mandatory_workspace_tokens} tokens; budget is "
            f"{packing.max_workspace_tokens}"
        )
    _require_no_gist_budgets(measurement, packing, policy)

    candidate_ids = tuple(
        event.event_id
        for event in sorted(
            store.events,
            key=lambda event: max(event.source_indices),
            reverse=True,
        )
        if event.event_id not in mandatory_set
    )
    refilled: list[str] = []
    skipped: list[dict[str, Any]] = []
    for event_id in candidate_ids:
        candidate_raw = set(view.raw_event_ids)
        candidate_raw.add(event_id)
        candidate = replace(
            view,
            raw_event_ids=_ordered_event_ids(store, candidate_raw),
            omitted_event_ids=tuple(
                item for item in view.omitted_event_ids if item != event_id
            ),
        )
        candidate_measurement = _measure(
            store, candidate, tokenizer, policy, max_new_tokens, tools
        )
        reasons = _budget_reasons(candidate_measurement, packing, policy)
        if reasons:
            skipped.append(
                {
                    "event_id": event_id,
                    "source_indices": list(store.event(event_id).source_indices),
                    "reasons": reasons,
                    "history_bytes": candidate_measurement.history_bytes,
                    "evidence_bytes": candidate_measurement.evidence_bytes,
                    "sequence_tokens": candidate_measurement.sequence_tokens,
                }
            )
            continue
        view = candidate
        measurement = candidate_measurement
        refilled.append(event_id)

    metadata = _metadata(
        store,
        view,
        measurement,
        packing,
        policy,
        mode="no_gist",
        max_new_tokens=max_new_tokens,
    )
    metadata.update(
        {
            "refill_candidate_event_ids": list(candidate_ids),
            "refilled_event_ids": refilled,
            "skipped_refill_events": skipped,
            "mandatory_workspace_tokens": mandatory_workspace_tokens,
            "workspace_refill_may_exceed_training_cap": True,
        }
    )
    return PreparedRawControl(measurement.memory, metadata)


def _build_full(
    store: EventStore,
    tokenizer: Any,
    packing: _PackingConfig,
    policy: _RawPolicy,
    mode: str,
    evidence_ids: tuple[str, ...],
    max_new_tokens: int,
    tools: Sequence[Mapping[str, Any]] | None,
) -> PreparedRawControl:
    all_ids = tuple(event.event_id for event in store.events)
    view = RuntimeMemoryView(
        gist_event_ids=(),
        raw_event_ids=all_ids,
        evidence_event_ids=evidence_ids,
        omitted_event_ids=(),
        mandatory_raw_event_ids=all_ids,
        raw_control_layout=_LAYOUTS[mode],
    )
    measurement = _measure(store, view, tokenizer, policy, max_new_tokens, tools)
    _require_system_limit(measurement, packing)
    _require_sequence_limit(measurement, packing)
    if mode == "full_shared" and measurement.evidence_bytes > policy.workspace_budget_bytes:
        raise PackingBudgetError(
            "workspace evidence packet needs "
            f"{measurement.evidence_bytes} planned KV bytes; budget is "
            f"{policy.workspace_budget_bytes}"
        )
    metadata = _metadata(
        store,
        view,
        measurement,
        packing,
        policy,
        mode=mode,
        max_new_tokens=max_new_tokens,
    )
    metadata["workspace_refill_may_exceed_training_cap"] = False
    return PreparedRawControl(measurement.memory, metadata)


def _measure(
    store: EventStore,
    view: RuntimeMemoryView,
    tokenizer: Any,
    policy: _RawPolicy,
    max_new_tokens: int,
    tools: Sequence[Mapping[str, Any]] | None,
) -> _Measurement:
    memory = _pack_raw_control(store, view, tokenizer, tools)
    rendered = render_raw_control_messages(store, view)
    without_packet = render_raw_control_messages(
        store, view, include_evidence=False
    )
    full_tokens = len(native_ids(tokenizer, rendered, tools=tools, generation=True))
    host_tokens = len(
        native_ids(tokenizer, without_packet, tools=tools, generation=True)
    )
    if full_tokens < host_tokens:
        raise ValueError("Adding the evidence packet reduced native token length")
    baseline_indices = _current_input_baseline_indices(store)
    baseline_messages = tuple(
        visible_message(store.messages[index]) for index in baseline_indices
    )
    baseline_tokens = (
        len(native_ids(tokenizer, baseline_messages, tools=tools, generation=True))
        if baseline_messages
        else 0
    )
    history_tokens = max(0, full_tokens - baseline_tokens)
    evidence_tokens = full_tokens - host_tokens
    resident_tokens = len(memory.system_input_ids) + len(memory.workspace_input_ids)
    if resident_tokens != full_tokens:
        raise RuntimeError("Packed raw tokens disagree with the rendered native request")
    return _Measurement(
        memory=memory,
        full_tokens=full_tokens,
        host_tokens=host_tokens,
        baseline_tokens=baseline_tokens,
        history_tokens=history_tokens,
        history_bytes=history_tokens * policy.kv_bytes_per_token,
        evidence_tokens=evidence_tokens,
        evidence_bytes=evidence_tokens * policy.kv_bytes_per_token,
        sequence_tokens=resident_tokens + max_new_tokens,
    )


def _pack_raw_control(
    store: EventStore,
    view: RuntimeMemoryView,
    tokenizer: Any,
    tools: Sequence[Mapping[str, Any]] | None,
) -> PackedMemory:
    if view.raw_control_layout != _LAYOUTS["full_shared"]:
        return pack_memory(
            store,
            view,
            tokenizer,
            tools=tools,
        )

    base_full_view = replace(
        view,
        evidence_event_ids=(),
        raw_control_layout=_LAYOUTS["full_original"],
    )
    base_full = pack_memory(
        store,
        base_full_view,
        tokenizer,
        tools=tools,
    )
    final_ids = native_ids(
        tokenizer,
        render_raw_control_messages(store, view),
        tools=tools,
        generation=True,
    )
    prefix = base_full.system_input_ids
    if final_ids[: len(prefix)] != prefix:
        raise ValueError("Native system/tools prefix changed after adding evidence")
    return replace(
        base_full,
        view=view,
        workspace_input_ids=final_ids[len(prefix) :],
    )


def _metadata(
    store: EventStore,
    view: RuntimeMemoryView,
    measurement: _Measurement,
    packing: _PackingConfig,
    policy: _RawPolicy,
    *,
    mode: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    actual_costs = {
        "system_tokens": len(measurement.memory.system_input_ids),
        "raw_tokens": len(measurement.memory.workspace_input_ids),
        "gist_tokens": 0,
        "resident_kv_tokens": measurement.full_tokens,
        "max_new_tokens": max_new_tokens,
        "sequence_tokens": measurement.sequence_tokens,
        "fixed_baseline_tokens": measurement.baseline_tokens,
        "history_tokens": measurement.history_tokens,
        "history_bytes": measurement.history_bytes,
        "evidence_host_tokens": measurement.host_tokens,
        "evidence_tokens": measurement.evidence_tokens,
        "evidence_bytes": measurement.evidence_bytes,
    }
    return {
        "raw_control_version": RAW_CONTROL_VERSION,
        "session_id": store.session_id,
        "mode": mode,
        "raw_control_layout": view.raw_control_layout,
        "raw_event_ids": list(view.raw_event_ids),
        "gist_event_ids": [],
        "omitted_event_ids": list(view.omitted_event_ids),
        "mandatory_raw_event_ids": list(view.mandatory_raw_event_ids),
        "evidence_event_ids": list(view.evidence_event_ids),
        "shared_evidence_event_ids": list(view.evidence_event_ids),
        "raw_source_indices": list(measurement.memory.raw_source_indices),
        "full_source_indices": list(range(len(store.messages))),
        "full_source_coverage": not view.omitted_event_ids,
        "static_raw_event_ids": list(
            select_view(
                store, recent_tool_events=packing.recent_tool_events
            ).raw_event_ids
        ),
        "fixed_baseline_source_indices": list(
            _current_input_baseline_indices(store)
        ),
        "fixed_baseline_tokens": measurement.baseline_tokens,
        "system_tokens": actual_costs["system_tokens"],
        "raw_tokens": actual_costs["raw_tokens"],
        "gist_tokens": 0,
        "resident_kv_tokens": measurement.full_tokens,
        "max_new_tokens": max_new_tokens,
        "sequence_tokens": measurement.sequence_tokens,
        "history_tokens": measurement.history_tokens,
        "history_bytes": measurement.history_bytes,
        "evidence_tokens": measurement.evidence_tokens,
        "evidence_bytes": measurement.evidence_bytes,
        "actual_costs": actual_costs,
        "kv_bytes_per_token": policy.kv_bytes_per_token,
        "history_budget_bytes": policy.history_budget_bytes,
        "workspace_budget_bytes": policy.workspace_budget_bytes,
        "max_system_tokens": packing.max_system_tokens,
        "max_workspace_tokens": packing.max_workspace_tokens,
        "max_sequence_tokens": packing.max_sequence_tokens,
        "history_budget_definition": HISTORY_BUDGET_DEFINITION,
        "workspace_budget_definition": WORKSPACE_BUDGET_DEFINITION,
        "current_input_baseline": CURRENT_INPUT_BASELINE,
        "accounting_scope": "actual final native tokenization and checkpoint KV geometry",
        "training_parity": False,
        "training_parity_note": (
            "A-only runtime raw control; not B event-native training parity"
        ),
    }


def _require_system_limit(
    measurement: _Measurement, packing: _PackingConfig
) -> None:
    system_tokens = len(measurement.memory.system_input_ids)
    if system_tokens > packing.max_system_tokens:
        raise PackingBudgetError(
            f"System/tools need {system_tokens} tokens; budget is "
            f"{packing.max_system_tokens}"
        )


def _require_sequence_limit(
    measurement: _Measurement, packing: _PackingConfig
) -> None:
    if measurement.sequence_tokens > packing.max_sequence_tokens:
        raise PackingBudgetError(
            "sequence needs "
            f"{measurement.sequence_tokens} tokens; budget is "
            f"{packing.max_sequence_tokens}"
        )


def _require_no_gist_budgets(
    measurement: _Measurement,
    packing: _PackingConfig,
    policy: _RawPolicy,
) -> None:
    reasons = _budget_reasons(measurement, packing, policy)
    if "history_budget" in reasons:
        raise PackingBudgetError(
            f"Raw history needs {measurement.history_bytes} planned KV bytes; "
            f"budget is {policy.history_budget_bytes}"
        )
    if "workspace_budget" in reasons:
        raise PackingBudgetError(
            "workspace evidence packet needs "
            f"{measurement.evidence_bytes} planned KV bytes; budget is "
            f"{policy.workspace_budget_bytes}"
        )
    _require_sequence_limit(measurement, packing)


def _budget_reasons(
    measurement: _Measurement,
    packing: _PackingConfig,
    policy: _RawPolicy,
) -> list[str]:
    reasons = []
    if measurement.history_bytes > policy.history_budget_bytes:
        reasons.append("history_budget")
    if measurement.evidence_bytes > policy.workspace_budget_bytes:
        reasons.append("workspace_budget")
    if measurement.sequence_tokens > packing.max_sequence_tokens:
        reasons.append("sequence_budget")
    return reasons


def _parse_policy(value: Mapping[str, Any]) -> _RawPolicy:
    if not isinstance(value, Mapping):
        raise TypeError("policy must be a mapping")
    missing = [name for name in _POLICY_FIELDS if name not in value]
    if missing:
        raise ValueError(f"policy lacks required fields: {missing!r}")
    for name in (
        "history_budget_bytes",
        "workspace_budget_bytes",
        "lease_decisions",
        "max_retrieved_events",
    ):
        if not _nonnegative_int(value[name]):
            raise ValueError(f"policy.{name} must be a nonnegative integer")
    if not _positive_int(value["kv_bytes_per_token"]):
        raise ValueError("policy.kv_bytes_per_token must be a positive integer")
    expected = {
        "source_commit": POLICY_SOURCE_COMMIT,
        "history_budget_definition": HISTORY_BUDGET_DEFINITION,
        "workspace_budget_definition": WORKSPACE_BUDGET_DEFINITION,
        "current_input_baseline": CURRENT_INPUT_BASELINE,
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(
                f"policy.{name}={value[name]!r}; expected {expected_value!r}"
            )
    return _RawPolicy(
        history_budget_bytes=value["history_budget_bytes"],
        workspace_budget_bytes=value["workspace_budget_bytes"],
        kv_bytes_per_token=value["kv_bytes_per_token"],
    )


def _validate_evidence_ids(
    store: EventStore, value: Sequence[str]
) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError("evidence_event_ids must be a sequence of strings")
    if any(not isinstance(event_id, str) or not event_id for event_id in value):
        raise ValueError("evidence_event_ids must contain nonempty strings")
    if len(value) != len(set(value)):
        raise ValueError("Duplicate evidence event")
    selected = set(value)
    ordered = _ordered_event_ids(store, selected)
    if len(ordered) != len(selected):
        unknown = selected - {event.event_id for event in store.events}
        raise ValueError(f"Evidence event is not in the visible prefix: {sorted(unknown)!r}")
    for event_id in ordered:
        event = store.event(event_id)
        if not event.complete or event.kind == "instruction":
            raise ValueError("Evidence events must be complete non-instruction events")
    return ordered


def _validate_tools(
    tools: Sequence[Mapping[str, Any]] | None,
) -> Sequence[Mapping[str, Any]] | None:
    if tools is None:
        return None
    if (
        not isinstance(tools, Sequence)
        or isinstance(tools, (str, bytes, bytearray))
        or any(not isinstance(tool, Mapping) for tool in tools)
    ):
        raise TypeError("tools must be a sequence of mappings or None")
    return tuple(dict(tool) for tool in tools)


def _ordered_event_ids(
    store: EventStore, selected: set[str]
) -> tuple[str, ...]:
    return tuple(
        event.event_id for event in store.events if event.event_id in selected
    )


def _positive_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _nonnegative_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


__all__ = [
    "PreparedRawControl",
    "RuntimeMemoryView",
    "build_raw_control",
    "render_raw_control_messages",
]
