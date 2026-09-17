"""Pure CPU packing shared by training and serving adapters.

Event extraction uses a stable, typed envelope because native Qwen templates
omit call IDs. The raw workspace uses the native chat template. This module
plans token and RoPE positions; it does not extract, rotate or cache tensors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .encoding_scope import (
    event_record_spans,
    plan_encoding_scope,
    record_common_spans,
    structural_tool_result_record_spans,
    validate_encoding_scope,
)
from .events import EventStore, Message

PACKING_VERSION = "history-event-v1"
RAW_LAYOUT_PROFILE = "event-native-evidence-v1"


class PackingBudgetError(ValueError):
    """The complete representation cannot fit; no content was truncated."""


class EncodingScopeCapacityError(PackingBudgetError):
    """One atomic encoder unit exceeds the declared model/extractor capacity."""


@dataclass(frozen=True)
class MemoryView:
    gist_event_ids: tuple[str, ...]
    raw_event_ids: tuple[str, ...]
    evidence_event_ids: tuple[str, ...] = ()

    def validate(self, store: EventStore) -> None:
        known = {event.event_id for event in store.events}
        for ids in (self.gist_event_ids, self.raw_event_ids, self.evidence_event_ids):
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate event within a memory view component")
        selected = set(self.gist_event_ids) | set(self.raw_event_ids)
        if selected != known:
            raise ValueError(f"View must cover exactly the visible events; missing={known - selected}, unknown={selected - known}")
        if not set(self.evidence_event_ids) <= set(self.raw_event_ids):
            raise ValueError("Evidence events must be a subset of raw events")
        if any(store.event(event_id).kind == "instruction" for event_id in self.evidence_event_ids):
            raise ValueError("System instructions must remain in the native prefix")
        for event_id in self.gist_event_ids:
            event = store.event(event_id)
            if not event.complete or event.kind == "instruction":
                raise ValueError(f"Incomplete events and instructions must remain raw: {event_id}")


def select_view(
    store: EventStore, *, recent_tool_events: int = 1,
    restored_event_ids: Iterable[str] = (), pinned_event_ids: Iterable[str] = (),
) -> MemoryView:
    """A fixed recent workspace plus caller-selected observable evidence.

    The current user request, latest message's event, instructions, pending
    calls and the most recently COMPLETED tool events are protected. A owns
    the rules for additional live bindings and persistent evidence leases.
    """
    if recent_tool_events < 0:
        raise ValueError("recent_tool_events must be nonnegative")
    extra = set(restored_event_ids) | set(pinned_event_ids)
    for event_id in extra:
        store.event(event_id)
    raw = {event.event_id for event in store.events if not event.complete or event.kind == "instruction"}
    users = [event for event in store.events if event.kind == "user"]
    if users:
        raw.add(users[-1].event_id)
    if store.events:
        raw.add(max(store.events, key=lambda event: max(event.source_indices)).event_id)
    completed = sorted(
        (event for event in store.events if event.kind == "tool_event" and event.complete),
        key=lambda event: max(event.source_indices),
    )
    if recent_tool_events:
        raw.update(event.event_id for event in completed[-recent_tool_events:])
    evidence = extra - raw
    raw.update(extra)
    view = MemoryView(
        gist_event_ids=tuple(event.event_id for event in store.events if event.event_id not in raw),
        raw_event_ids=tuple(event.event_id for event in store.events if event.event_id in raw),
        evidence_event_ids=tuple(event.event_id for event in store.events if event.event_id in evidence),
    )
    view.validate(store)
    return view


def visible_message(message: Message | Mapping[str, Any]) -> dict[str, Any]:
    """Keep model-visible fields; parse JSON arguments exactly once.

    Source snapshots retain their original argument strings. A valid JSON
    object becomes an object for templating, avoiding double encoding. Invalid
    historical arguments remain verbatim so error-recovery traces are usable.
    Non-text content requires a source adapter instead of silently vanishing
    in a text-only template. Metadata such as reward is not model input.
    """
    source = message.to_dict() if isinstance(message, Message) else dict(message)
    content = source.get("content")
    if content is not None and not isinstance(content, str):
        raise ValueError("Non-text content requires an explicit source adapter")
    if source["role"] == "developer":
        raise ValueError("The Qwen text packing profile requires developer-role adaptation")
    result = {key: source[key] for key in ("role", "content", "name", "tool_call_id") if key in source}
    if source.get("tool_calls"):
        calls = []
        for call in source["tool_calls"]:
            function = call["function"]
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    decoded = json.loads(arguments)
                    if isinstance(decoded, dict):
                        arguments = decoded
                except json.JSONDecodeError:
                    pass
            calls.append({
                "id": call["id"], "type": call.get("type", "function"),
                "function": {"name": function["name"], "arguments": arguments},
            })
        result["tool_calls"] = calls
    return result


def native_ids(tokenizer: Any, messages: Sequence[Mapping[str, Any]], *, tools=None, generation=False) -> tuple[int, ...]:
    if not messages:
        raise ValueError("The native text template needs at least one message")
    ids = tokenizer.apply_chat_template(
        list(messages), tools=tools, tokenize=True,
        add_generation_prompt=generation, enable_thinking=False,
        truncation=False,
    )
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    return tuple(int(token) for token in ids)


def event_encoder_messages(store: EventStore, event_id: str) -> tuple[dict[str, Any], ...]:
    """Content depends only on this completed event, never its current view."""
    event = store.event(event_id)
    if not event.complete or event.kind == "instruction":
        raise ValueError("Only complete history events can be independently encoded")
    envelope = {
        "type": "history_event", "kind": event.kind,
        "messages": [visible_message(message) for message in store.event_messages(event_id)],
    }
    return ({"role": "user", "content": json.dumps(envelope, ensure_ascii=False, separators=(",", ":"), allow_nan=False)},)


def raw_workspace_messages(store: EventStore, view: MemoryView) -> tuple[dict[str, Any], ...]:
    """Place A's shared evidence packet before the native current workspace.

    Evidence-owned messages appear only inside that packet. System messages
    remain at the front. The source store and independent event encodings are
    unaffected by this presentation choice.
    """
    # The shared evidence module imports visible_message from this module.
    from .evidence import evidence_message

    view.validate(store)
    native_events = set(view.raw_event_ids) - set(view.evidence_event_ids)
    native_indices = sorted({index for event in store.events if event.event_id in native_events for index in event.source_indices})
    messages = [visible_message(store.messages[index]) for index in native_indices]
    evidence = evidence_message(store, view.evidence_event_ids)
    if evidence is not None:
        prefix_length = 0
        while prefix_length < len(messages) and messages[prefix_length]["role"] == "system":
            prefix_length += 1
        messages.insert(prefix_length, evidence)
    return tuple(messages)


@dataclass(frozen=True)
class EncoderChunk:
    event_id: str
    part_index: int
    source_indices: tuple[int, ...]
    source_token_start: int
    source_token_end: int
    token_ids: tuple[int, ...]

    def encoding_key(self, *, parameter_version: str | int, ratio: int) -> tuple[Any, ...]:
        """An exact-input key; the caller advances version after optimizer.step.

        Runtime RoPE offsets are deliberately excluded: cached gist keys must
        be pre-RoPE, then rotated for the current layout by the model adapter.
        """
        if ratio <= 0:
            raise ValueError("ratio must be positive")
        return (PACKING_VERSION, parameter_version, ratio, self.token_ids)


def encode_event_chunks(
    store: EventStore, event_id: str, tokenizer: Any, *, max_chunk_tokens: int = 768,
    chunk_overlap: int = 64,
) -> tuple[EncoderChunk, ...]:
    if max_chunk_tokens <= 0 or not 0 <= chunk_overlap < max_chunk_tokens:
        raise ValueError("Require max_chunk_tokens > chunk_overlap >= 0")
    ids = native_ids(tokenizer, event_encoder_messages(store, event_id))
    chunks = []
    start = 0
    while start < len(ids):
        end = min(start + max_chunk_tokens, len(ids))
        chunks.append(EncoderChunk(event_id, len(chunks), store.event(event_id).source_indices, start, end, ids[start:end]))
        if end == len(ids):
            break
        start = end - chunk_overlap
    return tuple(chunks)


def encode_scope_chunks(
    store: EventStore,
    event_ids: Iterable[str],
    tokenizer: Any,
    *,
    encoding_scope: str,
    max_chunk_tokens: int = 768,
    chunk_overlap: int = 64,
    atomic_unit_token_limit: int | None = None,
    event_groups: Sequence[Sequence[str]] | None = None,
) -> tuple[EncoderChunk, ...]:
    """Tokenize the actual encoder calls for one G scope.

    ``current`` and source-bound fallbacks retain the frozen overlapping chunk
    behavior. Other scopes make exactly one chunk per atomic unit. Such units
    are never split; callers must provide the actual model/extractor limit
    when it is known.
    """

    scope = validate_encoding_scope(encoding_scope)
    selected = tuple(event_ids)
    selected_set = set(selected)
    if scope == "current":
        return tuple(
            chunk
            for event in store.events
            if event.event_id in selected_set
            for chunk in encode_event_chunks(
                store,
                event.event_id,
                tokenizer,
                max_chunk_tokens=max_chunk_tokens,
                chunk_overlap=chunk_overlap,
            )
        )
    if atomic_unit_token_limit is not None and (
        isinstance(atomic_unit_token_limit, bool)
        or not isinstance(atomic_unit_token_limit, int)
        or atomic_unit_token_limit <= 0
    ):
        raise ValueError("atomic_unit_token_limit must be a positive integer or None")

    if event_groups is None:
        plan = plan_encoding_scope(store, selected, scope)
        groups = plan.event_groups
        if plan.pending_event_ids:
            raise ValueError(
                "Pending encoding-scope events must remain raw: "
                f"{plan.pending_event_ids!r}"
            )
    else:
        groups = _validate_scope_groups(store, selected, event_groups, scope)

    chunks = []
    for group in groups:
        if (
            scope in {"record_bound", "record_bound_structural"}
            and not (
                structural_tool_result_record_spans(store, group[0])
                if scope == "record_bound_structural"
                else event_record_spans(store, group[0])
            )
        ):
            chunks.extend(
                encode_event_chunks(
                    store,
                    group[0],
                    tokenizer,
                    max_chunk_tokens=max_chunk_tokens,
                    chunk_overlap=chunk_overlap,
                )
            )
            continue
        for unit_id, source_indices, messages in _atomic_encoder_units(
            store, group, scope
        ):
            ids = native_ids(tokenizer, messages)
            if atomic_unit_token_limit is not None and len(ids) > atomic_unit_token_limit:
                raise EncodingScopeCapacityError(
                    f"Atomic {scope} encoder unit {unit_id!r} needs {len(ids)} tokens; "
                    f"model/extractor capacity is {atomic_unit_token_limit}"
                )
            chunks.append(
                EncoderChunk(
                    unit_id,
                    0,
                    source_indices,
                    0,
                    len(ids),
                    ids,
                )
            )
    return tuple(chunks)


def _validate_scope_groups(store, event_ids, event_groups, scope):
    selected = set(event_ids)
    groups = tuple(tuple(group) for group in event_groups)
    flattened = tuple(event_id for group in groups for event_id in group)
    if len(flattened) != len(set(flattened)) or set(flattened) != selected:
        raise ValueError("encoding event groups must cover selected events exactly once")
    expected_size = 2 if scope == "adjacent_pair" else 1
    if any(len(group) != expected_size for group in groups):
        raise ValueError(f"{scope} encoding groups must contain {expected_size} event(s)")
    expected = plan_encoding_scope(store, flattened, scope).event_groups
    if groups != expected:
        raise ValueError("encoding event groups must follow visible source order")
    return groups


def _atomic_encoder_units(store, group, scope):
    if scope == "event":
        event_id = group[0]
        event = store.event(event_id)
        return ((event_id, event.source_indices, event_encoder_messages(store, event_id)),)
    if scope == "adjacent_pair":
        events = [store.event(event_id) for event_id in group]
        envelope = {
            "type": "history_event_pair",
            "events": [
                {
                    "kind": event.kind,
                    "messages": [
                        visible_message(message)
                        for message in store.event_messages(event.event_id)
                    ],
                }
                for event in events
            ],
        }
        unit_id = (
            f"{store.session_id}:pair:"
            f"{min(events[0].source_indices)}-{min(events[1].source_indices)}"
        )
        source_indices = tuple(
            index for event in events for index in event.source_indices
        )
        messages = ({
            "role": "user",
            "content": json.dumps(
                envelope,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ),
        },)
        return ((unit_id, source_indices, messages),)
    if scope in {"record_bound", "record_bound_structural"}:
        event_id = group[0]
        records = (
            structural_tool_result_record_spans(store, event_id)
            if scope == "record_bound_structural"
            else event_record_spans(store, event_id)
        )
        return _bound_record_encoder_units(store, event_id, records, scope)
    if scope != "record":
        raise ValueError(f"Unsupported atomic encoding scope: {scope!r}")

    event_id = group[0]
    event = store.event(event_id)
    structural = []
    for source_index in event.source_indices:
        message = visible_message(store.messages[source_index])
        if message.get("tool_calls"):
            structural.append((source_index, _tool_call_context(message)))
    units = []
    for record_index, record in enumerate(event_record_spans(store, event_id)):
        record_message = visible_message(store.messages[record.source_index])
        span = record.span
        if record.container_path == ("content",):
            if record_message.get("tool_calls"):
                record_message = _tool_call_context(record_message)
            record_message["content"] = span.text
        else:
            call_index = record.container_path[1]
            arguments = span.text
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass
            record_message = _tool_call_context(record_message)
            selected_context = record_message["tool_calls"][call_index]
            selected_context["function"]["arguments"] = arguments
            record_message["tool_calls"] = [selected_context]
            record_message["content"] = None
        messages = [
            message for index, message in structural if index != record.source_index
        ]
        messages.append(record_message)
        field_path = (*record.container_path, *span.field_path)
        envelope = {
            "type": "history_record",
            "kind": event.kind,
            "field_path": list(field_path),
            "messages": messages,
        }
        sibling_context = _record_sibling_context(store, record)
        if sibling_context:
            envelope["sibling_context"] = sibling_context
        path = "/".join(
            str(part).replace("~", "~0").replace("/", "~1")
            for part in field_path
        )
        unit_id = (
            f"{event_id}:record:{record.source_index}:"
            f"{path or 'root'}:{record_index}"
        )
        provenance = tuple(
            dict.fromkeys([*(index for index, _ in structural), record.source_index])
        )
        units.append((
            unit_id,
            provenance,
            ({
                "role": "user",
                "content": json.dumps(
                    envelope,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            },),
        ))
    return tuple(units)


def _bound_record_encoder_units(store, event_id, records, scope):
    """Build source-only record inputs with exact producer and parent binding."""

    units = []
    for record_index, record in enumerate(records):
        source = store.messages[record.source_index].to_dict()
        field_path = (*record.container_path, *record.span.field_path)
        source_header = {
            key: source[key]
            for key in ("role", "name", "tool_call_id")
            if key in source
        }
        binding = {
            "message_header": source_header,
            "path": list(field_path),
        }
        producer = _original_producer_call(store, event_id, record)
        provenance = [record.source_index]
        if producer is not None:
            producer_index, producer_call = producer
            binding["producer_call"] = producer_call
            provenance.insert(0, producer_index)

        common_header = []
        container_text = _record_container_text(source, record.container_path)
        for common in record_common_spans(container_text, record.span):
            common_header.append(
                {
                    "path": [*record.container_path, *common.field_path],
                    "text": common.text,
                }
            )
        if common_header:
            binding["common_header"] = common_header

        envelope = {
            "type": "history_record_bound",
            "binding": binding,
            "record_text": record.span.text,
        }
        path = "/".join(
            str(part).replace("~", "~0").replace("/", "~1")
            for part in field_path
        )
        unit_id = (
            f"{event_id}:{scope}:{record.source_index}:"
            f"{path or 'root'}:{record_index}"
        )
        units.append(
            (
                unit_id,
                tuple(dict.fromkeys(provenance)),
                ({
                    "role": "user",
                    "content": json.dumps(
                        envelope,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                },),
            )
        )
    return tuple(units)


def _record_container_text(source, container_path):
    value = source
    for part in container_path:
        value = value[part]
    if not isinstance(value, str):
        raise ValueError("Record container is no longer an exact source string")
    return value


def _original_producer_call(store, event_id, record):
    source = store.messages[record.source_index].to_dict()
    if (
        len(record.container_path) >= 2
        and record.container_path[0] == "tool_calls"
        and isinstance(record.container_path[1], int)
    ):
        calls = source.get("tool_calls")
        call_index = record.container_path[1]
        if isinstance(calls, list) and 0 <= call_index < len(calls):
            call = calls[call_index]
            if isinstance(call, dict):
                return record.source_index, call
        return None

    call_id = source.get("tool_call_id")
    if source.get("role") != "tool" or not isinstance(call_id, str):
        return None
    for source_index in store.event(event_id).source_indices:
        message = store.messages[source_index].to_dict()
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if isinstance(call, dict) and call.get("id") == call_id:
                return source_index, call
    return None


def _tool_call_context(message):
    context = {
        key: value
        for key, value in message.items()
        if key not in {"content", "tool_calls"}
    }
    context["content"] = None
    context["tool_calls"] = [
        {
            "id": call["id"],
            "type": call.get("type", "function"),
            "function": {"name": call["function"]["name"]},
        }
        for call in message["tool_calls"]
    ]
    return context


def _record_sibling_context(store, record):
    span_path = record.span.field_path
    if (
        len(span_path) != 2
        or not isinstance(span_path[0], str)
        or not isinstance(span_path[1], int)
    ):
        return None
    message = store.messages[record.source_index].to_dict()
    value = message
    for part in record.container_path:
        value = value[part]
    try:
        parent = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(parent, dict):
        return None
    record_fields = {
        key
        for key, child in parent.items()
        if isinstance(child, list)
        and child
        and all(isinstance(item, dict) for item in child)
    }
    return {key: child for key, child in parent.items() if key not in record_fields}


@dataclass(frozen=True)
class GistPlacement:
    chunk: EncoderChunk
    source_position_start: int
    position_ids: tuple[int, ...]


@dataclass(frozen=True)
class PackedMemory:
    view: MemoryView
    system_input_ids: tuple[int, ...]
    workspace_input_ids: tuple[int, ...]
    raw_source_indices: tuple[int, ...]
    chunks: tuple[EncoderChunk, ...]
    raw_layout_profile: str = RAW_LAYOUT_PROFILE

    def gist_layout(self, ratio: int) -> tuple[GistPlacement, ...]:
        """Match dynamic-interleave's source-span RoPE, not compressed offsets."""
        if ratio <= 0:
            raise ValueError("ratio must be positive")
        offset = len(self.system_input_ids)
        placements = []
        for chunk in self.chunks:
            length = len(chunk.token_ids)
            positions = tuple(offset + min(start + ratio, length) - 1 for start in range(0, length, ratio))
            placements.append(GistPlacement(chunk, offset, positions))
            offset += length
        return tuple(placements)

    @property
    def workspace_position_start(self) -> int:
        return len(self.system_input_ids) + sum(len(chunk.token_ids) for chunk in self.chunks)

    def costs(self, ratio: int) -> dict[str, int]:
        gist_tokens = sum(len(placement.position_ids) for placement in self.gist_layout(ratio))
        presented = sum(len(chunk.token_ids) for chunk in self.chunks)
        unique = sum(
            max(chunk.source_token_end for chunk in self.chunks if chunk.event_id == event_id)
            for event_id in {chunk.event_id for chunk in self.chunks}
        )
        return {
            "system_tokens": len(self.system_input_ids), "raw_tokens": len(self.workspace_input_ids),
            "gist_tokens": gist_tokens, "presented_encoder_tokens": presented,
            "encoder_overlap_tokens": presented - unique,
            "resident_kv_tokens": len(self.system_input_ids) + len(self.workspace_input_ids) + gist_tokens,
            "raw_gist_overlap_events": len(set(self.view.raw_event_ids) & set(self.view.gist_event_ids)),
        }

    def causal_mask(self, ratio: int, *, target_tokens: int = 0) -> tuple[tuple[bool, ...], ...]:
        """Small CPU reference: ordinary queries see all memory and causal raw.

        Tensor adapters should materialize an equivalent efficient mask. Event
        extraction remains separate per chunk; no event sees another event.
        """
        if target_tokens < 0:
            raise ValueError("target_tokens must be nonnegative")
        past = len(self.system_input_ids) + sum(len(p.position_ids) for p in self.gist_layout(ratio))
        ordinary = len(self.workspace_input_ids) + target_tokens
        return tuple(tuple(key < past + query + 1 for key in range(past + ordinary)) for query in range(ordinary))


def pack_memory(
    store: EventStore, view: MemoryView, tokenizer: Any, *, tools=None,
    max_chunk_tokens: int = 768, chunk_overlap: int = 64, max_chunks: int | None = None,
    max_raw_tokens: int | None = None,
    derived_workspace_prefix_messages: Sequence[Mapping[str, Any]] = (),
    encoding_scope: str = "current",
    atomic_unit_token_limit: int | None = None,
    encoding_event_groups: Sequence[Sequence[str]] | None = None,
) -> PackedMemory:
    """Pack all selected content or raise; no first-plus-tail selection occurs.

    Chunks follow event source order regardless of the caller's ID ordering.
    Native raw events retain source order; restored events use A's evidence
    packet before the current workspace. The system/tools prefix
    is separated at a verified native-template boundary; the model adapter
    inserts gist KV between this prefix and the exact workspace.
    """
    view.validate(store)
    scope = validate_encoding_scope(encoding_scope)
    selected_gist = set(view.gist_event_ids)
    selected_groups = (
        None
        if encoding_event_groups is None
        else tuple(
            tuple(event_id for event_id in group)
            for group in encoding_event_groups
            if set(group) <= selected_gist
        )
    )
    chunks = encode_scope_chunks(
        store,
        view.gist_event_ids,
        tokenizer,
        encoding_scope=scope,
        max_chunk_tokens=max_chunk_tokens,
        chunk_overlap=chunk_overlap,
        atomic_unit_token_limit=atomic_unit_token_limit,
        event_groups=selected_groups,
    )
    if max_chunks is not None and len(chunks) > max_chunks:
        raise PackingBudgetError(f"Complete events need {len(chunks)} chunks; budget is {max_chunks}")
    raw_indices = tuple(sorted({index for event in store.events if event.event_id in view.raw_event_ids for index in event.source_indices}))
    raw_messages = list(raw_workspace_messages(store, view))
    if not raw_messages:
        raise ValueError("A decision view requires an observable raw message")
    # Derived observations belong to the charged workspace, never the source
    # event store, encoder chunks, or shared system/tools prefix.
    derived = []
    for message in derived_workspace_prefix_messages:
        if (set(message) != {"role", "content"} or message["role"] != "user"
                or not isinstance(message["content"], str) or not message["content"]):
            raise ValueError("Derived workspace observations require a nonempty user text message")
        derived.append(dict(message))
    if derived:
        prefix_length = 0
        while prefix_length < len(raw_messages) and raw_messages[prefix_length]["role"] == "system":
            prefix_length += 1
        raw_messages[prefix_length:prefix_length] = derived
    full_ids = native_ids(tokenizer, raw_messages, tools=tools, generation=True)
    prefix_messages = []
    for message in raw_messages:
        if message["role"] != "system":
            break
        prefix_messages.append(message)
    dummy = {"role": "user", "content": ""}
    dummy_ids = native_ids(tokenizer, [dummy])
    prefix_and_dummy = native_ids(tokenizer, prefix_messages + [dummy], tools=tools)
    if not dummy_ids or prefix_and_dummy[-len(dummy_ids):] != dummy_ids:
        raise ValueError("Native template does not support a separable system/tools prefix")
    prefix_ids = prefix_and_dummy[:-len(dummy_ids)]
    if full_ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError("Native system/tools prefix changed with workspace content")
    workspace_ids = full_ids[len(prefix_ids):]
    if max_raw_tokens is not None and len(full_ids) > max_raw_tokens:
        raise PackingBudgetError(f"System/tools plus complete raw workspace need {len(full_ids)} tokens; budget is {max_raw_tokens}")
    return PackedMemory(view, prefix_ids, workspace_ids, raw_indices, chunks)


def pack_target(
    tokenizer: Any, target: Message | Mapping[str, Any], *, max_target_tokens: int | None = None,
) -> tuple[int, ...]:
    """Native assistant continuation including its terminator, never sliced."""
    message = visible_message(target)
    if message["role"] != "assistant":
        raise ValueError("Only assistant messages are supervised")
    dummy = {"role": "user", "content": ""}
    prompt = native_ids(tokenizer, [dummy], generation=True)
    completion = native_ids(tokenizer, [dummy, message])
    if completion[:len(prompt)] != prompt:
        raise ValueError("Native assistant target is not a continuation of the generation prompt")
    target_ids = completion[len(prompt):]
    if max_target_tokens is not None and len(target_ids) > max_target_tokens:
        raise PackingBudgetError(f"Complete target needs {len(target_ids)} tokens; budget is {max_target_tokens}")
    return target_ids


def training_sequence(memory: PackedMemory, target_ids: Sequence[int]) -> dict[str, tuple[int, ...]]:
    """Unpadded ordinary inputs; CE is normalized per decision by the trainer."""
    if not target_ids:
        raise ValueError("An assistant target must contain tokens")
    ids = memory.workspace_input_ids + tuple(target_ids)
    return {
        "input_ids": ids,
        "labels": (-100,) * len(memory.workspace_input_ids) + tuple(target_ids),
        "attention_mask": (1,) * len(ids),
        "position_ids": tuple(range(memory.workspace_position_start, memory.workspace_position_start + len(ids))),
    }
