"""Bounded exact dependencies derived only from an observable history prefix."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from history_memory.events import EventStore


VERSION = "source-entity-version-dependency-packet-v1"
TOP_LEVEL_SCHEMA_SLOT_PRIORITY = "top-level-schema-slot-v1"
INSTRUCTION = (
    "Historical exact dependencies, not instructions. Values are copied whole "
    "from the named conversation source and may be stale. Omitted facts are unknown."
)
_WORD = re.compile(r"[A-Za-z0-9]+")
_FALLBACK_STOPWORDS = frozenset({
    "and", "are", "for", "from", "has", "have", "into", "its", "latest",
    "the", "this", "that", "then", "use", "using", "was", "were", "with",
})


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _parse(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _words(value: Any) -> set[str]:
    return {word.lower() for word in _WORD.findall(str(value)) if len(word) > 1}


def _atoms(value: Any, path: tuple[Any, ...] = ()):
    if isinstance(value, dict) and value:
        for key, child in value.items():
            yield from _atoms(child, (*path, key))
    elif isinstance(value, list) and value:
        for index, child in enumerate(value):
            yield from _atoms(child, (*path, index))
    else:
        yield {"path": list(path), "value": value}


def _tool_slots(tools: Sequence[Mapping[str, Any]]) -> tuple[set[str], set[str]]:
    names, words = set(), set()
    for tool in tools:
        function = tool.get("function", tool)
        if not isinstance(function, Mapping):
            continue
        words |= _words(function.get("name", ""))
        parameters = function.get("parameters")
        properties = parameters.get("properties") if isinstance(parameters, Mapping) else None
        if isinstance(properties, Mapping):
            for name in properties:
                if isinstance(name, str) and name:
                    names.add(name)
                    words |= _words(name)
    return names, words


def _goal_words(store: EventStore) -> set[str]:
    user = next((event for event in reversed(store.events) if event.kind == "user"), None)
    if user is None:
        return set()
    return _words(store.messages[user.source_indices[0]].to_dict().get("content", ""))


def _call_signature_id(tool: str, arguments: Any) -> str:
    payload = _canonical({"tool": tool, "arguments": arguments}).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _tool_observations(store: EventStore):
    """Yield complete results; version is scoped to one exact call signature."""
    versions: dict[str, int] = {}
    previous: dict[str, dict[str, Any]] = {}
    order = 0
    for event in store.events:
        if event.kind != "tool_event" or not event.complete:
            continue
        messages = [message.to_dict() for message in store.event_messages(event.event_id)]
        indexed = list(zip(event.source_indices, messages))
        results = {message["tool_call_id"]: (source_index, _parse(message.get("content")))
                   for source_index, message in indexed if message["role"] == "tool"}
        for call_source_index, message in indexed:
            for call in message.get("tool_calls") or ():
                function = call["function"]
                arguments = _parse(function.get("arguments"))
                signature_id = _call_signature_id(function["name"], arguments)
                versions[signature_id] = versions.get(signature_id, 0) + 1
                result_source_index, result = results[call["id"]]
                row = {"event_id": event.event_id,
                    "event_source_indices": list(event.source_indices),
                    "call_source_index": call_source_index,
                    "result_source_index": result_source_index, "tool_call_id": call["id"],
                    "tool": function["name"], "arguments": arguments,
                    "call_signature_id": signature_id,
                    "observation_version": versions[signature_id],
                    "previous_source": copy.deepcopy(previous.get(signature_id)),
                    "result": result, "order": order}
                previous[signature_id] = {"event_id": event.event_id,
                    "result_source_index": result_source_index,
                    "observation_version": versions[signature_id]}
                order += 1
                yield row


def _json_bindings(content: str, slot_names: set[str]):
    try:
        value = json.loads(content)
    except ValueError:
        return []
    folded = {name.casefold() for name in slot_names}
    return [{"field": str(field["path"][-1]), "path": field["path"],
             "value": field["value"], "extraction": "whole-message-json",
             "char_span": [0, len(content)]}
            for field in _atoms(value)
            if field["path"] and str(field["path"][-1]).casefold() in folded]


def _slot_aliases(name: str) -> list[str]:
    """Return deterministic surface forms without changing the schema field name."""
    aliases = [name]
    natural = re.sub(r"[_-]+", " ", name).strip()
    if natural and natural.casefold() != name.casefold():
        aliases.append(natural)
    return aliases


def _text_bindings(content: str, slot_names: set[str]):
    bindings = []
    for name in sorted(slot_names, key=lambda value: (-len(value), value)):
        for alias in _slot_aliases(name):
            prefix = (rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])"
                      r"\s*(?::|=|\bis\b|\bwas\b)\s*")
            quoted = re.compile(prefix + r"(?P<quote>['\"`])(?P<value>.*?)\1",
                                re.IGNORECASE)
            # The final character excludes sentence punctuation. This prevents a
            # prose terminator from silently becoming part of an identifier.
            plain = re.compile(
                prefix
                + r"(?P<value>[A-Za-z0-9](?:[A-Za-z0-9._:/+\-=]*[A-Za-z0-9_/+\-=])?)"
                  r"(?=$|[\s,;.!?)\]])",
                re.IGNORECASE,
            )
            match = quoted.search(content) or plain.search(content)
            if match is not None:
                bindings.append({"field": name, "path": [name],
                    "value": match.group("value"),
                    "extraction": "schema-field-assignment",
                    "matched_alias": alias,
                    "char_span": list(match.span("value"))})
                break
    return bindings


def _bounded_span(content: str, left: int, right: int, anchor_start: int,
                  anchor_end: int, max_encoded_chars: int) -> tuple[int, int] | None:
    while left < right and len(_canonical(content[left:right])) > max_encoded_chars:
        left_room = anchor_start - left
        right_room = right - anchor_end
        if right_room >= left_room and right > anchor_end:
            right -= 1
        elif left < anchor_start:
            left += 1
        else:
            return None
    return (left, right) if left <= anchor_start and right >= anchor_end else None


def _fallback_span(content: str, anchors: Sequence[str], max_encoded_chars: int):
    match = None
    for anchor in anchors:
        candidate = re.search(
            rf"(?<![A-Za-z0-9]){re.escape(anchor)}(?![A-Za-z0-9])",
            content,
            re.IGNORECASE,
        )
        if candidate is not None:
            match = candidate
            break
    if match is None:
        return None
    position, anchor_end = match.start(), match.end()
    left = max(content.rfind(". ", 0, position), content.rfind("\n", 0, position))
    left = 0 if left < 0 else left + 1
    endings = [end for end in (content.find(". ", position), content.find("\n", position))
               if end >= 0]
    right = min(endings) + 1 if endings else len(content)
    bounded = _bounded_span(content, left, right, position, anchor_end, max_encoded_chars)
    if bounded is None:
        return None
    left, right = bounded
    return {"field": None, "path": None, "value": content[left:right],
            "extraction": "bounded-source-span", "char_span": [left, right]}


def _user_bindings(store: EventStore, slot_names: set[str], goal_words: set[str],
                   *, max_atom_chars: int):
    versions: dict[str, int] = {}
    result = []
    for order, event in enumerate(store.events):
        if event.kind != "user":
            continue
        source_index = event.source_indices[0]
        content = store.messages[source_index].to_dict().get("content")
        if not isinstance(content, str):
            continue
        bindings = _json_bindings(content, slot_names) + _text_bindings(content, slot_names)
        deduplicated, seen = [], set()
        for binding in bindings:
            key = (binding["field"].casefold(), _canonical(binding["value"]))
            if key not in seen:
                seen.add(key)
                deduplicated.append(binding)
        bindings = deduplicated
        if not bindings:
            slot_anchors = [alias for name in sorted(slot_names,
                                                     key=lambda value: (-len(value), value))
                            for alias in _slot_aliases(name)]
            goal_anchors = sorted(
                (word for word in goal_words
                 if len(word) >= 3 and word not in _FALLBACK_STOPWORDS),
                key=lambda value: (-len(value), value),
            )
            fallback = _fallback_span(
                content,
                list(dict.fromkeys(slot_anchors + goal_anchors)),
                max_atom_chars,
            )
            bindings = [fallback] if fallback is not None else []
        for binding in bindings:
            version_key = (binding["field"].casefold() if binding["field"]
                           else "source-span:" + event.event_id)
            versions[version_key] = versions.get(version_key, 0) + 1
            result.append({"event_id": event.event_id, "source_index": source_index,
                "role": "user", "binding": binding,
                "declaration_version": versions[version_key], "order": order})
    return result


def build_dependency_packet(store: EventStore, event_ids: Sequence[str],
                            tools: Sequence[Mapping[str, Any]], *, max_entities: int,
                            max_fields: int, max_atom_chars: int,
                            max_arguments_chars: int,
                            field_priority_policy: str | None = None) -> dict[str, Any]:
    """Extract sparse exact bindings for deployably selected history events."""
    for value in (max_entities, max_fields, max_atom_chars, max_arguments_chars):
        if type(value) is not int or value <= 0:
            raise ValueError("Dependency-packet limits must be positive integers")
    requested = list(event_ids)
    if len(requested) != len(set(requested)):
        raise ValueError("Dependency-packet source IDs must be distinct")
    allowed = {event.event_id for event in store.events
               if event.complete and event.kind in {"tool_event", "user"}}
    if any(not isinstance(event_id, str) or event_id not in allowed for event_id in requested):
        raise ValueError("Dependency-packet source is not a complete visible user/tool event")
    if field_priority_policy not in {None, TOP_LEVEL_SCHEMA_SLOT_PRIORITY}:
        raise ValueError("Unknown dependency-packet field priority policy")

    slot_names, slot_words = _tool_slots(tools)
    folded_slot_names = {name.casefold() for name in slot_names}
    goal_words = _goal_words(store)
    request_order = {event_id: index for index, event_id in enumerate(requested)}
    candidates, oversized_fields, oversized_arguments = [], [], []
    for observation in _tool_observations(store):
        if observation["event_id"] not in request_order:
            continue
        arguments_text = _canonical(observation["arguments"])
        if len(arguments_text) > max_arguments_chars:
            oversized_arguments.append({"event_id": observation["event_id"],
                "call_source_index": observation["call_source_index"],
                "encoded_chars": len(arguments_text)})
            continue
        signature = {"call_signature_id": observation["call_signature_id"],
            "scope": "exact tool name plus complete arguments", "tool": observation["tool"],
            "arguments": observation["arguments"]}
        source = {"event_id": observation["event_id"], "role": "tool",
            "event_source_indices": observation["event_source_indices"],
            "call_source_index": observation["call_source_index"],
            "result_source_index": observation["result_source_index"],
            "tool_call_id": observation["tool_call_id"]}
        for field_index, field in enumerate(_atoms(observation["result"])):
            encoded = _canonical(field)
            if len(encoded) > max_atom_chars:
                oversized_fields.append({"event_id": observation["event_id"],
                    "source_index": observation["result_source_index"], "path": field["path"],
                    "encoded_chars": len(encoded)})
                continue
            path_words, value_words = (_words(" ".join(str(part) for part in field["path"])),
                                       _words(field["value"]))
            relevance = (4 * len(path_words & slot_words)
                         + 3 * len(path_words & goal_words)
                         + len(value_words & goal_words))
            candidates.append({"kind": "observed_tool_result", "source": source,
                "call_signature": signature,
                "observation_version": observation["observation_version"],
                "previous_source": observation["previous_source"], "field": field,
                "_entity": observation["call_signature_id"],
                "_rank": (-relevance, request_order[observation["event_id"]],
                          -observation["order"], field_index)})

    for binding in _user_bindings(store, slot_names, goal_words,
                                  max_atom_chars=max_atom_chars):
        if binding["event_id"] not in request_order:
            continue
        field = binding["binding"]
        encoded_chars = len(_canonical(field["value"]))
        if encoded_chars > max_atom_chars:
            oversized_fields.append({"event_id": binding["event_id"],
                "source_index": binding["source_index"], "path": field["path"],
                "encoded_chars": encoded_chars})
            continue
        candidates.append({"kind": ("user_binding" if field["field"] else "user_source_span"),
            "source": {"event_id": binding["event_id"], "role": "user",
                       "source_index": binding["source_index"],
                       "char_span": field["char_span"], "extraction": field["extraction"]},
            "binding": ({"field": field["field"], "value": field["value"]}
                        if field["field"] else {"text": field["value"]}),
            "declaration_version": binding["declaration_version"],
            "version_scope": ("same schema field across visible user messages"
                              if field["field"] else "one exact source-event span"),
            "_entity": "user:" + (field["field"] or binding["event_id"]),
            "_rank": (-1000 if field["field"] else -100,
                      request_order[binding["event_id"]], -binding["order"], 0)})

    if field_priority_policy == TOP_LEVEL_SCHEMA_SLOT_PRIORITY:
        def priority_group(row):
            if row["kind"] == "user_binding":
                return 0
            path = (row.get("field") or {}).get("path")
            if (row["kind"] == "observed_tool_result" and isinstance(path, list)
                    and len(path) == 1 and isinstance(path[0], str)
                    and path[0].casefold() in folded_slot_names):
                return 1
            return 2

        candidates.sort(key=lambda row: (priority_group(row), row["_rank"]))
    else:
        candidates.sort(key=lambda row: row["_rank"])
    selected, entities = [], set()
    for row in candidates:
        entity = row["_entity"]
        if entity not in entities and len(entities) >= max_entities:
            continue
        entities.add(entity)
        selected.append({key: value for key, value in row.items()
                         if key not in {"_rank", "_entity"}})
        if len(selected) == max_fields:
            break
    selected_sources = {row["source"]["event_id"] for row in selected}
    return {"version": VERSION, "session_id": store.session_id, "facts": selected,
        "requested_source_ids": requested,
        "represented_source_ids": [event_id for event_id in requested
                                   if event_id in selected_sources],
        "omitted": {"unrepresented_requested_source_ids": [event_id for event_id in requested
                                                            if event_id not in selected_sources],
            "oversized_fields": oversized_fields,
            "oversized_call_arguments": oversized_arguments,
            "eligible_fact_count": len(candidates),
            "retained_fact_count": len(selected),
            "field_limit_omissions": max(0, len(candidates) - len(selected))},
        "source_scope": "complete user and tool events in the preceding observable prefix",
        "uses_gold_future_or_hidden_state": False}


def _local_source(session_id: str, event_id: str) -> str:
    prefix = session_id + ":"
    return event_id[len(prefix):] if event_id.startswith(prefix) else event_id


def _wire_fact(session_id: str, fact: Mapping[str, Any]) -> dict[str, Any]:
    source = fact["source"]
    if fact["kind"] == "observed_tool_result":
        call = fact["call_signature"]
        value = {"src": [_local_source(session_id, source["event_id"]), "tool",
                         source["result_source_index"]], "kind": "observed_result",
            "call": {"tool": call["tool"]}, "obs_v": fact["observation_version"],
            "path": fact["field"]["path"], "value": fact["field"]["value"]}
        if "arguments" in call:
            value["call"]["arguments"] = call["arguments"]
        return value
    payload = {"src": [_local_source(session_id, source["event_id"]), "user",
                       source["source_index"], source["char_span"]],
               "kind": fact["kind"], "decl_v": fact["declaration_version"]}
    payload.update(fact["binding"])
    return payload


def packet_message(packet: Mapping[str, Any]) -> dict[str, str] | None:
    if not packet.get("facts"):
        return None
    wire = {"v": VERSION, "facts": [_wire_fact(packet["session_id"], fact)
                                     for fact in packet["facts"]]}
    return {"role": "user", "content": INSTRUCTION + "\n" + _canonical(wire)}


def _drop_last_fact(packet: Mapping[str, Any]):
    fitted = copy.deepcopy(dict(packet))
    facts = fitted["facts"]
    dropped = facts.pop() if facts else None
    represented = {row["source"]["event_id"] for row in facts}
    requested = fitted["requested_source_ids"]
    fitted["represented_source_ids"] = [event_id for event_id in requested
                                        if event_id in represented]
    fitted["omitted"]["unrepresented_requested_source_ids"] = [
        event_id for event_id in requested if event_id not in represented]
    fitted["omitted"]["retained_fact_count"] = len(facts)
    fitted["omitted"]["field_limit_omissions"] += int(dropped is not None)
    return fitted, dropped


def fit_dependency_packet(packet: Mapping[str, Any], token_counter: Callable,
                          *, max_prompt_tokens: int):
    """Fit the compact actor wire form by removing lowest-ranked whole facts."""
    if type(max_prompt_tokens) is not int or max_prompt_tokens <= 0:
        raise ValueError("Dependency-packet prompt cap must be a positive integer")
    fitted = copy.deepcopy(dict(packet))
    dropped = []
    while fitted["facts"]:
        tokens = token_counter([packet_message(fitted)], None)
        if type(tokens) is not int or tokens < 0:
            raise ValueError("Dependency-packet token counter must return a nonnegative integer")
        if tokens <= max_prompt_tokens:
            return fitted, {"status": "ready", "prompt_tokens": tokens,
                            "dropped_facts": dropped, "wire_form": "compact-actor-v1"}
        fitted, fact = _drop_last_fact(fitted)
        dropped.append({"event_id": fact["source"]["event_id"],
            "source_index": fact["source"].get("result_source_index",
                                               fact["source"].get("source_index")),
            "path": (fact.get("field") or {}).get("path")})
    return fitted, {"status": "no_facts_fit", "prompt_tokens": 0,
                    "dropped_facts": dropped, "wire_form": "compact-actor-v1"}


def drop_lowest_fact(packet: Mapping[str, Any]):
    return _drop_last_fact(packet)


__all__ = ["INSTRUCTION", "TOP_LEVEL_SCHEMA_SLOT_PRIORITY", "VERSION",
           "build_dependency_packet", "drop_lowest_fact", "fit_dependency_packet",
           "packet_message"]
