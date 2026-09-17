"""CPU contracts for bounded recovery evidence units and presentation."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest


RUNTIME = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RUNTIME), str(RUNTIME / "python")]

from benchmarks.memory_runtime.recovery.evidence_units import (  # noqa: E402
    build_catalog,
    deduplicate_units,
    expand_units,
    render_units,
    unit_is_covered,
)
from history_memory.events import EventStore  # noqa: E402


class OffsetTokenizer:
    """A variable-width tokenizer exposing the production offset contract."""

    @staticmethod
    def _pieces(text):
        return [match.span() for match in re.finditer(r"\s+|[\w]+|[^\w\s]", text)]

    def encode(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        return list(range(len(self._pieces(text))))

    def __call__(
        self, text, *, add_special_tokens=False, return_offsets_mapping=False
    ):
        assert add_special_tokens is False
        offsets = self._pieces(text)
        result = {"input_ids": list(range(len(offsets)))}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


class UnicodeByteTokenizer:
    """Mimic byte pieces sharing one Unicode character offset."""

    @staticmethod
    def _offsets(text):
        offsets = []
        for index, char in enumerate(text):
            offsets.extend([(index, index + 1)] * (3 if ord(char) > 127 else 1))
        return offsets

    def encode(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        return list(range(len(self._offsets(text))))

    def __call__(
        self, text, *, add_special_tokens=False, return_offsets_mapping=False
    ):
        offsets = self._offsets(text)
        result = {"input_ids": list(range(len(offsets)))}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


def tool_event(call_id, name, arguments, result):
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(result),
        },
    ]


def binding_store():
    messages = [{"role": "user", "content": "Process the records."}]
    messages.extend(
        tool_event(
            "create",
            "create_ticket",
            {"label": "source"},
            {"ticket_id": "opaque-A", "count": 17, "state": "pending"},
        )
    )
    messages.extend(
        tool_event(
            "schedule",
            "schedule_ticket",
            {"ticket_id": "opaque-A", "count": 17},
            {"job_id": "opaque-B", "state": "pending"},
        )
    )
    messages.extend(
        tool_event(
            "finish",
            "finish_job",
            {"job_id": "opaque-B", "count": 17},
            {"ok": True},
        )
    )
    messages.append({"role": "user", "content": "Continue."})
    return EventStore.from_messages("binding", messages)


def test_event_catalog_is_stable_prefix_only_and_receipt_omits_source_text():
    tokenizer = OffsetTokenizer()
    incomplete = [
        {"role": "system", "content": "Source policy."},
        {"role": "user", "content": "Earlier task."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "pending",
                    "type": "function",
                    "function": {"name": "wait", "arguments": "{}"},
                }
            ],
        },
    ]
    first = EventStore.from_messages("prefix", incomplete)
    first_catalog = build_catalog(first, tokenizer, "event")
    assert [unit.event_id for unit in first_catalog] == ["prefix:m1"]

    completed = EventStore.from_messages(
        "prefix",
        [
            *incomplete,
            {"role": "tool", "tool_call_id": "pending", "content": '{"ok":true}'},
            {"role": "user", "content": "Current request."},
        ],
    )
    second_catalog = build_catalog(completed, tokenizer, "event")
    assert first_catalog[0].unit_id == second_catalog[0].unit_id
    assert {unit.event_id for unit in second_catalog} == {
        "prefix:m1",
        "prefix:m2",
        "prefix:m4",
    }
    receipt = first_catalog[0].to_receipt()
    assert "text" not in receipt
    assert receipt["text_sha256"] == hashlib.sha256(
        first_catalog[0].text.encode("utf-8")
    ).hexdigest()


def test_token_windows_use_actual_offsets_and_respect_256_token_cap():
    tokenizer = OffsetTokenizer()
    text = " ".join(f"word{index}" for index in range(700))
    store = EventStore.from_messages("windows", [{"role": "user", "content": text}])
    units = build_catalog(store, tokenizer, "tokens_256")

    assert len(units) > 2
    assert all(unit.token_count <= 256 for unit in units)
    assert all(
        unit.token_count
        == len(tokenizer.encode(unit.text, add_special_tokens=False))
        for unit in units
    )
    assert "".join(unit.text for unit in units) == store.messages[0].json_text
    assert all(
        span.sha256
        == hashlib.sha256(
            store.messages[span.source_index].json_text[
                span.char_start : span.char_end
            ].encode("utf-8")
        ).hexdigest()
        for unit in units
        for span in unit.provenance
    )


def test_token_windows_keep_overlapping_unicode_byte_offsets_atomic():
    tokenizer = UnicodeByteTokenizer()
    text = "a" * 240 + "你" + "b" * 80
    store = EventStore.from_messages("unicode", [{"role": "user", "content": text}])
    units = build_catalog(store, tokenizer, "tokens_256")
    assert "".join(unit.text for unit in units) == store.messages[0].json_text
    assert all(unit.token_count <= 256 for unit in units)
    assert any("你" in unit.text for unit in units)


def test_record_and_field_units_keep_exact_spans_parent_and_call_association():
    tokenizer = OffsetTokenizer()
    content = (
        '{"records": [{"id": "α-1", "nested": {"status": "ready"}}, '
        '{"id": "β-2", "note": "comma, and } brace"}], "page": 1}'
    )
    store = EventStore.from_messages(
        "records",
        [
            {"role": "user", "content": "Inspect."},
            *tool_event(
                "list",
                "list_records",
                {"account_id": "acct-1", "filter": {"state": "all"}},
                json.loads(content),
            ),
            {"role": "assistant", "content": "Plain paragraph one.\n\nParagraph two."},
            {"role": "user", "content": "{\"valid\": true} trailing text"},
        ],
    )
    records = build_catalog(store, tokenizer, "record")
    result_records = [
        unit for unit in records if unit.metadata["role"] == "tool"
    ]
    assert [json.loads(unit.text)["id"] for unit in result_records] == ["α-1", "β-2"]
    assert [unit.metadata["record_path"][-2:] for unit in result_records] == [
        ["records", 0],
        ["records", 1],
    ]
    assert [unit.text for unit in records if unit.event_id == "records:m3"] == [
        "Plain paragraph one.",
        "Paragraph two.",
    ]
    assert not [unit for unit in records if unit.event_id == "records:m4"]

    fields = build_catalog(store, tokenizer, "field")
    arguments = [
        unit
        for unit in fields
        if unit.metadata["container_kind"] == "tool_call_arguments"
    ]
    account = next(unit for unit in arguments if unit.metadata["field_name"] == "account_id")
    state = next(unit for unit in arguments if unit.metadata["field_name"] == "state")
    assert json.loads(account.text) == "acct-1"
    assert account.metadata["tool_call_id"] == "list"
    assert account.metadata["tool_name"] == "list_records"
    assert account.association_id != state.association_id
    for unit in fields:
        for span in unit.provenance:
            message = store.messages[span.source_index].to_dict()
            container = message
            for part in span.container_path:
                container = container[part]
            assert container[span.char_start : span.char_end] == unit.text
            assert hashlib.sha256(unit.text.encode("utf-8")).hexdigest() == span.sha256


def test_predecessors_follow_only_unique_observed_string_argument_references():
    store = binding_store()
    catalog = build_catalog(store, OffsetTokenizer(), "field")
    selected = next(
        unit
        for unit in catalog
        if unit.metadata["tool_name"] == "finish_job"
        and unit.metadata["field_name"] == "job_id"
    )

    source = expand_units([selected], catalog, store, "source")
    one = expand_units([selected], catalog, store, "predecessor_1")
    two = expand_units([selected], catalog, store, "predecessor_2")
    assert source == [selected]
    assert {json.loads(unit.text) for unit in one if unit.metadata["value_type"] == "string"} == {
        "opaque-B"
    }
    strings = {
        json.loads(unit.text)
        for unit in two
        if unit.metadata["value_type"] == "string"
    }
    assert strings == {"opaque-A", "opaque-B"}
    assert all(unit.text != "17" for unit in two)
    assert all(unit.text != '"pending"' for unit in two)


def test_adjacency_span_dedup_and_presentation_order_are_data_only():
    store = binding_store()
    tokenizer = OffsetTokenizer()
    fields = build_catalog(store, tokenizer, "field")
    records = build_catalog(store, tokenizer, "record")
    result_field = next(
        unit
        for unit in fields
        if unit.event_id == "binding:m1"
        and unit.metadata["container_kind"] == "tool_call_arguments"
    )
    adjacent = expand_units([result_field], fields, store, "adjacent")
    assert 1 < len(adjacent) <= 3

    produced_field = next(
        unit
        for unit in fields
        if unit.event_id == "binding:m1"
        and unit.metadata["role"] == "tool"
        and unit.metadata["field_name"] == "ticket_id"
    )
    produced_record = next(
        unit
        for unit in records
        if unit.event_id == "binding:m1" and unit.metadata["role"] == "tool"
    )
    assert unit_is_covered(produced_field, [produced_record])
    assert deduplicate_units([produced_field, produced_record]) == [produced_record]

    later = next(unit for unit in records if unit.event_id == "binding:m5")
    relevance = render_units([later, produced_record], store, "structured", "relevance")
    chronological = render_units(
        [later, produced_record], store, "quoted", "chronological"
    )
    assert [message["role"] for message in relevance] == ["user"]
    payload = json.loads(relevance[0]["content"])
    assert [row["event_id"] for row in payload["units"]] == [
        later.event_id,
        produced_record.event_id,
    ]
    produced = payload["units"][1]
    assert produced["binding"]["historical_call"] == {
        "name": "create_ticket",
        "arguments": {"label": "source"},
    }
    assert chronological[0]["content"].find(
        produced_record.unit_id
    ) < chronological[0]["content"].find(later.unit_id)
    for rendered in (relevance, chronological):
        assert all(set(message) == {"role", "content"} for message in rendered)
        assert all("tool_calls" not in message for message in rendered)


def test_tokens_units_refuse_a_tokenizer_without_source_offsets():
    class NoOffsets:
        def encode(self, text, *, add_special_tokens=False):
            return [1] * len(text.split())

    store = EventStore.from_messages("no-offsets", [{"role": "user", "content": "a b"}])
    with pytest.raises(TypeError, match="callable tokenizer with offsets"):
        build_catalog(store, NoOffsets(), "tokens_256")
