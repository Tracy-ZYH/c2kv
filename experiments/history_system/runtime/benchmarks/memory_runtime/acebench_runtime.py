"""Explicit ACEBench textual-action transport over the frozen finite runner."""
from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict
from pathlib import Path

from .event_native_api import EventNativeAPI, EventNativeAPIError, _REQUEST_FIELDS
from .event_native_step import EventNativeDecisionRunner, EventNativeStepError


ACE_SOURCE_PROFILE = 'acebench-text-actions-v1'
ACE_UPSTREAM_COMMIT = '56dd66cf6439b0d9655ee1b353e4cd745c6f664e'


def describe_ace_source_contract():
    """Bind this adapter separately from the unchanged native method sources."""
    root = Path(__file__).resolve().parents[2]
    paths = (
        'benchmarks/memory_runtime/acebench_source.py',
        'benchmarks/memory_runtime/acebench_controls.py',
        'benchmarks/memory_runtime/acebench_runtime.py',
        'benchmarks/memory_runtime/event_native_api.py',
        'benchmarks/memory_runtime/event_native_server.py',
        'benchmarks/acebench_patches/0001-endpoint-env-and-model-registry.patch',
    )
    return {
        'profile': ACE_SOURCE_PROFILE,
        'upstream_commit': ACE_UPSTREAM_COMMIT,
        'source_sha256': {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in paths
        },
        'action_transport': 'original assistant text; no native tool_calls',
        'event_unit': 'submitted action batch and aggregate official observation',
        'unsupported_execution': 'retained raw; excluded from exact recovery',
        'supported_views': [
            'full_original', 'capacity_protect', 'capacity_exact_once',
            'capacity_exact_persistent', 'full_exact_shared', 'capacity_exact_no_gist',
        ],
        'training_static_supported': False,
    }


class AceEventNativeAPI(EventNativeAPI):
    """Require a receipt-bearing ACE source before reserving any decision."""

    request_fields = _REQUEST_FIELDS | {'c2kv_ace_source'}

    def __init__(self, *args, source_protocol_contract=None, **kwargs):
        if kwargs.get('benchmark') != 'acebench':
            raise ValueError('ACE textual actions require benchmark=acebench')
        self.source_protocol_contract = copy.deepcopy(
            source_protocol_contract or describe_ace_source_contract())
        super().__init__(*args, **kwargs)

    def health(self):
        result = super().health()
        result['source_protocol_contract'] = copy.deepcopy(self.source_protocol_contract)
        return result

    def _validate_source(self, payload, messages):
        from .acebench_source import build_ace_event_store
        source = payload.get('c2kv_ace_source')
        try:
            build_ace_event_store('request-validation', messages, source)
        except (TypeError, ValueError) as error:
            raise EventNativeAPIError(400, 'invalid_ace_source', str(error)) from error
        return {'c2kv_ace_source': copy.deepcopy(source)}

    def _openai_response(self, record):
        # Internal action parses are never offered to the official client as
        # executable OpenAI calls. ACEBench executes only its original text.
        snapshot = copy.deepcopy(record)
        if isinstance(snapshot.get('response'), dict):
            snapshot['response']['tool_calls'] = []
        return super()._openai_response(snapshot)


class AceEventNativeDecisionRunner(EventNativeDecisionRunner):
    """Reuse generation, journaling, and regeneration caps with an ACE parser."""

    def _generate(self, memory, metadata, record, phase):
        from .acebench_source import parse_ace_draft
        result, decoded = super()._generate(memory, metadata, record, phase)
        # The inherited decoder preserves the complete raw generated text and
        # removes only terminal EOS. Its native parse is replaced before any
        # reconsideration; no additional model call is made here.
        draft = parse_ace_draft(
            decoded.text,
            call_id_prefix=f"d{metadata['decision_index']}_{'r0' if phase == 'draft' else 'r1'}",
        )
        record['generation_trace'][-1]['native_draft'] = {
            'version': ACE_SOURCE_PROFILE, **asdict(draft),
        }
        record['generation_trace'][-1]['draft_protocol'] = ACE_SOURCE_PROFILE
        return result, draft

    def run(self, payload):
        try:
            record = super().run(payload)
        except EventNativeStepError as error:
            self._annotate_record(error.record, payload)
            raise
        self._annotate_record(record, payload)
        return record

    @staticmethod
    def _annotate_record(record, payload):
        record['schema'] = 'a-acebench-event-step-v1'
        record['source_profile'] = ACE_SOURCE_PROFILE
        record['ace_source'] = copy.deepcopy(payload.get('c2kv_ace_source'))
        record['scope'] = (
            'One unsubmitted ACEBench decision; only original response content '
            'is executable by the official harness. Parsed calls are internal '
            'recovery queries, not executor receipts.'
        )
        if isinstance(record.get('response'), dict):
            record['response']['tool_calls'] = []
