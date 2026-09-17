"""One unsubmitted event-native decision with configured bounded recovery."""
from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict
from typing import Any

from .attempt_journal import AttemptJournal
from .event_native import memory_to_dict
from .event_native_draft import NATIVE_DRAFT_VERSION, decode_native_generation


class EventNativeStepError(RuntimeError):
    """A terminal local step failure, including all observed generation costs."""

    def __init__(self, message, record):
        super().__init__(message)
        self.record = record


class EventNativeDecisionRunner:
    """Keep the draft private and journal each actual call before submission.

    The caller supplies visible prefixes and receives only the final assistant
    response for execution. Generation records retain discarded drafts for
    cost/provenance. No tool, scorer, oracle, retry, or continuation loop runs
    here; a failure terminates this runner.
    """

    def __init__(self, controller, generator, tokenizer, *, ratio: int,
                 max_new_tokens: int, max_generation_calls: int,
                 journal: AttemptJournal):
        if type(max_generation_calls) is not int or max_generation_calls <= 0:
            raise ValueError('max_generation_calls must be a positive finite cap')
        if not isinstance(journal, AttemptJournal):
            raise TypeError('a durable AttemptJournal is required')
        self.controller, self.generator, self.tokenizer = controller, generator, tokenizer
        self.ratio, self.max_new_tokens = ratio, max_new_tokens
        self.max_generation_calls, self.journal = max_generation_calls, journal
        self.generation_calls = 0
        self._completed = {}
        self._terminal_error = None

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._terminal_error is not None:
            raise RuntimeError('This runner stopped after a terminal failure; automatic retry is disabled')
        key = (payload['session_id'], payload['decision_key'])
        signature = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
        cached = self._completed.get(key)
        if cached is not None:
            if cached[0] != signature:
                self.close()
                raise ValueError('decision key reused with different visible input')
            return copy.deepcopy(cached[1])
        started = time.perf_counter()
        keep_session = False
        record = {
            'schema': 'a-event-native-exact-step-v1', 'status': 'started',
            'session_id': key[0], 'decision_key': key[1],
            'ratio': self.ratio, 'max_new_tokens': self.max_new_tokens,
            'generation_trace': [], 'exact_recovery': None, 'response': None,
            'controller_timing': {'prepare_seconds': None, 'reconsider_seconds': None},
            'decision_runtime_seconds': None,
            'scope': 'One unsubmitted decision; only response is executable, no tools or scorer were invoked.',
        }
        try:
            prepare_started = time.perf_counter()
            try:
                prepared = self.controller.prepare(payload, ratio=self.ratio,
                                                   max_new_tokens=self.max_new_tokens)
            finally:
                record['controller_timing']['prepare_seconds'] = time.perf_counter() - prepare_started
            with self.generator.decision_scope(session_id=key[0]):
                result, draft = self._generate(prepared.memory, prepared.metadata, record, 'draft',
                    compression_chunks=getattr(prepared, 'eligible_chunks', None))
                observer = getattr(self.controller, 'observe_draft_features', None)
                if callable(observer):
                    stats = getattr(result, 'stats', None)
                    observer(
                        session_id=key[0], decision_key=key[1],
                        shadow_features=(stats.get('shadow_features')
                                         if isinstance(stats, dict) else None))
                reconsider_started = time.perf_counter()
                try:
                    reconsidered = self.controller.reconsider(
                        prepared, list(draft.tool_calls), draft_text=draft.text,
                        parse_error=draft.reason if draft.status == 'malformed' else None)
                finally:
                    record['controller_timing']['reconsider_seconds'] = time.perf_counter() - reconsider_started
                record['exact_recovery'] = copy.deepcopy(reconsidered['decision'])
                rounds = []
                max_rounds = getattr(self.controller, 'max_recovery_rounds', 1)
                checks = [copy.deepcopy(reconsidered['decision'])]
                if hasattr(self.controller, 'max_recovery_rounds'):
                    record['recovery_checks'] = checks
                    record['recovery_rounds'] = rounds
                while reconsidered['regenerate']:
                    rounds.append(copy.deepcopy(reconsidered['decision']))
                    record['generation_trace'][-1]['discarded'] = True
                    result, draft = self._generate(
                        reconsidered['memory'], reconsidered['metadata'], record, 'regeneration')
                    if len(rounds) >= max_rounds or self.generation_calls >= self.max_generation_calls:
                        break
                    advance = getattr(self.controller, 'advance_recovery', None)
                    if not callable(advance):
                        break
                    advance(prepared, shadow_features=(getattr(result, 'stats', {}) or {}).get('shadow_features'))
                    reconsider_started = time.perf_counter()
                    try:
                        reconsidered = self.controller.reconsider(
                            prepared, list(draft.tool_calls), draft_text=draft.text,
                            parse_error=draft.reason if draft.status == 'malformed' else None)
                        checks.append(copy.deepcopy(reconsidered['decision']))
                    finally:
                        record['controller_timing']['reconsider_seconds'] += time.perf_counter() - reconsider_started
                if hasattr(self.controller, 'max_recovery_rounds'):
                    record['exact_recovery'] = copy.deepcopy(rounds[-1] if rounds else reconsidered['decision'])
                    record['exact_recovery']['recovery_round_count'] = len(rounds)
                    record['exact_recovery']['termination'] = (
                        reconsidered['decision']['reason'] if not reconsidered['regenerate']
                        else 'recovery_or_generation_limit')
                record['response'] = {
                    'role': 'assistant', 'content': draft.content,
                    'tool_calls': list(draft.tool_calls),
                    'reasoning_content': draft.reasoning_content,
                    'native_parse_status': draft.status, 'native_parse_reason': draft.reason,
                    'finish_reason': result.finish_reason,
                }
            record['session_cache_after'] = self.generator.session_cache_info()
            record['decision_runtime_seconds'] = time.perf_counter() - started
            record['status'] = 'ok'
            self._totals(record)
            self._completed[key] = (signature, copy.deepcopy(record))
            keep_session = True
            return record
        except Exception as error:
            record['status'] = 'failed'
            record['response'] = None
            record['error'] = {'type': type(error).__name__, 'message': str(error)}
            record['decision_runtime_seconds'] = time.perf_counter() - started
            self._totals(record)
            # Store scalar diagnostics, not a traceback that could retain KV tensors.
            self._terminal_error = copy.deepcopy(record['error'])
            raise EventNativeStepError(str(error), record) from error
        finally:
            if not keep_session:
                self.close()
                record['session_cache_after_close'] = self.generator.session_cache_info()

    def close(self):
        """Release the generator's committed device and host session cache."""
        self.generator.close_session()

    def _generate(self, memory, metadata, record, phase, *, compression_chunks=None):
        from .budget_guard import history_budget_receipt
        budget = history_budget_receipt(memory, metadata, self.controller,
                                       ratio=self.ratio, phase=phase)
        record.setdefault('pre_generation_budget_checks', []).append(budget)
        if budget['status'] != 'passed':
            raise ValueError('Pre-generation history budget rejected: ' + ', '.join(budget['errors']))
        if self.generation_calls >= self.max_generation_calls:
            raise RuntimeError('Finite generation-call cap exhausted before submission')
        request_id = json.dumps([record['session_id'], record['decision_key']], separators=(',', ':'))
        trace = {
            'phase': phase, 'status': 'started', 'discarded': False,
            'prepared_input': memory_to_dict(memory), 'controller': copy.deepcopy(metadata),
            'planned_resident_prompt_tokens': memory.costs(self.ratio)['resident_kv_tokens'],
            'usage': None, 'generation': None, 'native_draft': None,
        }
        handle = self.journal.start('generation', self.generation_calls + 1, request_id,
                                    {'task_id': record['session_id'], 'decision_id': record['decision_key']})
        self.generation_calls += 1
        trace.update(attempt_uid=handle.attempt_uid, attempt_index=handle.attempt_index)
        record['generation_trace'].append(trace)
        kwargs = {}
        if compression_chunks is not None:
            kwargs['compression_chunks'] = compression_chunks
        if getattr(self.generator, 'cache_trace_schema', None) == 'event-native-cache-trace-v1':
            kwargs['trace_context'] = {
                'attempt_uid': handle.attempt_uid, 'session_id': record['session_id'],
                'decision_key': record['decision_key'], 'phase': phase,
            }
        try:
            result = self.generator.generate(memory, ratio=self.ratio,
                                             max_new_tokens=self.max_new_tokens, **kwargs)
        except Exception:
            trace['status'] = 'failed'
            partial = getattr(self.generator, 'last_generation_trace', None)
            if isinstance(partial, dict) and partial.get('attempt_uid') == handle.attempt_uid:
                # Scope cleanup may still append completed release operations.
                trace['cache_trace'] = partial
            self.journal.finish(handle, 'failed')
            raise
        usage = {
            'prompt_tokens': trace['planned_resident_prompt_tokens'],
            'completion_tokens': len(result.token_ids),
            'total_tokens': trace['planned_resident_prompt_tokens'] + len(result.token_ids),
        }
        trace.update(status='completed', usage=usage, generation={
            'token_ids': list(result.token_ids), 'finish_reason': result.finish_reason,
            # The active scope finalizes scalar cache commit diagnostics on exit.
            'token_logprobs': list(result.token_logprobs), 'stats': result.stats,
        })
        self.journal.finish(handle, 'completed', usage=usage)
        draft = decode_native_generation(
            self.tokenizer, result,
            call_id_prefix=f"d{metadata['decision_index']}_r{len(record['generation_trace']) - 1}")
        trace['native_draft'] = {'version': NATIVE_DRAFT_VERSION, **asdict(draft)}
        selection_observer = getattr(self.controller, 'observe_selection_draft', None)
        if callable(selection_observer):
            selection_observer(session_id=record['session_id'], decision_key=record['decision_key'],
                               token_logprobs=list(result.token_logprobs))
        return result, draft

    @staticmethod
    def _totals(record):
        trace = record['generation_trace']
        record['generation_attempts'] = len(trace)
        record['generation_completed'] = sum(item['status'] == 'completed' for item in trace)
        record['generation_usage_total'] = {}
        record['generation_usage_known'] = {}
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            values = [(item['usage'] or {}).get(key) for item in trace]
            record['generation_usage_total'][key] = sum(values) if values and all(v is not None for v in values) else None
            record['generation_usage_known'][key] = sum(value for value in values if value is not None)
        record['usage_scope'] = (
            'Prompt counts resident model input per generation; encoder extraction and repeated raw '
            'forward work are separate generation.stats counters. Failed calls have unknown usage.')
