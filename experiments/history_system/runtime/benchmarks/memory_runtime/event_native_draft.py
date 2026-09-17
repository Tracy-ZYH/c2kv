"""Parse complete native Qwen tool drafts without executing their actions."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


NATIVE_DRAFT_VERSION = 'a-qwen-native-draft-v1'
_OPEN = '<tool_call>'
_CLOSE = '</tool_call>'


@dataclass(frozen=True)
class NativeDraft:
    text: str
    content: str
    tool_calls: tuple[dict[str, Any], ...]
    status: str
    reason: str
    reasoning_content: str | None = None


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f'non-JSON constant: {value}')


def _json(value):
    return json.loads(value, object_pairs_hook=_object, parse_constant=_constant)


def parse_native_draft(text: str, *, call_id_prefix: str) -> NativeDraft:
    """Read only explicit tool blocks; malformed output admits no partial calls.

    Tool names are not checked against a scorer or target action. Duplicate
    keys, non-JSON values, incomplete blocks, and non-object arguments remain
    parse failures instead of being repaired into executable actions.
    """
    if not isinstance(text, str):
        raise TypeError('native draft must be decoded text')
    if not isinstance(call_id_prefix, str) or not call_id_prefix:
        raise ValueError('native draft requires an explicit call ID prefix')
    body = text
    reasoning = None
    if body.lstrip().startswith('<think>'):
        start = body.index('<think>') + len('<think>')
        end = body.find('</think>', start)
        if end < 0:
            return NativeDraft(text, text, (), 'malformed', 'unfinished_thinking')
        reasoning = body[start:end]
        body = body[end + len('</think>'):]
    calls = []
    content = []
    cursor = 0
    try:
        while True:
            start = body.find(_OPEN, cursor)
            closing = body.find(_CLOSE, cursor)
            if start < 0:
                if closing >= 0:
                    raise ValueError('unmatched_tool_close')
                content.append(body[cursor:])
                break
            if 0 <= closing < start:
                raise ValueError('unmatched_tool_close')
            content.append(body[cursor:start])
            end = body.find(_CLOSE, start + len(_OPEN))
            if end < 0:
                raise ValueError('incomplete_tool_block')
            payload_text = body[start + len(_OPEN):end]
            if _OPEN in payload_text:
                raise ValueError('nested_tool_block')
            payload = _json(payload_text)
            if not isinstance(payload, dict) or set(payload) != {'name', 'arguments'}:
                raise ValueError('invalid_tool_object')
            if not isinstance(payload['name'], str) or not payload['name']:
                raise ValueError('invalid_tool_name')
            arguments = payload['arguments']
            if isinstance(arguments, str):
                arguments = _json(arguments)
            if not isinstance(arguments, dict):
                raise ValueError('arguments_not_object')
            calls.append({
                'id': f'{call_id_prefix}_{len(calls)}', 'type': 'function',
                'function': {'name': payload['name'], 'arguments': json.dumps(
                    arguments, ensure_ascii=False, separators=(',', ':'), allow_nan=False)},
            })
            cursor = end + len(_CLOSE)
    except (TypeError, ValueError, OverflowError) as error:
        return NativeDraft(text, body, (), 'malformed', str(error), reasoning)
    return NativeDraft(text, ''.join(content).strip(), tuple(calls),
                       'tool_calls' if calls else 'text',
                       'complete_native_calls' if calls else 'no_native_tool_calls', reasoning)


def decode_native_generation(tokenizer, result, *, call_id_prefix: str) -> NativeDraft:
    """Preserve protocol special tokens while removing only the terminal EOS."""
    ids = tuple(result.token_ids)
    eos_ids = result.stats.get('eos_token_ids', ())
    if result.finish_reason == 'stop' and ids and ids[-1] in eos_ids:
        ids = ids[:-1]
    text = tokenizer.decode(list(ids), skip_special_tokens=False,
                            clean_up_tokenization_spaces=False)
    return parse_native_draft(text, call_id_prefix=call_id_prefix)
