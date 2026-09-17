"""A reference inference entry point for the committed event-native profile.

The input is either PackedMemory or visible prefixes prepared with the
checkpoint's fixed selection and byte-budget contract. The legacy 1088 proxy
keeps its separate generation path.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

# benchmarks/memory_runtime is two directories below the repository root.
PYTHON_ROOT = Path(__file__).resolve().parents[2] / 'python'
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from history_memory.events import EventStore
from history_memory.packing import EncoderChunk, MemoryView, PackedMemory, pack_memory

EXPECTED_PROFILE = {
    'history_memory_training_profile': 'history-event-base-query-v1',
    'history_memory_packing_version': 'history-event-v1',
    'history_memory_raw_layout': 'event-native-evidence-v1',
    'history_memory_evidence_version': 'history-evidence-v1',
    'history_memory_normal_query': 'base',
    'gist_param': 'qkv',
    'gist_type': 'dynamic-interleave',
    'gist_residual_type': 'embed-mean',
}
EXACT_VIEW_MODES = frozenset({
    'capacity_exact_once', 'capacity_exact_persistent',
    'full_exact_shared', 'capacity_exact_no_gist',
})


def inspect_checkpoint(checkpoint: str | Path) -> dict[str, Any]:
    checkpoint = Path(checkpoint).resolve()
    config = json.loads((checkpoint / 'config.json').read_text(encoding='utf-8'))
    for field, required in EXPECTED_PROFILE.items():
        if config.get(field) != required:
            raise ValueError(f'event-native checkpoint requires {field}={required!r}; got {config.get(field)!r}')
    if config.get('model_type') != 'qwen3' or config.get('architectures') != ['Qwen3ForCausalLM']:
        raise ValueError('event-native reference generation requires the repository Qwen3ForCausalLM')
    ratios = config.get('history_memory_supported_ratios')
    if not isinstance(ratios, list) or not ratios or any(type(ratio) is not int or ratio <= 0 for ratio in ratios):
        raise ValueError('checkpoint must declare history_memory_supported_ratios')
    gist_token_id = config.get('gist_token_id')
    if type(gist_token_id) is not int or not 0 <= gist_token_id < config.get('vocab_size', 0):
        raise ValueError('checkpoint must declare an in-vocabulary gist_token_id')
    state_path = checkpoint / 'trainer_state.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}
    version = state.get('parameter_version', 0)
    if type(version) is not int or version < 0:
        raise ValueError('checkpoint parameter_version must be a nonnegative integer')
    corpus_identity = (state.get('contract') or {}).get('corpus_identity')
    return {
        'checkpoint': str(checkpoint), 'profile': EXPECTED_PROFILE.copy(),
        'training_arm': config.get('history_memory_arm'),
        'parameter_version': version, 'corpus_identity': corpus_identity,
        'synthetic_cpu_smoke': corpus_identity == 'synthetic-tiny-history-v1',
        'synthetic_inference_fixture': config.get('a_inference_fixture') is True,
        'training_completed': state.get('completed'),
        'declared_supported_ratios': list(ratios),
        'packing_contract': config.get('history_memory_packing'),
        'policy_contract': config.get('history_memory_policy'),
        'model_geometry': {name: config.get(name) for name in (
            'num_hidden_layers', 'num_key_value_heads', 'num_attention_heads',
            'head_dim', 'hidden_size', 'max_position_embeddings')},
        'scope': 'Profile fields checked from checkpoint; this is not a training-data or task-performance certification.',
    }


def prepare_memory(payload: Mapping[str, Any], tokenizer, *, max_chunk_tokens: int,
                   chunk_overlap: int, max_chunks: int | None = None,
                   max_raw_tokens: int | None = None) -> PackedMemory:
    """Compile caller-selected visible events through the shared B packer."""
    session_id = payload.get('session_id')
    if not isinstance(session_id, str) or not session_id:
        raise ValueError('session_id must be explicit and nonempty')
    store = EventStore.from_messages(session_id, payload['messages'])
    spec = payload['view']
    view = MemoryView(tuple(spec['gist_event_ids']), tuple(spec['raw_event_ids']),
                      tuple(spec.get('evidence_event_ids', ())))
    return pack_memory(store, view, tokenizer, tools=payload.get('tools'),
                       max_chunk_tokens=max_chunk_tokens, chunk_overlap=chunk_overlap,
                       max_chunks=max_chunks, max_raw_tokens=max_raw_tokens)


def memory_to_dict(memory: PackedMemory) -> dict[str, Any]:
    from dataclasses import asdict
    from .event_native_raw import RuntimeMemoryView
    schema = ('a-event-native-packed-input-v2' if isinstance(memory.view, RuntimeMemoryView)
              else 'a-event-native-packed-input-v1')
    return {'schema': schema, **asdict(memory)}


def memory_from_dict(value: Mapping[str, Any]) -> PackedMemory:
    """Read an explicitly tokenized input, including provenance and positions."""
    schema = value.get('schema')
    if schema not in {'a-event-native-packed-input-v1', 'a-event-native-packed-input-v2'}:
        raise ValueError('unsupported event-native packed-input schema')
    view = value['view']
    base_fields = {'gist_event_ids', 'raw_event_ids', 'evidence_event_ids'}
    if schema == 'a-event-native-packed-input-v1':
        if set(view) - base_fields:
            raise ValueError('v1 packed input cannot carry runtime omissions or a custom renderer')
        restored_view = MemoryView(tuple(view['gist_event_ids']), tuple(view['raw_event_ids']),
                                   tuple(view.get('evidence_event_ids', ())))
    else:
        from .event_native_raw import RuntimeMemoryView
        runtime_fields = {'omitted_event_ids', 'mandatory_raw_event_ids', 'raw_control_layout'}
        if set(view) != base_fields | runtime_fields:
            raise ValueError('v2 packed input must preserve the complete runtime view contract')
        restored_view = RuntimeMemoryView(
            gist_event_ids=tuple(view['gist_event_ids']), raw_event_ids=tuple(view['raw_event_ids']),
            evidence_event_ids=tuple(view['evidence_event_ids']),
            omitted_event_ids=tuple(view['omitted_event_ids']),
            mandatory_raw_event_ids=tuple(view['mandatory_raw_event_ids']),
            raw_control_layout=view['raw_control_layout'])
    return PackedMemory(
        view=restored_view,
        system_input_ids=tuple(value['system_input_ids']),
        workspace_input_ids=tuple(value['workspace_input_ids']),
        raw_source_indices=tuple(value['raw_source_indices']),
        chunks=tuple(EncoderChunk(
            event_id=chunk['event_id'], part_index=chunk['part_index'],
            source_indices=tuple(chunk['source_indices']),
            source_token_start=chunk['source_token_start'], source_token_end=chunk['source_token_end'],
            token_ids=tuple(chunk['token_ids'])) for chunk in value['chunks']),
        raw_layout_profile=value['raw_layout_profile'],
    )


def load_generator(checkpoint: str | Path, *, device: str = 'cpu', dtype: str = 'float32',
                   decode_strategy: str = 'incremental', prefill_chunk_size: int | None = None,
                   max_extraction_calls: int | None = None, retain_raw_snapshot: bool = True):
    """Load local weights only; leave the service/checkpoint training untouched."""
    if decode_strategy not in {'incremental', 'full_recompute'}:
        raise ValueError('unsupported inference decode strategy')
    if prefill_chunk_size is not None:
        if type(prefill_chunk_size) is not int or prefill_chunk_size <= 0:
            raise ValueError('prefill_chunk_size must be a positive integer or None')
        if decode_strategy != 'incremental':
            raise ValueError('prefill chunking requires incremental decode')
    if max_extraction_calls is not None and (type(max_extraction_calls) is not int or max_extraction_calls <= 0):
        raise ValueError('max_extraction_calls must be a positive integer or None')
    profile = inspect_checkpoint(checkpoint)
    import torch
    from models.qwen3 import Qwen3Config, Qwen3ForCausalLM
    from history_memory.runtime import HistoryMemoryModel
    from history_memory.inference import EventNativeGenerator
    if dtype not in {'float32', 'bfloat16', 'float16'}:
        raise ValueError('unsupported inference dtype')
    if dtype != 'float32' and not hasattr(Qwen3ForCausalLM, '_keep_in_fp32_modules_strict'):
        raise RuntimeError('this transformers runtime cannot preserve FP32 gist weights during mixed-dtype loading')

    class FP32GistQwen3ForCausalLM(Qwen3ForCausalLM):
        # Casting every checkpoint tensor to the base dtype would erase the
        # FP32 gist updates before HistoryMemoryModel promotes them again.
        _keep_in_fp32_modules_strict = {
            'gist_embed_tokens', 'gist_q_proj', 'gist_k_proj', 'gist_v_proj',
        }

    config = Qwen3Config.from_pretrained(str(checkpoint), local_files_only=True)
    model = FP32GistQwen3ForCausalLM.from_pretrained(
        str(checkpoint), config=config, local_files_only=True,
        dtype=getattr(torch, dtype), attn_implementation='eager').to(device)
    runtime = HistoryMemoryModel(model)
    runtime.restore_parameter_version(profile['parameter_version'])
    generator = EventNativeGenerator(runtime, decode_strategy=decode_strategy,
        retain_raw_snapshot=retain_raw_snapshot,
        **({"prefill_chunk_size": prefill_chunk_size} if prefill_chunk_size is not None else {}),
        **({"max_extraction_calls": max_extraction_calls} if max_extraction_calls is not None else {}))
    profile['inference_decode_strategy'] = decode_strategy
    profile['inference_prefill_chunk_size'] = prefill_chunk_size
    profile['inference_max_extraction_calls'] = max_extraction_calls
    profile['inference_session_cache_policy'] = generator.session_cache_policy
    return generator, profile


def validate_inference_byte_profile(profile: Mapping[str, Any], dtype: str) -> int:
    """Keep the checkpoint's B/W byte contract tied to the execution dtype."""
    element_bytes = {'float32': 4, 'bfloat16': 2, 'float16': 2}.get(dtype)
    if element_bytes is None:
        raise ValueError('unsupported inference dtype')
    geometry = profile['model_geometry']
    head_dim = geometry.get('head_dim')
    if head_dim is None:
        hidden, heads = geometry.get('hidden_size'), geometry.get('num_attention_heads')
        if type(hidden) is not int or type(heads) is not int or heads <= 0 or hidden % heads:
            raise ValueError('checkpoint lacks a valid KV head dimension')
        head_dim = hidden // heads
    dimensions = (geometry.get('num_hidden_layers'), geometry.get('num_key_value_heads'), head_dim)
    if any(type(value) is not int or value <= 0 for value in dimensions):
        raise ValueError('checkpoint lacks positive KV geometry')
    required = 2 * dimensions[0] * dimensions[1] * dimensions[2] * element_bytes
    policy = profile.get('policy_contract')
    if not isinstance(policy, Mapping) or policy.get('kv_bytes_per_token') != required:
        raise ValueError(
            f'inference dtype {dtype} requires {required} KV bytes per token; '
            'checkpoint policy declares a different byte contract'
        )
    return required


def prepare_request_sequence(profile, tokenizer, payload, *, view_mode, ratio, max_new_tokens):
    """Preflight a finite visible-prefix sequence before loading model weights."""
    from .event_native_policy import EventNativeController
    from .event_native_raw import build_raw_control
    if payload.get('schema') != 'a-event-native-request-sequence-v1':
        raise ValueError('unsupported event-native request-sequence schema')
    requests = payload.get('requests')
    if not isinstance(requests, list) or not requests:
        raise ValueError('request sequence must be a nonempty list')
    packing = profile.get('packing_contract')
    if not isinstance(packing, Mapping) or packing.get('ratios') != profile['declared_supported_ratios']:
        raise ValueError('checkpoint packing ratios must match its declared supported ratios')
    if ratio not in profile['declared_supported_ratios']:
        raise ValueError('requested ratio is absent from the checkpoint profile')
    if view_mode not in {'static', 'policy', 'no_gist', 'full_original', 'full_shared'}:
        raise ValueError('unsupported event-native inference view mode')
    controller = None if view_mode == 'full_original' else EventNativeController(
        tokenizer, packing=dict(packing), policy=profile.get('policy_contract'),
        view_mode=view_mode if view_mode in {'static', 'policy'} else 'policy')
    prepared = []
    for request in requests:
        selection = controller.prepare(request, ratio=ratio, max_new_tokens=max_new_tokens) if controller else None
        if view_mode in {'static', 'policy'}:
            item = selection
        else:
            if not isinstance(request, Mapping) or set(request) - {'session_id', 'decision_key', 'messages', 'tools'}:
                raise ValueError('raw control request must contain visible input fields only')
            if any(not isinstance(request.get(field), str) or not request[field]
                   for field in ('session_id', 'decision_key')):
                raise ValueError('raw control request needs explicit session_id and decision_key')
            store = EventStore.from_messages(request['session_id'], request['messages'])
            evidence = selection.memory.view.evidence_event_ids if selection else ()
            raw_packing = dict(packing)
            context_limit = profile.get('model_geometry', {}).get('max_position_embeddings')
            if type(context_limit) is int and context_limit > 0:
                raw_packing['max_sequence_tokens'] = min(raw_packing['max_sequence_tokens'], context_limit)
            item = build_raw_control(
                store, tokenizer, packing=raw_packing, policy=profile.get('policy_contract'),
                mode=view_mode, evidence_event_ids=evidence, max_new_tokens=max_new_tokens,
                tools=request.get('tools'))
            item.metadata.update(session_id=request['session_id'], decision_key=request['decision_key'])
            item.metadata['configured_max_sequence_tokens'] = packing['max_sequence_tokens']
            if selection:
                item.metadata['shared_controller'] = selection.metadata
        context_limit = profile.get('model_geometry', {}).get('max_position_embeddings')
        if type(context_limit) is int and context_limit > 0:
            logical_end = item.memory.workspace_position_start + len(item.memory.workspace_input_ids) + max_new_tokens
            physical_end = item.memory.costs(ratio)['resident_kv_tokens'] + max_new_tokens
            if max(logical_end, physical_end) > context_limit:
                raise ValueError('prepared request exceeds the checkpoint model context')
        prepared.append(item)
    return prepared


def run_exact_cli_sequence(args, profile, tokenizer, payload, planned_kv_bytes):
    """Execute journaled decisions with the route's optional exact recovery."""
    import os
    from .attempt_journal import AttemptJournal, summarize_attempt_journal
    from .event_native_controls import build_event_native_controller, describe_event_native_route
    from .event_native_eval_policy import load_eval_policy, resolve_event_native_eval_policy
    from .event_native_step import EventNativeDecisionRunner, EventNativeStepError
    from .event_native_costs import summarize_event_native_steps
    from .event_native_policy import EventNativeController
    if type(args.max_generation_calls) is not int or args.max_generation_calls <= 0:
        raise ValueError('finite decision routes require a positive --max-generation-calls cap')
    if not isinstance(payload, Mapping) or set(payload) != {'schema', 'requests'}:
        raise ValueError('exact sequence must contain only schema and visible requests')
    requests = payload['requests']
    if payload['schema'] != 'a-event-native-request-sequence-v1' or not isinstance(requests, list) or not requests:
        raise ValueError('invalid exact request sequence')
    packing = profile.get('packing_contract')
    if not isinstance(packing, Mapping) or packing.get('ratios') != profile['declared_supported_ratios']:
        raise ValueError('checkpoint packing ratios must match its declared supported ratios')
    if type(args.max_new_tokens) is not int or not 0 < args.max_new_tokens <= packing['max_target_tokens']:
        raise ValueError('exact generation reservation must fit the checkpoint target cap')
    context = profile['model_geometry'].get('max_position_embeddings')
    if type(context) is not int or context <= 0:
        raise ValueError('exact inference requires an explicit checkpoint model context')
    # Validate source identities before loading weights. Selection/admission is
    # per decision because an actual earlier draft may create a retained lease.
    previous = {}
    seen = {}
    for request in requests:
        if not isinstance(request, Mapping) or set(request) - {'session_id', 'decision_key', 'messages', 'tools'}:
            raise ValueError('exact requests must contain visible input fields only')
        if any(not isinstance(request.get(field), str) or not request[field]
               for field in ('session_id', 'decision_key')):
            raise ValueError('exact request requires explicit session_id and decision_key')
        messages_value = request.get('messages')
        if not isinstance(messages_value, list) or not messages_value or any(not isinstance(message, Mapping) for message in messages_value):
            raise ValueError('exact request messages must be a nonempty list of mappings')
        store = EventStore.from_messages(request['session_id'], messages_value)
        messages = tuple(message.json_text for message in store.messages)
        tools = request.get('tools')
        tools = [] if tools is None else tools
        if not isinstance(tools, list) or any(not isinstance(tool, Mapping) for tool in tools):
            raise ValueError('exact request tools must be a list of mappings')
        tools_json = json.dumps(tools, sort_keys=True, allow_nan=False)
        old = previous.get(request['session_id'])
        if old:
            EventNativeController._validate_monotone_prefix(old[0], messages)
            if old[1] != tools_json:
                raise ValueError('tools changed within an exact session')
        key = (request['session_id'], request['decision_key'])
        signature = (messages, tools_json)
        if key in seen and seen[key] != signature:
            raise ValueError('exact decision key reused with different input')
        seen[key] = signature
        previous[request['session_id']] = signature
    journal_path = args.output.with_suffix('.attempts.jsonl')
    steps_path = args.output.with_suffix('.steps.jsonl')
    if journal_path.exists() or steps_path.exists():
        raise ValueError('exact sequence artifacts already exist; automatic rerun is disabled')
    eval_path = getattr(args, 'eval_policy', None)
    runtime_policy = getattr(args, '_runtime_policy_contract', None)
    if runtime_policy is None:
        runtime_policy = resolve_event_native_eval_policy(profile, view_mode=args.view_mode,
            policy_override=load_eval_policy(eval_path) if eval_path is not None else None,
            source_path=str(eval_path.resolve()) if eval_path is not None else None)
    controller = build_event_native_controller(
        tokenizer, packing=packing, policy=runtime_policy['effective_policy'],
        view_mode=args.view_mode, model_context=context)
    generator, profile = load_generator(args.checkpoint, device=args.device, dtype=args.dtype,
                                        decode_strategy=args.decode_strategy)
    if generator.kv_bytes_per_token() != planned_kv_bytes:
        raise ValueError('loaded model KV geometry differs from the checkpoint budget contract')
    runner = EventNativeDecisionRunner(
        controller, generator, tokenizer, ratio=args.ratio, max_new_tokens=args.max_new_tokens,
        max_generation_calls=args.max_generation_calls, journal=AttemptJournal(journal_path))
    steps_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    status = 'completed'
    try:
        with steps_path.open('x', encoding='utf-8', newline='\n') as handle:
            for request in requests:
                try:
                    record = runner.run(request)
                except EventNativeStepError as error:
                    record, status = error.record, 'failed'
                except Exception as error:
                    record = {'status': 'failed', 'session_id': request['session_id'],
                              'decision_key': request['decision_key'], 'response': None,
                              'generation_trace': [], 'generation_attempts': 0,
                              'preparation_error': {'type': type(error).__name__, 'message': str(error)}}
                    status = 'failed'
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                handle.flush()
                os.fsync(handle.fileno())
                if status == 'failed':
                    break
    finally:
        cache_before_close = generator.session_cache_info()
        runner.close()
    output = {
        'schema': 'a-event-native-exact-sequence-v1', 'status': status,
        'checkpoint': profile, 'input': str(args.request_input.resolve()),
        'view_mode': args.view_mode, 'ratio': args.ratio, 'max_new_tokens': args.max_new_tokens,
        'route_contract': describe_event_native_route(args.view_mode),
        'runtime_policy_contract': runtime_policy,
        'decode_strategy': args.decode_strategy,
        'session_cache_policy': generator.session_cache_policy,
        'session_cache_before_close': cache_before_close,
        'session_cache_after_close': generator.session_cache_info(),
        'max_generation_calls': args.max_generation_calls,
        'generation_calls_reserved': runner.generation_calls, 'records': records,
        'attempt_journal': str(journal_path), 'step_records': str(steps_path),
        'journal_summary': summarize_attempt_journal(journal_path) if journal_path.exists() else None,
        'cost_summary': summarize_event_native_steps(records,
            attempt_journal=journal_path if journal_path.exists() else None),
        'scope': 'Journaled final native response per fixed visible prefix. Recovery is disabled for one-pass controls and allowed at most once for exact routes; no tools, scorer, or benchmark rollout.',
    }
    return output, 0 if status == 'completed' else 1


def main():
    from .event_native_controls import FINITE_VIEW_MODES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--packed-input', type=Path)
    inputs.add_argument('--request-input', type=Path)
    parser.add_argument('--view-mode', choices=sorted(FINITE_VIEW_MODES | {'policy', 'no_gist', 'full_shared'}))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ratio', type=int, required=True)
    parser.add_argument('--max-new-tokens', type=int, required=True)
    parser.add_argument('--max-generation-calls', type=int)
    parser.add_argument('--eval-policy', type=Path,
                        help='Explicit A evaluation budget/lease policy; requires a finite non-static request route.')
    parser.add_argument('--decode-strategy', choices=('incremental', 'full_recompute'), default='incremental')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dtype', default='float32')
    args = parser.parse_args()
    exit_code = 0
    if args.eval_policy is not None:
        if args.request_input is None or args.view_mode not in FINITE_VIEW_MODES - {'static'}:
            raise ValueError('--eval-policy requires a finite non-static --request-input route')
        if type(args.max_generation_calls) is not int or args.max_generation_calls <= 0:
            raise ValueError('--eval-policy requires a positive --max-generation-calls cap')
    if args.output.exists():
        raise ValueError(f'output already exists: {args.output}')
    profile = inspect_checkpoint(args.checkpoint)
    if args.ratio not in profile['declared_supported_ratios']:
        raise ValueError('requested ratio is absent from the checkpoint profile')
    from dataclasses import asdict
    if args.view_mode not in FINITE_VIEW_MODES and args.max_generation_calls is not None:
        raise ValueError('--max-generation-calls requires a supported finite decision route')
    if args.request_input:
        if args.view_mode is None:
            raise ValueError('--request-input requires explicit --view-mode')
        planned_kv_bytes = validate_inference_byte_profile(profile, args.dtype)
        if args.eval_policy is not None:
            from .event_native_eval_policy import load_eval_policy, resolve_event_native_eval_policy
            args._runtime_policy_contract = resolve_event_native_eval_policy(
                profile, view_mode=args.view_mode, policy_override=load_eval_policy(args.eval_policy),
                source_path=str(args.eval_policy.resolve()))
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint), local_files_only=True)
        payload = json.loads(args.request_input.read_text(encoding='utf-8'))
        if (args.view_mode in EXACT_VIEW_MODES | {'capacity_protect'}
                or args.max_generation_calls is not None):
            output, exit_code = run_exact_cli_sequence(args, profile, tokenizer, payload, planned_kv_bytes)
            summary = {'status': output['status'], 'requests': len(output['records']),
                       'generation_calls_reserved': output['generation_calls_reserved']}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
            print(json.dumps({'output': str(args.output), **summary}))
            if exit_code:
                raise SystemExit(exit_code)
            return
        prepared = prepare_request_sequence(
            profile, tokenizer, payload, view_mode=args.view_mode,
            ratio=args.ratio, max_new_tokens=args.max_new_tokens)
        generator, profile = load_generator(args.checkpoint, device=args.device, dtype=args.dtype,
                                            decode_strategy=args.decode_strategy)
        if generator.kv_bytes_per_token() != planned_kv_bytes:
            raise ValueError('loaded model KV geometry differs from the checkpoint budget contract')
        records = []
        try:
            for request, item in zip(payload['requests'], prepared, strict=True):
                with generator.decision_scope(session_id=request['session_id']):
                    result = generator.generate(item.memory, ratio=args.ratio, max_new_tokens=args.max_new_tokens)
                records.append({'controller': item.metadata, 'prepared_input': memory_to_dict(item.memory),
                                'result': asdict(result), 'session_cache_after': generator.session_cache_info()})
        finally:
            cache_before_close = generator.session_cache_info()
            generator.close_session()
        output = {'schema': 'a-event-native-controlled-generation-v1', 'checkpoint': profile,
                  'input': str(args.request_input.resolve()), 'view_mode': args.view_mode,
                  'ratio': args.ratio, 'max_new_tokens': args.max_new_tokens, 'records': records,
                  'decode_strategy': args.decode_strategy,
                  'session_cache_policy': generator.session_cache_policy,
                  'session_cache_before_close': cache_before_close,
                  'session_cache_after_close': generator.session_cache_info(),
                  'scope': 'Fixed visible-prefix sequence. Static/policy use training-parity packing; raw controls use explicit A representation contracts with shared pre-draft evidence. No tool execution, official scorer, or post-draft exact recovery.'}
        summary = {'requests': len(records), 'generated_tokens': sum(len(record['result']['token_ids']) for record in records)}
    else:
        if args.view_mode is not None:
            raise ValueError('--view-mode applies only to --request-input')
        memory = memory_from_dict(json.loads(args.packed_input.read_text(encoding='utf-8')))
        generator, profile = load_generator(args.checkpoint, device=args.device, dtype=args.dtype,
                                            decode_strategy=args.decode_strategy)
        result = generator.generate(memory, ratio=args.ratio, max_new_tokens=args.max_new_tokens)
        output = {'schema': 'a-event-native-generation-v1', 'checkpoint': profile,
                  'input': str(args.packed_input.resolve()), 'ratio': args.ratio,
                  'max_new_tokens': args.max_new_tokens, 'result': asdict(result),
                  'decode_strategy': args.decode_strategy,
                  'scope': 'Reference greedy generation from caller-selected memory; no tool execution, scorer, or A budget-controller validation.'}
        summary = {'finish_reason': result.finish_reason, 'generated_tokens': len(result.token_ids)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output), **summary}))


if __name__ == '__main__':
    main()
