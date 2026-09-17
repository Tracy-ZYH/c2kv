"""Finite loopback serving for the event-native exact decision runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

from .attempt_journal import AttemptJournal, summarize_attempt_journal
from .event_native import inspect_checkpoint, load_generator, validate_inference_byte_profile
from .event_native_controls import (
    ALL_VIEW_MODES, build_event_native_controller, describe_event_native_route,
)
from .always_compress import ALWAYS_COMPRESSION_POLICY
from .event_native_always import NATIVE_ALWAYS_ROUTE_MODES
from .event_native_eval_policy import load_eval_policy, resolve_event_native_eval_policy
from .event_native_eval_packing import resolve_eval_packing
from .event_native_step import EventNativeDecisionRunner
from .event_native_costs import read_event_native_steps, summarize_event_native_steps


GENERATION_BACKENDS = ('native', 'sglang')


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return parsed


def positive_seconds(value):
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError('must be positive and finite')
    return parsed


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--checkpoint', type=Path, required=True)
    result.add_argument('--out', type=Path, required=True)
    result.add_argument('--run-id', required=True)
    result.add_argument('--model-name', default='c2kv-event-native')
    result.add_argument('--benchmark', choices=('bfcl', 'acebench', 'tau2', 'toolsandbox', 'acon_appworld'), default='bfcl',
                        help='Frozen task/session namespace; requests must match.')
    result.add_argument('--source-profile', choices=('native-v1', 'acebench-text-actions-v1', 'openai-single-task-v1'),
                        default='native-v1', help='Explicit history and draft protocol adapter.')
    result.add_argument('--view-mode', choices=sorted(ALL_VIEW_MODES), required=True)
    result.add_argument('--compression-policy', choices=(ALWAYS_COMPRESSION_POLICY,),
                        help='Required explicit opt-in for the new always-compress routes.')
    result.add_argument('--history-view-protocol', choices=('fixed-budget-main',),
                        default='fixed-budget-main')
    result.add_argument('--ratio', type=positive_int, required=True)
    result.add_argument('--max-new-tokens', type=positive_int, required=True)
    result.add_argument('--decode-strategy', choices=('incremental', 'full_recompute'), default='incremental')
    result.add_argument('--prefill-chunk-size', type=positive_int,
                        help='Bound system and raw incremental prefill; no input truncation.')
    result.add_argument('--task-ids', required=True, help='Frozen comma-separated official task IDs.')
    result.add_argument('--max-decisions', type=positive_int, required=True)
    result.add_argument('--max-generation-calls', type=positive_int, required=True)
    result.add_argument('--max-extraction-calls', type=positive_int,
                        help='Cumulative real encoder-call cap; cache hits are free.')
    result.add_argument('--eval-policy', type=Path,
                        help='Explicit A evaluation budget/lease policy; static keeps its training policy.')
    result.add_argument('--eval-capacity', type=Path,
                        help='Explicit evaluation workspace/sequence caps; keeps training chunk geometry.')
    result.add_argument('--s0-config', type=Path,
                        help='Required explicit controller configuration for the native S0 route.')
    result.add_argument('--shadow-feature-config', type=Path,
                        help='Explicit optional generation-feature capture configuration.')
    result.add_argument('--generation-backend', choices=GENERATION_BACKENDS, default='native',
                        help='Generation implementation. D3/G--P controllers require sglang.')
    result.add_argument('--sglang-backend-url',
                        help='Bare SGLang engine base URL, for example http://127.0.0.1:36100.')
    result.add_argument('--sglang-timeout-seconds', type=positive_seconds, default=10800.0,
                        help='Per-request timeout for the external SGLang engine.')
    result.add_argument('--max-wall-seconds', type=positive_seconds, required=True)
    result.add_argument('--device', default='cpu')
    result.add_argument('--no-raw-snapshot', action='store_true',
                        help='Keep gist/system memo but do not retain raw KV across requests.')
    result.add_argument('--npu-allocator-metrics', action='store_true',
                        help='Measure process-local NPU allocator peaks around each generation.')
    result.add_argument('--dtype', choices=('float32', 'bfloat16', 'float16'), default='float32')
    result.add_argument('--host', choices=('127.0.0.1', '::1', 'localhost'), default='127.0.0.1')
    result.add_argument('--port', type=int, default=0)
    result.add_argument('--torch-threads', type=positive_int, default=1)
    result.add_argument('--serve-child', action='store_true', help=argparse.SUPPRESS)
    return result


def _saved_cost_summary(out):
    steps_path, journal_path = out / 'steps.jsonl', out / 'attempts.jsonl'
    steps = read_event_native_steps(steps_path) if steps_path.exists() else {
        'records': [], 'truncated_tail': False}
    return summarize_event_native_steps(steps['records'],
        attempt_journal=journal_path if journal_path.exists() else None,
        steps_truncated_tail=steps['truncated_tail'])


def _validate_allocator_device(args):
    if getattr(args, 'npu_allocator_metrics', False) and args.device.split(':', 1)[0] != 'npu':
        raise ValueError('NPU allocator metrics require an npu device')


def _read_s0_configuration(args):
    path = getattr(args, 's0_config', None)
    if path is None:
        return None, None
    config_bytes = path.read_bytes()
    config = json.loads(config_bytes)
    if not isinstance(config, dict):
        raise ValueError('s0-config must contain a JSON object')
    return config, {
        'config': config,
        'source': str(path.resolve()),
        'sha256': hashlib.sha256(config_bytes).hexdigest(),
    }


def _controller_requires_sglang(config):
    return isinstance(config, dict) and (
        'post_draft_recovery' in config or 'gp_experiments' in config
    )


def _normalize_sglang_url(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('SGLang generation requires --sglang-backend-url')
    value = value.strip()
    parsed = urlsplit(value)
    if (
        parsed.scheme != 'http'
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ('', '/')
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError('sglang-backend-url must be a bare HTTP base URL')
    return value.rstrip('/')


def _validate_generation_backend(args, *, s0_config=None):
    backend = getattr(args, 'generation_backend', 'native')
    if backend not in GENERATION_BACKENDS:
        raise ValueError(f'Unsupported generation backend: {backend!r}')
    requires_sglang = _controller_requires_sglang(s0_config)
    if requires_sglang and backend != 'sglang':
        raise ValueError('D3 post_draft_recovery and G--P require generation-backend=sglang')
    url = getattr(args, 'sglang_backend_url', None)
    if backend == 'sglang':
        args.sglang_backend_url = _normalize_sglang_url(url)
        if getattr(args, 'decode_strategy', 'incremental') != 'incremental':
            raise ValueError('SGLang generation requires decode-strategy=incremental')
        if getattr(args, 'npu_allocator_metrics', False):
            raise ValueError(
                'Process-local NPU allocator metrics cannot measure the external SGLang engine'
            )
        if getattr(args, 'device', 'cpu') != 'cpu':
            raise ValueError('SGLang controller process requires device=cpu')
        if getattr(args, 'max_extraction_calls', None) is None:
            raise ValueError('SGLang generation requires an explicit max-extraction-calls cap')
    elif url is not None:
        raise ValueError('sglang-backend-url requires generation-backend=sglang')
    return backend


def _shadow_feature_configuration(args, tokenizer):
    if getattr(args, 'shadow_feature_config', None) is None:
        return None, None
    from history_memory.shadow_features import ShadowFeatureConfig
    feature_bytes = args.shadow_feature_config.read_bytes()
    feature_config = json.loads(feature_bytes)
    allowed = {'enabled', 'prefill_layer', 'memgen_layer'}
    if not isinstance(feature_config, dict) or set(feature_config) - allowed:
        raise ValueError('Unsupported shadow feature configuration fields')
    tokenizer_path = args.checkpoint / 'tokenizer.json'
    tokenizer_binding = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    shadow = ShadowFeatureConfig(
        **feature_config,
        decode_token_ids=lambda ids: tokenizer.decode(
            list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False),
        model_binding=str(args.checkpoint.resolve()),
        tokenizer_binding=tokenizer_binding,
    )
    contract = {
        'config': feature_config,
        'config_sha256': hashlib.sha256(feature_bytes).hexdigest(),
        'checkpoint': str(args.checkpoint.resolve()),
        'tokenizer_sha256': tokenizer_binding,
        'protocol_special_tokens_preserved': True,
    }
    return shadow, contract


def _validate_eos_token_ids(value, *, source):
    values = (value,) if type(value) is int else tuple(value or ())
    if not values or any(type(item) is not int or item < 0 for item in values):
        raise ValueError(f'{source} must declare valid eos_token_id values')
    return values


def _checkpoint_eos_token_ids(checkpoint, tokenizer):
    generation_path = checkpoint / 'generation_config.json'
    if generation_path.is_file():
        generation_config = json.loads(generation_path.read_text(encoding='utf-8'))
        if not isinstance(generation_config, dict):
            raise ValueError('checkpoint generation_config.json must contain an object')
        source = 'checkpoint_generation_config.eos_token_id'
        return _validate_eos_token_ids(
            generation_config.get('eos_token_id'), source=source
        ), source
    source = 'checkpoint_tokenizer.eos_token_id'
    return _validate_eos_token_ids(tokenizer.eos_token_id, source=source), source


def _build_generator(
    args,
    *,
    profile,
    model_context,
    tokenizer,
    journal_path,
    s0_config,
    shadow_feature_config,
):
    backend = args.generation_backend
    if backend == 'native':
        generator, loaded_profile = load_generator(
            args.checkpoint,
            device=args.device,
            dtype=args.dtype,
            retain_raw_snapshot=not getattr(args, 'no_raw_snapshot', False),
            decode_strategy=args.decode_strategy,
            **(
                {'prefill_chunk_size': args.prefill_chunk_size}
                if getattr(args, 'prefill_chunk_size', None) is not None else {}
            ),
            **(
                {'max_extraction_calls': args.max_extraction_calls}
                if getattr(args, 'max_extraction_calls', None) is not None else {}
            ),
        )
        if shadow_feature_config is not None:
            generator.configure_shadow_features(shadow_feature_config)
        return generator, loaded_profile

    from history_memory.sglang_generator import SGLangEventNativeGenerator
    gp = s0_config.get('gp_experiments') if isinstance(s0_config, dict) else None
    encoding_scope = gp.get('G', 'current') if isinstance(gp, dict) else 'current'
    eos_token_ids, eos_source = _checkpoint_eos_token_ids(args.checkpoint, tokenizer)
    generator = SGLangEventNativeGenerator(
        args.sglang_backend_url,
        expected_model_path=args.checkpoint.resolve(),
        model_context=model_context,
        max_new_tokens=args.max_new_tokens,
        max_generation_calls=args.max_generation_calls,
        max_extraction_calls=args.max_extraction_calls,
        timeout_seconds=args.sglang_timeout_seconds,
        eos_token_ids=eos_token_ids,
        eos_source=eos_source,
        journal_path=journal_path.with_name('sglang_http.jsonl'),
        sampling_params={'temperature': 0.0, 'seed': 0},
        shadow_feature_config=shadow_feature_config,
        encoding_scope=encoding_scope,
    )
    return generator, profile


def _serve(args):
    s0_config, s0_contract = _read_s0_configuration(args)
    generation_backend = _validate_generation_backend(args, s0_config=s0_config)
    _validate_allocator_device(args)
    started = time.monotonic()
    deadline = started + args.max_wall_seconds
    task_ids = tuple(item.strip() for item in args.task_ids.split(','))
    if not args.run_id.strip() or not args.model_name.strip():
        raise ValueError('run and model identities must be nonempty')
    if not all(task_ids) or len(task_ids) != len(set(task_ids)):
        raise ValueError('task IDs must be nonempty and unique')
    if not 0 <= args.port <= 65535:
        raise ValueError('invalid loopback port')
    source_profile = getattr(args, 'source_profile', 'native-v1')
    compression_policy = getattr(args, 'compression_policy', None)
    history_view_protocol = getattr(args, 'history_view_protocol', 'fixed-budget-main')
    route_contract = describe_event_native_route(
        args.view_mode,
        compression_policy=compression_policy,
        history_view_protocol=history_view_protocol,
    )
    if source_profile == 'acebench-text-actions-v1':
        if args.benchmark != 'acebench':
            raise ValueError('ACE textual source requires the acebench namespace')
        if args.view_mode == 'static':
            raise ValueError('ACE textual source has no training-static adapter')
        if args.view_mode in NATIVE_ALWAYS_ROUTE_MODES:
            raise ValueError('Always-compress P0 supports only source_profile=native-v1')
    args.out.mkdir(parents=True, exist_ok=False)
    journal_path = args.out / 'attempts.jsonl'
    manifest = {
        'schema': 'a-event-native-server-v1', 'status': 'loading',
        'run_id': args.run_id, 'model_name': args.model_name, 'view_mode': args.view_mode,
        'benchmark': args.benchmark,
        'source_profile': source_profile,
        'route_contract': route_contract,
        'compression_policy': compression_policy,
        'history_view_protocol': history_view_protocol,
        'allowed_task_ids': list(task_ids), 'checkpoint_path': str(args.checkpoint.resolve()),
        'ratio': args.ratio, 'max_new_tokens': args.max_new_tokens,
        'eval_policy_path': str(args.eval_policy.resolve()) if args.eval_policy is not None else None,
        'decode_strategy': args.decode_strategy,
        'max_decisions': args.max_decisions, 'max_generation_calls': args.max_generation_calls,
        'max_wall_seconds': args.max_wall_seconds, 'device': args.device, 'dtype': args.dtype,
        'generation_backend': generation_backend,
        'sglang_backend_url': (
            args.sglang_backend_url if generation_backend == 'sglang' else None
        ),
        'sampling': {'mode': 'greedy', 'temperature': 0, 'seed': 0},
        'attempt_journal': str(journal_path.resolve()),
        'sglang_http_journal': (
            str(journal_path.with_name('sglang_http.jsonl').resolve())
            if generation_backend == 'sglang' else None
        ),
        'scope': 'Final event-native assistant transport; tools and scoring remain in the external official harness.',
    }
    save_json(args.out / 'startup.json', manifest)
    server = api = generator = None
    stop_requested = False
    old_handlers = {}

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    try:
        profile = inspect_checkpoint(args.checkpoint)
        expected_bytes = validate_inference_byte_profile(profile, args.dtype)
        if args.ratio not in profile['declared_supported_ratios']:
            raise ValueError('ratio is absent from checkpoint contract')
        if args.max_new_tokens > profile['packing_contract']['max_target_tokens']:
            raise ValueError('generation reservation exceeds checkpoint target cap')
        context = profile['model_geometry'].get('max_position_embeddings')
        if type(context) is not int or context <= 0:
            raise ValueError('checkpoint must declare model context')
        runtime_policy = resolve_event_native_eval_policy(profile, view_mode=args.view_mode,
            policy_override=load_eval_policy(args.eval_policy) if args.eval_policy is not None else None,
            source_path=str(args.eval_policy.resolve()) if args.eval_policy is not None else None)
        manifest['runtime_policy_contract'] = runtime_policy
        runtime_packing = resolve_eval_packing(profile, getattr(args, 'eval_capacity', None))
        manifest['runtime_packing_contract'] = runtime_packing
        save_json(args.out / 'startup.json', manifest)
        from transformers import AutoTokenizer
        from .event_native_api import EventNativeAPI, make_server
        controller_factory = build_event_native_controller
        runner_type, api_type = EventNativeDecisionRunner, EventNativeAPI
        source_kwargs = {}
        if source_profile == 'openai-single-task-v1':
            from .single_task_harness_api import SingleTaskHarnessAPI
            api_type = SingleTaskHarnessAPI
        if source_profile == 'acebench-text-actions-v1':
            from .acebench_controls import build_acebench_controller
            from .acebench_runtime import (
                AceEventNativeAPI, AceEventNativeDecisionRunner, describe_ace_source_contract,
            )
            controller_factory = build_acebench_controller
            runner_type, api_type = AceEventNativeDecisionRunner, AceEventNativeAPI
            manifest['source_protocol_contract'] = describe_ace_source_contract()
            source_kwargs['source_protocol_contract'] = manifest['source_protocol_contract']
        if generation_backend == 'native':
            import torch
            if args.device.split(':', 1)[0] == 'npu':
                import torch_npu  # Register the explicitly selected optional device backend.
            torch.set_num_threads(args.torch_threads)
        tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint), local_files_only=True)
        s0_kwargs = {}
        if s0_config is not None:
            from .event_native_always import NATIVE_S0_MODE
            if args.view_mode != NATIVE_S0_MODE:
                raise ValueError('s0-config requires the native S0 route')
            s0_kwargs['s0_config'] = s0_config
            manifest['s0_controller_contract'] = s0_contract
        controller = controller_factory(tokenizer, packing=runtime_packing['effective_packing'],
            policy=runtime_policy['effective_policy'], view_mode=args.view_mode, model_context=context,
            **s0_kwargs,
            **({'compression_policy': compression_policy,
                'history_view_protocol': history_view_protocol}
               if source_profile in ('native-v1', 'openai-single-task-v1') else {}))
        shadow_feature_config, shadow_contract = _shadow_feature_configuration(args, tokenizer)
        generator, profile = _build_generator(
            args,
            profile=profile,
            model_context=context,
            tokenizer=tokenizer,
            journal_path=journal_path,
            s0_config=s0_config,
            shadow_feature_config=shadow_feature_config,
        )
        if shadow_contract is not None:
            manifest['shadow_feature_contract'] = shadow_contract
        if getattr(args, 'npu_allocator_metrics', False):
            from .event_native_allocator import NpuAllocatorMeasuredGenerator
            generator = NpuAllocatorMeasuredGenerator(generator)
            manifest['allocator_measurement'] = generator.measurement_contract
        elif generation_backend == 'sglang':
            manifest['allocator_measurement'] = {
                'status': 'not_measured',
                'scope': 'external_sglang_engine',
                'reason': 'Process-local allocator counters do not cover the remote engine.',
            }
            manifest['backend_accounting'] = {
                'source': 'sglang_native_generation_response',
                'scope': 'exact encoded/reused chunks and logical KV bytes reported by the engine',
            }
        manifest['session_cache_policy'] = generator.session_cache_policy
        if generator.kv_bytes_per_token() != expected_bytes:
            raise ValueError('loaded KV geometry differs from the declared budget')
        if time.monotonic() >= deadline:
            raise TimeoutError('wall budget expired while loading the model')
        runner = runner_type(controller, generator, tokenizer, ratio=args.ratio,
            max_new_tokens=args.max_new_tokens, max_generation_calls=args.max_generation_calls,
            journal=AttemptJournal(journal_path))
        api = api_type(runner, run_id=args.run_id, model_name=args.model_name,
            benchmark=args.benchmark,
            view_mode=args.view_mode, max_new_tokens=args.max_new_tokens,
            allowed_task_ids=task_ids, max_decisions=args.max_decisions,
            deadline_monotonic=deadline, steps_path=args.out / 'steps.jsonl',
            runtime_policy_contract=runtime_policy,
            **({'compression_policy': compression_policy,
                'history_view_protocol': history_view_protocol}
               if source_profile in ('native-v1', 'openai-single-task-v1') else {}),
            **source_kwargs)
        server = make_server(api, host=args.host, port=args.port)
        server.timeout = 0.25
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.signal(signum, request_stop)
        host, port = server.server_address[:2]
        url_host = f'[{host}]' if ':' in host else host
        manifest.update(status='ready', checkpoint=profile, base_url=f'http://{url_host}:{port}/v1',
                        ready_elapsed_seconds=time.monotonic() - started)
        save_json(args.out / 'ready.json', manifest)
        print(json.dumps({'status': 'ready', 'base_url': manifest['base_url'],
                          'ready_file': str((args.out / 'ready.json').resolve())}), flush=True)
        while not stop_requested and time.monotonic() < deadline:
            health = api.health()
            if health['terminal'] or health['decisions_reserved'] >= args.max_decisions:
                break
            server.handle_request()
        health = api.health()
        manifest.update(status='stopped', api_health=health,
                        stop_reason='signal' if stop_requested else
                        health['terminal_reason'] if health['terminal'] else
                        'decision_cap' if health['decisions_reserved'] >= args.max_decisions else 'wall_cap')
    except BaseException as error:
        manifest.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        if server is not None:
            server.server_close()
        if generator is not None:
            manifest['session_cache_before_close'] = generator.session_cache_info()
            generator.close_session()
            manifest['session_cache_after_close'] = generator.session_cache_info()
        for signum, old in old_handlers.items():
            signal.signal(signum, old)
        manifest['wall_seconds'] = time.monotonic() - started
        manifest['wall_seconds_final'] = True
        manifest['journal_summary'] = summarize_attempt_journal(journal_path) if journal_path.exists() else None
        if api is not None:
            manifest['api_health'] = api.health()
        try:
            manifest['cost_summary'] = _saved_cost_summary(args.out)
        except Exception as error:
            manifest['cost_summary_error'] = {'type': type(error).__name__, 'message': str(error)}
            manifest['status'] = 'failed'
        save_json(args.out / 'final.json', manifest)
    print(json.dumps({'status': manifest['status'], 'stop_reason': manifest['stop_reason'],
                      'output': str(args.out.resolve())}), flush=True)
    if manifest['status'] == 'failed':
        raise RuntimeError('Server stopped with invalid persisted cost inventory; see final.json')


def _supervisor_path(out):
    return out.parent / f'{out.name}.supervisor.json'


def _child_command(args):
    command = [
        sys.executable,
        '-m',
        'benchmarks.memory_runtime.event_native_server',
        '--serve-child',
        '--checkpoint', str(args.checkpoint.resolve()),
        '--out', str(args.out.resolve()),
        '--run-id', args.run_id,
        '--model-name', args.model_name,
        '--benchmark', args.benchmark,
        '--source-profile', getattr(args, 'source_profile', 'native-v1'),
        '--view-mode', args.view_mode,
        '--ratio', str(args.ratio),
        '--max-new-tokens', str(args.max_new_tokens),
        '--decode-strategy', args.decode_strategy,
        '--task-ids', args.task_ids,
        '--max-decisions', str(args.max_decisions),
        '--max-generation-calls', str(args.max_generation_calls),
        '--max-wall-seconds', str(args.max_wall_seconds),
        '--generation-backend', getattr(args, 'generation_backend', 'native'),
        '--device', args.device,
        '--dtype', args.dtype,
        '--host', args.host,
        '--port', str(args.port),
        '--torch-threads', str(args.torch_threads),
    ]
    if getattr(args, 'compression_policy', None) is not None:
        command.extend(['--compression-policy', args.compression_policy])
    command.extend([
        '--history-view-protocol',
        getattr(args, 'history_view_protocol', 'fixed-budget-main'),
    ])
    if args.eval_policy is not None:
        command.extend(['--eval-policy', str(args.eval_policy.resolve())])
    if getattr(args, 'max_extraction_calls', None) is not None:
        command.extend(['--max-extraction-calls', str(args.max_extraction_calls)])
    if getattr(args, 'prefill_chunk_size', None) is not None:
        command.extend(['--prefill-chunk-size', str(args.prefill_chunk_size)])
    if getattr(args, 'eval_capacity', None) is not None:
        command.extend(['--eval-capacity', str(args.eval_capacity.resolve())])
    if getattr(args, 's0_config', None) is not None:
        command.extend(['--s0-config', str(args.s0_config.resolve())])
    if getattr(args, 'shadow_feature_config', None) is not None:
        command.extend(['--shadow-feature-config', str(args.shadow_feature_config.resolve())])
    if getattr(args, 'generation_backend', 'native') == 'sglang':
        command.extend([
            '--sglang-backend-url', args.sglang_backend_url,
            '--sglang-timeout-seconds', str(args.sglang_timeout_seconds),
        ])
    if getattr(args, 'no_raw_snapshot', False):
        command.append('--no-raw-snapshot')
    if getattr(args, 'npu_allocator_metrics', False):
        command.append('--npu-allocator-metrics')
    return command


def _hard_stop_owned_child(process):
    if process.poll() is None:
        if os.name == 'posix':
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    return process.wait(timeout=5)


def _supervise(args, *, command=None):
    s0_config, _ = _read_s0_configuration(args)
    _validate_generation_backend(args, s0_config=s0_config)
    _validate_allocator_device(args)
    started = time.monotonic()
    out = args.out.resolve()
    receipt_path = _supervisor_path(out)
    if out.exists():
        raise FileExistsError(f'output already exists: {out}')
    if receipt_path.exists():
        raise FileExistsError(f'supervisor receipt already exists: {receipt_path}')
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    child_command = list(command if command is not None else _child_command(args))
    receipt = {
        'schema': 'a-event-native-server-supervisor-v1',
        'status': 'prepared',
        'output': str(out),
        'maximum_wall_seconds': args.max_wall_seconds,
        'deadline_owner': 'parent_process',
        'owned_child_only': True,
        'child_command': child_command,
        'child_pid': None,
        'child_returncode': None,
        'hard_cutoff': None,
        'wall_seconds': 0.0,
        'wall_seconds_final': False,
    }
    save_json(receipt_path, receipt)
    process = None
    status = 'failed'
    caught = None
    try:
        root = Path(__file__).resolve().parents[2]
        environment = os.environ.copy()
        source_paths = [str(root / 'python'), str(root)]
        inherited_path = environment.get('PYTHONPATH')
        if inherited_path:
            source_paths.append(inherited_path)
        environment['PYTHONPATH'] = os.pathsep.join(source_paths)
        remaining = args.max_wall_seconds - (time.monotonic() - started)
        if remaining <= 0:
            cutoff_wall = time.monotonic() - started
            status = 'hard_wall_cutoff'
            receipt['hard_cutoff'] = {
                'reason': 'maximum_wall_seconds',
                'wall_seconds_at_cutoff': cutoff_wall,
                'process_action': 'not_started',
            }
        else:
            process = subprocess.Popen(
                child_command,
                cwd=root,
                env=environment,
                start_new_session=os.name == 'posix',
                creationflags=(
                    getattr(subprocess, 'CREATE_NO_WINDOW', 0)
                    if os.name == 'nt'
                    else 0
                ),
            )
            receipt.update(status='running', child_pid=process.pid)
            save_json(receipt_path, receipt)
            remaining = max(
                0.0,
                args.max_wall_seconds - (time.monotonic() - started),
            )
            try:
                returncode = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                cutoff_wall = time.monotonic() - started
                status = 'hard_wall_cutoff'
                receipt['hard_cutoff'] = {
                    'reason': 'maximum_wall_seconds',
                    'wall_seconds_at_cutoff': cutoff_wall,
                    'process_action': (
                        'SIGKILL' if os.name == 'posix' else 'TerminateProcess'
                    ),
                }
                returncode = _hard_stop_owned_child(process)
            else:
                status = 'completed' if returncode == 0 else 'failed'
            receipt['child_returncode'] = returncode
    except BaseException as error:
        caught = error
        receipt['error'] = {'type': type(error).__name__, 'message': str(error)}
    finally:
        if process is not None and process.poll() is None:
            receipt['child_returncode'] = _hard_stop_owned_child(process)
        if status == 'hard_wall_cutoff' and out.exists():
            try:
                receipt['cost_summary'] = _saved_cost_summary(out)
            except Exception as error:
                receipt['cost_summary_error'] = {'type': type(error).__name__, 'message': str(error)}
        receipt.update(
            status=status,
            wall_seconds=time.monotonic() - started,
            wall_seconds_final=True,
        )
        save_json(receipt_path, receipt)
    if caught is not None:
        raise caught
    return receipt


def main(argv=None):
    args = parser().parse_args(argv)
    if args.serve_child:
        _serve(args)
        return
    receipt = _supervise(args)
    if receipt['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
