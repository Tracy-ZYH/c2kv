from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_agent_tool_definition_c2kv import (  # noqa: E402
    _build_tool_cache,
    _build_tool_chunks,
    _extract_tool_name,
    _generate_from_input_ids,
    _load_model,
    _normalize_text,
    _prefill_system,
    _prefill_tokens_with_cache,
    _setup_device,
)
from eval_agent_tool_definition_hybrid_router import (  # noqa: E402
    _as_tool_list,
    _rank_tools,
    _render_tool_definition,
    _split_topk_tools,
    _tool_name,
)
from train.train_data_multiturn import _chat_template_ids  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _json_loads(value: Any, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default if default is not None else value


def _iter_samples(data_dir: Path, file_glob: str, limit: Optional[int] = None) -> Iterable[tuple[Path, Dict[str, Any]]]:
    seen = 0
    for path in sorted(data_dir.glob(file_glob)):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                yield path, json.loads(line)
                seen += 1
                if limit is not None and seen >= limit:
                    return


def _config(sample: Dict[str, Any]) -> Dict[str, Any]:
    config = _json_loads(sample.get("config"), {}) or {}
    return config if isinstance(config, dict) else {}


def _status(sample: Dict[str, Any]) -> Dict[str, Any]:
    status = _json_loads(sample.get("task_status"), {}) or {}
    return status if isinstance(status, dict) else {}


def _messages(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    messages = _json_loads(sample.get("messages"), []) or []
    return messages if isinstance(messages, list) else []


def _tools(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    tool_calls = _json_loads(sample.get("tool_calls"), {}) or {}
    if isinstance(tool_calls, dict) and isinstance(tool_calls.get("tools"), list):
        return tool_calls["tools"]
    if isinstance(tool_calls, list):
        return tool_calls
    return []


def _function_payload(call: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    name = function.get("name") or call.get("name") or call.get("tool_name") or call.get("function_name")
    arguments = function.get("arguments") or call.get("arguments") or call.get("args") or {}
    if isinstance(arguments, str):
        if not arguments.strip():
            arguments = {}
        else:
            try:
                arguments = json.loads(arguments)
            except Exception:
                arguments = arguments
    if not name:
        return None
    return {"name": str(name), "arguments": arguments}


def _first_tool_call(messages: Sequence[Dict[str, Any]]) -> Optional[tuple[int, Dict[str, Any]]]:
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if isinstance(call, dict):
                payload = _function_payload(call)
                if payload:
                    return index, payload
    return None


def _normal_content(message: Dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False) if content is not None else ""


def _tool_call_instruction_message() -> Dict[str, str]:
    return {
        "role": "user",
        "content": (
            "Predict the next tool call for this task. Output only the tool call in exactly this format:\n"
            "Action:\n"
            "<tool_call>\n"
            "{\"name\":\"tool_name\",\"arguments\":{}}\n"
            "</tool_call>\n"
            "Do not explain or include any other text."
        ),
    }


def _prompt_messages(
    messages: Sequence[Dict[str, Any]],
    first_tool_index: int,
    add_instruction: bool = True,
) -> List[Dict[str, str]]:
    rendered = []
    for message in messages[:first_tool_index]:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "system":
            continue
        if role == "tool":
            role = "user"
        content = _normal_content(message).strip()
        if content:
            rendered.append({"role": role or "user", "content": content})
    if add_instruction:
        rendered.append(_tool_call_instruction_message())
    return rendered


def _router_query(config: Dict[str, Any], prompt_messages: Sequence[Dict[str, str]], args: argparse.Namespace) -> str:
    task = str(config.get("task_str") or config.get("task") or "")
    user_messages = [message["content"] for message in prompt_messages if message.get("role") == "user"]
    if args.router_scope == "task":
        return task
    if args.router_scope == "last_user":
        return user_messages[-1] if user_messages else task
    if args.router_scope == "all_prompt":
        return "\n".join(user_messages)
    return "\n".join(item for item in [task, *user_messages] if item)


def _target_answer(payload: Dict[str, Any]) -> str:
    return "Action:\n<tool_call>\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n</tool_call>"


def _parse_pred_call(text: str) -> Optional[Dict[str, Any]]:
    blocks = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text or "", flags=re.S)
    candidates = blocks or [text]
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except Exception:
            value = None
        if isinstance(value, dict):
            function = value.get("function") if isinstance(value.get("function"), dict) else {}
            name = value.get("name") or function.get("name") or value.get("tool_name") or value.get("function_name")
            arguments = value.get("arguments") or function.get("arguments") or value.get("args") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except Exception:
                    pass
            if name:
                return {"name": str(name), "arguments": arguments}
    name = _extract_tool_name(text)
    return {"name": name, "arguments": {}} if name else None


def _flatten_args(value: Any, prefix: str = "") -> Dict[str, str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return {prefix or "$": value}
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_args(item, child))
        return out
    if isinstance(value, list):
        out = {}
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]" if prefix else f"[{index}]"
            out.update(_flatten_args(item, child))
        return out
    return {prefix or "$": json.dumps(value, ensure_ascii=False, sort_keys=True)}


def _f1(pred: set[str], gold: set[str]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    hit = len(pred & gold)
    if hit == 0:
        return 0.0
    precision = hit / len(pred)
    recall = hit / len(gold)
    return 2 * precision * recall / (precision + recall)


def _arg_f1s(target: Dict[str, Any], pred: Optional[Dict[str, Any]]) -> tuple[float, float]:
    if pred is None:
        return 0.0, 0.0
    pred_args = _flatten_args(pred.get("arguments", {}) if pred else {})
    gold_args = _flatten_args(target.get("arguments", {}))
    name_f1 = _f1(set(pred_args.keys()), set(gold_args.keys()))
    value_f1 = _f1(
        {f"{key}={value}" for key, value in pred_args.items()},
        {f"{key}={value}" for key, value in gold_args.items()},
    )
    return name_f1, value_f1


def _system_prompt(config: Dict[str, Any]) -> str:
    prompts = config.get("system_prompts") if isinstance(config.get("system_prompts"), dict) else {}
    base = prompts.get("agent") or "You are a helpful assistant."
    return (
        base.rstrip()
        + "\n\nWhen asked to predict or perform a tool call, output only:\n"
        + "Action:\n<tool_call>\n{\"name\":\"tool_name\",\"arguments\":{}}\n</tool_call>"
    )


def _tool_definition_ids(tokenizer: Any, tools: Sequence[Dict[str, Any]], title: str = "Tool definitions") -> List[int]:
    return _chat_template_ids(
        tokenizer,
        [{"role": "user", "content": f"{title}:\n" + _render_tool_definition(tools)}],
    )


def _build_full_prefix(model: Any, tokenizer: Any, system_prompt: str, tools: Sequence[Dict[str, Any]], args: argparse.Namespace):
    system_ids = _chat_template_ids(
        tokenizer,
        [{"role": "system", "content": system_prompt}],
        keep_bos=True,
        max_length=args.max_system_length,
    )
    system_input_ids = torch.tensor([system_ids], dtype=torch.long, device=model.device)
    system_cache, system_length, system_prefill_sec = _prefill_system(model, system_input_ids, args.system_attn_impl)
    tool_ids = _tool_definition_ids(tokenizer, tools)
    if len(tool_ids) > args.max_tool_definition_tokens:
        return None, f"tool_definition_tokens>{args.max_tool_definition_tokens}"
    keep = len(tool_ids)
    if args.method == "truncate":
        keep = max(1, (len(tool_ids) + args.ratio - 1) // args.ratio)
        tool_ids = tool_ids[:keep]
    elif args.max_full_tool_tokens is not None and len(tool_ids) > args.max_full_tool_tokens:
        return None, f"full_tool_tokens>{args.max_full_tool_tokens}"
    tool_input_ids = torch.tensor([tool_ids], dtype=torch.long, device=model.device)
    tool_cache, tool_length, full_prefill_sec = _prefill_tokens_with_cache(
        model,
        tool_input_ids,
        past_key_values=system_cache,
        past_length=system_length,
        attn_impl=args.generate_attn_impl,
    )
    ratio = len(_tool_definition_ids(tokenizer, tools)) / keep if keep else 0.0
    return {
        "cache": tool_cache,
        "system_length": system_length,
        "tool_length": tool_length,
        "cache_length": tool_cache.get_seq_length(),
        "system_prefill_sec": system_prefill_sec,
        "full_prefill_sec": full_prefill_sec,
        "tool_compress_sec": 0.0,
        "blend_sec": 0.0,
        "use_gist": False,
        "compression_ratio": ratio,
        "tool_definition_tokens": len(_tool_definition_ids(tokenizer, tools)),
    }, None


def _build_c2kv_prefix(model: Any, tokenizer: Any, system_prompt: str, tools: Sequence[Dict[str, Any]], args: argparse.Namespace):
    system_ids = _chat_template_ids(
        tokenizer,
        [{"role": "system", "content": system_prompt}],
        keep_bos=True,
        max_length=args.max_system_length,
    )
    system_input_ids = torch.tensor([system_ids], dtype=torch.long, device=model.device)
    system_cache, system_length, system_prefill_sec = _prefill_system(model, system_input_ids, args.system_attn_impl)
    tool_definition = _render_tool_definition(tools)
    context_input_ids, doc_tokens, doc_chunks, skip_reason = _build_tool_chunks(
        tokenizer,
        tool_definition,
        max_doc_length=args.max_doc_length,
        max_doc_num=args.max_doc_num,
        max_tool_definition_tokens=args.max_tool_definition_tokens,
        truncate_tool_definition=False,
    )
    if context_input_ids is None:
        return None, skip_reason
    tool_cache, tool_length, gist_tokens, actual_ratio, compress_sec, blend_sec = _build_tool_cache(
        model,
        context_input_ids,
        system_cache,
        system_length,
        args.gist_attn_impl,
        args.ratio,
    )
    return {
        "cache": tool_cache,
        "system_length": system_length,
        "tool_length": tool_length,
        "cache_length": tool_cache.get_seq_length(),
        "system_prefill_sec": system_prefill_sec,
        "full_prefill_sec": 0.0,
        "tool_compress_sec": compress_sec,
        "blend_sec": blend_sec,
        "use_gist": True,
        "compression_ratio": actual_ratio,
        "tool_definition_tokens": doc_tokens,
        "doc_chunks": doc_chunks,
        "gist_tokens": gist_tokens,
    }, None


def _build_hybrid_prefix(model: Any, tokenizer: Any, system_prompt: str, tools: Sequence[Dict[str, Any]], query: str, args: argparse.Namespace):
    top_tools, rest_tools, _ = _split_topk_tools(tools, query, args.top_k)
    system_ids = _chat_template_ids(
        tokenizer,
        [{"role": "system", "content": system_prompt}],
        keep_bos=True,
        max_length=args.max_system_length,
    )
    system_input_ids = torch.tensor([system_ids], dtype=torch.long, device=model.device)
    system_cache, system_length, system_prefill_sec = _prefill_system(model, system_input_ids, args.system_attn_impl)
    prefix_cache = system_cache
    top_length = 0
    top_prefill_sec = 0.0
    top_tokens = 0
    if top_tools:
        top_ids = _tool_definition_ids(tokenizer, top_tools, "Top-k tool definitions")
        top_tokens = len(top_ids)
        if args.max_full_tool_tokens is not None and top_tokens > args.max_full_tool_tokens:
            return None, f"top_full_tool_tokens>{args.max_full_tool_tokens}"
        top_input_ids = torch.tensor([top_ids], dtype=torch.long, device=model.device)
        prefix_cache, top_length, top_prefill_sec = _prefill_tokens_with_cache(
            model,
            top_input_ids,
            past_key_values=system_cache,
            past_length=system_length,
            attn_impl=args.generate_attn_impl,
        )
    rest_length = 0
    gist_tokens = 0
    rest_tokens = 0
    compress_sec = 0.0
    blend_sec = 0.0
    use_gist = bool(rest_tools)
    if rest_tools:
        rest_definition = _render_tool_definition(rest_tools)
        context_input_ids, rest_tokens, _, skip_reason = _build_tool_chunks(
            tokenizer,
            rest_definition,
            max_doc_length=args.max_doc_length,
            max_doc_num=args.max_doc_num,
            max_tool_definition_tokens=args.max_tool_definition_tokens,
            truncate_tool_definition=False,
        )
        if context_input_ids is None:
            return None, "rest_" + str(skip_reason)
        prefix_cache, rest_length, gist_tokens, _, compress_sec, blend_sec = _build_tool_cache(
            model,
            context_input_ids,
            prefix_cache,
            system_length + top_length,
            args.gist_attn_impl,
            args.ratio,
        )
    compressed_tokens = top_length + gist_tokens
    raw_tokens = top_tokens + rest_tokens
    return {
        "cache": prefix_cache,
        "system_length": system_length,
        "tool_length": top_length + rest_length,
        "cache_length": prefix_cache.get_seq_length(),
        "system_prefill_sec": system_prefill_sec,
        "full_prefill_sec": top_prefill_sec,
        "tool_compress_sec": compress_sec,
        "blend_sec": blend_sec,
        "use_gist": use_gist,
        "compression_ratio": raw_tokens / compressed_tokens if compressed_tokens else 0.0,
        "tool_definition_tokens": len(_tool_definition_ids(tokenizer, tools)),
        "top_tool_tokens": top_tokens,
        "rest_tool_tokens": rest_tokens,
        "gist_tokens": gist_tokens,
    }, None


def _build_prefix(model: Any, tokenizer: Any, system_prompt: str, tools: Sequence[Dict[str, Any]], query: str, args: argparse.Namespace):
    if args.method in ("full", "truncate"):
        return _build_full_prefix(model, tokenizer, system_prompt, tools, args)
    if args.method in ("c2kv", "c2kv_untrained"):
        return _build_c2kv_prefix(model, tokenizer, system_prompt, tools, args)
    if args.method == "hybrid":
        return _build_hybrid_prefix(model, tokenizer, system_prompt, tools, query, args)
    raise ValueError(f"Unknown method: {args.method}")


def _select_candidate_tools(
    tools: Sequence[Dict[str, Any]],
    target_tool_name: str,
    query: str,
    args: argparse.Namespace,
) -> tuple[List[Dict[str, Any]], List[str], bool, bool]:
    ranked = _rank_tools(tools, query)
    if args.candidate_top_k <= 0 or args.candidate_top_k >= len(tools):
        selected_indices = list(range(len(tools)))
    else:
        selected_indices = ranked[: args.candidate_top_k]
    target_index = None
    for index, tool in enumerate(tools):
        if _tool_name(tool) == target_tool_name:
            target_index = index
            break
    router_top_names = [_tool_name(tools[index]) for index in ranked[: max(0, args.candidate_top_k or args.top_k)]]
    router_hit = target_index in set(selected_indices) if target_index is not None else False
    if args.oracle_include_target_tool and target_index is not None and target_index not in selected_indices:
        if selected_indices:
            selected_indices[-1] = target_index
        else:
            selected_indices = [target_index]
    selected_set = set(selected_indices)
    selected_tools = [tool for index, tool in enumerate(tools) if index in selected_set]
    target_in_context = target_index in selected_set if target_index is not None else False
    return selected_tools, router_top_names, router_hit, target_in_context


def _generate_one(model: Any, tokenizer: Any, sample: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    config = _config(sample)
    messages = _messages(sample)
    tools = _tools(sample)
    first = _first_tool_call(messages)
    if not tools or first is None:
        return {"skipped": True, "skip_reason": "missing_tools_or_first_call"}
    first_index, target_payload = first
    router_messages = _prompt_messages(messages, first_index, add_instruction=False)
    if not router_messages:
        return {"skipped": True, "skip_reason": "empty_prompt"}
    prompt_messages = [*router_messages, _tool_call_instruction_message()]
    query = _router_query(config, router_messages, args)
    selected_tools, candidate_tool_names, candidate_router_hit, target_in_context = _select_candidate_tools(
        tools,
        target_payload["name"],
        query,
        args,
    )
    _, _, router_top_tool_names = _split_topk_tools(selected_tools, query, args.top_k)
    router_hit = target_payload["name"] in set(router_top_tool_names)
    prefix, skip_reason = _build_prefix(model, tokenizer, _system_prompt(config), selected_tools, query, args)
    if prefix is None:
        return {"skipped": True, "skip_reason": skip_reason}
    prompt_ids = _chat_template_ids(tokenizer, prompt_messages, add_generation_prompt=True)
    if len(prompt_ids) > args.max_prompt_tokens:
        prompt_ids = prompt_ids[-args.max_prompt_tokens :]
    prompt_input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
    mock_cache_ids = prompt_input_ids.new_zeros((1, prefix["cache_length"]))
    input_ids = torch.cat([mock_cache_ids, prompt_input_ids], dim=1)
    position_ids = torch.arange(
        prefix["system_length"] + prefix["tool_length"],
        prefix["system_length"] + prefix["tool_length"] + prompt_input_ids.shape[1],
        dtype=torch.long,
        device=model.device,
    ).unsqueeze(0)
    prediction, generate_sec, generated_tokens, tbt_sec = _generate_from_input_ids(
        model,
        tokenizer,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        attn_impl=args.generate_attn_impl,
        use_gist=prefix["use_gist"],
        position_ids=position_ids,
        past_key_values=prefix["cache"],
    )
    target = _target_answer(target_payload)
    pred_payload = _parse_pred_call(prediction)
    arg_name_f1, arg_value_f1 = _arg_f1s(target_payload, pred_payload)
    online_ttft = prefix["system_prefill_sec"] + prefix["full_prefill_sec"] + prefix["tool_compress_sec"] + prefix["blend_sec"]
    cached_ttft = prefix["system_prefill_sec"] + prefix["full_prefill_sec"] + prefix["blend_sec"]
    return {
        "modelname_run": sample.get("modelname_run"),
        "task_name": sample.get("task_name"),
        "method": args.method,
        "ratio": args.ratio,
        "top_k": args.top_k if args.method == "hybrid" else None,
        "candidate_top_k": args.candidate_top_k,
        "router_top_k": args.top_k,
        "router_top_tools": router_top_tool_names,
        "router_hit": router_hit,
        "candidate_top_tools": candidate_tool_names,
        "candidate_router_hit": candidate_router_hit,
        "target_tool_in_context": target_in_context,
        "total_tool_count": len(tools),
        "selected_tool_count": len(selected_tools),
        "skipped": False,
        "target_tool_name": target_payload["name"],
        "prediction_tool_name": pred_payload.get("name") if pred_payload else None,
        "first_tool_accuracy": bool(pred_payload and pred_payload.get("name") == target_payload["name"]),
        "tool_call_rate": pred_payload is not None,
        "argument_name_f1": arg_name_f1,
        "argument_value_f1": arg_value_f1,
        "exact_match": _normalize_text(prediction) == _normalize_text(target),
        "tool_definition_tokens": prefix["tool_definition_tokens"],
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": generated_tokens,
        "actual_compression_ratio": round(prefix["compression_ratio"], 4),
        "system_prefill_sec": round(prefix["system_prefill_sec"], 4),
        "full_prefill_sec": round(prefix["full_prefill_sec"], 4),
        "tool_compress_sec": round(prefix["tool_compress_sec"], 4),
        "blend_sec": round(prefix["blend_sec"], 4),
        "online_ttft_sec": round(online_ttft, 4),
        "cached_ttft_sec": round(cached_ttft, 4),
        "generate_sec": round(generate_sec, 4),
        "tbt_sec": round(tbt_sec, 6),
        "online_total_sec": round(online_ttft + generate_sec, 4),
        "cached_total_sec": round(cached_ttft + generate_sec, 4),
        "prediction": prediction,
        "target": target,
    }


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [row for row in rows if not row.get("skipped")]
    skips = Counter(row.get("skip_reason", "unknown") for row in rows if row.get("skipped"))
    total_generated = sum(row.get("generated_tokens", 0) for row in valid)
    return {
        "method": valid[0].get("method") if valid else None,
        "ratio": valid[0].get("ratio") if valid else None,
        "top_k": valid[0].get("top_k") if valid else None,
        "num_examples": len(rows),
        "num_valid": len(valid),
        "num_skipped": len(rows) - len(valid),
        "skip_reasons": dict(skips),
        "router_hit_rate": sum(1 for row in valid if row.get("router_hit")) / len(valid) if valid else 0.0,
        "candidate_router_hit_rate": sum(1 for row in valid if row.get("candidate_router_hit")) / len(valid) if valid else 0.0,
        "target_in_context_rate": sum(1 for row in valid if row.get("target_tool_in_context")) / len(valid) if valid else 0.0,
        "first_tool_accuracy": sum(1 for row in valid if row["first_tool_accuracy"]) / len(valid) if valid else 0.0,
        "tool_call_rate": sum(1 for row in valid if row["tool_call_rate"]) / len(valid) if valid else 0.0,
        "argument_name_f1": sum(row["argument_name_f1"] for row in valid) / len(valid) if valid else 0.0,
        "argument_value_f1": sum(row["argument_value_f1"] for row in valid) / len(valid) if valid else 0.0,
        "exact_match": sum(1 for row in valid if row["exact_match"]) / len(valid) if valid else 0.0,
        "avg_tool_definition_tokens": sum(row["tool_definition_tokens"] for row in valid) / len(valid) if valid else 0.0,
        "avg_total_tool_count": sum(row.get("total_tool_count", 0) for row in valid) / len(valid) if valid else 0.0,
        "avg_selected_tool_count": sum(row.get("selected_tool_count", 0) for row in valid) / len(valid) if valid else 0.0,
        "avg_prompt_tokens": sum(row["prompt_tokens"] for row in valid) / len(valid) if valid else 0.0,
        "avg_generated_tokens": total_generated / len(valid) if valid else 0.0,
        "avg_actual_compression_ratio": sum(row["actual_compression_ratio"] for row in valid) / len(valid) if valid else 0.0,
        "avg_online_ttft_sec": sum(row["online_ttft_sec"] for row in valid) / len(valid) if valid else 0.0,
        "avg_cached_ttft_sec": sum(row["cached_ttft_sec"] for row in valid) / len(valid) if valid else 0.0,
        "avg_generate_sec": sum(row["generate_sec"] for row in valid) / len(valid) if valid else 0.0,
        "avg_tbt_sec": sum(row["tbt_sec"] for row in valid) / len(valid) if valid else 0.0,
        "token_weighted_tbt_sec": sum(row["generate_sec"] for row in valid) / total_generated if total_generated else 0.0,
        "avg_online_total_sec": sum(row["online_total_sec"] for row in valid) / len(valid) if valid else 0.0,
        "avg_cached_total_sec": sum(row["cached_total_sec"] for row in valid) / len(valid) if valid else 0.0,
    }


def _clear_device_cache(device: str) -> None:
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "npu" and hasattr(torch, "npu"):
        torch.npu.empty_cache()


def _is_oom_error(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "oom" in message


def _common_subset_skip_reason(sample: Dict[str, Any], tokenizer: Any, args: argparse.Namespace) -> Optional[str]:
    if args.common_subset == "none":
        return None
    tools = _tools(sample)
    messages = _messages(sample)
    first = _first_tool_call(messages)
    if not tools or first is None:
        return "common_missing_tools_or_first_call"
    first_index, _ = first
    _, target_payload = first
    prompt_messages = _prompt_messages(messages, first_index, add_instruction=False)
    if not prompt_messages:
        return "common_empty_prompt"
    selected_tools, _, _, target_in_context = _select_candidate_tools(
        tools,
        target_payload["name"],
        _router_query(_config(sample), prompt_messages, args),
        args,
    )
    if not target_in_context and args.require_target_in_context:
        return "common_target_not_in_context"
    tool_tokens = len(_tool_definition_ids(tokenizer, selected_tools))
    if tool_tokens > args.max_tool_definition_tokens:
        return f"common_tool_definition_tokens>{args.max_tool_definition_tokens}"
    if args.common_subset == "full" and args.max_full_tool_tokens is not None:
        if tool_tokens > args.max_full_tool_tokens:
            return f"common_full_tool_tokens>{args.max_full_tool_tokens}"
    return None


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    device = _setup_device(args.device_type)
    tokenizer_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        local_files_only=True,
        padding_side="right",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model_args = copy.copy(args)
    model_args.mode = "c2kv"
    model_args.untrained_c2kv = args.method == "c2kv_untrained"
    model = _load_model(model_args, tokenizer, device)

    rows = []
    selection_skips: Counter[str] = Counter()
    for _, sample in tqdm(list(_iter_samples(Path(args.data_dir), args.file_glob, args.scan_limit)), desc=f"toolathlon-{args.method}"):
        status = _status(sample)
        if args.only_success and status.get("evaluation") is not True:
            continue
        common_skip = _common_subset_skip_reason(sample, tokenizer, args)
        if common_skip is not None:
            selection_skips[common_skip] += 1
            continue
        try:
            rows.append(_generate_one(model, tokenizer, sample, args))
        except RuntimeError as error:
            if not _is_oom_error(error):
                raise
            logger.warning("Skipping sample after OOM: task=%s method=%s", sample.get("task_name"), args.method)
            rows.append({
                "task_name": sample.get("task_name"),
                "modelname_run": sample.get("modelname_run"),
                "method": args.method,
                "ratio": args.ratio,
                "top_k": args.top_k if args.method == "hybrid" else None,
                "skipped": True,
                "skip_reason": "oom",
            })
            _clear_device_cache(device)
        else:
            _clear_device_cache(device)
        if args.max_examples is not None and len(rows) >= args.max_examples:
            break
    summary = {
        "model": args.model,
        "data_dir": args.data_dir,
        "file_glob": args.file_glob,
        "only_success": args.only_success,
        "common_subset": args.common_subset,
        "selection_skips": dict(selection_skips),
        "result": _summarize(rows),
    }
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        output_path.with_suffix(".summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate first Toolathlon tool call with Tool-C2KV variants.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base_model")
    parser.add_argument("--tokenizer")
    parser.add_argument("--data_dir", default="./datasets/toolathlon")
    parser.add_argument("--file_glob", default="*.jsonl")
    parser.add_argument("--output_file", default="./outputs/toolathlon_first_tool_eval.jsonl")
    parser.add_argument("--method", choices=["full", "truncate", "c2kv", "c2kv_untrained", "hybrid"], default="full")
    parser.add_argument("--ratio", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--candidate_top_k", type=int, default=0)
    parser.add_argument("--oracle_include_target_tool", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--require_target_in_context", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument(
        "--router_scope",
        choices=["task", "last_user", "all_prompt", "task_plus_prompt"],
        default="task_plus_prompt",
    )
    parser.add_argument("--only_success", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--max_examples", type=int, default=100)
    parser.add_argument("--scan_limit", type=int)
    parser.add_argument("--common_subset", choices=["none", "full"], default="full")
    parser.add_argument("--max_tool_definition_tokens", type=int, default=20000)
    parser.add_argument("--max_full_tool_tokens", type=int, default=10000)
    parser.add_argument("--max_doc_length", type=int, default=1024)
    parser.add_argument("--max_doc_num", type=int, default=20)
    parser.add_argument("--max_prompt_tokens", type=int, default=2048)
    parser.add_argument("--max_system_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device_type", choices=["auto", "cuda", "npu", "cpu"], default="auto")
    parser.add_argument("--system_attn_impl", default="eager")
    parser.add_argument("--gist_attn_impl", default="eager")
    parser.add_argument("--generate_attn_impl", default="eager")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--baseline_model_class", choices=["gist", "auto"], default="gist")
    parser.add_argument("--untrained_c2kv", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(evaluate(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
