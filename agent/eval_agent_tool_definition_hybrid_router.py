from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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
from train.train_data_multiturn import _chat_template_ids  # noqa: E402
from train_agent_tool_definition_c2kv import (  # noqa: E402
    AgentLLMTracesSource,
    AgentToolDefinitionDataArgs,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _jsonl_write(path: str, rows: List[Dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _json_loads(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _as_tool_list(tool_definition: str) -> List[Dict[str, Any]]:
    parsed = _json_loads(tool_definition, [])
    if isinstance(parsed, dict):
        if isinstance(parsed.get("tools"), list):
            parsed = parsed["tools"]
        elif isinstance(parsed.get("functions"), list):
            parsed = parsed["functions"]
        else:
            parsed = [parsed]
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _tool_name(tool: Dict[str, Any]) -> str:
    function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
    return str(
        function.get("name")
        or tool.get("name")
        or tool.get("tool_name")
        or tool.get("function_name")
        or ""
    )


def _tool_search_text(tool: Dict[str, Any]) -> str:
    function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
    fields = [
        _tool_name(tool),
        function.get("description", ""),
        tool.get("description", ""),
        function.get("parameters", ""),
        tool.get("parameters", ""),
        tool.get("input_schema", ""),
        tool.get("schema", ""),
    ]
    return " ".join(
        item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
        for item in fields
        if item
    )


def _compact_parameters(parameters: Any) -> Any:
    if not isinstance(parameters, dict):
        return parameters
    required = parameters.get("required") if isinstance(parameters.get("required"), list) else []
    properties = parameters.get("properties") if isinstance(parameters.get("properties"), dict) else {}
    compact_properties: Dict[str, Any] = {}
    for name in required:
        if not isinstance(name, str):
            continue
        value = properties.get(name, {})
        if isinstance(value, dict):
            compact = {
                key: value[key]
                for key in ("type", "description", "enum")
                if key in value
            }
            compact_properties[name] = compact or value
        else:
            compact_properties[name] = value
    return {
        "type": parameters.get("type", "object"),
        "required": required,
        "properties": compact_properties,
    }


def _compact_tool_definition(tool: Dict[str, Any]) -> Dict[str, Any]:
    function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
    parameters = (
        function.get("parameters")
        or tool.get("parameters")
        or tool.get("input_schema")
        or tool.get("schema")
    )
    compact_function = {
        "name": _tool_name(tool),
    }
    description = function.get("description") or tool.get("description")
    if description:
        compact_function["description"] = description
    if parameters:
        compact_function["parameters"] = _compact_parameters(parameters)
    if function or tool.get("type") == "function":
        return {
            "type": tool.get("type", "function"),
            "function": compact_function,
        }
    compact_tool = dict(compact_function)
    if parameters:
        compact_tool["parameters"] = _compact_parameters(parameters)
    return compact_tool


def _render_tool_definition(tools: Sequence[Dict[str, Any]], schema_mode: str = "full") -> str:
    if schema_mode == "compact":
        tools = [_compact_tool_definition(tool) for tool in tools]
    elif schema_mode != "full":
        raise ValueError(f"Unknown tool schema mode: {schema_mode}")
    return json.dumps(list(tools), ensure_ascii=False, separators=(",", ":"))


def _truncate_debug_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...<truncated {len(text) - max_chars} chars>"


def _debug_tool_items(definition: str) -> List[Dict[str, Any]]:
    parsed = _json_loads(definition, [])
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    return []


def _add_hybrid_debug_fields(
    row: Dict[str, Any],
    args: argparse.Namespace,
    *,
    full_definition: str,
    top_definition: str,
    rest_definition: str,
    numerator_tokens: int,
    denominator_tokens: int,
    top_tokens: int,
    rest_original_tokens: int,
    rest_compressed_tokens: int,
) -> Dict[str, Any]:
    if not (
        getattr(args, "debug_hybrid_tokens", False)
        or getattr(args, "dump_hybrid_definitions", False)
    ):
        return row
    top_debug_tools = _debug_tool_items(top_definition)
    rest_debug_tools = _debug_tool_items(rest_definition)
    row["debug_top_tool_count_from_definition"] = len(top_debug_tools)
    row["debug_rest_tool_count_from_definition"] = len(rest_debug_tools)
    row["debug_top_tool_names_from_definition"] = [_tool_name(tool) for tool in top_debug_tools]
    row["debug_rest_tool_names_from_definition"] = [_tool_name(tool) for tool in rest_debug_tools]
    if getattr(args, "dump_hybrid_definitions", False):
        max_chars = int(getattr(args, "debug_definition_chars", 4000))
        row["debug_full_tool_definition"] = _truncate_debug_text(full_definition, max_chars)
        row["debug_top_tool_definition"] = _truncate_debug_text(top_definition, max_chars)
        row["debug_rest_tool_definition"] = _truncate_debug_text(rest_definition, max_chars)
    row["hybrid_debug_log"] = "\n".join([
        f"num_tools: {row.get('num_tools')}",
        f"num_top_tools: {row.get('num_top_tools')}",
        f"num_rest_tools: {row.get('num_rest_tools')}",
        f"top_tool_names: {row.get('top_tool_names')}",
        f"top tokens: {row.get('top_doc_tokens')}",
        f"rest tokens: {row.get('rest_doc_tokens')}",
        f"debug top chars: {len(row.get('debug_top_tool_definition', ''))}",
        f"debug full chars: {len(row.get('debug_full_tool_definition', ''))}",
    ])
    return row


def _row_compressed_tool_tokens(row: Dict[str, Any]) -> float:
    return float(row.get("top_doc_tokens", 0) or 0) + float(row.get("rest_gist_tokens", 0) or 0)


def _message_text(message: Dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False) if content is not None else ""


def _query_text(messages: Sequence[Dict[str, Any]], router_scope: str) -> str:
    if router_scope == "all":
        return "\n".join(_message_text(message) for message in messages)
    for message in reversed(messages):
        if message.get("role") == "user":
            return _message_text(message)
    return _message_text(messages[-1]) if messages else ""


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _rank_tools(tools: Sequence[Dict[str, Any]], query: str) -> List[int]:
    query_tokens = set(_tokens(query))
    if not query_tokens:
        return list(range(len(tools)))
    scored = []
    for index, tool in enumerate(tools):
        name_tokens = set(_tokens(_tool_name(tool)))
        text_tokens = set(_tokens(_tool_search_text(tool)))
        name_overlap = len(query_tokens & name_tokens)
        text_overlap = len(query_tokens & text_tokens)
        score = 4.0 * name_overlap + float(text_overlap)
        scored.append((-score, index))
    scored.sort()
    return [index for _, index in scored]


def _split_ranked_tools(
    tools: Sequence[Dict[str, Any]],
    ranked: Sequence[int],
    top_k: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    top_indices = set(ranked[: max(0, top_k)])
    top_tools = [tool for index, tool in enumerate(tools) if index in top_indices]
    rest_tools = [tool for index, tool in enumerate(tools) if index not in top_indices]
    return top_tools, rest_tools, [_tool_name(tools[index]) for index in ranked[: max(0, top_k)]]


def _split_topk_tools(
    tools: Sequence[Dict[str, Any]],
    query: str,
    top_k: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    return _split_ranked_tools(tools, _rank_tools(tools, query), top_k)


def _rank_in_tool_order(
    tools: Sequence[Dict[str, Any]],
    ranked: Sequence[int],
    tool_name: Optional[str],
) -> Optional[int]:
    if not tool_name:
        return None
    for rank, index in enumerate(ranked, start=1):
        if _tool_name(tools[index]) == tool_name:
            return rank
    return None


def _rerank_lexical_pool_by_attention(
    lexical_ranked: Sequence[int],
    attention_scores: Sequence[float],
    top_k: int,
    pool_size: int,
) -> List[int]:
    pool_size = max(top_k, min(pool_size, len(lexical_ranked)))
    pool = list(lexical_ranked[:pool_size])
    pool_ranks = {index: rank for rank, index in enumerate(pool)}
    pool.sort(key=lambda index: (-attention_scores[index], pool_ranks[index]))
    pool_set = set(pool)
    return pool + [index for index in lexical_ranked if index not in pool_set]


def _att_rerank_replacement(
    lexical_ranked: Sequence[int],
    head_rankings: Sequence[Dict[str, Any]],
    top_k: int,
    pool_size: int,
    min_heads: int,
    min_margin: float,
    min_score_gain: float,
) -> tuple[List[int], Optional[Dict[str, Any]]]:
    if top_k <= 0 or len(lexical_ranked) <= top_k:
        return list(lexical_ranked), None
    pool_size = max(top_k + 1, min(pool_size, len(lexical_ranked)))
    base_top = list(lexical_ranked[:top_k])
    candidate_indices = set(lexical_ranked[top_k:pool_size])
    replace_index = base_top[-1]

    votes: Dict[int, Dict[str, Any]] = {}
    for head in head_rankings:
        ranked = head.get("ranked_indices") or []
        scores_by_index = head.get("scores_by_index") or {}
        if not ranked:
            continue
        top_index = ranked[0]
        if top_index not in candidate_indices:
            continue
        top_score = float(scores_by_index.get(top_index, 0.0) or 0.0)
        second_score = float(scores_by_index.get(ranked[1], 0.0) or 0.0) if len(ranked) > 1 else 0.0
        margin = top_score - second_score
        if margin < min_margin:
            continue
        entry = votes.setdefault(
            top_index,
            {
                "num_heads": 0,
                "margin_sum": 0.0,
                "score_sum": 0.0,
                "replace_score_sum": 0.0,
                "max_margin": 0.0,
                "support_heads": [],
            },
        )
        replace_score = float(scores_by_index.get(replace_index, 0.0) or 0.0)
        entry["num_heads"] += 1
        entry["margin_sum"] += margin
        entry["score_sum"] += top_score
        entry["replace_score_sum"] += replace_score
        entry["max_margin"] = max(entry["max_margin"], margin)
        entry["support_heads"].append({
            "layer": head.get("layer"),
            "head": head.get("head"),
            "margin": round(margin, 8),
            "top_score": round(top_score, 8),
            "replace_score": round(replace_score, 8),
        })

    if not votes:
        return list(lexical_ranked), None
    best_index, best = max(
        votes.items(),
        key=lambda item: (
            item[1]["num_heads"],
            item[1]["margin_sum"],
            item[1]["score_sum"],
            -list(lexical_ranked).index(item[0]),
        ),
    )
    replace_score = float(best["replace_score_sum"])
    score_gain = float(best["score_sum"]) - replace_score
    accepted = best["num_heads"] >= min_heads and score_gain >= min_score_gain
    debug = {
        "candidate_tool_index": best_index,
        "candidate_lexical_rank": list(lexical_ranked).index(best_index) + 1,
        "replace_tool_index": replace_index,
        "replace_lexical_rank": top_k,
        "num_support_heads": best["num_heads"],
        "margin_sum": round(float(best["margin_sum"]), 8),
        "score_sum": round(float(best["score_sum"]), 8),
        "replace_score_sum": round(replace_score, 8),
        "score_gain": round(score_gain, 8),
        "accepted": accepted,
        "support_heads": best["support_heads"][:20],
    }
    if not accepted:
        return list(lexical_ranked), debug
    final = list(lexical_ranked)
    final[top_k - 1] = best_index
    final = final[:top_k] + [
        index for index in lexical_ranked
        if index not in set(final[:top_k])
    ]
    return final, debug


def _split_random_topk_tools(
    tools: Sequence[Dict[str, Any]],
    top_k: int,
    seed_text: str,
    seed: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    rng = random.Random(f"{seed}:{seed_text}:{top_k}:{len(tools)}")
    ranked = list(range(len(tools)))
    rng.shuffle(ranked)
    top_indices = set(ranked[: max(0, top_k)])
    top_tools = [tool for index, tool in enumerate(tools) if index in top_indices]
    rest_tools = [tool for index, tool in enumerate(tools) if index not in top_indices]
    return top_tools, rest_tools, [_tool_name(tools[index]) for index in ranked[: max(0, top_k)]]


def _tool_token_lengths(tokenizer: Any, tools: Sequence[Dict[str, Any]]) -> List[int]:
    lengths = []
    for tool in tools:
        text = json.dumps(tool, ensure_ascii=False, separators=(",", ":"))
        lengths.append(max(1, len(tokenizer.encode(text, add_special_tokens=False))))
    return lengths


def _gist_spans_from_tool_lengths(tool_lengths: Sequence[int], gist_tokens: int) -> List[tuple[int, int]]:
    total = sum(tool_lengths)
    if total <= 0 or gist_tokens <= 0:
        return [(0, 0) for _ in tool_lengths]
    spans = []
    cursor = 0
    for length in tool_lengths:
        start = int(cursor * gist_tokens / total)
        cursor += length
        end = int((cursor * gist_tokens + total - 1) / total)
        if end <= start:
            end = min(gist_tokens, start + 1)
        spans.append((max(0, start), min(gist_tokens, end)))
    return spans


@torch.inference_mode()
def _build_full_tool_cache_with_spans(
    model: Any,
    tokenizer: Any,
    tools: Sequence[Dict[str, Any]],
    system_cache: Any,
    system_length: int,
    attn_impl: str,
) -> tuple[Any, int, List[tuple[int, int]], float]:
    prefix_cache = system_cache
    tool_length = 0
    spans: List[tuple[int, int]] = []
    prefill_sec = 0.0
    for index, tool in enumerate(tools):
        tool_text = _render_tool_definition([tool])
        tool_doc = {
            "role": "user",
            "content": f"Tool definition {index}:\n{tool_text}",
        }
        tool_ids = _chat_template_ids(tokenizer, [tool_doc])
        tool_input_ids = torch.tensor([tool_ids], dtype=torch.long, device=model.device)
        prefix_cache, length, elapsed = _prefill_tokens_with_cache(
            model,
            tool_input_ids,
            past_key_values=prefix_cache,
            past_length=system_length + tool_length,
            attn_impl=attn_impl,
        )
        spans.append((tool_length, tool_length + length))
        tool_length += length
        prefill_sec += elapsed
    return prefix_cache, tool_length, spans, prefill_sec


def _clear_device_cache(device: str) -> None:
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device == "npu" and hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.empty_cache()


@torch.inference_mode()
def _rank_tools_by_attention(
    model: Any,
    tokenizer: Any,
    example: Any,
    args: argparse.Namespace,
    tools: Sequence[Dict[str, Any]],
    ratio: int,
) -> tuple[List[int], List[float], float, List[Dict[str, Any]]]:
    system_ids = _chat_template_ids(
        tokenizer,
        [{"role": "system", "content": example.system_prompt}],
        keep_bos=True,
        max_length=args.max_system_length,
    )
    system_input_ids = torch.tensor([system_ids], dtype=torch.long, device=model.device)
    system_cache, system_length, system_prefill_sec = _prefill_system(
        model,
        system_input_ids,
        args.system_attn_impl,
    )

    cache_mode = args.attention_router_cache_mode
    if cache_mode == "full":
        prefix_cache, tool_length, spans, tool_compress_sec = _build_full_tool_cache_with_spans(
            model,
            tokenizer,
            tools,
            system_cache,
            system_length,
            args.attention_router_attn_impl,
        )
        blend_sec = 0.0
        tool_key_tokens = tool_length
        use_gist_for_query = False
    else:
        tool_definition = _render_tool_definition(tools)
        context_input_ids, _, _, skip_reason = _build_tool_chunks(
            tokenizer,
            tool_definition,
            max_doc_length=args.max_doc_length,
            max_doc_num=args.max_doc_num,
            max_tool_definition_tokens=args.max_tool_definition_tokens,
            truncate_tool_definition=args.truncate_tool_definition,
        )
        if context_input_ids is None:
            raise ValueError(f"attention_router_{skip_reason}")
        (
            prefix_cache,
            tool_length,
            gist_tokens,
            _,
            tool_compress_sec,
            blend_sec,
        ) = _build_tool_cache(
            model,
            context_input_ids,
            system_cache,
            system_length,
            args.gist_attn_impl,
            ratio,
        )
        if gist_tokens <= 0:
            raise ValueError("attention_router_empty_gist")
        spans = _gist_spans_from_tool_lengths(_tool_token_lengths(tokenizer, tools), gist_tokens)
        tool_key_tokens = gist_tokens
        use_gist_for_query = True

    query = _query_text(example.input_messages, args.router_scope)
    query_messages = [{"role": "user", "content": query}]
    query_ids = _chat_template_ids(tokenizer, query_messages, add_generation_prompt=True)
    if args.attention_router_max_query_tokens and len(query_ids) > args.attention_router_max_query_tokens:
        query_ids = query_ids[-args.attention_router_max_query_tokens :]
    query_input_ids = torch.tensor([query_ids], dtype=torch.long, device=model.device)
    query_len = query_input_ids.shape[1]
    attention_mask = torch.ones(
        (1, prefix_cache.get_seq_length() + query_len),
        dtype=torch.long,
        device=model.device,
    )
    position_ids = torch.arange(
        system_length + tool_length,
        system_length + tool_length + query_len,
        dtype=torch.long,
        device=model.device,
    ).unsqueeze(0)

    layer_scores: List[List[float]] = []
    head_rankings: List[Dict[str, Any]] = []

    def _score_tool_attention(tool_attn: torch.Tensor, span_len: int) -> float:
        if args.attention_router_score_mode == "sum":
            score = tool_attn.sum(dim=-1).mean()
        elif args.attention_router_score_mode == "sqrt_len":
            score = tool_attn.sum(dim=-1).mean() / (span_len ** 0.5)
        else:
            score = tool_attn.mean()
        return float(score.item())

    def make_hook(layer_index: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            attn_weights = output[1] if isinstance(output, tuple) and len(output) > 1 else None
            if attn_weights is None:
                return
            cache_attn = attn_weights[0, :, :, system_length : system_length + tool_key_tokens].float()
            layer_head_scores = []
            for head_index in range(cache_attn.shape[0]):
                head_attn = cache_attn[head_index]
                scores = []
                for start, end in spans:
                    if end <= start:
                        scores.append(0.0)
                    else:
                        scores.append(_score_tool_attention(head_attn[:, start:end], end - start))
                ranked = sorted(range(len(tools)), key=lambda index: (-scores[index], index))
                head_rankings.append({
                    "layer": layer_index,
                    "head": head_index,
                    "ranked_indices": ranked,
                    "scores_by_index": {
                        index: scores[index] for index in range(len(tools))
                    },
                    "top_margin": (
                        scores[ranked[0]] - scores[ranked[1]]
                        if len(ranked) > 1 else scores[ranked[0]]
                    ) if ranked else 0.0,
                })
                layer_head_scores.append(scores)
            if layer_head_scores:
                layer_scores.append([
                    sum(head_scores[index] for head_scores in layer_head_scores) / len(layer_head_scores)
                    for index in range(len(tools))
                ])
        return hook

    num_layers = len(model.model.layers)
    last_layers = max(1, min(args.attention_router_layers, num_layers))
    handles = [
        model.model.layers[index].self_attn.register_forward_hook(make_hook(index))
        for index in range(num_layers - last_layers, num_layers)
    ]
    original_attn_impl = model.model.config._attn_implementation
    model.model.config._attn_implementation = args.attention_router_attn_impl
    try:
        forward_kwargs = {
            "input_ids": query_input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": prefix_cache,
            "use_cache": True,
            "logits_to_keep": 1,
        }
        if use_gist_for_query:
            forward_kwargs["use_gist"] = True
        model(**forward_kwargs)
    finally:
        model.model.config._attn_implementation = original_attn_impl
        for handle in handles:
            handle.remove()
        prefix_cache = None
        system_cache = None
        _clear_device_cache(str(model.device).split(":")[0])

    if not layer_scores:
        raise RuntimeError(
            "Attention router did not capture attention weights. Try --attention_router_attn_impl eager."
        )
    scores = [
        sum(layer[index] for layer in layer_scores) / len(layer_scores)
        for index in range(len(tools))
    ]
    ranked = sorted(range(len(tools)), key=lambda index: (-scores[index], index))
    return ranked, scores, system_prefill_sec + tool_compress_sec + blend_sec, head_rankings


def _parse_cases(cases: str) -> List[tuple[int, int]]:
    parsed = []
    for item in cases.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            top_k, ratio = item.split(":", 1)
        elif "x" in item.lower():
            top_k, ratio = item.lower().split("x", 1)
        else:
            raise ValueError(f"Invalid case {item!r}; expected TOPK:RATIO, e.g. 3:4")
        parsed.append((int(top_k), int(ratio)))
    return parsed


@torch.inference_mode()
def _generate_one_hybrid(
    model: Any,
    tokenizer: Any,
    example: Any,
    args: argparse.Namespace,
    top_k: int,
    ratio: int,
) -> Dict[str, Any]:
    total_start = time.perf_counter()
    hybrid_mode = getattr(args, "hybrid_mode", "hybrid")
    top_schema_mode = getattr(args, "top_schema_mode", "full")
    tools = _as_tool_list(example.tool_definition)
    if not tools:
        return {
            "qid": example.qid,
            "session_id": example.session_id,
            "mode": "hybrid",
            "hybrid_mode": hybrid_mode,
            "router_strategy": args.router_strategy,
            "top_schema_mode": top_schema_mode,
            "top_k": top_k,
            "ratio": ratio,
            "skipped": True,
            "skip_reason": "no_parseable_tools",
        }

    query = _query_text(example.input_messages, args.router_scope)
    target = example.answer.strip()
    target_tool = _extract_tool_name(target)
    lexical_ranked = _rank_tools(tools, query)
    attention_ranked: Optional[List[int]] = None
    final_ranked: Optional[List[int]] = None
    att_rerank_debug: Optional[Dict[str, Any]] = None
    attention_router_sec = 0.0
    attention_tool_scores: Optional[List[float]] = None
    if args.router_strategy == "random":
        top_tools, rest_tools, top_tool_names = _split_random_topk_tools(
            tools,
            top_k,
            seed_text=example.qid,
            seed=args.router_seed,
        )
    elif args.router_strategy == "attention":
        attention_ranked, attention_tool_scores, attention_router_sec, _ = _rank_tools_by_attention(
            model,
            tokenizer,
            example,
            args,
            tools,
            ratio,
        )
        final_ranked = attention_ranked
        top_tools, rest_tools, top_tool_names = _split_ranked_tools(tools, final_ranked, top_k)
    elif args.router_strategy == "lex_attention":
        attention_ranked, attention_tool_scores, attention_router_sec, _ = _rank_tools_by_attention(
            model,
            tokenizer,
            example,
            args,
            tools,
            ratio,
        )
        final_ranked = _rerank_lexical_pool_by_attention(
            lexical_ranked,
            attention_tool_scores,
            top_k,
            args.attention_router_lexical_pool,
        )
        top_tools, rest_tools, top_tool_names = _split_ranked_tools(tools, final_ranked, top_k)
    elif args.router_strategy == "att_rerank":
        attention_ranked, attention_tool_scores, attention_router_sec, head_rankings = _rank_tools_by_attention(
            model,
            tokenizer,
            example,
            args,
            tools,
            ratio,
        )
        final_ranked, att_rerank_debug = _att_rerank_replacement(
            lexical_ranked,
            head_rankings,
            top_k,
            args.att_rerank_pool,
            args.att_rerank_min_heads,
            args.att_rerank_min_margin,
            args.att_rerank_min_score_gain,
        )
        top_tools, rest_tools, top_tool_names = _split_ranked_tools(tools, final_ranked, top_k)
    else:
        final_ranked = lexical_ranked
        top_tools, rest_tools, top_tool_names = _split_ranked_tools(tools, final_ranked, top_k)
    selected_top_tools = top_tools
    selected_rest_tools = rest_tools
    router_hit = target_tool in set(top_tool_names) if target_tool else False
    target_lexical_rank = _rank_in_tool_order(tools, lexical_ranked, target_tool)
    target_attention_rank = _rank_in_tool_order(tools, attention_ranked or [], target_tool)
    target_final_rank = _rank_in_tool_order(tools, final_ranked or [], target_tool)
    if args.router_hit_filter == "hit" and not router_hit:
        return {
            "qid": example.qid,
            "session_id": example.session_id,
            "mode": "hybrid",
            "hybrid_mode": hybrid_mode,
            "router_strategy": args.router_strategy,
            "top_schema_mode": top_schema_mode,
            "top_k": top_k,
            "ratio": ratio,
            "skipped": True,
            "skip_reason": "router_miss_filtered",
            "num_tools": len(tools),
            "top_tool_names": top_tool_names,
            "target_tool_name": target_tool,
            "router_hit": router_hit,
        }
    if args.router_hit_filter == "miss" and router_hit:
        return {
            "qid": example.qid,
            "session_id": example.session_id,
            "mode": "hybrid",
            "hybrid_mode": hybrid_mode,
            "router_strategy": args.router_strategy,
            "top_schema_mode": top_schema_mode,
            "top_k": top_k,
            "ratio": ratio,
            "skipped": True,
            "skip_reason": "router_hit_filtered",
            "num_tools": len(tools),
            "top_tool_names": top_tool_names,
            "target_tool_name": target_tool,
            "router_hit": router_hit,
        }

    if hybrid_mode == "drop_selected":
        top_tools = []
        rest_tools = selected_rest_tools
    elif hybrid_mode == "topk_only":
        top_tools = selected_top_tools
        rest_tools = []
    elif hybrid_mode != "hybrid":
        raise ValueError(f"Unknown hybrid_mode: {hybrid_mode}")
    top_definition = _render_tool_definition(top_tools, top_schema_mode)
    rest_definition = _render_tool_definition(rest_tools)

    system_ids = _chat_template_ids(
        tokenizer,
        [{"role": "system", "content": example.system_prompt}],
        keep_bos=True,
        max_length=args.max_system_length,
    )
    system_input_ids = torch.tensor([system_ids], dtype=torch.long, device=model.device)
    system_cache, system_length, system_prefill_sec = _prefill_system(
        model, system_input_ids, args.system_attn_impl
    )

    top_ids = []
    top_prefill_sec = 0.0
    top_length = 0
    prefix_cache = system_cache
    if top_tools:
        top_doc = {"role": "user", "content": "Top-k tool definitions:\n" + top_definition}
        top_ids = _chat_template_ids(tokenizer, [top_doc])
        tool_input_ids = torch.tensor([top_ids], dtype=torch.long, device=model.device)
        prefix_cache, top_length, top_prefill_sec = _prefill_tokens_with_cache(
            model,
            tool_input_ids,
            past_key_values=system_cache,
            past_length=system_length,
            attn_impl=args.generate_attn_impl,
        )

    rest_length = 0
    rest_doc_tokens = 0
    rest_doc_chunks = 0
    rest_gist_tokens = 0
    rest_actual_ratio = 0.0
    rest_compress_sec = 0.0
    blend_sec = 0.0
    has_c2kv_rest = bool(rest_tools)
    if rest_tools:
        context_input_ids, rest_doc_tokens, rest_doc_chunks, skip_reason = _build_tool_chunks(
            tokenizer,
            rest_definition,
            max_doc_length=args.max_doc_length,
            max_doc_num=args.max_doc_num,
            max_tool_definition_tokens=args.max_tool_definition_tokens,
            truncate_tool_definition=args.truncate_tool_definition,
        )
        if context_input_ids is None:
            return {
                "qid": example.qid,
                "session_id": example.session_id,
                "mode": "hybrid",
                "hybrid_mode": hybrid_mode,
                "router_strategy": args.router_strategy,
                "top_schema_mode": top_schema_mode,
                "top_k": top_k,
                "ratio": ratio,
                "skipped": True,
                "skip_reason": "rest_" + str(skip_reason),
                "num_tools": len(tools),
                "top_tool_names": top_tool_names,
                "rest_doc_tokens": rest_doc_tokens,
            }
        (
            prefix_cache,
            rest_length,
            rest_gist_tokens,
            rest_actual_ratio,
            rest_compress_sec,
            blend_sec,
        ) = _build_tool_cache(
            model,
            context_input_ids,
            prefix_cache,
            system_length + top_length,
            args.gist_attn_impl,
            ratio,
        )

    prompt_ids = _chat_template_ids(
        tokenizer,
        example.input_messages,
        add_generation_prompt=True,
    )
    if args.max_prompt_tokens and len(prompt_ids) > args.max_prompt_tokens:
        prompt_ids = prompt_ids[-args.max_prompt_tokens :]
    prompt_input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
    cache_length = prefix_cache.get_seq_length()
    mock_cache_ids = prompt_input_ids.new_zeros((1, cache_length))
    input_ids = torch.cat([mock_cache_ids, prompt_input_ids], dim=1)
    original_prefix_length = system_length + top_length + rest_length
    position_ids = torch.arange(
        original_prefix_length,
        original_prefix_length + prompt_input_ids.shape[1],
        dtype=torch.long,
        device=model.device,
    ).unsqueeze(0)

    prediction, generate_sec, generated_tokens, tbt_sec = _generate_from_input_ids(
        model,
        tokenizer,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        attn_impl=args.generate_attn_impl,
        use_gist=has_c2kv_rest,
        position_ids=position_ids,
        past_key_values=prefix_cache,
    )
    pred_tool = _extract_tool_name(prediction)
    full_doc_tokens = len(
        _chat_template_ids(
            tokenizer,
            [{"role": "user", "content": "Tool definitions:\n" + example.tool_definition}],
        )
    )
    online_ttft_sec = system_prefill_sec + top_prefill_sec + rest_compress_sec + blend_sec
    cached_ttft_sec = system_prefill_sec + top_prefill_sec + blend_sec
    tool_only_cached_ttft_sec = top_prefill_sec + blend_sec
    cached_total_sec = cached_ttft_sec + generate_sec
    total_sec = time.perf_counter() - total_start
    compressed_tool_tokens = top_length + rest_gist_tokens
    hybrid_ratio = (
        full_doc_tokens / compressed_tool_tokens
        if compressed_tool_tokens else 0.0
    )
    row = {
        "qid": example.qid,
        "session_id": example.session_id,
        "mode": "hybrid",
        "hybrid_mode": hybrid_mode,
        "router_strategy": args.router_strategy,
        "top_schema_mode": top_schema_mode,
        "top_k": top_k,
        "ratio": ratio,
        "skipped": False,
        "num_tools": len(tools),
        "num_top_tools": len(top_tools),
        "num_rest_tools": len(rest_tools),
        "top_tool_names": top_tool_names,
        "top_tool_token_lengths": [
            _tool_token_lengths(tokenizer, [tool])[0] for tool in top_tools
        ],
        "attention_top_tool_scores": (
            [
                round(attention_tool_scores[index], 6)
                for index, tool in enumerate(tools)
                if _tool_name(tool) in set(top_tool_names)
            ]
            if attention_tool_scores is not None else None
        ),
        "attention_score_mode": (
            args.attention_router_score_mode
            if args.router_strategy in {"attention", "lex_attention", "att_rerank"} else None
        ),
        "attention_cache_mode": (
            args.attention_router_cache_mode
            if args.router_strategy in {"attention", "lex_attention", "att_rerank"} else None
        ),
        "attention_router_sec": round(attention_router_sec, 4),
        "attention_lexical_pool": (
            args.attention_router_lexical_pool
            if args.router_strategy == "lex_attention" else None
        ),
        "att_rerank_pool": (
            args.att_rerank_pool if args.router_strategy == "att_rerank" else None
        ),
        "att_rerank_min_heads": (
            args.att_rerank_min_heads if args.router_strategy == "att_rerank" else None
        ),
        "att_rerank_min_margin": (
            args.att_rerank_min_margin if args.router_strategy == "att_rerank" else None
        ),
        "att_rerank_min_score_gain": (
            args.att_rerank_min_score_gain if args.router_strategy == "att_rerank" else None
        ),
        "att_rerank_replaced": (
            bool(att_rerank_debug and att_rerank_debug.get("accepted"))
            if args.router_strategy == "att_rerank" else None
        ),
        "att_rerank_debug": att_rerank_debug,
        "target_lexical_rank": target_lexical_rank,
        "target_attention_rank": target_attention_rank,
        "target_final_rank": target_final_rank,
        "router_scope": args.router_scope,
        "router_strategy": args.router_strategy,
        "router_hit": router_hit,
        "doc_tokens": full_doc_tokens,
        "top_doc_tokens": len(top_ids),
        "rest_doc_tokens": rest_doc_tokens,
        "rest_doc_chunks": rest_doc_chunks,
        "rest_gist_tokens": rest_gist_tokens,
        "actual_compression_ratio": round(hybrid_ratio, 4),
        "rest_actual_compression_ratio": round(rest_actual_ratio, 4),
        "prompt_tokens": len(prompt_ids),
        "target_tokens": len(tokenizer.encode(target, add_special_tokens=False)),
        "generated_tokens": generated_tokens,
        "latency_sec": round(generate_sec, 4),
        "system_prefill_sec": round(system_prefill_sec, 4),
        "top_full_prefill_sec": round(top_prefill_sec, 4),
        "tool_compress_sec": round(rest_compress_sec, 4),
        "full_prefill_sec": round(top_prefill_sec, 4),
        "blend_sec": round(blend_sec, 4),
        "generate_sec": round(generate_sec, 4),
        "ttft_sec": round(online_ttft_sec, 4),
        "online_ttft_sec": round(online_ttft_sec, 4),
        "cached_ttft_sec": round(cached_ttft_sec, 4),
        "tool_only_cached_ttft_sec": round(tool_only_cached_ttft_sec, 4),
        "tbt_sec": round(tbt_sec, 6),
        "total_sec": round(total_sec, 4),
        "cached_total_sec": round(cached_total_sec, 4),
        "target_tool_name": target_tool,
        "prediction_tool_name": pred_tool,
        "tool_name_match": target_tool is not None and target_tool == pred_tool,
        "has_tool_call": "<tool_call>" in prediction or "Action:" in prediction,
        "exact_match": _normalize_text(prediction) == _normalize_text(target),
        "prediction": prediction,
        "target": target,
    }
    return _add_hybrid_debug_fields(
        row,
        args,
        full_definition=example.tool_definition,
        top_definition=top_definition,
        rest_definition=rest_definition,
        numerator_tokens=full_doc_tokens,
        denominator_tokens=compressed_tool_tokens,
        top_tokens=top_length,
        rest_original_tokens=rest_length,
        rest_compressed_tokens=rest_gist_tokens,
    )


def _summarize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries = []
    keys = sorted({
        (
            row.get("hybrid_mode", "hybrid"),
            row.get("router_strategy", "lexical"),
            row.get("top_schema_mode", "full"),
            row.get("top_k"),
            row.get("ratio"),
        )
        for row in rows
    })
    for hybrid_mode, router_strategy, top_schema_mode, top_k, ratio in keys:
        group = [
            row for row in rows
            if row.get("hybrid_mode", "hybrid") == hybrid_mode
            and row.get("router_strategy", "lexical") == router_strategy
            and row.get("top_schema_mode", "full") == top_schema_mode
            and row.get("top_k") == top_k
            and row.get("ratio") == ratio
        ]
        valid_rows = [row for row in group if not row.get("skipped")]
        skip_reasons = Counter(row.get("skip_reason", "unknown") for row in group if row.get("skipped"))
        total_generated = sum(row.get("generated_tokens", 0) for row in valid_rows)
        compressed_tool_total = sum(_row_compressed_tool_tokens(row) for row in valid_rows)
        no_rest_rows = [row for row in valid_rows if row.get("num_rest_tools") == 0]
        summaries.append({
            "mode": "hybrid",
            "hybrid_mode": hybrid_mode,
            "router_strategy": router_strategy,
            "top_schema_mode": top_schema_mode,
            "top_k": top_k,
            "ratio": ratio,
            "num_examples": len(group),
            "num_valid": len(valid_rows),
            "num_skipped": len(group) - len(valid_rows),
            "skip_reasons": dict(skip_reasons),
            "router_hit_rate": (
                sum(1 for row in valid_rows if row.get("router_hit")) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "exact_match": (
                sum(1 for row in valid_rows if row.get("exact_match")) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "tool_name_accuracy": (
                sum(1 for row in valid_rows if row.get("tool_name_match")) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "tool_call_rate": (
                sum(1 for row in valid_rows if row.get("has_tool_call")) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "call_accuracy": (
                sum(1 for row in valid_rows if row.get("tool_name_match"))
                / sum(1 for row in valid_rows if row.get("has_tool_call"))
                if any(row.get("has_tool_call") for row in valid_rows) else 0.0
            ),
            "avg_num_tools": (
                sum(row.get("num_tools", 0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_top_doc_tokens": (
                sum(row.get("top_doc_tokens", 0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_rest_doc_tokens": (
                sum(row.get("rest_doc_tokens", 0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "num_no_rest_tools": len(no_rest_rows),
            "no_rest_tool_rate": (
                len(no_rest_rows) / len(valid_rows) if valid_rows else 0.0
            ),
            "avg_actual_compression_ratio": (
                sum(row.get("actual_compression_ratio", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "token_weighted_actual_compression_ratio": (
                sum(row.get("doc_tokens", 0) for row in valid_rows) / compressed_tool_total
                if compressed_tool_total else 0.0
            ),
            "avg_online_ttft_sec": (
                sum(row.get("online_ttft_sec", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_cached_ttft_sec": (
                sum(row.get("cached_ttft_sec", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_tool_only_cached_ttft_sec": (
                sum(row.get("tool_only_cached_ttft_sec", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_generate_sec": (
                sum(row.get("generate_sec", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_attention_router_sec": (
                sum(row.get("attention_router_sec", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows and any("attention_router_sec" in row for row in valid_rows) else 0.0
            ),
            "avg_generated_tokens": (
                total_generated / len(valid_rows) if valid_rows else 0.0
            ),
            "avg_tbt_sec": (
                sum(row.get("tbt_sec", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "token_weighted_tbt_sec": (
                sum(row.get("generate_sec", 0.0) for row in valid_rows) / total_generated
                if total_generated else 0.0
            ),
            "avg_online_total_sec": (
                sum(row.get("total_sec", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_cached_total_sec": (
                sum(row.get("cached_total_sec", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
        })
    return summaries


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

    data_args = AgentToolDefinitionDataArgs(
        dataset_path=args.dataset_path,
        eval_ratio=args.eval_ratio,
        split_seed=args.split_seed,
        split_manifest_file=args.split_manifest_file,
        split_manifest_name=args.split_manifest_name,
        max_samples_per_session=args.max_samples_per_session,
        max_doc_length=args.max_doc_length,
        max_doc_num=args.max_doc_num,
        max_tool_definition_tokens=args.max_tool_definition_tokens,
        max_length=args.max_length,
        max_system_length=args.max_system_length,
        truncate_tool_definition=args.truncate_tool_definition,
        require_tool_call=args.require_tool_call,
        min_target_tokens=args.min_target_tokens,
    )
    source = AgentLLMTracesSource(data_args)
    source_examples = list(source.iter_examples(args.split))
    if args.max_source_examples is not None:
        source_examples = source_examples[: args.max_source_examples]

    examples = []
    selection_skips: Counter[str] = Counter()
    for example in source_examples:
        num_tools = len(_as_tool_list(example.tool_definition))
        if args.min_num_tools > 0 and num_tools < args.min_num_tools:
            selection_skips[f"num_tools<{args.min_num_tools}"] += 1
            continue
        # Keep the same selection as C2KV eval by default so hybrid is compared
        # on examples whose full tool definition fits the existing C2KV budget.
        if args.selection_filter == "c2kv":
            _, _, _, skip_reason = _build_tool_chunks(
                tokenizer,
                example.tool_definition,
                max_doc_length=args.max_doc_length,
                max_doc_num=args.max_doc_num,
                max_tool_definition_tokens=args.max_tool_definition_tokens,
                truncate_tool_definition=args.truncate_tool_definition,
            )
            if skip_reason is not None:
                selection_skips[skip_reason] += 1
                continue
        examples.append(example)
        if args.max_examples is not None and args.max_examples > 0 and len(examples) >= args.max_examples:
            break

    cases = _parse_cases(args.hybrid_cases)
    logger.info(
        "Selected %d examples from %d source examples; cases=%s; selection_skips=%s",
        len(examples),
        len(source_examples),
        cases,
        dict(selection_skips),
    )
    model_args = copy.copy(args)
    model_args.mode = "c2kv"
    model_args.untrained_c2kv = False
    model = _load_model(model_args, tokenizer, device)

    rows: List[Dict[str, Any]] = []
    for top_k, ratio in cases:
        desc = f"hybrid_top{top_k}_c2kv{ratio}x"
        for example in tqdm(examples, desc=desc):
            rows.append(_generate_one_hybrid(model, tokenizer, example, args, top_k, ratio))

    summaries = _summarize_rows(rows)
    summary = {
        "model": args.model,
        "dataset_path": args.dataset_path,
        "split": args.split,
        "router_scope": args.router_scope,
        "router_strategy": args.router_strategy,
        "hybrid_mode": args.hybrid_mode,
        "top_schema_mode": args.top_schema_mode,
        "hybrid_cases": args.hybrid_cases,
        "selection_skips": dict(selection_skips),
        "results": summaries,
        "num_rows": len(rows),
    }
    if args.output_file:
        _jsonl_write(args.output_file, rows)
        summary_path = str(Path(args.output_file).with_suffix(".summary.json"))
        Path(summary_path).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Wrote predictions to %s", args.output_file)
        logger.info("Wrote summary to %s", summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate top-k full tools + rest C2KV hybrid routing.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base_model")
    parser.add_argument("--tokenizer")
    parser.add_argument("--dataset_path", default="./datasets/agent-llm-traces")
    parser.add_argument("--output_file", default="./outputs/agent_tooldef_hybrid_router_eval.jsonl")
    parser.add_argument("--split", choices=["train", "eval"], default="eval")
    parser.add_argument("--hybrid_cases", default="1:4,3:4,5:8")
    parser.add_argument("--hybrid_mode", choices=["hybrid", "drop_selected", "topk_only"], default="hybrid")
    parser.add_argument("--router_scope", choices=["last_user", "all"], default="last_user")
    parser.add_argument(
        "--router_strategy",
        choices=["lexical", "random", "attention", "lex_attention", "att_rerank"],
        default="lexical",
    )
    parser.add_argument("--top_schema_mode", choices=["full", "compact"], default="full")
    parser.add_argument("--router_hit_filter", choices=["all", "hit", "miss"], default="all")
    parser.add_argument("--router_seed", type=int, default=42)
    parser.add_argument("--attention_router_layers", type=int, default=4)
    parser.add_argument("--attention_router_attn_impl", default="eager")
    parser.add_argument("--attention_router_max_query_tokens", type=int, default=512)
    parser.add_argument(
        "--attention_router_score_mode",
        choices=["mean", "sqrt_len", "sum"],
        default="mean",
        help="How to aggregate query attention over each tool's KV span.",
    )
    parser.add_argument(
        "--attention_router_cache_mode",
        choices=["c2kv", "full"],
        default="c2kv",
        help="Whether attention routing reads attention over C2KV gist tokens or full tool KV.",
    )
    parser.add_argument(
        "--attention_router_lexical_pool",
        type=int,
        default=10,
        help="For lex_attention, rerank this many lexical candidates with attention.",
    )
    parser.add_argument("--att_rerank_pool", type=int, default=10)
    parser.add_argument("--att_rerank_min_heads", type=int, default=3)
    parser.add_argument("--att_rerank_min_margin", type=float, default=0.0)
    parser.add_argument("--att_rerank_min_score_gain", type=float, default=0.0)
    parser.add_argument(
        "--debug_hybrid_tokens",
        action="store_true",
        help="Write hybrid compression numerator/denominator token breakdowns into each row.",
    )
    parser.add_argument(
        "--dump_hybrid_definitions",
        action="store_true",
        help="Also write selected/full tool definitions into each debug row. Use with small max_examples.",
    )
    parser.add_argument("--debug_definition_chars", type=int, default=4000)
    parser.add_argument("--max_examples", type=int, default=50, help="Maximum examples; <=0 means all selected examples.")
    parser.add_argument("--max_source_examples", type=int)
    parser.add_argument("--selection_filter", choices=["c2kv", "none"], default="c2kv")
    parser.add_argument("--min_num_tools", type=int, default=0)
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--split_manifest_file")
    parser.add_argument("--split_manifest_name", default="toolset_disjoint")
    parser.add_argument("--max_samples_per_session", type=int, default=4)
    parser.add_argument("--max_doc_length", type=int, default=1024)
    parser.add_argument("--max_doc_num", type=int, default=10)
    parser.add_argument("--max_tool_definition_tokens", type=int, default=10000)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_system_length", type=int, default=256)
    parser.add_argument("--max_prompt_tokens", type=int, default=1920)
    parser.add_argument("--min_target_tokens", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--truncate_tool_definition", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--require_tool_call", type=lambda x: str(x).lower() == "true", default=True)
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
    summary = evaluate(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
