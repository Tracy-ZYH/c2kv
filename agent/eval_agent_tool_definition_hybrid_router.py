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


def _text_tokens(text: str) -> List[str]:
    return re.findall(r"\w+", _normalize_text(text))


def _text_token_f1(target: str, prediction: str) -> float:
    target_tokens = _text_tokens(target)
    prediction_tokens = _text_tokens(prediction)
    if not target_tokens and not prediction_tokens:
        return 1.0
    if not target_tokens or not prediction_tokens:
        return 0.0
    overlap = sum((Counter(target_tokens) & Counter(prediction_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def _rouge_l_f1(target: str, prediction: str) -> float:
    target_tokens = _text_tokens(target)
    prediction_tokens = _text_tokens(prediction)
    if not target_tokens and not prediction_tokens:
        return 1.0
    if not target_tokens or not prediction_tokens:
        return 0.0
    overlap = _lcs_length(target_tokens, prediction_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def _row_text_token_f1(row: Dict[str, Any]) -> float:
    if "text_token_f1" in row:
        return float(row.get("text_token_f1") or 0.0)
    return _text_token_f1(row.get("target", ""), row.get("prediction", ""))


def _row_rouge_l_f1(row: Dict[str, Any]) -> float:
    if "rouge_l_f1" in row:
        return float(row.get("rouge_l_f1") or 0.0)
    return _rouge_l_f1(row.get("target", ""), row.get("prediction", ""))


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
    if "compressed_tool_tokens" in row:
        return float(row.get("compressed_tool_tokens", 0) or 0)
    return float(row.get("top_doc_tokens", 0) or 0) + float(row.get("rest_gist_tokens", 0) or 0)


def _row_tool_original_tokens(row: Dict[str, Any]) -> float:
    if "tool_original_tokens" in row:
        return float(row.get("tool_original_tokens", 0) or 0)
    if any(key in row for key in ("full_tool_tokens", "c2kv2_doc_tokens", "c2kv4_doc_tokens")):
        full_schema_tokens = (
            float(row.get("full_schema_tool_tokens", 0) or 0)
            if "full_schema_tool_tokens" in row
            else float(row.get("full_tool_tokens", 0) or 0)
            - float(row.get("stable_summary_full_tokens", 0) or 0)
        )
        return (
            full_schema_tokens
            + float(row.get("c2kv2_doc_tokens", 0) or 0)
            + float(row.get("c2kv4_doc_tokens", 0) or 0)
        )
    return float(row.get("doc_tokens", 0) or 0)


def _row_actual_compression_ratio(row: Dict[str, Any]) -> float:
    original_tokens = _row_tool_original_tokens(row)
    compressed_tokens = _row_compressed_tool_tokens(row)
    if original_tokens > 0 and compressed_tokens > 0:
        return original_tokens / compressed_tokens
    return float(row.get("actual_compression_ratio", 0.0) or 0.0)


def _row_response_type_match(row: Dict[str, Any]) -> bool:
    if "response_type_match" in row:
        return bool(row.get("response_type_match"))
    target_has_tool_call = bool(row.get("target_tool_name")) or "<tool_call>" in row.get("target", "") or "Action:" in row.get("target", "")
    return target_has_tool_call == bool(row.get("has_tool_call"))


def _basic_metric_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    called_rows = [row for row in rows if row.get("has_tool_call")]
    return {
        "num_samples": len(rows),
        "exact_match": (
            sum(1 for row in rows if row.get("exact_match")) / len(rows)
            if rows else 0.0
        ),
        "avg_text_token_f1": (
            sum(_row_text_token_f1(row) for row in rows) / len(rows)
            if rows else 0.0
        ),
        "avg_rouge_l_f1": (
            sum(_row_rouge_l_f1(row) for row in rows) / len(rows)
            if rows else 0.0
        ),
        "response_type_accuracy": (
            sum(1 for row in rows if _row_response_type_match(row)) / len(rows)
            if rows else 0.0
        ),
        "tool_name_accuracy": (
            sum(1 for row in rows if row.get("tool_name_match")) / len(rows)
            if rows else 0.0
        ),
        "tool_call_rate": (
            len(called_rows) / len(rows)
            if rows else 0.0
        ),
        "call_accuracy": (
            sum(1 for row in called_rows if row.get("tool_name_match")) / len(called_rows)
            if called_rows else 0.0
        ),
    }


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


DEFAULT_STABLE_RETRIEVAL_HEADS = (
    (23, 10),
    (20, 15),
    (21, 11),
    (22, 4),
    (15, 9),
    (21, 18),
    (23, 13),
    (17, 4),
    (18, 15),
    (20, 16),
    (24, 13),
    (19, 12),
    (18, 19),
    (19, 13),
    (13, 2),
    (17, 27),
)


def _parse_stable_heads(spec: str) -> List[tuple[int, int]]:
    if not spec:
        return list(DEFAULT_STABLE_RETRIEVAL_HEADS)
    heads: List[tuple[int, int]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            layer, head = item.split(":", 1)
        elif "." in item:
            layer, head = item.split(".", 1)
        else:
            raise ValueError(f"Invalid stable head {item!r}; expected layer:head")
        heads.append((int(layer), int(head)))
    return heads


def _normalized_entropy_confidence(scores: Sequence[float]) -> float:
    positives = [max(0.0, float(score)) for score in scores]
    total = sum(positives)
    if total <= 0.0 or len(positives) <= 1:
        return 0.0
    probs = [score / total for score in positives if score > 0.0]
    if not probs:
        return 0.0
    entropy = -sum(prob * torch.log(torch.tensor(prob)).item() for prob in probs)
    max_entropy = torch.log(torch.tensor(float(len(positives)))).item()
    if max_entropy <= 0.0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - entropy / max_entropy))


def _select_vote_heads(
    head_rankings: Sequence[Dict[str, Any]],
    pool: Sequence[int],
    *,
    stable: bool,
    stable_heads: Sequence[tuple[int, int]],
    stable_head_count: int,
) -> List[Dict[str, Any]]:
    if not stable:
        return [dict(head, vote_weight=1.0) for head in head_rankings]
    lookup = {(int(head.get("layer")), int(head.get("head"))): head for head in head_rankings}
    selected = [dict(lookup[key], vote_weight=1.0) for key in stable_heads if key in lookup]
    if selected:
        return selected[:stable_head_count]

    def pool_margin(head: Dict[str, Any]) -> float:
        scores_by_index = head.get("scores_by_index") or {}
        values = sorted(
            [float(scores_by_index.get(index, 0.0) or 0.0) for index in pool],
            reverse=True,
        )
        if len(values) <= 1:
            return values[0] if values else 0.0
        return values[0] - values[1]

    fallback = sorted(head_rankings, key=pool_margin, reverse=True)[:stable_head_count]
    return [
        dict(head, vote_weight=max(pool_margin(head), 1e-8))
        for head in fallback
    ]


def _rank_lexical_pool_by_head_rrf(
    lexical_ranked: Sequence[int],
    head_rankings: Sequence[Dict[str, Any]],
    *,
    top_k: int,
    pool_size: int,
    rrf_k: float,
    stable: bool,
    stable_heads: Sequence[tuple[int, int]],
    stable_head_count: int,
) -> tuple[List[int], Dict[str, Any]]:
    pool_size = max(top_k, min(pool_size, len(lexical_ranked)))
    pool = list(lexical_ranked[:pool_size])
    pool_set = set(pool)
    selected_heads = _select_vote_heads(
        head_rankings,
        pool,
        stable=stable,
        stable_heads=stable_heads,
        stable_head_count=stable_head_count,
    )
    rrf_scores = {index: 0.0 for index in pool}
    confidence_values = []
    head_votes = []
    for head in selected_heads:
        ranked = [index for index in (head.get("ranked_indices") or []) if index in pool_set]
        scores_by_index = head.get("scores_by_index") or {}
        confidence = _normalized_entropy_confidence([
            float(scores_by_index.get(index, 0.0) or 0.0)
            for index in pool
        ])
        weight = float(head.get("vote_weight", 1.0) or 1.0)
        confidence_values.append(confidence)
        for rank, index in enumerate(ranked, start=1):
            rrf_scores[index] += weight * confidence / (rrf_k + rank)
        if ranked:
            head_votes.append({
                "layer": head.get("layer"),
                "head": head.get("head"),
                "top_tool_index": ranked[0],
                "weight": round(weight, 8),
                "confidence": round(confidence, 6),
            })
    lexical_pool_rank = {index: rank for rank, index in enumerate(pool)}
    voted_pool = sorted(pool, key=lambda index: (-rrf_scores[index], lexical_pool_rank[index]))
    voted_set = set(voted_pool)
    debug = {
        "vote_pool_size": pool_size,
        "vote_num_heads": len(selected_heads),
        "vote_stable": stable,
        "vote_rrf_k": rrf_k,
        "vote_avg_confidence": (
            sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        ),
        "vote_top_indices": voted_pool[:top_k],
        "vote_top_scores": [round(rrf_scores[index], 8) for index in voted_pool[:top_k]],
        "vote_head_votes": head_votes[:20],
    }
    return voted_pool + [index for index in lexical_ranked if index not in voted_set], debug


def _select_stable_extra_candidate(
    lexical_ranked: Sequence[int],
    head_rankings: Sequence[Dict[str, Any]],
    *,
    base_top_k: int,
    pool_size: int,
    rrf_k: float,
    stable_heads: Sequence[tuple[int, int]],
    stable_head_count: int,
) -> tuple[Optional[int], Optional[int], Dict[str, Any]]:
    pool_size = max(base_top_k + 1, min(pool_size, len(lexical_ranked)))
    candidates = list(lexical_ranked[base_top_k:pool_size])
    if not candidates:
        return None, None, {
            "stable_plus_pool_size": pool_size,
            "stable_plus_num_candidates": 0,
            "reason": "empty_candidate_pool",
        }
    voted_ranked, debug = _rank_lexical_pool_by_head_rrf(
        candidates,
        head_rankings,
        top_k=1,
        pool_size=len(candidates),
        rrf_k=rrf_k,
        stable=True,
        stable_heads=stable_heads,
        stable_head_count=stable_head_count,
    )
    candidate = voted_ranked[0] if voted_ranked else None
    lexical_rank = list(lexical_ranked).index(candidate) + 1 if candidate is not None else None
    debug.update({
        "stable_plus_pool_size": pool_size,
        "stable_plus_candidate_indices": candidates,
        "stable_plus_candidate_index": candidate,
        "stable_plus_candidate_lexical_rank": lexical_rank,
    })
    return candidate, lexical_rank, debug


def _parameter_names(parameters: Any) -> List[str]:
    if not isinstance(parameters, dict):
        return []
    names = []
    properties = parameters.get("properties")
    if isinstance(properties, dict):
        names.extend(str(name) for name in properties.keys())
    required = parameters.get("required")
    if isinstance(required, list):
        for name in required:
            name = str(name)
            if name not in names:
                names.append(name)
    return names


def _tool_router_summary(tool: Dict[str, Any]) -> Dict[str, Any]:
    function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
    parameters = (
        function.get("parameters")
        or tool.get("parameters")
        or tool.get("input_schema")
        or tool.get("schema")
    )
    summary = {
        "tool_name": _tool_name(tool),
        "description": function.get("description") or tool.get("description") or "",
        "parameter_names": _parameter_names(parameters),
    }
    return summary


def _conservative_vote_replacement(
    lexical_ranked: Sequence[int],
    voted_ranked: Sequence[int],
    top_k: int,
) -> tuple[List[int], Dict[str, Any]]:
    if top_k < 3 or len(lexical_ranked) < top_k:
        return list(lexical_ranked), {"accepted": False, "reason": "top_k<3"}
    fixed = list(lexical_ranked[:2])
    replacement = next((index for index in voted_ranked if index not in set(fixed)), lexical_ranked[2])
    accepted = replacement != lexical_ranked[2]
    selected = fixed + [replacement]
    selected_set = set(selected)
    final = selected + [index for index in lexical_ranked if index not in selected_set]
    return final, {
        "accepted": accepted,
        "replace_tool_index": lexical_ranked[2],
        "candidate_tool_index": replacement,
        "candidate_lexical_rank": list(lexical_ranked).index(replacement) + 1,
    }


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


def _sync_device(device: Any) -> None:
    device_type = getattr(device, "type", str(device))
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device_type == "npu" and hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()


@torch.inference_mode()
def _prefill_tokens_with_cache_maybe_gist(
    model: Any,
    input_ids: torch.Tensor,
    past_key_values: Any,
    past_length: int,
    attn_impl: str,
    *,
    use_gist: bool,
) -> tuple[Any, int, float]:
    if input_ids.shape[1] == 0:
        return past_key_values, 0, 0.0
    original_attn_impl = model.model.config._attn_implementation
    model.model.config._attn_implementation = attn_impl
    input_length = input_ids.shape[1]
    cache_length = (
        past_key_values.get_seq_length()
        if past_key_values is not None and hasattr(past_key_values, "get_seq_length")
        else past_length
    )
    attention_mask = torch.ones(
        (input_ids.shape[0], cache_length + input_length),
        dtype=torch.long,
        device=input_ids.device,
    )
    position_ids = torch.arange(
        past_length,
        past_length + input_length,
        dtype=torch.long,
        device=input_ids.device,
    ).unsqueeze(0)
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_values": past_key_values,
        "use_cache": True,
        "logits_to_keep": 1,
    }
    if use_gist:
        kwargs["use_gist"] = True
    _sync_device(input_ids.device)
    start = time.perf_counter()
    outputs = model(**kwargs)
    _sync_device(input_ids.device)
    elapsed = time.perf_counter() - start
    model.model.config._attn_implementation = original_attn_impl
    return outputs.past_key_values, input_length, elapsed


def _tool_single_doc_ids(
    tokenizer: Any,
    tool: Dict[str, Any],
    *,
    label: str = "Tool definition",
    schema_mode: str = "full",
) -> List[int]:
    tool_doc = {
        "role": "user",
        "content": f"{label}:\n" + _render_tool_definition([tool], schema_mode),
    }
    return _chat_template_ids(tokenizer, [tool_doc])


def _summary_single_doc_ids(tokenizer: Any, tool: Dict[str, Any], tool_index: int) -> List[int]:
    summary_doc = {
        "role": "user",
        "content": (
            f"Router summary for original tool index {tool_index}:\n"
            + json.dumps(_tool_router_summary(tool), ensure_ascii=False, separators=(",", ":"))
        ),
    }
    return _chat_template_ids(tokenizer, [summary_doc])


@torch.inference_mode()
def _append_full_tool_segment(
    model: Any,
    tokenizer: Any,
    prefix_cache: Any,
    logical_length: int,
    tool: Dict[str, Any],
    args: argparse.Namespace,
    *,
    use_gist: bool,
    label: str = "Tool definition",
) -> tuple[Any, int, float]:
    ids = _tool_single_doc_ids(tokenizer, tool, label=label)
    input_ids = torch.tensor([ids], dtype=torch.long, device=model.device)
    cache, length, elapsed = _prefill_tokens_with_cache_maybe_gist(
        model,
        input_ids,
        past_key_values=prefix_cache,
        past_length=logical_length,
        attn_impl=args.generate_attn_impl,
        use_gist=use_gist,
    )
    return cache, length, elapsed


@torch.inference_mode()
def _append_full_summary_segment(
    model: Any,
    tokenizer: Any,
    prefix_cache: Any,
    logical_length: int,
    tool: Dict[str, Any],
    tool_index: int,
    args: argparse.Namespace,
    *,
    use_gist: bool,
) -> tuple[Any, int, float]:
    ids = _summary_single_doc_ids(tokenizer, tool, tool_index)
    input_ids = torch.tensor([ids], dtype=torch.long, device=model.device)
    cache, length, elapsed = _prefill_tokens_with_cache_maybe_gist(
        model,
        input_ids,
        past_key_values=prefix_cache,
        past_length=logical_length,
        attn_impl=args.generate_attn_impl,
        use_gist=use_gist,
    )
    return cache, length, elapsed


@torch.inference_mode()
def _append_c2kv_tool_segment(
    model: Any,
    tokenizer: Any,
    prefix_cache: Any,
    logical_length: int,
    tool: Dict[str, Any],
    args: argparse.Namespace,
    *,
    ratio: int,
) -> tuple[Any, int, int, int, float, float, float, Optional[str]]:
    definition = _render_tool_definition([tool])
    context_input_ids, doc_tokens, doc_chunks, skip_reason = _build_tool_chunks(
        tokenizer,
        definition,
        max_doc_length=args.max_doc_length,
        max_doc_num=args.max_doc_num,
        max_tool_definition_tokens=args.max_tool_definition_tokens,
        truncate_tool_definition=args.truncate_tool_definition,
        document_mode="per_tool",
    )
    if context_input_ids is None:
        return prefix_cache, 0, doc_tokens, doc_chunks, 0.0, 0.0, 0.0, skip_reason
    cache, length, gist_tokens, actual_ratio, compress_sec, blend_sec = _build_tool_cache(
        model,
        context_input_ids,
        prefix_cache,
        logical_length,
        args.gist_attn_impl,
        ratio,
    )
    return cache, length, doc_tokens, doc_chunks, gist_tokens, actual_ratio, compress_sec + blend_sec, None


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
            document_mode=args.tool_document_eval_mode,
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
        elif args.attention_router_score_mode == "top4_mean":
            flat = tool_attn.reshape(-1)
            top_n = min(max(1, args.attention_router_span_top_tokens), flat.numel())
            score = torch.topk(flat, top_n).values.mean()
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


STABLE_PLUS_STRATEGIES = {
    "lex_top3_original_order",
    "lex_top3_plus_stable1_full",
    "lex_top3_plus_stable1_c2kv2",
    "lex_top3_plus_stable1_name_desc_full",
}


@torch.inference_mode()
def _generate_one_stable_plus_hybrid(
    model: Any,
    tokenizer: Any,
    example: Any,
    args: argparse.Namespace,
    top_k: int,
    ratio: int,
) -> Dict[str, Any]:
    total_start = time.perf_counter()
    tools = _as_tool_list(example.tool_definition)
    if not tools:
        return {
            "qid": example.qid,
            "session_id": example.session_id,
            "mode": "hybrid",
            "hybrid_mode": getattr(args, "hybrid_mode", "hybrid"),
            "router_strategy": args.router_strategy,
            "top_k": top_k,
            "ratio": ratio,
            "skipped": True,
            "skip_reason": "no_parseable_tools",
        }

    query = _query_text(example.input_messages, args.router_scope)
    target = example.answer.strip()
    target_tool = _extract_tool_name(target)
    lexical_ranked = _rank_tools(tools, query)
    if args.router_strategy == "lex_top3_original_order":
        attention_ranked: List[int] = []
        attention_router_sec = 0.0
        stable_candidate = None
        stable_candidate_lexical_rank = None
        vote_debug = {
            "reason": "lexical_only_original_order_baseline",
            "stable_plus_pool_size": 0,
            "stable_plus_num_candidates": 0,
        }
    else:
        attention_ranked, _, attention_router_sec, head_rankings = _rank_tools_by_attention(
            model, tokenizer, example, args, tools, ratio
        )
        stable_candidate, stable_candidate_lexical_rank, vote_debug = _select_stable_extra_candidate(
            lexical_ranked,
            head_rankings,
            base_top_k=top_k,
            pool_size=args.attention_router_lexical_pool,
            rrf_k=args.attention_rrf_k,
            stable_heads=_parse_stable_heads(args.attention_stable_heads),
            stable_head_count=args.attention_stable_head_count,
        )
    lexical_top_indices = set(lexical_ranked[:top_k])
    stable_set = {stable_candidate} if stable_candidate is not None else set()
    recovered_indices = lexical_top_indices | stable_set
    lexical_top3_tool_names = [_tool_name(tools[index]) for index in lexical_ranked[:top_k]]
    stable_candidate_name = _tool_name(tools[stable_candidate]) if stable_candidate is not None else None
    top_tool_names = [
        _tool_name(tool)
        for index, tool in enumerate(tools)
        if index in recovered_indices
    ]
    target_lexical_rank = _rank_in_tool_order(tools, lexical_ranked, target_tool)
    target_attention_rank = _rank_in_tool_order(tools, attention_ranked, target_tool)
    final_ranked = list(lexical_ranked[:top_k])
    if stable_candidate is not None and stable_candidate not in final_ranked:
        final_ranked.append(stable_candidate)
    final_ranked.extend(index for index in lexical_ranked if index not in set(final_ranked))
    target_final_rank = _rank_in_tool_order(tools, final_ranked, target_tool)
    lexical_top3_hit = bool(target_lexical_rank is not None and target_lexical_rank <= top_k)
    stable_candidate_hit = bool(target_tool and stable_candidate_name == target_tool)
    final_recovery_hit = bool(lexical_top3_hit or stable_candidate_hit)
    lexical_recall = {
        f"lexical_hit_at_{k}": bool(target_lexical_rank is not None and target_lexical_rank <= k)
        for k in (1, 3, 5, 10, 20)
    }

    if args.router_hit_filter == "hit" and not final_recovery_hit:
        return {
            "qid": example.qid,
            "session_id": example.session_id,
            "mode": "hybrid",
            "hybrid_mode": getattr(args, "hybrid_mode", "hybrid"),
            "router_strategy": args.router_strategy,
            "top_k": top_k,
            "ratio": ratio,
            "skipped": True,
            "skip_reason": "router_miss_filtered",
            "num_tools": len(tools),
            "top_tool_names": top_tool_names,
            "target_tool_name": target_tool,
            "router_hit": final_recovery_hit,
        }
    if args.router_hit_filter == "miss" and final_recovery_hit:
        return {
            "qid": example.qid,
            "session_id": example.session_id,
            "mode": "hybrid",
            "hybrid_mode": getattr(args, "hybrid_mode", "hybrid"),
            "router_strategy": args.router_strategy,
            "top_k": top_k,
            "ratio": ratio,
            "skipped": True,
            "skip_reason": "router_hit_filtered",
            "num_tools": len(tools),
            "top_tool_names": top_tool_names,
            "target_tool_name": target_tool,
            "router_hit": final_recovery_hit,
        }

    system_ids = _chat_template_ids(
        tokenizer,
        [{"role": "system", "content": example.system_prompt}],
        keep_bos=True,
        max_length=args.max_system_length,
    )
    system_input_ids = torch.tensor([system_ids], dtype=torch.long, device=model.device)
    prefix_cache, system_length, system_prefill_sec = _prefill_system(
        model, system_input_ids, args.system_attn_impl
    )
    logical_length = system_length
    full_tool_tokens = 0
    full_schema_tool_tokens = 0
    stable_summary_full_tokens = 0
    c2kv2_doc_tokens = 0
    c2kv2_gist_tokens = 0
    c2kv4_doc_tokens = 0
    c2kv4_gist_tokens = 0
    c2kv_doc_chunks = 0
    full_prefill_sec = 0.0
    c2kv2_compress_sec = 0.0
    c2kv4_compress_sec = 0.0
    c2kv2_ratios: List[float] = []
    c2kv4_ratios: List[float] = []
    has_c2kv_segment = False
    tool_precision_by_index: List[Dict[str, Any]] = []

    for index, tool in enumerate(tools):
        if index in lexical_top_indices or (
            args.router_strategy == "lex_top3_plus_stable1_full" and index in stable_set
        ):
            prefix_cache, length, elapsed = _append_full_tool_segment(
                model,
                tokenizer,
                prefix_cache,
                logical_length,
                tool,
                args,
                use_gist=has_c2kv_segment,
                label=f"Tool definition {index}",
            )
            logical_length += length
            full_tool_tokens += length
            full_schema_tool_tokens += length
            full_prefill_sec += elapsed
            precision = "full"
        elif args.router_strategy == "lex_top3_plus_stable1_name_desc_full" and index in stable_set:
            prefix_cache, summary_length, elapsed = _append_full_summary_segment(
                model, tokenizer, prefix_cache, logical_length, tool, index, args, use_gist=has_c2kv_segment
            )
            logical_length += summary_length
            stable_summary_full_tokens += summary_length
            full_tool_tokens += summary_length
            full_prefill_sec += elapsed
            (
                prefix_cache,
                original_length,
                doc_tokens,
                doc_chunks,
                gist_tokens,
                actual_ratio,
                elapsed,
                skip_reason,
            ) = _append_c2kv_tool_segment(
                model, tokenizer, prefix_cache, logical_length, tool, args, ratio=ratio
            )
            if skip_reason is not None:
                return {
                    "qid": example.qid,
                    "session_id": example.session_id,
                    "mode": "hybrid",
                    "hybrid_mode": getattr(args, "hybrid_mode", "hybrid"),
                    "router_strategy": args.router_strategy,
                    "top_k": top_k,
                    "ratio": ratio,
                    "skipped": True,
                    "skip_reason": "tool_" + str(skip_reason),
                    "num_tools": len(tools),
                    "tool_index": index,
                }
            logical_length += original_length
            c2kv4_doc_tokens += doc_tokens
            c2kv4_gist_tokens += gist_tokens
            c2kv_doc_chunks += doc_chunks
            c2kv4_compress_sec += elapsed
            c2kv4_ratios.append(actual_ratio)
            has_c2kv_segment = True
            precision = "summary_full+c2kv4"
        else:
            segment_ratio = 2 if (
                args.router_strategy == "lex_top3_plus_stable1_c2kv2" and index in stable_set
            ) else ratio
            (
                prefix_cache,
                original_length,
                doc_tokens,
                doc_chunks,
                gist_tokens,
                actual_ratio,
                elapsed,
                skip_reason,
            ) = _append_c2kv_tool_segment(
                model, tokenizer, prefix_cache, logical_length, tool, args, ratio=segment_ratio
            )
            if skip_reason is not None:
                return {
                    "qid": example.qid,
                    "session_id": example.session_id,
                    "mode": "hybrid",
                    "hybrid_mode": getattr(args, "hybrid_mode", "hybrid"),
                    "router_strategy": args.router_strategy,
                    "top_k": top_k,
                    "ratio": ratio,
                    "skipped": True,
                    "skip_reason": "tool_" + str(skip_reason),
                    "num_tools": len(tools),
                    "tool_index": index,
                }
            logical_length += original_length
            c2kv_doc_chunks += doc_chunks
            has_c2kv_segment = True
            if segment_ratio == 2:
                c2kv2_doc_tokens += doc_tokens
                c2kv2_gist_tokens += gist_tokens
                c2kv2_compress_sec += elapsed
                c2kv2_ratios.append(actual_ratio)
                precision = "c2kv2"
            else:
                c2kv4_doc_tokens += doc_tokens
                c2kv4_gist_tokens += gist_tokens
                c2kv4_compress_sec += elapsed
                c2kv4_ratios.append(actual_ratio)
                precision = "c2kv4"
        tool_precision_by_index.append({
            "index": index,
            "name": _tool_name(tool),
            "precision": precision,
        })

    prompt_ids = _chat_template_ids(tokenizer, example.input_messages, add_generation_prompt=True)
    if args.max_prompt_tokens and len(prompt_ids) > args.max_prompt_tokens:
        prompt_ids = prompt_ids[-args.max_prompt_tokens :]
    prompt_input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
    cache_length = prefix_cache.get_seq_length()
    mock_cache_ids = prompt_input_ids.new_zeros((1, cache_length))
    input_ids = torch.cat([mock_cache_ids, prompt_input_ids], dim=1)
    position_ids = torch.arange(
        logical_length,
        logical_length + prompt_input_ids.shape[1],
        dtype=torch.long,
        device=model.device,
    ).unsqueeze(0)
    prediction, generate_sec, generated_tokens, tbt_sec = _generate_from_input_ids(
        model,
        tokenizer,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        attn_impl=args.generate_attn_impl,
        use_gist=has_c2kv_segment,
        position_ids=position_ids,
        past_key_values=prefix_cache,
    )
    pred_tool = _extract_tool_name(prediction)
    prediction_has_tool_call = "<tool_call>" in prediction or "Action:" in prediction
    target_has_tool_call = bool(target_tool) or "<tool_call>" in target or "Action:" in target
    combined_full_doc_tokens = len(_chat_template_ids(
        tokenizer,
        [{"role": "user", "content": "Tool definitions:\n" + example.tool_definition}],
    ))
    tool_original_tokens = full_schema_tool_tokens + c2kv2_doc_tokens + c2kv4_doc_tokens
    compressed_tool_tokens = full_tool_tokens + c2kv2_gist_tokens + c2kv4_gist_tokens
    hybrid_ratio = tool_original_tokens / compressed_tool_tokens if compressed_tool_tokens else 0.0
    online_ttft_sec = system_prefill_sec + full_prefill_sec + c2kv2_compress_sec + c2kv4_compress_sec
    cached_ttft_sec = system_prefill_sec + full_prefill_sec
    total_sec = time.perf_counter() - total_start
    top_definition = _render_tool_definition([
        tools[index] for index in range(len(tools)) if index in recovered_indices
    ])
    rest_definition = _render_tool_definition([
        tools[index] for index in range(len(tools)) if index not in recovered_indices
    ])
    full_tool_names = [
        item["name"] for item in tool_precision_by_index
        if item["precision"] == "full"
    ]
    c2kv2_tool_names = [
        item["name"] for item in tool_precision_by_index
        if item["precision"] == "c2kv2"
    ]
    c2kv4_tool_names = [
        item["name"] for item in tool_precision_by_index
        if item["precision"] in {"c2kv4", "summary_full+c2kv4"}
    ]
    row = {
        "qid": example.qid,
        "session_id": example.session_id,
        "mode": "hybrid",
        "hybrid_mode": getattr(args, "hybrid_mode", "hybrid"),
        "router_strategy": args.router_strategy,
        "top_schema_mode": getattr(args, "top_schema_mode", "full"),
        "top_k": top_k,
        "ratio": ratio,
        "skipped": False,
        "num_tools": len(tools),
        "num_top_tools": len(recovered_indices),
        "num_rest_tools": len(tools) - len(recovered_indices),
        "top_tool_names": top_tool_names,
        "lexical_top3_tool_names": lexical_top3_tool_names,
        "stable_candidate_tool_name": stable_candidate_name,
        "stable_candidate_lexical_rank": stable_candidate_lexical_rank,
        "full_tool_names": full_tool_names,
        "c2kv2_tool_names": c2kv2_tool_names,
        "c2kv4_tool_names": c2kv4_tool_names,
        "lexical_top3_hit": lexical_top3_hit,
        "stable_candidate_hit": stable_candidate_hit,
        "final_recovery_hit": final_recovery_hit,
        "promotion_gain": bool((not lexical_top3_hit) and stable_candidate_hit),
        "demotion_loss": False,
        "lexical_hit_at_topk": lexical_top3_hit,
        "final_hit_at_topk": final_recovery_hit,
        **lexical_recall,
        "target_lexical_rank": target_lexical_rank,
        "target_attention_rank": target_attention_rank,
        "target_final_rank": target_final_rank,
        "router_scope": args.router_scope,
        "router_hit": final_recovery_hit,
        "attention_score_mode": args.attention_router_score_mode,
        "attention_cache_mode": args.attention_router_cache_mode,
        "attention_router_sec": round(attention_router_sec, 4),
        "attention_lexical_pool": args.attention_router_lexical_pool,
        "vote_debug": vote_debug,
        "tool_precision_by_index": tool_precision_by_index,
        "doc_tokens": tool_original_tokens,
        "combined_full_doc_tokens": combined_full_doc_tokens,
        "tool_original_tokens": tool_original_tokens,
        "top_doc_tokens": full_tool_tokens,
        "rest_doc_tokens": c2kv2_doc_tokens + c2kv4_doc_tokens,
        "rest_doc_chunks": c2kv_doc_chunks,
        "rest_gist_tokens": c2kv2_gist_tokens + c2kv4_gist_tokens,
        "full_tool_tokens": full_tool_tokens,
        "full_schema_tool_tokens": full_schema_tool_tokens,
        "stable_summary_full_tokens": stable_summary_full_tokens,
        "c2kv2_doc_tokens": c2kv2_doc_tokens,
        "c2kv2_gist_tokens": c2kv2_gist_tokens,
        "c2kv4_doc_tokens": c2kv4_doc_tokens,
        "c2kv4_gist_tokens": c2kv4_gist_tokens,
        "compressed_tool_tokens": compressed_tool_tokens,
        "tool_kv_tokens": compressed_tool_tokens,
        "prefix_kv_length": cache_length,
        "final_kv_length": cache_length + len(prompt_ids),
        "actual_compression_ratio": round(hybrid_ratio, 4),
        "combined_full_doc_actual_compression_ratio": (
            round(combined_full_doc_tokens / compressed_tool_tokens, 4)
            if compressed_tool_tokens else 0.0
        ),
        "c2kv2_actual_compression_ratio": (
            round(sum(c2kv2_ratios) / len(c2kv2_ratios), 4) if c2kv2_ratios else 0.0
        ),
        "c2kv4_actual_compression_ratio": (
            round(sum(c2kv4_ratios) / len(c2kv4_ratios), 4) if c2kv4_ratios else 0.0
        ),
        "rest_actual_compression_ratio": (
            round((c2kv2_doc_tokens + c2kv4_doc_tokens) / (c2kv2_gist_tokens + c2kv4_gist_tokens), 4)
            if (c2kv2_gist_tokens + c2kv4_gist_tokens) else 0.0
        ),
        "prompt_tokens": len(prompt_ids),
        "target_tokens": len(tokenizer.encode(target, add_special_tokens=False)),
        "generated_tokens": generated_tokens,
        "latency_sec": round(generate_sec, 4),
        "system_prefill_sec": round(system_prefill_sec, 4),
        "top_full_prefill_sec": round(full_prefill_sec, 4),
        "tool_compress_sec": round(c2kv2_compress_sec + c2kv4_compress_sec, 4),
        "c2kv2_compress_sec": round(c2kv2_compress_sec, 4),
        "c2kv4_compress_sec": round(c2kv4_compress_sec, 4),
        "full_prefill_sec": round(full_prefill_sec, 4),
        "blend_sec": 0.0,
        "generate_sec": round(generate_sec, 4),
        "ttft_sec": round(online_ttft_sec, 4),
        "online_ttft_sec": round(online_ttft_sec, 4),
        "cached_ttft_sec": round(cached_ttft_sec, 4),
        "tool_only_cached_ttft_sec": round(full_prefill_sec, 4),
        "tbt_sec": round(tbt_sec, 6),
        "total_sec": round(total_sec, 4),
        "cached_total_sec": round(cached_ttft_sec + generate_sec, 4),
        "target_tool_name": target_tool,
        "prediction_tool_name": pred_tool,
        "tool_name_match": target_tool is not None and target_tool == pred_tool,
        "has_tool_call": prediction_has_tool_call,
        "target_has_tool_call": target_has_tool_call,
        "response_type_match": target_has_tool_call == prediction_has_tool_call,
        "exact_match": _normalize_text(prediction) == _normalize_text(target),
        "text_token_f1": round(_text_token_f1(target, prediction), 4),
        "rouge_l_f1": round(_rouge_l_f1(target, prediction), 4),
        "prediction": prediction,
        "target": target,
    }
    return _add_hybrid_debug_fields(
        row,
        args,
        full_definition=example.tool_definition,
        top_definition=top_definition,
        rest_definition=rest_definition,
        numerator_tokens=tool_original_tokens,
        denominator_tokens=compressed_tool_tokens,
        top_tokens=full_tool_tokens,
        rest_original_tokens=c2kv2_doc_tokens + c2kv4_doc_tokens,
        rest_compressed_tokens=c2kv2_gist_tokens + c2kv4_gist_tokens,
    )


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
    if args.router_strategy in STABLE_PLUS_STRATEGIES:
        return _generate_one_stable_plus_hybrid(model, tokenizer, example, args, top_k, ratio)

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
    vote_debug: Optional[Dict[str, Any]] = None
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
    elif args.router_strategy in {"vote_all", "stable_vote", "conservative_vote"}:
        attention_ranked, attention_tool_scores, attention_router_sec, head_rankings = _rank_tools_by_attention(
            model,
            tokenizer,
            example,
            args,
            tools,
            ratio,
        )
        voted_ranked, vote_debug = _rank_lexical_pool_by_head_rrf(
            lexical_ranked,
            head_rankings,
            top_k=top_k,
            pool_size=args.attention_router_lexical_pool,
            rrf_k=args.attention_rrf_k,
            stable=args.router_strategy in {"stable_vote", "conservative_vote"},
            stable_heads=_parse_stable_heads(args.attention_stable_heads),
            stable_head_count=args.attention_stable_head_count,
        )
        if args.router_strategy == "conservative_vote":
            final_ranked, conservative_debug = _conservative_vote_replacement(
                lexical_ranked,
                voted_ranked,
                top_k,
            )
            vote_debug["conservative_replacement"] = conservative_debug
            att_rerank_debug = conservative_debug
        else:
            final_ranked = voted_ranked
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
    lexical_hit_at_topk = target_lexical_rank is not None and target_lexical_rank <= top_k
    lexical_recall = {
        f"lexical_hit_at_{k}": bool(target_lexical_rank is not None and target_lexical_rank <= k)
        for k in (1, 3, 5, 10, 20)
    }
    promotion_gain = bool((not lexical_hit_at_topk) and router_hit)
    demotion_loss = bool(lexical_hit_at_topk and not router_hit)
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
            document_mode=args.tool_document_eval_mode,
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
    prediction_has_tool_call = "<tool_call>" in prediction or "Action:" in prediction
    target_has_tool_call = bool(target_tool) or "<tool_call>" in target or "Action:" in target
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
            if args.router_strategy in {
                "attention",
                "lex_attention",
                "att_rerank",
                "vote_all",
                "stable_vote",
                "conservative_vote",
            } else None
        ),
        "attention_cache_mode": (
            args.attention_router_cache_mode
            if args.router_strategy in {
                "attention",
                "lex_attention",
                "att_rerank",
                "vote_all",
                "stable_vote",
                "conservative_vote",
            } else None
        ),
        "attention_router_sec": round(attention_router_sec, 4),
        "attention_lexical_pool": (
            args.attention_router_lexical_pool
            if args.router_strategy in {
                "lex_attention",
                "vote_all",
                "stable_vote",
                "conservative_vote",
            } else None
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
            if args.router_strategy in {"att_rerank", "conservative_vote"} else None
        ),
        "att_rerank_debug": att_rerank_debug,
        "vote_debug": vote_debug,
        "promotion_gain": promotion_gain,
        "demotion_loss": demotion_loss,
        "lexical_hit_at_topk": lexical_hit_at_topk,
        "final_hit_at_topk": router_hit,
        **lexical_recall,
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
        "has_tool_call": prediction_has_tool_call,
        "target_has_tool_call": target_has_tool_call,
        "response_type_match": target_has_tool_call == prediction_has_tool_call,
        "exact_match": _normalize_text(prediction) == _normalize_text(target),
        "text_token_f1": round(_text_token_f1(target, prediction), 4),
        "rouge_l_f1": round(_rouge_l_f1(target, prediction), 4),
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
        router_hit_rows = [row for row in valid_rows if row.get("router_hit")]
        lexical_hit_rows = [row for row in valid_rows if row.get("lexical_top3_hit")]
        attention_promotion_rows = [
            row for row in valid_rows
            if not row.get("lexical_top3_hit") and row.get("stable_candidate_hit")
        ]
        final_miss_rows = [row for row in valid_rows if row.get("final_recovery_hit") is False]
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
            "lexical_recall_at_1": (
                sum(1 for row in valid_rows if row.get("lexical_hit_at_1")) / len(valid_rows)
                if valid_rows and any("lexical_hit_at_1" in row for row in valid_rows) else 0.0
            ),
            "lexical_recall_at_3": (
                sum(1 for row in valid_rows if row.get("lexical_hit_at_3")) / len(valid_rows)
                if valid_rows and any("lexical_hit_at_3" in row for row in valid_rows) else 0.0
            ),
            "lexical_recall_at_5": (
                sum(1 for row in valid_rows if row.get("lexical_hit_at_5")) / len(valid_rows)
                if valid_rows and any("lexical_hit_at_5" in row for row in valid_rows) else 0.0
            ),
            "lexical_recall_at_10": (
                sum(1 for row in valid_rows if row.get("lexical_hit_at_10")) / len(valid_rows)
                if valid_rows and any("lexical_hit_at_10" in row for row in valid_rows) else 0.0
            ),
            "lexical_recall_at_20": (
                sum(1 for row in valid_rows if row.get("lexical_hit_at_20")) / len(valid_rows)
                if valid_rows and any("lexical_hit_at_20" in row for row in valid_rows) else 0.0
            ),
            "promotion_gain_count": sum(1 for row in valid_rows if row.get("promotion_gain")),
            "demotion_loss_count": sum(1 for row in valid_rows if row.get("demotion_loss")),
            "delta_hit_at_3": (
                (
                    sum(1 for row in valid_rows if row.get("promotion_gain"))
                    - sum(1 for row in valid_rows if row.get("demotion_loss"))
                ) / len(valid_rows)
                if valid_rows and any("promotion_gain" in row for row in valid_rows) else 0.0
            ),
            "hit_utilization": (
                sum(1 for row in router_hit_rows if row.get("tool_name_match")) / len(router_hit_rows)
                if router_hit_rows else 0.0
            ),
            "lexical_hit_utilization": (
                sum(1 for row in lexical_hit_rows if row.get("tool_name_match")) / len(lexical_hit_rows)
                if lexical_hit_rows else 0.0
            ),
            "attention_promotion_utilization": (
                sum(1 for row in attention_promotion_rows if row.get("tool_name_match")) / len(attention_promotion_rows)
                if attention_promotion_rows else 0.0
            ),
            "miss_recovery_rate": (
                sum(1 for row in final_miss_rows if row.get("tool_name_match")) / len(final_miss_rows)
                if final_miss_rows else 0.0
            ),
            "stable_candidate_hit_rate": (
                sum(1 for row in valid_rows if row.get("stable_candidate_hit")) / len(valid_rows)
                if valid_rows and any("stable_candidate_hit" in row for row in valid_rows) else 0.0
            ),
            "avg_stable_candidate_lexical_rank": (
                sum(float(row.get("stable_candidate_lexical_rank")) for row in valid_rows if row.get("stable_candidate_lexical_rank") is not None)
                / sum(1 for row in valid_rows if row.get("stable_candidate_lexical_rank") is not None)
                if valid_rows and any(row.get("stable_candidate_lexical_rank") is not None for row in valid_rows) else 0.0
            ),
            "recovery_group_results": {
                "lexical_top3_hit": _basic_metric_summary(lexical_hit_rows),
                "stable_candidate_promotion": _basic_metric_summary(attention_promotion_rows),
                "final_recovery_miss": _basic_metric_summary(final_miss_rows),
            },
            "exact_match": (
                sum(1 for row in valid_rows if row.get("exact_match")) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_text_token_f1": (
                sum(_row_text_token_f1(row) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_rouge_l_f1": (
                sum(_row_rouge_l_f1(row) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "response_type_accuracy": (
                sum(1 for row in valid_rows if _row_response_type_match(row)) / len(valid_rows)
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
            "avg_full_tool_tokens": (
                sum(row.get("full_tool_tokens", row.get("top_doc_tokens", 0)) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_full_schema_tool_tokens": (
                sum(row.get("full_schema_tool_tokens", row.get("full_tool_tokens", row.get("top_doc_tokens", 0))) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_stable_summary_full_tokens": (
                sum(row.get("stable_summary_full_tokens", 0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_tool_original_tokens": (
                sum(_row_tool_original_tokens(row) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_tool_kv_tokens": (
                sum(row.get("tool_kv_tokens", _row_compressed_tool_tokens(row)) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_combined_full_doc_tokens": (
                sum(row.get("combined_full_doc_tokens", row.get("doc_tokens", 0)) for row in valid_rows) / len(valid_rows)
                if valid_rows and any("combined_full_doc_tokens" in row for row in valid_rows) else 0.0
            ),
            "avg_combined_full_doc_actual_compression_ratio": (
                sum(row.get("combined_full_doc_actual_compression_ratio", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows and any("combined_full_doc_actual_compression_ratio" in row for row in valid_rows) else 0.0
            ),
            "avg_c2kv2_doc_tokens": (
                sum(row.get("c2kv2_doc_tokens", 0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_c2kv2_gist_tokens": (
                sum(row.get("c2kv2_gist_tokens", 0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_c2kv4_doc_tokens": (
                sum(row.get("c2kv4_doc_tokens", row.get("rest_doc_tokens", 0)) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_c2kv4_gist_tokens": (
                sum(row.get("c2kv4_gist_tokens", row.get("rest_gist_tokens", 0)) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_final_kv_length": (
                sum(row.get("final_kv_length", 0) for row in valid_rows) / len(valid_rows)
                if valid_rows and any("final_kv_length" in row for row in valid_rows) else 0.0
            ),
            "num_no_rest_tools": len(no_rest_rows),
            "no_rest_tool_rate": (
                len(no_rest_rows) / len(valid_rows) if valid_rows else 0.0
            ),
            "avg_actual_compression_ratio": (
                sum(_row_actual_compression_ratio(row) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "token_weighted_actual_compression_ratio": (
                sum(_row_tool_original_tokens(row) for row in valid_rows) / compressed_tool_total
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
                document_mode=args.tool_document_eval_mode,
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
        "tool_document_eval_mode": args.tool_document_eval_mode,
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
        choices=[
            "lexical",
            "random",
            "attention",
            "lex_attention",
            "att_rerank",
            "vote_all",
            "stable_vote",
            "conservative_vote",
            "lex_top3_original_order",
            "lex_top3_plus_stable1_full",
            "lex_top3_plus_stable1_c2kv2",
            "lex_top3_plus_stable1_name_desc_full",
        ],
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
        choices=["mean", "sqrt_len", "sum", "top4_mean"],
        default="mean",
        help="How to aggregate query attention over each tool's KV span.",
    )
    parser.add_argument(
        "--attention_router_span_top_tokens",
        type=int,
        default=4,
        help="For top4_mean scoring, average this many highest query-to-tool-span attention entries.",
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
    parser.add_argument("--attention_rrf_k", type=float, default=60.0)
    parser.add_argument(
        "--attention_stable_heads",
        default="",
        help="Comma-separated layer:head list for stable-vote routing. Defaults to the smoke-test top-16 heads.",
    )
    parser.add_argument("--attention_stable_head_count", type=int, default=16)
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
    parser.add_argument(
        "--tool_document_eval_mode",
        choices=["full", "per_tool"],
        default="full",
        help=(
            "How to build C2KV eval documents from tool schemas. full keeps one "
            "combined tool-definition document; per_tool makes each tool schema an "
            "independent C2KV document."
        ),
    )
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
