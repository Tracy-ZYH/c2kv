from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _display_mode(row: Dict[str, Any]) -> str:
    mode = str(row.get("mode"))
    hybrid_mode = row.get("hybrid_mode")
    router_strategy = row.get("router_strategy")
    top_k = row.get("top_k")
    top_schema_mode = row.get("top_schema_mode", "full")
    if mode == "hybrid" and hybrid_mode:
        if hybrid_mode == "hybrid" and router_strategy == "attention":
            attention_score_mode = row.get("attention_score_mode")
            attention_cache_mode = row.get("attention_cache_mode", "c2kv")
            prefix = "att_fullkv_hybrid" if attention_cache_mode == "full" else "att_hybrid"
            if attention_score_mode:
                return f"{prefix}_{attention_score_mode}"
            return prefix
        if hybrid_mode == "hybrid" and router_strategy == "lex_attention":
            attention_score_mode = row.get("attention_score_mode")
            attention_cache_mode = row.get("attention_cache_mode", "c2kv")
            prefix = "lex_att_fullkv_hybrid" if attention_cache_mode == "full" else "lex_att_hybrid"
            if attention_score_mode:
                return f"{prefix}_{attention_score_mode}"
            return prefix
        if hybrid_mode == "hybrid" and router_strategy == "att_rerank":
            attention_score_mode = row.get("attention_score_mode")
            attention_cache_mode = row.get("attention_cache_mode", "c2kv")
            prefix = "hybrid_fullkv_att_rerank" if attention_cache_mode == "full" else "hybrid_att_rerank"
            if attention_score_mode:
                return f"{prefix}_{attention_score_mode}"
            return prefix
        if hybrid_mode == "hybrid" and router_strategy in {"vote_all", "stable_vote", "conservative_vote"}:
            attention_score_mode = row.get("attention_score_mode")
            attention_cache_mode = row.get("attention_cache_mode", "c2kv")
            prefix = router_strategy
            if attention_cache_mode == "full":
                prefix = f"fullkv_{prefix}"
            if attention_score_mode:
                return f"{prefix}_{attention_score_mode}"
            return prefix
        if hybrid_mode == "hybrid" and router_strategy in {
            "lex_top3_original_order",
            "lex_top3_plus_stable1_full",
            "lex_top3_plus_stable1_c2kv2",
            "lex_top3_plus_stable1_name_desc_full",
        }:
            return str(router_strategy)
        if hybrid_mode == "hybrid" and router_strategy == "random":
            return "random_hybrid"
        if hybrid_mode == "hybrid" and router_strategy == "bm25":
            if top_k is not None and top_k != 3:
                return f"bm25_hybrid_top{top_k}"
            return "bm25_hybrid"
        if hybrid_mode == "hybrid":
            if top_schema_mode == "compact":
                return "c2kv_hybrid_compact"
            if top_k is not None and top_k != 3:
                return f"c2kv_hybrid_top{top_k}"
            return "c2kv_hybrid"
        return str(hybrid_mode)
    return mode


def _group_name(row: Dict[str, Any]) -> str:
    return f"{_display_mode(row)}@{row.get('ratio')}"


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def _has_tool_call(text: str) -> bool:
    return "<tool_call>" in (text or "") or "Action:" in (text or "")


def _text_tokens(text: str) -> List[str]:
    return re.findall(r"\w+", _normalize_text(text))


def _target_has_tool_call(row: Dict[str, Any]) -> bool:
    if "target_has_tool_call" in row:
        return bool(row.get("target_has_tool_call"))
    return bool(row.get("target_tool_name")) or _has_tool_call(row.get("target", ""))


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


def _lcs_length(left: List[str], right: List[str]) -> int:
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


def _row_compressed_tool_tokens(row: Dict[str, Any]) -> float:
    if "compressed_tool_tokens" in row:
        return float(row.get("compressed_tool_tokens", 0) or 0)
    if "top_doc_tokens" in row or "rest_gist_tokens" in row:
        return float(row.get("top_doc_tokens", 0) or 0) + float(row.get("rest_gist_tokens", 0) or 0)
    if "gist_tokens" in row:
        return float(row.get("gist_tokens", 0) or 0)
    if "kept_tool_tokens" in row:
        return float(row.get("kept_tool_tokens", 0) or 0)
    ratio = float(row.get("actual_compression_ratio", 0.0) or 0.0)
    doc_tokens = float(row.get("doc_tokens", 0) or 0)
    return doc_tokens / ratio if ratio > 0 and doc_tokens > 0 else 0.0


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
    return bool(row.get("response_type_match", _target_has_tool_call(row) == bool(row.get("has_tool_call"))))


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


def _summarize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries = []
    keys = sorted({_group_key(row) for row in rows})
    for mode, ratio in keys:
        group = [row for row in rows if _group_key(row) == (mode, ratio)]
        valid_rows = [row for row in group if not row.get("skipped")]
        skip_reasons = Counter(row.get("skip_reason", "unknown") for row in group if row.get("skipped"))
        generated_total = sum(row.get("generated_tokens", 0) for row in valid_rows)
        called_rows = [row for row in valid_rows if row.get("has_tool_call")]
        tool_targets = [row for row in valid_rows if _target_has_tool_call(row)]
        non_tool_targets = [row for row in valid_rows if not _target_has_tool_call(row)]
        compressed_tool_total = sum(_row_compressed_tool_tokens(row) for row in valid_rows)
        no_rest_rows = [row for row in valid_rows if row.get("num_rest_tools") == 0]
        lexical_rank_rows = [row for row in valid_rows if row.get("target_lexical_rank") is not None]
        attention_rank_rows = [row for row in valid_rows if row.get("target_attention_rank") is not None]
        final_rank_rows = [row for row in valid_rows if row.get("target_final_rank") is not None]
        att_rerank_rows = [row for row in valid_rows if row.get("att_rerank_replaced") is not None]
        router_hit_rows = [row for row in valid_rows if row.get("router_hit")]
        lexical_hit_rows = [row for row in valid_rows if row.get("lexical_top3_hit")]
        attention_promotion_rows = [
            row for row in valid_rows
            if not row.get("lexical_top3_hit") and row.get("stable_candidate_hit")
        ]
        final_miss_rows = [row for row in valid_rows if row.get("final_recovery_hit") is False]
        summaries.append({
            "mode": mode,
            "ratio": ratio,
            "num_examples": len(group),
            "num_valid": len(valid_rows),
            "num_skipped": len(group) - len(valid_rows),
            "skip_reasons": dict(skip_reasons),
            "num_tool_targets": len(tool_targets),
            "num_non_tool_targets": len(non_tool_targets),
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
                sum(
                    1 for row in valid_rows
                    if row.get("response_type_match", _target_has_tool_call(row) == bool(row.get("has_tool_call")))
                ) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "target_tool_call_rate": (
                len(tool_targets) / len(valid_rows) if valid_rows else 0.0
            ),
            "tool_name_accuracy": (
                sum(1 for row in valid_rows if row.get("tool_name_match")) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "tool_name_accuracy_on_tool_targets": (
                sum(1 for row in tool_targets if row.get("tool_name_match")) / len(tool_targets)
                if tool_targets else 0.0
            ),
            "tool_call_rate": (
                len(called_rows) / len(valid_rows) if valid_rows else 0.0
            ),
            "tool_call_rate_on_tool_targets": (
                sum(1 for row in tool_targets if row.get("has_tool_call")) / len(tool_targets)
                if tool_targets else 0.0
            ),
            "call_accuracy": (
                sum(1 for row in called_rows if row.get("tool_name_match")) / len(called_rows)
                if called_rows else 0.0
            ),
            "non_tool_exact_match": (
                sum(1 for row in non_tool_targets if row.get("exact_match")) / len(non_tool_targets)
                if non_tool_targets else 0.0
            ),
            "non_tool_text_token_f1": (
                sum(_row_text_token_f1(row) for row in non_tool_targets) / len(non_tool_targets)
                if non_tool_targets else 0.0
            ),
            "non_tool_rouge_l_f1": (
                sum(_row_rouge_l_f1(row) for row in non_tool_targets) / len(non_tool_targets)
                if non_tool_targets else 0.0
            ),
            "non_tool_false_tool_call_rate": (
                sum(1 for row in non_tool_targets if row.get("has_tool_call")) / len(non_tool_targets)
                if non_tool_targets else 0.0
            ),
            "router_hit_rate": (
                sum(1 for row in valid_rows if row.get("router_hit")) / len(valid_rows)
                if any("router_hit" in row for row in valid_rows) else 0.0
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
                sum(
                    float(row.get("stable_candidate_lexical_rank"))
                    for row in valid_rows
                    if row.get("stable_candidate_lexical_rank") is not None
                )
                / sum(1 for row in valid_rows if row.get("stable_candidate_lexical_rank") is not None)
                if valid_rows and any(row.get("stable_candidate_lexical_rank") is not None for row in valid_rows)
                else 0.0
            ),
            "recovery_group_results": {
                "lexical_top3_hit": _basic_metric_summary(lexical_hit_rows),
                "stable_candidate_promotion": _basic_metric_summary(attention_promotion_rows),
                "final_recovery_miss": _basic_metric_summary(final_miss_rows),
            },
            "avg_target_lexical_rank": (
                sum(float(row.get("target_lexical_rank")) for row in lexical_rank_rows) / len(lexical_rank_rows)
                if lexical_rank_rows else 0.0
            ),
            "avg_target_attention_rank": (
                sum(float(row.get("target_attention_rank")) for row in attention_rank_rows) / len(attention_rank_rows)
                if attention_rank_rows else 0.0
            ),
            "avg_target_final_rank": (
                sum(float(row.get("target_final_rank")) for row in final_rank_rows) / len(final_rank_rows)
                if final_rank_rows else 0.0
            ),
            "att_rerank_replacement_rate": (
                sum(1 for row in att_rerank_rows if row.get("att_rerank_replaced")) / len(att_rerank_rows)
                if att_rerank_rows else 0.0
            ),
            "avg_doc_tokens": (
                sum(_row_tool_original_tokens(row) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_prompt_tokens": (
                sum(row.get("prompt_tokens", 0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_generated_tokens": (
                generated_total / len(valid_rows) if valid_rows else 0.0
            ),
            "avg_actual_compression_ratio": (
                sum(_row_actual_compression_ratio(row) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "token_weighted_actual_compression_ratio": (
                sum(_row_tool_original_tokens(row) for row in valid_rows) / compressed_tool_total
                if compressed_tool_total else 0.0
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
            "avg_num_tools": (
                sum(row.get("num_tools", 0) for row in valid_rows) / len(valid_rows)
                if valid_rows and any("num_tools" in row for row in valid_rows) else 0.0
            ),
            "num_no_rest_tools": len(no_rest_rows),
            "no_rest_tool_rate": (
                len(no_rest_rows) / len(valid_rows)
                if valid_rows and any("num_rest_tools" in row for row in valid_rows) else 0.0
            ),
            "avg_system_prefill_sec": (
                sum(row.get("system_prefill_sec", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_tool_compress_sec": (
                sum(row.get("tool_compress_sec", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_full_prefill_sec": (
                sum(row.get("full_prefill_sec", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_blend_sec": (
                sum(row.get("blend_sec", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "avg_ttft_sec": (
                sum(row.get("ttft_sec", 0.0) for row in valid_rows) / len(valid_rows)
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
            "avg_tbt_sec": (
                sum(row.get("tbt_sec", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "token_weighted_tbt_sec": (
                sum(row.get("generate_sec", 0.0) for row in valid_rows) / generated_total
                if generated_total else 0.0
            ),
            "avg_total_sec": (
                sum(row.get("total_sec", 0.0) for row in valid_rows) / len(valid_rows)
                if valid_rows else 0.0
            ),
        })
    return summaries


def _group_key(row: Dict[str, Any]) -> tuple[Any, Any]:
    return _display_mode(row), row.get("ratio")


def _sample_key(row: Dict[str, Any]) -> Optional[str]:
    qid = row.get("qid")
    if qid is None:
        return None
    session_id = row.get("session_id")
    return f"{session_id}\t{qid}" if session_id is not None else str(qid)


def _common_subset(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = sorted({_group_key(row) for row in rows})
    valid_samples_by_key: Dict[str, set[str]] = {}
    duplicate_valid_rows_by_key: Dict[str, int] = {}
    for mode, ratio in keys:
        group = [row for row in rows if _group_key(row) == (mode, ratio)]
        valid_sample_counts = Counter(
            key
            for row in group
            if not row.get("skipped")
            for key in [_sample_key(row)]
            if key is not None
        )
        sample_keys = {
            key for key, count in valid_sample_counts.items() if count > 0
        }
        group_name = f"{mode}@{ratio}"
        valid_samples_by_key[group_name] = sample_keys
        duplicate_valid_rows_by_key[group_name] = sum(
            count - 1 for count in valid_sample_counts.values() if count > 1
        )
    if not valid_samples_by_key:
        common_samples: set[str] = set()
    else:
        common_samples = set.intersection(*valid_samples_by_key.values())
    return {
        "num_groups": len(valid_samples_by_key),
        "num_common_samples": len(common_samples),
        "valid_samples_by_group": {
            key: len(value) for key, value in valid_samples_by_key.items()
        },
        "duplicate_valid_rows_by_group": duplicate_valid_rows_by_key,
        "common_sample_keys": sorted(common_samples),
    }


def _dedupe_common_rows(rows: List[Dict[str, Any]], common_samples: set[str]) -> List[Dict[str, Any]]:
    selected: Dict[tuple[Any, Any, str], Dict[str, Any]] = {}
    for row in rows:
        if row.get("skipped"):
            continue
        sample_key = _sample_key(row)
        if sample_key not in common_samples:
            continue
        key = (*_group_key(row), sample_key)
        if key not in selected:
            selected[key] = row
    return list(selected.values())


def _summarize_common_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    common = _common_subset(rows)
    common_samples = set(common["common_sample_keys"])
    if not common_samples:
        return []
    common_rows = _dedupe_common_rows(rows, common_samples)
    return _summarize_rows(common_rows)


def _load_common_sample_keys(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if "common_subset" in payload:
            return set(payload["common_subset"].get("common_sample_keys", []))
        if "common_sample_keys" in payload:
            return set(payload.get("common_sample_keys", []))
    if isinstance(payload, list):
        return {str(item) for item in payload}
    raise ValueError(
        f"Could not find common sample keys in {path}. Expected a merge summary JSON "
        "with common_subset.common_sample_keys."
    )


def _fixed_subset(rows: List[Dict[str, Any]], sample_keys: set[str], source: Optional[str] = None) -> Dict[str, Any]:
    keys = sorted({_group_key(row) for row in rows})
    valid_samples_by_key: Dict[str, set[str]] = {}
    duplicate_valid_rows_by_key: Dict[str, int] = {}
    for mode, ratio in keys:
        group = [row for row in rows if _group_key(row) == (mode, ratio)]
        valid_sample_counts = Counter(
            key
            for row in group
            if not row.get("skipped")
            for key in [_sample_key(row)]
            if key in sample_keys
        )
        group_name = f"{mode}@{ratio}"
        valid_samples_by_key[group_name] = set(valid_sample_counts)
        duplicate_valid_rows_by_key[group_name] = sum(
            count - 1 for count in valid_sample_counts.values() if count > 1
        )
    subset = {
        "num_groups": len(valid_samples_by_key),
        "num_common_samples": len(sample_keys),
        "valid_samples_by_group": {
            key: len(value) for key, value in valid_samples_by_key.items()
        },
        "missing_valid_samples_by_group": {
            key: len(sample_keys - value) for key, value in valid_samples_by_key.items()
        },
        "duplicate_valid_rows_by_group": duplicate_valid_rows_by_key,
        "common_sample_keys": sorted(sample_keys),
    }
    if source is not None:
        subset["source"] = source
    return subset


def _summarize_fixed_subset_rows(rows: List[Dict[str, Any]], sample_keys: set[str]) -> List[Dict[str, Any]]:
    if not sample_keys:
        return []
    common_rows = _dedupe_common_rows(rows, sample_keys)
    return _summarize_rows(common_rows)


def _common_fairness_check(common_results: List[Dict[str, Any]], common_subset: Dict[str, Any]) -> Dict[str, Any]:
    expected = common_subset.get("num_common_samples", 0)
    counts = {
        f"{item.get('mode')}@{item.get('ratio')}": item.get("num_valid", 0)
        for item in common_results
    }
    return {
        "expected_num_valid_per_group": expected,
        "num_valid_by_group": counts,
        "is_fair_common_subset": bool(counts) and all(value == expected for value in counts.values()),
    }


def _compact_common_subset(payload: Dict[str, Any], include_keys: bool) -> Dict[str, Any]:
    if include_keys or "common_sample_keys" not in payload:
        return payload
    compact = dict(payload)
    compact["common_sample_keys_omitted"] = True
    compact["num_common_sample_keys_omitted"] = len(payload.get("common_sample_keys") or [])
    compact.pop("common_sample_keys", None)
    return compact


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge agent tool-definition reuse baseline eval shards.")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--input_files", nargs="+", required=True)
    parser.add_argument("--model")
    parser.add_argument("--base_model")
    parser.add_argument("--reuse_model")
    parser.add_argument("--dataset_path")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--tool_document_eval_mode", default="full")
    parser.add_argument("--modes")
    parser.add_argument("--ratios")
    parser.add_argument("--cacheblend_recompute_ratio", type=float, default=0.15)
    parser.add_argument(
        "--common_subset_file",
        help=(
            "Optional previous summary JSON. When set, common_subset_results are "
            "computed on that file's common_subset.common_sample_keys instead of "
            "recomputing the intersection across all current groups."
        ),
    )
    parser.add_argument(
        "--include_common_sample_keys",
        action="store_true",
        help="Include full common_sample_keys lists in the summary JSON. Hidden by default because they are long.",
    )
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    missing_files = []
    input_file_rows: Dict[str, int] = {}
    input_file_groups: Dict[str, Dict[str, int]] = {}
    for input_file in args.input_files:
        path = Path(input_file)
        if not path.exists():
            missing_files.append(str(path))
            continue
        file_rows = _read_jsonl(path)
        rows.extend(file_rows)
        input_file_rows[str(path)] = len(file_rows)
        input_file_groups[str(path)] = dict(Counter(_group_name(row) for row in file_rows))

    output_path = Path(args.output_file)
    _write_jsonl(output_path, rows)
    computed_common_subset = _common_subset(rows)
    if args.common_subset_file:
        reference_path = Path(args.common_subset_file)
        common_sample_keys = _load_common_sample_keys(reference_path)
        common_subset = _fixed_subset(rows, common_sample_keys, str(reference_path))
        common_subset_results = _summarize_fixed_subset_rows(rows, common_sample_keys)
    else:
        common_subset = computed_common_subset
        common_subset_results = _summarize_common_rows(rows)
    computed_common_subset_for_output = _compact_common_subset(
        computed_common_subset,
        args.include_common_sample_keys,
    )
    common_subset_for_output = _compact_common_subset(
        common_subset,
        args.include_common_sample_keys,
    )
    summary = {
        "model": args.model,
        "base_model": args.base_model,
        "reuse_model": args.reuse_model,
        "dataset_path": args.dataset_path,
        "split": args.split,
        "tool_document_eval_mode": args.tool_document_eval_mode,
        "modes": [item.strip() for item in (args.modes or "").split(",") if item.strip()],
        "ratios": [item.strip() for item in (args.ratios or "").split(",") if item.strip()],
        "num_rows": len(rows),
        "missing_files": missing_files,
        "input_file_rows": input_file_rows,
        "input_file_groups": input_file_groups,
        "computed_common_subset": computed_common_subset_for_output,
        "notes": {
            "epic_leading32": "PyTorch selective recompute with recompute_type=leading-32.",
            "cacheblend_vdiff": f"PyTorch value-difference selective recompute with recompute_type=vdiff-{args.cacheblend_recompute_ratio}; not the vLLM+LMCache expr_cacheblend.py path.",
            "snapkv_reuse": "Uses reuse_pipeline SnapKV compression, currently hard-coded to roughly 4x in compress_kv.",
            "epic_leading32_snapkv": "EPIC leading-32 selective recompute on top of SnapKV-compressed document KV.",
            "cacheblend_vdiff_snapkv": f"Value-difference selective recompute on top of SnapKV-compressed document KV with recompute_type=vdiff-{args.cacheblend_recompute_ratio}.",
            "snapkv_hybrid": "Hybrid top-k full tool schemas plus SnapKV-compressed rest tool schemas.",
            "epic_leading32_snapkv_hybrid": "Hybrid top-k full tool schemas plus EPIC leading-32 selective recompute on SnapKV-compressed rest schemas.",
            "cacheblend_vdiff_snapkv_hybrid": "Hybrid top-k full tool schemas plus value-difference selective recompute on SnapKV-compressed rest schemas.",
            "c2kv_aug_hybrid": "All tool schemas C2KV-compressed plus an extra full top-k tool-schema prefix.",
            "snapkv_aug_hybrid": "All tool schemas SnapKV-compressed plus an extra full top-k tool-schema prefix.",
            "epic_leading32_snapkv_aug_hybrid": "All tool schemas SnapKV-compressed with EPIC leading-32 selective recompute plus an extra full top-k tool-schema prefix.",
            "cacheblend_vdiff_snapkv_aug_hybrid": "All tool schemas SnapKV-compressed with value-difference selective recompute plus an extra full top-k tool-schema prefix.",
            "common_subset_results": (
                "Metrics recomputed on the internal common sample keys. If "
                "common_subset_file is set, these qids come from that reference "
                "summary; otherwise they are valid for every present mode/ratio group. "
                "Full common_sample_keys are omitted from output unless "
                "--include_common_sample_keys is set."
            ),
        },
        "results": _summarize_rows(rows),
        "common_subset": common_subset_for_output,
        "common_subset_results": common_subset_results,
        "common_subset_fairness_check": _common_fairness_check(common_subset_results, common_subset),
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
