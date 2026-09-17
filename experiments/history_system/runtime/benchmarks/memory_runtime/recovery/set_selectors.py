"""Mutually exclusive C0--C5 implementations of one evidence-set interface."""
from __future__ import annotations

import math

from .set_protocol import choose_parameter_source
from .set_retrieval import enrich_candidates


def public_candidates(candidates):
    return [{key: value for key, value in row.items() if key != "unit"} for row in candidates]


def select_evidence_set(
    context,
    candidates,
    actions,
    config,
    *,
    models,
    tokenizer,
    trained=None,
    legacy_gate=None,
):
    selector = config.get("set_selector", "candidate_rule")
    receipt = {"schema": "recovery-set-selector-v1", "selector": selector,
        "n_presented_to_selector": len(candidates), "legal_action_count": len(actions),
        "selected_ids": [], "action_id": 0, "available": True, "reason": "empty_set",
        "score": 0.0, "score_semantics": None}
    if len(actions) <= 1:
        receipt["reason"] = "no_feasible_evidence_sets"
        return (), receipt
    chosen, score = (), 0.0
    if selector == "candidate_rule":
        chosen = next((tuple(action) for action in actions if len(action) == 1), ())
        receipt["score_semantics"] = "first_feasible_ranked_single"
    elif selector == "legacy_prefill":
        if not isinstance(legacy_gate, dict) or legacy_gate.get("type") != "prefill_linear_head":
            raise ValueError("legacy_prefill requires a prefill_linear_head gate receipt")
        score = legacy_gate.get("score")
        receipt.update(
            available=score is not None,
            reason=legacy_gate["reason"],
            score_semantics="frozen_legacy_prefill_failure_risk",
        )
        if legacy_gate.get("triggered"):
            chosen = next((tuple(action) for action in actions if len(action) == 1), ())
    elif selector == "parameter_source":
        chosen, score = choose_parameter_source(context, candidates, actions)
        receipt["score_semantics"] = "unsupported_typed_parameter_coverage"
    elif selector == "reranker":
        enrich_candidates(
            context,
            candidates,
            models,
            rerank=True,
            semantic=trained is not None,
            overflow_policy=config.get("semantic_query_overflow_policy", "error"),
        )
        scores = {row["unit_id"]: row["reranker_score"] for row in candidates}
        if trained is not None:
            for row in candidates:
                prediction = trained.model.predict_candidate(context, row)
                if not prediction.available or prediction.score is None:
                    receipt.update(available=False, reason=prediction.reason,
                        missing_fields=list(prediction.missing_fields), score_semantics="t01_calibrated_relevance")
                    return (), receipt
                scores[row["unit_id"]] = prediction.score
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores.values()):
            raise ValueError("reranker scores must be finite values between zero and one")
        threshold = config.get("selector_threshold", 0.5)
        action_scores = [sum(scores[identifier] - threshold for identifier in action) for action in actions]
        best = min(range(len(actions)), key=lambda index: (-action_scores[index], len(actions[index]), index))
        chosen, score = tuple(actions[best]), action_scores[best]
        receipt.update(score_semantics="sum_calibrated_relevance_minus_threshold" if trained is not None else "sum_relevance_minus_threshold", threshold=threshold,
            candidate_scores=scores, action_scores=action_scores)
    elif selector == "local_llm":
        payload = {"q_t": {key: value for key, value in context.items()
                    if key not in {"prefill_hidden", "draft_logprobs"}},
            "d_t": {"kind": "STOP" if context["is_stop"] else "CALLS" if context["parse_ok"] else "MALFORMED",
                    "text": context["draft_text"], "tool_calls": context["draft_tool_calls"]},
            "candidates": public_candidates(candidates),
            "allowed_sets": [{"action_id": index, "selected_ids": list(action)} for index, action in enumerate(actions)]}
        action_id = models.choose_action(context_payload=payload, allowed_actions=actions)
        if type(action_id) is not int or not 0 <= action_id < len(actions):
            raise ValueError("local selector returned an action outside the legal action catalog")
        chosen = tuple(actions[action_id])
        receipt["score_semantics"] = "constrained_action_id"
    elif selector in {"risk", "gain_turn", "gain_task"}:
        if trained is None:
            raise ValueError("trained selector artifact was not loaded")
        if selector != "risk":
            enrich_candidates(
                context,
                candidates,
                models,
                rerank=True,
                semantic=True,
                overflow_policy=config.get("semantic_query_overflow_policy", "error"),
            )
        prediction = trained.select(context, candidates, legal_actions=actions,
            **({"threshold": config.get("selector_threshold", 0.5)} if selector == "risk"
               else {"tokenizer": tokenizer, "delta": config.get("gain_delta", 0.0)}))
        chosen, score = tuple(prediction.selected_ids), prediction.score
        receipt.update(available=prediction.available, reason=prediction.reason,
            score_semantics="current_turn_failure_risk" if selector == "risk" else "estimated_" + selector)
    else:
        raise ValueError(f"Unknown set selector: {selector}")
    chosen = tuple(chosen)
    if chosen not in actions:
        raise ValueError("selector returned a set outside the admissible action catalog")
    receipt.update(selected_ids=list(chosen), action_id=actions.index(chosen), score=score)
    if receipt["available"]:
        receipt["reason"] = "selected_legal_evidence_set" if chosen else "selector_empty_set"
    return chosen, receipt
