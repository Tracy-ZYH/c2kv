"""Archive retrieval and real, append-only feasibility before selection."""
from __future__ import annotations

from .evidence_units import build_catalog, unit_is_covered
from .selection import _catalog_records, _lexical_score, _sort_ranked, _cosine, _validate_embeddings
from .set_protocol import canonical_text, uncovered_units


def query_texts(context):
    task = context["goal"] + "\n" + canonical_text(context["last_action_observation"])
    kind = "STOP" if context["is_stop"] else "CALLS" if context["parse_ok"] else "MALFORMED"
    # Serializing the held calls retains parameter keys as well as typed values.
    draft = kind + "\n" + canonical_text(context["draft_tool_calls"]) + "\n" + context["draft_text"]
    return task, draft


def retrieve_archive(catalog, context, config, models=None):
    records = _catalog_records(list(catalog))
    task, draft = query_texts(context)
    route_limit = config.get("retrieval_route_limit", 16)
    limit = config.get("retrieval_limit", 24)
    receipt = {"query_mode": config["Q"], "routes": {}, "retrieved_ids": []}
    if not records:
        return [], receipt
    routes = []
    for name, query in (("task_lexical", task), ("draft_lexical", draft)):
        rows = []
        for record in records:
            score, components = _lexical_score(query, record)
            if score > 0:
                rows.append({**record, "score": score, "components": components})
        route = _sort_ranked(rows)[:route_limit]
        routes.append(route)
        receipt["routes"][name] = [row["unit_id"] for row in route]
    if config["Q"] == "lexical":
        rows = []
        for record in records:
            score, components = _lexical_score(task + "\n" + draft, record)
            if score > 0:
                rows.append({**record, "score": score, "components": components})
        ranked = _sort_ranked(rows)[:limit]
        receipt["routes"] = {"combined_lexical": [row["unit_id"] for row in ranked]}
    else:
        if models is None:
            raise ValueError("archive_rrf requires a local embedding model or injected local backend")
        embed_queries = getattr(models, "embed_retrieval_queries", None)
        if embed_queries is None:
            raise TypeError(
                "archive_rrf requires a local backend with embed_retrieval_queries"
            )
        query_vectors, query_limits = embed_queries(
            task=task,
            draft=draft,
            overflow_policy=config.get("semantic_query_overflow_policy", "error"),
        )
        queries = _validate_embeddings(query_vectors, 3)
        receipt["semantic_query_limits"] = query_limits
        docs = _validate_embeddings(models.embed(texts=[record["text"] for record in records],
            purpose="document", config={}), len(records))
        similarities = {record["unit_id"]: {
            "task_similarity": _cosine(queries[0], vector),
            "draft_similarity": _cosine(queries[1], vector),
            "combined_similarity": _cosine(queries[2], vector),
        } for record, vector in zip(records, docs, strict=True)}
        semantic = _sort_ranked([{**record, "score": similarities[record["unit_id"]]["combined_similarity"],
            "components": {}} for record in records])[:route_limit]
        routes.append(semantic)
        receipt["routes"]["semantic"] = [row["unit_id"] for row in semantic]
        fused = {}
        k = config.get("rrf_k", 60)
        for name, route in zip(("task_lexical", "draft_lexical", "semantic"), routes, strict=True):
            for rank, row in enumerate(route, 1):
                entry = fused.setdefault(row["unit_id"], {**row, "score": 0.0, "components": {}})
                entry["score"] += 1.0 / (k + rank)
                entry["components"][name + "_rank"] = rank
                entry.update(similarities[row["unit_id"]])
        ranked = _sort_ranked(list(fused.values()))[:limit]
    receipt["retrieved_ids"] = [row["unit_id"] for row in ranked]
    return ranked, receipt


def supply_candidates(prepared, tokenizer, config, context, admissible, models=None):
    """Scan past rejected candidates; optionally try fixed fine source windows."""
    store = prepared._store
    catalog = build_catalog(store, tokenizer, config["U"])
    ranked, retrieval = retrieve_archive(catalog, context, config, models)
    eligible = set(prepared.metadata["eligible_extraction"]["eligible_event_ids"])
    cancelled = set(prepared.metadata.get("revision_cancelled_event_ids") or ())
    raw = prepared.memory.raw_source_indices
    visible = list(prepared._gp_visible)
    fallback = build_catalog(store, tokenizer, config["fallback_unit"]) if config.get("fallback_unit") else []
    receipt = {"schema": "recovery-candidate-supply-v2", "stage": "evaluated", "n_archive_units": len(catalog),
        "n_retrieved": len(ranked), "n_after_source_check": 0, "n_feasible": 0,
        "n_presented_to_selector": 0, "n_distinct_sources": 0,
        "retrieval": retrieval, "rejections": [], "fallback_attempts": 0,
        "feasibility_scan_complete": True}
    candidates, seen = [], set()
    draft_names = {(call.get("function") or {}).get("name") for call in context["draft_tool_calls"]}
    draft_names.discard(None)
    def consider(unit, rank_row, parent_id=None):
        for piece in uncovered_units(unit, store, tokenizer, raw, visible):
            if piece.unit_id in seen:
                continue
            seen.add(piece.unit_id)
            fits, admission = admissible([piece])
            if not fits:
                receipt["rejections"].append({"unit_id": piece.unit_id,
                    "reason": "candidate_not_admitted", "admission": admission})
                continue
            row = {"unit": piece, **piece.to_receipt(), "text": piece.text,
                "score": rank_row["score"], "retrieval_components": rank_row.get("components", {}),
                "admission": admission, "feasible": True, "fallback_parent_id": parent_id}
            source_names = {(call.get("function") or {}).get("name")
                for message in store.event_messages(piece.event_id)
                for call in (message.to_dict().get("tool_calls") or [])}
            row["tool_name_match"] = len(source_names & draft_names) / len(draft_names) if draft_names else 0.0
            for name in ("task_similarity", "draft_similarity"):
                if name in rank_row and piece.unit_id == rank_row["unit_id"]:
                    row[name] = rank_row[name]
            candidates.append(row)
            receipt["n_feasible"] += 1
            if len(candidates) >= config["candidate_limit"]:
                break
    for index, row in enumerate(ranked):
        unit = row["unit"]
        if unit.event_id not in eligible or unit.event_id in cancelled:
            receipt["rejections"].append({"unit_id": unit.unit_id, "reason": "source_not_eligible"})
            continue
        receipt["n_after_source_check"] += 1
        pieces = uncovered_units(unit, store, tokenizer, raw, visible)
        if not pieces:
            receipt["rejections"].append({"unit_id": unit.unit_id, "reason": "already_exact_visible"})
            continue
        before = len(candidates)
        consider(unit, row)
        if len(candidates) == before and fallback:
            children = [child for child in fallback if child.event_id == unit.event_id
                and unit_is_covered(child, [unit])]
            # All subunits were created statically from this immutable source.
            for child in children:
                receipt["fallback_attempts"] += 1
                consider(child, row, unit.unit_id)
                if len(candidates) >= config["candidate_limit"]:
                    break
        if len(candidates) >= config["candidate_limit"]:
            receipt["feasibility_scan_complete"] = False
            break
    receipt["n_presented_to_selector"] = len(candidates)
    receipt["n_distinct_sources"] = len({row["event_id"] for row in candidates})
    receipt["presented_ids"] = [row["unit_id"] for row in candidates]
    return candidates, receipt


def enrich_candidates(
    context,
    candidates,
    models,
    *,
    rerank=False,
    semantic=False,
    overflow_policy="error",
):
    """Compute frozen scores for the actual visible candidate text, not parents."""
    if not candidates:
        return
    task, draft = query_texts(context)
    documents = [row["text"] for row in candidates]
    if rerank:
        rerank_candidates = getattr(models, "rerank_retrieval_candidates", None)
        if rerank_candidates is None:
            raise TypeError(
                "recovery reranking requires a local backend with "
                "rerank_retrieval_candidates"
            )
        scores = rerank_candidates(
            task=task,
            draft=draft,
            documents=documents,
            overflow_policy=overflow_policy,
        )
        if len(scores) != len(candidates):
            raise ValueError("reranker returned a different number of scores")
        for row, score in zip(candidates, scores, strict=True):
            row["reranker_score"] = float(score)
    if semantic:
        queries = _validate_embeddings(models.embed(texts=[task, draft], purpose="query", config={}), 2)
        vectors = _validate_embeddings(models.embed(texts=documents, purpose="document", config={}), len(documents))
        for row, vector in zip(candidates, vectors, strict=True):
            row["task_similarity"] = _cosine(queries[0], vector)
            row["draft_similarity"] = _cosine(queries[1], vector)
