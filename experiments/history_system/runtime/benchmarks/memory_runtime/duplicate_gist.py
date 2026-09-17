"""Prune duplicate gist after allocation while preserving the main gist reservation."""
import copy


def prune_duplicate_gist(messages, counts, tools, token_counter):
    from memory_runtime.always_compress import coverage_accounting, ratio_accounting
    out, updated = copy.deepcopy(messages), copy.deepcopy(counts)
    meta = updated["memory_runtime"]
    base = counts["memory_runtime"]
    raw = [m for m in messages if not m.get("c2kv_key_hash")]
    if token_counter(raw, tools) != base["total_raw_prompt_tokens"]:
        raise ValueError("Original raw tokenizer accounting differs")
    blocks, records = base["block_refs"], updated["compressed_records"]
    if len(blocks) != len(records):
        raise ValueError("Block-to-carrier mapping is incomplete")
    raw_sources = set(base["selected_source_indices"])
    duplicate = [i for i,b in enumerate(blocks)
                 if b["source_indices"] and set(b["source_indices"]) <= raw_sources]
    removed = set(duplicate)
    preserved_first = bool(blocks) and len(removed) == len(blocks)
    if preserved_first:
        removed.remove(0)
    remove_positions = set()
    for i, (b, r) in enumerate(zip(blocks, records)):
        position = r["out_index"]
        if (r["record"]["key_hash"] != b["key_hash"]
                or messages[position].get("c2kv_key_hash") != b["key_hash"]):
            raise ValueError("Captured gist carrier identity differs")
        if i in removed:
            remove_positions.add(position)
    out = [m for i,m in enumerate(out) if i not in remove_positions]
    def remap(i):
        return i - sum(position < i for position in remove_positions)
    kept = [dict(r, out_index=remap(r["out_index"])) for i,r in enumerate(records) if i not in removed]
    retained_blocks = [b for i,b in enumerate(blocks) if i not in removed]
    gist = sum(b["gist_tokens"] for b in retained_blocks)
    removed_tokens = base["gist_tokens"] - gist
    active = base["active_history_bytes"] - removed_tokens * base["bytes_per_kv_token"]
    coverage = coverage_accounting(
        eligible_sources=frozenset(base["source_coverage"]["eligible_source_indices"]),
        raw_sources=raw_sources, retained_blocks=retained_blocks,
        packing_fragments=updated.get("history_packing_fragments") if records else [])
    if coverage["unrepresented_source_indices"] != base["source_coverage"]["unrepresented_source_indices"]:
        raise ValueError("Removing duplicate gist changed source coverage")
    original_tokens = sum(r["record"]["original_seq_len"] for r in kept)
    updated.update(compressed_records=kept, gist_tokens=gist, original_tokens=original_tokens,
        n_gist_messages=len(kept), compressed=len(kept), n_docs=len(kept),
        dropped_docs=updated["dropped_docs"]+len(removed),
        current_start_out_index=remap(updated["current_start_out_index"]))
    packed = updated.get("history_packed_original_tokens")
    if packed is not None:
        updated.update(history_dropped_original_tokens=packed-original_tokens,
                       history_retained_fraction=original_tokens/packed if packed else None)
    meta.update(block_refs=retained_blocks, gist_tokens=gist, active_history_bytes=active,
                source_coverage=coverage, byte_geometry_verified_by_backend=False)
    workspace = meta["pre_generation_workspace"]
    workspace["native_workspace_out_indices"] = [remap(i) for i in workspace["native_workspace_out_indices"]]
    reservation = meta["gist_reservation"]
    reservation["satisfied"] = not reservation["required"] or gist > 0
    ratio = base["compression_ratio"]
    meta["compression_ratio"] = ratio_accounting(
        {"active_history_bytes":ratio["full_history_bytes"],
         "common_raw_prompt_tokens":base["common_raw_prompt_tokens"]}, meta)
    meta["compression_ratio"]["includes_coverage_loss"] = bool(coverage["unrepresented_source_indices"])
    meta["gist_pruning"] = {
        "version":"duplicate-gist-reservation-v1",
        "eligible_duplicate_block_indices":duplicate,
        "removed_block_indices":sorted(removed), "removed_gist_tokens":removed_tokens,
        "removed_kv_equivalent_bytes":removed_tokens*base["bytes_per_kv_token"],
        "original_active_history_bytes":base["active_history_bytes"],
        "raw_refill":False, "raw_messages_unchanged":[m for m in out if not m.get("c2kv_key_hash")] == raw,
        "source_coverage_unchanged":True,
        "preserved_first_block_for_reservation":preserved_first,
        "minimum_active_blocks_when_eligible":1,
        "scope":"Post-allocation pruning; source selection and raw content are unchanged."}
    if ([m for m in out if not m.get("c2kv_key_hash")] != raw
            or token_counter(raw, tools) != meta["total_raw_prompt_tokens"]
            or active != (meta["raw_history_tokens"]+gist)*meta["bytes_per_kv_token"]
            or active > min(meta["history_budget_bytes"],meta["workspace_budget_bytes"])
            or (not meta["no_eligible_history"] and gist <= 0)):
        raise ValueError("Fixed raw input or byte accounting changed unexpectedly")
    return out, updated
