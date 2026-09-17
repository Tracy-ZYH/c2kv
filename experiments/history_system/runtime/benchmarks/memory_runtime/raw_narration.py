"""Remove older gist-backed narration after the shared lexical allocation."""

import copy
import json
import re

from history_memory.events import EventStore
from .always_compress import coverage_accounting, ratio_accounting

NARRATION_VERSION = "older-gist-backed-narration-v1"
TOOL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def prune_older_narration(source, messages, counts, tools, token_counter):
    out, updated = copy.deepcopy(messages), copy.deepcopy(counts)
    base, meta = counts["memory_runtime"], updated["memory_runtime"]
    raw = [m for m in messages if not m.get("c2kv_key_hash")]
    if token_counter(raw, tools) != base["total_raw_prompt_tokens"]:
        raise ValueError("Original narration-view token accounting differs")
    store = EventStore.from_messages(base["task_id"], source)
    protected = {i for e in store.events if e.event_id in base["protected_event_ids"]
                 for i in e.source_indices}
    complete = {i for e in store.events if e.kind == "tool_event" and e.complete
                for i in e.source_indices}
    gist_backed = set(base["source_coverage"]["gist_fully_represented_source_indices"])
    workspace = base["pre_generation_workspace"]
    indices, positions = workspace["restored_source_indices"], workspace["native_workspace_out_indices"]
    if len(indices) != len(positions):
        raise ValueError("Native narration source mapping is incomplete")
    partial, skipped = set(), []
    for index, position in zip(indices, positions):
        original = source[index]
        if (index in protected or index not in complete or index not in gist_backed
                or original.get("role") != "assistant" or not original.get("tool_calls")):
            continue
        content = messages[position].get("content")
        if not isinstance(content, str):
            skipped.append({"source_index": index, "reason": "non_text_rendering"})
            continue
        matches = list(TOOL_BLOCK.finditer(content))
        expected = [{"name": c["function"]["name"],
                     "arguments": json.loads(c["function"]["arguments"])}
                    for c in original["tool_calls"]]
        try:
            actual = [json.loads(m.group(1)) for m in matches]
        except ValueError:
            actual = None
        # Unrecognized renderings remain intact; only the audited serialization
        # is eligible. Preserve each serialized call block byte for byte.
        if (actual != expected or not matches
                or TOOL_BLOCK.sub("", content[matches[0].start():]).strip()):
            skipped.append({"source_index": index, "reason": "unsupported_tool_serialization"})
            continue
        replacement = "Action:\n" + "\n".join(m.group(0) for m in matches)
        if replacement != content:
            out[position]["content"] = replacement
            partial.add(index)
    raw_tokens = token_counter([m for m in out if not m.get("c2kv_key_hash")], tools)
    saved = base["total_raw_prompt_tokens"] - raw_tokens
    if saved < 0:
        skipped.append({"source_indices": sorted(partial), "reason": "complete_input_token_increase"})
        out, partial, raw_tokens, saved = copy.deepcopy(messages), set(), base["total_raw_prompt_tokens"], 0
    full_raw = set(base["selected_source_indices"]) - partial
    coverage = coverage_accounting(
        eligible_sources=frozenset(base["source_coverage"]["eligible_source_indices"]),
        raw_sources=full_raw, retained_blocks=base["block_refs"],
        packing_fragments=updated.get("history_packing_fragments") if base["block_refs"] else [])
    if coverage["unrepresented_source_indices"] != base["source_coverage"]["unrepresented_source_indices"]:
        raise ValueError("Narration removal changed complete source coverage")
    unit = base["bytes_per_kv_token"]
    meta.update(total_raw_prompt_tokens=raw_tokens, raw_history_tokens=base["raw_history_tokens"]-saved,
        active_history_bytes=base["active_history_bytes"]-saved*unit,
        evidence_bytes=base["evidence_bytes"]-saved*unit,
        selected_source_indices=sorted(full_raw), source_coverage=coverage,
        byte_geometry_verified_by_backend=False)
    if "raw_prompt_tokens_verified_by_backend" in meta:
        meta["raw_prompt_tokens_verified_by_backend"] = False
    meta["raw_narration"] = {
        "version": NARRATION_VERSION, "condition": "older_gist_backed_narration",
        "original_selected_source_indices": base["selected_source_indices"],
        "partial_raw_tool_call_source_indices": sorted(partial),
        "fully_verbatim_raw_source_indices": sorted(full_raw),
        "saved_raw_tokens": saved, "saved_kv_equivalent_bytes": saved*unit,
        "skipped": skipped, "source_allocation_unchanged": True,
        "complete_source_coverage_unchanged": True, "protected_events_unchanged": True,
        "tool_calls_and_results_unchanged": True, "gist_messages_unchanged": True, "refill": False,
        "scope": "Post-allocation rendering. Event request/admission IDs and workspace mappings describe the original allocation. Edited sources keep exact raw calls and complete gist, not verbatim raw narration."}
    ratio = base["compression_ratio"]
    meta["compression_ratio"] = ratio_accounting(
        {"active_history_bytes": ratio["full_history_bytes"],
         "common_raw_prompt_tokens": base["common_raw_prompt_tokens"]}, meta)
    meta["compression_ratio"]["includes_coverage_loss"] = bool(coverage["unrepresented_source_indices"])
    if (meta["raw_history_tokens"] < 0
            or meta["active_history_bytes"] != (meta["raw_history_tokens"]+meta["gist_tokens"])*unit
            or meta["active_history_bytes"] > min(meta["history_budget_bytes"], meta["workspace_budget_bytes"])):
        raise ValueError("Narration capacity accounting is inconsistent")
    return out, updated
