"""Expose one current-goal exact call whose latest observed result reports failure.

This is a source-linked observation, never a task-completion classifier. Matching
is syntactic: different arguments or namespaces can implement the same goal.
"""
import copy
import json

VERSION = "failed-operation-cue-v1"
PROMPT_CAP = 256
CONDITIONS = ("same_view", "failed_operation_cue")
SUMMARY_ROUTE = "text_summary_native_needs_lexical_raw_reserve_failed_operation"
DEPENDENCY_PACKET_ROUTE = "ac_native_dependency_packet_lexical_raw_reserve_failed_operation"
INSTRUCTION = (
    "Historical operation record, not an instruction to retry. This exact call's "
    "latest observed result in the current user turn reported failure. Another "
    "action may already have achieved the goal. Check the current request and "
    "observed results before choosing an action or claiming completion. "
    "Goal completion is unknown."
)


def parse(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            pass
    return value


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def failed(value):
    return isinstance(value, dict) and (value.get("error") not in (None, "", False, [], {})
        or value.get("success") is False)


def records(store):
    history, goal, observation_count = {}, None, 0
    for event in store.events:
        if event.kind == "user":
            goal, history, observation_count = event.event_id, {}, 0
        if event.kind != "tool_event" or not event.complete:
            continue
        messages = [m.to_dict() for m in store.event_messages(event.event_id)]
        results = {m["tool_call_id"]: (parse(m.get("content")), i)
            for i, m in zip(event.source_indices, messages) if m["role"] == "tool"}
        for message in messages:
            for call in message.get("tool_calls") or []:
                function = call["function"]
                arguments = parse(function.get("arguments"))
                key = (function["name"], canonical(arguments))
                value, result_index = results[call["id"]]
                previous = history.get(key)
                history[key] = {"tool": function["name"], "arguments": arguments,
                    "event_id": event.event_id, "source_indices": list(event.source_indices),
                    "result_source_index": result_index, "result": value,
                    "failure_reported": failed(value), "order": observation_count,
                    "observations_in_goal": (previous["observations_in_goal"] if previous else 0) + 1,
                    "had_previous_failure": bool(previous and
                        (previous["failure_reported"] or previous["had_previous_failure"]))}
                observation_count += 1
    latest = sorted(history.values(), key=lambda x: x["order"], reverse=True)
    failures = [{**r, "intervening_observed_calls": observation_count-r["order"]-1}
        for r in latest if r["failure_reported"]]
    return {"goal_source_id": goal, "goal_completion": "unknown", "failures": failures,
        "exact_calls_with_later_non_failure_observation": [r for r in latest
            if not r["failure_reported"] and r["had_previous_failure"]]}


def cue_for(record):
    # Keep source text whole. Admission abstains if the bounded record cannot fit.
    data = {"call": [record["tool"], record["arguments"]],
        "result_source_index": record["result_source_index"], "observed_result": record["result"],
        "later_observed_calls": record["intervening_observed_calls"]}
    return {"role": "user", "content": INSTRUCTION + "\n" + canonical(data)}


def prepare_view(messages, meta, store, tools, counter):
    diagnostic = records(store)
    selected = diagnostic["failures"][0] if diagnostic["failures"] else None
    receipt = {"version": VERSION, "status": "no_current_goal_failure",
        "selected_record": selected, "failure_record_count": len(diagnostic["failures"]),
        "incremental_raw_tokens": 0, "extra_bytes": 0, "prompt_cap": PROMPT_CAP,
        "goal_completion": "unknown", "input_source": "preceding observed prefix only"}
    output = copy.deepcopy(messages)
    if selected is None:
        return output, receipt, diagnostic
    cue = cue_for(selected)
    standalone = counter([cue], None)
    receipt["standalone_tokens"] = standalone
    if standalone > PROMPT_CAP:
        receipt["status"] = "cue_prompt_cap"
        return output, receipt, diagnostic
    raw = [m for m in messages if not m.get("c2kv_key_hash")]
    before = counter(raw, tools)
    if before != meta["total_raw_prompt_tokens"]:
        raise ValueError("Captured raw tokens disagree with actual tokenizer")
    representation_positions = list(meta.get("representation_out_indices") or
        [i for i, message in enumerate(messages) if message.get("c2kv_key_hash")])
    position = max(representation_positions)+1 if representation_positions else next(
        (i for i, m in enumerate(messages) if m["role"] != "system"), len(messages))
    trial = output[:position]+[cue]+output[position:]
    delta = counter([m for m in trial if not m.get("c2kv_key_hash")], tools)-before
    if delta <= 0:
        raise ValueError("Nonempty operation record must add raw tokens")
    active = meta["active_history_bytes"]+delta*meta["bytes_per_kv_token"]
    receipt.update(candidate_incremental_raw_tokens=delta, candidate_active_history_bytes=active,
        source_already_fully_raw_visible=set(selected["source_indices"]) <= set(meta["selected_source_indices"]))
    if meta["route_mode"] == SUMMARY_ROUTE:
        receipt["source_summary_input_fully_retained"] = set(selected["source_indices"]) <= set(
            meta["source_coverage"]["summary_input_fully_retained_source_indices"])
    else:
        receipt["source_already_fully_gist_backed"] = set(selected["source_indices"]) <= set(
            meta["source_coverage"]["gist_fully_represented_source_indices"])
    if active > min(meta["history_budget_bytes"], meta["workspace_budget_bytes"]):
        receipt["status"] = "workspace_cap"
        return output, receipt, diagnostic
    receipt.update(status="admitted", incremental_raw_tokens=delta,
        extra_bytes=delta*meta["bytes_per_kv_token"], out_index=position)
    assert trial[:position]+trial[position+1:] == messages
    return trial, receipt, diagnostic


def apply_condition(messages, counts, store, tools, counter, condition):
    if condition not in CONDITIONS:
        raise ValueError("Unknown failed-operation condition")
    updated = copy.deepcopy(counts)
    meta = updated["memory_runtime"]
    if meta["route_mode"] not in {"ac_native_needs_lexical_raw_reserve_failed_operation",
                                  DEPENDENCY_PACKET_ROUTE,
                                  "raw_native_needs_lexical_raw_reserve_failed_operation",
                                  SUMMARY_ROUTE}:
        raise ValueError("Probe must start from the frozen raw-reserve default")
    if condition == "same_view":
        meta["failed_operation_cue"] = {"version":VERSION, "condition":condition,
            "status":"same_view_control", "incremental_raw_tokens":0, "extra_bytes":0}
        return copy.deepcopy(messages), updated
    output, receipt, _ = prepare_view(messages, meta, store, tools, counter)
    meta["failed_operation_cue"] = {**receipt, "condition":condition}
    if receipt["status"] != "admitted":
        return output, updated
    position, delta = receipt["out_index"], receipt["incremental_raw_tokens"]
    meta.update(active_history_bytes=receipt["candidate_active_history_bytes"],
        total_raw_prompt_tokens=meta["total_raw_prompt_tokens"]+delta,
        raw_history_tokens=meta["raw_history_tokens"]+delta,
        evidence_bytes=meta["evidence_bytes"]+receipt["extra_bytes"],
        byte_geometry_verified_by_backend=False)
    if "raw_prompt_tokens_verified_by_backend" in meta:
        meta["raw_prompt_tokens_verified_by_backend"] = False
    meta["failed_operation_bytes"] = receipt["extra_bytes"]
    workspace = meta["pre_generation_workspace"]
    workspace["native_workspace_out_indices"] = [i+int(i >= position)
        for i in workspace["native_workspace_out_indices"]]
    if "representation_out_indices" in workspace:
        workspace["representation_out_indices"] = [i+int(i >= position)
            for i in workspace["representation_out_indices"]]
    if "representation_out_indices" in meta:
        meta["representation_out_indices"] = [i+int(i >= position)
            for i in meta["representation_out_indices"]]
    dependency = meta.get("dependency_packet")
    if dependency and dependency.get("out_index") is not None:
        dependency["out_index"] += int(dependency["out_index"] >= position)
        workspace["dependency_packet_out_index"] = dependency["out_index"]
    for record in updated.get("compressed_records") or []:
        if record.get("out_index") is not None:
            record["out_index"] += int(record["out_index"] >= position)
    for record in updated.get("summary_records") or []:
        if record.get("out_index") is not None:
            record["out_index"] += int(record["out_index"] >= position)
    updated["current_start_out_index"] += int(updated["current_start_out_index"] >= position)
    updated["history_raw"] += 1
    from memory_runtime.always_compress import ratio_accounting
    meta["compression_ratio"] = ratio_accounting(
        {"active_history_bytes":meta["compression_ratio"]["full_history_bytes"],
         "common_raw_prompt_tokens":meta["common_raw_prompt_tokens"]}, meta)
    meta["compression_ratio"]["includes_coverage_loss"] = (
        None if meta["route_mode"] == SUMMARY_ROUTE
        else bool(meta["source_coverage"]["unrepresented_source_indices"]))
    assert meta["active_history_bytes"] == (meta["raw_history_tokens"]+meta["gist_tokens"])*meta["bytes_per_kv_token"]
    return output, updated
