from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from transformers import AutoTokenizer

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_agent_tool_definition_c2kv import _extract_tool_name  # noqa: E402
from eval_agent_tool_definition_hybrid_router import _render_tool_definition, _tool_name  # noqa: E402
from train.train_data_multiturn import _chat_template_ids  # noqa: E402
from train_agent_tool_definition_c2kv import (  # noqa: E402
    AgentLLMTracesSource,
    AgentToolDefinitionDataArgs,
)


def _json_loads(value: Any, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default if default is not None else value


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


def _schema_obj(tool: Dict[str, Any]) -> Dict[str, Any]:
    function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
    schema = (
        function.get("parameters")
        or tool.get("parameters")
        or tool.get("input_schema")
        or tool.get("schema")
        or {}
    )
    return schema if isinstance(schema, dict) else {}


def _parameter_signature(schema: Dict[str, Any]) -> tuple[tuple[str, str], ...]:
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    out = []
    for name, value in properties.items():
        if isinstance(value, dict):
            typ = value.get("type") or value.get("anyOf") or value.get("oneOf") or value.get("items") or "unknown"
        else:
            typ = "unknown"
        out.append((str(name), json.dumps(typ, ensure_ascii=False, sort_keys=True)))
    return tuple(sorted(out))


def _tool_signature(tool: Dict[str, Any]) -> Dict[str, Any]:
    schema = _schema_obj(tool)
    required = schema.get("required") if isinstance(schema.get("required"), list) else []
    return {
        "name": _tool_name(tool),
        "parameters": _parameter_signature(schema),
        "required": tuple(sorted(str(item) for item in required)),
    }


def _hash_json(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _toolset_key(tools: Sequence[Dict[str, Any]]) -> str:
    signatures = sorted((_tool_signature(tool) for tool in tools), key=lambda item: item["name"])
    return _hash_json(signatures)


def _tool_names(tools: Sequence[Dict[str, Any]]) -> set[str]:
    return {name for name in (_tool_name(tool) for tool in tools) if name}


def _namespace(name: str) -> str:
    for sep in ("__", "-", ".", "/"):
        if sep in name:
            return name.split(sep, 1)[0]
    return name.split("_", 1)[0] if "_" in name else name


def _namespace_set(tools: Sequence[Dict[str, Any]]) -> set[str]:
    return {_namespace(name) for name in _tool_names(tools)}


def _stats(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"min": 0, "avg": 0.0, "p50": 0, "p95": 0, "max": 0}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))
    return {
        "min": ordered[0],
        "avg": round(float(statistics.mean(ordered)), 4),
        "p50": ordered[len(ordered) // 2],
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


def _split_groups(
    groups: Dict[str, set[str]],
    eval_ratio: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)
    eval_count = max(1, int(round(len(keys) * eval_ratio))) if keys else 0
    eval_group_keys = set(keys[:eval_count])
    eval_sessions = set().union(*(groups[key] for key in eval_group_keys)) if eval_group_keys else set()
    train_sessions = set().union(*(groups[key] for key in keys[eval_count:])) if len(keys) > eval_count else set()
    return train_sessions, eval_sessions


def _session_split(sessions: Sequence[Dict[str, Any]], eval_ratio: float, seed: int) -> tuple[set[str], set[str]]:
    ids = sorted(session["session_id"] for session in sessions)
    rng = random.Random(seed)
    rng.shuffle(ids)
    eval_count = max(1, int(round(len(ids) * eval_ratio))) if ids else 0
    return set(ids[eval_count:]), set(ids[:eval_count])


def _greedy_namespace_split(
    sessions: Sequence[Dict[str, Any]],
    eval_ratio: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    namespace_to_sessions: Dict[str, set[str]] = defaultdict(set)
    session_namespaces: Dict[str, set[str]] = {}
    for session in sessions:
        sid = session["session_id"]
        namespaces = session.get("namespaces", set())
        session_namespaces[sid] = namespaces
        for namespace in namespaces:
            namespace_to_sessions[namespace].add(sid)
    namespace_keys = sorted(namespace_to_sessions, key=lambda key: (-len(namespace_to_sessions[key]), key))
    rng = random.Random(seed)
    rng.shuffle(namespace_keys)
    target_eval = max(1, int(round(len(sessions) * eval_ratio))) if sessions else 0
    eval_namespaces = set()
    eval_sessions = set()
    for namespace in namespace_keys:
        candidate = eval_sessions | namespace_to_sessions[namespace]
        if len(eval_sessions) < target_eval or abs(len(candidate) - target_eval) < abs(len(eval_sessions) - target_eval):
            eval_namespaces.add(namespace)
            eval_sessions = candidate
    train_sessions = {
        session["session_id"]
        for session in sessions
        if not (session_namespaces[session["session_id"]] & eval_namespaces)
    }
    return train_sessions, eval_sessions


def _overlap_report(
    sessions: Sequence[Dict[str, Any]],
    train_ids: set[str],
    eval_ids: set[str],
) -> Dict[str, Any]:
    train = [session for session in sessions if session["session_id"] in train_ids]
    eval_ = [session for session in sessions if session["session_id"] in eval_ids]
    train_toolsets = {session["toolset_key"] for session in train}
    eval_toolsets = {session["toolset_key"] for session in eval_}
    train_names = set().union(*(session["tool_names"] for session in train)) if train else set()
    eval_names = set().union(*(session["tool_names"] for session in eval_)) if eval_ else set()
    train_namespaces = set().union(*(session["namespaces"] for session in train)) if train else set()
    eval_namespaces = set().union(*(session["namespaces"] for session in eval_)) if eval_ else set()
    return {
        "train_sessions": len(train),
        "eval_sessions": len(eval_),
        "train_toolsets": len(train_toolsets),
        "eval_toolsets": len(eval_toolsets),
        "toolset_overlap": len(train_toolsets & eval_toolsets),
        "tool_name_overlap": len(train_names & eval_names),
        "eval_tool_name_overlap_rate": round(len(train_names & eval_names) / len(eval_names), 6) if eval_names else 0.0,
        "namespace_overlap": len(train_namespaces & eval_namespaces),
        "eval_namespace_overlap_rate": round(len(train_namespaces & eval_namespaces) / len(eval_namespaces), 6)
        if eval_namespaces else 0.0,
    }


def _count_tokens(tokenizer: Any, text: str) -> int:
    if tokenizer is None:
        return max(1, len(text) // 4)
    return len(tokenizer.encode(text, add_special_tokens=False))


def _tool_doc_tokens(tokenizer: Any, tools: Sequence[Dict[str, Any]]) -> int:
    text = "Tool definitions:\n" + _render_tool_definition(tools)
    if tokenizer is None:
        return max(1, len(text) // 4)
    return len(_chat_template_ids(tokenizer, [{"role": "user", "content": text}]))


def _target_tool_position(tools: Sequence[Dict[str, Any]], target_tool: Optional[str]) -> Optional[int]:
    if not target_tool:
        return None
    names = [_tool_name(tool) for tool in tools]
    try:
        return names.index(target_tool)
    except ValueError:
        return None


def _truncate_contains_target(
    tokenizer: Any,
    tools: Sequence[Dict[str, Any]],
    target_tool: Optional[str],
    ratio: int,
) -> Optional[bool]:
    pos = _target_tool_position(tools, target_tool)
    if pos is None:
        return None
    full_tokens = _tool_doc_tokens(tokenizer, tools)
    budget = max(1, (full_tokens + ratio - 1) // ratio)
    running = _count_tokens(tokenizer, "Tool definitions:\n")
    for index, tool in enumerate(tools):
        running += _count_tokens(tokenizer, json.dumps(tool, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        if index == pos:
            return running <= budget
    return False


def _shuffle_json_keys(value: Any, rng: random.Random) -> Any:
    if isinstance(value, dict):
        items = list(value.items())
        rng.shuffle(items)
        return {key: _shuffle_json_keys(item, rng) for key, item in items}
    if isinstance(value, list):
        return [_shuffle_json_keys(item, rng) for item in value]
    return value


def _canonical_markup(tools: Sequence[Dict[str, Any]]) -> str:
    blocks = []
    for tool in tools:
        signature = _tool_signature(tool)
        blocks.append("<TOOL>")
        blocks.append(f"<NAME> {signature['name']}")
        function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
        description = function.get("description") or tool.get("description") or ""
        if description:
            blocks.append(f"<DESCRIPTION> {description}")
        blocks.append("<PARAMETERS>")
        for name, typ in signature["parameters"]:
            required = str(name in set(signature["required"])).lower()
            blocks.append(f'  <PARAM name="{name}" type="{typ}" required="{required}"/>')
        blocks.append("</PARAMETERS>")
        blocks.append("</TOOL>")
    return "\n".join(blocks)


def _format_variant_lengths(tokenizer: Any, tools: Sequence[Dict[str, Any]], seed: int) -> Dict[str, int]:
    rng = random.Random(seed)
    shuffled_tools = list(tools)
    rng.shuffle(shuffled_tools)
    shuffled_keys = _shuffle_json_keys(list(tools), rng)
    variants = {
        "json_minified": json.dumps(list(tools), ensure_ascii=False, separators=(",", ":")),
        "json_pretty": json.dumps(list(tools), ensure_ascii=False, indent=2, sort_keys=True),
        "json_shuffled_keys": json.dumps(shuffled_keys, ensure_ascii=False, separators=(",", ":")),
        "json_shuffled_tools": json.dumps(shuffled_tools, ensure_ascii=False, separators=(",", ":")),
        "canonical_markup": _canonical_markup(tools),
    }
    return {name: _count_tokens(tokenizer, value) for name, value in variants.items()}


def _chunk_stats(tokenizer: Any, tools: Sequence[Dict[str, Any]], max_doc_length: int) -> Dict[str, Any]:
    full_tokens = _tool_doc_tokens(tokenizer, tools)
    tool_tokens = [_tool_doc_tokens(tokenizer, [tool]) for tool in tools]
    namespace_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for tool in tools:
        namespace_groups[_namespace(_tool_name(tool))].append(tool)
    app_tokens = [_tool_doc_tokens(tokenizer, group) for group in namespace_groups.values()]
    field_tokens = []
    for tool in tools:
        function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
        pieces = [
            _tool_name(tool),
            str(function.get("description") or tool.get("description") or ""),
            json.dumps(_schema_obj(tool), ensure_ascii=False, sort_keys=True),
        ]
        field_tokens.extend(_count_tokens(tokenizer, piece) for piece in pieces if piece)
    return {
        "fixed_token_chunks": (full_tokens + max_doc_length - 1) // max_doc_length,
        "tool_level_docs": len(tools),
        "tool_level_oversize_docs": sum(1 for value in tool_tokens if value > max_doc_length),
        "tool_level_token_stats": _stats(tool_tokens),
        "app_level_docs": len(app_tokens),
        "app_level_oversize_docs": sum(1 for value in app_tokens if value > max_doc_length),
        "app_level_token_stats": _stats(app_tokens),
        "field_piece_token_stats": _stats(field_tokens),
    }


def _build_sessions(args: argparse.Namespace, tokenizer: Any) -> List[Dict[str, Any]]:
    data_args = AgentToolDefinitionDataArgs(
        dataset_path=args.dataset_path,
        eval_ratio=args.eval_ratio,
        split_seed=args.split_seed,
        max_samples_per_session=0,
        max_tool_definition_tokens=args.max_tool_definition_tokens,
        max_doc_length=args.max_doc_length,
        max_doc_num=args.max_doc_num,
        truncate_tool_definition=False,
        require_tool_call=args.require_tool_call,
        min_target_tokens=args.min_target_tokens,
    )
    source = AgentLLMTracesSource(data_args)
    sessions = []
    for session in source.sessions:
        examples = source._session_examples(session["session_id"], session["spans"])
        if not examples:
            continue
        example = examples[0]
        tools = _as_tool_list(example.tool_definition)
        if not tools:
            continue
        target_tools = [_extract_tool_name(item.answer) for item in examples]
        sessions.append({
            "session_id": session["session_id"],
            "num_examples": len(examples),
            "tools": tools,
            "tool_names": _tool_names(tools),
            "namespaces": _namespace_set(tools),
            "toolset_key": _toolset_key(tools),
            "tool_tokens": _tool_doc_tokens(tokenizer, tools),
            "target_tools": [item for item in target_tools if item],
        })
    if args.max_sessions:
        sessions = sessions[: args.max_sessions]
    return sessions


def diagnose(args: argparse.Namespace) -> Dict[str, Any]:
    tokenizer = None
    if args.tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer,
            trust_remote_code=True,
            local_files_only=True,
            padding_side="right",
        )
    sessions = _build_sessions(args, tokenizer)
    session_train, session_eval = _session_split(sessions, args.eval_ratio, args.split_seed)
    toolset_groups: Dict[str, set[str]] = defaultdict(set)
    for session in sessions:
        toolset_groups[session["toolset_key"]].add(session["session_id"])
    toolset_train, toolset_eval = _split_groups(toolset_groups, args.eval_ratio, args.split_seed)
    namespace_train, namespace_eval = _greedy_namespace_split(sessions, args.eval_ratio, args.split_seed)

    truncate_hits: Dict[str, List[int]] = {f"ratio_{ratio}": [] for ratio in args.ratios}
    target_positions = []
    variant_lengths: Dict[str, List[int]] = defaultdict(list)
    chunk_summaries: Dict[str, List[float]] = defaultdict(list)
    for session in sessions:
        tools = session["tools"]
        target_tool = session["target_tools"][0] if session["target_tools"] else None
        pos = _target_tool_position(tools, target_tool)
        if pos is not None:
            target_positions.append(pos)
        for ratio in args.ratios:
            hit = _truncate_contains_target(tokenizer, tools, target_tool, ratio)
            if hit is not None:
                truncate_hits[f"ratio_{ratio}"].append(int(hit))
        for name, value in _format_variant_lengths(tokenizer, tools, args.split_seed).items():
            variant_lengths[name].append(value)
        chunk_stats = _chunk_stats(tokenizer, tools, args.max_doc_length)
        for key, value in chunk_stats.items():
            if isinstance(value, (int, float)):
                chunk_summaries[key].append(value)

    split_report = {
        "session_disjoint": _overlap_report(sessions, session_train, session_eval),
        "toolset_disjoint": _overlap_report(sessions, toolset_train, toolset_eval),
        "namespace_disjoint_proxy": _overlap_report(sessions, namespace_train, namespace_eval),
    }
    split_manifests = {
        "session_disjoint": {
            "train_session_ids": sorted(session_train),
            "eval_session_ids": sorted(session_eval),
        },
        "toolset_disjoint": {
            "train_session_ids": sorted(toolset_train),
            "eval_session_ids": sorted(toolset_eval),
        },
        "namespace_disjoint_proxy": {
            "train_session_ids": sorted(namespace_train),
            "eval_session_ids": sorted(namespace_eval),
        },
    }
    if args.manifest_file:
        manifest_path = Path(args.manifest_file)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(split_manifests, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return {
        "dataset_path": args.dataset_path,
        "token_count_mode": "tokenizer" if tokenizer else "rough_char_div_4",
        "num_sessions": len(sessions),
        "num_toolsets": len({session["toolset_key"] for session in sessions}),
        "tool_tokens": _stats([session["tool_tokens"] for session in sessions]),
        "tools_per_session": _stats([len(session["tools"]) for session in sessions]),
        "split_report": split_report,
        "manifest_file": args.manifest_file,
        "target_tool_position": _stats(target_positions),
        "truncate_target_survival_rate": {
            key: round(sum(values) / len(values), 6) if values else 0.0
            for key, values in truncate_hits.items()
        },
        "format_variant_token_lengths": {
            key: _stats(values) for key, values in variant_lengths.items()
        },
        "chunk_strategy_numeric_stats": {
            key: _stats(values) for key, values in chunk_summaries.items()
        },
        "interpretation_hints": [
            "If session_disjoint has high toolset/tool-name overlap, the original split is IID over toolsets.",
            "If truncate_target_survival_rate is high for a ratio, truncate may be strong because target tools appear early.",
            "If format variant lengths differ heavily, rerun eval with these rendered variants to test format dependence.",
            "If tool_level_oversize_docs is low, tool-level chunking is a strong next training/eval candidate.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Agent tool-definition split leakage and chunk/format risks.")
    parser.add_argument("--dataset_path", default="./datasets/agent-llm-traces")
    parser.add_argument("--tokenizer")
    parser.add_argument("--output_file", default="./outputs/agent_tooldef_generalization_diagnosis.json")
    parser.add_argument("--manifest_file", default="./outputs/agent_tooldef_split_manifests.json")
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--max_sessions", type=int)
    parser.add_argument("--max_tool_definition_tokens", type=int, default=10000)
    parser.add_argument("--max_doc_length", type=int, default=1024)
    parser.add_argument("--max_doc_num", type=int, default=10)
    parser.add_argument("--min_target_tokens", type=int, default=128)
    parser.add_argument("--require_tool_call", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--ratios", type=lambda x: [int(item) for item in str(x).split(",") if item], default=[2, 4, 8])
    args = parser.parse_args()
    result = diagnose(args)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
