from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import torch
from transformers import DataCollatorWithPadding, HfArgumentParser

from gist_args import ModelArgs, TrainingArgs
from models import format_numel_str, get_model_and_tokenizer
from train.trainer import GistMultiDocTrainer

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_agent_tool_definition_hybrid_router import _render_tool_definition, _tool_name  # noqa: E402
from train_agent_tool_definition_c2kv import (  # noqa: E402
    AgentLLMTracesSource,
    AgentToolDefinitionDataArgs,
    AgentToolDefinitionDataset,
    AgentToolDefinitionExample,
    _json_loads,
    _setup_device,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class UnifiedNextActionDataArgs:
    agent_dataset_path: str = "./datasets/agent-llm-traces"
    toolathlon_dataset_path: str = "./datasets/toolathlon"
    hermes_dataset_path: str = "./datasets/hermes-agent-reasoning-traces"
    hermes_configs: str = "kimi,glm-5.1"
    split_seed: int = 42
    train_ratio: float = 0.7
    validation_ratio: float = 0.1
    test_ratio: float = 0.2
    source_mix: str = "agent_llm_traces:0.3,toolathlon:0.4,hermes:0.3"
    split: str = "train"
    max_train_examples: Optional[int] = 20000
    max_eval_examples: Optional[int] = 2000
    max_steps_per_trajectory: int = 6
    max_history_steps: int = 6
    max_observation_chars: int = 1200
    max_history_chars: int = 12000
    max_tools_per_sample: int = 32
    same_namespace_negative_tools: int = 8
    random_negative_tools: int = 24
    canonical_format_prob: float = 0.7
    minified_json_prob: float = 0.2
    shuffled_json_prob: float = 0.1
    truncate_description_chars: int = 600
    shuffle_tools: bool = True
    max_doc_length: int = 1024
    max_doc_num: int = 32
    max_length: int = 2048
    max_system_length: int = 256
    max_tool_definition_tokens: int = 32000
    truncate_tool_definition: bool = True
    min_target_tokens: int = 32
    device_type: str = "auto"
    npu_attn_impl: str = "eager"
    output_manifest_file: str = "./outputs/unified_next_action_split_manifest.json"


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _hash_json(value: Any) -> str:
    return hashlib.sha1(_json_dumps(value).encode("utf-8")).hexdigest()


def _namespace(name: str) -> str:
    for sep in ("__", ".", "/", "-"):
        if sep in name:
            return name.split(sep, 1)[0]
    return name.split("_", 1)[0] if "_" in name else name


def _schema_obj(tool: Dict[str, Any]) -> Dict[str, Any]:
    function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
    schema = function.get("parameters") or tool.get("parameters") or tool.get("input_schema") or tool.get("schema") or {}
    return schema if isinstance(schema, dict) else {}


def _parameter_signature(tool: Dict[str, Any]) -> List[Dict[str, Any]]:
    schema = _schema_obj(tool)
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    required = set(str(item) for item in schema.get("required", []) if isinstance(schema.get("required"), list))
    rows = []
    for name, value in properties.items():
        value = value if isinstance(value, dict) else {}
        typ = value.get("type") or value.get("anyOf") or value.get("oneOf") or value.get("items") or "unknown"
        rows.append({
            "name": str(name),
            "type": json.dumps(typ, ensure_ascii=False, sort_keys=True) if not isinstance(typ, str) else typ,
            "required": str(name) in required,
        })
    return sorted(rows, key=lambda item: item["name"])


def _toolset_hash(tools: Sequence[Dict[str, Any]]) -> str:
    signatures = []
    for tool in tools:
        signatures.append({
            "name": _tool_name(tool),
            "parameters": _parameter_signature(tool),
        })
    return _hash_json(sorted(signatures, key=lambda item: item["name"]))


def _tool_description(tool: Dict[str, Any], limit: int) -> str:
    function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
    description = str(function.get("description") or tool.get("description") or "")
    return description[:limit].rstrip() if limit and len(description) > limit else description


def _canonical_tool_doc(tool: Dict[str, Any], description_limit: int) -> str:
    name = _tool_name(tool)
    blocks = [
        "<TOOL>",
        f"<NAMESPACE> {_namespace(name)}",
        f"<NAME> {name}",
    ]
    description = _tool_description(tool, description_limit)
    if description:
        blocks.append(f"<DESCRIPTION> {description}")
    blocks.append("<PARAMETERS>")
    for param in _parameter_signature(tool):
        blocks.append(
            f'<PARAM name="{param["name"]}" type="{param["type"]}" '
            f'required="{str(param["required"]).lower()}">'
        )
    blocks.extend(["</PARAMETERS>", "</TOOL>"])
    return "\n".join(blocks)


def _shuffle_json_keys(value: Any, rng: random.Random) -> Any:
    if isinstance(value, dict):
        items = list(value.items())
        rng.shuffle(items)
        return {key: _shuffle_json_keys(item, rng) for key, item in items}
    if isinstance(value, list):
        return [_shuffle_json_keys(item, rng) for item in value]
    return value


def _render_tool_documents(
    tools: Sequence[Dict[str, Any]],
    rng: random.Random,
    args: UnifiedNextActionDataArgs,
) -> tuple[str, List[str]]:
    tools = list(tools)
    if args.shuffle_tools:
        rng.shuffle(tools)
    p = rng.random()
    if p < args.canonical_format_prob:
        docs = [_canonical_tool_doc(tool, args.truncate_description_chars) for tool in tools]
    elif p < args.canonical_format_prob + args.minified_json_prob:
        docs = [json.dumps(tool, ensure_ascii=False, separators=(",", ":")) for tool in tools]
    else:
        docs = [
            json.dumps(_shuffle_json_keys(tool, rng), ensure_ascii=False, separators=(",", ":"))
            for tool in tools
        ]
    return "\n\n".join(docs), docs


def _function_payload(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    function = value.get("function") if isinstance(value.get("function"), dict) else {}
    name = value.get("name") or value.get("tool") or value.get("tool_name") or function.get("name")
    arguments = value.get("arguments") or value.get("args") or function.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except Exception:
            pass
    if not name:
        return None
    return {"tool": str(name), "arguments": arguments if isinstance(arguments, dict) else arguments}


def _target_answer(action: Dict[str, Any]) -> str:
    payload = {"name": action["tool"], "arguments": action.get("arguments", {})}
    return "Action:\n<tool_call>\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n</tool_call>"


def _step_text(step: Dict[str, Any], index: int, max_observation_chars: int) -> str:
    action = step.get("action", {})
    observation = str(step.get("observation") or "")
    if max_observation_chars and len(observation) > max_observation_chars:
        observation = observation[:max_observation_chars].rstrip()
    return (
        f'<STEP index="{index}">\n'
        f'<ACTION tool="{action.get("tool", "")}">\n'
        f"<ARGUMENTS>{json.dumps(action.get('arguments', {}), ensure_ascii=False, separators=(',', ':'))}</ARGUMENTS>\n"
        f"<OBSERVATION status=\"{step.get('status', 'unknown')}\">\n"
        f"{observation}\n"
        "</OBSERVATION>\n"
        "</STEP>"
    )


def _prompt_messages(task_instruction: str, history: Sequence[Dict[str, Any]], args: UnifiedNextActionDataArgs) -> List[Dict[str, str]]:
    kept = list(history[-args.max_history_steps :]) if args.max_history_steps else list(history)
    rendered_steps = []
    total_chars = 0
    for step_index, step in enumerate(kept, start=max(1, len(history) - len(kept) + 1)):
        text = _step_text(step, step_index, args.max_observation_chars)
        total_chars += len(text)
        if args.max_history_chars and total_chars > args.max_history_chars:
            break
        rendered_steps.append(text)
    content = "Task instruction:\n" + task_instruction.strip()
    if rendered_steps:
        content += "\n\nPrevious action-observation history:\n" + "\n\n".join(rendered_steps)
    content += (
        "\n\nPredict the next action. Output only:\n"
        "Action:\n<tool_call>\n{\"name\":\"tool_name\",\"arguments\":{}}\n</tool_call>"
    )
    return [{"role": "user", "content": content}]


def _select_tools(
    tools: Sequence[Dict[str, Any]],
    target_tool: str,
    rng: random.Random,
    args: UnifiedNextActionDataArgs,
) -> List[Dict[str, Any]]:
    if not args.max_tools_per_sample or len(tools) <= args.max_tools_per_sample:
        return list(tools)
    target = [tool for tool in tools if _tool_name(tool) == target_tool]
    target_namespace = _namespace(target_tool)
    same_namespace = [
        tool for tool in tools
        if _tool_name(tool) != target_tool and _namespace(_tool_name(tool)) == target_namespace
    ]
    others = [
        tool for tool in tools
        if _tool_name(tool) != target_tool and _namespace(_tool_name(tool)) != target_namespace
    ]
    rng.shuffle(same_namespace)
    rng.shuffle(others)
    selected = target[:1]
    selected.extend(same_namespace[: args.same_namespace_negative_tools])
    remaining = max(0, args.max_tools_per_sample - len(selected))
    selected.extend(others[: min(args.random_negative_tools, remaining)])
    remaining = max(0, args.max_tools_per_sample - len(selected))
    if remaining:
        selected.extend((same_namespace[args.same_namespace_negative_tools :] + others[args.random_negative_tools :])[:remaining])
    return selected[: args.max_tools_per_sample]


def _choose_step_indices(num_steps: int, max_steps: int, rng: random.Random) -> List[int]:
    if num_steps <= max_steps:
        return list(range(num_steps))
    buckets = [
        list(range(0, max(1, num_steps // 3))),
        list(range(max(1, num_steps // 3), max(2, 2 * num_steps // 3))),
        list(range(max(2, 2 * num_steps // 3), num_steps)),
    ]
    chosen = []
    per_bucket = max(1, max_steps // 3)
    for bucket in buckets:
        rng.shuffle(bucket)
        chosen.extend(bucket[:per_bucket])
    remaining = [item for item in range(num_steps) if item not in set(chosen)]
    rng.shuffle(remaining)
    chosen.extend(remaining[: max_steps - len(chosen)])
    return sorted(chosen[:max_steps])


def _split_for_group(group_id: str, args: UnifiedNextActionDataArgs) -> str:
    value = int(hashlib.sha1(f"{args.split_seed}:{group_id}".encode("utf-8")).hexdigest(), 16) / (2**160)
    if value < args.train_ratio:
        return "train"
    if value < args.train_ratio + args.validation_ratio:
        return "eval"
    return "test"


def _make_examples_from_trajectory(
    source: str,
    task_group_id: str,
    trajectory_id: str,
    task_instruction: str,
    tools: Sequence[Dict[str, Any]],
    trajectory: Sequence[Dict[str, Any]],
    split: str,
    args: UnifiedNextActionDataArgs,
) -> List[AgentToolDefinitionExample]:
    if not task_instruction or not tools or not trajectory:
        return []
    rng = random.Random(f"{args.split_seed}:{source}:{trajectory_id}:{split}")
    indices = _choose_step_indices(len(trajectory), args.max_steps_per_trajectory, rng)
    rows = []
    for step_index in indices:
        action = trajectory[step_index].get("action", {})
        target_tool = str(action.get("tool") or "")
        if not target_tool:
            continue
        selected_tools = _select_tools(tools, target_tool, rng, args)
        tool_definition, tool_documents = _render_tool_documents(selected_tools, rng, args)
        rows.append(
            AgentToolDefinitionExample(
                qid=f"{source}:{trajectory_id}:{step_index}",
                session_id=trajectory_id,
                tool_definition=tool_definition,
                tool_documents=tool_documents,
                input_messages=_prompt_messages(task_instruction, trajectory[:step_index], args),
                answer=_target_answer(action),
                has_tool_call=True,
                system_prompt=(
                    "You are a tool-calling assistant. Use the available tool definitions and "
                    "the previous action-observation history to predict the next action."
                ),
            )
        )
    return rows


def _toolathlon_rows(path: Path) -> Iterator[Dict[str, Any]]:
    for jsonl_file in sorted(path.glob("*.jsonl")):
        with jsonl_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    row["_file"] = jsonl_file.name
                    yield row


def _toolathlon_tools(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    traj_log = row.get("traj_log") if isinstance(row.get("traj_log"), dict) else row
    tool_calls = _json_loads(traj_log.get("tool_calls"), {}) or {}
    if isinstance(tool_calls, dict) and isinstance(tool_calls.get("tools"), list):
        return [item for item in tool_calls["tools"] if isinstance(item, dict)]
    if isinstance(tool_calls, list):
        return [item for item in tool_calls if isinstance(item, dict)]
    return []


def _toolathlon_trajectory(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    traj_log = row.get("traj_log") if isinstance(row.get("traj_log"), dict) else row
    messages = _json_loads(traj_log.get("messages"), []) or []
    steps = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            payload = _function_payload(call)
            if not payload:
                continue
            observation = ""
            status = "unknown"
            for next_message in messages[index + 1 : index + 4]:
                if isinstance(next_message, dict) and next_message.get("role") == "tool":
                    observation = str(next_message.get("content") or "")
                    status = "failure" if "error" in observation.lower() else "success"
                    break
            steps.append({"action": payload, "observation": observation, "status": status})
    return steps


def _load_toolathlon(args: UnifiedNextActionDataArgs, split: str) -> List[AgentToolDefinitionExample]:
    rows = []
    for row in _toolathlon_rows(Path(args.toolathlon_dataset_path)):
        task_group_id = str(row.get("task_name") or row.get("task_id") or row.get("id") or "")
        if _split_for_group(f"toolathlon:{task_group_id}", args) != split:
            continue
        traj_log = row.get("traj_log") if isinstance(row.get("traj_log"), dict) else row
        config = _json_loads(traj_log.get("config"), {}) or {}
        task_instruction = str(config.get("task_str") or row.get("task") or row.get("task_name") or "")
        trajectory_id = f"toolathlon:{row.get('modelname_run') or row.get('_file')}:{task_group_id}"
        rows.extend(
            _make_examples_from_trajectory(
                "toolathlon",
                task_group_id,
                trajectory_id,
                task_instruction,
                _toolathlon_tools(row),
                _toolathlon_trajectory(row),
                split,
                args,
            )
        )
    return rows


def _hermes_rows(path: Path, configs: Sequence[str]) -> Iterator[Dict[str, Any]]:
    import pyarrow.parquet as pq

    for config in configs:
        parquet_file = path / "data" / config / "train.parquet"
        if not parquet_file.exists():
            logger.warning("Hermes parquet missing: %s", parquet_file)
            continue
        pf = pq.ParquetFile(parquet_file)
        for batch in pf.iter_batches(batch_size=512):
            for row in batch.to_pylist():
                row["_config"] = config
                yield row


def _hermes_tools(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    value = _json_loads(row.get("tools"), [])
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _hermes_trajectory(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    conversations = row.get("conversations") or []
    steps = []
    for index, message in enumerate(conversations):
        if not isinstance(message, dict) or message.get("from") != "gpt":
            continue
        blocks = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", str(message.get("value") or ""), flags=re.S)
        for block in blocks:
            try:
                payload = _function_payload(json.loads(block))
            except Exception:
                payload = None
            if not payload:
                continue
            observation = ""
            status = "unknown"
            for next_message in conversations[index + 1 : index + 4]:
                if isinstance(next_message, dict) and next_message.get("from") == "tool":
                    observation = str(next_message.get("value") or "")
                    status = "failure" if "error" in observation.lower() else "success"
                    break
            steps.append({"action": payload, "observation": observation, "status": status})
    return steps


def _load_hermes(args: UnifiedNextActionDataArgs, split: str) -> List[AgentToolDefinitionExample]:
    rows = []
    configs = [item.strip() for item in args.hermes_configs.split(",") if item.strip()]
    for row in _hermes_rows(Path(args.hermes_dataset_path), configs):
        task_group_id = str(row.get("task_id") or row.get("id") or row.get("task") or "")
        if _split_for_group(f"hermes:{task_group_id}", args) != split:
            continue
        trajectory_id = f"hermes:{row.get('_config')}:{row.get('id')}"
        rows.extend(
            _make_examples_from_trajectory(
                "hermes",
                task_group_id,
                trajectory_id,
                str(row.get("task") or ""),
                _hermes_tools(row),
                _hermes_trajectory(row),
                split,
                args,
            )
        )
    return rows


def _load_agent_traces(args: UnifiedNextActionDataArgs, split: str) -> List[AgentToolDefinitionExample]:
    data_args = AgentToolDefinitionDataArgs(
        dataset_path=args.agent_dataset_path,
        eval_ratio=args.validation_ratio + args.test_ratio,
        split_seed=args.split_seed,
        max_samples_per_session=0,
        max_doc_length=args.max_doc_length,
        max_doc_num=args.max_doc_num,
        max_tool_definition_tokens=args.max_tool_definition_tokens,
        max_length=args.max_length,
        max_system_length=args.max_system_length,
        truncate_tool_definition=args.truncate_tool_definition,
        require_tool_call=True,
        min_target_tokens=args.min_target_tokens,
    )
    source = AgentLLMTracesSource(data_args)
    examples = []
    for session in source.sessions:
        task_group_id = session["session_id"]
        row_split = _split_for_group(f"agent_llm_traces:{task_group_id}", args)
        if row_split != split:
            continue
        candidates = source._session_examples(task_group_id, session["spans"])
        rng = random.Random(f"{args.split_seed}:agent_llm_traces:{task_group_id}:{split}")
        if args.max_steps_per_trajectory and len(candidates) > args.max_steps_per_trajectory:
            candidates = [candidates[index] for index in _choose_step_indices(len(candidates), args.max_steps_per_trajectory, rng)]
        for item in candidates:
            action = _extract_action_from_target(item.answer)
            if not action:
                continue
            tools = _tools_from_definition(item.tool_definition)
            if not tools:
                continue
            selected_tools = _select_tools(tools, action["tool"], rng, args)
            tool_definition, tool_documents = _render_tool_documents(selected_tools, rng, args)
            examples.append(
                AgentToolDefinitionExample(
                    qid="agent_llm_traces:" + item.qid,
                    session_id=item.session_id,
                    tool_definition=tool_definition,
                    tool_documents=tool_documents,
                    input_messages=item.input_messages,
                    answer=_target_answer(action),
                    has_tool_call=True,
                    system_prompt=item.system_prompt,
                )
            )
    return examples


def _tools_from_definition(value: str) -> List[Dict[str, Any]]:
    parsed = _json_loads(value, [])
    if isinstance(parsed, dict):
        if isinstance(parsed.get("tools"), list):
            parsed = parsed["tools"]
        elif isinstance(parsed.get("functions"), list):
            parsed = parsed["functions"]
        else:
            parsed = [parsed]
    return [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []


def _extract_action_from_target(text: str) -> Optional[Dict[str, Any]]:
    blocks = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text or "", flags=re.S)
    for block in blocks:
        try:
            payload = _function_payload(json.loads(block))
        except Exception:
            payload = None
        if payload:
            return payload
    return None


def _parse_source_mix(value: str) -> Dict[str, float]:
    out = {}
    for item in value.split(","):
        if not item.strip():
            continue
        name, weight = item.split(":", 1)
        out[name.strip()] = float(weight)
    total = sum(out.values()) or 1.0
    return {key: value / total for key, value in out.items()}


def _sample_mix(
    pools: Dict[str, List[AgentToolDefinitionExample]],
    max_examples: Optional[int],
    mix: Dict[str, float],
    seed: int,
) -> List[AgentToolDefinitionExample]:
    rng = random.Random(seed)
    for pool in pools.values():
        rng.shuffle(pool)
    if max_examples is None:
        rows = [item for source in sorted(pools) for item in pools[source]]
        rng.shuffle(rows)
        return rows
    selected = []
    remaining_sources = set(pools)
    for source, weight in mix.items():
        take = min(len(pools.get(source, [])), int(round(max_examples * weight)))
        selected.extend(pools.get(source, [])[:take])
    if len(selected) < max_examples:
        leftovers = []
        for source in remaining_sources:
            already = sum(1 for item in selected if item.qid.startswith(source + ":"))
            leftovers.extend(pools.get(source, [])[already:])
        rng.shuffle(leftovers)
        selected.extend(leftovers[: max_examples - len(selected)])
    rng.shuffle(selected)
    return selected[:max_examples]


def _load_unified_examples(args: UnifiedNextActionDataArgs, split: str) -> List[AgentToolDefinitionExample]:
    pools = {
        "agent_llm_traces": _load_agent_traces(args, split),
        "toolathlon": _load_toolathlon(args, split),
        "hermes": _load_hermes(args, split),
    }
    logger.info("Unified %s pools: %s", split, {key: len(value) for key, value in pools.items()})
    max_examples = args.max_train_examples if split == "train" else args.max_eval_examples
    rows = _sample_mix(pools, max_examples, _parse_source_mix(args.source_mix), args.split_seed + (0 if split == "train" else 1))
    logger.info(
        "Unified %s sampled %d examples by source=%s",
        split,
        len(rows),
        dict(Counter(row.qid.split(":", 1)[0] for row in rows)),
    )
    return rows


def main() -> None:
    parser = HfArgumentParser([ModelArgs, TrainingArgs, UnifiedNextActionDataArgs])
    model_args, training_args, data_args = parser.parse_args_into_dataclasses()
    device_type = _setup_device(model_args, data_args)

    if model_args.gist_gradient_checkpointing:
        import models.gist_utils as _gist_utils

        _gist_utils.GIST_GRADIENT_CHECKPOINTING = True

    model, tokenizer = get_model_and_tokenizer(
        model_args,
        device=device_type,
        evaluation_mode=not training_args.do_train,
    )
    if model_args.enable_gist and training_args.only_train_gist:
        for name, param in model.named_parameters():
            param.requires_grad_("gist" in name)

    logger.info(f"Total Model params: {format_numel_str(sum(p.numel() for p in model.parameters()))}")
    logger.info(
        "Trainable Model params: "
        f"{format_numel_str(sum(p.numel() for p in model.parameters() if p.requires_grad))}"
    )

    # The unified source scans three datasets and can take several minutes.
    # Building it inside `main_process_first` makes non-zero ranks wait at an
    # HCCL barrier long enough to hit socket timeouts on Ascend. Every rank
    # builds the same deterministic sample list instead; this is more I/O, but
    # avoids distributed startup deadlocks.
    train_examples = _load_unified_examples(data_args, "train")
    eval_examples = _load_unified_examples(data_args, "eval")
    train_dataset = AgentToolDefinitionDataset(
        train_examples,
        tokenizer=tokenizer,
        max_doc_length=data_args.max_doc_length,
        max_doc_num=data_args.max_doc_num,
        max_length=data_args.max_length,
        max_system_length=data_args.max_system_length,
        truncate_tool_definition=data_args.truncate_tool_definition,
        min_target_tokens=data_args.min_target_tokens,
        max_tool_definition_tokens=data_args.max_tool_definition_tokens,
    )
    eval_dataset = AgentToolDefinitionDataset(
        eval_examples,
        tokenizer=tokenizer,
        max_doc_length=data_args.max_doc_length,
        max_doc_num=data_args.max_doc_num,
        max_length=data_args.max_length,
        max_system_length=data_args.max_system_length,
        truncate_tool_definition=data_args.truncate_tool_definition,
        min_target_tokens=data_args.min_target_tokens,
        max_tool_definition_tokens=data_args.max_tool_definition_tokens,
    )

    if len(train_dataset) == 0:
        raise RuntimeError("No train samples remained after filtering")

    trainer = GistMultiDocTrainer(
        model=model,
        args=training_args,
        max_doc_length=train_dataset.max_doc_length,
        model_args=model_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if len(eval_dataset) else None,
        data_collator=DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=True,
            return_tensors="pt",
        ),
    )

    if training_args.do_train:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
    else:
        if len(eval_dataset) == 0:
            raise ValueError("Evaluation requested but eval dataset is empty")
        logger.info("Evaluation result: %s", trainer.evaluate())


if __name__ == "__main__":
    main()
