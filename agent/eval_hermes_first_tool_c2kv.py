from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pyarrow.parquet as pq
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_agent_tool_definition_hybrid_router import _render_tool_definition  # noqa: E402
from eval_toolathlon_first_tool_c2kv import (  # noqa: E402
    _arg_f1s,
    _build_prefix,
    _clear_device_cache,
    _is_oom_error,
    _parse_pred_call,
    _target_answer,
)
from eval_agent_tool_definition_c2kv import (  # noqa: E402
    _generate_from_input_ids,
    _load_model,
    _normalize_text,
    _setup_device,
)
from train.train_data_multiturn import _chat_template_ids  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _json_loads(value: Any, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default if default is not None else value


def _tools(value: Any) -> List[Dict[str, Any]]:
    parsed = _json_loads(value, [])
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


def _function_payload(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    function = value.get("function") if isinstance(value.get("function"), dict) else {}
    name = value.get("name") or value.get("tool_name") or value.get("function_name") or function.get("name")
    arguments = value.get("arguments") or value.get("args") or function.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except Exception:
            pass
    if not name:
        return None
    return {"name": str(name), "arguments": arguments}


def _first_tool_call(conversations: List[Dict[str, Any]]) -> Optional[tuple[int, Dict[str, Any]]]:
    for index, message in enumerate(conversations):
        if not isinstance(message, dict) or message.get("from") != "gpt":
            continue
        value = str(message.get("value") or "")
        blocks = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", value, flags=re.S)
        for block in blocks:
            try:
                payload = _function_payload(json.loads(block))
            except Exception:
                payload = None
            if payload:
                return index, payload
    return None


def _system_prompt(conversations: List[Dict[str, Any]]) -> str:
    for message in conversations:
        if isinstance(message, dict) and message.get("from") == "system":
            value = str(message.get("value") or "").strip()
            if value:
                return (
                    value
                    + "\n\nWhen asked to predict or perform a tool call, output only:\n"
                    + "Action:\n<tool_call>\n{\"name\":\"tool_name\",\"arguments\":{}}\n</tool_call>"
                )
    return (
        "You are a helpful tool-calling assistant.\n\n"
        "When asked to predict or perform a tool call, output only:\n"
        "Action:\n<tool_call>\n{\"name\":\"tool_name\",\"arguments\":{}}\n</tool_call>"
    )


def _prompt_messages(conversations: List[Dict[str, Any]], first_tool_index: int) -> List[Dict[str, str]]:
    role_map = {
        "human": "user",
        "gpt": "assistant",
        "tool": "user",
    }
    messages: List[Dict[str, str]] = []
    for message in conversations[:first_tool_index]:
        if not isinstance(message, dict):
            continue
        role = message.get("from")
        if role == "system":
            continue
        mapped_role = role_map.get(str(role), "user")
        content = str(message.get("value") or "").strip()
        if not content:
            continue
        if role == "tool":
            content = "Tool response:\n" + content
        messages.append({"role": mapped_role, "content": content})
    messages.append({
        "role": "user",
        "content": (
            "Predict the next tool call for this trajectory. Output only:\n"
            "Action:\n<tool_call>\n{\"name\":\"tool_name\",\"arguments\":{}}\n</tool_call>"
        ),
    })
    return messages


def _query_text(task: str, prompt_messages: List[Dict[str, str]]) -> str:
    user_text = "\n".join(message["content"] for message in prompt_messages if message.get("role") == "user")
    return "\n".join(item for item in [task, user_text] if item)


def _iter_rows(data_dir: Path, configs: List[str], limit: Optional[int]) -> Iterable[Dict[str, Any]]:
    seen = 0
    for config in configs:
        path = data_dir / "data" / config / "train.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=256):
            for row in batch.to_pylist():
                row["_config"] = config
                yield row
                seen += 1
                if limit is not None and seen >= limit:
                    return


def _tool_doc_tokens(tokenizer: Any, tools: List[Dict[str, Any]]) -> int:
    return len(_chat_template_ids(tokenizer, [{"role": "user", "content": "Tool definitions:\n" + _render_tool_definition(tools)}]))


@torch.inference_mode()
def _generate_one(model: Any, tokenizer: Any, row: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    conversations = row.get("conversations") or []
    tools = _tools(row.get("tools"))
    first = _first_tool_call(conversations)
    if not tools or first is None:
        return {
            "id": row.get("id"),
            "config": row.get("_config"),
            "skipped": True,
            "skip_reason": "missing_tools_or_first_call",
        }
    first_index, target_payload = first
    prompt_messages = _prompt_messages(conversations, first_index)
    if not prompt_messages:
        return {
            "id": row.get("id"),
            "config": row.get("_config"),
            "skipped": True,
            "skip_reason": "empty_prompt",
        }
    tool_tokens = _tool_doc_tokens(tokenizer, tools)
    if tool_tokens > args.max_tool_definition_tokens:
        return {
            "id": row.get("id"),
            "config": row.get("_config"),
            "method": args.method,
            "ratio": args.ratio,
            "skipped": True,
            "skip_reason": f"tool_definition_tokens>{args.max_tool_definition_tokens}",
            "tool_definition_tokens": tool_tokens,
        }
    if args.method == "full" and args.max_full_tool_tokens is not None and tool_tokens > args.max_full_tool_tokens:
        return {
            "id": row.get("id"),
            "config": row.get("_config"),
            "method": args.method,
            "ratio": args.ratio,
            "skipped": True,
            "skip_reason": f"full_tool_tokens>{args.max_full_tool_tokens}",
            "tool_definition_tokens": tool_tokens,
        }

    query = _query_text(str(row.get("task") or ""), prompt_messages)
    prefix, skip_reason = _build_prefix(model, tokenizer, _system_prompt(conversations), tools, query, args)
    if prefix is None:
        return {
            "id": row.get("id"),
            "config": row.get("_config"),
            "method": args.method,
            "ratio": args.ratio,
            "skipped": True,
            "skip_reason": skip_reason,
            "tool_definition_tokens": tool_tokens,
        }

    prompt_ids = _chat_template_ids(tokenizer, prompt_messages, add_generation_prompt=True)
    if len(prompt_ids) > args.max_prompt_tokens:
        prompt_ids = prompt_ids[-args.max_prompt_tokens :]
    prompt_input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
    mock_cache_ids = prompt_input_ids.new_zeros((1, prefix["cache_length"]))
    input_ids = torch.cat([mock_cache_ids, prompt_input_ids], dim=1)
    position_ids = torch.arange(
        prefix["system_length"] + prefix["tool_length"],
        prefix["system_length"] + prefix["tool_length"] + prompt_input_ids.shape[1],
        dtype=torch.long,
        device=model.device,
    ).unsqueeze(0)
    prediction, generate_sec, generated_tokens, tbt_sec = _generate_from_input_ids(
        model,
        tokenizer,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        attn_impl=args.generate_attn_impl,
        use_gist=prefix["use_gist"],
        position_ids=position_ids,
        past_key_values=prefix["cache"],
    )
    target = _target_answer(target_payload)
    pred_payload = _parse_pred_call(prediction)
    arg_name_f1, arg_value_f1 = _arg_f1s(target_payload, pred_payload)
    online_ttft = prefix["system_prefill_sec"] + prefix["full_prefill_sec"] + prefix["tool_compress_sec"] + prefix["blend_sec"]
    cached_ttft = prefix["system_prefill_sec"] + prefix["full_prefill_sec"] + prefix["blend_sec"]
    return {
        "id": row.get("id"),
        "config": row.get("_config"),
        "category": row.get("category"),
        "subcategory": row.get("subcategory"),
        "method": args.method,
        "ratio": args.ratio,
        "skipped": False,
        "target_tool_name": target_payload["name"],
        "prediction_tool_name": pred_payload.get("name") if pred_payload else None,
        "first_tool_accuracy": bool(pred_payload and pred_payload.get("name") == target_payload["name"]),
        "tool_call_rate": pred_payload is not None,
        "argument_name_f1": arg_name_f1,
        "argument_value_f1": arg_value_f1,
        "exact_match": _normalize_text(prediction) == _normalize_text(target),
        "tool_definition_tokens": prefix["tool_definition_tokens"],
        "tools_count": len(tools),
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": generated_tokens,
        "actual_compression_ratio": round(prefix["compression_ratio"], 4),
        "system_prefill_sec": round(prefix["system_prefill_sec"], 4),
        "full_prefill_sec": round(prefix["full_prefill_sec"], 4),
        "tool_compress_sec": round(prefix["tool_compress_sec"], 4),
        "blend_sec": round(prefix["blend_sec"], 4),
        "online_ttft_sec": round(online_ttft, 4),
        "cached_ttft_sec": round(cached_ttft, 4),
        "generate_sec": round(generate_sec, 4),
        "tbt_sec": round(tbt_sec, 6),
        "online_total_sec": round(online_ttft + generate_sec, 4),
        "cached_total_sec": round(cached_ttft + generate_sec, 4),
        "prediction": prediction,
        "target": target,
    }


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [row for row in rows if not row.get("skipped")]
    skips = Counter(row.get("skip_reason", "unknown") for row in rows if row.get("skipped"))
    total_generated = sum(row.get("generated_tokens", 0) for row in valid)
    return {
        "method": valid[0].get("method") if valid else None,
        "ratio": valid[0].get("ratio") if valid else None,
        "num_examples": len(rows),
        "num_valid": len(valid),
        "num_skipped": len(rows) - len(valid),
        "skip_reasons": dict(skips),
        "first_tool_accuracy": sum(1 for row in valid if row["first_tool_accuracy"]) / len(valid) if valid else 0.0,
        "tool_call_rate": sum(1 for row in valid if row["tool_call_rate"]) / len(valid) if valid else 0.0,
        "argument_name_f1": sum(row["argument_name_f1"] for row in valid) / len(valid) if valid else 0.0,
        "argument_value_f1": sum(row["argument_value_f1"] for row in valid) / len(valid) if valid else 0.0,
        "exact_match": sum(1 for row in valid if row["exact_match"]) / len(valid) if valid else 0.0,
        "avg_tool_definition_tokens": sum(row["tool_definition_tokens"] for row in valid) / len(valid) if valid else 0.0,
        "avg_tools_count": sum(row["tools_count"] for row in valid) / len(valid) if valid else 0.0,
        "avg_prompt_tokens": sum(row["prompt_tokens"] for row in valid) / len(valid) if valid else 0.0,
        "avg_generated_tokens": total_generated / len(valid) if valid else 0.0,
        "avg_actual_compression_ratio": sum(row["actual_compression_ratio"] for row in valid) / len(valid) if valid else 0.0,
        "avg_online_ttft_sec": sum(row["online_ttft_sec"] for row in valid) / len(valid) if valid else 0.0,
        "avg_cached_ttft_sec": sum(row["cached_ttft_sec"] for row in valid) / len(valid) if valid else 0.0,
        "avg_generate_sec": sum(row["generate_sec"] for row in valid) / len(valid) if valid else 0.0,
        "avg_tbt_sec": sum(row["tbt_sec"] for row in valid) / len(valid) if valid else 0.0,
        "token_weighted_tbt_sec": sum(row["generate_sec"] for row in valid) / total_generated if total_generated else 0.0,
        "avg_online_total_sec": sum(row["online_total_sec"] for row in valid) / len(valid) if valid else 0.0,
        "avg_cached_total_sec": sum(row["cached_total_sec"] for row in valid) / len(valid) if valid else 0.0,
    }


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    device = _setup_device(args.device_type)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model,
        trust_remote_code=True,
        local_files_only=True,
        padding_side="right",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model_args = copy.copy(args)
    model_args.mode = "c2kv"
    model_args.untrained_c2kv = args.method == "c2kv_untrained"
    model = _load_model(model_args, tokenizer, device)

    rows: List[Dict[str, Any]] = []
    configs = [item.strip() for item in args.configs.split(",") if item.strip()]
    iterator = _iter_rows(Path(args.data_dir), configs, args.scan_limit)
    for row in tqdm(iterator, desc=f"hermes-{args.method}"):
        try:
            rows.append(_generate_one(model, tokenizer, row, args))
        except RuntimeError as error:
            if not _is_oom_error(error):
                raise
            logger.warning("Skipping Hermes sample after OOM: id=%s method=%s", row.get("id"), args.method)
            rows.append({
                "id": row.get("id"),
                "config": row.get("_config"),
                "method": args.method,
                "ratio": args.ratio,
                "skipped": True,
                "skip_reason": "oom",
            })
            _clear_device_cache(device)
        else:
            _clear_device_cache(device)
        if args.max_examples is not None and len(rows) >= args.max_examples:
            break

    summary = {
        "model": args.model,
        "data_dir": args.data_dir,
        "configs": configs,
        "result": _summarize(rows),
    }
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        output_path.with_suffix(".summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Hermes first tool call with Tool-C2KV variants.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base_model")
    parser.add_argument("--tokenizer")
    parser.add_argument("--data_dir", default="./datasets/hermes-agent-reasoning-traces")
    parser.add_argument("--configs", default="kimi,glm-5.1")
    parser.add_argument("--output_file", default="./outputs/hermes_first_tool_eval.jsonl")
    parser.add_argument("--method", choices=["full", "truncate", "c2kv", "c2kv_untrained"], default="full")
    parser.add_argument("--ratio", type=int, default=4)
    parser.add_argument("--max_examples", type=int, default=100)
    parser.add_argument("--scan_limit", type=int)
    parser.add_argument("--max_tool_definition_tokens", type=int, default=10000)
    parser.add_argument("--max_full_tool_tokens", type=int, default=10000)
    parser.add_argument("--max_doc_length", type=int, default=1024)
    parser.add_argument("--max_doc_num", type=int, default=10)
    parser.add_argument("--max_prompt_tokens", type=int, default=2048)
    parser.add_argument("--max_system_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=128)
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
    print(json.dumps(evaluate(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
