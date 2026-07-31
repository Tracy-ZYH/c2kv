from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence

import torch
from transformers import AutoTokenizer

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_agent_tool_definition_c2kv import (  # noqa: E402
    _build_tool_cache,
    _build_tool_chunks,
    _load_model,
    _prefill_system,
    _setup_device,
)
from eval_agent_tool_definition_hybrid_router import (  # noqa: E402
    _build_full_tool_cache_with_spans,
    _clear_device_cache,
    _gist_spans_from_tool_lengths,
    _rank_in_tool_order,
    _rank_tools_by_attention,
    _render_tool_definition,
    _tool_name,
    _tool_token_lengths,
)
from train.train_data_multiturn import _chat_template_ids  # noqa: E402


def _smoke_tools() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_flight",
                "description": "Search for flights between cities.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "origin": {"type": "string"},
                        "destination": {"type": "string"},
                    },
                    "required": ["origin", "destination"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "cancel_order",
                "description": "Cancel an existing customer order.",
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                },
            },
        },
    ]


def _router_args(args: argparse.Namespace, cache_mode: str) -> argparse.Namespace:
    return SimpleNamespace(
        model=args.model,
        base_model=args.base_model,
        mode="c2kv",
        untrained_c2kv=False,
        dtype=args.dtype,
        baseline_model_class="gist",
        system_attn_impl=args.attn_impl,
        gist_attn_impl=args.attn_impl,
        generate_attn_impl=args.attn_impl,
        max_system_length=args.max_system_length,
        max_doc_length=args.max_doc_length,
        max_doc_num=args.max_doc_num,
        max_tool_definition_tokens=args.max_tool_definition_tokens,
        truncate_tool_definition=False,
        router_scope="last_user",
        attention_router_layers=args.attention_router_layers,
        attention_router_attn_impl=args.attention_router_attn_impl,
        attention_router_max_query_tokens=args.attention_router_max_query_tokens,
        attention_router_score_mode=args.attention_router_score_mode,
        attention_router_cache_mode=cache_mode,
    )


@torch.inference_mode()
def _rank_tools_by_attention_heads(
    model: Any,
    tokenizer: Any,
    example: Any,
    args: argparse.Namespace,
    tools: Sequence[Dict[str, Any]],
    ratio: int,
    target_tool: str,
) -> Dict[str, Any]:
    system_ids = _chat_template_ids(
        tokenizer,
        [{"role": "system", "content": example.system_prompt}],
        keep_bos=True,
        max_length=args.max_system_length,
    )
    system_input_ids = torch.tensor([system_ids], dtype=torch.long, device=model.device)
    system_cache, system_length, _ = _prefill_system(
        model,
        system_input_ids,
        args.system_attn_impl,
    )

    if args.attention_router_cache_mode == "full":
        prefix_cache, tool_length, spans, _ = _build_full_tool_cache_with_spans(
            model,
            tokenizer,
            tools,
            system_cache,
            system_length,
            args.attention_router_attn_impl,
        )
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
        )
        if context_input_ids is None:
            raise ValueError(f"attention_router_{skip_reason}")
        prefix_cache, tool_length, gist_tokens, _, _, _ = _build_tool_cache(
            model,
            context_input_ids,
            system_cache,
            system_length,
            args.gist_attn_impl,
            ratio,
        )
        spans = _gist_spans_from_tool_lengths(_tool_token_lengths(tokenizer, tools), gist_tokens)
        tool_key_tokens = gist_tokens
        use_gist_for_query = True

    query_ids = _chat_template_ids(
        tokenizer,
        example.input_messages,
        add_generation_prompt=True,
    )
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

    tool_names = [_tool_name(tool) for tool in tools]
    per_head: List[Dict[str, Any]] = []

    def make_hook(layer_index: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            attn_weights = output[1] if isinstance(output, tuple) and len(output) > 1 else None
            if attn_weights is None:
                return
            cache_attn = attn_weights[0, :, :, system_length : system_length + tool_key_tokens].float()
            for head_index in range(cache_attn.shape[0]):
                head_attn = cache_attn[head_index]
                scores = []
                for start, end in spans:
                    if end <= start:
                        scores.append(0.0)
                        continue
                    tool_attn = head_attn[:, start:end]
                    if args.attention_router_score_mode == "sum":
                        score = tool_attn.sum(dim=-1).mean()
                    elif args.attention_router_score_mode == "sqrt_len":
                        score = tool_attn.sum(dim=-1).mean() / ((end - start) ** 0.5)
                    else:
                        score = tool_attn.mean()
                    scores.append(float(score.item()))
                ranked = sorted(range(len(tools)), key=lambda index: (-scores[index], index))
                per_head.append({
                    "layer": layer_index,
                    "head": head_index,
                    "target_rank": _rank_in_tool_order(tools, ranked, target_tool),
                    "target_score": (
                        round(scores[tool_names.index(target_tool)], 8)
                        if target_tool in tool_names else None
                    ),
                    "top_tool": tool_names[ranked[0]] if ranked else None,
                    "top_score": round(scores[ranked[0]], 8) if ranked else None,
                    "ranked_tool_names": [tool_names[index] for index in ranked],
                    "scores_by_rank": [round(scores[index], 8) for index in ranked],
                })
        return hook

    num_layers = len(model.model.layers)
    last_layers = max(1, min(args.attention_router_layers, num_layers))
    layer_indices = list(range(num_layers - last_layers, num_layers))
    handles = [
        model.model.layers[index].self_attn.register_forward_hook(make_hook(index))
        for index in layer_indices
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

    target_top1_heads = [
        item for item in per_head
        if item.get("target_rank") == 1
    ]
    return {
        "cache_mode": args.attention_router_cache_mode,
        "layers": layer_indices,
        "num_heads_checked": len(per_head),
        "num_target_top1_heads": len(target_top1_heads),
        "has_target_top1_head": bool(target_top1_heads),
        "target_top1_heads": target_top1_heads,
        "per_layer_head_ranks": sorted(
            per_head,
            key=lambda item: (
                item.get("target_rank") if item.get("target_rank") is not None else 10**9,
                -float(item.get("target_score") or 0.0),
                item.get("layer"),
                item.get("head"),
            ),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test attention tool routing on a 3-tool example.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base_model")
    parser.add_argument("--tokenizer")
    parser.add_argument("--device_type", choices=["auto", "cuda", "npu", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--ratio", type=int, default=4)
    parser.add_argument("--cache_modes", default="full")
    parser.add_argument("--target_tool", default="search_flight")
    parser.add_argument("--query", default="Please call the search_flight tool.")
    parser.add_argument("--attn_impl", default="eager")
    parser.add_argument("--attention_router_attn_impl", default="eager")
    parser.add_argument("--attention_router_layers", type=int, default=4)
    parser.add_argument("--attention_router_max_query_tokens", type=int, default=512)
    parser.add_argument(
        "--attention_router_score_mode",
        choices=["mean", "sqrt_len", "sum"],
        default="mean",
    )
    parser.add_argument("--max_system_length", type=int, default=256)
    parser.add_argument("--max_doc_length", type=int, default=1024)
    parser.add_argument("--max_doc_num", type=int, default=10)
    parser.add_argument("--max_tool_definition_tokens", type=int, default=10000)
    parser.add_argument(
        "--debug_heads",
        type=lambda value: str(value).lower() in {"1", "true", "yes"},
        default=True,
        help="Also print per-layer per-head target ranks.",
    )
    args = parser.parse_args()

    device = _setup_device(args.device_type)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model,
        trust_remote_code=True,
        local_files_only=True,
        padding_side="right",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model_args = _router_args(args, "c2kv")
    model = _load_model(model_args, tokenizer, device)

    tools = _smoke_tools()
    example = SimpleNamespace(
        qid="attention_router_smoke",
        session_id="attention_router_smoke",
        system_prompt="You are a tool-calling assistant. Select the requested tool.",
        input_messages=[{"role": "user", "content": args.query}],
    )

    results = {}
    for cache_mode in [item.strip() for item in args.cache_modes.split(",") if item.strip()]:
        ranked, scores, elapsed, _ = _rank_tools_by_attention(
            model,
            tokenizer,
            example,
            _router_args(args, cache_mode),
            tools,
            args.ratio,
        )
        names = [_tool_name(tools[index]) for index in ranked]
        results[cache_mode] = {
            "ranked_tool_names": names,
            "scores_by_rank": [round(scores[index], 8) for index in ranked],
            "target_tool": args.target_tool,
            "target_rank": _rank_in_tool_order(tools, ranked, args.target_tool),
            "target_top1": names[:1] == [args.target_tool],
            "attention_router_sec": round(elapsed, 4),
        }
        if args.debug_heads:
            results[cache_mode]["head_debug"] = _rank_tools_by_attention_heads(
                model,
                tokenizer,
                example,
                _router_args(args, cache_mode),
                tools,
                args.ratio,
                args.target_tool,
            )

    print(json.dumps({
        "query": args.query,
        "score_mode": args.attention_router_score_mode,
        "results": results,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
