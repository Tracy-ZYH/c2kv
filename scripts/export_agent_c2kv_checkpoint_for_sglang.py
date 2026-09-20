#!/usr/bin/env python3
"""Export an adapter-style Agent History C2KV checkpoint for SGLang serving.

The 60k trainer stores only trainable gist/C2KV weights in c2kv_adapter.bin.
SGLang's Qwen3 loader expects those tensors to appear in the model-path weight
iterator alongside the frozen base weights.  This script creates a lightweight
serving directory by symlinking the base Qwen files, writing the adapter as one
safetensors shard, and extending model.safetensors.index.json to include the
gist tensors.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


PROFILE_FILENAME = "c2kv_checkpoint_profile.json"
ADAPTER_SHARD = "c2kv_adapter.safetensors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-doc-length", type=int, default=1024)
    parser.add_argument("--max-doc-num", type=int, default=16)
    parser.add_argument("--compression-ratios", default="2,4,8")
    parser.add_argument("--query-projection", choices=("base", "gist"), default="base")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    base_model = args.base_model.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    adapter_bin = checkpoint / "c2kv_adapter.bin"
    base_index = base_model / "model.safetensors.index.json"
    if not adapter_bin.is_file():
        raise FileNotFoundError(f"missing adapter checkpoint: {adapter_bin}")
    if not base_index.is_file():
        raise FileNotFoundError(f"missing base safetensors index: {base_index}")
    if output.exists() and any(output.iterdir()) and not args.force:
        raise FileExistsError(f"{output} already exists and is not empty; pass --force to update")
    output.mkdir(parents=True, exist_ok=True)

    skip = {"model.safetensors.index.json"}
    for src in base_model.iterdir():
        if src.name in skip:
            continue
        dst = output / src.name
        if src.is_file():
            link_or_copy(src, dst)

    state = torch.load(adapter_bin, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"expected dict in {adapter_bin}, got {type(state).__name__}")
    adapter_state = {
        str(k): v.detach().cpu().contiguous()
        for k, v in state.items()
        if "gist" in str(k)
    }
    if not adapter_state:
        raise ValueError(f"no gist/C2KV tensors found in {adapter_bin}")
    save_file(adapter_state, output / ADAPTER_SHARD)

    index = json.loads(base_index.read_text(encoding="utf-8"))
    weight_map = dict(index.get("weight_map") or {})
    for name in sorted(adapter_state):
        weight_map[name] = ADAPTER_SHARD
    metadata = dict(index.get("metadata") or {})
    metadata["c2kv_adapter_tensors"] = len(adapter_state)
    metadata["c2kv_adapter_source"] = str(adapter_bin)
    index["metadata"] = metadata
    index["weight_map"] = weight_map
    (output / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    ratios = [int(item.strip()) for item in args.compression_ratios.split(",") if item.strip()]
    profile = {
        "schema_version": 1,
        "profile_kind": "exported_adapter",
        "model": {
            "initial_model": str(base_model),
            "adapter_checkpoint": str(checkpoint),
            "gist_param": "qkv",
            "gist_type": "dynamic-interleave",
        },
        "training": {
            "doc_mode": "history_only",
            "tools_in_system": True,
            "doc_packing": "turn",
            "max_doc_length": args.max_doc_length,
            "max_doc_num": args.max_doc_num,
            "compression_ratios": ratios,
            "history_selection": "all_completed_history_compressed",
        },
        "serving": {
            "compatible": True,
            "compatibility_reason": "base Qwen weights plus exported Agent History C2KV gist adapter",
            "query_projection": args.query_projection,
            "doc_packing": "turn",
            "max_doc_length": args.max_doc_length,
            "max_doc_num": args.max_doc_num,
            "compression_ratios": ratios,
        },
        "evaluation_surfaces": {
            "serving_e2e": {
                "compatible": True,
                "reason": "SGLang Qwen3 loader consumes gist_q/k/v tensors from the extended safetensors index",
            }
        },
        "artifacts": {
            "base_model": str(base_model),
            "adapter_bin": str(adapter_bin),
            "adapter_shard": str(output / ADAPTER_SHARD),
        },
    }
    (output / PROFILE_FILENAME).write_text(
        json.dumps(profile, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output),
        "adapter_tensors": len(adapter_state),
        "adapter_shard": str(output / ADAPTER_SHARD),
        "profile": str(output / PROFILE_FILENAME),
    }, indent=2))


if __name__ == "__main__":
    main()
