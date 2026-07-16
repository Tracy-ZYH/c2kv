#!/usr/bin/env bash
set -euo pipefail

# Selection ablations for top-k routing:
# - hybrid:        selected top-k tools are full, rest tools are C2KV.
# - drop_selected: selected top-k tools are removed, rest tools are C2KV.
# - topk_only:     selected top-k tools are full, rest tools are removed.
#
# Run with ROUTER_STRATEGIES=lexical,random to compare router-selected top-k
# against random top-k on the same selected examples.

export MODEL_PATH="${MODEL_PATH:-./checkpoints/qwen3-4b-agent-tooldef-npu}"
export BASE_MODEL="${BASE_MODEL:-./models/Qwen3-4B-Instruct-2507}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
export DATASET_PATH="${DATASET_PATH:-./datasets/agent-llm-traces}"
export OUTPUT_FILE="${OUTPUT_FILE:-./outputs/agent_tooldef_selection_ablation_eval_npu.jsonl}"

export HYBRID_CASES="${HYBRID_CASES:-3:4}"
export HYBRID_MODES="${HYBRID_MODES:-hybrid,drop_selected,topk_only}"
export ROUTER_STRATEGIES="${ROUTER_STRATEGIES:-lexical,random}"
export ROUTER_HIT_FILTER="${ROUTER_HIT_FILTER:-all}"
export ROUTER_SCOPE="${ROUTER_SCOPE:-last_user}"
export ROUTER_SEED="${ROUTER_SEED:-42}"
export MAX_EXAMPLES="${MAX_EXAMPLES:-106}"

bash agent/eval_agent_tool_definition_hybrid_router_npu.sh
