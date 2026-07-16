#!/usr/bin/env bash
set -euo pipefail

# Fair router comparison:
# - Run lexical and random routers on the same selected examples.
# - Do not pre-filter hit/miss samples during generation.
# - Merge script recomputes common-sample overall, hit, miss, and paired hit-outcome metrics.

export MODEL_PATH="${MODEL_PATH:-./checkpoints/qwen3-4b-agent-tooldef-npu}"
export BASE_MODEL="${BASE_MODEL:-./models/Qwen3-4B-Instruct-2507}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
export DATASET_PATH="${DATASET_PATH:-./datasets/agent-llm-traces}"
export OUTPUT_FILE="${OUTPUT_FILE:-./outputs/agent_tooldef_hybrid_router_fair_eval_npu.jsonl}"

export HYBRID_CASES="${HYBRID_CASES:-3:4}"
export ROUTER_STRATEGIES="${ROUTER_STRATEGIES:-lexical,random}"
export ROUTER_HIT_FILTER="${ROUTER_HIT_FILTER:-all}"
export ROUTER_SCOPE="${ROUTER_SCOPE:-last_user}"
export ROUTER_SEED="${ROUTER_SEED:-42}"
export MAX_EXAMPLES="${MAX_EXAMPLES:-106}"

bash agent/eval_agent_tool_definition_hybrid_router_npu.sh
