#!/usr/bin/env bash
set -euo pipefail

# Compare lexical vs attention-routed hybrid on cards 0,1:
# - Full: all tool definitions are full-prefilled.
# - C2KV: all tool definitions are C2KV-compressed.
# - Lex-hybrid: lexical top-k tools are full, rest tools are C2KV.
# - Att-hybrid: attention-routed top-k tools are full, rest tools are C2KV.
# - FullKV-att-hybrid: attention-routed top-k from full tool KV, rest tools are C2KV.
# - Lex-att-hybrid: lexical candidate pool reranked by attention.

export MODEL_PATH="${MODEL_PATH:-./checkpoints/qwen3-4b-agent-tooldef-npu}"
export BASE_MODEL="${BASE_MODEL:-./models/Qwen3-4B-Instruct-2507}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
export DATASET_PATH="${DATASET_PATH:-./datasets/agent-llm-traces}"
export OUTPUT_FILE="${OUTPUT_FILE:-./outputs/ablation_0726_full_c2kv_lex_att_fullkv_att_mean_hybrid_common.jsonl}"
export SPLIT="${SPLIT:-eval}"

export HYBRID_CASES="${HYBRID_CASES:-3:4}"
export ROUTER_HIT_FILTER="${ROUTER_HIT_FILTER:-all}"
export ROUTER_SCOPE="${ROUTER_SCOPE:-last_user}"
export ROUTER_SEED="${ROUTER_SEED:-42}"
export MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
export SELECTION_FILTER="${SELECTION_FILTER:-c2kv}"
export MIN_NUM_TOOLS="${MIN_NUM_TOOLS:-4}"
export SPLIT_MANIFEST_FILE="${SPLIT_MANIFEST_FILE:-}"
export SPLIT_NAME="${SPLIT_NAME:-toolset_disjoint}"

export MAX_DOC_LENGTH="${MAX_DOC_LENGTH:-1024}"
export MAX_DOC_NUM="${MAX_DOC_NUM:-10}"
export MAX_TOOL_DEFINITION_TOKENS="${MAX_TOOL_DEFINITION_TOKENS:-10000}"
export MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-4096}"
export MAX_BASELINE_INPUT_TOKENS="${MAX_BASELINE_INPUT_TOKENS:-0}"
export MAX_HYBRID_DECODE_TOKENS="${MAX_HYBRID_DECODE_TOKENS:-0}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
export NPU_ATTN_IMPL="${NPU_ATTN_IMPL:-eager}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-max_split_size_mb:512}"

export ATTENTION_ROUTER_LAYERS="${ATTENTION_ROUTER_LAYERS:-4}"
export ATTENTION_ROUTER_ATTN_IMPL="${ATTENTION_ROUTER_ATTN_IMPL:-eager}"
export ATTENTION_ROUTER_MAX_QUERY_TOKENS="${ATTENTION_ROUTER_MAX_QUERY_TOKENS:-512}"
export ATTENTION_ROUTER_SCORE_MODE="${ATTENTION_ROUTER_SCORE_MODE:-mean}"
export ATTENTION_ROUTER_CACHE_MODE="${ATTENTION_ROUTER_CACHE_MODE:-c2kv}"
export ATTENTION_ROUTER_LEXICAL_POOL="${ATTENTION_ROUTER_LEXICAL_POOL:-10}"

export DEBUG_HYBRID_TOKENS="${DEBUG_HYBRID_TOKENS:-True}"
export DUMP_HYBRID_DEFINITIONS="${DUMP_HYBRID_DEFINITIONS:-False}"
export DEBUG_DEFINITION_CHARS="${DEBUG_DEFINITION_CHARS:-4000}"

BASELINE_DEVICES="${BASELINE_DEVICES:-0,1}"
LEXICAL_DEVICE="${LEXICAL_DEVICE:-0}"
ATTENTION_DEVICE="${ATTENTION_DEVICE:-1}"
LEX_ATTENTION_DEVICE="${LEX_ATTENTION_DEVICE:-0}"
FULLKV_ATTENTION_DEVICE="${FULLKV_ATTENTION_DEVICE:-1}"

OUTPUT_STEM="${OUTPUT_FILE%.jsonl}"
RUN_DIR="${RUN_DIR:-${OUTPUT_STEM}.runs}"
mkdir -p "${RUN_DIR}"

BASELINE_OUTPUT="${RUN_DIR}/full_c2kv.jsonl"
LEXICAL_OUTPUT="${RUN_DIR}/c2kv_hybrid.jsonl"
ATTENTION_OUTPUT="${RUN_DIR}/att_hybrid.jsonl"
LEX_ATTENTION_OUTPUT="${RUN_DIR}/lex_att_hybrid.jsonl"
FULLKV_ATTENTION_OUTPUT="${RUN_DIR}/att_fullkv_hybrid.jsonl"

echo "MODEL_PATH=${MODEL_PATH}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "TOKENIZER_PATH=${TOKENIZER_PATH}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "OUTPUT_FILE=${OUTPUT_FILE}"
echo "SPLIT=${SPLIT}"
echo "MAX_EXAMPLES=${MAX_EXAMPLES}"
echo "HYBRID_CASES=${HYBRID_CASES}"
echo "SELECTION_FILTER=${SELECTION_FILTER}"
echo "MIN_NUM_TOOLS=${MIN_NUM_TOOLS}"
echo "BASELINE_DEVICES=${BASELINE_DEVICES}"
echo "LEXICAL_DEVICE=${LEXICAL_DEVICE}"
echo "ATTENTION_DEVICE=${ATTENTION_DEVICE}"
echo "LEX_ATTENTION_DEVICE=${LEX_ATTENTION_DEVICE}"
echo "FULLKV_ATTENTION_DEVICE=${FULLKV_ATTENTION_DEVICE}"
echo "ATTENTION_ROUTER_LAYERS=${ATTENTION_ROUTER_LAYERS}"
echo "ATTENTION_ROUTER_MAX_QUERY_TOKENS=${ATTENTION_ROUTER_MAX_QUERY_TOKENS}"
echo "ATTENTION_ROUTER_SCORE_MODE=${ATTENTION_ROUTER_SCORE_MODE}"
echo "ATTENTION_ROUTER_CACHE_MODE=${ATTENTION_ROUTER_CACHE_MODE}"
echo "ATTENTION_ROUTER_LEXICAL_POOL=${ATTENTION_ROUTER_LEXICAL_POOL}"

PIDS=()

echo "[launch] Full + C2KV on ${BASELINE_DEVICES}"
(
  export ASCEND_RT_VISIBLE_DEVICES="${BASELINE_DEVICES}"
  export OUTPUT_FILE="${BASELINE_OUTPUT}"
  export TMP_DIR="${RUN_DIR}/full_c2kv.parts"
  export COMPARE_MODES="full,c2kv"
  export RATIOS="4"
  export PARALLEL_EVAL=True
  bash agent/eval_agent_tool_definition_reuse_baselines_npu.sh
) > "${RUN_DIR}/full_c2kv.log" 2>&1 &
PIDS+=("$!")

for pid in "${PIDS[@]}"; do
  wait "${pid}"
done

PIDS=()

echo "[launch] Lex-hybrid on ${LEXICAL_DEVICE}"
(
  export ASCEND_RT_VISIBLE_DEVICES="${LEXICAL_DEVICE}"
  export OUTPUT_FILE="${LEXICAL_OUTPUT}"
  export TMP_DIR="${RUN_DIR}/c2kv_hybrid.parts"
  export HYBRID_MODES="hybrid"
  export ROUTER_STRATEGIES="lexical"
  export TOP_SCHEMA_MODE="full"
  export PARALLEL_EVAL=True
  bash agent/eval_agent_tool_definition_hybrid_router_npu.sh
) > "${RUN_DIR}/c2kv_hybrid.log" 2>&1 &
PIDS+=("$!")

echo "[launch] Att-hybrid on ${ATTENTION_DEVICE}"
(
  export ASCEND_RT_VISIBLE_DEVICES="${ATTENTION_DEVICE}"
  export OUTPUT_FILE="${ATTENTION_OUTPUT}"
  export TMP_DIR="${RUN_DIR}/att_hybrid.parts"
  export HYBRID_MODES="hybrid"
  export ROUTER_STRATEGIES="attention"
  export TOP_SCHEMA_MODE="full"
  export PARALLEL_EVAL=True
  bash agent/eval_agent_tool_definition_hybrid_router_npu.sh
) > "${RUN_DIR}/att_hybrid.log" 2>&1 &
PIDS+=("$!")

for pid in "${PIDS[@]}"; do
  wait "${pid}"
done

echo "[launch] Lex-att-hybrid on ${LEX_ATTENTION_DEVICE}"
(
  export ASCEND_RT_VISIBLE_DEVICES="${LEX_ATTENTION_DEVICE}"
  export OUTPUT_FILE="${LEX_ATTENTION_OUTPUT}"
  export TMP_DIR="${RUN_DIR}/lex_att_hybrid.parts"
  export HYBRID_MODES="hybrid"
  export ROUTER_STRATEGIES="lex_attention"
  export TOP_SCHEMA_MODE="full"
  export PARALLEL_EVAL=True
  bash agent/eval_agent_tool_definition_hybrid_router_npu.sh
) > "${RUN_DIR}/lex_att_hybrid.log" 2>&1

echo "[launch] FullKV-att-hybrid on ${FULLKV_ATTENTION_DEVICE}"
(
  export ASCEND_RT_VISIBLE_DEVICES="${FULLKV_ATTENTION_DEVICE}"
  export OUTPUT_FILE="${FULLKV_ATTENTION_OUTPUT}"
  export TMP_DIR="${RUN_DIR}/att_fullkv_hybrid.parts"
  export HYBRID_MODES="hybrid"
  export ROUTER_STRATEGIES="attention"
  export ATTENTION_ROUTER_CACHE_MODE="full"
  export TOP_SCHEMA_MODE="full"
  export PARALLEL_EVAL=True
  bash agent/eval_agent_tool_definition_hybrid_router_npu.sh
) > "${RUN_DIR}/att_fullkv_hybrid.log" 2>&1

python agent/merge_agent_tool_definition_reuse_baselines_eval.py \
  --output_file "${OUTPUT_FILE}" \
  --model "${MODEL_PATH}" \
  --base_model "${BASE_MODEL}" \
  --reuse_model "${BASE_MODEL}" \
  --dataset_path "${DATASET_PATH}" \
  --split "${SPLIT}" \
  --modes "full,c2kv,c2kv_hybrid,att_hybrid_${ATTENTION_ROUTER_SCORE_MODE},att_fullkv_hybrid_${ATTENTION_ROUTER_SCORE_MODE},lex_att_hybrid_${ATTENTION_ROUTER_SCORE_MODE}" \
  --ratios "1,4" \
  --input_files \
    "${BASELINE_OUTPUT}" \
    "${LEXICAL_OUTPUT}" \
    "${ATTENTION_OUTPUT}" \
    "${FULLKV_ATTENTION_OUTPUT}" \
    "${LEX_ATTENTION_OUTPUT}"

echo "Done."
echo "Merged rows: ${OUTPUT_FILE}"
echo "Summary: ${OUTPUT_STEM}.summary.json"
echo "Logs are under ${RUN_DIR}/"
