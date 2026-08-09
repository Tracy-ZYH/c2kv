#!/usr/bin/env bash
set -euo pipefail

# Full selection ablation for top-k routing:
# - Full:          all tool definitions are full-prefilled.
# - C2KV-All:      all tool definitions are C2KV-compressed.
# - C2KV-hybrid:   selected top-k tools are full, rest tools are C2KV.
# - Random-Hybrid: random top-k tools are full, rest tools are C2KV.
# - Drop-Selected: selected top-k tools are removed, rest tools are C2KV.
# - Top-K Only:    selected top-k tools are full, rest tools are removed.
#
# Defaults use cards 2-7: 2,3 for Full/C2KV-All and 4-7 for the four
# hybrid ablations. The final summary recomputes metrics on the common valid
# sample subset across all six groups.

export MODEL_PATH="${MODEL_PATH:-./checkpoints/qwen3-4b-agent-tooldef-npu}"
export BASE_MODEL="${BASE_MODEL:-./models/Qwen3-4B-Instruct-2507}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
export DATASET_PATH="${DATASET_PATH:-./datasets/agent-llm-traces}"
export OUTPUT_FILE="${OUTPUT_FILE:-./outputs/ablation_0725_full_c2kv_hybrid_top1_compact_common.jsonl}"
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
export TOOL_DOCUMENT_EVAL_MODE="${TOOL_DOCUMENT_EVAL_MODE:-full}"
export MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-4096}"
export MAX_BASELINE_INPUT_TOKENS="${MAX_BASELINE_INPUT_TOKENS:-0}"
export MAX_HYBRID_DECODE_TOKENS="${MAX_HYBRID_DECODE_TOKENS:-0}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
export NPU_ATTN_IMPL="${NPU_ATTN_IMPL:-eager}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-max_split_size_mb:512}"

export DEBUG_HYBRID_TOKENS="${DEBUG_HYBRID_TOKENS:-True}"
export DUMP_HYBRID_DEFINITIONS="${DUMP_HYBRID_DEFINITIONS:-False}"
export DEBUG_DEFINITION_CHARS="${DEBUG_DEFINITION_CHARS:-4000}"

BASELINE_DEVICES="${BASELINE_DEVICES:-2,3}"
HYBRID_DEVICE_LEXICAL="${HYBRID_DEVICE_LEXICAL:-4}"
HYBRID_DEVICE_RANDOM="${HYBRID_DEVICE_RANDOM:-5}"
HYBRID_DEVICE_DROP_SELECTED="${HYBRID_DEVICE_DROP_SELECTED:-6}"
HYBRID_DEVICE_TOPK_ONLY="${HYBRID_DEVICE_TOPK_ONLY:-7}"

OUTPUT_STEM="${OUTPUT_FILE%.jsonl}"
RUN_DIR="${RUN_DIR:-${OUTPUT_STEM}.runs}"
mkdir -p "${RUN_DIR}"

BASELINE_OUTPUT="${RUN_DIR}/full_c2kv.jsonl"
LEXICAL_OUTPUT="${RUN_DIR}/c2kv_hybrid.jsonl"
LEXICAL_TOP1_OUTPUT="${RUN_DIR}/c2kv_hybrid_top1.jsonl"
COMPACT_OUTPUT="${RUN_DIR}/c2kv_hybrid_compact.jsonl"
RANDOM_OUTPUT="${RUN_DIR}/random_hybrid.jsonl"
DROP_OUTPUT="${RUN_DIR}/drop_selected.jsonl"
TOPK_ONLY_OUTPUT="${RUN_DIR}/topk_only.jsonl"

echo "MODEL_PATH=${MODEL_PATH}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "TOKENIZER_PATH=${TOKENIZER_PATH}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "OUTPUT_FILE=${OUTPUT_FILE}"
echo "SPLIT=${SPLIT}"
echo "MAX_EXAMPLES=${MAX_EXAMPLES}"
echo "HYBRID_CASES=${HYBRID_CASES}"
echo "SELECTION_FILTER=${SELECTION_FILTER}"
echo "TOOL_DOCUMENT_EVAL_MODE=${TOOL_DOCUMENT_EVAL_MODE}"
echo "MIN_NUM_TOOLS=${MIN_NUM_TOOLS}"
echo "BASELINE_DEVICES=${BASELINE_DEVICES}"
echo "HYBRID_DEVICE_LEXICAL=${HYBRID_DEVICE_LEXICAL}"
echo "HYBRID_DEVICE_RANDOM=${HYBRID_DEVICE_RANDOM}"
echo "HYBRID_DEVICE_DROP_SELECTED=${HYBRID_DEVICE_DROP_SELECTED}"
echo "HYBRID_DEVICE_TOPK_ONLY=${HYBRID_DEVICE_TOPK_ONLY}"
echo "DEBUG_HYBRID_TOKENS=${DEBUG_HYBRID_TOKENS}"
echo "DUMP_HYBRID_DEFINITIONS=${DUMP_HYBRID_DEFINITIONS}"

PIDS=()

echo "[launch] Full + C2KV-All on ${BASELINE_DEVICES}"
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

echo "[launch] C2KV-hybrid lexical on ${HYBRID_DEVICE_LEXICAL}"
(
  export ASCEND_RT_VISIBLE_DEVICES="${HYBRID_DEVICE_LEXICAL}"
  export OUTPUT_FILE="${LEXICAL_OUTPUT}"
  export TMP_DIR="${RUN_DIR}/c2kv_hybrid.parts"
  export HYBRID_MODES="hybrid"
  export ROUTER_STRATEGIES="lexical"
  export TOP_SCHEMA_MODE="full"
  export PARALLEL_EVAL=True
  bash agent/eval_agent_tool_definition_hybrid_router_npu.sh
) > "${RUN_DIR}/c2kv_hybrid.log" 2>&1 &
PIDS+=("$!")

echo "[launch] Random-Hybrid on ${HYBRID_DEVICE_RANDOM}"
(
  export ASCEND_RT_VISIBLE_DEVICES="${HYBRID_DEVICE_RANDOM}"
  export OUTPUT_FILE="${RANDOM_OUTPUT}"
  export TMP_DIR="${RUN_DIR}/random_hybrid.parts"
  export HYBRID_MODES="hybrid"
  export ROUTER_STRATEGIES="random"
  export TOP_SCHEMA_MODE="full"
  export PARALLEL_EVAL=True
  bash agent/eval_agent_tool_definition_hybrid_router_npu.sh
) > "${RUN_DIR}/random_hybrid.log" 2>&1 &
PIDS+=("$!")

echo "[launch] Drop-Selected on ${HYBRID_DEVICE_DROP_SELECTED}"
(
  export ASCEND_RT_VISIBLE_DEVICES="${HYBRID_DEVICE_DROP_SELECTED}"
  export OUTPUT_FILE="${DROP_OUTPUT}"
  export TMP_DIR="${RUN_DIR}/drop_selected.parts"
  export HYBRID_MODES="drop_selected"
  export ROUTER_STRATEGIES="lexical"
  export TOP_SCHEMA_MODE="full"
  export PARALLEL_EVAL=True
  bash agent/eval_agent_tool_definition_hybrid_router_npu.sh
) > "${RUN_DIR}/drop_selected.log" 2>&1 &
PIDS+=("$!")

echo "[launch] Top-K Only on ${HYBRID_DEVICE_TOPK_ONLY}"
(
  export ASCEND_RT_VISIBLE_DEVICES="${HYBRID_DEVICE_TOPK_ONLY}"
  export OUTPUT_FILE="${TOPK_ONLY_OUTPUT}"
  export TMP_DIR="${RUN_DIR}/topk_only.parts"
  export HYBRID_MODES="topk_only"
  export ROUTER_STRATEGIES="lexical"
  export TOP_SCHEMA_MODE="full"
  export PARALLEL_EVAL=True
  bash agent/eval_agent_tool_definition_hybrid_router_npu.sh
) > "${RUN_DIR}/topk_only.log" 2>&1 &
PIDS+=("$!")

for pid in "${PIDS[@]}"; do
  wait "${pid}"
done

PIDS=()

echo "[launch] C2KV-hybrid top-1 lexical on ${HYBRID_DEVICE_LEXICAL}"
(
  export ASCEND_RT_VISIBLE_DEVICES="${HYBRID_DEVICE_LEXICAL}"
  export OUTPUT_FILE="${LEXICAL_TOP1_OUTPUT}"
  export TMP_DIR="${RUN_DIR}/c2kv_hybrid_top1.parts"
  export HYBRID_CASES="1:4"
  export HYBRID_MODES="hybrid"
  export ROUTER_STRATEGIES="lexical"
  export TOP_SCHEMA_MODE="full"
  export PARALLEL_EVAL=True
  bash agent/eval_agent_tool_definition_hybrid_router_npu.sh
) > "${RUN_DIR}/c2kv_hybrid_top1.log" 2>&1 &
PIDS+=("$!")

echo "[launch] C2KV-hybrid top-3 compact lexical on ${HYBRID_DEVICE_RANDOM}"
(
  export ASCEND_RT_VISIBLE_DEVICES="${HYBRID_DEVICE_RANDOM}"
  export OUTPUT_FILE="${COMPACT_OUTPUT}"
  export TMP_DIR="${RUN_DIR}/c2kv_hybrid_compact.parts"
  export HYBRID_CASES="3:4"
  export HYBRID_MODES="hybrid"
  export ROUTER_STRATEGIES="lexical"
  export TOP_SCHEMA_MODE="compact"
  export PARALLEL_EVAL=True
  bash agent/eval_agent_tool_definition_hybrid_router_npu.sh
) > "${RUN_DIR}/c2kv_hybrid_compact.log" 2>&1 &
PIDS+=("$!")

for pid in "${PIDS[@]}"; do
  wait "${pid}"
done

python agent/merge_agent_tool_definition_reuse_baselines_eval.py \
  --output_file "${OUTPUT_FILE}" \
  --model "${MODEL_PATH}" \
  --base_model "${BASE_MODEL}" \
  --reuse_model "${BASE_MODEL}" \
  --dataset_path "${DATASET_PATH}" \
  --split "${SPLIT}" \
  --tool_document_eval_mode "${TOOL_DOCUMENT_EVAL_MODE}" \
  --modes "full,c2kv,c2kv_hybrid,c2kv_hybrid_top1,c2kv_hybrid_compact,random_hybrid,drop_selected,topk_only" \
  --ratios "1,4" \
  --input_files \
    "${BASELINE_OUTPUT}" \
    "${LEXICAL_OUTPUT}" \
    "${LEXICAL_TOP1_OUTPUT}" \
    "${COMPACT_OUTPUT}" \
    "${RANDOM_OUTPUT}" \
    "${DROP_OUTPUT}" \
    "${TOPK_ONLY_OUTPUT}"

echo "Done."
echo "Merged rows: ${OUTPUT_FILE}"
echo "Summary: ${OUTPUT_STEM}.summary.json"
echo "Logs are under ${RUN_DIR}/"
