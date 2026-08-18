#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="$(pwd)/python:$(pwd)/python/inference:$(pwd)/agent:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-4,5,6,7}"
export ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-max_split_size_mb:128}"

MODEL_PATH="${MODEL_PATH:-./checkpoints/qwen3-4b-mixed-mdoc-c2kv-r16-npu-12k/checkpoint-1125}"
BASE_MODEL="${BASE_MODEL:-./models/Qwen3-4B-Instruct-2507}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
DATASET="${DATASET:-wikimqa}"
DATASET_PATH="${DATASET_PATH:-./datasets/longbench_2wikimqa_test}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/mdoc_r16_wikimqa_direct}"
MAX_EXAMPLES="${MAX_EXAMPLES:-200}"
RATIO="${RATIO:-16}"
HYBRID_TOP_K="${HYBRID_TOP_K:-3}"
MAX_DOC_LENGTH="${MAX_DOC_LENGTH:-2048}"
MAX_DOC_NUM="${MAX_DOC_NUM:-0}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-0}"
MAX_QUERY_TOKENS="${MAX_QUERY_TOKENS:-1024}"
DOC_SELECTION="${DOC_SELECTION:-head}"
SYSTEM_ATTN_IMPL="${SYSTEM_ATTN_IMPL:-eager}"
GIST_ATTN_IMPL="${GIST_ATTN_IMPL:-npu_fusion_attention}"
GENERATE_ATTN_IMPL="${GENERATE_ATTN_IMPL:-npu_fusion_attention}"
DTYPE="${DTYPE:-bf16}"
MODES="${MODES:-full,c2kv,hybrid}"
RANK_PLANS="${RANK_PLANS:-}"
TARGET_COMPRESSION_RATIO="${TARGET_COMPRESSION_RATIO:-8}"
RECOVERY_CANDIDATE_DOCS="${RECOVERY_CANDIDATE_DOCS:-4}"
RECOVERY_SPAN_TOKENS="${RECOVERY_SPAN_TOKENS:-256,128,64}"
RECOVERY_MAX_SPANS="${RECOVERY_MAX_SPANS:-2}"

mkdir -p "${OUTPUT_DIR}"

echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "TOKENIZER_PATH=${TOKENIZER_PATH}"
echo "DATASET=${DATASET}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "MAX_EXAMPLES=${MAX_EXAMPLES}"
echo "RATIO=${RATIO}"
echo "HYBRID_TOP_K=${HYBRID_TOP_K}"
echo "MAX_DOC_LENGTH=${MAX_DOC_LENGTH}"
echo "MAX_DOC_NUM=${MAX_DOC_NUM}"
echo "MAX_CONTEXT_TOKENS=${MAX_CONTEXT_TOKENS}"
echo "RANK_PLANS=${RANK_PLANS}"
echo "TARGET_COMPRESSION_RATIO=${TARGET_COMPRESSION_RATIO}"
echo "RECOVERY_CANDIDATE_DOCS=${RECOVERY_CANDIDATE_DOCS}"
echo "RECOVERY_SPAN_TOKENS=${RECOVERY_SPAN_TOKENS}"
echo "RECOVERY_MAX_SPANS=${RECOVERY_MAX_SPANS}"

IFS=',' read -ra VISIBLE_NPUS <<< "${ASCEND_RT_VISIBLE_DEVICES}"
IFS=',' read -ra RUN_MODES <<< "${MODES}"
IFS=';' read -ra RUN_RANK_PLANS <<< "${RANK_PLANS}"

PIDS=()
INDEX=0
RANK_PLAN_INDEX=0
for MODE in "${RUN_MODES[@]}"; do
  MODE="${MODE// /}"
  RANK_PLAN=""
  MODE_NAME="${MODE}"
  if [[ "${MODE}" == rank_plan:* ]]; then
    MODE_NAME="${MODE#rank_plan:}"
    if (( RANK_PLAN_INDEX < ${#RUN_RANK_PLANS[@]} )); then
      RANK_PLAN="${RUN_RANK_PLANS[${RANK_PLAN_INDEX}]}"
    fi
    RANK_PLAN_INDEX=$((RANK_PLAN_INDEX + 1))
    MODE="rank_plan"
  fi
  DEVICE="${VISIBLE_NPUS[$((INDEX % ${#VISIBLE_NPUS[@]}))]}"
  OUTPUT_FILE="${OUTPUT_DIR}/${DATASET}_${MODE_NAME}_r${RATIO}.jsonl"
  LOG_FILE="${OUTPUT_DIR}/${DATASET}_${MODE_NAME}_r${RATIO}.log"
  echo "[launch] mode=${MODE} name=${MODE_NAME} rank_plan=${RANK_PLAN} device=${DEVICE} output=${OUTPUT_FILE}"
  (
    export ASCEND_RT_VISIBLE_DEVICES="${DEVICE}"
    export ASCEND_VISIBLE_DEVICES="${DEVICE}"
    python python/inference/expr_mdoc_c2kv_baselines.py \
      --model "${MODEL_PATH}" \
      --base_model "${BASE_MODEL}" \
      --tokenizer "${TOKENIZER_PATH}" \
      --dataset "${DATASET}" \
      --dataset_path "${DATASET_PATH}" \
      --mode "${MODE}" \
      --override_ratio "${RATIO}" \
      --hybrid_top_k "${HYBRID_TOP_K}" \
      --rank_plan "${RANK_PLAN:-1:full,2-:c2kv${RATIO}}" \
      --target_compression_ratio "${TARGET_COMPRESSION_RATIO}" \
      --recovery_candidate_docs "${RECOVERY_CANDIDATE_DOCS}" \
      --recovery_span_tokens "${RECOVERY_SPAN_TOKENS}" \
      --recovery_max_spans "${RECOVERY_MAX_SPANS}" \
      --max_examples "${MAX_EXAMPLES}" \
      --output_file "${OUTPUT_FILE}" \
      --max_doc_length "${MAX_DOC_LENGTH}" \
      --max_doc_num "${MAX_DOC_NUM}" \
      --max_context_tokens "${MAX_CONTEXT_TOKENS}" \
      --max_query_tokens "${MAX_QUERY_TOKENS}" \
      --doc_selection "${DOC_SELECTION}" \
      --device_type npu \
      --system_attn_impl "${SYSTEM_ATTN_IMPL}" \
      --gist_attn_impl "${GIST_ATTN_IMPL}" \
      --generate_attn_impl "${GENERATE_ATTN_IMPL}" \
      --dtype "${DTYPE}"
  ) > "${LOG_FILE}" 2>&1 &
  PIDS+=("$!")
  INDEX=$((INDEX + 1))
done

FAILED=0
for PID in "${PIDS[@]}"; do
  if ! wait "${PID}"; then
    FAILED=1
  fi
done

echo "==== summaries ===="
for MODE in "${RUN_MODES[@]}"; do
  MODE="${MODE// /}"
  MODE_NAME="${MODE}"
  if [[ "${MODE}" == rank_plan:* ]]; then
    MODE_NAME="${MODE#rank_plan:}"
  fi
  SUMMARY_FILE="${OUTPUT_DIR}/${DATASET}_${MODE_NAME}_r${RATIO}.summary.json"
  echo "---- ${SUMMARY_FILE} ----"
  if [[ -f "${SUMMARY_FILE}" ]]; then
    cat "${SUMMARY_FILE}"
  else
    echo "missing summary; check ${OUTPUT_DIR}/${DATASET}_${MODE}_r${RATIO}.log"
  fi
done

exit "${FAILED}"
