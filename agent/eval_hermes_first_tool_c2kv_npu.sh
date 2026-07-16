#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)/python:$(pwd)/agent:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

MODEL_PATH="${MODEL_PATH:-./checkpoints/qwen3-4b-agent-tooldef-npu}"
BASE_MODEL="${BASE_MODEL:-./models/Qwen3-4B-Instruct-2507}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
HERMES_DATASET_PATH="${HERMES_DATASET_PATH:-./datasets/hermes-agent-reasoning-traces}"
HERMES_OUTPUT_DIR="${HERMES_OUTPUT_DIR:-./outputs/hermes_first_tool}"
CONFIGS="${CONFIGS:-kimi,glm-5.1}"
METHODS="${METHODS:-full,truncate,c2kv_untrained,c2kv}"
RATIOS="${RATIOS:-2,4,8}"
MAX_EXAMPLES="${MAX_EXAMPLES:-100}"
SCAN_LIMIT="${SCAN_LIMIT:-}"
MAX_TOOL_DEFINITION_TOKENS="${MAX_TOOL_DEFINITION_TOKENS:-10000}"
MAX_FULL_TOOL_TOKENS="${MAX_FULL_TOOL_TOKENS:-10000}"
MAX_DOC_LENGTH="${MAX_DOC_LENGTH:-1024}"
MAX_DOC_NUM="${MAX_DOC_NUM:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
NPU_ATTN_IMPL="${NPU_ATTN_IMPL:-eager}"

mkdir -p "${HERMES_OUTPUT_DIR}"

if [[ -n "${DATASET_PATH:-}" && "${DATASET_PATH}" != "${HERMES_DATASET_PATH}" ]]; then
  echo "NOTE: ignoring DATASET_PATH=${DATASET_PATH}; use HERMES_DATASET_PATH for this script."
fi
if [[ -n "${OUTPUT_DIR:-}" && "${OUTPUT_DIR}" != "${HERMES_OUTPUT_DIR}" ]]; then
  echo "NOTE: ignoring OUTPUT_DIR=${OUTPUT_DIR}; use HERMES_OUTPUT_DIR for this script."
fi

if [[ ! -f "${HERMES_DATASET_PATH}/data/kimi/train.parquet" && ! -f "${HERMES_DATASET_PATH}/data/glm-5.1/train.parquet" ]]; then
  echo "ERROR: no Hermes parquet files found under HERMES_DATASET_PATH=${HERMES_DATASET_PATH}" >&2
  exit 1
fi

echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TOKENIZER_PATH=${TOKENIZER_PATH}"
echo "HERMES_DATASET_PATH=${HERMES_DATASET_PATH}"
echo "HERMES_OUTPUT_DIR=${HERMES_OUTPUT_DIR}"
echo "CONFIGS=${CONFIGS}"
echo "METHODS=${METHODS}"
echo "RATIOS=${RATIOS}"
echo "MAX_EXAMPLES=${MAX_EXAMPLES}"
echo "SCAN_LIMIT=${SCAN_LIMIT}"
echo "MAX_TOOL_DEFINITION_TOKENS=${MAX_TOOL_DEFINITION_TOKENS}"
echo "MAX_FULL_TOOL_TOKENS=${MAX_FULL_TOOL_TOKENS}"

IFS=',' read -ra _visible_npus <<< "${ASCEND_RT_VISIBLE_DEVICES}"
IFS=',' read -ra _methods <<< "${METHODS}"
IFS=',' read -ra _ratios <<< "${RATIOS}"
BATCH_SIZE="${#_visible_npus[@]}"
CASE_INDEX=0
SUMMARY_FILES=()

SCAN_ARGS=()
if [[ -n "${SCAN_LIMIT}" ]]; then
  SCAN_ARGS=(--scan_limit "${SCAN_LIMIT}")
fi

for method in "${_methods[@]}"; do
  method="${method// /}"
  if [[ "${method}" == "full" ]]; then
    run_ratios=(1)
  else
    run_ratios=("${_ratios[@]}")
  fi
  for ratio in "${run_ratios[@]}"; do
    ratio="${ratio// /}"
    device="${_visible_npus[$((CASE_INDEX % BATCH_SIZE))]}"
    output_file="${HERMES_OUTPUT_DIR}/${method}_r${ratio}.jsonl"
    summary_file="${HERMES_OUTPUT_DIR}/${method}_r${ratio}.summary.json"
    log_file="${HERMES_OUTPUT_DIR}/${method}_r${ratio}.log"
    rm -f "${output_file}" "${summary_file}" "${log_file}"
    SUMMARY_FILES+=("${summary_file}")
    echo "[launch] method=${method} ratio=${ratio} device=${device} output=${output_file}"
    (
      export ASCEND_RT_VISIBLE_DEVICES="${device}"
      python agent/eval_hermes_first_tool_c2kv.py \
        --device_type npu \
        --model "${MODEL_PATH}" \
        --base_model "${BASE_MODEL}" \
        --tokenizer "${TOKENIZER_PATH}" \
        --data_dir "${HERMES_DATASET_PATH}" \
        --configs "${CONFIGS}" \
        --output_file "${output_file}" \
        --method "${method}" \
        --ratio "${ratio}" \
        --max_examples "${MAX_EXAMPLES}" \
        "${SCAN_ARGS[@]}" \
        --max_tool_definition_tokens "${MAX_TOOL_DEFINITION_TOKENS}" \
        --max_full_tool_tokens "${MAX_FULL_TOOL_TOKENS}" \
        --max_doc_length "${MAX_DOC_LENGTH}" \
        --max_doc_num "${MAX_DOC_NUM}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --system_attn_impl "${NPU_ATTN_IMPL}" \
        --gist_attn_impl "${NPU_ATTN_IMPL}" \
        --generate_attn_impl "${NPU_ATTN_IMPL}"
    ) > "${log_file}" 2>&1 &
    CASE_INDEX=$((CASE_INDEX + 1))
    if (( CASE_INDEX % BATCH_SIZE == 0 )); then
      wait
    fi
  done
done

wait

echo "Summaries:"
for summary in "${SUMMARY_FILES[@]}"; do
  echo "==== ${summary} ===="
  if [[ -f "${summary}" ]]; then
    cat "${summary}"
  else
    echo "MISSING summary file. Check log: ${summary%.summary.json}.log"
  fi
done
