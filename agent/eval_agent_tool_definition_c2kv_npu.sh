#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)/python:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

MODEL_PATH="${MODEL_PATH:-./checkpoints/qwen3-4b-agent-tooldef-npu}"
BASE_MODEL="${BASE_MODEL:-./models/Qwen3-4B-Instruct-2507}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
DATASET_PATH="${DATASET_PATH:-./datasets/agent-llm-traces}"
OUTPUT_FILE="${OUTPUT_FILE:-./outputs/agent_tooldef_c2kv_eval_npu.jsonl}"
SPLIT_MANIFEST_FILE="${SPLIT_MANIFEST_FILE:-}"
SPLIT_NAME="${SPLIT_NAME:-toolset_disjoint}"
MAX_EXAMPLES="${MAX_EXAMPLES:-50}"
MAX_DOC_LENGTH="${MAX_DOC_LENGTH:-1024}"
MAX_DOC_NUM="${MAX_DOC_NUM:-10}"
MAX_TOOL_DEFINITION_TOKENS="${MAX_TOOL_DEFINITION_TOKENS:-10000}"
TOOL_DOCUMENT_EVAL_MODE="${TOOL_DOCUMENT_EVAL_MODE:-full}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
OVERRIDE_RATIO="${OVERRIDE_RATIO:-4}"
RATIOS="${RATIOS:-2,4,8}"
COMPARE_MODES="${COMPARE_MODES:-c2kv,truncate,full}"
NPU_ATTN_IMPL="${NPU_ATTN_IMPL:-eager}"
PARALLEL_EVAL="${PARALLEL_EVAL:-True}"
OUTPUT_STEM="${OUTPUT_FILE%.jsonl}"
TMP_DIR="${TMP_DIR:-${OUTPUT_STEM}.parts}"

echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "TOKENIZER_PATH=${TOKENIZER_PATH}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "OUTPUT_FILE=${OUTPUT_FILE}"
echo "SPLIT_MANIFEST_FILE=${SPLIT_MANIFEST_FILE}"
echo "SPLIT_NAME=${SPLIT_NAME}"
echo "MAX_EXAMPLES=${MAX_EXAMPLES}"
echo "TOOL_DOCUMENT_EVAL_MODE=${TOOL_DOCUMENT_EVAL_MODE}"
echo "COMPARE_MODES=${COMPARE_MODES}"
echo "RATIOS=${RATIOS}"
echo "PARALLEL_EVAL=${PARALLEL_EVAL}"

SPLIT_ARGS=(--split_manifest_name "${SPLIT_NAME}")
if [[ -n "${SPLIT_MANIFEST_FILE}" ]]; then
  SPLIT_ARGS=(--split_manifest_file "${SPLIT_MANIFEST_FILE}" --split_manifest_name "${SPLIT_NAME}")
fi

if [[ "${PARALLEL_EVAL}" != "True" && "${PARALLEL_EVAL}" != "true" && "${PARALLEL_EVAL}" != "1" ]]; then
  python agent/eval_agent_tool_definition_c2kv.py \
    --device_type npu \
    --model "${MODEL_PATH}" \
    --base_model "${BASE_MODEL}" \
    --tokenizer "${TOKENIZER_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    --output_file "${OUTPUT_FILE}" \
    "${SPLIT_ARGS[@]}" \
    --max_examples "${MAX_EXAMPLES}" \
    --max_doc_length "${MAX_DOC_LENGTH}" \
    --max_doc_num "${MAX_DOC_NUM}" \
    --max_tool_definition_tokens "${MAX_TOOL_DEFINITION_TOKENS}" \
    --tool_document_eval_mode "${TOOL_DOCUMENT_EVAL_MODE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --override_ratio "${OVERRIDE_RATIO}" \
    --ratios "${RATIOS}" \
    --compare_modes "${COMPARE_MODES}" \
    --system_attn_impl "${NPU_ATTN_IMPL}" \
    --gist_attn_impl "${NPU_ATTN_IMPL}" \
    --generate_attn_impl "${NPU_ATTN_IMPL}" \
    --truncate_tool_definition False \
    --require_tool_call True
  exit 0
fi

mkdir -p "${TMP_DIR}"
IFS=',' read -ra _visible_npus <<< "${ASCEND_RT_VISIBLE_DEVICES}"
IFS=',' read -ra _modes <<< "${COMPARE_MODES}"
IFS=',' read -ra _ratios <<< "${RATIOS}"

CASE_OUTPUTS=()
CASE_INDEX=0
BATCH_SIZE="${#_visible_npus[@]}"

for mode in "${_modes[@]}"; do
  mode="${mode// /}"
  case_ratios=("${_ratios[@]}")
  if [[ "${mode}" == "full" ]]; then
    case_ratios=("1")
  fi
  for ratio in "${case_ratios[@]}"; do
    ratio="${ratio// /}"
    device="${_visible_npus[$((CASE_INDEX % BATCH_SIZE))]}"
    case_name="${mode}_r${ratio}"
    case_output="${TMP_DIR}/${case_name}.jsonl"
    case_log="${TMP_DIR}/${case_name}.log"
    CASE_OUTPUTS+=("${case_output}")
    echo "[launch] case=${case_name} device=${device} output=${case_output}"
    (
      export ASCEND_RT_VISIBLE_DEVICES="${device}"
      python agent/eval_agent_tool_definition_c2kv.py \
        --device_type npu \
        --model "${MODEL_PATH}" \
        --base_model "${BASE_MODEL}" \
        --tokenizer "${TOKENIZER_PATH}" \
        --dataset_path "${DATASET_PATH}" \
        --output_file "${case_output}" \
        "${SPLIT_ARGS[@]}" \
        --max_examples "${MAX_EXAMPLES}" \
        --max_doc_length "${MAX_DOC_LENGTH}" \
        --max_doc_num "${MAX_DOC_NUM}" \
        --max_tool_definition_tokens "${MAX_TOOL_DEFINITION_TOKENS}" \
        --tool_document_eval_mode "${TOOL_DOCUMENT_EVAL_MODE}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --override_ratio "${ratio}" \
        --mode "${mode}" \
        --system_attn_impl "${NPU_ATTN_IMPL}" \
        --gist_attn_impl "${NPU_ATTN_IMPL}" \
        --generate_attn_impl "${NPU_ATTN_IMPL}" \
        --truncate_tool_definition False \
        --require_tool_call True
    ) > "${case_log}" 2>&1 &

    CASE_INDEX=$((CASE_INDEX + 1))
    if (( CASE_INDEX % BATCH_SIZE == 0 )); then
      wait
    fi
  done
done

wait

python agent/merge_agent_tool_definition_eval.py \
  --output_file "${OUTPUT_FILE}" \
  --model "${MODEL_PATH}" \
  --base_model "${BASE_MODEL}" \
  --dataset_path "${DATASET_PATH}" \
  --split eval \
  --tool_document_eval_mode "${TOOL_DOCUMENT_EVAL_MODE}" \
  --input_files "${CASE_OUTPUTS[@]}"
