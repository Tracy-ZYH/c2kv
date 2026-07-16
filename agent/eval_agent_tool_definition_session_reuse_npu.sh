#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)/python:$(pwd)/agent:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

MODEL_PATH="${MODEL_PATH:-./checkpoints/qwen3-4b-agent-tooldef-npu}"
BASE_MODEL="${BASE_MODEL:-./models/Qwen3-4B-Instruct-2507}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
DATASET_PATH="${DATASET_PATH:-./datasets/agent-llm-traces}"
OUTPUT_FILE="${OUTPUT_FILE:-./outputs/agent_tooldef_session_reuse_eval_npu.jsonl}"
MAX_SESSIONS="${MAX_SESSIONS:-20}"
MIN_SPANS_PER_SESSION="${MIN_SPANS_PER_SESSION:-2}"
MAX_SPANS_PER_SESSION="${MAX_SPANS_PER_SESSION:-0}"
MAX_DOC_LENGTH="${MAX_DOC_LENGTH:-1024}"
MAX_DOC_NUM="${MAX_DOC_NUM:-10}"
MAX_TOOL_DEFINITION_TOKENS="${MAX_TOOL_DEFINITION_TOKENS:-10000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
RATIOS="${RATIOS:-2,4,8}"
COMPARE_MODES="${COMPARE_MODES:-c2kv,truncate,full}"
COMPARE_REUSE="${COMPARE_REUSE:-False}"
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
echo "MAX_SESSIONS=${MAX_SESSIONS}"
echo "MAX_SPANS_PER_SESSION=${MAX_SPANS_PER_SESSION}"
echo "COMPARE_MODES=${COMPARE_MODES}"
echo "RATIOS=${RATIOS}"
echo "COMPARE_REUSE=${COMPARE_REUSE}"
echo "PARALLEL_EVAL=${PARALLEL_EVAL}"

if [[ "${PARALLEL_EVAL}" != "True" && "${PARALLEL_EVAL}" != "true" && "${PARALLEL_EVAL}" != "1" ]]; then
  python agent/eval_agent_tool_definition_session_reuse.py \
    --device_type npu \
    --model "${MODEL_PATH}" \
    --base_model "${BASE_MODEL}" \
    --tokenizer "${TOKENIZER_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    --output_file "${OUTPUT_FILE}" \
    --max_sessions "${MAX_SESSIONS}" \
    --min_spans_per_session "${MIN_SPANS_PER_SESSION}" \
    --max_spans_per_session "${MAX_SPANS_PER_SESSION}" \
    --max_doc_length "${MAX_DOC_LENGTH}" \
    --max_doc_num "${MAX_DOC_NUM}" \
    --max_tool_definition_tokens "${MAX_TOOL_DEFINITION_TOKENS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --ratios "${RATIOS}" \
    --compare_modes "${COMPARE_MODES}" \
    --compare_reuse "${COMPARE_REUSE}" \
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

if [[ "${COMPARE_REUSE}" == "True" || "${COMPARE_REUSE}" == "true" || "${COMPARE_REUSE}" == "1" ]]; then
  _reuse_values=("False" "True")
else
  _reuse_values=("True")
fi

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
    for reuse in "${_reuse_values[@]}"; do
      device="${_visible_npus[$((CASE_INDEX % BATCH_SIZE))]}"
      case_name="${mode}_r${ratio}_reuse${reuse}"
      case_output="${TMP_DIR}/${case_name}.jsonl"
      case_log="${TMP_DIR}/${case_name}.log"
      CASE_OUTPUTS+=("${case_output}")
      echo "[launch] case=${case_name} device=${device} output=${case_output}"
      (
        export ASCEND_RT_VISIBLE_DEVICES="${device}"
        python agent/eval_agent_tool_definition_session_reuse.py \
          --device_type npu \
          --model "${MODEL_PATH}" \
          --base_model "${BASE_MODEL}" \
          --tokenizer "${TOKENIZER_PATH}" \
          --dataset_path "${DATASET_PATH}" \
          --output_file "${case_output}" \
          --max_sessions "${MAX_SESSIONS}" \
          --min_spans_per_session "${MIN_SPANS_PER_SESSION}" \
          --max_spans_per_session "${MAX_SPANS_PER_SESSION}" \
          --max_doc_length "${MAX_DOC_LENGTH}" \
          --max_doc_num "${MAX_DOC_NUM}" \
          --max_tool_definition_tokens "${MAX_TOOL_DEFINITION_TOKENS}" \
          --max_new_tokens "${MAX_NEW_TOKENS}" \
          --mode "${mode}" \
          --override_ratio "${ratio}" \
          --compare_reuse False \
          --reuse "${reuse}" \
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
done

wait

python agent/merge_agent_tool_definition_session_reuse_eval.py \
  --output_file "${OUTPUT_FILE}" \
  --model "${MODEL_PATH}" \
  --base_model "${BASE_MODEL}" \
  --dataset_path "${DATASET_PATH}" \
  --split eval \
  --input_files "${CASE_OUTPUTS[@]}"
