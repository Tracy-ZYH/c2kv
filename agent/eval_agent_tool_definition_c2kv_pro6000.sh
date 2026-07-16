#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/zhuyuhan/project/c2kv}"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/python:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/qwen3-4b-agent-tooldef-pro6000}"
BASE_MODEL="${BASE_MODEL:-${PROJECT_ROOT}/models/Qwen3-4B-Instruct-2507}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/datasets/agent-llm-traces}"
OUTPUT_FILE="${OUTPUT_FILE:-${PROJECT_ROOT}/outputs/agent_tooldef_c2kv_eval_pro6000.jsonl}"
MAX_EXAMPLES="${MAX_EXAMPLES:-50}"
MAX_DOC_LENGTH="${MAX_DOC_LENGTH:-1024}"
MAX_DOC_NUM="${MAX_DOC_NUM:-10}"
MAX_TOOL_DEFINITION_TOKENS="${MAX_TOOL_DEFINITION_TOKENS:-10000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
OVERRIDE_RATIO="${OVERRIDE_RATIO:-4}"
RATIOS="${RATIOS:-2,4,8}"
COMPARE_MODES="${COMPARE_MODES:-c2kv,c2kv_untrained,truncate}"
GIST_ATTN_IMPL="${GIST_ATTN_IMPL:-flex_attention}"
GENERATE_ATTN_IMPL="${GENERATE_ATTN_IMPL:-flash_attention_2}"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "TOKENIZER_PATH=${TOKENIZER_PATH}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "OUTPUT_FILE=${OUTPUT_FILE}"
echo "MAX_EXAMPLES=${MAX_EXAMPLES}"
echo "COMPARE_MODES=${COMPARE_MODES}"
echo "RATIOS=${RATIOS}"

python agent/eval_agent_tool_definition_c2kv.py \
  --device_type cuda \
  --model "${MODEL_PATH}" \
  --base_model "${BASE_MODEL}" \
  --tokenizer "${TOKENIZER_PATH}" \
  --dataset_path "${DATASET_PATH}" \
  --output_file "${OUTPUT_FILE}" \
  --max_examples "${MAX_EXAMPLES}" \
  --max_doc_length "${MAX_DOC_LENGTH}" \
  --max_doc_num "${MAX_DOC_NUM}" \
  --max_tool_definition_tokens "${MAX_TOOL_DEFINITION_TOKENS}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --override_ratio "${OVERRIDE_RATIO}" \
  --ratios "${RATIOS}" \
  --compare_modes "${COMPARE_MODES}" \
  --system_attn_impl "${GENERATE_ATTN_IMPL}" \
  --gist_attn_impl "${GIST_ATTN_IMPL}" \
  --generate_attn_impl "${GENERATE_ATTN_IMPL}" \
  --truncate_tool_definition False \
  --require_tool_call True
