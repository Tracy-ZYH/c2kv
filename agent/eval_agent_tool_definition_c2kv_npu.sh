#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)/python:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

MODEL_PATH="${MODEL_PATH:-./checkpoints/qwen3-4b-agent-tooldef-npu}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
DATASET_PATH="${DATASET_PATH:-./datasets/agent-llm-traces}"
OUTPUT_FILE="${OUTPUT_FILE:-./outputs/agent_tooldef_c2kv_eval_npu.jsonl}"
MAX_EXAMPLES="${MAX_EXAMPLES:-50}"
MAX_DOC_LENGTH="${MAX_DOC_LENGTH:-1024}"
MAX_DOC_NUM="${MAX_DOC_NUM:-10}"
MAX_TOOL_DEFINITION_TOKENS="${MAX_TOOL_DEFINITION_TOKENS:-10000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
OVERRIDE_RATIO="${OVERRIDE_RATIO:-4}"
NPU_ATTN_IMPL="${NPU_ATTN_IMPL:-eager}"

echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TOKENIZER_PATH=${TOKENIZER_PATH}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "OUTPUT_FILE=${OUTPUT_FILE}"
echo "MAX_EXAMPLES=${MAX_EXAMPLES}"

python agent/eval_agent_tool_definition_c2kv.py \
  --device_type npu \
  --model "${MODEL_PATH}" \
  --tokenizer "${TOKENIZER_PATH}" \
  --dataset_path "${DATASET_PATH}" \
  --output_file "${OUTPUT_FILE}" \
  --max_examples "${MAX_EXAMPLES}" \
  --max_doc_length "${MAX_DOC_LENGTH}" \
  --max_doc_num "${MAX_DOC_NUM}" \
  --max_tool_definition_tokens "${MAX_TOOL_DEFINITION_TOKENS}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --override_ratio "${OVERRIDE_RATIO}" \
  --system_attn_impl "${NPU_ATTN_IMPL}" \
  --gist_attn_impl "${NPU_ATTN_IMPL}" \
  --generate_attn_impl "${NPU_ATTN_IMPL}" \
  --truncate_tool_definition False \
  --require_tool_call True
