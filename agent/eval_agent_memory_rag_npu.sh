#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)/python:$(pwd)/agent:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

MODEL_PATH="${MODEL_PATH:-./checkpoints/qwen3-4b-agent-tooldef-npu}"
BASE_MODEL="${BASE_MODEL:-./models/Qwen3-4B-Instruct-2507}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
DATASET_PATH="${DATASET_PATH:-./datasets/agent-llm-traces}"
OUTPUT_FILE="${OUTPUT_FILE:-./outputs/agent_memory_rag_eval_npu.jsonl}"
METHODS="${METHODS:-no_memory,full_rag,hybrid_rag,hybrid_summary}"
TOP_M="${TOP_M:-3}"
MAX_MEMORY_ITEM_CHARS="${MAX_MEMORY_ITEM_CHARS:-800}"
MAX_MEMORY_TOTAL_CHARS="${MAX_MEMORY_TOTAL_CHARS:-2400}"
MAX_FULL_HISTORY_MESSAGES="${MAX_FULL_HISTORY_MESSAGES:-8}"
HYBRID_TOP_K="${HYBRID_TOP_K:-3}"
HYBRID_RATIO="${HYBRID_RATIO:-4}"
MAX_EXAMPLES="${MAX_EXAMPLES:-50}"
MAX_DOC_LENGTH="${MAX_DOC_LENGTH:-1024}"
MAX_DOC_NUM="${MAX_DOC_NUM:-10}"
MAX_TOOL_DEFINITION_TOKENS="${MAX_TOOL_DEFINITION_TOKENS:-10000}"
MAX_BASELINE_TOOL_TOKENS="${MAX_BASELINE_TOOL_TOKENS:-10000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
NPU_ATTN_IMPL="${NPU_ATTN_IMPL:-eager}"

echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TOKENIZER_PATH=${TOKENIZER_PATH}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "OUTPUT_FILE=${OUTPUT_FILE}"
echo "METHODS=${METHODS}"
echo "TOP_M=${TOP_M}"
echo "MAX_MEMORY_ITEM_CHARS=${MAX_MEMORY_ITEM_CHARS}"
echo "MAX_MEMORY_TOTAL_CHARS=${MAX_MEMORY_TOTAL_CHARS}"
echo "HYBRID_TOP_K=${HYBRID_TOP_K}"
echo "HYBRID_RATIO=${HYBRID_RATIO}"
echo "MAX_EXAMPLES=${MAX_EXAMPLES}"

python agent/eval_agent_memory_rag.py \
  --device_type npu \
  --model "${MODEL_PATH}" \
  --base_model "${BASE_MODEL}" \
  --tokenizer "${TOKENIZER_PATH}" \
  --dataset_path "${DATASET_PATH}" \
  --output_file "${OUTPUT_FILE}" \
  --methods "${METHODS}" \
  --top_m "${TOP_M}" \
  --max_memory_item_chars "${MAX_MEMORY_ITEM_CHARS}" \
  --max_memory_total_chars "${MAX_MEMORY_TOTAL_CHARS}" \
  --max_full_history_messages "${MAX_FULL_HISTORY_MESSAGES}" \
  --hybrid_top_k "${HYBRID_TOP_K}" \
  --hybrid_ratio "${HYBRID_RATIO}" \
  --max_examples "${MAX_EXAMPLES}" \
  --max_doc_length "${MAX_DOC_LENGTH}" \
  --max_doc_num "${MAX_DOC_NUM}" \
  --max_tool_definition_tokens "${MAX_TOOL_DEFINITION_TOKENS}" \
  --max_baseline_tool_tokens "${MAX_BASELINE_TOOL_TOKENS}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --system_attn_impl "${NPU_ATTN_IMPL}" \
  --gist_attn_impl "${NPU_ATTN_IMPL}" \
  --generate_attn_impl "${NPU_ATTN_IMPL}" \
  --truncate_tool_definition False \
  --require_tool_call True
