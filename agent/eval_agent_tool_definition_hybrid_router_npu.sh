#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)/python:$(pwd)/agent:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

MODEL_PATH="${MODEL_PATH:-./checkpoints/qwen3-4b-agent-tooldef-npu}"
BASE_MODEL="${BASE_MODEL:-./models/Qwen3-4B-Instruct-2507}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
DATASET_PATH="${DATASET_PATH:-./datasets/agent-llm-traces}"
OUTPUT_FILE="${OUTPUT_FILE:-./outputs/agent_tooldef_hybrid_router_eval_npu.jsonl}"
MAX_EXAMPLES="${MAX_EXAMPLES:-106}"
MAX_DOC_LENGTH="${MAX_DOC_LENGTH:-1024}"
MAX_DOC_NUM="${MAX_DOC_NUM:-10}"
MAX_TOOL_DEFINITION_TOKENS="${MAX_TOOL_DEFINITION_TOKENS:-10000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
HYBRID_CASES="${HYBRID_CASES:-1:4,3:4,5:8}"
HYBRID_MODES="${HYBRID_MODES:-hybrid}"
ROUTER_SCOPE="${ROUTER_SCOPE:-last_user}"
ROUTER_STRATEGIES="${ROUTER_STRATEGIES:-lexical}"
ROUTER_HIT_FILTER="${ROUTER_HIT_FILTER:-all}"
ROUTER_SEED="${ROUTER_SEED:-42}"
NPU_ATTN_IMPL="${NPU_ATTN_IMPL:-eager}"
PARALLEL_EVAL="${PARALLEL_EVAL:-True}"
OUTPUT_STEM="${OUTPUT_FILE%.jsonl}"
TMP_DIR="${TMP_DIR:-${OUTPUT_STEM}.parts}"

echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TOKENIZER_PATH=${TOKENIZER_PATH}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "OUTPUT_FILE=${OUTPUT_FILE}"
echo "MAX_EXAMPLES=${MAX_EXAMPLES}"
echo "HYBRID_CASES=${HYBRID_CASES}"
echo "HYBRID_MODES=${HYBRID_MODES}"
echo "ROUTER_SCOPE=${ROUTER_SCOPE}"
echo "ROUTER_STRATEGIES=${ROUTER_STRATEGIES}"
echo "ROUTER_HIT_FILTER=${ROUTER_HIT_FILTER}"
echo "PARALLEL_EVAL=${PARALLEL_EVAL}"

if [[ "${PARALLEL_EVAL}" != "True" && "${PARALLEL_EVAL}" != "true" && "${PARALLEL_EVAL}" != "1" ]]; then
  python agent/eval_agent_tool_definition_hybrid_router.py \
    --device_type npu \
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
    --hybrid_cases "${HYBRID_CASES}" \
    --hybrid_mode "${HYBRID_MODES}" \
    --router_scope "${ROUTER_SCOPE}" \
    --router_strategy "${ROUTER_STRATEGIES}" \
    --router_hit_filter "${ROUTER_HIT_FILTER}" \
    --router_seed "${ROUTER_SEED}" \
    --system_attn_impl "${NPU_ATTN_IMPL}" \
    --gist_attn_impl "${NPU_ATTN_IMPL}" \
    --generate_attn_impl "${NPU_ATTN_IMPL}" \
    --truncate_tool_definition False \
    --require_tool_call True
  exit 0
fi

mkdir -p "${TMP_DIR}"
IFS=',' read -ra _visible_npus <<< "${ASCEND_RT_VISIBLE_DEVICES}"
IFS=',' read -ra _cases <<< "${HYBRID_CASES}"
IFS=',' read -ra _hybrid_modes <<< "${HYBRID_MODES}"
IFS=',' read -ra _strategies <<< "${ROUTER_STRATEGIES}"

CASE_OUTPUTS=()
CASE_INDEX=0
BATCH_SIZE="${#_visible_npus[@]}"

for hybrid_mode in "${_hybrid_modes[@]}"; do
  hybrid_mode="${hybrid_mode// /}"
  for strategy in "${_strategies[@]}"; do
    strategy="${strategy// /}"
    for case_spec in "${_cases[@]}"; do
      case_spec="${case_spec// /}"
      device="${_visible_npus[$((CASE_INDEX % BATCH_SIZE))]}"
      case_name="hybrid_${hybrid_mode}_${strategy}_${case_spec/:/_r}"
      case_output="${TMP_DIR}/${case_name}.jsonl"
      case_log="${TMP_DIR}/${case_name}.log"
      rm -f "${case_output}" "${case_output%.jsonl}.summary.json" "${case_log}"
      CASE_OUTPUTS+=("${case_output}")
      echo "[launch] hybrid_mode=${hybrid_mode} strategy=${strategy} case=${case_spec} device=${device} output=${case_output}"
      (
        export ASCEND_RT_VISIBLE_DEVICES="${device}"
        python agent/eval_agent_tool_definition_hybrid_router.py \
          --device_type npu \
          --model "${MODEL_PATH}" \
          --base_model "${BASE_MODEL}" \
          --tokenizer "${TOKENIZER_PATH}" \
          --dataset_path "${DATASET_PATH}" \
          --output_file "${case_output}" \
          --max_examples "${MAX_EXAMPLES}" \
          --max_doc_length "${MAX_DOC_LENGTH}" \
          --max_doc_num "${MAX_DOC_NUM}" \
          --max_tool_definition_tokens "${MAX_TOOL_DEFINITION_TOKENS}" \
          --max_new_tokens "${MAX_NEW_TOKENS}" \
          --hybrid_cases "${case_spec}" \
          --hybrid_mode "${hybrid_mode}" \
          --router_scope "${ROUTER_SCOPE}" \
          --router_strategy "${strategy}" \
          --router_hit_filter "${ROUTER_HIT_FILTER}" \
          --router_seed "${ROUTER_SEED}" \
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

python agent/merge_agent_tool_definition_hybrid_eval.py \
  --output_file "${OUTPUT_FILE}" \
  --model "${MODEL_PATH}" \
  --dataset_path "${DATASET_PATH}" \
  --split eval \
  --router_scope "${ROUTER_SCOPE}" \
  --router_strategy "${ROUTER_STRATEGIES}" \
  --hybrid_modes "${HYBRID_MODES}" \
  --input_files "${CASE_OUTPUTS[@]}"
