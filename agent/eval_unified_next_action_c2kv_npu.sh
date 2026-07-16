#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)/python:$(pwd)/agent:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

MODEL_PATH="${MODEL_PATH:-./checkpoints/qwen3-4b-unified-next-action-c2kv-npu}"
BASE_MODEL="${BASE_MODEL:-./models/Qwen3-4B-Instruct-2507}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
OUTPUT_FILE="${OUTPUT_FILE:-./outputs/unified_next_action_eval_npu.jsonl}"
SPLIT="${SPLIT:-eval}"
COMPARE_MODES="${COMPARE_MODES:-full,truncate,c2kv_untrained,c2kv}"
RATIOS="${RATIOS:-2,4,8}"
MAX_EXAMPLES="${MAX_EXAMPLES:-200}"

AGENT_DATASET_PATH="${AGENT_DATASET_PATH:-./datasets/agent-llm-traces}"
TOOLATHLON_DATASET_PATH="${TOOLATHLON_DATASET_PATH:-./datasets/toolathlon}"
HERMES_DATASET_PATH="${HERMES_DATASET_PATH:-./datasets/hermes-agent-reasoning-traces}"
HERMES_CONFIGS="${HERMES_CONFIGS:-kimi,glm-5.1}"
SOURCE_MIX="${SOURCE_MIX:-agent_llm_traces:0.3,toolathlon:0.4,hermes:0.3}"
SPLIT_SEED="${SPLIT_SEED:-42}"

MAX_DOC_LENGTH="${MAX_DOC_LENGTH:-1024}"
MAX_DOC_NUM="${MAX_DOC_NUM:-32}"
MAX_TOOL_DEFINITION_TOKENS="${MAX_TOOL_DEFINITION_TOKENS:-32000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
MAX_TOOLS_PER_SAMPLE="${MAX_TOOLS_PER_SAMPLE:-32}"
NPU_ATTN_IMPL="${NPU_ATTN_IMPL:-eager}"

OUTPUT_STEM="${OUTPUT_FILE%.jsonl}"
TMP_DIR="${TMP_DIR:-${OUTPUT_STEM}.parts}"

echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "OUTPUT_FILE=${OUTPUT_FILE}"
echo "SPLIT=${SPLIT}"
echo "COMPARE_MODES=${COMPARE_MODES}"
echo "RATIOS=${RATIOS}"
echo "MAX_EXAMPLES=${MAX_EXAMPLES}"

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
    rm -f "${case_output}" "${case_output%.jsonl}.summary.json" "${case_log}"
    CASE_OUTPUTS+=("${case_output}")
    echo "[launch] case=${case_name} device=${device} output=${case_output}"
    (
      export ASCEND_RT_VISIBLE_DEVICES="${device}"
      python agent/eval_unified_next_action_c2kv.py \
        --device_type npu \
        --model "${MODEL_PATH}" \
        --base_model "${BASE_MODEL}" \
        --tokenizer "${TOKENIZER_PATH}" \
        --output_file "${case_output}" \
        --split "${SPLIT}" \
        --mode "${mode}" \
        --compare_modes "${mode}" \
        --override_ratio "${ratio}" \
        --ratios "${ratio}" \
        --max_examples "${MAX_EXAMPLES}" \
        --agent_dataset_path "${AGENT_DATASET_PATH}" \
        --toolathlon_dataset_path "${TOOLATHLON_DATASET_PATH}" \
        --hermes_dataset_path "${HERMES_DATASET_PATH}" \
        --hermes_configs "${HERMES_CONFIGS}" \
        --source_mix "${SOURCE_MIX}" \
        --split_seed "${SPLIT_SEED}" \
        --max_doc_length "${MAX_DOC_LENGTH}" \
        --max_doc_num "${MAX_DOC_NUM}" \
        --max_tool_definition_tokens "${MAX_TOOL_DEFINITION_TOKENS}" \
        --max_tools_per_sample "${MAX_TOOLS_PER_SAMPLE}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --system_attn_impl "${NPU_ATTN_IMPL}" \
        --gist_attn_impl "${NPU_ATTN_IMPL}" \
        --generate_attn_impl "${NPU_ATTN_IMPL}"
    ) > "${case_log}" 2>&1 &

    CASE_INDEX=$((CASE_INDEX + 1))
    if (( CASE_INDEX % BATCH_SIZE == 0 )); then
      wait
    fi
  done
done

wait

python agent/merge_unified_next_action_eval.py \
  --output_file "${OUTPUT_FILE}" \
  --model "${MODEL_PATH}" \
  --base_model "${BASE_MODEL}" \
  --split "${SPLIT}" \
  --input_files "${CASE_OUTPUTS[@]}"
