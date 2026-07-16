#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)/python:$(pwd)/python/inference:$(pwd)/agent:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

MODEL_PATH="${MODEL_PATH:-./checkpoints/qwen3-4b-agent-tooldef-npu}"
BASE_MODEL="${BASE_MODEL:-./models/Qwen3-4B-Instruct-2507}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
DATASET_PATH="${DATASET_PATH:-./datasets/agent-llm-traces}"
OUTPUT_FILE="${OUTPUT_FILE:-./outputs/agent_tooldef_reuse_baselines_eval_npu.jsonl}"
SPLIT="${SPLIT:-eval}"

COMPARE_MODES="${COMPARE_MODES:-full,snapkv_reuse,epic_leading32_snapkv,cacheblend_vdiff_snapkv,c2kv,hybrid}"
RATIOS="${RATIOS:-4}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
HYBRID_TOP_K="${HYBRID_TOP_K:-3}"
CACHEBLEND_RECOMPUTE_RATIO="${CACHEBLEND_RECOMPUTE_RATIO:-0.15}"
SELECTION_FILTER="${SELECTION_FILTER:-c2kv}"
STRICT_4X_BASELINES="${STRICT_4X_BASELINES:-True}"

SPLIT_MANIFEST_FILE="${SPLIT_MANIFEST_FILE:-}"
SPLIT_NAME="${SPLIT_NAME:-toolset_disjoint}"
MAX_DOC_LENGTH="${MAX_DOC_LENGTH:-1024}"
MAX_DOC_NUM="${MAX_DOC_NUM:-10}"
MAX_TOOL_DEFINITION_TOKENS="${MAX_TOOL_DEFINITION_TOKENS:-10000}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-0}"
MAX_BASELINE_INPUT_TOKENS="${MAX_BASELINE_INPUT_TOKENS:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
NPU_ATTN_IMPL="${NPU_ATTN_IMPL:-eager}"
PARALLEL_EVAL="${PARALLEL_EVAL:-True}"
OUTPUT_STEM="${OUTPUT_FILE%.jsonl}"
TMP_DIR="${TMP_DIR:-${OUTPUT_STEM}.parts}"

SPLIT_ARGS=(--split_manifest_name "${SPLIT_NAME}")
if [[ -n "${SPLIT_MANIFEST_FILE}" ]]; then
  SPLIT_ARGS=(--split_manifest_file "${SPLIT_MANIFEST_FILE}" --split_manifest_name "${SPLIT_NAME}")
fi

echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "TOKENIZER_PATH=${TOKENIZER_PATH}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "OUTPUT_FILE=${OUTPUT_FILE}"
echo "SPLIT=${SPLIT}"
echo "COMPARE_MODES=${COMPARE_MODES}"
echo "RATIOS=${RATIOS}"
echo "MAX_EXAMPLES=${MAX_EXAMPLES}"
echo "SELECTION_FILTER=${SELECTION_FILTER}"
echo "STRICT_4X_BASELINES=${STRICT_4X_BASELINES}"
echo "MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS}"
echo "MAX_BASELINE_INPUT_TOKENS=${MAX_BASELINE_INPUT_TOKENS}"
echo "PARALLEL_EVAL=${PARALLEL_EVAL}"

if [[ "${STRICT_4X_BASELINES}" == "True" || "${STRICT_4X_BASELINES}" == "true" || "${STRICT_4X_BASELINES}" == "1" ]]; then
  IFS=',' read -ra _strict_modes <<< "${COMPARE_MODES}"
  for mode in "${_strict_modes[@]}"; do
    mode="${mode// /}"
    if [[ "${mode}" == "reuse" || "${mode}" == "epic_leading32" || "${mode}" == "cacheblend_vdiff" ]]; then
      echo "ERROR: ${mode} is a 1x baseline. For fair 4x comparison use snapkv_reuse, epic_leading32_snapkv, cacheblend_vdiff_snapkv, c2kv, hybrid." >&2
      echo "Set STRICT_4X_BASELINES=False only if you intentionally want mixed 1x/4x results." >&2
      exit 1
    fi
  done
fi

if [[ "${PARALLEL_EVAL}" != "True" && "${PARALLEL_EVAL}" != "true" && "${PARALLEL_EVAL}" != "1" ]]; then
  python agent/eval_agent_tool_definition_reuse_baselines.py \
    --device_type npu \
    --model "${MODEL_PATH}" \
    --base_model "${BASE_MODEL}" \
    --reuse_model "${BASE_MODEL}" \
    --tokenizer "${TOKENIZER_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    --output_file "${OUTPUT_FILE}" \
    --split "${SPLIT}" \
    "${SPLIT_ARGS[@]}" \
    --compare_modes "${COMPARE_MODES}" \
    --ratios "${RATIOS}" \
    --hybrid_top_k "${HYBRID_TOP_K}" \
    --cacheblend_recompute_ratio "${CACHEBLEND_RECOMPUTE_RATIO}" \
    --max_examples "${MAX_EXAMPLES}" \
    --selection_filter "${SELECTION_FILTER}" \
    --max_doc_length "${MAX_DOC_LENGTH}" \
    --max_doc_num "${MAX_DOC_NUM}" \
    --max_tool_definition_tokens "${MAX_TOOL_DEFINITION_TOKENS}" \
    --max_prompt_tokens "${MAX_PROMPT_TOKENS}" \
    --max_baseline_input_tokens "${MAX_BASELINE_INPUT_TOKENS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
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
SUMMARY_FILES=()
CASE_INDEX=0
BATCH_SIZE="${#_visible_npus[@]}"

for mode in "${_modes[@]}"; do
  mode="${mode// /}"
  case_ratios=("${_ratios[@]}")
  if [[ "${mode}" == "full" || "${mode}" == "reuse" || "${mode}" == "epic_leading32" || "${mode}" == "cacheblend_vdiff" ]]; then
    case_ratios=("1")
  elif [[ "${mode}" == "snapkv_reuse" || "${mode}" == "epic_leading32_snapkv" || "${mode}" == "cacheblend_vdiff_snapkv" ]]; then
    case_ratios=("4")
  fi
  for ratio in "${case_ratios[@]}"; do
    ratio="${ratio// /}"
    device="${_visible_npus[$((CASE_INDEX % BATCH_SIZE))]}"
    case_name="${mode}_r${ratio}"
    case_output="${TMP_DIR}/${case_name}.jsonl"
    case_summary="${TMP_DIR}/${case_name}.summary.json"
    case_log="${TMP_DIR}/${case_name}.log"
    rm -f "${case_output}" "${case_summary}" "${case_log}"
    CASE_OUTPUTS+=("${case_output}")
    SUMMARY_FILES+=("${case_summary}")
    echo "[launch] case=${case_name} device=${device} output=${case_output}"
    (
      export ASCEND_RT_VISIBLE_DEVICES="${device}"
      python agent/eval_agent_tool_definition_reuse_baselines.py \
        --device_type npu \
        --model "${MODEL_PATH}" \
        --base_model "${BASE_MODEL}" \
        --reuse_model "${BASE_MODEL}" \
        --tokenizer "${TOKENIZER_PATH}" \
        --dataset_path "${DATASET_PATH}" \
        --output_file "${case_output}" \
        --split "${SPLIT}" \
        "${SPLIT_ARGS[@]}" \
        --compare_modes "${mode}" \
        --ratios "${ratio}" \
        --hybrid_top_k "${HYBRID_TOP_K}" \
        --cacheblend_recompute_ratio "${CACHEBLEND_RECOMPUTE_RATIO}" \
        --max_examples "${MAX_EXAMPLES}" \
        --selection_filter "${SELECTION_FILTER}" \
        --max_doc_length "${MAX_DOC_LENGTH}" \
        --max_doc_num "${MAX_DOC_NUM}" \
        --max_tool_definition_tokens "${MAX_TOOL_DEFINITION_TOKENS}" \
        --max_prompt_tokens "${MAX_PROMPT_TOKENS}" \
        --max_baseline_input_tokens "${MAX_BASELINE_INPUT_TOKENS}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
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

python agent/merge_agent_tool_definition_reuse_baselines_eval.py \
  --output_file "${OUTPUT_FILE}" \
  --model "${MODEL_PATH}" \
  --base_model "${BASE_MODEL}" \
  --reuse_model "${BASE_MODEL}" \
  --dataset_path "${DATASET_PATH}" \
  --split "${SPLIT}" \
  --modes "${COMPARE_MODES}" \
  --ratios "${RATIOS}" \
  --cacheblend_recompute_ratio "${CACHEBLEND_RECOMPUTE_RATIO}" \
  --input_files "${CASE_OUTPUTS[@]}"

echo "Shard summaries:"
for summary in "${SUMMARY_FILES[@]}"; do
  echo "==== ${summary} ===="
  if [[ -f "${summary}" ]]; then
    cat "${summary}"
  else
    echo "MISSING summary file. Check log: ${summary%.summary.json}.log"
  fi
done
