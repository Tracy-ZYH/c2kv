#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)/python:$(pwd)/python/inference:$(pwd)/agent:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,3,4,5,6,7}"

MODEL_PATH="${MODEL_PATH:-./checkpoints/qwen3-4b-agent-history-c2kv-npu}"
BASE_MODEL="${BASE_MODEL:-./models/Qwen3-4B-Instruct-2507}"
TOKENIZER_PATH="${HISTORY_TOKENIZER_PATH:-${BASE_MODEL}}"
DATASET_PATH="${DATASET_PATH:-./datasets/agent-llm-traces}"
OUTPUT_FILE="${OUTPUT_FILE:-./outputs/agent_history_c2kv_eval_npu.jsonl}"
SPLIT="${SPLIT:-eval}"

COMPARE_MODES="${COMPARE_MODES:-full,truncate,c2kv,hybrid}"
RATIOS="${RATIOS:-4}"
HYBRID_TOP_K="${HYBRID_TOP_K:-3}"
ATTENTION_ROUTER_LAYERS="${ATTENTION_ROUTER_LAYERS:-32}"
ATTENTION_ROUTER_ATTN_IMPL="${ATTENTION_ROUTER_ATTN_IMPL:-${NPU_ATTN_IMPL:-eager}}"
ATTENTION_ROUTER_MAX_QUERY_TOKENS="${ATTENTION_ROUTER_MAX_QUERY_TOKENS:-512}"
ATTENTION_ROUTER_SCORE_MODE="${ATTENTION_ROUTER_SCORE_MODE:-mean}"
ATT_RERANK_POOL="${ATT_RERANK_POOL:-10}"
ATT_RERANK_MIN_HEADS="${ATT_RERANK_MIN_HEADS:-30}"
ATT_RERANK_MIN_MARGIN="${ATT_RERANK_MIN_MARGIN:-0.0}"
ATT_RERANK_MIN_SCORE_GAIN="${ATT_RERANK_MIN_SCORE_GAIN:-0.0}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
MAX_SOURCE_EXAMPLES="${MAX_SOURCE_EXAMPLES:-}"
SELECTION_FILTER="${SELECTION_FILTER:-c2kv}"

SPLIT_MANIFEST_FILE="${SPLIT_MANIFEST_FILE:-}"
SPLIT_NAME="${SPLIT_NAME:-subset_disjoint}"
SPLIT_SEED="${SPLIT_SEED:-42}"
EVAL_RATIO="${EVAL_RATIO:-0.1}"
MAX_SAMPLES_PER_SESSION="${MAX_SAMPLES_PER_SESSION:-4}"
REQUIRE_TOOL_CALL="${REQUIRE_TOOL_CALL:-False}"
INCLUDE_TOOLS="${INCLUDE_TOOLS:-True}"

MAX_DOC_LENGTH="${HISTORY_MAX_DOC_LENGTH:-768}"
MAX_DOC_NUM="${HISTORY_MAX_DOC_NUM:-16}"
MIN_DOC_NUM="${HISTORY_MIN_DOC_NUM:-1}"
MAX_HISTORY_TOKENS="${MAX_HISTORY_TOKENS:-12288}"
MAX_LENGTH="${MAX_LENGTH:-1536}"
MAX_SYSTEM_LENGTH="${MAX_SYSTEM_LENGTH:-4096}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-1536}"
MAX_BASELINE_INPUT_TOKENS="${MAX_BASELINE_INPUT_TOKENS:-16000}"
RAW_FIRST_N_TURNS="${RAW_FIRST_N_TURNS:-15}"
RAW_PREFIX_N_TURNS="${RAW_PREFIX_N_TURNS:-8}"
RAW_PREFIX_NEXT_TARGET_SCOPE="${RAW_PREFIX_NEXT_TARGET_SCOPE:-turn}"
DUMP_RAW_HISTORY_DOCS="${DUMP_RAW_HISTORY_DOCS:-False}"
RAW_HISTORY_DOC_DEBUG_CHARS="${RAW_HISTORY_DOC_DEBUG_CHARS:-2000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
HISTORY_SELECTION="${HISTORY_SELECTION:-tail}"
TRUNCATE_SELECTION="${TRUNCATE_SELECTION:-tail}"
MAX_INPUT_CHARS="${MAX_INPUT_CHARS:-}"
MAX_ANSWER_CHARS="${MAX_ANSWER_CHARS:-}"
PREFIX_HISTORY_DOC_NUM="${PREFIX_HISTORY_DOC_NUM:-}"
PREFIX_HISTORY_EXACT="${PREFIX_HISTORY_EXACT:-False}"
SPLIT_OVERSIZED_HISTORY_DOCS="${SPLIT_OVERSIZED_HISTORY_DOCS:-True}"
HYBRID_FULL_AFTER_C2KV="${HYBRID_FULL_AFTER_C2KV:-False}"
NPU_ATTN_IMPL="${NPU_ATTN_IMPL:-eager}"
PARALLEL_EVAL="${PARALLEL_EVAL:-True}"
OUTPUT_STEM="${OUTPUT_FILE%.jsonl}"
TMP_DIR="${TMP_DIR:-${OUTPUT_STEM}.parts}"

SPLIT_ARGS=(--split_manifest_name "${SPLIT_NAME}")
if [[ -n "${SPLIT_MANIFEST_FILE}" ]]; then
  SPLIT_ARGS=(--split_manifest_file "${SPLIT_MANIFEST_FILE}" --split_manifest_name "${SPLIT_NAME}")
fi

OPTIONAL_ARGS=()
if [[ -n "${MAX_SOURCE_EXAMPLES}" ]]; then
  OPTIONAL_ARGS+=(--max_source_examples "${MAX_SOURCE_EXAMPLES}")
fi
if [[ -n "${MAX_INPUT_CHARS}" ]]; then
  OPTIONAL_ARGS+=(--max_input_chars "${MAX_INPUT_CHARS}")
fi
if [[ -n "${MAX_ANSWER_CHARS}" ]]; then
  OPTIONAL_ARGS+=(--max_answer_chars "${MAX_ANSWER_CHARS}")
fi
if [[ -n "${PREFIX_HISTORY_DOC_NUM}" ]]; then
  OPTIONAL_ARGS+=(--prefix_history_doc_num "${PREFIX_HISTORY_DOC_NUM}")
fi
OPTIONAL_ARGS+=(--prefix_history_exact "${PREFIX_HISTORY_EXACT}")
OPTIONAL_ARGS+=(--split_oversized_history_docs "${SPLIT_OVERSIZED_HISTORY_DOCS}")
OPTIONAL_ARGS+=(--hybrid_full_after_c2kv "${HYBRID_FULL_AFTER_C2KV}")
if [[ "${DUMP_RAW_HISTORY_DOCS}" == "True" || "${DUMP_RAW_HISTORY_DOCS}" == "true" || "${DUMP_RAW_HISTORY_DOCS}" == "1" ]]; then
  OPTIONAL_ARGS+=(--dump_raw_history_docs --raw_history_doc_debug_chars "${RAW_HISTORY_DOC_DEBUG_CHARS}")
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
echo "HYBRID_TOP_K=${HYBRID_TOP_K}"
echo "MAX_EXAMPLES=${MAX_EXAMPLES}"
echo "MAX_DOC_LENGTH=${MAX_DOC_LENGTH}"
echo "MAX_DOC_NUM=${MAX_DOC_NUM}"
echo "MAX_HISTORY_TOKENS=${MAX_HISTORY_TOKENS}"
echo "MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS}"
echo "MAX_BASELINE_INPUT_TOKENS=${MAX_BASELINE_INPUT_TOKENS}"
echo "RAW_FIRST_N_TURNS=${RAW_FIRST_N_TURNS}"
echo "RAW_PREFIX_N_TURNS=${RAW_PREFIX_N_TURNS}"
echo "RAW_PREFIX_NEXT_TARGET_SCOPE=${RAW_PREFIX_NEXT_TARGET_SCOPE}"
echo "DUMP_RAW_HISTORY_DOCS=${DUMP_RAW_HISTORY_DOCS}"
echo "HISTORY_SELECTION=${HISTORY_SELECTION}"
echo "TRUNCATE_SELECTION=${TRUNCATE_SELECTION}"
echo "PREFIX_HISTORY_DOC_NUM=${PREFIX_HISTORY_DOC_NUM}"
echo "PREFIX_HISTORY_EXACT=${PREFIX_HISTORY_EXACT}"
echo "SPLIT_OVERSIZED_HISTORY_DOCS=${SPLIT_OVERSIZED_HISTORY_DOCS}"
echo "HYBRID_FULL_AFTER_C2KV=${HYBRID_FULL_AFTER_C2KV}"
echo "INCLUDE_TOOLS=${INCLUDE_TOOLS}"
echo "PARALLEL_EVAL=${PARALLEL_EVAL}"

COMMON_ARGS=(
  --device_type npu
  --model "${MODEL_PATH}"
  --base_model "${BASE_MODEL}"
  --tokenizer "${TOKENIZER_PATH}"
  --dataset_path "${DATASET_PATH}"
  --split "${SPLIT}"
  "${SPLIT_ARGS[@]}"
  --hybrid_top_k "${HYBRID_TOP_K}"
  --attention_router_layers "${ATTENTION_ROUTER_LAYERS}"
  --attention_router_attn_impl "${ATTENTION_ROUTER_ATTN_IMPL}"
  --attention_router_max_query_tokens "${ATTENTION_ROUTER_MAX_QUERY_TOKENS}"
  --attention_router_score_mode "${ATTENTION_ROUTER_SCORE_MODE}"
  --att_rerank_pool "${ATT_RERANK_POOL}"
  --att_rerank_min_heads "${ATT_RERANK_MIN_HEADS}"
  --att_rerank_min_margin "${ATT_RERANK_MIN_MARGIN}"
  --att_rerank_min_score_gain "${ATT_RERANK_MIN_SCORE_GAIN}"
  --max_examples "${MAX_EXAMPLES}"
  --selection_filter "${SELECTION_FILTER}"
  --eval_ratio "${EVAL_RATIO}"
  --split_seed "${SPLIT_SEED}"
  --max_samples_per_session "${MAX_SAMPLES_PER_SESSION}"
  --require_tool_call "${REQUIRE_TOOL_CALL}"
  --include_tools "${INCLUDE_TOOLS}"
  --max_doc_length "${MAX_DOC_LENGTH}"
  --min_doc_num "${MIN_DOC_NUM}"
  --max_doc_num "${MAX_DOC_NUM}"
  --max_history_tokens "${MAX_HISTORY_TOKENS}"
  --max_length "${MAX_LENGTH}"
  --max_system_length "${MAX_SYSTEM_LENGTH}"
  --max_prompt_tokens "${MAX_PROMPT_TOKENS}"
  --max_baseline_input_tokens "${MAX_BASELINE_INPUT_TOKENS}"
  --raw_first_n_turns "${RAW_FIRST_N_TURNS}"
  --raw_prefix_n_turns "${RAW_PREFIX_N_TURNS}"
  --raw_prefix_next_target_scope "${RAW_PREFIX_NEXT_TARGET_SCOPE}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --history_selection "${HISTORY_SELECTION}"
  --truncate_selection "${TRUNCATE_SELECTION}"
  --system_attn_impl "${NPU_ATTN_IMPL}"
  --gist_attn_impl "${NPU_ATTN_IMPL}"
  --generate_attn_impl "${NPU_ATTN_IMPL}"
  "${OPTIONAL_ARGS[@]}"
)

if [[ "${PARALLEL_EVAL}" != "True" && "${PARALLEL_EVAL}" != "true" && "${PARALLEL_EVAL}" != "1" ]]; then
  python agent/eval_agent_history_c2kv.py \
    --output_file "${OUTPUT_FILE}" \
    --compare_modes "${COMPARE_MODES}" \
    --ratios "${RATIOS}" \
    "${COMMON_ARGS[@]}"
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
FULL_RATIO_MODES=",full,history_full,all_full,original_replay_full,reconstructed_contiguous_full,raw_first15_full,raw_first15_full_same_model,raw_first8_full,raw_first8_full_same_model,raw_prefix8_exact_full,raw_prefix8_exact_full_same_model,raw_prefix_next_full,raw_prefix_next_full_same_model,split_full_kv,sequential_full_kv,current_only,"

for mode in "${_modes[@]}"; do
  mode="${mode// /}"
  case_ratios=("${_ratios[@]}")
  if [[ "${FULL_RATIO_MODES}" == *",${mode},"* ]]; then
    case_ratios=("1")
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
      python agent/eval_agent_history_c2kv.py \
        --output_file "${case_output}" \
        --compare_modes "${mode}" \
        --ratios "${ratio}" \
        "${COMMON_ARGS[@]}"
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
  --reuse_model "${MODEL_PATH}" \
  --dataset_path "${DATASET_PATH}" \
  --split "${SPLIT}" \
  --modes "${COMPARE_MODES}" \
  --ratios "${RATIOS}" \
  --input_files "${CASE_OUTPUTS[@]}"

echo "Shard summaries:"
for summary in "${SUMMARY_FILES[@]}"; do
  echo "==== ${summary} ===="
  if [[ -f "${summary}" ]]; then
    cat "${summary}"
  else
    summary_log="${summary%.summary.json}.log"
    echo "MISSING summary file. Check log: ${summary_log}"
  fi
done
