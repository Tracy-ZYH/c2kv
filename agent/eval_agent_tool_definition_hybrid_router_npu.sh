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
SPLIT="${SPLIT:-eval}"
MAX_EXAMPLES="${MAX_EXAMPLES:-106}"
MAX_DOC_LENGTH="${MAX_DOC_LENGTH:-1024}"
MAX_DOC_NUM="${MAX_DOC_NUM:-10}"
MAX_TOOL_DEFINITION_TOKENS="${MAX_TOOL_DEFINITION_TOKENS:-10000}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
MAX_SOURCE_EXAMPLES="${MAX_SOURCE_EXAMPLES:-}"
SELECTION_FILTER="${SELECTION_FILTER:-c2kv}"
TOOL_DOCUMENT_EVAL_MODE="${TOOL_DOCUMENT_EVAL_MODE:-full}"
MIN_NUM_TOOLS="${MIN_NUM_TOOLS:-0}"
EVAL_RATIO="${EVAL_RATIO:-0.1}"
SPLIT_SEED="${SPLIT_SEED:-42}"
MAX_SAMPLES_PER_SESSION="${MAX_SAMPLES_PER_SESSION:-4}"
MIN_TARGET_TOKENS="${MIN_TARGET_TOKENS:-128}"
HYBRID_CASES="${HYBRID_CASES:-1:4,3:4,5:8}"
HYBRID_MODES="${HYBRID_MODES:-hybrid}"
ROUTER_SCOPE="${ROUTER_SCOPE:-last_user}"
ROUTER_STRATEGIES="${ROUTER_STRATEGIES:-lexical}"
ROUTER_HIT_FILTER="${ROUTER_HIT_FILTER:-all}"
ROUTER_SEED="${ROUTER_SEED:-42}"
TOP_SCHEMA_MODE="${TOP_SCHEMA_MODE:-full}"
ATTENTION_ROUTER_LAYERS="${ATTENTION_ROUTER_LAYERS:-4}"
ATTENTION_ROUTER_ATTN_IMPL="${ATTENTION_ROUTER_ATTN_IMPL:-eager}"
ATTENTION_ROUTER_MAX_QUERY_TOKENS="${ATTENTION_ROUTER_MAX_QUERY_TOKENS:-512}"
ATTENTION_ROUTER_SCORE_MODE="${ATTENTION_ROUTER_SCORE_MODE:-mean}"
ATTENTION_ROUTER_SPAN_TOP_TOKENS="${ATTENTION_ROUTER_SPAN_TOP_TOKENS:-4}"
ATTENTION_ROUTER_CACHE_MODE="${ATTENTION_ROUTER_CACHE_MODE:-c2kv}"
ATTENTION_ROUTER_LEXICAL_POOL="${ATTENTION_ROUTER_LEXICAL_POOL:-10}"
ATT_RERANK_POOL="${ATT_RERANK_POOL:-10}"
ATT_RERANK_MIN_HEADS="${ATT_RERANK_MIN_HEADS:-3}"
ATT_RERANK_MIN_MARGIN="${ATT_RERANK_MIN_MARGIN:-0.0}"
ATT_RERANK_MIN_SCORE_GAIN="${ATT_RERANK_MIN_SCORE_GAIN:-0.0}"
ATTENTION_RRF_K="${ATTENTION_RRF_K:-60.0}"
ATTENTION_STABLE_HEADS="${ATTENTION_STABLE_HEADS:-}"
ATTENTION_STABLE_HEAD_COUNT="${ATTENTION_STABLE_HEAD_COUNT:-16}"
NPU_ATTN_IMPL="${NPU_ATTN_IMPL:-eager}"
PARALLEL_EVAL="${PARALLEL_EVAL:-True}"
DEBUG_HYBRID_TOKENS="${DEBUG_HYBRID_TOKENS:-False}"
DUMP_HYBRID_DEFINITIONS="${DUMP_HYBRID_DEFINITIONS:-False}"
DEBUG_DEFINITION_CHARS="${DEBUG_DEFINITION_CHARS:-4000}"
SPLIT_MANIFEST_FILE="${SPLIT_MANIFEST_FILE:-}"
SPLIT_NAME="${SPLIT_NAME:-toolset_disjoint}"
OUTPUT_STEM="${OUTPUT_FILE%.jsonl}"
TMP_DIR="${TMP_DIR:-${OUTPUT_STEM}.parts}"

SPLIT_ARGS=(--split "${SPLIT}" --split_manifest_name "${SPLIT_NAME}")
if [[ -n "${SPLIT_MANIFEST_FILE}" ]]; then
  SPLIT_ARGS=(--split "${SPLIT}" --split_manifest_file "${SPLIT_MANIFEST_FILE}" --split_manifest_name "${SPLIT_NAME}")
fi

echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TOKENIZER_PATH=${TOKENIZER_PATH}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "OUTPUT_FILE=${OUTPUT_FILE}"
echo "SPLIT=${SPLIT}"
echo "MAX_EXAMPLES=${MAX_EXAMPLES}"
echo "MAX_SOURCE_EXAMPLES=${MAX_SOURCE_EXAMPLES}"
echo "SELECTION_FILTER=${SELECTION_FILTER}"
echo "TOOL_DOCUMENT_EVAL_MODE=${TOOL_DOCUMENT_EVAL_MODE}"
echo "MIN_NUM_TOOLS=${MIN_NUM_TOOLS}"
echo "HYBRID_CASES=${HYBRID_CASES}"
echo "HYBRID_MODES=${HYBRID_MODES}"
echo "ROUTER_SCOPE=${ROUTER_SCOPE}"
echo "ROUTER_STRATEGIES=${ROUTER_STRATEGIES}"
echo "ROUTER_HIT_FILTER=${ROUTER_HIT_FILTER}"
echo "TOP_SCHEMA_MODE=${TOP_SCHEMA_MODE}"
echo "ATTENTION_ROUTER_LAYERS=${ATTENTION_ROUTER_LAYERS}"
echo "ATTENTION_ROUTER_ATTN_IMPL=${ATTENTION_ROUTER_ATTN_IMPL}"
echo "ATTENTION_ROUTER_MAX_QUERY_TOKENS=${ATTENTION_ROUTER_MAX_QUERY_TOKENS}"
echo "ATTENTION_ROUTER_SCORE_MODE=${ATTENTION_ROUTER_SCORE_MODE}"
echo "ATTENTION_ROUTER_SPAN_TOP_TOKENS=${ATTENTION_ROUTER_SPAN_TOP_TOKENS}"
echo "ATTENTION_ROUTER_CACHE_MODE=${ATTENTION_ROUTER_CACHE_MODE}"
echo "ATTENTION_ROUTER_LEXICAL_POOL=${ATTENTION_ROUTER_LEXICAL_POOL}"
echo "ATT_RERANK_POOL=${ATT_RERANK_POOL}"
echo "ATT_RERANK_MIN_HEADS=${ATT_RERANK_MIN_HEADS}"
echo "ATT_RERANK_MIN_MARGIN=${ATT_RERANK_MIN_MARGIN}"
echo "ATT_RERANK_MIN_SCORE_GAIN=${ATT_RERANK_MIN_SCORE_GAIN}"
echo "ATTENTION_RRF_K=${ATTENTION_RRF_K}"
echo "ATTENTION_STABLE_HEADS=${ATTENTION_STABLE_HEADS}"
echo "ATTENTION_STABLE_HEAD_COUNT=${ATTENTION_STABLE_HEAD_COUNT}"
echo "MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS}"
echo "DEBUG_HYBRID_TOKENS=${DEBUG_HYBRID_TOKENS}"
echo "DUMP_HYBRID_DEFINITIONS=${DUMP_HYBRID_DEFINITIONS}"
echo "PARALLEL_EVAL=${PARALLEL_EVAL}"

DEBUG_ARGS=()
if [[ "${DEBUG_HYBRID_TOKENS}" == "True" || "${DEBUG_HYBRID_TOKENS}" == "true" || "${DEBUG_HYBRID_TOKENS}" == "1" ]]; then
  DEBUG_ARGS+=(--debug_hybrid_tokens)
fi
if [[ "${DUMP_HYBRID_DEFINITIONS}" == "True" || "${DUMP_HYBRID_DEFINITIONS}" == "true" || "${DUMP_HYBRID_DEFINITIONS}" == "1" ]]; then
  DEBUG_ARGS+=(--dump_hybrid_definitions --debug_definition_chars "${DEBUG_DEFINITION_CHARS}")
fi

SOURCE_ARGS=()
if [[ -n "${MAX_SOURCE_EXAMPLES}" ]]; then
  SOURCE_ARGS+=(--max_source_examples "${MAX_SOURCE_EXAMPLES}")
fi

if [[ "${PARALLEL_EVAL}" != "True" && "${PARALLEL_EVAL}" != "true" && "${PARALLEL_EVAL}" != "1" ]]; then
  python agent/eval_agent_tool_definition_hybrid_router.py \
    --device_type npu \
    --model "${MODEL_PATH}" \
    --base_model "${BASE_MODEL}" \
    --tokenizer "${TOKENIZER_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    --output_file "${OUTPUT_FILE}" \
    "${SPLIT_ARGS[@]}" \
    --max_examples "${MAX_EXAMPLES}" \
    "${SOURCE_ARGS[@]}" \
    --selection_filter "${SELECTION_FILTER}" \
    --tool_document_eval_mode "${TOOL_DOCUMENT_EVAL_MODE}" \
    --min_num_tools "${MIN_NUM_TOOLS}" \
    --eval_ratio "${EVAL_RATIO}" \
    --split_seed "${SPLIT_SEED}" \
    --max_samples_per_session "${MAX_SAMPLES_PER_SESSION}" \
    --max_doc_length "${MAX_DOC_LENGTH}" \
    --max_doc_num "${MAX_DOC_NUM}" \
    --max_tool_definition_tokens "${MAX_TOOL_DEFINITION_TOKENS}" \
    --max_prompt_tokens "${MAX_PROMPT_TOKENS}" \
    --min_target_tokens "${MIN_TARGET_TOKENS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --hybrid_cases "${HYBRID_CASES}" \
    --hybrid_mode "${HYBRID_MODES}" \
    --router_scope "${ROUTER_SCOPE}" \
    --router_strategy "${ROUTER_STRATEGIES}" \
    --top_schema_mode "${TOP_SCHEMA_MODE}" \
    --attention_router_layers "${ATTENTION_ROUTER_LAYERS}" \
    --attention_router_attn_impl "${ATTENTION_ROUTER_ATTN_IMPL}" \
    --attention_router_max_query_tokens "${ATTENTION_ROUTER_MAX_QUERY_TOKENS}" \
    --attention_router_score_mode "${ATTENTION_ROUTER_SCORE_MODE}" \
    --attention_router_span_top_tokens "${ATTENTION_ROUTER_SPAN_TOP_TOKENS}" \
    --attention_router_cache_mode "${ATTENTION_ROUTER_CACHE_MODE}" \
    --attention_router_lexical_pool "${ATTENTION_ROUTER_LEXICAL_POOL}" \
    --att_rerank_pool "${ATT_RERANK_POOL}" \
    --att_rerank_min_heads "${ATT_RERANK_MIN_HEADS}" \
    --att_rerank_min_margin "${ATT_RERANK_MIN_MARGIN}" \
    --att_rerank_min_score_gain "${ATT_RERANK_MIN_SCORE_GAIN}" \
    --attention_rrf_k "${ATTENTION_RRF_K}" \
    --attention_stable_heads "${ATTENTION_STABLE_HEADS}" \
    --attention_stable_head_count "${ATTENTION_STABLE_HEAD_COUNT}" \
    --router_hit_filter "${ROUTER_HIT_FILTER}" \
    --router_seed "${ROUTER_SEED}" \
    --system_attn_impl "${NPU_ATTN_IMPL}" \
    --gist_attn_impl "${NPU_ATTN_IMPL}" \
    --generate_attn_impl "${NPU_ATTN_IMPL}" \
    "${DEBUG_ARGS[@]}" \
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
          "${SPLIT_ARGS[@]}" \
          --max_examples "${MAX_EXAMPLES}" \
          "${SOURCE_ARGS[@]}" \
          --selection_filter "${SELECTION_FILTER}" \
          --tool_document_eval_mode "${TOOL_DOCUMENT_EVAL_MODE}" \
          --min_num_tools "${MIN_NUM_TOOLS}" \
          --eval_ratio "${EVAL_RATIO}" \
          --split_seed "${SPLIT_SEED}" \
          --max_samples_per_session "${MAX_SAMPLES_PER_SESSION}" \
          --max_doc_length "${MAX_DOC_LENGTH}" \
          --max_doc_num "${MAX_DOC_NUM}" \
          --max_tool_definition_tokens "${MAX_TOOL_DEFINITION_TOKENS}" \
          --max_prompt_tokens "${MAX_PROMPT_TOKENS}" \
          --min_target_tokens "${MIN_TARGET_TOKENS}" \
          --max_new_tokens "${MAX_NEW_TOKENS}" \
          --hybrid_cases "${case_spec}" \
          --hybrid_mode "${hybrid_mode}" \
          --router_scope "${ROUTER_SCOPE}" \
          --router_strategy "${strategy}" \
          --top_schema_mode "${TOP_SCHEMA_MODE}" \
          --attention_router_layers "${ATTENTION_ROUTER_LAYERS}" \
          --attention_router_attn_impl "${ATTENTION_ROUTER_ATTN_IMPL}" \
          --attention_router_max_query_tokens "${ATTENTION_ROUTER_MAX_QUERY_TOKENS}" \
          --attention_router_score_mode "${ATTENTION_ROUTER_SCORE_MODE}" \
          --attention_router_span_top_tokens "${ATTENTION_ROUTER_SPAN_TOP_TOKENS}" \
          --attention_router_cache_mode "${ATTENTION_ROUTER_CACHE_MODE}" \
          --attention_router_lexical_pool "${ATTENTION_ROUTER_LEXICAL_POOL}" \
          --att_rerank_pool "${ATT_RERANK_POOL}" \
          --att_rerank_min_heads "${ATT_RERANK_MIN_HEADS}" \
          --att_rerank_min_margin "${ATT_RERANK_MIN_MARGIN}" \
          --att_rerank_min_score_gain "${ATT_RERANK_MIN_SCORE_GAIN}" \
          --attention_rrf_k "${ATTENTION_RRF_K}" \
          --attention_stable_heads "${ATTENTION_STABLE_HEADS}" \
          --attention_stable_head_count "${ATTENTION_STABLE_HEAD_COUNT}" \
          --router_hit_filter "${ROUTER_HIT_FILTER}" \
          --router_seed "${ROUTER_SEED}" \
          --system_attn_impl "${NPU_ATTN_IMPL}" \
          --gist_attn_impl "${NPU_ATTN_IMPL}" \
          --generate_attn_impl "${NPU_ATTN_IMPL}" \
          "${DEBUG_ARGS[@]}" \
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
  --split "${SPLIT}" \
  --tool_document_eval_mode "${TOOL_DOCUMENT_EVAL_MODE}" \
  --router_scope "${ROUTER_SCOPE}" \
  --router_strategy "${ROUTER_STRATEGIES}" \
  --hybrid_modes "${HYBRID_MODES}" \
  --input_files "${CASE_OUTPUTS[@]}"
