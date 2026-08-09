#!/usr/bin/env bash
set -euo pipefail

# Tool-definition router vote suite on NPU.
# Compares:
# - full / c2kv
# - lexical hybrid
# - Current: lexical top-10 + all-head mean direct top-3
# - Vote-All: lexical top-10 + all-head confidence-weighted RRF
# - Stable-Vote: lexical top-10 + stable top-16 heads confidence-weighted RRF
# - Conservative-Vote: lexical top-10 + stable top-16 heads, keep lexical top-2 and replace rank-3 only

export MODEL_PATH="${MODEL_PATH:-./checkpoints/qwen3-4b-agent-tooldoc-hardneg-npu}"
export BASE_MODEL="${BASE_MODEL:-./models/Qwen3-4B-Instruct-2507}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${BASE_MODEL}}"
export DATASET_PATH="${DATASET_PATH:-./datasets/agent-llm-traces}"
export OUTPUT_FILE="${OUTPUT_FILE:-./outputs/ablation_0804_tooldef_vote_router_suite.jsonl}"
export SPLIT="${SPLIT:-eval}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-4,5,6,7}"
export HYBRID_CASES="${HYBRID_CASES:-3:4}"
export ROUTER_SCOPE="${ROUTER_SCOPE:-last_user}"
export ROUTER_HIT_FILTER="${ROUTER_HIT_FILTER:-all}"
export MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
export SELECTION_FILTER="${SELECTION_FILTER:-c2kv}"
export MIN_NUM_TOOLS="${MIN_NUM_TOOLS:-4}"
export SPLIT_MANIFEST_FILE="${SPLIT_MANIFEST_FILE:-}"
export SPLIT_NAME="${SPLIT_NAME:-toolset_disjoint}"
export MAX_DOC_LENGTH="${MAX_DOC_LENGTH:-1024}"
export MAX_DOC_NUM="${MAX_DOC_NUM:-10}"
export MAX_TOOL_DEFINITION_TOKENS="${MAX_TOOL_DEFINITION_TOKENS:-10000}"
export MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-4096}"
export MAX_BASELINE_INPUT_TOKENS="${MAX_BASELINE_INPUT_TOKENS:-0}"
export MAX_HYBRID_DECODE_TOKENS="${MAX_HYBRID_DECODE_TOKENS:-0}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
export NPU_ATTN_IMPL="${NPU_ATTN_IMPL:-eager}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-max_split_size_mb:512}"

export ATTENTION_ROUTER_LAYERS="${ATTENTION_ROUTER_LAYERS:-32}"
export ATTENTION_ROUTER_ATTN_IMPL="${ATTENTION_ROUTER_ATTN_IMPL:-eager}"
export ATTENTION_ROUTER_MAX_QUERY_TOKENS="${ATTENTION_ROUTER_MAX_QUERY_TOKENS:-512}"
export ATTENTION_ROUTER_SCORE_MODE="${ATTENTION_ROUTER_SCORE_MODE:-top4_mean}"
export ATTENTION_ROUTER_SPAN_TOP_TOKENS="${ATTENTION_ROUTER_SPAN_TOP_TOKENS:-4}"
export ATTENTION_ROUTER_CACHE_MODE="${ATTENTION_ROUTER_CACHE_MODE:-full}"
export ATTENTION_ROUTER_LEXICAL_POOL="${ATTENTION_ROUTER_LEXICAL_POOL:-10}"
export ATTENTION_RRF_K="${ATTENTION_RRF_K:-60.0}"
export ATTENTION_STABLE_HEADS="${ATTENTION_STABLE_HEADS:-}"
export ATTENTION_STABLE_HEAD_COUNT="${ATTENTION_STABLE_HEAD_COUNT:-16}"

export DEBUG_HYBRID_TOKENS="${DEBUG_HYBRID_TOKENS:-True}"
export DUMP_HYBRID_DEFINITIONS="${DUMP_HYBRID_DEFINITIONS:-False}"

OUTPUT_STEM="${OUTPUT_FILE%.jsonl}"
RUN_DIR="${RUN_DIR:-${OUTPUT_STEM}.runs}"
mkdir -p "${RUN_DIR}"

BASELINE_OUTPUT="${RUN_DIR}/full_c2kv.jsonl"
LEXICAL_OUTPUT="${RUN_DIR}/c2kv_hybrid_lexical.jsonl"
CURRENT_OUTPUT="${RUN_DIR}/current_lex_attention.jsonl"
VOTE_ALL_OUTPUT="${RUN_DIR}/vote_all.jsonl"
STABLE_VOTE_OUTPUT="${RUN_DIR}/stable_vote.jsonl"
CONSERVATIVE_OUTPUT="${RUN_DIR}/conservative_vote.jsonl"

echo "OUTPUT_FILE=${OUTPUT_FILE}"
echo "RUN_DIR=${RUN_DIR}"
echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "HYBRID_CASES=${HYBRID_CASES}"
echo "ATTENTION_ROUTER_SCORE_MODE=${ATTENTION_ROUTER_SCORE_MODE}"
echo "ATTENTION_ROUTER_CACHE_MODE=${ATTENTION_ROUTER_CACHE_MODE}"
echo "ATTENTION_ROUTER_LEXICAL_POOL=${ATTENTION_ROUTER_LEXICAL_POOL}"
echo "ATTENTION_STABLE_HEAD_COUNT=${ATTENTION_STABLE_HEAD_COUNT}"

IFS=',' read -ra DEVICES <<< "${ASCEND_RT_VISIBLE_DEVICES}"
if (( ${#DEVICES[@]} < 4 )); then
  echo "Need at least 4 visible NPU ids for the default suite." >&2
  exit 1
fi

COMMON_ENV=(
  MODEL_PATH="${MODEL_PATH}"
  BASE_MODEL="${BASE_MODEL}"
  TOKENIZER_PATH="${TOKENIZER_PATH}"
  DATASET_PATH="${DATASET_PATH}"
  SPLIT="${SPLIT}"
  HYBRID_CASES="${HYBRID_CASES}"
  ROUTER_SCOPE="${ROUTER_SCOPE}"
  ROUTER_HIT_FILTER="${ROUTER_HIT_FILTER}"
  MAX_EXAMPLES="${MAX_EXAMPLES}"
  SELECTION_FILTER="${SELECTION_FILTER}"
  MIN_NUM_TOOLS="${MIN_NUM_TOOLS}"
  SPLIT_MANIFEST_FILE="${SPLIT_MANIFEST_FILE}"
  SPLIT_NAME="${SPLIT_NAME}"
  MAX_DOC_LENGTH="${MAX_DOC_LENGTH}"
  MAX_DOC_NUM="${MAX_DOC_NUM}"
  MAX_TOOL_DEFINITION_TOKENS="${MAX_TOOL_DEFINITION_TOKENS}"
  MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS}"
  MAX_BASELINE_INPUT_TOKENS="${MAX_BASELINE_INPUT_TOKENS}"
  MAX_HYBRID_DECODE_TOKENS="${MAX_HYBRID_DECODE_TOKENS}"
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS}"
  NPU_ATTN_IMPL="${NPU_ATTN_IMPL}"
  ATTENTION_ROUTER_LAYERS="${ATTENTION_ROUTER_LAYERS}"
  ATTENTION_ROUTER_ATTN_IMPL="${ATTENTION_ROUTER_ATTN_IMPL}"
  ATTENTION_ROUTER_MAX_QUERY_TOKENS="${ATTENTION_ROUTER_MAX_QUERY_TOKENS}"
  ATTENTION_ROUTER_SCORE_MODE="${ATTENTION_ROUTER_SCORE_MODE}"
  ATTENTION_ROUTER_SPAN_TOP_TOKENS="${ATTENTION_ROUTER_SPAN_TOP_TOKENS}"
  ATTENTION_ROUTER_CACHE_MODE="${ATTENTION_ROUTER_CACHE_MODE}"
  ATTENTION_ROUTER_LEXICAL_POOL="${ATTENTION_ROUTER_LEXICAL_POOL}"
  ATTENTION_RRF_K="${ATTENTION_RRF_K}"
  ATTENTION_STABLE_HEADS="${ATTENTION_STABLE_HEADS}"
  ATTENTION_STABLE_HEAD_COUNT="${ATTENTION_STABLE_HEAD_COUNT}"
  DEBUG_HYBRID_TOKENS="${DEBUG_HYBRID_TOKENS}"
  DUMP_HYBRID_DEFINITIONS="${DUMP_HYBRID_DEFINITIONS}"
)

PIDS=()

echo "[launch] full+c2kv on ${DEVICES[0]},${DEVICES[1]}"
(
  export ASCEND_RT_VISIBLE_DEVICES="${DEVICES[0]},${DEVICES[1]}"
  export OUTPUT_FILE="${BASELINE_OUTPUT}"
  export TMP_DIR="${RUN_DIR}/full_c2kv.parts"
  export COMPARE_MODES="full,c2kv"
  export RATIOS="4"
  export PARALLEL_EVAL=True
  env "${COMMON_ENV[@]}" bash agent/eval_agent_tool_definition_reuse_baselines_npu.sh
) > "${RUN_DIR}/full_c2kv.log" 2>&1 &
PIDS+=("$!")

echo "[launch] lexical/current/vote-all/stable-vote on ${DEVICES[2]}"
(
  export ASCEND_RT_VISIBLE_DEVICES="${DEVICES[2]}"
  export OUTPUT_FILE="${RUN_DIR}/router_batch_a.jsonl"
  export TMP_DIR="${RUN_DIR}/router_batch_a.parts"
  export HYBRID_MODES="hybrid"
  export ROUTER_STRATEGIES="lexical,lex_attention,vote_all,stable_vote"
  export TOP_SCHEMA_MODE="full"
  export PARALLEL_EVAL=True
  env "${COMMON_ENV[@]}" bash agent/eval_agent_tool_definition_hybrid_router_npu.sh
) > "${RUN_DIR}/router_batch_a.log" 2>&1 &
PIDS+=("$!")

echo "[launch] conservative-vote on ${DEVICES[3]}"
(
  export ASCEND_RT_VISIBLE_DEVICES="${DEVICES[3]}"
  export OUTPUT_FILE="${CONSERVATIVE_OUTPUT}"
  export TMP_DIR="${RUN_DIR}/conservative_vote.parts"
  export HYBRID_MODES="hybrid"
  export ROUTER_STRATEGIES="conservative_vote"
  export TOP_SCHEMA_MODE="full"
  export PARALLEL_EVAL=True
  env "${COMMON_ENV[@]}" bash agent/eval_agent_tool_definition_hybrid_router_npu.sh
) > "${RUN_DIR}/conservative_vote.log" 2>&1 &
PIDS+=("$!")

for pid in "${PIDS[@]}"; do
  wait "${pid}"
done

python agent/merge_agent_tool_definition_reuse_baselines_eval.py \
  --output_file "${OUTPUT_FILE}" \
  --model "${MODEL_PATH}" \
  --base_model "${BASE_MODEL}" \
  --reuse_model "${BASE_MODEL}" \
  --dataset_path "${DATASET_PATH}" \
  --split "${SPLIT}" \
  --modes "full,c2kv,c2kv_hybrid,current,vote_all,stable_vote,conservative_vote" \
  --ratios "1,4" \
  --input_files \
    "${BASELINE_OUTPUT}" \
    "${RUN_DIR}/router_batch_a.jsonl" \
    "${CONSERVATIVE_OUTPUT}"

echo "Done."
echo "Merged rows: ${OUTPUT_FILE}"
echo "Summary: ${OUTPUT_STEM}.summary.json"
echo "Logs are under ${RUN_DIR}/"
