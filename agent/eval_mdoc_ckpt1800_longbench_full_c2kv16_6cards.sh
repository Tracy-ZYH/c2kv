#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="$(pwd)/python:$(pwd)/python/inference:$(pwd)/agent:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

MODEL_PATH="${MODEL_PATH:-/home/zhuyuhan/project/c2kv/checkpoints/qwen3-4b-mixed-mdoc-c2kv-r4-8-16-npu-10k-30k/checkpoint-1800}"
BASE_MODEL="${BASE_MODEL:-/home/zhuyuhan/project/c2kv/models/Qwen3-4B-Instruct-2507}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/zhuyuhan/project/c2kv/outputs/mdoc_longbench_ckpt1800_full_c2kv16}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LONGBENCH_RAW_DIR="${LONGBENCH_RAW_DIR:-/home/zhuyuhan/project/c2kv/datasets/longbench_raw}"
LONGBENCH_DATA_DIR="${LONGBENCH_DATA_DIR:-${LONGBENCH_RAW_DIR}/data}"
DEVICE_TYPE="${DEVICE_TYPE:-npu}"

DEVICES_CSV="${DEVICES:-2,3,4,5,6,7}"
DATASETS_CSV="${DATASETS:-hotpotqa,wikimqa,multi_news,triviaqa,musique}"
MODES_CSV="${MODES:-full,c2kv}"
RANK_PLANS="${RANK_PLANS:-}"
RATIO="${RATIO:-16}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
MAX_DOC_LENGTH="${MAX_DOC_LENGTH:-2048}"
MAX_DOC_NUM="${MAX_DOC_NUM:-0}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-0}"
MAX_QUERY_TOKENS="${MAX_QUERY_TOKENS:-1024}"
DOC_SELECTION="${DOC_SELECTION:-head}"
HYBRID_CHUNK_TOP_K="${HYBRID_CHUNK_TOP_K:-8}"
HYBRID_CHUNK_TOKENS="${HYBRID_CHUNK_TOKENS:-256}"
HYBRID_CHUNK_OVERLAP="${HYBRID_CHUNK_OVERLAP:-64}"
HYBRID_CHUNK_RANKER="${HYBRID_CHUNK_RANKER:-lexical}"
ATTENTION_ROUTER_LAYERS="${ATTENTION_ROUTER_LAYERS:-4}"
ATTENTION_ROUTER_ATTN_IMPL="${ATTENTION_ROUTER_ATTN_IMPL:-eager}"
ATTENTION_ROUTER_MAX_QUERY_TOKENS="${ATTENTION_ROUTER_MAX_QUERY_TOKENS:-512}"
ATTENTION_ROUTER_SCORE_MODE="${ATTENTION_ROUTER_SCORE_MODE:-top4_mean}"
ATTENTION_ROUTER_SPAN_TOP_TOKENS="${ATTENTION_ROUTER_SPAN_TOP_TOKENS:-4}"
SYSTEM_ATTN_IMPL="${SYSTEM_ATTN_IMPL:-eager}"
if [[ "${DEVICE_TYPE}" == "cuda" ]]; then
  GIST_ATTN_IMPL="${GIST_ATTN_IMPL:-eager}"
  GENERATE_ATTN_IMPL="${GENERATE_ATTN_IMPL:-eager}"
else
  export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-max_split_size_mb:128}"
  GIST_ATTN_IMPL="${GIST_ATTN_IMPL:-npu_fusion_attention}"
  GENERATE_ATTN_IMPL="${GENERATE_ATTN_IMPL:-npu_fusion_attention}"
fi
DTYPE="${DTYPE:-bf16}"

mkdir -p "${OUTPUT_DIR}"

DATASET_SPECS=(
  "hotpotqa|${LONGBENCH_DATA_DIR}/hotpotqa.jsonl"
  "wikimqa|${LONGBENCH_DATA_DIR}/2wikimqa.jsonl"
  "multi_news|${LONGBENCH_DATA_DIR}/multi_news.jsonl"
  "triviaqa|${LONGBENCH_DATA_DIR}/triviaqa.jsonl"
  "musique|${LONGBENCH_DATA_DIR}/musique.jsonl"
)

IFS=',' read -r -a DEVICES <<< "${DEVICES_CSV}"
IFS=',' read -r -a SELECTED_DATASETS <<< "${DATASETS_CSV}"
IFS=',' read -r -a MODES <<< "${MODES_CSV}"
IFS=';' read -r -a RUN_RANK_PLANS <<< "${RANK_PLANS}"

if [[ "${#DEVICES[@]}" -eq 0 ]]; then
  echo "DEVICES is empty" >&2
  exit 1
fi

echo "MODEL_PATH=${MODEL_PATH}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "TOKENIZER_PATH=${TOKENIZER_PATH}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "LONGBENCH_RAW_DIR=${LONGBENCH_RAW_DIR}"
echo "LONGBENCH_DATA_DIR=${LONGBENCH_DATA_DIR}"
echo "DEVICE_TYPE=${DEVICE_TYPE}"
echo "DEVICES=${DEVICES_CSV}"
echo "DATASETS=${DATASETS_CSV}"
echo "MODES=${MODES_CSV}"
echo "RANK_PLANS=${RANK_PLANS}"
echo "RATIO=${RATIO}"
echo "MAX_EXAMPLES=${MAX_EXAMPLES}"
echo "MAX_DOC_LENGTH=${MAX_DOC_LENGTH}"
echo "MAX_DOC_NUM=${MAX_DOC_NUM}"
echo "MAX_CONTEXT_TOKENS=${MAX_CONTEXT_TOKENS}"
echo "HYBRID_CHUNK_TOP_K=${HYBRID_CHUNK_TOP_K}"
echo "HYBRID_CHUNK_TOKENS=${HYBRID_CHUNK_TOKENS}"
echo "HYBRID_CHUNK_OVERLAP=${HYBRID_CHUNK_OVERLAP}"
echo "HYBRID_CHUNK_RANKER=${HYBRID_CHUNK_RANKER}"
echo "ATTENTION_ROUTER_LAYERS=${ATTENTION_ROUTER_LAYERS}"
echo "ATTENTION_ROUTER_ATTN_IMPL=${ATTENTION_ROUTER_ATTN_IMPL}"
echo "ATTENTION_ROUTER_MAX_QUERY_TOKENS=${ATTENTION_ROUTER_MAX_QUERY_TOKENS}"
echo "ATTENTION_ROUTER_SCORE_MODE=${ATTENTION_ROUTER_SCORE_MODE}"
echo "ATTENTION_ROUTER_SPAN_TOP_TOKENS=${ATTENTION_ROUTER_SPAN_TOP_TOKENS}"

PIDS=()
PID_LABELS=()
ACTIVE=0
NEXT_DEVICE=0
FAILED=0

wait_one() {
  local pid="${PIDS[0]}"
  local label="${PID_LABELS[0]}"
  if wait "${pid}"; then
    echo "[done] ${label}"
  else
    echo "[failed] ${label}" >&2
    FAILED=1
  fi
  PIDS=("${PIDS[@]:1}")
  PID_LABELS=("${PID_LABELS[@]:1}")
  ACTIVE=$((ACTIVE - 1))
}

launch_job() {
  if (( $# < 4 )); then
    echo "[internal error] launch_job expects dataset, dataset_path, mode_spec, mode_position" >&2
    exit 1
  fi
  local dataset="$1"
  local dataset_path="$2"
  local mode_spec="${3:-}"
  local mode_position="${4:-0}"
  if [[ -z "${mode_spec}" ]]; then
    echo "[internal error] empty mode_spec for dataset=${dataset}" >&2
    exit 1
  fi
  local device="${DEVICES[$((NEXT_DEVICE % ${#DEVICES[@]}))]}"
  NEXT_DEVICE=$((NEXT_DEVICE + 1))

  local mode="${mode_spec}"
  local mode_name="${mode_spec}"
  local rank_plan="${RANK_PLAN:-1:full,2-:c2kv${RATIO}}"
  local chunk_top_k="${HYBRID_CHUNK_TOP_K}"
  local chunk_ranker="${HYBRID_CHUNK_RANKER}"
  if [[ "${mode_spec}" == rank_plan:* ]]; then
    mode="rank_plan"
    mode_name="${mode_spec#rank_plan:}"
    if (( mode_position < ${#RUN_RANK_PLANS[@]} )); then
      rank_plan="${RUN_RANK_PLANS[${mode_position}]}"
    fi
  fi
  if [[ "${mode_spec}" == chunk_hybrid:* ]]; then
    mode="chunk_hybrid"
    mode_name="${mode_spec#chunk_hybrid:}"
    if [[ "${mode_name}" == *attention_fullkv* ]] || [[ "${mode_name}" == *att_fullkv* ]]; then
      chunk_ranker="attention_fullkv"
    elif [[ "${mode_name}" == *attention_c2kv* ]] || [[ "${mode_name}" == *att_c2kv* ]] || [[ "${mode_name}" == *compressedkv* ]]; then
      chunk_ranker="attention_c2kv"
    elif [[ "${mode_name}" == *bm25* ]]; then
      chunk_ranker="bm25"
    elif [[ "${mode_name}" == *lex* ]] || [[ "${mode_name}" == *overlap* ]]; then
      chunk_ranker="lexical"
    fi
    if [[ "${mode_name}" =~ top([0-9]+) ]]; then
      chunk_top_k="${BASH_REMATCH[1]}"
    fi
  fi

  local output_file="${OUTPUT_DIR}/${dataset}_${mode_name}_r${RATIO}.jsonl"
  local log_file="${OUTPUT_DIR}/${dataset}_${mode_name}_r${RATIO}.log"
  local label="${dataset}/${mode_name}/${DEVICE_TYPE}${device}"

  echo "[launch] ${label} mode=${mode} rank_plan=${rank_plan} chunk_top_k=${chunk_top_k} chunk_ranker=${chunk_ranker} output=${output_file}"
  (
    if [[ "${DEVICE_TYPE}" == "cuda" ]]; then
      export CUDA_VISIBLE_DEVICES="${device}"
      unset ASCEND_RT_VISIBLE_DEVICES || true
      unset ASCEND_VISIBLE_DEVICES || true
    else
      export ASCEND_RT_VISIBLE_DEVICES="${device}"
      export ASCEND_VISIBLE_DEVICES="${device}"
    fi
    "${PYTHON_BIN}" python/inference/expr_mdoc_c2kv_baselines.py \
      --model "${MODEL_PATH}" \
      --base_model "${BASE_MODEL}" \
      --tokenizer "${TOKENIZER_PATH}" \
      --dataset "${dataset}" \
      --dataset_path "${dataset_path}" \
      --mode "${mode}" \
      --override_ratio "${RATIO}" \
      --rank_plan "${rank_plan}" \
      --max_examples "${MAX_EXAMPLES}" \
      --output_file "${output_file}" \
      --max_doc_length "${MAX_DOC_LENGTH}" \
      --max_doc_num "${MAX_DOC_NUM}" \
      --max_context_tokens "${MAX_CONTEXT_TOKENS}" \
      --max_query_tokens "${MAX_QUERY_TOKENS}" \
      --doc_selection "${DOC_SELECTION}" \
      --hybrid_chunk_top_k "${chunk_top_k}" \
      --hybrid_chunk_tokens "${HYBRID_CHUNK_TOKENS}" \
      --hybrid_chunk_overlap "${HYBRID_CHUNK_OVERLAP}" \
      --hybrid_chunk_ranker "${chunk_ranker}" \
      --attention_router_layers "${ATTENTION_ROUTER_LAYERS}" \
      --attention_router_attn_impl "${ATTENTION_ROUTER_ATTN_IMPL}" \
      --attention_router_max_query_tokens "${ATTENTION_ROUTER_MAX_QUERY_TOKENS}" \
      --attention_router_score_mode "${ATTENTION_ROUTER_SCORE_MODE}" \
      --attention_router_span_top_tokens "${ATTENTION_ROUTER_SPAN_TOP_TOKENS}" \
      --device_type "${DEVICE_TYPE}" \
      --system_attn_impl "${SYSTEM_ATTN_IMPL}" \
      --gist_attn_impl "${GIST_ATTN_IMPL}" \
      --generate_attn_impl "${GENERATE_ATTN_IMPL}" \
      --dtype "${DTYPE}"
  ) > "${log_file}" 2>&1 &

  PIDS+=("$!")
  PID_LABELS+=("${label}")
  ACTIVE=$((ACTIVE + 1))
}

dataset_selected() {
  local candidate="$1"
  local selected
  for selected in "${SELECTED_DATASETS[@]}"; do
    selected="${selected// /}"
    if [[ "${selected}" == "${candidate}" ]]; then
      return 0
    fi
  done
  return 1
}

for spec in "${DATASET_SPECS[@]}"; do
  dataset="${spec%%|*}"
  dataset_path="${spec#*|}"
  if ! dataset_selected "${dataset}"; then
    continue
  fi
  if [[ ! -e "${dataset_path}" ]]; then
    echo "[skip] missing dataset_path=${dataset_path}" >&2
    FAILED=1
    continue
  fi
  for mode_index in "${!MODES[@]}"; do
    mode="${MODES[${mode_index}]}"
    mode="${mode// /}"
    if [[ -z "${mode}" ]]; then
      continue
    fi
    launch_job "${dataset}" "${dataset_path}" "${mode}" "${mode_index}"
    if (( ACTIVE >= ${#DEVICES[@]} )); then
      wait_one
    fi
  done
done

while (( ACTIVE > 0 )); do
  wait_one
done

echo "==== summaries ===="
for spec in "${DATASET_SPECS[@]}"; do
  dataset="${spec%%|*}"
  if ! dataset_selected "${dataset}"; then
    continue
  fi
  for mode in "${MODES[@]}"; do
    mode="${mode// /}"
    mode_name="${mode}"
    if [[ "${mode}" == rank_plan:* ]]; then
      mode_name="${mode#rank_plan:}"
    elif [[ "${mode}" == chunk_hybrid:* ]]; then
      mode_name="${mode#chunk_hybrid:}"
    fi
    summary="${OUTPUT_DIR}/${dataset}_${mode_name}_r${RATIO}.summary.json"
    echo "---- ${summary} ----"
    if [[ -f "${summary}" ]]; then
      cat "${summary}"
    else
      echo "missing summary; check ${OUTPUT_DIR}/${dataset}_${mode_name}_r${RATIO}.log"
    fi
  done
done

exit "${FAILED}"
