#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/zhuyuhan/project/c2kv}"
SGLANG_DIR="${SGLANG_DIR:-/home/zhuyuhan/project/kvoffload-sglang}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhuyuhan/miniconda3/envs/sglang/bin/python}"

MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/checkpoints/qwen3-4b-agent-tooldoc-hardneg-npu}"
BASE_MODEL="${BASE_MODEL:-${ROOT_DIR}/models/Qwen3-4B-Instruct-2507}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${BASE_MODEL}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-agent-tooldef}"
DATASET_PATH="${DATASET_PATH:-${ROOT_DIR}/datasets/agent-llm-traces}"
DEVICES_CSV="${DEVICES:-0,1,2}"
MODES_CSV="${MODES:-full,c2kv,hybrid}"
HOST="${HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-32000}"
RATIO="${RATIO:-4}"
MAX_EXAMPLES="${MAX_EXAMPLES:-20}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.7}"
RUN_NAME="${RUN_NAME:-agent_tooldef_hardneg_sglang_api}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/outputs/${RUN_NAME}}"
START_STAGGER_SECONDS="${START_STAGGER_SECONDS:-2}"

SPLIT="${SPLIT:-eval}"
SPLIT_NAME="${SPLIT_NAME:-toolset_disjoint}"
SPLIT_MANIFEST_FILE="${SPLIT_MANIFEST_FILE:-}"
MAX_DOC_LENGTH="${MAX_DOC_LENGTH:-1024}"
MAX_DOC_NUM="${MAX_DOC_NUM:-64}"
MAX_TOOL_DEFINITION_TOKENS="${MAX_TOOL_DEFINITION_TOKENS:-32000}"
MAX_SAMPLES_PER_SESSION="${MAX_SAMPLES_PER_SESSION:-4}"
MIN_TARGET_TOKENS="${MIN_TARGET_TOKENS:-64}"
MIN_NUM_TOOLS="${MIN_NUM_TOOLS:-4}"
DATASET_TOOL_DOCUMENT_MODE="${DATASET_TOOL_DOCUMENT_MODE:-full}"
TOOL_DOCUMENT_EVAL_MODE="${TOOL_DOCUMENT_EVAL_MODE:-per_tool}"
HARD_NEGATIVE_NUM="${HARD_NEGATIVE_NUM:-15}"
HARD_NEGATIVE_ROUTER_SCOPE="${HARD_NEGATIVE_ROUTER_SCOPE:-last_user}"
SHUFFLE_TOOL_DOCUMENTS="${SHUFFLE_TOOL_DOCUMENTS:-True}"
BALANCE_SUBSETS="${BALANCE_SUBSETS:-True}"
HYBRID_TOP_K="${HYBRID_TOP_K:-3}"
ROUTER_SCOPE="${ROUTER_SCOPE:-last_user}"

LOCAL_NO_PROXY="127.0.0.1,localhost,::1"
if [[ -n "${NO_PROXY:-}" ]]; then
  export NO_PROXY="${LOCAL_NO_PROXY},${NO_PROXY}"
else
  export NO_PROXY="${LOCAL_NO_PROXY}"
fi
if [[ -n "${no_proxy:-}" ]]; then
  export no_proxy="${LOCAL_NO_PROXY},${no_proxy}"
else
  export no_proxy="${LOCAL_NO_PROXY}"
fi
export PYTHONPATH="${ROOT_DIR}/python:${ROOT_DIR}/python/inference:${ROOT_DIR}/agent:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

echo "MODEL_PATH=${MODEL_PATH}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "TOKENIZER_PATH=${TOKENIZER_PATH}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "DEVICES=${DEVICES_CSV}"
echo "MODES=${MODES_CSV}"
echo "SPLIT_NAME=${SPLIT_NAME}"
echo "TOOL_DOCUMENT_EVAL_MODE=${TOOL_DOCUMENT_EVAL_MODE}"
echo "DATASET_TOOL_DOCUMENT_MODE=${DATASET_TOOL_DOCUMENT_MODE}"
echo "MAX_DOC_LENGTH=${MAX_DOC_LENGTH}"
echo "MAX_DOC_NUM=${MAX_DOC_NUM}"
echo "MAX_TOOL_DEFINITION_TOKENS=${MAX_TOOL_DEFINITION_TOKENS}"
echo "MIN_NUM_TOOLS=${MIN_NUM_TOOLS}"
echo "MAX_EXAMPLES=${MAX_EXAMPLES}"

echo "Preparing Ascend environment..."
set +u
source /usr/local/Ascend/cann-8.5.0/set_env.sh || {
  echo "Failed to source CANN set_env.sh" >&2
  exit 1
}
source /usr/local/Ascend/nnal/atb/set_env.sh || {
  echo "Failed to source ATB set_env.sh" >&2
  exit 1
}
set -u

IFS=',' read -r -a DEVICES <<< "${DEVICES_CSV}"
IFS=',' read -r -a MODES <<< "${MODES_CSV}"
if [[ "${#DEVICES[@]}" -lt "${#MODES[@]}" ]]; then
  echo "Need one device per mode: devices=${#DEVICES[@]}, modes=${#MODES[@]}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}/parts" "${OUT_DIR}/logs"

declare -a CHILD_PIDS
cleanup() {
  for pid in "${CHILD_PIDS[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

wait_for_server() {
  local base_url="$1"
  local server_pid="$2"
  local server_log="$3"
  local deadline=$((SECONDS + 900))
  until curl --noproxy '*' -fsS "${base_url}/health" >/dev/null 2>&1; do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      local status=0
      wait "${server_pid}" || status=$?
      echo "SGLang server exited before healthy at ${base_url} (exit=${status})." >&2
      tail -80 "${server_log}" >&2 || true
      return 1
    fi
    if [[ ${SECONDS} -ge ${deadline} ]]; then
      echo "Timed out waiting for ${base_url}/health" >&2
      tail -80 "${server_log}" >&2 || true
      return 1
    fi
    sleep 2
  done
}

run_mode() {
  local device="$1"
  local port="$2"
  local mode="$3"
  local base_url="http://${HOST}:${port}"
  local server_log="${OUT_DIR}/logs/server_${mode}_device${device}.log"
  local eval_log="${OUT_DIR}/logs/eval_${mode}_device${device}.log"
  local output_file="${OUT_DIR}/parts/${mode}.jsonl"

  echo "[launch] mode=${mode} device=${device} port=${port}"
  (
    server_pid=""
    inner_cleanup() {
      if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
        kill "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
      fi
    }
    trap inner_cleanup EXIT

    cd "${SGLANG_DIR}"
    SGLANG_DEBUG_MEMORY_POOL="${SGLANG_DEBUG_MEMORY_POOL:-1}" \
    SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE="${SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE:-0}" \
    ASCEND_LAUNCH_BLOCKING="${ASCEND_LAUNCH_BLOCKING:-1}" \
    ASCEND_RT_VISIBLE_DEVICES="${device}" \
    "${PYTHON_BIN}" -m sglang.launch_server \
      --model-path "${MODEL_PATH}" \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --model-impl sglang \
      --device npu \
      --attention-backend ascend \
      --tool-call-parser qwen25 \
      --enable-c2kv \
      --dtype bfloat16 \
      --mem-fraction-static "${MEM_FRACTION_STATIC}" \
      --host "${HOST}" \
      --port "${port}" \
      >"${server_log}" 2>&1 &
    server_pid=$!

    wait_for_server "${base_url}" "${server_pid}" "${server_log}"

    eval_args=(
      "${ROOT_DIR}/agent/eval_agent_tool_definition_sglang_api.py"
      --base-url "${base_url}"
      --model "${SERVED_MODEL_NAME}"
      --tokenizer "${TOKENIZER_PATH}"
      --dataset-path "${DATASET_PATH}"
      --output-file "${output_file}"
      --split "${SPLIT}"
      --mode "${mode}"
      --ratio "${RATIO}"
      --max-examples "${MAX_EXAMPLES}"
      --max-new-tokens "${MAX_NEW_TOKENS}"
      --split-manifest-name "${SPLIT_NAME}"
      --max-doc-length "${MAX_DOC_LENGTH}"
      --max-doc-num "${MAX_DOC_NUM}"
      --max-tool-definition-tokens "${MAX_TOOL_DEFINITION_TOKENS}"
      --max-samples-per-session "${MAX_SAMPLES_PER_SESSION}"
      --min-target-tokens "${MIN_TARGET_TOKENS}"
      --min-num-tools "${MIN_NUM_TOOLS}"
      --dataset-tool-document-mode "${DATASET_TOOL_DOCUMENT_MODE}"
      --tool-document-eval-mode "${TOOL_DOCUMENT_EVAL_MODE}"
      --hard-negative-num "${HARD_NEGATIVE_NUM}"
      --hard-negative-router-scope "${HARD_NEGATIVE_ROUTER_SCOPE}"
      --shuffle-tool-documents "${SHUFFLE_TOOL_DOCUMENTS}"
      --balance-subsets "${BALANCE_SUBSETS}"
      --hybrid-top-k "${HYBRID_TOP_K}"
      --router-scope "${ROUTER_SCOPE}"
    )
    if [[ -n "${SPLIT_MANIFEST_FILE}" ]]; then
      eval_args+=(--split-manifest-file "${SPLIT_MANIFEST_FILE}")
    fi

    cd "${ROOT_DIR}"
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 "${PYTHON_BIN}" "${eval_args[@]}" >"${eval_log}" 2>&1
  )
}

for ((i = 0; i < ${#MODES[@]}; i++)); do
  mode="${MODES[i]}"
  device="${DEVICES[i]}"
  port=$((BASE_PORT + i))
  run_mode "${device}" "${port}" "${mode}" &
  CHILD_PIDS+=("$!")
  sleep "${START_STAGGER_SECONDS}"
done

failed=0
for pid in "${CHILD_PIDS[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
trap - EXIT

if [[ "${failed}" != "0" ]]; then
  echo "At least one mode failed. Logs are under ${OUT_DIR}/logs" >&2
  exit 1
fi

"${PYTHON_BIN}" "${ROOT_DIR}/agent/merge_agent_tool_definition_sglang_api.py" \
  --input-glob "${OUT_DIR}/parts/*.jsonl" \
  --output-dir "${OUT_DIR}"

echo "Report: ${OUT_DIR}/report.md"
echo "Summary: ${OUT_DIR}/summary.json"
