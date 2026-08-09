#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/zhuyuhan/project/c2kv}"
SGLANG_DIR="${SGLANG_DIR:-/home/zhuyuhan/project/kvoffload-sglang}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhuyuhan/miniconda3/envs/sglang/bin/python}"

MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/checkpoints/qwen3-4b-mixed-mdoc-c2kv-r16-npu-12k/checkpoint-1125}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-4b}"
DATASET_PATH="${DATASET_PATH:-${ROOT_DIR}/datasets/longbench_hotpotqa_test}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
BASE_URL="${BASE_URL:-http://${HOST}:${PORT}}"
ASCEND_DEVICE="${ASCEND_DEVICE:-7}"
COMPRESSION_RATIO="${COMPRESSION_RATIO:-16}"
EVAL_MODE="${EVAL_MODE:-c2kv}"
OUTPUT_FILE="${OUTPUT_FILE:-${ROOT_DIR}/results/sglang_c2kv/qwen3-4b-mixed-mdoc-c2kv-r16-npu-12k_checkpoint-1125/hotpotqa_r16_${EVAL_MODE}.jsonl}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
WORKERS="${WORKERS:-1}"
MAX_TOKENS="${MAX_TOKENS:-16}"
CUT_LENGTH="${CUT_LENGTH:-}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.7}"
START_SERVER="${START_SERVER:-1}"
SERVER_LOG="${SERVER_LOG:-${ROOT_DIR}/outputs/sglang_c2kv_hotpotqa_r16_npu_server.log}"

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

mkdir -p "$(dirname "${OUTPUT_FILE}")"
mkdir -p "$(dirname "${SERVER_LOG}")"

server_pid=""

cleanup() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}

wait_for_server() {
  local deadline=$((SECONDS + 900))
  until curl --noproxy '*' -fsS "${BASE_URL}/health" >/dev/null 2>&1; do
    if [[ -n "${server_pid}" ]] && ! kill -0 "${server_pid}" 2>/dev/null; then
      local server_status=0
      wait "${server_pid}" || server_status=$?
      echo "SGLang server exited before becoming healthy (exit=${server_status})." >&2
      echo "Last server log lines:" >&2
      tail -80 "${SERVER_LOG}" >&2 || true
      return 1
    fi
    if [[ ${SECONDS} -ge ${deadline} ]]; then
      echo "Timed out waiting for ${BASE_URL}/health" >&2
      echo "Last server log lines:" >&2
      tail -80 "${SERVER_LOG}" >&2 || true
      return 1
    fi
    sleep 2
  done
}

if [[ "${START_SERVER}" == "1" ]]; then
  trap cleanup EXIT

  if curl --noproxy '*' -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
    echo "Server already healthy at ${BASE_URL}; reusing it."
  else
    echo "Starting SGLang server on ${BASE_URL}"
    (
      cd "${SGLANG_DIR}"
      SGLANG_DEBUG_MEMORY_POOL="${SGLANG_DEBUG_MEMORY_POOL:-1}" \
      SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE="${SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE:-0}" \
      ASCEND_LAUNCH_BLOCKING="${ASCEND_LAUNCH_BLOCKING:-1}" \
      ASCEND_RT_VISIBLE_DEVICES="${ASCEND_DEVICE}" \
      exec "${PYTHON_BIN}" -m sglang.launch_server \
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
        --port "${PORT}"
    ) >"${SERVER_LOG}" 2>&1 &
    server_pid=$!
    wait_for_server
  fi
else
  wait_for_server
fi

eval_args=(
  "${ROOT_DIR}/python/inference/eval_sglang_c2kv_longbench_hotpotqa.py"
  --base-url "${BASE_URL}"
  --model "${SERVED_MODEL_NAME}"
  --dataset-path "${DATASET_PATH}"
  --output-file "${OUTPUT_FILE}"
  --compression-ratio "${COMPRESSION_RATIO}"
  --mode "${EVAL_MODE}"
  --workers "${WORKERS}"
  --max-tokens "${MAX_TOKENS}"
)

if [[ -n "${MAX_EXAMPLES}" ]]; then
  eval_args+=(--max-examples "${MAX_EXAMPLES}")
fi

if [[ -n "${CUT_LENGTH}" ]]; then
  eval_args+=(--cut-length "${CUT_LENGTH}")
fi

cd "${ROOT_DIR}"
echo "Running HotpotQA C2KV evaluation..."
echo "Dataset: ${DATASET_PATH}"
echo "Mode: ${EVAL_MODE}"
echo "Output: ${OUTPUT_FILE}"
"${PYTHON_BIN}" "${eval_args[@]}"

echo "Wrote results to ${OUTPUT_FILE}"
echo "Wrote summary to ${OUTPUT_FILE/.jsonl/.summary.json}"
