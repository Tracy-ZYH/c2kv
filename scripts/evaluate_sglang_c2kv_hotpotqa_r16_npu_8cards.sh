#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/zhuyuhan/project/c2kv}"
SGLANG_DIR="${SGLANG_DIR:-/home/zhuyuhan/project/kvoffload-sglang}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhuyuhan/miniconda3/envs/sglang/bin/python}"

MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/checkpoints/qwen3-4b-mixed-mdoc-c2kv-r16-npu-12k/checkpoint-1125}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-4b}"
DATASET_PATH="${DATASET_PATH:-${ROOT_DIR}/datasets/longbench_hotpotqa_test}"

DEVICES_CSV="${DEVICES:-0,1,2,3,4,5,6,7}"
MODES_CSV="${MODES:-full,c2kv,combined-c2kv}"
HOST="${HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-31000}"
COMPRESSION_RATIO="${COMPRESSION_RATIO:-16}"
MAX_EXAMPLES="${MAX_EXAMPLES:-200}"
WORKERS="${WORKERS:-1}"
MAX_TOKENS="${MAX_TOKENS:-16}"
CUT_LENGTH="${CUT_LENGTH:-}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.7}"
START_STAGGER_SECONDS="${START_STAGGER_SECONDS:-2}"
RUN_NAME="${RUN_NAME:-hotpotqa_r16_8cards}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/results/sglang_c2kv/qwen3-4b-mixed-mdoc-c2kv-r16-npu-12k_checkpoint-1125/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/outputs/${RUN_NAME}}"

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

IFS=',' read -r -a DEVICES <<< "${DEVICES_CSV}"
IFS=',' read -r -a MODES <<< "${MODES_CSV}"
if [[ "${#DEVICES[@]}" -eq 0 ]]; then
  echo "DEVICES is empty." >&2
  exit 1
fi
if [[ "${#MODES[@]}" -eq 0 ]]; then
  echo "MODES is empty." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}/shards" "${OUT_DIR}/merged" "${LOG_DIR}"

if [[ "${#DEVICES[@]}" -lt "${#MODES[@]}" ]]; then
  echo "Need at least one device per mode: devices=${#DEVICES[@]}, modes=${#MODES[@]}." >&2
  exit 1
fi

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
      local server_status=0
      wait "${server_pid}" || server_status=$?
      echo "SGLang server exited before healthy at ${base_url} (exit=${server_status})." >&2
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

run_job() {
  local device="$1"
  local port="$2"
  local mode="$3"
  local shard_index="$4"
  local num_shards="$5"
  local base_url="http://${HOST}:${port}"
  local server_log="${LOG_DIR}/server_device${device}_${mode}_shard${shard_index}.log"
  local eval_log="${LOG_DIR}/eval_device${device}_${mode}_shard${shard_index}.log"
  local output_file="${OUT_DIR}/shards/${mode}_shard${shard_index}_of${num_shards}_device${device}.jsonl"

  echo "[device ${device}] mode=${mode} shard=${shard_index}/${num_shards} port=${port}"
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
      "${ROOT_DIR}/python/inference/eval_sglang_c2kv_longbench_hotpotqa.py"
      --base-url "${base_url}"
      --model "${SERVED_MODEL_NAME}"
      --dataset-path "${DATASET_PATH}"
      --output-file "${output_file}"
      --compression-ratio "${COMPRESSION_RATIO}"
      --mode "${mode}"
      --max-examples "${MAX_EXAMPLES}"
      --num-shards "${num_shards}"
      --shard-index "${shard_index}"
      --workers "${WORKERS}"
      --max-tokens "${MAX_TOKENS}"
    )
    if [[ -n "${CUT_LENGTH}" ]]; then
      eval_args+=(--cut-length "${CUT_LENGTH}")
    fi

    cd "${ROOT_DIR}"
    "${PYTHON_BIN}" "${eval_args[@]}" >"${eval_log}" 2>&1
  )
}

for ((i = 0; i < ${#MODES[@]}; i++)); do
  device="${DEVICES[i]}"
  mode="${MODES[i]}"
  shard_index=0
  num_shards=1
  port=$((BASE_PORT + i))

  run_job "${device}" "${port}" "${mode}" "${shard_index}" "${num_shards}" &
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
  echo "At least one shard failed. Logs are under ${LOG_DIR}" >&2
  exit 1
fi

summary_json="${OUT_DIR}/summary.json"
report_md="${OUT_DIR}/report.md"
"${PYTHON_BIN}" "${ROOT_DIR}/python/inference/merge_sglang_c2kv_results.py" \
  --input-glob "${OUT_DIR}/shards/*.jsonl" \
  --merged-dir "${OUT_DIR}/merged" \
  --summary-json "${summary_json}" \
  --report-md "${report_md}"

echo "Merged shard JSONL files: ${OUT_DIR}/merged"
echo "Summary JSON: ${summary_json}"
echo "Report: ${report_md}"
