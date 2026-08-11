#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/zhuyuhan/project/c2kv}"
SGLANG_DIR="${SGLANG_DIR:-/home/zhuyuhan/project/kvoffload-sglang}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhuyuhan/miniconda3/envs/sglang/bin/python}"

MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/checkpoints/qwen3-4b-agent-history-c2kv-toolcall-npu-v2/checkpoint-1000}"
TOKENIZER_PATH="${HISTORY_TOKENIZER_PATH:-${ROOT_DIR}/models/Qwen3-4B-Instruct-2507}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-history-c2kv}"

AGENT_DATASET_PATH="${AGENT_DATASET_PATH:-${ROOT_DIR}/datasets/agent-llm-traces}"
TOOLATHLON_DATASET_PATH="${TOOLATHLON_DATASET_PATH:-${ROOT_DIR}/datasets/toolathlon}"
DATASET_NAMES_CSV="${DATASET_NAMES:-agent_llm_traces,toolathlon}"
DATASET_PATHS_CSV="${DATASET_PATHS:-${AGENT_DATASET_PATH},${TOOLATHLON_DATASET_PATH}}"
MODES_CSV="${MODES:-full,c2kv,hybrid}"
DEVICES_CSV="${DEVICES:-0,1,2,3,4,5}"

HOST="${HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-34000}"
RATIO="${RATIO:-4}"
HYBRID_TOP_K="${HYBRID_TOP_K:-3}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
MAX_SOURCE_EXAMPLES="${MAX_SOURCE_EXAMPLES:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.55}"
RUN_NAME="${RUN_NAME:-agent_history_sglang_api_6cards}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/outputs/${RUN_NAME}}"
START_STAGGER_SECONDS="${START_STAGGER_SECONDS:-8}"

SPLIT="${SPLIT:-eval}"
SPLIT_NAME="${SPLIT_NAME:-subset_disjoint}"
SPLIT_MANIFEST_FILE="${SPLIT_MANIFEST_FILE:-}"
SPLIT_SEED="${SPLIT_SEED:-42}"
EVAL_RATIO="${EVAL_RATIO:-0.1}"
MAX_SAMPLES_PER_SESSION="${MAX_SAMPLES_PER_SESSION:-4}"
SELECTION_FILTER="${SELECTION_FILTER:-c2kv}"
REQUIRE_TOOL_CALL="${REQUIRE_TOOL_CALL:-False}"
INCLUDE_TOOLS="${INCLUDE_TOOLS:-True}"

MAX_DOC_LENGTH="${HISTORY_MAX_DOC_LENGTH:-768}"
MAX_DOC_NUM="${HISTORY_MAX_DOC_NUM:-16}"
MIN_DOC_NUM="${HISTORY_MIN_DOC_NUM:-1}"
MAX_HISTORY_TOKENS="${MAX_HISTORY_TOKENS:-12288}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-1536}"
MAX_BASELINE_INPUT_TOKENS="${MAX_BASELINE_INPUT_TOKENS:-16000}"
HISTORY_SELECTION="${HISTORY_SELECTION:-tail}"
SPLIT_OVERSIZED_HISTORY_DOCS="${SPLIT_OVERSIZED_HISTORY_DOCS:-True}"
PREFIX_HISTORY_DOC_NUM="${PREFIX_HISTORY_DOC_NUM:-}"
PREFIX_HISTORY_EXACT="${PREFIX_HISTORY_EXACT:-False}"
MAX_INPUT_CHARS="${MAX_INPUT_CHARS:-}"
MAX_ANSWER_CHARS="${MAX_ANSWER_CHARS:-}"

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
export PYTHONPATH="${ROOT_DIR}/python:${ROOT_DIR}/python/inference:${ROOT_DIR}/agent:${ROOT_DIR}/agent/api:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

resolve_model_path() {
  local model_path="$1"
  if [[ -f "${model_path}/config.json" ]] && grep -q '"model_type"' "${model_path}/config.json"; then
    printf '%s\n' "${model_path}"
    return
  fi
  if [[ ! -d "${model_path}" ]]; then
    printf '%s\n' "${model_path}"
    return
  fi

  local latest=""
  while IFS= read -r candidate; do
    if [[ -f "${candidate}/config.json" ]] && grep -q '"model_type"' "${candidate}/config.json"; then
      latest="${candidate}"
    fi
  done < <(
    find "${model_path}" -maxdepth 1 -type d -name 'checkpoint-*' -print |
      sort -t- -k2,2n
  )

  if [[ -n "${latest}" ]]; then
    printf '%s\n' "${latest}"
    return
  fi

  printf '%s\n' "${model_path}"
}

RESOLVED_MODEL_PATH="$(resolve_model_path "${MODEL_PATH}")"
if [[ "${RESOLVED_MODEL_PATH}" != "${MODEL_PATH}" ]]; then
  echo "Resolved MODEL_PATH=${MODEL_PATH} -> ${RESOLVED_MODEL_PATH}"
fi
MODEL_PATH="${RESOLVED_MODEL_PATH}"

echo "MODEL_PATH=${MODEL_PATH}"
echo "TOKENIZER_PATH=${TOKENIZER_PATH}"
echo "DATASET_NAMES=${DATASET_NAMES_CSV}"
echo "DATASET_PATHS=${DATASET_PATHS_CSV}"
echo "MODES=${MODES_CSV}"
echo "DEVICES=${DEVICES_CSV}"
echo "MAX_EXAMPLES=${MAX_EXAMPLES}"
echo "MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC}"
echo "OUT_DIR=${OUT_DIR}"

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

IFS=',' read -r -a DATASET_NAMES <<< "${DATASET_NAMES_CSV}"
IFS=',' read -r -a DATASET_PATHS <<< "${DATASET_PATHS_CSV}"
IFS=',' read -r -a MODES <<< "${MODES_CSV}"
IFS=',' read -r -a DEVICES <<< "${DEVICES_CSV}"

EXPECTED_JOBS=$((${#DATASET_NAMES[@]} * ${#MODES[@]}))
if [[ "${#DATASET_NAMES[@]}" -ne "${#DATASET_PATHS[@]}" ]]; then
  echo "DATASET_NAMES and DATASET_PATHS must have the same length." >&2
  exit 1
fi
if [[ "${#DEVICES[@]}" -lt "${EXPECTED_JOBS}" ]]; then
  echo "Need one device per dataset/mode job: devices=${#DEVICES[@]}, jobs=${EXPECTED_JOBS}" >&2
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

run_job() {
  local dataset_name="$1"
  local dataset_path="$2"
  local mode="$3"
  local device="$4"
  local port="$5"
  local base_url="http://${HOST}:${port}"
  local job_name="${dataset_name}_${mode}"
  local server_log="${OUT_DIR}/logs/server_${job_name}_device${device}.log"
  local eval_log="${OUT_DIR}/logs/eval_${job_name}_device${device}.log"
  local output_file="${OUT_DIR}/parts/${job_name}.jsonl"

  echo "[launch] dataset=${dataset_name} mode=${mode} device=${device} port=${port}"
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
    SGLANG_EMPTY_CACHE_INTERVAL="${SGLANG_EMPTY_CACHE_INTERVAL:-1}" \
    ASCEND_LAUNCH_BLOCKING="${ASCEND_LAUNCH_BLOCKING:-1}" \
    TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}" \
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
      "${ROOT_DIR}/agent/api/eval_agent_history_sglang_api.py"
      --base-url "${base_url}"
      --model "${SERVED_MODEL_NAME}"
      --tokenizer "${TOKENIZER_PATH}"
      --dataset-path "${dataset_path}"
      --output-file "${output_file}"
      --split "${SPLIT}"
      --mode "${mode}"
      --ratio "${RATIO}"
      --hybrid-top-k "${HYBRID_TOP_K}"
      --max-examples "${MAX_EXAMPLES}"
      --max-new-tokens "${MAX_NEW_TOKENS}"
      --selection-filter "${SELECTION_FILTER}"
      --eval-ratio "${EVAL_RATIO}"
      --split-seed "${SPLIT_SEED}"
      --split-manifest-name "${SPLIT_NAME}"
      --max-samples-per-session "${MAX_SAMPLES_PER_SESSION}"
      --require-tool-call "${REQUIRE_TOOL_CALL}"
      --include-tools "${INCLUDE_TOOLS}"
      --max-doc-length "${MAX_DOC_LENGTH}"
      --min-doc-num "${MIN_DOC_NUM}"
      --max-doc-num "${MAX_DOC_NUM}"
      --max-history-tokens "${MAX_HISTORY_TOKENS}"
      --max-prompt-tokens "${MAX_PROMPT_TOKENS}"
      --max-baseline-input-tokens "${MAX_BASELINE_INPUT_TOKENS}"
      --history-selection "${HISTORY_SELECTION}"
      --split-oversized-history-docs "${SPLIT_OVERSIZED_HISTORY_DOCS}"
      --prefix-history-exact "${PREFIX_HISTORY_EXACT}"
    )
    if [[ -n "${MAX_SOURCE_EXAMPLES}" ]]; then
      eval_args+=(--max-source-examples "${MAX_SOURCE_EXAMPLES}")
    fi
    if [[ -n "${SPLIT_MANIFEST_FILE}" ]]; then
      eval_args+=(--split-manifest-file "${SPLIT_MANIFEST_FILE}")
    fi
    if [[ -n "${PREFIX_HISTORY_DOC_NUM}" ]]; then
      eval_args+=(--prefix-history-doc-num "${PREFIX_HISTORY_DOC_NUM}")
    fi
    if [[ -n "${MAX_INPUT_CHARS}" ]]; then
      eval_args+=(--max-input-chars "${MAX_INPUT_CHARS}")
    fi
    if [[ -n "${MAX_ANSWER_CHARS}" ]]; then
      eval_args+=(--max-answer-chars "${MAX_ANSWER_CHARS}")
    fi

    cd "${ROOT_DIR}"
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 "${PYTHON_BIN}" "${eval_args[@]}" >"${eval_log}" 2>&1
  )
}

job_index=0
for ((d = 0; d < ${#DATASET_NAMES[@]}; d++)); do
  dataset_name="${DATASET_NAMES[d]// /}"
  dataset_path="${DATASET_PATHS[d]}"
  for mode in "${MODES[@]}"; do
    mode="${mode// /}"
    device="${DEVICES[job_index]}"
    port=$((BASE_PORT + job_index))
    run_job "${dataset_name}" "${dataset_path}" "${mode}" "${device}" "${port}" &
    CHILD_PIDS+=("$!")
    job_index=$((job_index + 1))
    sleep "${START_STAGGER_SECONDS}"
  done
done

failed=0
for pid in "${CHILD_PIDS[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
trap - EXIT

if [[ "${failed}" != "0" ]]; then
  echo "At least one job failed. Logs are under ${OUT_DIR}/logs" >&2
  exit 1
fi

"${PYTHON_BIN}" "${ROOT_DIR}/agent/api/merge_agent_history_sglang_api.py" \
  --input-glob "${OUT_DIR}/parts/*.jsonl" \
  --output-dir "${OUT_DIR}" \
  --common-valid-subset

echo "Report: ${OUT_DIR}/report.md"
echo "Summary: ${OUT_DIR}/summary.json"
