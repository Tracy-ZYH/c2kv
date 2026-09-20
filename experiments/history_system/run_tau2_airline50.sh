#!/usr/bin/env bash
set -Eeuo pipefail

# One-card tau2 airline50 runner for a fair C1000 C1-vs-C2KV comparison.
# This script starts and owns exactly one PR #5 SGLang process group.

METHOD="${METHOD:-proposed}"
DETECTOR="${DETECTOR:-t02_risk}"
DEVICE="${DEVICE:?set DEVICE to one physical NPU id}"
ENGINE_PORT="${ENGINE_PORT:?set ENGINE_PORT}"
CONTROLLER_PORT="${CONTROLLER_PORT:?set CONTROLLER_PORT}"
OUTPUT="${OUTPUT:?set OUTPUT to a new directory}"

case "${METHOD}" in
  proposed)
    case "${DETECTOR}" in
      t02_risk|legacy_prefill) ;;
      *) echo "DETECTOR must be t02_risk or legacy_prefill" >&2; exit 2 ;;
    esac
    SERVED_MODEL="c1_${DETECTOR}"
    DETECTOR_ARGS=(--detector "${DETECTOR}")
    EMBEDDING_DEVICE=npu:0
    ;;
  c2kv_only) SERVED_MODEL=c2kv_only; DETECTOR_ARGS=(); EMBEDDING_DEVICE=cpu ;;
  *) echo "METHOD must be proposed or c2kv_only" >&2; exit 2 ;;
esac

C2KV_ROOT="${C2KV_ROOT:-/home/zhuyuhan/project/c2kv}"
SGLANG_ROOT="${SGLANG_ROOT:-/home/zhuyuhan/project/kvoffload-sglang-c2kv-pr5-runtime}"
BFCL_ROOT="${BFCL_ROOT:-/home/zhuyuhan/project/bfcl-c2kv}"
TAU2_ROOT="${TAU2_ROOT:-/home/zhuyuhan/benchmarks/tau2}"
CHECKPOINT="${CHECKPOINT:-/home/zhuyuhan/project/model/arm-C/checkpoint-1000}"
EMBEDDING="${EMBEDDING:-/home/liuyancheng/c2kv-evidence-sets-20260916/models/Qwen3-Embedding-0.6B}"
SGLANG_PYTHON="${SGLANG_PYTHON:-/home/liuyancheng/envs/sgl/bin/python}"
C2KV_PYTHON="${C2KV_PYTHON:-/home/liuyancheng/envs/c2kv/bin/python}"
TAU2_PYTHON="${TAU2_PYTHON:-/home/liuyancheng/envs/bench312/bin/python}"
ENGINE_LOG="${OUTPUT}.engine.log"
MODEL_INFO="${OUTPUT}.model_info.json"

[[ ! -e "${OUTPUT}" ]] || { echo "refusing to overwrite ${OUTPUT}" >&2; exit 1; }
[[ ! -e "${ENGINE_LOG}" ]] || { echo "refusing to overwrite ${ENGINE_LOG}" >&2; exit 1; }
mkdir -p "$(dirname "${OUTPUT}")"

set +u
source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

export ASCEND_RT_VISIBLE_DEVICES="${DEVICE}"
export HCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost

ENGINE_PID=""
cleanup() {
  local pid="${ENGINE_PID:-}" pgid=""
  [[ -n "${pid}" ]] || return 0
  if ! kill -0 "${pid}" 2>/dev/null; then wait "${pid}" 2>/dev/null || true; return 0; fi
  pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d '[:space:]')"
  if [[ "${pgid}" == "${pid}" ]]; then kill -TERM -- "-${pgid}" 2>/dev/null || true
  else kill -TERM "${pid}" 2>/dev/null || true; fi
  for _ in $(seq 1 60); do
    kill -0 "${pid}" 2>/dev/null || { wait "${pid}" 2>/dev/null || true; return 0; }
    sleep 0.5
  done
  if [[ "${pgid}" == "${pid}" ]]; then kill -KILL -- "-${pgid}" 2>/dev/null || true
  else kill -KILL "${pid}" 2>/dev/null || true; fi
  wait "${pid}" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

(
  cd "${SGLANG_ROOT}"
  exec setsid env \
    PYTHONPATH="${SGLANG_ROOT}/python" \
    "${SGLANG_PYTHON}" -m sglang.launch_server \
      --model-path "${CHECKPOINT}" \
      --served-model-name "${SERVED_MODEL}" \
      --model-impl sglang --device npu --attention-backend ascend --dtype bfloat16 \
      --enable-c2kv --c2kv-gist-type dynamic-interleave --c2kv-gist-param qkv \
      --c2kv-query-proj base --c2kv-pool-fraction 0.05 \
      --c2kv-shadow-feature-layer -2 --enable-return-hidden-states \
      --mem-fraction-static 0.55 --max-total-tokens 65536 --context-length 131072 \
      --max-running-requests 1 --page-size 128 --chunked-prefill-size 256 \
      --disable-radix-cache --disable-cuda-graph \
      --host 127.0.0.1 --port "${ENGINE_PORT}"
) >"${ENGINE_LOG}" 2>&1 &
ENGINE_PID=$!

ready=0
for _ in $(seq 1 900); do
  if ! kill -0 "${ENGINE_PID}" 2>/dev/null; then
    echo "SGLang exited before readiness; see ${ENGINE_LOG}" >&2
    exit 1
  fi
  if curl --noproxy '*' -fsS "http://127.0.0.1:${ENGINE_PORT}/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
done
[[ "${ready}" == 1 ]] || { echo "SGLang readiness timeout; see ${ENGINE_LOG}" >&2; exit 1; }
curl --noproxy '*' -fsS "http://127.0.0.1:${ENGINE_PORT}/model_info" >"${MODEL_INFO}"

TASK_ARGS=()
for task_id in $(seq 0 49); do TASK_ARGS+=(--tau2-task-id "${task_id}"); done

cd "${C2KV_ROOT}"
"${C2KV_PYTHON}" experiments/history_system/run_c1.py \
  --benchmark tau2 --method "${METHOD}" "${DETECTOR_ARGS[@]}" \
  --checkpoint "${CHECKPOINT}" \
  --embedding-model "${EMBEDDING}" --embedding-device "${EMBEDDING_DEVICE}" \
  --sglang-backend-url "http://127.0.0.1:${ENGINE_PORT}" \
  --user-base-url "http://127.0.0.1:${ENGINE_PORT}" \
  --sglang-root "${SGLANG_ROOT}" --portable-root "${BFCL_ROOT}" \
  --tau2-dir "${TAU2_ROOT}" --tau2-python "${TAU2_PYTHON}" \
  --task-set airline --tau2-max-steps 100 --task-timeout 1200 \
  --port "${CONTROLLER_PORT}" --out "${OUTPUT}" \
  "${TASK_ARGS[@]}"

echo "completed ${METHOD}: ${OUTPUT}"
