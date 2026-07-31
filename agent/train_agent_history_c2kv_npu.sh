#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)/python:$(pwd)/agent:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-7200}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1}"
export C2KV_GIST_DOC_MICROBATCH="${C2KV_GIST_DOC_MICROBATCH:-1}"
export C2KV_GIST_TRAIN_RATIOS="${C2KV_GIST_TRAIN_RATIOS:-4,8,16}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-max_split_size_mb:128}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-./models/Qwen3-4B-Instruct-2507}"
DATASET_PATH="${DATASET_PATH:-./datasets/agent-llm-traces}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/qwen3-4b-agent-history-c2kv-npu}"
NPU_ATTN_IMPL="${NPU_ATTN_IMPL:-npu_fusion_attention}"

SPLIT_MANIFEST_FILE="${SPLIT_MANIFEST_FILE:-}"
SPLIT_NAME="${SPLIT_NAME:-subset_disjoint}"
SPLIT_SEED="${SPLIT_SEED:-42}"
EVAL_RATIO="${EVAL_RATIO:-0.1}"

MAX_DOC_LENGTH="${HISTORY_MAX_DOC_LENGTH:-512}"
MAX_DOC_NUM="${HISTORY_MAX_DOC_NUM:-12}"
MIN_DOC_NUM="${HISTORY_MIN_DOC_NUM:-1}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_SYSTEM_LENGTH="${MAX_SYSTEM_LENGTH:-4096}"
MAX_SAMPLES_PER_SESSION="${MAX_SAMPLES_PER_SESSION:-4}"
NUM_SAMPLES="${NUM_SAMPLES:-6000}"
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-128}"
HISTORY_SELECTION="${HISTORY_SELECTION:-tail}"
REQUIRE_TOOL_CALL="${REQUIRE_TOOL_CALL:-False}"
INCLUDE_TOOLS="${INCLUDE_TOOLS:-True}"
MAX_INPUT_CHARS="${MAX_INPUT_CHARS:-}"
MAX_ANSWER_CHARS="${MAX_ANSWER_CHARS:-}"

LEARNING_RATE="${LEARNING_RATE:-5e-7}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
GIST_GRADIENT_CHECKPOINTING="${GIST_GRADIENT_CHECKPOINTING:-True}"
EVAL_STEPS="${EVAL_STEPS:-25}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
DATASET_SHUFFLE_SEED="${DATASET_SHUFFLE_SEED:-2948}"
DO_EVAL="${DO_EVAL:-False}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-2}"
MASTER_PORT="${MASTER_PORT:-29567}"
TORCHRUN_LOG_DIR="${TORCHRUN_LOG_DIR:-./outputs/torchrun_agent_history_logs}"
TORCHRUN_REDIRECTS="${TORCHRUN_REDIRECTS:-3}"
TORCHRUN_TEE="${TORCHRUN_TEE:-0}"
DDP_TIMEOUT="${DDP_TIMEOUT:-7200}"
# Empty by default: two-card DDP is simpler and avoids ZeRO batch-size assertions.
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG-}"
if [[ -z "${C2KV_GIST_CHECKPOINT_USE_REENTRANT+x}" ]]; then
  if [[ -n "${DEEPSPEED_CONFIG}" && "${DEEPSPEED_CONFIG}" != "none" && "${DEEPSPEED_CONFIG}" != "None" ]]; then
    export C2KV_GIST_CHECKPOINT_USE_REENTRANT=True
  else
    export C2KV_GIST_CHECKPOINT_USE_REENTRANT=False
  fi
fi

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  IFS=',' read -ra _visible_npus <<< "${ASCEND_RT_VISIBLE_DEVICES}"
  NPROC_PER_NODE="${#_visible_npus[@]}"
fi

if ! find "${DATASET_PATH}" -name '*.parquet' -type f -print -quit 2>/dev/null | grep -q .; then
  echo "ERROR: no parquet files found under DATASET_PATH=${DATASET_PATH}" >&2
  echo "Expected files like: ${DATASET_PATH}/data/train-00000-of-00039.parquet" >&2
  exit 1
fi

if [[ -n "${SPLIT_MANIFEST_FILE}" && -f "${SPLIT_MANIFEST_FILE}" ]]; then
  python - "${SPLIT_MANIFEST_FILE}" "${SPLIT_NAME}" <<'PY'
import json
import sys

path, split_name = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as f:
    manifest = json.load(f)
if "train_session_ids" in manifest and "eval_session_ids" in manifest:
    sys.exit(0)
if split_name not in manifest:
    available = sorted(key for key in manifest if key != "metadata")
    raise SystemExit(
        f"ERROR: split {split_name!r} not found in {path}. "
        f"Available splits: {available}"
    )
PY
fi

SOURCE_ARGS=(
  --source_type agent_llm_traces
  --split_seed "${SPLIT_SEED}"
  --eval_ratio "${EVAL_RATIO}"
  --split_manifest_name "${SPLIT_NAME}"
  --max_samples_per_session "${MAX_SAMPLES_PER_SESSION}"
  --require_tool_call "${REQUIRE_TOOL_CALL}"
  --include_tools "${INCLUDE_TOOLS}"
)
if [[ -n "${SPLIT_MANIFEST_FILE}" ]]; then
  SOURCE_ARGS+=(--split_manifest_file "${SPLIT_MANIFEST_FILE}")
fi
if [[ -n "${MAX_INPUT_CHARS}" ]]; then
  SOURCE_ARGS+=(--max_input_chars "${MAX_INPUT_CHARS}")
fi
if [[ -n "${MAX_ANSWER_CHARS}" ]]; then
  SOURCE_ARGS+=(--max_answer_chars "${MAX_ANSWER_CHARS}")
fi
SAMPLE_ARGS=(--eval_num_samples "${EVAL_NUM_SAMPLES}")
if [[ -n "${NUM_SAMPLES}" ]]; then
  SAMPLE_ARGS+=(--num_samples "${NUM_SAMPLES}")
fi
MAX_STEPS_ARGS=()
if [[ -n "${MAX_STEPS}" ]]; then
  MAX_STEPS_ARGS=(--max_steps "${MAX_STEPS}")
fi
RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  RESUME_ARGS=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

EVAL_ARGS=(--eval_strategy steps --eval_steps "${EVAL_STEPS}")
if [[ "${DO_EVAL}" != "True" && "${DO_EVAL}" != "true" && "${DO_EVAL}" != "1" ]]; then
  EVAL_ARGS=(--eval_strategy no)
fi

DATALOADER_ARGS=(--dataloader_num_workers "${DATALOADER_NUM_WORKERS}")
if (( DATALOADER_NUM_WORKERS > 0 )); then
  DATALOADER_ARGS+=(--dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR}")
fi

DEEPSPEED_ARGS=()
if [[ -n "${DEEPSPEED_CONFIG}" && "${DEEPSPEED_CONFIG}" != "none" && "${DEEPSPEED_CONFIG}" != "None" ]]; then
  DEEPSPEED_ARGS=(--deepspeed "${DEEPSPEED_CONFIG}")
fi

echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "SPLIT_MANIFEST_FILE=${SPLIT_MANIFEST_FILE}"
echo "SPLIT_NAME=${SPLIT_NAME}"
echo "MAX_DOC_LENGTH=${MAX_DOC_LENGTH}"
echo "MAX_DOC_NUM=${MAX_DOC_NUM}"
echo "MIN_DOC_NUM=${MIN_DOC_NUM}"
echo "MAX_LENGTH=${MAX_LENGTH}"
echo "MAX_SYSTEM_LENGTH=${MAX_SYSTEM_LENGTH}"
echo "MAX_SAMPLES_PER_SESSION=${MAX_SAMPLES_PER_SESSION}"
echo "NUM_SAMPLES=${NUM_SAMPLES}"
echo "EVAL_NUM_SAMPLES=${EVAL_NUM_SAMPLES}"
echo "HISTORY_SELECTION=${HISTORY_SELECTION}"
echo "REQUIRE_TOOL_CALL=${REQUIRE_TOOL_CALL}"
echo "INCLUDE_TOOLS=${INCLUDE_TOOLS}"
echo "LEARNING_RATE=${LEARNING_RATE}"
echo "NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
echo "GIST_GRADIENT_CHECKPOINTING=${GIST_GRADIENT_CHECKPOINTING}"
echo "C2KV_GIST_DOC_MICROBATCH=${C2KV_GIST_DOC_MICROBATCH}"
echo "C2KV_GIST_TRAIN_RATIOS=${C2KV_GIST_TRAIN_RATIOS}"
echo "C2KV_GIST_CHECKPOINT_USE_REENTRANT=${C2KV_GIST_CHECKPOINT_USE_REENTRANT}"
echo "PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF}"
echo "DO_EVAL=${DO_EVAL}"
echo "SAVE_STEPS=${SAVE_STEPS}"
echo "SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT}"
echo "RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT}"
echo "DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "TORCHRUN_LOG_DIR=${TORCHRUN_LOG_DIR}"
echo "DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG}"

mkdir -p "${TORCHRUN_LOG_DIR}"

torchrun \
  --master_port "${MASTER_PORT}" \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --log_dir "${TORCHRUN_LOG_DIR}" \
  --redirects "${TORCHRUN_REDIRECTS}" \
  --tee "${TORCHRUN_TEE}" \
  -m train.train_compress_history \
  --device_type npu \
  --npu_attn_impl "${NPU_ATTN_IMPL}" \
  --attn_impl "${NPU_ATTN_IMPL}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  "${MAX_STEPS_ARGS[@]}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --padding_side right \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --lr_scheduler_type cosine \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay 0.1 \
  --enable_gist True \
  --gist_param qkv \
  --gist_type dynamic-interleave \
  --gist_overlap 64 \
  --gist_residual_type embed-mean \
  --gist_gradient_checkpointing "${GIST_GRADIENT_CHECKPOINTING}" \
  --only_train_gist True \
  --train_data "${DATASET_PATH}" \
  "${SOURCE_ARGS[@]}" \
  "${SAMPLE_ARGS[@]}" \
  --max_doc_length "${MAX_DOC_LENGTH}" \
  --min_doc_num "${MIN_DOC_NUM}" \
  --max_doc_num "${MAX_DOC_NUM}" \
  --max_length "${MAX_LENGTH}" \
  --max_system_length "${MAX_SYSTEM_LENGTH}" \
  --history_selection "${HISTORY_SELECTION}" \
  --output_dir "${OUTPUT_DIR}" \
  --logging_steps 1 \
  --logging_nan_inf_filter False \
  --remove_unused_columns False \
  "${DEEPSPEED_ARGS[@]}" \
  --ddp_timeout "${DDP_TIMEOUT}" \
  --do_train True \
  "${EVAL_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  "${DATALOADER_ARGS[@]}" \
  --bf16 True \
  --dataset_shuffle_seed "${DATASET_SHUFFLE_SEED}"
