#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)/python:$(pwd)/agent:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-7200}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export C2KV_GIST_DOC_MICROBATCH="${C2KV_GIST_DOC_MICROBATCH:-1}"
export C2KV_GIST_TRAIN_RATIOS="${C2KV_GIST_TRAIN_RATIOS:-4,8,16}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-./models/Qwen3-4B-Instruct-2507}"
DATASET_PATH="${DATASET_PATH:-./datasets/agent-llm-traces}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/qwen3-4b-agent-tooldoc-hardneg-npu}"
NPU_ATTN_IMPL="${NPU_ATTN_IMPL:-npu_fusion_attention}"

SPLIT_MANIFEST_FILE="${SPLIT_MANIFEST_FILE:-}"
SPLIT_NAME="${SPLIT_NAME:-subset_disjoint}"
SPLIT_SEED="${SPLIT_SEED:-42}"
EVAL_RATIO="${EVAL_RATIO:-0.1}"

MAX_DOC_LENGTH="${TOOLDOC_MAX_DOC_LENGTH:-2048}"
MAX_DOC_NUM="${TOOLDOC_MAX_DOC_NUM:-16}"
MAX_TOOL_DEFINITION_TOKENS="${TOOLDOC_MAX_TOOL_DEFINITION_TOKENS:-131072}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_SYSTEM_LENGTH="${MAX_SYSTEM_LENGTH:-256}"
MAX_SAMPLES_PER_SESSION="${MAX_SAMPLES_PER_SESSION:-4}"
MAX_SAMPLES_PER_SUBSET="${MAX_SAMPLES_PER_SUBSET:-}"
REQUIRE_TOOL_CALL="${REQUIRE_TOOL_CALL:-True}"
TRUNCATE_TOOL_DEFINITION="${TRUNCATE_TOOL_DEFINITION:-False}"
MIN_TARGET_TOKENS="${MIN_TARGET_TOKENS:-128}"

HARD_NEGATIVE_NUM="${HARD_NEGATIVE_NUM:-15}"
HARD_NEGATIVE_ROUTER_SCOPE="${HARD_NEGATIVE_ROUTER_SCOPE:-last_user}"
SHUFFLE_TOOL_DOCUMENTS="${SHUFFLE_TOOL_DOCUMENTS:-True}"
BALANCE_SUBSETS="${BALANCE_SUBSETS:-True}"

LEARNING_RATE="${LEARNING_RATE:-5e-7}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-5}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
GIST_GRADIENT_CHECKPOINTING="${GIST_GRADIENT_CHECKPOINTING:-True}"
EVAL_STEPS="${EVAL_STEPS:-25}"
SAVE_STEPS="${SAVE_STEPS:-100}"
DATASET_SHUFFLE_SEED="${DATASET_SHUFFLE_SEED:-2948}"
DO_EVAL="${DO_EVAL:-True}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-4}"
MASTER_PORT="${MASTER_PORT:-29500}"
TORCHRUN_LOG_DIR="${TORCHRUN_LOG_DIR:-./outputs/torchrun_tooldoc_hardneg_logs}"
TORCHRUN_REDIRECTS="${TORCHRUN_REDIRECTS:-3}"
TORCHRUN_TEE="${TORCHRUN_TEE:-0}"
DDP_TIMEOUT="${DDP_TIMEOUT:-7200}"
# Use ${VAR-default} so DEEPSPEED_CONFIG="" really disables DeepSpeed.
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG-./configs/ds_config_npu.json}"
if [[ -z "${C2KV_GIST_CHECKPOINT_USE_REENTRANT+x}" ]]; then
  if [[ -n "${DEEPSPEED_CONFIG}" && "${DEEPSPEED_CONFIG}" != "none" && "${DEEPSPEED_CONFIG}" != "None" ]]; then
    export C2KV_GIST_CHECKPOINT_USE_REENTRANT=True
  else
    export C2KV_GIST_CHECKPOINT_USE_REENTRANT=False
  fi
fi

if (( HARD_NEGATIVE_NUM + 1 > MAX_DOC_NUM )); then
  echo "ERROR: HARD_NEGATIVE_NUM + 1 must be <= MAX_DOC_NUM for target+hard-neg document construction." >&2
  echo "Got HARD_NEGATIVE_NUM=${HARD_NEGATIVE_NUM}, MAX_DOC_NUM=${MAX_DOC_NUM}." >&2
  echo "Use TOOLDOC_MAX_DOC_NUM=$((HARD_NEGATIVE_NUM + 1)) or reduce HARD_NEGATIVE_NUM." >&2
  exit 1
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

SPLIT_ARGS=(--split_manifest_name "${SPLIT_NAME}")
if [[ -n "${SPLIT_MANIFEST_FILE}" ]]; then
  SPLIT_ARGS=(--split_manifest_file "${SPLIT_MANIFEST_FILE}" --split_manifest_name "${SPLIT_NAME}")
fi

SUBSET_ARGS=(--balance_subsets "${BALANCE_SUBSETS}")
if [[ -n "${MAX_SAMPLES_PER_SUBSET}" ]]; then
  SUBSET_ARGS+=(--max_samples_per_subset "${MAX_SAMPLES_PER_SUBSET}")
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
echo "MAX_TOOL_DEFINITION_TOKENS=${MAX_TOOL_DEFINITION_TOKENS}"
echo "HARD_NEGATIVE_NUM=${HARD_NEGATIVE_NUM}"
echo "BALANCE_SUBSETS=${BALANCE_SUBSETS}"
echo "MAX_SAMPLES_PER_SUBSET=${MAX_SAMPLES_PER_SUBSET}"
echo "LEARNING_RATE=${LEARNING_RATE}"
echo "NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}"
echo "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
echo "GIST_GRADIENT_CHECKPOINTING=${GIST_GRADIENT_CHECKPOINTING}"
echo "C2KV_GIST_DOC_MICROBATCH=${C2KV_GIST_DOC_MICROBATCH}"
echo "C2KV_GIST_TRAIN_RATIOS=${C2KV_GIST_TRAIN_RATIOS}"
echo "C2KV_GIST_CHECKPOINT_USE_REENTRANT=${C2KV_GIST_CHECKPOINT_USE_REENTRANT}"
echo "DO_EVAL=${DO_EVAL}"
echo "DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS}"
echo "DATALOADER_PREFETCH_FACTOR=${DATALOADER_PREFETCH_FACTOR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "DDP_TIMEOUT=${DDP_TIMEOUT}"
echo "HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT}"
echo "TORCHRUN_LOG_DIR=${TORCHRUN_LOG_DIR}"
echo "DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG}"

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

mkdir -p "${TORCHRUN_LOG_DIR}"

torchrun \
  --master_port "${MASTER_PORT}" \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --log_dir "${TORCHRUN_LOG_DIR}" \
  --redirects "${TORCHRUN_REDIRECTS}" \
  --tee "${TORCHRUN_TEE}" \
  agent/train_agent_tool_definition_c2kv.py \
  --device_type npu \
  --npu_attn_impl "${NPU_ATTN_IMPL}" \
  --attn_impl "${NPU_ATTN_IMPL}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
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
  --dataset_path "${DATASET_PATH}" \
  "${SPLIT_ARGS[@]}" \
  --split_seed "${SPLIT_SEED}" \
  --eval_ratio "${EVAL_RATIO}" \
  --tool_document_mode target_hard_negatives \
  --hard_negative_num "${HARD_NEGATIVE_NUM}" \
  --hard_negative_router_scope "${HARD_NEGATIVE_ROUTER_SCOPE}" \
  --shuffle_tool_documents "${SHUFFLE_TOOL_DOCUMENTS}" \
  "${SUBSET_ARGS[@]}" \
  --max_doc_length "${MAX_DOC_LENGTH}" \
  --max_doc_num "${MAX_DOC_NUM}" \
  --max_tool_definition_tokens "${MAX_TOOL_DEFINITION_TOKENS}" \
  --max_length "${MAX_LENGTH}" \
  --max_system_length "${MAX_SYSTEM_LENGTH}" \
  --max_samples_per_session "${MAX_SAMPLES_PER_SESSION}" \
  --truncate_tool_definition "${TRUNCATE_TOOL_DEFINITION}" \
  --min_target_tokens "${MIN_TARGET_TOKENS}" \
  --require_tool_call "${REQUIRE_TOOL_CALL}" \
  --output_dir "${OUTPUT_DIR}" \
  --logging_steps 1 \
  --logging_nan_inf_filter False \
  --remove_unused_columns False \
  "${DEEPSPEED_ARGS[@]}" \
  --ddp_timeout "${DDP_TIMEOUT}" \
  --do_train True \
  "${EVAL_ARGS[@]}" \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  "${DATALOADER_ARGS[@]}" \
  --bf16 True \
  --dataset_shuffle_seed "${DATASET_SHUFFLE_SEED}"
