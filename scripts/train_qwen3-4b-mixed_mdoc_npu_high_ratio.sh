#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)/python:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-21600}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}"
export C2KV_GIST_TRAIN_RATIOS="${C2KV_GIST_TRAIN_RATIOS:-16}"
export C2KV_GIST_DOC_MICROBATCH="${C2KV_GIST_DOC_MICROBATCH:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-max_split_size_mb:128}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-./models/Qwen3-4B-Instruct-2507}"
DATASET_PATH="${DATASET_PATH:-/home/zhuyuhan/project/c2kv/datasets_cleaned}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/qwen3-4b-mixed-mdoc-c2kv-r16-npu}"
NPU_ATTN_IMPL="${NPU_ATTN_IMPL:-npu_fusion_attention}"

TRAIN_SOURCES="${TRAIN_SOURCES:-hotpotqa,wikimqa,longmagpie}"
EVAL_SOURCES="${EVAL_SOURCES:-hotpotqa,wikimqa}"
TRAIN_SOURCE_SIZES="${TRAIN_SOURCE_SIZES:-3000,3000,3000}"
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-512}"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-5e-7}"
MAX_DOC_LENGTH="${MAX_DOC_LENGTH:-1024}"
MAX_DOC_NUM="${MAX_DOC_NUM:-10}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_SYSTEM_LENGTH="${MAX_SYSTEM_LENGTH:-256}"
DATASET_SHUFFLE_SEED="${DATASET_SHUFFLE_SEED:-2948}"
GIST_GRADIENT_CHECKPOINTING="${GIST_GRADIENT_CHECKPOINTING:-True}"

DO_EVAL="${DO_EVAL:-True}"
EVAL_STEPS="${EVAL_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-32}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}"
DATASET_LOAD_FROM_CACHE_FILE="${DATASET_LOAD_FROM_CACHE_FILE:-True}"
PREPROCESS_CACHE_VERSION="${PREPROCESS_CACHE_VERSION:-mdoc_answer_preserve_v2}"
MASTER_PORT="${MASTER_PORT:-29588}"
TORCHRUN_LOG_DIR="${TORCHRUN_LOG_DIR:-./outputs/torchrun_mdoc_high_ratio_logs}"
TORCHRUN_RDZV_ID="${TORCHRUN_RDZV_ID:-mdoc_high_ratio_r16_$(date +%Y%m%d_%H%M%S)}"
TORCHRUN_REDIRECTS="${TORCHRUN_REDIRECTS:-3}"
TORCHRUN_TEE="${TORCHRUN_TEE:-3}"
DDP_TIMEOUT="${DDP_TIMEOUT:-7200}"
# Qwen3-4B fits on 910B when only gist parameters are trained. Plain DDP avoids
# ZeRO-3 parameter all-gather hangs that can appear on Ascend/HCCL.
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-none}"

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  IFS=',' read -ra _visible_npus <<< "${ASCEND_RT_VISIBLE_DEVICES}"
  NPROC_PER_NODE="${#_visible_npus[@]}"
fi

for required_dir in \
  "${DATASET_PATH}/hotpotqa_train_cleaned" \
  "${DATASET_PATH}/wikimqa_train_cleaned" \
  "${DATASET_PATH}/longmagpie_cleaned"; do
  if [[ ! -d "${required_dir}" ]]; then
    echo "ERROR: required dataset directory not found: ${required_dir}" >&2
    exit 1
  fi
done

if [[ -z "${C2KV_GIST_CHECKPOINT_USE_REENTRANT+x}" ]]; then
  if [[ -n "${DEEPSPEED_CONFIG}" && "${DEEPSPEED_CONFIG}" != "none" && "${DEEPSPEED_CONFIG}" != "None" ]]; then
    export C2KV_GIST_CHECKPOINT_USE_REENTRANT=True
  else
    export C2KV_GIST_CHECKPOINT_USE_REENTRANT=False
  fi
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
echo "ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "TRAIN_SOURCES=${TRAIN_SOURCES}"
echo "TRAIN_SOURCE_SIZES=${TRAIN_SOURCE_SIZES}"
echo "C2KV_GIST_TRAIN_RATIOS=${C2KV_GIST_TRAIN_RATIOS}"
echo "NPU_ATTN_IMPL=${NPU_ATTN_IMPL}"
echo "DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG}"
echo "DATASET_NUM_PROC=${DATASET_NUM_PROC}"
echo "DATASET_LOAD_FROM_CACHE_FILE=${DATASET_LOAD_FROM_CACHE_FILE}"
echo "PREPROCESS_CACHE_VERSION=${PREPROCESS_CACHE_VERSION}"
echo "TORCHRUN_RDZV_ID=${TORCHRUN_RDZV_ID}"

mkdir -p "${TORCHRUN_LOG_DIR}"

torchrun \
  --master_port "${MASTER_PORT}" \
  --rdzv_id "${TORCHRUN_RDZV_ID}" \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --log_dir "${TORCHRUN_LOG_DIR}" \
  --redirects "${TORCHRUN_REDIRECTS}" \
  --tee "${TORCHRUN_TEE}" \
  -m train.train_mdoc \
  --device_type npu \
  --npu_attn_impl "${NPU_ATTN_IMPL}" \
  --attn_impl "${NPU_ATTN_IMPL}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --padding_side right \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 16 \
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
  --output_dir "${OUTPUT_DIR}" \
  --logging_steps 10 \
  "${DEEPSPEED_ARGS[@]}" \
  --ddp_timeout "${DDP_TIMEOUT}" \
  --do_train True \
  "${EVAL_ARGS[@]}" \
  --only_train_gist True \
  --train_data "${DATASET_PATH}" \
  --train_data_cleaned "${DATASET_PATH}" \
  --train_sources "${TRAIN_SOURCES}" \
  --eval_sources "${EVAL_SOURCES}" \
  --train_source_sizes "${TRAIN_SOURCE_SIZES}" \
  --eval_num_samples "${EVAL_NUM_SAMPLES}" \
  --max_doc_length "${MAX_DOC_LENGTH}" \
  --max_doc_num "${MAX_DOC_NUM}" \
  --max_length "${MAX_LENGTH}" \
  --max_system_length "${MAX_SYSTEM_LENGTH}" \
  --dataset_num_proc "${DATASET_NUM_PROC}" \
  --dataset_load_from_cache_file "${DATASET_LOAD_FROM_CACHE_FILE}" \
  --preprocess_cache_version "${PREPROCESS_CACHE_VERSION}" \
  "${DATALOADER_ARGS[@]}" \
  --bf16 True \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --dataset_shuffle_seed "${DATASET_SHUFFLE_SEED}" \
  --remove_unused_columns False
