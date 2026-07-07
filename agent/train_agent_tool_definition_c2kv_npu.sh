#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)/python:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
DATASET_PATH="${DATASET_PATH:-./datasets/agent-llm-traces}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/qwen3-4b-agent-tooldef-npu}"
NPU_ATTN_IMPL="${NPU_ATTN_IMPL:-eager}"
MAX_DOC_LENGTH="${MAX_DOC_LENGTH:-1024}"
MAX_DOC_NUM="${MAX_DOC_NUM:-10}"
MAX_TOOL_DEFINITION_TOKENS="${MAX_TOOL_DEFINITION_TOKENS:-10000}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_SAMPLES_PER_SESSION="${MAX_SAMPLES_PER_SESSION:-4}"
REQUIRE_TOOL_CALL="${REQUIRE_TOOL_CALL:-True}"
TRUNCATE_TOOL_DEFINITION="${TRUNCATE_TOOL_DEFINITION:-False}"
MIN_TARGET_TOKENS="${MIN_TARGET_TOKENS:-128}"
LEARNING_RATE="${LEARNING_RATE:-5e-7}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-5}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
EVAL_STEPS="${EVAL_STEPS:-25}"
SAVE_STEPS="${SAVE_STEPS:-100}"

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  IFS=',' read -ra _visible_npus <<< "${ASCEND_RT_VISIBLE_DEVICES}"
  NPROC_PER_NODE="${#_visible_npus[@]}"
fi

echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "MAX_DOC_LENGTH=${MAX_DOC_LENGTH}"
echo "MAX_DOC_NUM=${MAX_DOC_NUM}"
echo "MAX_TOOL_DEFINITION_TOKENS=${MAX_TOOL_DEFINITION_TOKENS}"
echo "MAX_LENGTH=${MAX_LENGTH}"
echo "MIN_TARGET_TOKENS=${MIN_TARGET_TOKENS}"
echo "LEARNING_RATE=${LEARNING_RATE}"
echo "NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}"
echo "WARMUP_STEPS=${WARMUP_STEPS}"
echo "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"

if ! find "${DATASET_PATH}" -name '*.parquet' -type f -print -quit 2>/dev/null | grep -q .; then
  echo "ERROR: no parquet files found under DATASET_PATH=${DATASET_PATH}" >&2
  echo "Expected files like: ${DATASET_PATH}/data/train-00000-of-00039.parquet" >&2
  echo "Set DATASET_PATH to the real dataset directory, or copy/sync agent-llm-traces there." >&2
  exit 1
fi

torchrun --nproc_per_node "${NPROC_PER_NODE}" \
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
  --gist_gradient_checkpointing True \
  --only_train_gist True \
  --dataset_path "${DATASET_PATH}" \
  --max_doc_length "${MAX_DOC_LENGTH}" \
  --max_doc_num "${MAX_DOC_NUM}" \
  --max_tool_definition_tokens "${MAX_TOOL_DEFINITION_TOKENS}" \
  --max_length "${MAX_LENGTH}" \
  --max_system_length 256 \
  --max_samples_per_session "${MAX_SAMPLES_PER_SESSION}" \
  --truncate_tool_definition "${TRUNCATE_TOOL_DEFINITION}" \
  --min_target_tokens "${MIN_TARGET_TOKENS}" \
  --eval_ratio 0.1 \
  --require_tool_call "${REQUIRE_TOOL_CALL}" \
  --output_dir "${OUTPUT_DIR}" \
  --logging_steps 1 \
  --logging_nan_inf_filter False \
  --deepspeed ./configs/ds_config_npu.json \
  --do_train True \
  --eval_strategy steps \
  --eval_steps "${EVAL_STEPS}" \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --dataloader_num_workers 4 \
  --dataloader_prefetch_factor 4 \
  --bf16 True \
  --dataset_shuffle_seed 2948
