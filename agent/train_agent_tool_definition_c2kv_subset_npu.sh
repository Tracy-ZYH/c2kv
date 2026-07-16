#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)/python:$(pwd)/agent:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-./models/Qwen3-4B-Instruct-2507}"
DATASET_PATH="${DATASET_PATH:-./datasets/agent-llm-traces}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/qwen3-4b-agent-tooldef-subset-npu}"
NPU_ATTN_IMPL="${NPU_ATTN_IMPL:-eager}"

SPLIT_NAME="${SPLIT_NAME:-subset_disjoint}"
EVAL_SUBSETS="${EVAL_SUBSETS:-swebench}"
SPLIT_MANIFEST_FILE="${SPLIT_MANIFEST_FILE:-./outputs/agent_tooldef_subset_split_manifest.json}"
REBUILD_SPLIT_MANIFEST="${REBUILD_SPLIT_MANIFEST:-True}"
SPLIT_SEED="${SPLIT_SEED:-42}"
EVAL_RATIO="${EVAL_RATIO:-0.1}"

MAX_DOC_LENGTH="${MAX_DOC_LENGTH:-1024}"
MAX_DOC_NUM="${MAX_DOC_NUM:-10}"
MAX_TOOL_DEFINITION_TOKENS="${MAX_TOOL_DEFINITION_TOKENS:-10000}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_SYSTEM_LENGTH="${MAX_SYSTEM_LENGTH:-256}"
MAX_SAMPLES_PER_SESSION="${MAX_SAMPLES_PER_SESSION:-4}"
REQUIRE_TOOL_CALL="${REQUIRE_TOOL_CALL:-True}"
TRUNCATE_TOOL_DEFINITION="${TRUNCATE_TOOL_DEFINITION:-False}"
MIN_TARGET_TOKENS="${MIN_TARGET_TOKENS:-128}"

LEARNING_RATE="${LEARNING_RATE:-5e-7}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-9}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
EVAL_STEPS="${EVAL_STEPS:-25}"
SAVE_STEPS="${SAVE_STEPS:-100}"
DATASET_SHUFFLE_SEED="${DATASET_SHUFFLE_SEED:-2948}"

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  IFS=',' read -ra _visible_npus <<< "${ASCEND_RT_VISIBLE_DEVICES}"
  NPROC_PER_NODE="${#_visible_npus[@]}"
fi

if ! find "${DATASET_PATH}" -name '*.parquet' -type f -print -quit 2>/dev/null | grep -q .; then
  echo "ERROR: no parquet files found under DATASET_PATH=${DATASET_PATH}" >&2
  echo "Expected files like: ${DATASET_PATH}/data/train-00000-of-00039.parquet" >&2
  exit 1
fi

if [[ "${REBUILD_SPLIT_MANIFEST}" == "True" || ! -f "${SPLIT_MANIFEST_FILE}" ]]; then
  echo "Building subset-disjoint split manifest: ${SPLIT_MANIFEST_FILE}"
  python agent/build_agent_llm_traces_subset_manifest.py \
    --dataset_path "${DATASET_PATH}" \
    --output_file "${SPLIT_MANIFEST_FILE}" \
    --split_name "${SPLIT_NAME}" \
    --eval_subsets "${EVAL_SUBSETS}"
fi

echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "SPLIT_MANIFEST_FILE=${SPLIT_MANIFEST_FILE}"
echo "SPLIT_NAME=${SPLIT_NAME}"
echo "EVAL_SUBSETS=${EVAL_SUBSETS}"
echo "MAX_DOC_LENGTH=${MAX_DOC_LENGTH}"
echo "MAX_DOC_NUM=${MAX_DOC_NUM}"
echo "MAX_TOOL_DEFINITION_TOKENS=${MAX_TOOL_DEFINITION_TOKENS}"
echo "MAX_LENGTH=${MAX_LENGTH}"
echo "MAX_SYSTEM_LENGTH=${MAX_SYSTEM_LENGTH}"
echo "MAX_SAMPLES_PER_SESSION=${MAX_SAMPLES_PER_SESSION}"
echo "MIN_TARGET_TOKENS=${MIN_TARGET_TOKENS}"
echo "LEARNING_RATE=${LEARNING_RATE}"
echo "NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}"
echo "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"

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
  --split_manifest_file "${SPLIT_MANIFEST_FILE}" \
  --split_manifest_name "${SPLIT_NAME}" \
  --split_seed "${SPLIT_SEED}" \
  --eval_ratio "${EVAL_RATIO}" \
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
  --deepspeed ./configs/ds_config_npu.json \
  --do_train True \
  --eval_strategy steps \
  --eval_steps "${EVAL_STEPS}" \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --dataloader_num_workers 4 \
  --dataloader_prefetch_factor 4 \
  --bf16 True \
  --dataset_shuffle_seed "${DATASET_SHUFFLE_SEED}"
