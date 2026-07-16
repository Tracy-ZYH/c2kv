#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)/python:$(pwd)/agent:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-./models/Qwen3-4B-Instruct-2507}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/qwen3-4b-unified-next-action-c2kv-npu}"
AGENT_DATASET_PATH="${AGENT_DATASET_PATH:-./datasets/agent-llm-traces}"
TOOLATHLON_DATASET_PATH="${TOOLATHLON_DATASET_PATH:-./datasets/toolathlon}"
HERMES_DATASET_PATH="${HERMES_DATASET_PATH:-./datasets/hermes-agent-reasoning-traces}"
HERMES_CONFIGS="${HERMES_CONFIGS:-kimi,glm-5.1}"
SOURCE_MIX="${SOURCE_MIX:-agent_llm_traces:0.3,toolathlon:0.4,hermes:0.3}"
SPLIT_SEED="${SPLIT_SEED:-42}"

NPU_ATTN_IMPL="${NPU_ATTN_IMPL:-eager}"
MAX_DOC_LENGTH="${MAX_DOC_LENGTH:-1024}"
MAX_DOC_NUM="${MAX_DOC_NUM:-32}"
MAX_TOOL_DEFINITION_TOKENS="${MAX_TOOL_DEFINITION_TOKENS:-32000}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_SYSTEM_LENGTH="${MAX_SYSTEM_LENGTH:-256}"
MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-20000}"
MAX_EVAL_EXAMPLES="${MAX_EVAL_EXAMPLES:-2000}"
MAX_STEPS_PER_TRAJECTORY="${MAX_STEPS_PER_TRAJECTORY:-6}"
MAX_HISTORY_STEPS="${MAX_HISTORY_STEPS:-6}"
MAX_OBSERVATION_CHARS="${MAX_OBSERVATION_CHARS:-1200}"
MAX_HISTORY_CHARS="${MAX_HISTORY_CHARS:-12000}"
MAX_TOOLS_PER_SAMPLE="${MAX_TOOLS_PER_SAMPLE:-32}"
MIN_TARGET_TOKENS="${MIN_TARGET_TOKENS:-32}"

LEARNING_RATE="${LEARNING_RATE:-5e-7}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
EVAL_STEPS="${EVAL_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-500}"

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  IFS=',' read -ra _visible_npus <<< "${ASCEND_RT_VISIBLE_DEVICES}"
  NPROC_PER_NODE="${#_visible_npus[@]}"
fi

echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "AGENT_DATASET_PATH=${AGENT_DATASET_PATH}"
echo "TOOLATHLON_DATASET_PATH=${TOOLATHLON_DATASET_PATH}"
echo "HERMES_DATASET_PATH=${HERMES_DATASET_PATH}"
echo "SOURCE_MIX=${SOURCE_MIX}"
echo "MAX_TRAIN_EXAMPLES=${MAX_TRAIN_EXAMPLES}"
echo "MAX_EVAL_EXAMPLES=${MAX_EVAL_EXAMPLES}"
echo "MAX_DOC_LENGTH=${MAX_DOC_LENGTH}"
echo "MAX_DOC_NUM=${MAX_DOC_NUM}"
echo "MAX_TOOL_DEFINITION_TOKENS=${MAX_TOOL_DEFINITION_TOKENS}"
echo "MAX_LENGTH=${MAX_LENGTH}"
echo "MAX_TOOLS_PER_SAMPLE=${MAX_TOOLS_PER_SAMPLE}"

torchrun --nproc_per_node "${NPROC_PER_NODE}" \
  agent/train_unified_next_action_c2kv.py \
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
  --agent_dataset_path "${AGENT_DATASET_PATH}" \
  --toolathlon_dataset_path "${TOOLATHLON_DATASET_PATH}" \
  --hermes_dataset_path "${HERMES_DATASET_PATH}" \
  --hermes_configs "${HERMES_CONFIGS}" \
  --source_mix "${SOURCE_MIX}" \
  --split_seed "${SPLIT_SEED}" \
  --max_train_examples "${MAX_TRAIN_EXAMPLES}" \
  --max_eval_examples "${MAX_EVAL_EXAMPLES}" \
  --max_steps_per_trajectory "${MAX_STEPS_PER_TRAJECTORY}" \
  --max_history_steps "${MAX_HISTORY_STEPS}" \
  --max_observation_chars "${MAX_OBSERVATION_CHARS}" \
  --max_history_chars "${MAX_HISTORY_CHARS}" \
  --max_tools_per_sample "${MAX_TOOLS_PER_SAMPLE}" \
  --max_doc_length "${MAX_DOC_LENGTH}" \
  --max_doc_num "${MAX_DOC_NUM}" \
  --max_tool_definition_tokens "${MAX_TOOL_DEFINITION_TOKENS}" \
  --max_length "${MAX_LENGTH}" \
  --max_system_length "${MAX_SYSTEM_LENGTH}" \
  --truncate_tool_definition True \
  --min_target_tokens "${MIN_TARGET_TOKENS}" \
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
