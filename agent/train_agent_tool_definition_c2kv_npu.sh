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
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NPU_ATTN_IMPL="${NPU_ATTN_IMPL:-eager}"

torchrun --nproc_per_node "${NPROC_PER_NODE}" \
  agent/train_agent_tool_definition_c2kv.py \
  --device_type npu \
  --npu_attn_impl "${NPU_ATTN_IMPL}" \
  --attn_impl "${NPU_ATTN_IMPL}" \
  --num_train_epochs 1 \
  --warmup_steps 100 \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --padding_side right \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --lr_scheduler_type cosine \
  --learning_rate 5e-6 \
  --weight_decay 0.1 \
  --enable_gist True \
  --gist_param qkv \
  --gist_type dynamic-interleave \
  --gist_overlap 64 \
  --gist_residual_type embed-mean \
  --gist_gradient_checkpointing True \
  --only_train_gist True \
  --dataset_path "${DATASET_PATH}" \
  --max_doc_length 4096 \
  --max_doc_num 1 \
  --max_length 2048 \
  --max_system_length 256 \
  --max_samples_per_session 4 \
  --eval_ratio 0.1 \
  --require_tool_call True \
  --output_dir "${OUTPUT_DIR}" \
  --logging_steps 1 \
  --deepspeed ./configs/ds_config_npu.json \
  --do_train True \
  --eval_strategy steps \
  --eval_steps 100 \
  --save_strategy steps \
  --save_steps 500 \
  --dataloader_num_workers 4 \
  --dataloader_prefetch_factor 4 \
  --bf16 True \
  --dataset_shuffle_seed 2948
