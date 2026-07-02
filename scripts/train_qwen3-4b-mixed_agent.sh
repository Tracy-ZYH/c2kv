export PYTHONPATH=`pwd`/python:$PYTHONPATH
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=64 torchrun --nproc_per_node 8 -m train.train_compress_history \
    --num_train_epochs 1 \
    --warmup_steps 200 \
    --model_name_or_path checkpoints/qwen3-4b-mixed/260625-dyn-overlap64-embed_residual/checkpoint-8000 \
    --padding_side right \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --weight_decay 0.1 \
    --enable_gist True \
    --gist_param qkv \
    --gist_type dynamic-interleave \
    --gist_overlap 64 \
    --gist_residual_type embed-mean \
    --gist_gradient_checkpointing True \
    --output_dir ./checkpoints/qwen3-4b-mixed/open-swe-traces/260701 \
    --logging_steps 1 \
    --deepspeed ./configs/ds_config.json \
    --do_train True \
    --only_train_gist True \
    --dataloader_num_workers 8 \
    --dataloader_prefetch_factor 32 \
    --train_data /mnt/nas1/duchuheng/datasets/nvidia--Open-SWE-Traces--train \
    --bf16 True \
    --save_strategy steps \
    --save_steps 1000 \
    --dataset_shuffle_seed 2948 \
    --source_type open_swe \
    --max_samples_per_trace 8 \
    --max_doc_num 10 \
    --max_doc_length 1024 \
    --max_system_length 8192 \
    --max_length 16384 \
    --resolved_only True \
    --recent_message_num 4 \
    --num_samples 130000 \
    --num_proc 128
    # --gradient_checkpointing True
    # --device_map auto \
