export PYTHONPATH=`pwd`/python:$PYTHONPATH
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=64 torchrun --nproc_per_node 8 -m train.train_mdoc \
    --resume_from_checkpoint ./checkpoints/qwen3-32b-mixed/260507-tulu3-dynamic_interleave-1024-embed_residual/checkpoint-2500 \
    --num_train_epochs 1 \
    --warmup_ratio 0.05 \
    --model_name_or_path Qwen/Qwen3-32B \
    --padding_side right \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --learning_rate 5e-6 \
    --weight_decay 0.1 \
    --enable_gist True \
    --gist_param qkv \
    --gist_type dynamic-interleave \
    --gist_residual_type embed-mean \
    --gist_gradient_checkpointing True \
    --output_dir ./checkpoints/qwen3-32b-mixed/260507-tulu3-dynamic_interleave-1024-embed_residual \
    --logging_dir ./logs/qwen3-32b-mixed/260507-tulu3-dynamic_interleave-1024-embed_residual \
    --logging_steps 1 \
    --deepspeed ./configs/ds_config_32b.json \
    --do_train True \
    --eval_strategy steps \
    --eval_steps 50 \
    --only_train_gist True \
    --dataloader_num_workers 8 \
    --dataloader_prefetch_factor 32 \
    --train_data /mnt/nas1/duchuheng/datasets \
    --bf16 True \
    --save_strategy steps \
    --save_steps 500 \
    --dataset_shuffle_seed 7561
    # --gradient_checkpointing True
    # --device_map auto \
