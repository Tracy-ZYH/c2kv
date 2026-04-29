export PYTHONPATH=`pwd`/python:$PYTHONPATH
export OUTPUT_DIR=/home/admin/workspace/aop_lab/app_data/checkpoints/qwen3-32b-mixed/
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=64 torchrun --nproc_per_node 8 -m train.train_mdoc \
    --num_train_epochs 1 \
    --warmup_ratio 0.06 \
    --model_name_or_path Qwen/Qwen3-32B \
    --padding_side right \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --enable_gist True \
    --gist_param qkv \
    --gist_type dynamic-interleave \
    --gist_residual_type mean \
    --gist_gradient_checkpointing True \
    --output_dir $OUTPUT_DIR/260429-tulu3-dynamic_interleave-1024-residual \
    --logging_dir ./logs/qwen3-32b-mixed/260429-tulu3-dynamic_interleave-1024-residual \
    --logging_steps 1 \
    --deepspeed ./configs/ds_config.json \
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
