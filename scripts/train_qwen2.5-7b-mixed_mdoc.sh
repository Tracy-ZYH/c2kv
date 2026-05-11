export PYTHONPATH=`pwd`/python:$PYTHONPATH
export OUTPUT_DIR=./checkpoints/qwen2.5-7b-mixed/
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=64 torchrun --nproc_per_node 8 -m train.train_mdoc \
    --num_train_epochs 1 \
    --warmup_ratio 0.06 \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --padding_side right \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --weight_decay 0.1 \
    --enable_gist True \
    --gist_param qkv \
    --gist_type dynamic-interleave \
    --gist_residual_type embed-mean \
    --output_dir $OUTPUT_DIR/260430-tulu3-dynamic_interleave-1024-embed_residual \
    --logging_dir ./logs/qwen2.5-7b-mixed/260430-tulu3-dynamic_interleave-1024-embed_residual \
    --logging_steps 10 \
    --deepspeed ./configs/ds_config.json \
    --do_train True \
    --eval_strategy steps \
    --eval_steps 100 \
    --only_train_gist True \
    --dataloader_num_workers 8 \
    --dataloader_prefetch_factor 32 \
    --train_data /mnt/nas1/duchuheng/datasets \
    --bf16 True \
    --save_strategy steps \
    --save_steps 2000 \
    --dataset_shuffle_seed 2948
    # --gradient_checkpointing True
    # --device_map auto \
