export PYTHONPATH=`pwd`/python:$PYTHONPATH
export OUTPUT_DIR=/home/admin/workspace/aop_lab/app_data/checkpoints/qwen3-14b-mixed/
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=64 torchrun --nproc_per_node 8 -m train.train_mdoc \
    --num_train_epochs 1 \
    --warmup_ratio 0.06 \
    --model_name_or_path Qwen/Qwen3-14B \
    --padding_side right \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --weight_decay 0.1 \
    --enable_gist True \
    --gist_param qkv \
    --gist_type interleave-8 \
    --gist_residual_type mean \
    --output_dir $OUTPUT_DIR/260318-sft-8x-512-residual \
    --logging_dir ./logs/qwen3-14b-mixed/260318-sft-8x-512-residual \
    --logging_steps 1 \
    --deepspeed ./configs/ds_config.json \
    --do_train True \
    --eval_strategy steps \
    --eval_steps 200 \
    --only_train_gist True \
    --dataloader_num_workers 8 \
    --dataloader_prefetch_factor 32 \
    --train_data /mnt/nas1/duchuheng/datasets \
    --bf16 True \
    --save_strategy steps \
    --save_steps 2000 \
    --dataset_shuffle_seed 2948
    # --device_map auto \
