export PYTHONPATH=`pwd`/python:$PYTHONPATH
export OUTPUT_DIR=/home/admin/workspace/aop_lab/app_data/checkpoints/qwen3-8b-mixed/
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=64 torchrun --nproc_per_node 8 -m train.train_mdoc \
    --num_train_epochs 1 \
    --warmup_steps 200 \
    --model_name_or_path Qwen/Qwen3-8B \
    --padding_side right \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --weight_decay 0.1 \
    --enable_gist True \
    --gist_param qkv \
    --gist_reconstruct_loss_coef 1.0 \
    --gist_type interleave-16 \
    --output_dir $OUTPUT_DIR/260119-reconstruct-16x \
    --logging_dir ./logs/qwen3-8b-mixed/260119-reconstruct-16x \
    --logging_steps 1 \
    --deepspeed ./configs/ds_config.json \
    --do_train True \
    --eval_strategy steps \
    --eval_steps 100 \
    --only_train_gist True \
    --dataloader_num_workers 8 \
    --dataloader_prefetch_factor 32 \
    --train_data /mnt/nas1/duchuheng/datasets/longmagpie_processed \
    --bf16 True \
    --save_strategy steps \
    --save_steps 1000 \
    --dataset_shuffle_seed 2948
    # --gradient_checkpointing True
    # --device_map auto \
