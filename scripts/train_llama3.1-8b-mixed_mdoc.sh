export PYTHONPATH=`pwd`/python:$PYTHONPATH
export OUTPUT_DIR=/home/admin/workspace/aop_lab/app_data/checkpoints/llama3.1-8b-mixed/
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=64 torchrun --nproc_per_node 8 -m train.train_mdoc \
    --num_train_epochs 1 \
    --warmup_steps 400 \
    --resume_from_checkpoint $OUTPUT_DIR/260120-distill-4x/checkpoint-1000 \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --padding_side right \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --weight_decay 0.1 \
    --enable_gist True \
    --gist_param qkv \
    --gist_self_distill_coef 1.0 \
    --gist_self_distill_temperature 1.0 \
    --gist_type interleave-4 \
    --output_dir $OUTPUT_DIR/260120-distill-4x \
    --logging_dir ./logs/llama3.1-8b-mixed/260120-distill-4x \
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
    --dataset_shuffle_seed 7367
    # --gradient_checkpointing True
    # --device_map auto \
