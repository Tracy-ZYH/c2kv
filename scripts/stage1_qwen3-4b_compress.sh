export PYTHONPATH=`pwd`/python:$PYTHONPATH
export OUTPUT_DIR=/home/admin/workspace/aop_lab/app_data/checkpoints/qwen3-4b-inst/
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=64 torchrun --nproc_per_node 8 -m train.stage1 \
    --num_train_epochs 4 \
    --warmup_steps 128 \
    --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
    --padding_side right \
    --dataset_min_length 3072 \
    --dataset_max_length 4096 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --enable_gist True \
    --gist_param qkv \
    --gist_type interleave-64 \
    --gist_mode 1024,1536,2048,2560-1 \
    --gist_reconstruct_loss_coef 1. \
    --output_dir $OUTPUT_DIR/compress-64x \
    --logging_dir ./logs/qwen3-4b-inst/compress-64x \
    --logging_steps 1 \
    --deepspeed ./configs/ds_config.json \
    --do_train True \
    --eval_strategy steps \
    --eval_steps 32 \
    --only_train_gist True \
    --train_data /mnt/nas1/alsc_supply_tech_SlimPajama-627B_20240926201127 \
    --bf16 True \
    --save_strategy steps \
    --save_steps 512 \
    --dataset_shuffle_seed 2348 
    # --gradient_checkpointing True
    # --device_map auto \