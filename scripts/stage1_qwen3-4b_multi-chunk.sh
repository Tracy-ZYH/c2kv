export PYTHONPATH=`pwd`/python:$PYTHONPATH
export OUTPUT_DIR=/home/admin/workspace/aop_lab/app_data/checkpoints/qwen3-4b/
OMP_NUM_THREADS=64 torchrun --nproc_per_node 8 -m train.stage1 \
    --max_steps 40000 \
    --model_name_or_path $OUTPUT_DIR/32x-single-chunk/checkpoint-15000 \
    --padding_side right \
    --pretrain_min_length 4096 \
    --pretrain_max_length 8192 \
    --per_device_train_batch_size 2 \
    --enable_gist True \
    --gist_type interleave-32 \
    --gist_mode 512,768,1024,1280,1536-4 \
    --output_dir $OUTPUT_DIR/32x-multi-chunk \
    --logging_dir ./logs/qwen3-4b/32x-multi-chunk \
    --logging_steps 10 \
    --deepspeed ./configs/ds_config.json \
    --do_train True \
    --only_train_gist True \
    --train_data /mnt/nas1/alsc_supply_tech_SlimPajama-627B_20240926201127 \
    --fp16 True \
    --save_strategy steps \
    --save_steps 5000 \
    --dataset_shuffle_seed 9384
    # --resume_from_checkpoint $OUTPUT_DIR/32x-multi-chunk/checkpoint-35000 \
    # --gradient_checkpointing True
    # --device_map auto \
    # --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
