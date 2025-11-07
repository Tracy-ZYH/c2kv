export PYTHONPATH=`pwd`/python:$PYTHONPATH
OMP_NUM_THREADS=64 torchrun --nproc_per_node 8 -m train.stage1 \
    --max_steps 100000 \
    --model_name_or_path ./outputs/qwen3-4b/32x-single-chunk/checkpoint-50000 \
    --padding_side right \
    --pretrain_min_length 2048 \
    --pretrain_max_length 6144 \
    --per_device_train_batch_size 3 \
    --enable_gist True \
    --gist_type interleave-32 \
    --gist_mode 128,256,512,1024-4 \
    --output_dir ./outputs/qwen3-4b/32x-multi-chunk \
    --logging_dir ./logs/qwen3-4b/32x-multi-chunk \
    --logging_steps 10 \
    --deepspeed ./configs/ds_config.json \
    --do_train True \
    --only_train_gist True \
    --train_data /mnt/nas1/alsc_supply_tech_SlimPajama-627B_20240926201127 \
    --fp16 True \
    --save_strategy steps \
    --save_steps 5000
    # --gradient_checkpointing True
    # --device_map auto \
    # --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
