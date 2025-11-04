export PYTHONPATH=`pwd`/python:$PYTHONPATH
OMP_NUM_THREADS=64 torchrun --nproc_per_node 8 -m train.stage1 \
    --max_steps 100000 \
    --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
    --padding_side right \
    --pretrain_min_length 1024 \
    --pretrain_max_length 4096 \
    --per_device_train_batch_size 2 \
    --enable_gist True \
    --gist_type interleave-4 \
    --gist_mode 512,1024,1536,2048-1 \
    --output_dir ./outputs/qwen3/single-chunk \
    --logging_dir ./logs/qwen3/single-chunk \
    --logging_steps 10 \
    --deepspeed ./configs/ds_config.json \
    --do_train True \
    --only_train_gist True \
    --train_data /mnt/nas1/alsc_supply_tech_SlimPajama-627B_20240926201127 \
    --fp16 True \
    --save_strategy steps \
    --save_steps 10000 
    # --gradient_checkpointing True
    # --device_map auto \