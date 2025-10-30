OMP_NUM_THREADS=4 torchrun --nproc_per_node 4 -m train.stage1 \
    --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
    --padding_side right \
    --max_length 4096 \
    --per_device_train_batch_size 2 \
    --enable_gist True \
    --gist_type interleave-4 \
    --gist_mode 1024-8 \
    --output_dir ./checkpoints/qwen3 \
    --deepspeed ./train/ds_config.json \
    --do_train True \
    --logging_dir ./runs \
    --only_train_gist True \
    --train_data /mnt/nas1/alsc_supply_tech_SlimPajama-627B_20240926201127 \
    --max_steps 10000 \
    --fp16 True
    # --device_map auto \
