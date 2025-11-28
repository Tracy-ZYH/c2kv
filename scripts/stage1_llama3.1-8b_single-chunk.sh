export PYTHONPATH=`pwd`/python:$PYTHONPATH
export OUTPUT_DIR=/home/admin/workspace/aop_lab/app_data/checkpoints/llama3.1-8b-inst/
OMP_NUM_THREADS=64 torchrun --nproc_per_node 8 -m train.stage1 \
    --max_steps 50000 \
    --warmup_steps 5000 \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --padding_side right \
    --dataset_min_length 2048 \
    --dataset_max_length 4096 \
    --per_device_train_batch_size 3 \
    --enable_gist True \
    --gist_type interleave-16 \
    --gist_mode 1024,1536,2048,2560,3072-2 \
    --output_dir $OUTPUT_DIR/16x-single-chunk \
    --logging_dir ./logs/llama3.1-8b-inst/16x-single-chunk \
    --logging_steps 10 \
    --deepspeed ./configs/ds_config.json \
    --do_train True \
    --eval_strategy steps \
    --eval_steps 100 \
    --only_train_gist True \
    --train_data /mnt/nas1/alsc_supply_tech_SlimPajama-627B_20240926201127 \
    --bf16 True \
    --save_strategy steps \
    --save_steps 5000
    # --gradient_checkpointing True
    # --device_map auto \