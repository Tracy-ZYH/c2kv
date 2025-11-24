export PYTHONPATH=`pwd`/python:$PYTHONPATH
export OUTPUT_DIR=/home/admin/workspace/aop_lab/app_data/checkpoints/qwen3-4b-inst/
OMP_NUM_THREADS=64 torchrun --nproc_per_node 4 -m train.stage2 \
    --num_train_epochs 5 \
    --warmup_steps 100 \
    --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
    --padding_side right \
    --dataset_min_length 8192 \
    --dataset_max_length 12228 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --enable_gist True \
    --gist_type interleave-16 \
    --gist_mode 512,768,1024-16 \
    --output_dir $OUTPUT_DIR/sft-16x \
    --logging_dir ./logs/qwen3-4b-inst/sft-16x \
    --logging_steps 1 \
    --deepspeed ./configs/ds_config.json \
    --do_train True \
    --eval_strategy steps \
    --eval_steps 20 \
    --only_train_gist True \
    --train_data Yukang/LongAlpaca-16k-length \
    --bf16 True \
    --save_strategy steps \
    --save_steps 1000
    # --gradient_checkpointing True
    # --device_map auto \