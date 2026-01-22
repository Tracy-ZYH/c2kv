export PYTHONPATH=`pwd`/python:$PYTHONPATH
export OUTPUT_DIR=/home/admin/workspace/aop_lab/app_data/checkpoints/qwen3-4b-inst/
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=64 torchrun --nproc_per_node 8 -m train.train_sft \
    --num_train_epochs 5 \
    --warmup_steps 100 \
    --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
    --padding_side right \
    --dataset_min_length 6144 \
    --dataset_max_length 16384 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --enable_gist True \
    --gist_param QkV \
    --gist_type interleave-16 \
    --gist_mode 512,768,1024,1280-24 \
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
    --save_strategy epoch \
    --dataset_shuffle_seed 2837
    # --gradient_checkpointing True
    # --device_map auto \