export PYTHONPATH=`pwd`/python:$PYTHONPATH
export OUTPUT_DIR=/home/admin/workspace/aop_lab/app_data/checkpoints/qwen3-4b-musique/
OMP_NUM_THREADS=64 torchrun --nproc_per_node 4 -m train.stage3 \
    --num_train_epochs 5 \
    --warmup_steps 100 \
    --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
    --padding_side right \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --enable_gist True \
    --gist_type interleave-16 \
    --output_dir $OUTPUT_DIR/sft-16x \
    --logging_dir ./logs/qwen3-4b-musique/sft-16x \
    --logging_steps 1 \
    --deepspeed ./configs/ds_config.json \
    --do_train True \
    --eval_strategy steps \
    --eval_steps 50 \
    --only_train_gist True \
    --train_data ../datasets/musique_ans_v1.0_train.jsonl \
    --bf16 True \
    --save_strategy steps \
    --save_steps 1000
    # --gradient_checkpointing True
    # --device_map auto \