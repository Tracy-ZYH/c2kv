export PYTHONPATH=`pwd`/python:$PYTHONPATH
export OUTPUT_DIR=/home/admin/workspace/aop_lab/app_data/checkpoints/qwen3-4b-musique/
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=64 torchrun --nproc_per_node 8 -m train.stage3 \
    --num_train_epochs 5 \
    --warmup_steps 100 \
    --model_name_or_path /home/admin/workspace/aop_lab/app_data/checkpoints/qwen3-4b-inst/compress-16x/checkpoint-4096 \
    --padding_side right \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --enable_gist True \
    --gist_param qkv \
    --gist_type interleave-16 \
    --output_dir $OUTPUT_DIR/compressed_sft-16x \
    --logging_dir ./logs/qwen3-4b-musique/compressed_sft-16x \
    --logging_steps 1 \
    --deepspeed ./configs/ds_config.json \
    --do_train True \
    --eval_strategy steps \
    --eval_steps 20 \
    --only_train_gist True \
    --train_data ../datasets/musique_ans_v1.0_train.jsonl \
    --bf16 True \
    --save_strategy steps \
    --save_steps 500 \
    --dataset_shuffle_seed 2948
    # --gradient_checkpointing True
    # --device_map auto \