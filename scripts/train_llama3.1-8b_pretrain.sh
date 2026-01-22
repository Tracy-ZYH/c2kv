export PYTHONPATH=`pwd`/python:$PYTHONPATH
export OUTPUT_DIR=/home/admin/workspace/aop_lab/app_data/checkpoints/llama3.1-8b-inst/
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=64 torchrun --nproc_per_node 8 -m train.train_pretrain \
    --num_train_epochs 4 \
    --warmup_steps 512 \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --padding_side right \
    --dataset_min_length 8192 \
    --dataset_max_length 12288 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --enable_gist True \
    --gist_param qkv \
    --gist_type interleave-16 \
    --gist_mode 512,1024-30 \
    --gist_self_distill_coef 0.8 \
    --gist_self_distill_temperature 1.0 \
    --output_dir $OUTPUT_DIR/distill0.8-16x-pretrain \
    --logging_dir ./logs/llama3.1-8b-inst/distill0.8-16x-pretrain \
    --logging_steps 1 \
    --deepspeed ./configs/ds_config.json \
    --do_train True \
    --eval_strategy steps \
    --eval_steps 64 \
    --only_train_gist True \
    --dataloader_num_workers 8 \
    --dataloader_prefetch_factor 4 \
    --train_data /mnt/nas1/alsc_supply_tech_SlimPajama-627B_20240926201127 \
    --bf16 True \
    --save_strategy epoch \
    --dataset_shuffle_seed 3846 
    # --gradient_checkpointing True
    # --device_map auto \