MODEL=qwen3-4b/260625-dyn-overlap64-embed_residual
WORKERS=4
CUT_LENGTH=16384
DATASETS=("wikimqa" "hotpotqa" "musique" "multinews" "triviaqa")
RATIO=2

MODEL_PATH=checkpoints/${MODEL}
OUTPUT_PATH=results/${MODEL}
mkdir -p $OUTPUT_PATH

for dataset in "${DATASETS[@]}"; do
    echo "Evaluating dataset: $dataset"
    # python python/inference/expr_c2kv_api.py \
    #     --compression-ratio $RATIO \
    #     --workers $WORKERS \
    #     --cut_length $CUT_LENGTH \
    #     --model $MODEL \
    #     --dataset $dataset \
    #     --output_file $OUTPUT_PATH/c2kv/${dataset}.jsonl
    # python python/inference/expr_c2kv_mixed_reuse_api.py \
    #     --compression-ratio $RATIO \
    #     --workers $WORKERS \
    #     --cut_length $CUT_LENGTH \
    #     --model $MODEL \
    #     --dataset $dataset \
    #     --random-trials 4 \
    #     --output_file $OUTPUT_PATH/mixed_reuse/${dataset}.jsonl
    # python python/inference/expr_fullcompute_api.py \
    #     --workers $WORKERS \
    #     --model $MODEL \
    #     --dataset $dataset \
    #     --output_file $OUTPUT_PATH/fullcompute/${dataset}.jsonl
done
