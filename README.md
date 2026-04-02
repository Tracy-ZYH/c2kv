## C2KV: Concatenable and Compressible KV Cache

C2KV enables **independently prefilled text segments to be compressed into compact KV Cache representations** that can be correctly concatenated and recognized by downstream tasks. The core idea is to freeze the original model weights, introduce additional QKV Projection layers, and use C2KV as Memory Slots. Each text chunk is independently compressed into C2KV entries with properly repositioned RoPE (Rotary Position Embedding), so that multiple C2KV caches can be seamlessly stitched together for downstream inference.

### Supported Models

| Architecture | Example Model |
|---|---|
| Llama | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| Qwen2.5 | `Qwen/Qwen2.5-7B-Instruct` |
| Qwen3 | `Qwen/Qwen3-4B-Instruct-2507`, `Qwen/Qwen3-14B` |

### Code Organization

```
├── python/
│   ├── gist_args.py                  # Dataclass definitions for ModelArgs & TrainingArgs
│   ├── models/
│   │   ├── __init__.py               # Public API: blend_gist_key_values, get_model_class, etc.
│   │   ├── model_utils.py            # Model/tokenizer loading, architecture dispatch, regularization loss
│   │   ├── gist_utils.py             # C2KV core: attention mask construction, RoPE repositioning, KV blending
│   │   ├── lora_utils.py             # LoRA utilities
│   │   ├── llama                     # Llama model classes
│   │   ├── qwen2_5                   # Qwen 2.5 model classes
│   │   └── qwen3                     # Qwen 3 model classes
│   ├── train/
│   │   ├── configs.py                # QA query prompt templates & continuation prompts
│   │   ├── train_data.py             # Dataset loading & tokenization (GistDataset, PretrainDataset, MDQADataset)
│   │   ├── train_mdoc.py             # Multi-document QA fine-tuning entry point
│   │   ├── train_pretrain.py         # Continual pre-training entry point
│   │   ├── train_sft.py              # SFT training entry point
│   │   ├── trainer.py                # Custom Trainer classes (GistPretrainTrainer, GistMultiDocTrainer)
│   │   └── clean_data.py            # Data cleaning via LLM-based answer verification
│   └── inference/
│       ├── expr_c2kv.py              # **Main C2KV evaluation script**
│       ├── expr_fullcompute.py       # Full-compute baseline evaluation
│       ├── expr_reuse.py             # KV cache reuse baseline (+ EPIC variants)
│       ├── expr_blockattention.py    # Block Attention baseline evaluation
│       ├── expr_cacheblend.py        # CacheBlend evaluation
│       ├── reuse_pipeline.py         # KV cache reuse pipeline utilities
│       ├── rope_reposition.py        # RoPE repositioning for concatenated KV caches
│       ├── mdocdataset.py            # Dataset loaders (HotpotQA, MuSiQue, WikiMQA, MultiNews, SAMSum, etc.)
│       ├── longbench_metrics.py      # F1 / ROUGE evaluation metrics, copied from LongBench
│       ├── expr_timer.py             # Profiling & timing utilities
│       └── serve_gistmodel.py        # Model serving utilities
├── scripts/
│   ├── train_qwen3-4b-mixed_mdoc.sh  # Train Qwen3-4B on mixed multi-doc QA data
│   ├── train_qwen3-14b-mixed_mdoc.sh # Train Qwen3-14B on mixed multi-doc QA data
│   ├── train_qwen2.5-7b-mixed_mdoc.sh# Train Qwen2.5-7B on mixed multi-doc QA data
│   ├── train_llama3.1-8b-mixed_mdoc.sh# Train Llama-3.1-8B on mixed multi-doc QA data
│   ├── generate_commands.py          # Generate batch evaluation commands
│   └── execute_parallel.py           # Parallel command execution
├── configs/
│   ├── ds_config.json                # DeepSpeed ZeRO-3 config (CPU offload)
│   └── ds2moe_config.json            # DeepSpeed config for MoE models
├── requirements.txt
└── kvcache_inspect.ipynb             # Interactive notebook for KV cache inspection
```

### Installation

```bash
pip install -r requirements.txt
```

**Key dependencies**: `transformers==4.57.1`, `torch==2.9.0`, `deepspeed==0.18.1`, `flash-attn` (required for `flash_attention_2` backend).

> **Important**: With `transformers==4.57.1`, the `model.generate()` API has a known bug with custom `position_ids`. See [huggingface/transformers#36510](https://github.com/huggingface/transformers/issues/36510) for the fix.

### Quick Start

The pipeline consists of three stages: **Data Preparation → Training → Evaluation**.

#### Step 1: Prepare Data

**Training datasets** (download or use HuggingFace datasets):

| Dataset | Source | Usage |
|---|---|---|
| LongMagpie | `Seerkfang/LongMagpie_multidoc_longcontext_dataset` | Multi-doc long-context QA |
| HotpotQA | [hotpotqa.github.io](https://hotpotqa.github.io) | Multi-hop QA |
| MuSiQue [Optional] | `dgslibisey/MuSiQue` | Multi-step QA |
| 2WikiMultihopQA | `xanhho/2WikiMultihopQA` | Multi-hop QA |

Organize the training data under a single directory (e.g., `./datasets/`):

```
datasets/
├── hotpotqa_train.jsonl          # HotpotQA training split
├── musique_ans_v1.0_train.jsonl  # MuSiQue training split
└── longmagpie_1024/              # LongMagpie processed (HF dataset on disk)
```

**[Optional] Data Cleaning**: Use a strong LLM to verify answer quality and filter low-quality samples:

```bash
export PYTHONPATH=$(pwd)/python:$PYTHONPATH

python -m train.clean_data \
    --input_path ./datasets/hotpotqa_train.jsonl \
    --output_path ./datasets_cleaned/hotpotqa_train_cleaned \
    --dataset_type hotpotqa \
    --api_base http://localhost:8000/v1 \
    --model Qwen/Qwen3-235B-A22B-Instruct-2507 \
    --max_concurrent 8 \
    --f1_threshold 0.8 \
    --max_tokens 32
```

After cleaning, the training script expects the following layout under `--train_data`:

```
<train_data>/
├── longmagpie_1024/              # LongMagpie processed data
<train_data>_cleaned/
├── hotpotqa_train_cleaned/       # Cleaned HotpotQA
└── wikimqa_train_cleaned/        # Cleaned 2WikiMultihopQA
```

#### Step 2: Train

All training scripts set `PYTHONPATH` and launch distributed training via `torchrun`. Key arguments:

| Argument | Description |
|---|---|
| `--model_name_or_path` | Base model (HuggingFace ID or local path) |
| `--enable_gist True` | Enable C2KV gist projections |
| `--gist_type <gist_type>` | C2KV compression strategy |
| `--only_train_gist True` | Freeze base model, only train C2KV parameters |
| `--train_data <path>` | Root directory of training datasets |
| `--output_dir <path>` | Checkpoint output directory |
| `--deepspeed ./configs/ds_config.json` | DeepSpeed ZeRO-3 config |

**Example** (Qwen3-4B, 8 GPUs):

```bash
export PYTHONPATH=$(pwd)/python:$PYTHONPATH
export OUTPUT_DIR=./checkpoints/qwen3-4b/

HF_HUB_OFFLINE=1 torchrun --nproc_per_node 8 -m train.train_mdoc \
    --num_train_epochs 2 \
    --warmup_ratio 0.06 \
    --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
    --padding_side right \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --weight_decay 0.1 \
    --enable_gist True \
    --gist_param qkv \
    --gist_type dynamic-interleave \
    --output_dir $OUTPUT_DIR \
    --deepspeed ./configs/ds_config.json \
    --do_train True \
    --eval_strategy steps \
    --eval_steps 200 \
    --only_train_gist True \
    --train_data ./datasets \
    --bf16 True \
    --save_strategy steps \
    --save_steps 2000
```

See `scripts/train_*_mdoc.sh` for per-model configurations.

#### Step 3: Evaluate

**Evaluation dataset**: [LongBench](https://huggingface.co/datasets/zai-org/LongBench) (auto-downloaded via HuggingFace `datasets`).

Supported evaluation datasets: `hotpotqa`, `musique`, `wikimqa`, `samsum`, `multinews`, `needle`, `gsm8k`, `qasper`, `gov_report`, `qmsum`.

**Evaluate C2KV model**:

```bash
export PYTHONPATH=$(pwd)/python:$PYTHONPATH

python python/inference/expr_c2kv.py \
    --model <path_to_trained_checkpoint> \
    --dataset hotpotqa \
    --output_file results/hotpotqa/c2kv_result.jsonl \
    --max_examples 500
```

**Key arguments for `expr_c2kv.py`**:

| Argument | Description |
|---|---|
| `--model` | Path to trained C2KV model checkpoint |
| `--dataset` | Dataset name (`hotpotqa`, `musique`, `wikimqa`, etc.) |
| `--dataset_path` | Custom dataset path (optional, defaults to LongBench) |
| `--output_file` | Output JSONL file for predictions |
| `--max_examples` | Limit number of evaluation examples |
| `--cut_length` | Maximum document length before splitting |
| `--profile` | Enable latency profiling |
| `--override-ratio` | Override compression ratio for `dynamic-interleave` models |
| `--only_supporting` | MuSiQue only: use only supporting paragraphs |
| `--cot` | Enable chain-of-thought prompting |

**Evaluate baselines** (full-compute, naive reuse, CacheBlend, EPIC, Block Attention):

```bash
# Full compute (no compression)
python python/inference/expr_fullcompute.py \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --dataset hotpotqa \
    --output_file results/hotpotqa/qwen3-4b_fullcompute.jsonl

# KV cache reuse with SnapKV compression
python python/inference/expr_reuse.py \
    --compress snapkv \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --dataset hotpotqa \
    --output_file results/hotpotqa/qwen3-4b_reuse_snapkv.jsonl
```

See `scripts/evaluate_baselines.sh` for the full set of baseline commands.

### How C2KV Works

1. **Freeze base model**: All original model parameters are frozen during training.
2. **Add Gist QKV Projections**: Extra `gist_q_proj`, `gist_k_proj`, `gist_v_proj` layers are added to each attention layer. Only these parameters are trained.
3. **Interleaved Memory Slots**: Input tokens are divided into chunks. For each chunk, a C2KV memory slot is inserted that attends to the chunk tokens and compresses them. The compression ratio is controlled by `gist_type` (e.g., `interleave-4` = 4:1 compression, `dynamic-interleave` = adaptive, currently randomly sample from [4, 8, 16] in training, refer to `gist_utils.py`).
4. **RoPE Repositioning**: When concatenating independently prefilled C2KV caches, the Key cache RoPE positions are rotated to maintain correct positional relationships (`rope_reposition.py`).
5. **KV Blending**: The compressed C2KV entries from multiple documents are blended into a single KV cache with correct position IDs (`blend_gist_key_values` in `gist_utils.py`).

### License

Please refer to the license terms of the base models used (Llama, Qwen).
