#!/bin/bash
# Generate RULER benchmark data for all tasks and context lengths
# This script uses NVIDIA's RULER toolkit to generate synthetic evaluation data

set -e

# Configuration
RULER_REPO="https://github.com/hsiehjackson/RULER.git"
RULER_DIR="./RULER_toolkit"
OUTPUT_DIR="./datasets/ruler"
TOKENIZER_PATH="${1:-Qwen/Qwen3-4B-Instruct-2507}"  # Default tokenizer, can override with arg

# Context lengths to generate (in tokens)
CONTEXT_LENGTHS=(4096 8192 16384 32768 65536)

# Tasks to generate (from synthetic.yaml)
TASKS=(
    "niah_single_1"      # NIAH with noise haystack, words->numbers
    "niah_single_2"      # NIAH with essay haystack, words->numbers
    "niah_single_3"      # NIAH with essay haystack, words->uuids
    "niah_multikey_1"    # NIAH with 4 keys
    "niah_multikey_2"    # NIAH with needle haystack
    "niah_multikey_3"    # NIAH with uuid keys and values
    "niah_multivalue"    # NIAH with 4 values per key
    "niah_multiquery"    # NIAH with 4 queries
    "vt"                 # Variable tracking
    "cwe"                # Common words extraction
    "fwe"                # Frequency words extraction
    "qa_1"               # QA with SQuAD
    "qa_2"               # QA with HotpotQA
)

echo "============================================"
echo "RULER Data Generation Script"
echo "============================================"
echo "Tokenizer: ${TOKENIZER_PATH}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Context lengths: ${CONTEXT_LENGTHS[@]}"
echo "Tasks: ${TASKS[@]}"
echo "============================================"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Clone RULER if not exists
if [ ! -d "${RULER_DIR}" ]; then
    echo "Cloning RULER repository..."
    git clone ${RULER_REPO} ${RULER_DIR}
    cd ${RULER_DIR}
    pip install -e .
    cd ..
else
    echo "RULER repository already exists at ${RULER_DIR}"
fi

# Generate data for each context length and task
for length in "${CONTEXT_LENGTHS[@]}"; do
    echo ""
    echo "============================================"
    echo "Generating data for context length: ${length}"
    echo "============================================"

    for task in "${TASKS[@]}"; do
        echo "Generating task: ${task}"

        output_file="${OUTPUT_DIR}/${task}_${length}.jsonl"

        # Skip if file already exists
        if [ -f "${output_file}" ]; then
            echo "  ✓ Already exists: ${output_file}"
            continue
        fi

        # Generate data using RULER's script
        python ${RULER_DIR}/scripts/data/prepare.py \
            --save_dir ${OUTPUT_DIR}/temp_${length} \
            --benchmark synthetic \
            --task ${task} \
            --tokenizer_path ${TOKENIZER_PATH} \
            --tokenizer_type hf \
            --max_seq_length ${length} \
            --num_samples 100 \
            --random_seed 42 \
            --model_template_type base

        # Move generated file to final location
        if [ -f "${OUTPUT_DIR}/temp_${length}/${task}/validation.jsonl" ]; then
            mv "${OUTPUT_DIR}/temp_${length}/${task}/validation.jsonl" "${output_file}"
            echo "  ✓ Generated: ${output_file}"
        elif [ -f "${OUTPUT_DIR}/temp_${length}/${task}.jsonl" ]; then
            mv "${OUTPUT_DIR}/temp_${length}/${task}.jsonl" "${output_file}"
            echo "  ✓ Generated: ${output_file}"
        else
            echo "  ✗ Failed to generate: ${task}"
            echo "  Looking for files in ${OUTPUT_DIR}/temp_${length}/"
            ls -la "${OUTPUT_DIR}/temp_${length}/" 2>/dev/null || echo "  Directory not found"
        fi

        # Clean up temp directory
        rm -rf "${OUTPUT_DIR}/temp_${length}"
    done
done

echo ""
echo "============================================"
echo "Data generation complete!"
echo "============================================"
echo "Generated files:"
ls -lh ${OUTPUT_DIR}/*.jsonl | wc -l
echo "files in ${OUTPUT_DIR}"
echo ""
