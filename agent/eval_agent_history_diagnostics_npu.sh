#!/usr/bin/env bash
set -euo pipefail

# Diagnostic baselines for history compression:
# 1. original_replay_full: original system/tools/messages rendered as one prompt.
# 2. reconstructed_contiguous_full: current history_docs + current_messages rendered as one prompt.
# 3. split_full_kv: same history docs, independently full-prefilled then KV-concatenated.
# 4. sequential_full_kv: same history docs, incrementally full-prefilled with past KV.
# 5. contiguous_history_c2kv: same history docs, concatenated as one C2KV document.
# 6. split_c2kv: same history docs, independently C2KV-compressed.
# 7. current_only: system/tools + current messages only.
# Extra:
# - recent1_hybrid / recent2_hybrid: latest N history docs full, older docs C2KV.
# - tail_truncate: tail history kept full, older docs dropped.

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}"
export MODEL_PATH="${MODEL_PATH:-./checkpoints/qwen3-4b-agent-history-c2kv-toolcall-npu-v2/checkpoint-1088}"
export BASE_MODEL="${BASE_MODEL:-./models/Qwen3-4B-Instruct-2507}"
export HISTORY_TOKENIZER_PATH="${HISTORY_TOKENIZER_PATH:-${BASE_MODEL}}"
export OUTPUT_FILE="${OUTPUT_FILE:-./outputs/diagnostic_0803_history_replay_split_c2kv.jsonl}"
export COMPARE_MODES="${COMPARE_MODES:-original_replay_full,reconstructed_contiguous_full,sequential_full_kv,split_full_kv,contiguous_history_c2kv,current_only,tail_truncate,split_c2kv,recent1_hybrid,recent2_hybrid}"
export RATIOS="${RATIOS:-4}"
export SPLIT_NAME="${SPLIT_NAME:-subset_disjoint}"
export REQUIRE_TOOL_CALL="${REQUIRE_TOOL_CALL:-False}"
export INCLUDE_TOOLS="${INCLUDE_TOOLS:-True}"
export HISTORY_MAX_DOC_LENGTH="${HISTORY_MAX_DOC_LENGTH:-768}"
export HISTORY_MAX_DOC_NUM="${HISTORY_MAX_DOC_NUM:-16}"
export MAX_HISTORY_TOKENS="${MAX_HISTORY_TOKENS:-12288}"
export MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-0}"
export MAX_BASELINE_INPUT_TOKENS="${MAX_BASELINE_INPUT_TOKENS:-16000}"
export HISTORY_SELECTION="${HISTORY_SELECTION:-tail}"
export TRUNCATE_SELECTION="${TRUNCATE_SELECTION:-tail}"
export NPU_ATTN_IMPL="${NPU_ATTN_IMPL:-eager}"
export PARALLEL_EVAL="${PARALLEL_EVAL:-True}"

bash agent/eval_agent_history_c2kv_npu.sh
