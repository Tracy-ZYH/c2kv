# Agent evaluation

This directory is independent of the multi-document evaluation stack.
`AgentDataset` examples contain:

- `system_prompt` (falls back to `You are a helpful assistant.`);
- `tools` in Hugging Face function-tool format;
- `messages` without the leading system message;
- expected tool calls for evaluation.

The full-compute path calls `apply_chat_template(messages, tools=tools)` once,
so Qwen3 renders the system prompt and tool definitions in one system message.

The C2KV path renders three independent pieces:

1. the system prompt as a normal system message;
2. tool definitions as another system message, produced by passing only
   `tools=` to the chat template;
3. the remaining conversation and assistant generation prompt.

The rendered system/tool segments have a small LRU cache. Identical definitions
can therefore reuse their prefilled/extracted KV across examples.

Use `--max-tool-tokens N` to skip examples whose independently rendered tool
system message is longer than `N` tokens. `--max-examples` is applied after
this filtering.

```bash
export PYTHONPATH=$PWD/python

python python/agent/expr_agent_full.py \
  --model <base-or-checkpoint-path> \
  --dataset-path /path/to/agent-llm-traces \
  --output-file results/agent_full.jsonl

python python/agent/expr_agent_c2kv.py \
  --model <c2kv-checkpoint-path> \
  --dataset-path /path/to/agent-llm-traces \
  --override-ratio 4 \
  --reuse-cache-size 1 \
  --output-file results/agent_c2kv.jsonl
```
