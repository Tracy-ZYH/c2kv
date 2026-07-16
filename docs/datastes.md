# Agent Tool-Calling Datasets

本文档汇总当前项目中三个 agent / tool-calling 数据集的基本情况、数据格式、规模，以及它们对 Tool-C2KV 实验的适用性。

本地数据目录：

```text
C2KV/datasets/
├── agent-llm-traces
├── toolathlon
└── hermes-agent-reasoning-traces
```

> 注：本地 `exp_env` conda 环境已安装 `pyarrow` 和 `datasets`，可以读取 parquet。`toolathlon` 已经用 JSONL 实际读取并统计；`hermes-agent-reasoning-traces` 已用 `pyarrow` 验证 parquet 行数与字段；`agent-llm-traces` 的字段说明主要来自 README 与项目中已有解析代码。

## 1. 总览对比

| 数据集 | 本地格式 | 规模 | 主要字段 | 工具定义位置 | 真实工具调用位置 | 当前用途 |
|---|---|---:|---|---|---|---|
| `agent-llm-traces` | Parquet, 39 shards | 1,781 traces | OpenTelemetry span | `attributes.gen_ai.tool.definitions` | `attributes.gen_ai.output.messages` | 已用于训练第一版 Tool-C2KV |
| `toolathlon` | JSONL, 66 files | 7,116 trajectories, 108 tasks | trajectory record | top-level `tool_calls.tools` | `messages[].tool_calls` | 用于跨数据集泛化、长工具集评测 |
| `hermes-agent-reasoning-traces` | Parquet, 4 files | 14,701 samples | ShareGPT conversation | top-level `tools` string | `conversations` 中的 `<tool_call>` | 可用于多轮 tool-call SFT / reasoning trace 训练 |

## 2. agent-llm-traces

本地路径：

```text
C2KV/datasets/agent-llm-traces
```

文件情况：

| 项 | 数值 |
|---|---:|
| parquet 文件数 | 39 |
| 本地 parquet 总大小 | 983,592,848 bytes |
| README 标注 traces | 1,781 |
| benchmarks | 6 |
| agent frameworks | 5 |
| models | 5 |

README 中的 benchmark 分布：

| Benchmark | Traces | Workload |
|---|---:|---|
| appworld | 406 | Personal assistant |
| browsecompplus | 133 | Deep research |
| swebench | 391 | Software engineering |
| tau2_airline | 196 | Customer service |
| tau2_retail | 469 | Customer service |
| tau2_telecom | 186 | Technical support |

### 数据格式

该数据集是 OpenTelemetry traces。每个 trace 中包含多个 span，每个 span 记录一次 LLM 或工具相关操作。

典型 span 字段：

```json
{
  "trace_id": "...",
  "span_id": "...",
  "parent_span_id": "...",
  "name": "...",
  "start_time": "...",
  "end_time": "...",
  "attributes": {
    "gen_ai.input.messages": "...",
    "gen_ai.output.messages": "...",
    "gen_ai.tool.definitions": "...",
    "gen_ai.usage.input_tokens": 123,
    "gen_ai.usage.output_tokens": 45
  },
  "status": {
    "code": 1,
    "message": ""
  }
}
```

项目中的构造方式：

- 按 `trace_id / session_id` 聚合 session。
- 从 span 的 `attributes.gen_ai.tool.definitions` 读取工具定义。
- 从 `attributes.gen_ai.input.messages` 读取当前模型输入。
- 从 `attributes.gen_ai.output.messages` 提取 assistant 回复和 tool call。

### 当前发现的问题

对该数据集做 generalization diagnosis 后发现，原来的 `session_disjoint` split 存在明显工具泄漏：

| Split | Train sessions | Eval sessions | Train toolsets | Eval toolsets | Toolset overlap | Tool-name overlap rate | Namespace overlap rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| session_disjoint | 1096 | 122 | 98 | 18 | 16 | 1.0000 | 1.0000 |
| toolset_disjoint | 950 | 268 | 90 | 10 | 0 | 0.6307 | 1.0000 |
| namespace_disjoint_proxy | 994 | 224 | 96 | 4 | 0 | 0.5983 | 0.9091 |

这说明第一版 Tool-C2KV 在该数据集上的结果更接近 IID toolset 表现，不能直接证明通用工具定义压缩能力。

## 3. Toolathlon

本地路径：

```text
C2KV/datasets/toolathlon
```

实际读取统计：

| 项 | 数值 |
|---|---:|
| JSONL 文件数 | 66 |
| 轨迹数 | 7,116 |
| 任务数 | 108 |
| 成功轨迹 `evaluation=true` | 1,634 |
| 失败轨迹 `evaluation=false` | 5,228 |
| 状态未知/未完成 | 254 |

每条 trajectory 的基本统计：

| 指标 | min | avg | p50 | p95 | max |
|---|---:|---:|---:|---:|---:|
| messages | 0 | 47.99 | 38 | 126 | 779 |
| actual tool calls | 0 | 25.44 | 19 | 67 | 751 |
| available tools | 0 | 69.26 | 67 | 122 | 147 |
| tool definition tokens | 1 | 10,534.32 | 9,912 | 20,121 | 21,689 |
| full trajectory tokens | 1 | 45,848.65 | 27,413 | 140,045 | 5,144,476 |
| observation tokens | 1 | 40,248.40 | 21,068 | 131,847 | 5,118,054 |

> 上表 token 是本地粗略 `char / 4` 估算；精确统计应在服务器上用目标 tokenizer 重新计算。

### 数据格式

每行 JSONL 是一条完整任务轨迹，顶层字段包括：

```json
{
  "modelname_run": "...",
  "task_name": "...",
  "task_status": "...",
  "config": "...",
  "request_id": "...",
  "initial_run_time": "...",
  "completion_time": "...",
  "tool_calls": "...",
  "messages": "...",
  "key_stats": "...",
  "agent_cost": "..."
}
```

需要注意：大部分字段本身是 JSON 字符串，需要再次 `json.loads`。

关键字段含义：

- `tool_calls`：可用工具定义集合，不是真实调用轨迹。
- `tool_calls.tools`：工具 schema 列表。
- `messages`：完整执行轨迹。
- `messages[].tool_calls`：assistant 发出的真实工具调用。
- `role="tool"` 的 message：工具 observation。
- `config.task_str`：任务描述。
- `config.needed_mcp_servers`：任务需要的 MCP servers。
- `task_status.evaluation`：任务是否成功。

### MCP server 分布

出现最多的 MCP server：

| MCP server | Trajectories |
|---|---:|
| filesystem | 5,563 |
| terminal | 3,097 |
| emails | 1,560 |
| fetch | 1,551 |
| playwright_with_chunk | 1,540 |
| excel | 1,497 |
| pdf-tools | 1,305 |
| google_sheet | 707 |
| yahoo-finance | 642 |
| memory | 590 |

### 对 Tool-C2KV 的意义

Toolathlon 更适合评测跨任务、长工具集、长轨迹下的工具选择能力。

当前发现：

- 工具集合明显更大。
- 工具定义更长。
- 目标工具常位于工具列表后部。
- 简单 fixed-token context 很容易看不到目标工具。
- agent-traces-trained C2KV 迁移到 Toolathlon 后表现很差，说明第一版模型没有学到足够通用的工具定义压缩能力。

## 4. Hermes Agent Reasoning Traces

本地路径：

```text
C2KV/datasets/hermes-agent-reasoning-traces
```

文件情况：

| 文件 | 大小 bytes | 说明 |
|---|---:|---|
| `data/kimi/train.parquet` | 508,262,558 | Kimi-K2.5 traces |
| `data/glm-5.1/train.parquet` | 599,571,360 | GLM-5.1-FP8 traces |
| `data/train-00000-of-00002.parquet` | 304,227,899 | 本地额外 train shard |
| `data/train-00001-of-00002.parquet` | 204,036,725 | 本地额外 train shard |

README 标注配置，并已用 `exp_env + pyarrow` 验证：

| Config | Source model | Samples |
|---|---|---:|
| `kimi` | Moonshot AI Kimi-K2.5 | 7,646 |
| `glm-5.1` | ZhipuAI GLM-5.1-FP8 | 7,055 |
| total | - | 14,701 |

本地还存在默认 train 分片：

| File | Rows | Columns |
|---|---:|---|
| `data/train-00000-of-00002.parquet` | 3,823 | `id`, `conversations`, `tools`, `category`, `subcategory`, `task` |
| `data/train-00001-of-00002.parquet` | 3,823 | `id`, `conversations`, `tools`, `category`, `subcategory`, `task` |

这两个分片合计 7,646 行，行数与 `data/kimi/train.parquet` 一致，推测是默认 config 的 Kimi 分片副本；做统计或训练时应避免和 `data/kimi/train.parquet` 重复计数。

官方统计：

| Metric | kimi | glm-5.1 |
|---|---:|---:|
| samples | 7,646 | 7,055 |
| total turns | 185,798 | 134,918 |
| total tool calls | 106,222 | 68,328 |
| avg turns/sample | 24.3 | 19.1 |
| avg tool calls/sample | 13.9 | 9.7 |
| avg think depth words | 414 | 70 |

本地用 `exp_env + pyarrow` 实测的工具定义粗略 token 统计（`len(text) / 4` 估算）：

| Config | Samples | Avg tool tokens | P50 | P95 | Max | Avg tools/sample | Avg turns/sample | Avg tool calls/sample |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| kimi | 7,646 | 2,577.61 | 2,177 | 6,219 | 6,219 | 7.96 | 24.34 | 15.89 |
| glm-5.1 | 7,055 | 3,992.73 | 2,254 | 6,290 | 6,290 | 10.59 | 19.12 | 11.69 |
| combined | 14,701 | 3,256.73 | 2,254 | 6,290 | 6,290 | 9.23 | 21.84 | 13.87 |

统计文件：

```text
C2KV/outputs/hermes_agent_reasoning_stats.json
```

对应脚本：

```text
C2KV/agent/inspect_hermes_dataset.py
```

### 数据格式

本地 parquet schema：

| Field | Type | Description |
|---|---|---|
| `id` | string | UUID identifier |
| `conversations` | list | ShareGPT multi-turn dialogue |
| `tools` | string | JSON tool definitions available to the agent |
| `category` | string | high-level task category |
| `subcategory` | string | fine-grained task type |
| `task` | string | task description |

Conversation 使用 ShareGPT 格式：

```json
{
  "from": "system|human|gpt|tool",
  "value": "..."
}
```

消息内容中包含：

- `<think>`：模型推理过程。
- `<tool_call>`：函数/工具调用。
- `<tool_response>`：真实工具执行结果。

### 类别分布

README 标注共有 9 类任务：

| Category | kimi | glm-5.1 |
|---|---:|---:|
| Terminal & Coding | 2,010 | 2,237 |
| Agent Tools | 1,474 | 2,775 |
| Repository Tasks | 1,109 | 1,022 |
| Browser Automation | 1,048 | 639 |
| Multi-Tool | 807 | 52 |
| File Operations | 757 | 134 |
| Scheduling | 204 | 104 |
| Planning & Organization | 201 | 92 |
| Conversational | 36 | 0 |

### 对 Tool-C2KV 的意义

Hermes 更接近 ShareGPT/SFT 格式，天然适合构造多轮监督训练样本：

```text
system + tools + task/history -> next assistant message / next tool call
```

它与 Toolathlon 的区别：

- Hermes 直接给出 `tools` 字段和 ShareGPT conversation，格式更接近常见 SFT。
- Toolathlon 有独立执行式任务评测和完整 task status，更适合做任务级成功率/工具调用泛化评测。
- Hermes 有 `<think>`，可以研究 reasoning trace 对工具调用的影响，但训练时要考虑是否保留 chain-of-thought。

## 5. 三个数据集的实验定位

| 数据集 | 优点 | 风险 | 推荐用途 |
|---|---|---|---|
| `agent-llm-traces` | OpenTelemetry span 细，包含工具定义、输入输出、token usage、模型信息 | 原 session split 存在工具集泄漏；数据来源复杂，span 解析成本高 | 初步训练、span-level tool definition compression |
| `toolathlon` | 长时程任务、真实执行、任务状态明确、工具集合大 | 工具定义极长；直接 full baseline 容易 OOM；需要 router/候选工具选择 | 跨数据集泛化、long toolset compression、router + C2KV |
| `hermes-agent-reasoning-traces` | ShareGPT 格式清晰，含 reasoning/tool_response，样本量 14.7k | 本机尚未逐行解析 parquet；CoT 是否训练需谨慎 | 多轮 SFT、next tool-call prediction、reasoning trace ablation |

## 6. 后续建议

1. 对 `agent-llm-traces` 重建 split：
   - Session-disjoint
   - Toolset-disjoint
   - Namespace / tool-name-disjoint

2. 对 Toolathlon 引入候选工具路由：
   - `query -> top-k tools`
   - top-k full
   - rest C2KV 或直接忽略

3. 对 Hermes 做 parquet 逐行读取脚本：
   - 统计 `tools` token 长度。
   - 统计 `<tool_call>` 数量。
   - 构造 `task + tools + history -> next tool_call` 样本。

4. 对 chunk 策略做对比：
   - fixed-token chunk
   - tool-level chunk
   - app/namespace-level chunk
   - field-aware chunk
