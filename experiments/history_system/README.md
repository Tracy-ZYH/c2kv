# C1 end-to-end delivery

这个目录提供独立的 C1 controller，并通过 SGLang C2KV native serving 运行官方 BFCL、τ²-bench 或 ToolSandbox。源码包含实际使用的 archive、gist/raw packing、Prefill detector、evidence-set retrieval、B0 admission 和 append/regenerate 流程。三者共用同一 controller 配置；benchmark 参数只切换 source profile 和官方 harness worker。

默认 `t02_risk` 使用随包提供的 **T02 C1 risk head**（`artifacts/c1_risk.t02_v1.json`），对应实验3的 H0/C1/R1。它使用 Prefill PCA8、草稿平均 NLL、STOP 和解析状态，只有风险严格大于 `0.5` 才恢复排名第一的合法单候选。模型已经训练，本次交付不重新训练；模型文件 hash、训练来源和冻结配置见相邻 provenance 文件。

更新后默认启动无需新增参数；旧命令若显式写了 `--detector legacy_prefill`，删去该参数即可使用新默认。若本机保存的是同一 C1000 checkpoint 的另一份副本，入口会自动核对权重、tokenizer 和配置并适配路径，不需要手工修改 detector 文件。

`--detector legacy_prefill` 显式使用旧 D3 Prefill head，保留原阈值和候选框架。`--selector-artifact PATH` 可以覆盖新 risk head；缺失或不兼容的 artifact 报错，不自动换回旧 detector。

每次 decision 的流程：

1. 根据已观察到的历史构建当前 raw/gist 工作区，生成尚未提交的 draft。
2. 对已观察到的 archive 做多路检索，最多检索 24 个单元，检查源文本、去重和 B0 准入，保留最多 8 个候选。
3. 有合法候选时，从实际 Prefill hidden state 和草稿特征计算风险；无合法候选时直接保留 draft。
4. detector 触发且存在合法证据时追加一个候选，再生成一次并提交最终动作；否则提交 draft。
5. 证据按 `next_decision` 生命周期释放。工具只执行最终提交的动作。

ratio、B0、任务 generation/extraction 限额来自 `configs/current_algorithm.json` 和 `runtime/configs/`。H0、C1000、ratio8、R1 与评测源保持一致；没有启用 H1、C5 proposal、C4、256-token fallback 或额外恢复预留。兼容版沿用旧 head 的阈值，不重新拟合。新 T02 的风险标签与旧 head 的训练目标不同，比较结果时需保留 detector 版本。

## Dependencies

- Python 环境需提供 `torch`、`transformers`、`numpy`、`safetensors`、`requests`；NPU 另需可用的 `torch_npu` 和 Ascend 环境。
- SGLang checkout 必须实现 `POST /v1/c2kv/native_generate` 和相应 `/model_info` capability。普通 OpenAI `/v1/chat/completions` 接口不足以承载 gist KV 与 Prefill feature。
- 当前 multi-benchmark native-serving 基线要求 `Tracy-ZYH/kvoffload-sglang-c2kv` 的 PR #5（包含 `7e55632`）或更新版本。
- 使用选定的 C1000 checkpoint 和 `Qwen3-Embedding-0.6B` 本地目录。大模型权重另行提供；小型 C1 detector JSON 已随源码提供。
- `--benchmark-dir` 指向含 `bfcl_eval/` 的 BFCL checkout；`--bfcl-python` 指向它的依赖环境。wrapper 会核对实际导入的源码路径。

## Run on NPU

先加载该机器的 Ascend 环境，并设置自己的路径。以下两个进程可以共用一张空闲 NPU；运行前检查占用。

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
export HCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost

export SGLANG_ROOT=/path/to/compatible/sglang
export SGLANG_PYTHON=/path/to/sglang/env/bin/python
export CHECKPOINT=/path/to/checkpoint-1000
export EMBEDDING=/path/to/Qwen3-Embedding-0.6B
export BFCL_ROOT=/path/to/bfcl-c2kv
export BFCL_PYTHON=/path/to/bfcl/env/bin/python
```

启动 engine：

```bash
PYTHONPATH="$SGLANG_ROOT/python" "$SGLANG_PYTHON" -m sglang.launch_server \
  --model-path "$CHECKPOINT" --served-model-name c1_t02_risk \
  --model-impl sglang --device npu --attention-backend ascend --dtype bfloat16 \
  --enable-c2kv --c2kv-gist-type dynamic-interleave --c2kv-gist-param qkv \
  --c2kv-query-proj base --c2kv-pool-fraction 0.05 \
  --c2kv-shadow-feature-layer -2 --enable-return-hidden-states \
  --mem-fraction-static 0.55 --max-total-tokens 65536 --context-length 131072 \
  --max-running-requests 1 --page-size 128 --chunked-prefill-size 256 \
  --disable-radix-cache --disable-cuda-graph --host 127.0.0.1 --port 38800
```

等 engine 打印 ready 后，先确认 native C1 capability；`run_c1.py` 的非 preview
运行也会执行同一类只读预检，并在创建输出目录前拒绝未启动、checkpoint 不匹配
或缺少 native-packed/Prefill feature 能力的后端。

```bash
curl -fsS --noproxy '*' http://127.0.0.1:38800/model_info | "$SGLANG_PYTHON" -m json.tool
```

在另一终端加载相同环境，从仓库根目录运行完整 BFCL task loop 和官方 scoring：

```bash
"$SGLANG_PYTHON" experiments/history_system/run_c1.py \
  --checkpoint "$CHECKPOINT" --embedding-model "$EMBEDDING" --embedding-device npu:0 \
  --sglang-backend-url http://127.0.0.1:38800 \
  --benchmark-dir "$BFCL_ROOT" --bfcl-python "$BFCL_PYTHON" \
  --task-id multi_turn_base_0 \
  --out outputs/c1_bfcl_base0
```

重复 `--task-id` 可以顺序运行多个 base/long-context task。每次使用新的输出目录；程序不会覆盖旧结果或自动重跑。加 `--preview` 只验证配置并打印 profile，不调用模型。

τ² 使用 `--benchmark tau2 --tau2-task-id ID`，ToolSandbox 使用
`--benchmark toolsandbox --ts-scenario NAME`。这两类普通 OpenAI harness 使用
`openai-single-task-v1`，每个 controller 只绑定一个 task/scenario；多个 ID 会顺序运行多个全新 controller。必须显式提供 `--user-base-url` 指向 raw Full engine，且它不能是 C1 controller 地址，从而只压缩 agent、不压缩 user simulator。官方评分仍分别由 `tau2 evaluate-trajs` 和 `tool_sandbox` CLI 产生。

新 head 已随包提供，默认无需额外参数。需要复现旧交付版时只切换：

```bash
--detector legacy_prefill
```

同 checkpoint、8x event packing 和 generation 配置下的纯 C2KV 对照使用
`--method c2kv_only`。该模式从同一个 S0 controller 配置中只移除
`post_draft_recovery`/detector，并且不请求 shadow detector feature；不要用绑定
旧 checkpoint-1088 profile 的 portable `c2kv` arm 代替这个 C1000 对照。

## Outputs and validation

`profile.json` 保存 detector、checkpoint、controller 和任务身份；`result.json` 保存运行状态和官方成绩。每题的 `task_shards/TASK/server/` 保留 engine HTTP、模型调用、detector/恢复决策与最终成本；相邻的 `bfcl/`、`tau2/` 或 `toolsandbox/` 保留对应 native harness 输出。根目录的 `unified_summary.json/csv` 只汇总官方 score 和 C1 runtime telemetry，不重新定义 benchmark accuracy。

运行成功要求官方生成和评分完成、controller 正常结束、模型调用没有失败或悬空。题目答错可以是正常的模型结果；HTTP 错误或 actor 崩溃不能冒充正常的零分。压缩率与缓存命中保留真实测量，低压缩率或短任务没有缓存复用不等于运行失败。

交付 smoke 的成绩仅表示选定题目的功能验收，标为 `preliminary, n=1`，不作为完整 benchmark 质量成绩。

本次 T02 默认入口在 2026-09-18 的 NPU 验证见 [tracy_t02_npu_smoke.json](validation/tracy_t02_npu_smoke.json)：不传 detector/artifact 参数，BFCL `multi_turn_base_26` 完成 10 个 decision、11 次 HTTP 200，记录 5 次真实风险评分、5 次无合法候选免评分和 1 次追加再生成。官方结果为 **1/1，preliminary, n=1**，仅是功能 smoke。原入口在评分后因压缩率 `0.90175` 不大于 1 误报失败；修正验收条件后，用不变的终态证据 CPU 重评通过，原失败记录保留，未重跑任务。78 项测试及 4 项 subtests 通过；服务器现有 checkpoint 副本的自动路径适配通过，测试 NPU 已释放。

旧 `legacy_prefill` 交付版在 2026-09-17 的实机验收见 [tracy_npu_smoke.json](validation/tracy_npu_smoke.json)：BFCL base/long-context 的 5 个任务均到达官方评分终态，86 次 native 请求全部 HTTP 200，取得 83 次真实 Prefill score，发生 3 次证据追加与再生成。每次生成都通过 B0 检查，held draft 没有作为最终动作提交，模型调用无失败或悬空，task controller 正常退出。108 项 runtime 回归、30 项 SGLang contract tests、5 项实机 hidden-capture 检查通过。

这组 smoke 的官方成绩为 **0/5，preliminary, n=1**：两题因模型达到 BFCL step cap 结束，两题 execution-response mismatch，一题 instance-state mismatch。这组旧 detector 验收证明当时链路可运行，不能代替本次 T02 默认入口的验收。测试 engine 已关闭。

历史完整任务证据见 [t02_c1_evaluation.json](validation/t02_c1_evaluation.json)：H0/C1/R1 在固定 D128 上为 **23/128，preliminary, n=1**，其中 122 题正常完成、6 题经审计为容量失败。同轮 D20 为 **6/20，preliminary, n=1**；没有旧交付版 detector 的匹配 D128 对照，因此不声称新 detector 优于旧版。这里的历史成绩与本次交付功能验证分别保留。
