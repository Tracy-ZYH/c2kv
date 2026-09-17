# C1 end-to-end delivery

这个目录提供独立的 C1 controller，并通过 SGLang C2KV native serving 运行官方 BFCL。源码包含实际使用的 archive、gist/raw packing、Prefill detector、evidence-set retrieval、B0 admission 和 append/regenerate 流程。

默认 `legacy_prefill` 使用已训练的旧 Prefill head，加上当前 evidence-set 候选检索和首个可准入 singleton 选择。它是 **C1 的旧 detector 兼容版**。新 T02 risk head 尚未随此版本交付；`--detector t02_risk --selector-artifact PATH` 才会启用新 C1，缺少 artifact 会报错。

每次 decision 的流程：

1. 根据已观察到的历史构建当前 raw/gist 工作区，生成尚未提交的 draft。
2. 从实际 Prefill hidden state 计算 detector score，使用 head 保存的阈值和模型绑定。
3. 对已观察到的 archive 检索候选，检查源文本、去重和 B0 准入。
4. detector 触发且存在合法证据时追加一个候选，再生成一次并提交最终动作；否则提交 draft。
5. 证据按 `next_decision` 生命周期释放。工具只执行最终提交的动作。

ratio、B0、任务 generation/extraction 限额来自 `configs/current_algorithm.json` 和 `runtime/configs/`。兼容版沿用旧 head 的阈值，不重新拟合。新 T02 的风险标签与旧 head 的训练目标不同，比较结果时需保留 detector 版本。

## Dependencies

- Python 环境需提供 `torch`、`transformers`、`numpy`、`safetensors`、`requests`；NPU 另需可用的 `torch_npu` 和 Ascend 环境。
- SGLang checkout 必须实现 `POST /v1/c2kv/native_generate` 和相应 `/model_info` capability。普通 OpenAI `/v1/chat/completions` 接口不足以承载 gist KV 与 Prefill feature。
- 已验证的 native-serving 实现为 [66ec2db10](https://github.com/setsuna113/kvoffload-sglang-c2kv/commit/66ec2db101fe9db9ab41b1a39311444cb0ef971b)，在雨涵现有 SGLang checkout 的独立副本上测试。
- 使用选定的 C1000 checkpoint 和 `Qwen3-Embedding-0.6B` 本地目录。模型权重另行提供，不包含在源码 PR 中。
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
  --model-path "$CHECKPOINT" --served-model-name c1-c1000 \
  --model-impl sglang --device npu --attention-backend ascend --dtype bfloat16 \
  --enable-c2kv --c2kv-gist-type dynamic-interleave --c2kv-gist-param qkv \
  --c2kv-query-proj base --c2kv-pool-fraction 0.05 \
  --c2kv-shadow-feature-layer -2 --enable-return-hidden-states \
  --mem-fraction-static 0.55 --max-total-tokens 65536 --context-length 131072 \
  --max-running-requests 1 --page-size 128 --chunked-prefill-size 256 \
  --disable-radix-cache --disable-cuda-graph --host 127.0.0.1 --port 38800
```

在另一终端加载相同环境，从仓库根目录运行完整 BFCL task loop 和官方 scoring：

```bash
"$SGLANG_PYTHON" experiments/history_system/run_c1.py \
  --checkpoint "$CHECKPOINT" --embedding-model "$EMBEDDING" --embedding-device npu:0 \
  --sglang-backend-url http://127.0.0.1:38800 \
  --benchmark-dir "$BFCL_ROOT" --bfcl-python "$BFCL_PYTHON" \
  --detector legacy_prefill --task-id multi_turn_base_0 \
  --out outputs/c1_bfcl_base0
```

重复 `--task-id` 可以顺序运行多个 base/long-context task。每次使用新的输出目录；程序不会覆盖旧结果或自动重跑。加 `--preview` 只验证配置并打印 profile，不调用模型。

新 head 就绪后的入口保持相同，只替换 detector 参数：

```bash
--detector t02_risk --selector-artifact /path/to/c1_risk.json
```

## Outputs and validation

`profile.json` 保存 detector、checkpoint、controller 和任务身份；`result.json` 保存运行状态和官方成绩。每题的 `task_shards/TASK/server/` 保留 engine HTTP、模型调用、detector/恢复决策与最终成本；`task_shards/TASK/bfcl/` 保留 BFCL contract、真实源码绑定、完整轨迹和官方评分。

运行成功要求官方生成和评分完成、controller 正常结束、模型调用没有失败或悬空。题目答错可以是正常的模型结果；HTTP 错误或 actor 崩溃不能冒充正常的零分。

交付 smoke 的成绩仅表示选定题目的功能验收，标为 `preliminary, n=1`，不作为完整 benchmark 质量成绩。

2026-09-17 在 `ssh tracy` 的实机验收见 [tracy_npu_smoke.json](validation/tracy_npu_smoke.json)：BFCL base/long-context 的 5 个任务均到达官方评分终态，86 次 native 请求全部 HTTP 200，取得 83 次真实 Prefill score，发生 3 次证据追加与再生成。每次生成都通过 B0 检查，held draft 没有作为最终动作提交，模型调用无失败或悬空，task controller 正常退出。108 项 runtime 回归、30 项 SGLang contract tests、5 项实机 hidden-capture 检查通过。

这组 smoke 的官方成绩为 **0/5，preliminary, n=1**：两题因模型达到 BFCL step cap 结束，两题 execution-response mismatch，一题 instance-state mismatch。这版交付证明链路可运行，当前样本没有显示整题质量收益。测试 engine 已关闭。
