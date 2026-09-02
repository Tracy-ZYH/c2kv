# D-KV-Repair 共享包（restore-vs-sham KV 修复 · 信号层）

C2KV gist 压缩后 KV cache 的 in-place 修复实验核心。设计一句话：对"压缩压错"的触发点（C→W transition，压缩前对、压缩后错），用三个修复臂对两个对照臂，在同一失败集上测修复率与噪声地板。

本包刻意精简：只有臂实现 + eval 接入参考 + 两个数据无关工具。AppWorld 侧的 driver、冻结 manifest、行级结果、预注册全文是 harness/数据特定的，未包含（需要时可单独聊）。

## 臂定义表

| arm | harness mode | 注册 | 含义 |
|---|---|---|---|
| E-none | `c2kv` | yes | 未动的压缩前缀（下界） |
| E-sham | `d_sham_neutral` | yes | 等长中性 span 走同一注入路径（噪声地板） |
| E-corr | `d_corr` | yes | append-only erratum：doc k* 的 raw KV 追加到 full-grid gist 后（双重覆盖） |
| E-corr+re | `d_corr_recompute` | yes | docs 0..k* gist + 同一 raw slice + docs k*+1..T-1 在修正前缀上重算 |
| E-full | `full` | yes | 未压缩上界 |
| (诊断) | `d_corr_all` | **no** | 追加所有 doc 的 raw KV；只回答"append-only 通道是否活着" |
| (守卫) | `d_sham_mech` | **no** | 机械拆装重组；必须与 `c2kv` 逐 token 一致 |

两个关键设计点：

- **所有臂的 decode 位置完全一致**（会计字段里 `d_*_prefill_sec` 只是注入侧成本，不影响打分位置）。
- **single-variable claim**：`d_corr` 与 `d_corr_recompute` 的上游半边位级相同（网格行是压缩确定性的），两臂唯一变量是下游表示。

## 文件

| 文件 | 说明 |
|---|---|
| `kv_repair_arms.py` | 臂构造**参考实现**（292 行，从 `agent/eval_agent_history_c2kv.py` 原样提取 + 接口契约头）；活代码见下方 Step 3 |
| `extract_cw_triggers.py` | C→W trigger 提取（纯 stdlib，零三方依赖） |
| `d_sham_plan.py` | sham/corr 修复计划生成（纯 stdlib） |
| `d_neutral_corpus.txt` | sham 臂中性语料（sha256 绑定进 plan） |
| `d_witness_core.py` | witness 选块算法（纯 stdlib，冻结；用法见下方「Witness 选块」） |

## 接 BFCL 的三步流程

**Step 1 — 两臂电池。** 用你们的 BFCL runner 在同一 qid 集上跑 `full` / `c2kv` 两臂，产出行级 jsonl（每行至少：qid、输出文本、correct 判定；行格式约定见 `extract_cw_triggers.py` 头部 docstring）。我们 fork 的 `metrology/bfcl_hf_runner.py` 已有 c2kv 臂（含 doc_mode/ratio 参数与 untrained 对照），可作起点（获取方式见 Step 3）。

**Step 2 — 提触发点 + 生成修复计划。**

```bash
python extract_cw_triggers.py \
    --full_rows full.jsonl --compressed_rows c2kv.jsonl \
    --batch bfcl_r1 --s_metric tool_name_match \
    --out_manifest cw_manifest.json --out_doc_table d_doc_ids.json \
    --ckpt_path <ckpt> --model_sha <sha> --eval_code_sha <git_sha>

python d_sham_plan.py \
    --manifest cw_manifest.json --doc_table d_doc_ids.json \
    --corpus d_neutral_corpus.txt \
    --tokenizer <model_path> --out d_sham_plan.json
```

`--s_metric` 按 BFCL 的判定方式选择（可选值见 `--help`）；plan 自带 budget/neutrality 两道门，门不过会拒绝产出可用计划。

**Step 3 — 跑臂（活代码，无需自行移植）。** 五个臂已作为 `c2kv_d_*` 条件实现在我们的 BFCL runner 里（`metrology/d_repair_arms.py` + runner 接线，与 `c2kv` 条件共用同一 gist 生成栈；`kv_repair_arms.py` 保留为 AppWorld 侧的语义出处对照）。可跑的完整栈在 setsuna113/c2kv 的 `share/d-kv-repair` 分支（本目录之外还含 `metrology/` 与 python 侧依赖）：

```bash
git fetch https://github.com/setsuna113/c2kv.git share/d-kv-repair
git checkout FETCH_HEAD
```

```bash
python -m metrology.bfcl_hf_runner \
    --bfcl_pkg_path <bfcl_eval 包路径> --model <Qwen3-4B-Instruct-2507> \
    --condition c2kv_d_corr --c2kv_checkpoint <c2kv ckpt> \
    --c2kv_doc_mode joint --c2kv_ratio 8 \
    --d_plan d_sham_plan.json \
    --cap_tier default --output d_corr.jsonl
```

- condition 与臂名 1:1：`c2kv_d_corr` / `c2kv_d_corr_recompute` / `c2kv_d_corr_all` / `c2kv_d_sham_neutral`（必需 `--d_plan`）/ `c2kv_d_sham_mech`（守卫）。
- `--d_plan` 的 JSON 与 AppWorld per-qid plan 同形 `{qid: {k_star, span_len, sham_token_ids}}`；`d_sham_plan.py` 的产物可直接喂入（BFCL 侧 qid = entry id）。corr 系臂不需要 payload，提供时只做 k* 交叉校验。
- 一处有意的语义差异：k* 取**合并网格（工具文档 + 历史文档）中点**，AppWorld 只数历史文档——BFCL 的压缩网格本来就含工具块，中点是"被压缩文档序列正中间那份"的自然类比（`d_repair_arms.py` docstring 有完整说明）。
- 臂前提不满足（plan 缺失 / k* 不符 / sham 长度不符）会落 error 行而非静默退化，对齐 AppWorld 的 fatal skip。

**接入后先跑守卫臂**：`c2kv_d_sham_mech` 的输出必须与 `c2kv` 基线臂逐 token 一致。不一致 = 手术管线有机械损伤，先修管线，再谈任何修复率数字。

若要迁回自己的 harness 而非直接用 runner，`kv_repair_arms.py` 头部的接口契约仍是移植清单（宿主侧需要提供的接口现成 harness 里都有对应物）。

## Witness 选块（oracle 定位器，`d_witness_core.py`）

修复臂修**哪一块**历史，和怎么放 raw KV 是两个独立变量。AppWorld 单步电池上的结论（93 个 C→W 触发题，fixed_joint，ratio 8）：

| 选块策略 | 救回率（tool_name_match，单 seed，n=93） |
|---|---:|
| 中位块（旧默认） | 33/93 = 35.5% |
| **witness 块** | **71/93 = 76.3%** |
| 乱修一块（823 次对照） | 25.0% |
| 最优块天花板 | 81/93 = 87.1% |

也就是说在那个电池上，块选对了修复率从 ~35% 到 ~76%，布局（原位 / 追加 / 删 gist）只差 1–5pp。这个包把选块算法单独交出来，供 BFCL 侧复用。

**算法（prereg v2.2，冻结，`d_witness_core.py` 全文即算法）**：把目标动作拆成"工具名 + 参数叶子值"，每个值在多少个 doc 里出现记 df，每个 doc 的分数 = Σ 1/df（只算出现在该 doc 里的值），取 argmax；没有任何值出现过则返回 None。全部 doc 都有的值给每块加同样的 1/n，在 argmax 里抵消；只有一块有的值加 1.0。匹配在**模型实际看到的 doc 文本**上做（渲染后、截断后），不在原始 JSON 上做。

**它是 oracle**：用到了目标动作（gold / reference action），所以测的是"定位的价值上界"，不是可上线的定位器。在线定位器是另一个问题，我们那边的信号层实验全部阴性，这里不重复。

**接进 BFCL harness（约十行）**。`bfcl_history_kv_repair.py::_build_request_messages` 里 `texts = [_render_history_unit(unit) for unit in units]` 之后、`_recent_repair_indices` 之前，用触发本次修复的 reference step 的动作替换 `latest_index`：

```python
from d_witness_core import select_k_star, target_values

if os.environ.get("REPAIR_LOCATOR", "recent") == "witness" and ref_action is not None:
    # ref_action: 触发修复的 reference step 的 decoded tool call, 形如
    # {"name": ..., "arguments": {...}}（oracle 触发时 runner 已经拿着它）
    values = target_values(ref_action["name"], ref_action.get("arguments") or {})
    k = select_k_star(texts, values)   # None = 历史里没有任何字面 witness
    latest_index = k if k is not None else len(units) - 1   # 回退到 recent
```

`_recent_repair_indices` 不用改：window 仍以 `latest_index` 为右端，W1 就是"只修 witness 块"。`k is None` 是算法的正常输出（目标参数是自由文本、历史里没有字面依据），按 recent 回退并在日志里记一列。

**建议的对照**：在现有 `replace_w1`（recent）之外加 `replace_w1@witness` 和 `replace_w1@first` 两臂。三者一起能把 W1→W2→W4→All 的阶梯拆开：如果 witness@W1 追上 W4/All，说明阶梯涨的是"盖住了对的块"而不是"raw 越多越好"；如果追不上，说明 BFCL 上的损伤不是块定位问题。两种结果都有信息量。

**口径**：与 sham / corr 臂一样，请把 `k_star`、`df`、`score` 落到 details 行里（AppWorld 侧的 witness 表每题记 `n_docs, k_witness, k_median, df, score, target_doc_values`），方便对拍。

## 权重获取

D 修复的是 c2kv gist 压缩后的 KV，需要 c2kv 训练权重（Qwen3-4B-Instruct-2507 基座）：

- 我们训好的 checkpoint 在 NPU 服务器（8×910B3），路径找言成要；
- 或按 C2KV 官方配方自训：120K 样本（3×40K）/ 1 epoch / global batch 32 / LR 5e-5 / warmup 0.06 / wd 0.1（论文 2607.17715 §Training Setup；注意 README 示例写 2 epochs 与论文矛盾，以论文为准）。

## 范围与分工

本包是 C 线"信号层"（哪里该修 / 怎么修 / 怎么设对照）。"成本层"（修复的延迟与显存账、downstream persistence）在另一条线，代码与预注册都在，需要可另聊。
