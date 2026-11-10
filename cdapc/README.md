# CD-APC 实验结果包

本文档的原始归档对应2026-11-09时 `cdapc-experiments`（MMaDA / CD-APC 新框架）与 `cdapc-mme`（MME 独立源码快照）中已完成的实验，包含源码节选、配置、文档、报告、逐样本规范化回答、manifest与校验和。
**2026-11-10新增：[COCO K4/L256补充包](pope_coco_k4_l256/README.md)**，包括直答、think EOS前缀版、think原候选安全EOS版及逐字节原始token/commit-log产物。新think原候选官方Accuracy/F1为69.23%/73.97%，每条件9000评测行。
原始归档仍不含raw runs；补充包单独保存其原始记录。两者均不含模型权重、LLM Judge HTTP缓存、凭据和瞬态文件。

- **原始归档未包含、现已部分补充**：K4 L256/I128在2026-11-09打包时尚未结束。COCO三个已结束条件现位于 `pope_coco_k4_l256/`；A-OKVQA/GQA的后续结果不在该补充包中。历史未结束运行及小样本EOS试验仍按原归档范围保留。
- **已取消、未包含**：`pope_aokvqa_reasoning_r50`（取消原因 `user_requested_stop_r50`），无报告与输出。
- 验证文件：`cd cdapc && sha256sum -c SHA256SUMS`。

## 目录结构

```text
cdapc/
  README.md            本文档
  manifest.json        全部文件的来源路径与 SHA256
  SHA256SUMS           本目录校验和
  source/              cdapc-experiments 源码节选（src/scripts/configs/manifests/docs/patches/tests + 根文件）
  source/mme/          cdapc-mme 快照：README、scripts、configs、manifests、tests、source_snapshot.json，
                       以及与 experiments 不同的 3 个 src/cdapc_lab 文件（其余 src 文件两边逐字节相同，见 manifest）
  reports/             已完成实验的报告 JSON/MD/log、Judge 逐输入结果、K pilot 结果、MME 官方评分导出与校验日志
  answers/             23 个条件的逐样本 JSONL + 可读 TXT + manifest.json + 本说明
```

## 公共协议

所有实验使用同一 MMaDA 多模态输入布局（与根 `README.md` / `mmmu/` 一致）：

```text
<|mmu|><|soi|>[1024 个 MAGVITv2 视觉 code，每个 + len(tokenizer)=126349]<|eoi|>[tokenizer chat-template 文本]
```

图像预处理：RGB → 短边 Resize(512, bicubic) → CenterCrop(512×512) → 0.5 归一化；视觉 code 共 1024 个。chat template 由 checkpoint 自带 `tokenizer_config.json` 生成（BOS `<|startoftext|>` 保留一次），无独立 system message。MMaDA checkpoint revision `3ee0085f0c42541f1134aae30482954451952406`，MAGVITv2 revision `5c3fa78f8b3523347c5cd1a4c97f3c4e96f33d5d`，batch size 1，BF16，SDPA，双向注意力。

CD-APC 解码参数（各条件相同部分）：alpha=0.5，beta=0.1，u=0.1，窗口 W=16，停止规则 `eos_prefix_complete`，variant `full`，branch_execution `sequential`，cache `disabled`，fallback `position_rule=normalized_gate_readiness / token_rule=base_top1`。

各任务的确切 prompt（`prompt_id` 见各自配置）：

| 任务 | prompt_id | user 文本 |
| --- | --- | --- |
| POPE 直答 | `pope_yesno_v1` | `{question} Please answer yes or no.` |
| POPE think | `pope_yesno_reasoning_v1` | `OFFICIAL_THINKING_INSTRUCTION + "\n" + {question}\nAfter </think>, end with exactly Final answer: Yes or Final answer: No.` |
| CHAIR | `chair_caption_v1` | `Please describe this image in detail.` |
| MMMU | `mmmu_wino_reasoning_v1` | `OFFICIAL_THINKING_INSTRUCTION + "\n" + {question}\n{options}`（WINO 式上下文，选项为 `A. …` 行） |
| MME | `mme_yesno_v1` | 官方题目原文（已含 `Please answer yes or no.`），无额外推理指令 |

`OFFICIAL_THINKING_INSTRUCTION`：`You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here`。`model_reasoning_instruction=True` 的任务（POPE think、MMMU）在 user 文本前加该指令加换行。

## MMMU（cdapc_r70，L256/I128）

- 数据：`lmms-lab/MMMU` validation，revision `364f2e2eb107b36e07ff4c5a15f5947a759cef47`；900 题 / 900 唯一输入（43 道多图题只用一张图：题干+选项中按名称排序的第一个 `<image n>` 占位符，没有则 `image_1`）。
- 生成：画布 L=256，迭代上限 I=128（**预算受限**，8 条 `budget_exhausted` 计错），K=16，r=0.7，seed `20261105`。
- 评分：Official（官方评测器，含随机 fallback）、Strict（不猜答案）、LLM Judge（`deepseek-flash`，WINO 的 `JUDGE_RULES` 提示词，回复含 "1" 记对；主指标）三轨。
- Native 基线：已有 `mmmu/` 包（`mmmu_wino_reasoning_v1` 同协议，L=256、steps=128、block_length=64）。注意旧 CD-APC r90/r50 为 I256，本 r70 为 I128，并非只改 r 阈值的对照。
- 报告：`reports/mmmu_cdapc_r70_L256_I128.json`、`reports/mmmu_r70_256_128_runs.json`；输出 `answers/mmmu_cdapc_r70_L256_I128.*`。

| 方法 | Official % | Strict % | Judge % | 平均步数 | 平均秒数 | 平均输出 tokens |
|---|---:|---:|---:|---:|---:|---:|
| Native（mmmu/） | 27.78 | 18.89 | 25.22 | 128.00 | 19.92 | 102.42 |
| r90（mmmu/，I256） | 29.56 | 22.78 | 27.44 | 77.83 | 28.92 | 114.62 |
| r50（mmmu/，I256） | 29.67 | 18.22 | 24.33 | 31.37 | 11.69 | 95.44 |
| **r70 L256/I128（本包）** | **28.67** | **19.89** | **25.11** | 49.14 | 18.37 | — |

## POPE 直答（128-token 画布，L=I=128，K=16，W=16）

- 数据：RUCAIBox/POPE SEEM test。COCO：`lmms-lab/POPE` rev `4db1276663dfa5eb8ad16a52d24c31a09e470896`（与 RUCAIBox `08d957b9…` 逐条核对一致）；A-OKVQA/GQA：rev `08d957b917e5a378a2f99d35b6293c536a66298b`。每数据集 9,000 条评测记录（random/popular/adversarial 各 3,000），去重后唯一输入：COCO 5,127、A-OKVQA 5,145、GQA 5,171。
- 评分：官方 Yes/No 重写规则（pinned `upstream/pope/evaluate.py` 应答改写循环）+ `strict_yes_no_v1` 解析；主指标为三子集等权 Accuracy/F1。
- Native 基线：仅 COCO 有（本仓库 `pope/native` 记录，L=128、steps=64、block_length=32；`reports/pope-native-recompute.json` 复算一致）；A-OKVQA/GQA 无 Native。
- r 扫描只在 POPE-COCO test 上完整进行（r90/r80/r70/r60/r50）；A-OKVQA/GQA 取 r90/r70/r50。
- 报告：`reports/pope_yesno_mmada_cdapc_L128*.json`（COCO 五档；其中无后缀的 `pope_yesno_mmada_cdapc_L128.json` 即 r90，`_r80/_r70/_r60/_r50` 依次对应）、`reports/pope_{aokvqa,gqa}_r{90,70,50}.json`、`reports/pope_rgate_sweep*.json`、`reports/pope_rgate_behavior.*`；输出 `answers/pope_*_direct_r*.jsonl/txt`。

| 方法 | COCO Acc / F1 % | A-OKVQA Acc / F1 % | GQA Acc / F1 % |
|---|---:|---:|---:|
| Native | 80.33 / 80.33 | — | — |
| r90 | 80.44 / 81.25 | 78.29 / 80.41 | 78.94 / 80.97 |
| r80 | 80.47 / 81.29 | — | — |
| r70 | 80.52 / 81.38 | 77.98 / 80.12 | 79.06 / 81.03 |
| r60 | 80.29 / 81.14 | — | — |
| r50 | 79.94 / 80.84 | 78.43 / 80.29 | 79.51 / 80.96 |

| 数据集 | 条件 | 平均步数 | 平均秒数 |
|---|---|---:|---:|
| coco | r90 / r80 / r70 / r60 / r50 | 9.40 / 7.06 / 5.13 / 3.34 / 1.93 | 2.54 / 1.92 / 1.39 / 0.91 / 0.53 |
| aokvqa | r90 / r70 / r50 | 9.37 / 5.36 / 1.93 | 2.53 / 1.44 / 0.52 |
| gqa | r90 / r70 / r50 | 9.43 / 5.34 / 1.87 | 2.54 / 1.44 / 0.51 |

COCO 各档 seed `20261105`；A-OKVQA/GQA 各档 seed `42`。COCO r70 平均 Accuracy 仅比 r90 高 0.078 个百分点（净 7/9,000 条），各子集配对 95% 区间含零，r70 不是跨任务通用最优。

## POPE think（128-token 画布，L=I=128，K=16，W=16）

- 数据同 POPE 直答（A-OKVQA 5,145 / GQA 5,171 唯一输入，9,000 条记录/数据集）。
- 评分主指标：`reasoned_yes_no`（`pope_final_yes_no_v1`）只解析最后一个 `</think>` 之后（可含 `Final answer:` 标记）的区域，未解析/歧义计错；另有 `raw_official` 文本诊断与独立 LLM Judge（`deepseek-flash`，不接收图片与 gold，逐输入结果在 `reports/pope_think_llm_judge/`）。文本审计口径另见 `reports/pope_think_answer_audit.*`、`pope_think_answer_manual.json`。
- 无 Native 基线。A-OKVQA think 仅 r90/r70 完成（r50 已取消）；GQA think r90/r70/r50 完成。
- 报告：`reports/pope_{aokvqa,gqa}_reasoning_r*.json`、`reports/pope_think_llm_judge/*.json`（不含 cache 与 runner.lock）；输出 `answers/pope_*_think_r*.jsonl/txt`。

| 条件 | 严格最终答案 Acc % | 原未解析率 % | 文本审计 Acc % | LLM Judge Acc % | 平均秒数 |
|---|---:|---:|---:|---:|---:|
| pope_aokvqa_reasoning_r90 | 42.97 | 37.02 | 62.70 | 62.41 | 7.52 |
| pope_aokvqa_reasoning_r70 | 31.32 | 56.30 | — | 57.71 | 4.12 |
| pope_aokvqa_reasoning_r50 | cancelled | — | — | — | — |
| pope_gqa_reasoning_r90 | 44.60 | 38.46 | 64.18 | 63.96 | 8.42 |
| pope_gqa_reasoning_r70 | 30.57 | 58.21 | 57.24 | 57.02 | 4.63 |
| pope_gqa_reasoning_r50 | 12.18 | 84.07 | — | 43.61 | 1.90 |

各档 seed `42`。未解析记录中 A-OKVQA r90、GQA r90/r70 约 67%/65%/59% 存在可辨认答案（文本审计口径），放宽格式后仍有标签空壳与实际答错；未审计组保持 —，不外推。

## CHAIR（64-token 画布，L=I=64，K=16，W=16）

- 数据：CHAIR-COCO-val2014-500，revision `d06a661f5d1262a5856a577ed0a53f2c209481a8ac17b182ef59a5a5a778726c`；500 张 COCO val2014 图 / 500 条描述。
- 评分：pinned CHAIR 评测器（rev `4087a26211aa2339b9a76307cb8f0321ef691d0a`），COCO ground-truth 物体集合经 `chair_reference_verified.json` 核对；指标 CHAIRs/CHAIRi/Recall。
- Native 基线：本仓库 `chair/native` 记录（L=64、steps=32、block_length=16；`reports/chair_reference_verified.json` 复算一致）。
- 报告：`reports/chair_r{90,70,50}.json`（含逐图 `sentences`）、`chair_r*_answers.json`、`chair_pope_extensions_runs.json`；输出 `answers/chair_r*.jsonl/txt`。各档 seed `42`。

| 方法 | CHAIRs ↓ % | CHAIRi ↓ % | Recall ↑ % | 平均步数 | 平均秒数 |
|---|---:|---:|---:|---:|---:|
| Native | 21.80 | 9.90 | 54.92 | — | — |
| r90 | 25.00 | 9.92 | 57.42 | 47.67 | 11.96 |
| r70 | 23.60 | 9.57 | 57.17 | 33.88 | 8.50 |
| r50 | 22.40 | 7.56 | 55.98 | 22.05 | 5.56 |

r50 的 CHAIRi 比 Native 低 2.34 个百分点，但 CHAIRs 高 0.60 个百分点；物体提及级与描述级幻觉需分开报告。

## GQA think K pilot（30 图子集，L=I=128，W=16，r=0.7）

- 固定 30 张随机 GQA 图、306 个唯一输入、540 条评测记录；比较每轮最大提交数 K=16（复用 `pope_gqa_think_r70` 全量输出）/K=4/K=2，其余设置相同。**只是 30 图子集，不与全量分数直接比较。**
- 报告：`reports/k_commit_pilot/`（`results.json` 逐输入、`rows.json` 冻结行、`index.json`、`protocol.json`、`baseline_records.json`、`results.md`）；输出 `answers/gqa_think_k{4,2}_pilot.jsonl/txt`（K16 见 `answers/pope_gqa_think_r70.jsonl` 过滤）。各档 seed `42`。

| K | LLM Judge Acc % | LLM Judge F1 % | Strict Acc % | 空壳 % | Judge缺答 % | Judge冲突 % |
|---|---:|---:|---:|---:|---:|---:|
| 16（复用全量） | 56.67 | 76.83 | 29.07 | 28.10 | 28.10 | 0.00 |
| 4 | 63.15 | 77.42 | 31.30 | 19.93 | 19.93 | 0.00 |
| 2 | 62.78 | 75.56 | 41.85 | 16.01 | 16.01 | 0.00 |

图像级配对差值区间见 `reports/k_commit_pilot/results.md`；逐输入提交行为（`mechanism` 字段）在 JSONL 中。

## MME（cdapc-mme 独立快照，r70，L256/I128）

- 数据：`lmms-lab/MME` revision `d6c9023f017b564f7b3ccccf5348166bce8fdbcd`；官方 14 类、2,374 题、1,187 个图像问题对；官方 Yes/No 提示词原文。
- 生成：画布 L=256，迭代上限 I=128（**预算受限**），K=16，r=0.7，W=16，seed `20261105`；预算耗尽计错且保留分母。
- 评分：官方 `calculation.py`（commit `dd2950902889cd614d4edf606827240166d29381`）Acc/Acc+，已用未修改官方脚本逐类核对（`reports/mme_official_scorer.log`、`mme_official_text/`）。
- 无 Native 基线。MME 在独立源码快照 `cdapc-mme` 中运行，与本包 `source/mme/` 对应；与 experiments 的 `src/cdapc_lab` 仅 `config.py`、`data/records.py`、`experiments/pope_report.py` 三个文件不同，其余逐字节相同（清单见 `manifest.json`）。
- 报告：`reports/mme_cdapc_r70_L256_I128.{json,md}`、`reports/mme_r70_256_128_runs.json`（与快照 `reports/index.json` 同内容，另存为 `reports/mme_index.json`）、`reports/mme_official_text/`、`reports/mme_official_scorer.log`；输出 `answers/mme_r70_L256_I128.*`。

| 分区 | 得分 |
|---|---:|
| Perception | 997.37 / 2000 |
| Cognition | 228.57 / 800 |
| **总分** | **1225.95 / 2800** |

逐类得分见 `reports/mme_cdapc_r70_L256_I128.md`。

## 验证与复现

- `answers/` 由 `source/scripts/export_cdapc_upload.py` 离线生成：逐条件重算指标并与已存报告逐项相等断言（POPE 各子集 Acc/F1/Precision/Recall、CHAIRs/CHAIRi/Recall、MMMU 三轨、MME 逐类、K pilot 全部聚合），并核对 run 清单、预测/评分状态、ID 覆盖、Judge 请求哈希与缓存解析一致性。字段与归一化规则见 `answers/README.md`。
- 接受度证据：`reports/mmada_acceptance.json`、`lavida_acceptance.json`、`mmada_ccaw_v1.json`、`lavida_ccaw_v1.json`（含 `.log`）、`mechanism16.json`、`mechanism16-replay.json`、`final_audit.json`、`mmada_native_*` / `mmada_adapter_*` / `lavida_native_*` / `lavida_adapter_*` 各 run 记录。
- 解码行为证据：`reports/pope_rgate_behavior.*`、`reports/pope_think_answer_audit.*`、`pope_think_answer_manual.json`、`chair_reference_verified.json`、`pope-native-recompute.json`。
- `manifest.json` 记录每个文件的 `source_path` 与 SHA256；`source_git_commit` 为 `cdapc-experiments` 的 HEAD，工作区含未提交实现（`source_includes_uncommitted_experiment_implementation: true`）。MME 快照源路径以 `source/mme/` 前缀对应 `cdapc-mme/`。

## 注意事项

- **MMMU r70 为 I128**，旧 r90/r50 为 I256；r 阈值与迭代预算同时变化，不是单变量对照。
- **r 扫描在 POPE-COCO test 上进行**；A-OKVQA/GQA 只取 r90/r70/r50 三档，不能外推全阈值曲线。
- **POPE think 严格未解析率 37%–84%**（随 r 下降而升高）；严格分数把无法解析计错，Judge 分数是独立口径，文本审计又是第三口径，三者不可混用。
- **CHAIR 64-token 画布截断多数描述**：Native 与 CD-APC 平均输出都约 60–63 tokens，以句末标点结束的仅 Native 148/500、r90 113/500、r70 125/500、r50 194/500；CHAIR 对长度敏感，差异部分来自长度。
- **K pilot 是 30 图子集**（306 输入 / 540 记录），K16 复用全量 GQA think r70 输出；配对区间按图像重采样，小样本结论仅供方向参考。
- **K4/L256后续结果**：COCO已结束条件的配置、源码、报告及原始预测已补充至 `pope_coco_k4_l256/`；不据此宣称A-OKVQA/GQA后续队列已完成。历史187输入EOS小样本不在原始归档内，补充包提供的是全量COCO及相关诊断。
- `pope_aokvqa_reasoning_r50` 已取消，所有表格中该行只有状态标记。
- 平均秒数不含模型加载与输入预处理；预算受限条件（MMMU r70、MME）的 `budget_exhausted` 样本计错并保留分母。
