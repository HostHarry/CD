# CDAPC 结果补充（2026-11-12）

本目录收录 2026-11-10 之后在本机完成的 CDAPC 实验，以及与 CD 基线（`baselines/records`，
commit 见 `comparison_256_128/COMMIT`）的 256/128 统一对比。除特别说明外，CDAPC 设置为
MMaDA-8B-MixCoT、alpha 0.5、beta 0.1、u 0.1、r 0.7、W16/K4、L256/I128、快速 EOS 前缀候选
（候选限于已提交 EOS 之前，EOS 前缀完整时安全停止）。生成失败、缺答、歧义一律保留分母计错。

## 1. POPE think，无 Final answer 后缀（`pope_think_no_final/`）

提示：MMaDA 官方思考指令＋原题，与基线 `pope_think_v1` 一致。官方规则取最后一个 `</think>` 后文本，
未闭合用全文；DeepSeek 为 `pope_text_llm_judge_v1`（deepseek-flash，spec 7c519833…，与基线同一协议）。

| 数据集 | 官方 Acc / F1 | DeepSeek Acc / F1 |
|---|---:|---:|
| COCO | 75.99 / 77.77 | 76.20 / 77.98 |
| A-OKVQA | 74.63 / 76.83 | 74.60 / 76.91 |
| GQA | 76.74 / 78.44 | 76.83 / 78.54 |

COCO 复用64图 pilot 的661个输入，其余为新生成；`coco/official_scopes.md` 另列不含 pilot 的436图结果。
A-OKVQA/GQA 沿用各自原配置 seed 42；COCO 为 20261105。
`pope_coco_vs_vcd_deepseek/`：与基线 VCD 在 COCO 全部9000行上的同协议 DeepSeek 配对比较。

## 2. 256/128 各方法统一对比（`comparison_256_128/`）

所有方法逐行用同一规则重新评分（云端 Native/VCD/VIGOR 的官方 Acc/F1 逐 split 复现一致），
配对 bootstrap 2000 次、seed 42。主要结论：POPE 直答各方法差异很小；POPE think 下 CDAPC
显著优于 Native 与 VIGOR，与 VCD 持平；MMMU 仅 Judge 口径下 CDAPC K4 显著高于 VCD。

## 3. MMMU validation，K4（`mmmu_k4_l256_i128/`）

900题，WINO 思考提示。Official 27.89%，Strict 19.89%，DeepSeek Judge 27.33%（spec ffda0193…）。
4题预算耗尽计错。与历史 K16（原候选范围）的配对区间均跨0。

## 4. MME，K4（`mme_direct_k4/`、`mme_think_k4/`）

2374题、1187图像对，官方 Acc+Acc+ 评分并经未修改的官方脚本逐类核对。

| 条件 | Perception | Cognition | 总分 |
|---|---:|---:|---:|
| 直答 K4 | 987.94 | 258.93 | 1246.86 |
| think K4 | 1018.18 | 252.50 | 1270.68 |

直答与历史直答 K16 的完整 prompt IDs 与 seed 逐题一致；各项差值的图像对 bootstrap 区间均跨0。

## 5. HallusionBench（`hallusionbench/`）

VLMEvalKit TSV 全部1129行（951图像、178纯文本）；VLMEvalKit 原版评分，另有 DeepSeek 答案提取。
纯文本题无视觉分支，CDAPC 沿用同提示词 Native 输出（与基线 VCD/VIGOR 相同处理）。
本地 Native 为 256/128/64，基线为 256/128/32。

| 组别 | 官方 aAcc | DeepSeek aAcc |
|---|---:|---:|
| CDAPC 直答 | 44.38 | 44.91 |
| Native 直答（B64） | 42.16 | 43.05 |
| CDAPC think | 38.44 | 41.45 |
| Native think（B64） | 33.30 | 39.68 |
| 基线 Native / VCD / VIGOR（B32，直答） | 42.60 / 40.48 / 40.12 | 43.76 / 41.63 / 42.16 |

图像题上 CDAPC 直答显著优于本地 Native、VCD、VIGOR；think 组的官方分数下降主要来自解析失败。

## 6. LLaDA-V 提示词小样本（`lladav_prompt_pilot/`）

24个 POPE-COCO 输入。LLaDA-V 在任何提示词下都会写满整个画布（结束符落在最后一格），
因此 256 画布无法提前停止。64/32 时 Native 可写完并给出 Final answer。
LLaDA-V 全量实验已改为 64/64 并仍在运行，**本目录不含其结果**。

## 文件说明

各子目录含结果 JSON/Markdown、逐题答案或预测、预检与源快照哈希、配置和所用脚本。
原始 prediction 文件（含 commit log）体积过大未上传，保存在本机对应工作区，
其路径与 SHA256 记录在各 `prediction_inventory.json` / `inventory.json` / `result.json` 中。
`SHA256SUMS` 覆盖本目录全部文件。
