# MMMU 输出失败模式

对 Native、r90、r50 各 900 条实际回答进行结构分类，并核对具体内容错误。全部回答见 `../answers/README.md`，逐题分类和带原文的人工核对案例见 `failure_modes.json`。

## 全量结构分类

每格为“题数（其中 Judge 判对数）”。以下类别互斥，每列合计 900。

| 输出形式 | Native | r90 | r50 |
| --- | ---: | ---: | ---: |
| 完全空字符串 | 0 | 0 | 0 |
| 只有 think 标签，没有正文和答案 | 84（0） | 81（0） | 115（0） |
| 没有完整 think 标签 | 20（3） | 19（3） | 21（1） |
| 有关闭标签，没有打开标签 | 0 | 0 | 1（0） |
| think 有内容但未正确闭合 | 124（36） | 80（18） | 263（68） |
| think 已闭合，其后没有文本 | 217（52） | 104（31） | 86（20） |
| think 已闭合，其后有文本 | 455（136） | 616（195） | 414（130） |

本次先单独归类“标签空壳”。它包括仅 `<think>`，以及空的 `<think></think>`：Native 为 83+1，r90 为 80+1，r50 为 113+2。因此与此前只统计“恰好为 `<think>`”或直接统计闭合标签的数字有所不同。

“关闭标签后有文本”只是一项结构属性，不保证该文本是有效答案。所有导出记录都生成成功，没有缺失样本；这里的空字符串检查针对评分时的归一化文本。

## 对用户关心的几类问题逐项判断

**没有 think：不一定是回答失败。** r90 的 `validation_Psychology_22` 只输出 `B`，与 gold 一致；`validation_Electronics_12` 只输出 `10`，也正确。无 think 标签中也存在 `1think.5` 等损坏文本，因此应同时查看回答和得分。

**think 完没有外部答案：不一定没有给出答案。** r90 的 104 条此类输出中，31 条被 Judge 判对，因为答案可能已经写在 think 内。不能把 104 条全部当作漏答。

真正未给出结论的案例是 Native 的 `validation_Accounting_11`：

```text
<think>
To determine the maximum possible loss per share for the long strangle
strategy, we need to consider the exercise of the put and call options. The

</think>
```

原文没有给出数值或选项，gold 是 A / $9.00。正文停在未完成的 `The` 后，却出现了关闭标签。

**没有任何输出：空字符串为 0；标签空壳分别为 84、81、115 条，全部判错。** 例如 r50 的 `validation_Accounting_2` 只有 `<think>`，既没有推理也没有答案。这是本批数据中更常见的“无实质输出”。

**think 错误：已有可直接确认的内容错误，但不能把所有 Judge=0 都标成推理错误。**

| 方法与题目 | 原文证据 | 可以确认的问题 |
| --- | --- | --- |
| r90 / `validation_Accounting_1` | `10 \times 30 = 30` | 算术错误；末尾 D、gold B |
| r50 / `validation_Accounting_1` | `$10,000 for 25,000 units, which is $4 per unit` | 按其写出的数字应为 $0.4；末尾还有损坏的 `</thinkthink>` |
| r90 / `validation_Accounting_29` | 同时称 A、B 波动率为 5%，随后说 A 的 5% 高于 B | 内部比较矛盾；答案 A 写在 think 内，gold B |
| r50 / `validation_Clinical_Medicine_21` | 大量连续重复 `abdominal` | 推理退化；末尾 B 仍被 Judge 判对 |
| Native / `validation_Diagnostics_and_Laboratory_Medicine_2` | `The answer is D. </think> B` | 正文与最终选项冲突；gold B，因此 Judge 仍给 1 |

人工核对的八个案例，包括正确的无 think 回答、空 think、漏答、算术错误、逻辑矛盾和重复，均在 JSON 中保留了完整问题、选项、gold、输出和说明。

## 格式问题解释不了全部错误

r90 中，**616 条有闭合 think 且后面有文本，仍有 421 条被 Judge 判错**；这占 r90 全部 653 条错误的约 64.5%。r50 对应为 414 条中 284 条错误。

因此，r50 更突出的结构问题是空 think 和未闭合 think；r90 的大量失败则发生在表面结构已经完整的输出中，需要继续检查读图、算术、逻辑、答案映射等内容。

这 421/284 条是内容核验的候选集合，不能直接当作“已确认推理错误”的数量：答案错误也可能来自图像理解、选项映射或评分问题；而答案正确的样本也可能包含错误推理。

## 自动筛查范围

另做了两个保守探针：

- 连续至少 8 次重复同一个、长度至少两字母的词：Native 1 条、r90 0 条、r50 10 条。它只覆盖一种机械重复；r90 为 0 不意味着没有其他重复模式。
- think 内最后一次明确的英文“answer is X”与 think 外明确字母不一致：Native 3 条、r90 0 条、r50 0 条。这只是显式字母矛盾，不覆盖数字与选项内容不一致等语义冲突。

本次没有新增模型推理或 LLM 判分。结构统计覆盖全部 2,700 条；“推理错误”的确认仅覆盖文中人工核对案例，不声称提供了全量推理正确率。

复现：在原项目运行 `python scripts/mmmu_failure_modes.py`，输入为已导出的三组答案和原始题目信息。
