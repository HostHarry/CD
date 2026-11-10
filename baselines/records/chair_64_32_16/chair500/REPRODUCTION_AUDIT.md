# 完整运行后的复现审计

## 结论

当前证据支持将本轮称为**统一 MMaDA 协议下的 VCD/VIGOR 适配对照**。
未发现漏样本、源码漂移、原生结果混用、EOS 提前结束破坏答案前缀、
VIGOR 接错解码循环或 POPE 计分错误。

确实存在需要区分的三类问题：

1. **已确认的实验条件差异**：VCD 噪声、采样和随机种子协议，
   VIGOR 日程，以及提示词/评测范围没有完整对齐论文或作者脚本。
2. **未核实的算法适配选择**：扩散 VCD 的提交排序采用经过 APC 的
   对比分布置信度。作者原版 AR 代码不定义这个多位置排序问题，
   VIGOR 附录没有提供足够明确的排序公式。
3. **已确认的评分器缺陷**：固定 Pattern 版本把 `bus` 单数化为 `bu`。
   独立修正这个词并重算，主要方法排序没有反转。

这些事实不能证明性能下降必然是代码错误，也不能支持已经复现论文收益。
本次没有改写正式预测、评分器、模型权重、运行源码或已完成的 manifest。
新增检查脚本和诊断结果在 `results/reproduction_audit/`。

## 1. 运行与数据完整性：通过

`results/reproduction_audit/integrity.py` 对完成的数据进行只读审计：

- 每种方法覆盖全部 5127 个 POPE 唯一输入、全部 9000 个评测行以及
  500 张 CHAIR 图像，无缺失或额外预测键。
- 当前运行源码、保存的源码快照、上游固定文件、样本文件及所有图像
  与记录的 SHA256 一致。
- VCD、VIGOR 两个数据库内的 native payload 逐字节一致。
- 全部提交轨迹能够重建实际生成状态；已提交位置没有被再次覆盖。
- 解码文本与保存的 token 前缀一致；EOS 的 tokenizer/model ID 一致。
- 提前结束记录的 EOS 之前没有 MASK，尾部未计算位置仍明确保留在
  `generation_state_ids`，没有伪造完整生成序列。
- 全部 POPE 输出均为 Yes/No。独立重算各子集混淆计数、Accuracy 和 F1，
  与正式结果完全一致。
- CHAIR 的答案文件哈希、评分器哈希和生成 manifest 哈希匹配。
  另用原评分器重新计算全部 native/VCD/VIGOR CHAIR 分数，结果一致；
  从逐图对象记录独立汇总 CHAIRs/CHAIRi 也一致。

证据：`integrity.json`、`chair_sensitivity.json`。
这排除了上述工程错误；它不等于证明未执行的 EOS 尾部 token 相同。

## 2. VCD：优先核实提交排序，并标明参数差异

### 已确认的协议差异

| 项目 | 作者论文/源码 | 本轮 |
|---|---|---|
| POPE 图像加噪 | VCD 附录 A：T=999；Python 参数默认 500 | 500 |
| 主实验解码 | 从 softmax 分布直接采样 | temperature=0 贪心 |
| 重复实验 | POPE/MME 报告 5 次运行的均值、标准差 | 一个固定种子协议 |
| RNG 使用 | 作者 LLaVA 入口启动时设置一次 seed | 每个唯一输入重新设 seed=42 |
| 视觉条件 | 作者 AR 模型的视觉编码 | MMaDA 的 MAGVIT 离散码 |
| 位置提交 | AR 每步只有一个解码位置 | 按截断后对比分布置信度选位置 |

alpha=1、beta=0.1 与论文一致。作者 shell 示例 beta 默认 0.2，
Python parser 默认 0.1；本轮确实选择了后者，没有把两者混写成同一默认值。
VIGOR 是温度零的确定性方法，不能把 VCD 的五次随机运行要求直接套给它。
现有 image-bootstrap 区间也不是五次 VCD 推理重复实验的替代品。

作者噪声函数的数值系数：

- step=500：约 `0.8630 * image + 0.5052 * noise`。
- step=999：约 `0.2850 * image + 0.9585 * noise`。

本轮 POPE 的 noisy MAGVIT 码平均有 **99.04%** 与 clean 码不同。
这说明离散编码对该扰动很敏感；**不能解释为 99% 的视觉语义丢失**，
也不能据此断定噪声实现有错。未将 999 当作必然改善效果的“修复”。

### 提交排序的具体问题

当前 `vcd/decode.py` 用作者原版公式构造候选分布：

```text
contrast_logits = (1 + alpha) * clean_logits - alpha * noisy_logits
APC: 根据 clean_logits 的 log(beta) + max(clean_logits) 截断候选
candidate = argmax(APC(contrast_logits))
commit_score = softmax(APC(contrast_logits))[candidate]
```

因此 VCD 既可能改变选词，也可能改变提交顺序。这是多位置扩散适配中
需要明确说明的选择，不能直接从作者 AR 代码推导出唯一答案。
VIGOR 附录 C.1 的 “clean side alone” 没有给出排序置信度的明确公式，
不应将其擅自解释成必须使用对比前的 clean 概率，也不应宣称当前排序
已被证明与该附录实现相同。

实测诊断：

- 先用两位置概率例子确认：alpha=0 时，APC 重新归一化仍可以反转位置
  置信度大小，虽然两个位置的候选 token 没变。
- 再取 POPE 前两张图的前两个问题，共四个固定输入，在实际 MMaDA 上
  运行 native、alpha=0 且保留 APC、alpha=0 且关闭 APC 三个条件。
- **四个输入全部出现提交轨迹变化**；其中一个输入的安全 EOS 步数
  从 native 的 16 步变为 3 步。
- **四个答案均未改变**。关闭 APC 后，答案与提交轨迹都恢复为 native。

这证明当前适配里 `alpha=0` 不是“完全无干预”的对照；没有证明这一选择
导致本轮分数下降。也不能把它未经作者确认就改成另一套排序并称作
“修复后的原版 VCD”。

证据：`vcd_apc_ordering_demo.json`、`vcd_alpha_zero_probe.py/json`。
关闭 APC 仅是诊断进程中的临时控制，正式代码和预测没有改动。

## 3. VIGOR：循环一致，实验条件仍不同

本次重新取得作者匿名发布的 POPE 和 CHAIR 脚本，保留下载地址和 SHA256。
对 `baseline=False`、合法日程和 `remasking=visual_ablation` 专门化后，
其 POPE 解码函数与当前固定 GitHub AMBER 文件中的函数体 **AST 相同**。
作者 CHAIR 脚本也导入这一 POPE 函数。

全量轨迹还确认：

- 提交的 token 来自正常分支的 argmax，没有误接 VCD 对比候选。
- 提交位置满足记录的 visual-gain 分数 top-k 条件。
- 视觉消融为 MASK image codes，SOI/EOI 保持原生格式。
- 每步重算消融分支，FP64 概率，`clamp(min=1e-8)` 与作者代码一致。
  论文伪代码的 epsilon 加法与发布代码的 clamp 有差别；本轮跟随代码。

| 条件 | 作者脚本默认/显式分支 | 本轮 |
|---|---|---|
| POPE 长度/步数/块长 | CLI 默认 16/8/16 | 128/64/32 |
| CHAIR 64-token 档 | 64/32/64 | 64/32/16 |
| CHAIR 512-token 档 | 512/256/128 | 未覆盖 |
| POPE 额外后缀 | `--prompt_suffix` 默认空 | 9000 题均加 yes/no 后缀 |
| CHAIR 提示词 | 见下文 | `Please describe this image in detail.` |

作者 CHAIR 默认提示词是：

```text
Provide a detailed caption describing all objects, attributes, and actions visible in the image.
```

上述 POPE 数值和后缀来自脚本默认值，**不是已恢复的论文实际 launch 命令**。
当前 128/64/32 是用户明确选择的 MMaDA 比较预算；不能将这个合法的比较
选择说成作者原配置。CHAIR 的较小块长限制了每一步可重排的位置范围，
提示词也改变了输入条件；影响大小需要另一个预先定义的匹配实验确认。

证据：`vigor_author_loop_equivalence.json`、
`vigor_pope_amber_ast.diff`、下载的三个作者脚本。

## 4. CHAIR 评分缺陷及独立敏感性检查

原评分器会把 `bus` 单数化为 `bu`，导致普通的 bus 提及不被正确识别。
native、VCD、VIGOR 分别有 8、10、6 条 caption 包含这个词。
诊断只为这个词增加例外，重新构建同一 500 张图的标注词表并评分：

| 方法 | 正式 CHAIRs / CHAIRi | bus 例外诊断 CHAIRs / CHAIRi |
|---|---:|---:|
| native | 21.80 / 9.90 | 22.00 / 9.83 |
| VCD | 25.20 / 10.10 | 25.40 / 10.01 |
| VIGOR | 22.80 / 9.38 | 22.80 / 9.27 |

单位为百分比。主要排序没有反转；这个词形缺陷不足以解释 VCD 的整体退步。
诊断评分没有替换正式评分。采用新的评分规则时，应把三种方法的修订评分
一起单独发布，不需要重新生成 caption。

另一个范围限制是生成上限：native/VCD/VIGOR 分别有 **71.4% / 50.8% /
60.4%** 的 caption 达到 64 token 上限而没有 EOS。它们不能代表论文的
512-token 实验，CHAIRi 的小幅差异也应连同回答长度一起解释。

## 5. 推荐的后续处理

1. 保留本轮结果，明确附带本文的适配说明和评分器敏感性结果。
2. 优先取得/确认扩散 VCD 的提交置信度公式。取得前，当前版本应明确标为
   “VCD + APC 后对比置信度排序的 MMaDA 适配”，不要将 clean 排序猜测
   当作经过验证的修复。
3. 若目标改为对齐作者实验条件，预先固定另一个协议：明确 VCD 噪声设置、
   随机协议；明确 VIGOR 日程与提示词，再生成匹配 native 对照。
   改了 prompt/计划长度/块长后，不能复用当前 native 预测。
4. 新配置先做固定小集合的正确性检查，再决定评测规模；不要根据测试集
   上哪种设置分数更高来反向选择“正确实现”。

这次仅进行只读全量审计、评分重算和四输入机制验证，没有启动新的全量推理。

## 来源与证据

- VCD 论文：<https://arxiv.org/pdf/2311.16922>，重点 §4.1、附录 A。
- VIGOR 论文：<https://arxiv.org/html/2611.05066v1>，重点 §4.1、附录 B/C。
- 作者脚本链接及下载哈希：`results/reproduction_audit/download_sources.json`。
- 全量完整性检查：`results/reproduction_audit/integrity.json`。
- 评分重算：`results/reproduction_audit/chair_sensitivity.json`。
- 四输入机制验证：`results/reproduction_audit/vcd_alpha_zero_probe.json`。

原固定 GitHub 仓库未提供可核实的扩散 VCD 排序代码；本次也未取得该实现。
匿名发布的目录枚举返回 HTTP 401，未绕过访问限制。已可公开读取的
POPE/CHAIR 文件足以完成 VIGOR 循环比对。
PDF 文本解析依赖仅安装在审计目录的 `pdf_vendor/` 中，
没有更改实验环境或共享 Python 包。
