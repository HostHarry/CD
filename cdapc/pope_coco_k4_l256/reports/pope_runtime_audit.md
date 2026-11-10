# POPE 运行时间核查与 COCO 评分

核查时间：2026-11-10 17:33，北京时间。

## CD 仓库可用范围

已执行远端 fetch 并核对远端分支。`HostHarry/CD` 的唯一远端分支为
`main`，当前提交为 `01b832e0d672d1b1932893b7a47f5e89ccd7f684`，
提交时间 2026-11-09 18:50:10 +08:00。
本机 Git 工作区为 `/root/autodl-tmp/cd-mmmu-upload`。
`/root/autodl-tmp/HostHarry-CD` 是较早导出的存档目录，并非 Git 工作区。

当前远端包含历史 native/DEGAP 和 CDAPC 存档；未包含用户所述本轮
VCD/VIGOR 实现、配置或逐题输出。全文检索 VCD/VIGOR 仅找到说明文档中的引用。
因此不能核实这两种实现的输入长度、消融方式、前向次数、缓存、输出投影范围、
EOS 停止逻辑、计时边界或本轮硬件配置。历史 native 参数不代表本轮参数。

用户提供的 native 6.5 秒、VCD/VIGOR 11.9 秒仅作外部报告引用，
未将其标记为本机重算结果。

## 相同输入上的 CDAPC 耗时

从新旧 COCO think 运行冻结 4,769 对已完成输入，每条均保存路径、SHA256、
generation fingerprint、耗时和提交统计。当前新组尚未完成 5,127 个输入；
这是一份运行时间快照，不是新组全量质量报告。

| 指标 | 原候选范围＋安全 EOS | EOS 前缀候选 |
|---|---:|---:|
| 输入数 | 4,769 | 4,769 |
| 平均生成秒数 | 27.9378 | 9.1768 |
| 平均迭代轮数 | 76.8840 | 25.3057 |
| 平均完整模型前向次数 | 153.7681 | 50.6114 |
| 总生成时间／总轮数，秒 | 0.36338 | 0.36264 |
| 已知 EOS 后仅填尾部的轮次占比 | 66.2191% | 0% |
| 平均仅填尾部轮数 | 50.9119 | 0 |
| 平均提交位置数 | 255.6622 | 49.8593 |
| 平均可见输出 token 数 | 42.7148 | 42.8092 |
| 平均 fallback 轮数 | 8.4867 | 8.4793 |

配对平均耗时之比为 **3.0444×**。两组每轮成本基本相同，原候选版多做了
约 51.58 轮，其中平均 50.91 轮只提交已经确定的 EOS 后方位置。
这里“仅填尾部”指：本轮开始前已有已提交 EOS，本轮所有提交位置均在
当时最早已知 EOS 之后。它不以最终输出位置倒推定义。

原候选快照有 4,761 条 success、8 条 budget_exhausted；
finish reason 为 4,735 canvas_filled、26 eos_prefix_complete、8 budget_exhausted。
耗时统计包含全部已完成终态记录。

该结果解释了原候选版相对前缀版的大量额外生成工作。
它不能替代与外部 VCD/VIGOR 的同环境实测，也不能给其他实现开销分配时间占比。
前缀版 9.18 秒低于外部报告的 11.9 秒，仅是尚未对齐计时条件的数值比较。

## 已核实的 CDAPC 输入输出结构

1. 每题图像预处理执行一次：512×512 图像经 MAGVITv2 得到 1,024 个离散图像 token。
   这批原候选输入预处理均值约 0.109 秒。
2. 输入为 `[任务标记、图像 token、聊天模板与问题、256 个响应位置]`；
   这批输入总长约 1,369–1,371，均值 1,369.23。
   thinking instruction 和 `Final answer: Yes/No` 要求包含在文本提示中。
3. 每轮依次进行有图、图像访问消融两次完整前向。两者保留同样的输入 token；
   消融通过 attention bias 阻止非图像 query 直接访问图像 key，
   并未删除图像位置或缩短输入。`use_cache=False`。
4. 模型对完整输入生成 logits，词表维度为 134,656，之后才截取响应区，
   得到每分支 `[1, 256, 134656]` 的 logits。
5. `make_proposals` 对整个 256 位置的合法文本词表进行 FP32 概率、增益、
   候选池与门控分数计算，再按 W=16 选择提交候选；W 不缩减模型前向或这一步张量范围。
6. 位置排序在 Python 中提取 GPU 标量；每个提交 token 的日志还会重复计算
   两分支 log-softmax、排序得到 top-5，并提取多个标量。
   `commit_log=true`，这些操作位于生成计时区域内。
   磁盘保存原始预测 JSON 位于生成计时结束之后，不能把落盘耗时直接归入这里的秒数。

这些是确认存在的工作量；没有单独 profile 或关闭日志的配对实验，
不能声称日志、attention backend 或某个张量操作分别占了多少时间。
是否构成 VCD/VIGOR 的相对开销，需要取得其本轮实现后判断。

源码依据（均为运行工作区版本）：

- `cdapc-pope-safe-eos/src/cdapc_lab/models/mmada.py`
- `cdapc-pope-safe-eos/src/cdapc_lab/models/base.py`
- `cdapc-pope-safe-eos/src/cdapc_lab/decoding/engine.py`
- `cdapc-pope-safe-eos/src/cdapc_lab/decoding/proposal.py`
- `cdapc-pope-safe-eos/src/cdapc_lab/decoding/selection.py`
- `cdapc-pope-safe-eos/src/cdapc_lab/experiments/infer.py`

逐输入证据与源码哈希见 `pope_runtime_audit.json`。

## 已完成 COCO 的官方规则复核

取最后一个 `</think>` 后的文本，未闭合则回退全文；去掉首尾空白后，
执行固定 POPE 官方关键词规则。闭合后为空仍默认 yes；
基础设施失败／预算耗尽保留分母计错。本次已完成前缀版两组均无生成失败。

每组 5,127 个唯一输入映射到全部 9,000 行，三个 split 各 3,000 行。
F1 为正类 yes 的 F1，平均行为 split 等权平均。

| split | CDAPC 直答 Acc / F1，% | CDAPC think Acc / F1，% |
|---|---:|---:|
| random | 82.63 / 81.35 | 69.57 / 74.35 |
| popular | 82.40 / 81.14 | 67.93 / 73.34 |
| adversarial | 76.70 / 76.47 | 67.10 / 72.83 |
| mean | 80.58 / 79.65 | 68.20 / 73.51 |

本次逐题重算与此前已用完整官方脚本验证的所有 split 完全一致。
独立报告位于 `cdapc-pope-eos-full/reports/`，文件名后缀为
`_after_think_official.json` 与 `.md`。

17:37 检查时，新原候选版 COCO 为 4,796 / 5,127，尚在生成。
额外的本地规则评分等待进程曾启动，但执行工具随后报告该后台 shell
被用户终止，17:37 已核实进程退出。对应 `.status.json` 已更新为 stopped，
尚未生成新组的答案区域官方评分。原实验生成控制器仍正常运行。
`scripts/score_pope_answer_region.py` 已在完整 COCO 直答／think 两组验证通过，
可在新组完成后直接使用，无需修改生成代码或调用 Judge。

要独立重算用户所述 Native/VCD/VIGOR 本轮 COCO，需要该轮逐题输出和数据映射；
当前远端存档没有这些文件。
