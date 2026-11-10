# POPE-COCO：CDAPC K4 / L256 / I128

2026-11-10 补充上传的三个已结束条件。每个条件含5127个唯一输入、
9000条评测行（Random / Popular / Adversarial各3000），输入集合完全一致。
这是此前 `cdapc/README.md` 中尚未收录的K4/L256实验的COCO部分。

| 条件目录 | 答案区域官方 Accuracy % | F1 % | 生成秒/输入 |
|---|---:|---:|---:|
| `direct_eos_prefix` | 80.58 | 79.65 | 0.77 |
| `think_eos_prefix` | 68.20 | 73.51 | 9.14 |
| `think_original_safe_eos` | 69.23 | 73.97 | 27.91 |

指标为三个split等权平均，Yes为正类。生成耗时不含模型加载和输入预处理。
Judge为独立的已有文本答案提取结果，存于各目录的 `judge.json`；
其分数不与本表官方规则分数混用。本次导出没有重新调用Judge。

## 生成协议

MMaDA-8B-MixCoT，MAGVITv2，512像素center crop，1024视觉码，BF16，
双向SDPA，batch=1，temperature=0。参数为
alpha=0.5、beta=0.1、u=0.1、r=0.7、W=16、K=4、L=256、I=128，
budget_limited=true，seed=20261105；每轮两次顺序、无缓存前向。
模型、tokenizer和官方代码的固定revision/哈希在各run manifest中。

三个条件都只在实际EOS前缀已提交完整时安全停止。
`eos_prefix`源码在已有EOS后将候选限于EOS前的缺口；
`original`源码仍允许候选窗口进入EOS后的区域。这是候选策略差异，
并非把“预测到EOS”直接当作停止条件。

直答user文本：`{question} Please answer yes or no.`

think在user文本前添加官方思考指令，再接：
`{question}\nAfter </think>, end with exactly Final answer: Yes or Final answer: No.`
完整指令在各源码快照的 `models/mmada.py`，实际输入token ID保存在每条预测内。

## 评分与预算失败

先取最后一个 `</think>` 后的文字；未闭合则用全文；去除首尾空白。
再按固定POPE官方规则取第一个英文句点前的文本，删逗号，按普通空格分词；
含独立的 `No`、`no` 或 `not` 判No，否则Yes。成功生成但答案区域为空时默认Yes。

原候选版有8个预算耗尽输入，对应22条评测行，全部保留分母计错。
其余5119输入成功；两个前缀条件均为5127成功。
**原始run manifest因此标记为`incomplete`，本包保留该字段原值。**
这里“已结束”指所有计划输入均已有终态记录、9000行均已评分，
不表示8个失败被删除或改写为成功。验证脚本核验计划输入与实际记录完全一致。

每个条件的 `official.json` 是本表评分来源。
`generation_report.json` 原样保留历史严格解析等统计；
其中旧字段名称不应被当作本表答案区域规则的替代指标。

## 文件与可移植验证

- `rows.json`：完整9000行问题、标签、图像名/SHA256和input_key映射。
- 各条件 `answers.jsonl` / `answers.txt`：每个唯一输入的完整可见回答；
  JSONL另含截取答案、状态、耗时及规则/Judge判定。
- 各条件 `records/part-*.tar.gz`：**原始prediction和score文件逐字节保存**，
  包括prompt IDs、raw response IDs、commit log、成本、EOS信息及规范化文本。
  每包至多256个输入，gzip时间戳固定；不是裁剪或重写后的轨迹。
- 各条件 `records_index.json`：input_key到压缩包、成员路径、原文件SHA256的映射。
- 各条件 `runs/`：三个分片的原始run/dataset/evaluation manifest及resolved config。
- `source/eos_prefix/`、`source/original/`：独立源码快照、配置YAML、分片输入ID、
  样本元数据和上游锁文件；生成相关源码哈希与相应run manifest核对一致。
- `reports/`：全量EOS配对比较、源码/耗时审计与质量诊断。
- `analysis_scripts/`：产生上述报告的本机分析脚本及本次导出脚本。
- `manifest.json`、`SHA256SUMS`：条件索引、复制文件的原路径及哈希、全包校验。

从仓库根目录运行，仅需Python标准库：

```bash
python3 cdapc/pope_coco_k4_l256/verify.py
```

验证覆盖全包文件哈希、全部原始prediction/score成员哈希、输入覆盖、
生成源码指纹、commit log重建完整画布、安全EOS、每轮两次前向，
以及9000行的官方和Judge各split指标重算。

报告和配置中的绝对路径是历史来源；读取已上传数据应使用
`records_index.json` 的相对archive/member路径。历史分析脚本保留本机布局，
在其他机器重跑分析时需据索引适配路径；验证脚本不依赖原机器或模型。
模型权重、图像、Judge HTTP缓存、凭据和中间推理快照均未打包。

## 诊断结论与边界

恢复原候选相对前缀版Accuracy提高1.03个百分点，图像配对95%区间
[+0.49,+1.57]个百分点，但生成耗时约3.05倍。
解析损失和明确的错误Yes回答同时存在，见
[`reports/pope_coco_quality_diagnosis.md`](reports/pope_coco_quality_diagnosis.md)。
Native/VCD/VIGOR外部think整轮记录当时未导出，
192题Native提示词诊断的预算和提示不同，不是隔离算法的因果对照。
原始参考诊断已在仓库 `baselines/records/pope_prompt_diagnostic/`。

本补充包不含A-OKVQA/GQA运行结果；其正在执行的队列未因上传而改变。
