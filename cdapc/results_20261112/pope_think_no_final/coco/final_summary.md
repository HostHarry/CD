# COCO新prompt全量验证与VCD对比

2026-11-11完成生成、官方评分、DeepSeek评分及VCD全量配对比较。
共500张图、5127个唯一输入、9000行；三个split各3000行且Yes/No各1500行。
以下Acc/F1均为三个split等权平均。

## 同一快速EOS策略：仅删除Final answer要求

两组均保留官方MMaDA think指令，参数r70/W16/K4/L256/I128、seed 20261105不变。
候选限制在已知最早EOS之前，前缀完整才停止。带后缀控制组复用已有结果。

| Prompt | 官方Acc / F1 | DeepSeek Acc / F1 |
|---|---:|---:|
| think＋问题＋Final answer要求 | 68.20% / 73.51% | 70.56% / 75.67% |
| think＋问题 | **75.99% / 77.77%** | **76.20% / 77.98%** |

去后缀的图像配对效应（2000次bootstrap，seed=42）：

- 官方Acc：+7.79个百分点，95%区间[+6.51, +9.18]；F1：+4.26，[+3.15, +5.44]。
- DeepSeek Acc：+5.64个百分点，[+4.33, +6.96]；F1：+2.31，[+1.08, +3.49]。

不包含先前64图pilot的新增436张图（4466输入、7848行）：

| 评分 | 带后缀Acc / F1 | 去后缀Acc / F1 | ΔAcc [95% CI] |
|---|---:|---:|---:|
| 官方 | 68.72% / 73.91% | 76.44% / 78.19% | +7.72 [+6.33, +9.15] |
| DeepSeek | 70.90% / 75.88% | 76.63% / 78.40% | +5.73 [+4.29, +7.20] |

改善在新增部分仍存在。提示协议根据64图探索选择；不称为完全未经测试集分析的独立benchmark。
历史“裸问题”实验没有think指令，与此次条件不同。

## 与VCD对齐完整9000行，统一DeepSeek

VCD使用CD@fcc76d6的完整输出，两种方法均为think指令＋原问题，无额外Final answer要求。
所有行的图像SHA、问题、标签已核验；两种方法保留各自配置和seed协议。

| Split | CDAPC Acc / F1 | VCD Acc / F1 |
|---|---:|---:|
| Random | 79.57% / 80.46% | 78.73% / 79.39% |
| Popular | 75.77% / 77.65% | 76.30% / 77.59% |
| Adversarial | 73.27% / 75.84% | 72.40% / 74.77% |
| 平均 | **76.20% / 77.98%** | **75.81% / 77.25%** |

CDAPC−VCD：Acc +0.39个百分点，配对95%区间[-0.92, +1.67]；
F1 +0.74，区间[-0.48, +1.91]。两者差距区间均跨0，目前表现接近，不能确认稳定领先。
CDAPC独对804行、VCD独对769行，净多对35行。

此前6951行部分报告保持不变。补齐余下No题后，Acc领先由部分宏均值的1.24点收窄到0.39点。

## 格式、耗时与审计

- CDAPC闭合`</think>`：4397/5127（85.76%）→5081/5127（99.10%）。
- 官方Yes比例：70.00%→57.73%；负例误报Yes行数：2331→1428。
- 平均生成时间：9.14→14.58秒/输入；平均迭代：25.21→40.34。
- 新prompt生成：5126成功，1预算耗尽；该失败始终保留分母计错。
- 新prompt DeepSeek：5117 answered、8 ambiguous、1 missing、1 budget_exhausted。
- DeepSeek使用既有`pope_text_llm_judge_v1`/`deepseek-flash`协议；不给图像或gold，答案提取后在本地比标签。API错误为0。
- 661个pilot输出逐字节复用；源快照、完整分母、独立Acc/F1复算及跨报告缓存一致性均通过。
- 旧prompt的A-OKVQA/GQA后续队列已取消，历史输出保留；本轮结束后三张GPU已释放。

来源与复现文件：

- `/root/autodl-tmp/cdapc-pope-coco-prefix-no-final/reports/comparison.json`
- `/root/autodl-tmp/cdapc-pope-coco-prefix-no-final/reports/official_scopes.json`
- `/root/autodl-tmp/cdapc-pope-coco-prefix-no-final/reports/fast_official_deepseek.json`
- `/root/autodl-tmp/cdapc-pope-coco-prefix-no-final/reports/completion_verification.json`
- `/root/autodl-tmp/cdapc-pope-vcd-full-judge/reports/comparison.json`
