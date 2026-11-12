# 新prompt快速EOS：COCO / A-OKVQA / GQA

官方MMaDA think指令＋原问题，不加Final answer要求；r70/W16/K4/L256/I128。
每个数据集500张图、9000行；Acc/F1均为三个split等权平均。

| 数据集 | 唯一输入 | 官方Acc / F1 | DeepSeek Acc / F1 |
|---|---:|---:|---:|
| COCO | 5127 | 75.99% / 77.77% | 76.20% / 77.98% |
| A-OKVQA | 5145 | 74.63% / 76.83% | 74.60% / 76.91% |
| GQA | 5171 | 76.74% / 78.44% | 76.83% / 78.54% |

A-OKVQA和GQA全量输出哈希、生成指纹、分母、逐split及平均Acc/F1已核验。
预算耗尽、缺答和歧义始终保留分母。API错误为0；请求或格式重试次数保留在各组JSON中。
COCO沿用其冻结seed 20261105；A-OKVQA/GQA沿用各自原配置seed 42和数据版本。
旧prompt输出保持不变；本队列不自动重启其他实验。

- aokvqa: generation={'success': 5145}; judge={'ambiguous': 10, 'answered': 5129, 'missing': 6}; 平均生成14.87秒/输入。
- gqa: generation={'success': 5170, 'budget_exhausted': 1}; judge={'ambiguous': 12, 'answered': 5157, 'budget_exhausted': 1, 'missing': 1}; 平均生成15.10秒/输入。

详情：各数据集目录下official.json、deepseek.json、inventory.json和summary.md；
审计：completion_verification.json与final.sha256。
