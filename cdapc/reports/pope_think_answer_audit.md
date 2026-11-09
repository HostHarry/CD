# POPE think：缺答案与解析拒绝审计

审计覆盖已完成的 A-OKVQA think r90、GQA think r90/r70，共 15,487 个唯一输入。使用 first_eos_v1 实际送评分的完整文本；不把 EOS 后未展示的画布算作答案。先用固定规则分类，再逐例核对 35 条特殊输出。提取规则不使用 gold；提取后才与 gold 比较。没有改动原始解析器、评分文件或实验队列。

结论：未解析不等于没有答案。多数未解析记录中有可辨认的 Yes/No，但标签未闭合、答案带解释、标签损坏或答案只出现在 think 内。同时也有大量只有标签的空壳，必须继续记为无答案。

以下按每组 9,000 条评测记录计数，与运行报告的未解析率同口径。跨子集复用输入按原权重计数；完整 JSON 同时保存唯一输入计数。

| 未解析类型 | A-OKVQA r90 | GQA r90 | GQA r70 |
|---|---:|---:|---:|
| 标签空壳，确实无答案 | 1104 | 1221 | 2111 |
| 仅 Yes/No 加标签，结构不完整 | 1649 | 1558 | 1760 |
| 正常闭合后的 Yes/No 带解释 | 558 | 651 | 1030 |
| 其他明确答案：标签错误、正文结论、自然语言 | 18 | 30 | 294 |
| 残片、无法可靠确定或冲突 | 3 | 1 | 44 |
| 原规则未解析合计 | 3332 | 3461 | 5239 |

| 口径 | A-OKVQA r90 | GQA r90 | GQA r70 |
|---|---:|---:|---:|
| 原规则 Accuracy % | 42.97 | 44.60 | 30.57 |
| 仅允许正常闭合后 Yes/No 带解释 % | 47.98 | 50.78 | 39.96 |
| 再接受只有标签加 Yes/No 的明确回答 % | 62.57 | 63.89 | 54.79 |
| 再接受明确正文答案/可辨标签错误 % | 62.69 | 64.10 | 56.94 |
| 再加入逐例核对的自然语言答案 % | 62.70 | 64.18 | 57.24 |
| 对应直答 Accuracy % | 78.29 | 78.94 | 79.06 |

后四行是离线审计口径，不替代原来的最终答案格式评分。接受 think 内回答后，评测目标从“合规的最终回答”扩展为“文本里可辨认的回答”。无答案、无法确定及冲突仍计错，不随机猜测，也不把没有 Yes 自动当作 No。即使采用最宽的审计口径，结果仍低于对应直答；格式限制解释了部分降幅，但不能把全部问题归给解析器。

解析器的两个关键限制位于 `src/cdapc_lab/evaluation/parsers.py::reasoned_yes_no`：打开 think 后没有正确关闭，直接返回 unclosed_think；最终区域必须完整匹配 `yes|no` 加可选句号/感叹号，附加解释会匹配失败。因此 `</think> Yes, there is a toilet in the image.` 也被标成未解析。

全部标签空壳都有 EOS；可见文本中确实没有答案词。另有 GQA r70 的一个唯一输入（2 条评测记录）输出`Yes, there is no person in the image.`，含相反判断，保留为冲突未解析。

## 唯一输入计数

| 类别 | A-OKVQA r90 | GQA r90 | GQA r70 |
|---|---:|---:|---:|
| 只有 think 标签/空文本，确实无答案 | 683 | 733 | 1327 |
| 仅 Yes/No 加标签，但格式不完整 | 1065 | 1034 | 1116 |
| 正常闭合后明确 Yes/No，因附加文本被拒绝 | 241 | 273 | 446 |
| 标签或多余符号异常，但可辨最终 Yes/No | 2 | 1 | 58 |
| 明确 Yes/No 结论仅在 think 正文中 | 8 | 9 | 74 |
| 最终句用自然语言明确回答，未用独立 Yes/No | 1 | 4 | 6 |
| 正文明确表达目标物体有/无 | 1 | 0 | 12 |
| 只有无明确判断的残片 | 1 | 1 | 19 |
| 有内容，但未对所问物体给出可靠可提取的判断 | 2 | 0 | 4 |
| 答案存在冲突，无法唯一提取 | 0 | 0 | 1 |
| 未解析唯一输入合计 | 2004 | 2055 | 3063 |
| 全部唯一输入 | 5145 | 5171 | 5171 |

## 原文案例

### pope_aokvqa_reasoning_r90 / 1c09f4a195dbc4e56d014ced165512caefd995e06722ea1f5d6ed2f747d06a89

Question: Is there a tv in the image?
Gold: no; 分类: 仅 Yes/No 加标签，但格式不完整; 审计提取: no; 原解析原因: unclosed_think

```text
<think>No
```

### pope_gqa_reasoning_r70 / b60638be3419447db6615efd854def053b907ea89fc81399c514bdc224a96bc6

Question: Is there a toilet in the image?
Gold: yes; 分类: 正常闭合后明确 Yes/No，因附加文本被拒绝; 审计提取: yes; 原解析原因: no_unique_final_answer

```text
<think> The image shows a bathroom with a tiled floor, a toilet, and a trash can. The toilet is clearly visible in the center of the image. </think> Yes, there is a toilet in the image.
```

### pope_gqa_reasoning_r70 / 112fee29f9d24cf8952cff8c794122c427dbca3e1076a5c9f8379d9e3ad8be17

Question: Is there a fence in the image?
Gold: no; 分类: 只有 think 标签/空文本，确实无答案; 审计提取: None; 原解析原因: unclosed_think

```text
<think>
```

### pope_gqa_reasoning_r70 / 3b640d37bfd5a6ce2a018e10e15d7b0082c856600644c58b00c64502590270a7

Question: Is there a person in the image?
Gold: no; 分类: 答案存在冲突，无法唯一提取; 审计提取: None; 原解析原因: no_unique_final_answer

```text
<think> Yes, there is no person in the image. </think> Yes, there is no person in the image.
```

## 复现与文件

运行 `.venv/bin/python scripts/pope_think_answer_audit.py`。
脚本逐条验证原解析结果和评分，并核对 9,000 条加权准确率/未解析率与原报告一致。

- `reports/pope_think_answer_audit.json`：所有输出、逐条分类、权重、原文案例和不同口径结果。
- `reports/pope_think_answer_manual.json`：35 条特殊输出的判定、理由及原文 SHA256。
- `reports/pope_think_answer_review.json`：尚待核对的记录；本轮为空。
- `reports/pope_think_answer_audit.sha256`：文件校验清单。
