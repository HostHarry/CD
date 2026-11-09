# CD-APC 逐题输出文本

这里是全部已完成实验的 **逐样本规范化回答文本**，由 `cdapc-experiments/scripts/export_cdapc_upload.py` 离线导出：只读取已保存的 run 清单、预测、评分、Judge 输出与报告 JSON，不重新推理、不调用 API、不改写文本。未完成的 POPE think K4 L256/I128 实验与已取消的 `pope_aokvqa_reasoning_r50` 不在本目录。

## 条件与文件对照

| 条件 | JSONL | 可读文本 | 唯一输入 | 评测记录 |
| --- | --- | --- | --- | ---: |
| MMMU cdapc_r70 L256/I128 | [mmmu_cdapc_r70_L256_I128.jsonl](mmmu_cdapc_r70_L256_I128.jsonl) | [.txt](mmmu_cdapc_r70_L256_I128.txt) | 900 | 900 |
| POPE-COCO 直答 r90 | [pope_coco_direct_r90.jsonl](pope_coco_direct_r90.jsonl) | [.txt](pope_coco_direct_r90.txt) | 5,127 | 9,000 |
| POPE-COCO 直答 r80 | [pope_coco_direct_r80.jsonl](pope_coco_direct_r80.jsonl) | [.txt](pope_coco_direct_r80.txt) | 5,127 | 9,000 |
| POPE-COCO 直答 r70 | [pope_coco_direct_r70.jsonl](pope_coco_direct_r70.jsonl) | [.txt](pope_coco_direct_r70.txt) | 5,127 | 9,000 |
| POPE-COCO 直答 r60 | [pope_coco_direct_r60.jsonl](pope_coco_direct_r60.jsonl) | [.txt](pope_coco_direct_r60.txt) | 5,127 | 9,000 |
| POPE-COCO 直答 r50 | [pope_coco_direct_r50.jsonl](pope_coco_direct_r50.jsonl) | [.txt](pope_coco_direct_r50.txt) | 5,127 | 9,000 |
| POPE-AOKVQA 直答 r90 | [pope_aokvqa_direct_r90.jsonl](pope_aokvqa_direct_r90.jsonl) | [.txt](pope_aokvqa_direct_r90.txt) | 5,145 | 9,000 |
| POPE-AOKVQA 直答 r70 | [pope_aokvqa_direct_r70.jsonl](pope_aokvqa_direct_r70.jsonl) | [.txt](pope_aokvqa_direct_r70.txt) | 5,145 | 9,000 |
| POPE-AOKVQA 直答 r50 | [pope_aokvqa_direct_r50.jsonl](pope_aokvqa_direct_r50.jsonl) | [.txt](pope_aokvqa_direct_r50.txt) | 5,145 | 9,000 |
| POPE-GQA 直答 r90 | [pope_gqa_direct_r90.jsonl](pope_gqa_direct_r90.jsonl) | [.txt](pope_gqa_direct_r90.txt) | 5,171 | 9,000 |
| POPE-GQA 直答 r70 | [pope_gqa_direct_r70.jsonl](pope_gqa_direct_r70.jsonl) | [.txt](pope_gqa_direct_r70.txt) | 5,171 | 9,000 |
| POPE-GQA 直答 r50 | [pope_gqa_direct_r50.jsonl](pope_gqa_direct_r50.jsonl) | [.txt](pope_gqa_direct_r50.txt) | 5,171 | 9,000 |
| POPE-AOKVQA think r90 | [pope_aokvqa_think_r90.jsonl](pope_aokvqa_think_r90.jsonl) | [.txt](pope_aokvqa_think_r90.txt) | 5,145 | 9,000 |
| POPE-AOKVQA think r70 | [pope_aokvqa_think_r70.jsonl](pope_aokvqa_think_r70.jsonl) | [.txt](pope_aokvqa_think_r70.txt) | 5,145 | 9,000 |
| POPE-GQA think r90 | [pope_gqa_think_r90.jsonl](pope_gqa_think_r90.jsonl) | [.txt](pope_gqa_think_r90.txt) | 5,171 | 9,000 |
| POPE-GQA think r70 | [pope_gqa_think_r70.jsonl](pope_gqa_think_r70.jsonl) | [.txt](pope_gqa_think_r70.txt) | 5,171 | 9,000 |
| POPE-GQA think r50 | [pope_gqa_think_r50.jsonl](pope_gqa_think_r50.jsonl) | [.txt](pope_gqa_think_r50.txt) | 5,171 | 9,000 |
| CHAIR r90 | [chair_r90.jsonl](chair_r90.jsonl) | [.txt](chair_r90.txt) | 500 | 500 |
| CHAIR r70 | [chair_r70.jsonl](chair_r70.jsonl) | [.txt](chair_r70.txt) | 500 | 500 |
| CHAIR r50 | [chair_r50.jsonl](chair_r50.jsonl) | [.txt](chair_r50.txt) | 500 | 500 |
| GQA think K=4 pilot | [gqa_think_k4_pilot.jsonl](gqa_think_k4_pilot.jsonl) | [.txt](gqa_think_k4_pilot.txt) | 306 | 540 |
| GQA think K=2 pilot | [gqa_think_k2_pilot.jsonl](gqa_think_k2_pilot.jsonl) | [.txt](gqa_think_k2_pilot.txt) | 306 | 540 |
| MME r70 L256/I128 | [mme_r70_L256_I128.jsonl](mme_r70_L256_I128.jsonl) | [.txt](mme_r70_L256_I128.txt) | 2,374 | 2,374 |

K pilot 的 K16 基线复用已完成的 `pope_gqa_think_r70` 全量运行：把 [pope_gqa_think_r70.jsonl](pope_gqa_think_r70.jsonl) 按 `reports/k_commit_pilot/rows.json` 中的 306 个 `input_key` 过滤即可得到同一子集的 K16 输出，本目录不重复导出。

## 字段说明

所有 JSONL 每行一条记录，`text` 为规范化回答原文。

- `sample_id`：唯一输入 ID。POPE 为 `input_key`（图像 SHA256 + 带后缀问题的哈希），CHAIR 为 `chair:<image_id>`，MMMU 为 `validation_*`，MME 为 `mme:<group>:<idx>`。
- `group_id` / `image` / `image_sha256`：图像分组与文件名、内容哈希。
- `question`：送入模板的原始问题；`prompt`：实际拼接后的完整 user 文本（POPE 直答为 `{question} Please answer yes or no.`；POPE think 为 MMaDA 思考指令 + 换行 + `{question}\nAfter </think>, end with exactly Final answer: Yes or Final answer: No.`；CHAIR 为 `Please describe this image in detail.`；MME 为官方题目原文）。
- `text`：规范化回答。POPE/MMMU 逐字复制各 run 评分文件的 `normalized.text`；CHAIR/MME 由 `normalize_output` 对保存的 `raw_response_ids` 重算并与已存产物逐条断言相等。
- `generation_status` / `decode_iterations` / `emitted_tokens_before_eos` / `finish_reason`：生成状态、迭代步数、首个 EOS 前 token 数、结束原因，来自预测文件。
- `source_run`：来源 run 目录（相对各自工作区根）。

### POPE 行（直答 / think / K pilot）

- `rows[]`：该输入映射到的评测记录，含 `row_id`（`split:question_id`）、`split`、`question_id`、`label`、`scores`。
- 直答 `scores.official`：官方 Yes/No 重写规则在该行标签上的 0/1（与报告 `cdapc.splits` 一致）；另有 `official_prediction`、`strict_status`、`strict_prediction`、`strict_correct` 顶层字段。
- think 与 K pilot `scores.strict` / `scores.judge`：严格最终答案解析与 LLM Judge 在该行标签上的 0/1；顶层另有 `strict_*`、`judge_prediction`、`judge_status`、`judge_correct`、`judge_evidence`、`judge_request_sha256`。K pilot 另有 `commit_cap` 与 `mechanism`（提交/步、batch 直方图、尾部 EOS 轮次等）。

### MMMU 行

`gold`、`question_type`、`judge_correct`（LLM judge 0/1）、`official_correct`（官方评测器，含随机 fallback）、`strict_correct` / `strict_status`（严格轨，不猜答案）、`method` 固定为 `mmmu_cdapc_r70_L256_I128`。

### CHAIR 行

`image_id`、`metrics`（该图在 pinned CHAIR 评测器下的 `CHAIRs` / `CHAIRi` / `Recall`，逐图复制自 `reports/chair_r*.json` 的 `sentences`）。

### MME 行

`category`、`label`（官方 Yes/No 答案）、`prediction`（官方 `parse_pred_ans` 结果；预算耗尽为 `other` 计错）、`group_id` 为图像问题对。

## 归一化规则

沿用原评分器：只取首个 EOS 前的 response tokens，`skip_special_tokens=True` 解码并去除外侧空白（`cdapc_lab.evaluation.normalize.normalize_output` / `first_eos_v1`）。POPE think 的严格分数只看最后一个 `</think>` 之后（可含 `Final answer:` 标记）的区域；无法解析计错。文本审计口径见 `reports/pope_think_answer_audit.*`，不属于本目录字段。

## 校验

导出脚本对每个条件重新计算指标并与已存报告逐项比对（POPE 各子集 Accuracy/F1/Precision/Recall、CHAIRs/CHAIRi/Recall、MMMU Judge/Official/Strict、MME 逐类 Acc/Acc+ 与总分、K pilot 全部聚合指标），全部相等才写出文件；同时断言 run 清单 complete（预算受限 run 允许 `incomplete` 但全部样本已结算）、预测状态合法、ID 与 manifest 一致、无重复、Judge 请求哈希与缓存解析一致。逐条件结果见 `manifest.json` 的 `conditions` 与导出脚本输出。

## 原始记录位置

每条记录的 `source_run` 指向原实验服务器上的 run 目录：

```text
<source_run>/predictions/<sha256(sample_id)>.json   raw_response_ids / commit_log
<source_run>/scores/<evaluation_fingerprint>/<sha256(sample_id)>.json   normalized.text 等评分字段
<source_run>/judge/<hash>/<sha256(sample_id)>.json   MMMU judge 分数
reports/pope_think_llm_judge/pope_*_reasoning_r*.json   POPE think Judge 逐输入结果
reports/k_commit_pilot/results.json                  K pilot 逐输入结果与缓存键
```

CHAIR 与 MME 不写 `scores/` 目录：前者由 `scripts/score_chair_runs.py` 直接从预测评分（产物为 `reports/chair_r*_answers.json` 与报告 `sentences`），后者由 `cdapc-mme/scripts/score_mme.py` 直接评分（产物为 `reports/mme_official_text/`）。各 run 与评分指纹的对应关系见 `manifest.json` 的 `conditions.<name>.evaluations`。
