# MMMU r70 256/128 与 POPE think LLM Judge

本轮新增两个独立任务。已有推理进程与历史分数保留。

## MMMU r70

- 数据：MMMU validation 全部 900 题，3 个互斥分片，各 300 题。
- 模型、图像输入、seed、prompt、EOS 规则沿用现有 MMMU r90。
- `canvas_length=256`、`iteration_limit=128`、`budget_limited=true`。
- `r_gate.threshold=0.7`；其余为 `alpha=0.5, beta=0.1, u=0.1, W=K=16`。
- `stop_rule=eos_prefix_complete`；到 128 次迭代仍未完成时记录 budget_exhausted。
- 配置：`configs/experiments/mmada_mmmu_cdapc_r70_L256_I128_shard{0,1,2}.yaml`。

旧 MMMU r90/r50 是 L256/I256，Native 是 L256/128 steps。新增条件同时改变
r 阈值与 CDAPC 迭代预算，不能仅将其分差归因于 r70。

运行 `scripts/run_mmmu_r70_256_128.py`。控制器等待现有 CHAIR/POPE 队列完成，
通过同一个 GPU 队列锁领取三张 GPU，支持已有同配置 worker 接管和断点恢复。
随后自动执行 Official/Strict、现有 MMMU DeepSeek Judge，并生成独立结果报告。
MMMU Judge 的 API 错误可以重试，保留之前的失败记录。

- 状态：`reports/mmmu_r70_256_128_runs.json`
- 报告：`reports/mmmu_cdapc_r70_L256_I128.json`（完成后）
- 日志：`logs/mmmu/r70_L256_I128_driver.log` 及分片日志
- 最终输出与提交日志：状态文件中各 `run` 指向的 predictions/scores 目录

### 2026-11-09 恢复记录

迁移后的主机内核从 5.15.0-94 变为 5.15.0-119，GPU 报告的可用总显存也有变化，
因此生成指纹不同。控制器保留原来的 313 条预测及原 run，通过状态文件的
`previous_runs` 记录来源，并在当前环境下按原配置新建 run、重新生成全部 900 题。
不修改旧预测的指纹，也不把两个环境下的部分输出拼成同一个 run。

`budget_limited=true` 时，整组输入均产生 success 或 budget_exhausted 后允许进入评分。
预算耗尽样本保留原状态，Official、Strict、Judge 均计零且保留在分母中；
其耗时、迭代数和停止原因计入统计，避免队列因正常预算结束而反复重试。

## POPE think LLM Judge

原计划覆盖 A-OKVQA、GQA 的 r90/r70/r50。2026-11-08 用户取消了尚未生成的
A-OKVQA think r50；当前保留五组，共 25,803 条唯一生成输入、45,000 条评测记录。
控制器自动接入后续完成的 think 条件，取消的条件不计入分数，也不阻塞后续队列。
已完成的 GQA r50 及其 Judge 结果保留。

独立协议 `pope_text_llm_judge_v1`，API 模型 `deepseek-flash`，
temperature=0、thinking disabled、max_tokens=192。请求仅包含题目及
first_eos_v1 的完整可见输出，不提供图片和 gold。

裁判返回严格 JSON：

```json
{"answer": "yes", "status": "answered", "evidence": "Yes"}
```

`answer` 可为 yes/no/null，`status` 为 answered/missing/ambiguous。
有答案时必须提供原文连续引用。若格式或引用不合要求，用验证反馈请求修正，
每次实际请求和原始响应都保留。API 错误与格式错误不当作模型答错。

判定优先级：

1. 优先读取明确最终答案；明确结论可覆盖前文推理。
2. 无外部答案时，允许读取 think 内明确回答，包括未闭合的 `<think>No`。
3. 允许明确说明目标物体存在或不存在的自然语言回答。
4. 标签空壳、无结论残片、无关重复记 missing，不通过“没提到物体”推断 No。
5. 最终答案自身冲突，例如 `Yes, there is no person`，记 ambiguous。

模型提取之后才在本地与 gold 比较。missing/ambiguous 计错，API 错误保持待处理。
保存 Accuracy、F1、Precision、Recall、Yes ratio、缺答/冲突数，以及对原严格评分
的逐题转移。指标使用原 9,000 条记录的权重；只在整组完成后发布宏平均主分数。
这衡量回答中表达的答案，不验证读图或推理过程是否正确。

按完整请求内容 SHA256 复用完全相同的请求，包括跨样本/阈值的重复文本。
相同裁判结果仍逐样本与各自 gold 比较，并按各子集原始记录计分。
因此 API 请求数、唯一生成输入数和 9,000 条评分记录数分别统计。

运行 `scripts/run_pope_think_judge.py --workers 8`；支持缓存恢复。
`--once` 只处理当时已完成条件，`--limit N` 用于接口检查。

- 状态：`reports/pope_think_llm_judge/index.json`
- 固定协议：同目录 `judge_manifest.json`
- 请求及所有响应：同目录 `cache/<request_sha256>.json`
- 逐条件结果：同目录 `pope_{gqa,aokvqa}_reasoning_r{90,70,50}.json`
- 日志：`logs/pope_think_judge_queue.log`

保留旧 POPE 严格分数、规则/逐例文本审计分数；LLM Judge 是第三个独立口径。
每个 LLM 条件完成及 MMMU 新组完成后，自动刷新 `reports/experiment_summary.md`
和 JSON 汇总；未完成的 LLM 判分不填入最终 Accuracy 列。
