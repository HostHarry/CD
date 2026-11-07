# MMMU 逐题输出文本

这里是已经完成的三组实验的 **2,700 条模型回答，每组 900 条**，包括推理文本与最终答案。

| 方法 | 直接阅读原文 | 程序读取 |
| --- | --- | --- |
| Native | [native.txt](native.txt) | [native.jsonl](native.jsonl) |
| CDAPC r90 | [cdapc_r90.txt](cdapc_r90.txt) | [cdapc_r90.jsonl](cdapc_r90.jsonl) |
| CDAPC r50 | [cdapc_r50.txt](cdapc_r50.txt) | [cdapc_r50.jsonl](cdapc_r50.jsonl) |

`timing_only_crossed` 是分析报告中建议的后续对照，尚未运行 MMMU 实验，因此本目录没有该组输出。

## 文本与字段

- JSONL 的 `text` 字段逐字复制原评分文件的 `normalized.text`，没有重新生成、改写或筛选。
- 文本保留 `<think>...</think>` 和模型实际写出的答案；只输出 `<think>`、标签不完整、内容重复等情况也完整保留。
- 归一化沿用原实验：只取首个 EOS 前的 response tokens，解码时跳过特殊 token，再去除外侧空白。
- TXT 文件在每条回答前增加题目 ID、gold、三个评分、步数和 token 数，正文与 JSONL 的 `text` 相同。
- `judge_correct` 为已有 LLM judge 的 0/1 结果；`official_correct` 与 `strict_correct` 为已有规则评分，`strict_status` 标注规则解析是否成功。
- `decode_iterations` 是实际迭代次数；`emitted_tokens_before_eos` 是首个 EOS 前的 token 数。
- 三组使用相同题目顺序，可以通过 `sample_id` 对齐。

例如，查看同一道题的三个回答：

```python
import json
from pathlib import Path

sample_id = "validation_Basic_Medical_Science_5"
for method in ("native", "cdapc_r90", "cdapc_r50"):
    path = Path("mmmu/answers") / f"{method}.jsonl"
    with path.open(encoding="utf-8") as stream:
        rows = (json.loads(line) for line in stream)
        row = next(row for row in rows if row["sample_id"] == sample_id)
    print(method, "gold:", row["gold"], "judge:", row["judge_correct"])
    print(row["text"])
```

可以在 TXT 文件中直接搜索以下题目 ID：

- `validation_Basic_Medical_Science_5`：r90 输出 92 tokens、运行 239 步的尾部 EOS 等待案例。
- `validation_Clinical_Medicine_21`：r50 正文大量重复 `abdominal`，末尾选项 B 判对的案例。
- `validation_Biology_6`：r50 在末尾选项处把同状态 base B 替换为 C 的案例。
- `validation_Chemistry_4`：r50 输出 64 tokens、运行 129 步的尾部等待案例。

## 原始记录位置与复现

每条 JSONL 都记录了 `source_run`。在原实验服务器上：

```text
<source_run>/scores/<evaluation_fingerprint>/<sha256(sample_id)>.json
    normalized.text  ← 本目录导出的正文

<source_run>/predictions/<sha256(sample_id)>.json
    raw_response_ids ← 完整 response token 画布
    commit_log       ← CDAPC 的逐次提交记录
```

各 run 与 evaluation fingerprint 的对应关系见本目录 `manifest.json` 中的 `run_index`。该文件也记录六个导出文件的 SHA256。

导出脚本是原项目中的 `scripts/export_mmmu_answers.py`，仓库保留其源码节选于 `../source/scripts/export_mmmu_answers.py`。重新导出需要原始 run 文件和样本 ID manifest，不需要模型推理或联网评分。
