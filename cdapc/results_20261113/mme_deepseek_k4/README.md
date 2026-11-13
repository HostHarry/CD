# MME K4：DeepSeek 答案提取评分（2026-11-13）

对已完成的 CDAPC think K4、直答 K4 做补充评分，不重新生成回答。
两组均为 MMaDA-8B-MixCoT，L256/I128、r70、W16/K4、快速 EOS，
各覆盖 2,374 题、1,187 个图像对、14 类。原官方评分见
[`results_20261112`](../../results_20261112/README.md)；本目录保留该次生成的逐题输入及来源哈希。

| 条件 | 解析方式 | Perception /2000 | Cognition /800 | 总分 /2800 |
|---|---|---:|---:|---:|
| think K4 | DeepSeek | 1098.23 | 293.57 | 1391.80 |
| think K4 | 官方规则 | 1018.18 | 252.50 | 1270.68 |
| 直答 K4 | DeepSeek / 官方规则一致 | 987.94 | 258.93 | 1246.86 |

DeepSeek 下 think − 直答总分为 **+144.94**，95% 配对 bootstrap 区间
**[+44.00, +242.83]**。按类别分层、以图像对为单位重采样 2,000 次，seed 42。
think 比官方解析新增 91 道正确答案，其中 90 道原来无法解析；没有正确答案变错。
直答逐题判断与官方规则一致。

## 协议与失败处理

使用 `pope_text_llm_judge_v1`、`deepseek-flash`、temperature 0：
API 只接收问题和完整的首 EOS 归一化回答，不提供图像或标准答案，不要求判断事实真伪。
提取模型表达的 Yes/No 后，在本地比对标准答案，按 MME 每类
`100 × (Acc + Acc+)` 计算感知、认知和总分。这是补充解析口径，不替代官方结果。

缺答、歧义和生成失败均保留在固定分母中计错。think 最终有 20 道缺答、
1 道歧义、6 道预算耗尽；直答有 1 道缺答。
API/格式失败会重试，并阻止最终评分发布，不混作模型答错。
4,403 个唯一请求最终全部成功。一条请求的证据引用反复改动原文拼写，
额外重试时仅将证据长度约束为至多 12 字符，答案提取规则和逐字引用校验不变；
所有实际请求、响应及失败记录均保留。报告中的缓存命中数 4,402 是最后一次续跑时的命中数，
包含此前本次任务已完成的请求，并非开始评估前已有的缓存。

## 文件

- `reports/deepseek.md`：汇总、分项分数与配对区间。
- `reports/deepseek.json`：逐题判断、证据、请求哈希、原官方判断及统计。
- `reports/deepseek_requests.json`：去重后的 API 请求、响应和重试记录，不含认证信息。
- `inputs/{think_k4,direct_k4}/`：原官方逐题结果、配置、预检、生成源码快照和预测文件清单。
- `data/mme/rows.json`：完整评测行及本地使用的标准答案，不作为 API 输入。
- `scripts/`、`src/`、`upstream/mme/calculation.py`：实际评分脚本、所用库源码及官方 MME 评分器。
- `export_manifest.json`：本次导出文件的原路径、SHA256 和大小。
- `SHA256SUMS`：本上传目录内所有其他文件的相对路径校验清单。

文件均从原工作区按字节复制。`scripts/judge.py` 是实际执行脚本，保留原工作区布局、
共享缓存和凭据加载位置；异地再次调用 API 需配置对应路径及自己的凭据。
审计已有分数无需调用 API：用 `reports/deepseek.json` 的逐题 `prediction`
对照 `data/mme/rows.json`，并使用 `scripts/reference_score_mme.py` 的 `metrics` 函数；
配对区间使用 `scripts/run.py` 的 `paired_bootstrap`。
原 `reports/deepseek.sha256` 保留本机绝对路径；下载后应校验本目录的相对路径清单：

```sh
sha256sum -c SHA256SUMS
```
