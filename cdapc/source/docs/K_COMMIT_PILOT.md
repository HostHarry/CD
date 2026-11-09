# GQA think r70 的 K 提交上限对照

本轮为探索性配对试验，检验减小单轮提交上限是否改善答案与输出结构。
它不是全量 POPE 分数，不用本试验分数与此前全量 9,000 条分数直接比较。

- 数据：GQA 500 张图像中，以 seed=20261108 均匀抽取 30 张图。
- 样本：保留这些图的全部问题，共 306 个唯一输入、540 条评测记录。
- 三个子集各 180 条，均为 90 Yes、90 No。选图不依据输出或 gold，选定后校验标签平衡。
- 生成 seed=42，保留原始 GPU 分片归属（101/103/102 个输入）。
- 条件：K16/K4/K2。固定 r=0.7、W=16、L=128、iteration_limit=128；
  模型、提示词、u/alpha/beta、EOS 规则、采样输入等完全相同。
- K16 复用同题的已有结果；K4/K2 重新生成。配置校验只允许 K、运行名和
  子集 manifest 路径变化；评分前逐题核对 prompt IDs、输入指纹和 sample seed。

运行 `scripts/run_k_commit_pilot.py --prepare` 可重验配置和冻结的选样。
不带参数则启动持久控制器。它等待已排队的 MMMU r70 生成结束，再使用相同 GPU
队列锁，依次运行 K4、K2。支持现有 worker 接管和已完成结果恢复。
GPU 生成完成即释放 GPU 锁；评分阶段与原 POPE Judge 控制器串行共享缓存锁。

Judge 使用原 `pope_text_llm_judge_v1`，读取完整文本，不提供 gold 或图片。
缺答、冲突计错，API/格式错误保持待处理。所有请求和响应引用原共享 cache。
原严格最终答案分数、Judge 分数、空标签壳比例、耗时、步数、实际提交数、
fallback、切换次数、停止原因、仅填已知 EOS 后区域的轮次分别保存。

Accuracy/F1 采用该子集的 540 条原始记录权重；生成成本和输出结构按 306 个
唯一输入统计。Accuracy 差值做 2,000 次图像级配对 bootstrap，seed=42。
同一图片的重复问题及跨采样子集记录保持在一起，不当作独立样本。
30 个图像簇的区间可能较宽；K16 的耗时来自较早运行，仅作参考。
区间为探索性的逐项区间，未做多重比较校正。

2026-11-09 恢复后，K4/K2 使用当前主机环境，K16 继续复用冻结的历史结果。
GPU 型号和模型软件版本一致，但内核版本及 GPU 报告的总显存与历史环境不同。
最终 JSON 的 `generation_environments` 保留各组运行环境；耗时不能视为严格控制
硬件环境后的 K 单因素对照，质量差异也需带上这项环境限制。

文件：

- `reports/k_commit_pilot/protocol.json`：冻结选样、参数与选样校验。
- `reports/k_commit_pilot/rows.json`：实际评测行。
- `reports/k_commit_pilot/index.json`：排队与各分片运行状态。
- `reports/k_commit_pilot/baseline_records.json`：本次复用的 K16 逐题记录。
- `reports/k_commit_pilot/results.json` / `results.md`：全部完成后的对照结果。
- `logs/k_commit_pilot/`：控制器与分片日志。

解释：同一轮提交都依据旧画布。降低 K 可以增加后续重新观察的机会，但也可能
让高分 EOS 占掉名额，延长正文等待；只根据当前 K=16 结果无法预判净效果。
