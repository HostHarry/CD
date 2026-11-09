# 施工验收记录 · 2026-11-05

新工程已部署到 `/root/autodl-tmp/cdapc-experiments`；本地镜像为
`/Users/hostharry/科研/cdapc-experiments`。按 ponytail ultra 限定为指南要求的
框架、小样本验证和独立 CCAW 实现，没有启动正式 benchmark 或参数搜索。

当前可用协议是 **顺序执行的两个完整前向、无跨轮缓存**。合批路径的真实
BF16 决策一致性未通过，配置层已拒绝启用；不能将本次验收描述为所有执行
路径均通过。真实数据只产生 4 个有效 content-swap 配对，不能声称机制收益。

## 已执行的检查

| 检查 | 结果与证据 |
| --- | --- |
| 单元与集成测试 | 服务器当前 74 passed（新增 MMaDA 官方推理指令原文校验；`final_audit.json` 记录的是 `df31f79` 时的 73 passed）；覆盖公式、四种门开关、fallback 写回、状态、解析、续跑、指纹、受控分叉及 CCAW。最终服务器输出见 `reports/final_audit.json`。 |
| 官方 Native | MMaDA 与 LaViDa 包装输出分别等于独立官方入口，采用显式 `native_no_cache`、双向 normal attention。 |
| 正常与消融 adapter | 两模型正常 logits 与独立官方前向误差 0；替换视觉内容后消融文本 logits 误差 0；实际捕获 32 层 attention kernel 的阻断 mask；调用次序检查一致。 |
| 合批与顺序比较 | 两模型 `paired_passed=false`。MMaDA（512 输入、官方推理指令）出现 2 个 gate 差异，提交位置与写入均不同；LaViDa 还出现 argmax 和 gate 差异。生产配置只允许 sequential。 |
| 固定窗口及诊断 | 每模型 Native、full、gain_zero、timing_only_crossed、base_candidate_visual 各 2 样本，20 条预测成功；full 先运行 1 题再续跑，已完成离线评分与汇总。 |
| 无实际改选的对照 | 按 full 日志选取的 2/2 样本，两模型分别满足 full/crossed token 与成本一致。 |
| 受控共同状态分叉 | 合成数值 fixture 验证只改变一个目标位置，之后用同一 full+fixed continuation，B 分支复现原轨迹。真实覆盖率见下文。 |
| CCAW | 边界、步长、EMA、half-up、相邻轮历史、单次更新、初始 W、冻结状态候选与门均验证；两模型真实 min=max=initial=16 时 token 和成本均与 fixed 一致。 |
| 最终离线重评 | 15 个选定 run、44 条已成功预测全部重评，官方与 strict 计数保持一致，原始预测 hash 不变；旧评分保留。tokenizer/config 与生成时记录不符时明确拒绝重评。 |
| 环境 | RTX 5090 32 GB；Python 3.12.3、Torch 2.8.0+cu128、Transformers 4.50.3；`pip check` 无依赖冲突。 |

两模型详细报告：`reports/mmada_acceptance.json` 与
`reports/lavida_acceptance.json`。早期失败报告保留在 reports 中，不能把每个
文件都解释成通过；以上结果以这两个汇总和 `final_audit.json` 为准。

## 真实机制覆盖：4/16

独立预登记的 16 个 M³CoT test ID，使用 MMaDA full+fixed、L=512，保持
alpha=0.5、beta=0.1、u=0.1、r=0.9、W=K=16。ID 按固定哈希顺序选取，
没有根据预测结果挑选，也没有为触发事件放宽门槛。

- 16/16 完成。官方计分 5/16，strict 4/16（parsed 11、unparsed 5）；它们是
  诊断结果，不是方法比较。
- 加入官方推理指令后，首个 EOS 位于响应第 54–313 个 token，没有一题在第
  1–2 个 token 内结束；未加指令的 512 输入为 11/16，256 输入为 1/16。
  15/16 输出含 think 结束标签。
- 窗口内观察到 5,831/27,976 次 proposal flip，合格 flip 为 4/7,409；
  fallback 轮次 783/1,782。
- 实际写入改选为 4/8,192，first eligible flip 为 4/16（无事件 12、源预测失败 0），
  valid pairs 为 4/4，replay 不一致为 0。
- 4 对的 `paired_strict` 为两边都对 1、都错 3、差值 0（95% CI [0, 0]）。
  样本量太小，不能据此估计分叉效应，也不能写成零效应。

源 run：`runs/20261105T230723830026Z_b1e4190dc7df`，评分指纹
`b084625d16c130dd2264b06b101dfa43b1014dbebe49a7a288d59049f9e9121d`。
完整证据：`reports/mechanism16.json`、`reports/mechanism16-replay.json`。
后者另有 replay 源码、gold 与 scorer hash，明确区分无事件和源预测失败。

## 独立 CCAW 小样本比较

使用同一组 2 个预定 smoke ID、L=128，只改变窗口控制器。数值为两题合计，
生成时延经过 CUDA 同步，排除模型加载和预处理；不作为稳定测速结论。

| 模型 / 控制器 | 实际 W 范围 | 分支评估数 | 生成秒数 | official / strict |
| --- | --- | ---: | ---: | --- |
| MMaDA fixed | 16 | 268 | 47.42 | 0/2 / 0/2 |
| MMaDA ccaw_v1 | 16–44 | 304 | 54.01 | 0/2 / 0/2 |
| LaViDa fixed | 16 | 196 | 25.90 | 1/2 / 0/2 |
| LaViDa ccaw_v1 | 16–44 | 204 | 27.13 | 1/2 / 0/2 |

两模型各一题额外验证固定范围退化，token 与真实计算计数均相等。
报告为 `reports/mmada_ccaw_v1.json`、`reports/lavida_ccaw_v1.json`。
L=128 可能截断推理，official 的宽松正文匹配与 strict 末尾答案解析也会不同。
此处不能推断准确率改进或普遍加速。

| 模型 | fixed run | ccaw_v1 run |
| --- | --- | --- |
| MMaDA | `20261105T233638415607Z_27cad5128acf` | `20261105T233726686159Z_e26b90010824` |
| LaViDa | `20261105T184733587137Z_857539ce320d` | `20261105T184800495823Z_77c87440f651` |

## 运行与追溯

在远程工程目录执行，例如：

```bash
.venv/bin/python -m cdapc_lab doctor --config configs/experiments/mmada_m3cot_fixed_smoke.yaml
.venv/bin/python -m cdapc_lab infer --config configs/experiments/mmada_m3cot_ccaw_v1_smoke.yaml
.venv/bin/python -m cdapc_lab score --run runs/RUN_ID
.venv/bin/python -m cdapc_lab summarize --run runs/RUN_ID --evaluation FULL_EVALUATION_FINGERPRINT
```

infer 返回新 run 路径，score 返回完整评分指纹。已有 run 可能含多个评分版本，
summarize 必须选定所需指纹。最新离线指纹索引在 `reports/final_audit.json`。

源码阶段：`54c046b` 初始固定窗口与 MMaDA，`a421d5a` 离线 LaViDa 与 16 题
预登记，`abdc2a6` 零事件审计及独立 replay 指纹，`b8f1f8d` CCAW，
`1068870` 离线 tokenizer 校验。CCAW 改动会改变生成源码指纹；历史 run
保留原始记录，不与新源码混合续跑。需要重放历史状态时，应使用对应阶段
源码和同一锁定环境。只改评分实现不会要求重新生成。

本次 MMaDA 输入改为 README `inference_mmu` 协议的 512 中心裁剪（1024 个
MAGVITv2 码），并在共享题目前加 MMaDA 官方推理指令（DECISIONS 第 13、14
条）。此前的 MMaDA 结果（256 输入，源码 commit `df31f79`；以及未加指令的
512 中间 run，ID 以 `20261105T2242` 至 `20261105T2251` 开头）为历史记录，
不再续跑或与新 run 混合。`reports/final_audit.json` 只覆盖到 `df31f79` 为止
的 run，当前 MMaDA 结果以 `reports/mmada_acceptance.json`、
`reports/mechanism16*.json` 和 `reports/mmada_ccaw_v1.json` 为准。

上游代码、模型配置、tokenizer、视觉模型与数据版本在 `upstream.lock.json`
固定；只复用与官方 LFS SHA-256 相符的外部权重二进制，没有复用旧实验源码。
权重软链接依赖服务器共享存储，迁移主机前需重新准备资产并通过 doctor。
本地已备份源代码、所有小样本输入、原始预测、评分与日志，不含大模型权重。

尚未执行：正式大样本主表、独立重复测速、外部解码基线、额外数据集、跨硬件
一致性和真实触发事件的分叉效应估计。本阶段没有创建 DCD、缓存优化或新机制。

## POPE (MMaDA, 5,127 deduplicated inputs, HostHarry/CD-aligned)

τ_r=0.9 CD-APC run `reports/pope_yesno_mmada_cdapc_L128.json` (runs
`runs/20261106T023052820346Z_4d2ebf7ab2ec`, `…T023537693655Z_340d074c4d19`,
`…T023537971082Z_4f14d8e9e308`; eval `0f130057…`/`a0edc9bd…`/`ffd2fa10…`):
all 5,127 prompt_ids equal the native export; same_text 4,355,
same_official_prediction 4,851. Native metrics recompute exactly.

### Per-split metrics over τ_r (native first row)

| split | config | acc | precision | recall | F1 | yes_ratio |
|---|---|---|---|---|---|---|
| random | native | 0.8303 | — | — | 0.8249 | — |
| | τ 0.9 | 0.8343 | 0.8289 | 0.8427 | 0.8357 | 0.5083 |
| | τ 0.8 | 0.8337 | 0.8269 | 0.8440 | 0.8354 | 0.5103 |
| | τ 0.7 | 0.8327 | 0.8236 | 0.8467 | 0.8350 | 0.5140 |
| | τ 0.6 | 0.8300 | 0.8214 | 0.8433 | 0.8322 | 0.5133 |
| | τ 0.5 | 0.8237 | 0.8122 | 0.8420 | 0.8268 | 0.5183 |
| popular | native | 0.8197 | — | — | 0.8159 | — |
| | τ 0.9 | 0.8237 | 0.8118 | 0.8427 | 0.8270 | 0.5190 |
| | τ 0.8 | 0.8243 | 0.8121 | 0.8440 | 0.8277 | 0.5197 |
| | τ 0.7 | 0.8253 | 0.8120 | 0.8467 | 0.8290 | 0.5213 |
| | τ 0.6 | 0.8243 | 0.8125 | 0.8433 | 0.8276 | 0.5190 |
| | τ 0.5 | 0.8213 | 0.8086 | 0.8420 | 0.8250 | 0.5207 |
| adversarial | native | 0.7600 | — | — | 0.7691 | — |
| | τ 0.9 | 0.7553 | 0.7174 | 0.8427 | 0.7750 | 0.5873 |
| | τ 0.8 | 0.7560 | 0.7177 | 0.8440 | 0.7757 | 0.5880 |
| | τ 0.7 | 0.7577 | 0.7187 | 0.8467 | 0.7775 | 0.5890 |
| | τ 0.6 | 0.7543 | 0.7159 | 0.8433 | 0.7744 | 0.5890 |
| | τ 0.5 | 0.7533 | 0.7152 | 0.8420 | 0.7734 | 0.5887 |

Δacc/ΔF1 95% paired CIs vs native per split (report
`reports/pope_yesno_mmada_cdapc_L128[_r{80,70,60,50}].json`):

- random: 0.9 [−0.004,0.0123]/[0.0026,0.0195]; 0.8 [−0.0047,0.0113]/[0.0022,0.0191];
  0.7 [−0.0057,0.0107]/[0.0018,0.0187]; 0.6 [−0.0083,0.008]/[−0.0009,0.0158];
  0.5 [−0.015,0.002]/[−0.0066,0.0109].
- popular: 0.9 [−0.0037,0.0113]/[0.0032,0.0187]; 0.8 [−0.0033,0.0123]/[0.0039,0.0199];
  0.7 [−0.002,0.0133]/[0.0054,0.0212]; 0.6 [−0.003,0.012]/[0.004,0.0196];
  0.5 [−0.007,0.0107]/[0.0006,0.018].
- adversarial: 0.9 [−0.013,0.0033]/[−0.0019,0.014]; 0.8 [−0.0127,0.0043]/[−0.0012,0.0146];
  0.7 [−0.0107,0.006]/[0.0005,0.0165]; 0.6 [−0.014,0.003]/[−0.0023,0.0135];
  0.5 [−0.016,0.003]/[−0.0041,0.0135].

### Mechanism and cost per τ_r

| τ | qualified_flips | actual_switches | changed inputs (switch / no-switch) | fixed / broken | sign flips (switched/blocked_pool/blocked_other) | mean s | canvas_filled / eos_prefix |
|---|---|---|---|---|---|---|---|
| 0.9 | 0 | 0 | 276 (0/276) | 97/179 | 153 (0/0/153) | 2.54 | 4648/479 |
| 0.8 | 1 | 1 | 283 (0/283) | 99/184 | 149 (0/0/149) | 1.92 | 2970/2157 |
| 0.7 | 69 | 69 | 284 (0/284) | 99/185 | 141 (8/0/133) | 1.39 | 1781/3346 |
| 0.6 | 360 | 360 | 295 (10/285) | 99/196 | 159 (47/0/112) | 0.91 | 825/4302 |
| 0.5 | 896 | 896 | 349 (76/273) | 115/234 | 246 (186/0/60) | 0.53 | 216/4911 |

Most `actual_switches` only change the case of the same answer word at the
answer token (τ=0.5: `No`→`no` 479, `Yes`→`yes` 215), which the official rule
scores identically. Contrast replacements that change the answer word (base
word → written word at the answer token, per unique input) are:

| τ | yes→no | no→yes | written word correct / wrong |
|---|---|---|---|
| 0.9 | 0 | 0 | 0 / 0 |
| 0.8 | 0 | 0 | 0 / 0 |
| 0.7 | 5 | 0 | 5 / 0 |
| 0.6 | 30 | 9 | 30 / 9 |
| 0.5 | 109 | 60 | 115 / 54 |

These counts compare the written word with the base word of the same
decoding state, not with native. Lowering τ also moves the answer commit
earlier (more `eos_prefix_complete` stops), so the τ effect mixes contrast
content with commit timing.

Per-τ run dirs and evaluation fingerprints are recorded in
`reports/pope_rgate_sweep_runs.json`; the aggregated table is
`reports/pope_rgate_sweep.json` (`scripts/pope_rgate_summary.py`).
Lower τ activates contrast replacement (896 switches at τ=0.5) and much
earlier EOS stopping (4,911 vs 479 inputs), but accuracy declines — the
τ=0.5-vs-0.9 paired CIs exclude zero for Δacc and ΔF1 on random
([−0.0197,−0.002]/[−0.0174,−0.0008]); other splits are within noise.
This sweep ran on the POPE evaluation set itself (DECISIONS 17); no grid
point is a tuned default.
