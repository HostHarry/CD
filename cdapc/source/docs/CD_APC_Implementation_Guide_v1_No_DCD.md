# CD-APC 实验框架重建施工指南

> 版本：v1.1  
> 日期：2026-11-05  
> 项目：大模型 / 第一篇 CD-APC  
> 文档性质：给开发者或编码代理执行的施工规范；本文中的目录、接口和命令是待实现的目标，不代表程序已经存在或实验已经完成。  
> 当前明确约束：**从空目录重新编写实验框架；不复用旧实验文件；本阶段不实现 DCD。**

---

## 0. 开工指令与范围

### 0.1 直接交给施工者的指令

在一个新的独立目录 `cdapc-experiments/` 中，按本文顺序实现第一篇 CD-APC 的实验框架。

第一批交付必须支持：

1. 固定上游版本的官方 Native 生成路径。
2. MMaDA 与 LaViDa 两个模型适配器。
3. 正常分支和视觉访问消融分支的两个完整前向。
4. 固定窗口、独立双门、确定性排序、原始候选 fallback 的 CD-APC。
5. 生成、输出规范化、答案解析、计分相互独立。
6. 完整生效配置、数据样本清单、逐样本预测和实际计算成本。
7. `g=0`、crossed timing-only、base-candidate visual scheduling 三种机制诊断预设。
8. 同一解码状态下的一次候选替换分叉实验。

固定窗口版本验收通过后，再实现本文定义的 `ccaw_v1`。新增创新点的实验策略只在机制归因完成后接入。

### 0.2 本阶段不施工的内容

- **DCD 的算法实现、移植、封装、配置、CLI 注册项和占位测试均不创建。**
- 不把 `g=0` 诊断或某个置信度提交变体命名为 DCD 或 Native。
- 不移植旧仓库的 `_pick_transfer`、`dispatch_dcd_decode_text` 等解码帮助函数。
- 不实现第二篇的单前向门控差分、选层、共享前层或局部重跑。
- 首个版本不实现 KV cache、prefix cache、动态 cache refresh。
- 首个版本不加入第三个 `g` 门、历史分布重加权、固定等待轮数、学习式打分器。
- 不提前把“发生改选后优先单独提交”设为主方法行为。
- VIGOR、VISAGE 等外部对照后续独立接入，不阻塞本阶段验收。
- 先完成小样本验收，再按冻结后的正式实验配置运行大规模实验。

### 0.3 “不复用旧文件”的具体含义

禁止把 `HostHarry/APC` 或旧服务器实验目录中的 decoder、runner、模型修改版、评测补丁、配置和字节码包装器复制、导入或软链接到新工程。

允许作为外部依赖使用：

- 固定版本的官方模型实现和 checkpoint；
- 固定版本的 tokenizer、视觉编码器、image processor；
- 固定版本的数据集及官方 scorer；
- 通用科学计算与测试库。

官方依赖从其正式来源重新获取。若必须修正 attention mask 的传递，基于固定官方源码编写最小补丁，记录补丁 hash；不得整体搬运旧 `modeling_llada.py`。

旧代码只用于理解历史行为、核对问题和追溯旧结果。新实验不以复现某个旧汇总分数作为程序正确性的唯一标准。

---

## 1. 本次重建要解决的已知问题

审计对象实际为 [HostHarry/APC](https://github.com/HostHarry/APC)。用户给出的 `hostharry/cd-apc` 路径在审计时返回 404。主要检查的固定版本如下：

| 分支 | Commit | 用途 |
|---|---|---|
| `upload-cdapc-historyvchd` | `6dd311f7e3402297778bd10931b4c1ee786101af` | 默认入口和旧实验路径 |
| `22231-mmada` | `e751eda6757145f6ba0560152e36be11a03ce953` | MMaDA 工作分支 |
| `22231-lavida` | `56a6d2b65e63b00450f0b24e031011fa69fcc31f` | LaViDa 工作分支 |
| `43030-lavida` | `aaa0170d24c0cc13d783e57d4dfd70edcf86864a` | 少量评测配置与归档对照 |

已确认的问题及新框架要求：

| 已审行为 | 新工程必须满足的要求 |
|---|---|
| 旧 `cv_dcd + cd_apc` 与后来的 `vchd` 具有不同提交规则 | 方法注册与生效策略必须唯一；未知名称直接报错 |
| 部分关键 `.py` 文件通过旧 `.pyc` 加载，且存在接口版本错配 | 自有算法必须有完整可读源码；不能以旧 bytecode 代替源码 |
| fallback 默认可能写对比候选 | 主方法 fallback 明确写原始 top-1，测试最终写回动作 |
| 部分门控开关受路由影响，配置关闭不等于实际关闭 | 四种 u/r 开关组合必须通过行为测试 |
| CCAW 有多种压力定义和重复窗口更新 | 新版本采用唯一公式、唯一更新函数、明确版本号 |
| Original 与 VCHD 的 EOS 后处理不同 | 共用输出规范化器，保留原始 token |
| VCHD 忽略原生 steps、block 或温度参数 | 方法不支持的字段报错，不能打印一个未执行的参数 |
| attention 参数存在但可能没有传入实际 kernel | 验证视觉阻断实际生效 |
| M³CoT 的新 parser 仍可能把推理中的选项或冠词当成答案 | parser 独立、无 gold 依赖，有固定反例测试 |
| 结果与代码版本、任务过滤规则难以一一对应 | 保存实际 source hash、数据 ID 清单、生效配置和评分指纹 |

这些结论来自源码和静态字节码分析。旧 M³CoT parser 的两个合成反例已执行验证；完整模型推理和旧 benchmark 分数没有在本次审计中重跑。证据链接见附录 A。

---

## 2. 首个可交付版本的完成定义

以下全部满足，才可称“固定窗口实验框架完成”：

- [ ] 自有代码无旧实验目录依赖、无旧字节码加载。
- [ ] 唯一方法注册只有 `native` 和 `cdapc`；诊断变体属于 `cdapc.variant`。
- [ ] 官方 Native 包装与直接官方调用通过输入和输出核对。
- [ ] 两个模型适配器都能提供同一类型的正常/消融响应 logits。
- [ ] attention 干预有效性、paired/separate 一致性通过。
- [ ] 固定窗口 CD-APC 的公式、双门、排序、fallback、停止规则通过测试。
- [ ] 在同一协议下，Native 与 CD-APC 共用样本、问题文本和评分流程。
- [ ] 对一个实际小样本 manifest 完成生成、离线评分、汇总、中断续跑。
- [ ] 基础成本统计始终存在；详细 trace 关闭时结果不变。
- [ ] 三种诊断预设和共同状态分叉可运行。
- [ ] 所有结果明确标记“新框架版本”，不继承旧结果目录或旧方法别名。

此时不要求新增创新点已经有效，不要求论文主表跑完，也不要求接入外部解码对照。

---

## 3. 工程目录与文件责任

建议使用 Python package `cdapc_lab`。自有逻辑优先采用小型函数和 dataclass；不建立复杂插件框架，不通过复制 engine 增加变体。

| 路径 | 责任 |
|---|---|
| `pyproject.toml` | 安装、CLI、运行与开发依赖 |
| `upstream.lock.json` | 模型源码、checkpoint、tokenizer、视觉组件版本 |
| `docs/ALGORITHM_SPEC.md` | 本文算法合同的最终工程版本 |
| `docs/PROTOCOL_SPEC.md` | 各数据集输入、输出、评分协议 |
| `docs/DECISIONS.md` | 与旧实现/稿件存在差异的工程决定及原因 |
| `src/cdapc_lab/cli.py` | 所有命令的薄入口 |
| `src/cdapc_lab/__main__.py` | 调用 CLI，使 `python -m cdapc_lab` 可执行 |
| `src/cdapc_lab/config.py` | 严格解析、组合校验、生效配置 |
| `src/cdapc_lab/types.py` | 跨模块 dataclass/类型 |
| `src/cdapc_lab/models/base.py` | Adapter 与 NativeRunner 接口 |
| `src/cdapc_lab/models/mmada.py` | MMaDA 输入与真实前向 |
| `src/cdapc_lab/models/lavida.py` | LaViDa 输入与真实前向 |
| `src/cdapc_lab/decoding/proposal.py` | 合法词表、APC、对比分数及统计量 |
| `src/cdapc_lab/decoding/gates.py` | 门控检查 |
| `src/cdapc_lab/decoding/selection.py` | 正常选点、排序、fallback |
| `src/cdapc_lab/decoding/state.py` | MASK 状态、写回、状态快照 |
| `src/cdapc_lab/decoding/window.py` | fixed；后续增加 ccaw_v1 |
| `src/cdapc_lab/decoding/engine.py` | 唯一 CD-APC 循环 |
| `src/cdapc_lab/data/records.py` | Sample 与 manifest |
| `src/cdapc_lab/data/tasks/` | 数据集加载与任务协议 |
| `src/cdapc_lab/evaluation/normalize.py` | 响应区提取、首 EOS 截断 |
| `src/cdapc_lab/evaluation/parsers.py` | 无 gold 的严格答案提取 |
| `src/cdapc_lab/evaluation/scorers.py` | 官方 scorer 与逐例计分 |
| `src/cdapc_lab/experiments/infer.py` | 生成任务与续跑 |
| `src/cdapc_lab/experiments/replay.py` | 共同状态分叉 |
| `src/cdapc_lab/experiments/score.py` | 只重评、不调用模型 |
| `src/cdapc_lab/analysis/paired.py` | 配对差值、置信区间、四格统计 |
| `src/cdapc_lab/analysis/efficiency.py` | 质量与实际成本汇总 |
| `src/cdapc_lab/io/artifacts.py` | 指纹、原子写入、结果检查 |
| `configs/experiments/` | 可追踪的实验配置 |
| `tests/unit/` | 公式、状态、门控、窗口与配置测试 |
| `tests/evaluation/` | 解析器、输出规范化、评分测试 |
| `tests/integration/` | 小模型/真实 adapter 验证 |
| `runs/` | 运行产物；不作为源码提交 |

不得创建 `dcd.py`、DCD 配置或 DCD 注册占位项。首版也不要预建一堆尚未实现的方法空文件。

### 3.1 参数来源

参数只允许来自：

1. 明确的 schema 默认值；
2. 本次实验配置；
3. 显式 CLI 覆盖。

解析后保存唯一 `ResolvedConfig`。未知字段、无效组合和未消费字段均报错。

环境变量仅用于设备、数据/模型根目录和必要凭据。算法的 alpha、门阈值、窗口、fallback、参考分支不能由隐藏环境变量修改。

### 3.2 依赖锁定

在真实运行前填入并核对：

- 官方模型源码 URL 和 commit；
- checkpoint ID/path、revision 或本地文件指纹；
- tokenizer revision；
- 视觉编码器/VQ 模型/image processor revision；
- Python、PyTorch、CUDA、transformers 等实际版本；
- dtype、attention backend、normal attention semantics；
- 本项目 source hash、Git commit 与 dirty 状态。

不要根据旧字节码选择运行环境。`PIN_REQUIRED` 必须使 preflight 失败，不能自动解释为 `latest`。

---

## 4. 数据对象与接口合同

下列代码为接口草案，不是可直接运行的实现。字段名称可以小幅调整，但语义不能省略。

### 4.1 Sample 与 GoldRecord 分开

~~~python
@dataclass(frozen=True)
class Sample:
    sample_id: str
    group_id: str | None
    question: str
    options: dict[str, str] | None
    image_paths: tuple[str, ...]
    public_context: str | None
    task_id: str

@dataclass(frozen=True)
class GoldRecord:
    sample_id: str
    answer: object
    metric_metadata: dict
~~~

模型输入构造器只接收 `Sample`，不能接收正确答案、正确选项索引或 gold rationale。公开给模型的题目 context 与标注 rationale 必须由 loader 明确区分。

### 4.2 PreparedInput

最少包含：

- `sample_id`；
- 官方预处理生成的模型输入；
- prompt token IDs 或可复原的编码记录；
- 图像张量/视觉特征及其 shape；
- 展开后真实视觉 key 位置掩码；
- response 位置映射；
- mask/EOS/PAD IDs；
- 合法文本输出域；
- 基础 attention mask/bias；
- 输入指纹和预处理版本。

视觉位置必须根据模型实际展开结构获取。不能写死“第 2 到 1026 位”，不能假定所有模型的视觉 token 都是一个连续区间。

### 4.3 DecodeState

~~~python
@dataclass
class DecodeState:
    response_ids: Tensor          # [L]，只表示响应区
    pending: Tensor               # [L]，bool
    iteration: int
    window_width: int
    pressure_ema: float
    previous_candidate_ids: Tensor
    previous_observed: Tensor
    finish_reason: str | None
~~~

必要规则：

- prompt 和视觉输入不属于可写状态。
- 状态写回只能发生在 pending 位置。
- 首版状态不包含可复用的 KV cache。
- 快照要包含所有会影响后续决策的状态；需要随机数时另存 RNG 状态。
- 模型实际输入由固定 PreparedInput 与当前 DecodeState 组合。
- 初始状态：响应区全为 MASK，`pending` 全 true，`iteration=0`，`pressure_ema=0.0`，`previous_observed` 全 false，`finish_reason=None`；fixed 的初始 W 等于 width，CCAW 的初始 W 等于 initial。未观测的历史候选 ID 无语义，不允许参与比较。
- 候选历史只表示紧邻上一轮的有效观测。每轮写回后重建 `previous_observed`：仅本轮观察过且仍为 pending 的位置为 true，其他位置置 false；不能用累积 OR 保留更早轮次的标记。

### 4.4 ForwardResult 与计算记录

~~~python
@dataclass(frozen=True)
class ForwardResult:
    response_logits: Tensor       # [1, L, V_raw]，未被策略重写
    cost: ComputeCost

@dataclass(frozen=True)
class PairedForwardResult:
    visual: ForwardResult
    ablated: ForwardResult
    execution_cost: ComputeCost   # 该次实际执行总量，不能再重复加子项
~~~

实际计数由 adapter 在模型调用处产生。若两个分支合批，不能给两个子结果各记一次调用后又在总记录中重复加总。

### 4.5 ProposalStats

每个待观察位置至少保留：

- `base_id = b`；
- `contrast_id = c`；
- `logp_I(b), logp_I(c)`；
- `gain_measured(b), gain_measured(c)`；
- `gain_used` 的模式；
- `pool_mass = m`、`pool_size`；
- `u_b, u_c`；
- `q_b, q_c`；
- `r_b, r_c`；
- `proposal_changed = (c != b)`；
- `h(c) - h(b)`。

GPU 中可以临时持有 dense 张量；默认日志不保存每步全词表概率。需要节约显存时，可以按响应位置分块计算逐行 softmax，验收其数值和决策一致性。

### 4.6 GateResult 与 CommitDecision

~~~python
@dataclass(frozen=True)
class GateResult:
    target: str                  # "contrast" 或 "base"
    pass_u: Tensor
    pass_r: Tensor
    eligible: Tensor

@dataclass(frozen=True)
class CommitDecision:
    positions: Tensor
    written_ids: Tensor
    base_ids: Tensor
    contrast_ids: Tensor
    reason: str                  # "qualified" 或 "fallback"
    gate_target: str
    eligible_count: int
~~~

日志里的 `actual_switch` 由 `written_id != base_id` 判断。不要把 proposal 发生变化、正常提交改选和 fallback 写入混成一个指标。

---

## 5. CD-APC 固定窗口的唯一数学规范

### 5.1 合法词表与概率

设同一状态下两个分支原始响应 logits 为 \(\ell^I,\ell^A\)。

在同一个合法文本词表 \(\mathcal V\) 上计算：

\[
\log p^I=\operatorname{logsoftmax}(\ell^I|_{\mathcal V}),
\qquad
\log p^A=\operatorname{logsoftmax}(\ell^A|_{\mathcal V}).
\]

规范：

- 使用 FP32 计算 log-softmax、APC mass、门控统计量。
- MASK 和视觉 code 不允许被写入文本响应。
- EOS 应保留为合法输出；其他特殊 token 的处理必须按模型协议锁定。
- 非法位置用 mask 排除；不得对 \(-\infty-(-\infty)\) 求 gain。
- 使用 log-softmax 差，不先 softmax 再取可能下溢的 log。
- 正常分支 parity 检查比较策略处理前的原始 logits，避免把词表裁剪混入模型计算一致性。

### 5.2 原始候选与 APC pool

\[
b_i=\arg\max_{v\in\mathcal V}p_i^I(v),
\]

\[
\mathcal A_i=
\{v\in\mathcal V:
\log p_i^I(v)\ge \max_w\log p_i^I(w)+\log\beta\}.
\]

要求 \(0<\beta\le1\)，并验证 \(b_i\in\mathcal A_i\)。

\[
m_i=\sum_{v\in\mathcal A_i}p_i^I(v).
\]

pool mass 来自未做 APC 重归一化的正常分支概率。不能把 pool 内再次归一化后的概率和当作 mass。

### 5.3 对比候选

\[
g_i(v)=\log p_i^I(v)-\log p_i^A(v),
\]

\[
h_i(v)=\log p_i^I(v)+\alpha g_{i,\mathrm{used}}(v),
\quad v\in\mathcal A_i,
\]

\[
q_i=\operatorname{softmax}_{\mathcal A_i}(h_i),
\qquad c_i=\arg\max_{v\in\mathcal A_i}h_i(v).
\]

默认 \(\alpha\ge0\)。真实 gain 与用于决策的 gain 分开记录：

- `full`：`gain_used = gain_measured`；
- `gain_zero`：两分支仍执行，但 `gain_used = 0`；
- \(\alpha=0\)：真实 gain 可以保留用于诊断，候选重加权作用为零。

### 5.4 双门

对门控目标 token \(z_i\)：

\[
u_i(z_i)=p_i^I(z_i),\qquad
r_i(z_i)=m_iq_i(z_i).
\]

\[
\operatorname{pass}_{u,i}
=(\neg e_u)\lor[u_i(z_i)\ge\tau_u],
\]

\[
\operatorname{pass}_{r,i}
=(\neg e_r)\lor[r_i(z_i)\ge\tau_r].
\]

\[
E_t=\{i\in C_t:\operatorname{pass}_{u,i}\land
\operatorname{pass}_{r,i}\}.
\]

开启的门要求 \(0<\tau\le1\)。关闭门使用布尔开关，不用阈值 0 模拟关闭。

**不得改写为 \(\min(u,r)\ge0.9\)。** 关闭某个门也不得自动修改 proposal 或排序分数。

\(r\) 表示对比 proposal 的集中程度，经 APC mass 修正后仍不是正确概率。测试 fixture 的“通过门控”不能被写成“正确性保证”。

### 5.5 搜索窗口与正常提交

令 \(M_t\) 是响应区尚未提交的位置：

\[
C_t=\operatorname{leftmost}_{W_t}(M_t).
\]

fixed 模式 \(W_t=W\)。它表示最靠左的 \(W\) 个 pending 位置，不是长度为 W 的连续物理切片；首版不再叠加隐藏的 physical-span 上限。

对 \(E_t\) 做确定性排序：

1. \(r(z_i)\) 降序；
2. \(u(z_i)\) 降序；
3. 响应位置升序。

取前 \(\min(K_{\max},|E_t|)\) 个位置。候选 token 内部 argmax 并列时固定取合法词表 ID 最小者。

### 5.6 fallback

当 \(E_t=\varnothing\) 时：

\[
\rho_i=\min_{a\in G_{\rm enabled}}
\frac{s_{a,i}}{\tau_a},
\]

其中 \(s_u=u(z_i)\)、\(s_r=r(z_i)\)。只使用开启的门；两门都关闭时所有候选有资格，因此不会进入 fallback。

fallback 位置按以下顺序选一个：

1. \(\rho_i\) 降序；
2. \(r(z_i)\) 降序；
3. 响应位置升序。

**实际写入始终为该位置的原始候选 \(b_i\)。**

位置选择仍可使用对比统计量。因此文档只能说“fallback 写原始候选”，不能说“fallback 完全不使用对比信息”。

### 5.7 进度与停止

CD-APC 首版采用 `fill_canvas`：

- 未完成的一轮必须至少提交一个合法非 MASK token。
- 已提交 token 不覆盖。
- 整个响应 buffer 无 MASK 时结束。
- 每轮至少减少一个 MASK，故最多 L 轮；实现保留明确的安全上限。
- 若外部限制提前耗尽且仍有 MASK，记录 `budget_exhausted`。不能静默删 MASK、补答案或把失败记录丢掉。
- EOS 只在完成后的共同规范化步骤处理；不引入真正的 EOS early-stop。

---

## 6. 方法和机制诊断的绑定表

CLI 首版只注册：

- `method: native`；
- `method: cdapc`。

CD-APC 的变体由下表约束。输入配置选择变体，解析后的配置必须展开并保存 gain、gate、write 三种实际行为。

| `cdapc.variant` | gain_used | 门与排序对象 | 正常提交写入 | fallback |
|---|---|---|---|---|
| `full` | 真实 gain | \(c\) | \(c\) | \(b\) |
| `gain_zero` | 0 | \(b=c\) | \(b\) | \(b\) |
| `timing_only_crossed` | 真实 gain | \(c\) | \(b\) | \(b\) |
| `base_candidate_visual` | 真实 gain，用于计算 \(q\) | \(b\) | \(b\) | \(b\) |

对 `base_candidate_visual`，必须计算：

\[
u_b=p^I(b),\qquad r_b=mq(b).
\]

不能拿 \(r(c)\) 判断 \(b\) 的提交资格。

### 6.1 必须保持的等价性

在相同冻结状态、fixed 窗口及相同参数下：

- `full` 与 `timing_only_crossed` 的候选统计、门、排序和提交位置相同；写入值可以不同。
- 一旦写入不同，后续轨迹允许分叉。该对照测的是策略总效应，不能称为固定全部未来时序的纯直接效应。
- 在 full 的完整轨迹上从未发生 `written_id != base_id` 的确定性样本，full 与 crossed 应有相同轨迹与输出。不能按 crossed 的日志筛选此子集，因为 crossed 本来始终写 b。
- `gain_zero` 与 \(\alpha=0\) 的决策应相同；二者仍属于 CD-APC 诊断，不属于 Native。

本阶段 `gain_zero` 仍执行两个完整分支，并如实记录成本。不把删除参考分支的加速混入此项归因。

---

## 7. 唯一 CD-APC engine 的执行顺序

~~~python
def decode_cdapc(adapter, prepared, config, observer):
    state = initialize_response(prepared, config)

    while state.pending.any():
        if state.iteration >= config.iteration_limit:
            state.finish_reason = "budget_exhausted"
            break
        snapshot = snapshot_state(state)

        pair = adapter.forward_pair(
            prepared,
            snapshot,
            execution=config.branch_execution,
            cache="disabled",
        )
        stats = make_proposals(pair, prepared.allowed_text_vocab, config)

        candidate_positions = choose_window(snapshot, config.window)
        gates = evaluate_gates(stats, candidate_positions, config)
        decision = choose_commits(stats, gates, candidate_positions, config)

        # 只观察，不改变决策。用于 trace 与共同状态分叉快照。
        observer.before_commit(snapshot, stats, gates, decision)

        # fixed 为恒等；CCAW 在这里计算唯一一次下一轮状态。
        next_window_state = compute_next_window_state(
            snapshot, stats, gates, candidate_positions, config.window
        )

        state = apply_commit(snapshot, decision)
        state = update_observation_history(state, stats, candidate_positions)
        state = install_window_state(state, next_window_state)
        state.iteration += 1

        observer.after_commit(state, decision, pair.execution_cost)

    if not state.pending.any():
        state.finish_reason = "canvas_filled"
    return RawGeneration.from_state(state)
~~~

施工要求：

- 每一轮的 \(p^I,p^A\) 必须对应同一个提交前 state。
- 窗口压力在提交前计算，候选历史在压力计算后更新。
- 写回后不能复用上一轮其他位置的 proposal，下一轮必须重新前向。
- observer 的开关不能影响 proposal、门、排序、窗口或 RNG。
- 只保留一个 engine；变体通过合同绑定，不复制四份循环。
- Native 不调用这个 engine，保留官方生成入口。
- `budget_exhausted` 仍返回并保存带剩余 MASK 的原始结果和状态；它不是成功完成，不能在后处理中静默删除 MASK 后改标为成功。

---

## 8. 模型适配器施工细则

### 8.1 Native 的身份与两个不同的 parity

必须分开完成：

**A. Native 端到端 parity**

同一官方 checkpoint、输入、生成配置和环境下，新包装调用与直接官方调用的响应 token 一致。Native 保留官方 block/steps/remasking 等调度。

**B. visual 单次前向 parity**

冻结同一个完整响应状态，在相同 attention 语义下，adapter 正常分支与直接官方模型前向的 logits 一致。

B 不意味着 CD-APC 最终输出应该与 Native 相同。不要用一个测试替代另一个测试。

### 8.2 无缓存模式

CD-APC 首版：

- 每一轮从完整当前 state 计算两个分支；
- 不传入跨轮 `past_key_values`；
- 不创建可供后续轮使用的 K/V；
- 不共享旧 visual cache 给 reference；
- 不以“缓存了 prompt”替代两个完整计算。

Native 使用官方支持的明确配置。若官方默认含缓存，首轮采用无缓存配置时必须标识为 `native_no_cache` 执行配置；不能声称它是未经变化的官方默认。

**是否无缓存与是否 prefix-LM 是两个不同问题。** 两种方法在一个比较协议中采用相同的 normal attention 语义，且该语义在 manifest 中显式记录。不能因为关 cache 就自动切换 prefix-LM。

### 8.3 视觉访问消融

参考分支沿用同一输入、位置编码和已提交文本，修改当前前向的访问关系：

- query：所有非视觉位置；
- key：实际视觉 token/patch 位置；
- 层与头：本版本定义为全部对应 self-attention 层与头；
- 保留正常分支的 padding、prefix 等原有约束；
- 对非视觉 query→视觉 key 连接施加不可访问 mask。

用视觉 key 布尔向量支持多个不连续视觉区间。

这项干预切断的是当前前向的视觉访问。历史已提交文本可能携带先前视觉信息，不能声称 reference 完全没有任何视觉信息。

### 8.4 验证 mask 实际生效

至少完成：

1. 一个受控小模型/attention fixture：视觉值变化能影响正常分支；指定路径阻断后该影响消失。
2. 固定文本和 response state、固定视觉 token 形状，只替换视觉内容；验证参考分支文本 logits 的预期不变性。
3. 确认实际 kernel 接收了构造的 mask，而非只在高层函数接收。
4. paired batch-2 与两个独立分支的一致性。
5. 单独正常前向与 paired 的 visual 部分一致性。

不能仅以“两分支输出不同”证明消融正确；也不能要求每个真实样本的两分支都必须不同。

已审官方 LaViDa 版本的 SDPA 路径有 `attn_mask=None`。优先选择经验证支持 mask 的官方路径；若需补丁，只修改必要传递逻辑，并对正常分支做回归。

### 8.5 数值一致性标准

- 模型处于 `eval()` 与无梯度推理模式；明确 autocast/dtype，不让 dropout 或训练态进入解码。
- 同一设备、dtype、backend、形状下要求一致到预先规定的误差容限。
- 单独记录 max/mean logit error、argmax 不一致数、门边界不一致数。
- 对 near-tie/near-threshold 事件报告差值，不能只看张量平均误差。
- batch-2 与顺序执行若因 kernel 形状产生决策差异，先固定一条执行路径作为主协议；不能笼统宣称二者完全等价。
- 不要求跨硬件 bitwise 一致。

---

## 9. 数据协议、输出规范化与评分

### 9.1 首批任务

第一批完整接入 `M³CoT`；第二个任务接入具有明确离散输出协议的视觉判断集，以检查任务外泛化。随后扩展 MMMU、MMBench、HallusionBench、VLind-Bench。

每个 task adapter 必须声明：

- 数据来源与 revision；
- split 与精确过滤条件；
- image 路径与多图策略；
- prompt ID 与模板正文 hash；
- 允许的答案域；
- 官方 scorer revision；
- strict parser ID；
- 聚合单位与 group_id；
- 无效样本和解析失败处理。

不允许仅把一个本地 TSV 文件命名成 test split 就认为已完成 split 对齐。preflight 要输出样本 ID manifest 与过滤后数量。

### 9.2 输入一致性

- 方法间使用同一任务问题文本、选项排列、公开 context 和图像。
- 模型间保留其官方 chat template，不强行把 tokenizer 层面的 token 序列做成一样。
- prompt 不能根据方法名改变。
- 图像预处理由模型协议决定，不能根据图片文件名包含 `geo`、`llava` 等字符串切换。
- 图片读取失败报错；不使用白图替代。
- 多图输入不支持时明确报错或采用预先声明的统一协议，不静默取第一张。
- Smoke 子集按固定 seed/预先指定 IDs 生成，与模型预测无关；不用于选最优参数。

### 9.3 统一输出规范化

~~~python
normalize_output(raw_generation, tokenizer, normalizer_config)
    -> NormalizedOutput
~~~

顺序固定：

1. 按 adapter 的明确响应映射提取 response token IDs；
2. 只在响应区寻找第一个 EOS；
3. 在 EOS 前截断；
4. 按同一特殊 token 规则解码；
5. 执行版本化、最小化的空白处理；
6. 保存规范化文本和所用版本。

始终保存完整原始响应 IDs、首次 EOS 位置、有效长度和完整 buffer 长度。不对某方法单独去掉 `!`、裁剪推理或删除重复。

若官方 Native 返回的原始结果已在 EOS 处结束，记录该事实；不编造不存在的后续 buffer。比较时保留 Native 内部停止行为。

### 9.4 parser 与 scorer 必须分离

~~~python
official_score(normalized_text, allowed_options, gold, official_config)
    -> ScoreRecord

strict_parse(normalized_text, allowed_options, strict_config)
    -> ParsedAnswer
strict_score(parsed_answer, gold, metric_config)
    -> ScoreRecord
~~~

两条轨道接收同一份规范化文本。官方轨道封装官方的答案提取与计分流程，不能先通过 strict parser，再将其输出冒充官方原始预测输入。严格轨道的提取与计分按上面的两个接口分开。

`ParsedAnswer` 至少包含：

- `prediction`；
- `status: parsed / unparsed / ambiguous / invalid`；
- `evidence_span`；
- `rule_id`；
- `parser_revision`。

parser 不能接受 gold。正确答案只进入 scorer。

### 9.5 两条评分轨道

**official**

固定上游 scorer 和任务规则。若上游包含随机兜底，记录其随机设置与 source revision，确保可重放；不能在保留 official 名称时静默改规则。

**strict_final_v1**

用于规范化的选择题输出：

- 优先识别响应末尾的明确最终答案，而非全文任何一次选项出现。
- 支持末尾 `Final answer: (B)`、末尾 `(B)`、末尾完整 `\boxed{B}`，以及整条回答只有合法字母。
- “末尾”允许标签之后出现空白或一个句末标点（`.`、`。`、`!`、`！`）；允许规则写入 parser 版本，不能用任意删标点扩大匹配。
- 不把正文里的 `option B`、`choice A` 或冠词 `a` 当最终答案。
- 更早的 boxed 或选项讨论不能覆盖明确末尾答案。
- 对最终答案区域内部的多标签冲突、非法标签、截断标记，返回明确失败/歧义状态。
- 不随机猜测；未解析/歧义按该严格轨道的预定义规则计错并单独报告比例。
- 自由文本、数值和是非题分别采用 task-specific parser，不能套用 MCQ 字母规则。

primary metric 在运行前写入协议。默认同时输出 official 与 strict 诊断结果；不得看哪个分数更高再选主指标。若严格轨道被选为主指标，论文必须明确其协议，所有方法一致使用。

修改 parser 或 normalizer 后，只离线重评同一批 raw predictions，不重新生成。新评分写入新的 evaluation fingerprint 目录。

### 9.6 必须通过的解析反例

| 文本/情境 | 预期 |
|---|---|
| `Analysis of option B. The final result is (A).` | 提取 A |
| `The answer is a red ball. Final: (B).` | 提取 B |
| 早期 `\boxed{A}`，末尾 `Final answer: (C)` | 提取 C |
| 正文讨论 A/B/C，末尾只有 `(D)` | 提取 D |
| 响应只有合法字母 B | 提取 B |
| 最终区域出现无法消解的多个答案 | ambiguous |
| 截断的 boxed、无最终标签、空回答 | unparsed/invalid，不随机猜 |
| EOS 后出现另一个答案 | 该答案不进入解析文本 |
| prompt 内有 EOS | 不影响响应截断 |

这些是新实现的待验收测试；旧 parser 的前两个失败结果不代表新 parser 已完成。

---

## 10. 配置模板与生效规则

以下是拟实现的完整单实验配置示例。`PIN_REQUIRED` 与本地路径由施工者在 P0/P1 核对后填写。配置未锁定前，doctor 必须失败。

~~~yaml
schema_version: 1

run:
  name: mmada_m3cot_fixed_smoke_v1
  seed: 20261105
  sample_ids_file: manifests/m3cot_smoke_ids.json
  output_root: runs
  fail_on_error: true

model:
  backend: mmada
  checkpoint: PIN_REQUIRED
  checkpoint_revision: PIN_REQUIRED
  upstream_source_revision: PIN_REQUIRED
  tokenizer_revision: PIN_REQUIRED
  vision_component_revision: PIN_REQUIRED
  dtype: bfloat16
  attention_backend: PIN_REQUIRED
  attention_semantics: PIN_REQUIRED

runtime:
  device: cuda:0
  sample_batch_size: 1

protocol:
  task: m3cot
  dataset_revision: PIN_REQUIRED
  data_root: PIN_REQUIRED
  split: test
  prompt_id: m3cot_cot_final_v1
  canvas_length: 512
  normalizer_id: first_eos_v1
  strict_parser_id: strict_final_v1
  official_scorer_revision: PIN_REQUIRED
  scorer_profile: official_and_strict
  primary_metric: official

method: cdapc

cdapc:
  variant: full
  alpha: 0.5
  beta: 0.1
  u_gate:
    enabled: true
    threshold: 0.1
  r_gate:
    enabled: true
    threshold: 0.9
  max_commit_per_iteration: 16
  window:
    kind: fixed
    width: 16
  fallback:
    position_rule: normalized_gate_readiness
    token_rule: base_top1
  branch_execution: sequential
  cache: disabled
  stop_rule: fill_canvas
  iteration_limit: 512

recording:
  trace_level: sampled
  trace_sample_ids_file: manifests/m3cot_trace_ids.json
  snapshot_first_eligible_flip: true
~~~

### 10.1 配置校验规则

- `method=native` 时，配置包含明确的 `native` 生成配置块，不得携带会被忽略的 `cdapc` 块。
- `method=cdapc` 时，不得接受 `native.steps`、`step_per_block` 等原生专用参数。
- CD-APC 首版只支持 argmax；不提供一个会被忽略的 temperature 参数。
- `branch_execution` 只支持经验证的 `sequential` 或 `paired_batch`。
- `cdapc.cache` 首版只允许 `disabled`；Native 的缓存设置由其官方生成配置显式声明，并遵守第8.2节的比较协议。
- enabled 门阈值必须在 \((0,1]\)。
- \(K_{\max}\ge1\)，fixed 的 \(W\ge1\)；初版配置要求 \(K_{\max}\le W\)。
- canvas_length、iteration_limit、W 和 K 都必须为正整数。CCAW 要求整数 \(1\le W_{\min}\le W_0\le W_{\max}\)、\(K_{\max}\le W_{\min}\)、\(0\le\gamma<1\)，expand_step 与 shrink_step 为正整数。
- iteration_limit 小于 L 时必须显式标记预算限制实验，不能当默认完整解码。
- 全部实际字段及派生的 gain/gate/write 行为写入 `resolved_config.json`。

### 10.2 起始参数

| 项目 | MMaDA 起点 | LaViDa 起点 | 说明 |
|---|---:|---:|---|
| alpha | 0.5 | 0.25 | 参考已审运行设置，显式区分；不是已验证最优值 |
| beta | 0.1 | 0.1 | APC pool |
| tau_u | 0.1 | 0.1 | 独立基础门 |
| tau_r | 0.9 | 0.9 | 独立对比门 |
| fixed W | 16 | 16 | 最左 pending 数 |
| K_max | 16 | 16 | 每轮提交上限 |
| 短回答 L | 128 | 128 | 初始工程配置，检查截断与 EOS |
| CoT L | 512 | 512 | 初始工程配置，不能直接视为公平最优预算 |

参数敏感性在框架通过后开展。不得把 smoke 样本上的最高分参数带入主表而不说明选择过程；不需要为本框架引入新的训练模块或额外校准流程。

需要对齐 VISAGE 的四项主实验时，采用下节的任务级配置，不用本节通用的短回答／CoT 起始长度覆盖它们。

### 10.3 VISAGE 主实验对齐预设：MMaDA

本预设记录 VISAGE 论文主实验的 MMaDA 生成配置，供后续复现其 Native 基线与接入外部方法对照时使用。\(L\) 为响应区生成长度，\(T\) 为整段生成的计划扩散总步数，\(B\) 为 `block_length`。

| 数据集 | 生成长度 L | 总步数 T | 块长 B | 对应配置字段 |
|---|---:|---:|---:|---|
| POPE | 256 | 256 | 32 | `protocol.canvas_length=256`、`native.steps=256`、`native.block_length=32` |
| HallusionBench | 256 | 256 | 32 | 同上 |
| MME | 256 | 256 | 32 | 同上 |
| MMMU-val | 256 | 128 | 64 | `protocol.canvas_length=256`、`native.steps=128`、`native.block_length=64` |

来源：VISAGE《Seeing to Ground: Visual Attention for Hallucination-Resilient MDLLMs》[§5 Implementation Details](https://arxiv.org/html/2603.25711v1#S5)，与用户提供的 [ECCV 正式版](https://link.springer.com/chapter/10.1007/978-3-032-37447-9_23) PDF 第10页／印刷第422页一致。同一 benchmark 内，论文的 MMaDA Base、VCD 和 VISAGE 共享这组生成配置。

施工时遵守以下规则：

1. **明确参数归属。** 上表的 T 是总步数，不能当成每块步数传入模型；若官方接口使用每块步数，应显式换算并记录映射。B 是 Native 的解码块长，不得自动写入 CD-APC 的搜索窗口 W。
2. **CD-APC 对齐长度与评测协议。** 在这四项任务上与上述 Native 预设比较时，CD-APC 同样使用 `protocol.canvas_length=256`，共用样本、输入和评分协议；其提交过程仍由 W、K、双门和 fallback 决定，不接收或静默忽略 `native.steps`、`native.block_length`。默认完整解码的 `iteration_limit` 同步设为256；如另设更小上限，应按预算限制实验标记。
3. **不将长度对齐称为等计算量。** 实际迭代数、模型调用数、分支评估数与时延按第11节统计。Native 的计划 T 与 CD-APC 的实际迭代数、双前向成本分别报告。
4. **区分主表和测速配置。** MMMU-val 准确率主表使用 `L=256,T=128,B=64`；正式版 Table 5 的延迟实验另使用 `L=256,T=256`，该表未另行披露 B，不得用测速设置覆盖主表预设。
5. **限定适用范围。** 以上是 MMaDA 的 VISAGE 主实验对齐预设，不自动外推到 LaViDa、LLaDA-V 或 Lumina-DiMOO；M³CoT 的 `L=512` 工程起点保持独立。这项补充不要求提前实现 VISAGE 或 DCD。

---

## 11. 计算成本与时间测量

### 11.1 始终保存的计数

~~~python
@dataclass
class ComputeCost:
    model_forward_calls: int
    denoiser_branch_evaluations: int
    full_sequence_branch_evaluations: int
    prefill_calls: int
    logical_query_positions: int
~~~

另在 run/sample 层保存：

- `decode_iterations`；
- `committed_buffer_tokens`；
- `emitted_tokens_before_eos`；
- `first_eos_position`；
- `preprocess_seconds`；
- `generation_seconds`；
- `peak_memory_bytes`；
- attention backend、dtype、实际序列长度。

计数定义：

| 实际操作 | model calls | branch evaluations |
|---|---:|---:|
| 单分支完整前向 | 1 | 1 |
| 两分支 batch-2 完整前向 | 1 | 2 |
| 两分支分别完整前向 | 2 | 2 |

模型内若存在额外 CFG 分支或其他计算，也要记录。首版 CD-APC 不支持额外 CFG，必须在配置层拒绝；Native 按真实官方配置计量。

`logical_query_positions` 用于描述实际处理规模，不等价于 FLOPs。不能只根据 W 缩小宣称前向 FLOPs 降低。

### 11.2 测时边界

- 排除模型加载、数据下载、答案 judge 和结果写盘。
- 图像预处理/编码与生成时间分开，并给出是否计入端到端时间的明确表头。
- GPU 测时在统一开始/结束位置同步；不把异步提交耗时当完整计算时间。
- warm-up 与正式样本分开。
- 详细 trace 的运行用于机制分析；正式测速关闭详细 trace，基础计数保留。
- 在小规模固定计时样本上重复测量稳定性；确定性重复运行不能作为独立准确率样本。
- 不能用中断续跑的“剩余部分耗时”除以全部样本数。

输出至少同时提供质量、实际 branch evaluations、生成时延和有效输出长度。不要把 API 调用次数单独命名为 NFE。

---

## 12. 产物、指纹和断点续跑

建议每次运行保存：

| 文件/目录 | 内容 |
|---|---|
| `run_manifest.json` | 代码、依赖、模型、协议、硬件、运行状态 |
| `resolved_config.json` | 实际生效配置与派生行为 |
| `dataset_manifest.json` | 数据 revision、split、ID 顺序、group IDs、校验和 |
| `predictions/<sample_id>.json` | 原始响应 IDs、状态、计算量、输入指纹 |
| `scores/<eval_fingerprint>/<sample_id>.json` | 规范化输出、解析与逐例评分 |
| `traces/<sample_id>.jsonl` | 抽样轨迹 |
| `snapshots/<sample_id>/` | 首次 eligible flip 的可复原状态 |
| `summary/<eval_fingerprint>.json` | 聚合结果与配对统计 |

### 12.1 生成与评分指纹分开

`generation_fingerprint` 包括：

- 有效生成配置；
- 模型、tokenizer、视觉组件与预处理版本；
- 实际生成依赖的 `generation_source_hash`，包括 adapter、decoder、输入构造和必要的上游补丁；
- 数据 ID manifest；
- 问题、选项、公开 context 和图像实际内容的校验和；不能只记录可被替换的路径；
- prompt 模板；
- attention 语义、backend、dtype、cache policy；
- 随机设置。

生成指纹使用“生成配置投影”，不能直接 hash 整份 `ResolvedConfig`：排除 normalizer/parser/scorer、评分主指标、run 名称、输出目录和不影响决策的纯记录开关。源码也只纳入影响原始生成的依赖；全仓库 Git commit 与 dirty 状态仍保存为 provenance，但不单独决定预测复用资格。修改 prompt、adapter、decoder 或有效生成参数必须改变生成指纹；仅修改评分代码则不应改变。

`evaluation_fingerprint` 包括：

- 预测产物 hash；
- normalizer、parser、scorer 的实际 `evaluation_source_hash`；
- metric 配置和 scorer 随机设置；
- gold 标注、允许答案域和聚合元数据的版本及校验和。

将评分协议部分另存为 `evaluation_spec_hash`，供跨 run 比较使用。两个 run 的预测不同，其完整 `evaluation_fingerprint` 通常也不同；配对比较应核对评分协议兼容，不能要求包含预测 hash 的完整指纹相等。

不要把 parser 改动当成需要重新生成的理由，也不能在 prompt 已变化后复用旧预测。

### 12.2 续跑规则

- 样本 ID 必须唯一，目录文件名使用安全编码。
- 单例产物先写临时文件，再原子替换。
- 只有 sample_id、generation fingerprint 匹配且状态成功时才可跳过。
- 配置变化、输入变化、代码变化不能混入原 run。
- 缺失、重复、错误和成功样本分别报告。
- 最终按 dataset manifest 顺序聚合，不按文件枚举顺序。
- preflight 发现数据缺失时先终止；不能生成一部分后静默缩小样本分母。
- 计时环境变化时，不混合汇总为同一硬件时延结果。

需要随机数的 Native 或其他路径，使用独立的逐样本 seed：由固定的 `run.seed`、`sample_id` 和 `replicate_id` 经版本化 SHA-256 规则派生，不能使用 Python 进程随机化的 `hash()`，也不能依赖已经处理过多少题。保存派生值，并在每题开始前重置实际使用的 Python、NumPy、PyTorch/CUDA RNG；分叉从同一快照 RNG 状态继续。这样续跑不因跳过前面样本而改变后面样本。当前 CD-APC argmax 不需要人为注入采样噪声。

---

## 13. 机制轨迹与指标口径

### 13.1 每轮最小轨迹

抽样 trace 保存：

- sample_id、iteration、state hash；
- pending 数、搜索位置、requested W、实际 \(|C_t|\)；
- proposal 的 b/c、u/r、pool mass、pool size；
- gate target、各门通过与否；
- eligible 数、计划提交位置；
- 实际 written_ids、reason；
- 是否发生 actual switch；
- 本轮与累计计算量；
- 若为 CCAW：压力分量、支持样本数、EMA、目标 W、下一轮 W。

### 13.2 不混用的比例

分别报告：

1. proposal flip rate：被观察的 position-state 中 \(c\ne b\) 的比例；
2. qualified flip rate：正常门控合格事件中的 flip；
3. actual committed switch rate：实际写入值不同于当时 b 的比例；
4. sample coverage：至少出现一次 actual switch 的题目比例；
5. fallback rate；
6. 最终输出差异比例；
7. Full-only correct 与对照-only correct。

每个指标必须同时保存分子、分母和作用范围。

全 buffer 与最终 EOS 前位置分开统计。由于双向交互，最终 EOS 后曾参与计算的位置不能被直接断言“对前文没有影响”；它们可以作为单独分析层，不应事后从所有轨迹统计中无说明删除。

---

## 14. 同一状态的一次改选分叉实验

这是一项实验驱动器，不注册为新的推理方法。

### 14.1 事件选择

使用预先固定的样本集合。每题至多选一次：

- `full + fixed` 正常运行；
- 第一次 `reason=qualified` 且原计划提交集合内有 \(c_i\ne b_i\) 的事件；
- 多个位置满足时，选择按现有排序排在最前的那个；
- 不能按最终答案是否改善来挑事件。

没有事件的样本记录 `no_eligible_flip`，报告覆盖率，不能通过放宽门阈值追求触发率。

### 14.2 快照

保存：

- PreparedInput 的可复原引用与 hash；
- 完整 DecodeState；
- 本轮候选/门控/计划提交摘要；
- 目标位置 i、b、c；
- RNG 状态与环境信息；
- 前缀已发生的计算量；
- continuation policy。

离线 replay 重新计算该状态，核对候选与事件标识。若不能重现，报告 `replay_mismatch`，不能强行沿用不一致事件。

### 14.3 第一批只做两分支

| 分支 | 目标位置 i | 其余原计划提交 |
|---|---|---|
| A | 写 b | 与 B 完全相同 |
| B | 写 c | 与 A 完全相同 |

之后都使用 `full + fixed` 继续。

统计：

\[
\Delta_{\rm paired}
=\frac{N_{\text{B正确A错误}}-N_{\text{A正确B错误}}}{N_{\rm valid\ pairs}}.
\]

分开保存 `triggered_events`、`replay_mismatch`、`execution_failure` 和 `valid_pairs`。公式分母为有效配对的四格计数之和；无法解析但按预定评分规则计错的答案仍属于有效配对，不能当作技术失败剔除。技术失败与无事件样本都要明确报告。

同时报告事件触发数占所有预先选定样本的比例，以及 valid_pairs 占 triggered_events 的比例。事件子集结果不能直接冒充整个测试集提升。

保存四格计数、配对置信区间、后续计算量。统计单位为题目；同一题的多个 token 不作为独立样本。

### 14.4 后续可选四分支

只有要验证“改选后优先单独提交”时，才增加：

- C：只写 b，其他位置留 MASK，下一轮重算；
- D：只写 c，其他位置留 MASK，下一轮重算。

比较 \((D-B)-(C-A)\)，并检查实际成本。这是后续研究实验，初版推理算法不包含该提交策略。

分叉实验的额外研究计算量与普通部署推理成本分别记录。

---

## 15. CCAW 后续施工规范：ccaw_v1

**本节在固定窗口和分叉实验通过后施工。这里明确选择一份重建后的控制规范，不宣称它等于任何旧分支，也不把规范修正本身当成新增创新点。所有使用该版本的性能需要重新测量。**

### 15.1 先冻结控制对象

- 只调整下一轮搜索容量 W。
- 不改变当前 logits、candidate、u/r 或 gate。
- 不同时调整 K_max。
- 不改变正常/消融分支输入长度。
- 不在 selector 内额外扩窗寻找合格候选。
- 没有合格候选时按共同 fallback 规则推进。

### 15.2 压力观测与归一化

在本轮提交前的 \(C_t\) 上定义：

\[
f_{\rm def}
=\frac{|C_t\setminus E_t|}{|C_t|},
\]

\[
f_{\rm flip}
=\frac{|\{i\in E_t:c_i\ne b_i\}|}{|C_t|}.
\]

令 \(H_t\) 为本轮窗口内、且上一轮具有有效候选观测的未提交位置：

\[
f_{\rm inst}=
\begin{cases}
\frac{|\{i\in H_t:c_i^{(t)}\ne c_i^{(t-1)}\}|}{|H_t|},
& |H_t|>0,\\
0,& |H_t|=0.
\end{cases}
\]

保存 `instability_support = |H_t|`；首次无观测不能解释为有证据证明稳定。

`previous_observed` 每轮按第4.3节重建。因此某位置离开窗口后再次进入时，不能把数轮之前的候选当作上一轮候选；没有相邻轮有效观测的该位置不进入本轮 `H_t`。

具体更新为 \(\texttt{previous\_observed}_{t+1}=\mathbf1_{C_t}\land\texttt{pending}_{t+1}\)。“本轮观测”特指进入 \(C_t\)，不能因为完整前向算出了所有响应位置的 logits，就把窗口外位置标为已观测。

因前两项互斥，满足：

\[
0\le f_{\rm def}+f_{\rm flip}\le1,\qquad 0\le f_{\rm inst}\le1.
\]

本重建规范选择：

\[
d_t=\frac{f_{\rm def}+f_{\rm flip}+f_{\rm inst}}{2}\in[0,1].
\]

**本节重新冻结了观测量和归一化，不能把它描述成只将旧代码的除数 3 改成 2。** 对本节定义的三个分量，若直接除以 3，压力最大只有 \(2/3\)；这一上界依赖于本节前两项的互斥定义，不能直接套用于旧实现的 JSD 加权分量。本文选择除以 2，是一项需要记录和消融的工程定义，不是已有实验结论，也不保证实际轨迹一定遍历完整窗口范围。

不要从旧代码加入 JSD 权重、history confidence penalty 或额外 pressure scale。若后续研究需要它们，创建新 controller version 并分别比较。

### 15.3 唯一状态更新

\[
\bar d_t=\gamma\bar d_{t-1}+(1-\gamma)d_t,
\]

\[
W_t^\star=W_{\max}
-\operatorname{round}_{\rm half\ up}
[(W_{\max}-W_{\min})\bar d_t],
\]

\[
\Delta_t=\operatorname{clip}
(W_t^\star-W_t,-s_{\rm shrink},s_{\rm expand}),
\]

\[
W_{t+1}=\operatorname{clip}(W_t+\Delta_t,W_{\min},W_{\max}).
\]

- 只影响下一轮。
- 每轮只调用一次 update。
- 下一轮开始禁止重新用 EMA 直接覆盖 W。
- 正数 half-up 使用明确的 `floor(x + 0.5)` 规则，不能依赖 Python bankers rounding。
- 先计算压力，再更新候选观测历史。
- 没有候选窗口时已经结束，不更新压力。

建议起点：

~~~yaml
window:
  kind: ccaw_v1
  initial: 16
  minimum: 16
  maximum: 64
  ema_decay: 0.8
  expand_step: 4
  shrink_step: 8
~~~

`K_max=16` 保持不变。这些是新实验起点，不是对旧论文参数的追认。

### 15.4 CCAW 验收

- 持续低压、高压、交替压力序列均满足边界和步长限制。
- 初始 W 确实等于 initial，不自动跳到 maximum。
- EMA 确实进入 target 计算。
- 当前冻结状态的候选与 gate 不随 controller 开关变化。
- `minimum=maximum=initial=W_fixed` 时，全轨迹退化为 fixed。
- 压力各分量、支持数、EMA、target、实际 W 全部可追踪。
- 某位置离开窗口一轮后再次进入时，不进入本轮 H；对应 history 测试必须覆盖该过程。
- fixed 与 ccaw_v1 比较时，除窗口控制外其余配置相同。

---

## 16. 测试矩阵

只写能够发现具体风险的测试，不以“测试数量多”作为目标。

| ID | 测试 | 必须验证的结果 |
|---|---|---|
| U01 | 两分支相同 | c=b，r=u |
| U02 | alpha=0 与 gain_zero | 候选、门、排序、写回一致 |
| U03 | APC singleton | m=p_I(b)，q(b)=1，r=u |
| U04 | 合格改选数值 fixture | 第二候选通过 .1/.9 双门 |
| U05 | 四种 u/r 开关 | 真值表正确，候选与分数不被开关改写 |
| U06 | 全部门失败且 c≠b | fallback 最终只写一个 b |
| U07 | 稳定并列 | 排序与选点可复现 |
| U08 | W 与 K 分离 | W=16、K=4 时最多写4个 |
| U09 | 状态保护 | prompt不变、已提交不覆盖、MASK严格减少 |
| U10 | 非法配置/方法 | 明确错误，不能落入 Native |
| U11 | 数值边界 | 极小 p_A、非法词表、alpha=0 无NaN |
| E01 | 旧 parser 两个反例 | 提取明确末尾答案 |
| E02 | EOS 后另一个答案 | 不影响评分文本 |
| E03 | 无EOS/首位EOS/多EOS | 规范化结果按合同 |
| E04 | 无法解析 | 明确状态，无随机严格兜底 |
| E05 | gold 隔离 | parser 不接收 gold，input builder 不接收 GoldRecord |
| I01 | Native parity | 包装调用与直接官方调用一致 |
| I02 | visual forward parity | 同状态、同注意力语义 logits 一致 |
| I03 | mask 有效性 | 对应视觉路径被实际阻断 |
| I04 | paired/separate | 数值和决策差异在预定标准内 |
| I05 | 无隐藏缓存 | 相同 state 结果不受先前调用次序影响 |
| R01 | 中断续跑 | 与完整运行的样本集合和结果一致 |
| R02 | 指纹变化 | 拒绝复用不匹配的预测 |
| R03 | trace 开关 | 相同 token 轨迹 |
| A01 | full/crossed 冻结状态 | 提交位置相同，仅写回值可变 |
| A02 | full 轨迹上无实际改选的样本 | full/crossed 完整轨迹一致；不能按 crossed 筛选 |
| A03 | 分叉控制 | 只改变指定位置，随后同一 continuation |
| W01 | CCAW 更新与 history | 单次更新、EMA生效、参数域/扩缩边界正确；离窗再入窗不复用过期观测 |
| W02 | 固定范围退化 | CCAW min=max 时等于 fixed |

U04 固定 fixture：

\[
p^I=(0.7,0.2,0.1),\quad
p^A=(0.8999,0.0001,0.1),\quad
\alpha=0.5,\ \beta=0.1.
\]

预期：

\[
b=0,\quad c=1,\quad u_c=0.2,\quad r_c\approx0.92575.
\]

它只验证公式和门控允许改选，不验证候选正确性。

---

## 17. 分阶段施工任务与验收门

### P0：建立规范和依赖清单

交付：

- 新工程目录和 package；
- `ALGORITHM_SPEC.md`、`PROTOCOL_SPEC.md`、`DECISIONS.md`；
- `upstream.lock.json`；
- 严格配置 schema 与方法注册。

验收：

- 不含 DCD 实现/注册/占位；
- 没有旧 bundle 路径依赖；
- 所有 PIN_REQUIRED 可见，未填时 doctor 失败；
- unknown/unused 参数报错；
- Native 与 CD-APC 的参数域分开。

### P1：先建立数据和独立评分

交付：

- M³CoT loader、Sample/GoldRecord 分离；
- 固定 smoke IDs；
- 输出规范化、official scorer 绑定、strict parser；
- run artifacts 与指纹；
- 两个旧 parser 反例及 EOS 测试。

验收：

- 不加载模型也能完成 synthetic prediction 的评分；
- 数据过滤和 ID 清单一致；
- 解析失败、歧义和样本错误不被静默删除；
- 修改 parser 后生成指纹保持不变、评分指纹变化。

### P2：实现纯数学核心和状态机

交付：

- proposal、gates、selection、state；
- fixed window；
- full 与三种诊断变体的绑定表；
- 使用 mock logits 的 engine 测试。

验收：

- U01–U11、A01 通过；
- fallback 写入实际 b；
- gate 开关不被其他路由覆盖；
- 所有非终止步有进展；
- 不调用任何旧解码模块。

### P3：接入官方 Native 与两个模型 adapter

可以并行开发 MMaDA 与 LaViDa adapter，但共享接口和核心保持单一。

交付：

- NativeRunner；
- 正常/参考完整前向；
- attention 语义和 mask 传递；
- 执行点成本计数；
- 必要的最小上游补丁及说明。

验收：

- I01–I05 通过；
- Native 的 block/steps 等参数实际由官方路径执行；
- CD-APC 的参考分支确实生效；
- 基础计数无需开启 trace；
- 真实小样本可生成并离线评分。

### P4：跑通最小实验闭环

先用预先固定的小样本 manifest 验证工程，之后再运行主诊断数据集。

交付：

- Native、full fixed；
- gain_zero、timing_only_crossed、base_candidate_visual；
- 中断续跑；
- per-sample scores、配对统计、成本汇总；
- 明确的失败样本清单。

验收：

- 同一协议下输入和评分一致；
- no-switch 一致性通过；
- 所有统计分母明确；
- 全部预测可由 manifest 和 source hash 追溯；
- 不使用旧表格中的分数补齐新结果。

### P5：共同状态分叉归因

交付：

- 首次 eligible flip 快照；
- b/c 两分支 replay；
- 四格正确性表；
- 触发覆盖率与后续计算量；
- replay mismatch 报告。

验收：

- A02、A03 通过；
- 每题至多一个预先规定事件；
- 不挑成功案例；
- 允许实验结论为无净收益，不以增加模块掩盖负结果。

### P6：后置接入 ccaw_v1

交付：

- 本文第15节的唯一控制器；
- fixed退化和压力序列测试；
- fixed/ccaw_v1 配对实验；
- 对归一化变化的显式说明。

验收：

- W01–W02 通过；
- controller只改变搜索窗口；
- 无selector内部隐式扩窗；
- 不把新结果标为旧CCAW配置复现。

### P7：扩展正式实验与创新点筛选

交付：

- MMMU、MMBench、HallusionBench、VLind-Bench等任务接入；
- 两基座一致算法下的主表；
- 预先规定的参数敏感性；
- 质量—成本曲线；
- 失败类型与格式敏感性；
- 如有必要，独立的新机制实验。

本阶段仍不把 DCD 当作依赖。未来接入外部对照时另开施工任务，不回头修改已经冻结的主方法语义。

---

## 18. 拟实现的 CLI 与操作顺序

**以下命令是施工完成后应提供的目标接口，目前不是已存在的可执行程序。**

### 18.1 安装与预检查

~~~bash
python -m pip install -e '.[dev]'
python -m cdapc_lab doctor \
  --config configs/experiments/mmada_m3cot_fixed_smoke.yaml
~~~

doctor 应验证版本锁、文件存在、数据数量、图片有效性、未知参数、mask/EOS IDs、设备、backend能力及输出目录，不启动完整评测。

### 18.2 CPU 验收

~~~bash
python -m pytest -q tests/unit tests/evaluation
~~~

纯数学和解析测试不依赖真实模型权重。GPU模型验证单独运行，不能因CPU测试通过而宣称adapter正确。

### 18.3 模型验收

~~~bash
python -m cdapc_lab verify-native \
  --config configs/experiments/mmada_native_smoke.yaml

python -m cdapc_lab verify-adapter \
  --config configs/experiments/mmada_m3cot_fixed_smoke.yaml

python -m cdapc_lab verify-native \
  --config configs/experiments/lavida_native_smoke.yaml

python -m cdapc_lab verify-adapter \
  --config configs/experiments/lavida_m3cot_fixed_smoke.yaml
~~~

### 18.4 生成与评分分开

~~~bash
python -m cdapc_lab infer \
  --config configs/experiments/mmada_m3cot_fixed_smoke.yaml

python -m cdapc_lab score \
  --run runs/RUN_ID \
  --profile official_and_strict

python -m cdapc_lab summarize \
  --run runs/RUN_ID \
  --evaluation EVAL_ID
~~~

`RUN_ID` 由 infer 返回，`EVAL_ID` 是 score 返回的完整 `evaluation_fingerprint`。score 同时返回 `evaluation_spec_hash`。不能用含义模糊的“最新目录”自动猜测；只有唯一匹配时才允许省略评分版本参数，没有匹配或有多个匹配均报错。

### 18.5 续跑与分叉

~~~bash
python -m cdapc_lab infer \
  --config configs/experiments/mmada_m3cot_fixed_smoke.yaml \
  --resume-run runs/RUN_ID

python -m cdapc_lab replay \
  --run runs/RUN_ID \
  --event first_eligible_flip \
  --design content_swap
~~~

分叉驱动器必须输出自己的研究任务记录，不能覆盖原 run 的 predictions。

### 18.6 配对比较

~~~bash
python -m cdapc_lab compare \
  --run-a runs/RUN_A \
  --evaluation-a EVAL_A \
  --run-b runs/RUN_B \
  --evaluation-b EVAL_B \
  --unit sample_id \
  --metric strict_accuracy
~~~

`EVAL_A`、`EVAL_B` 分别是两边 score 返回的评分指纹。`strict_accuracy` 是评分产物中的指标名，`strict_final_v1` 是 parser ID，`official_and_strict` 是评分任务 profile；三者不能混用。compare 读取明确选定的既有评分记录，不调用模型，也不自动选最新评分。

若任务存在同题选项置换或共享统计单元，协议要求使用 group_id 聚类重采样。先验证两个 run 的样本 ID、gold 版本和 `evaluation_spec_hash` 兼容，不能直接按行号相减。

---

## 19. 实验扩展顺序与报告规则

### 19.1 推荐顺序

1. Synthetic fixtures 和 parser。
2. 官方 Native / adapter parity。
3. 固定 smoke manifest 的端到端闭环。
4. M³CoT 固定窗口机制诊断。
5. b/c 状态分叉。
6. LaViDa 与 MMaDA 的跨基座复核。
7. CCAW独立比较。
8. 其他任务与质量—效率曲线。
9. 有证据后再筛选一个新增机制。

不得把所有模型、所有任务、所有消融做成一个未经验证的大网格直接启动。

### 19.2 参数敏感性

首轮固定参数。后续只围绕具体风险做小范围预先规定的敏感性：

- alpha 对改选覆盖率、净收益和有害改选的影响；
- beta 对 pool singleton、pool mass和候选可达性的影响；
- tau_u/tau_r 对门通过率、fallback、轮数与准确率的影响；
- W 与 K 的独立作用；
- L 对截断、EOS 后计算和答案解析的影响。

所有结果报告完整预设范围。若选择某组参数用于主表，记录选择依据和使用的数据；不得用最终测试集的最高点伪装成默认配置。

### 19.3 正式结果最低内容

- 精确数据集版本、split、样本量；
- 模型/视觉组件与prompt版本；
- 生成与评分指纹；
- 两个分支的定义；
- 实际生效参数；
- official与strict评分及解析失败率；
- actual switch覆盖率、fallback率；
- 实际迭代数、模型调用数、分支评估数；
- 时延、显存、输出长度；
- 配对差值与置信区间；
- 缺失/失败/不支持样本说明。

正向、零效果和负向实验都保留原始记录。没有确认的机制不能在论文中改写成确定因果解释。

---

## 20. 给施工者的最终交付清单

### 必交

- [ ] 完整新源码及安装说明。
- [ ] 固定官方依赖/模型版本的 lock 文件。
- [ ] 唯一算法规范与协议规范。
- [ ] Native 和 CD-APC 两种方法入口。
- [ ] MMaDA 与 LaViDa adapter。
- [ ] 固定窗口、双门、raw fallback。
- [ ] 三种机制诊断预设。
- [ ] 无gold依赖的parser与独立scorer。
- [ ] 统一EOS规范化。
- [ ] 两种指纹、原子产物和断点续跑。
- [ ] 真实执行成本计数。
- [ ] 首次eligible flip快照与b/c replay。
- [ ] 必要测试及逐项验收报告。
- [ ] 一个可完整复原的小样本运行。
- [ ] 后续ccaw_v1独立提交与实验记录。

### 不得用来宣告完成

- 仅跑通 imports 或只做语法检查；
- 仅与旧表格某个分数接近；
- 只检查配置中门开关的值；
- 只检查attention mask的shape；
- 只保存最终accuracy，不保存逐例输出；
- 使用旧预测填补新run；
- 把未执行参数打印出来当实验设置；
- 将一次batch-2调用写成单分支计算；
- 将机制诊断伪装成Native或DCD。

### 施工顺序硬门

**输入与评分正确 → 纯数学核心正确 → Native与adapter一致性 → 固定窗口闭环 → 改选归因 → CCAW → 正式实验和新增机制。**

---

## 附录 A：审计证据链接

以下链接用于核对旧行为，不能作为新工程的文件导入来源。

1. [旧 MMaDA generic CD-APC runner](https://github.com/HostHarry/APC/blob/e751eda6757145f6ba0560152e36be11a03ce953/MMaDA_DCD_cloud_bundle_20260711/MMaDA/evaluation/VLMEvalKit/scripts/run_cd_apc_generic.sh)  
   `cv_dcd + cd_apc + min_base_blended` 与后续双门路径不同。

2. [MMaDA 模型分发](https://github.com/HostHarry/APC/blob/e751eda6757145f6ba0560152e36be11a03ce953/MMaDA_DCD_cloud_bundle_20260711/MMaDA/models/modeling_mmada.py)  
   核对真实策略分支，不能以脚本名称推断算法。

3. [MMaDA decoder 字节码包装器](https://github.com/HostHarry/APC/blob/e751eda6757145f6ba0560152e36be11a03ce953/MMaDA_DCD_cloud_bundle_20260711/MMaDA/decoding/decoder.py) 与 [history 源码](https://github.com/HostHarry/APC/blob/e751eda6757145f6ba0560152e36be11a03ce953/MMaDA_DCD_cloud_bundle_20260711/MMaDA/decoding/history.py)  
   关键逻辑源码缺失以及所需字段/参数的版本错配。

4. [LaViDa 配置](https://github.com/HostHarry/APC/blob/56a6d2b65e63b00450f0b24e031011fa69fcc31f/llava/decoding/config.py)、[decoder](https://github.com/HostHarry/APC/blob/56a6d2b65e63b00450f0b24e031011fa69fcc31f/llava/decoding/decoder.py) 和 [运行脚本](https://github.com/HostHarry/APC/blob/56a6d2b65e63b00450f0b24e031011fa69fcc31f/eval/run_vchd_llada.sh)  
   fallback默认False、整buffer完成后EOS截断，以及实际参数。

5. [LaViDa 窗口控制](https://github.com/HostHarry/APC/blob/56a6d2b65e63b00450f0b24e031011fa69fcc31f/llava/decoding/window.py)  
   压力定义、EMA与有界更新；需结合decoder调用顺序审查。

6. [LaViDa adapter](https://github.com/HostHarry/APC/blob/56a6d2b65e63b00450f0b24e031011fa69fcc31f/llava/decoding/lavida_adapter.py)  
   成对分支与prefix mask语义。两套cache确实分离，不能误写为reference直接复用visual cache。

7. [固定官方 LaViDa attention 实现](https://github.com/jacklishufan/LaViDa/blob/7ca2f5aa1ddc6b07f0bff902ff9338e66c1fc70b/llava/model/language_model/llada/modeling_llada.py)  
   已审SDPA路径显式传入`attn_mask=None`；新adapter必须验证实际干预。

8. [LaViDa M³CoT parser](https://github.com/HostHarry/APC/blob/56a6d2b65e63b00450f0b24e031011fa69fcc31f/eval/lmms_eval/tasks/m3cot/utils.py)  
   前文两个合成反例来自该版本的实际函数验证。

9. [43030 分支的 M³CoT task](https://github.com/HostHarry/APC/blob/aaa0170d24c0cc13d783e57d4dfd70edcf86864a/eval/lmms_eval/tasks/m3cot/m3cot_full.yaml) 与 [数据过滤代码](https://github.com/HostHarry/APC/blob/aaa0170d24c0cc13d783e57d4dfd70edcf86864a/eval/lmms_eval/tasks/m3cot/utils.py)  
   同名任务在不同分支具有不同过滤行为，数据manifest不能省略。

---

## 附录 B：给编码代理的短任务模板

> 在新的 `cdapc-experiments/` 目录施工。不得复制或导入旧实验代码；不实现DCD，不创建DCD占位。第一篇保留两个完整前向，关闭cache、第三门和历史分布重加权。先完成本文P0–P4，再做状态分叉；固定窗口验收后才接入独立版本的ccaw_v1。Native调用固定官方生成器，不能用gain_zero替代。所有配置严格校验，所有参数必须实际消费。生成与评分分离，所有方法共用数据和输出规范。先用合成与小样本验收，不启动未经配置的大规模实验。每完成一阶段，提交实际修改文件、通过的验收、尚存限制与下一阶段工作；不得把未运行的测试或未完成的模型实验写成成功。
