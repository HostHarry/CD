# LLaDA-V CDAPC：POPE-COCO 64/64 实现

本目录发布正在运行的 LLaDA-V 64/64 实验源码、六个分片配置和固定输入 ID。
`src/`、`scripts/run.py`、六个运行配置及输入 ID 与本机
`cdapc-lladav-64` 逐字节一致。全量实验尚未完成，本包不包含其最终分数。
历史提示词 pilot 见 [lladav_prompt_pilot](../results_20261112/lladav_prompt_pilot/)。

## 实现与协议

- `src/cdapc_lab/models/lladav.py`：官方 LLaDA-V 多模态模板、any-resolution 图像
  预处理及视觉位置映射；使用 float16、SDPA、双向注意力，禁用 KV cache。
- `models/base.py`、`config.py`、`data/records.py`、`experiments/infer.py`：
  backend 注册、配置校验、提示词及模型/数据/源码指纹校验。
- `decoding/`：共享 CDAPC 引擎及快速 EOS 前缀候选策略。视觉消融的 additive
  bias 经 `isfinite` 转为官方 LLaDA-V 所需的二值 attention mask。
- EOS 使用 `<|eot_id|>`（126348），MASK 为 126336；按官方 Native 实现抑制
  126081、126080、126346、126347。该 EOS 与 tokenizer 默认 EOS 不同。
- 输入为 POPE-COCO 的 5127 个唯一图像/问题输入，映射至 9000 行、
  random/popular/adversarial 各 3000 行。seed 为 20261105。
- 两组使用同一 `pope_yesno_reasoning_v1`：原问题、Final answer 要求以及
  LLaDA-V `/think` 标记。它与 MMaDA 的无 Final answer 后缀实验提示不同。
- Native：画布 64、64 步、block length 64、temperature 0。
  CDAPC：L64/I64、alpha 0.5、beta 0.1、u 0.1、r 0.7、W16/K4，
  sequential 分支、`eos_prefix_complete` 停止。

## 环境与固定依赖

本机使用 Python 3.12.3、PyTorch 2.8.0+cu128、Transformers 4.50.3。
三张 RTX 5090 各运行一个分片，单卡实际占用约 30–31 GiB 显存。
先安装适合机器的 CUDA PyTorch，再安装本包：

```bash
cd cdapc/lladav_64
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
DS_BUILD_OPS=0 python -m pip install -r requirements-inference.txt
python -m pip install -e '.[dev]'
mkdir -p upstream models logs reports
git clone https://github.com/ML-GSAI/LLaDA-V upstream/lladav
git -C upstream/lladav checkout f8b02ce04b09f4f271fe55a3652059a73bbc7a32
huggingface-cli download GSAI-ML/LLaDA-V \
  --revision a10ba1790083e10e07cfcab6e7be93e1823f5792 \
  --include '*.json' '*.safetensors' --local-dir models/lladav
huggingface-cli download google/siglip2-so400m-patch14-384 \
  --revision e8e487298228002f3d8a82e0cd5c8ea9c567f57f \
  --include '*.json' '*.safetensors' '*.model' --local-dir models/siglip2
python scripts/prepare_pope.py
```

`prepare_pope.py` 下载固定 revision 的 POPE parquet 和 RUCAIBox 问题/评分脚本，
并对照本仓库 `protocol/images.tsv` 与 `protocol/pope_coco.manifest.json`
验证图片和问题哈希。它默认从本目录定位 CD 仓库，也支持 `CD_REPO_ROOT`；
下载缓存遵循 `HF_HUB_CACHE`。这两处路径是相对本机原脚本的可移植性修改。

`upstream.lock.json` 保存原始源提交、模型文件 SHA256、数据 revision 和本机
验证回执；其中历史绝对路径仅为来源记录，运行时使用配置中的相对路径。
换机器后 preflight 会重新核验模型内容并更新文件时间/大小回执。
模型权重、图片和上游第三方代码需按以上固定版本获取，不包含在本包中。

## 验证与运行

```bash
python -m pytest -q tests/unit tests/integration/test_lladav.py
nohup .venv/bin/python -u scripts/run.py > logs/controller.log 2>&1 < /dev/null &
cat reports/status.json
```

控制器按 Native → CDAPC → 评分顺序执行，每组使用 GPU 0/1/2。
若分片失败，控制器记录 `needs_attention` 并退出。确认旧控制器和 worker
已退出、GPU 可用后，可用同一条命令续跑：它扫描现有预测，仅调度缺失 ID。
续跑时须保留配置、模型、数据和源码不变；不要把不同设置的输出混入 `runs/`。

两组齐全后也可单独运行 `python scripts/run.py --score`。评分优先解析
`Final answer: Yes/No`，其次解析答案区开头 Yes/No，最后使用 POPE 官方首句规则。
预算耗尽项计错并保留分母；三个 split 等权平均；配对 bootstrap 以图像为单位，
重复 2000 次。输出为 `reports/result.json` 和 `reports/summary.md`。

`validation/lladav_acceptance.json` 和 `lladav_cpu_sanity.json` 是此前适配器的
工程验证证据。前者仅为两个固定 M3CoT 输入、L128 验证，不是 POPE 64/64
评测；replay 有效配对数为 0，不构成 content-swap 效果证据。
测试中的配置路径已改为本包实际的 64/64 配置。

`source_manifest.json` 记录本机来源文件及哈希；`SHA256SUMS` 覆盖发布内容
（不含自身）。在环境准备及运行修改 lock 文件之前，可执行
`sha256sum -c SHA256SUMS` 验证发布包。
