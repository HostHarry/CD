# MMMU `mmmu_wino_reasoning_v1` 输入复现

本文对应已完成的 Native / CDAPC r90 / CDAPC r50 三组 MMMU validation 实验。三组使用相同输入构造。以下路径均相对于 `cdapc-experiments` 项目根目录；上传包中的原始文件位于 `source/` 下。

## 正文与选项格式

放在 **user message 内**的完整推理指令是：

```text
You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here
```

多选题的正文为：

```text
{上述指令}
{question}
A. {option_A}
B. {option_B}
C. {option_C}
D. {option_D}
```

选项数取决于题目，字母从 A 连续编号。开放题只有指令和 question。上下文构造保留末尾换行，随后 tokenizer 的 chat template 会对 message content 做 `trim`。这套 MMMU prompt 没有附加 `Final answer: (X)`；项目里该要求属于另一任务。

以下 Python 代码复现正文和选图规则，可独立复制：

```python
import ast
import re
from string import ascii_uppercase

REASONING = (
    "You should first think about the reasoning process in the mind and then "
    "provide the user with the answer. The reasoning process is enclosed within "
    "<think> </think> tags, i.e. <think> reasoning process here </think> answer here"
)

def build_mmmu_input_parts(row):
    # 原数据的 options 列是 Python list 的字符串表示。
    choices = (ast.literal_eval(row["options"])
               if row["question_type"] == "multiple-choice" else [])
    options = {ascii_uppercase[i]: str(value) for i, value in enumerate(choices)}
    option_text = "\n".join(f"{letter}. {value}" for letter, value in options.items())

    # 必须在清理占位符之前选图；题干和选项都参与选图。
    source = row["question"] + "\n" + option_text
    placeholders = sorted(set(re.findall(r"<image \d+>", source)))
    image_key = (placeholders[0].strip("<>").replace(" ", "_")
                 if placeholders else "image_1")
    image_record = row[image_key]
    if not isinstance(image_record, dict):
        image_record = row["image_1"]
    image_bytes = image_record["bytes"]

    context = (f'{row["question"]}\n{option_text}\n'
               if options else f'{row["question"]}\n')
    context = context.replace("<image>", "")
    context = re.sub(r"<image \d+>", "", context)
    context = re.sub(r"<image\d+>", "", context)
    user_content = REASONING + "\n" + context
    return user_content, image_bytes, image_key
```

## 多图题的实际处理

- 搜索范围是原始题干 **加全部选项**。
- 选图只匹配带空格的 `<image n>`，先去重，再按字符串排序，取第一项。不是按它在题目中第一次出现的位置选择。
- 没有匹配项时选 `image_1`；所选字段不是字典时，数据准备脚本回退到 `image_1`。
- 模型实际只接收这一张图，不拼图、不传入其余图片。
- 文本清理会删除三种占位符：`<image>`、`<image n>`、`<imagen>`。因此选图匹配规则与文本清理规则并不完全相同。
- 当前 900 条 validation 数据有 **43 条多图题、857 条单图题**；保存的 `gold.json` 中，900 条的 `used_image` 均为 `image_1`。

这是当前实验的单图协议。另一服务器若对多图题传入全部图片，所得结果就不再是本次输入协议的复现。

## tokenizer 与视觉输入

实际使用模型自带的 tokenizer chat template：

```python
text_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": user_content}],
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to(device)
```

没有独立 system message。此 checkpoint 的模板会添加 BOS、user header、`<|eot_id|>` 以及 assistant generation header，且会 trim user content。请使用同一 checkpoint 的 `tokenizer_config.json`，不要另外手写一套 header。

图像与完整输入的构造如下，`vision` 为 MAGVITv2，`prompting` 为 MMaDA 的 `UniversalPrompting`：

```python
from io import BytesIO
from PIL import Image
import torch
from training.utils import image_transform

with Image.open(BytesIO(image_bytes)) as image:
    pixels = image_transform(image.convert("RGB"), resolution=512)
pixels = pixels.unsqueeze(0).to(device)
image_ids = vision.get_code(pixels) + len(tokenizer)
assert image_ids.shape == (1, 1024)

token = lambda name: prompting.sptids_dict[name].reshape(1, 1).to(device)
prompt_ids = torch.cat(
    [token("<|mmu|>"), token("<|soi|>"), image_ids,
     token("<|eoi|>"), text_ids],
    dim=1,
).long()
```

图像变换是 RGB → 短边 Resize(512, bicubic) → CenterCrop(512×512) → ToTensor → mean/std 均为 0.5 的归一化。它保留缩放时的宽高比，再中心裁剪。

输入结构为：

```text
<|mmu|><|soi|>[1024 个视觉 code，偏移量为 len(tokenizer)]<|eoi|>[chat-template 文本]
```

本实现没有额外手动插入 `<|sot|>`。chat template 自带的 BOS `<|startoftext|>` 会保留一次，其 ID 为 126080，与 MMaDA 的 `<|sot|>` 别名相同。CDAPC 的 visual-key mask 只标记这 1024 个 code，不包括 `<|soi|>` / `<|eoi|>`。生成时在完整 prompt 后追加 256 个 response mask；256 是输出画布长度，不是含图像和问题的总序列长度。

## 原始实现入口

| 文件 | 入口 / 作用 |
| --- | --- |
| `src/cdapc_lab/data/mmmu.py` | `mmmu_context()`、`strip_image_placeholders()`、`choose_image()` |
| `src/cdapc_lab/data/records.py` | `TASKS["mmmu"]`、`build_question()`、`uses_model_reasoning_instruction()` |
| `scripts/prepare_mmmu.py` | 从 parquet 读取题目和选项，选择并保存单张图片 |
| `src/cdapc_lab/models/mmada.py` | `OFFICIAL_THINKING_INSTRUCTION`、`MMaDAAdapter.prepare()` |
| `upstream/mmada/training/utils.py` | `image_transform()` |
| `models/mmada/tokenizer_config.json` | checkpoint 自带的 chat template |

版本与生成预算：

- 数据：`lmms-lab/MMMU`，revision `364f2e2eb107b36e07ff4c5a15f5947a759cef47`，`data/validation-00000-of-00001.parquet`。
- MMaDA checkpoint/tokenizer revision：`3ee0085f0c42541f1134aae30482954451952406`。
- MMaDA upstream revision：`3cdb8709635beb29f66e4e23cd66060872ac3996`。
- MAGVITv2 revision：`5c3fa78f8b3523347c5cd1a4c97f3c4e96f33d5d`。
- 随机种子：`20261105`；batch size 1；BF16；SDPA；双向注意力。
- Native：128 steps，block length 64，temperature 0，low-confidence remasking，CFG 0。
- CDAPC：最大 256 次迭代，画布 256，W=K=16，alpha=0.5，beta=0.1，u=0.1，r90/r50 分别取 r=0.9/0.5；按 `eos_prefix_complete` 停止。

本说明描述当前执行过的输入协议。WINO 的题目正文和推理指令是参考来源，但完整多模态输入布局使用本项目 MMaDA README 协议，不能据此声称完整复现 WINO 的输入实现。
