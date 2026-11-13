"""LLaDA-V unit/integration tests; upstream `llava` code runs in a subprocess to avoid
colliding with upstream/lavida's own top-level `llava` package."""
from pathlib import Path
import os
import subprocess
import sys
import torch
import pytest
from cdapc_lab.models.base import Adapter, allowed_vocab
from cdapc_lab.types import PreparedInput

ROOT = Path(__file__).resolve().parents[2]
LLADAV = ROOT / "models/lladav"


def _tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(LLADAV, local_files_only=True)


def test_allowed_vocab_eos_override():
    tokenizer = _tokenizer()
    mask_id = tokenizer.convert_tokens_to_ids("<|mdm_mask|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    suppressed = {tokenizer.convert_tokens_to_ids(t) for t in
                  ("<|endoftext|>", "<|startoftext|>", "<|start_header_id|>", "<|end_header_id|>")}
    assert suppressed == {126081, 126080, 126346, 126347}
    vocab = allowed_vocab(tokenizer, mask_id, "cpu", suppressed, eos=eot_id)
    ids = set(vocab.tolist())
    assert eot_id in ids
    assert mask_id not in ids
    assert not (suppressed & ids)
    assert not (set(tokenizer.all_special_ids) - {eot_id}) & ids
    assert tokenizer.eos_token_id == 126081 and 126081 not in ids


def test_allowed_vocab_default_unchanged():
    tokenizer = _tokenizer()
    mask_id = tokenizer.convert_tokens_to_ids("<|mdm_mask|>")
    default = allowed_vocab(tokenizer, mask_id, "cpu")
    ids = set(default.tolist())
    # default eos is 126081 (<|endoftext|>); retained; <|eot_id|> is not in
    # all_special_ids for this tokenizer so the default also retains it
    assert 126081 in ids
    assert tokenizer.convert_tokens_to_ids("<|eot_id|>") in ids
    assert mask_id not in ids


class _StubAdapter(Adapter):
    def __init__(self, ids):
        self._ids = ids
        self.denoiser = torch.nn.Linear(2, 2)

    def direct_native(self, prepared, config):
        return self._ids


def _prepared(mask_id=3, eos_ids=(9,), canvas=8):
    return PreparedInput("t", torch.tensor([[1]]), torch.tensor([True]),
                         torch.tensor([1, 2, 9]), mask_id, eos_ids, None, canvas, "synthetic")


def test_native_finish_reasons():
    prepared = _prepared()
    full = _StubAdapter(torch.tensor([5, 6, 7, 8, 9, 1, 2, 4]))
    assert full.native(prepared, {}).finish_reason == "canvas_filled"
    eos_prefix = _StubAdapter(torch.tensor([5, 6, 7, 9, 3, 3, 3, 3]))
    assert eos_prefix.native(prepared, {}).finish_reason == "eos_prefix_complete"
    budget = _StubAdapter(torch.tensor([5, 6, 3, 9, 3, 3, 3, 3]))
    assert budget.native(prepared, {}).finish_reason == "budget_exhausted"


def test_lladav_mask_semantics_subprocess():
    script = r'''
import sys, torch
sys.path.insert(0, r"%s")
from llava.model.language_model.modeling_llada import LLaDAModel, LLaDASdpaAttention
from llava.model.language_model.configuration_llada import LLaDAConfig
sys.path.insert(0, r"%s")
from cdapc_lab.models.base import visual_ablation_bias

cfg = LLaDAConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64,
                  _attn_implementation="sdpa", use_cache=False)
torch.manual_seed(0)
model = LLaDAModel(cfg).eval()
assert all(isinstance(l.self_attn, LLaDASdpaAttention) for l in model.layers)

T, L = 12, 5
visual_keys = torch.zeros(T - L, dtype=torch.bool)
visual_keys[2:6] = True
embeds = torch.randn(1, T, cfg.hidden_size)
out_none = model(inputs_embeds=embeds, use_cache=False)[0]
out_ones = model(inputs_embeds=embeds, attention_mask=torch.ones((1, 1, T, T)), use_cache=False)[0]
assert torch.equal(out_none, out_ones), "mask=None must equal all-ones binary mask"

bias = visual_ablation_bias(visual_keys, L, dtype=embeds.dtype)
binary = torch.isfinite(bias).to(embeds.dtype)
out_visual = model(inputs_embeds=embeds, attention_mask=binary, use_cache=False)[0]
changed = embeds.clone()
idx = visual_keys.nonzero().flatten()
changed[:, idx] = changed[:, idx].roll(1, 1) + 0.25
out_changed = model(inputs_embeds=changed, attention_mask=binary, use_cache=False)[0]
assert torch.equal(out_visual[:, -L:], out_changed[:, -L:]), "ablated branch must be invariant"
out_changed_nm = model(inputs_embeds=changed, use_cache=False)[0]
assert not torch.equal(out_visual[:, -L:], out_changed_nm[:, -L:]), "visual branch must change"
print("PASS")
''' % (ROOT / "upstream/lladav/train", ROOT / "src")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env)
    assert result.returncode == 0 and "PASS" in result.stdout, result.stderr


def test_llava_llada_template_subprocess():
    script = r'''
import sys
sys.path.insert(0, r"%s")
import copy
from transformers import AutoTokenizer
from llava.conversation import conv_templates
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.mm_utils import tokenizer_image_token

tokenizer = AutoTokenizer.from_pretrained(r"%s", local_files_only=True)
conv = copy.deepcopy(conv_templates["llava_llada"])
conv.tokenizer = tokenizer
conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\nQUESTION_BODY")
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
assert "helpful language and vision assistant" in prompt
assert "<|start_header_id|>assistant<|end_header_id|>" in prompt
assert prompt.rstrip().endswith("<|start_header_id|>assistant<|end_header_id|>")
ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
assert (ids == IMAGE_TOKEN_INDEX).sum().item() == 1
assert prompt.count("<image>") == 1
print("PASS")
''' % (ROOT / "upstream/lladav/train", LLADAV)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env)
    assert result.returncode == 0 and "PASS" in result.stdout, result.stderr


def test_pope_reasoning_backend_pairing():
    from cdapc_lab.config import resolve_config
    for shard in range(3):
        for arm in ("native", "cdapc"):
            path = ROOT / f"configs/{arm}_shard{shard}.yaml"
            config = resolve_config(path)
            assert config["protocol"]["prompt_id"] == "pope_yesno_reasoning_v1"
            assert config["model"]["backend"] == "lladav"
    import yaml
    raw = yaml.safe_load((ROOT / "configs/native_shard0.yaml").read_text())
    raw["model"]["backend"] = "lavida"
    raw["model"]["dtype"] = "bfloat16"
    target = ROOT / "configs/experiments/.tmp_lavida_pope_reasoning.yaml"
    target.write_text(yaml.safe_dump(raw))
    try:
        with pytest.raises(ValueError):
            resolve_config(target)
    finally:
        target.unlink()


def test_lladav_think_suffix_subprocess():
    script = THINK_SCRIPT % (ROOT / "upstream/lladav/train", ROOT / "src", LLADAV,
                             ROOT / "data/pope_yesno", ROOT / "data/pope_yesno")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env)
    assert result.returncode == 0 and "PASS" in result.stdout, result.stderr


THINK_SCRIPT = r'''
import sys, json
from types import SimpleNamespace
sys.path.insert(0, r"%s")
sys.path.insert(0, r"%s")
import torch
import llava.mm_utils
from transformers import AutoTokenizer
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from cdapc_lab.models.lladav import LLaDAVAdapter
from cdapc_lab.types import Sample
from cdapc_lab.io.artifacts import read_json

llava.mm_utils.process_images = lambda images, proc, cfg: [torch.zeros(1, 3, 4, 4)]
captured = {}
def fake_prepare(self, input_ids, position_ids, attention_mask, past_key_values, labels,
                 images, modalities=["image"], image_sizes=None, is_llada=False):
    captured["prompt"] = input_ids
    expanded, lab = [], []
    for t in input_ids[0].tolist():
        if t == IMAGE_TOKEN_INDEX:
            expanded.extend([0, 0, 0]); lab.extend([IGNORE_INDEX] * 3)
        else:
            expanded.append(t); lab.append(t)
    e = torch.zeros(1, len(expanded), 8)
    l = torch.tensor(lab)[None] if labels is not None else None
    return None, None, None, None, e, l

adapter = LLaDAVAdapter.__new__(LLaDAVAdapter)
adapter.config = {"model": {"dtype": "float16"}, "protocol": {"prompt_id": None, "canvas_length": 8}}
adapter.device = torch.device("cpu")
adapter.dtype = torch.float32
adapter.tokenizer = AutoTokenizer.from_pretrained(r"%s", local_files_only=True)
adapter.image_processor = object()
adapter.mask_id = 126336
adapter.eot_id = 126348
records = read_json(r"%s/samples.json")
row = records[0]
image = r"%s/" + row["image_paths"][0]

def run(prompt_id):
    adapter.config["protocol"]["prompt_id"] = prompt_id
    captured.clear()
    sample = Sample(sample_id=row["sample_id"], question=row["question"], options=row["options"],
                    image_paths=(image,), task_id="pope", group_id=row["group_id"],
                    public_context=row.get("public_context"))
    adapter.model = SimpleNamespace(prepare_inputs_labels_for_multimodal=fake_prepare.__get__(adapter),
                                    config=SimpleNamespace())
    prepared = adapter.prepare(sample)
    prompt_ids = captured["prompt"][0].tolist()
    text = adapter.tokenizer.decode([t for t in prompt_ids if t >= 0])
    return prepared, text

prepared, text = run("pope_yesno_reasoning_v1")
assert prepared.preprocessing_version == "lladav_official_anyres_think_v1"
assert text.count(" /think") == 1
assert text.index(" /think") > text.index("No.")
prepared, text = run("pope_yesno_v1")
assert prepared.preprocessing_version == "lladav_official_anyres_v1"
assert " /think" not in text
print("PASS")
'''
