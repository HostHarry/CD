import sys
from pathlib import Path
import torch
from PIL import Image
from cdapc_lab.types import PreparedInput
from cdapc_lab.data.records import build_question, uses_model_reasoning_instruction
from .base import Adapter, sample_fingerprint, allowed_vocab

OFFICIAL_THINKING_INSTRUCTION = "You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here"


class MMaDAAdapter(Adapter):
    def __init__(self, config):
        m = config["model"]
        sys.path.insert(0, str(Path(m["upstream_root"]).resolve()))
        from models import MMadaModelLM, MAGVITv2
        from training.prompting_utils import UniversalPrompting
        from transformers import AutoTokenizer
        self.config = config
        self.device = torch.device(config["runtime"]["device"])
        self.dtype = getattr(torch, m["dtype"])
        self.tokenizer = AutoTokenizer.from_pretrained(m["checkpoint"], local_files_only=True)
        self.prompting = UniversalPrompting(self.tokenizer, max_text_len=config["protocol"]["canvas_length"], use_reserved_token=True)
        self.model = MMadaModelLM.from_pretrained(m["checkpoint"], torch_dtype=self.dtype, local_files_only=True).to(self.device).eval()
        self.model.requires_grad_(False)
        self.vision = MAGVITv2.from_pretrained(m["vision_checkpoint"], local_files_only=True).to(self.device).eval()
        self.vision.requires_grad_(False)
        self.denoiser = self.model
        self.mask_id = self.tokenizer.convert_tokens_to_ids("<|mdm_mask|>")
        if self.mask_id is None or self.mask_id == self.tokenizer.unk_token_id:
            raise ValueError("Missing official MASK token")

    @torch.inference_mode()
    def prepare(self, sample):
        from training.utils import image_transform
        if len(sample.image_paths) != 1:
            raise ValueError("MMaDA protocol requires one image")
        with Image.open(sample.image_paths[0]) as image:
            pixels = image_transform(image.convert("RGB"), resolution=512).unsqueeze(0).to(self.device)
        image_ids = self.vision.get_code(pixels) + len(self.tokenizer)
        if image_ids.shape != (1, 1024):
            raise ValueError("MMaDA inference_mmu protocol expects 1024 MAGVITv2 codes")
        prompt_id = self.config["protocol"]["prompt_id"]
        reasoning = uses_model_reasoning_instruction(sample, prompt_id)
        content = (OFFICIAL_THINKING_INSTRUCTION + "\n" if reasoning else "") + build_question(sample, prompt_id)
        text_ids = self.tokenizer.apply_chat_template([{"role": "user", "content": content}],
                                                      tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.device)
        token = lambda name: self.prompting.sptids_dict[name].reshape(1, 1).to(self.device)
        parts = [token("<|mmu|>"), token("<|soi|>"), image_ids, token("<|eoi|>"), text_ids]
        prompt = torch.cat(parts, dim=1).long()
        visual = torch.cat([torch.full((part.shape[1],), i == 2, device=self.device, dtype=torch.bool) for i, part in enumerate(parts)])
        excluded = [int(v.item()) for k, v in self.prompting.sptids_dict.items() if k not in ("<|eot|>",)]
        return PreparedInput(sample.sample_id, prompt, visual, allowed_vocab(self.tokenizer, self.mask_id, self.device, excluded),
                             self.mask_id, (self.tokenizer.eos_token_id,), self.tokenizer.pad_token_id,
                             self.config["protocol"]["canvas_length"], sample_fingerprint(sample),
                             {"image_shape": list(pixels.shape)},
                             "mmada_inference_mmu_512_thinking_v1" if reasoning else "mmada_inference_mmu_512_v1")

    @torch.inference_mode()
    def forward_logits(self, prepared, state, bias=None, batch=1):
        ids = torch.cat((prepared.prompt_ids, state.response_ids[None]), dim=1).expand(batch, -1)
        return self.model(ids, attention_bias=bias, use_cache=False).logits[:, -prepared.canvas_length:]

    @torch.inference_mode()
    def direct_native(self, prepared, config):
        output = self.model.mmu_generate(prepared.prompt_ids, max_new_tokens=prepared.canvas_length,
                                         steps=config["steps"], block_length=config["block_length"],
                                         temperature=config["temperature"], remasking=config["remasking"],
                                         cfg_scale=config["cfg_scale"], mask_id=prepared.mask_id)
        return output[0, prepared.prompt_ids.shape[1]:]
