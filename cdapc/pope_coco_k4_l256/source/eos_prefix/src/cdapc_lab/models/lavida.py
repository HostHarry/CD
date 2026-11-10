import sys
from pathlib import Path
import copy
import torch
from PIL import Image
from cdapc_lab.types import PreparedInput
from cdapc_lab.data.records import build_question
from .base import Adapter, sample_fingerprint, allowed_vocab


class LaViDaAdapter(Adapter):
    def __init__(self, config):
        m = config["model"]
        sys.path.insert(0, str(Path(m["upstream_root"]).resolve()))
        from transformers import AutoTokenizer
        from llava.model.language_model.llava_llada import LlavaLladaForMaskedDiffusion, LlavaLladaConfig
        self.config = config
        self.device = torch.device(config["runtime"]["device"])
        self.dtype = getattr(torch, m["dtype"])
        self.tokenizer = AutoTokenizer.from_pretrained(m["checkpoint"], local_files_only=True)
        model_config = LlavaLladaConfig.from_pretrained(m["checkpoint"], local_files_only=True)
        model_config.mm_vision_tower = str(Path(m["vision_checkpoint"]).resolve())
        model_config.use_cache = False
        self.model = LlavaLladaForMaskedDiffusion.from_pretrained(m["checkpoint"], config=model_config,
                         torch_dtype=self.dtype, local_files_only=True).to(self.device).eval()
        self.model.tie_weights()
        self.model.requires_grad_(False)
        self.denoiser = self.model.get_model()
        self.denoiser.set_activation_checkpointing(None)
        self.image_processor = self.model.get_vision_tower().image_processor
        self.mask_id = self.tokenizer.convert_tokens_to_ids("<|mdm_mask|>")
        if self.mask_id is None or self.mask_id == self.tokenizer.unk_token_id:
            raise ValueError("Missing official MASK token")

    @torch.inference_mode()
    def prepare(self, sample):
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates
        if len(sample.image_paths) != 1:
            raise ValueError("LaViDa protocol requires one image")
        conv = copy.deepcopy(conv_templates["llada"])
        conv.tokenizer = self.tokenizer
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + build_question(sample))
        conv.append_message(conv.roles[1], None)
        prompt = tokenizer_image_token(conv.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")[None].to(self.device)
        with Image.open(sample.image_paths[0]) as source:
            image = source.convert("RGB")
        pixels = process_images([image], self.image_processor, self.model.config)
        pixels = [p.to(device=self.device, dtype=self.dtype) for p in pixels]
        prepared = self.model.prepare_inputs_labels_for_multimodal(prompt, None, None, None, None, pixels,
                             ["image"], image_sizes=[image.size], return_inputs=True)
        embeddings, expanded_ids = prepared[4], prepared[6]
        visual = expanded_ids[0] == IMAGE_TOKEN_INDEX
        if not visual.any() or expanded_ids.shape[1] != embeddings.shape[1]:
            raise ValueError("Official expanded visual-position map missing")
        return PreparedInput(sample.sample_id, expanded_ids, visual, allowed_vocab(self.tokenizer, self.mask_id, self.device),
                             self.mask_id, (self.tokenizer.eos_token_id,), self.tokenizer.pad_token_id,
                             self.config["protocol"]["canvas_length"], sample_fingerprint(sample),
                             {"prompt_embeddings": embeddings, "original_prompt_ids": prompt,
                              "image_shapes": [list(p.shape) for p in pixels], "image_sizes": [image.size], "pixels": pixels},
                             "lavida_official_anyres_v1")

    @torch.inference_mode()
    def forward_logits(self, prepared, state, bias=None, batch=1):
        response = self.denoiser.transformer.wte(state.response_ids[None])
        embeddings = torch.cat((prepared.model_inputs["prompt_embeddings"], response), dim=1).expand(batch, -1, -1)
        return self.denoiser(None, input_embeddings=embeddings, attention_bias=bias,
                             use_cache=False).logits[:, -prepared.canvas_length:]

    @torch.inference_mode()
    def direct_native(self, prepared, config):
        from llava.model.language_model.llada.generate import generate
        blocks = prepared.canvas_length // config["block_length"]
        # Official code overwrites steps; step_per_block is its consumed control.
        if config["steps"] // blocks > config["block_length"]:
            raise ValueError("Official LaViDa clamps steps above block length; refusing ignored budget")
        output = generate(self.denoiser, inputs_embeds=prepared.model_inputs["prompt_embeddings"],
                          max_new_tokens=prepared.canvas_length, block_length=config["block_length"],
                          step_per_block=config["steps"] // blocks, temperature=config["temperature"],
                          remasking=config["remasking"], cfg_scale=config["cfg_scale"],
                          mask_id=prepared.mask_id, prefix_lm=False, verbose=False)
        return output[0, prepared.prompt_ids.shape[1]:]
