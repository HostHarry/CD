import sys
from pathlib import Path
import copy
import torch
from PIL import Image
from cdapc_lab.types import PreparedInput
from cdapc_lab.data.records import build_question, uses_model_reasoning_instruction
from .base import Adapter, sample_fingerprint, allowed_vocab

OFFICIAL_SUPPRESSED_TOKENS = ("<|endoftext|>", "<|startoftext|>", "<|start_header_id|>", "<|end_header_id|>")
OFFICIAL_SUPPRESSED_IDS = (126081, 126080, 126346, 126347)


class LLaDAVAdapter(Adapter):
    def __init__(self, config):
        m = config["model"]
        sys.path.insert(0, str(Path(m["upstream_root"]).resolve()))
        from transformers import AutoTokenizer
        from llava.model.language_model.llava_llada import LlavaLLaDAModelLM, LlavaLLaDAConfig
        from llava.model.language_model.modeling_llada import LLaDASdpaAttention
        self.config = config
        self.device = torch.device(config["runtime"]["device"])
        self.dtype = getattr(torch, m["dtype"])
        self.tokenizer = AutoTokenizer.from_pretrained(m["checkpoint"], local_files_only=True)
        model_config = LlavaLLaDAConfig.from_pretrained(m["checkpoint"], local_files_only=True)
        model_config.mm_vision_tower = str(Path(m["vision_checkpoint"]).resolve())
        model_config.use_cache = False
        model, loading_info = LlavaLLaDAModelLM.from_pretrained(
            m["checkpoint"], config=model_config, torch_dtype=self.dtype,
            attn_implementation="sdpa", low_cpu_mem_usage=True, local_files_only=True,
            output_loading_info=True)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            if loading_info.get(key):
                raise ValueError(f"Official checkpoint loading info nonempty: {key}={loading_info[key]}")
        self.model = model
        vision_tower = self.model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        self.model.to(device=self.device, dtype=self.dtype).eval()
        self.model.requires_grad_(False)
        if self.model.config._attn_implementation != "sdpa" or \
                any(not isinstance(layer.self_attn, LLaDASdpaAttention) for layer in self.denoiser_layers()):
            raise ValueError("Official SDPA decoder attention required")
        self.denoiser = self.model.get_model()
        self.image_processor = vision_tower.image_processor
        self.mask_id = self.tokenizer.convert_tokens_to_ids("<|mdm_mask|>")
        if self.mask_id != 126336:
            raise ValueError("Official MASK token id mismatch")
        self.eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if self.eot_id != 126348:
            raise ValueError("Official <|eot_id|> token id mismatch")
        suppressed = tuple(self.tokenizer.convert_tokens_to_ids(t) for t in OFFICIAL_SUPPRESSED_TOKENS)
        if suppressed != OFFICIAL_SUPPRESSED_IDS:
            raise ValueError(f"Official suppressed token ids mismatch: {suppressed}")

    def denoiser_layers(self):
        return self.model.get_model().layers

    @torch.inference_mode()
    def prepare(self, sample):
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, IGNORE_INDEX
        from llava.conversation import conv_templates
        if len(sample.image_paths) != 1:
            raise ValueError("LLaDA-V protocol requires one image")
        conv = copy.deepcopy(conv_templates["llava_llada"])
        conv.tokenizer = self.tokenizer
        prompt_id = self.config["protocol"]["prompt_id"]
        reasoning = uses_model_reasoning_instruction(sample, prompt_id)
        content = DEFAULT_IMAGE_TOKEN + "\n" + build_question(sample, prompt_id)
        if reasoning:
            content += " /think"
        conv.append_message(conv.roles[0], content)
        conv.append_message(conv.roles[1], None)
        prompt = tokenizer_image_token(conv.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")[None].to(self.device)
        with Image.open(sample.image_paths[0]) as source:
            image = source.convert("RGB")
        pixels = process_images([image], self.image_processor, self.model.config)
        pixels = [p.to(device=self.device, dtype=self.dtype) for p in pixels]
        inputs = self.model.prepare_inputs_labels_for_multimodal(
            prompt, None, None, None, prompt.clone(), pixels, ["image"], image_sizes=[image.size])
        embeddings, labels = inputs[4], inputs[5]
        visual = labels[0] == IGNORE_INDEX
        reference = self.model.prepare_inputs_labels_for_multimodal(
            prompt, None, None, None, None, pixels, ["image"], image_sizes=[image.size])
        text_ids = prompt[0][prompt[0] != IMAGE_TOKEN_INDEX]
        if embeddings.shape[1] != labels.shape[1] or not visual.any() or \
                not torch.equal(labels[0][~visual], text_ids) or not torch.equal(embeddings, reference[4]):
            raise ValueError("Official expanded visual-position map missing")
        return PreparedInput(sample.sample_id, labels, visual,
                             allowed_vocab(self.tokenizer, self.mask_id, self.device,
                                           OFFICIAL_SUPPRESSED_IDS, eos=self.eot_id),
                             self.mask_id, (self.eot_id,), self.tokenizer.pad_token_id,
                             self.config["protocol"]["canvas_length"], sample_fingerprint(sample),
                             {"prompt_embeddings": embeddings, "original_prompt_ids": prompt,
                              "image_shapes": [list(p.shape) for p in pixels], "image_sizes": [image.size],
                              "pixels": pixels},
                             "lladav_official_anyres_think_v1" if reasoning else "lladav_official_anyres_v1")

    @torch.inference_mode()
    def forward_logits(self, prepared, state, bias=None, batch=1):
        response = self.denoiser.embed_tokens(state.response_ids[None])
        embeddings = torch.cat((prepared.model_inputs["prompt_embeddings"], response), dim=1).expand(batch, -1, -1)
        mask = None if bias is None else torch.isfinite(bias).to(self.dtype)
        with torch.autocast("cuda", dtype=torch.float16):
            hidden = self.denoiser(inputs_embeds=embeddings, attention_mask=mask, use_cache=False)[0]
        return self.model.lm_head(hidden)[:, -prepared.canvas_length:].float()

    @torch.inference_mode()
    def direct_native(self, prepared, config):
        output = self.model.generate_with_embeds(
            inputs_embeds=prepared.model_inputs["prompt_embeddings"],
            steps=config["steps"], gen_length=prepared.canvas_length,
            block_length=config["block_length"], temperature=config["temperature"],
            cfg_scale=config["cfg_scale"], remasking=config["remasking"],
            mask_id=self.mask_id, tokenizer=self.tokenizer, stopping_criteria=["<|eot_id|>"])
        tokens = output[0]
        if tokens.numel() > prepared.canvas_length:
            raise ValueError("Official generation exceeded canvas")
        canvas = torch.full((prepared.canvas_length,), self.mask_id, dtype=torch.long, device=tokens.device)
        canvas[:tokens.numel()] = tokens
        if tokens.numel() < prepared.canvas_length:
            canvas[tokens.numel()] = self.eot_id
        return canvas
