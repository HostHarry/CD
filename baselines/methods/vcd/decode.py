"""VCD image distortion and contrastive decoding."""
import torch
import torch.nn.functional as F
from PIL import Image

from baseline_sources import upstream_functions
from common.decoding import checked_logits, record_step, validate_generation
from models.modeling_mmada import get_num_transfer_tokens
from training.utils import image_transform, image_transform_squash


@torch.inference_mode()
def generate_vcd(model, prompt, distorted_prompt, visual, generation, options, trace):
    """Author VCD logits/APC, with MMaDA's original transfer-count/ranking schedule."""
    mask_id = model.config.mask_token_id
    validate_generation(prompt, visual, generation, mask_id)
    if distorted_prompt.shape != prompt.shape:
        raise ValueError("distorted prompt shape differs")
    nonvisual = torch.ones(prompt.shape[1], dtype=torch.bool, device=prompt.device)
    nonvisual[visual] = False
    if not torch.equal(prompt[:, nonvisual], distorted_prompt[:, nonvisual]):
        raise ValueError("VCD distorted prompt changed nonvisual tokens")
    length, steps, block_length = (generation[k] for k in ("max_new_tokens", "steps", "block_length"))
    prompt_len = prompt.shape[1]
    x = torch.full((1, prompt_len + length), mask_id, device=prompt.device, dtype=torch.long)
    x[:, :prompt_len] = prompt
    steps_per_block = steps // (length // block_length)
    combine = upstream_functions()["vcd"]
    for block in range(length // block_length):
        block_start, block_end = prompt_len + block * block_length, prompt_len + (block + 1) * block_length
        counts = get_num_transfer_tokens(x[:, block_start:block_end] == mask_id, steps_per_block)
        for step in range(steps_per_block):
            masked = x == mask_id
            active = torch.where(masked[0, prompt_len:block_end])[0] + prompt_len
            normal = checked_logits(model, x)
            reference_input = x.clone()
            reference_input[:, visual] = distorted_prompt[:, visual]
            reference = checked_logits(model, reference_input)
            logits = combine(normal, reference, options.cd_alpha, options.cd_beta)
            if torch.isnan(logits).any() or torch.isposinf(logits).any() or torch.isneginf(logits).all(-1).any():
                raise ValueError("invalid VCD logits (APC negative infinity is expected)")
            candidates = logits.argmax(-1)
            probability = F.softmax(logits.to(torch.float64), dim=-1)
            confidence = probability.gather(-1, candidates.unsqueeze(-1)).squeeze(-1)
            confidence[:, block_end:] = -torch.inf
            confidence = torch.where(masked, confidence, -torch.inf)
            _, selected = torch.topk(confidence[0], k=int(counts[0, step]))
            trace.append(record_step(
                x, candidates, confidence[0], selected, active, prompt_len, mask_id, block, step, block_end,
                {"method": "vcd", "normal_top1": normal[0, active].argmax(-1).tolist(),
                 "contrast_top1": candidates[0, active].tolist(),
                 "apc_excluded_tokens": torch.isneginf(logits[0, active]).sum(-1).tolist()}))
            x[0, selected] = candidates[0, selected]
    return x


@torch.inference_mode()
def distorted_prompt(self, sample, prompt, visual, options):
    # Official VCD adds noise on the preprocessed CPU image before model transfer.
    with Image.open(sample["image"]) as image:
        transform = image_transform if self.resize == "crop" else image_transform_squash
        pixels = transform(image.convert("RGB"), resolution=self.resolution)
    noisy = upstream_functions()["noise"](pixels, options.noise_step)
    if not torch.isfinite(noisy).all():
        raise ValueError("nonfinite distorted pixels")
    codes = self.vq.get_code(noisy.unsqueeze(0).to(self.device)) + len(self.tokenizer)
    if codes.shape != (1, len(visual)):
        raise ValueError("distorted image code shape differs")
    distorted = prompt.clone()
    distorted[:, visual] = codes
    return distorted
