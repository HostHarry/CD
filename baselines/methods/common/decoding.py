"""Common validation and trace fields, with no intervention algorithm."""
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class BaselineConfig:
    mode: str
    cd_alpha: float = 1.0
    cd_beta: float = 0.1
    noise_step: int = 500
    # Retained for compatibility with the verified imported predictions.
    chunk_size: int = 16

    def __post_init__(self):
        if self.mode not in {"vcd", "vigor"}:
            raise ValueError("baseline must be vcd or vigor")
        if not math.isfinite(self.cd_alpha) or self.cd_alpha < 0:
            raise ValueError("VCD alpha must be finite and nonnegative")
        if not math.isfinite(self.cd_beta) or not 0 < self.cd_beta <= 1:
            raise ValueError("VCD beta must be in (0,1]")
        if not isinstance(self.noise_step, int) or not 0 <= self.noise_step < 1000:
            raise ValueError("VCD noise step must be an integer in [0,999]")


def validate_generation(prompt, visual, generation, mask_id, *, allow_text_only=False):
    if prompt.ndim != 2 or prompt.shape[0] != 1 or (prompt == mask_id).any():
        raise ValueError("baselines require batch=1 and no MASK inside the prompt")
    if (generation["temperature"] != 0 or generation["cfg_scale"] != 0
            or generation["remasking"] != "low_confidence"):
        raise ValueError("this matched-protocol bridge requires temperature=0, CFG=0, low_confidence")
    length, steps, block = (generation[k] for k in ("max_new_tokens", "steps", "block_length"))
    if (min(length, steps, block) < 1 or length % block or steps % (length // block)
            or steps > length):
        raise ValueError("invalid fixed generation schedule")
    visual = list(visual)
    if allow_text_only and not visual:
        return
    if (not visual or visual != list(range(visual[0], visual[-1] + 1))
            or visual[0] < 0 or visual[-1] >= prompt.shape[1]):
        raise ValueError("expected one contiguous image-code span inside prompt")


def checked_logits(model, tokens):
    logits = model(tokens, use_cache=False).logits
    if not torch.isfinite(logits).all():
        raise ValueError("nonfinite model logits")
    return logits


def record_step(x, candidates, confidence, selected, active, prompt_len, mask_id,
                block, step, block_end, extra):
    selected = selected.sort().values
    remaining = int((x[:, prompt_len:] == mask_id).sum())
    remaining -= int((candidates[0, selected] != mask_id).sum())
    return {
        **extra, "block": block, "step": step, "block_end": block_end,
        "active_positions": active.tolist(),
        "committed_positions": [selected.tolist()],
        "committed_tokens": [candidates[0, selected].tolist()],
        "committed_confidence": [confidence[selected].tolist()],
        "remaining_masks": remaining,
    }
