"""VIGOR author-loop bridge and observational trace collection."""
import types

import torch

from baseline_sources import upstream_functions
from common.decoding import checked_logits, record_step, validate_generation


class _VigorObserver:
    """Observe author topk calls; all tensors returned to the author code are untouched."""
    def __init__(self, model, prompt_len, generation, trace):
        self.model, self.prompt_len, self.generation, self.trace = model, prompt_len, generation, trace
        self.calls = 0

    def __call__(self, tokens, infer=True):
        logits = checked_logits(self.model, tokens)
        if self.calls % 2 == 0:
            self.tokens = tokens
            self.candidates = logits.argmax(-1)
        self.calls += 1
        return types.SimpleNamespace(logits=logits)

    def topk(self, confidence, k):
        result = torch.topk(confidence, k=k)
        g = self.generation
        steps_per_block = g["steps"] // (g["max_new_tokens"] // g["block_length"])
        block, step = divmod(len(self.trace), steps_per_block)
        block_end = self.prompt_len + (block + 1) * g["block_length"]
        active = torch.where((self.tokens[0, self.prompt_len:block_end] == self.model.config.mask_token_id))[0]
        active += self.prompt_len
        if not torch.isfinite(confidence[active]).all():
            raise ValueError("nonfinite VIGOR ranking score")
        self.trace.append(record_step(
            self.tokens, self.candidates, confidence, result.indices, active, self.prompt_len,
            self.model.config.mask_token_id, block, step, block_end,
            {"method": "vigor", "normal_top1": self.candidates[0, active].tolist(),
             "vigor_scores": confidence[active].tolist(),
             "confidence_semantics": "visual-gain ordering score, not probability"}))
        return result


class _ObservedTorch:
    def __init__(self, observer):
        self.topk = observer.topk

    def __getattr__(self, name):
        return getattr(torch, name)


@torch.inference_mode()
def generate_vigor(model, prompt, visual, generation, trace):
    """Execute the complete author Lumina token-ID loop with an MMaDA call bridge."""
    mask_id = model.config.mask_token_id
    validate_generation(prompt, visual, generation, mask_id)
    x = torch.full((1, prompt.shape[1] + generation["max_new_tokens"]),
                   mask_id, device=prompt.device, dtype=torch.long)
    x[:, :prompt.shape[1]] = prompt
    observer = _VigorObserver(model, prompt.shape[1], generation, trace)
    # Unwrap only @torch.no_grad (we are already in inference mode).
    original = upstream_functions()["vigor"].__wrapped__
    function = types.FunctionType(
        original.__code__, {**original.__globals__, "torch": _ObservedTorch(observer)}, original.__name__)
    return function(
        observer, x, steps=generation["steps"], gen_length=generation["max_new_tokens"],
        block_length=generation["block_length"], temperature=0.0, mask_id=mask_id,
        code_start=prompt.shape[1], probe_interval=1, img_start=visual[0], img_end=visual[-1] + 1)
