"""Observe the untouched official MMaDA generator, without changing its tensors."""
import types

import torch

from .decoding import record_step, validate_generation


class NativeObserver:
    def __init__(self, model, prompt_len, generation, trace):
        self.model, self.prompt_len, self.generation, self.trace = model, prompt_len, generation, trace

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, tokens, **kwargs):
        output = self.model(tokens, **kwargs)
        if not torch.isfinite(output.logits).all():
            raise ValueError("nonfinite native logits")
        self.tokens, self.candidates = tokens, output.logits.argmax(-1)
        return output

    def topk(self, confidence, k):
        result = torch.topk(confidence, k=k)
        g = self.generation
        per_block = g["steps"] // (g["max_new_tokens"] // g["block_length"])
        block, step = divmod(len(self.trace), per_block)
        end = self.prompt_len + (block + 1) * g["block_length"]
        active = torch.where(self.tokens[0, self.prompt_len:end] == self.model.config.mask_token_id)[0]
        active += self.prompt_len
        self.trace.append(record_step(
            self.tokens, self.candidates, confidence, result.indices, active, self.prompt_len,
            self.model.config.mask_token_id, block, step, end, {"method": "native"}))
        return result


class ObservedTorch:
    def __init__(self, observer):
        self.topk = observer.topk

    def __getattr__(self, name):
        return getattr(torch, name)


@torch.inference_mode()
def generate_native(model, prompt, visual, generation, trace):
    validate_generation(prompt, visual, generation, model.config.mask_token_id, allow_text_only=True)
    observer = NativeObserver(model, prompt.shape[1], generation, trace)
    original = model.mmu_generate.__wrapped__
    function = types.FunctionType(
        original.__code__, {**original.__globals__, "torch": ObservedTorch(observer)}, original.__name__,
        original.__defaults__)
    return function(observer, prompt, **generation, mask_id=model.config.mask_token_id)
