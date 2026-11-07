from __future__ import annotations
from contextlib import contextmanager
from dataclasses import asdict
import torch
from cdapc_lab.types import ComputeCost, ForwardResult, PairedForwardResult, RawGeneration
from cdapc_lab.io.artifacts import digest, file_hash


def sample_fingerprint(sample):
    record = asdict(sample)
    record["image_paths"] = [file_hash(path) for path in sample.image_paths]
    return digest(record)


def visual_ablation_bias(visual_keys, response_length, dtype=torch.float32):
    if visual_keys.ndim != 1 or visual_keys.dtype != torch.bool or not visual_keys.any():
        raise ValueError("Expected nonempty actual visual key positions")
    visual = torch.cat((visual_keys, torch.zeros(response_length, dtype=torch.bool, device=visual_keys.device)))
    blocked = (~visual[:, None]) & visual[None, :]
    bias = torch.zeros((1, 1, len(visual), len(visual)), dtype=dtype, device=visual.device)
    return bias.masked_fill(blocked, -torch.inf)


def allowed_vocab(tokenizer, mask_id, device, extra_excluded=()):
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Tokenizer must declare EOS")
    excluded = (set(tokenizer.all_special_ids) | set(extra_excluded) | {mask_id}) - {eos}
    return torch.tensor([i for i in range(len(tokenizer)) if i not in excluded], device=device, dtype=torch.long)


class Adapter:
    @contextmanager
    def count_forwards(self):
        cost = ComputeCost()
        def count(module, args, kwargs):
            if kwargs.get("use_cache", False) or kwargs.get("past_key_values") is not None:
                raise ValueError("Unexpected cache in no-cache protocol")
            tensor = args[0] if args and isinstance(args[0], torch.Tensor) else None
            if tensor is None:
                tensor = next((kwargs[k] for k in ("input_ids", "input_embeddings", "inputs_embeds") if isinstance(kwargs.get(k), torch.Tensor)), None)
            if tensor is None:
                raise RuntimeError("Cannot measure actual model input")
            batch, length = tensor.shape[:2]
            cost.model_forward_calls += 1
            cost.denoiser_branch_evaluations += batch
            cost.full_sequence_branch_evaluations += batch
            cost.logical_query_positions += batch * length
        handle = self.denoiser.register_forward_pre_hook(count, with_kwargs=True)
        try:
            yield cost
        finally:
            handle.remove()

    @torch.inference_mode()
    def forward_pair(self, prepared, state, execution="sequential", cache="disabled"):
        if cache != "disabled":
            raise ValueError("Only complete no-cache forwards supported")
        bias = visual_ablation_bias(prepared.visual_keys, prepared.canvas_length)
        with self.count_forwards() as cost:
            if execution == "sequential":
                visual = self.forward_logits(prepared, state, None)
                ablated = self.forward_logits(prepared, state, bias)
            elif execution == "paired_batch":
                logits = self.forward_logits(prepared, state, torch.cat((torch.zeros_like(bias), bias)), batch=2)
                visual, ablated = logits[:1], logits[1:]
            else:
                raise ValueError("Unknown execution mode")
        return PairedForwardResult(ForwardResult(visual, ComputeCost()), ForwardResult(ablated, ComputeCost()), cost)

    @torch.inference_mode()
    def native(self, prepared, config):
        with self.count_forwards() as cost:
            ids = self.direct_native(prepared, config)
        reason = "canvas_filled" if not (ids == prepared.mask_id).any() else "budget_exhausted"
        return RawGeneration(ids, reason, cost.model_forward_calls, cost)


def load_adapter(config):
    if config["model"]["backend"] == "mmada":
        from .mmada import MMaDAAdapter
        return MMaDAAdapter(config)
    if config["model"]["backend"] == "lavida":
        from .lavida import LaViDaAdapter
        return LaViDaAdapter(config)
    raise ValueError("Unknown backend")
