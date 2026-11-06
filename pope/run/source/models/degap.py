from dataclasses import dataclass
import math
import time

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DEGAPConfig:
    alpha: float = 1.0
    mode: str = "degap_token"
    fixed_boundary: int = 16
    lens: str = "native"
    query_scope: str = "all"
    chunk_size: int = 16
    record_entropy: bool = False

    def __post_init__(self):
        if not math.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        if self.mode not in {"degap_token", "degap_step_mean", "fixed_b"}:
            raise ValueError("unsupported DEGAP mode")
        if self.lens not in {"native", "raw_linear"}:
            raise ValueError("unsupported logit lens")
        if self.query_scope not in {"all", "nonvisual"}:
            raise ValueError("unsupported query scope")
        if self.chunk_size < 1 or self.fixed_boundary < 0:
            raise ValueError("invalid chunk size or boundary")


def select_boundaries(entropy):
    if entropy.ndim != 2 or entropy.shape[1] < 1:
        raise ValueError("entropy must have shape [positions, layers]")
    if not torch.isfinite(entropy).all():
        raise ValueError("nonfinite entropy")
    device = entropy.device
    y = entropy.detach().to(device="cpu", dtype=torch.float64)
    result = torch.full((y.shape[0],), -1, dtype=torch.long)
    minima = y.argmin(-1)
    for m in minima.unique().tolist():
        if m < 3:
            continue
        rows = torch.where(minima == m)[0]
        values = y[rows, :m + 1]
        values = values - values[:, :1]
        x = torch.arange(m + 1, dtype=torch.float64)
        k = torch.arange(1, m - 1)
        sy = values.cumsum(-1)
        syy = values.square().cumsum(-1)
        sxy = (values * x).cumsum(-1)
        sx, sxx = x.cumsum(0), x.square().cumsum(0)

        def mse(n, sum_x, sum_xx, sum_y, sum_yy, sum_xy):
            centered_xx = sum_xx - sum_x.square() / n
            centered_xy = sum_xy - sum_x * sum_y / n
            sse = sum_yy - sum_y.square() / n - centered_xy.square() / centered_xx
            return sse.clamp_min(0) / n

        errors = mse(k + 1, sx[k], sxx[k], sy[:, k], syy[:, k], sxy[:, k])
        errors += mse(
            m - k, sx[m] - sx[k], sxx[m] - sxx[k],
            sy[:, m:m + 1] - sy[:, k], syy[:, m:m + 1] - syy[:, k],
            sxy[:, m:m + 1] - sxy[:, k],
        )
        if not torch.isfinite(errors).all():
            raise ValueError("nonfinite split objective")
        tolerance = 64 * torch.finfo(torch.float64).eps * values.square().mean(-1, keepdim=True).clamp_min(1)
        best = errors.min(-1, keepdim=True).values
        result[rows] = (errors <= best + tolerance).to(torch.int64).argmax(-1) + 1
    return result.to(device)


def visual_bias(length, visual_positions, query_scope, device, base_bias=None):
    indices = torch.as_tensor(visual_positions, dtype=torch.long, device=device)
    if indices.ndim != 1 or indices.numel() == 0:
        raise ValueError("visual positions must be a nonempty vector")
    if indices.min() < 0 or indices.max() >= length or indices.unique().numel() != indices.numel():
        raise ValueError("invalid or duplicate visual positions")
    visual = torch.zeros(length, dtype=torch.bool, device=device)
    visual[indices] = True
    if visual.all():
        raise ValueError("attention must retain at least one nonvisual key")
    if query_scope not in {"all", "nonvisual"}:
        raise ValueError("unsupported query scope")
    forbidden = visual[None, :].expand(length, -1)
    if query_scope == "nonvisual":
        forbidden = forbidden & ~visual[:, None]
    bias = torch.zeros((1, 1, length, length), device=device, dtype=torch.float32)
    if base_bias is not None:
        if base_bias.dtype == torch.bool:
            bias.masked_fill_(~base_bias, -torch.inf)
        else:
            bias += base_bias
    bias.masked_fill_(forbidden[None, None], -torch.inf)
    if torch.isneginf(bias).all(-1).any() or torch.isnan(bias).any() or torch.isposinf(bias).any():
        raise ValueError("invalid attention bias or fully blocked query")
    return bias


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


class DEGAP:
    def __init__(self, model, config):
        self.wrapper = model
        self.model = model.model
        self.config = config if isinstance(config, DEGAPConfig) else DEGAPConfig(**config)
        c = self.model.config
        if model.training or not c.rope or c.alibi or c.block_group_size != 1:
            raise ValueError("DEGAP requires eval mode, RoPE, non-ALiBi, block_group_size=1")
        self.blocks = self.model.transformer.blocks
        if self.config.mode == "fixed_b" and self.config.fixed_boundary >= len(self.blocks):
            raise ValueError("fixed boundary outside decoder")
        self.entropy_seconds = 0.0

    def readout(self, hidden, lens="native"):
        t, c = self.model.transformer, self.model.config
        if lens == "native":
            hidden = t.ln_f(hidden)
        logits = F.linear(hidden, t.wte.weight) if c.weight_tying else t.ff_out(hidden)
        return logits / math.sqrt(c.d_model) if c.scale_logits else logits

    @torch.no_grad()
    def observe(self, tokens, positions, attention_bias=None, collect_entropy=True):
        cache, entropies, handles, timings = {}, [], [], []
        self.entropy_seconds = 0.0

        def hook(index, block, inputs, output):
            cache[index] = inputs[0]
            if not collect_entropy:
                return
            if tokens.is_cuda:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
            else:
                start = time.perf_counter()
            values = []
            for chunk in positions.split(self.config.chunk_size):
                logits = self.readout(output[0][:, chunk], self.config.lens)[0].float()
                lp = logits.log_softmax(-1)
                values.append(-(lp.exp() * lp).sum(-1))
            entropies.append(torch.cat(values))
            if tokens.is_cuda:
                end.record()
                timings.append((start, end))
            else:
                self.entropy_seconds += time.perf_counter() - start

        try:
            for index, block in enumerate(self.blocks):
                handles.append(block.register_forward_hook(
                    lambda module, inputs, output, index=index: hook(index, module, inputs, output)))
            normal = self.wrapper(tokens, attention_bias=attention_bias, use_cache=False).logits
        finally:
            for handle in handles:
                handle.remove()
        if timings:
            synchronize(tokens.device)
            self.entropy_seconds = sum(start.elapsed_time(end) for start, end in timings) / 1000
        entropy = torch.stack(entropies, dim=-1) if entropies else None
        return normal, entropy, cache

    @torch.no_grad()
    def reference(self, hidden, positions, boundary, bias):
        h = hidden
        bias = bias.to(hidden.dtype) if bias is not None else None
        for block in self.blocks[boundary:]:
            h, _ = block(h, attention_bias=bias, use_cache=False)
        return torch.cat([
            self.readout(h[:, chunk])[0] for chunk in positions.split(self.config.chunk_size)
        ], dim=0)

    @torch.no_grad()
    def __call__(self, tokens, positions, visual_positions, attention_bias=None):
        if tokens.shape[0] != 1 or positions.numel() == 0:
            raise ValueError("DEGAP requires one sample and nonempty active positions")
        synchronize(tokens.device)
        started = time.perf_counter()
        normal, entropy, cache = self.observe(
            tokens, positions, attention_bias, self.config.mode != "fixed_b")
        if not torch.isfinite(normal[0, positions]).all():
            raise ValueError("nonfinite normal logits")
        if self.config.mode == "fixed_b":
            boundaries = torch.full_like(positions, self.config.fixed_boundary)
        elif self.config.mode == "degap_step_mean":
            boundaries = select_boundaries(entropy.mean(0, keepdim=True)).expand_as(positions)
        else:
            boundaries = select_boundaries(entropy)
        bias = visual_bias(tokens.shape[1], visual_positions, self.config.query_scope,
                           tokens.device, attention_bias)
        logits = normal.float()
        normal_top = normal[0, positions].argmax(-1)
        unique = boundaries[boundaries >= 0].unique().tolist()
        jsd = torch.zeros(positions.numel(), device=tokens.device)
        for b in unique:
            rows = torch.where(boundaries == b)[0]
            selected = positions[rows]
            reference = self.reference(cache[b], selected, b, bias).float()
            original = normal[0, selected].float()
            if not torch.isfinite(reference).all():
                raise ValueError("nonfinite reference logits")
            combined = (1 + self.config.alpha) * original - self.config.alpha * reference
            p, q = original.log_softmax(-1), reference.log_softmax(-1)
            mixture = torch.logaddexp(p, q) - math.log(2)
            jsd[rows] = ((p.exp() * (p - mixture)).sum(-1) +
                         (q.exp() * (q - mixture)).sum(-1)) / 2
            logits[0, selected] = combined
        if not torch.isfinite(logits[0, positions]).all():
            raise ValueError("nonfinite contrast logits")
        synchronize(tokens.device)
        trace = {
            "active_positions": positions.tolist(),
            "boundaries": boundaries.tolist(),
            "unique_boundaries": unique,
            "no_valid_split": int((boundaries < 0).sum()),
            "suffix_layers": sum(len(self.blocks) - b for b in unique),
            "normal_top1": normal_top.tolist(),
            "contrast_top1": logits[0, positions].argmax(-1).tolist(),
            "reference_jsd": jsd.tolist(),
            "entropy_seconds": self.entropy_seconds,
            "forward_seconds": time.perf_counter() - started,
        }
        if self.config.record_entropy and entropy is not None:
            trace["entropy"] = entropy.tolist()
        return logits, trace
