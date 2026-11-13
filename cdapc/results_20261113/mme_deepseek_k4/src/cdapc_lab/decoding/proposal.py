import math
import torch
from cdapc_lab.types import ProposalStats


def make_proposals(visual_logits, ablated_logits, allowed_ids, config):
    if visual_logits.shape != ablated_logits.shape or visual_logits.ndim != 3 or visual_logits.shape[0] != 1:
        raise ValueError("Expected matching [1, L, V] logits")
    if allowed_ids.ndim != 1 or allowed_ids.dtype != torch.long or allowed_ids.numel() == 0:
        raise ValueError("Allowed vocabulary must be nonempty int64 IDs")
    allowed_ids = allowed_ids.sort().values
    if torch.any(allowed_ids[1:] == allowed_ids[:-1]) or allowed_ids[0] < 0 or allowed_ids[-1] >= visual_logits.shape[-1]:
        raise ValueError("Invalid/duplicate allowed vocabulary IDs")
    visual = visual_logits[0, :, allowed_ids].float()
    ablated = ablated_logits[0, :, allowed_ids].float()
    if not torch.isfinite(visual).all() or not torch.isfinite(ablated).all():
        raise ValueError("Legal vocabulary logits must be finite")
    lp, la = visual.log_softmax(-1), ablated.log_softmax(-1)
    gain = lp - la
    base = lp.argmax(-1)
    pool = lp >= lp.max(-1, keepdim=True).values + math.log(config.beta)
    mass = (lp.exp() * pool).sum(-1)
    # Avoid 0 * extreme gain: alpha=0 and gain_zero share exactly the same path.
    h = lp if config.alpha == 0 or config.variant == "gain_zero" else lp + config.alpha * gain
    if not torch.isfinite(h).all():
        raise ValueError("Contrast scores overflowed")
    h = h.masked_fill(~pool, -torch.inf)
    contrast = h.argmax(-1)
    q = h.softmax(-1)
    rows = torch.arange(lp.shape[0], device=lp.device)
    pb, pc = lp[rows, base], lp[rows, contrast]
    qb, qc = q[rows, base], q[rows, contrast]
    return ProposalStats(allowed_ids[base], allowed_ids[contrast], pb, pc,
                         gain[rows, base], gain[rows, contrast], config.binding["gain_used"],
                         mass, pool.sum(-1), pb.exp(), pc.exp(), qb, qc,
                         mass * qb, mass * qc, h[rows, contrast] - h[rows, base])
