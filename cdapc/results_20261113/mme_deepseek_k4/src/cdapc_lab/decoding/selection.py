import torch
from cdapc_lab.types import CommitDecision
from .gates import target_scores


def choose_commits(stats, gates, positions, config):
    if not positions.numel():
        raise ValueError("Cannot commit from an empty window")
    u, r = target_scores(stats, config)
    eligible = positions[gates.eligible].tolist()
    if eligible:
        selected = sorted(eligible, key=lambda i: (-float(r[i]), -float(u[i]), i))[:config.max_commit_per_iteration]
        reason = "qualified"
    else:
        readiness = []
        if config.u_gate.enabled:
            readiness.append(u / config.u_gate.threshold)
        if config.r_gate.enabled:
            readiness.append(r / config.r_gate.threshold)
        if not readiness:
            raise RuntimeError("Disabled gates must qualify every position")
        rho = torch.stack(readiness).amin(0)
        selected = [min(positions.tolist(), key=lambda i: (-float(rho[i]), -float(r[i]), i))]
        reason = "fallback"
    selected = torch.tensor(selected, dtype=torch.long, device=positions.device)
    base, contrast = stats.base_ids[selected], stats.contrast_ids[selected]
    written = contrast if reason == "qualified" and config.binding["write_target"] == "contrast" else base
    return CommitDecision(selected, written, base, contrast, reason, gates.target, len(eligible))
