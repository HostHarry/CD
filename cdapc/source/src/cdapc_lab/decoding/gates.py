from cdapc_lab.types import GateResult


def target_scores(stats, config):
    if config.binding["gate_target"] == "base":
        return stats.u_b, stats.r_b
    return stats.u_c, stats.r_c


def evaluate_gates(stats, positions, config):
    u, r = target_scores(stats, config)
    pass_u = (u[positions] >= config.u_gate.threshold) | (not config.u_gate.enabled)
    pass_r = (r[positions] >= config.r_gate.threshold) | (not config.r_gate.enabled)
    return GateResult(config.binding["gate_target"], pass_u, pass_r, pass_u & pass_r)
