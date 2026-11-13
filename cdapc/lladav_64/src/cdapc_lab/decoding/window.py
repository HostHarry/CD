import math


def choose_window(state):
    return state.pending.nonzero(as_tuple=True)[0][:state.window_width]


def compute_next_window_state(state, stats, gates, positions, window):
    if window.kind == "fixed":
        return {"next_width": state.window_width, "ema": state.pressure_ema}
    if not positions.numel():
        raise ValueError("An empty window has already terminated; do not update")
    count = positions.numel()
    deferred = float((~gates.eligible).sum()) / count
    flipped = float(((stats.contrast_ids[positions] != stats.base_ids[positions]) & gates.eligible).sum()) / count
    observed = state.previous_observed[positions] & state.pending[positions]
    history = positions[observed]
    support = history.numel()
    instability = float((stats.contrast_ids[history] != state.previous_candidate_ids[history]).sum()) / support if support else 0.0
    pressure = (deferred + flipped + instability) / 2
    ema = window.ema_decay * state.pressure_ema + (1 - window.ema_decay) * pressure
    target = window.maximum - math.floor((window.maximum - window.minimum) * ema + .5)
    change = max(-window.shrink_step, min(window.expand_step, target - state.window_width))
    width = max(window.minimum, min(window.maximum, state.window_width + change))
    return {"f_def": deferred, "f_flip": flipped, "f_inst": instability, "instability_support": support,
            "pressure": pressure, "ema": ema, "target": target, "previous_width": state.window_width,
            "next_width": width, "controller": "ccaw_v1"}
