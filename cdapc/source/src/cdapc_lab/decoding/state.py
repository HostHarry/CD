import torch
from cdapc_lab.types import DecodeState


def initialize_response(prepared, config):
    ids = torch.full((prepared.canvas_length,), prepared.mask_id, dtype=torch.long, device=prepared.prompt_ids.device)
    width = config.window.width if config.window.kind == "fixed" else config.window.initial
    return DecodeState(ids, torch.ones_like(ids, dtype=torch.bool), 0, width,
                       0.0, torch.zeros_like(ids), torch.zeros_like(ids, dtype=torch.bool))


def apply_commit(state, decision, prepared):
    positions, ids = decision.positions, decision.written_ids
    if positions.ndim != 1 or positions.numel() == 0 or positions.unique().numel() != positions.numel():
        raise ValueError("Commit positions must be nonempty and unique")
    if ids.shape != positions.shape or (positions < 0).any() or (positions >= state.response_ids.numel()).any():
        raise ValueError("Invalid commit shape or position")
    if not state.pending[positions].all() or (ids == prepared.mask_id).any() or not torch.isin(ids, prepared.allowed_text_vocab).all():
        raise ValueError("Cannot overwrite committed positions or write illegal tokens")
    out = state.clone()
    out.response_ids[positions] = ids
    out.pending[positions] = False
    return out


def update_observation_history(state, stats, positions):
    state.previous_observed.zero_()
    state.previous_observed[positions] = state.pending[positions]
    state.previous_candidate_ids[positions] = stats.contrast_ids[positions]
    return state
