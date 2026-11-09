from types import SimpleNamespace
from dataclasses import replace
import pytest
import torch
from cdapc_lab.config import AdaptiveWindow, CDConfig, Gate, parse_cd
from cdapc_lab.types import PreparedInput, ForwardResult, PairedForwardResult, ComputeCost
from cdapc_lab.decoding.state import initialize_response, update_observation_history
from cdapc_lab.decoding.window import compute_next_window_state
from cdapc_lab.decoding.engine import decode_cdapc


def state_and_stats(window):
    p = PreparedInput("s", torch.tensor([[9]]), torch.tensor([True]), torch.arange(3), 3, (2,), None, 32, "s")
    state = initialize_response(p, CDConfig(window=window, max_commit_per_iteration=1))
    stats = SimpleNamespace(base_ids=torch.zeros(32, dtype=torch.long), contrast_ids=torch.ones(32, dtype=torch.long))
    return p, state, stats


@pytest.mark.parametrize("pressures", [[0] * 30, [1] * 30, [0, 1] * 15])
def test_bounded_pressure_sequences(pressures):
    window = AdaptiveWindow()
    _, state, stats = state_and_stats(window)
    assert state.window_width == 16 and state.pressure_ema == 0
    for pressure in pressures:
        state.previous_observed[:] = bool(pressure)
        stats.contrast_ids[:] = pressure
        state.previous_candidate_ids[:] = 0
        gates = SimpleNamespace(eligible=torch.full((4,), not bool(pressure)))
        before = state.window_width
        update = compute_next_window_state(state, stats, gates, torch.arange(4), window)
        assert update["pressure"] == pressure
        assert window.minimum <= update["next_width"] <= window.maximum
        assert -window.shrink_step <= update["next_width"] - before <= window.expand_step
        state.window_width, state.pressure_ema = update["next_width"], update["ema"]


def test_half_up_ema_and_history_expiry():
    window = AdaptiveWindow(initial=2, minimum=1, maximum=6, ema_decay=0., expand_step=9, shrink_step=9)
    _, state, stats = state_and_stats(window)
    gates = SimpleNamespace(eligible=torch.ones(2, dtype=torch.bool))
    update = compute_next_window_state(state, stats, gates, torch.arange(2), window)
    assert update["pressure"] == .5 and update["target"] == 3  # 6 - floor(2.5 + .5), not bankers rounding.
    slower = compute_next_window_state(state, stats, gates, torch.arange(2), replace(window, ema_decay=.8))
    assert slower["target"] != update["target"]
    update_observation_history(state, stats, torch.tensor([0, 1]))
    update_observation_history(state, stats, torch.tensor([2, 3]))
    stats.contrast_ids[:] = 2
    update = compute_next_window_state(state, stats, gates, torch.arange(2), window)
    assert update["instability_support"] == 0 and update["f_inst"] == 0


class Adapter:
    def forward_pair(self, prepared, state, execution, cache):
        v = torch.tensor([.7, .2, .1]).log().repeat(1, 32, 1)
        a = torch.tensor([.8999, .0001, .1]).log().repeat(1, 32, 1)
        return PairedForwardResult(ForwardResult(v, ComputeCost()), ForwardResult(a, ComputeCost()), ComputeCost(2, 2, 2, 0, 66))


def test_fixed_degeneracy_and_one_update_per_iteration():
    p, _, _ = state_and_stats(AdaptiveWindow())
    cfg = CDConfig(max_commit_per_iteration=2, u_gate=Gate(True, .9))
    fixed = decode_cdapc(Adapter(), p, cfg, trace=True)
    adaptive = decode_cdapc(Adapter(), p, replace(cfg, window=AdaptiveWindow(initial=16, minimum=16, maximum=16)), trace=True)
    assert torch.equal(fixed.response_ids, adaptive.response_ids)
    assert fixed.cost == adaptive.cost
    assert [r["decision"] for r in fixed.traces] == [r["decision"] for r in adaptive.traces]
    assert len(adaptive.metrics["window_updates"]) == adaptive.decode_iterations
    assert all(r["requested_width"] == 16 for r in adaptive.traces)
    expanding = decode_cdapc(Adapter(), p, replace(cfg, window=AdaptiveWindow()), trace=True)
    for field in ("proposals", "gates", "decision"):
        assert expanding.traces[0][field] == fixed.traces[0][field]


@pytest.mark.parametrize("window", [{"kind": "ccaw_v1", "minimum": 0}, {"kind": "ccaw_v1", "initial": 8},
                                    {"kind": "ccaw_v1", "ema_decay": 1}, {"kind": "ccaw_v1", "expand_step": 0},
                                    {"kind": "ccaw_v1", "width": 16}, {"kind": "fixed", "initial": 16}])
def test_window_validation(window):
    with pytest.raises(ValueError):
        parse_cd({"window": window})
