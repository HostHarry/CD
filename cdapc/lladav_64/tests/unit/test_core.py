from dataclasses import replace
import torch
import pytest
from cdapc_lab.config import CDConfig, Gate, FixedWindow, parse_cd
from cdapc_lab.types import PreparedInput, ForwardResult, PairedForwardResult, ComputeCost
from cdapc_lab.decoding.proposal import make_proposals
from cdapc_lab.decoding.gates import evaluate_gates
from cdapc_lab.decoding.selection import choose_commits
from cdapc_lab.decoding.state import initialize_response, apply_commit, update_observation_history
from cdapc_lab.decoding.engine import decode_cdapc
from cdapc_lab.decoding.window import choose_window


def proposals(config=None, rows=1, same=False):
    v = torch.tensor([.7, .2, .1]).log().repeat(1, rows, 1)
    a = v if same else torch.tensor([.8999, .0001, .1]).log().repeat(1, rows, 1)
    return make_proposals(v, a, torch.arange(3), config or CDConfig())


def prepared(length=16):
    return PreparedInput("test", torch.tensor([[7, 8]]), torch.tensor([True, False]),
                         torch.arange(3), 3, (2,), None, length, "synthetic")


class Adapter:
    def forward_pair(self, prepared, state, execution, cache):
        assert cache == "disabled"
        v = torch.tensor([.7, .2, .1]).log().repeat(1, prepared.canvas_length, 1)
        a = torch.tensor([.8999, .0001, .1]).log().repeat(1, prepared.canvas_length, 1)
        cost = ComputeCost(2 if execution == "sequential" else 1, 2, 2, 0, 2 * (prepared.canvas_length + 2))
        return PairedForwardResult(ForwardResult(v, ComputeCost()), ForwardResult(a, ComputeCost()), cost)


def decide(stats, cfg):
    positions = torch.arange(len(stats.base_ids))
    return choose_commits(stats, evaluate_gates(stats, positions, cfg), positions, cfg)


def test_equal_branches():
    stats = proposals(same=True)
    assert torch.equal(stats.base_ids, stats.contrast_ids)
    torch.testing.assert_close(stats.r_c, stats.u_c)


def test_alpha_zero_gain_zero():
    a, b = CDConfig(alpha=0), CDConfig(variant="gain_zero")
    for key in ("base_ids", "contrast_ids", "r_c", "u_c"):
        torch.testing.assert_close(getattr(proposals(a), key), getattr(proposals(b), key))
    assert torch.equal(decide(proposals(a), a).written_ids, decide(proposals(b), b).written_ids)


def test_singleton_mass_and_fixture():
    s = proposals(CDConfig(beta=1))
    assert s.pool_size.item() == 1
    torch.testing.assert_close(s.pool_mass, s.u_b)
    torch.testing.assert_close(s.r_b, s.u_b)
    s = proposals()
    assert (s.base_ids.item(), s.contrast_ids.item()) == (0, 1)
    assert s.u_c.item() == pytest.approx(.2)
    assert s.r_c.item() == pytest.approx(.92575, abs=1e-5)
    assert decide(s, CDConfig()).written_ids.item() == 1


@pytest.mark.parametrize("u,r,eligible", [(True, True, False), (True, False, False), (False, True, True), (False, False, True)])
def test_gate_switches(u, r, eligible):
    cfg = CDConfig(u_gate=Gate(u, .9), r_gate=Gate(r, .9))
    s = proposals(cfg)
    assert evaluate_gates(s, torch.arange(1), cfg).eligible.item() == eligible
    assert s.contrast_ids.item() == 1 and s.r_c.item() == pytest.approx(.92575, abs=1e-5)


def test_fallback_writes_base_and_protects_state():
    cfg = CDConfig(u_gate=Gate(True, .9), r_gate=Gate(True, .99))
    s = proposals(cfg, rows=16)
    d = decide(s, cfg)
    assert d.reason == "fallback" and d.positions.tolist() == [0]
    assert d.written_ids.tolist() == [0] and d.contrast_ids.tolist() == [1]
    p = prepared()
    state = initialize_response(p, cfg)
    new = apply_commit(state, d, p)
    assert new.response_ids[0] == 0 and new.pending.sum() == 15
    assert state.pending.sum() == 16 and p.prompt_ids.tolist() == [[7, 8]]
    with pytest.raises(ValueError):
        apply_commit(new, d, p)
    with pytest.raises(ValueError):
        apply_commit(state, replace(d, written_ids=torch.tensor([3])), p)


def test_stable_ties_window_and_k():
    cfg = CDConfig(max_commit_per_iteration=4)
    s = proposals(cfg, rows=16)
    d = decide(s, cfg)
    assert d.positions.tolist() == [0, 1, 2, 3]
    state = initialize_response(prepared(32), cfg)
    state.pending[1::2] = False
    assert choose_window(state).tolist() == list(range(0, 32, 2))
    logits = torch.zeros(1, 1, 5)
    s = make_proposals(logits, logits, torch.tensor([4, 1, 3]), cfg)
    assert s.base_ids.item() == 1


def test_crossed_and_base_target():
    full, crossed = CDConfig(), CDConfig(variant="timing_only_crossed")
    a, b = decide(proposals(full, 4), full), decide(proposals(crossed, 4), crossed)
    assert torch.equal(a.positions, b.positions)
    assert a.written_ids.tolist() == [1] * 4 and b.written_ids.tolist() == [0] * 4
    base = CDConfig(variant="base_candidate_visual")
    assert decide(proposals(base), base).reason == "fallback"


def test_engine_progress_cost_trace_budget():
    cfg = CDConfig(max_commit_per_iteration=4)
    a = decode_cdapc(Adapter(), prepared(), cfg)
    b = decode_cdapc(Adapter(), prepared(), cfg, trace=True)
    assert torch.equal(a.response_ids, b.response_ids)
    assert a.decode_iterations == 4 and a.cost.model_forward_calls == 8
    assert a.cost.denoiser_branch_evaluations == 8
    assert a.metrics["actual_switches"] == 16 and a.first_flip["target_position"] == 0
    assert all(row["pending_count"] == 16 - 4 * i for i, row in enumerate(b.traces))
    limited = decode_cdapc(Adapter(), prepared(), replace(cfg, iteration_limit=1, budget_limited=True))
    assert limited.finish_reason == "budget_exhausted" and (limited.response_ids == 3).sum() == 12


def test_history_expires_outside_window():
    state = initialize_response(prepared(), CDConfig())
    stats = proposals(rows=16)
    update_observation_history(state, stats, torch.tensor([0, 1]))
    assert state.previous_observed[0]
    update_observation_history(state, stats, torch.tensor([2, 3]))
    assert not state.previous_observed[0]


@pytest.mark.parametrize("kwargs", [{"variant": "unknown"}, {"alpha": -1}, {"alpha": float("nan")}, {"beta": 0}, {"temperature": 0}, {"cache": "enabled"}, {"max_commit_per_iteration": 17}, {"iteration_limit": True}, {"u_gate": {"enabled": "false"}}, {"window": {"kind": "unknown"}}])
def test_bad_configs(kwargs):
    with pytest.raises((TypeError, ValueError)):
        parse_cd(kwargs)


def test_extreme_logits_illegal_vocab():
    cfg = CDConfig(alpha=0)
    v, a = torch.tensor([[[1., 2., 3., float("nan")]]]), torch.tensor([[[-1e30, 0., 2., float("nan")]]])
    stats = make_proposals(v, a, torch.arange(3), cfg)
    assert torch.isfinite(stats.r_c).all()
    with pytest.raises(ValueError):
        make_proposals(v, a, torch.tensor([], dtype=torch.long), cfg)
    with pytest.raises(ValueError):
        make_proposals(v, a, torch.arange(4), cfg)
