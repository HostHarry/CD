from dataclasses import replace
import torch
import pytest
from cdapc_lab.config import CDConfig
from cdapc_lab.types import PreparedInput, ForwardResult, PairedForwardResult, ComputeCost
from cdapc_lab.decoding.engine import decode_cdapc


class PositionAdapter:
    """Constant per-position visual/ablated logits."""

    def __init__(self, visual_rows, ablated_rows=None):
        self.visual = torch.stack([torch.tensor(row).log() for row in visual_rows])[None]
        self.ablated = torch.stack([torch.tensor(row).log() for row in ablated_rows or visual_rows])[None]

    def forward_pair(self, prepared, state, execution="sequential", cache="disabled"):
        v = self.visual[:, :prepared.canvas_length].clone()
        a = self.ablated[:, :prepared.canvas_length].clone()
        return PairedForwardResult(ForwardResult(v, ComputeCost()), ForwardResult(a, ComputeCost()),
                                   ComputeCost(2, 2, 2, 0, 2 * prepared.canvas_length))


def prepared(length=4):
    return PreparedInput("test", torch.tensor([[7, 8]]), torch.tensor([True, False]),
                         torch.arange(3), 3, (2,), None, length, "synthetic")


def prefix_then_eos():
    # position 0: non-EOS top token; positions 1-3: EOS (id 2) top token
    return PositionAdapter([[.9, .05, .05], [.05, .05, .8], [.05, .05, .8], [.05, .05, .8]])


def test_eos_prefix_stop_matches_fill_canvas_prefix():
    cfg = CDConfig(max_commit_per_iteration=1)
    fill = decode_cdapc(PositionAdapter([[.9, .05, .05]] * 4), prepared(), cfg, trace=True)
    stopped = decode_cdapc(prefix_then_eos(), prepared(), replace(cfg, stop_rule="eos_prefix_complete"), trace=True)
    fill_same = decode_cdapc(prefix_then_eos(), prepared(), cfg, trace=True)
    assert fill_same.finish_reason == "canvas_filled" and fill_same.decode_iterations == 4
    assert stopped.finish_reason == "eos_prefix_complete"
    assert stopped.decode_iterations < fill_same.decode_iterations
    first_eos = int(stopped.response_ids.tolist().index(2))
    assert torch.equal(stopped.response_ids[:first_eos + 1], fill_same.response_ids[:first_eos + 1])
    assert (stopped.response_ids[first_eos + 1:] == 3).all()
    normalize = lambda ids: ids[:ids.index(2)] if 2 in ids else ids
    assert normalize(stopped.response_ids.tolist()) == normalize(fill_same.response_ids.tolist())
    for early, late in zip(stopped.traces, fill_same.traces):
        assert early["decision"]["positions"] == late["decision"]["positions"]
        assert early["decision"]["written_ids"] == late["decision"]["written_ids"]


def test_eos_with_earlier_pending_keeps_decoding():
    # position 0 has the lowest readiness and a non-EOS top token; it commits last
    adapter = PositionAdapter([[.34, .33, .33], [.05, .05, .8], [.05, .05, .8], [.05, .05, .8]])
    result = decode_cdapc(adapter, prepared(), replace(CDConfig(max_commit_per_iteration=1),
                                                       stop_rule="eos_prefix_complete"), trace=True)
    first_commit = result.traces[0]["decision"]["positions"][0]
    assert first_commit > 0
    assert result.response_ids[first_commit].item() == 2
    assert result.decode_iterations == 4 and result.finish_reason == "canvas_filled"
    assert result.response_ids.tolist() == [0, 2, 2, 2]


def test_unknown_stop_rule_rejected():
    with pytest.raises(ValueError):
        CDConfig(stop_rule="bogus")


def test_commit_log_matches_decisions_and_is_optional():
    cfg = CDConfig(max_commit_per_iteration=2)
    adapter = prefix_then_eos()
    plain = decode_cdapc(adapter, prepared(), cfg)
    logged = decode_cdapc(adapter, prepared(), cfg, commit_log=True)
    assert torch.equal(plain.response_ids, logged.response_ids)
    assert plain.metrics == logged.metrics and plain.commit_log == []
    decisions = [(d["positions"], d["written_ids"], d["base_ids"], d["contrast_ids"], d["reason"])
                 for d in (t["decision"] for t in logged.traces)] if logged.traces else None
    traced = decode_cdapc(adapter, prepared(), cfg, commit_log=True, trace=True)
    assert [e["position"] for e in logged.commit_log] == \
        [p for t in traced.traces for p in t["decision"]["positions"]]
    for t in traced.traces:
        for i, p in enumerate(t["decision"]["positions"]):
            entry = next(e for e in logged.commit_log if e["iteration"] == t["iteration"] and e["position"] == p)
            assert entry["written_id"] == t["decision"]["written_ids"][i]
            assert entry["base_id"] == t["decision"]["base_ids"][i]
            assert entry["contrast_id"] == t["decision"]["contrast_ids"][i]
            assert entry["reason"] == t["decision"]["reason"]
    for entry in logged.commit_log:
        lps = [t["logp_visual"] for t in entry["top5"]]
        assert lps == sorted(lps, reverse=True) and len(entry["top5"]) <= 5
        assert entry["top5"][0]["id"] == entry["base_id"]
