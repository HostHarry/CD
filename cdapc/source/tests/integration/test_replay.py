from dataclasses import replace
import torch
from cdapc_lab.config import CDConfig
from cdapc_lab.types import PreparedInput, ForwardResult, PairedForwardResult, ComputeCost
from cdapc_lab.decoding.engine import decode_cdapc
from cdapc_lab.experiments.replay import fork_once


class LogitsAdapter:
    def __init__(self, switch=True):
        self.switch = switch

    def forward_pair(self, prepared, state, execution="sequential", cache="disabled"):
        v = torch.tensor([.7, .2, .1]).log().repeat(1, 4, 1)
        a = torch.tensor([.8999, .0001, .1]).log().repeat(1, 4, 1) if self.switch else v
        return PairedForwardResult(ForwardResult(v, ComputeCost()), ForwardResult(a, ComputeCost()), ComputeCost(2, 2, 2, 0, 12))


def prepared():
    return PreparedInput("test", torch.tensor([[7, 8]]), torch.tensor([True, False]), torch.arange(3), 3, (2,), None, 4, "test")


def test_content_swap_and_mismatch():
    p, cfg, adapter = prepared(), CDConfig(max_commit_per_iteration=2), LogitsAdapter()
    full = decode_cdapc(adapter, p, cfg)
    result = fork_once(adapter, p, cfg, full.first_flip)
    assert result["status"] == "valid_pair"
    assert result["initial_states"]["A"] == [0, 1, 3, 3]
    assert result["initial_states"]["B"] == [1, 1, 3, 3]
    assert torch.equal(result["outputs"]["B"].response_ids, full.response_ids)
    corrupted = dict(full.first_flip, base_id=2)
    assert fork_once(adapter, p, cfg, corrupted)["status"] == "replay_mismatch"


def test_full_no_switch_subset_matches_crossed():
    adapter, p = LogitsAdapter(False), prepared()
    cfg = CDConfig()
    full = decode_cdapc(adapter, p, cfg, trace=True)
    crossed = decode_cdapc(adapter, p, replace(cfg, variant="timing_only_crossed"), trace=True)
    assert full.metrics["actual_switches"] == 0
    assert torch.equal(full.response_ids, crossed.response_ids)
    assert full.traces == crossed.traces


def test_missing_source_is_not_a_no_flip_observation(tmp_path, monkeypatch):
    from cdapc_lab.experiments import replay as driver
    from cdapc_lab.io.artifacts import atomic_json
    from cdapc_lab.types import Sample
    config = {"method": "cdapc", "cdapc": {"variant": "full", "window": {"kind": "fixed", "width": 16}},
              "protocol": {"data_root": str(tmp_path / "data")}}
    atomic_json(tmp_path / "resolved_config.json", config)
    atomic_json(tmp_path / "run_manifest.json", {"generation_fingerprint": "fixed"})
    atomic_json(tmp_path / "data/gold.json", [{"sample_id": "missing", "answer": "A", "metric_metadata": {}}])
    samples = [Sample("missing", "q", {"A": "a"}, ("unused",))]
    monkeypatch.setattr(driver, "preflight", lambda _: (samples, {}, {"generation_fingerprint": "fixed"}))
    result = driver.replay(tmp_path)
    assert result["counts"]["source_failure"] == 1
    assert result["counts"]["no_eligible_flip"] == 0 and result["triggered_events"] == 0
    assert result["replay_source_hash"]
