from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
import pytest
from cdapc_lab.types import Sample, PreparedInput, ForwardResult, PairedForwardResult, ComputeCost
from cdapc_lab.io.artifacts import read_json, atomic_json, safe_id, file_hash
from cdapc_lab.experiments import infer as runner
from cdapc_lab.experiments.score import score_run, summarize


class ToyAdapter:
    def prepare(self, sample):
        return PreparedInput(sample.sample_id, torch.tensor([[7, 8]]), torch.tensor([True, False]),
                             torch.arange(3), 3, (2,), None, 4, "toy")

    def forward_pair(self, prepared, state, execution, cache):
        v = torch.tensor([.7, .2, .1]).log().repeat(1, 4, 1)
        a = torch.tensor([.8999, .0001, .1]).log().repeat(1, 4, 1)
        return PairedForwardResult(ForwardResult(v, ComputeCost()), ForwardResult(a, ComputeCost()), ComputeCost(2, 2, 2, 0, 12))


def test_resume_offline_score_and_fingerprints(tmp_path, monkeypatch):
    samples = [Sample(x, "question", {"A": "cat", "B": "dog"}, ("unused",)) for x in ("a", "b")]
    manifest = {"samples": [{"sample_id": x.sample_id, "group_id": x.sample_id} for x in samples]}
    checkpoint = tmp_path / "checkpoint"
    atomic_json(checkpoint / "tokenizer.json", {"fixture": "original"})
    provenance = {"generation_fingerprint": "frozen", "environment": {"device": "cpu"},
                  "assets": {"model_files": {"tokenizer.json": file_hash(checkpoint / "tokenizer.json")}}}
    monkeypatch.setattr(runner, "preflight", lambda _: (samples, manifest, provenance))
    cfg = {"run": {"seed": 7, "replicate_id": 0, "output_root": str(tmp_path / "runs"), "fail_on_error": True},
           "runtime": {"device": "cpu"}, "method": "cdapc", "cdapc": {"max_commit_per_iteration": 2},
           "recording": {"trace_level": "off", "snapshot_first_eligible_flip": True},
           "model": {"checkpoint": str(checkpoint)}, "protocol": {"data_root": str(tmp_path / "data"),
                  "normalizer_id": "first_eos_v1", "strict_parser_id": "strict_final_v1", "primary_metric": "official"}}
    interrupted = runner.infer(cfg, adapter=ToyAdapter(), max_samples=1)
    assert read_json(interrupted / "run_manifest.json")["status"] == "incomplete"
    original = read_json(interrupted / "predictions" / (safe_id("a") + ".json"))
    runner.infer(cfg, interrupted, adapter=ToyAdapter())
    assert read_json(interrupted / "predictions" / (safe_id("a") + ".json")) == original
    complete = runner.infer(cfg, adapter=ToyAdapter())
    for sid in ("a", "b"):
        a = read_json(interrupted / "predictions" / (safe_id(sid) + ".json"))
        b = read_json(complete / "predictions" / (safe_id(sid) + ".json"))
        assert a["raw_response_ids"] == b["raw_response_ids"] and a["sample_seed"] == b["sample_seed"]
        assert a["cost"] == b["cost"]
    atomic_json(tmp_path / "data/gold.json", [{"sample_id": sid, "answer": "B", "metric_metadata": {}} for sid in ("a", "b")])
    tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: "Final answer: (B)")
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer)))
    evaluation = score_run(interrupted)
    summary = summarize(interrupted, evaluation["evaluation_fingerprint"])
    assert summary["metrics"]["strict_accuracy"]["numerator"] == 2
    assert summary["metrics"]["official"]["denominator"] == 2
    assert read_json(interrupted / "predictions" / (safe_id("a") + ".json")) == original
    from cdapc_lab.experiments import score as scoring
    monkeypatch.setattr(scoring, "source_hash", lambda *a: "changed_parser_source")
    revised = score_run(interrupted)
    assert revised["evaluation_fingerprint"] != evaluation["evaluation_fingerprint"]
    assert read_json(interrupted / "run_manifest.json")["generation_fingerprint"] == "frozen"
    with pytest.raises(ValueError, match="explicit"):
        summarize(interrupted)
    atomic_json(checkpoint / "tokenizer.json", {"fixture": "changed"})
    with pytest.raises(ValueError, match="tokenizer/config changed"):
        score_run(interrupted)
    monkeypatch.setattr(runner, "preflight", lambda _: (samples, manifest, {"generation_fingerprint": "changed"}))
    with pytest.raises(ValueError, match="fingerprint"):
        runner.infer(cfg, interrupted, adapter=ToyAdapter())


def test_queue_accepts_only_finished_budget_limited_runs(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    import run_chair_pope_extensions as queue
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    run = tmp_path / "runs/bounded"
    config = {"run": {"name": "bounded"}, "cdapc": {"budget_limited": True}}
    manifest = {"status": "incomplete", "sample_count": 3,
                "sample_status_counts": {"success": 2, "budget_exhausted": 1},
                "generation_fingerprint": "old"}
    atomic_json(run / "resolved_config.json", config)
    atomic_json(run / "run_manifest.json", manifest)
    assert queue.completed("runs/bounded")
    assert queue.locate_run("bounded", "old") == "runs/bounded"
    assert queue.locate_run("bounded", "new") is None
    for counts in ({"success": 1, "budget_exhausted": 1},
                   {"success": 1, "budget_exhausted": 1, "error": 1},
                   {"success": 1, "budget_exhausted": 1, "missing": 1}):
        atomic_json(run / "run_manifest.json", manifest | {"sample_status_counts": counts})
        assert not queue.completed("runs/bounded")
    atomic_json(run / "run_manifest.json", manifest | {"status": "running"})
    assert not queue.completed("runs/bounded")
    atomic_json(run / "run_manifest.json", manifest)
    config["cdapc"]["budget_limited"] = False
    atomic_json(run / "resolved_config.json", config)
    assert not queue.completed("runs/bounded")


def test_mmmu_runtime_change_keeps_old_run_and_selects_matching_run(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    import run_chair_pope_extensions as queue
    import run_mmmu_r70_256_128 as controller
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    monkeypatch.setattr(controller, "ROOT", tmp_path)
    config = {"run": {"name": "bounded"}}
    old = tmp_path / "runs/old"
    original = {"generation_fingerprint": "old", "environment": {"platform": "old"}}
    atomic_json(old / "resolved_config.json", config)
    atomic_json(old / "run_manifest.json", original)
    atomic_json(old / "predictions/sample.json", {"status": "success"})
    monkeypatch.setattr(controller, "resolve_config", lambda path: config)
    monkeypatch.setattr(controller, "preflight", lambda cfg: (
        [], {}, {"generation_fingerprint": "new", "environment": {"platform": "new"}}))
    item = {"run": "runs/old", "name": "bounded", "generated": 1}
    controller.select_generation_run(item, old / "resolved_config.json")
    assert item["run"] is None and item["generated"] == 0
    assert item["previous_runs"][0]["run"] == "runs/old"
    assert item["previous_runs"][0]["saved_predictions"] == 1
    assert read_json(old / "run_manifest.json") == original
    assert read_json(old / "predictions/sample.json") == {"status": "success"}
    new = tmp_path / "runs/new"
    atomic_json(new / "resolved_config.json", config)
    atomic_json(new / "run_manifest.json",
                {"generation_fingerprint": "new", "environment": {"platform": "new"}})
    controller.select_generation_run(item, new / "resolved_config.json")
    assert item["run"] == "runs/new"
    assert len(item["previous_runs"]) == 1
