from copy import deepcopy
import pytest
from cdapc_lab.io.artifacts import atomic_json, read_json, digest, generation_projection, reusable_prediction, sample_seed, safe_id
from cdapc_lab.analysis.paired import paired_statistics


def test_generation_projection_separates_evaluation_and_recording():
    cfg = {"run": {"seed": 7, "replicate_id": 0, "name": "a", "output_root": "runs"},
           "model": {"checkpoint": "/a", "upstream_root": "/b", "vision_checkpoint": "/c", "checkpoint_revision": "v1"},
           "protocol": {"normalizer_id": "a", "strict_parser_id": "b", "primary_metric": "official", "prompt_id": "v1", "data_root": "data"},
           "recording": {"trace_level": "off"}}
    changed = deepcopy(cfg)
    changed["protocol"].update(normalizer_id="changed", strict_parser_id="changed", primary_metric="strict")
    changed["recording"]["trace_level"] = "full"
    changed["run"]["name"] = "b"
    assert digest(generation_projection(cfg)) == digest(generation_projection(changed))
    changed["protocol"]["prompt_id"] = "v2"
    assert digest(generation_projection(cfg)) != digest(generation_projection(changed))


def test_atomic_identity_and_failed_resume(tmp_path):
    path = tmp_path / "predictions" / (safe_id("../../unsafe") + ".json")
    atomic_json(path, {"sample_id": "../../unsafe", "generation_fingerprint": "a", "status": "error"})
    assert not reusable_prediction(path, "../../unsafe", "a")
    with pytest.raises(ValueError):
        reusable_prediction(path, "../../unsafe", "b")
    atomic_json(path, {"sample_id": "../../unsafe", "generation_fingerprint": "a", "status": "success"})
    assert reusable_prediction(path, "../../unsafe", "a")
    assert not list(path.parent.glob(".write-*"))
    assert sample_seed(7, "sample") == sample_seed(7, "sample")
    assert sample_seed(7, "sample") != sample_seed(7, "sample", 1)


def test_paired_groups_and_denominators():
    result = paired_statistics([0, 1, 0, 1], [1, 0, 0, 1], ["a", "a", "b", "c"], repeats=100)
    assert result["sample_count"] == 4 and result["unit_count"] == 3
    assert result["four_cells"] == {"both_correct": 1, "a_only": 1, "b_only": 1, "both_wrong": 1}
    assert result["paired_difference_b_minus_a"] == 0
