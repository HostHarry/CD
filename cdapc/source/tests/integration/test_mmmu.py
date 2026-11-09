import json
from pathlib import Path
from unittest.mock import patch

import pytest

from cdapc_lab.types import Sample
from cdapc_lab.data.records import TASKS, build_question, uses_model_reasoning_instruction
from cdapc_lab.data.mmmu import mmmu_context, mmmu_judge_question, choose_image, strip_image_placeholders
from cdapc_lab.evaluation.scorers import load_mmmu_official, mmmu_random_for, _StrictRandom, MmmuUnparsed
from cdapc_lab.experiments.mmmu_report import question_bootstrap
from cdapc_lab.config import resolve_config

ROOT = Path(__file__).resolve().parents[2]


def mc_sample():
    return Sample("validation_Test_1", "What is shown? <image 1>", {"A": "cat", "B": "dog"},
                  ("img.png",), task_id="mmmu")


def open_sample():
    return Sample("validation_Test_2", "Compute <image 1> the sum.", {}, ("img.png",), task_id="mmmu")


def test_wino_prompt_strings():
    assert mmmu_context("Q <image 1>?", {"A": "x", "B": "y"}) == "Q ?\nA. x\nB. y\n"
    assert mmmu_context("Q <image 2>.", {}) == "Q .\n"
    assert mmmu_judge_question("Q <image 1>?", {"A": "x"}) == "Q <image 1>?\nA. x\n"
    assert build_question(mc_sample()) == "What is shown? \nA. cat\nB. dog\n"
    assert build_question(open_sample()) == "Compute  the sum.\n"
    assert uses_model_reasoning_instruction(mc_sample())


def test_placeholder_variants():
    assert strip_image_placeholders("<image 1> a <image2> b <image>") == " a  b "


def test_choose_image():
    assert choose_image("<image 2> q <image 1>") == "image_1"
    assert choose_image("nothing") == "image_1"
    assert choose_image("<image 3>") == "image_3"


@pytest.fixture(scope="module")
def scorer():
    return load_mmmu_official(ROOT / "upstream/mmmu/eval/eval_utils.py",
                              ROOT / "upstream/mmmu/eval/data_utils.py")


def test_official_binding_and_seeding(scorer):
    opts = {"A": "x1", "B": "x2", "C": "x3"}
    assert scorer(mmmu_random_for("s1"))("It is (B).", opts, "B", "multiple-choice")["correct"]
    a = scorer(mmmu_random_for("zz"))("no answer here at all folks", opts, "A", "multiple-choice")
    b = scorer(mmmu_random_for("zz"))("no answer here at all folks", opts, "A", "multiple-choice")
    assert a == b  # deterministic per-sample seed
    assert scorer(_StrictRandom())("no answer here at all folks", opts, "A", "multiple-choice")["unparsed"]
    assert scorer(mmmu_random_for("o"))("The answer is 42.", {}, "42", "open")["correct"]
    with pytest.raises(ValueError):
        load_mmmu_official(ROOT / "upstream/m3cot/utils/metric.py",
                           ROOT / "upstream/mmmu/eval/data_utils.py")


def test_judge_request_and_parse():
    from cdapc_lab.evaluation.llm_judge import judge_spec, build_request, judge_prediction, ANSWER_RE
    spec, spec_hash = judge_spec("deepseek-flash")
    assert spec["model"] == "deepseek-flash" and len(spec_hash) == 64
    req = build_request(spec, "Q?", "A", "pred")
    assert req["messages"][0]["content"].startswith("You are a helpful and precise")
    assert "Q?" in req["messages"][1]["content"] and "pred" in req["messages"][1]["content"]
    fake = {"choices": [{"message": {"content": "1"}}], "model": "deepseek-flash", "usage": {}}
    with patch("cdapc_lab.evaluation.llm_judge._post", return_value=fake):
        out = judge_prediction(spec, "Q?", "A", "x", "KEY")
    assert out["score"] == 1 and out["strict_judge"] and out["request_sha256"]
    fake["choices"][0]["message"]["content"] = "maybe"
    with patch("cdapc_lab.evaluation.llm_judge._post", return_value=fake):
        out = judge_prediction(spec, "Q?", "A", "x", "KEY")
    assert out["score"] == 0 and out["strict_judge"] is False
    match = ANSWER_RE.search("blah <answer>  B  </answer> tail")
    assert match.group(1) == "B"


def test_question_bootstrap():
    a = {"1": 1.0, "2": 0.0, "3": 1.0}
    b = {"1": 1.0, "2": 1.0, "3": 1.0}
    res = question_bootstrap(a, b)
    assert res["n"] == 3 and res["mean_delta"] == pytest.approx(1 / 3)
    assert res["ci95"][0] <= res["mean_delta"] <= res["ci95"][1]
    assert question_bootstrap(a, a)["mean_delta"] == 0
    assert question_bootstrap(a, a)["ci95"] == [0.0, 0.0]


def test_config_pairing():
    cfg = resolve_config(ROOT / "configs/experiments/mmada_mmmu_cdapc_r90_smoke.yaml")
    assert cfg["protocol"]["task"] == "mmmu" and cfg["protocol"]["prompt_id"] == "mmmu_wino_reasoning_v1"
    assert cfg["protocol"]["strict_parser_id"] == "mmmu_rule_noguess_v1"
    import tempfile, yaml
    raw = yaml.safe_load(open(ROOT / "configs/experiments/mmada_mmmu_cdapc_r90_smoke.yaml"))
    raw["protocol"]["prompt_id"] = "pope_yesno_v1"
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(raw, f)
        with pytest.raises(ValueError):
            resolve_config(f.name)


def test_budget_exhaustion_stays_in_accuracy_and_cost(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from cdapc_lab.experiments import mmmu_report as reporting, score as scoring
    from cdapc_lab.io.artifacts import atomic_json, safe_id

    rows = [
        {"sample_id": sid, "generation_status": status, "official": correct,
         "strict_accuracy": correct}
        for sid, status, correct in (("ok", "success", 1), ("budget", "budget_exhausted", 0),
                                     ("missing", "missing", 0))
    ]
    monkeypatch.setattr(scoring, "select_evaluation",
                        lambda *args: ({"evaluation_fingerprint": "eval"}, rows))
    monkeypatch.setattr(reporting, "judge_spec", lambda name: ({}, "judge"))
    for sid, status, seconds, steps in (("ok", "success", 1, 2),
                                       ("budget", "budget_exhausted", 4, 128)):
        filename = safe_id(sid) + ".json"
        atomic_json(tmp_path / "predictions" / filename, {
            "sample_id": sid, "status": status, "generation_seconds": seconds,
            "decode_iterations": steps, "finish_reason": status})
        atomic_json(tmp_path / "judge/judge" / filename,
                    {"status": "success", "score": 1, "strict_judge": True})
    ids = [row["sample_id"] for row in rows]
    gold = {sid: SimpleNamespace(metric_metadata={"question_type": "multiple-choice"})
            for sid in ids}
    block = reporting.method_block([tmp_path], None, "test", ids, gold, [])
    assert block["blocks"]["all"]["official"] == {"n": 3, "accuracy": pytest.approx(1 / 3)}
    assert block["blocks"]["all"]["strict"]["n"] == 3
    assert block["per_sample"]["budget"]["judge"] == 0
    assert block["judged"] == 2
    assert block["cost"]["count"] == 2
    assert block["cost"]["generation_seconds"] == 5
    assert block["finish_reason_counts"]["budget_exhausted"] == 1
    assert block["mechanism_totals"]["decode_iterations"] == 130
