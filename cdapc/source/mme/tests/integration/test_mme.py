import copy
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.prepare_mme import validate_pairs
from scripts.score_mme import metrics, official_protocol
from scripts.run_mme_256_128 import k_generation_finished, finished
from cdapc_lab.config import resolve_config
from cdapc_lab.data.records import build_question, uses_model_reasoning_instruction, load_manifest
from cdapc_lab.io.artifacts import atomic_json, read_json
from cdapc_lab.types import Sample


@pytest.mark.parametrize("text,expected", [
    ("yes", "yes"), ("No.", "no"), ("YES, it is.", "yes"), ("Nope", "no"),
    ("not sure", "no"), ("I think yes", "other"), ("", "other"),
    ("<think>yes</think> yes", "other"),
])
def test_original_official_answer_parser(text, expected):
    _, parse = official_protocol()
    assert parse(text) == expected


def paired_rows():
    return [{"sample_id": f"{category}:{group}:{answer}", "group_id": f"{category}:{group}",
             "category": category, "answer": answer, "question": f"Question {answer}?",
             "image_sha256": f"sha-{category}-{group}"}
            for category in ("existence", "code_reasoning")
            for group in (1, 2) for answer in ("yes", "no")]


def test_acc_plus_keeps_missing_answers_and_uses_image_pairs():
    rows = paired_rows()
    answers = {r["sample_id"]: r["answer"] for r in rows}
    answers.pop("existence:2:no")
    answers["code_reasoning:1:yes"] = "no"
    answers["code_reasoning:1:no"] = "yes"
    result = metrics(rows, answers, {"Perception": ["existence"], "Cognition": ["code_reasoning"]})
    assert result["perception"] == 125
    assert result["cognition"] == 100
    assert result["total"] == 225
    assert result["per_category"]["existence"]["acc"] == .75
    assert result["per_category"]["existence"]["acc_plus"] == .5
    assert result["per_category"]["existence"]["other"] == 1
    with pytest.raises(ValueError, match="Duplicate"):
        metrics(rows + rows[:1], answers, {"Perception": ["existence"], "Cognition": ["code_reasoning"]})
    with pytest.raises(ValueError, match="category"):
        metrics(rows[:4], answers, {"Perception": ["existence"], "Cognition": ["code_reasoning"]})


def test_category_scores_match_unmodified_official_script(tmp_path):
    categories, _ = official_protocol()
    rows, answers = [], {}
    for category in [c for values in categories.values() for c in values]:
        lines = []
        for i, gold in enumerate(("yes", "no")):
            sid = f"{category}:{i}"
            rows.append({"sample_id": sid, "group_id": category, "category": category, "answer": gold})
            answers[sid] = "yes"
            lines.append(f"one.png\tQuestion {i}?\t{gold}\tYes\n")
        (tmp_path / f"{category}.txt").write_text("".join(lines))
    local = metrics(rows, answers, categories)
    official = subprocess.run(
        [sys.executable, str(ROOT / "upstream/mme/calculation.py"), "--results_dir", str(tmp_path)],
        capture_output=True, text=True, check=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "vendor")})
    observed = dict(re.findall(r"^[ \t]+(\w+)[ \t]+score:\s+([\d.eE+-]+)", official.stdout, re.M))
    assert set(observed) == set(local["per_category"])
    assert all(float(observed[k]) == row["score"] == 50 for k, row in local["per_category"].items())
    assert local["perception"] == 500 and local["cognition"] == 200


def test_preparation_rejects_broken_image_pairs():
    rows = paired_rows()
    assert len(validate_pairs(rows)) == 4
    for field, value in (("answer", "yes"), ("image_sha256", "different"), ("question", "Question yes?")):
        broken = copy.deepcopy(rows)
        broken[1][field] = value
        with pytest.raises(ValueError, match="Invalid MME"):
            validate_pairs(broken)
    with pytest.raises(ValueError, match="Invalid MME"):
        validate_pairs(rows[:-1])


def test_mme_uses_original_question_and_256_128():
    config = resolve_config(ROOT / "configs/experiments/mmada_mme_cdapc_r70_L256_I128_shard0.yaml")
    assert config["protocol"]["canvas_length"] == 256
    assert config["cdapc"]["iteration_limit"] == 128 and config["cdapc"]["budget_limited"]
    assert config["cdapc"]["r_gate"]["threshold"] == .7
    sample = Sample("test", "Is it red? Please answer yes or no.", {}, ("image.png",), task_id="mme")
    assert build_question(sample, "mme_yesno_v1") == sample.question
    assert not uses_model_reasoning_instruction(sample, "mme_yesno_v1")


def test_full_manifest_has_disjoint_shards_and_keeps_pairs_together():
    rows = read_json(ROOT / "data/mme/rows.json")
    groups = validate_pairs(rows)
    assert len(rows) == 2374 and len(groups) == 1187
    ids = set(read_json(ROOT / "manifests/mme_all_ids.json"))
    seen = set()
    row_by_id = {r["sample_id"]: r for r in rows}
    for shard in range(3):
        selected = read_json(ROOT / f"manifests/mme_shard{shard}_ids.json")
        assert not seen.intersection(selected)
        seen.update(selected)
        subset = {sid: row_by_id[sid] for sid in selected}
        assert len(validate_pairs(list(subset.values()))) * 2 == len(selected)
    assert seen == ids


def test_queue_waits_for_both_k_conditions():
    assert not k_generation_finished({"conditions": {}})
    state = {"conditions": {"4": {"status": "generated"}, "2": {"status": "generating"}}}
    assert not k_generation_finished(state)
    state["conditions"]["2"]["status"] = "generated"
    assert k_generation_finished(state)


def test_budget_exhaustion_is_terminal_but_errors_are_not(tmp_path):
    manifest = {"status": "incomplete", "sample_count": 2,
                "sample_status_counts": {"success": 1, "budget_exhausted": 1}}
    atomic_json(tmp_path / "run_manifest.json", manifest)
    assert finished(str(tmp_path))
    manifest["sample_status_counts"] = {"error": 1, "budget_exhausted": 1}
    atomic_json(tmp_path / "run_manifest.json", manifest)
    assert not finished(str(tmp_path))
