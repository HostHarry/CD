import gzip
import json
from pathlib import Path
import pytest
import yaml

from cdapc_lab.config import resolve_config
from cdapc_lab.data.pope import SUFFIX, pope_input_key, split_metrics, paired_bootstrap
from cdapc_lab.data.records import build_question, load_manifest, uses_model_reasoning_instruction
from cdapc_lab.evaluation.parsers import strict_yes_no, reasoned_yes_no
from cdapc_lab.evaluation.scorers import load_pope_official_rule, score_pope_text
from cdapc_lab.experiments.score import pope_metrics
from cdapc_lab.experiments.pope_report import transition_cells, answer_switch, confidence_shift
from cdapc_lab.types import Sample

ROOT = Path(__file__).parents[2]
POPE_SCORER = ROOT / "upstream/pope/evaluate.py"
HOSTHARRY = Path("/root/autodl-tmp/HostHarry-CD")


@pytest.mark.parametrize("text,prediction", [
    ("Yes, there is a dog.", "yes"), ("No, there is no dog.", "no"), ("YES", "yes")])
def test_strict_yes_no_parsed(text, prediction):
    parsed = strict_yes_no(text)
    assert (parsed.prediction, parsed.status) == (prediction, "parsed")
    assert parsed.parser_revision == "strict_yes_no_v1"
    assert text[parsed.evidence_span[0]:parsed.evidence_span[1]].lower() in ("yes", "no")


@pytest.mark.parametrize("text,status", [
    ("Yes. No.", "ambiguous"), ("There is a cat.", "unparsed"),
    ("Not sure.", "unparsed"), ("nobody", "unparsed")])
def test_strict_yes_no_not_parsed(text, status):
    assert strict_yes_no(text).status == status


def test_pope_official_rule_pinned():
    if not POPE_SCORER.is_file():
        pytest.skip("Official source checkout absent")
    rule = load_pope_official_rule(POPE_SCORER)
    assert rule("Yes, there is a dog.") == "yes"
    assert rule("No, there is not.") == "no"
    assert rule("There is not a dog in the image. Yes.") == "no"
    assert rule("There is a dog.") == "yes"
    assert rule("") == "yes"


def test_pope_official_rule_structure_change(tmp_path):
    mutated = tmp_path / "evaluate.py"
    mutated.write_text("answers = []\nfor answer in answers:\n    pass\n")
    with pytest.raises(ValueError, match="structure changed"):
        load_pope_official_rule(mutated)


def test_build_question_pope_suffix():
    sample = Sample(sample_id="s", question="Is there a dog in the image?", options={},
                    image_paths=("x.png",), task_id="pope")
    assert build_question(sample) == "Is there a dog in the image?" + SUFFIX
    assert not uses_model_reasoning_instruction(sample)
    mcq = Sample(sample_id="m", question="Q", options={"A": "a", "B": "b"},
                 image_paths=("x.png",), task_id="m3cot")
    assert uses_model_reasoning_instruction(mcq)
    assert build_question(mcq) == "Q\n(A) a\n(B) b\nThink step by step, then end with Final answer: (X), using exactly one option letter."


def test_reasoning_and_caption_prompt_scope():
    sample = Sample("s", "Is there a dog?", {}, ("x.png",), task_id="pope")
    assert uses_model_reasoning_instruction(sample, "pope_yesno_reasoning_v1")
    assert build_question(sample, "pope_yesno_reasoning_v1").startswith("Is there a dog?\n")
    assert "Final answer: Yes" in build_question(sample, "pope_yesno_reasoning_v1")
    assert not uses_model_reasoning_instruction(sample, "pope_yesno_v1")
    caption = Sample("c", "Please describe this image in detail.", {}, ("x.png",), task_id="chair")
    assert build_question(caption, "chair_caption_v1") == caption.question
    assert not uses_model_reasoning_instruction(caption, "chair_caption_v1")


@pytest.mark.parametrize("text,expected", [
    ("<think>No dog is visible, but there is a cat.</think> Final answer: Yes", "yes"),
    ("<think>Yes, something is visible.</think>\nFinal answer: No.", "no"),
    ("<think>Final answer: Yes", None),
    ("<think>There is a dog.</think>", None),
    ("<think></think>Final answer: Yes or No", None),
    ("<think>Text</think><think>Final answer: Yes", None),
    ("Final answer: Yes", "yes"),
    ("<think>Text</think>\nNo", "no"),
    ("<think>Text</think>There is not a dog.", None),
])
def test_reasoning_answer_scope(text, expected):
    parsed = reasoned_yes_no(text)
    assert parsed.prediction == expected
    assert parsed.parser_revision == "pope_final_yes_no_v1"
    if expected:
        assert text[slice(*parsed.evidence_span)].lower() == expected


def test_reasoning_scoring_does_not_credit_missing_no():
    from cdapc_lab.experiments.pope_report import reasoning_split_metrics
    rule = load_pope_official_rule(POPE_SCORER)
    text = "<think>There is not a dog.</think>Final answer: Yes"
    score = score_pope_text(text, "yes", rule, "pope_final_yes_no_v1")
    assert score["official"] == 1 and score["raw_official"] == 0
    empty = score_pope_text("<think></think>", "no", rule, "pope_final_yes_no_v1")
    assert empty["official"] == empty["strict_accuracy"] == 0
    metrics = reasoning_split_metrics([{"text": "<think></think>", "label": "no"}])
    assert metrics["accuracy"] == 0 and metrics["invalid_ratio"] == 1
    exhausted = reasoning_split_metrics([
        {"text": "Final answer: No", "label": "no", "generation_status": "budget_exhausted"},
        {"text": "Final answer: Yes", "label": "yes", "generation_status": "success"},
    ])
    assert exhausted["n"] == 2 and exhausted["accuracy"] == .5
    assert exhausted["invalid_ratio"] == .5


def test_reasoning_parser_config_pairing(tmp_path):
    path = _config(tmp_path, prompt_id="pope_yesno_reasoning_v1", strict_parser_id="pope_final_yes_no_v1")
    assert resolve_config(path)["protocol"]["strict_parser_id"] == "pope_final_yes_no_v1"
    path = _config(tmp_path, prompt_id="pope_yesno_reasoning_v1")
    with pytest.raises(ValueError, match="strict_parser_id"):
        resolve_config(path)


@pytest.fixture
def extension_queue():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "extension_queue", ROOT / "scripts/run_chair_pope_extensions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extension_queue_r70_coverage_and_direct_before_reasoning(extension_queue):
    plan = extension_queue.stage_plan()
    keys = [extension_queue.stage_key(entry) for entry in plan]
    assert len(keys) == len(set(keys)) == 15
    assert {extension_queue.stage_key(e) for e in plan if e["tau"] == 70} == {
        "chair_r70", "pope_aokvqa_r70", "pope_gqa_r70",
        "pope_aokvqa_reasoning_r70", "pope_gqa_reasoning_r70",
    }
    assert max(i for i, e in enumerate(plan) if not e["reasoning"]) < min(
        i for i, e in enumerate(plan) if e["reasoning"])
    for reasoning in (False, True):
        assert max(i for i, e in enumerate(plan)
                   if e["dataset"] == "pope_gqa" and e["reasoning"] == reasoning) < min(
            i for i, e in enumerate(plan)
            if e["dataset"] == "pope_aokvqa" and e["reasoning"] == reasoning)


def test_extension_queue_restart_finishes_active_before_added_stages(
        extension_queue, tmp_path, monkeypatch):
    q = extension_queue
    active = "pope_aokvqa_reasoning_r90"
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"active": active, "status": "running",
                                 "stages": {active: {"status": "generating", "shards": {}}}}))
    monkeypatch.setattr(q, "INDEX", index)
    monkeypatch.setattr(q, "LOGS", tmp_path / "logs")
    observed = []

    def finish_stage(state, **entry):
        key = q.stage_key(entry)
        observed.append(key)
        state["stages"][key]["status"] = "complete"

    monkeypatch.setattr(q, "run_stage", finish_stage)
    q.main()
    assert observed[0] == active
    assert observed.count(active) == 1
    assert len(observed) == len(set(observed)) == 15
    assert observed.index("pope_gqa_r70") < observed.index("pope_aokvqa_r70")
    assert observed.index("pope_gqa_reasoning_r50") < observed.index("pope_aokvqa_reasoning_r70")
    assert observed.index("pope_gqa_r70") < observed.index("pope_aokvqa_reasoning_r70")
    assert observed.index("pope_gqa_r70") < observed.index("pope_aokvqa_reasoning_r50")
    assert json.loads(index.read_text())["status"] == "complete"
    assert json.loads(index.read_text())["order"] == ["chair", "pope_gqa", "pope_aokvqa"]


def test_extension_queue_unknown_active_stage_rejected(extension_queue):
    with pytest.raises(ValueError, match="Active stage missing"):
        extension_queue.stage_plan("unrecognized_running_experiment")


def _config(tmp_path, **protocol):
    base = yaml.safe_load((ROOT / "configs/experiments/mmada_pope_yesno_shard0.yaml").read_text())
    base["protocol"].update(protocol)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(base))
    return path


def test_config_pope_pairings(tmp_path):
    good = _config(tmp_path)
    assert resolve_config(good)["protocol"]["task"] == "pope"
    with pytest.raises(ValueError):
        resolve_config(_config(tmp_path, prompt_id="m3cot_cot_final_v1"))
    with pytest.raises(ValueError):
        resolve_config(_config(tmp_path, task="m3cot"))
    with pytest.raises(ValueError):
        resolve_config(_config(tmp_path, task="pope", strict_parser_id="strict_final_v1"))


def _pope_root(tmp_path, options):
    from PIL import Image
    root = tmp_path / "pope"
    (root / "images").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2)).save(root / "images/a.png")
    sample = {"sample_id": "x" * 64, "question": "Is there a dog?", "options": options,
              "image_paths": ["images/a.png"], "task_id": "pope", "group_id": "img.jpg",
              "public_context": None}
    (root / "samples.json").write_text(json.dumps([sample]))
    (root / "gold.json").write_text(json.dumps([{"sample_id": "x" * 64, "answer": "yes"}]))
    (root / "provenance.json").write_text(json.dumps({"revision": "r1", "split": "test"}))
    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps(["x" * 64]))
    return root, ids


def test_load_manifest_pope(tmp_path):
    root, ids = _pope_root(tmp_path, {})
    samples, manifest = load_manifest(root, ids, "r1", "test", "pope")
    assert samples[0].task_id == "pope"
    from cdapc_lab.io.artifacts import digest
    assert manifest["prompt_hash"] == digest("{question}" + SUFFIX)
    with pytest.raises(ValueError):
        load_manifest(root, ids, "r1", "test", "m3cot")
    root, ids = _pope_root(tmp_path, {"A": "a"})
    with pytest.raises(ValueError):
        load_manifest(root, ids, "r1", "test", "pope")


def test_pope_metrics_exact():
    pairs = [("yes", "yes"), ("yes", "no"), ("no", "no"), ("no", "yes"),
             (None, "yes"), (None, "no")]
    m = pope_metrics(pairs)
    assert (m["TP"], m["FP"], m["TN"], m["FN"], m["unresolved"]) == (1, 1, 1, 2, 1)
    assert m["accuracy"] == pytest.approx(2 / 6)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(1 / 3)
    assert m["f1"] == pytest.approx(2 * 0.5 * (1 / 3) / (0.5 + 1 / 3))
    assert m["yes_ratio"] == pytest.approx(2 / 6)
    assert pope_metrics([])["f1"] is None


def test_input_key_matches_native_export():
    export = HOSTHARRY / "pope/native/predictions-000.jsonl.gz"
    rows_path = ROOT / "data/pope_yesno/rows.json"
    if not export.is_file() or not rows_path.is_file():
        pytest.skip("Native export or prepared rows absent")
    record = json.loads(gzip.open(export, "rt").readline())
    rows = json.loads(rows_path.read_text())
    row = next(r for r in rows if r["input_key"] == record["input_key"])
    question = row["question"] + SUFFIX
    assert row["image_sha256"] == record["payload"]["input"]["image_sha256"]
    assert pope_input_key(row["image_sha256"], question) == record["input_key"]


def test_split_metrics_synthetic():
    rule = lambda text: "no" if "not" in text else "yes"
    rows = [
        {"id": "a", "split": "random", "label": "yes", "image_sha256": "i1", "text": "Yes"},       # tp
        {"id": "b", "split": "random", "label": "no", "image_sha256": "i1", "text": "Yes"},        # fp
        {"id": "c", "split": "random", "label": "yes", "image_sha256": "i2", "text": "not sure"},  # fn
        {"id": "d", "split": "random", "label": "no", "image_sha256": "i2", "text": "not ok"},     # tn
        {"id": "e", "split": "random", "label": "yes", "image_sha256": "i3", "text": "maybe"},     # tp + invalid
        {"id": "f", "split": "random", "label": "no", "image_sha256": "i3", "text": ""},           # fp(empty->yes) + invalid + empty
    ]
    m = split_metrics(rows, rule)
    assert m["n"] == 6
    assert m["confusion"]["tp"] == 2 and m["confusion"]["fp"] == 2
    assert m["confusion"]["tn"] == 1 and m["confusion"]["fn"] == 1
    assert m["accuracy"] == pytest.approx(3 / 6)
    assert m["f1"] == pytest.approx(4 / 7)
    assert m["yes_ratio"] == pytest.approx(4 / 6)
    assert m["invalid_ratio"] == pytest.approx(4 / 6)
    assert m["empty_answers"] == 1


def test_paired_bootstrap_identical_zero():
    rule = lambda text: "no" if "not" in text else "yes"
    rows = [{"id": f"r{i}", "split": "random", "label": "yes" if i % 2 else "no",
             "image_sha256": f"img{i % 3}", "text": "Yes"} for i in range(9)]
    first = paired_bootstrap(rows, [dict(r) for r in rows], rule)
    second = paired_bootstrap(rows, [dict(r) for r in rows], rule)
    assert first == second
    assert first["delta_accuracy_ci95"] == [0.0, 0.0]
    assert first["delta_f1_ci95"] == [0.0, 0.0]
    assert first["repeats"] == 2000 and first["seed"] == 42


def test_transition_cells_exact():
    pairs = [("yes", "yes", "yes"), ("yes", "no", "yes"), ("no", "yes", "no"),
             ("no", "no", "yes"), ("yes", "no", "no"), ("no", "yes", "yes")]
    cells = transition_cells(pairs)
    assert cells["yes->yes"] == 1 and cells["yes->no"] == 2
    assert cells["no->yes"] == 2 and cells["no->no"] == 1
    assert cells["both_correct"] == 1 and cells["fixed"] == 2
    assert cells["broken"] == 2 and cells["both_wrong"] == 1


def test_answer_switch_and_confidence():
    log = [{"position": 0, "written_id": 10, "base_id": 10, "contrast_id": 11, "top5": []},
           {"position": 3, "written_id": 11, "base_id": 10, "contrast_id": 11, "top5": []},
           {"position": 5, "written_id": 11, "base_id": 10, "contrast_id": 11, "top5": []}]
    assert answer_switch(log, 4)          # position 3 < eos 4 switched
    assert not answer_switch(log, 3)      # switches at positions 3,5 are at/after eos
    assert answer_switch(log, None)       # no EOS: all positions count
    assert not answer_switch([], None)


def test_confidence_shift_math():
    decode = {1: " yes ", 2: "no", 3: "maybe"}.get
    entry = {"position": 1, "written_id": 1, "base_id": 1, "contrast_id": 2, "top5": [
        {"id": 3, "logp_visual": -0.1, "logp_ablated": -0.2, "in_pool": True},
        {"id": 1, "logp_visual": -0.4, "logp_ablated": -1.0, "in_pool": True},
        {"id": 2, "logp_visual": -0.9, "logp_ablated": -0.3, "in_pool": False}]}
    v = confidence_shift(entry, decode, 0.5)
    lo_i = -0.4 - (-0.9)
    delta = 0.5 * ((-0.4 - (-1.0)) - (-0.9 - (-0.3)))
    assert v["LO_I"] == pytest.approx(lo_i)
    assert v["delta"] == pytest.approx(delta)
    assert v["LO_h"] == pytest.approx(lo_i + delta)
    assert v["yes_in_pool"] is True and v["no_in_pool"] is False
    assert confidence_shift({"top5": [{"id": 3, "logp_visual": -0.1, "logp_ablated": -0.2, "in_pool": True}]},
                            decode, 0.5) is None


@pytest.mark.parametrize("text,answer,status,evidence", [
    ("<think>No", "no", "answered", "No"),
    ("<think>", None, "missing", ""),
    ("Yes, there is no person.", None, "ambiguous", "Yes, there is no person."),
])
def test_pope_llm_judge_structured_answer(text, answer, status, evidence):
    import json
    from cdapc_lab.evaluation.pope_llm_judge import parse_response, request_for
    value = {"answer": answer, "status": status, "evidence": evidence}
    assert parse_response(json.dumps(value), text) == value
    request = request_for("Is there a person in the image?", text)
    assert json.loads(request["messages"][1]["content"]) == {
        "question": "Is there a person in the image?", "candidate_response": text}


@pytest.mark.parametrize("value", [
    {"answer": "yes", "status": "missing", "evidence": ""},
    {"answer": None, "status": "answered", "evidence": "No"},
    {"answer": "no", "status": "answered", "evidence": "invented quotation"},
    {"answer": "no", "status": "answered", "evidence": ""},
])
def test_pope_llm_judge_rejects_inconsistent_or_ungrounded_output(value):
    import json
    from cdapc_lab.evaluation.pope_llm_judge import parse_response
    with pytest.raises(ValueError):
        parse_response(json.dumps(value), "<think>No")


def test_pope_llm_judge_retry_and_exact_request_cache(tmp_path, monkeypatch):
    import json
    from cdapc_lab.evaluation import pope_llm_judge as judge
    outputs = iter(["invalid JSON", json.dumps({
        "answer": "no", "status": "answered", "evidence": "No"})])
    calls = []
    def post(url, request, key):
        calls.append(request)
        return {"model": "test", "choices": [{"message": {"content": next(outputs)}}]}
    monkeypatch.setattr(judge, "_post", post)
    monkeypatch.setattr(judge.time, "sleep", lambda _: None)
    request = judge.request_for("Is there a person?", "<think>No")
    result = judge.judge_request(request, tmp_path, "test-credential")
    assert result["status"] == "success" and len(result["attempts"]) == 2
    assert result["parsed"]["answer"] == "no"
    assert judge.judge_request(request, tmp_path, "test-credential") == result
    assert len(calls) == 2


def test_pope_llm_judge_cache_reuse_keeps_gold_and_row_weights_separate(tmp_path, monkeypatch):
    from scripts import run_pope_think_judge as runner
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "OUTPUT", tmp_path / "reports")
    monkeypatch.setattr(runner, "STAGES", ("fixture",))
    source = {"records": {
        "a": {"request_sha256": "shared", "gold": "no", "strict_prediction": None},
        "b": {"request_sha256": "shared", "gold": "yes", "strict_prediction": None}},
        "rows": [{"input_key": sid, "label": label, "split": split}
                 for split in ("random", "popular", "adversarial")
                 for sid, label in (("a", "no"), ("b", "yes"))],
        "source_report": "fixture"}
    ready = {"fixture": source}
    results = {"shared": {"status": "success",
                         "parsed": {"answer": "no", "status": "answered", "evidence": "No"}}}
    index = {"status": "complete", "active": None}
    summary = runner.reports(ready, results, index, tmp_path / "cache", "judging", 1)
    stage = summary["stages"]["fixture"]
    assert stage["status"] == "complete" and stage["judged_unique"] == 2
    assert stage["macro"]["accuracy"] == .5
    assert stage["paired_transitions_vs_strict"] == {"strict_0_judge_1": 3, "strict_0_judge_0": 3}
    source["records"]["b"]["request_sha256"] = "pending"
    summary = runner.reports(ready, results, index, tmp_path / "cache", "judging", 1)
    stage = summary["stages"]["fixture"]
    assert stage["status"] == "judging" and stage["judged_unique"] == 1
    assert stage["metric_scope"] == "judged_subset_only" and "macro" not in stage


def test_k_pilot_tail_wait_uses_eos_known_before_batch():
    from scripts.k_commit_report import mechanism
    pred = {
        "commit_log": [
            {"iteration": 0, "position": 3, "written_id": 99},
            {"iteration": 0, "position": 4, "written_id": 99},
            {"iteration": 1, "position": 5, "written_id": 99},
            {"iteration": 1, "position": 6, "written_id": 99},
            *[{"iteration": 2, "position": p, "written_id": 1} for p in (0, 1, 2)]],
        "decode_iterations": 3, "generation_seconds": 1.5, "emitted_tokens_before_eos": 3,
        "eos_ids": [99], "finish_reason": "eos_prefix_complete",
        "metrics": {"committed": 7, "fallback_iterations": 0, "actual_switches": 0},
    }
    result = mechanism(pred)
    assert result["tail_only_iterations"] == 1  # The first batch introduces EOS.
    assert result["batch_histogram"] == {2: 2, 3: 1}


def test_k_pilot_paired_bootstrap_keeps_duplicate_rows_in_image_clusters():
    from scripts.k_commit_report import paired_accuracy
    rows = [{"input_key": "a", "image": "one"}, {"input_key": "a", "image": "one"},
            {"input_key": "b", "image": "two"}]
    first = {"a": {"judge_correct": 0}, "b": {"judge_correct": 1}}
    second = {"a": {"judge_correct": 1}, "b": {"judge_correct": 0}}
    result = paired_accuracy(rows, first, second)
    assert result["images"] == 2 and result["delta"] == pytest.approx(1 / 3)
    assert result["ci95"] == [-1, 1]
    identical = paired_accuracy(rows, first, first)
    assert identical["delta"] == 0 and identical["ci95"] == [0, 0]


def test_k_pilot_report_scores_all_conditions_on_identical_rows(tmp_path, monkeypatch):
    from scripts import k_commit_report as report
    from cdapc_lab.io.artifacts import atomic_json
    out, cache = tmp_path / "pilot", tmp_path / "judge"
    out.mkdir()
    cache.mkdir()
    (out / "protocol.json").write_text('{"fixture":true}')
    monkeypatch.setattr(report, "OUT", out)
    monkeypatch.setattr(report, "JUDGE", cache)
    monkeypatch.setattr(report, "ROOT", tmp_path)
    atomic_json(tmp_path / "reports/pope_gqa_reasoning_r70.json",
                {"cdapc": {"per_run": [{"run": "runs/k16"}]}})
    for k in ("16", "4", "2"):
        atomic_json(tmp_path / f"runs/k{k}/run_manifest.json",
                    {"environment": {"platform": "old" if k == "16" else "new"}})
    state = {"conditions": {
        k: {"shards": {"0": {"run": f"runs/k{k}"}}} for k in ("4", "2")}}
    rows = [{"input_key": sid, "label": gold, "image": sid, "split": split}
            for split in ("random", "popular", "adversarial")
            for sid, gold in (("a", "yes"), ("b", "no")) for _ in range(90)]
    predictions = {"16": ("missing", "no"), "4": ("yes", "no"), "2": ("yes", "yes")}
    records = {
        k: {sid: dict(request_sha256=answer, gold=gold, strict_prediction=None,
                      iterations=2, seconds=1., commits=3, output_tokens=1,
                      tail_only_iterations=0, empty_shell=answer == "missing",
                      finish_reason="eos_prefix_complete")
            for sid, gold, answer in zip(("a", "b"), ("yes", "no"), answers)}
        for k, answers in predictions.items()}
    monkeypatch.setattr(report, "collect", lambda state: (
        rows, records, {answer: answer for answer in ("missing", "yes", "no")}))
    monkeypatch.setattr(report, "judge_key", lambda: "fake")
    monkeypatch.setattr(report, "judge_request", lambda request, cache, key: {
        "status": "success", "parsed": {
            "answer": None if request == "missing" else request,
            "status": "missing" if request == "missing" else "answered",
            "evidence": "" if request == "missing" else request}})
    report.evaluate(state)
    result = json.loads((out / "results.json").read_text())
    assert {k: r["judge_accuracy"] for k, r in result["methods"].items()} == {
        "16": .5, "4": 1., "2": .5}
    assert result["methods"]["16"]["judge_f1"] is None
    assert result["methods"]["16"]["judge_missing_rate"] == .5
    assert result["paired_judge_accuracy"]["4_minus_16"]["delta"] == .5
    assert result["paired_judge_accuracy"]["2_minus_16"]["delta"] == 0
    assert result["generation_environments"]["16"][0]["environment"]["platform"] == "old"
    assert result["generation_environments"]["4"][0]["environment"]["platform"] == "new"
    assert "—" in (out / "results.md").read_text()


def test_cancelled_generation_never_launches_workers(monkeypatch):
    from scripts import run_chair_pope_extensions as queue
    monkeypatch.setattr(queue, "stage_configs", lambda *a, **kw: pytest.fail("Cancelled stage must not launch"))
    index = {"stages": {"pope_aokvqa_reasoning_r50": {"status": "cancelled", "shards": {}}}}
    queue.run_stage(index, "pope_aokvqa", 50, True)
    assert index["stages"]["pope_aokvqa_reasoning_r50"]["status"] == "cancelled"


def test_cancelled_judge_stage_is_terminal_without_scores(tmp_path, monkeypatch):
    from scripts import run_pope_think_judge as runner
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "OUTPUT", tmp_path / "reports")
    monkeypatch.setattr(runner, "STAGES", ("cancelled", "pending"))
    index = {"status": "running", "active": "pending", "stages": {
        "cancelled": {"status": "cancelled", "cancellation_reason": "user_requested_stop_r50"},
        "pending": {"status": "generating"}}}
    summary = runner.reports({}, {}, index, tmp_path / "cache", "waiting_for_generation", 1)
    assert summary["stages"]["cancelled"] == {
        "status": "cancelled", "generation_status": "cancelled",
        "cancellation_reason": "user_requested_stop_r50"}
    assert summary["stages"]["pending"]["status"] == "waiting_for_generation"


def test_budget_limited_pope_report_keeps_all_rows_and_costs(tmp_path, monkeypatch):
    from cdapc_lab.experiments import pope_report as report
    from cdapc_lab.io.artifacts import atomic_json
    monkeypatch.setattr(report, "ROOT", tmp_path)
    monkeypatch.setattr(report, "load_pope_official_rule", lambda _: lambda text: "yes")
    run = tmp_path / "run"
    atomic_json(run / "resolved_config.json", {
        "protocol": {"task": "pope", "data_root": "data",
                     "prompt_id": "pope_yesno_reasoning_v1", "strict_parser_id": "pope_final_yes_no_v1"},
        "cdapc": {"budget_limited": True},
    })
    data = [{"input_key": sid, "row_id": f"{split}:{sid}", "split": split,
             "label": gold, "image_sha256": "image"}
            for split in report.SPLITS for sid, gold in (("ok", "yes"), ("budget", "no"))]
    atomic_json(tmp_path / "data/rows.json", data)
    scores = [{"sample_id": sid, "generation_status": status,
               "normalized": {"text": text}}
              for sid, status, text in (("ok", "success", "Final answer: Yes"),
                                        ("budget", "budget_exhausted", "Final answer: No"))]
    monkeypatch.setattr(report, "select_evaluation",
                        lambda *args: ({"evaluation_fingerprint": "fixture"}, scores))
    for sid, status, seconds in (("ok", "success", 1), ("budget", "budget_exhausted", 4)):
        atomic_json(run / f"predictions/{sid}.json",
                    {"sample_id": sid, "status": status, "generation_seconds": seconds,
                     "decode_iterations": 2 if sid == "ok" else 128, "finish_reason": status})
    result = report.pope_report([run])
    assert not result["partial"]
    assert result["cdapc"]["inputs"] == 2 and result["cdapc"]["rows"] == 6
    assert all(s["n"] == 2 and s["accuracy"] == .5 for s in result["cdapc"]["splits"].values())
    assert result["cdapc"]["cost"]["count"] == 2
    assert result["cdapc"]["cost"]["generation_seconds"] == 5
    assert result["cdapc"]["generation_status_counts"] == {"success": 1, "budget_exhausted": 1}
    assert result["cdapc"]["mechanism_totals"]["decode_iterations"] == 130
    scores[1]["generation_status"] = "error"
    with pytest.raises(ValueError, match="cover"):
        report.pope_report([run])


def test_budget_exhausted_judge_is_zero_without_api_and_preserves_9000_rows(tmp_path, monkeypatch):
    from scripts import run_pope_think_judge as runner
    from cdapc_lab.io.artifacts import atomic_json
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "OUTPUT", tmp_path / "judged")
    monkeypatch.setattr(runner, "STAGES", ("fixture",))
    run = tmp_path / "runs/run"
    atomic_json(run / "resolved_config.json", {
        "protocol": {"prompt_id": "pope_yesno_reasoning_v1", "data_root": "data"}})
    data = [{"input_key": sid, "question": f"Question {sid}?", "label": gold, "split": split}
            for split in ("random", "popular", "adversarial") for _ in range(1500)
            for sid, gold in (("ok", "yes"), ("budget", "no"))]
    atomic_json(tmp_path / "data/rows.json", data)
    for sid, gold, status in (("ok", "yes", "success"), ("budget", "no", "budget_exhausted")):
        atomic_json(run / f"scores/eval/{sid}.json", {
            "sample_id": sid, "generation_status": status, "gold": gold,
            "normalized": {"text": "Final answer: " + gold}, "parsed": {"prediction": gold}})
    atomic_json(tmp_path / "report.json", {
        "runs": ["runs/run"], "cdapc": {"per_run": [{"run": "runs/run", "evaluation_fingerprint": "eval"}]}})
    path = tmp_path / "new_queue.json"
    atomic_json(path, {"status": "complete", "active": None,
                       "stages": {"fixture": {"status": "complete", "report": "report.json"}}})
    index, ready, requests = runner.collect(path)
    assert len(requests) == 1
    assert ready["fixture"]["records"]["budget"]["request_sha256"] is None
    results = {sha: {"status": "success",
                     "parsed": {"answer": "yes", "status": "answered", "evidence": "yes"}}
               for sha in requests}
    summary = runner.reports(ready, results, index, tmp_path / "cache", "judging", 1)
    stage = summary["stages"]["fixture"]
    assert stage["status"] == "complete" and stage["judged_unique"] == 2
    assert stage["api_judged_unique"] == 1 and stage["budget_exhausted_inputs"] == 1
    assert stage["api_error_inputs"] == 0
    assert all(s["n"] == 3000 and s["accuracy"] == .5 and s["budget_exhausted"] == 1500
               for s in stage["splits"].values())
    assert stage["paired_transitions_vs_strict"] == {"strict_1_judge_1": 4500, "strict_0_judge_0": 4500}


def test_full_pope_k4_configs_cover_three_datasets_at_256_128():
    from scripts.run_pope_k4_256_128 import DATASETS, stage_name
    from cdapc_lab.io.artifacts import read_json
    expected = {"coco": 5127, "aokvqa": 5145, "gqa": 5171}
    assert DATASETS == ("coco", "aokvqa", "gqa")
    for dataset in DATASETS:
        seen = set()
        for shard in range(3):
            cfg = resolve_config(ROOT / f"configs/experiments/mmada_{stage_name(dataset)}_shard{shard}.yaml")
            assert cfg["protocol"]["canvas_length"] == 256
            assert cfg["protocol"]["prompt_id"] == "pope_yesno_reasoning_v1"
            assert cfg["protocol"]["strict_parser_id"] == "pope_final_yes_no_v1"
            assert cfg["cdapc"]["max_commit_per_iteration"] == 4
            assert cfg["cdapc"]["iteration_limit"] == 128 and cfg["cdapc"]["budget_limited"]
            assert cfg["cdapc"]["window"] == {"kind": "fixed", "width": 16}
            assert cfg["cdapc"]["r_gate"]["threshold"] == .7
            ids = read_json(ROOT / cfg["run"]["sample_ids_file"])
            assert not seen.intersection(ids)
            seen.update(ids)
        assert len(seen) == expected[dataset]
