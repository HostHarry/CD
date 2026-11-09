"""Score CHAIR using the pinned evaluator and verified exported COCO object sets."""
import argparse
from contextlib import redirect_stdout
import gzip
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
import types

from cdapc_lab.evaluation.normalize import normalize_output
from cdapc_lab.io.artifacts import atomic_json, file_hash
from prepare_eval_extensions import ROOT, EXPORT, CHAIR_HASH, CHAIR_REV


def load_evaluator():
    path = ROOT / "upstream/chair/chair.py"
    assert file_hash(path) == CHAIR_HASH
    inflect_path = importlib.metadata.distribution("patternfork-nosql").locate_file("pattern/text/en/inflect.py")
    spec = importlib.util.spec_from_file_location("chair_inflect", inflect_path)
    inflect = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inflect)
    bridge = types.ModuleType("pattern.en")
    bridge.singularize = inflect.singularize
    previous = sys.modules.get("pattern.en")
    sys.modules["pattern.en"] = bridge
    try:
        spec = importlib.util.spec_from_file_location("pinned_chair", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            del sys.modules["pattern.en"]
        else:
            sys.modules["pattern.en"] = previous
    reference = json.loads((EXPORT / "chair/run/native_chair.json").read_text())
    other = json.loads((EXPORT / "chair/run/degap_token_chair.json").read_text())
    objects = {row["image_id"]: set(row["mscoco_gt_words"]) for row in reference["sentences"]}
    assert len(objects) == 500
    assert objects == {row["image_id"]: set(row["mscoco_gt_words"]) for row in other["sentences"]}

    class CachedChair(module.CHAIR):
        def get_annotations(self):
            self.imid_to_objects = {key: set(values) for key, values in objects.items()}

    evaluator = CachedChair("exported COCO ground-truth object union")
    reproduced = evaluator.compute_chair(str(EXPORT / "chair/run/native_answers.json"), "image_id", "text")
    assert reproduced["overall_metrics"] == reference["overall_metrics"]
    expected = {r["image_id"]: r for r in reference["sentences"]}
    for row in reproduced["sentences"]:
        original = expected[row["image_id"]]
        assert row["metrics"] == original["metrics"]
        assert row["mscoco_generated_words"] == original["mscoco_generated_words"]
        assert json.loads(json.dumps(row["mscoco_hallucinated_words"])) == original["mscoco_hallucinated_words"]
    provenance = {
        "evaluator_revision": CHAIR_REV, "evaluator_sha256": CHAIR_HASH,
        "ground_truth_source": "complete mscoco_gt_words object sets from the exported native CHAIR evaluation",
        "ground_truth_cache_sha256": file_hash(EXPORT / "chair/run/native_chair.json"),
        "original_annotation_provenance": reference["provenance"],
        "native_500_per_image_and_aggregate_reproduction": True,
        "pattern_inflect_sha256": file_hash(inflect_path),
        "nltk_version": importlib.metadata.version("nltk"),
    }
    assert provenance["pattern_inflect_sha256"] == reference["provenance"]["pattern_inflect_sha256"]
    return evaluator, reference["overall_metrics"], provenance


def report(runs, output):
    output = Path(output)
    with (ROOT / "logs/chair_evaluator.log").open("a") as log, redirect_stdout(log):
        evaluator, native, provenance = load_evaluator()
    if not runs:
        result = {"native_reproduced": native, "provenance": provenance}
        atomic_json(output, result)
        return result
    from transformers import AutoTokenizer
    configs = [json.loads((Path(run) / "resolved_config.json").read_text()) for run in runs]
    assert all(c["protocol"]["task"] == "chair" and c["protocol"] == configs[0]["protocol"] for c in configs)
    assert all(c["cdapc"] == configs[0]["cdapc"] for c in configs)
    tokenizer = AutoTokenizer.from_pretrained(configs[0]["model"]["checkpoint"], local_files_only=True)
    expected = json.loads((ROOT / "manifests/chair_all_ids.json").read_text())
    reference_prompts = {}
    for path in (EXPORT / "chair/native").glob("predictions-*.jsonl.gz"):
        with gzip.open(path, "rt") as stream:
            for line in stream:
                record = json.loads(line)["payload"]
                reference_prompts[record["input"]["image_sha256"]] = record["input"]["prompt_ids"]
    images = {r["id"]: r for r in json.loads((ROOT / "data/chair/rows.json").read_text())}
    answers, ids, seconds, steps, forwards, same_prompts = [], set(), 0.0, 0, 0, 0
    for run in map(Path, runs):
        assert json.loads((run / "run_manifest.json").read_text())["status"] == "complete"
        for path in (run / "predictions").glob("*.json"):
            pred = json.loads(path.read_text())
            assert pred["status"] == "success" and pred["sample_id"] not in ids
            sid = pred["sample_id"]
            ids.add(sid)
            norm = normalize_output(pred["raw_response_ids"], tokenizer, pred["eos_ids"])
            answers.append({"sample_id": sid, "image_id": images[sid]["image_id"], "text": norm["text"],
                            "answer_token_length": norm["emitted_tokens_before_eos"],
                            "decode_iterations": pred["decode_iterations"], "finish_reason": pred["finish_reason"]})
            prompt = pred["prompt_ids"]
            prompt = prompt[0] if prompt and isinstance(prompt[0], list) else prompt
            old = reference_prompts[images[sid]["image_sha256"]]
            old = old[0] if old and isinstance(old[0], list) else old
            same_prompts += prompt == old
            seconds += pred["generation_seconds"]
            steps += pred["decode_iterations"]
            forwards += pred["cost"]["denoiser_branch_evaluations"]
    assert ids == set(expected) and same_prompts == len(expected)
    answers.sort(key=lambda row: row["image_id"])
    answer_path = output.with_name(output.stem + "_answers.json")
    atomic_json(answer_path, answers)
    with (ROOT / "logs/chair_evaluator.log").open("a") as log, redirect_stdout(log):
        result = evaluator.compute_chair(str(answer_path), "image_id", "text")
    result.update(provenance=provenance, runs=list(map(str, runs)), native=native,
                  config=configs[0]["protocol"], cdapc=configs[0]["cdapc"],
                  coverage={"images": len(ids), "prompt_ids_equal_to_native": same_prompts},
                  cost={"mean_iterations": steps / len(ids), "mean_seconds": seconds / len(ids),
                        "mean_branch_evaluations": forwards / len(ids)},
                  mean_answer_tokens=sum(r["answer_token_length"] for r in answers) / len(ids),
                  empty_answers=sum(not r["text"] for r in answers))
    atomic_json(output, result)
    return {k: v for k, v in result.items() if k != "sentences"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(report(args.runs, args.output), indent=2))
