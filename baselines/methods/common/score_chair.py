import argparse
from contextlib import redirect_stdout
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
import types

from .runtime import save_json, sha256


EVALUATOR_REVISION = "4087a26211aa2339b9a76307cb8f0321ef691d0a"
EVALUATOR_SHA256 = "98728cd2b7fb8b97420b5f3bf14789dcad1d7d980de9c089b82856217387770d"


def load_evaluator(path):
    inflect_path = importlib.metadata.distribution("patternfork-nosql").locate_file("pattern/text/en/inflect.py")
    inflect_spec = importlib.util.spec_from_file_location("baseline_pattern_inflect", inflect_path)
    inflect = importlib.util.module_from_spec(inflect_spec)
    inflect_spec.loader.exec_module(inflect)
    bridge = types.ModuleType("pattern.en")
    bridge.singularize = inflect.singularize
    previous = sys.modules.get("pattern.en")
    sys.modules["pattern.en"] = bridge
    try:
        spec = importlib.util.spec_from_file_location("baseline_chair_evaluator", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            del sys.modules["pattern.en"]
        else:
            sys.modules["pattern.en"] = previous
    return module, inflect_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--nltk-data", type=Path)
    args = parser.parse_args()
    if sha256(args.evaluator) != EVALUATOR_SHA256:
        raise ValueError("CHAIR evaluator differs from the pinned revision")
    import nltk
    if args.nltk_data:
        nltk.data.path.insert(0, str(args.nltk_data.resolve()))
    nltk.word_tokenize("Verify tokenizer data.")
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    status = json.loads((args.run_dir / "status.json").read_text())
    if not status["selected_run_complete"] or manifest["benchmark"] != "CHAIR":
        raise ValueError("requires a completed CHAIR generation run")
    answers = {name: json.loads((args.run_dir / f"{name}_answers.json").read_text())
               for name in manifest["methods"]}
    ids = {row["image_id"] for row in next(iter(answers.values()))}
    if not ids or any({row["image_id"] for row in rows} != ids or len(rows) != len(ids)
                      for rows in answers.values()):
        raise ValueError("methods must have identical unique image IDs")
    metadata = {
        "implementation": "Maxlinn/CHAIR-metric-standalone (Python3 derivative of LisaAnne/Hallucination)",
        "reference_revision": EVALUATOR_REVISION, "evaluator_sha256": sha256(args.evaluator),
        "annotation_sha256": {
            name: sha256(args.annotations / name)
            for name in ("instances_train2014.json", "instances_val2014.json",
                         "captions_train2014.json", "captions_val2014.json")
        },
        "nltk": importlib.metadata.version("nltk"),
        "patternfork-nosql": importlib.metadata.version("patternfork-nosql"),
        "generation_manifest_sha256": sha256(args.run_dir / "manifest.json"),
        "partial_dataset": not status["full_dataset_complete"],
        "filter": "unchanged scorer with annotation records restricted to evaluated image IDs",
    }
    module, inflect_path = load_evaluator(args.evaluator)
    metadata["pattern_inflect_sha256"] = sha256(inflect_path)
    metadata["pattern_import"] = "unchanged inflect.py singularize imported directly; unrelated WordNet side effects bypassed"
    for name in ("combine_coco_instances", "combine_coco_captions"):
        original = getattr(module, name)
        def restricted(path, original=original):
            data = original(path)
            data["annotations"] = [row for row in data["annotations"] if row["image_id"] in ids]
            data["images"] = [row for row in data["images"] if row["id"] in ids]
            return data
        setattr(module, name, restricted)
    with (args.run_dir / "chair_evaluator.log").open("a") as log, redirect_stdout(log):
        evaluator = module.CHAIR(str(args.annotations))
        if ids - evaluator.imid_to_objects.keys():
            raise ValueError("missing ground truth objects")
        metrics = {}
        for method, rows in answers.items():
            result = evaluator.compute_chair(str(args.run_dir / f"{method}_answers.json"), "image_id", "text")
            result["provenance"] = metadata
            result["answer_token_length_mean"] = sum(r["answer_token_length"] for r in rows) / len(rows)
            result["answer_sha256"] = sha256(args.run_dir / f"{method}_answers.json")
            save_json(args.run_dir / f"{method}_chair.json", result)
            metrics[method] = {**result["overall_metrics"],
                               "answer_token_length_mean": result["answer_token_length_mean"]}
    save_json(args.run_dir / "chair_metrics.json", {"provenance": metadata, "methods": metrics})
    status["chair_scored"] = True
    save_json(args.run_dir / "status.json", status)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
