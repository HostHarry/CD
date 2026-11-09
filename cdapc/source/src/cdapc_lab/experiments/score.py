from pathlib import Path
from dataclasses import asdict
from cdapc_lab.io.artifacts import read_json, atomic_json, digest, file_hash, source_hash, safe_id
from cdapc_lab.data.records import load_gold
from cdapc_lab.evaluation.normalize import normalize_output
from cdapc_lab.evaluation.scorers import load_official_scorer, load_pope_official_rule, score_text, score_pope_text, load_mmmu_official, mmmu_random_for, _StrictRandom

ROOT = Path(__file__).resolve().parents[3]


def score_run(run, profile="official_and_strict"):
    if profile != "official_and_strict":
        raise ValueError("Unknown scoring profile")
    from transformers import AutoTokenizer
    run = Path(run)
    config = read_json(run / "resolved_config.json")
    assets = read_json(run / "run_manifest.json")["assets"]["model_files"]
    tokenizer_assets = {name: sha for name, sha in assets.items() if not name.endswith(".safetensors")}
    if not tokenizer_assets:
        raise ValueError("No frozen tokenizer/config assets in generation provenance")
    for name, expected in tokenizer_assets.items():
        if file_hash(Path(config["model"]["checkpoint"]) / name) != expected:
            raise ValueError(f"Scoring tokenizer/config changed: {name}")
    manifest = read_json(run / "dataset_manifest.json")
    ids = [s["sample_id"] for s in manifest["samples"]]
    gold = load_gold(config["protocol"]["data_root"], ids)
    predictions = {}
    for sid in ids:
        path = run / "predictions" / (safe_id(sid) + ".json")
        predictions[sid] = read_json(path) if path.is_file() else {"status": "missing", "sample_id": sid}
        if predictions[sid]["sample_id"] != sid:
            raise ValueError("Prediction ID mismatch")
    task = config["protocol"].get("task", "m3cot")
    official_paths = {"pope": (ROOT / "upstream/pope/evaluate.py",),
                      "mmmu": (ROOT / "upstream/mmmu/eval/eval_utils.py", ROOT / "upstream/mmmu/eval/data_utils.py")}
    official_path = official_paths.get(task, (ROOT / "upstream/m3cot/utils/metric.py",))[0]
    source = source_hash(ROOT / "src/cdapc_lab", ["evaluation", "experiments/score.py"])
    spec = {"profile": profile, "evaluation_source_hash": source,
            "official_source_hash": [file_hash(p) for p in official_paths.get(task, (official_path,))] if task == "mmmu" else file_hash(official_path), "normalizer": config["protocol"]["normalizer_id"],
            "parser": config["protocol"]["strict_parser_id"], "primary_metric": config["protocol"]["primary_metric"],
            "gold": {sid: asdict(gold[sid]) for sid in ids},
            "allowed_options": {sid: predictions[sid].get("options") for sid in ids},
            "failure_policy": "unparsed/ambiguous/budget_exhausted/error/missing count incorrect; fixed manifest denominator",
            "scorer_rng": "official judge_answer is deterministic"}
    if task != "m3cot":
        spec["task"] = task
    if config["protocol"]["strict_parser_id"] == "pope_final_yes_no_v1":
        spec["answer_scope"] = "final region after closing think; raw official retained separately"
    spec_hash = digest(spec)
    fingerprint = digest({"predictions": predictions, "evaluation_spec_hash": spec_hash,
                          "tokenizer_assets": tokenizer_assets})
    output = run / "scores" / fingerprint
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["checkpoint"], local_files_only=True)
    official = None
    if task == "pope":
        official = load_pope_official_rule(official_path)
    elif task != "mmmu":
        official = load_official_scorer(official_path)
    mmmu_scorer = None
    if task == "mmmu":
        make = load_mmmu_official(*official_paths["mmmu"])
        mmmu_scorer = {"official": make, "strict": None}
    for sid in ids:
        prediction = predictions[sid]
        result = {"sample_id": sid, "group_id": next(x["group_id"] for x in manifest["samples"] if x["sample_id"] == sid),
                  "generation_status": prediction["status"], "strict_accuracy": 0, "official": 0}
        if task == "pope":
            result.update(official_prediction=None, gold=gold[sid].answer,
                          category=gold[sid].metric_metadata.get("category"))
        elif task == "mmmu":
            result.update(parsed_pred=None, gold=gold[sid].answer,
                          question_type=gold[sid].metric_metadata.get("question_type"),
                          strict_status="unparsed")
        if "raw_response_ids" in prediction:
            norm = normalize_output(prediction["raw_response_ids"], tokenizer, prediction["eos_ids"])
            if task == "pope":
                scores = score_pope_text(norm["text"], gold[sid].answer, official,
                                         config["protocol"]["strict_parser_id"])
                result["official_prediction"] = scores["official_prediction"]
                if "raw_official" in scores:
                    result.update(raw_official=scores["raw_official"],
                                  raw_official_prediction=scores["raw_official_prediction"])
            elif task == "mmmu":
                meta = gold[sid].metric_metadata
                official_score = mmmu_scorer["official"](mmmu_random_for(sid))(
                    norm["text"], prediction["options"], gold[sid].answer, meta["question_type"])
                strict_score = mmmu_scorer["official"](_StrictRandom())(
                    norm["text"], prediction["options"], gold[sid].answer, meta["question_type"])
                result.update(parsed_pred=official_score["parsed_pred"],
                              strict_status="unparsed" if strict_score["unparsed"] else "parsed",
                              strict_prediction=strict_score["parsed_pred"])
                if prediction["status"] == "success":
                    result.update(official=official_score["correct"], strict_accuracy=int(strict_score["correct"]))
            else:
                scores = score_text(norm["text"], prediction["options"], gold[sid].answer, official)
            result.update(normalized=norm)
            if task != "mmmu":
                result["parsed"] = scores["parsed"]
            if prediction["status"] == "success" and task != "mmmu":
                result.update(strict_accuracy=scores["strict_accuracy"], official=scores["official"])
        atomic_json(output / (safe_id(sid) + ".json"), result)
    atomic_json(output / "evaluation_manifest.json", {"evaluation_fingerprint": fingerprint,
                "evaluation_spec_hash": spec_hash, "spec": spec, "sample_ids": ids,
                "tokenizer_assets": tokenizer_assets})
    return {"evaluation_fingerprint": fingerprint, "evaluation_spec_hash": spec_hash}


def select_evaluation(run, evaluation=None):
    root = Path(run) / "scores"
    if evaluation is None:
        choices = sorted(p.name for p in root.iterdir() if (p / "evaluation_manifest.json").is_file()) if root.exists() else []
        if len(choices) != 1:
            raise ValueError("Select an explicit evaluation fingerprint (zero or multiple evaluations)")
        evaluation = choices[0]
    if len(evaluation) != 64 or any(c not in "0123456789abcdef" for c in evaluation):
        raise ValueError("Expected full evaluation fingerprint")
    output = root / evaluation
    meta = read_json(output / "evaluation_manifest.json")
    return meta, [read_json(output / (safe_id(sid) + ".json")) for sid in meta["sample_ids"]]


def pope_metrics(pairs):
    """pairs: iterable of (prediction|'yes'|'no'|None, gold 'yes'|'no')."""
    pairs = list(pairs)
    tp = fp = tn = fn = unresolved = 0
    n = 0
    for pred, gold in pairs:
        n += 1
        if pred == "yes" and gold == "yes":
            tp += 1
        elif pred == "yes" and gold == "no":
            fp += 1
        elif pred == "no" and gold == "no":
            tn += 1
        elif pred == "no" and gold == "yes":
            fn += 1
        elif gold == "yes":
            fn += 1
        else:
            unresolved += 1
    safe = lambda a, b: a / b if b else None
    precision, recall = safe(tp, tp + fp), safe(tp, tp + fn)
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "unresolved": unresolved,
            "accuracy": safe(tp + tn, n), "precision": precision, "recall": recall,
            "f1": safe(2 * precision * recall, precision + recall) if precision is not None and recall is not None else None,
            "yes_ratio": safe(sum(1 for p, _ in pairs if p == "yes"), n)}


def summarize(run, evaluation=None):
    run = Path(run)
    meta, rows = select_evaluation(run, evaluation)
    total = len(rows)
    predictions = [read_json(run / "predictions" / (safe_id(r["sample_id"]) + ".json"))
                   for r in rows if (run / "predictions" / (safe_id(r["sample_id"]) + ".json")).is_file()]
    successful = [p for p in predictions if p["status"] == "success"]
    result = {"evaluation_fingerprint": meta["evaluation_fingerprint"], "evaluation_spec_hash": meta["evaluation_spec_hash"],
              "sample_count": total, "success_count": len(successful), "failures": [r["sample_id"] for r in rows if r["generation_status"] != "success"],
              "metrics": {m: {"numerator": sum(r[m] for r in rows), "denominator": total,
                              "value": sum(r[m] for r in rows) / total} for m in ("official", "strict_accuracy")},
              "parse_statuses": {s: sum(r.get("parsed", {}).get("status", "not_generated") == s for r in rows)
                                 for s in ("parsed", "unparsed", "ambiguous", "invalid", "not_generated")}}
    result["cost"] = {"denominator": len(successful), "scope": "successful generated samples; sum per-sample durations across resume", **{
        name: sum(p.get(name, 0) for p in successful) for name in ("generation_seconds", "preprocess_seconds", "decode_iterations", "emitted_tokens_before_eos")}}
    result["cost"]["branch_evaluations"] = sum(p["cost"]["denoiser_branch_evaluations"] for p in successful)
    result["cost"]["model_forward_calls"] = sum(p["cost"]["model_forward_calls"] for p in successful)
    metric_pairs = {"proposal_flip_rate": ("proposal_flips", "observed"), "qualified_flip_rate": ("qualified_flips", "eligible"),
                    "actual_committed_switch_rate": ("actual_switches", "committed")}
    result["mechanism"] = {}
    for name, (num, den) in metric_pairs.items():
        n, d = (sum(p.get("metrics", {}).get(k, 0) for p in successful) for k in (num, den))
        result["mechanism"][name] = {"numerator": n, "denominator": d, "scope": "full canvas", "value": n / d if d else None}
    result["mechanism"]["sample_switch_coverage"] = {"numerator": sum(p.get("metrics", {}).get("actual_switches", 0) > 0 for p in successful), "denominator": len(successful)}
    result["mechanism"]["fallback_rate"] = {"numerator": sum(p.get("metrics", {}).get("fallback_iterations", 0) for p in successful), "denominator": sum(p.get("decode_iterations", 0) for p in successful)}
    if read_json(run / "resolved_config.json")["protocol"].get("task", "m3cot") == "pope":
        result["pope"] = {
            "official": pope_metrics((r.get("official_prediction") if r["generation_status"] == "success" else None, r["gold"]) for r in rows),
            "strict": pope_metrics((r.get("parsed", {}).get("prediction") if r["generation_status"] == "success" and r.get("parsed", {}).get("status") == "parsed" else None, r["gold"]) for r in rows)}
    atomic_json(run / "summary" / (meta["evaluation_fingerprint"] + ".json"), result)
    return result
