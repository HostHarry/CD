"""Pinned official MMMU scoring plus an explicit no-random-guess diagnostic."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sqlite3
import statistics
import types

from .mmmu_protocol import DATA
from .run import input_key
from .runtime import ROOT, save_json, sha256

SCORER_REVISION = "51ce7f3e829c16bb44bc5445782686b4c3508794"
SCORER_SHA = "cc1a89b4a27f697702d340bc6a7578b5237135754ddaa9ef105239a3d9847367"


class ChoiceRNG:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.used = False

    def choice(self, options):
        self.used = True
        return self.rng.choice(options)


class OfficialScorer:
    def __init__(self):
        path = ROOT / "data/evaluators/mmmu/eval_utils.py"
        if sha256(path) != SCORER_SHA:
            raise ValueError("official scorer source changed")
        spec = importlib.util.spec_from_file_location("_mmmu_official_eval", path)
        self.module = importlib.util.module_from_spec(spec)
        state = random.getstate()
        try:
            spec.loader.exec_module(self.module)
        finally:
            random.setstate(state)

    def score(self, sample, answer, text, seed=20261105):
        original_answer = answer
        if isinstance(answer, str) and answer.startswith("["):
            aliases = ast.literal_eval(answer)
            if not isinstance(aliases, list) or not aliases or not all(isinstance(v, str) for v in aliases):
                raise ValueError("unexpected MMMU answer aliases")
            answer = aliases
        fallback_seed = int.from_bytes(hashlib.sha256(
            f"mmmu-official-choice-v1:{seed}:{sample['id']}".encode()).digest()[:4], "big")
        rng = ChoiceRNG(fallback_seed)
        if sample["question_type"] == "multiple-choice":
            original = self.module.parse_multi_choice_response
            parser = types.FunctionType(original.__code__, {**original.__globals__, "random": rng})
            prediction = parser(text, list(sample["options"]), sample["options"])
            correct = self.module.eval_multi_choice(answer, prediction)
        else:
            prediction = sorted(self.module.parse_open_response(text), key=repr)
            correct = self.module.eval_open(answer, prediction)
        return {
            "official_prediction": prediction, "official_correct": bool(correct),
            "no_guess_prediction": None if rng.used else prediction,
            "no_guess_correct": bool(correct and not rng.used),
            "used_random_guess": rng.used, "fallback_seed": fallback_seed,
            "gold": answer, "gold_aliases_decoded": answer != original_answer,
        }


def aggregate(rows):
    return {
        "rows": len(rows),
        "official_accuracy": statistics.mean(r["official_correct"] for r in rows) if rows else None,
        "no_guess_accuracy": statistics.mean(r["no_guess_correct"] for r in rows) if rows else None,
        "random_guesses": sum(r["used_random_guess"] for r in rows),
    }


def score_runs(output, allow_partial=False):
    from verify_eos import audit_record

    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((DATA / "manifest.json").read_text())
    if sha256(DATA / "samples.jsonl") != manifest["samples_sha256"]:
        raise ValueError("prepared samples changed")
    if sha256(DATA / "gold.jsonl") != manifest["gold_sha256"]:
        raise ValueError("gold changed")
    samples = [json.loads(line) for line in (DATA / "samples.jsonl").read_text().splitlines()]
    gold = {g["id"]: g for g in map(json.loads, (DATA / "gold.jsonl").read_text().splitlines())}
    if set(gold) != {s["id"] for s in samples}:
        raise ValueError("gold/sample IDs differ")
    model_config = json.loads((Path("/root/autodl-fs/models/MMaDA-8B-MixCoT") / "config.json").read_text())
    scorer, summaries, scored_by_method = OfficialScorer(), {}, {}
    for method in ("vcd", "vigor"):
        payloads = {}
        for shard in range(2):
            run_dir = ROOT / method / "results/mmmu_wino_v1" / f"shard{shard}"
            database_path = run_dir / "predictions.sqlite"
            if not database_path.exists():
                if allow_partial:
                    continue
                raise ValueError(f"missing prediction database: {database_path}")
            run_manifest = json.loads((run_dir / "manifest.json").read_text())
            if run_manifest["samples_sha256"] != manifest["shards"][shard]["sha256"]:
                raise ValueError("run samples differ from the prepared shard")
            with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as db:
                if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    raise ValueError("corrupted prediction database")
                for name, key, text in db.execute("SELECT method,input_key,payload FROM predictions"):
                    if name != method or key in payloads:
                        raise ValueError("unexpected method or duplicate prediction")
                    payloads[key] = json.loads(text)
        expected = {input_key(s) for s in samples}
        if not set(payloads) <= expected or (not allow_partial and set(payloads) != expected):
            raise ValueError("prediction coverage differs from the 900-sample manifest")
        rows = []
        for sample in samples:
            result = payloads.get(input_key(sample))
            if result is None:
                continue
            if result["input"]["image_sha256"] != sample["image_sha256"]:
                raise ValueError("prediction image differs from source")
            if len(result["input"]["prompt_ids"]) != sample["expected_prompt_tokens"]:
                raise ValueError("prediction prompt length differs from the validated input")
            steps = result["executed_steps"]
            if steps != len(result["trace"]) or result["remaining_masks"]:
                raise ValueError("invalid trajectory length or incomplete answer")
            first_closed = audit_record(result, model_config["mask_token_id"], model_config["eos_token_id"])
            if first_closed is not None and first_closed != steps:
                raise ValueError("safe EOS did not stop at the first complete prefix")
            rows.append({
                "id": sample["id"], "subject": sample["subject"],
                "question_type": sample["question_type"], "text": result["text"],
                "normal_prompt_sha256": hashlib.sha256(json.dumps(result["input"]["prompt_ids"]).encode()).hexdigest(),
                **scorer.score(sample, gold[sample["id"]]["answer"], result["text"]),
                "answer_token_length": result["answer_token_length"],
                "executed_steps": steps, "denoiser_forwards": 2 * steps,
                "vision_encodings": 2 if method == "vcd" else 1,
                "end_to_end_seconds": result["end_to_end_seconds"],
                "decode_seconds": result["decode_seconds"],
                "peak_allocated_bytes": result["peak_allocated_bytes"],
                "eos_found": result["eos_found"], "early_stopped": result["early_stopped"],
                "image_tokens_in_output": result["image_token_count"],
            })
        summary = {
            **aggregate(rows), "expected_rows": 900, "complete": len(rows) == 900,
            "by_question_type": {kind: aggregate([r for r in rows if r["question_type"] == kind])
                                 for kind in ("multiple-choice", "open")},
            "by_subject": {subject: aggregate([r for r in rows if r["subject"] == subject])
                           for subject in sorted({r["subject"] for r in rows})},
            "eos_found": sum(r["eos_found"] for r in rows),
            "early_stopped": sum(r["early_stopped"] for r in rows),
            "image_tokens_in_output": sum(r["image_tokens_in_output"] for r in rows),
        }
        if rows:
            for field in ("answer_token_length", "executed_steps", "denoiser_forwards",
                          "vision_encodings", "end_to_end_seconds", "decode_seconds"):
                summary[field] = {"mean": statistics.mean(r[field] for r in rows),
                                  "total": sum(r[field] for r in rows)}
            summary["peak_allocated_bytes"] = max(r["peak_allocated_bytes"] for r in rows)
        save_json(output / f"{method}_scores.json", rows)
        summaries[method] = summary
        scored_by_method[method] = {r["id"]: r for r in rows}
    common = set(scored_by_method["vcd"]) & set(scored_by_method["vigor"])
    for sample_id in common:
        if (scored_by_method["vcd"][sample_id]["normal_prompt_sha256"] !=
                scored_by_method["vigor"][sample_id]["normal_prompt_sha256"]):
            raise ValueError("VCD/VIGOR normal prompt IDs differ")
    paired = {"rows": len(common)}
    for metric in ("official_correct", "no_guess_correct"):
        paired[metric] = {
            "vcd_only_correct": sum(scored_by_method["vcd"][i][metric] and
                                    not scored_by_method["vigor"][i][metric] for i in common),
            "vigor_only_correct": sum(scored_by_method["vigor"][i][metric] and
                                      not scored_by_method["vcd"][i][metric] for i in common),
        }
    report = {
        "methods": summaries, "paired": paired,
        "scorer_revision": SCORER_REVISION, "scorer_sha256": SCORER_SHA,
        "wrapper_sha256": sha256(__file__), "data_manifest_sha256": sha256(DATA / "manifest.json"),
        "normalization": "first EOS; tokenizer decode with skip_special_tokens; strip surrounding whitespace",
        "official_track": "unchanged official functions on the entire normalized response, including think",
        "fallback_seed_protocol": "first 32 bits SHA256(mmmu-official-choice-v1:20261105:sample_id)",
        "diagnostic_track": "same official rules with random fallback counted incorrect; not the unpublished reference strict wrapper",
        "gold_protocol": "decode the three stringified answer-alias lists before official eval_open",
        "comparison_note": "Local rule scores are not directly comparable to the reference report's LLM-judge accuracy.",
    }
    save_json(output / "metrics.json", report)
    table = ["# MMMU VCD / VIGOR", "", "Protocol: mmmu_wino_reasoning_v1, 256/128/64, seed 20261105.",
             "Reference single-image policy; two fixed 450-row shards.", "",
             "| Method | Scored | Official accuracy | No-guess accuracy | Random fallbacks | Mean answer tokens | Mean steps |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for method, value in summaries.items():
        if value["rows"]:
            table.append(f"| {method} | {value['rows']}/900 | {value['official_accuracy']:.2%} | "
                         f"{value['no_guess_accuracy']:.2%} | {value['random_guesses']} | "
                         f"{value['answer_token_length']['mean']:.2f} | {value['executed_steps']['mean']:.2f} |")
    table += ["", report["comparison_note"], "", "Detailed scoring, subject breakdowns and compute costs: metrics.json.",
              "Raw IDs, full generation states, input IDs and commitment traces remain in the per-shard SQLite databases."]
    (output / "RESULTS.md").write_text("\n".join(table) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/mmmu_wino_v1")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    report = score_runs(args.output, args.allow_partial)
    print(json.dumps({m: aggregate(json.loads((args.output / f"{m}_scores.json").read_text()))
                      for m in report["methods"]}, ensure_ascii=False))
