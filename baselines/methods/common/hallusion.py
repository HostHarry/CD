"""HallusionBench preparation and pinned VLMEvalKit exact-matching scoring."""
import argparse
import ast
import base64
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import statistics
import sys

import numpy as np
import pandas as pd
from PIL import Image
from omegaconf import OmegaConf

from .run import input_key
from .runtime import ROOT, save_json, sha256

DATA = ROOT / "data/hallusion_full_original"
CONFIG = ROOT / "configs/hallusion_256_128.yaml"
RUN = ROOT / "results/hallusion_full_original_256_128"
METHODS = ("native", "vcd", "vigor")
SCORER = ROOT / "data/evaluators/hallusion"
REVISION = "3c192143007e4eb3f5c1a36558a4f8c1451980cb"
TSV_MD5 = "0c23ac0dc9ef46832d7a24504f2a0c7c"
SCORER_HASHES = {
    "yorn.py": "692d72ee0c8fe64c7dd93b466f1b47c432303d8fba57503c56aae770b67ef59e",
    "misc.py": "407ec5eaaf84227368365c6c7c93725166becfe755ed6a24eb93260d65464258",
}
PROMPT_ID = "hallusion_original_question_v1"


def write_locked(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"refusing changed artifact: {path}")
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)


def jsonl_bytes(rows):
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prepare(tsv):
    """Copy only benchmark data; runtime/method code stays in this project."""
    data = tsv.read_bytes()
    if hashlib.md5(data).hexdigest() != TSV_MD5:
        raise ValueError("HallusionBench TSV differs from the upstream MD5")
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "images").mkdir(exist_ok=True)
    write_locked(DATA / "HallusionBench.tsv", data)
    csv.field_size_limit(sys.maxsize)
    rows = list(csv.DictReader(io.StringIO(data.decode()), delimiter="\t"))
    if len(rows) != 1129 or len({r["index"] for r in rows}) != 1129:
        raise ValueError("unexpected source count or duplicate IDs")
    samples, no_image_ids = [], []
    for row in rows:
        path, image_sha = None, None
        if row["image"]:
            payload = base64.b64decode(row["image"], validate=True)
            image_sha = hashlib.sha256(payload).hexdigest()
            with Image.open(io.BytesIO(payload)) as image:
                extension = Image.registered_extensions()
                suffix = next(k for k in (".png", ".jpg", ".webp") if extension[k] == image.format)
                image.verify()
            path = DATA / "images" / (image_sha + suffix)
            write_locked(path, payload)
        else:
            no_image_ids.append(row["index"])
        if row["answer"] not in ("Yes", "No"):
            raise ValueError("unexpected HallusionBench gold answer")
        samples.append({
            "id": row["index"], "index": row["index"], "dataset": "HallusionBench",
            "question": row["question"], "original_question": row["question"],
            "answer": row["answer"], "category": row["category"],
            "l2-category": row["l2-category"],
            "image": str(path) if path is not None else None, "image_sha256": image_sha,
            "input_mode": "image" if path is not None else "text_only",
            "source_image_path": row["image_path"], "prompt_id": PROMPT_ID,
        })
    if len(samples) != 1129 or len(no_image_ids) != 178:
        raise ValueError("unexpected full dataset or text-only subset size")
    write_locked(DATA / "samples.jsonl", jsonl_bytes(samples))
    shards = []
    for shard in range(2):
        selected = samples[shard::2]
        path = DATA / f"shard_{shard}.jsonl"
        write_locked(path, jsonl_bytes(selected))
        shards.append({"rows": len(selected), "sha256": sha256(path)})
    longest = max((row for row in samples if row["image"]), key=lambda row: len(row["question"]))
    other = next(row for row in samples
                 if row["image"] and row["answer"] != longest["answer"]
                 and row["category"] != longest["category"])
    text_only = max((row for row in samples if row["image"] is None), key=lambda row: len(row["question"]))
    write_locked(DATA / "preflight.jsonl", jsonl_bytes([longest, other, text_only]))
    manifest = {
        "dataset_url": "https://opencompass.openxlab.space/utils/VLMEval/HallusionBench.tsv",
        "tsv_md5": TSV_MD5, "tsv_sha256": sha256(DATA / "HallusionBench.tsv"),
        "source_rows": len(rows), "selected_rows": len(samples),
        "no_image_ids": no_image_ids, "excluded_rows": 0, "selection": "all 1129 source rows",
        "text_only_policy": "pure chat template, no image delimiters; all methods use native decoding",
        "prompt_suffix": "", "prompt_id": PROMPT_ID,
        "prompt_note": "Unmodified TSV question in the MMaDA standalone chat template",
        "samples_sha256": sha256(DATA / "samples.jsonl"),
        "preflight_sha256": sha256(DATA / "preflight.jsonl"), "shards": shards,
        "category_counts": dict(Counter(row["category"] for row in samples)),
        "answer_counts": dict(Counter(row["answer"] for row in samples)),
        "preparer_sha256": sha256(__file__),
    }
    write_locked(DATA / "manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
    print(json.dumps({k: manifest[k] for k in ("source_rows", "selected_rows", "category_counts",
                                             "answer_counts")}), flush=True)


def official_functions():
    """Execute unchanged pure scorer functions, without importing judge clients."""
    namespace = {"np": np, "pd": pd, "defaultdict": defaultdict,
                 "load": lambda frame: frame.copy()}
    for filename, names in (
        ("misc.py", {"process_punctuation"}),
        ("yorn.py", {"YOrN_Extraction", "Hallusion_rating"}),
    ):
        path = SCORER / filename
        if sha256(path) != SCORER_HASHES[filename]:
            raise ValueError(f"pinned scorer changed: {path}")
        functions = [node for node in ast.parse(path.read_text()).body
                     if isinstance(node, ast.FunctionDef) and node.name in names]
        if {node.name for node in functions} != names:
            raise ValueError("missing pinned scorer functions")
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["YOrN_Extraction"], namespace["Hallusion_rating"]


def rate(rows):
    extract, rating = official_functions()
    scored = [{**row, "extracted": extract(row["prediction"])} for row in rows]
    for row in scored:
        row["score"] = row["extracted"] == row["answer"]
    frame = rating(pd.DataFrame(scored))
    frame = pd.concat([frame[frame["split"] == "Overall"],
                       frame[frame["split"] != "Overall"].sort_values("split")], ignore_index=True)
    overall = frame[frame["split"] == "Overall"].iloc[0]
    return scored, {
        "rows": len(scored), "correct": sum(row["score"] for row in scored),
        "accuracy_percent": float(overall["aAcc"]),
        "figure_accuracy_percent": float(overall["fAcc"]),
        "question_accuracy_percent": float(overall["qAcc"]),
        "avg_percent": float(overall["Avg"]),
        "unknown_answers": sum(row["extracted"] == "Unknown" for row in scored),
        "empty_answers": sum(not row["prediction"].strip() for row in scored),
        "breakdown": frame.to_dict(orient="records"),
    }


def load_predictions(run_dir, samples, sources):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    config = OmegaConf.to_container(OmegaConf.load(CONFIG), resolve=True)
    if (manifest["sources"] != sources or manifest["config"] != config
            or manifest["generation"] != config["generation"]
            or manifest.get("early_eos") != "safe_eos_v1"
            or manifest["samples_sha256"] != hashlib.sha256(jsonl_bytes(samples)).hexdigest()
            or manifest["selected_rows"] != len(samples) or manifest["limit"] != 0
            or set(manifest["methods"]) != set(METHODS)):
        raise ValueError("run manifest differs from the locked HallusionBench protocol")
    expected = {(method, input_key(sample)) for method in METHODS for sample in samples}
    with sqlite3.connect(f"file:{run_dir}/predictions.sqlite?mode=ro", uri=True) as db:
        if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("prediction database is corrupt")
        records = {(method, key): json.loads(payload) for method, key, payload in db.execute(
            "SELECT method,input_key,payload FROM predictions")}
    if set(records) != expected:
        raise ValueError("incomplete or unexpected prediction coverage")
    return records


def audit_predictions(samples, records):
    from transformers import AutoTokenizer
    from training.prompting_utils import UniversalPrompting
    from verify_eos import audit_record

    config = OmegaConf.load(CONFIG)
    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint, local_files_only=True)
    prompting = UniversalPrompting(tokenizer, use_reserved_token=True)
    mask, eos = tokenizer.convert_tokens_to_ids("<|mdm_mask|>"), tokenizer.eos_token_id
    special = prompting.sptids_dict
    for sample in samples:
        expected_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": sample["question"]}], tokenize=True, add_generation_prompt=True)
        inputs = []
        for method in METHODS:
            record = records[method, input_key(sample)]
            ids = record["input"]["prompt_ids"]
            if sample["input_mode"] == "text_only":
                if (ids != expected_text or record["input"]["visual_positions"]
                        or record.get("effective_method") != "native"
                        or record.get("requested_method") != method
                        or record.get("text_only_method_policy") != "native_without_visual_intervention"):
                    raise ValueError("text-only prompt or decoder policy mismatch")
            elif (ids[:2] != [int(special["<|mmu|>"]), int(special["<|soi|>"])]
                  or ids[1026] != int(special["<|eoi|>"]) or ids[1027:] != expected_text
                  or record["input"]["visual_positions"] != list(range(2, 1026))):
                raise ValueError("visual prompt differs from prepared sample")
            if record["input"]["image_sha256"] != sample["image_sha256"]:
                raise ValueError("prompt/image provenance differs from prepared sample")
            inputs.append(ids)
            steps = record["executed_steps"]
            if (record["planned_steps"] != 128 or record["planned_generated_positions"] != 256
                    or steps != len(record["trace"]) or not 0 < steps <= 128
                    or record["remaining_masks"] or record["image_token_count"]):
                raise ValueError("invalid generation schedule or output")
            first_closed = audit_record(record, mask, eos)
            if first_closed is not None and first_closed != steps:
                raise ValueError("generation did not stop at its first committed EOS prefix")
        if not inputs[0] == inputs[1] == inputs[2]:
            raise ValueError("normal input differs across methods")
        if sample["input_mode"] == "text_only":
            native = records["native", input_key(sample)]
            for method in ("vcd", "vigor"):
                actual = records[method, input_key(sample)]
                if any(native[field] != actual[field] for field in ("generated_ids", "text", "trace")):
                    raise ValueError("text-only fallback differs from native decoding")


def score_runs():
    protocol = json.loads((RUN / "manifest.json").read_text())
    data_manifest = json.loads((DATA / "manifest.json").read_text())
    if (sha256(DATA / "manifest.json") != protocol["data_manifest_sha256"]
            or sha256(DATA / "samples.jsonl") != data_manifest["samples_sha256"]):
        raise ValueError("prepared data changed")
    samples = read_rows(DATA / "samples.jsonl")
    all_records = {}
    for shard in range(2):
        path = DATA / f"shard_{shard}.jsonl"
        if sha256(path) != data_manifest["shards"][shard]["sha256"]:
            raise ValueError("prepared shard changed")
        selected = read_rows(path)
        records = load_predictions(RUN / f"shard{shard}", selected, protocol["sources"])
        for key, record in records.items():
            if key in all_records:
                # Repeated identical inputs may occur across deterministic shards.
                old = all_records[key]
                if any(old[field] != record[field] for field in ("input", "generated_ids", "text", "trace")):
                    raise ValueError("duplicate inputs produced different predictions")
            all_records[key] = record
    audit_predictions(samples, all_records)
    report = {
        "benchmark": "HallusionBench", "primary_metric": "aAcc (per-answer accuracy, percent)",
        "selected_rows": len(samples), "source_rows": data_manifest["source_rows"],
        "image_rows": len(samples) - len(data_manifest["no_image_ids"]),
        "text_only_rows": len(data_manifest["no_image_ids"]), "excluded_rows": 0,
        "protocol": protocol, "scoring": "VLMEvalKit exact_matching; Unknown counts as incorrect",
        "scorer_revision": REVISION, "scorer_sha256": SCORER_HASHES, "methods": {},
    }
    for method in METHODS:
        rows = []
        for sample in samples:
            record = all_records[method, input_key(sample)]
            rows.append({**sample, "prediction": record["text"],
                         "answer_token_length": record["answer_token_length"],
                         "executed_steps": record["executed_steps"],
                         "end_to_end_seconds": record["end_to_end_seconds"],
                         "early_stopped": record["early_stopped"]})
        scored, metrics = rate(rows)
        metrics.update(complete=len(rows) == 1129,
                       mean_answer_tokens=statistics.mean(r["answer_token_length"] for r in rows),
                       mean_executed_steps=statistics.mean(r["executed_steps"] for r in rows),
                       mean_end_to_end_seconds=statistics.mean(r["end_to_end_seconds"] for r in rows))
        metrics["by_input_mode"] = {
            mode: rate([row for row in rows if row["input_mode"] == mode])[1]
            for mode in ("image", "text_only")
        }
        save_json(RUN / f"{method}_scores.json", scored)
        report["methods"][method] = metrics
    save_json(RUN / "metrics.json", report)
    lines = ["# HallusionBench: 256/128/32", "",
             "Full dataset: 1129 rows, including 951 image and 178 text-only questions.",
             "Original TSV questions in the MMaDA chat template; safe EOS stopping.",
             "Text-only rows use identical native decoding for all three methods.",
             "Pinned VLMEvalKit exact matching; Unknown answers count as incorrect.", "",
             "| Method | Scored | Accuracy (aAcc %) | fAcc % | qAcc % | Unknown | Mean steps |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for method, m in report["methods"].items():
        lines.append(f"| {method} | {m['rows']}/1129 | {m['accuracy_percent']:.2f} | "
                     f"{m['figure_accuracy_percent']:.2f} | {m['question_accuracy_percent']:.2f} | "
                     f"{m['unknown_answers']} | {m['mean_executed_steps']:.2f} |")
    (RUN / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path, metavar="TSV")
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare)
    elif args.score:
        score_runs()
    else:
        parser.error("choose --prepare TSV or --score")
