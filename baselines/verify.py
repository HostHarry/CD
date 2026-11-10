#!/usr/bin/env python3
"""Verify baselines/ records without the original repository.

For every exported experiment under baselines/records/:
  1. every SHA256SUMS file (method dirs and baselines/methods/) is checked;
  2. records_index.json payload hashes are checked against the gzip JSONL
     prediction chunks (verbatim payload guarantee);
  3. POPE experiments are re-scored from the exported predictions with the
     unchanged official parser (common.metrics.pope_metrics) and compared
     per split, including the paired image-resampled bootstrap, to the
     stored metrics.json;
  4. CHAIR is re-aggregated from the exported chair.json sentences
     (CHAIRs / CHAIRi / Recall formulas verified against
     data/evaluators/chair.py) and compared to chair_metrics.json;
  5. the prompt diagnostic is re-scored from predictions + samples and
     compared to metrics.json (metrics, diagnostics, think stats, paired
     flips vs variant A);
  6. MMMU/HallusionBench get integrity + coverage checks (their scores come
     from external judges and are not recomputed here).

Requires numpy (paired bootstrap). Run from anywhere:
    python3 baselines/verify.py
"""
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import sys
from collections import Counter

import numpy as np

BASE = Path(__file__).resolve().parent
REPO = BASE.parent
RECORDS = BASE / "records"
INDEX_PATH = BASE / "records_index.json"

SPLITS = ("random", "popular", "adversarial")
INSTRUCTION = (
    "You should first think about the reasoning process in the mind and then "
    "provide the user with the answer. The reasoning process is enclosed within "
    "<think> </think> tags, i.e. <think> reasoning process here </think> answer here"
)
SUFFIX = " Please answer yes or no."

# Experiments whose metrics.json is the flat common.run shape {method: {...pope...}}.
POPE_EXPERIMENTS = ("pope_yesno_128_64_32", "pope_yesno_256_128_32", "pope_think_256_128_64")
THINK_EXPERIMENTS = ("pope_think_256_128_64",)
CHAIR_EXPERIMENT = "chair_64_32_16"
DIAG_EXPERIMENT = "pope_prompt_diagnostic"
EXTERNAL_SCORE_EXPERIMENTS = ("mmmu_wino_v1", "hallusion_full_original_256_128")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def input_key(sample):
    # Identical to common.run.input_key.
    return sha256_bytes(json.dumps(
        [sample["image_sha256"], sample.get("messages", sample.get("question"))],
        ensure_ascii=False, sort_keys=True).encode())


# --- metric functions ported verbatim from common/metrics.py ---

def official_pope_answer(text):
    words = text.split(".")[0].replace(",", "").split(" ")
    return "no" if any(word in words for word in ("No", "not", "no")) else "yes"


def pope_metrics(rows):
    counts = Counter()
    for row in rows:
        pred = official_pope_answer(row["text"]) == "yes"
        truth = row["label"] == "yes"
        counts["tp" if pred and truth else "fp" if pred else "fn" if truth else "tn"] += 1
        counts["invalid"] += re.match(r"^(yes|no)\b", row["text"].strip(), re.I) is None
        counts["empty"] += not row["text"].strip()
    tp, fp, fn, tn = (counts[k] for k in ("tp", "fp", "fn", "tn"))
    n = len(rows)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    return {
        "n": n, "accuracy": (tp + tn) / n if n else None,
        "precision": precision, "recall": recall,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
        "yes_ratio": (tp + fp) / n if n else None,
        "invalid_ratio": counts["invalid"] / n if n else None,
        "empty_answers": counts["empty"], "confusion": dict(counts),
    }


def paired_bootstrap(rows_a, rows_b, repeats=2000, seed=42):
    a, b = {r["id"]: r for r in rows_a}, {r["id"]: r for r in rows_b}
    if not a or a.keys() != b.keys() or len(a) != len(rows_a) or len(b) != len(rows_b):
        raise ValueError("bootstrap requires identical, nonempty, unique sample IDs")
    images = sorted({r["image_sha256"] for r in a.values()})
    index = {image: i for i, image in enumerate(images)}
    stats = np.zeros((2, len(images), 4), dtype=np.float64)
    for side, rows in enumerate((a, b)):
        for key, row in rows.items():
            if row["image_sha256"] != a[key]["image_sha256"] or row["label"] != a[key]["label"]:
                raise ValueError("paired samples differ")
            pred = official_pope_answer(row["text"]) == "yes"
            truth = row["label"] == "yes"
            cell = 0 if pred and truth else 1 if pred else 2 if truth else 3
            stats[side, index[row["image_sha256"]], cell] += 1
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repeats):
        totals = stats[:, rng.integers(0, len(images), len(images))].sum(1)
        tp, fp, fn, tn = totals.T
        acc = (tp + tn) / totals.sum(1)
        f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(2), where=(2 * tp + fp + fn) != 0)
        differences.append([acc[1] - acc[0], f1[1] - f1[0]])
    bounds = np.quantile(differences, [0.025, 0.975], axis=0)
    return {"unit": "image", "images": len(images), "repeats": repeats, "seed": seed,
            "delta_accuracy_ci95": bounds[:, 0].tolist(), "delta_f1_ci95": bounds[:, 1].tolist()}


# --- loading helpers ---

def jsonl_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_index():
    return json.loads(INDEX_PATH.read_text())


def dataset_dir(record_key):
    # record_key is "experiment/dataset"
    exp, ds = record_key.split("/", 1)
    return RECORDS / exp / ds


def stream_predictions(record_key, method):
    """Yield (input_key, payload_str) for every exported prediction, hash-checked."""
    index = load_index()[record_key][method]
    expected = {e["input_key"]: e["payload_sha256"] for e in index["entries"]}
    by_shard = {}
    for e in index["entries"]:
        by_shard.setdefault(e["shard"], set()).add(e["input_key"])
    method_dir = dataset_dir(record_key) / method
    seen = set()
    for shard in sorted(by_shard):
        wanted = by_shard[shard]
        with gzip.open(method_dir / shard, "rb") as stream:
            for line in stream:
                prefix, raw = line.split(b', "payload": ', 1)
                key = json.loads(prefix + b"}")["input_key"]
                if key not in wanted:
                    continue
                raw = raw[:-2]
                assert sha256_bytes(raw) == expected[key], (record_key, method, key)
                seen.add(key)
                yield key, raw.decode()
    assert seen == set(expected), (record_key, method, "missing predictions")


def all_predictions(record_key, method):
    return {key: json.loads(raw) for key, raw in stream_predictions(record_key, method)}


def check_sha256sums():
    failures = []
    checked = 0
    for sums in sorted(BASE.rglob("SHA256SUMS")):
        for line in sums.read_text().splitlines():
            if not line.strip():
                continue
            digest, rel = line[:64], line[66:]
            path = sums.parent / rel
            checked += 1
            if not path.is_file() or sha256_file(path) != digest:
                failures.append(str(path))
    assert not failures, f"SHA256SUMS mismatches: {failures[:10]}"
    print(f"sha256sums: {checked} files verified")


# --- POPE re-scoring ---

def pope_dataset_rows(record_key, method, samples):
    """Per-sample rows with text taken from the exported predictions."""
    preds = all_predictions(record_key, method)
    rows = []
    for sample in samples:
        key = input_key(sample)
        assert key in preds, (record_key, method, sample.get("id"), "missing prediction")
        rows.append({"id": sample["id"], "text": preds[key]["text"],
                     "label": sample["label"], "split": sample["split"],
                     "image_sha256": sample["image_sha256"]})
    return rows, len(preds)


def verify_pope_experiment(exp):
    for ds_dir in sorted((RECORDS / exp).iterdir()):
        if not ds_dir.is_dir():
            continue
        ds = ds_dir.name
        record_key = f"{exp}/{ds}"
        # samples_manifest lists the real sample files (exclude files are separate)
        manifest = json.loads((ds_dir / "samples_manifest.json").read_text())
        sample_names = {Path(s).name for s in manifest["source_files"]}
        samples = []
        for f in sorted(ds_dir.glob("*.jsonl")):
            if f.name in sample_names:
                samples.extend(jsonl_rows(f))
        metrics = json.loads((ds_dir / "metrics.json").read_text())
        rows_by_method = {}
        for method in metrics:
            if method not in ("native", "vcd", "vigor"):
                continue
            rows, n_preds = pope_dataset_rows(record_key, method, samples)
            rows_by_method[method] = rows
            stored = metrics[method]
            assert stored["evaluated_rows"] == len(rows), (record_key, method, "evaluated_rows")
            assert stored["unique_predictions"] == n_preds, (record_key, method, "unique_predictions")
            for split in SPLITS:
                subset = [r for r in rows if r["split"] == split]
                got = pope_metrics(subset)
                want = dict(stored["pope"][split])
                paired = want.pop("paired_vs_native", None)
                assert got == want, (record_key, method, split, "metrics differ")
                if paired is not None:
                    native_rows = [r for r in rows_by_method["native"] if r["split"] == split]
                    assert len(native_rows) == len(subset)
                    got_paired = paired_bootstrap(native_rows, subset)
                    assert got_paired == paired, (record_key, method, split, "paired bootstrap differs")
            print(f"{record_key}/{method}: {len(rows)} rows re-scored, "
                  f"per-split metrics + paired bootstrap match stored metrics.json")


def think_answer(text):
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip()
    return text


def verify_think_primary(exp):
    """Re-score the post-</think> primary track from answers.json."""
    for ds_dir in sorted((RECORDS / exp).iterdir()):
        if not (ds_dir / "primary_metrics.json").exists():
            continue
        stored = json.loads((ds_dir / "primary_metrics.json").read_text())
        scored = {}
        for method in ("native", "vcd", "vigor"):
            answers = json.loads((ds_dir / method / "answers.json").read_text())
            scored[method] = [{"id": r["id"], "image_sha256": r["image_sha256"],
                               "label": r["label"], "split": r["split"],
                               "text": think_answer(r["text"])} for r in answers]
            for split in SPLITS:
                got = pope_metrics([r for r in scored[method] if r["split"] == split])
                assert got == stored[method]["splits"][split], (exp, ds_dir.name, method, split)
        for method in ("vcd", "vigor"):
            for split in SPLITS:
                got = paired_bootstrap([r for r in scored["native"] if r["split"] == split],
                                       [r for r in scored[method] if r["split"] == split])
                assert got == stored[method]["paired_vs_native_primary"][split], (
                    exp, ds_dir.name, method, split, "primary paired")
        print(f"{exp}/{ds_dir.name}: post-</think> primary metrics + paired bootstrap match")


def judge_pope_metrics(pairs):
    """cdapc_lab.experiments.score.pope_metrics: None predictions are wrong."""
    tp = fp = tn = fn = unresolved = 0
    for pred, gold in pairs:
        if pred == "yes":
            tp += gold == "yes"
            fp += gold == "no"
        elif pred == "no":
            tn += gold == "no"
            fn += gold == "yes"
        elif gold == "yes":
            fn += 1
        else:
            unresolved += 1
    n = len(pairs)
    safe = lambda a, b: a / b if b else None
    precision, recall = safe(tp, tp + fp), safe(tp, tp + fn)
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "unresolved": unresolved,
            "accuracy": safe(tp + tn, n), "precision": precision, "recall": recall,
            "f1": safe(2 * precision * recall, precision + recall)
            if precision is not None and recall is not None else None,
            "yes_ratio": safe(sum(p == "yes" for p, _ in pairs), n)}


def verify_think_judge(exp):
    """Re-derive DeepSeek-judge metrics from api_records + *_judged.json."""
    for judge_dir in sorted((RECORDS / exp).glob("*/judge")):
        summary = json.loads((judge_dir / "summary.json").read_text())
        records = {}
        with gzip.open(judge_dir / "api_records.jsonl.gz", "rt") as handle:
            for line in handle:
                rec = json.loads(line)
                records[rec["request_sha256"]] = rec
        answers_dir = judge_dir.parent
        for method in ("native", "vcd", "vigor"):
            judged = json.loads((judge_dir / f"{method}_judged.json").read_text())
            answers = json.loads((answers_dir / method / "answers.json").read_text())
            assert len(judged) == len(answers)
            for row, ans in zip(judged, answers):
                assert (row["id"], row["split"]) == (ans["id"], ans["split"])
                assert row["text_sha256"] == hashlib.sha256(ans["text"].encode()).hexdigest()
                rec = records[row["request_sha256"]]
                assert rec["status"] == "success"
                assert json.loads(rec["request"]["messages"][1]["content"])["candidate_response"] == ans["text"]
                assert rec["parsed"]["answer"] == row["judge_prediction"]
            for split in SPLITS:
                sub = [r for r in judged if r["split"] == split]
                got = judge_pope_metrics([(r["judge_prediction"], r["label"]) for r in sub])
                want = {k: v for k, v in summary["methods"][method]["splits"][split].items()
                        if k not in ("n", "statuses")}
                assert got == want, (judge_dir, method, split)
        print(f"{judge_dir.relative_to(RECORDS)}: {len(records)} API records, judge metrics match")


# --- CHAIR re-scoring ---

def chair_overall(chair):
    """Re-aggregate overall_metrics from sentences (formulas verified against
    data/evaluators/chair.py compute_chair)."""
    sentences = chair["sentences"]
    num_caps = len(sentences)
    num_hallucinated = sum(1 for s in sentences if s["metrics"]["CHAIRs"] == 1)
    hallucinated_words = sum(len(s["mscoco_hallucinated_words"]) for s in sentences)
    node_words = sum(len(s["mscoco_generated_words"]) for s in sentences)
    recall_hits = sum(len(set(s["mscoco_generated_words"]) & set(s["mscoco_gt_words"]))
                      for s in sentences)
    gt_objects = sum(len(s["mscoco_gt_words"]) for s in sentences)
    return {"CHAIRs": num_hallucinated / num_caps,
            "CHAIRi": hallucinated_words / node_words,
            "Recall": recall_hits / gt_objects}


def verify_chair():
    exp = CHAIR_EXPERIMENT
    ds_dir = RECORDS / exp / "chair500"
    record_key = f"{exp}/chair500"
    stored = json.loads((ds_dir / "chair_metrics.json").read_text())
    expected = {}
    for run_tag in ("vcd_run", "vigor_run"):
        for method, entry in stored[run_tag]["methods"].items():
            expected.setdefault(method, []).append(entry)
    for method in ("native", "vcd", "vigor"):
        entries = expected[method]
        assert all(e == entries[0] for e in entries), (method, "inconsistent stored entries")
        chair = json.loads((ds_dir / method / "chair.json").read_text())
        recomputed = chair_overall(chair)
        assert recomputed == chair["overall_metrics"], (method, "chair.json self-inconsistent")
        want = {k: entries[0][k] for k in ("CHAIRs", "CHAIRi", "Recall", "answer_token_length_mean")}
        got = {**recomputed, "answer_token_length_mean": chair["answer_token_length_mean"]}
        assert got == want, (method, got, want)
        # cross-check answer_token_length_mean against exported answers
        answers = json.loads((ds_dir / method / "answers.json").read_text())
        mean_len = sum(r["answer_token_length"] for r in answers) / len(answers)
        assert abs(mean_len - want["answer_token_length_mean"]) < 1e-9, method
        print(f"{record_key}/{method}: CHAIRs={recomputed['CHAIRs']:.4f} "
              f"CHAIRi={recomputed['CHAIRi']:.5f} Recall={recomputed['Recall']:.5f} "
              f"(recomputed from {len(chair['sentences'])} sentences, matches stored)")


# --- prompt diagnostic re-scoring (ports score.py logic) ---

def variant_question(variant, q0):
    if variant == "A":
        return q0
    if variant == "B":
        return INSTRUCTION + "\n" + q0
    if variant == "C":
        return q0
    if variant == "D":
        return q0 + SUFFIX
    raise ValueError(variant)


def variant_sample(row, variant):
    return {"id": row["id"], "split": row["split"], "label": row["label"],
            "image": row["image"], "image_sha256": row["image_sha256"],
            "question": variant_question(variant, row["q0"])}


def extract_answer(variant, record):
    text = record["text"]
    if variant != "B":
        return text, {"think_closed": None}
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip(), {"think_closed": True}
    return text, {"think_closed": False}


def diag_metrics_block(scored, answer_key="answer"):
    rows = [{"text": r[answer_key], "label": r["label"], "split": r["split"]} for r in scored]
    return {"rows": len(rows), "overall": pope_metrics(rows),
            "splits": {s: pope_metrics([r for r in rows if r["split"] == s]) for s in SPLITS}}


def diag_diagnostics(scored):
    yes_start_no = no_start_yes = neither = 0
    for row in scored:
        text = row["answer"].strip().lower()
        if text.startswith("yes") and row["parsed"] == "no":
            yes_start_no += 1
        if text.startswith("no") and row["parsed"] == "yes":
            no_start_yes += 1
        if not text.startswith(("yes", "no")):
            neither += 1
    n = len(scored)
    if not n:
        return {"rows": 0}
    return {
        "rows": n, "yes_start_parsed_no": yes_start_no, "no_start_parsed_yes": no_start_yes,
        "not_starting_with_yes_or_no": neither,
        "answer_token_length_mean": statistics.mean(r["answer_token_length"] for r in scored),
        "answer_token_length_median": statistics.median(r["answer_token_length"] for r in scored),
        "eos_found_rate": sum(bool(r["eos_found"]) for r in scored) / n,
        "executed_steps_mean": statistics.mean(r["executed_steps"] for r in scored),
        "end_to_end_seconds_mean": statistics.mean(r["end_to_end_seconds"] for r in scored),
    }


def diag_think_stats(scored):
    n = len(scored)
    if not n:
        return {"rows": 0}
    with_open = sum("<think>" in r["generated_text"] for r in scored)
    with_close = sum("</think>" in r["generated_text"] for r in scored)
    return {"rows": n, "with_think_tag": with_open,
            "with_think_tag_ratio": with_open / n, "with_think_close": with_close,
            "with_think_close_ratio": with_close / n,
            "think_close_missing": sum(r["think_closed"] is False for r in scored)}


def diag_paired_vs_a(a_scored, v_scored):
    a_parsed = {r["id"]: r["parsed"] for r in a_scored}
    out = {"changed": 0, "fixed": 0, "broken": 0,
           "by_label": {label: {"fixed": 0, "broken": 0} for label in ("yes", "no")},
           "changed_ids": []}
    for row in v_scored:
        parsed_a = a_parsed.get(row["id"])
        if parsed_a is None or parsed_a == row["parsed"]:
            continue
        a_correct = (parsed_a == "yes") == (row["label"] == "yes")
        v_correct = (row["parsed"] == "yes") == (row["label"] == "yes")
        kind = "fixed" if not a_correct and v_correct else "broken"
        out["changed"] += 1
        out[kind] += 1
        out["by_label"][row["label"]][kind] += 1
        out["changed_ids"].append(row["id"])
    return out


def diag_examples(scored, limit=3):
    out = []
    for row in scored:
        if row["label"] != "no":
            continue
        out.append({key: row[key] for key in
                    ("id", "split", "label", "q0", "answer", "generated_text", "parsed",
                     "eos_found", "executed_steps", "end_to_end_seconds")})
        if len(out) == limit:
            break
    return out


def verify_diagnostic():
    record_key = f"{DIAG_EXPERIMENT}/coco_192"
    ds_dir = dataset_dir(record_key)
    samples = jsonl_rows(ds_dir / "samples.jsonl")
    stored = json.loads((ds_dir / "metrics.json").read_text())
    scored_by_variant = {}
    for variant in ("A", "B", "C", "D"):
        preds = all_predictions(record_key, variant)
        scored, missing = [], []
        for row in samples:
            key = input_key(variant_sample(row, variant))
            record = preds.get(key)
            if record is None:
                missing.append(row["id"])
                continue
            answer, info = extract_answer(variant, record)
            scored.append({
                "id": row["id"], "split": row["split"], "label": row["label"], "q0": row["q0"],
                "answer": answer, "generated_text": record["text"],
                "parsed": official_pope_answer(answer),
                "think_closed": info["think_closed"],
                "answer_token_length": record["answer_token_length"],
                "eos_found": record["eos_found"],
                "executed_steps": record["executed_steps"],
                "end_to_end_seconds": record["end_to_end_seconds"],
            })
        scored_by_variant[variant] = scored
        want = stored["variants"][variant]
        assert want["rows_scored"] == len(scored), (variant, "rows_scored")
        assert want["missing_ids"] == missing, (variant, "missing_ids")
        assert want["complete"] == (not missing), (variant, "complete")
        got_metrics = diag_metrics_block(scored)
        assert got_metrics == want["metrics"], (variant, "metrics differ")
        assert diag_diagnostics(scored) == want["diagnostics"], (variant, "diagnostics differ")
        assert diag_examples(scored) == want["examples_label_no"], (variant, "examples differ")
        if variant == "B":
            assert diag_think_stats(scored) == want["think"], (variant, "think stats differ")
            full_scored = [dict(r, answer=r["generated_text"],
                                parsed=official_pope_answer(r["generated_text"])) for r in scored]
            got_full = {"metrics": diag_metrics_block(full_scored),
                        "diagnostics": diag_diagnostics(full_scored)}
            assert got_full == want["fulltext_secondary"], (variant, "fulltext secondary differs")
        print(f"{record_key}/{variant}: {len(scored)} rows re-scored, all stored fields match")
    for variant in ("B", "C", "D"):
        got = diag_paired_vs_a(scored_by_variant["A"], scored_by_variant[variant])
        assert got == stored["variants"][variant]["paired_vs_A"], (variant, "paired_vs_A differs")
    print(f"{record_key}: paired_vs_A flips verified for B/C/D")


# --- external-judge experiments: integrity + coverage only ---

def verify_external_experiment(exp):
    for ds_dir in sorted((RECORDS / exp).iterdir()):
        if not ds_dir.is_dir():
            continue
        ds = ds_dir.name
        record_key = f"{exp}/{ds}"
        index = load_index()[record_key]
        for method, info in index.items():
            if not isinstance(info, dict) or "entries" not in info:
                continue
            n = sum(1 for _ in stream_predictions(record_key, method))
            assert n == info["predictions"] == len(info["entries"]), (record_key, method, n)
            answers = json.loads((ds_dir / method / "answers.json").read_text())
            assert len(answers) == info["answers_rows"], (record_key, method, "answers rows")
        manifest = json.loads((ds_dir / "samples_manifest.json").read_text())
        total = sum(f["rows"] for f in manifest["files"])
        print(f"{record_key}: payload hashes ok for {sorted(index)}; "
              f"{total} sample rows covered (external scorer outputs not recomputed)")


# --- optional prompt rebuild check (needs the MMaDA-Baselines checkout) ---

def record_samples(ds_dir):
    """Sample rows for a dataset dir, keyed by input_key, in file order."""
    manifest = json.loads((ds_dir / "samples_manifest.json").read_text())
    names = {Path(s).name for s in manifest["source_files"]}
    rows = []
    for f in sorted(ds_dir.glob("*.jsonl")):
        if f.name in names:
            rows.extend(jsonl_rows(f))
    return rows


def stored_input_fields(record_key, method, wanted_keys):
    """Return {input_key: input-dict} for the wanted keys only."""
    index = load_index()[record_key][method]
    wanted = set(wanted_keys)
    out = {}
    by_shard = {}
    for e in index["entries"]:
        if e["input_key"] in wanted:
            by_shard.setdefault(e["shard"], set()).add(e["input_key"])
    method_dir = dataset_dir(record_key) / method
    for shard, keys in by_shard.items():
        with gzip.open(method_dir / shard, "rb") as stream:
            for line in stream:
                prefix, raw = line.split(b', "payload": ', 1)
                key = json.loads(prefix + b"}")["input_key"]
                if key in keys:
                    out[key] = json.loads(raw[:-2].decode())["input"]
    return out


def source_prompt_ids(src_root, record_key, method, input_key):
    """Original prompt_ids/visual_positions from the run's SQLite payloads."""
    import sqlite3
    info = load_index()[record_key][method]
    for source in info.get("sources", []):
        path = src_root / source["path"]
        if not path.exists():
            continue
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            cols = [r[1] for r in con.execute("PRAGMA table_info(predictions)")]
            if "method" in cols:
                row = con.execute(
                    "SELECT payload FROM predictions WHERE method=? AND input_key=?",
                    (method, input_key)).fetchone()
            else:
                row = con.execute(
                    "SELECT payload FROM predictions WHERE input_key=?",
                    (input_key,)).fetchone()
        finally:
            con.close()
        if row:
            inp = json.loads(row[0])["input"]
            return inp["prompt_ids"], inp.get("visual_positions") or []
    return None, None


def rebuild_prompts(per_record=5):
    """Two checks per sampled stored input (>=per_record rows per
    record/method):

    1. transform fidelity (exact): the exported prompt_ids_sha256 /
       prompt_ids_length / visual_positions equal the sha256/length/range
       recipe applied to the original SQLite prompt_ids;
    2. end-to-end rebuild: common.runtime.Runtime.prepare(image+question) on
       CPU reproduces the prompt; image VQ codes may differ in a small
       fraction of positions vs the GPU-produced originals (CPU fp32 vs GPU
       TF32 numerics) — reported, not failed, when diffs stay inside the
       visual span and the text tail is identical.

    Requires MMADA_BASELINES checkout + its .venv; CPU-only, no GPU use."""
    src = Path(os.environ.get("MMADA_BASELINES", "/root/autodl-tmp/MMaDA-Baselines"))
    sys.path.insert(0, str(src))
    from common.runtime import Runtime
    runtime = Runtime("/root/autodl-fs/models/MMaDA-8B-MixCoT",
                      "/root/autodl-fs/models/magvitv2",
                      device="cpu", resolution=512, resize="crop")
    # collect (record_key, method) -> sampled input_keys -> sample row
    jobs = {}   # input_key -> {"sample": row, "expects": [(record_key, method, stored_input)]}
    for record_key, methods in load_index().items():
        exp, ds = record_key.split("/", 1)
        ds_dir = dataset_dir(record_key)
        if exp == DIAG_EXPERIMENT:
            continue  # diagnostic samples need variant reconstruction; handled below
        samples = record_samples(ds_dir)
        if not samples:
            continue
        # deterministic spread across the file
        picks = [samples[int(i * (len(samples) - 1) / max(per_record - 1, 1))]
                 for i in range(per_record)]
        keys = [input_key(s) for s in picks]
        for method in methods:
            stored = stored_input_fields(record_key, method, keys)
            for sample, key in zip(picks, keys):
                if key not in stored:
                    continue
                job = jobs.setdefault(key, {"sample": sample, "expects": []})
                job["expects"].append((record_key, method, stored[key], False))
    # diagnostic: rebuild per variant with variant_question
    ds_dir = dataset_dir(f"{DIAG_EXPERIMENT}/coco_192")
    diag_samples = jsonl_rows(ds_dir / "samples.jsonl")
    for variant in ("A", "B", "C", "D"):
        picks = [diag_samples[int(i * (len(diag_samples) - 1) / max(per_record - 1, 1))]
                 for i in range(per_record)]
        rows = [variant_sample(r, variant) for r in picks]
        keys = [input_key(r) for r in rows]
        stored = stored_input_fields(f"{DIAG_EXPERIMENT}/coco_192", variant, keys)
        for row, key in zip(rows, keys):
            if key not in stored:
                continue
            job = jobs.setdefault(key, {"sample": row, "expects": []})
            # variant C uses the custom training-text prepare(), not
            # common.runtime — fidelity is still checked, rebuild is skipped.
            job["expects"].append((f"{DIAG_EXPERIMENT}/coco_192", variant,
                                   stored[key], variant == "C"))

    fidelity_fail = rebuild_exact = rebuild_numeric = rebuild_bad = rebuild_skipped = 0
    diff_positions = []
    for key, job in sorted(jobs.items()):
        sample = job["sample"]
        _, _, provenance = runtime.prepare(sample)
        rebuilt_ids = provenance["prompt_ids"]
        rebuilt_visual = provenance["visual_positions"]
        rebuilt_digest = hashlib.sha256(
            json.dumps(rebuilt_ids, separators=(",", ":")).encode()).hexdigest()
        for record_key, method, stored_input, skip_rebuild in job["expects"]:
            # 1. exact transform fidelity vs the source SQLite payload
            orig_ids, orig_visual = source_prompt_ids(src, record_key, method, key)
            assert orig_ids is not None, (record_key, method, key, "no source payload")
            orig_digest = hashlib.sha256(
                json.dumps(orig_ids, separators=(",", ":")).encode()).hexdigest()
            orig_len = len(orig_ids)
            want_visual = ([orig_visual[0], orig_visual[0] + len(orig_visual)]
                           if orig_visual
                           and orig_visual == list(range(orig_visual[0],
                                                        orig_visual[0] + len(orig_visual)))
                           else orig_visual)
            if not (stored_input.get("prompt_ids_sha256") == orig_digest
                    and stored_input.get("prompt_ids_length") == orig_len
                    and stored_input.get("visual_positions") == want_visual):
                fidelity_fail += 1
                print(f"FIDELITY MISMATCH {record_key}/{method} key={key[:12]}")
            # 2. end-to-end rebuild vs original ids (skipped for custom
            #    prompt protocols such as diagnostic variant C)
            if skip_rebuild:
                rebuild_skipped += 1
                continue
            if len(rebuilt_ids) != len(orig_ids):
                rebuild_bad += 1
                print(f"REBUILD LENGTH MISMATCH {record_key}/{method} key={key[:12]}")
                continue
            diffs = [i for i, (a, b) in enumerate(zip(rebuilt_ids, orig_ids)) if a != b]
            visual_set = set(orig_visual)
            outside = [i for i in diffs if i not in visual_set]
            text_start = max(orig_visual) + 1 if orig_visual else 0
            tail_same = rebuilt_ids[text_start:] == orig_ids[text_start:]
            if not diffs:
                rebuild_exact += 1
            elif not outside and tail_same and len(diffs) <= max(8, len(orig_visual) // 50):
                rebuild_numeric += 1
                diff_positions.append(len(diffs))
            else:
                rebuild_bad += 1
                print(f"REBUILD MISMATCH {record_key}/{method} key={key[:12]} "
                      f"diffs={len(diffs)} outside_visual={len(outside)} tail_same={tail_same}")
    total = fidelity_fail + rebuild_exact + rebuild_numeric + rebuild_bad
    checked = sum(len(j["expects"]) for j in jobs.values())
    print(f"rebuild-prompts: {checked} stored inputs checked "
          f"({len(jobs)} unique prompts rebuilt on CPU)")
    print(f"  transform fidelity vs source sqlite: "
          f"{checked - fidelity_fail}/{checked} exact")
    print(f"  end-to-end rebuild: exact={rebuild_exact} "
          f"cpu-vs-gpu code diffs only={rebuild_numeric} bad={rebuild_bad} "
          f"skipped(custom protocol)={rebuild_skipped}"
          + (f" (diff positions per row: min={min(diff_positions)} "
             f"max={max(diff_positions)})" if diff_positions else ""))
    assert fidelity_fail == 0, "export transform did not reproduce source hashes"
    assert rebuild_bad == 0, "prompt rebuild diverged beyond visual-code numerics"


def main():
    check_sha256sums()
    index = load_index()
    for record_key in index:
        exp = record_key.split("/", 1)[0]
        if exp in POPE_EXPERIMENTS:
            continue  # handled per-experiment below
        if exp == CHAIR_EXPERIMENT:
            continue
        if exp == DIAG_EXPERIMENT:
            continue
        if exp in EXTERNAL_SCORE_EXPERIMENTS:
            verify_external_experiment(exp)
    for exp in POPE_EXPERIMENTS:
        verify_pope_experiment(exp)
    for exp in THINK_EXPERIMENTS:
        verify_think_primary(exp)
        verify_think_judge(exp)
    verify_chair()
    verify_diagnostic()
    if "--rebuild-prompts" in sys.argv:
        rebuild_prompts()
    print("all checks passed")


if __name__ == "__main__":
    sys.exit(main())
