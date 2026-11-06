"""Check exported records and recompute POPE metrics.json without the original repository."""
import gzip
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def input_key(sample):
    return hashlib.sha256(json.dumps(
        [sample["image_sha256"], sample.get("messages", sample.get("question"))],
        ensure_ascii=False, sort_keys=True).encode()).hexdigest()


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
    images = sorted({r["image_sha256"] for r in a.values()})
    index = {image: i for i, image in enumerate(images)}
    stats = np.zeros((2, len(images), 4), dtype=np.float64)
    for side, rows in enumerate((a, b)):
        for row in rows.values():
            pred = official_pope_answer(row["text"]) == "yes"
            truth = row["label"] == "yes"
            stats[side, index[row["image_sha256"]], 0 if pred and truth else 1 if pred else 2 if truth else 3] += 1
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


def load(benchmark, method, index):
    records = {}
    expected = {e["input_key"]: e["payload_sha256"] for e in index[benchmark][method]["entries"]}
    for shard in sorted((ROOT / benchmark / method).glob("predictions-*.jsonl.gz")):
        with gzip.open(shard, "rb") as stream:
            for line in stream:
                prefix, raw = line.split(b', "payload": ', 1)
                key = json.loads(prefix + b"}")["input_key"]
                raw = raw[:-2]
                assert hashlib.sha256(raw).hexdigest() == expected[key], (benchmark, method, key)
                records[key] = json.loads(raw)
    assert records.keys() == expected.keys(), (benchmark, method)
    return records


def main():
    failures = [line for line in (ROOT / "SHA256SUMS").read_text().splitlines()
                if hashlib.sha256((ROOT / line[66:]).read_bytes()).hexdigest() != line[:64]]
    assert not failures, failures
    index = json.loads((ROOT / "records_index.json").read_text())
    for benchmark, samples_file in (("pope", "pope_coco_yesno.jsonl"), ("chair", "chair500.jsonl")):
        samples = [json.loads(line) for line in (ROOT / "protocol" / samples_file).read_text().splitlines()]
        keys = {input_key(s) for s in samples}
        rows = {}
        for method in ("native", "degap_token"):
            records = load(benchmark, method, index)
            assert records.keys() == keys, (benchmark, method, "input keys differ from protocol samples")
            assert all(r["remaining_masks"] == 0 for r in records.values())
            rows[method] = [{**s, "text": records[input_key(s)]["text"]} for s in samples]
            print(f"{benchmark} {method}: {len(records)} unique inputs, {len(rows[method])} rows, payload hashes ok")
        if benchmark == "pope":
            stored = json.loads((ROOT / "pope/run/metrics.json").read_text())
            for method in ("native", "degap_token"):
                for split in ("adversarial", "popular", "random"):
                    a = [r for r in rows["native"] if r["split"] == split]
                    b = [r for r in rows[method] if r["split"] == split]
                    got = pope_metrics(b)
                    if method != "native":
                        got["paired_vs_native"] = paired_bootstrap(a, b)
                    assert got == stored[method]["pope"][split], (method, split)
                    print(f"pope {method} {split}: metrics.json reproduced exactly")
    print("all checks passed")


if __name__ == "__main__":
    sys.exit(main())
