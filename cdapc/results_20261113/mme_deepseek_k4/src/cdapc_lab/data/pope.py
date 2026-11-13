import hashlib
import json
from collections import Counter
import re

import numpy as np

SUFFIX = " Please answer yes or no."


def pope_input_key(image_sha256, question_with_suffix):
    return hashlib.sha256(json.dumps(
        [image_sha256, question_with_suffix],
        ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def split_metrics(rows, rule):
    """Port of HostHarry/CD degap pope_metrics: rows carry id/split/label/image_sha256/text."""
    counts = Counter()
    for row in rows:
        pred = rule(row["text"]) == "yes"
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


def paired_bootstrap(rows_a, rows_b, rule, repeats=2000, seed=42):
    """Port of HostHarry/CD degap paired_bootstrap; deltas are B - A (A = native)."""
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
            pred = rule(row["text"]) == "yes"
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
