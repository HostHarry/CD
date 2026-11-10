"""Compare the two completed COCO think runs under the same answer-region rule."""
from collections import Counter
from datetime import datetime
from pathlib import Path
import json

import numpy as np
from cdapc_lab.io.artifacts import atomic_json, file_hash, read_json

ROOT = Path(__file__).resolve().parents[1]


def metrics(totals):
    tp, fp, tn, fn, unresolved = np.moveaxis(totals, -1, 0)
    accuracy = (tp + tn) / (tp + fp + tn + fn + unresolved)
    denominator = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0)
    return np.stack((accuracy.mean(-1), f1.mean(-1)), axis=-1)


def main():
    old_path = ROOT.parent / "cdapc-pope-eos-full/reports/pope_coco_think_eos_prefix_r70_k4_L256_I128_after_think_official.json"
    new_path = ROOT.parent / "cdapc-pope-safe-eos/reports/pope_coco_think_safe_eos_r70_k4_L256_I128_after_think_official.json"
    rows_path = ROOT.parent / "cdapc-pope-safe-eos/data/pope_yesno/rows.json"
    reports = [read_json(p) for p in (old_path, new_path)]
    rows = read_json(rows_path)
    assert len(rows) == 9000
    assert all(r["evaluation_rows"] == 9000 and r["rows_sha256"] == file_hash(rows_path) for r in reports)
    ids = {row["input_key"] for row in rows}
    assert all(set(r["per_input"]) == ids for r in reports)
    images = sorted({row["image_sha256"] for row in rows})
    splits = ("random", "popular", "adversarial")
    image_index = {image: i for i, image in enumerate(images)}
    cells = np.zeros((2, 3, len(images), 5), dtype=np.int64)
    transitions = Counter()
    for row in rows:
        correct = []
        for side, report in enumerate(reports):
            p = report["per_input"][row["input_key"]]
            assert p["gold"] == row["label"]
            pred, gold = p["effective_prediction"], row["label"]
            cell = (0 if pred == "yes" and gold == "yes" else
                    1 if pred == "yes" else 2 if pred == "no" and gold == "no" else
                    3 if gold == "yes" else 4)
            cells[side, splits.index(row["split"]), image_index[row["image_sha256"]], cell] += 1
            correct.append(pred == gold)
        transitions[f"{int(correct[0])}_to_{int(correct[1])}"] += 1
    point = metrics(cells.sum(axis=2))
    for side, report in enumerate(reports):
        np.testing.assert_allclose(point[side], [report["macro"]["accuracy"], report["macro"]["f1"]], atol=1e-12)
    rng = np.random.default_rng(42)
    differences = []
    for _ in range(2000):
        draws = rng.integers(0, len(images), len(images))
        sampled = metrics(cells[:, :, draws, :].sum(axis=2))
        differences.append(sampled[1] - sampled[0])
    bounds = np.quantile(differences, [0.025, 0.975], axis=0)
    result = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "inputs": {str(p): file_hash(p) for p in (old_path, new_path, rows_path)},
        "script_sha256": file_hash(Path(__file__)),
        "unique_inputs": len(ids), "evaluation_rows": len(rows),
        "protocol": reports[1]["protocol"], "failure_policy": reports[1]["failure_policy"],
        "bootstrap": {"unit": "image_sha256", "images": len(images), "repeats": 2000, "seed": 42,
                      "metric": "equal mean over the three split metrics"},
        "eos_prefix": reports[0]["macro"], "original_candidates_safe_eos": reports[1]["macro"],
        "delta_new_minus_old": dict(zip(("accuracy", "f1"), (point[1] - point[0]).tolist())),
        "ci95_new_minus_old": dict(zip(("accuracy", "f1"), bounds.T.tolist())),
        "correctness_transitions_rows": dict(transitions),
        "selected_text_changes_unique": sum(reports[0]["per_input"][sid]["selected_text"] !=
                                            reports[1]["per_input"][sid]["selected_text"] for sid in ids),
        "effective_prediction_changes_unique": sum(reports[0]["per_input"][sid]["effective_prediction"] !=
                                                   reports[1]["per_input"][sid]["effective_prediction"] for sid in ids),
        "cost": {"eos_prefix": reports[0]["cost"], "original_candidates": reports[1]["cost"]},
        "generation_statuses": {r["stage"]: r["generation_statuses"] for r in reports},
    }
    target = ROOT / "reports/pope_coco_safe_vs_prefix_full_official.json"
    atomic_json(target, result)
    target.with_suffix(".sha256").write_text("".join(
        f"{file_hash(p)}  {p}\n" for p in (Path(__file__), target)))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
