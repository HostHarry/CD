from collections import defaultdict
import numpy as np
from cdapc_lab.experiments.score import select_evaluation


def paired_statistics(a, b, groups, seed=20261105, repeats=2000):
    if not a or len(a) != len(b) or len(groups) != len(a):
        raise ValueError("Expected nonempty matching paired data")
    differences = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    clusters = defaultdict(list)
    for i, group in enumerate(groups):
        clusters[group].append(i)
    units = list(clusters.values())
    rng = np.random.default_rng(seed)
    boot = [float(differences[np.concatenate([units[i] for i in rng.integers(len(units), size=len(units))])].mean()) for _ in range(repeats)]
    return {"paired_difference_b_minus_a": float(differences.mean()), "ci95": np.quantile(boot, [.025, .975]).tolist(),
            "unit_count": len(units), "sample_count": len(a), "bootstrap_repeats": repeats, "seed": seed,
            "four_cells": {"both_correct": sum(bool(x) and bool(y) for x, y in zip(a, b)),
                           "a_only": sum(bool(x) and not bool(y) for x, y in zip(a, b)),
                           "b_only": sum(not bool(x) and bool(y) for x, y in zip(a, b)),
                           "both_wrong": sum(not bool(x) and not bool(y) for x, y in zip(a, b))}}


def compare(run_a, evaluation_a, run_b, evaluation_b, metric="strict_accuracy", unit="sample_id"):
    if metric not in ("strict_accuracy", "official") or unit not in ("sample_id", "group_id"):
        raise ValueError("Unknown metric or resampling unit")
    ma, a = select_evaluation(run_a, evaluation_a)
    mb, b = select_evaluation(run_b, evaluation_b)
    if ma["evaluation_spec_hash"] != mb["evaluation_spec_hash"]:
        raise ValueError("Incompatible evaluation specs/gold/options")
    by_id = {r["sample_id"]: r for r in b}
    if set(by_id) != {r["sample_id"] for r in a}:
        raise ValueError("Sample ID mismatch")
    groups = [r.get("group_id") or r["sample_id"] for r in a]
    if len(set(groups)) != len(groups) and unit != "group_id":
        raise ValueError("Shared question groups require group_id cluster bootstrap")
    return paired_statistics([r[metric] for r in a], [by_id[r["sample_id"]][metric] for r in a],
                             groups if unit == "group_id" else [r["sample_id"] for r in a])
