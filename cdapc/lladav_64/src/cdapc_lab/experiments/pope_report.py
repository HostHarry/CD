import gzip
import json
from collections import Counter
from pathlib import Path

from cdapc_lab.data.pope import split_metrics, paired_bootstrap
from cdapc_lab.evaluation.scorers import load_pope_official_rule
from cdapc_lab.io.artifacts import read_json, atomic_json
from cdapc_lab.experiments.score import select_evaluation, pope_metrics
from cdapc_lab.evaluation.parsers import reasoned_yes_no

ROOT = Path(__file__).resolve().parents[3]
SPLITS = ("random", "popular", "adversarial")


def reasoning_split_metrics(rows):
    predictions = [reasoned_yes_no(r["text"]).prediction
                   if r.get("generation_status", "success") == "success" else None for r in rows]
    metrics = pope_metrics(zip(predictions, (r["label"] for r in rows)))
    return {**metrics, "n": len(rows),
            "invalid_ratio": sum(p is None for p in predictions) / len(rows) if rows else None,
            "empty_answers": sum(not r["text"].strip() for r in rows)}


def load_native_export(export_root):
    payloads = {}
    for path in sorted((Path(export_root) / "pope/native").glob("predictions-*.jsonl.gz")):
        with gzip.open(path, "rt") as stream:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    payloads[record["input_key"]] = record["payload"]
    if not payloads:
        raise ValueError(f"No native export records under {export_root}")
    return payloads


def split_rows(all_rows, texts_by_key, generation_statuses=None):
    rows = []
    for row in all_rows:
        if row["input_key"] in texts_by_key:
            rows.append({"id": row["row_id"], "split": row["split"], "label": row["label"],
                         "image_sha256": row["image_sha256"], "text": texts_by_key[row["input_key"]]})
            if generation_statuses is not None:
                rows[-1]["generation_status"] = generation_statuses[row["input_key"]]
    return rows


def mechanism_totals(predictions, statuses=("success",)):
    totals = {"proposal_flips": 0, "qualified_flips": 0, "actual_switches": 0,
              "fallback_iterations": 0, "decode_iterations": 0}
    for prediction in predictions:
        if prediction.get("status") not in statuses:
            continue
        metrics = prediction.get("metrics", {})
        for key in ("proposal_flips", "qualified_flips", "actual_switches", "fallback_iterations"):
            totals[key] += metrics.get(key, 0)
        totals["decode_iterations"] += prediction.get("decode_iterations", 0)
    return totals


def transition_cells(pairs):
    """pairs: iterable of (native_pred, cdapc_pred, label) with preds 'yes'/'no'."""
    cells = {"yes->yes": 0, "yes->no": 0, "no->yes": 0, "no->no": 0,
             "both_correct": 0, "fixed": 0, "broken": 0, "both_wrong": 0}
    for native, cdapc, label in pairs:
        cells[f"{native}->{cdapc}"] += 1
        n_ok, c_ok = native == label, cdapc == label
        cells["both_correct" if n_ok and c_ok else "fixed" if c_ok else
              "broken" if n_ok else "both_wrong"] += 1
    return cells


def answer_switch(commit_log, first_eos):
    """True if some entry before the first EOS wrote a non-base token."""
    for entry in commit_log or []:
        if first_eos is not None and entry["position"] >= first_eos:
            continue
        if entry["written_id"] != entry["base_id"]:
            return True
    return False


def confidence_shift(entry, decode, alpha):
    """LO_I/LO_h at the answer position from a commit_log entry's top5."""
    lp_yes = lp_no = la_yes = la_no = None
    in_pool_other = None
    yes_entry = no_entry = None
    for token in entry["top5"]:
        text = decode(token["id"]).strip().lower()
        if text == "yes" and yes_entry is None:
            yes_entry = token
        if text == "no" and no_entry is None:
            no_entry = token
    if yes_entry is None or no_entry is None:
        return None
    lp_yes, lp_no = yes_entry["logp_visual"], no_entry["logp_visual"]
    la_yes, la_no = yes_entry["logp_ablated"], no_entry["logp_ablated"]
    lo_i = lp_yes - lp_no
    delta = alpha * ((lp_yes - la_yes) - (lp_no - la_no))
    return {"LO_I": lo_i, "delta": delta, "LO_h": lo_i + delta,
            "yes_in_pool": yes_entry["in_pool"], "no_in_pool": no_entry["in_pool"]}


def confidence_summary(values, decode_unused=None):
    import numpy as np
    deltas = [v["delta"] for v in values]
    abs_deltas = [abs(v["delta"]) for v in values]
    sig = lambda x: 1 / (1 + np.exp(-x))
    return {"n": len(values),
            "delta_mean": float(np.mean(deltas)) if deltas else None,
            "delta_median": float(np.median(deltas)) if deltas else None,
            "delta_p5": float(np.quantile(deltas, 0.05)) if deltas else None,
            "delta_p95": float(np.quantile(deltas, 0.95)) if deltas else None,
            "abs_delta_mean": float(np.mean(abs_deltas)) if deltas else None,
            "abs_delta_median": float(np.median(abs_deltas)) if deltas else None,
            "abs_delta_p5": float(np.quantile(abs_deltas, 0.05)) if deltas else None,
            "abs_delta_p95": float(np.quantile(abs_deltas, 0.95)) if deltas else None,
            "mean_sigmoid_LO_I": float(np.mean([sig(v["LO_I"]) for v in values])) if values else None,
            "mean_sigmoid_LO_h": float(np.mean([sig(v["LO_h"]) for v in values])) if values else None}


def pope_report(runs, evaluations=None, native_export=None, allow_partial=False, output=None):
    rule = load_pope_official_rule(ROOT / "upstream/pope/evaluate.py")
    runs = [Path(r) for r in runs]
    evaluations = evaluations or [None] * len(runs)
    if len(evaluations) != len(runs):
        raise ValueError("--evaluations must align with --runs")
    report = {"runs": [str(r) for r in runs], "native_export": str(native_export) if native_export else None,
              "partial": allow_partial}
    texts_by_key, covered, overlap, predictions_by_key = {}, set(), set(), {}
    generation_statuses = {}
    costs = {"generation_seconds": 0.0, "decode_iterations": 0, "denoiser_branch_evaluations": 0,
             "peak_memory_bytes": 0, "count": 0}
    per_run = []
    for run, evaluation in zip(runs, evaluations):
        config = read_json(run / "resolved_config.json")
        if config["protocol"].get("task") != "pope":
            raise ValueError(f"Run is not a POPE task: {run}")
        meta, rows = select_evaluation(run, evaluation)
        per_run.append({"run": str(run), "evaluation_fingerprint": meta["evaluation_fingerprint"],
                        "scored": len(rows)})
        for row in rows:
            terminal = ("success", "budget_exhausted") if (
                config["protocol"]["strict_parser_id"] == "pope_final_yes_no_v1"
                and config.get("cdapc", {}).get("budget_limited", False)) else ("success",)
            if row["generation_status"] not in terminal:
                continue
            key = row["sample_id"]
            if key in covered:
                overlap.add(key)
            covered.add(key)
            texts_by_key[key] = row.get("normalized", {}).get("text", "")
            generation_statuses[key] = row["generation_status"]
        for path in (run / "predictions").glob("*.json"):
            prediction = read_json(path)
            predictions_by_key[prediction["sample_id"]] = prediction
            if prediction.get("status") in ("success", "budget_exhausted"):
                costs["generation_seconds"] += prediction.get("generation_seconds", 0)
                costs["decode_iterations"] += prediction.get("decode_iterations", 0)
                costs["denoiser_branch_evaluations"] += prediction.get("cost", {}).get("denoiser_branch_evaluations", 0)
                costs["peak_memory_bytes"] = max(costs["peak_memory_bytes"], prediction.get("peak_memory_bytes", 0))
                costs["count"] += 1
    if overlap:
        raise ValueError(f"Runs overlap on {len(overlap)} inputs")
    data_root = ROOT / (read_json(runs[0] / "resolved_config.json")["protocol"]["data_root"] if runs else "data/pope_yesno")
    reasoning = bool(runs) and read_json(runs[0] / "resolved_config.json")["protocol"]["strict_parser_id"] == "pope_final_yes_no_v1"
    if reasoning and native_export:
        raise ValueError("Reasoning reports require a matching reasoning baseline; do not mix the plain native export")
    if runs:
        protocols = [read_json(run / "resolved_config.json")["protocol"] for run in runs]
        if any(p != protocols[0] for p in protocols[1:]):
            raise ValueError("Cannot combine runs with different protocols")
        report["prompt_id"] = protocols[0]["prompt_id"]
        report["answer_scope"] = "extracted_final_answer" if reasoning else "raw_official"
    all_rows = read_json(data_root / "rows.json")
    expected = {r["input_key"] for r in all_rows}
    if runs:
        missing = expected - covered
        if missing and not allow_partial:
            raise ValueError(f"Runs cover {len(covered & expected)}/{len(expected)} inputs; use --allow-partial")
        report["partial"] = bool(missing)
        if missing:
            report["uncovered_inputs"] = len(missing)
    if runs:
        cdapc_rows = split_rows(all_rows, texts_by_key, generation_statuses)
        report["cdapc"] = {"inputs": len(covered & expected), "rows": len(cdapc_rows),
                           "splits": {split: (reasoning_split_metrics([r for r in cdapc_rows if r["split"] == split]) if reasoning else
                                              split_metrics([r for r in cdapc_rows if r["split"] == split], rule))
                                      for split in SPLITS},
                           "cost": {**costs, "mean_generation_seconds":
                                    costs["generation_seconds"] / costs["count"] if costs["count"] else None},
                           "mechanism_totals": mechanism_totals(
                               predictions_by_key.values(), statuses=("success", "budget_exhausted")),
                           "generation_status_counts": dict(Counter(generation_statuses.values())),
                           "finish_reason_counts": dict(Counter(
                               p.get("finish_reason") for p in predictions_by_key.values())),
                           "per_run": per_run}
        if reasoning:
            report["raw_official_splits"] = {split: split_metrics([r for r in cdapc_rows if r["split"] == split], rule)
                                            for split in SPLITS}
            report["raw_official_scope"] = (
                "Text-only diagnostic, including budget-exhausted text; not the primary score. "
                "Primary final-answer metrics count budget exhaustion as incorrect.")
    if native_export:
        payloads = load_native_export(native_export)
        native_texts = {key: payload["text"] for key, payload in payloads.items()}
        native_rows = split_rows(all_rows, native_texts)
        native_metrics = {split: split_metrics([r for r in native_rows if r["split"] == split], rule)
                          for split in SPLITS}
        reference = json.loads((Path(native_export) / "pope/run/metrics.json").read_text())["native"]["pope"]
        comparison = {split: {"recomputed": native_metrics[split], "reference": reference[split],
                              "equal": native_metrics[split] == reference[split]} for split in SPLITS}
        report["native"] = {"records": len(payloads), "covered_inputs": len(set(payloads) & expected),
                            "splits": native_metrics, "matches_stored_metrics": comparison,
                            "all_splits_equal": all(c["equal"] for c in comparison.values())}
        if runs:
            prompt_equal = prompt_different = 0
            same_text = same_prediction = 0
            for path_row in (r for run in runs for r in (run / "predictions").glob("*.json")):
                prediction = read_json(path_row)
                key = prediction["sample_id"]
                if key not in payloads or prediction.get("status") != "success":
                    continue
                ours = prediction.get("prompt_ids")
                ours = ours[0] if ours and isinstance(ours[0], list) else ours
                if ours == payloads[key]["input"]["prompt_ids"]:
                    prompt_equal += 1
                else:
                    prompt_different += 1
                text = texts_by_key.get(key, "")
                same_text += text == native_texts[key]
                same_prediction += rule(text) == rule(native_texts[key])
            report["parity"] = {"prompt_ids_equal": prompt_equal, "prompt_ids_different": prompt_different,
                                "same_text": same_text, "same_official_prediction": same_prediction}
            shared_keys = covered & set(payloads) & expected
            shared_rows = [r for r in all_rows if r["input_key"] in shared_keys]
            # (a) answer transitions vs native
            native_preds = {key: rule(native_texts[key]) for key in shared_keys}
            cdapc_preds = {key: rule(texts_by_key.get(key, "")) for key in shared_keys}
            changed = sorted(key for key in shared_keys if native_preds[key] != cdapc_preds[key])
            per_input_pairs = {key: (native_preds[key], cdapc_preds[key],
                                     next(r["label"] for r in shared_rows if r["input_key"] == key))
                               for key in shared_keys}
            report["transitions"] = {
                "per_input": transition_cells(per_input_pairs.values()),
                "per_split": {split: transition_cells(
                    (native_preds[r["input_key"]], cdapc_preds[r["input_key"]], r["label"])
                    for r in shared_rows if r["split"] == split) for split in SPLITS},
                "changed_input_keys": changed}
            # (b) answer_switch attribution
            switches = {key: answer_switch(predictions_by_key.get(key, {}).get("commit_log"),
                                         predictions_by_key.get(key, {}).get("first_eos_position"))
                        for key in shared_keys}
            report["answer_switch_attribution"] = {
                "changed_with_switch": sum(k in changed and switches[k] for k in shared_keys),
                "changed_without_switch": sum(k in changed and not switches[k] for k in shared_keys),
                "switch_without_change": sum(k not in changed and switches[k] for k in shared_keys)}
            # (c) confidence shift at the answer token
            alpha = read_json(runs[0] / "resolved_config.json")["cdapc"]["alpha"]
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                read_json(runs[0] / "resolved_config.json")["model"]["checkpoint"], local_files_only=True)
            decode = lambda tid: tokenizer.decode([tid])
            shifts, missing = {}, 0
            for key in sorted(shared_keys):
                prediction = predictions_by_key.get(key, {})
                first_eos = prediction.get("first_eos_position")
                commits = [e for e in prediction.get("commit_log", [])
                           if first_eos is None or e["position"] < first_eos]
                if not commits:
                    missing += 1
                    continue
                entry = min(commits, key=lambda e: e["position"])
                value = confidence_shift(entry, decode, alpha)
                if value is None:
                    missing += 1
                    continue
                shifts[key] = value
            flips = {k: v for k, v in shifts.items()
                     if (v["LO_h"] < 0) != (v["LO_I"] < 0)}
            blocked_pool = blocked_other = actually_switched = 0
            for key, value in flips.items():
                contrast_prefers_yes = value["LO_h"] > 0
                other_in_pool = value["yes_in_pool"] if contrast_prefers_yes else value["no_in_pool"]
                if not other_in_pool:
                    blocked_pool += 1
                elif switches[key]:
                    actually_switched += 1
                else:
                    blocked_other += 1
            report["confidence_shift"] = {
                **confidence_summary(list(shifts.values())),
                "n_missing": missing,
                "sign_flips": len(flips),
                "sign_flip_switched": actually_switched,
                "sign_flip_blocked_not_in_pool": blocked_pool,
                "sign_flip_blocked_other": blocked_other,
                "per_input": shifts}
            report["paired_vs_native"] = {
                split: paired_bootstrap(
                    [{"id": r["row_id"], "label": r["label"], "image_sha256": r["image_sha256"],
                      "text": native_texts[r["input_key"]]} for r in shared_rows if r["split"] == split],
                    [{"id": r["row_id"], "label": r["label"], "image_sha256": r["image_sha256"],
                      "text": texts_by_key[r["input_key"]]} for r in shared_rows if r["split"] == split],
                    rule) for split in SPLITS
                if any(r["split"] == split for r in shared_rows)}
    if output:
        atomic_json(Path(output), report)
    return report
