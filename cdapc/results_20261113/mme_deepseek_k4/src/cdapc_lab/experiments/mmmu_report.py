"""MMMU-val report: judge/official/strict accuracy per method + comparison vs native."""
import json
from pathlib import Path

import numpy as np

from cdapc_lab.io.artifacts import atomic_json, read_json, safe_id
from cdapc_lab.experiments.pope_report import mechanism_totals, answer_switch
from cdapc_lab.evaluation.llm_judge import judge_spec, load_judge_rules


def question_bootstrap(rows_a, rows_b, repeats=2000, seed=42):
    """rows: {id: correct (0/1/None-missing)}; CI of mean(B-A) over resampled ids."""
    keys = sorted(set(rows_a) & set(rows_b))
    keys = [k for k in keys if rows_a[k] is not None and rows_b[k] is not None]
    if not keys:
        return None
    a = np.array([rows_a[k] for k in keys], dtype=float)
    b = np.array([rows_b[k] for k in keys], dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(keys), (repeats, len(keys)))
    deltas = (b[indices] - a[indices]).mean(1)
    return {"n": len(keys), "mean_delta": float(b.mean() - a.mean()),
            "ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
            "unit": "question", "repeats": repeats, "seed": seed}


def method_block(runs, evaluations, judge_name, all_ids, gold, samples_meta):
    from cdapc_lab.experiments.score import select_evaluation
    spec, spec_hash = judge_spec(judge_name)
    per = {}
    costs = {"generation_seconds": 0.0, "decode_iterations": 0,
             "denoiser_branch_evaluations": 0, "count": 0, "peak_memory_bytes": 0}
    finish = {}
    trunc = {"no_eos_in_canvas": 0, "contains_think_close": 0}
    predictions_by_id = {}
    judge_errors = judge_invalid = judged = 0
    for i, run in enumerate(runs):
        run = Path(run)
        meta, rows = select_evaluation(run, evaluations[i] if evaluations else None)
        ev_dir = run / "scores" / meta["evaluation_fingerprint"]
        judge_dir = run / "judge" / spec_hash
        for row in rows:
            sid = row["sample_id"]
            entry = per.setdefault(sid, {})
            meta = gold[sid].metric_metadata
            entry["question_type"] = meta.get("question_type")
            entry["n_images"] = meta.get("n_images", 1)
            successful = row["generation_status"] == "success"
            entry["official"] = float(row["official"]) if successful else 0.0
            entry["strict"] = float(row["strict_accuracy"]) if successful else 0.0
            jpath = judge_dir / (safe_id(sid) + ".json")
            if jpath.is_file():
                jr = read_json(jpath)
                if jr.get("status") == "judge_error":
                    judge_errors += 1
                else:
                    entry["judge"] = float(jr["score"]) if successful else 0.0
                    judged += 1
                    judge_invalid += int(not jr.get("strict_judge", False))
        for path in (run / "predictions").glob("*.json"):
            p = read_json(path)
            predictions_by_id[p["sample_id"]] = p
            if p.get("status") in ("success", "budget_exhausted"):
                costs["generation_seconds"] += p.get("generation_seconds", 0)
                costs["decode_iterations"] += p.get("decode_iterations", 0)
                costs["denoiser_branch_evaluations"] += p.get("cost", {}).get("denoiser_branch_evaluations", 0)
                costs["peak_memory_bytes"] = max(costs["peak_memory_bytes"], p.get("peak_memory_bytes", 0))
                costs["count"] += 1
                finish[p.get("finish_reason", "missing")] = finish.get(p.get("finish_reason", "missing"), 0) + 1
                sfile = ev_dir / (safe_id(p["sample_id"]) + ".json")
                text = read_json(sfile).get("normalized", {}).get("text", "") if sfile.is_file() else ""
                trunc["contains_think_close"] += int("</think>" in text)
                if p.get("first_eos_position") is None:
                    trunc["no_eos_in_canvas"] += 1
    covered = set(per) & set(all_ids)
    def subset(cond):
        rows = {sid: per[sid] for sid in covered if cond(per[sid])}
        out = {}
        for metric in ("judge", "official", "strict"):
            vals = [r[metric] for r in rows.values() if metric in r]
            out[metric] = {"n": len(vals), "accuracy": float(np.mean(vals)) if vals else None}
        out["n"] = len(rows)
        return out
    blocks = {"all": subset(lambda r: True),
              "single_image": subset(lambda r: r["n_images"] == 1),
              "per_type": {qt: subset(lambda r, q=qt: r["question_type"] == q)
                           for qt in ("multiple-choice", "open")}}
    switches = {sid: answer_switch(predictions_by_id.get(sid, {}).get("commit_log"),
                                   predictions_by_id.get(sid, {}).get("first_eos_position"))
                for sid in covered}
    return {"covered": len(covered), "per_sample": per, "blocks": blocks,
            "mechanism_totals": mechanism_totals(
                predictions_by_id.values(), statuses=("success", "budget_exhausted")),
            "answer_switch_inputs": sum(switches.values()),
            "cost": {**costs, "mean_generation_seconds":
                     costs["generation_seconds"] / costs["count"] if costs["count"] else None},
            "finish_reason_counts": finish, "truncation": trunc,
            "judged": judged, "judge_errors": judge_errors, "judge_invalid": judge_invalid}


def mmmu_report(runs, evaluations=None, native_runs=None, native_evaluations=None,
                judge="deepseek-flash", allow_partial=False, output=None):
    if not runs and not native_runs:
        raise ValueError("Provide --runs and/or --native-runs")
    any_run = Path(runs[0] if runs else native_runs[0])
    protocol = read_json(any_run / "resolved_config.json")["protocol"]
    if protocol.get("task") != "mmmu":
        raise ValueError("mmmu-report requires MMMU runs")
    root = Path(protocol["data_root"])
    gold = {r["sample_id"]: r for r in read_json(root / "gold.json")}
    gold = {k: type("G", (), {"answer": v["answer"], "metric_metadata": v["metric_metadata"]}) for k, v in gold.items()}
    samples_meta = read_json(root / "samples.json")
    all_ids = read_json(Path(any_run).parent.parent / "manifests" / "mmmu_val_all_ids.json")
    report = {"runs": [str(r) for r in runs], "native_runs": [str(r) for r in (native_runs or [])],
              "judge": judge}
    if runs:
        report["cdapc"] = method_block(runs, evaluations, judge, all_ids, gold, samples_meta)
        if not allow_partial and report["cdapc"]["covered"] != len(all_ids):
            raise ValueError(f"Coverage {report['cdapc']['covered']} != {len(all_ids)} (use --allow-partial)")
    if native_runs:
        report["native"] = method_block(native_runs, native_evaluations, judge, all_ids, gold, samples_meta)
    if runs and native_runs:
        shared = set(report["cdapc"]["per_sample"]) & set(report["native"]["per_sample"]) & set(all_ids)
        report["paired_vs_native"] = {}
        for metric in ("judge", "official", "strict"):
            report["paired_vs_native"][metric] = {
                "all": question_bootstrap({s: report["native"]["per_sample"][s].get(metric) for s in shared},
                                          {s: report["cdapc"]["per_sample"][s].get(metric) for s in shared}),
                "single_image": question_bootstrap(
                    {s: report["native"]["per_sample"][s].get(metric) for s in shared
                     if report["native"]["per_sample"][s]["n_images"] == 1},
                    {s: report["cdapc"]["per_sample"][s].get(metric) for s in shared
                     if report["cdapc"]["per_sample"][s]["n_images"] == 1})}
        cells = {"both_correct": 0, "fixed": 0, "broken": 0, "both_wrong": 0}
        for sid in shared:
            n, c = report["native"]["per_sample"][sid].get("judge"), report["cdapc"]["per_sample"][sid].get("judge")
            if n is None or c is None:
                continue
            cells["both_correct" if n and c else "fixed" if c else "broken" if n else "both_wrong"] += 1
        report["judge_transitions_vs_native"] = cells
    if output:
        atomic_json(output, report)
    return report
