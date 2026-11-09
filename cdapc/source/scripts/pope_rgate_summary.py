#!/usr/bin/env python3
"""Summarize the POPE r-gate sweep across finished tau values."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cdapc_lab.data.pope import paired_bootstrap
from cdapc_lab.evaluation.scorers import load_pope_official_rule
from cdapc_lab.io.artifacts import read_json, safe_id, atomic_json

TAUS = {"0.9": "reports/pope_yesno_mmada_cdapc_L128.json",
        "0.8": "reports/pope_yesno_mmada_cdapc_L128_r80.json",
        "0.7": "reports/pope_yesno_mmada_cdapc_L128_r70.json",
        "0.6": "reports/pope_yesno_mmada_cdapc_L128_r60.json",
        "0.5": "reports/pope_yesno_mmada_cdapc_L128_r50.json"}
SPLITS = ("random", "popular", "adversarial")


def texts_for(report):
    texts = {}
    for run in report["runs"]:
        for path in (ROOT / run / "predictions").glob("*.json"):
            prediction = read_json(path)
            if prediction.get("status") == "success":
                sid = prediction["sample_id"]
                texts[sid] = prediction.get("raw_response_ids")
    return texts


def normalized_texts(report):
    # use the score rows written by the evaluation (normalized text)
    texts = {}
    for i, run in enumerate(report["runs"]):
        ev = report["cdapc"]["per_run"][i]["evaluation_fingerprint"]
        score_dir = ROOT / run / "scores" / ev
        for path in score_dir.glob("*.json"):
            if path.name == "evaluation_manifest.json":
                continue
            row = read_json(path)
            if row["generation_status"] == "success":
                texts[row["sample_id"]] = row.get("normalized", {}).get("text", "")
    return texts


def main():
    rule = load_pope_official_rule(ROOT / "upstream/pope/evaluate.py")
    all_rows = read_json(ROOT / "data/pope_yesno/rows.json")
    reports, texts = {}, {}
    for tau, path in TAUS.items():
        if not (ROOT / path).is_file():
            continue
        reports[tau] = json.loads((ROOT / path).read_text())
        texts[tau] = normalized_texts(reports[tau])
    out = {"taus_finished": sorted(reports)}
    base_tau = "0.9"
    for tau, report in reports.items():
        entry = {"runs": report["runs"],
                 "splits": report["cdapc"]["splits"],
                 "mechanism_totals": report["cdapc"]["mechanism_totals"],
                 "cost": report["cdapc"]["cost"],
                 "transitions": report["transitions"]["per_input"],
                 "transitions_per_split": report["transitions"]["per_split"],
                 "answer_switch_attribution": report["answer_switch_attribution"],
                 "confidence_shift": {k: v for k, v in report["confidence_shift"].items()
                                      if k != "per_input"},
                 "paired_vs_native": report.get("paired_vs_native"),
                 "parity": report.get("parity")}
        if base_tau in texts and tau != base_tau and set(texts[base_tau]) == set(texts[tau]):
            shared = set(texts[base_tau])
            entry["paired_vs_tau_0.9"] = {
                split: paired_bootstrap(
                    [{"id": r["row_id"], "label": r["label"], "image_sha256": r["image_sha256"],
                      "text": texts[base_tau].get(r["input_key"], "")} for r in all_rows if r["split"] == split and r["input_key"] in shared],
                    [{"id": r["row_id"], "label": r["label"], "image_sha256": r["image_sha256"],
                      "text": texts[tau].get(r["input_key"], "")} for r in all_rows if r["split"] == split and r["input_key"] in shared],
                    rule)
                for split in SPLITS}
        # finish_reason counts
        counts = {}
        for run in report["runs"]:
            for path in (ROOT / run / "predictions").glob("*.json"):
                fr = read_json(path).get("finish_reason", "missing")
                counts[fr] = counts.get(fr, 0) + 1
        entry["finish_reason_counts"] = counts
        out[tau] = entry
    atomic_json(ROOT / "reports/pope_rgate_sweep.json", out)
    print(json.dumps(out["taus_finished"]))


if __name__ == "__main__":
    main()
