"""Score the four prompt variants of the POPE prompt diagnostic.

Reads samples.jsonl plus each variant's resumable SQLite and writes
metrics.json + RESULTS.md in this directory. Official parsing is unchanged:
common.metrics.official_pope_answer / pope_metrics.

Usage: python results/pope_prompt_diagnostic/score.py [--determinism-only]
"""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(ROOT))

import run_diag
from common.eos import answer_ids
from common.metrics import official_pope_answer, pope_metrics

SPLITS = ("random", "popular", "adversarial")
LABELS = ("yes", "no")
PAUSED_SHARDS = (ROOT / "results/pope_coco_nosuffix_256_256/coco_shard0/predictions.sqlite",
                 ROOT / "results/pope_coco_nosuffix_256_256/coco_shard1/predictions.sqlite")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_records(variant):
    path = BASE / variant / "predictions.sqlite"
    if not path.exists():
        return {}
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    try:
        return {key: json.loads(payload) for key, payload in connection.execute(
            "SELECT input_key,payload FROM predictions")}
    finally:
        connection.close()


def extract_answer(variant, record):
    """Return (answer_text, info). B strips through the last </think>."""
    text = record["text"]
    if variant != "B":
        return text, {"think_closed": None}
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip(), {"think_closed": True}
    return text, {"think_closed": False}


def build_scored(variant, records):
    scored, missing = [], []
    for row in run_diag.read_samples():
        key = run_diag.input_key(run_diag.variant_sample(row, variant))
        record = records.get(key)
        if record is None:
            missing.append(row["id"])
            continue
        answer, info = extract_answer(variant, record)
        scored.append({
            "id": row["id"], "split": row["split"], "label": row["label"], "q0": row["q0"],
            "answer": answer, "generated_text": record["text"], "parsed": official_pope_answer(answer),
            "think_closed": info["think_closed"],
            "answer_token_length": record["answer_token_length"],
            "eos_found": record["eos_found"],
            "executed_steps": record["executed_steps"],
            "end_to_end_seconds": record["end_to_end_seconds"],
            "early_stopped": record.get("early_stopped"),
            "stop_reason": record.get("stop_reason"),
        })
    return scored, missing


def pope_rows(scored, answer_key="answer"):
    return [{"text": r[answer_key], "label": r["label"], "split": r["split"]} for r in scored]


def metrics_block(scored, answer_key="answer"):
    rows = pope_rows(scored, answer_key)
    return {
        "rows": len(rows),
        "overall": pope_metrics(rows),
        "splits": {s: pope_metrics([r for r in rows if r["split"] == s]) for s in SPLITS},
    }


def diagnostics(scored):
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
        "rows": n,
        "yes_start_parsed_no": yes_start_no,
        "no_start_parsed_yes": no_start_yes,
        "not_starting_with_yes_or_no": neither,
        "answer_token_length_mean": statistics.mean(r["answer_token_length"] for r in scored),
        "answer_token_length_median": statistics.median(r["answer_token_length"] for r in scored),
        "eos_found_rate": sum(bool(r["eos_found"]) for r in scored) / n,
        "executed_steps_mean": statistics.mean(r["executed_steps"] for r in scored),
        "end_to_end_seconds_mean": statistics.mean(r["end_to_end_seconds"] for r in scored),
    }


def think_stats(scored):
    n = len(scored)
    if not n:
        return {"rows": 0}
    with_open = sum("<think>" in r["generated_text"] for r in scored)
    with_close = sum("</think>" in r["generated_text"] for r in scored)
    return {
        "rows": n,
        "with_think_tag": with_open,
        "with_think_tag_ratio": with_open / n,
        "with_think_close": with_close,
        "with_think_close_ratio": with_close / n,
        "think_close_missing": sum(r["think_closed"] is False for r in scored),
    }


def paired_vs_a(a_scored, v_scored):
    a_parsed = {r["id"]: r["parsed"] for r in a_scored}
    out = {"changed": 0, "fixed": 0, "broken": 0,
           "by_label": {label: {"fixed": 0, "broken": 0} for label in LABELS},
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


def examples(scored, limit=3):
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


def determinism_report():
    """Compare variant-A generated_ids (through EOS) with the paused run shards."""
    records = load_records("A")
    if not records:
        return {"a_committed": 0, "compared": 0, "identical": 0, "note": "no A predictions yet"}
    a_answer_ids = {key: answer_ids(record) for key, record in records.items()}
    compared = identical = 0
    mismatches = []
    for shard_path in PAUSED_SHARDS:
        if not shard_path.exists():
            continue
        connection = sqlite3.connect(f"file:{shard_path}?mode=ro", uri=True, timeout=30)
        try:
            for key, payload in connection.execute(
                    "SELECT input_key,payload FROM predictions WHERE method='native'"):
                if key not in a_answer_ids:
                    continue
                compared += 1
                if answer_ids(json.loads(payload)) == a_answer_ids[key]:
                    identical += 1
                else:
                    mismatches.append(key)
        finally:
            connection.close()
    return {"a_committed_unique": len(records), "compared": compared,
            "identical": identical, "mismatched": len(mismatches),
            "mismatch_keys": mismatches[:10]}


def fmt_pct(value):
    return f"{100 * value:.2f}" if value is not None else "-"


def metric_cells(block):
    m = block["overall"]
    return [fmt_pct(m["accuracy"]), fmt_pct(m["f1"]), fmt_pct(m["precision"]),
            fmt_pct(m["recall"]), fmt_pct(m["yes_ratio"]), fmt_pct(m["invalid_ratio"])]


def write_results(report):
    lines = ["# POPE prompt diagnostic — native, 256/256/32", "",
             f"{report['samples']} rows (32 per split x label), native only, seed 42, "
             "crop 512, temperature 0, CFG 0, low_confidence, safe EOS stopping.",
             "Image: <|mmu|><|soi|>[1024 codes]<|eoi|> + chat/training text as in run_diag.py.",
             "Scorer: unchanged official POPE parser (common.metrics.pope_metrics).",
             "Values are percentages; per-split rows plus an all-rows row per variant.", ""]
    lines += ["## Overall metrics", "",
              "| Variant | Rows | Accuracy | F1 | Precision | Recall | Yes ratio | Invalid ratio |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for variant, entry in report["variants"].items():
        flag = "" if entry["complete"] else f" (incomplete: {entry['rows_scored']}/{report['samples']})"
        lines.append(f"| {variant}{flag} | {entry['rows_scored']} | "
                     + " | ".join(metric_cells(entry["metrics"])) + " |")
    lines += ["", "## Per split", "",
              "| Variant | Split | Accuracy | F1 | Precision | Recall | Yes ratio | Invalid ratio |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for variant, entry in report["variants"].items():
        for split in SPLITS:
            m = entry["metrics"]["splits"][split]
            lines.append(f"| {variant} | {split} | " + " | ".join([
                fmt_pct(m["accuracy"]), fmt_pct(m["f1"]), fmt_pct(m["precision"]),
                fmt_pct(m["recall"]), fmt_pct(m["yes_ratio"]), fmt_pct(m["invalid_ratio"])]) + " |")
        m = entry["metrics"]["overall"]
        lines.append(f"| {variant} | all | " + " | ".join([
            fmt_pct(m["accuracy"]), fmt_pct(m["f1"]), fmt_pct(m["precision"]),
            fmt_pct(m["recall"]), fmt_pct(m["yes_ratio"]), fmt_pct(m["invalid_ratio"])]) + " |")
    lines += ["", "## Answer diagnostics", "",
              "| Variant | yes-start parsed no | no-start parsed yes | neither start | "
              "tok len mean | tok len median | EOS rate | steps mean | e2e mean s |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for variant, entry in report["variants"].items():
        d = entry["diagnostics"]
        if d.get("rows"):
            lines.append("| {} | {} | {} | {} | {:.2f} | {} | {:.3f} | {:.1f} | {:.2f} |".format(
                variant, d["yes_start_parsed_no"], d["no_start_parsed_yes"],
                d["not_starting_with_yes_or_no"], d["answer_token_length_mean"],
                d["answer_token_length_median"], d["eos_found_rate"],
                d["executed_steps_mean"], d["end_to_end_seconds_mean"]))
        else:
            lines.append(f"| {variant} | - | - | - | - | - | - | - | - |")
    if "B" in report["variants"] and report["variants"]["B"].get("think"):
        t = report["variants"]["B"]["think"]
        sec = report["variants"]["B"].get("fulltext_secondary", {}).get("metrics", {}).get("overall", {})
        lines += ["", "## Variant B thinking format", "",
                  f"- rows with `<think>`: {t['with_think_tag']}/{t['rows']} "
                  f"({fmt_pct(t['with_think_tag_ratio'])}%)",
                  f"- rows with `</think>`: {t['with_think_close']}/{t['rows']} "
                  f"({fmt_pct(t['with_think_close_ratio'])}%)",
                  f"- think_close_missing (full text used as answer): {t['think_close_missing']}",
                  f"- secondary parse of FULL B text — accuracy {fmt_pct(sec.get('accuracy'))}, "
                  f"F1 {fmt_pct(sec.get('f1'))}, yes ratio {fmt_pct(sec.get('yes_ratio'))}"]
    lines += ["", "## Paired parsed-answer changes vs A", "",
              "| Variant | Changed | Fixed | Broken | yes-label fixed | yes-label broken | "
              "no-label fixed | no-label broken |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for variant in ("B", "C", "D"):
        entry = report["variants"].get(variant)
        if not entry or "paired_vs_A" not in entry:
            continue
        p = entry["paired_vs_A"]
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            variant, p["changed"], p["fixed"], p["broken"],
            p["by_label"]["yes"]["fixed"], p["by_label"]["yes"]["broken"],
            p["by_label"]["no"]["fixed"], p["by_label"]["no"]["broken"]))
    lines += ["", "## Example outputs (first 3 label=no rows per variant)", ""]
    for variant, entry in report["variants"].items():
        lines.append(f"### {variant}")
        for ex in entry["examples_label_no"]:
            lines.append(f"- `{ex['id']}` ({ex['split']}, label=no) q0: {ex['q0']}")
            lines.append(f"  - answer: {ex['answer']!r} parsed={ex['parsed']} "
                         f"eos={ex['eos_found']} steps={ex['executed_steps']} "
                         f"e2e={ex['end_to_end_seconds']:.2f}s")
            if ex["generated_text"] != ex["answer"]:
                lines.append(f"  - generated: {ex['generated_text']!r}")
        lines.append("")
    det = report.get("determinism_vs_paused_run")
    if det:
        lines += ["## Determinism vs paused no-suffix run", "",
                  f"- A committed unique inputs: {det.get('a_committed_unique', det.get('a_committed'))}",
                  f"- compared: {det.get('compared')}, identical: {det.get('identical')}, "
                  f"mismatched: {det.get('mismatched')}"]
        if det.get("mismatch_keys"):
            lines.append(f"- mismatch keys: {det['mismatch_keys']}")
        lines.append("")
    (BASE / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--determinism-only", action="store_true")
    args = parser.parse_args()
    if args.determinism_only:
        print(json.dumps(determinism_report(), indent=2))
        return
    samples = run_diag.read_samples()
    scored_by_variant = {}
    report = {"experiment": "pope_prompt_diagnostic", "samples": len(samples),
              "generation": run_diag.GENERATION, "seed": run_diag.SEED,
              "score_sha256": digest(__file__), "variants": {}}
    for variant in run_diag.VARIANTS:
        records = load_records(variant)
        scored, missing = build_scored(variant, records)
        scored_by_variant[variant] = scored
        status_path = BASE / variant / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        entry = {
            "complete": not missing,
            "rows_scored": len(scored), "missing_ids": missing,
            "unique_committed": status.get("committed"),
            "metrics": metrics_block(scored),
            "diagnostics": diagnostics(scored),
            "examples_label_no": examples(scored),
        }
        if variant == "B":
            entry["think"] = think_stats(scored)
            full_scored = [dict(row, answer=row["generated_text"],
                                 parsed=official_pope_answer(row["generated_text"]))
                           for row in scored]
            entry["fulltext_secondary"] = {
                "metrics": metrics_block(full_scored), "diagnostics": diagnostics(full_scored)}
        report["variants"][variant] = entry
    if scored_by_variant.get("A"):
        for variant in ("B", "C", "D"):
            if scored_by_variant.get(variant):
                report["variants"][variant]["paired_vs_A"] = paired_vs_a(
                    scored_by_variant["A"], scored_by_variant[variant])
    report["determinism_vs_paused_run"] = determinism_report()
    report["manifest"] = json.loads((BASE / "manifest.json").read_text())
    (BASE / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    write_results(report)
    for variant, entry in report["variants"].items():
        m = entry["metrics"]["overall"]
        print(f"{variant}: rows={entry['rows_scored']} acc={fmt_pct(m['accuracy'])} "
              f"f1={fmt_pct(m['f1'])} yes_ratio={fmt_pct(m['yes_ratio'])} "
              f"complete={entry['complete']}")
    print("wrote metrics.json and RESULTS.md")


if __name__ == "__main__":
    main()
