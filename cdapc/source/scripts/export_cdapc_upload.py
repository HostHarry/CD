#!/usr/bin/env python3
"""Export normalized per-sample answers for every completed CD-APC condition.

Offline package exporter for the cd-mmmu-upload/cdapc results archive. It reads
saved run artifacts only -- run manifests, resolved configs, prediction files,
score files, judge outputs, and the stored report JSONs used for cross-checks --
and writes JSONL + TXT answer files plus answers/manifest.json into the
cd-mmmu-upload/cdapc/answers tree.

It never writes into cdapc-experiments or cdapc-mme, never invokes inference,
scoring, or judge commands, and performs no network or GPU work. Run it with
the project virtualenv:

    .venv/bin/python scripts/export_cdapc_upload.py
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
MME = ROOT.parent / "cdapc-mme"
DEFAULT_OUT = Path("/root/autodl-tmp/cd-mmmu-upload/cdapc/answers")
REPORTS = ROOT / "reports"
JUDGE_DIR = REPORTS / "pope_think_llm_judge"
PILOT_DIR = REPORTS / "k_commit_pilot"

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(MME / "src"))

from cdapc_lab.data.pope import split_metrics  # noqa: E402
from cdapc_lab.evaluation.scorers import load_pope_official_rule  # noqa: E402
from cdapc_lab.evaluation.pope_llm_judge import request_for  # noqa: E402
from cdapc_lab.experiments.pope_report import reasoning_split_metrics  # noqa: E402
from cdapc_lab.experiments.score import pope_metrics  # noqa: E402
from cdapc_lab.io.artifacts import digest, safe_id  # noqa: E402

SPLITS = ("random", "popular", "adversarial")
POPE_RULE = load_pope_official_rule(ROOT / "upstream/pope/evaluate.py")

# Protocol constants mirrored from src/cdapc_lab (verified against source below).
POPE_DIRECT_TEMPLATE = "{question} Please answer yes or no."
POPE_THINK_TEMPLATE = ("{question}\nAfter </think>, end with exactly "
                       "Final answer: Yes or Final answer: No.")
CHAIR_QUESTION = "Please describe this image in detail."


def thinking_instruction():
    """OFFICIAL_THINKING_INSTRUCTION from models/mmada.py, read not copied."""
    source = (ROOT / "src/cdapc_lab/models/mmada.py").read_text()
    match = re.search(r'OFFICIAL_THINKING_INSTRUCTION\s*=\s*"((?:[^"\\]|\\.)*)"', source)
    if not match:
        raise ValueError("OFFICIAL_THINKING_INSTRUCTION not found in mmada.py")
    return match.group(1)


INSTRUCTION = thinking_instruction()


def jload(path):
    return json.loads(Path(path).read_text())


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def rel(path, root=ROOT):
    path = Path(path)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def isclose(a, b, tol=1e-12):
    if a is None or b is None:
        return a is b
    return abs(a - b) <= tol


def check_run_config(run, expected, label):
    """Assert the resolved config fields that define the condition."""
    config = jload(Path(run) / "resolved_config.json")
    protocol, cdapc = config["protocol"], config["cdapc"]
    checks = {
        "task": protocol.get("task"),
        "prompt_id": protocol.get("prompt_id"),
        "strict_parser_id": protocol.get("strict_parser_id"),
        "canvas_length": protocol.get("canvas_length"),
        "data_root": protocol.get("data_root"),
        "r_gate": cdapc.get("r_gate", {}).get("threshold"),
        "iteration_limit": cdapc.get("iteration_limit"),
        "max_commit": cdapc.get("max_commit_per_iteration"),
        "window": cdapc.get("window", {}).get("width"),
        "alpha": cdapc.get("alpha"),
        "beta": cdapc.get("beta"),
        "u_gate": cdapc.get("u_gate", {}).get("threshold"),
        "stop_rule": cdapc.get("stop_rule"),
        "budget_limited": cdapc.get("budget_limited"),
    }
    for key, want in expected.items():
        got = checks[key]
        if got != want:
            raise AssertionError(f"{label} {run}: {key}={got!r} expected {want!r}")
    return config


def scan_shard(run, manifest_ids, expected_config, label, allowed_statuses,
               workspace=ROOT, evaluation=None):
    """Read one shard run; return {sample_id: (prediction, score)}.

    Asserts run_manifest status complete, resolved config matches the
    condition, prediction ids equal the shard manifest ids with no duplicates,
    every prediction status is allowed, and score files agree with predictions.
    """
    run = Path(run)
    root = Path(workspace)
    manifest = jload(run / "run_manifest.json")
    if manifest.get("status") != "complete":
        raise AssertionError(f"{label} {run}: run_manifest status "
                             f"{manifest.get('status')!r} != 'complete'")
    check_run_config(run, expected_config, label)
    if evaluation is not None:
        score_dir = run / "scores" / evaluation
        if not (score_dir / "evaluation_manifest.json").is_file():
            raise AssertionError(f"{label} {run}: missing evaluation {evaluation}")
    else:
        score_dirs = [p for p in (run / "scores").iterdir()
                      if (p / "evaluation_manifest.json").is_file()]
        if len(score_dirs) != 1:
            raise AssertionError(f"{label} {run}: {len(score_dirs)} score fingerprints")
        score_dir = score_dirs[0]
    found = {}
    for path in sorted((run / "predictions").glob("*.json")):
        prediction = jload(path)
        sid = prediction["sample_id"]
        if sid in found:
            raise AssertionError(f"{label} {run}: duplicate sample {sid}")
        if prediction.get("status") not in allowed_statuses:
            raise AssertionError(f"{label} {run}/{sid}: status "
                                 f"{prediction.get('status')!r}")
        score_path = score_dir / (safe_id(sid) + ".json")
        if not score_path.is_file():
            raise AssertionError(f"{label} {run}/{sid}: missing score file")
        score = jload(score_path)
        if score.get("sample_id") != sid:
            raise AssertionError(f"{label} {run}/{sid}: score id mismatch")
        if score.get("generation_status") != prediction.get("status"):
            raise AssertionError(f"{label} {run}/{sid}: generation_status mismatch")
        found[sid] = (prediction, score)
    expected = set(manifest_ids)
    if set(found) != expected:
        raise AssertionError(
            f"{label} {run}: coverage {len(found)}/{len(expected)}; "
            f"missing={len(expected - set(found))} extra={len(set(found) - expected)}")
    return found, score_dir.name, rel(run, root)


def shard_ids(config, workspace=ROOT):
    ids_file = config["run"]["sample_ids_file"]
    return jload(Path(workspace) / ids_file)


def load_eval_rows(data_root, workspace=ROOT):
    rows = jload(Path(workspace) / data_root / "rows.json")
    by_key = defaultdict(list)
    for row in rows:
        by_key[row["input_key"]].append(row)
    return rows, by_key


def check_key_rows(key_rows, label):
    """All eval rows of one input share question/image/label."""
    first = key_rows[0]
    for row in key_rows[1:]:
        for field in ("question", "image_sha256", "image", "label"):
            if row[field] != first[field]:
                raise AssertionError(f"{label} {key_rows[0]['input_key']}: "
                                     f"{field} differs across eval rows")
    return first


def pope_prompt(question, prompt_id):
    if prompt_id == "pope_yesno_v1":
        return POPE_DIRECT_TEMPLATE.format(question=question)
    if prompt_id == "pope_yesno_reasoning_v1":
        return INSTRUCTION + "\n" + POPE_THINK_TEMPLATE.format(question=question)
    raise ValueError(f"Unknown POPE prompt_id {prompt_id}")


def pope_row(sample_id, key_rows, first, prediction, score, prompt_id,
             dataset, condition, judge_entry=None, extra=None):
    """Build one exported POPE JSONL record."""
    text = score["normalized"]["text"]
    status = score["generation_status"]
    parsed = score.get("parsed") or {}
    record = {
        "sample_id": sample_id,
        "group_id": score.get("group_id"),
        "image_sha256": first["image_sha256"],
        "image": first["image"],
        "question": first["question"],
        "prompt": pope_prompt(first["question"], prompt_id),
        "prompt_id": prompt_id,
        "dataset": dataset,
        "condition": condition,
        "text": text,
        "generation_status": status,
        "decode_iterations": prediction["decode_iterations"],
        "emitted_tokens_before_eos": prediction["emitted_tokens_before_eos"],
        "finish_reason": prediction["finish_reason"],
        "source_run": rel(prediction["_run"]),
    }
    rows = []
    for row in key_rows:
        entry = {"row_id": row["row_id"], "split": row["split"],
                 "question_id": row["question_id"], "label": row["label"]}
        if prompt_id == "pope_yesno_v1":
            entry["scores"] = {"official": score["official"]}
            record.setdefault("official_prediction", score.get("official_prediction"))
            record.setdefault("strict_status", parsed.get("status"))
            record.setdefault("strict_prediction", parsed.get("prediction"))
            record.setdefault("strict_correct", score.get("strict_accuracy"))
        else:
            strict_ok = int(parsed.get("prediction") is not None
                            and parsed.get("prediction") == row["label"]) \
                if status == "success" else 0
            scores = {"strict": strict_ok}
            if judge_entry is not None:
                scores["judge"] = judge_entry["_row_scores"].get(row["row_id"], 0)
            entry["scores"] = scores
            record.setdefault("strict_status", parsed.get("status"))
            record.setdefault("strict_prediction", parsed.get("prediction"))
            record.setdefault("strict_correct", score.get("strict_accuracy"))
            if judge_entry is not None:
                record.setdefault("judge_prediction", judge_entry["judge_prediction"])
                record.setdefault("judge_status", judge_entry["judge_status"])
                record.setdefault("judge_correct", judge_entry["judge_correct"])
                record.setdefault("judge_evidence", judge_entry.get("evidence", ""))
                record.setdefault("judge_request_sha256", judge_entry.get("request_sha256"))
        rows.append(entry)
    record["rows"] = rows
    if extra:
        record.update(extra)
    return record


def assemble_pope_condition(name, dataset, kind, tau_or_k, report_path,
                            judge_path=None, expected_unique=None):
    """Export one POPE direct/think condition; returns (records, meta, checks)."""
    report = jload(report_path)
    cdapc = report["cdapc"]
    prompt_id = cdapc.get("prompt_id") or report.get("prompt_id")
    if prompt_id is None:
        config = jload(ROOT / cdapc["per_run"][0]["run"] / "resolved_config.json")
        prompt_id = config["protocol"]["prompt_id"]
    data_root = jload(ROOT / cdapc["per_run"][0]["run"] / "resolved_config.json")[
        "protocol"]["data_root"]
    all_rows, by_key = load_eval_rows(data_root)
    expected = {"task": "pope", "prompt_id": prompt_id,
                "strict_parser_id": ("strict_yes_no_v1" if kind == "direct"
                                     else "pope_final_yes_no_v1"),
                "canvas_length": 128, "data_root": data_root,
                "r_gate": tau_or_k if kind != "pilot" else 0.7,
                "iteration_limit": 128, "max_commit": 16 if kind != "pilot" else int(tau_or_k),
                "window": 16, "alpha": 0.5, "beta": 0.1, "u_gate": 0.1,
                "stop_rule": "eos_prefix_complete", "budget_limited": False}
    found, fingerprints, run_rels = {}, {}, []
    manifest_order = []
    seen_ids = set()
    for item in cdapc["per_run"]:
        run = ROOT / item["run"]
        config = jload(run / "resolved_config.json")
        ids = shard_ids(config)
        shard, fp, run_rel = scan_shard(
            run, ids, expected, name, ("success", "budget_exhausted"),
            evaluation=item.get("evaluation_fingerprint"))
        for sid, pair in shard.items():
            if sid in seen_ids:
                raise AssertionError(f"{name}: input {sid} covered by two runs")
            seen_ids.add(sid)
            found[sid] = pair
            prediction, _ = pair
            prediction["_run"] = run
        fingerprints[item["run"]] = fp
        run_rels.append(run_rel)
        manifest_order.extend(ids)
    if len(manifest_order) != len(set(manifest_order)):
        raise AssertionError(f"{name}: shard manifests overlap")
    if set(manifest_order) != set(found):
        raise AssertionError(f"{name}: manifest ids != scored ids")
    judge_entries = None
    if judge_path is not None:
        judge_entries = verify_think_judge(jload(judge_path), found, by_key, name)
    records = []
    for sid in manifest_order:
        prediction, score = found[sid]
        key_rows = by_key[sid]
        first = check_key_rows(key_rows, name)
        judge_entry = judge_entries.get(sid) if judge_entries else None
        if judge_entry is not None:
            for row in key_rows:
                judge_entry["_row_scores"][row["row_id"]] = int(
                    judge_entry["judge_prediction"] is not None
                    and judge_entry["judge_prediction"] == row["label"])
        records.append(pope_row(sid, key_rows, first, prediction, score,
                                prompt_id, dataset, name,
                                judge_entry=judge_entry))
    # Coverage of the full eval set: every eval-row input_key must be exported.
    all_keys = {row["input_key"] for row in all_rows}
    covered_keys = set(found)
    if covered_keys != all_keys:
        raise AssertionError(f"{name}: uncovered inputs "
                             f"{len(all_keys - covered_keys)}")
    if expected_unique is not None:
        assert len(records) == expected_unique, (name, len(records))
    meta = {"report": rel(report_path), "runs": run_rels,
            "evaluations": fingerprints, "prompt_id": prompt_id,
            "data_root": data_root,
            "unique_inputs": len(records),
            "evaluation_rows": len(all_rows),
            "covered_input_keys": len(covered_keys)}
    checks = {"strict": crosscheck_pope_strict(records, all_rows, by_key,
                                              kind, report, name),
              "official_splits": None}
    if kind == "direct":
        checks["official_splits"] = crosscheck_pope_official(
            records, all_rows, by_key, report, name)
    if judge_path is not None:
        checks["judge"] = crosscheck_pope_judge(records, all_rows, by_key,
                                                jload(judge_path), name)
    return records, meta, checks


def verify_think_judge(judge_report, found, by_key, name):
    """Verify judge per_input against cache + score files; return {sid: entry}."""
    if judge_report.get("status") != "complete":
        raise AssertionError(f"{name}: judge report status "
                             f"{judge_report.get('status')!r}")
    per_input = judge_report["per_input"]
    if judge_report.get("judged_unique") != len(per_input):
        raise AssertionError(f"{name}: judged_unique mismatch")
    cache = JUDGE_DIR / "cache"
    entries = {}
    for sid, entry in per_input.items():
        if sid not in found:
            raise AssertionError(f"{name}: judge input {sid} not in runs")
        prediction, score = found[sid]
        text = score["normalized"]["text"]
        if entry.get("text_sha256") != sha256_text(text):
            raise AssertionError(f"{name}/{sid}: judge text_sha256 mismatch")
        if entry.get("gold") != score.get("gold"):
            raise AssertionError(f"{name}/{sid}: judge gold mismatch")
        parsed = score.get("parsed") or {}
        want_strict = parsed.get("prediction") if score["generation_status"] == "success" else None
        if entry.get("strict_prediction") != want_strict:
            raise AssertionError(f"{name}/{sid}: judge strict_prediction mismatch")
        want_strict_correct = int(want_strict is not None and want_strict == entry["gold"])
        if entry.get("strict_correct") != want_strict_correct:
            raise AssertionError(f"{name}/{sid}: judge strict_correct mismatch")
        # Older judge reports predate the generation_status field; verify it
        # only when present.
        if ("generation_status" in entry
                and entry["generation_status"] != score["generation_status"]):
            raise AssertionError(f"{name}/{sid}: judge generation_status mismatch")
        if not str(entry.get("score_path", "")).endswith(safe_id(sid) + ".json"):
            raise AssertionError(f"{name}/{sid}: judge score_path mismatch")
        request_sha = entry.get("request_sha256")
        if request_sha is None:
            if entry.get("judge_status") != "budget_exhausted":
                raise AssertionError(f"{name}/{sid}: missing request_sha256")
        else:
            first = check_key_rows(by_key[sid], name)
            if digest(request_for(first["question"], text)) != request_sha:
                raise AssertionError(f"{name}/{sid}: request_sha256 mismatch")
            cached = jload(cache / (request_sha + ".json"))
            if cached.get("status") != "success":
                raise AssertionError(f"{name}/{sid}: judge cache status "
                                     f"{cached.get('status')!r}")
            parsed_j = cached.get("parsed") or {}
            if (entry.get("judge_prediction") != parsed_j.get("answer")
                    or entry.get("judge_status") != parsed_j.get("status")
                    or entry.get("evidence") != parsed_j.get("evidence")):
                raise AssertionError(f"{name}/{sid}: judge fields != cache parsed")
            want_jc = int(parsed_j.get("answer") is not None
                          and parsed_j.get("answer") == entry["gold"])
            if entry.get("judge_correct") != want_jc:
                raise AssertionError(f"{name}/{sid}: judge_correct mismatch")
        entries[sid] = dict(entry)
        entries[sid]["_row_scores"] = {}
    if set(entries) != set(found):
        raise AssertionError(f"{name}: judge coverage != run coverage")
    return entries


def crosscheck_pope_strict(records, all_rows, by_key, kind, report, name):
    """Recompute the report's primary per-split metrics from exported rows."""
    texts = {r["sample_id"]: r["text"] for r in records}
    statuses = {r["sample_id"]: r["generation_status"] for r in records}
    cdapc_rows = []
    for row in all_rows:
        sid = row["input_key"]
        if sid in texts:
            cdapc_rows.append({"id": row["row_id"], "split": row["split"],
                               "label": row["label"],
                               "image_sha256": row["image_sha256"],
                               "text": texts[sid],
                               "generation_status": statuses[sid]})
    want = report["cdapc"]["splits"]
    out = {}
    for split in SPLITS:
        subset = [r for r in cdapc_rows if r["split"] == split]
        got = (reasoning_split_metrics(subset) if kind != "direct"
               else split_metrics(subset, POPE_RULE))
        out[split] = compare_metrics(got, want[split], f"{name}/{split}")
    return out


def crosscheck_pope_official(records, all_rows, by_key, report, name):
    """Direct conditions: recompute official-rule per-split metrics."""
    texts = {r["sample_id"]: r["text"] for r in records}
    cdapc_rows = [{"id": row["row_id"], "split": row["split"],
                   "label": row["label"], "image_sha256": row["image_sha256"],
                   "text": texts[row["input_key"]]}
                  for row in all_rows if row["input_key"] in texts]
    want = report["cdapc"]["splits"]
    out = {}
    for split in SPLITS:
        subset = [r for r in cdapc_rows if r["split"] == split]
        got = split_metrics(subset, POPE_RULE)
        out[split] = compare_metrics(got, want[split], f"{name}/official/{split}")
    return out


def crosscheck_pope_judge(records, all_rows, by_key, judge_report, name):
    """Recompute judge per-split metrics + macro from exported judge fields."""
    judge = {r["sample_id"]: r for r in records}
    want_splits = judge_report["splits"]
    out = {}
    for split in SPLITS:
        rows = [r for r in all_rows if r["split"] == split
                and r["input_key"] in judge]
        pairs = [(judge[r["input_key"]].get("judge_prediction"), r["label"])
                 for r in rows]
        got = {**pope_metrics(pairs), "n": len(rows)}
        # Older judge reports lack some status-count keys; mirror what's stored.
        for status_key in ("missing", "ambiguous", "budget_exhausted"):
            if status_key in want_splits[split]:
                got[status_key] = sum(
                    judge[r["input_key"]].get("judge_status") == status_key
                    for r in rows)
        out[split] = compare_metrics(got, want_splits[split], f"{name}/judge/{split}")
    want_macro = judge_report.get("macro") or {}
    got_macro = {k: statistics.mean(out[split]["got"][k] for split in SPLITS)
                 for k in ("accuracy", "f1", "precision", "recall", "yes_ratio")
                 if all(out[split]["got"].get(k) is not None for split in SPLITS)}
    macro_ok = all(isclose(got_macro.get(k), want_macro.get(k)) for k in want_macro)
    out["macro"] = {"ok": macro_ok, "got": got_macro, "want": want_macro}
    return out


def compare_metrics(got, want, label):
    """Compare every numeric key of two metric dicts; return detail dict."""
    diffs = {}
    ok = True
    for key, want_val in want.items():
        if key == "confusion":
            got_val = got.get(key, {})
            for ck, cv in want_val.items():
                if got_val.get(ck) != cv:
                    diffs[f"confusion.{ck}"] = (got_val.get(ck), cv)
                    ok = False
            continue
        got_val = got.get(key)
        if isinstance(want_val, (int, float)) and isinstance(got_val, (int, float)):
            if not isclose(got_val, want_val):
                diffs[key] = (got_val, want_val)
                ok = False
        elif got_val != want_val:
            diffs[key] = (got_val, want_val)
            ok = False
    for key in got:
        if key not in want:
            diffs[f"extra.{key}"] = (got[key], None)
            ok = False
    if not ok:
        raise AssertionError(f"METRIC MISMATCH {label}: {diffs}")
    return {"ok": True, "diffs": diffs, "got": got}


def write_pair(out_dir, name, records, txt_fn, manifest_files, meta_extra=None):
    jsonl = out_dir / f"{name}.jsonl"
    jsonl.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                            for r in records))
    text = out_dir / f"{name}.txt"
    text.write_text(txt_fn(records))
    for path in (jsonl, text):
        manifest_files[path.name] = {
            "sha256": sha256_file(path), "bytes": path.stat().st_size,
            "samples": len(records)}
    return {"jsonl": jsonl.name, "txt": text.name}


def pope_txt(records):
    parts = []
    for row in records:
        header = (f"# {row['sample_id']}\n"
                  f"Group: {row['group_id']} | Image: {row['image']} | "
                  f"Rows: {len(row['rows'])}\n")
        scores = []
        for key in ("official_prediction", "strict_status", "strict_prediction",
                    "strict_correct", "judge_prediction", "judge_status",
                    "judge_correct"):
            if key in row:
                scores.append(f"{key}: {row[key]}")
        parts.append(header + " | ".join(scores) + "\n"
                     f"Steps: {row['decode_iterations']} | "
                     f"Tokens before EOS: {row['emitted_tokens_before_eos']} | "
                     f"Finish: {row['finish_reason']} | "
                     f"Status: {row['generation_status']}\n\n"
                     f"{row['text']}\n\n")
    return "".join(parts)


def mmmu_txt(records):
    parts = []
    for row in records:
        parts.append(
            f"# {row['sample_id']}\n"
            f"Gold: {row['gold']} | Judge: {row['judge_correct']} | "
            f"Official: {int(row['official_correct'])} | "
            f"Strict: {row['strict_correct']} ({row['strict_status']}) | "
            f"Status: {row['generation_status']}\n"
            f"Steps: {row['decode_iterations']} | "
            f"Tokens before EOS: {row['emitted_tokens_before_eos']}\n\n"
            f"{row['text']}\n\n")
    return "".join(parts)


def chair_txt(records):
    parts = []
    for row in records:
        m = row["metrics"]
        parts.append(
            f"# {row['sample_id']} (image_id {row['image_id']})\n"
            f"CHAIRs: {m['CHAIRs']} | CHAIRi: {m['CHAIRi']:.6f} | "
            f"Recall: {m['Recall']:.6f}\n"
            f"Steps: {row['decode_iterations']} | "
            f"Tokens: {row['emitted_tokens_before_eos']} | "
            f"Finish: {row['finish_reason']}\n\n"
            f"{row['text']}\n\n")
    return "".join(parts)


def mme_txt(records):
    parts = []
    for row in records:
        parts.append(
            f"# {row['sample_id']} [{row['category']}]\n"
            f"Label: {row['label']} | Prediction: {row['prediction']} | "
            f"Status: {row['generation_status']}\n"
            f"Steps: {row['decode_iterations']} | "
            f"Tokens before EOS: {row['emitted_tokens_before_eos']} | "
            f"Finish: {row['finish_reason']}\n\n"
            f"{row['text']}\n\n")
    return "".join(parts)


def export_mmmu(out_dir, manifest_files):
    """MMMU cdapc_r70 L256/I128 -- mirrors scripts/export_mmmu_answers.py."""
    name = "mmmu_cdapc_r70_L256_I128"
    index = jload(REPORTS / "mmmu_r70_256_128_runs.json")
    ids = jload(ROOT / "manifests/mmmu_val_all_ids.json")
    assert len(ids) == len(set(ids)) == 900
    expected = {"task": "mmmu", "prompt_id": "mmmu_wino_reasoning_v1",
                "strict_parser_id": "mmmu_rule_noguess_v1",
                "canvas_length": 256, "data_root": "data/mmmu_val",
                "r_gate": 0.7, "iteration_limit": 128, "max_commit": 16,
                "window": 16, "alpha": 0.5, "beta": 0.1, "u_gate": 0.1,
                "stop_rule": "eos_prefix_complete", "budget_limited": True}
    records, run_rels, fps = {}, [], {}
    for shard in range(3):
        run = ROOT / index["shards"][str(shard)]["run"]
        manifest = jload(run / "run_manifest.json")
        status = manifest.get("status")
        counts = manifest.get("sample_status_counts") or {}
        resolved = sum(counts.get(s, 0) for s in ("success", "budget_exhausted"))
        # Budget-limited runs legitimately end "incomplete" when samples hit
        # the iteration budget; every sample must still be accounted for.
        if status != "complete" and not (
                status == "incomplete"
                and resolved == manifest.get("sample_count")
                and set(counts) <= {"success", "budget_exhausted"}):
            raise AssertionError(f"{name} {run}: status {status} counts {counts}")
        check_run_config(run, expected, name)
        # The config's sample_ids_file is authoritative; read it directly.
        config = jload(run / "resolved_config.json")
        shard_ids = jload(ROOT / config["run"]["sample_ids_file"])
        score_dirs = [p for p in (run / "scores").iterdir()
                      if (p / "evaluation_manifest.json").is_file()]
        assert len(score_dirs) == 1
        score_dir = score_dirs[0]
        judge_dirs = [p for p in (run / "judge").iterdir() if p.is_dir()]
        assert len(judge_dirs) == 1
        judge_dir = judge_dirs[0]
        fps[rel(run)] = score_dir.name
        found = set()
        for path in sorted((run / "predictions").glob("*.json")):
            prediction = jload(path)
            sid = prediction["sample_id"]
            if sid in found:
                raise AssertionError(f"{name} {run}: duplicate {sid}")
            found.add(sid)
            if prediction.get("status") not in ("success", "budget_exhausted"):
                raise AssertionError(f"{name} {run}/{sid}: status "
                                     f"{prediction.get('status')!r}")
            score = jload(score_dir / path.name)
            judge = jload(judge_dir / path.name)
            if score.get("sample_id") != sid or judge.get("sample_id") != sid:
                raise AssertionError(f"{name} {run}/{sid}: id mismatch")
            if score.get("generation_status") != prediction.get("status"):
                raise AssertionError(f"{name} {run}/{sid}: status mismatch")
            if judge.get("score") not in (0, 1):
                raise AssertionError(f"{name} {run}/{sid}: judge score "
                                     f"{judge.get('score')!r}")
            if sid in records:
                raise AssertionError(f"{name}: input {sid} in two runs")
            records[sid] = {
                "sample_id": sid, "method": name,
                "question_type": score.get("question_type"),
                "gold": score["gold"], "text": score["normalized"]["text"],
                "judge_correct": judge["score"],
                "official_correct": bool(score["official"]),
                "strict_correct": int(score["strict_accuracy"]),
                "strict_status": score["strict_status"],
                "generation_status": score["generation_status"],
                "decode_iterations": prediction["decode_iterations"],
                "emitted_tokens_before_eos": prediction["emitted_tokens_before_eos"],
                "finish_reason": prediction["finish_reason"],
                "source_run": rel(run),
            }
        if found != set(shard_ids):
            raise AssertionError(f"{name} {run}: coverage "
                                 f"{len(found)}/{len(shard_ids)}")
        run_rels.append(rel(run))
    if set(records) != set(ids):
        raise AssertionError(f"{name}: coverage {len(records)}/{len(ids)}")
    ordered = [records[sid] for sid in ids]
    meta = {"report": rel(REPORTS / "mmmu_cdapc_r70_L256_I128.json"),
            "runs": run_rels, "evaluations": fps,
            "unique_inputs": len(ordered), "evaluation_rows": len(ordered)}
    checks = crosscheck_mmmu(ordered, REPORTS / "mmmu_cdapc_r70_L256_I128.json")
    files = write_pair(out_dir, name, ordered, mmmu_txt, manifest_files)
    meta.update(files)
    return meta, checks


def crosscheck_mmmu(records, report_path):
    report = jload(report_path)
    per_sample = report["cdapc"]["per_sample"]
    out = {}
    n = len(records)
    for metric, key in (("judge", "judge_correct"), ("official", "official_correct"),
                        ("strict", "strict_correct")):
        acc = sum(bool(r[key]) for r in records) / n
        want = report["cdapc"]["blocks"]["all"][metric]["accuracy"]
        if not isclose(acc, want):
            raise AssertionError(f"mmmu {metric}: {acc} != {want}")
        out[metric] = {"ok": True, "got": acc, "want": want}
    # Per-sample agreement with the stored report.
    mismatches = 0
    for r in records:
        stored = per_sample.get(r["sample_id"])
        if stored is None:
            mismatches += 1
            continue
        for metric, key in (("judge", "judge_correct"), ("official", "official_correct"),
                            ("strict", "strict_correct")):
            if not isclose(float(stored[metric]), float(bool(r[key]))):
                mismatches += 1
    if mismatches:
        raise AssertionError(f"mmmu per_sample mismatches: {mismatches}")
    out["per_sample"] = {"ok": True, "checked": n}
    return out


def export_chair(out_dir, manifest_files):
    """CHAIR r90/r70/r50 -- text + metrics verified against stored reports."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/mmada",
                                              local_files_only=True)
    from cdapc_lab.evaluation.normalize import normalize_output
    index = jload(REPORTS / "chair_pope_extensions_runs.json")
    rows = jload(ROOT / "data/chair/rows.json")
    images = {r["id"]: r for r in rows}
    expected_ids = jload(ROOT / "manifests/chair_all_ids.json")
    results = {}
    for tau in (90, 70, 50):
        name = f"chair_r{tau}"
        stage = index["stages"][name]
        report_path = REPORTS / stage["report"].split("reports/")[-1]
        report = jload(report_path)
        sentences = {s["image_id"]: s for s in report["sentences"]}
        answers_path = REPORTS / f"chair_r{tau}_answers.json"
        answers = {a["sample_id"]: a for a in jload(answers_path)}
        expected = {"task": "chair", "prompt_id": "chair_caption_v1",
                    "strict_parser_id": "chair_caption_v1",
                    "canvas_length": 64, "data_root": "data/chair",
                    "r_gate": tau / 100, "iteration_limit": 64, "max_commit": 16,
                    "window": 16, "alpha": 0.5, "beta": 0.1, "u_gate": 0.1,
                    "stop_rule": "eos_prefix_complete", "budget_limited": False}
        found, fps, run_rels = {}, {}, []
        for run_text in report["runs"]:
            run = Path(run_text)
            if not run.is_absolute():
                run = ROOT / run
            manifest = jload(run / "run_manifest.json")
            if manifest.get("status") != "complete":
                raise AssertionError(f"{name} {run}: status {manifest.get('status')}")
            check_run_config(run, expected, name)
            config = jload(run / "resolved_config.json")
            ids = shard_ids(config)
            # CHAIR runs are scored directly from predictions by
            # scripts/score_chair_runs.py; there is no scores/ fingerprint dir.
            shard_found = set()
            for path in sorted((run / "predictions").glob("*.json")):
                prediction = jload(path)
                prediction["_run"] = run
                sid = prediction["sample_id"]
                if sid in shard_found or sid in found:
                    raise AssertionError(f"{name} {run}: duplicate {sid}")
                shard_found.add(sid)
                if prediction.get("status") != "success":
                    raise AssertionError(f"{name} {run}/{sid}: status "
                                         f"{prediction.get('status')!r}")
                norm = normalize_output(prediction["raw_response_ids"],
                                        tokenizer, prediction["eos_ids"])
                saved = answers.get(sid)
                if saved is None or saved.get("sample_id") != sid:
                    raise AssertionError(f"{name}: no saved answer for {sid}")
                if (norm["text"] != saved["text"]
                        or norm["emitted_tokens_before_eos"]
                        != saved["answer_token_length"]
                        or prediction["decode_iterations"] != saved["decode_iterations"]
                        or prediction["finish_reason"] != saved["finish_reason"]):
                    raise AssertionError(f"{name}/{sid}: answers.json mismatch")
                found[sid] = (prediction, norm, saved)
            if shard_found != set(ids):
                raise AssertionError(f"{name} {run}: coverage "
                                     f"{len(shard_found)}/{len(ids)}")
            fps[rel(run)] = "direct score_chair_runs.py (no scores/ fingerprint)"
            run_rels.append(rel(run))
        if set(found) != set(expected_ids):
            raise AssertionError(f"{name}: coverage {len(found)}/{len(expected_ids)}")
        records = []
        for sid in sorted(found, key=lambda s: images[s]["image_id"]):
            prediction, norm, saved = found[sid]
            info = images[sid]
            sentence = sentences.get(info["image_id"])
            if sentence is None:
                raise AssertionError(f"{name}: no sentence for image "
                                     f"{info['image_id']}")
            if sentence["caption"] != saved["text"]:
                raise AssertionError(f"{name} image {info['image_id']}: "
                                     "caption != answers.json text")
            records.append({
                "sample_id": sid, "image_id": info["image_id"],
                "group_id": str(info["image_id"]),
                "image_sha256": info["image_sha256"],
                "image": Path(info["image"]).name,
                "question": info["question"], "prompt": info["question"],
                "prompt_id": "chair_caption_v1", "dataset": "chair",
                "condition": name, "text": saved["text"],
                "generation_status": "success",
                "decode_iterations": prediction["decode_iterations"],
                "emitted_tokens_before_eos": norm["emitted_tokens_before_eos"],
                "finish_reason": prediction["finish_reason"],
                "source_run": rel(prediction["_run"]),
                "metrics": dict(sentence["metrics"]),
            })
        # Cross-check overall metrics recomputed from sentences.
        checks = crosscheck_chair(report, name)
        meta = {"report": rel(report_path), "answers": rel(answers_path),
                "runs": run_rels, "evaluations": fps,
                "unique_inputs": len(records), "evaluation_rows": len(records),
                "prompt_id": "chair_caption_v1", "data_root": "data/chair"}
        files = write_pair(out_dir, name, records, chair_txt, manifest_files)
        meta.update(files)
        results[name] = {"meta": meta, "checks": checks}
    return results


def crosscheck_chair(report, name):
    sentences = report["sentences"]
    n = len(sentences)
    chair_s = sum(s["metrics"]["CHAIRs"] for s in sentences) / n
    hallucinated = sum(len(s["mscoco_hallucinated_words"]) for s in sentences)
    generated = sum(len(s["mscoco_generated_words"]) for s in sentences)
    chair_i = hallucinated / generated if generated else None
    recall_hits = sum(len(set(s["mscoco_generated_words"])
                          & set(s["mscoco_gt_words"])) for s in sentences)
    recall_total = sum(len(s["mscoco_gt_words"]) for s in sentences)
    recall = recall_hits / recall_total if recall_total else None
    got = {"CHAIRs": chair_s, "CHAIRi": chair_i, "Recall": recall}
    want = report["overall_metrics"]
    for key in ("CHAIRs", "CHAIRi", "Recall"):
        if not isclose(got[key], want[key]):
            raise AssertionError(f"{name} overall {key}: {got[key]} != {want[key]}")
    return {"overall_metrics": {"ok": True, "got": got, "want": want},
            "sentences": n,
            "empty_answers": report.get("empty_answers"),
            "mean_answer_tokens": report.get("mean_answer_tokens")}


def mechanism(pred):
    """Port of scripts/k_commit_report.py mechanism()."""
    from collections import defaultdict
    batches = defaultdict(list)
    for entry in pred["commit_log"]:
        batches[entry["iteration"]].append(entry)
    assert set(batches) == set(range(pred["decode_iterations"]))
    known_eos = None
    tail_only = 0
    sizes = []
    for iteration in sorted(batches):
        entries = batches[iteration]
        sizes.append(len(entries))
        tail_only += int(known_eos is not None and all(
            entry["position"] > known_eos for entry in entries))
        for entry in entries:
            if entry["written_id"] in pred["eos_ids"]:
                known_eos = (min(known_eos, entry["position"])
                             if known_eos is not None else entry["position"])
    assert sum(sizes) == pred["metrics"]["committed"]
    return {
        "iterations": pred["decode_iterations"],
        "seconds": pred["generation_seconds"],
        "output_tokens": pred["emitted_tokens_before_eos"],
        "commits": sum(sizes),
        "batch_histogram": dict(Counter(sizes)),
        "tail_only_iterations": tail_only,
        "finish_reason": pred["finish_reason"],
        "fallback_iterations": pred["metrics"]["fallback_iterations"],
        "switches": pred["metrics"]["actual_switches"],
    }


def export_k_pilot(out_dir, manifest_files):
    """GQA think K pilot K4/K2 (K16 is referenced, not duplicated)."""
    index = jload(PILOT_DIR / "index.json")
    results = jload(PILOT_DIR / "results.json")
    rows = jload(PILOT_DIR / "rows.json")
    by_key = defaultdict(list)
    for row in rows:
        by_key[row["input_key"]].append(row)
    pilot_ids = set(by_key)
    cache = JUDGE_DIR / "cache"
    all_meta, all_checks = {}, {}
    for k in ("4", "2"):
        name = f"gqa_think_k{k}_pilot"
        per_input = results["per_input"][k]
        if set(per_input) != pilot_ids:
            raise AssertionError(f"{name}: per_input coverage "
                                 f"{len(per_input)}/{len(pilot_ids)}")
        records_by_sid = {}
        run_ids = defaultdict(set)
        for sid, rec in per_input.items():
            score_rel = rec["score_path"]
            run_rel = str(Path(*Path(score_rel).parts[:2]))
            run_ids[run_rel].add(sid)
            score = jload(ROOT / score_rel)
            prediction = jload(ROOT / run_rel / "predictions"
                               / (safe_id(sid) + ".json"))
            if (score.get("sample_id") != sid
                    or score.get("generation_status") != "success"
                    or prediction.get("status") != "success"):
                raise AssertionError(f"{name}/{sid}: status mismatch")
            text = score["normalized"]["text"]
            if text != rec["text"]:
                raise AssertionError(f"{name}/{sid}: text != results.json")
            parsed = score.get("parsed") or {}
            if (parsed.get("prediction") != rec["strict_prediction"]
                    or score.get("strict_accuracy") != rec["strict_correct"]):
                raise AssertionError(f"{name}/{sid}: strict fields mismatch")
            mech = mechanism(prediction)
            for field, want in (("iterations", rec["iterations"]),
                                ("seconds", rec["seconds"]),
                                ("output_tokens", rec["output_tokens"]),
                                ("commits", rec["commits"]),
                                ("tail_only_iterations", rec["tail_only_iterations"]),
                                ("finish_reason", rec["finish_reason"]),
                                ("fallback_iterations", rec["fallback_iterations"]),
                                ("switches", rec["switches"])):
                got = mech[field]
                same = isclose(got, want) if isinstance(want, float) else got == want
                if not same:
                    raise AssertionError(f"{name}/{sid}: mechanism {field} "
                                         f"{got} != {want}")
            if ({str(k2): v for k2, v in mech["batch_histogram"].items()}
                    != rec["batch_histogram"]):
                raise AssertionError(f"{name}/{sid}: batch_histogram mismatch")
            if max(int(x) for x in rec["batch_histogram"]) > int(k):
                raise AssertionError(f"{name}/{sid}: commit exceeds K={k}")
            cached = jload(cache / (rec["request_sha256"] + ".json"))
            if cached.get("status") != "success":
                raise AssertionError(f"{name}/{sid}: judge cache status")
            parsed_j = cached.get("parsed") or {}
            if (parsed_j.get("answer") != rec["judge_prediction"]
                    or parsed_j.get("status") != rec["judge_status"]
                    or parsed_j.get("evidence") != rec["judge_evidence"]):
                raise AssertionError(f"{name}/{sid}: judge != cache parsed")
            first = check_key_rows(by_key[sid], name)
            if digest(request_for(first["question"], text)) != rec["request_sha256"]:
                raise AssertionError(f"{name}/{sid}: request_sha256 mismatch")
            want_jc = int(parsed_j.get("answer") is not None
                          and parsed_j.get("answer") == rec["gold"])
            if want_jc != rec["judge_correct"] or rec["gold"] != score["gold"]:
                raise AssertionError(f"{name}/{sid}: judge_correct/gold mismatch")
            records_by_sid[sid] = (rec, score, prediction, run_rel, mech)
        # Verify each run's manifest coverage.
        for run_rel, sids in run_ids.items():
            run = ROOT / run_rel
            manifest = jload(run / "run_manifest.json")
            if manifest.get("status") != "complete":
                raise AssertionError(f"{name} {run_rel}: status")
            config = jload(run / "resolved_config.json")
            expected = {"task": "pope", "prompt_id": "pope_yesno_reasoning_v1",
                        "strict_parser_id": "pope_final_yes_no_v1",
                        "canvas_length": 128, "data_root": "data/pope_gqa",
                        "r_gate": 0.7, "iteration_limit": 128,
                        "max_commit": int(k), "window": 16, "alpha": 0.5,
                        "beta": 0.1, "u_gate": 0.1,
                        "stop_rule": "eos_prefix_complete", "budget_limited": False}
            check_run_config(run, expected, name)
            ids = set(shard_ids(config))
            if ids != sids:
                raise AssertionError(f"{name} {run_rel}: shard ids {len(ids)} "
                                     f"!= records {len(sids)}")
        ordered = []
        seen_keys = set()
        for row in rows:
            sid = row["input_key"]
            if sid in seen_keys:
                continue
            seen_keys.add(sid)
            rec, score, prediction, run_rel, mech = records_by_sid[sid]
            key_rows = by_key[sid]
            first = check_key_rows(key_rows, name)
            parsed = score.get("parsed") or {}
            entry_rows = []
            for erow in key_rows:
                strict_ok = int(parsed.get("prediction") is not None
                                and parsed.get("prediction") == erow["label"])
                judge_ok = int(rec["judge_prediction"] is not None
                               and rec["judge_prediction"] == erow["label"])
                entry_rows.append({"row_id": erow["row_id"], "split": erow["split"],
                                   "question_id": erow["question_id"],
                                   "label": erow["label"],
                                   "scores": {"strict": strict_ok, "judge": judge_ok}})
            ordered.append({
                "sample_id": sid, "group_id": score.get("group_id"),
                "image_sha256": first["image_sha256"], "image": first["image"],
                "question": first["question"],
                "prompt": pope_prompt(first["question"], "pope_yesno_reasoning_v1"),
                "prompt_id": "pope_yesno_reasoning_v1", "dataset": "pope_gqa",
                "condition": name, "commit_cap": int(k),
                "text": score["normalized"]["text"],
                "generation_status": "success",
                "decode_iterations": prediction["decode_iterations"],
                "emitted_tokens_before_eos": prediction["emitted_tokens_before_eos"],
                "finish_reason": prediction["finish_reason"],
                "source_run": run_rel,
                "strict_status": parsed.get("status"),
                "strict_prediction": parsed.get("prediction"),
                "strict_correct": score["strict_accuracy"],
                "judge_prediction": rec["judge_prediction"],
                "judge_status": rec["judge_status"],
                "judge_correct": rec["judge_correct"],
                "judge_evidence": rec.get("judge_evidence", ""),
                "judge_request_sha256": rec["request_sha256"],
                "mechanism": {kk: mech[kk] for kk in
                              ("iterations", "seconds", "output_tokens", "commits",
                               "batch_histogram", "tail_only_iterations",
                               "fallback_iterations", "switches")},
                "empty_shell": rec.get("empty_shell"),
                "rows": entry_rows,
            })
        assert len(ordered) == len(pilot_ids)
        checks = crosscheck_k_pilot(ordered, rows, by_key, results["methods"][k], name)
        meta = {"report": rel(PILOT_DIR / "results.json"),
                "runs": sorted(run_ids), "commit_cap": int(k),
                "unique_inputs": len(ordered), "evaluation_rows": len(rows),
                "prompt_id": "pope_yesno_reasoning_v1", "data_root": "data/pope_gqa"}
        files = write_pair(out_dir, name, ordered, pope_txt, manifest_files)
        meta.update(files)
        all_meta[name] = meta
        all_checks[name] = checks
    return all_meta, all_checks


def crosscheck_k_pilot(records, rows, by_key, want, name):
    """Recompute methods[k] aggregates from exported records."""
    by_sid = {r["sample_id"]: r for r in records}
    splits = {}
    for split in SPLITS:
        selected = [row for row in rows if row["split"] == split]
        assert len(selected) == 180
        splits[split] = {
            "n": len(selected),
            "judge": pope_metrics(
                (by_sid[row["input_key"]]["judge_prediction"], row["label"])
                for row in selected),
            "strict": pope_metrics(
                (by_sid[row["input_key"]]["strict_prediction"], row["label"])
                for row in selected),
        }
    n = len(records)
    iterations = sum(r["mechanism"]["iterations"] for r in records)
    commits = sum(r["mechanism"]["commits"] for r in records)
    got = {
        "unique_inputs": n, "evaluation_rows": len(rows), "splits": splits,
        "judge_accuracy": statistics.mean(s["judge"]["accuracy"] for s in splits.values()),
        "judge_f1": (statistics.mean(s["judge"]["f1"] for s in splits.values())
                     if all(s["judge"]["f1"] is not None for s in splits.values()) else None),
        "strict_accuracy": statistics.mean(s["strict"]["accuracy"] for s in splits.values()),
        "mean_seconds": statistics.mean(r["mechanism"]["seconds"] for r in records),
        "mean_iterations": iterations / n,
        "commits_per_iteration": commits / iterations,
        "mean_output_tokens": statistics.mean(r["mechanism"]["output_tokens"] for r in records),
        "tail_only_iteration_rate": (sum(r["mechanism"]["tail_only_iterations"] for r in records) / iterations),
        "empty_shell_rate": sum(bool(r["empty_shell"]) for r in records) / n,
        "judge_missing_rate": sum(r["judge_status"] == "missing" for r in records) / n,
        "judge_ambiguous_rate": sum(r["judge_status"] == "ambiguous" for r in records) / n,
        "judge_statuses_unique": dict(Counter(r["judge_status"] for r in records)),
        "finish_reasons": dict(Counter(r["finish_reason"] for r in records)),
    }
    diffs = {}
    for key in ("unique_inputs", "evaluation_rows", "judge_accuracy", "judge_f1",
                "strict_accuracy", "mean_seconds", "mean_iterations",
                "commits_per_iteration", "mean_output_tokens",
                "tail_only_iteration_rate", "empty_shell_rate",
                "judge_missing_rate", "judge_ambiguous_rate",
                "judge_statuses_unique", "finish_reasons"):
        g, w = got[key], want[key]
        if isinstance(w, float):
            if not isclose(g, w):
                diffs[key] = (g, w)
        elif g != w:
            diffs[key] = (g, w)
    for split in SPLITS:
        for sub in ("judge", "strict"):
            for key, w in want["splits"][split][sub].items():
                g = got["splits"][split][sub].get(key)
                if isinstance(w, (int, float)) and isinstance(g, (int, float)):
                    if not isclose(g, w):
                        diffs[f"splits.{split}.{sub}.{key}"] = (g, w)
                elif g != w:
                    diffs[f"splits.{split}.{sub}.{key}"] = (g, w)
        if got["splits"][split]["n"] != want["splits"][split]["n"]:
            diffs[f"splits.{split}.n"] = (got["splits"][split]["n"],
                                          want["splits"][split]["n"])
    if diffs:
        raise AssertionError(f"{name} metric mismatch: {diffs}")
    return {"ok": True, "judge_accuracy": got["judge_accuracy"],
            "strict_accuracy": got["strict_accuracy"],
            "judge_f1": got["judge_f1"],
            "mean_seconds": got["mean_seconds"]}


def load_score_mme():
    path = MME / "scripts/score_mme.py"
    spec = importlib.util.spec_from_file_location("cdapc_mme_score", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def export_mme(out_dir, manifest_files):
    """MME r70 L256/I128 from the cdapc-mme snapshot workspace."""
    score_mme = load_score_mme()
    from transformers import AutoTokenizer
    from cdapc_lab.evaluation.normalize import normalize_output
    index = jload(MME / "reports/index.json")
    rows = jload(MME / "data/mme/rows.json")
    categories, parse = score_mme.official_protocol()
    report = jload(MME / "reports/mme_cdapc_r70_L256_I128.json")
    per_sample = report["per_sample"]
    expected = {"task": "mme", "prompt_id": "mme_yesno_v1",
                "strict_parser_id": "mme_official_v1",
                "canvas_length": 256, "data_root": "data/mme",
                "r_gate": 0.7, "iteration_limit": 128, "max_commit": 16,
                "window": 16, "alpha": 0.5, "beta": 0.1, "u_gate": 0.1,
                "stop_rule": "eos_prefix_complete", "budget_limited": True}
    found, run_rels, fps = {}, [], {}
    first_config = None
    for shard in ("0", "1", "2"):
        run = Path(index["shards"][shard]["run"])
        manifest = jload(run / "run_manifest.json")
        if manifest.get("status") != "complete":
            raise AssertionError(f"mme {run}: status {manifest.get('status')}")
        config = check_run_config(run, expected, "mme_r70")
        if first_config is None:
            first_config = config
            tokenizer = AutoTokenizer.from_pretrained(
                MME / config["model"]["checkpoint"], local_files_only=True)
        ids = jload(MME / config["run"]["sample_ids_file"])
        shard_found = set()
        for path in sorted((run / "predictions").glob("*.json")):
            prediction = jload(path)
            prediction["_run"] = run
            sid = prediction["sample_id"]
            if sid in shard_found or sid in found:
                raise AssertionError(f"mme: duplicate {sid}")
            shard_found.add(sid)
            if prediction.get("status") not in ("success", "budget_exhausted"):
                raise AssertionError(f"mme {run}/{sid}: status "
                                     f"{prediction.get('status')!r}")
            norm = normalize_output(prediction["raw_response_ids"],
                                    tokenizer, prediction["eos_ids"])
            stored = per_sample.get(sid)
            if stored is None:
                raise AssertionError(f"mme: no per_sample for {sid}")
            if norm != stored["normalized"]:
                raise AssertionError(f"mme/{sid}: normalized != report")
            want_text = norm["text"]
            want_pred = (parse(want_text) if prediction["status"] == "success"
                         else "other")
            if (stored["text"] != want_text or stored["prediction"] != want_pred
                    or stored["generation_status"] != prediction["status"]
                    or stored["finish_reason"] != prediction["finish_reason"]):
                raise AssertionError(f"mme/{sid}: per_sample mismatch")
            found[sid] = (prediction, norm, stored)
        if shard_found != set(ids):
            raise AssertionError(f"mme {run}: coverage "
                                 f"{len(shard_found)}/{len(ids)}")
        # MME is scored directly from predictions by scripts/score_mme.py;
        # there is no scores/ fingerprint dir.
        fps[rel(run, MME)] = "direct score_mme.py (no scores/ fingerprint)"
        run_rels.append(rel(run, MME))
    if set(found) != {r["sample_id"] for r in rows} or len(found) != 2374:
        raise AssertionError(f"mme: coverage {len(found)}")
    row_by_sid = {r["sample_id"]: r for r in rows}
    ordered = []
    for sid in [r["sample_id"] for r in rows]:
        prediction, norm, stored = found[sid]
        info = row_by_sid[sid]
        ordered.append({
            "sample_id": sid, "group_id": info["group_id"],
            "category": info["category"], "image_name": info["image_name"],
            "image_sha256": info["image_sha256"],
            "question": info["question"], "prompt": info["question"],
            "label": info["answer"],
            "text": stored["text"], "prediction": stored["prediction"],
            "generation_status": stored["generation_status"],
            "decode_iterations": prediction["decode_iterations"],
            "emitted_tokens_before_eos": prediction["emitted_tokens_before_eos"],
            "finish_reason": prediction["finish_reason"],
            "source_run": rel(prediction["_run"], MME),
        })
    answers = {r["sample_id"]: r["prediction"] for r in ordered}
    got_metrics = score_mme.metrics(rows, answers, categories)
    want_metrics = report["metrics"]
    for section in ("perception", "cognition", "total"):
        if not isclose(got_metrics[section], want_metrics[section]):
            raise AssertionError(f"mme {section}: {got_metrics[section]} != "
                                 f"{want_metrics[section]}")
    for cat, want in want_metrics["per_category"].items():
        got = got_metrics["per_category"].get(cat)
        if got is None:
            raise AssertionError(f"mme missing category {cat}")
        for key, w in want.items():
            g = got.get(key)
            if isinstance(w, float):
                if not isclose(g, w):
                    raise AssertionError(f"mme {cat}.{key}: {g} != {w}")
            elif g != w:
                raise AssertionError(f"mme {cat}.{key}: {g} != {w}")
    checks = {"metrics": {"ok": True,
                          "perception": got_metrics["perception"],
                          "cognition": got_metrics["cognition"],
                          "total": got_metrics["total"]},
              "questions": len(ordered), "image_pairs": report["image_pairs"]}
    meta = {"report": "mme/mme_cdapc_r70_L256_I128.json (cdapc-mme snapshot)",
            "runs_index": "mme/mme_index.json",
            "runs": run_rels, "evaluations": fps,
            "unique_inputs": len(ordered), "evaluation_rows": len(ordered),
            "prompt_id": "mme_yesno_v1", "data_root": "data/mme (cdapc-mme)"}
    files = write_pair(out_dir, "mme_r70_L256_I128", ordered, mme_txt,
                       manifest_files)
    meta.update(files)
    return meta, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT,
                        help="answers output directory")
    args = parser.parse_args()
    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions = {}
    checks = {}

    # MMMU r70 L256/I128.
    meta, chk = export_mmmu(out_dir, {})
    conditions["mmmu_cdapc_r70_L256_I128"] = meta
    checks["mmmu_cdapc_r70_L256_I128"] = chk

    # POPE COCO direct r90/r80/r70/r60/r50.
    coco_reports = {
        90: "pope_yesno_mmada_cdapc_L128.json",
        80: "pope_yesno_mmada_cdapc_L128_r80.json",
        70: "pope_yesno_mmada_cdapc_L128_r70.json",
        60: "pope_yesno_mmada_cdapc_L128_r60.json",
        50: "pope_yesno_mmada_cdapc_L128_r50.json",
    }
    for tau, filename in coco_reports.items():
        name = f"pope_coco_direct_r{tau}"
        records, meta, chk = assemble_pope_condition(
            name, "pope_yesno", "direct", tau / 100,
            REPORTS / filename, expected_unique=5127)
        files = write_pair(out_dir, name, records, pope_txt, {})
        meta.update(files)
        conditions[name] = meta
        checks[name] = chk

    # POPE A-OKVQA / GQA direct.
    for dataset in ("aokvqa", "gqa"):
        for tau in (90, 70, 50):
            name = f"pope_{dataset}_direct_r{tau}"
            records, meta, chk = assemble_pope_condition(
                name, f"pope_{dataset}", "direct", tau / 100,
                REPORTS / f"pope_{dataset}_r{tau}.json")
            files = write_pair(out_dir, name, records, pope_txt, {})
            meta.update(files)
            conditions[name] = meta
            checks[name] = chk

    # POPE think (completed stages only; aokvqa r50 was cancelled).
    for dataset, taus in (("aokvqa", (90, 70)), ("gqa", (90, 70, 50))):
        for tau in taus:
            name = f"pope_{dataset}_think_r{tau}"
            records, meta, chk = assemble_pope_condition(
                name, f"pope_{dataset}", "think", tau / 100,
                REPORTS / f"pope_{dataset}_reasoning_r{tau}.json",
                judge_path=JUDGE_DIR / f"pope_{dataset}_reasoning_r{tau}.json")
            files = write_pair(out_dir, name, records, pope_txt, {})
            meta.update(files)
            conditions[name] = meta
            checks[name] = chk

    # CHAIR.
    chair_results = export_chair(out_dir, {})
    for name, result in chair_results.items():
        conditions[name] = result["meta"]
        checks[name] = result["checks"]

    # GQA think K pilot K4/K2.
    pilot_meta, pilot_checks = export_k_pilot(out_dir, {})
    conditions.update(pilot_meta)
    checks.update(pilot_checks)

    # MME.
    mme_meta, mme_checks = export_mme(out_dir, {})
    conditions["mme_r70_L256_I128"] = mme_meta
    checks["mme_r70_L256_I128"] = mme_checks

    # answers/manifest.json
    all_files = {}
    for path in sorted(out_dir.iterdir()):
        if path.name in ("manifest.json", "README.md"):
            continue
        all_files[path.name] = {"sha256": sha256_file(path),
                                "bytes": path.stat().st_size}
    manifest = {
        "description": ("Exact normalized per-sample answer text for every "
                        "completed CD-APC condition, exported offline from "
                        "saved run artifacts by scripts/export_cdapc_upload.py. "
                        "The running POPE think K4 L256/I128 experiment and the "
                        "cancelled pope_aokvqa_reasoning_r50 stage are excluded."),
        "normalization": ("text is score['normalized']['text'] (POPE/MMMU) or "
                          "normalize_output(...) recomputed and asserted equal "
                          "to the saved artifact (CHAIR/MME): response tokens "
                          "before the first EOS, decoded with special tokens "
                          "skipped and outer whitespace stripped."),
        "k16_note": ("The K pilot's K16 baseline reuses the completed "
                     "pope_gqa_think_r70 condition; see "
                     "pope_gqa_think_r70.jsonl filtered to the 306 pilot "
                     "input_keys listed in reports/k_commit_pilot/rows.json. "
                     "No separate K16 file is emitted."),
        "conditions": conditions,
        "verification": "All stored-report metrics were recomputed from the "
                        "exported rows and asserted equal; see exporter output.",
        "files": all_files,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    # Cross-check summary table.
    print("\n=== CD-APC answer export cross-checks ===")
    for name in sorted(checks):
        summary = []
        for key, value in checks[name].items():
            if isinstance(value, dict) and value.get("ok"):
                summary.append(f"{key}:OK")
            elif isinstance(value, dict):
                summary.append(f"{key}:{value}")
        print(f"{name}: {'; '.join(summary) if summary else checks[name]}")
    print(f"\nWrote {len(all_files) + 1} files to {out_dir}")


if __name__ == "__main__":
    main()
