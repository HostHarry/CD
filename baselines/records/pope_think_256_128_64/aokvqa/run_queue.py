"""Run full POPE 256/128/64 on COCO + A-OKVQA + GQA, native/vcd/vigor, think prompt.

Self-contained experiment directory; there is no predecessor wait. Both GPUs
must have at least 22 GiB free before either shard starts. Prompt (prompt_id
"pope_think_v1", user-approved): the chat-template user message is
common.mmmu_protocol.INSTRUCTION + "\\n" + q0, where q0 is the original POPE
question without the trailing " Please answer yes or no." suffix. There is no
system message; the image is <|mmu|><|soi|>[1024 codes]<|eoi|> followed by the
chat text, exactly as common.runtime.Runtime.prepare builds it. For COCO the
derived rows are asserted identical to results/pope_coco_think_256_256/inputs
(and thus to diagnostic variant B); for A-OKVQA/GQA q0 must equal the row's
original_question. Scoring has a PRIMARY post-</think> track and a SECONDARY
unchanged full-text official track (common.run metrics).
"""
import argparse
from collections import Counter
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import statistics
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
CONFIG = BASE / "config.yaml"
METHODS = ("native", "vcd", "vigor")
DATASETS = {
    "coco": ROOT / "data/pope_coco_yesno.jsonl",
    "aokvqa": ROOT / "data/pope_extended/aokvqa_yesno.jsonl",
    "gqa": ROOT / "data/pope_extended/gqa_yesno.jsonl",
}
SUFFIX = " Please answer yes or no."
PROMPT_ID = "pope_think_v1"
THINK_COCO_INPUTS = ROOT / "results/pope_coco_think_256_256/inputs/coco_think.jsonl"
EXCLUDES = {
    "coco": ROOT / "data/calibration.jsonl",
    "aokvqa": ROOT / "data/pope_extended/excluded_samples.jsonl",
    "gqa": ROOT / "data/pope_extended/excluded_samples.jsonl",
}
SPLITS = ("random", "popular", "adversarial")
sys.path.insert(0, str(ROOT))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_locked(path, content):
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"refusing changed artifact: {path}")
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(path)


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def answer_rows(path):
    # common.run writes each method's answers as one JSON array.
    return json.loads(Path(path).read_text())


def input_key(sample):
    # Same key as common.run, checked against it during preparation.
    return hashlib.sha256(json.dumps(
        [sample["image_sha256"], sample.get("messages", sample.get("question"))],
        ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def think_question(name, sample):
    question = sample["question"]
    if not question.endswith(SUFFIX):
        raise ValueError(f"{name}: {sample['id']} lacks the exact yes/no suffix")
    q0 = question[:-len(SUFFIX)]
    if name != "coco" and q0 != sample.get("original_question"):
        raise ValueError(f"{name}: {sample['id']} suffix-stripped question differs from original_question")
    return q0


def prepare():
    from common.mmmu_protocol import INSTRUCTION
    from common.run import input_key as runtime_key
    from common.runtime import source_hashes
    from omegaconf import OmegaConf

    config = OmegaConf.to_container(OmegaConf.load(CONFIG), resolve=True)
    generation = dict(config["generation"])
    generation.update(config["benchmarks"]["POPE"])
    if [generation[k] for k in ("max_new_tokens", "steps", "block_length")] != [256, 128, 64]:
        raise ValueError("unexpected POPE budget")
    (BASE / "inputs").mkdir(exist_ok=True)
    plan = {
        "scope": "full POPE COCO, A-OKVQA and GQA; native/vcd/vigor; new 256/128/64 predictions",
        "methods": list(METHODS), "gpus": ["0", "1"],
        "generation": generation, "early_eos": "safe_eos_v1", "prompt_id": PROMPT_ID,
        "prompt": ("chat-template user message = common.mmmu_protocol.INSTRUCTION + '\\n' + q0 "
                   "(POPE question minus ' Please answer yes or no.'), no system message; "
                   "image as <|mmu|><|soi|>[1024 codes]<|eoi|> + chat text"),
        "sharding": "alternate unique input keys in first-occurrence order; all duplicate rows stay together",
        "sources": source_hashes(), "launcher_sha256": digest(__file__),
        "config_sha256": digest(CONFIG), "datasets": {},
    }
    think_coco = None
    images = {}
    for name, source in DATASETS.items():
        samples = rows(source)
        if len(samples) != 9000 or len({r["id"] for r in samples}) != 9000:
            raise ValueError(f"{name}: expected 9000 unique row IDs")
        counts = Counter((r["split"], r["label"]) for r in samples)
        if counts != {(split, label): 1500 for split in SPLITS for label in ("yes", "no")}:
            raise ValueError(f"{name}: unexpected split/label counts")
        derived = []
        for sample in samples:
            row = dict(sample)
            row["question"] = INSTRUCTION + "\n" + think_question(name, sample)
            row["removed_suffix"] = SUFFIX
            row["prompt_id"] = PROMPT_ID
            derived.append(row)
        entry = {"derived_from": str(source), "derived_from_sha256": digest(source)}
        if name == "coco":
            # The think prompt must reproduce the paused pope_coco_think_256_256
            # inputs exactly so input keys line up with that experiment.
            if think_coco is None:
                think_coco = {r["id"]: r for r in rows(THINK_COCO_INPUTS)}
            if len(think_coco) != 9000 or any(row != think_coco.get(row["id"]) for row in derived):
                raise ValueError("coco: derived rows differ from pope_coco_think_256_256 inputs")
            entry["matches_pope_coco_think_256_256_rows"] = len(derived)
        for row in derived:
            if input_key(row) != runtime_key(row):
                raise ValueError("launcher key differs from the inference runner")
            path_image = row["image"]
            if path_image not in images:
                images[path_image] = digest(path_image)
            if images[path_image] != row["image_sha256"]:
                raise ValueError(f"image hash differs: {path_image}")
        excluded = EXCLUDES[name]
        if {r["image_sha256"] for r in rows(excluded)} & {r["image_sha256"] for r in derived}:
            raise ValueError(f"{name}: excluded images overlap this evaluation")
        derived_path = BASE / "inputs" / f"{name}_think.jsonl"
        write_locked(derived_path, "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in derived).encode())
        unique = list(dict.fromkeys(input_key(row) for row in derived))
        assignment = {key: index % 2 for index, key in enumerate(unique)}
        shards = []
        for shard in range(2):
            selected = [row for row in derived if assignment[input_key(row)] == shard]
            target = BASE / "inputs" / f"{name}_{shard}.jsonl"
            content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected).encode()
            write_locked(target, content)
            shards.append({"samples": str(target), "sha256": digest(target), "rows": len(selected),
                           "unique_inputs": len({input_key(row) for row in selected})})
        plan["datasets"][name] = {
            "samples": str(derived_path), "sha256": digest(derived_path), "rows": len(derived),
            "unique_inputs": len(unique), "shards": shards,
            "exclude_samples": str(excluded), "exclude_sha256": digest(excluded), **entry,
        }
    write_locked(BASE / "plan.json", (json.dumps(plan, indent=2) + "\n").encode())
    print(json.dumps({name: {k: d[k] for k in ("rows", "unique_inputs")}
                      for name, d in plan["datasets"].items()}, indent=2))


def verify_plan(plan):
    from common.runtime import source_hashes

    if (plan["launcher_sha256"] != digest(__file__) or plan["config_sha256"] != digest(CONFIG)
            or plan["sources"] != source_hashes()):
        raise ValueError("queued launcher, configuration or inference sources changed")
    for data in plan["datasets"].values():
        for path, expected in [(data["samples"], data["sha256"]),
                               (data["exclude_samples"], data["exclude_sha256"]),
                               *[(s["samples"], s["sha256"]) for s in data["shards"]]]:
            if digest(path) != expected:
                raise ValueError(f"queued dataset changed: {path}")


def gpu_inventory():
    lines = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,memory.free", "--format=csv,noheader,nounits",
    ], text=True).splitlines()
    free = {line.split(",")[0].strip(): int(line.split(",")[-1]) for line in lines}
    return lines if all(free.get(gpu, 0) >= 22 * 1024 for gpu in ("0", "1")) else None


def command(samples, output, exclude, score_only=False):
    cmd = [sys.executable, "-u", "-m", "common.run", "--config", str(CONFIG),
           "--samples", str(samples), "--output", str(output), "--benchmark", "POPE",
           "--methods", *METHODS, "--early-eos", "--exclude-samples", str(exclude)]
    return cmd + (["--score-only"] if score_only else [])


def run_jobs(jobs, lock_fd):
    with ExitStack() as stack:
        children = []
        try:
            for argv, gpu, log_path in jobs:
                log = stack.enter_context(log_path.open("a"))
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "OMP_NUM_THREADS": "4",
                       "TOKENIZERS_PARALLELISM": "false", "PYTHONHASHSEED": "42"}
                children.append(subprocess.Popen(
                    argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True, pass_fds=(lock_fd,)))
            while any(child.poll() is None for child in children):
                if any(child.poll() not in (None, 0) for child in children):
                    raise RuntimeError("POPE worker failed; inspect the dataset logs")
                time.sleep(2)
            if any(child.returncode for child in children):
                raise RuntimeError("POPE worker failed; committed predictions remain resumable")
        finally:
            for child in children:
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    child.wait()


def merge_predictions(name, data, plan):
    """Merge exact payload bytes only after both disjoint shards are complete."""
    from verify_eos import audit_record

    target = BASE / name
    full = json.loads((target / "manifest.json").read_text())
    if full["sources"] != plan["sources"] or full["generation"] != plan["generation"]:
        raise ValueError("full-run manifest differs from the queued protocol")
    model = json.loads((Path(full["config"]["checkpoint"]) / "config.json").read_text())
    mask, eos = model["mask_token_id"], model["eos_token_id"]
    all_keys = {input_key(r) for r in rows(Path(data["samples"]))}
    expected_all = {(method, key) for method in METHODS for key in all_keys}
    receipts = []
    with (target / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with sqlite3.connect(target / "predictions.sqlite") as out:
            if not set(out.execute("SELECT method,input_key FROM predictions")) <= expected_all:
                raise ValueError("unexpected preexisting merged prediction")
            seen = set()
            for shard, metadata in enumerate(data["shards"]):
                directory = BASE / f"{name}_shard{shard}"
                manifest = json.loads((directory / "manifest.json").read_text())
                for field in ("sources", "environment", "config", "generation", "methods", "benchmark",
                              "seed_protocol", "early_eos", "excluded_samples_sha256", "checkpoints",
                              "warmups_per_method_per_session"):
                    if manifest[field] != full[field]:
                        raise ValueError(f"shard/full protocol mismatch: {field}")
                if (manifest["samples_sha256"] != metadata["sha256"] or manifest["limit"] != 0
                        or manifest["expected_rows"] != metadata["rows"]):
                    raise ValueError("shard data manifest mismatch")
                expected = {(method, input_key(r)) for method in METHODS
                            for r in rows(Path(metadata["samples"]))}
                if seen & expected:
                    raise ValueError("input keys overlap across the prepared shards")
                seen.update(expected)
                content_hash = hashlib.sha256()
                with sqlite3.connect(f"file:{directory}/predictions.sqlite?mode=ro", uri=True) as source:
                    if source.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                        raise ValueError("source prediction database is corrupt")
                    if set(source.execute("SELECT method,input_key FROM predictions")) != expected:
                        raise ValueError("incomplete prediction shard")
                    for method, key, payload in source.execute(
                        "SELECT method,input_key,payload FROM predictions ORDER BY method,input_key"
                    ):
                        record = json.loads(payload)
                        if (record["planned_generated_positions"] != 256 or record["planned_steps"] != 128
                                or record["executed_steps"] != len(record["trace"])
                                or not 0 < record["executed_steps"] <= 128 or record["remaining_masks"]):
                            raise ValueError("invalid POPE generation record")
                        first_closed = audit_record(record, mask, eos)
                        if first_closed is not None and first_closed != record["executed_steps"]:
                            raise ValueError("safe EOS prefix was not stopped at its first committed step")
                        content_hash.update(json.dumps([method, key, payload], ensure_ascii=False).encode())
                        existing = out.execute(
                            "SELECT payload FROM predictions WHERE method=? AND input_key=?", (method, key)
                        ).fetchone()
                        if existing is not None and existing[0] != payload:
                            raise ValueError("previously merged payload differs from its source")
                        if existing is None:
                            out.execute("INSERT INTO predictions VALUES (?,?,?)", (method, key, payload))
                receipts.append({"shard": shard, "predictions": len(expected),
                                 "ordered_payload_sha256": content_hash.hexdigest(),
                                 "manifest_sha256": digest(directory / "manifest.json")})
            if seen != expected_all or set(out.execute("SELECT method,input_key FROM predictions")) != expected_all:
                raise ValueError("merged prediction coverage differs from all 9000 questions")
    save(target / "merge_receipt.json", {
        "payloads_preserved_verbatim": True, "execution": "two independent GPU shards",
        "generation": plan["generation"], "shards": receipts,
        "rows_per_method": 9000, "unique_inputs_per_method": len(all_keys),
    })


def complete_metrics(name):
    path = BASE / name / "metrics.json"
    status_path = BASE / name / "status.json"
    if not path.exists() or not status_path.exists():
        return None
    status = json.loads(status_path.read_text())
    metrics = json.loads(path.read_text())
    if (status.get("full_dataset_complete") is not True or set(metrics) != set(METHODS)
            or any(not m.get("complete") or m.get("evaluated_rows") != 9000 for m in metrics.values())):
        return None
    return metrics


def think_answer(text):
    """Primary track: answer after the last </think>, else the full text."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip(), True
    return text, False


def scored_answer_rows(answer_list, primary):
    """Attach the scored answer text to each {method}_answers.json row."""
    out = []
    for row in answer_list:
        if primary:
            answer, closed = think_answer(row["text"])
        else:
            answer, closed = row["text"], None
        out.append({**row, "answer": answer, "think_closed": closed})
    return out


def pope_block(scored):
    """Official POPE metrics per split, pooled, and the equal 3-split mean."""
    from common.metrics import pope_metrics

    scored_rows = [{"text": r["answer"], "label": r["label"], "split": r["split"]}
                   for r in scored]
    per_split = {s: pope_metrics([r for r in scored_rows if r["split"] == s]) for s in SPLITS}
    pooled = pope_metrics(scored_rows)
    metrics = ("accuracy", "f1", "precision", "recall", "yes_ratio", "invalid_ratio")
    macro = {metric: 100 * sum(per_split[s][metric] for s in SPLITS) / 3 for metric in metrics}
    return {"splits": per_split, "pooled": pooled, "macro_percent": macro}


def think_diagnostics(scored):
    """Think-format and answer diagnostics over scored answers rows."""
    from common.metrics import official_pope_answer

    n = len(scored)
    if not n:
        return {"rows": 0}
    think_open = think_close = think_missing = yes_start_no = no_start_yes = neither = no_eos = 0
    lengths, steps, seconds = [], [], []
    for row in scored:
        text = row["text"]
        answer = row["answer"].strip().lower()
        parsed = official_pope_answer(row["answer"])
        think_open += "<think>" in text
        think_close += "</think>" in text
        think_missing += row["think_closed"] is False
        if answer.startswith("yes") and parsed == "no":
            yes_start_no += 1
        if answer.startswith("no") and parsed == "yes":
            no_start_yes += 1
        if not answer.startswith(("yes", "no")):
            neither += 1
        no_eos += not row["eos_found"]
        lengths.append(row["answer_token_length"])
        steps.append(row["executed_steps"])
        seconds.append(row["end_to_end_seconds"])
    return {
        "rows": n,
        "think_open_rate": think_open / n,
        "think_close_rate": think_close / n,
        "think_close_missing": think_missing,
        "answer_not_starting_with_yes_or_no": neither,
        "yes_start_parsed_no": yes_start_no,
        "no_start_parsed_yes": no_start_yes,
        "eos_found_rate": 1 - no_eos / n,
        "rows_without_eos_hit_256_limit": no_eos,
        "answer_token_length_mean": statistics.mean(lengths),
        "answer_token_length_median": statistics.median(lengths),
        "executed_steps_mean": statistics.mean(steps),
        "end_to_end_seconds_mean": statistics.mean(seconds),
    }


def primary_block(method_answers, native_answers, method):
    """PRIMARY track: post-</think> official metrics + diagnostics + paired bootstrap."""
    from common.metrics import paired_bootstrap

    scored = scored_answer_rows(method_answers, primary=True)
    block = pope_block(scored)
    block["diagnostics"] = think_diagnostics(scored)
    if method != "native":
        native_scored = scored_answer_rows(native_answers, primary=True)
        paired = {}
        for split in SPLITS:
            left = [{"id": r["id"], "image_sha256": r["image_sha256"],
                     "label": r["label"], "text": r["answer"]}
                    for r in native_scored if r["split"] == split]
            right = [{"id": r["id"], "image_sha256": r["image_sha256"],
                      "label": r["label"], "text": r["answer"]}
                     for r in scored if r["split"] == split]
            paired[split] = paired_bootstrap(left, right)
        block["paired_vs_native_primary"] = paired
    return block


def summarize():
    report = {
        "generation": [256, 128, 64],
        "prompt_id": PROMPT_ID,
        "prompt": ("chat-template user message = common.mmmu_protocol.INSTRUCTION + '\\n' + q0; "
                   "no system message; image <|mmu|><|soi|>[1024 codes]<|eoi|> + chat text"),
        "aggregation": "equal mean of the random, popular and adversarial split metrics",
        "tracks": {
            "primary_post_think": ("answer = text after last </think> (full text + think_close_missing "
                                   "otherwise), parsed by the unchanged official POPE parser; "
                                   "paired_vs_native_primary resamples images per split"),
            "secondary_raw_full_text": "unchanged common.run metrics: official parser on the full generated text",
        },
        "datasets": {},
    }
    for name in DATASETS:
        metrics = complete_metrics(name)
        if metrics is None:
            raise ValueError(f"refusing a summary before {name} is scored")
        native_answers = answer_rows(BASE / name / "native_answers.json")
        report["datasets"][name] = {}
        for method in METHODS:
            method_answers = answer_rows(BASE / name / f"{method}_answers.json")
            report["datasets"][name][method] = {
                "rows": len(method_answers),
                "primary_post_think": primary_block(method_answers, native_answers, method),
                "secondary_raw_full_text": metrics[method],
            }
    # Equal mean over the 9 dataset x split subsets, per method (primary track).
    # macro_percent is already a percentage averaged over the three splits, so
    # averaging it over the three datasets yields the 9-subset mean.
    report["nine_subset_mean"] = {}
    for method in METHODS:
        report["nine_subset_mean"][method] = {
            metric: sum(report["datasets"][name][method]["primary_post_think"]["macro_percent"][metric]
                        for name in DATASETS) / len(DATASETS)
            for metric in ("accuracy", "f1", "precision", "recall", "yes_ratio", "invalid_ratio")
        }
    save(BASE / "metrics.json", report)
    write_results(report)


ORDERED = ("accuracy", "f1", "precision", "recall", "yes_ratio", "invalid_ratio")


def metrics_table(track):
    """Markdown rows: three splits, equal 3-split mean, and pooled metrics."""
    lines = ["| Split | Accuracy | F1 | Precision | Recall | Yes ratio | Invalid ratio |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for label in SPLITS:
        m = track["splits"][label]
        lines.append(f"| {label} | " + " | ".join(f"{100 * m[k]:.2f}" for k in ORDERED) + " |")
    lines.append("| mean | " + " | ".join(f"{track['macro_percent'][k]:.2f}" for k in ORDERED) + " |")
    m = track["pooled"]
    lines.append("| pooled | " + " | ".join(f"{100 * m[k]:.2f}" for k in ORDERED) + " |")
    return lines


def macro_row(track):
    return " | ".join(f"{track['macro_percent'][k]:.2f}" for k in ORDERED)


def write_results(report):
    lines = ["# POPE 256/128/64 — think prompt (pope_think_v1), 3 datasets x 3 methods", "",
             "Full POPE COCO + A-OKVQA + GQA, 9000 rows per dataset and method.",
             "User message = official MMaDA thinking instruction + '\\n' + question without",
             "' Please answer yes or no.'; no system message; image <|mmu|><|soi|>[1024 codes]",
             "<|eoi|> + chat text. Seed 42, crop 512, temperature 0, CFG 0, low_confidence,",
             "VCD alpha=1.0 beta=0.1 noise_step=500, safe EOS stopping (safe_eos_v1).",
             "PRIMARY track: answer = text after the last </think> (full text counted as",
             "think_close_missing otherwise), parsed by the unchanged official POPE parser.",
             "SECONDARY track: unchanged common.run metrics on the full generated text.",
             "Values are percentages; 'mean' is the equal average of the three splits,",
             "'pooled' scores all rows jointly. F1 is not pooled.", ""]
    for name in DATASETS:
        lines += [f"## {name}", ""]
        lines += ["### Primary (post-think answer) — macro over splits", "",
                  "| Method | Accuracy | F1 | Precision | Recall | Yes ratio | Invalid ratio |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for method in METHODS:
            lines.append(f"| {method} | {macro_row(report['datasets'][name][method]['primary_post_think'])} |")
        lines += ["", "### Secondary (full generated text, official parser) — macro over splits", "",
                  "| Method | Accuracy | F1 | Precision | Recall | Yes ratio | Invalid ratio |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for method in METHODS:
            secondary = report["datasets"][name][method]["secondary_raw_full_text"]
            macro = {metric: 100 * sum(secondary["pope"][s][metric] for s in SPLITS) / 3
                     for metric in ORDERED}
            lines.append(f"| {method} | " + " | ".join(f"{macro[k]:.2f}" for k in ORDERED) + " |")
        lines += ["", "### Primary diagnostics", "",
                  "| Method | rows | </think> rate | think_close_missing | not yes/no | yes→no | no→yes | "
                  "EOS rate | no-EOS@256 | ans len mean/med | steps mean | sec mean |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
        for method in METHODS:
            d = report["datasets"][name][method]["primary_post_think"]["diagnostics"]
            lines.append(
                f"| {method} | {d['rows']} | {d['think_close_rate']:.4f} | {d['think_close_missing']} | "
                f"{d['answer_not_starting_with_yes_or_no']} | {d['yes_start_parsed_no']} | "
                f"{d['no_start_parsed_yes']} | {d['eos_found_rate']:.4f} | "
                f"{d['rows_without_eos_hit_256_limit']} | {d['answer_token_length_mean']:.1f}/"
                f"{d['answer_token_length_median']:.1f} | {d['executed_steps_mean']:.1f} | "
                f"{d['end_to_end_seconds_mean']:.2f} |")
        for method in ("vcd", "vigor"):
            paired = report["datasets"][name][method]["primary_post_think"].get("paired_vs_native_primary")
            if paired:
                lines += ["", f"### {method} vs native paired bootstrap (primary, image-resampled)", "",
                          "| Split | Δ accuracy 95% CI | Δ F1 95% CI |",
                          "|---|---|---|"]
                for split in SPLITS:
                    p = paired[split]
                    lines.append(f"| {split} | [{p['delta_accuracy_ci95'][0]:+.4f}, "
                                 f"{p['delta_accuracy_ci95'][1]:+.4f}] | [{p['delta_f1_ci95'][0]:+.4f}, "
                                 f"{p['delta_f1_ci95'][1]:+.4f}] |")
        lines.append("")
    lines += ["## Nine-subset mean (3 datasets x 3 splits), primary post-think", "",
              "| Method | Accuracy | F1 | Precision | Recall | Yes ratio | Invalid ratio |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        m = report["nine_subset_mean"][method]
        lines.append(f"| {method} | " + " | ".join(f"{m[k]:.2f}" for k in ORDERED) + " |")
    lines.append("")
    (BASE / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main():
    plan = json.loads((BASE / "plan.json").read_text())
    verify_plan(plan)
    with (BASE / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = {"state": "waiting_for_gpus", "complete": False, "pid": os.getpid(),
                  "started": time.time(), "stages": {}}

        def update(**fields):
            status.update(fields, updated=time.time())
            save(BASE / "queue_status.json", status)

        update()
        try:
            while not (inventory := gpu_inventory()):
                update()
                time.sleep(30)
            update(gpu_inventory=inventory)
            for name, data in plan["datasets"].items():
                verify_plan(plan)
                if complete_metrics(name) is not None:
                    status["stages"][name] = {"state": "generation_and_scoring_complete"}
                    continue
                if gpu_inventory() is None:
                    raise RuntimeError("GPUs are occupied; refusing to contend with another job")
                status["stages"][name] = {"state": "generating", "started": time.time()}
                update(state="running", current_dataset=name)
                jobs = [(command(shard["samples"], BASE / f"{name}_shard{i}", data["exclude_samples"]),
                         str(i), BASE / f"{name}_shard{i}.log")
                        for i, shard in enumerate(data["shards"])]
                run_jobs(jobs, lock.fileno())
                verify_plan(plan)
                status["stages"][name]["state"] = "merging_and_scoring"
                update()
                scoring = (command(data["samples"], BASE / name, data["exclude_samples"], score_only=True),
                           "0", BASE / f"{name}_score.log")
                # Establish the full-data manifest through the unchanged runner.
                run_jobs([scoring], lock.fileno())
                merge_predictions(name, data, plan)
                run_jobs([scoring], lock.fileno())
                if complete_metrics(name) is None:
                    raise ValueError(f"{name}: generation/scoring incomplete")
                status["stages"][name].update(state="generation_and_scoring_complete", finished=time.time())
                update()
            # Experiment-level think-aware scoring once every dataset is merged.
            summarize()
            update(state="generation_and_scoring_complete", complete=True, finished=time.time())
        except BaseException as error:
            update(state="blocked", error=repr(error))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.check:
        verify_plan(json.loads((BASE / "plan.json").read_text()))
        print("Queued source/config/data hashes match. No model was loaded.")
    else:
        def terminate(signum, frame):
            raise KeyboardInterrupt(f"received signal {signum}")
        signal.signal(signal.SIGTERM, terminate)
        main()
