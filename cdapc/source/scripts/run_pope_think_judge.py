"""Judge all six POPE think stages, including stages that finish later."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

from cdapc_lab.evaluation.llm_judge import judge_key
from cdapc_lab.evaluation.pope_llm_judge import judge_request, request_for, spec
from cdapc_lab.experiments.score import pope_metrics
from cdapc_lab.io.artifacts import atomic_json, digest, read_json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/pope_think_llm_judge"
STAGES = tuple(f"pope_{dataset}_reasoning_r{tau}" for dataset in ("gqa", "aokvqa") for tau in (90, 70, 50))


def collect(index_path=None):
    index = read_json(index_path or ROOT / "reports/chair_pope_extensions_runs.json")
    ready = {}
    requests = {}
    for name in STAGES:
        stage = index["stages"][name]
        if stage["status"] != "complete":
            continue
        report = read_json(ROOT / stage["report"])
        config = read_json(ROOT / report["runs"][0] / "resolved_config.json")
        assert config["protocol"]["prompt_id"] == "pope_yesno_reasoning_v1"
        data = read_json(ROOT / config["protocol"]["data_root"] / "rows.json")
        questions = {r["input_key"]: r["question"] for r in data}
        records = {}
        for item in report["cdapc"]["per_run"]:
            folder = ROOT / item["run"] / "scores" / item["evaluation_fingerprint"]
            for path in sorted(folder.glob("*.json")):
                if path.name == "evaluation_manifest.json":
                    continue
                row = read_json(path)
                sid = row["sample_id"]
                assert sid not in records and row["generation_status"] in ("success", "budget_exhausted")
                text = row["normalized"]["text"]
                key = None
                if row["generation_status"] == "success":
                    request = request_for(questions[sid], text)
                    key = digest(request)
                    requests[key] = request
                records[sid] = {
                    "request_sha256": key, "score_path": str(path.relative_to(ROOT)),
                    "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "generation_status": row["generation_status"],
                    "gold": row["gold"], "strict_prediction": (
                        row["parsed"]["prediction"] if row["generation_status"] == "success" else None),
                }
        assert set(records) == set(questions) and len(data) == 9000
        assert all(records[r["input_key"]]["gold"] == r["label"] for r in data)
        ready[name] = {"records": records, "rows": data, "source_report": stage["report"]}
    return index, ready, requests


def reports(ready, results, index, cache, run_status, workers):
    old = read_json(OUTPUT / "index.json") if (OUTPUT / "index.json").exists() else {}
    previously_complete = {name for name, stage in old.get("stages", {}).items()
                           if stage.get("status") == "complete"}
    summary = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": run_status, "runner_pid": os.getpid(), "workers": workers,
        "judge_spec": spec(), "judge_spec_hash": digest(spec()),
        "generation_queue_status": index["status"], "generation_active": index["active"],
        "cache": str(cache.relative_to(ROOT)), "stages": {},
    }
    for name in STAGES:
        if name not in ready:
            generation = index["stages"][name]
            summary["stages"][name] = {
                "status": "cancelled" if generation["status"] == "cancelled" else "waiting_for_generation",
                "generation_status": generation["status"],
            }
            if generation["status"] == "cancelled":
                summary["stages"][name]["cancellation_reason"] = generation.get("cancellation_reason")
            continue
        source = ready[name]
        judged = {}
        for sid, row in source["records"].items():
            if row.get("generation_status") == "budget_exhausted":
                judged[sid] = {
                    **row, "judge_prediction": None, "judge_status": "budget_exhausted",
                    "evidence": "", "judge_correct": 0, "strict_correct": 0,
                }
                continue
            result = results.get(row["request_sha256"])
            if result and result["status"] == "success":
                parsed = result["parsed"]
                judged[sid] = {
                    **row, "judge_prediction": parsed["answer"], "judge_status": parsed["status"],
                    "evidence": parsed["evidence"],
                    "judge_correct": int(parsed["answer"] is not None and parsed["answer"] == row["gold"]),
                    "strict_correct": int(row["strict_prediction"] == row["gold"]),
                }
        errors = sum(results.get(r["request_sha256"], {}).get("status") == "judge_error"
                     for r in source["records"].values())
        complete = len(judged) == len(source["records"])
        splits = {}
        transitions = Counter()
        for row in source["rows"]:
            entry = judged.get(row["input_key"])
            if entry:
                transitions[f"strict_{entry['strict_correct']}_judge_{entry['judge_correct']}"] += 1
        for split in ("random", "popular", "adversarial"):
            rows = [r for r in source["rows"] if r["split"] == split and r["input_key"] in judged]
            pairs = [(judged[r["input_key"]]["judge_prediction"], r["label"]) for r in rows]
            splits[split] = {
                **pope_metrics(pairs), "n": len(rows),
                "missing": sum(judged[r["input_key"]]["judge_status"] == "missing" for r in rows),
                "ambiguous": sum(judged[r["input_key"]]["judge_status"] == "ambiguous" for r in rows),
                "budget_exhausted": sum(
                    judged[r["input_key"]]["judge_status"] == "budget_exhausted" for r in rows),
            }
        stage = {
            "status": "complete" if complete else "judging",
            "expected_unique": len(source["records"]), "judged_unique": len(judged),
            "api_judged_unique": sum(r["judge_status"] != "budget_exhausted" for r in judged.values()),
            "budget_exhausted_inputs": sum(r["judge_status"] == "budget_exhausted" for r in judged.values()),
            "api_error_inputs": errors, "answer_statuses_unique": dict(Counter(
                row["judge_status"] for row in judged.values())),
            "metric_scope": "all_9000_rows" if complete else "judged_subset_only",
            "splits": splits, "paired_transitions_vs_strict": dict(transitions),
            "source_report": source["source_report"],
        }
        if complete:
            stage["macro"] = {k: statistics.mean(s[k] for s in splits.values())
                              for k in ("accuracy", "f1", "precision", "recall", "yes_ratio")
                              if all(s[k] is not None for s in splits.values())}
        atomic_json(OUTPUT / f"{name}.json", {**stage, "judge_spec_hash": digest(spec()),
                                             "per_input": judged})
        summary["stages"][name] = stage
    atomic_json(OUTPUT / "index.json", summary)
    newly_complete = {name for name, stage in summary["stages"].items()
                      if stage["status"] == "complete"} - previously_complete
    if newly_complete and (ROOT / "scripts/summarize_experiments.py").is_file():
        completed = subprocess.run([sys.executable, "scripts/summarize_experiments.py"], cwd=ROOT)
        if completed.returncode:
            print("SUMMARY_REFRESH_FAILED", completed.returncode, flush=True)
    return summary


def main():
    global OUTPUT, STAGES
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--generation-index", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--stages", nargs="+")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    if args.output:
        OUTPUT = ROOT / args.output
    if args.stages:
        STAGES = tuple(args.stages)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache = ROOT / args.cache if args.cache else OUTPUT / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    # All users of the same request cache share its existing lock.
    with (cache.parent / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key = judge_key()
        if not key:
            raise RuntimeError("DeepSeek API credential unavailable")
        atomic_json(OUTPUT / "judge_manifest.json", spec())
        submitted = 0
        try:
            while True:
                index, ready, requests = collect(
                    ROOT / args.generation_index if args.generation_index else None)
                results = {sha: read_json(cache / f"{sha}.json") for sha in requests
                           if (cache / f"{sha}.json").is_file()}
                pending = [sha for sha in requests if results.get(sha, {}).get("status") != "success"]
                stuck = [sha for sha in pending if len(results.get(sha, {}).get("attempts", [])) >= 9]
                if stuck:
                    reports(ready, results, index, cache, "needs_attention", args.workers)
                    raise RuntimeError(f"{len(stuck)} requests failed 9 attempts; inspect cached errors")
                if args.limit:
                    pending = pending[:max(0, args.limit - submitted)]
                for offset in range(0, len(pending), 64):
                    batch = pending[offset:offset + 64]
                    failures = 0
                    with ThreadPoolExecutor(max_workers=args.workers) as pool:
                        futures = {pool.submit(judge_request, requests[sha], cache, key): sha for sha in batch}
                        for future in as_completed(futures):
                            sha = futures[future]
                            results[sha] = future.result()
                            failures += results[sha]["status"] != "success"
                    submitted += len(batch)
                    reports(ready, results, index, cache, "judging", args.workers)
                    print("API_REQUESTS", submitted, "CACHE_SUCCEEDED",
                          sum(r["status"] == "success" for r in results.values()),
                          "BATCH_ERRORS", failures, flush=True)
                    if failures >= max(4, len(batch) // 2):
                        raise RuntimeError("High API/response-format failure rate; inspect cache")
                summary = reports(ready, results, index, cache, "waiting_for_generation", args.workers)
                if all(s["status"] in ("complete", "cancelled") for s in summary["stages"].values()):
                    summary["status"] = "complete"
                    atomic_json(OUTPUT / "index.json", summary)
                    print("ALL NON-CANCELLED THINK STAGES JUDGED", flush=True)
                    break
                if args.once or (args.limit and submitted >= args.limit):
                    summary["status"] = "paused"
                    atomic_json(OUTPUT / "index.json", summary)
                    break
                time.sleep(30)
        except Exception:
            state = read_json(OUTPUT / "index.json") if (OUTPUT / "index.json").exists() else {}
            state.update(status="failed", error=traceback.format_exc())
            atomic_json(OUTPUT / "index.json", state)
            raise


if __name__ == "__main__":
    main()
