"""Resume-safe MMMU r70 run after the existing CHAIR/POPE GPU queue."""
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import yaml

from cdapc_lab.config import resolve_config
from cdapc_lab.experiments.infer import preflight
from cdapc_lab.io.artifacts import atomic_json
from run_chair_pope_extensions import completed, existing_worker, locate_run, read

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "reports/mmmu_r70_256_128_runs.json"
DEPENDENCY = ROOT / "reports/chair_pope_extensions_runs.json"
LOGS = ROOT / "logs/mmmu"


def configs():
    result = []
    for shard in range(3):
        source = yaml.safe_load(
            (ROOT / f"configs/experiments/mmada_mmmu_cdapc_r90_shard{shard}.yaml").read_text())
        value = copy.deepcopy(source)
        value["run"]["name"] = f"mmada_mmmu_cdapc_r70_L256_I128_shard{shard}"
        value["cdapc"].update(iteration_limit=128, budget_limited=True)
        value["cdapc"]["r_gate"]["threshold"] = 0.7
        assert value["protocol"]["canvas_length"] == 256
        path = ROOT / f"configs/experiments/{value['run']['name']}.yaml"
        if path.exists():
            assert yaml.safe_load(path.read_text()) == value
        else:
            path.write_text(yaml.safe_dump(value, sort_keys=False))
        resolve_config(path)
        result.append((path, value))
    return result


def save(state):
    state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S%z")
    atomic_json(STATE, state)


def select_generation_run(item, path):
    """Start a separate run after a runtime change; never relabel old records."""
    _, _, provenance = preflight(resolve_config(path))
    fingerprint = provenance["generation_fingerprint"]
    previous_run = item.get("run")
    if previous_run:
        previous = read(ROOT / previous_run / "run_manifest.json")
        if previous["generation_fingerprint"] != fingerprint:
            item.setdefault("previous_runs", []).append({
                "run": previous_run,
                "generation_fingerprint": previous["generation_fingerprint"],
                "environment": previous["environment"],
                "saved_predictions": len(list((ROOT / previous_run / "predictions").glob("*.json"))),
                "reason": "generation_fingerprint_changed; original artifacts retained",
            })
            print("NEW RUN REQUIRED", previous_run, "generation fingerprint changed", flush=True)
            previous_run = None
    item.update(run=previous_run or locate_run(item["name"], fingerprint),
                generation_fingerprint=fingerprint, environment=provenance["environment"])
    item["generated"] = (len(list((ROOT / item["run"] / "predictions").glob("*.json")))
                         if item["run"] else 0)


def main():
    plan = configs()
    with (LOGS / "r70_L256_I128.lock").open("a") as own_lock:
        fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read(STATE) if STATE.exists() else {"status": "queued", "shards": {}}
        if state["status"] == "complete":
            return
        state.update(controller_pid=os.getpid(), dependency=str(DEPENDENCY.relative_to(ROOT)),
                     canvas_length=256, iteration_limit=128, budget_limited=True, r_gate=0.7,
                     expected_inputs=900, comparison_note=(
                         "Existing r90/r50 use L256/I256; this run also changes iteration budget."))
        for i, (path, value) in enumerate(plan):
            state["shards"].setdefault(str(i), {}).update(
                config=str(path.relative_to(ROOT)), name=value["run"]["name"])
        save(state)
        try:
            while read(DEPENDENCY)["status"] != "complete":
                state["status"] = "waiting_for_existing_gpu_queue"
                save(state)
                time.sleep(30)
            # Same lock as the existing controller; never contend with its workers.
            with (ROOT / "logs/benchmark_extensions/queue.lock").open("a") as gpu_lock:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX)
                state["status"] = "generating"
                for i, (path, _) in enumerate(plan):
                    item = state["shards"][str(i)]
                    if not completed(item.get("run")):
                        select_generation_run(item, path)
                save(state)
                for attempt in range(2):
                    workers = {}
                    for i, (path, value) in enumerate(plan):
                        item = state["shards"][str(i)]
                        item["run"] = item.get("run") or locate_run(
                            value["run"]["name"], item.get("generation_fingerprint"))
                        if completed(item["run"]):
                            continue
                        worker = existing_worker(path)
                        if worker is None:
                            command = [sys.executable, "-m", "cdapc_lab", "infer", "--config", str(path)]
                            if item["run"]:
                                command += ["--resume-run", item["run"]]
                            with (LOGS / f"r70_L256_I128_shard{i}.log").open("a") as log:
                                worker = subprocess.Popen(command, cwd=ROOT, stdout=log,
                                    stderr=subprocess.STDOUT, env={**os.environ,
                                    "CUDA_VISIBLE_DEVICES": str(i), "OMP_NUM_THREADS": "8",
                                    "MKL_NUM_THREADS": "8"})
                        item["worker_pid"] = worker.pid
                        workers[i] = worker
                    while workers:
                        for i in list(workers):
                            item = state["shards"][str(i)]
                            item["run"] = item["run"] or locate_run(
                                item["name"], item.get("generation_fingerprint"))
                            if item["run"]:
                                item["generated"] = len(list((ROOT / item["run"] / "predictions").glob("*.json")))
                            if workers[i].poll() is not None:
                                del workers[i]
                        save(state)
                        if workers:
                            time.sleep(30)
                    if all(completed(item["run"]) for item in state["shards"].values()):
                        break
                    if attempt == 1:
                        raise RuntimeError("MMMU generation incomplete after retry; see shard logs")
            runs = [state["shards"][str(i)]["run"] for i in range(3)]
            state["status"] = "scoring"
            save(state)
            for i, run in enumerate(runs):
                with (LOGS / f"r70_L256_I128_score{i}.log").open("a") as log:
                    subprocess.run([sys.executable, "-m", "cdapc_lab", "score", "--run", run],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            state["status"] = "judging"
            save(state)
            def judge(run):
                with (LOGS / f"r70_L256_I128_judge_{Path(run).name}.log").open("a") as log:
                    subprocess.run([sys.executable, "-m", "cdapc_lab", "mmmu-judge", "--run", run],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            for attempt in range(3):
                with ThreadPoolExecutor(max_workers=3) as pool:
                    list(pool.map(judge, runs))
                failures = sum(
                    read(path).get("status") == "judge_error"
                    for run in runs for path in (ROOT / run / "judge").glob("*/*.json")
                    if path.name != "judge_manifest.json")
                if not failures:
                    break
                if attempt < 2:
                    time.sleep(30)
            native = read(ROOT / "reports/mmmu_runs.json")["native"]
            output = "reports/mmmu_cdapc_r70_L256_I128.json"
            with (LOGS / "r70_L256_I128_report.log").open("a") as log:
                subprocess.run([sys.executable, "-m", "cdapc_lab", "mmmu-report", "--runs", *runs,
                    "--native-runs", *[native[f"shard{i}"] for i in range(3)], "--output", output],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = read(ROOT / output)
            if result["cdapc"]["judge_errors"] or result["cdapc"]["judged"] != 900:
                raise RuntimeError("Generation finished; MMMU Judge incomplete, inspect report")
            state.update(status="complete", report=output)
            save(state)
            subprocess.run([sys.executable, "scripts/summarize_experiments.py"],
                           cwd=ROOT, check=True)
            print("COMPLETE", output, flush=True)
        except Exception:
            state.update(status="failed", error=traceback.format_exc())
            save(state)
            raise


if __name__ == "__main__":
    main()
