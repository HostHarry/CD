"""Paired GQA think r70 K=16/4/2 pilot on a frozen random image subset."""
import argparse
import copy
import fcntl
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import yaml

from cdapc_lab.config import resolve_config
from cdapc_lab.io.artifacts import atomic_json, digest, read_json
from run_chair_pope_extensions import completed, existing_worker, locate_run

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/k_commit_pilot"
LOGS = ROOT / "logs/k_commit_pilot"
STATE = OUT / "index.json"
SEED = 20261108


def prepare():
    OUT.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    data = read_json(ROOT / "data/pope_gqa/rows.json")
    images = sorted({r["image"] for r in data})
    assert len(images) == 500
    chosen = sorted(random.Random(SEED).sample(images, 30))
    rows = [r for r in data if r["image"] in set(chosen)]
    ids = {r["input_key"] for r in rows}
    for split in ("random", "popular", "adversarial"):
        subset = [r for r in rows if r["split"] == split]
        assert len(subset) == 180 and sum(r["label"] == "yes" for r in subset) == 90
    protocol = {
        "selection_seed": SEED, "selection_unit": "image", "images": chosen,
        "unique_inputs": len(ids), "evaluation_rows": len(rows),
        "selected_rows_sha256": digest(rows),
        "sampling": "30 uniformly sampled images, all questions; independent of outputs/gold",
        "baseline": "reports/pope_gqa_reasoning_r70.json",
        "conditions": [16, 4, 2], "r_gate": .7, "window": 16, "canvas": 128, "iterations": 128,
        "note": "Pilot only; compare image-paired results, not against full-dataset scores",
    }
    for path, value in ((OUT / "protocol.json", protocol), (OUT / "rows.json", rows)):
        if path.exists():
            assert read_json(path) == value, "Frozen pilot selection changed"
        else:
            atomic_json(path, value)
    plans = {}
    seen = set()
    for shard in range(3):
        source_path = ROOT / f"configs/experiments/mmada_pope_gqa_reasoning_r70_shard{shard}.yaml"
        base = yaml.safe_load(source_path.read_text())
        selected = [sid for sid in read_json(ROOT / base["run"]["sample_ids_file"]) if sid in ids]
        assert selected and not seen.intersection(selected)
        seen.update(selected)
        manifest = f"manifests/gqa_think_r70_k_pilot_shard{shard}_ids.json"
        if (ROOT / manifest).exists():
            assert read_json(ROOT / manifest) == selected
        else:
            atomic_json(ROOT / manifest, selected)
        for k in (4, 2):
            config = copy.deepcopy(base)
            config["run"].update(name=f"mmada_gqa_think_r70_k{k}_pilot_shard{shard}",
                                 sample_ids_file=manifest)
            config["cdapc"]["max_commit_per_iteration"] = k
            path = ROOT / f"configs/experiments/{config['run']['name']}.yaml"
            if path.exists():
                assert yaml.safe_load(path.read_text()) == config
            else:
                path.write_text(yaml.safe_dump(config, sort_keys=False))
            checked = resolve_config(path)
            original = resolve_config(source_path)
            original["run"].update(name=checked["run"]["name"], sample_ids_file=manifest)
            original["cdapc"]["max_commit_per_iteration"] = k
            assert checked == original
            plans.setdefault(str(k), []).append((path, config))
    assert seen == ids
    state = read_json(STATE) if STATE.exists() else {
        "status": "prepared", "conditions": {str(k): {"status": "pending", "shards": {}}
                                            for k in (4, 2)}}
    state.update(protocol_sha256=digest(protocol), unique_inputs=len(ids), images=30,
                 evaluation_rows=len(rows), baseline_k16="reuse_existing_same_inputs")
    for k, plan in plans.items():
        for shard, (path, config) in enumerate(plan):
            state["conditions"][k]["shards"].setdefault(str(shard), {}).update(
                config=str(path.relative_to(ROOT)), name=config["run"]["name"],
                expected=len(read_json(ROOT / config["run"]["sample_ids_file"])))
    save(state)
    return state, plans


def save(state):
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    atomic_json(STATE, state)


def generate(state, k, plan):
    stage = state["conditions"][k]
    stage["status"] = "generating"
    state.update(status="generating", active_k=int(k))
    save(state)
    for attempt in range(2):
        workers = {}
        for i, (path, config) in enumerate(plan):
            item = stage["shards"][str(i)]
            item["run"] = item.get("run") or locate_run(config["run"]["name"])
            if completed(item["run"]):
                continue
            worker = existing_worker(path)
            if worker is None:
                command = [sys.executable, "-m", "cdapc_lab", "infer", "--config", str(path)]
                if item["run"]:
                    command += ["--resume-run", item["run"]]
                with (LOGS / f"k{k}_shard{i}.log").open("a") as log:
                    worker = subprocess.Popen(command, cwd=ROOT, stdout=log,
                        stderr=subprocess.STDOUT, env={**os.environ,
                        "CUDA_VISIBLE_DEVICES": str(i), "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"})
            item["worker_pid"] = worker.pid
            workers[i] = worker
        while workers:
            for i in list(workers):
                item = stage["shards"][str(i)]
                item["run"] = item["run"] or locate_run(item["name"])
                if item["run"]:
                    item["generated"] = len(list((ROOT / item["run"] / "predictions").glob("*.json")))
                if workers[i].poll() is not None:
                    del workers[i]
            save(state)
            if workers:
                time.sleep(30)
        if all(completed(item["run"]) for item in stage["shards"].values()):
            stage["status"] = "generated"
            save(state)
            return
    raise RuntimeError(f"K{k} generation incomplete after retry")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    LOGS.mkdir(exist_ok=True)
    with (LOGS / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state, plans = prepare()
        if args.prepare:
            print("PREPARED", state["unique_inputs"], "inputs", state["evaluation_rows"], "rows")
            return
        if state["status"] == "complete":
            return
        state["controller_pid"] = os.getpid()
        save(state)
        try:
            # Preserve the already scheduled MMMU job, then share the GPU queue lock.
            while True:
                dependency = read_json(ROOT / "reports/mmmu_r70_256_128_runs.json")
                shards = dependency.get("shards", {})
                if len(shards) == 3 and all(completed(x.get("run")) for x in shards.values()):
                    break
                state["status"] = "waiting_for_mmmu_generation"
                save(state)
                time.sleep(30)
            with (ROOT / "logs/benchmark_extensions/queue.lock").open("a") as gpu_lock:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX)
                for k, plan in plans.items():
                    generate(state, k, plan)
            state["status"] = "scoring_and_judging"
            save(state)
            from k_commit_report import evaluate
            evaluate(state)
            state.update(status="complete", report="reports/k_commit_pilot/results.json",
                         markdown="reports/k_commit_pilot/results.md")
            save(state)
            refreshed = subprocess.run([sys.executable, "scripts/summarize_experiments.py"], cwd=ROOT)
            state["summary_returncode"] = refreshed.returncode
            save(state)
            print("K COMMIT PILOT COMPLETE", flush=True)
        except Exception:
            state.update(status="failed", error=traceback.format_exc())
            save(state)
            raise


if __name__ == "__main__":
    main()
