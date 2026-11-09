"""Durable sequential benchmark queue; three GPU shards per experiment."""
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import yaml
from cdapc_lab.io.artifacts import atomic_json

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "reports/chair_pope_extensions_runs.json"
LOGS = ROOT / "logs/benchmark_extensions"
POPE_DATASETS = ("pope_gqa", "pope_aokvqa")


def read(path):
    return json.loads(Path(path).read_text())


def completed(run):
    if not run or not (ROOT / run / "run_manifest.json").exists():
        return False
    manifest = read(ROOT / run / "run_manifest.json")
    if manifest["status"] == "complete":
        return True
    # A deliberately bounded generation has finished even if some samples
    # exhausted the budget. They retain that status and score as failures.
    if manifest["status"] != "incomplete":
        return False
    config = read(ROOT / run / "resolved_config.json")
    counts = manifest.get("sample_status_counts", {})
    return (config.get("cdapc", {}).get("budget_limited", False)
            and counts.get("budget_exhausted", 0) > 0
            and set(counts) <= {"success", "budget_exhausted"}
            and sum(counts.values()) == manifest["sample_count"])


def locate_run(name, generation_fingerprint=None):
    matches = []
    for path in (ROOT / "runs").glob("*/resolved_config.json"):
        config = read(path)
        if config["run"]["name"] == name:
            if generation_fingerprint is not None:
                manifest = path.parent / "run_manifest.json"
                if (not manifest.exists()
                        or read(manifest)["generation_fingerprint"] != generation_fingerprint):
                    continue
            matches.append(str(path.parent.relative_to(ROOT)))
    return sorted(matches)[-1] if matches else None


def process_state(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[0], fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None, None


class AdoptedWorker:
    """Monitor an existing inference worker without restarting or signaling it."""
    def __init__(self, pid):
        self.pid = pid
        self.started = process_state(pid)[1]

    def poll(self):
        state, started = process_state(self.pid)
        return None if started == self.started and state not in (None, "Z", "X") else "exit_observed"


def existing_worker(config_path):
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = [s.decode(errors="surrogateescape") for s in (entry / "cmdline").read_bytes().split(b"\0") if s]
            if "cdapc_lab" not in args or "infer" not in args or "--config" not in args:
                continue
            if (entry / "cwd").resolve() != ROOT:
                continue
            path = Path(args[args.index("--config") + 1])
            if (ROOT / path).resolve() == config_path.resolve():
                worker = AdoptedWorker(int(entry.name))
                if worker.poll() is None:
                    matches.append(worker)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    if len(matches) > 1:
        raise RuntimeError(f"Multiple live workers for {config_path}")
    return matches[0] if matches else None


def stage_key(entry):
    return f"{entry['dataset']}_{'reasoning_' if entry['reasoning'] else ''}r{entry['tau']}"


def stage_plan(active=None):
    plan = ([{"dataset": "chair", "tau": tau, "reasoning": False} for tau in (90, 70, 50)]
            + [{"dataset": dataset, "tau": tau, "reasoning": reasoning}
               for reasoning in (False, True)
               for dataset in POPE_DATASETS
               for tau in (90, 70, 50)])
    if active is not None:
        current = next((entry for entry in plan if stage_key(entry) == active), None)
        if current is None:
            raise ValueError(f"Active stage missing from plan: {active}")
        # Existing workers own the GPUs until this stage finishes. Inserting an
        # earlier stage on restart must not launch another set alongside them.
        plan.remove(current)
        plan.insert(0, current)
    return plan


def stage_configs(dataset, tau, reasoning=False):
    configs = []
    for shard in range(3):
        source = ROOT / f"configs/experiments/mmada_{dataset}_r{tau}_shard{shard}.yaml"
        config = yaml.safe_load(source.read_text())
        if reasoning:
            config = copy.deepcopy(config)
            config["protocol"].update(prompt_id="pope_yesno_reasoning_v1", strict_parser_id="pope_final_yes_no_v1")
            config["run"]["name"] = f"mmada_{dataset}_reasoning_r{tau}_shard{shard}"
            source = ROOT / f"configs/experiments/{config['run']['name']}.yaml"
            if source.exists():
                assert yaml.safe_load(source.read_text()) == config
            else:
                source.write_text(yaml.safe_dump(config, sort_keys=False))
        configs.append((source, config))
    return configs


def run_stage(index, dataset, tau, reasoning=False):
    key = stage_key({"dataset": dataset, "tau": tau, "reasoning": reasoning})
    stage = index["stages"].setdefault(key, {"status": "pending", "shards": {}})
    if stage["status"] == "cancelled":
        print("SKIP CANCELLED", key, flush=True)
        return
    if stage["status"] == "complete" and (ROOT / stage["report"]).is_file():
        print("SKIP COMPLETED", key, flush=True)
        return
    configs = stage_configs(dataset, tau, reasoning)
    for shard, (source, config) in enumerate(configs):
        item = stage["shards"].setdefault(str(shard), {})
        item.update(config=str(source.relative_to(ROOT)), log=str((LOGS / f"{key}_shard{shard}.log").relative_to(ROOT)))
        if not item.get("run"):
            item["run"] = locate_run(config["run"]["name"])
    index.update(active=key, status="running")
    stage["status"] = "generating"
    atomic_json(INDEX, index)
    print("START", key, time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    for attempt in range(2):
        workers = {}
        for shard, (source, config) in enumerate(configs):
            item = stage["shards"][str(shard)]
            if completed(item.get("run")):
                continue
            worker = existing_worker(source)
            if worker is not None:
                workers[shard] = worker
                item.update(worker_pid=worker.pid, adopted=True)
                print("ADOPT", key, "shard", shard, "pid", worker.pid, flush=True)
                continue
            if completed(item.get("run")):
                continue
            command = [sys.executable, "-m", "cdapc_lab", "infer", "--config", str(source.relative_to(ROOT))]
            if item.get("run"):
                command += ["--resume-run", item["run"]]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(shard), "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"}
            with (ROOT / item["log"]).open("a") as log:
                workers[shard] = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                item.update(worker_pid=workers[shard].pid, adopted=False)
        while workers:
            for shard in list(workers):
                item = stage["shards"][str(shard)]
                if not item.get("run"):
                    item["run"] = locate_run(configs[shard][1]["run"]["name"])
                if item.get("run"):
                    item["predictions"] = len(list((ROOT / item["run"] / "predictions").glob("*.json")))
                code = workers[shard].poll()
                if code is not None:
                    if isinstance(code, int):
                        item["returncode"] = code
                    else:
                        item["adopted_exit_observed"] = True
                    del workers[shard]
            atomic_json(INDEX, index)
            if workers:
                print(key, "predictions", {s: x.get("predictions", 0) for s, x in stage["shards"].items()}, flush=True)
                time.sleep(30)
        if all(completed(item.get("run")) for item in stage["shards"].values()):
            break
        if attempt == 1:
            raise RuntimeError(f"Incomplete generation after retry: {key}; see {LOGS}")
    runs = [stage["shards"][str(shard)]["run"] for shard in range(3)]
    stage["status"] = "scoring"
    atomic_json(INDEX, index)
    output = f"reports/{key}.json"
    if dataset == "chair":
        command = [sys.executable, "scripts/score_chair_runs.py", "--runs", *runs, "--output", output]
    else:
        for shard, run in enumerate(runs):
            with (LOGS / f"{key}_score{shard}.log").open("a") as log:
                subprocess.run([sys.executable, "-m", "cdapc_lab", "score", "--run", run], cwd=ROOT,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        command = [sys.executable, "-m", "cdapc_lab", "pope-report", "--runs", *runs, "--output", output]
    with (LOGS / f"{key}_report.log").open("a") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = read(ROOT / output)
    stage.update(status="complete", report=output,
                 metrics=result["overall_metrics"] if dataset == "chair" else result["cdapc"]["splits"],
                 completed_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    atomic_json(INDEX, index)
    print("COMPLETE", key, stage["metrics"], flush=True)


def main():
    LOGS.mkdir(exist_ok=True)
    with (LOGS / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        index = read(INDEX) if INDEX.exists() else {"stages": {}, "threads_per_worker": 8, "gpus": [0, 1, 2]}
        active = index.get("active")
        if index["stages"].get(active, {}).get("status") in ("complete", "cancelled"):
            active = None
        index.update(plan=stage_plan(active), order=["chair", *POPE_DATASETS], queue_pid=os.getpid())
        for entry in index["plan"]:
            index["stages"].setdefault(stage_key(entry), {"status": "pending", "shards": {}})
        atomic_json(INDEX, index)
        try:
            for entry in index["plan"]:
                run_stage(index, **entry)
            index.update(status="complete", active=None)
            atomic_json(INDEX, index)
            print("ALL QUEUED BENCHMARKS FINISHED", flush=True)
        except Exception:
            index.update(status="failed", error=traceback.format_exc())
            atomic_json(INDEX, index)
            raise


if __name__ == "__main__":
    main()
