"""Durable MME queue after the existing K4/K2 GPU generation."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / "cdapc-experiments"
sys.path.insert(0, str(ROOT / "src"))
from cdapc_lab.config import resolve_config
from cdapc_lab.experiments.infer import preflight
from cdapc_lab.io.artifacts import atomic_json, read_json

STATE = BASE / "reports/mme_r70_256_128_runs.json"
DEPENDENCY = BASE / "reports/k_commit_pilot/index.json"


def save(state):
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    atomic_json(STATE, state)
    atomic_json(ROOT / "reports/index.json", state)


def finished(run):
    if not run or not (Path(run) / "run_manifest.json").exists():
        return False
    manifest = read_json(Path(run) / "run_manifest.json")
    if manifest["status"] == "complete":
        return True
    counts = manifest.get("sample_status_counts", {})
    return (manifest["status"] == "incomplete"
            and counts.get("budget_exhausted", 0) > 0
            and set(counts) <= {"success", "budget_exhausted"}
            and sum(counts.values()) == manifest["sample_count"])


def find_run(name, fingerprint):
    matches = []
    for path in (ROOT / "runs").glob("*/resolved_config.json"):
        manifest = path.parent / "run_manifest.json"
        if (read_json(path)["run"]["name"] == name and manifest.exists()
                and read_json(manifest)["generation_fingerprint"] == fingerprint):
            matches.append(str(path.parent))
    return max(matches) if matches else None


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[0], fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None, None


class ExistingWorker:
    def __init__(self, pid):
        self.pid = pid
        self.started = process_identity(pid)[1]

    def poll(self):
        state, started = process_identity(self.pid)
        return None if started == self.started and state not in (None, "Z", "X") else 0


def existing_worker(config):
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / "cmdline").read_bytes().decode(errors="replace").split("\0")
            if "cdapc_lab" not in args or "infer" not in args or "--config" not in args:
                continue
            if (entry / "cwd").resolve() != ROOT:
                continue
            if (ROOT / args[args.index("--config") + 1]).resolve() == config.resolve():
                worker = ExistingWorker(int(entry.name))
                if worker.poll() is None:
                    matches.append(worker)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    if len(matches) > 1:
        raise RuntimeError(f"Multiple workers for {config}")
    return matches[0] if matches else None


def k_generation_finished(state):
    return all(state.get("conditions", {}).get(k, {}).get("status") in ("generated", "complete")
               for k in ("4", "2"))


def main():
    with (ROOT / "logs/controller.lock").open("a") as own_lock:
        fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read_json(STATE) if STATE.exists() else {"status": "prepared", "shards": {}}
        if state["status"] == "complete":
            return
        state.update(controller_pid=os.getpid(), workspace=str(ROOT),
                     dependency=str(DEPENDENCY), expected_inputs=2374, expected_image_pairs=1187,
                     canvas_length=256, iteration_limit=128, r_gate=.7, max_commit=16,
                     prompt_id="mme_yesno_v1", evaluation="official Acc + Acc+; no API judge")
        plans = []
        for i in range(3):
            path = ROOT / f"configs/experiments/mmada_mme_cdapc_r70_L256_I128_shard{i}.yaml"
            config = resolve_config(path)
            plans.append((path, config))
            state["shards"].setdefault(str(i), {}).update(
                config=str(path), name=config["run"]["name"],
                expected=len(read_json(ROOT / config["run"]["sample_ids_file"])))
        save(state)
        try:
            while not k_generation_finished(read_json(DEPENDENCY)):
                state["status"] = "waiting_for_k_pilot_generation"
                save(state)
                time.sleep(30)
            with (BASE / "logs/benchmark_extensions/queue.lock").open("a") as gpu_lock:
                state["status"] = "waiting_for_gpu_lock"
                save(state)
                fcntl.flock(gpu_lock, fcntl.LOCK_EX)
                state["status"] = "generating"
                for i, (_, config) in enumerate(plans):
                    item = state["shards"][str(i)]
                    _, _, provenance = preflight(config)
                    fingerprint = provenance["generation_fingerprint"]
                    previous = item.get("run")
                    match = find_run(item["name"], fingerprint)
                    if previous and previous != match:
                        item.setdefault("previous_runs", []).append(previous)
                    item.update(run=match, generation_fingerprint=fingerprint)
                save(state)
                for attempt in range(2):
                    workers = {}
                    for i, (path, _) in enumerate(plans):
                        item = state["shards"][str(i)]
                        if finished(item["run"]):
                            continue
                        command = [sys.executable, "-m", "cdapc_lab", "infer", "--config", str(path)]
                        if item["run"]:
                            command += ["--resume-run", item["run"]]
                        worker = existing_worker(path)
                        if worker is None:
                            with (ROOT / f"logs/shard{i}.log").open("a") as log:
                                worker = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, "PYTHONPATH": str(ROOT / "src"),
                                         "CUDA_VISIBLE_DEVICES": str(i), "OMP_NUM_THREADS": "8",
                                         "MKL_NUM_THREADS": "8"})
                        else:
                            item["adopted"] = True
                        workers[i] = worker
                        item["worker_pid"] = worker.pid
                    while workers:
                        for i in list(workers):
                            item = state["shards"][str(i)]
                            item["run"] = item.get("run") or find_run(item["name"], item["generation_fingerprint"])
                            item["generated"] = (len(list((Path(item["run"]) / "predictions").glob("*.json")))
                                                 if item["run"] else 0)
                            code = workers[i].poll()
                            if code is not None:
                                item["worker_returncode"] = code
                                del workers[i]
                        save(state)
                        if workers:
                            time.sleep(30)
                    if all(finished(x.get("run")) for x in state["shards"].values()):
                        break
                else:
                    raise RuntimeError("MME generation incomplete after retry; inspect shard logs")
            state["status"] = "scoring"
            save(state)
            from score_mme import score
            score([state["shards"][str(i)]["run"] for i in range(3)])
            state.update(status="complete", report="reports/mme_cdapc_r70_L256_I128.json")
            save(state)
            subprocess.run([sys.executable, "scripts/summarize_experiments.py"], cwd=BASE, check=True)
            print("MME COMPLETE", flush=True)
        except Exception:
            state.update(status="failed", error=traceback.format_exc())
            save(state)
            raise


if __name__ == "__main__":
    main()
