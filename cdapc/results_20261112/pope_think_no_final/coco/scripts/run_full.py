"""Complete missing full-COCO inputs after the current paired pilot finishes."""
from datetime import datetime
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cdapc_lab.config import resolve_config
from cdapc_lab.experiments.infer import preflight
from cdapc_lab.io.artifacts import atomic_json, digest, file_hash, generation_projection, read_json


def save(state):
    state["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    atomic_json(ROOT / "reports/status.json", state)


def collect(plan):
    records = dict(plan["preserved_predictions"])
    for item in records.values():
        assert file_hash(item["path"]) == item["sha256"]
    for path in sorted((ROOT / "runs").glob("*/predictions/*.json")):
        raw = read_json(path)
        sid = raw["sample_id"]
        assert sid in plan["control"] and sid not in records
        assert raw["status"] in ("success", "budget_exhausted"), str(path)
        records[sid] = {"path": str(path), "sha256": file_hash(path), "status": raw["status"]}
    return records


def prepare():
    os.chdir(ROOT)
    plan = read_json(ROOT / "reports/plan.json")
    pilot = Path(plan["pilot_workspace"])
    assert file_hash(pilot / "reports/plan.json") == plan["pilot_plan_sha256"]
    assert file_hash(plan["control_report"]) == plan["control_report_sha256"]
    for path in (ROOT / "src").rglob("*.py"):
        assert file_hash(path) == file_hash(pilot / path.relative_to(ROOT)), str(path)
    existing = collect(plan)
    missing = set(plan["control"]) - set(existing)
    wave = digest(sorted(missing))[:12]
    old = read_json(pilot / "reports/preflight.json")["shards"]
    result = {"present": len(existing), "missing": len(missing), "shards": {},
              "plan_sha256": file_hash(ROOT / "reports/plan.json")}
    for key, reference in old.items():
        ids = [sid for sid in read_json(ROOT / f"manifests/all_shard{key}.json") if sid in missing]
        if not ids:
            continue
        ids_path = ROOT / f"manifests/{wave}_missing_shard{key}.json"
        atomic_json(ids_path, ids)
        config = yaml.safe_load((pilot / reference["config"]).read_text())
        config["run"].update(name=f"full_coco_prefix_no_final_{wave}_shard{key}",
                             sample_ids_file=str(ids_path), output_root=str(ROOT / "runs"))
        path = ROOT / f"configs/{wave}_shard{key}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        resolved = resolve_config(path)
        prior = resolve_config(pilot / reference["config"])
        assert generation_projection(resolved) == generation_projection(prior)
        assert resolved["recording"] == prior["recording"]
        samples, _, provenance = preflight(resolved)
        for field in ("generation_source_hash", "environment", "assets"):
            assert provenance[field] == reference[field], field
        result["shards"][key] = {
            "config": str(path), "config_sha256": file_hash(path),
            "ids": str(ids_path), "ids_sha256": file_hash(ids_path),
            "expected": len(samples), **provenance}
    assert sum(v["expected"] for v in result["shards"].values()) == len(missing)
    atomic_json(ROOT / "reports/preflight.json", result)
    print("PREPARED", result["present"], "preserved;", result["missing"], "missing", flush=True)
    return result


def verify_snapshot():
    frozen = read_json(ROOT / "reports/source_snapshot.json")
    assert all(file_hash(ROOT / name) == sha for name, sha in frozen.items()), "Frozen full-COCO files changed"


def main():
    os.chdir(ROOT)
    if "--prepare" in sys.argv:
        prepare()
        return
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    plan = read_json(ROOT / "reports/plan.json")
    verify_snapshot()
    wait = Path(plan["wait_workspace"])
    workers = {}
    state = {"status": "waiting_for_current_pilot", "controller_pid": os.getpid(),
             "expected_inputs": plan["inputs"], "expected_rows": plan["rows"], "preserved": 661}
    with (ROOT / "logs/controller.lock").open("a") as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while True:
                dependency = read_json(wait / "reports/status.json")
                state["dependency_status"] = dependency["status"]
                save(state)
                if dependency["status"] == "complete":
                    break
                if dependency["status"] in ("interrupted_or_failed", "paused"):
                    raise RuntimeError("Current paired pilot is blocked; full COCO not started")
                pid = Path(f"/proc/{dependency['controller_pid']}/cmdline")
                if not pid.exists() or not pid.read_bytes():
                    if read_json(wait / "reports/status.json")["status"] != "complete":
                        raise RuntimeError("Paired pilot controller disappeared")
                time.sleep(30)
            lock = ROOT.parent / "cdapc-experiments/logs/benchmark_extensions/queue.lock"
            with lock.open("a") as gpu_lock:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX)
                verify_snapshot()
                prepared = prepare()
                assert file_hash(ROOT / "reports/plan.json") == prepared["plan_sha256"]
                previous_runs = set((ROOT / "runs").iterdir())
                state.update(status="generating", preserved=prepared["present"], shards={})
                for key, item in prepared["shards"].items():
                    assert file_hash(item["config"]) == item["config_sha256"]
                    assert file_hash(item["ids"]) == item["ids_sha256"]
                    with (ROOT / f"logs/{Path(item['config']).stem}.log").open("a") as log:
                        process = subprocess.Popen(
                            [sys.executable, "-m", "cdapc_lab", "infer", "--config", item["config"]],
                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": key,
                                 "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"})
                    workers[key] = process
                    state["shards"][key] = {"worker_pid": process.pid, "expected": item["expected"]}
                while True:
                    for key, process in workers.items():
                        item = state["shards"][key]
                        matches = [p.parent for p in (ROOT / "runs").glob("*/run_manifest.json")
                                   if p.parent not in previous_runs and read_json(p)["generation_fingerprint"]
                                   == prepared["shards"][key]["generation_fingerprint"]]
                        if matches:
                            assert len(matches) == 1
                            item.update(run=str(matches[0]),
                                        generated=len(list((matches[0] / "predictions").glob("*.json"))))
                        item["returncode"] = process.poll()
                        if item["returncode"] not in (None, 0):
                            raise RuntimeError(f"Generation shard {key} failed")
                    state["generated_total"] = prepared["present"] + sum(
                        v.get("generated", 0) for v in state["shards"].values())
                    save(state)
                    if all(p.poll() is not None for p in workers.values()):
                        break
                    time.sleep(30)
                records = collect(plan)
                assert set(records) == set(plan["control"]) and len(records) == 5127
                atomic_json(ROOT / "reports/prediction_inventory.json", records)
                runs = sorted({str(Path(v["path"]).parents[1]) for v in records.values()})
                combined = {"shards": {str(i): {"run": run} for i, run in enumerate(runs)}}
                state["status"] = "scoring_official"
                save(state)
                from frozen_pilot_tools import score
                score(combined)
                from report_scopes import write_official_scopes
                write_official_scopes()
                state.update(status="complete", report="reports/comparison.md",
                             additional_images_report="reports/official_scopes.md",
                             deepseek_status="reports/judge/status.json")
                save(state)
                print("FULL_COCO_OFFICIAL_COMPLETE", flush=True)
        except BaseException as error:
            state.update(status="interrupted_or_failed", error=repr(error))
            save(state)
            raise
        finally:
            for process in workers.values():
                if process.poll() is None:
                    process.terminate()
            for process in workers.values():
                if process.poll() is None:
                    process.wait(timeout=60)


if __name__ == "__main__":
    main()
