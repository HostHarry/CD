"""Resume two HallusionBench shards and score Native, VCD and VIGOR automatically."""
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
import signal
import subprocess
import sys
import time

import pandas as pd

from common.hallusion import (
    CONFIG, DATA, METHODS, REVISION, RUN, SCORER_HASHES, audit_predictions,
    load_predictions, official_functions, read_rows, score_runs, write_locked,
)
from common.runtime import ROOT, save_json, sha256, source_hashes


def command(samples, output):
    return [sys.executable, "-u", "-m", "common.run", "--config", str(CONFIG),
            "--samples", str(samples), "--output", str(output), "--methods", *METHODS,
            "--benchmark", "custom", "--warmups", "3", "--early-eos"]


def available_gpus(gpus):
    rows = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,memory.free", "--format=csv,noheader,nounits"
    ], text=True).splitlines()
    inventory = {r.split(",")[0].strip(): r.split(",") for r in rows}
    if len(set(gpus)) != 2 or not set(gpus) <= set(inventory):
        raise ValueError("two distinct available GPUs are required")
    for gpu in gpus:
        if int(inventory[gpu][2]) < 22 * 1024:
            raise ValueError(f"GPU {gpu} has less than 22 GiB free")
    return rows


def parallel_jobs(jobs, lock_fd):
    with ExitStack() as stack:
        children = []
        try:
            for argv, gpu, log_path in jobs:
                log = stack.enter_context(log_path.open("a"))
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "OMP_NUM_THREADS": "4",
                       "TOKENIZERS_PARALLELISM": "false", "PYTHONHASHSEED": "42"}
                child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True,
                                         pass_fds=(lock_fd,))
                children.append(child)
            while any(child.poll() is None for child in children):
                if any(child.poll() not in (None, 0) for child in children):
                    raise RuntimeError("HallusionBench worker failed; see shard logs")
                time.sleep(2)
            if any(child.returncode for child in children):
                raise RuntimeError("HallusionBench worker failed; committed rows remain resumable")
        finally:
            for child in children:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait()


def main(args):
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        official_functions()
        protocol = {
            "sources": source_hashes(), "config_sha256": sha256(CONFIG),
            "data_manifest_sha256": sha256(DATA / "manifest.json"),
            "scorer_revision": REVISION, "scorer_sha256": SCORER_HASHES,
            "pandas_version": pd.__version__, "methods": list(METHODS),
            "rows_per_method": 1129, "shards": [565, 564],
            "gpus": args.gpus, "early_eos": "safe_eos_v1",
        }
        write_locked(RUN / "manifest.json", (json.dumps(protocol, indent=2) + "\n").encode())
        status = {"started": time.time(), "pid": os.getpid(), "state": "preflight", "complete": False}
        save_json(RUN / "queue_status.json", status)
        try:
            if not (RUN / "preflight_validation.json").exists():
                status["gpu_inventory"] = available_gpus(args.gpus)
                jobs = [(command(DATA / "preflight.jsonl", RUN / f"preflight{shard}"),
                         gpu, RUN / f"preflight{shard}.log") for shard, gpu in enumerate(args.gpus)]
                parallel_jobs(jobs, lock.fileno())
                samples = read_rows(DATA / "preflight.jsonl")
                records = []
                for shard in range(2):
                    record = load_predictions(RUN / f"preflight{shard}", samples, protocol["sources"])
                    audit_predictions(samples, record)
                    records.append(record)
                for key in records[0]:
                    for field in ("input", "generated_ids", "text", "trace"):
                        if records[0][key][field] != records[1][key][field]:
                            raise ValueError(f"preflight differs across GPUs: {key}, {field}")
                checks = [{"method": method, "input_key": key,
                           "executed_steps": r["executed_steps"], "text": r["text"],
                           "end_to_end_seconds": r["end_to_end_seconds"]}
                          for (method, key), r in records[0].items()]
                save_json(RUN / "preflight_validation.json",
                          {"passed": True, "protocol": protocol, "checks": checks,
                           "cross_gpu_tokens_inputs_traces_equal": True})
            validation = json.loads((RUN / "preflight_validation.json").read_text())
            if not validation["passed"] or validation["protocol"] != protocol:
                raise ValueError("preflight differs from the current protocol")
            if args.preflight_only:
                status["state"] = "preflight_passed"
                return
            if source_hashes() != protocol["sources"]:
                raise ValueError("source changed during preflight")
            status.update(state="generating", gpu_inventory=available_gpus(args.gpus))
            save_json(RUN / "queue_status.json", status)
            jobs = [(command(DATA / f"shard_{shard}.jsonl", RUN / f"shard{shard}"),
                     gpu, RUN / f"shard{shard}.log") for shard, gpu in enumerate(args.gpus)]
            parallel_jobs(jobs, lock.fileno())
            status["state"] = "scoring"
            save_json(RUN / "queue_status.json", status)
            report = score_runs()
            status["complete"] = all(m["complete"] for m in report["methods"].values())
            status["state"] = "complete" if status["complete"] else "incomplete"
        except BaseException as exc:
            status.update(state="failed", error=repr(exc))
            raise
        finally:
            status["updated"] = time.time()
            status["elapsed_seconds"] = status["updated"] - status["started"]
            save_json(RUN / "queue_status.json", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs=2, default=["0", "1"])
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    def terminate(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    main(args)
