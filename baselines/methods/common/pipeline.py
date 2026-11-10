"""Each baseline owns its resumable POPE/CHAIR queue and result directory."""
import argparse
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

from .runtime import ROOT, save_json, sha256, source_hashes


def main(method):
    parser = argparse.ArgumentParser(description=f"Run {method}: POPE, CHAIR, then CHAIR scoring")
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--early-eos", action="store_true")
    args = parser.parse_args()
    output = ROOT / method / "results"
    if args.early_eos:
        output /= "early_eos"
    output.mkdir(parents=True, exist_ok=True)
    child = None

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    with (output / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = {"method": method, "sources": source_hashes(),
                    "early_eos": args.early_eos,
                    "config_sha256": sha256(ROOT / method / "config.yaml"),
                    "data": {name: sha256(ROOT / "data" / name)
                             for name in ("pope_coco_yesno.jsonl", "chair500.jsonl", "calibration.jsonl")}}
        protocol_path = output / "protocol.json"
        if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("pipeline source/config/data changed")
        save_json(protocol_path, protocol)
        status_path = output / "queue_status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {"stages": {}}
        status.update(state="running", gpus=args.gpus)
        stages = []
        for benchmark, samples, directory in (
            ("POPE", "pope_coco_yesno.jsonl", "pope_full"),
            ("CHAIR", "chair500.jsonl", "chair_full"),
        ):
            stages.append((directory, [
                sys.executable, "-u", str(ROOT / "multigpu.py"), "--method", method,
                "--config", str(ROOT / method / "config.yaml"),
                "--run-dir", str(output / directory), "--samples", str(ROOT / "data" / samples),
                "--benchmark", benchmark, "--gpus", *args.gpus,
                "--exclude-samples", str(ROOT / "data/calibration.jsonl"),
            ] + (["--early-eos"] if args.early_eos else [])))
        stages.append(("chair_score", [
            sys.executable, "-u", "-m", "common.score_chair", "--run-dir", str(output / "chair_full"),
            "--annotations", str(ROOT / "data/annotations"), "--evaluator", str(ROOT / "data/evaluators/chair.py"),
            "--nltk-data", str(ROOT / "data/nltk_data"),
        ]))
        try:
            for name, command in stages:
                if status["stages"].get(name, {}).get("state") == "completed":
                    continue
                status["stages"][name] = {"state": "running", "started": time.time(), "command": command}
                save_json(status_path, status)
                with (output / f"{name}.log").open("a") as log:
                    child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                             pass_fds=(lock.fileno(),))
                    code = child.wait()
                status["stages"][name].update(state="completed" if code == 0 else "blocked",
                                             exit_code=code, finished=time.time())
                save_json(status_path, status)
                if code:
                    raise RuntimeError(f"{name} failed: inspect {output / (name + '.log')}")
            status["state"] = "generation_and_scoring_complete"
            save_json(status_path, status)
        except BaseException as exc:
            status.update(state="blocked", error=repr(exc))
            save_json(status_path, status)
            raise
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                child.wait(timeout=60)
