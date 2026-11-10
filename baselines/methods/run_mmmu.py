"""Two fixed 450-row MMMU shards, VCD then VIGOR, with automatic scoring."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import signal
import subprocess
import sys
import time
import uuid

from common.mmmu_protocol import CHECKPOINT, DATA, jsonl_bytes, write_locked
from common.mmmu_score import SCORER_SHA, score_runs
from common.run import input_key
from common.runtime import ROOT, save_json, sha256, source_hashes

RUN = ROOT / "results/mmmu_wino_v1"
CONFIG = ROOT / "configs/mmmu_wino_v1.yaml"
METHODS = ("vcd", "vigor")


def command(method, samples, output):
    return [sys.executable, "-u", "-m", "common.run", "--config", str(CONFIG),
            "--samples", str(samples), "--output", str(output), "--methods", method,
            "--benchmark", "custom", "--warmups", "1", "--early-eos"]


def worker(shard):
    for method in METHODS:
        output = ROOT / method / "results/mmmu_wino_v1" / f"shard{shard}"
        subprocess.run(command(method, DATA / f"shard_{shard}.jsonl", output),
                       check=True, cwd=ROOT)


def available_gpus(gpus):
    rows = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,memory.free", "--format=csv,noheader,nounits"
    ], text=True).splitlines()
    inventory = {r.split(",")[0].strip(): r.split(",") for r in rows}
    if len(gpus) != 2 or len(set(gpus)) != 2 or not set(gpus) <= set(inventory):
        raise ValueError("two distinct available GPUs are required")
    for gpu in gpus:
        if "5090" not in inventory[gpu][1] or int(inventory[gpu][2]) < 22 * 1024:
            raise ValueError(f"GPU {gpu} is not an available 5090 with 22 GiB free")
    return rows


def parallel_jobs(jobs):
    """Wait for both owned jobs; never terminate unrelated GPU processes."""
    with ExitStack() as stack:
        children = []
        try:
            for name, argv, gpu, log_path in jobs:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log = stack.enter_context(log_path.open("a"))
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "OMP_NUM_THREADS": "4",
                       "TOKENIZERS_PARALLELISM": "false", "PYTHONHASHSEED": "20261105"}
                child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                children.append((name, child))
            statuses = {name: child.wait() for name, child in children}
            if any(statuses.values()):
                raise RuntimeError(f"worker failure; committed rows remain resumable: {statuses}")
        finally:
            for _, child in children:
                if child.poll() is None:
                    # The process group contains this launcher's worker and its model child.
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    child.wait()


def validate_preflight(samples, session, protocol):
    from transformers import AutoTokenizer
    from training.prompting_utils import UniversalPrompting
    from verify_eos import audit_record

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True)
    prompting = UniversalPrompting(tokenizer, use_reserved_token=True)
    mask = tokenizer.convert_tokens_to_ids("<|mdm_mask|>")
    eos = tokenizer.eos_token_id
    records = {}
    checks = []
    for method in METHODS:
        path = ROOT / method / "results/mmmu_wino_v1/preflight" / session / "predictions.sqlite"
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
            records[method] = {key: json.loads(payload) for key, payload in db.execute(
                "SELECT input_key,payload FROM predictions WHERE method=?", (method,))}
        if set(records[method]) != {input_key(s) for s in samples}:
            raise ValueError("preflight coverage differs")
        for sample in samples:
            record = records[method][input_key(sample)]
            ids = record["input"]["prompt_ids"]
            text = tokenizer.apply_chat_template([{"role": "user", "content": sample["question"]}],
                                                 tokenize=True, add_generation_prompt=True)
            if ids[:2] != [int(prompting.sptids_dict[t]) for t in ("<|mmu|>", "<|soi|>")]:
                raise ValueError("MMMU vision prefix mismatch")
            if (ids[1026] != int(prompting.sptids_dict["<|eoi|>"]) or ids[1027:] != text or
                    record["input"]["visual_positions"] != list(range(2, 1026)) or
                    len(ids) != sample["expected_prompt_tokens"]):
                raise ValueError("MMMU prompt assembly differs from reference")
            first_closed = audit_record(record, mask, eos)
            if record["remaining_masks"] or record["image_token_count"]:
                raise ValueError("preflight returned incomplete or visual-code output")
            if record["executed_steps"] > 128:
                raise ValueError("generation exceeded its schedule")
            if first_closed is not None and first_closed != record["executed_steps"]:
                raise ValueError("EOS stopping was not at the first committed closed prefix")
            checks.append({
                "method": method, "id": sample["id"], "steps": record["executed_steps"],
                "answer_token_length": record["answer_token_length"],
                "seconds": record["end_to_end_seconds"],
                "peak_allocated_bytes": record["peak_allocated_bytes"],
                "prompt_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                "trace_reconstruction": True,
            })
    for sample in samples:
        key = input_key(sample)
        if records["vcd"][key]["input"]["prompt_ids"] != records["vigor"][key]["input"]["prompt_ids"]:
            raise ValueError("normal model inputs differ across methods/GPUs")
    report = {"passed": True, "session": session, "protocol": protocol, "checks": checks}
    save_json(RUN / "preflight_validation.json", report)
    return report


def complete_generation():
    for method in METHODS:
        for shard in range(2):
            path = ROOT / method / "results/mmmu_wino_v1" / f"shard{shard}" / "predictions.sqlite"
            if not path.exists():
                return False
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
                if db.execute("SELECT COUNT(*) FROM predictions WHERE method=?", (method,)).fetchone()[0] != 450:
                    return False
    return True


def main(args):
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = {
            "sources": source_hashes(), "config_sha256": sha256(CONFIG),
            "data_manifest_sha256": sha256(DATA / "manifest.json"),
            "official_scorer_sha256": SCORER_SHA,
            "methods": list(METHODS), "rows_per_method": 900, "shards": [450, 450],
            "gpu_assignment": args.gpus, "early_eos": "safe_eos_v1",
        }
        write_locked(RUN / "manifest.json", (json.dumps(protocol, indent=2) + "\n").encode())
        status = {"started": time.time(), "pid": os.getpid(), "state": "preflight", "complete": False}
        save_json(RUN / "queue_status.json", status)
        try:
            validation = RUN / "preflight_validation.json"
            if validation.exists():
                report = json.loads(validation.read_text())
                if not report["passed"] or report["protocol"] != protocol:
                    raise ValueError("preflight no longer matches current sources/protocol")
            else:
                status["gpu_inventory"] = available_gpus(args.gpus)
                samples = list(map(json.loads, (DATA / "samples.jsonl").read_text().splitlines()))
                selected = [max(samples, key=lambda s: s["expected_prompt_tokens"]), samples[0],
                            next(s for s in samples if s["question_type"] == "open"),
                            next(s for s in samples if s["source_image_count"] > 1)]
                write_locked(DATA / "preflight.jsonl", jsonl_bytes(selected))
                session = uuid.uuid4().hex
                status["preflight_session"] = session
                save_json(RUN / "queue_status.json", status)
                jobs = []
                for method, gpu in zip(METHODS, args.gpus):
                    directory = ROOT / method / "results/mmmu_wino_v1/preflight" / session
                    jobs.append((method, command(method, DATA / "preflight.jsonl", directory),
                                 gpu, RUN / f"preflight_{method}_{session}.log"))
                parallel_jobs(jobs)
                validate_preflight(selected, session, protocol)
            if args.preflight_only:
                status["state"] = "preflight_passed"
                return
            if not complete_generation():
                status["gpu_inventory"] = available_gpus(args.gpus)
                if source_hashes() != protocol["sources"]:
                    raise ValueError("sources changed during preflight")
                status["state"] = "generating"
                save_json(RUN / "queue_status.json", status)
                jobs = [(f"shard{shard}", [sys.executable, "-u", str(Path(__file__).resolve()),
                                         "--worker-shard", str(shard)],
                         gpu, RUN / f"shard{shard}.log") for shard, gpu in enumerate(args.gpus)]
                parallel_jobs(jobs)
            status["state"] = "scoring"
            save_json(RUN / "queue_status.json", status)
            report = score_runs(RUN)
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
    parser.add_argument("--worker-shard", type=int, choices=(0, 1))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.worker_shard is not None:
        worker(args.worker_shard)
    else:
        def terminate(signum, frame):
            raise KeyboardInterrupt(f"received signal {signum}")
        signal.signal(signal.SIGTERM, terminate)
        main(args)
