import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time
import uuid

from common.run import input_key
from common.runtime import Runtime, environment, save_json, sha256, source_hashes
from common.eos import answer_ids


def connect(path):
    connection = sqlite3.connect(path, timeout=60)
    connection.execute("PRAGMA busy_timeout=60000")
    return connection


def prepare_tasks(connection, tasks):
    if len(set(tasks)) != len(tasks):
        raise ValueError("duplicate task keys")
    expected = set(tasks)
    completed = set(connection.execute("SELECT method,input_key FROM predictions"))
    if not completed <= expected:
        raise ValueError("predictions contain tasks outside the manifest")
    with connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS gpu_tasks ("
            "method TEXT, input_key TEXT, ordinal INTEGER, state TEXT, worker TEXT,"
            "attempts INTEGER DEFAULT 0, error TEXT, updated REAL,"
            "PRIMARY KEY(method,input_key))")
        existing = set(connection.execute("SELECT method,input_key FROM gpu_tasks"))
        if not existing <= expected:
            raise ValueError("queued tasks differ from the manifest")
        for ordinal, (method, key) in enumerate(tasks):
            connection.execute(
                "INSERT OR IGNORE INTO gpu_tasks(method,input_key,ordinal,state) VALUES (?,?,?,'pending')",
                (method, key, ordinal))
        connection.execute(
            "UPDATE gpu_tasks SET state=CASE WHEN EXISTS "
            "(SELECT 1 FROM predictions p WHERE p.method=gpu_tasks.method AND p.input_key=gpu_tasks.input_key) "
            "THEN 'done' ELSE 'pending' END, worker=NULL, error=NULL")
    return len(completed)


def claim_task(connection, worker):
    connection.execute("BEGIN IMMEDIATE")
    try:
        row = connection.execute(
            "SELECT method,input_key FROM gpu_tasks WHERE state='pending' ORDER BY ordinal LIMIT 1").fetchone()
        if row is not None:
            connection.execute(
                "UPDATE gpu_tasks SET state='running',worker=?,attempts=attempts+1,updated=? "
                "WHERE method=? AND input_key=?", (worker, time.time(), *row))
        connection.commit()
        return row
    except BaseException:
        connection.rollback()
        raise


def finish_task(connection, worker, task, result):
    payload = json.dumps(result, ensure_ascii=False, allow_nan=False)
    connection.execute("BEGIN IMMEDIATE")
    try:
        owner = connection.execute(
            "SELECT state,worker FROM gpu_tasks WHERE method=? AND input_key=?", task).fetchone()
        if owner != ("running", worker):
            raise ValueError("task is not owned by this worker")
        connection.execute("INSERT INTO predictions VALUES (?,?,?)", (*task, payload))
        connection.execute(
            "UPDATE gpu_tasks SET state='done',updated=? WHERE method=? AND input_key=?", (time.time(), *task))
        connection.commit()
    except BaseException:
        connection.rollback()
        raise


def token_fingerprint(result, early_eos=False):
    value = [result["input"]["prompt_ids"], answer_ids(result) if early_eos else result["generated_ids"]]
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def worker(args):
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    if manifest["sources"] != source_hashes() or manifest["environment"] != environment():
        raise ValueError("worker source/environment differs from the locked manifest")
    if sha256(args.samples) != manifest["samples_sha256"]:
        raise ValueError("worker samples differ from the manifest")
    samples = [json.loads(line) for line in args.samples.read_text().splitlines() if line.strip()]
    if manifest["limit"]:
        samples = samples[:manifest["limit"]]
    unique = {input_key(sample): sample for sample in samples}
    config = manifest["config"]
    runtime = Runtime(config["checkpoint"], config["vq_checkpoint"], "cuda",
                      config["resolution"], config["resize"], early_eos=bool(manifest.get("early_eos")))
    first = next(iter(unique.values()))
    fingerprints = {}
    for method, options in manifest["methods"].items():
        for _ in range(max(1, manifest["warmups_per_method_per_session"])):
            result = runtime.generate(first, options, manifest["generation"], config["seed"])
        fingerprints[method] = token_fingerprint(result, bool(manifest.get("early_eos")))
    control = args.run_dir / "multi_gpu" / args.session
    save_json(control / f"gpu{args.worker}.ready.json", {
        "pid": os.getpid(), "physical_gpu": args.worker,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "fingerprints": fingerprints, "environment": environment(),
    })
    started = time.monotonic()
    while not (control / "release.json").exists():
        if os.getppid() != args.parent_pid or time.monotonic() - started > 600:
            raise RuntimeError("coordinator unavailable during startup")
        time.sleep(1)
    with connect(args.run_dir / "predictions.sqlite") as connection:
        while os.getppid() == args.parent_pid:
            task = claim_task(connection, args.worker)
            if task is None:
                break
            method, key = task
            save_json(control / f"gpu{args.worker}.progress.json", {
                "state": "running", "method": method, "input_key": key, "started": time.time()})
            try:
                result = runtime.generate(
                    unique[key], manifest["methods"][method], manifest["generation"], config["seed"])
                finish_task(connection, args.worker, task, result)
            except BaseException as exc:
                with connection:
                    connection.execute(
                        "UPDATE gpu_tasks SET state='failed',error=?,updated=? WHERE method=? AND input_key=?",
                        (repr(exc), time.time(), *task))
                raise
            print(f"GPU {args.worker} {method} {unique[key]['id']} committed", flush=True)
        save_json(control / f"gpu{args.worker}.progress.json", {"state": "finished", "updated": time.time()})


def validate_fingerprints(ready, references):
    values = [value["fingerprints"] for value in ready]
    if not values or any(value != values[0] for value in values[1:]):
        raise ValueError("token/prompt outputs differ across GPUs during startup")
    for method, expected in references.items():
        if values[0][method] != expected:
            raise ValueError(f"{method} startup differs from a preserved prediction")


def score_command(args):
    return [
        sys.executable, "-u", "-m", "common.run", "--config", str(args.config),
        "--samples", str(args.samples), "--output", str(args.run_dir),
        "--benchmark", args.benchmark, "--methods", "native", args.method,
        "--exclude-samples", str(args.exclude_samples), "--score-only",
    ] + (["--early-eos"] if args.early_eos else [])


def coordinator(args):
    if len(set(args.gpus)) != len(args.gpus) or not args.gpus:
        raise ValueError("GPU indices must be nonempty and unique")
    devices = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader",
    ], text=True).splitlines()
    available = {line.split(",")[0].strip() for line in devices}
    if not set(args.gpus) <= available:
        raise ValueError("requested GPU is unavailable")
    validation_env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpus[0]}
    subprocess.run(score_command(args), check=True, env=validation_env)
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    approved = ["native", args.method]
    if list(manifest["methods"]) != approved:
        raise ValueError("manifest methods differ from the selected comparison")
    samples = [json.loads(line) for line in args.samples.read_text().splitlines() if line.strip()]
    if manifest["limit"]:
        samples = samples[:manifest["limit"]]
    unique = {input_key(sample): sample for sample in samples}
    tasks = [(method, key) for key in unique for method in manifest["methods"]]
    with (args.run_dir / "run.lock").open("a") as lock, ExitStack() as stack:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        session = uuid.uuid4().hex
        control = args.run_dir / "multi_gpu" / session
        control.mkdir(parents=True)
        connection = stack.enter_context(connect(args.run_dir / "predictions.sqlite"))
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("prediction database failed integrity check")
        backup = args.run_dir / "multi_gpu/predictions.before_multigpu.sqlite"
        if not backup.exists():
            with connect(backup) as destination:
                connection.backup(destination)
        completed_before = prepare_tasks(connection, tasks)
        references = {}
        first_key = next(iter(unique))
        for method in manifest["methods"]:
            row = connection.execute(
                "SELECT payload FROM predictions WHERE method=? AND input_key=?", (method, first_key)).fetchone()
            if row:
                references[method] = token_fingerprint(json.loads(row[0]), bool(manifest.get("early_eos")))
        execution = {
            "session": session, "started": time.time(), "gpus": args.gpus,
            "gpu_inventory": devices, "pid": os.getpid(),
            "manifest_sha256": sha256(args.run_dir / "manifest.json"),
            "launcher_sha256": sha256(__file__), "preserved_predictions": completed_before,
            "target_predictions": len(tasks), "methods": list(manifest["methods"]),
            "state": "starting", "timing_scope": "mixed preserved single-GPU and new parallel samples",
            "early_eos": manifest.get("early_eos"),
            "fingerprint_scope": "prompt_and_answer_through_eos" if manifest.get("early_eos") else "prompt_and_full_tokens",
        }
        save_json(control / "execution.json", execution)
        save_json(args.run_dir / "multi_gpu/current.json", {"session": session, "gpus": args.gpus})
        children = []
        try:
            if completed_before < len(tasks):
                for gpu in args.gpus:
                    log = stack.enter_context((control / f"gpu{gpu}.log").open("a"))
                    command = [
                        sys.executable, "-u", str(Path(__file__).resolve()), "--worker", gpu,
                        "--run-dir", str(args.run_dir.resolve()), "--samples", str(args.samples.resolve()),
                        "--session", session, "--parent-pid", str(os.getpid()),
                        "--method", args.method,
                    ]
                    children.append(subprocess.Popen(
                        command, stdout=log, stderr=subprocess.STDOUT,
                        env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu},
                        pass_fds=(lock.fileno(),)))
                startup = time.monotonic()
                released = False
                while any(child.poll() is None for child in children):
                    if any(child.poll() not in (None, 0) for child in children):
                        raise RuntimeError(f"GPU worker failed; inspect {control}")
                    if not released:
                        paths = [control / f"gpu{gpu}.ready.json" for gpu in args.gpus]
                        if all(path.exists() for path in paths):
                            ready = [json.loads(path.read_text()) for path in paths]
                            validate_fingerprints(ready, references)
                            save_json(control / "release.json", {
                                "cross_gpu_outputs_equal": True, "preserved_reference_methods": list(references)})
                            execution["state"] = "running"
                            save_json(control / "execution.json", execution)
                            released = True
                        elif time.monotonic() - startup > 600:
                            raise RuntimeError("GPU startup timed out")
                    counts = dict(connection.execute("SELECT method,count(*) FROM predictions GROUP BY method"))
                    save_json(args.run_dir / "progress.json", {
                        "completed_unique_predictions": sum(counts.values()),
                        "target_unique_predictions": len(tasks), "completed_by_method": counts,
                        "gpus": args.gpus, "session": session,
                        "worker_state": "running" if released else "warming_up", "updated": time.time(),
                    })
                    time.sleep(2)
                if any(child.returncode != 0 for child in children):
                    raise RuntimeError("GPU worker failed")
            actual = set(connection.execute("SELECT method,input_key FROM predictions"))
            if actual != set(tasks):
                raise ValueError("missing or extra predictions after parallel inference")
            if connection.execute("SELECT count(*) FROM gpu_tasks WHERE state!='done'").fetchone()[0]:
                raise ValueError("unfinished queue entries")
            counts = dict(connection.execute("SELECT method,count(*) FROM predictions GROUP BY method"))
            save_json(args.run_dir / "progress.json", {
                "completed_unique_predictions": len(actual), "target_unique_predictions": len(tasks),
                "completed_by_method": counts, "gpus": args.gpus, "worker_state": "complete",
                "session": session, "updated": time.time(),
            })
            execution.update(state="complete", finished=time.time(), completed_by_method=counts)
            save_json(control / "execution.json", execution)
        except BaseException as exc:
            execution.update(state="blocked", error=repr(exc), finished=time.time())
            save_json(control / "execution.json", execution)
            raise
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate()
            for child in children:
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
    subprocess.run(score_command(args), check=True, env=validation_env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--benchmark", choices=["POPE", "CHAIR"], default="POPE")
    parser.add_argument("--exclude-samples", type=Path, default=Path("data/calibration.jsonl"))
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--worker")
    parser.add_argument("--session")
    parser.add_argument("--parent-pid", type=int)
    parser.add_argument("--method", choices=["vcd", "vigor"], required=True)
    parser.add_argument("--early-eos", action="store_true")
    args = parser.parse_args()
    if args.config is None:
        args.config = Path(args.method) / "config.yaml"
    if args.worker is not None:
        if args.session is None or args.parent_pid is None:
            parser.error("worker requires coordinator session and parent PID")
        worker(args)
    else:
        def interrupt(signum, frame):
            raise KeyboardInterrupt(f"signal {signum}")
        signal.signal(signal.SIGTERM, interrupt)
        coordinator(args)


if __name__ == "__main__":
    main()
