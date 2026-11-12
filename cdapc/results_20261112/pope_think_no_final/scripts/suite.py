"""Frozen A-OKVQA -> GQA generation, with independent official/DeepSeek scoring."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from cdapc_lab.data.records import build_question, uses_model_reasoning_instruction
from cdapc_lab.evaluation.llm_judge import judge_key
from cdapc_lab.evaluation.normalize import normalize_output
from cdapc_lab.evaluation.pope_llm_judge import judge_request, request_for, spec
from cdapc_lab.evaluation.scorers import load_pope_official_rule
from cdapc_lab.experiments.infer import preflight
from cdapc_lab.experiments.score import pope_metrics
from cdapc_lab.io.artifacts import atomic_json, digest, file_hash, generation_projection, read_json
from cdapc_lab.models.mmada import OFFICIAL_THINKING_INSTRUCTION

SPLITS = ("random", "popular", "adversarial")
CACHE = ROOT.parent / "cdapc-experiments/reports/pope_think_llm_judge/cache"


def save(path, value):
    value["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    atomic_json(path, value)


def inventory(dataset):
    result = {}
    for p in (ROOT / "runs" / dataset).glob("*/predictions/*.json"):
        raw = read_json(p)
        assert raw["sample_id"] not in result, "Duplicate terminal prediction"
        assert raw["status"] in ("success", "budget_exhausted"), str(p)
        result[raw["sample_id"]] = {"path": str(p), "sha256": file_hash(p), "status": raw["status"]}
    return result


def prepare():
    plan = read_json(ROOT / "reports/plan.json")
    reference = Path(plan["source_reference"])
    assert all(file_hash(p) == file_hash(reference / p.relative_to(ROOT))
               for p in (ROOT / "src").rglob("*.py"))
    result = {}
    coco = next(iter(read_json(reference / "reports/preflight.json")["shards"].values()))
    for dataset, stage in plan["stages"].items():
        ids, shards = set(), {}
        for k, name in enumerate(stage["configs"]):
            cfg = resolve_config(ROOT / name)
            samples, _, provenance = preflight(cfg)
            assert provenance["generation_source_hash"] == coco["generation_source_hash"]
            assert provenance["environment"] == coco["environment"]
            assert provenance["assets"] == coco["assets"]
            for sample in samples:
                assert sample.sample_id not in ids
                ids.add(sample.sample_id)
                assert uses_model_reasoning_instruction(sample, cfg["protocol"]["prompt_id"])
                assert build_question(sample, cfg["protocol"]["prompt_id"]) == sample.question
            shards[str(k)] = {"config": name, "expected": len(samples), **provenance}
        rows = read_json(ROOT / f"manifests/{dataset}_rows.json")
        assert ids == {r["input_key"] for r in rows}
        assert len(ids) == stage["inputs"] and len(rows) == 9000
        result[dataset] = {"shards": shards, "inputs": len(ids), "rows": len(rows)}
    atomic_json(ROOT / "reports/preflight.json", result)
    snapshot = {str(p.relative_to(ROOT)): file_hash(p)
                for folder in ("src", "scripts", "configs", "manifests")
                for p in (ROOT / folder).rglob("*")
                if p.is_file() and "__pycache__" not in str(p)}
    for name in ("reports/plan.json", "reports/preflight.json"):
        snapshot[name] = file_hash(ROOT / name)
    atomic_json(ROOT / "reports/source_snapshot.json", snapshot)
    print("SUITE_PREFLIGHT_OK", {k: v["inputs"] for k, v in result.items()}, flush=True)


def metrics(records, rows):
    parts = {s: pope_metrics((records[r["input_key"]]["prediction"], r["label"])
                            for r in rows if r["split"] == s) for s in SPLITS}
    return {"splits": parts,
            "pooled": pope_metrics((records[r["input_key"]]["prediction"], r["label"]) for r in rows),
            "macro": {m: sum(v[m] for v in parts.values()) / 3
                      for m in ("accuracy", "f1", "precision", "recall", "yes_ratio")}}


def official(dataset):
    from transformers import AutoTokenizer
    rows = read_json(ROOT / f"manifests/{dataset}_rows.json")
    questions = {r["input_key"]: r["question"] for r in rows}
    records = inventory(dataset)
    assert set(records) == set(questions)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/mmada", local_files_only=True)
    rule = load_pope_official_rule(ROOT / "upstream/pope/evaluate.py")
    parsed, tokens = {}, {}
    for sid, item in records.items():
        raw = read_json(item["path"])
        question = questions[sid]
        if question not in tokens:
            tokens[question] = tokenizer.apply_chat_template(
                [{"role": "user", "content": OFFICIAL_THINKING_INSTRUCTION + "\n" + question}],
                tokenize=True, add_generation_prompt=True)
        assert raw["prompt_ids"][0][1027:] == tokens[question]
        text = normalize_output(raw["raw_response_ids"], tokenizer, raw["eos_ids"])["text"]
        parsed[sid] = {
            "question": question, "text": text, "generation_status": raw["status"],
            "prediction": rule(text.rsplit("</think>", 1)[-1].strip()) if raw["status"] == "success" else None,
            "source_path": item["path"], "source_sha256": item["sha256"],
            "generation_seconds": raw["generation_seconds"], "iterations": raw["decode_iterations"]}
    report = {"status": "complete", "dataset": dataset, "inputs": len(parsed), "rows": len(rows),
              "metrics": metrics(parsed, rows), "per_input": parsed,
              "generation_statuses": dict(Counter(p["generation_status"] for p in parsed.values())),
              "closed_unique": sum("</think>" in p["text"] for p in parsed.values()),
              "mean_seconds": sum(p["generation_seconds"] for p in parsed.values()) / len(parsed),
              "source_snapshot_sha256": file_hash(ROOT / "reports/source_snapshot.json"),
              "official_rule_sha256": file_hash(ROOT / "upstream/pope/evaluate.py")}
    atomic_json(ROOT / f"reports/{dataset}/inventory.json", records)
    atomic_json(ROOT / f"reports/{dataset}/official.json", report)
    write_markdown(dataset)
    print("OFFICIAL_COMPLETE", dataset, report["metrics"]["macro"], flush=True)


def write_markdown(dataset):
    path = ROOT / "reports" / dataset
    report = read_json(path / "official.json")
    judge = read_json(path / "deepseek.json") if (path / "deepseek.json").exists() else None
    lines = [f"# {dataset.upper()} 新prompt快速EOS", "",
             f"{report['inputs']}个唯一输入，{report['rows']}行；生成失败和缺答保留分母。",
             "think指令＋问题；r70/W16/K4/L256/I128；沿用该数据集seed=42。",
             "", "| split | 官方Acc/F1 | DeepSeek Acc/F1 |", "|---|---:|---:|"]
    for split in (*SPLITS, "macro"):
        a = report["metrics"]["macro"] if split == "macro" else report["metrics"]["splits"][split]
        b = (judge["metrics"]["macro"] if split == "macro" else judge["metrics"]["splits"][split]) if judge else None
        lines.append(f"| {split} | {100*a['accuracy']:.2f}%/{100*a['f1']:.2f}% | "
                     + (f"{100*b['accuracy']:.2f}%/{100*b['f1']:.2f}%" if b else "等待评分") + " |")
    (path / "summary.md").write_text("\n".join(lines) + "\n")


def judge(dataset):
    path = ROOT / "reports" / dataset
    state = {"status": "waiting_for_official", "controller_pid": os.getpid()}
    save(path / "judge_status.json", state)
    try:
        while not (path / "official.json").exists():
            time.sleep(15)
        report = read_json(path / "official.json")
        rows = read_json(ROOT / f"manifests/{dataset}_rows.json")
        records = report["per_input"]
        requests = {digest(request_for(v["question"], v["text"])): request_for(v["question"], v["text"])
                    for v in records.values() if v["generation_status"] == "success"}
        state.update(status="waiting_for_cache_lock", unique_requests=len(requests))
        save(path / "judge_status.json", state)
        with (CACHE.parent / "runner.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            results = {sha: read_json(CACHE / (sha + ".json")) for sha in requests
                       if (CACHE / (sha + ".json")).exists()}
            hits = sum(v["status"] == "success" for v in results.values())
            before = sum(len(v.get("attempts", [])) for v in results.values())
            key = judge_key()
            while True:
                pending = [sha for sha in requests if results.get(sha, {}).get("status") != "success"]
                if not pending:
                    break
                if not key or any(len(results.get(sha, {}).get("attempts", [])) >= 9 for sha in pending):
                    raise RuntimeError("Missing credential or repeated judge failures")
                for offset in range(0, len(pending), 64):
                    batch = pending[offset:offset+64]
                    with ThreadPoolExecutor(max_workers=8) as pool:
                        futures = {pool.submit(judge_request, requests[sha], CACHE, key): sha for sha in batch}
                        for future in as_completed(futures):
                            results[futures[future]] = future.result()
                    failures = sum(results[sha]["status"] != "success" for sha in batch)
                    state.update(status="judging", cache_hits=hits,
                                 succeeded=sum(v["status"] == "success" for v in results.values()),
                                 pending=sum(results.get(sha, {}).get("status") != "success" for sha in requests))
                    save(path / "judge_status.json", state)
                    if failures >= max(4, len(batch)//2):
                        raise RuntimeError("High API/format error rate")
            parsed = {}
            for sid, item in records.items():
                sha = digest(request_for(item["question"], item["text"])) if item["generation_status"] == "success" else None
                if sha:
                    value = results[sha]
                    assert value["request_sha256"] == digest(value["request"]) == sha
                    answer = value["parsed"]
                else:
                    answer = {"answer": None, "status": item["generation_status"], "evidence": ""}
                parsed[sid] = {"prediction": answer["answer"], "status": answer["status"],
                               "evidence": answer["evidence"], "request_sha256": sha}
            result = {"status": "complete", "inputs": len(parsed), "rows": len(rows),
                      "metrics": metrics(parsed, rows), "per_input": parsed,
                      "statuses": dict(Counter(v["status"] for v in parsed.values())),
                      "judge_spec": spec(), "judge_spec_hash": digest(spec()),
                      "official_sha256": file_hash(path / "official.json"),
                      "api": {"errors": 0, "cache_hits": hits,
                              "attempts": sum(len(v.get("attempts", [])) for v in results.values())-before,
                              "returned_models": dict(Counter(v.get("returned_model") for v in results.values()))}}
            atomic_json(path / "deepseek.json", result)
        write_markdown(dataset)
        state.update(status="complete", report=str(path / "summary.md"))
        save(path / "judge_status.json", state)
    except BaseException as error:
        state.update(status="needs_attention", error=repr(error))
        save(path / "judge_status.json", state)
        raise


def generate():
    plan = read_json(ROOT / "reports/plan.json")
    frozen = read_json(ROOT / "reports/source_snapshot.json")
    prepared = read_json(ROOT / "reports/preflight.json")
    assert all(file_hash(ROOT / p) == sha for p, sha in frozen.items())
    state = {"status": "starting", "controller_pid": os.getpid(), "stages": {}}
    workers = {}
    judges = {}
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        with (ROOT / "logs/controller.lock").open("a") as own, \
             (ROOT.parent / "cdapc-experiments/logs/benchmark_extensions/queue.lock").open("a") as lock:
            fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock, fcntl.LOCK_EX)
            for dataset in plan["order"]:
                assert all(file_hash(ROOT / p) == sha for p, sha in frozen.items())
                present = inventory(dataset)
                expected = {r["input_key"] for r in read_json(ROOT / f"manifests/{dataset}_rows.json")}
                assert set(present) <= expected
                stage = {"status": "generating", "preserved": len(present), "expected": len(expected), "shards": {}}
                state["stages"][dataset] = stage
                state.update(status="generating", dataset=dataset)
                previous = set((ROOT / "runs" / dataset).iterdir())
                workers, active = {}, {}
                for k, reference in prepared[dataset]["shards"].items():
                    cfg = yaml.safe_load((ROOT / reference["config"]).read_text())
                    ids = [sid for sid in read_json(ROOT / cfg["run"]["sample_ids_file"]) if sid not in present]
                    if not ids:
                        continue
                    wave = digest(ids)[:12]
                    ids_path = ROOT / f"manifests/{dataset}_{wave}_{k}.json"
                    atomic_json(ids_path, ids)
                    cfg["run"].update(sample_ids_file=str(ids_path), name=f"{dataset}_{wave}_shard{k}")
                    cfg_path = ROOT / f"configs/{dataset}_{wave}_{k}.yaml"
                    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
                    resolved = resolve_config(cfg_path)
                    assert generation_projection(resolved) == generation_projection(resolve_config(ROOT / reference["config"]))
                    samples, _, provenance = preflight(resolved)
                    for field in ("generation_source_hash", "assets", "environment"):
                        assert provenance[field] == reference[field]
                    active[k] = {"config": str(cfg_path), "config_sha256": file_hash(cfg_path),
                                 "ids_sha256": file_hash(ids_path), **provenance}
                    with (ROOT / f"logs/{dataset}_{wave}_{k}.log").open("a") as log:
                        workers[k] = subprocess.Popen([sys.executable, "-m", "cdapc_lab", "infer", "--config", str(cfg_path)],
                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": k,
                                 "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"})
                    stage["shards"][k] = {"worker_pid": workers[k].pid, "expected": len(samples)}
                atomic_json(ROOT / f"reports/{dataset}/active_preflight.json", active)
                while True:
                    for k, process in workers.items():
                        item = stage["shards"][k]
                        matches = [p.parent for p in (ROOT / "runs" / dataset).glob("*/run_manifest.json")
                                   if p.parent not in previous and read_json(p)["generation_fingerprint"] == active[k]["generation_fingerprint"]]
                        if matches:
                            assert len(matches) == 1
                            item.update(run=str(matches[0]), generated=len(list((matches[0] / "predictions").glob("*.json"))))
                        item["returncode"] = process.poll()
                        if item["returncode"] not in (None, 0):
                            raise RuntimeError(f"{dataset} shard {k} failed")
                    stage["generated"] = len(present) + sum(v.get("generated", 0) for v in stage["shards"].values())
                    save(ROOT / "reports/status.json", state)
                    if all(p.poll() is not None for p in workers.values()):
                        break
                    time.sleep(30)
                assert set(inventory(dataset)) == expected
                official(dataset)
                stage["status"] = "official_complete"
                with (ROOT / f"logs/{dataset}_judge.log").open("a") as log:
                    judges[dataset] = subprocess.Popen([sys.executable, str(Path(__file__)), "judge", dataset],
                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
                stage["judge_pid"] = judges[dataset].pid
                save(ROOT / "reports/status.json", state)
        state["status"] = "waiting_for_judges"
        save(ROOT / "reports/status.json", state)
        for dataset, process in judges.items():
            if process.wait() != 0:
                raise RuntimeError(f"{dataset} judge needs attention; generation retained")
            assert read_json(ROOT / f"reports/{dataset}/judge_status.json")["status"] == "complete"
            state["stages"][dataset]["status"] = "complete"
        state["status"] = "complete"
        save(ROOT / "reports/status.json", state)
    except BaseException as error:
        state.update(status="needs_attention", error=repr(error))
        save(ROOT / "reports/status.json", state)
        raise
    finally:
        for process in workers.values():
            if process.poll() is None:
                process.terminate()
        for process in workers.values():
            if process.poll() is None:
                process.wait(timeout=60)


if __name__ == "__main__":
    os.chdir(ROOT)
    if sys.argv[1] == "prepare":
        prepare()
    elif sys.argv[1] == "judge":
        judge(sys.argv[2])
    elif sys.argv[1] == "score":
        official(sys.argv[2])
    else:
        generate()
