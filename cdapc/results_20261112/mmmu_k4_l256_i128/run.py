"""Frozen MMMU K4/EOS-prefix run; official, no-guess, and existing WINO judge."""
from collections import Counter
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cdapc_lab.config import resolve_config
from cdapc_lab.data.mmmu import mmmu_judge_question
from cdapc_lab.data.records import build_question, uses_model_reasoning_instruction
from cdapc_lab.evaluation.llm_judge import ANSWER_RE, build_request, judge_key, judge_spec
from cdapc_lab.experiments.infer import preflight
from cdapc_lab.experiments.mmmu_report import mmmu_report, question_bootstrap
from cdapc_lab.experiments.score import score_run, select_evaluation
from cdapc_lab.io.artifacts import atomic_json, digest, file_hash, generation_projection, read_json, safe_id


def save(state):
    state["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    atomic_json(ROOT / "reports/status.json", state)


def prepare():
    plan = read_json(ROOT / "reports/plan.json")
    reference = Path(plan["source_reference"])
    assert all(file_hash(p) == file_hash(reference / p.relative_to(ROOT))
               for p in (ROOT / "src").rglob("*.py"))
    common = next(iter(read_json(reference / "reports/preflight.json")["shards"].values()))
    shards, all_ids = {}, set()
    reference_inputs = {}
    for k, item in plan["shards"].items():
        cfg = resolve_config(ROOT / item["config"])
        old = resolve_config(item["reference_config"])
        current_projection, old_projection = generation_projection(cfg), generation_projection(old)
        old_projection["cdapc"]["max_commit_per_iteration"] = 4
        assert current_projection == old_projection
        samples, manifest, provenance = preflight(cfg)
        for field in ("generation_source_hash", "environment", "assets"):
            assert provenance[field] == common[field], field
        prior = Path(item["reference_run"])
        assert manifest == read_json(prior / "dataset_manifest.json")
        for sample in samples:
            assert sample.sample_id not in all_ids
            all_ids.add(sample.sample_id)
            assert uses_model_reasoning_instruction(sample, cfg["protocol"]["prompt_id"])
            assert build_question(sample, cfg["protocol"]["prompt_id"]) == build_question(sample, old["protocol"]["prompt_id"])
            path = prior / "predictions" / (safe_id(sample.sample_id) + ".json")
            raw = read_json(path)
            assert raw["sample_id"] == sample.sample_id
            reference_inputs[sample.sample_id] = {"path": str(path), "sha256": file_hash(path)}
        shards[k] = {"config": item["config"], "expected": len(samples), **provenance}
    assert all_ids == set(read_json(ROOT / "manifests/mmmu_val_all_ids.json"))
    assert len(all_ids) == 900
    gold = read_json(ROOT / "data/mmmu_val/gold.json")
    assert Counter(r["metric_metadata"]["question_type"] for r in gold) == plan["question_types"]
    spec, spec_hash = judge_spec("deepseek-flash")
    assert judge_key(), "DeepSeek credential not available"
    atomic_json(ROOT / "reports/reference_inputs.json", reference_inputs)
    atomic_json(ROOT / "reports/preflight.json", {"shards": shards, "judge_spec": spec, "judge_spec_hash": spec_hash,
                "historical_report_sha256": file_hash(plan["historical_report"]),
                "source_reference": str(reference), "inputs": 900, "question_types": plan["question_types"]})
    snapshot = {str(p.relative_to(ROOT)): file_hash(p)
                for folder in ("src", "scripts", "configs", "manifests")
                for p in (ROOT / folder).rglob("*")
                if p.is_file() and "__pycache__" not in str(p)}
    for name in ("plan", "preflight", "reference_inputs"):
        snapshot[f"reports/{name}.json"] = file_hash(ROOT / f"reports/{name}.json")
    atomic_json(ROOT / "reports/source_snapshot.json", snapshot)
    print("MMMU_PREFLIGHT_OK", len(all_ids), "same questions/images/options/prompt; K4 + EOS prefix", flush=True)


def collect():
    result = {}
    expected = set(read_json(ROOT / "manifests/mmmu_val_all_ids.json"))
    for path in (ROOT / "runs").glob("*/predictions/*.json"):
        raw = read_json(path)
        sid = raw["sample_id"]
        assert sid in expected and sid not in result
        assert raw["status"] in ("success", "budget_exhausted", "error")
        result[sid] = {"path": str(path), "sha256": file_hash(path), "status": raw["status"]}
    return result


def verify_inputs(records):
    reference = read_json(ROOT / "reports/reference_inputs.json")
    checked, failed = 0, []
    for sid, item in records.items():
        prior = reference[sid]
        assert file_hash(prior["path"]) == prior["sha256"]
        old, new = read_json(prior["path"]), read_json(item["path"])
        assert old["sample_seed"] == new["sample_seed"]
        if new["status"] == "error":
            failed.append(sid)
            continue
        assert old["prompt_ids"] == new["prompt_ids"], sid
        assert old["eos_ids"] == new["eos_ids"]
        checked += 1
    atomic_json(ROOT / "reports/actual_input_verification.json",
                {"checked": checked, "runtime_error_inputs": failed,
                 "comparison": "Exact model prompt IDs including image codes and per-input seeds vs historical MMMU"})
    return checked


def reuse_judgments(runs):
    plan = read_json(ROOT / "reports/plan.json")
    reference = read_json(ROOT / "reports/reference_inputs.json")
    spec, spec_hash = judge_spec("deepseek-flash")
    source = {r["sample_id"]: r for r in read_json(ROOT / "data/mmmu_val/samples.json")}
    gold = {r["sample_id"]: r for r in read_json(ROOT / "data/mmmu_val/gold.json")}
    reused = 0
    for run in runs:
        _, scored = select_evaluation(run)
        target_dir = Path(run) / "judge" / spec_hash
        target_dir.mkdir(parents=True, exist_ok=True)
        for row in scored:
            sid = row["sample_id"]
            old_run = Path(reference[sid]["path"]).parents[1]
            old_path = old_run / "judge" / spec_hash / (safe_id(sid) + ".json")
            target = target_dir / old_path.name
            if target.exists() or not old_path.exists():
                continue
            old = read_json(old_path)
            if old.get("status") == "judge_error":
                continue
            text = row.get("normalized", {}).get("text", "")
            match = ANSWER_RE.search(text)
            pred = match.group(1) if match else text
            request = build_request(spec, mmmu_judge_question(source[sid]["question"], source[sid]["options"]),
                                    gold[sid]["answer"], pred)
            request_sha = hashlib.sha256(json.dumps(request, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            if old["request"] == request and old["request_sha256"] == request_sha:
                copied = {**old, "cache_source": str(old_path), "cache_source_sha256": file_hash(old_path)}
                atomic_json(target, copied)
                reused += 1
    atomic_json(ROOT / "reports/judge_cache_reuse.json", {"reused": reused, "judge_spec_hash": spec_hash})
    return spec_hash


def finalize(runs):
    plan = read_json(ROOT / "reports/plan.json")
    prepared = read_json(ROOT / "reports/preflight.json")
    assert file_hash(plan["historical_report"]) == prepared["historical_report_sha256"]
    report = mmmu_report(runs)
    block = report["cdapc"]
    assert block["covered"] == block["judged"] == 900 and block["judge_errors"] == 0
    assert all(block["blocks"]["all"][m]["n"] == 900 for m in ("official", "strict", "judge"))
    historical = read_json(plan["historical_report"])["cdapc"]
    assert set(block["per_sample"]) == set(historical["per_sample"])
    pairs = {}
    for metric in ("official", "strict", "judge"):
        a = {sid: value[metric] for sid, value in historical["per_sample"].items()}
        b = {sid: value[metric] for sid, value in block["per_sample"].items()}
        assert all(v in (0, 1) for v in (*a.values(), *b.values()))
        pairs[metric] = question_bootstrap(a, b)
        assert pairs[metric]["n"] == 900
    records = collect()
    assert len(records) == 900
    status_counts = dict(Counter(v["status"] for v in records.values()))
    answer_rows = {}
    for run in runs:
        meta, scored = select_evaluation(run)
        for row in scored:
            sid = row["sample_id"]
            pred = read_json(records[sid]["path"])
            values = block["per_sample"][sid]
            assert pred["status"] == row["generation_status"]
            if pred["status"] != "success":
                assert all(values[m] == 0 for m in ("official", "strict", "judge"))
            answer_rows[sid] = {"sample_id": sid, "text": row.get("normalized", {}).get("text", ""),
                               "generation_status": pred["status"], "gold": row["gold"],
                               "official": values["official"], "strict": values["strict"],
                               "judge": values["judge"], "source_path": records[sid]["path"],
                               "source_sha256": records[sid]["sha256"], "evaluation_fingerprint": meta["evaluation_fingerprint"]}
    for metric in ("official", "strict", "judge"):
        assert sum(r[metric] for r in answer_rows.values()) / 900 == block["blocks"]["all"][metric]["accuracy"]
    assert all(file_hash(ROOT / p) == sha for p, sha in read_json(ROOT / "reports/source_snapshot.json").items())
    report.update(condition=plan["condition"], generation_statuses=status_counts,
                  comparison_note=plan["reference_note"], paired_vs_historical_k16=pairs,
                  historical_report=plan["historical_report"], source_snapshot_sha256=file_hash(ROOT / "reports/source_snapshot.json"))
    atomic_json(ROOT / "reports/result.json", report)
    ids = read_json(ROOT / "manifests/mmmu_val_all_ids.json")
    (ROOT / "reports/answers.jsonl").write_text("".join(json.dumps(answer_rows[sid], ensure_ascii=False) + "\n" for sid in ids))
    lines = ["# MMMU r70/W16/K4：L256/I128，快速EOS前缀", "",
             "完整900题（847选择题、53开放题）。既定WINO思考提示与图像选择/预处理不变。",
             "Official含无法解析时的随机fallback；Strict不猜答案；Judge沿用DeepSeek/WINO正确性判分。",
             "所有生成失败固定计错；API错误须解决后才能完成报告。", "",
             "| 口径 | 当前Accuracy | 历史K16 | 差值95% CI（百分点） |", "|---|---:|---:|---:|"]
    for metric in ("official", "strict", "judge"):
        current = block["blocks"]["all"][metric]["accuracy"]
        old = historical["blocks"]["all"][metric]["accuracy"]
        lo, hi = pairs[metric]["ci95"]
        lines.append(f"| {metric} | {100*current:.2f}% | {100*old:.2f}% | [{100*lo:+.2f}, {100*hi:+.2f}] |")
    lines += ["", "历史组为K16＋原候选范围，本轮同时改变K与EOS候选策略，不是单变量K对照。",
              "外部baseline的block_length=64不等于CDAPC滚动窗口W16；未在此表中混入另一服务器的结果。",
              f"生成状态：{status_counts}。平均生成{block['cost']['mean_generation_seconds']:.2f}秒/输入。",
              "900题分母、实际输入、源快照及逐题分数均已核验。"]
    (ROOT / "reports/summary.md").write_text("\n".join(lines) + "\n")
    paths = [ROOT / "reports" / name for name in ("result.json", "answers.jsonl", "summary.md",
             "prediction_inventory.json", "actual_input_verification.json", "judge_cache_reuse.json",
             "plan.json", "preflight.json", "source_snapshot.json")]
    (ROOT / "reports/final.sha256").write_text("".join(f"{file_hash(p)}  {p.relative_to(ROOT)}\n" for p in paths))


def main():
    os.chdir(ROOT)
    if "--prepare" in sys.argv:
        prepare()
        return
    state = {"status": "starting", "controller_pid": os.getpid(), "expected_inputs": 900}
    workers = {}
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        with (ROOT / "logs/controller.lock").open("a") as own:
            fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
            frozen = read_json(ROOT / "reports/source_snapshot.json")
            assert all(file_hash(ROOT / p) == sha for p, sha in frozen.items())
            prepared = read_json(ROOT / "reports/preflight.json")
            with (ROOT.parent / "cdapc-experiments/logs/benchmark_extensions/queue.lock").open("a") as gpu_lock:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX)
                present = collect()
                previous = set((ROOT / "runs").iterdir())
                active = {}
                state.update(status="generating", preserved=len(present), shards={})
                for k, ref in prepared["shards"].items():
                    cfg = yaml.safe_load((ROOT / ref["config"]).read_text())
                    ids = [sid for sid in read_json(ROOT / cfg["run"]["sample_ids_file"]) if sid not in present]
                    if not ids:
                        continue
                    wave = digest(ids)[:12]
                    ids_path = ROOT / f"manifests/{wave}_{k}.json"
                    atomic_json(ids_path, ids)
                    cfg["run"].update(name=f"mmmu_prefix_k4_{wave}_shard{k}", sample_ids_file=str(ids_path))
                    path = ROOT / f"configs/{wave}_{k}.yaml"
                    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
                    samples, _, provenance = preflight(resolve_config(path))
                    for field in ("generation_source_hash", "environment", "assets"):
                        assert provenance[field] == ref[field]
                    assert generation_projection(resolve_config(path)) == generation_projection(resolve_config(ROOT / ref["config"]))
                    active[k] = {"config": str(path), "expected": len(samples), "config_sha256": file_hash(path), **provenance}
                    with (ROOT / f"logs/generation_{wave}_{k}.log").open("a") as log:
                        workers[k] = subprocess.Popen([sys.executable, "-m", "cdapc_lab", "infer", "--config", str(path)],
                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": k,
                                 "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"})
                    state["shards"][k] = {"worker_pid": workers[k].pid, "expected": len(samples)}
                atomic_json(ROOT / "reports/active_preflight.json", active)
                while True:
                    for k, process in workers.items():
                        item = state["shards"][k]
                        matches = [p.parent for p in (ROOT / "runs").glob("*/run_manifest.json")
                                   if p.parent not in previous and read_json(p)["generation_fingerprint"] == active[k]["generation_fingerprint"]]
                        if matches:
                            assert len(matches) == 1
                            item.update(run=str(matches[0]), generated=len(list((matches[0] / "predictions").glob("*.json"))))
                        item["returncode"] = process.poll()
                        if item["returncode"] not in (None, 0):
                            raise RuntimeError(f"Shard {k} failed")
                    state["generated"] = len(present) + sum(v.get("generated", 0) for v in state["shards"].values())
                    save(state)
                    if all(p.poll() is not None for p in workers.values()):
                        break
                    time.sleep(20)
            records = collect()
            assert len(records) == 900
            atomic_json(ROOT / "reports/prediction_inventory.json", records)
            verify_inputs(records)
            runs = sorted({str(Path(v["path"]).parents[1]) for v in records.values()})
            state["status"] = "scoring"
            save(state)
            for run in runs:
                score_run(run)
            spec_hash = reuse_judgments(runs)
            state["status"] = "judging"
            save(state)
            for attempt in range(3):
                workers = {}
                for k, run in enumerate(runs):
                    with (ROOT / f"logs/judge_{Path(run).name}.log").open("a") as log:
                        workers[k] = subprocess.Popen([sys.executable, "-m", "cdapc_lab", "mmmu-judge", "--run", run],
                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": "",
                                 "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
                while True:
                    files = [p for run in runs for p in (Path(run) / "judge" / spec_hash).glob("*.json")
                             if p.name != "judge_manifest.json"]
                    states = [read_json(p).get("status", "success") for p in files]
                    state.update(judge_attempt=attempt+1, judged=sum(s != "judge_error" for s in states),
                                 judge_errors=states.count("judge_error"))
                    save(state)
                    if all(p.poll() is not None for p in workers.values()):
                        break
                    time.sleep(15)
                if any(p.returncode != 0 for p in workers.values()):
                    raise RuntimeError("MMMU judge worker failed")
                if state["judged"] == 900 and not state["judge_errors"]:
                    break
                if attempt == 2:
                    raise RuntimeError("MMMU judge incomplete; raw generations retained")
            finalize(runs)
            state.update(status="complete", report="reports/summary.md")
            save(state)
            print("MMMU_FULL_RESULT_COMPLETE", flush=True)
    except BaseException as error:
        state.update(status="needs_attention", error=repr(error))
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
