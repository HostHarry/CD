"""Run the frozen no-Final-answer arm, then score against the existing control."""
import fcntl
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from collections import Counter

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cdapc_lab.config import resolve_config
from cdapc_lab.data.records import build_question, uses_model_reasoning_instruction
from cdapc_lab.evaluation.normalize import normalize_output
from cdapc_lab.evaluation.scorers import load_pope_official_rule
from cdapc_lab.experiments.infer import preflight
from cdapc_lab.experiments.score import pope_metrics
from cdapc_lab.io.artifacts import atomic_json, file_hash, read_json
from cdapc_lab.models.mmada import OFFICIAL_THINKING_INSTRUCTION

SUFFIX = "\nAfter </think>, end with exactly Final answer: Yes or Final answer: No."
STATE = ROOT / "reports/status.json"


def save(state):
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    atomic_json(STATE, state)


def validate():
    plan = read_json(ROOT / "reports/plan.json")
    base = Path(plan["base_workspace"])
    changed = [str(p.relative_to(ROOT)) for p in (ROOT / "src").rglob("*.py")
               if file_hash(p) != file_hash(base / p.relative_to(ROOT))]
    assert sorted(changed) == sorted(plan["expected_source_changes"]), changed
    result, all_ids = {}, set()
    for key, item in plan["shards"].items():
        cfg = resolve_config(ROOT / item["config"])
        old = resolve_config(base / f"configs/experiments/mmada_pope_coco_think_eos_prefix_r70_k4_L256_I128_shard{key}.yaml")
        for field in ("model", "runtime", "recording", "cdapc", "method"):
            assert cfg[field] == old[field], field
        assert cfg["run"]["seed"] == old["run"]["seed"]
        assert cfg["run"]["replicate_id"] == old["run"]["replicate_id"]
        assert {k: v for k, v in cfg["protocol"].items() if k != "prompt_id"} == {
            k: v for k, v in old["protocol"].items() if k != "prompt_id"}
        samples, manifest, provenance = preflight(cfg)
        for sample in samples:
            assert sample.sample_id not in all_ids
            all_ids.add(sample.sample_id)
            assert uses_model_reasoning_instruction(sample, cfg["protocol"]["prompt_id"])
            assert build_question(sample, cfg["protocol"]["prompt_id"]) == sample.question
            assert build_question(sample, old["protocol"]["prompt_id"]) == sample.question + SUFFIX
        result[key] = {"config": item["config"], "inputs": len(samples), **provenance}
    assert all_ids == set(plan["control"])
    snapshot = {str(p.relative_to(ROOT)): file_hash(p) for folder in ("src", "scripts", "configs", "manifests")
                for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in str(p)}
    atomic_json(ROOT / "reports/preflight.json", {"shards": result, "source_changes": changed,
                                                 "snapshot": snapshot, "plan_sha256": file_hash(ROOT / "reports/plan.json")})
    print("PREFLIGHT_OK", len(all_ids), "inputs", flush=True)
    return result


def score(state):
    plan = read_json(ROOT / "reports/plan.json")
    rows = read_json(ROOT / "manifests/rows.json")
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/mmada", local_files_only=True)
    rule = load_pope_official_rule(ROOT / "upstream/pope/evaluate.py")
    b_paths = {}
    for item in state["shards"].values():
        for path in (ROOT / item["run"] / "predictions").glob("*.json"):
            sid = read_json(path)["sample_id"]
            assert sid not in b_paths
            b_paths[sid] = path
    assert set(b_paths) == set(plan["control"])
    questions = {r["input_key"]: r["question"] for r in rows}
    records = {"A": {}, "B": {}}
    for sid, control in plan["control"].items():
        saved = Path(control["source_score"])
        assert file_hash(saved) == control["source_score_sha256"]
        a_path = saved.parents[2] / "predictions" / saved.name
        for arm, path in (("A", a_path), ("B", b_paths[sid])):
            raw = read_json(path)
            normalized = normalize_output(raw["raw_response_ids"], tokenizer, raw["eos_ids"])
            text = normalized["text"]
            selected = text.rsplit("</think>", 1)[-1].strip()
            effective = rule(selected) if raw["status"] == "success" else None
            if arm == "A":
                assert effective == control["effective_prediction"]
                assert text == read_json(saved)["normalized"]["text"]
            prompt = OFFICIAL_THINKING_INSTRUCTION + "\n" + questions[sid] + (SUFFIX if arm == "A" else "")
            expected = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                    tokenize=True, add_generation_prompt=True)
            assert raw["prompt_ids"][0][max(raw["visual_key_positions"]) + 2:] == expected
            records[arm][sid] = {
                "text": text, "answer_region": selected, "prediction": effective,
                "status": raw["status"], "closed": "</think>" in text,
                "closed_empty": "</think>" in text and not selected,
                "generation_seconds": raw["generation_seconds"], "iterations": raw["decode_iterations"],
                "visible_tokens": normalized["emitted_tokens_before_eos"],
                "source_path": str(path), "source_sha256": file_hash(path), "sample_seed": raw["sample_seed"],
            }
        assert records["A"][sid]["sample_seed"] == records["B"][sid]["sample_seed"]
        a, b = read_json(a_path), read_json(b_paths[sid])
        assert a["prompt_ids"][0][:1027] == b["prompt_ids"][0][:1027], "Image input changed"
    metrics = {}
    splits = ("random", "popular", "adversarial")
    for arm in records:
        data = records[arm]
        split_metrics = {s: pope_metrics((data[r["input_key"]]["prediction"], r["label"])
                                        for r in rows if r["split"] == s) for s in splits}
        metrics[arm] = {
            "splits": split_metrics,
            "macro": {k: statistics.mean(v[k] for v in split_metrics.values())
                      for k in ("accuracy", "f1", "precision", "recall", "yes_ratio")},
            "generation_statuses": dict(Counter(v["status"] for v in data.values())),
            "closed_unique": sum(v["closed"] for v in data.values()),
            "closed_empty_unique": sum(v["closed_empty"] for v in data.values()),
            "failed_rows": sum(data[r["input_key"]]["status"] != "success" for r in rows),
            "false_yes_rows": sum(data[r["input_key"]]["prediction"] == "yes" and r["label"] == "no" for r in rows),
            "mean_seconds": statistics.mean(v["generation_seconds"] for v in data.values()),
            "mean_iterations": statistics.mean(v["iterations"] for v in data.values()),
            "mean_visible_tokens": statistics.mean(v["visible_tokens"] for v in data.values()),
        }
    images = sorted({r["image_sha256"] for r in rows})
    cells = ("TP", "FP", "TN", "FN", "unresolved")
    cube = np.array([[[[pope_metrics(
        (records[arm][r["input_key"]]["prediction"], r["label"])
        for r in rows if r["image_sha256"] == image and r["split"] == split)[c]
        for c in cells] for arm in ("A", "B")] for split in splits] for image in images])
    draws = np.random.default_rng(42).integers(0, len(images), size=(2000, len(images)))
    counts = cube[draws].sum(axis=1)
    accuracies = ((counts[..., 0] + counts[..., 2]) / counts.sum(axis=-1)).mean(axis=1)
    f1 = (2 * counts[..., 0] / (2 * counts[..., 0] + counts[..., 1] + counts[..., 3])).mean(axis=1)
    ci = {name: np.quantile(values[:, 1] - values[:, 0], [.025, .975]).tolist()
          for name, values in (("accuracy", accuracies), ("f1", f1))}
    result = {"scope": "Full 500-image COCO comparison; prompt selected after the 64-image pilot; control reused",
              "candidate_policy": "eos_prefix_only",
              "protocol": "last </think> suffix or full text, then pinned official POPE rule; failures incorrect",
              "images": len(images), "inputs": len(b_paths), "rows": len(rows),
              "plan_sha256": file_hash(ROOT / "reports/plan.json"),
              "official_scorer_sha256": file_hash(ROOT / "upstream/pope/evaluate.py"),
              "metrics": metrics, "paired_image_bootstrap_B_minus_A_95ci": ci,
              "bootstrap_repeats": 2000, "bootstrap_seed": 42, "per_input": records}
    atomic_json(ROOT / "reports/comparison.json", result)
    lines = ["# COCO think：Final answer后缀对照", "",
             f"{len(images)}张图，{len(b_paths)}个唯一输入，{len(rows)}条评测行。A复用旧结果；B仅去掉Final answer后缀。",
             "两组保留相同think指令、r70/W16/K4/L256/I128及EOS前缀候选安全早停。失败保留分母计错。",
             "", "| 条件 | Accuracy | F1 | 闭合输入 | 误报Yes行数 | 秒/输入 |",
             "|---|---:|---:|---:|---:|---:|"]
    for arm, label in (("A", "A：带Final answer后缀"), ("B", "B：不带Final answer后缀")):
        m = metrics[arm]
        lines.append(f"| {label} | {100*m['macro']['accuracy']:.2f}% | {100*m['macro']['f1']:.2f}% | "
                     f"{m['closed_unique']}/{len(b_paths)} | {m['false_yes_rows']} | {m['mean_seconds']:.2f} |")
    lines += ["", "图像配对bootstrap，B−A，2000次、seed=42："]
    for key, bounds in ci.items():
        delta = 100 * (metrics["B"]["macro"][key] - metrics["A"]["macro"][key])
        lines.append(f"- {key}: {delta:+.2f}个百分点，95%区间[{100*bounds[0]:+.2f}, {100*bounds[1]:+.2f}]。")
    lines += ["", "提示协议在64图探索后选择；新增436图另行统计，不称为未调参的独立benchmark。输入图像码、每题seed和实际prompt均逐条核验。"]
    (ROOT / "reports/comparison.md").write_text("\n".join(lines) + "\n")


def main():
    os.chdir(ROOT)
    if "--preflight" in sys.argv:
        validate()
        return
    frozen = read_json(ROOT / "reports/preflight.json")
    assert file_hash(ROOT / "reports/plan.json") == frozen["plan_sha256"]
    assert all(file_hash(ROOT / p) == sha for p, sha in frozen["snapshot"].items())
    state = {"status": "starting", "controller_pid": os.getpid(), "shards": {}}
    workers = {}
    with (ROOT / "logs/controller.lock").open("a") as own, \
            Path("/root/autodl-tmp/cdapc-experiments/logs/benchmark_extensions/queue.lock").open("a") as gpu_lock:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for key, item in frozen["shards"].items():
                matching = [p.parent for p in (ROOT / "runs").glob("*/run_manifest.json")
                            if read_json(p)["generation_fingerprint"] == item["generation_fingerprint"]]
                assert len(matching) <= 1
                command = [sys.executable, "-m", "cdapc_lab", "infer", "--config", item["config"]]
                if matching:
                    command += ["--resume-run", str(matching[0])]
                with (ROOT / f"logs/shard{key}.log").open("a") as log:
                    worker = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                        env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": key,
                             "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"})
                workers[key] = worker
                state["shards"][key] = {"worker_pid": worker.pid, "expected": item["inputs"],
                                       "generation_fingerprint": item["generation_fingerprint"]}
            state["status"] = "generating"
            while True:
                for key, worker in workers.items():
                    item = state["shards"][key]
                    runs = [p.parent for p in (ROOT / "runs").glob("*/run_manifest.json")
                            if read_json(p)["generation_fingerprint"] == item["generation_fingerprint"]]
                    if runs:
                        assert len(runs) == 1
                        item.update(run=str(runs[0].relative_to(ROOT)),
                                    generated=len(list((runs[0] / "predictions").glob("*.json"))))
                    item["returncode"] = worker.poll()
                    if item["returncode"] not in (None, 0):
                        raise RuntimeError(f"shard {key} failed; see its log")
                save(state)
                if all(w.poll() is not None for w in workers.values()):
                    break
                time.sleep(15)
            assert all(item.get("generated") == item["expected"] for item in state["shards"].values())
            state["status"] = "scoring"
            save(state)
            score(state)
            state.update(status="complete", report="reports/comparison.md")
            save(state)
            print("COMPLETE", state["report"], flush=True)
        except BaseException as error:
            state.update(status="interrupted_or_failed", error=repr(error))
            save(state)
            raise
        finally:
            for worker in workers.values():
                if worker.poll() is None:
                    worker.terminate()


if __name__ == "__main__":
    raise SystemExit("Scoring helpers only; use run_full.py")
