"""Frozen MME think run: r70/W16/K4/L256/I128, EOS-prefix candidates, official MME scoring."""
from collections import Counter
from datetime import datetime
import fcntl
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import yaml

ROOT = Path(__file__).resolve().parents[1]
PREFIX = ROOT.parent / "cdapc-pope-coco-prefix-no-final"
HISTORY = ROOT.parent / "cdapc-mme"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from cdapc_lab.config import resolve_config
from cdapc_lab.data.records import build_question, uses_model_reasoning_instruction
from cdapc_lab.evaluation.normalize import normalize_output
from cdapc_lab.experiments.infer import preflight
from cdapc_lab.io.artifacts import atomic_json, digest, file_hash, generation_projection, read_json, safe_id
from cdapc_lab.models.mmada import OFFICIAL_THINKING_INSTRUCTION
from reference_score_mme import metrics, official_protocol

SHARED = ("decoding", "models", "io", "types.py", "experiments/infer.py")


def save(state):
    state["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    atomic_json(ROOT / "reports/status.json", state)


def prepare():
    for name in SHARED:
        paths = [ROOT / "src/cdapc_lab" / name] if name.endswith(".py") else \
            sorted((ROOT / "src/cdapc_lab" / name).rglob("*.py"))
        for path in paths:
            assert file_hash(path) == file_hash(PREFIX / path.relative_to(ROOT)), str(path)
    history = read_json(HISTORY / "reports/mme_cdapc_r70_L256_I128.json")
    shards, ids = {}, set()
    for k in range(3):
        cfg = resolve_config(ROOT / f"configs/shard{k}.yaml")
        old = resolve_config(HISTORY / f"configs/experiments/mmada_mme_cdapc_r70_L256_I128_shard{k}.yaml")
        a, b = generation_projection(cfg), generation_projection(old)
        assert a["cdapc"]["max_commit_per_iteration"] == 4 and b["cdapc"]["max_commit_per_iteration"] == 16
        assert cfg["protocol"]["prompt_id"] == "mme_think_v1" and old["protocol"]["prompt_id"] == "mme_yesno_v1"
        for field in ("model", "runtime"):
            assert cfg[field] == old[field], field
        assert cfg["run"]["seed"] == old["run"]["seed"] and cfg["run"]["sample_ids_file"] == old["run"]["sample_ids_file"]
        samples, manifest, provenance = preflight(cfg)
        prior = read_json(Path(history["runs"][k]) / "dataset_manifest.json")
        assert [s["content_hash"] for s in manifest["samples"]] == [s["content_hash"] for s in prior["samples"]]
        for sample in samples:
            assert sample.sample_id not in ids
            ids.add(sample.sample_id)
            assert uses_model_reasoning_instruction(sample, cfg["protocol"]["prompt_id"])
            assert build_question(sample, cfg["protocol"]["prompt_id"]) == sample.question
        shards[str(k)] = {"config": f"configs/shard{k}.yaml", "expected": len(samples),
                          "reference_run": history["runs"][k], **provenance}
    rows = read_json(ROOT / "data/mme/rows.json")
    assert ids == {r["sample_id"] for r in rows} and len(ids) == 2374
    assert len({r["group_id"] for r in rows}) == 1187
    atomic_json(ROOT / "reports/preflight.json", {
        "shards": shards, "inputs": 2374, "image_pairs": 1187,
        "prompt": "MMaDA official thinking instruction + verbatim MME question (includes 'Please answer yes or no.')",
        "historical_report": str(HISTORY / "reports/mme_cdapc_r70_L256_I128.json"),
        "historical_report_sha256": file_hash(HISTORY / "reports/mme_cdapc_r70_L256_I128.json"),
        "shared_decoder_sources_match_prefix_workspace": True})
    snapshot = {str(p.relative_to(ROOT)): file_hash(p)
                for folder in ("src", "scripts", "configs", "manifests")
                for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in str(p)}
    snapshot["reports/preflight.json"] = file_hash(ROOT / "reports/preflight.json")
    snapshot["upstream.lock.json"] = file_hash(ROOT / "upstream.lock.json")
    atomic_json(ROOT / "reports/source_snapshot.json", snapshot)
    print("MME_PREFLIGHT_OK", len(ids), "inputs", flush=True)


def collect():
    result = {}
    for path in (ROOT / "runs").glob("*/predictions/*.json"):
        raw = read_json(path)
        assert raw["sample_id"] not in result
        assert raw["status"] in ("success", "budget_exhausted"), str(path)
        result[raw["sample_id"]] = {"path": str(path), "sha256": file_hash(path), "status": raw["status"]}
    return result


def verify_inputs(records):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/mmada", local_files_only=True)
    questions = {r["sample_id"]: r["question"] for r in read_json(ROOT / "data/mme/rows.json")}
    history = read_json(HISTORY / "reports/mme_cdapc_r70_L256_I128.json")
    old_paths = {sid: Path(v["run"]) / "predictions" / (safe_id(sid) + ".json") for sid, v in history["per_sample"].items()}
    checked = 0
    for sid, item in records.items():
        new, old = read_json(item["path"]), read_json(old_paths[sid])
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": OFFICIAL_THINKING_INSTRUCTION + "\n" + questions[sid]}],
            tokenize=True, add_generation_prompt=True)
        assert new["prompt_ids"][0][1027:] == text, sid
        assert new["prompt_ids"][0][:1027] == old["prompt_ids"][0][:1027], sid
        assert new["sample_seed"] == old["sample_seed"]
        checked += 1
    atomic_json(ROOT / "reports/actual_input_verification.json", {
        "checked": checked, "text": "think instruction + verbatim question tokens",
        "image": "image tokens identical to historical direct MME run", "seed": "identical per-sample seeds"})
    return checked


def score(records):
    from transformers import AutoTokenizer
    rows = read_json(ROOT / "data/mme/rows.json")
    provenance = read_json(ROOT / "data/mme/provenance.json")
    assert file_hash(ROOT / "upstream/mme/calculation.py") == provenance["scorer_sha256"]
    categories, parse = official_protocol()
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/mmada", local_files_only=True)
    answers, raw_answers, per, exported = {}, {}, {}, {}
    statuses, finish = Counter(), Counter()
    seconds = iterations = closed = empty = 0
    for sid, item in records.items():
        p = read_json(item["path"])
        text = normalize_output(p["raw_response_ids"], tokenizer, p["eos_ids"])["text"]
        is_closed = "</think>" in text
        selected = text.rsplit("</think>", 1)[-1].strip() if is_closed else text
        ok = p["status"] == "success"
        answers[sid] = parse(selected) if ok else "other"
        raw_answers[sid] = parse(text) if ok else "other"
        exported[sid] = selected.replace("\t", " ").replace("\r", " ").replace("\n", " ") if ok else ""
        closed += is_closed
        empty += is_closed and not selected
        statuses[p["status"]] += 1
        finish[p["finish_reason"]] += 1
        seconds += p["generation_seconds"]
        iterations += p["decode_iterations"]
        per[sid] = {"text": text, "selected_answer_text": selected, "prediction": answers[sid],
                    "raw_full_text_prediction": raw_answers[sid], "generation_status": p["status"],
                    "finish_reason": p["finish_reason"], "source_path": item["path"], "source_sha256": item["sha256"]}
    assert len(per) == 2374
    primary = metrics(rows, answers, categories)
    raw = metrics(rows, raw_answers, categories)
    export = ROOT / "reports/mme_official_text"
    export.mkdir(exist_ok=True)
    for category in primary["per_category"]:
        selected_rows = sorted((r for r in rows if r["category"] == category), key=lambda r: (r["group_id"], r["sample_id"]))
        (export / f"{category}.txt").write_text("".join(
            f"{r['image_name']}\t{r['question']}\t{r['answer']}\t{exported[r['sample_id']]}\n" for r in selected_rows))
    checked = subprocess.run([sys.executable, str(ROOT / "upstream/mme/calculation.py"), "--results_dir", str(export)],
                             capture_output=True, text=True, check=True,
                             env={**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "vendor"), str(ROOT / "src")])})
    (ROOT / "reports/mme_official_scorer.log").write_text(checked.stdout + checked.stderr)
    official = dict(re.findall(r"^[ \t]+(\w+)[ \t]+score:\s+([\d.eE+-]+)", checked.stdout, re.M))
    assert set(official) == set(primary["per_category"])
    for name, value in official.items():
        assert abs(float(value) - primary["per_category"][name]["score"]) < 1e-8
    history = read_json(HISTORY / "reports/mme_cdapc_r70_L256_I128.json")
    assert file_hash(HISTORY / "reports/mme_cdapc_r70_L256_I128.json") == read_json(ROOT / "reports/preflight.json")["historical_report_sha256"]
    yes_share = sum(a == "yes" for a in answers.values()) / len(answers)
    other = sum(a == "other" for a in answers.values())
    report = {"status": "complete", "inputs": 2374, "image_pairs": 1187,
              "condition": {"prompt_id": "mme_think_v1", "r_gate": 0.7, "window": 16, "max_commit": 4,
                            "canvas": 256, "iterations": 128, "eos": "prefix candidates", "seed": 20261105},
              "primary_scope": "Answer after last </think>; whole visible text when unclosed; official MME parser",
              "metrics": primary, "raw_full_text_diagnostic": raw,
              "historical_direct_k16": history["metrics"],
              "generation_status_counts": dict(statuses), "finish_reason_counts": dict(finish),
              "closed_think": closed, "closed_empty_answer": empty, "other_predictions": other, "yes_share": yes_share,
              "cost": {"mean_seconds": seconds / 2374, "mean_iterations": iterations / 2374},
              "official_scorer_verified": True, "per_sample": per,
              "source_snapshot_sha256": file_hash(ROOT / "reports/source_snapshot.json"),
              "notes": ["Historical direct K16 changes prompt, K and EOS candidate policy; not a single-variable comparison.",
                        "Budget-exhausted outputs count incorrect ('other') with fixed denominators.",
                        "No API judge; MME uses its official rule-based parser."]}
    atomic_json(ROOT / "reports/result.json", report)
    lines = ["# MME think：r70/W16/K4，L256/I128，快速EOS前缀", "",
             "全部2374题、1187个图像问题对、14类。MMaDA官方think指令＋MME原题（含Please answer yes or no.）。",
             "主口径：取最后一个`</think>`之后的回答，未闭合用全文，再用MME官方前4字符Yes/No解析；每类100×(Acc+Acc+)。",
             "", f"| 项目 | 本轮think K4 | 历史直答K16 |", "|---|---:|---:|"]
    for key, label in (("perception", "Perception /2000"), ("cognition", "Cognition /800"), ("total", "总分 /2800")):
        lines.append(f"| {label} | {primary[key]:.2f} | {history['metrics'][key]:.2f} |")
    lines += ["", "| 类别 | 题数 | Acc % | Acc+ % | 得分 | 历史直答K16得分 |", "|---|---:|---:|---:|---:|---:|"]
    for name, v in primary["per_category"].items():
        lines.append(f"| {name} | {v['questions']} | {100*v['acc']:.2f} | {100*v['acc_plus']:.2f} | "
                     f"{v['score']:.2f} | {history['metrics']['per_category'][name]['score']:.2f} |")
    lines += ["", f"`</think>`闭合 {closed}/2374；闭合后为空 {empty}；无法解析为Yes/No {other}；Yes比例 {100*yes_share:.2f}%。",
              f"全文直接解析诊断：总分 {raw['total']:.2f}（非主口径）。",
              f"生成状态：{dict(statuses)}；平均 {seconds/2374:.2f} 秒、{iterations/2374:.2f} 轮/题。",
              "历史组同时不同于提示词、K和EOS候选策略，不是单变量对照。官方未修改评分脚本已逐类核对。"]
    (ROOT / "reports/summary.md").write_text("\n".join(lines) + "\n")
    paths = [ROOT / "reports" / n for n in ("result.json", "summary.md", "prediction_inventory.json",
             "actual_input_verification.json", "preflight.json", "source_snapshot.json", "mme_official_scorer.log")]
    (ROOT / "reports/final.sha256").write_text("".join(f"{file_hash(p)}  {p.relative_to(ROOT)}\n" for p in paths))
    return report


def main():
    os.chdir(ROOT)
    if "--prepare" in sys.argv:
        prepare()
        return
    if "--score" in sys.argv:
        score(read_json(ROOT / "reports/prediction_inventory.json"))
        return
    state = {"status": "starting", "controller_pid": os.getpid(), "expected_inputs": 2374}
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
                    cfg["run"].update(name=f"mme_think_k4_{wave}_shard{k}", sample_ids_file=str(ids_path))
                    path = ROOT / f"configs/{wave}_{k}.yaml"
                    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
                    resolved = resolve_config(path)
                    assert generation_projection(resolved) == generation_projection(resolve_config(ROOT / ref["config"]))
                    samples, _, provenance = preflight(resolved)
                    for field in ("generation_source_hash", "environment", "assets"):
                        assert provenance[field] == ref[field], field
                    active[k] = {"config": str(path), "expected": len(samples), **provenance}
                    with (ROOT / f"logs/generation_{wave}_{k}.log").open("a") as log:
                        workers[k] = subprocess.Popen(
                            [sys.executable, "-m", "cdapc_lab", "infer", "--config", str(path)],
                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": k,
                                 "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"})
                    state["shards"][k] = {"worker_pid": workers[k].pid, "expected": len(samples)}
                atomic_json(ROOT / "reports/active_preflight.json", active)
                while True:
                    for k, process in workers.items():
                        item = state["shards"][k]
                        matches = [p.parent for p in (ROOT / "runs").glob("*/run_manifest.json")
                                   if p.parent not in previous
                                   and read_json(p)["generation_fingerprint"] == active[k]["generation_fingerprint"]]
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
            assert len(records) == 2374
            atomic_json(ROOT / "reports/prediction_inventory.json", records)
            state["status"] = "verifying_and_scoring"
            save(state)
            verify_inputs(records)
            report = score(records)
            state.update(status="complete", report="reports/summary.md", total=report["metrics"]["total"])
            save(state)
            print("MME_THINK_COMPLETE", report["metrics"]["total"], flush=True)
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
