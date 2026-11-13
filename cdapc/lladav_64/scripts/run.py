"""LLaDA-V POPE-COCO think at 64/64: Native (64 steps, block 64) then CDAPC (r70/W16/K4, L64/I64); then score."""
from collections import Counter
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cdapc_lab.io.artifacts import atomic_json, digest, read_json

ARMS = ("native", "cdapc")
SPLITS = ("random", "popular", "adversarial")
FINAL = re.compile(r"Final answer:\s*\**\s*(Yes|No)\b", re.I)


def save(state):
    state["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    atomic_json(ROOT / "reports/status.json", state)


def collect(arm):
    out = {}
    for p in (ROOT / "runs" / arm).glob("*/predictions/*.json"):
        r = read_json(p)
        assert r["sample_id"] not in out
        if r["status"] in ("success", "budget_exhausted"):
            out[r["sample_id"]] = str(p)
    return out


def generate(arm, state, workers):
    (ROOT / "runs" / arm).mkdir(parents=True, exist_ok=True)
    present = collect(arm)
    stage = state["arms"].setdefault(arm, {})
    stage.update(status="generating", preserved=len(present), shards={})
    for k in range(3):
        cfg = yaml.safe_load((ROOT / f"configs/{arm}_shard{k}.yaml").read_text())
        ids = [s for s in read_json(ROOT / cfg["run"]["sample_ids_file"]) if s not in present]
        if not ids:
            continue
        wave = digest(ids)[:12]
        ids_path = ROOT / f"manifests/{arm}_{wave}_{k}.json"
        atomic_json(ids_path, ids)
        cfg["run"].update(name=f"lladav64_{arm}_{wave}_shard{k}", sample_ids_file=str(ids_path))
        path = ROOT / f"configs/{arm}_{wave}_{k}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        with (ROOT / f"logs/{arm}_{wave}_{k}.log").open("a") as log:
            workers[k] = subprocess.Popen([sys.executable, "-m", "cdapc_lab", "infer", "--config", str(path)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": str(k),
                     "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "OMP_NUM_THREADS": "8"})
        stage["shards"][str(k)] = {"pid": workers[k].pid, "expected": len(ids)}
    while workers:
        stage["generated"] = len(collect(arm))
        save(state)
        for k, p in list(workers.items()):
            if p.poll() is not None:
                if p.returncode != 0:
                    raise RuntimeError(f"{arm} shard {k} exited {p.returncode}")
                del workers[k]
        time.sleep(30)
    stage.update(status="generated", generated=len(collect(arm)))
    assert stage["generated"] == 5127, stage["generated"]
    save(state)


def parse(text, rule):
    found = FINAL.findall(text)
    if found:
        return found[-1].lower(), "final_answer"
    head = re.match(r"^\W*(yes|no)\b", text.rsplit("</think>", 1)[-1].strip(), re.I)
    if head:
        return head.group(1).lower(), "leading_yes_no"
    return rule(text), "pope_official_first_sentence"


def metrics(rows, pred):
    out = {}
    for s in SPLITS:
        sel = [r for r in rows if r["split"] == s]
        tp = sum(pred[r["input_key"]] == "yes" and r["label"] == "yes" for r in sel)
        fp = sum(pred[r["input_key"]] == "yes" and r["label"] == "no" for r in sel)
        fn = sum(pred[r["input_key"]] != "yes" and r["label"] == "yes" for r in sel)
        out[s] = {"accuracy": sum(pred[r["input_key"]] == r["label"] for r in sel) / len(sel),
                  "f1": 2 * tp / (2 * tp + fp + fn), "yes_ratio": sum(pred[r["input_key"]] == "yes" for r in sel) / len(sel)}
    out["mean"] = {m: float(np.mean([out[s][m] for s in SPLITS])) for m in ("accuracy", "f1", "yes_ratio")}
    return out


def bootstrap(rows, a, b, repeats=2000, seed=42):
    images = sorted({r["image_sha256"] for r in rows})
    idx = {x: i for i, x in enumerate(images)}
    cube = np.zeros((len(images), 3, 2, 5))
    for r in rows:
        for j, pred in enumerate((a, b)):
            p, g = pred[r["input_key"]], r["label"]
            cube[idx[r["image_sha256"]], SPLITS.index(r["split"]), j] += [p == g, 1, p == "yes" and g == "yes",
                                                                          p == "yes" and g == "no", p != "yes" and g == "yes"]
    c = cube[np.random.default_rng(seed).integers(0, len(images), (repeats, len(images)))].sum(1)
    acc, f1 = c[..., 0] / c[..., 1], 2 * c[..., 2] / (2 * c[..., 2] + c[..., 3] + c[..., 4])
    return {m: np.quantile((v[:, :, 0] - v[:, :, 1]).mean(1), [0.025, 0.975]).tolist() for m, v in (("accuracy", acc), ("f1", f1))}


def score():
    from transformers import AutoTokenizer
    from cdapc_lab.evaluation.normalize import normalize_output
    from cdapc_lab.evaluation.scorers import load_pope_official_rule
    tok = AutoTokenizer.from_pretrained(ROOT / "models/lladav", local_files_only=True)
    rule = load_pope_official_rule(ROOT / "upstream/pope/evaluate.py")
    rows = read_json(ROOT / "data/pope_yesno/rows.json")
    report, preds = {"arms": {}}, {}
    for arm in ARMS:
        pred, src, stats = {}, Counter(), []
        for sid, path in collect(arm).items():
            r = read_json(path)
            n = normalize_output(r["raw_response_ids"], tok, r["eos_ids"])
            p, s = parse(n["text"], rule) if r["status"] == "success" else (None, "budget_exhausted")
            pred[sid], src[s] = p, src[s] + 1
            stats.append((r["status"], r["finish_reason"], r["decode_iterations"], r["generation_seconds"], n["first_eos_position"]))
        preds[arm] = pred
        report["arms"][arm] = {"metrics": metrics(rows, pred), "answer_sources": dict(src),
                               "statuses": dict(Counter(s[0] for s in stats)), "finish": dict(Counter(s[1] for s in stats)),
                               "mean_iterations": statistics.mean(s[2] for s in stats),
                               "mean_seconds": statistics.mean(s[3] for s in stats),
                               "eos_at_last_position": sum(s[4] == 63 for s in stats)}
    report["paired_cdapc_minus_native_ci95"] = bootstrap(rows, preds["cdapc"], preds["native"])
    atomic_json(ROOT / "reports/result.json", report)
    lines = ["# LLaDA-V POPE-COCO think，64/64（5127输入、9000行）", "",
             "提示：/think＋Final answer后缀（pope_yesno_reasoning_v1）；seed 20261105。",
             "Native：长度64、64步、块长64；CDAPC：r70/W16/K4，L64/I64。答案优先取Final answer，失败与未解析计错。", "",
             "| 方法 | Random | Popular | Adversarial | 平均 Acc / F1 | Yes比例 | 平均秒 | 平均轮数 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, b in report["arms"].items():
        m = b["metrics"]
        lines.append(f"| {arm} | " + " | ".join(f"{100*m[s]['accuracy']:.2f}/{100*m[s]['f1']:.2f}" for s in SPLITS)
                     + f" | **{100*m['mean']['accuracy']:.2f} / {100*m['mean']['f1']:.2f}** | {100*m['mean']['yes_ratio']:.1f}% | "
                     f"{b['mean_seconds']:.1f} | {b['mean_iterations']:.1f} |")
    ci = report["paired_cdapc_minus_native_ci95"]
    lines += ["", f"CDAPC−Native 图像配对95%区间：Acc [{100*ci['accuracy'][0]:+.2f}, {100*ci['accuracy'][1]:+.2f}]，"
              f"F1 [{100*ci['f1'][0]:+.2f}, {100*ci['f1'][1]:+.2f}] 个百分点。", ""]
    lines += [f"- {a}：状态 {b['statuses']}；答案来源 {b['answer_sources']}" for a, b in report["arms"].items()]
    (ROOT / "reports/summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main():
    os.chdir(ROOT)
    if "--score" in sys.argv:
        score()
        return
    state, workers = {"status": "starting", "controller_pid": os.getpid(), "arms": {}}, {}
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        with (ROOT / "logs/controller.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            state["status"] = "generating"
            for arm in ARMS:
                generate(arm, state, workers)
            state["status"] = "scoring"
            save(state)
            score()
            state["status"] = "complete"
            save(state)
    except BaseException as error:
        state.update(status="needs_attention", error=repr(error))
        save(state)
        raise
    finally:
        for p in workers.values():
            if p.poll() is None:
                p.terminate()


if __name__ == "__main__":
    main()
