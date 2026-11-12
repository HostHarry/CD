"""HallusionBench: CDAPC (r70/W16/K4/L256/I128, EOS-prefix) and Native (256/128/32, safe EOS), direct + think.

Text-only rows have no visual branch; CDAPC arms use the Native output with the same prompt for them
(the baseline's native_without_visual_intervention policy). Scoring uses unchanged VLMEvalKit functions.
"""
import ast
from collections import Counter, defaultdict
from datetime import datetime
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PREFIX = ROOT.parent / "cdapc-pope-coco-prefix-no-final"
sys.path.insert(0, str(ROOT / "src"))
from cdapc_lab.config import resolve_config
from cdapc_lab.data.records import build_question, uses_model_reasoning_instruction
from cdapc_lab.evaluation.normalize import normalize_output
from cdapc_lab.experiments.infer import preflight
from cdapc_lab.io.artifacts import atomic_json, digest, file_hash, generation_projection, read_json
from cdapc_lab.models.mmada import OFFICIAL_THINKING_INSTRUCTION

ARMS = {"cdapc_direct": ("cdapc", "direct", "image"), "native_direct": ("native", "direct", "all"),
        "cdapc_think": ("cdapc", "think", "image"), "native_think": ("native", "think", "all")}
PROMPTS = {"direct": "hallusion_original_question_v1", "think": "hallusion_think_v1"}
SHARED_IDENTICAL = ("decoding", "io", "types.py", "experiments/infer.py")
SCORER = {"yorn.py": "692d72ee0c8fe64c7dd93b466f1b47c432303d8fba57503c56aae770b67ef59e",
          "misc.py": "407ec5eaaf84227368365c6c7c93725166becfe755ed6a24eb93260d65464258"}


def save(state):
    state["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    atomic_json(ROOT / "reports/status.json", state)


def py_files(name, root):
    base = root / "src/cdapc_lab" / name
    return [base] if name.endswith(".py") else sorted(base.rglob("*.py"))


def prepare():
    for name in SHARED_IDENTICAL:
        for path in py_files(name, ROOT):
            assert file_hash(path) == file_hash(PREFIX / path.relative_to(ROOT)), str(path)
    changed = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "src").rglob("*.py")
                     if not (PREFIX / p.relative_to(ROOT)).exists() or file_hash(p) != file_hash(PREFIX / p.relative_to(ROOT)))
    rows = read_json(ROOT / "data/hallusion/rows.json")
    ids = [r["index"] for r in rows]
    image_ids = {r["index"] for r in rows if r["input_mode"] == "image"}
    result = {}
    for arm, (method, prompt, scope) in ARMS.items():
        seen, shards = set(), {}
        for k in range(3):
            cfg = resolve_config(ROOT / f"configs/{arm}_shard{k}.yaml")
            assert cfg["method"] == method and cfg["protocol"]["prompt_id"] == PROMPTS[prompt] and cfg["run"]["seed"] == 42
            if method == "cdapc":
                cd = cfg["cdapc"]
                assert (cd["r_gate"]["threshold"], cd["window"]["width"], cd["max_commit_per_iteration"],
                        cd["iteration_limit"], cfg["protocol"]["canvas_length"]) == (0.7, 16, 4, 128, 256)
            else:
                n = cfg["native"]
                assert (n["steps"], n["block_length"], n["early_eos"], cfg["protocol"]["canvas_length"]) == (128, 64, "safe_eos_v1", 256)
            samples, _, provenance = preflight(cfg)
            for sample in samples:
                assert sample.sample_id not in seen
                seen.add(sample.sample_id)
                assert uses_model_reasoning_instruction(sample, PROMPTS[prompt]) == (prompt == "think")
                assert build_question(sample, PROMPTS[prompt]) == sample.question
            shards[str(k)] = {"config": f"configs/{arm}_shard{k}.yaml", "expected": len(samples), **provenance}
        assert seen == (image_ids if scope == "image" else set(ids)), arm
        result[arm] = {"shards": shards, "inputs": len(seen)}
    atomic_json(ROOT / "reports/preflight.json", {
        "arms": result, "rows": len(rows), "image_rows": len(image_ids), "text_only_rows": len(rows) - len(image_ids),
        "source_changes_vs_prefix_workspace": changed,
        "shared_unchanged_vs_prefix_workspace": list(SHARED_IDENTICAL),
        "baseline_commit": (ROOT / "baseline/COMMIT").read_text().strip(),
        "native_schedule": "256/128/64 by request; CD baseline Native/VCD/VIGOR use 256/128/32",
        "smoke": "At block 32, local Native direct reproduced baseline Native text, prompt-ID SHA and executed steps "
                 "on 6 rows (validates data/prompt/image pipeline); the full run uses block 64"})
    snapshot = {str(p.relative_to(ROOT)): file_hash(p)
                for folder in ("src", "scripts", "configs", "manifests", "evaluator", "upstream/hallusion", "baseline")
                for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in str(p) and "smoke" not in p.name}
    for name in ("reports/preflight.json", "upstream.lock.json", "data/hallusion/samples.json",
                 "data/hallusion/gold.json", "data/hallusion/rows.json", "data/hallusion/provenance.json"):
        snapshot[name] = file_hash(ROOT / name)
    atomic_json(ROOT / "reports/source_snapshot.json", snapshot)
    print("HALLUSION_PREFLIGHT_OK", {a: v["inputs"] for a, v in result.items()}, flush=True)


def collect(arm):
    result = {}
    for path in (ROOT / "runs" / arm).glob("*/predictions/*.json"):
        raw = read_json(path)
        assert raw["sample_id"] not in result
        assert raw["status"] in ("success", "budget_exhausted"), str(path)
        result[raw["sample_id"]] = {"path": str(path), "sha256": file_hash(path), "status": raw["status"]}
    return result


def generate_arm(arm, prepared, state, workers):
    present = collect(arm)
    expected = prepared["arms"][arm]["inputs"]
    (ROOT / "runs" / arm).mkdir(parents=True, exist_ok=True)
    previous = set((ROOT / "runs" / arm).iterdir())
    stage = {"status": "generating", "preserved": len(present), "expected": expected, "shards": {}}
    state["arms"][arm] = stage
    state["current"] = arm
    active = {}
    for k, ref in prepared["arms"][arm]["shards"].items():
        cfg = yaml.safe_load((ROOT / ref["config"]).read_text())
        ids = [sid for sid in read_json(ROOT / cfg["run"]["sample_ids_file"]) if sid not in present]
        if not ids:
            continue
        wave = digest(ids)[:12]
        ids_path = ROOT / f"manifests/{arm}_{wave}_{k}.json"
        atomic_json(ids_path, ids)
        cfg["run"].update(name=f"{arm}_{wave}_shard{k}", sample_ids_file=str(ids_path))
        path = ROOT / f"configs/{arm}_{wave}_{k}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        resolved = resolve_config(path)
        assert generation_projection(resolved) == generation_projection(resolve_config(ROOT / ref["config"]))
        samples, _, provenance = preflight(resolved)
        for field in ("generation_source_hash", "environment", "assets"):
            assert provenance[field] == ref[field], field
        active[k] = {"config": str(path), "expected": len(samples), **provenance}
        with (ROOT / f"logs/{arm}_{wave}_{k}.log").open("a") as log:
            workers[k] = subprocess.Popen([sys.executable, "-m", "cdapc_lab", "infer", "--config", str(path)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": k,
                     "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"})
        stage["shards"][k] = {"worker_pid": workers[k].pid, "expected": len(samples)}
    atomic_json(ROOT / f"reports/{arm}_active_preflight.json", active)
    while workers:
        for k, process in list(workers.items()):
            item = stage["shards"][k]
            matches = [p.parent for p in (ROOT / "runs" / arm).glob("*/run_manifest.json")
                       if p.parent not in previous and read_json(p)["generation_fingerprint"] == active[k]["generation_fingerprint"]]
            if matches:
                assert len(matches) == 1
                item.update(run=str(matches[0]), generated=len(list((matches[0] / "predictions").glob("*.json"))))
            item["returncode"] = process.poll()
            if item["returncode"] not in (None, 0):
                raise RuntimeError(f"{arm} shard {k} failed")
        stage["generated"] = len(present) + sum(v.get("generated", 0) for v in stage["shards"].values())
        save(state)
        if all(p.poll() is not None for p in workers.values()):
            workers.clear()
            break
        time.sleep(20)
    records = collect(arm)
    assert len(records) == expected, (arm, len(records), expected)
    atomic_json(ROOT / f"reports/{arm}_inventory.json", records)
    stage["status"] = "generated"
    save(state)


def official_functions():
    namespace = {"np": np, "pd": pd, "defaultdict": defaultdict, "load": lambda frame: frame.copy()}
    for filename, names in (("misc.py", {"process_punctuation"}), ("yorn.py", {"YOrN_Extraction", "Hallusion_rating"})):
        path = ROOT / "evaluator" / filename
        assert file_hash(path) == SCORER[filename], filename
        nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
        assert {n.name for n in nodes} == names
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["YOrN_Extraction"], namespace["Hallusion_rating"]


def rate(rows, predictions, extract, rating):
    frame = pd.DataFrame([{**r, "prediction": predictions[r["index"]]} for r in rows])
    frame["extracted"] = [extract(p) for p in frame["prediction"]]
    frame["score"] = frame["extracted"] == frame["answer"]
    table = rating(frame)
    overall = table[table["split"] == "Overall"].iloc[0]
    return {"rows": len(frame), "correct": int(frame["score"].sum()),
            "aAcc": float(overall["aAcc"]), "fAcc": float(overall["fAcc"]), "qAcc": float(overall["qAcc"]),
            "Avg": float(overall["Avg"]), "unknown": int((frame["extracted"] == "Unknown").sum()),
            "yes_share": float((frame["extracted"] == "Yes").mean() * 100),
            "breakdown": table.sort_values("split").to_dict(orient="records")}, dict(zip(frame["index"], frame["score"]))


def set_bootstrap(rows, scores_a, scores_b, repeats=2000, seed=42):
    """Cluster bootstrap over HallusionBench question sets (l2-category + set_id); delta = A - B, percent."""
    sets = defaultdict(list)
    for r in rows:
        sets[f"{r['l2-category']}_{r['index'].split('_')[3]}"].append(r)
    keys = sorted(sets)
    stats = np.zeros((2, len(keys), 6))
    for j, scores in enumerate((scores_a, scores_b)):
        for i, key in enumerate(keys):
            figs, ques = defaultdict(list), defaultdict(list)
            for r in sets[key]:
                parts = r["index"].split("_")
                figs[parts[4]].append(scores[r["index"]])
                ques[parts[5]].append(scores[r["index"]])
            stats[j, i] = [len(sets[key]), sum(scores[r["index"]] for r in sets[key]),
                           len(figs), sum(all(v) for v in figs.values()), len(ques), sum(all(v) for v in ques.values())]
    def metrics(total):
        a, f, q = 100 * total[..., 1] / total[..., 0], 100 * total[..., 3] / total[..., 2], 100 * total[..., 5] / total[..., 4]
        return np.stack([a, f, q, (a + f + q) / 3], -1)
    point = metrics(stats.sum(1))
    idx = np.random.default_rng(seed).integers(0, len(keys), size=(repeats, len(keys)))
    draws = metrics(stats[:, idx].sum(2))
    delta = draws[0] - draws[1]
    return {name: {"a": float(point[0, i]), "b": float(point[1, i]), "delta": float(point[0, i] - point[1, i]),
                   "ci95": np.quantile(delta[:, i], [0.025, 0.975]).tolist()}
            for i, name in enumerate(("aAcc", "fAcc", "qAcc", "Avg"))} | {
        "unit": "question set (l2-category + set_id)", "sets": len(keys), "repeats": repeats, "seed": seed}


def score():
    from transformers import AutoTokenizer
    rows = read_json(ROOT / "data/hallusion/rows.json")
    by_index = {r["index"]: r for r in rows}
    extract, rating = official_functions()
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/mmada", local_files_only=True)
    texts, info = {}, {}
    for arm in ARMS:
        texts[arm], info[arm] = {}, {}
        for sid, item in read_json(ROOT / f"reports/{arm}_inventory.json").items():
            raw = read_json(item["path"])
            assert file_hash(item["path"]) == item["sha256"]
            question = by_index[sid]["question"]
            content = (OFFICIAL_THINKING_INSTRUCTION + "\n" if ARMS[arm][1] == "think" else "") + question
            expected = tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True)
            assert raw["prompt_ids"][0][-len(expected):] == expected, (arm, sid)
            assert len(raw["prompt_ids"][0]) == len(expected) + (1027 if by_index[sid]["input_mode"] == "image" else 0)
            texts[arm][sid] = normalize_output(raw["raw_response_ids"], tokenizer, raw["eos_ids"])["text"]
            info[arm][sid] = {"status": raw["status"], "finish": raw["finish_reason"], "seconds": raw["generation_seconds"],
                              "iterations": raw["decode_iterations"], "forwards": raw["cost"]["model_forward_calls"],
                              "source_path": item["path"], "source_sha256": item["sha256"],
                              "decoder": ARMS[arm][0]}
    composed = {}
    for arm, (method, prompt, scope) in ARMS.items():
        composed[arm] = {}
        for r in rows:
            sid = r["index"]
            source = arm if sid in texts[arm] else f"native_{prompt}"
            assert method == "cdapc" and r["input_mode"] == "text_only" if source != arm else True
            composed[arm][sid] = {"text": texts[source][sid], **info[source][sid], "source_arm": source}
    baseline = {m: {a["index"]: a["text"] for a in read_json(ROOT / f"baseline/{m}/answers.json")} for m in ("native", "vcd", "vigor")}
    reproduction = {"identical_text": sum(composed["native_direct"][s]["text"] == baseline["native"][s] for s in by_index),
                    "rows": len(rows)}
    base_prompts = {}
    for line in gzip.open(ROOT / "baseline/native/predictions-000.jsonl.gz", "rt"):
        payload = json.loads(line)["payload"]
        base_prompts.setdefault(payload["input"]["prompt_ids_sha256"], payload["executed_steps"])
    matched = 0
    for sid, item in info["native_direct"].items():
        ids = read_json(item["source_path"])["prompt_ids"][0]
        matched += hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest() in base_prompts
    reproduction["prompt_sha_matched"] = matched
    report = {"status": "complete", "rows": len(rows), "image_rows": sum(r["input_mode"] == "image" for r in rows),
              "text_only_rows": sum(r["input_mode"] == "text_only" for r in rows), "arms": {},
              "scoring": "VLMEvalKit YOrN_Extraction/Hallusion_rating (pinned 3c19214); Unknown incorrect; no API judge",
              "think_answer_scope": "text after last </think>; whole visible text when unclosed",
              "text_only_policy": "CDAPC arms use the Native output with the same prompt (no visual branch exists)",
              "native_direct_vs_baseline_native": {**reproduction,
                 "note": "local Native uses block 64, baseline Native block 32; text identity is diagnostic only"}}
    scores, predictions = {}, {}
    for arm, (method, prompt, _) in ARMS.items():
        pred = {}
        for sid, v in composed[arm].items():
            text = v["text"] if v["status"] == "success" else ""
            pred[sid] = text.rsplit("</think>", 1)[-1].strip() if prompt == "think" and "</think>" in text else text
        predictions[arm] = pred
        metrics, scores[arm] = rate(rows, pred, extract, rating)
        own = [v for v in composed[arm].values() if v["source_arm"] == arm]
        block = {"metrics": metrics,
                 "by_input_mode": {m: rate([r for r in rows if r["input_mode"] == m], pred, extract, rating)[0]
                                   for m in ("image", "text_only")},
                 "generation_status": dict(Counter(v["status"] for v in composed[arm].values())),
                 "finish_reasons": dict(Counter(v["finish"] for v in own)),
                 "own_generated_rows": len(own),
                 "mean_seconds_own_rows": statistics.mean(v["seconds"] for v in own),
                 "mean_iterations_own_rows": statistics.mean(v["iterations"] for v in own)}
        if prompt == "think":
            block["closed_think"] = sum("</think>" in v["text"] for v in composed[arm].values())
            block["closed_empty"] = sum("</think>" in v["text"] and not v["text"].rsplit("</think>", 1)[-1].strip()
                                        for v in composed[arm].values())
            block["raw_full_text_diagnostic"] = rate(rows, {s: (v["text"] if v["status"] == "success" else "")
                                                            for s, v in composed[arm].items()}, extract, rating)[0]
        report["arms"][arm] = block
    for m in ("native", "vcd", "vigor"):
        report.setdefault("baseline", {})[m], scores[f"baseline_{m}"] = rate(rows, baseline[m], extract, rating)
    published = read_json(ROOT / "baseline/metrics.json")["methods"]
    for m in ("native", "vcd", "vigor"):
        assert abs(report["baseline"][m]["aAcc"] - published[m]["accuracy_percent"]) < 1e-9, m
    image_rows = [r for r in rows if r["input_mode"] == "image"]
    pairs = {"cdapc_direct - native_direct": ("cdapc_direct", "native_direct"),
             "cdapc_think - native_think": ("cdapc_think", "native_think"),
             "cdapc_think - cdapc_direct": ("cdapc_think", "cdapc_direct"),
             "native_think - native_direct": ("native_think", "native_direct"),
             "native_direct(B64) - baseline_native(B32)": ("native_direct", "baseline_native"),
             "cdapc_direct - baseline_vcd": ("cdapc_direct", "baseline_vcd"),
             "cdapc_direct - baseline_vigor": ("cdapc_direct", "baseline_vigor")}
    report["paired"] = {name: {"all_rows": set_bootstrap(rows, scores[a], scores[b]),
                               "image_rows": set_bootstrap(image_rows, scores[a], scores[b])}
                        for name, (a, b) in pairs.items()}
    for name, (a, b) in pairs.items():
        for metric in ("aAcc", "fAcc", "qAcc"):
            ref = report["arms"][a]["metrics"] if a in report["arms"] else report["baseline"][a[9:]]
            assert abs(report["paired"][name]["all_rows"][metric]["a"] - ref[metric]) < 1e-9, (name, metric)
    report["per_row"] = {arm: {sid: {**v, "prediction_text_scored": predictions[arm][sid], "correct": bool(scores[arm][sid])}
                               for sid, v in composed[arm].items()} for arm in ARMS}
    report["source_snapshot_sha256"] = file_hash(ROOT / "reports/source_snapshot.json")
    atomic_json(ROOT / "reports/result.json", report)
    write_summary(report)
    paths = [ROOT / "reports" / n for n in ("result.json", "summary.md", "preflight.json", "source_snapshot.json",
                                            *[f"{a}_inventory.json" for a in ARMS])]
    (ROOT / "reports/final.sha256").write_text("".join(f"{file_hash(p)}  {p.relative_to(ROOT)}\n" for p in paths))
    return report


def write_summary(report):
    label = {"cdapc_direct": "CDAPC 直答", "native_direct": "Native 直答", "cdapc_think": "CDAPC think", "native_think": "Native think"}
    lines = ["# HallusionBench：CDAPC 与 Native，直答与 think", "",
             "全部1129题（951图像题、178纯文本题）。直答为TSV原题；think为MMaDA官方思考指令＋原题。",
             "CDAPC：r70/W16/K4/L256/I128，快速EOS前缀；本地Native：256/128/64，safe EOS；seed 42。",
             "CD基线Native/VCD/VIGOR为256/128/32（块长不同），仅直答。",
             "纯文本题没有视觉分支，CDAPC两组沿用同提示词的Native输出（与CD基线VCD/VIGOR相同的处理）。",
             "评分：VLMEvalKit原版YOrN_Extraction/Hallusion_rating；think取最后一个`</think>`之后的回答，未闭合用全文；Unknown计错。", "",
             "| 组别 | aAcc | fAcc | qAcc | Avg | 图像题aAcc | Unknown | Yes比例 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, block in report["arms"].items():
        m, i = block["metrics"], block["by_input_mode"]["image"]
        lines.append(f"| {label[arm]} | {m['aAcc']:.2f} | {m['fAcc']:.2f} | {m['qAcc']:.2f} | {m['Avg']:.2f} | "
                     f"{i['aAcc']:.2f} | {m['unknown']} | {m['yes_share']:.1f}% |")
    for m in ("native", "vcd", "vigor"):
        b = report["baseline"][m]
        lines.append(f"| CD基线 {m}（直答，B32） | {b['aAcc']:.2f} | {b['fAcc']:.2f} | {b['qAcc']:.2f} | {b['Avg']:.2f} | — | {b['unknown']} | {b['yes_share']:.1f}% |")
    rep = report["native_direct_vs_baseline_native"]
    lines += ["", f"本地Native直答（B64）与CD基线Native（B32）：prompt哈希一致 {rep['prompt_sha_matched']}/{rep['rows']}；"
              f"逐题文本相同 {rep['identical_text']}/{rep['rows']}（块长不同，仅作诊断）。",
              "", "配对差值（题组整组bootstrap，2000次，seed 42；单位：百分点）：", "",
              "| 比较 | 范围 | ΔaAcc [95% CI] | ΔfAcc [95% CI] | ΔqAcc [95% CI] |", "|---|---|---:|---:|---:|"]
    for name, scopes in report["paired"].items():
        for scope, v in scopes.items():
            cells = " | ".join(f"{v[k]['delta']:+.2f} [{v[k]['ci95'][0]:+.2f}, {v[k]['ci95'][1]:+.2f}]" for k in ("aAcc", "fAcc", "qAcc"))
            lines.append(f"| {name} | {'全部' if scope == 'all_rows' else '图像题'} | {cells} |")
    lines += [""]
    for arm, block in report["arms"].items():
        extra = f"；`</think>`闭合 {block['closed_think']}/1129，闭合后为空 {block['closed_empty']}" if "closed_think" in block else ""
        lines.append(f"- {label[arm]}：生成状态 {block['generation_status']}；本组实际生成 {block['own_generated_rows']} 题，"
                     f"平均 {block['mean_seconds_own_rows']:.2f} 秒、{block['mean_iterations_own_rows']:.2f} 轮{extra}。")
    (ROOT / "reports/summary.md").write_text("\n".join(lines) + "\n")


def main():
    os.chdir(ROOT)
    if "--prepare" in sys.argv:
        prepare()
        return
    if "--score" in sys.argv:
        score()
        return
    state = {"status": "starting", "controller_pid": os.getpid(), "arms": {}}
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
                state["status"] = "generating"
                for arm in ARMS:
                    generate_arm(arm, prepared, state, workers)
            state["status"] = "scoring"
            save(state)
            report = score()
            state.update(status="complete", report="reports/summary.md",
                         aAcc={a: b["metrics"]["aAcc"] for a, b in report["arms"].items()})
            save(state)
            print("HALLUSION_COMPLETE", state["aAcc"], flush=True)
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
