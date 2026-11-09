"""Score only the frozen pilot rows and audit saved commit timing."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import fcntl
from pathlib import Path
import re
import statistics

import numpy as np

from cdapc_lab.evaluation.llm_judge import judge_key
from cdapc_lab.evaluation.pope_llm_judge import judge_request, request_for, spec
from cdapc_lab.experiments.score import score_run, pope_metrics
from cdapc_lab.io.artifacts import atomic_json, digest, read_json, safe_id

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/k_commit_pilot"
JUDGE = ROOT / "reports/pope_think_llm_judge"


def mechanism(pred):
    batches = defaultdict(list)
    for entry in pred["commit_log"]:
        batches[entry["iteration"]].append(entry)
    assert set(batches) == set(range(pred["decode_iterations"]))
    known_eos = None
    tail_only = 0
    sizes = []
    for iteration in sorted(batches):
        entries = batches[iteration]
        sizes.append(len(entries))
        tail_only += int(known_eos is not None and all(
            entry["position"] > known_eos for entry in entries))
        for entry in entries:
            if entry["written_id"] in pred["eos_ids"]:
                known_eos = min(known_eos, entry["position"]) if known_eos is not None else entry["position"]
    assert sum(sizes) == pred["metrics"]["committed"]
    return {
        "iterations": pred["decode_iterations"], "seconds": pred["generation_seconds"],
        "output_tokens": pred["emitted_tokens_before_eos"], "commits": sum(sizes),
        "batch_histogram": dict(Counter(sizes)), "tail_only_iterations": tail_only,
        "finish_reason": pred["finish_reason"], "fallback_iterations": pred["metrics"]["fallback_iterations"],
        "switches": pred["metrics"]["actual_switches"],
    }


def paired_accuracy(rows, first, second, repeats=2000, seed=42):
    groups = defaultdict(list)
    for row in rows:
        sid = row["input_key"]
        groups[row["image"]].append(second[sid]["judge_correct"] - first[sid]["judge_correct"])
    image_sums = np.array([sum(v) for v in groups.values()], dtype=float)
    image_sizes = np.array([len(v) for v in groups.values()], dtype=float)
    indices = np.random.default_rng(seed).integers(0, len(groups), (repeats, len(groups)))
    draws = image_sums[indices].sum(axis=1) / image_sizes[indices].sum(axis=1)
    return {"delta": float(image_sums.sum() / image_sizes.sum()),
            "ci95": [float(v) for v in np.quantile(draws, [.025, .975])],
            "unit": "paired image cluster", "images": len(groups), "repeats": repeats, "seed": seed}


def collect(state, include_controls=True):
    rows = read_json(OUT / "rows.json")
    ids = {r["input_key"] for r in rows}
    questions = {r["input_key"]: r["question"] for r in rows}
    baseline = read_json(ROOT / "reports/pope_gqa_reasoning_r70.json")
    locations = {"16": [(item["run"], item["evaluation_fingerprint"])
                         for item in baseline["cdapc"]["per_run"]]}
    for k in (("4", "2") if include_controls else ()):
        locations[k] = []
        for item in state["conditions"][k]["shards"].values():
            scored = score_run(ROOT / item["run"])
            locations[k].append((item["run"], scored["evaluation_fingerprint"]))
    records = {}
    requests = {}
    for k, runs in locations.items():
        records[k] = {}
        for run, fingerprint in runs:
            score_root = ROOT / run / "scores" / fingerprint
            for sid in sorted(ids):
                score_path = score_root / (safe_id(sid) + ".json")
                if not score_path.exists():
                    continue
                score = read_json(score_path)
                pred = read_json(ROOT / run / "predictions" / score_path.name)
                assert sid not in records[k] and pred["status"] == "success"
                assert score["generation_status"] == "success"
                text = score["normalized"]["text"]
                request = request_for(questions[sid], text)
                sha = digest(request)
                requests[sha] = request
                record = {
                    "sample_id": sid, "question": questions[sid], "gold": score["gold"], "text": text,
                    "strict_prediction": score["parsed"]["prediction"],
                    "strict_correct": score["strict_accuracy"], "request_sha256": sha,
                    "score_path": str(score_path.relative_to(ROOT)),
                    "generation_fingerprint": pred["generation_fingerprint"],
                    "prompt_sha256": digest(pred["prompt_ids"]), "sample_seed": pred["sample_seed"],
                    "input_fingerprint": pred["input_fingerprint"],
                    "empty_shell": not re.sub(r"</?think>", "", text).strip(),
                    **mechanism(pred),
                }
                assert max(record["batch_histogram"]) <= int(k)
                if k != "16":
                    for field in ("prompt_sha256", "input_fingerprint", "sample_seed", "gold"):
                        assert record[field] == records["16"][sid][field], (k, sid, field)
                records[k][sid] = record
        assert set(records[k]) == ids
    return rows, records, requests


def evaluate(state):
    rows, records, requests = collect(state)
    cache = JUDGE / "cache"
    # Serialize access with the existing Judge controller; GPU lock is already released.
    with (JUDGE / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        key = judge_key()
        if not key:
            raise RuntimeError("DeepSeek API credential unavailable")
        results = {}
        for attempt in range(3):
            pending = [sha for sha in requests if results.get(sha, {}).get("status") != "success"]
            with ThreadPoolExecutor(max_workers=4) as pool:
                values = list(pool.map(lambda sha: judge_request(requests[sha], cache, key), pending))
            results.update(zip(pending, values))
            if all(result["status"] == "success" for result in results.values()):
                break
        if any(result["status"] != "success" for result in results.values()):
            raise RuntimeError("Pilot Judge errors remain; cached results retained for resume")
    report = {
        "protocol": read_json(OUT / "protocol.json"), "judge_spec": spec(),
        "scope": "same frozen 30 images and 540 rows for all three K values",
        "limitations": [
            "Small pilot; image-cluster paired intervals can be wide",
            "K16 outputs/timings reused from an earlier run; host runtime may differ, so timing is descriptive",
            "No full-dataset accuracy claims; no changes to r/window/prompt/iteration budget",
            "Pointwise exploratory confidence intervals; no multiple-comparison adjustment",
        ], "methods": {}, "per_input": records,
    }
    baseline = read_json(ROOT / "reports/pope_gqa_reasoning_r70.json")
    method_runs = {"16": [item["run"] for item in baseline["cdapc"]["per_run"]]}
    method_runs.update({k: [item["run"] for item in stage["shards"].values()]
                       for k, stage in state["conditions"].items()})
    report["generation_environments"] = {
        k: [{"run": run, "environment": read_json(ROOT / run / "run_manifest.json")["environment"]}
            for run in runs] for k, runs in method_runs.items()}
    for k, items in records.items():
        for record in items.values():
            parsed = results[record["request_sha256"]]["parsed"]
            record.update(judge_prediction=parsed["answer"], judge_status=parsed["status"],
                          judge_evidence=parsed["evidence"],
                          judge_correct=int(parsed["answer"] is not None and parsed["answer"] == record["gold"]))
        splits = {}
        for split in ("random", "popular", "adversarial"):
            selected = [row for row in rows if row["split"] == split]
            assert len(selected) == 180
            splits[split] = {
                "n": len(selected),
                "judge": pope_metrics((items[row["input_key"]]["judge_prediction"], row["label"])
                                      for row in selected),
                "strict": pope_metrics((items[row["input_key"]]["strict_prediction"], row["label"])
                                       for row in selected),
            }
        n = len(items)
        iterations = sum(r["iterations"] for r in items.values())
        commits = sum(r["commits"] for r in items.values())
        report["methods"][k] = {
            "unique_inputs": n, "evaluation_rows": len(rows), "splits": splits,
            "judge_accuracy": statistics.mean(s["judge"]["accuracy"] for s in splits.values()),
            "judge_f1": (statistics.mean(s["judge"]["f1"] for s in splits.values())
                         if all(s["judge"]["f1"] is not None for s in splits.values()) else None),
            "strict_accuracy": statistics.mean(s["strict"]["accuracy"] for s in splits.values()),
            "mean_seconds": statistics.mean(r["seconds"] for r in items.values()),
            "mean_iterations": iterations / n, "commits_per_iteration": commits / iterations,
            "mean_output_tokens": statistics.mean(r["output_tokens"] for r in items.values()),
            "tail_only_iteration_rate": sum(r["tail_only_iterations"] for r in items.values()) / iterations,
            "empty_shell_rate": sum(r["empty_shell"] for r in items.values()) / n,
            "judge_missing_rate": sum(r["judge_status"] == "missing" for r in items.values()) / n,
            "judge_ambiguous_rate": sum(r["judge_status"] == "ambiguous" for r in items.values()) / n,
            "judge_statuses_unique": dict(Counter(r["judge_status"] for r in items.values())),
            "finish_reasons": dict(Counter(r["finish_reason"] for r in items.values())),
        }
    report["paired_judge_accuracy"] = {
        "4_minus_16": paired_accuracy(rows, records["16"], records["4"]),
        "2_minus_16": paired_accuracy(rows, records["16"], records["2"]),
        "2_minus_4": paired_accuracy(rows, records["4"], records["2"]),
    }
    atomic_json(OUT / "results.json", report)
    lines = [
        "# GQA think r70：K=16/4/2 配对小规模试验", "",
        f"固定 30 张随机图像、{len(records['16'])} 个唯一输入、540 条评测记录。"
        "r=0.7、W=16、L=128、迭代上限128，其余设置一致。K16 复用相同题目的历史输出。",
        "", "| K | LLM Judge Acc % | LLM Judge F1 % | Strict Acc % | 空壳 % | Judge缺答 % | Judge冲突 % |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for k, r in report["methods"].items():
        f1 = f"{100*r['judge_f1']:.2f}" if r["judge_f1"] is not None else "—"
        lines.append(f"| {k} | {100*r['judge_accuracy']:.2f} | {f1} | "
                     f"{100*r['strict_accuracy']:.2f} | {100*r['empty_shell_rate']:.2f} | "
                     f"{100*r['judge_missing_rate']:.2f} | {100*r['judge_ambiguous_rate']:.2f} |")
    lines += ["", "Acc/F1 按评测记录计算；空壳/缺答/冲突按唯一输入计算。",
              "", "| K | 秒/条 | 步数/条 | 提交/轮 | EOS尾部轮次 % |",
              "|---|---:|---:|---:|---:|"]
    for k, r in report["methods"].items():
        lines.append(f"| {k} | {r['mean_seconds']:.2f} | {r['mean_iterations']:.2f} | "
                     f"{r['commits_per_iteration']:.2f} | "
                     f"{100*r['tail_only_iteration_rate']:.2f} |")
    lines += ["", "配对 Accuracy 差值以图像为重采样单位；以下单位为百分点。", ""]
    for name, result in report["paired_judge_accuracy"].items():
        lines.append(f"- {name}: {100*result['delta']:.2f}; "
                     f"95% CI [{100*result['ci95'][0]:.2f}, {100*result['ci95'][1]:.2f}]")
    lines += ["", "本轮为探索性小样本，不与全量 9,000 条分数直接比较。"
              "逐题输入校验、完整输出、提交行为与 Judge 缓存键均保存在 results.json。",
              "", "K16 复用历史结果；各组运行环境保存在 JSON 的 generation_environments。"
              "主机环境差异需随质量对照说明，历史耗时仅作参考。"]
    (OUT / "results.md").write_text("\n".join(lines) + "\n")
