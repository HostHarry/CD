"""DeepSeek text judge for POPE think 256/128/64, reusing the cloud CD-APC
protocol pope_text_llm_judge_v1 verbatim (HostHarry/CD cdapc/source).

The API receives only the plain POPE question and the complete decoded
response; no image, no gold. Scoring against gold is local.
Run from the project root: .venv/bin/python judge_pope_think.py coco
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import statistics
import sys

CDAPC_SRC = Path("/root/autodl-tmp/mmada-records/cdapc/source/src")
JUDGE_FILE_SHA256 = "1ab1a8ef942edc347f87f8165df6d870cff67fb040b976006fff2a03e846fa57"
sys.path.insert(0, str(CDAPC_SRC))
from cdapc_lab.evaluation.pope_llm_judge import judge_request, request_for, spec  # noqa: E402
from cdapc_lab.experiments.score import pope_metrics  # noqa: E402
from cdapc_lab.io.artifacts import atomic_json, digest, file_hash  # noqa: E402

RUN = Path(__file__).resolve().parent / "results/pope_think_256_128_64"
OUT = RUN / "judge"
METHODS = ("native", "vcd", "vigor")
SPLITS = ("random", "popular", "adversarial")
INSTRUCTION_END = "</think> answer here\n"


def plain_question(prompt):
    head, sep, q0 = prompt.partition(INSTRUCTION_END)
    assert sep and head.startswith("You should first think"), prompt[:80]
    return q0


def collect(ds):
    samples = [json.loads(line) for line in (RUN / "inputs" / f"{ds}_think.jsonl").open()]
    questions = {(s["split"], s["id"]): plain_question(s["question"]) for s in samples}
    rows, requests = {}, {}
    for method in METHODS:
        answers = json.loads((RUN / ds / f"{method}_answers.json").read_text())
        assert len(answers) == len(samples) == 9000
        out = []
        for a in answers:
            request = request_for(questions[(a["split"], a["id"])], a["text"])
            sha = digest(request)
            requests[sha] = request
            out.append({"id": a["id"], "split": a["split"], "label": a["label"],
                        "image_sha256": a["image_sha256"], "request_sha256": sha,
                        "text_sha256": hashlib.sha256(a["text"].encode()).hexdigest()})
        rows[method] = out
    return rows, requests


def paired_bootstrap(left, right, repeats=2000, seed=42):
    import numpy as np
    by_image = {}
    for a, b in zip(left, right):
        assert (a["id"], a["split"]) == (b["id"], b["split"])
        by_image.setdefault(a["image_sha256"], []).append((a["correct"], b["correct"]))
    groups = list(by_image.values())
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(repeats):
        pairs = [p for i in rng.integers(0, len(groups), len(groups)) for p in groups[i]]
        deltas.append(sum(b - a for a, b in pairs) / len(pairs))
    return {"unit": "image", "images": len(groups), "repeats": repeats, "seed": seed,
            "delta_accuracy": sum(b["correct"] - a["correct"] for a, b in zip(left, right)) / len(left),
            "delta_accuracy_ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]}


def report(ds, rows, results):
    summary = {"dataset": ds, "judge_spec": spec(), "judge_spec_hash": digest(spec()),
               "judge_source_sha256": JUDGE_FILE_SHA256,
               "question_sent": "plain POPE question (instruction and yes/no suffix removed)",
               "response_sent": "complete decoded response text ({method}_answers.json 'text')",
               "methods": {}}
    scored = {}
    for method in METHODS:
        out = []
        for r in rows[method]:
            res = results.get(r["request_sha256"])
            if not res or res["status"] != "success":
                continue
            p = res["parsed"]
            out.append({**r, "judge_prediction": p["answer"], "judge_status": p["status"],
                        "evidence": p["evidence"],
                        "correct": int(p["answer"] is not None and p["answer"] == r["label"])})
        scored[method] = out
        complete = len(out) == len(rows[method])
        splits = {}
        for s in SPLITS:
            sub = [r for r in out if r["split"] == s]
            splits[s] = {**pope_metrics([(r["judge_prediction"], r["label"]) for r in sub]),
                         "n": len(sub), "statuses": dict(Counter(r["judge_status"] for r in sub))}
        block = {"complete": complete, "judged_rows": len(out), "splits": splits}
        if complete:
            block["macro"] = {k: statistics.mean(splits[s][k] for s in SPLITS)
                              for k in ("accuracy", "f1", "precision", "recall", "yes_ratio")
                              if all(splits[s][k] is not None for s in SPLITS)}
        summary["methods"][method] = block
        atomic_json(OUT / ds / f"{method}_judged.json", out)
    if all(summary["methods"][m]["complete"] for m in METHODS):
        for method in ("vcd", "vigor"):
            summary["methods"][method]["paired_vs_native"] = {
                s: paired_bootstrap([r for r in scored["native"] if r["split"] == s],
                                    [r for r in scored[method] if r["split"] == s]) for s in SPLITS}
    usage = Counter()
    for res in results.values():
        for k, v in (res.get("usage") or {}).items():
            if isinstance(v, int):
                usage[k] += v
    summary["api_usage_totals"] = dict(usage)
    summary["api_errors"] = sum(r["status"] == "judge_error" for r in results.values())
    atomic_json(OUT / ds / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    assert file_hash(CDAPC_SRC / "cdapc_lab/evaluation/pope_llm_judge.py") == JUDGE_FILE_SHA256
    key = Path("/root/.deepseek_key").read_text().strip()
    cache = OUT / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    (OUT / args.dataset).mkdir(exist_ok=True)
    atomic_json(OUT / "judge_manifest.json", {"spec": spec(), "spec_hash": digest(spec()),
                                              "source_sha256": JUDGE_FILE_SHA256})
    rows, requests = collect(args.dataset)
    results = {sha: json.loads((cache / f"{sha}.json").read_text())
               for sha in requests if (cache / f"{sha}.json").is_file()}
    pending = [sha for sha in requests if results.get(sha, {}).get("status") != "success"]
    if args.limit:
        pending = pending[:args.limit]
    print("UNIQUE_REQUESTS", len(requests), "PENDING", len(pending), flush=True)
    done = 0
    for offset in range(0, len(pending), 256):
        batch = pending[offset:offset + 256]
        failures = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(judge_request, requests[sha], cache, key): sha for sha in batch}
            for future in as_completed(futures):
                results[futures[future]] = future.result()
                failures += results[futures[future]]["status"] != "success"
        done += len(batch)
        print("DONE", done, "BATCH_ERRORS", failures, flush=True)
        if failures >= max(8, len(batch) // 2):
            raise RuntimeError("High API/format failure rate; inspect cache")
    summary = report(args.dataset, rows, results)
    print(json.dumps({m: summary["methods"][m].get("macro") for m in METHODS}),
          "usage", summary["api_usage_totals"], "errors", summary["api_errors"], flush=True)


if __name__ == "__main__":
    main()
