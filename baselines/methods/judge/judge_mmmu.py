"""DeepSeek LLM judge for MMMU-val (mmmu_wino_v1), reusing the cloud CD-APC
MMMU judge verbatim: cdapc_lab.evaluation.llm_judge (WINO JUDGE_RULES from
LaViDa@24220c0 eval/lmms_eval/tasks/mmmu/utils.py, deepseek-flash, temperature
0, max_tokens 20, score = 1 if the reply contains "1").

Judges local VCD/VIGOR outputs and, as a same-setup control, the cloud native
outputs published in HostHarry/CD mmmu/answers/native.jsonl.
Run from the project root: .venv/bin/python judge/judge_mmmu.py
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
RECORDS = Path("/root/autodl-tmp/mmada-records")
CDAPC_SRC = RECORDS / "cdapc/source/src"
RULES_FILE = ROOT / "judge/upstream/lavida_mmmu_utils.py"
RULES_FILE_SHA256 = "cb865ea43a83bb63fd5cba0b72d776a5673638b597b634460c9c089e6eee85de"
OUT = ROOT / "results/mmmu_wino_v1/judge"
sys.path.insert(0, str(CDAPC_SRC))
import cdapc_lab.evaluation.llm_judge as lj  # noqa: E402
from cdapc_lab.data.mmmu import mmmu_judge_question  # noqa: E402
from cdapc_lab.io.artifacts import atomic_json, file_hash  # noqa: E402

lj.JUDGE_UTILS = RULES_FILE


def load_samples():
    rows = {}
    for shard in (0, 1):
        for line in (ROOT / f"data/mmmu_wino_v1/shard_{shard}.jsonl").open():
            r = json.loads(line)
            rows[r["id"]] = r
    assert len(rows) == 900
    return rows


def load_outputs():
    native = [json.loads(line) for line in (RECORDS / "mmmu/answers/native.jsonl").open()]
    gold = {r["sample_id"]: r["gold"] for r in native}
    outputs = {"native": {r["sample_id"]: r["text"] for r in native}}
    for method in ("vcd", "vigor"):
        texts = {}
        for shard in (0, 1):
            for r in json.loads((ROOT / f"{method}/results/mmmu_wino_v1/shard{shard}/{method}_answers.json").read_text()):
                texts[r["id"]] = r["text"]
        outputs[method] = texts
    for method, texts in outputs.items():
        assert set(texts) == set(gold), method
    return gold, outputs


def paired_bootstrap(a, b, repeats=2000, seed=42):
    import numpy as np
    ids = sorted(a)
    diff = np.array([b[i] - a[i] for i in ids], dtype=float)
    rng = np.random.default_rng(seed)
    means = [diff[rng.integers(0, len(ids), len(ids))].mean() for _ in range(repeats)]
    return {"unit": "question", "n": len(ids), "repeats": repeats, "seed": seed,
            "delta": float(diff.mean()),
            "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    assert file_hash(RULES_FILE) == RULES_FILE_SHA256
    key = Path("/root/.deepseek_key").read_text().strip()
    spec, spec_hash = lj.judge_spec("deepseek-flash")
    samples = load_samples()
    gold, outputs = load_outputs()
    cache = OUT / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "judge_manifest.json", {
        **spec, "judge_spec_hash": spec_hash, "rules_file": "LaViDa@24220c0 eval/lmms_eval/tasks/mmmu/utils.py",
        "llm_judge_source_sha256": file_hash(CDAPC_SRC / "cdapc_lab/evaluation/llm_judge.py"),
        "native_source": "HostHarry/CD mmmu/answers/native.jsonl (cloud native, same protocol)",
        "pred": "text inside <answer>...</answer> if present, else complete decoded response"})
    jobs = {}
    for method, texts in outputs.items():
        for sid, text in texts.items():
            r = samples[sid]
            question = mmmu_judge_question(r["original_question"], r["options"])
            match = lj.ANSWER_RE.search(text)
            pred = match.group(1) if match else text
            request = lj.build_request(spec, question, gold[sid], pred)
            sha = hashlib.sha256(json.dumps(request, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            jobs[(method, sid)] = (sha, question, pred)
    unique = {}
    for (method, sid), (sha, question, pred) in jobs.items():
        unique.setdefault(sha, (question, gold[sid], pred))
    results = {sha: json.loads((cache / f"{sha}.json").read_text())
               for sha in unique if (cache / f"{sha}.json").is_file()}
    pending = [sha for sha in unique if results.get(sha, {}).get("status") == "judge_error" or sha not in results]
    if args.limit:
        pending = pending[:args.limit]
    print("UNIQUE_REQUESTS", len(unique), "PENDING", len(pending), flush=True)

    def run(sha):
        question, answer, pred = unique[sha]
        result = lj.judge_prediction(spec, question, answer, pred, key)
        result.setdefault("status", "success")
        atomic_json(cache / f"{sha}.json", result)
        return sha, result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n, future in enumerate(as_completed([pool.submit(run, sha) for sha in pending]), 1):
            sha, result = future.result()
            results[sha] = result
            if n % 200 == 0:
                print("DONE", n, flush=True)
    summary = {"judge_spec_hash": spec_hash, "methods": {}}
    scores = {}
    for method in outputs:
        per = {}
        for sid in outputs[method]:
            res = results.get(jobs[(method, sid)][0])
            if res and res.get("status") != "judge_error":
                per[sid] = res["score"]
        scores[method] = per
        atomic_json(OUT / f"{method}_judged.json", [
            {"id": sid, "request_sha256": jobs[(method, sid)][0], "score": per.get(sid),
             "content": results.get(jobs[(method, sid)][0], {}).get("content")}
            for sid in sorted(outputs[method])])
        block = {"judged": len(per), "errors": 900 - len(per)}
        if len(per) == 900:
            block["accuracy"] = statistics.mean(per.values())
            block["strict_judge_replies"] = sum(
                bool(results[jobs[(method, s)][0]].get("strict_judge")) for s in per)
        summary["methods"][method] = block
    if all(len(s) == 900 for s in scores.values()):
        for method in ("vcd", "vigor"):
            summary["methods"][method]["paired_vs_native"] = paired_bootstrap(scores["native"], scores[method])
        cloud = {r["sample_id"]: r["judge_correct"]
                 for r in map(json.loads, (RECORDS / "mmmu/answers/native.jsonl").open())}
        summary["native_agreement_with_cloud_judge"] = sum(
            int(scores["native"][s] == cloud[s]) for s in cloud) / 900
    usage = {}
    for res in results.values():
        for k, v in (res.get("usage") or {}).items():
            if isinstance(v, int):
                usage[k] = usage.get(k, 0) + v
    summary["api_usage_totals"] = usage
    atomic_json(OUT / "summary.json", summary)
    print(json.dumps(summary, indent=1)[:1500], flush=True)


if __name__ == "__main__":
    main()
