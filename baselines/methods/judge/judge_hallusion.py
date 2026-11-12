"""DeepSeek text judge for HallusionBench think 256/128/64 (VCD, VIGOR), reusing
the cloud pope_text_llm_judge_v1 protocol verbatim (no image, no gold sent).
The extracted yes/no is fed to the pinned VLMEvalKit Hallusion_rating, so
aAcc/fAcc/qAcc keep their official definitions; missing/ambiguous -> Unknown.
Run from the project root: .venv/bin/python judge/judge_hallusion.py
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/root/autodl-tmp/mmada-records/cdapc/source/src")
JUDGE_FILE_SHA256 = "1ab1a8ef942edc347f87f8165df6d870cff67fb040b976006fff2a03e846fa57"
from cdapc_lab.evaluation.pope_llm_judge import judge_request, request_for, spec  # noqa: E402
from cdapc_lab.io.artifacts import atomic_json, digest, file_hash  # noqa: E402
from common.hallusion import rate  # noqa: E402

STAGE = ROOT / "results/think_benchmarks_256_128_64/hallusion_think"
OUT = STAGE / "judge"
CACHE = ROOT / "results/pope_think_256_128_64/judge/cache"
METHODS = ("vcd", "vigor")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    assert file_hash(Path("/root/autodl-tmp/mmada-records/cdapc/source/src/cdapc_lab/evaluation/pope_llm_judge.py")) == JUDGE_FILE_SHA256
    key = Path("/root/.deepseek_key").read_text().strip()
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "judge_manifest.json", {"spec": spec(), "spec_hash": digest(spec()),
                                              "source_sha256": JUDGE_FILE_SHA256,
                                              "question_sent": "original HallusionBench question (no instruction)",
                                              "response_sent": "complete decoded response"})
    answers = {m: json.loads((STAGE / f"{m}_answers.json").read_text()) for m in METHODS}
    requests, rows = {}, {}
    for method, items in answers.items():
        rows[method] = []
        for a in items:
            request = request_for(a["original_question"], a["text"])
            sha = digest(request)
            requests[sha] = request
            rows[method].append((a, sha))
    results = {sha: json.loads((CACHE / f"{sha}.json").read_text())
               for sha in requests if (CACHE / f"{sha}.json").is_file()}
    pending = [sha for sha in requests if results.get(sha, {}).get("status") != "success"]
    print("UNIQUE_REQUESTS", len(requests), "PENDING", len(pending), flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(judge_request, requests[sha], CACHE, key): sha for sha in pending}
        for n, future in enumerate(as_completed(futures), 1):
            results[futures[future]] = future.result()
            if n % 500 == 0:
                print("DONE", n, flush=True)
    summary = {"judge_spec_hash": digest(spec()), "methods": {}}
    for method in METHODS:
        judged = []
        for a, sha in rows[method]:
            res = results[sha]
            if res["status"] != "success":
                raise RuntimeError(f"judge error for {method} {a['id']}")
            p = res["parsed"]
            judged.append({**{k: v for k, v in a.items() if k not in ("text", "scored_text")},
                           "request_sha256": sha,
                           "text_sha256": hashlib.sha256(a["text"].encode()).hexdigest(),
                           "judge_prediction": p["answer"], "judge_status": p["status"],
                           "evidence": p["evidence"],
                           "prediction": {"yes": "Yes", "no": "No"}.get(p["answer"], "")})
        scored, metrics = rate(judged)
        metrics["by_input_mode"] = {mode: rate([r for r in judged if r["input_mode"] == mode])[1]
                                    for mode in ("image", "text_only")}
        metrics["statuses"] = {s: sum(r["judge_status"] == s for r in judged)
                               for s in ("answered", "missing", "ambiguous")}
        atomic_json(OUT / f"{method}_judged.json", scored)
        summary["methods"][method] = metrics
    usage = {}
    for sha in requests:
        for k, v in (results[sha].get("usage") or {}).items():
            if isinstance(v, int):
                usage[k] = usage.get(k, 0) + v
    summary["api_usage_totals"] = usage
    atomic_json(OUT / "summary.json", summary)
    for m, b in summary["methods"].items():
        print(m, round(b["accuracy_percent"], 2), round(b["figure_accuracy_percent"], 2),
              round(b["question_accuracy_percent"], 2), b["statuses"], flush=True)


if __name__ == "__main__":
    main()
