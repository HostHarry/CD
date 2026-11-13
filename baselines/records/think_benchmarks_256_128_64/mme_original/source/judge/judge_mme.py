"""Judge both completed MME conditions with the existing DeepSeek extraction protocol.

Run from the project root: .venv/bin/python judge/judge_mme.py
Only the original question and complete candidate response go to the API.
Gold comparison, fixed-denominator MME scoring and paired tests stay local.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "judge"))
from judge_hallusion import (  # noqa: E402
    CACHE, JUDGE_FILE_SHA256, atomic_json, digest, file_hash,
    judge_request, request_for, spec,
)
from cdapc_lab.evaluation.pope_llm_judge import parse_response  # noqa: E402
from cdapc_lab.evaluation.llm_judge import BASE_URL, _post  # noqa: E402
from experiments.think_benchmarks import (  # noqa: E402
    MME, RUN, STAGES, export_official_text, mme_metrics, mme_official, paired_bootstrap,
)

METHODS = ("native", "vcd", "vigor")
STAGE_NAMES = ("mme_think", "mme_original")
JUDGE_SOURCE = Path(
    "/root/autodl-tmp/mmada-records/cdapc/source/src/cdapc_lab/evaluation/pope_llm_judge.py"
)
FORMAT_FEEDBACK = (
    "Your evidence failed exact substring validation against candidate_response. "
    "Do not autocorrect or complete spelling, punctuation, or unmatched quotation "
    "marks in the candidate. Copy a SHORTER contiguous substring if necessary. "
    "Keep the answer/status justified by the candidate, and return only the required "
    "JSON object. Evidence must appear character-for-character in candidate_response."
)


def collect(stages):
    rows, requests = {}, {}
    for stage in stages:
        rows[stage] = {}
        samples = [json.loads(line) for line in
                   (STAGES[stage]["data"] / "samples.jsonl").read_text().splitlines()]
        gold = {r["id"]: r for r in samples}
        assert len(gold) == len(samples) == 2374
        for method in METHODS:
            answers = json.loads((RUN / stage / f"{method}_answers.json").read_text())
            assert len(answers) == len(gold)
            assert {r["id"] for r in answers} == set(gold)
            rows[stage][method] = []
            for a in sorted(answers, key=lambda r: r["id"]):
                for field in ("answer", "original_question", "group_id", "category"):
                    assert a[field] == gold[a["id"]][field]
                request = request_for(a["original_question"], a["text"])
                sha = digest(request)
                requests[sha] = request
                rows[stage][method].append((a, sha))
    return rows, requests


def validate_result(sha, request, result):
    assert result["request_sha256"] == sha == digest(result["request"])
    assert result["request"] == request
    text = json.loads(request["messages"][1]["content"])["candidate_response"]
    content = result["attempts"][-1]["response"]["choices"][0]["message"]["content"]
    assert parse_response(content, text) == result["parsed"]


def repair_evidence(request, previous, key):
    """Additional format-only retries; keep all actual payloads and failed responses."""
    result = dict(previous)
    attempts = list(previous["attempts"])
    text = json.loads(request["messages"][1]["content"])["candidate_response"]
    for _ in range(3):
        last = attempts[-1]
        if last.get("error") != "Evidence must occur verbatim in the candidate":
            break
        bad = last["response"]["choices"][0]["message"]["content"]
        payload = {**request, "messages": [
            *request["messages"], {"role": "assistant", "content": bad},
            {"role": "user", "content": FORMAT_FEEDBACK},
        ]}
        entry = {"requested_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "request": payload}
        try:
            response = _post(BASE_URL, payload, key)
            entry["response"] = response
            parsed = parse_response(response["choices"][0]["message"]["content"], text)
            result.update(status="success", parsed=parsed,
                          returned_model=response.get("model"), usage=response.get("usage"))
        except Exception as error:
            entry["error"] = str(error)[:500]
        attempts.append(entry)
        result["attempts"] = attempts
        atomic_json(CACHE / f"{digest(request)}.json", result)
        if result["status"] == "success":
            break
    return result


def verify_official(out, method_rows, metrics):
    export_official_text(out, method_rows)
    for method in METHODS:
        checked = subprocess.run(
            [sys.executable, str(MME / "upstream/calculation.py"),
             "--results_dir", str(out / "official_text" / method)],
            capture_output=True, text=True, check=True,
            env={**os.environ, "PYTHONPATH": str(MME / "vendor")},
        )
        (out / f"official_scorer_{method}.log").write_text(checked.stdout + checked.stderr)
        official = dict(re.findall(r"^[ \t]+(\w+)[ \t]+score:\s+([\d.eE+-]+)",
                                   checked.stdout, re.M))
        mine = metrics[method]["primary"]["per_category"]
        assert set(official) == set(mine)
        assert all(abs(float(v) - mine[k]["score"]) < 1e-8 for k, v in official.items())
        metrics[method]["official_scorer_verified"] = True


def report(stage, rows, results):
    out = RUN / stage / "judge"
    categories, _ = mme_official()
    method_rows, metrics = {}, {}
    for method, items in rows.items():
        judged = []
        for a, sha in items:
            p = results[sha]["parsed"]
            parsed = p["answer"] or "other"
            judged.append({
                **{k: a[k] for k in ("id", "group_id", "category", "image_name",
                                     "original_question", "answer", "image_sha256", "prompt_id")},
                "request_sha256": sha,
                "text_sha256": hashlib.sha256(a["text"].encode()).hexdigest(),
                "judge_prediction": p["answer"], "judge_status": p["status"],
                "evidence": p["evidence"], "parsed": parsed, "scored_text": parsed,
                "correct": parsed == a["answer"],
                "rule_prediction": a["parsed"], "rule_correct": a["correct"],
            })
        method_rows[method] = judged
        metrics[method] = {
            "primary": mme_metrics(judged, categories),
            "statuses": dict(Counter(r["judge_status"] for r in judged)),
            "rule_comparison": {
                "prediction_transitions": dict(Counter(
                    f"{r['rule_prediction']}->{r['parsed']}" for r in judged)),
                "rule_wrong_judge_correct": sum(not r["rule_correct"] and r["correct"] for r in judged),
                "rule_correct_judge_wrong": sum(r["rule_correct"] and not r["correct"] for r in judged),
            },
        }
        atomic_json(out / f"{method}_judged.json", judged)
    for method in ("vcd", "vigor"):
        metrics[method]["paired_vs_native"] = paired_bootstrap(
            method_rows["native"], method_rows[method], "group_id")
    verify_official(out, method_rows, metrics)
    hashes = sorted({sha for items in rows.values() for _, sha in items})
    usage = Counter()
    # Include all retained attempts, including format retries, rather than just the final call.
    for sha in hashes:
        for attempt in results[sha]["attempts"]:
            usage.update({k: v for k, v in (attempt.get("response", {}).get("usage") or {}).items()
                          if isinstance(v, int)})
    with gzip.open(out / "api_records.jsonl.gz", "wt", encoding="utf-8") as f:
        for sha in hashes:
            f.write(json.dumps(results[sha], ensure_ascii=False) + "\n")
    summary = {
        "stage": stage, "complete": True, "rows_per_method": 2374,
        "unique_requests": len(hashes), "api_errors": 0,
        "judge_spec": spec(), "judge_spec_hash": digest(spec()),
        "api_usage_totals_all_attempts": dict(usage), "methods": metrics,
    }
    atomic_json(out / "summary.json", summary)
    lines = [
        f"# {stage}: DeepSeek answer extraction",
        "",
        "Protocol: pope_text_llm_judge_v1; deepseek-flash; full response and original question.",
        "No gold or image sent. Missing/ambiguous count incorrect; API errors block final scoring.",
        "MME category Acc + Acc+ is rechecked using the unmodified official scorer.",
        "Paired intervals concern question accuracy, not MME total score.",
        "",
        "| Method | Perception /2000 | Cognition /800 | Total /2800 | Acc % | Unknown |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, block in metrics.items():
        p = block["primary"]
        unknown = sum(c["other"] for c in p["per_category"].values())
        lines.append(f"| {method} | {p['perception']:.2f} | {p['cognition']:.2f} | "
                     f"{p['total']:.2f} | {100*p['accuracy']:.2f} | {unknown} |")
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text("".join(
        f"{file_hash(p)}  {p.relative_to(out)}\n" for p in files))
    print(stage, json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="+", choices=STAGE_NAMES, default=list(STAGE_NAMES))
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--limit", type=int, help="Smoke test: at most this many uncached requests")
    args = parser.parse_args()
    assert file_hash(JUDGE_SOURCE) == JUDGE_FILE_SHA256
    rows, requests = collect(args.stages)
    CACHE.mkdir(parents=True, exist_ok=True)
    with (RUN / "mme_judge.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for stage in args.stages:
            out = RUN / stage / "judge"
            out.mkdir(parents=True, exist_ok=True)
            atomic_json(out / "judge_manifest.json", {
                "stage": stage, "spec": spec(), "spec_hash": digest(spec()),
                "judge_source_sha256": JUDGE_FILE_SHA256,
                "runner_sha256": file_hash(Path(__file__)),
                "scoring_source_sha256": file_hash(ROOT / "experiments/think_benchmarks.py"),
                "official_scorer_sha256": file_hash(MME / "upstream/calculation.py"),
                "samples_sha256": file_hash(STAGES[stage]["data"] / "samples.jsonl"),
                "answers_sha256": {m: file_hash(RUN / stage / f"{m}_answers.json") for m in METHODS},
                "question_sent": "original_question, including original yes/no wording",
                "response_sent": "complete decoded response; no stripping of think blocks",
                "additional_evidence_format_retry": FORMAT_FEEDBACK,
            })
        results = {sha: json.loads((CACHE / f"{sha}.json").read_text())
                   for sha in requests if (CACHE / f"{sha}.json").is_file()}
        for sha, result in results.items():
            if result["status"] == "success":
                validate_result(sha, requests[sha], result)
        pending = [sha for sha in requests if results.get(sha, {}).get("status") != "success"]
        print("UNIQUE_REQUESTS", len(requests), "PENDING", len(pending), flush=True)
        if args.limit is not None:
            pending = pending[:args.limit]
        key = Path("/root/.deepseek_key").read_text().strip() if pending else None
        for offset in range(0, len(pending), 256):
            batch = pending[offset:offset + 256]
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(judge_request, requests[sha], CACHE, key): sha for sha in batch}
                for future in as_completed(futures):
                    sha = futures[future]
                    result = future.result()
                    if result["status"] == "success":
                        validate_result(sha, requests[sha], result)
                    results[sha] = result
            errors = sum(results[sha]["status"] != "success" for sha in batch)
            print("DONE", min(offset + len(batch), len(pending)), "BATCH_ERRORS", errors, flush=True)
            if errors >= 10:
                raise RuntimeError("Repeated API errors; cached progress retained")
        for sha in pending:
            if results[sha]["status"] != "success":
                results[sha] = repair_evidence(requests[sha], results[sha], key)
                if results[sha]["status"] == "success":
                    validate_result(sha, requests[sha], results[sha])
        missing = sum(results.get(sha, {}).get("status") != "success" for sha in requests)
        print("REMAINING", missing, flush=True)
        if missing:
            if args.limit is None:
                raise RuntimeError("Judge incomplete; resume to retry pending requests")
            return
        for stage in args.stages:
            report(stage, rows[stage], results)


if __name__ == "__main__":
    main()
