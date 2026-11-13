"""Score completed MME K4 arms with the existing text-only DeepSeek extractor."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import fcntl
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run
from cdapc_lab.evaluation import pope_llm_judge
from cdapc_lab.evaluation.llm_judge import BASE_URL, _post, judge_key
from cdapc_lab.io.artifacts import atomic_json, digest, file_hash, read_json

ROOT = run.ROOT
CACHE = ROOT.parent / "cdapc-experiments/reports/pope_think_llm_judge/cache"
ARMS = {"think_k4": run.THINK, "direct_k4": ROOT}
LABELS = {"think_k4": "CDAPC think K4", "direct_k4": "CDAPC 直答 K4"}


def main():
    rows_path = ROOT / "data/mme/rows.json"
    rows = read_json(rows_path)
    questions = {r["sample_id"]: r["question"] for r in rows}
    assert len(questions) == 2374
    sources = {arm: read_json(path / "reports/result.json") for arm, path in ARMS.items()}
    categories, _ = run.official_protocol()
    requests, keys = {}, {}
    for arm, source in sources.items():
        assert read_json(ARMS[arm] / "data/mme/rows.json") == rows
        assert source["status"] == "complete" and set(source["per_sample"]) == set(questions)
        original = {sid: v["prediction"] for sid, v in source["per_sample"].items()}
        assert run.metrics(rows, original, categories) == source["metrics"]
        keys[arm] = {}
        for sid, value in source["per_sample"].items():
            if value["generation_status"] != "success":
                keys[arm][sid] = None
                continue
            request = pope_llm_judge.request_for(questions[sid], value["text"])
            h = digest(request)
            requests[h] = request
            keys[arm][sid] = h
    print("UNIQUE_REQUESTS", len(requests), flush=True)
    with (CACHE.parent / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        results = {h: read_json(CACHE / f"{h}.json") for h in requests if (CACHE / f"{h}.json").exists()}
        hits = sum(v["status"] == "success" for v in results.values())
        key = judge_key()
        assert key or hits == len(requests), "DeepSeek credential unavailable"
        print("CACHE_HITS", hits, flush=True)
        for attempt in range(4):
            pending = [h for h in requests if results.get(h, {}).get("status") != "success"
                       and len(results.get(h, {}).get("attempts", [])) < 12]
            if not pending:
                break
            print("ATTEMPT", attempt + 1, "PENDING", len(pending), flush=True)
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = {pool.submit(pope_llm_judge.judge_request, requests[h], CACHE, key): h for h in pending}
                for n, future in enumerate(as_completed(futures), 1):
                    results[futures[future]] = future.result()
                    if n % 64 == 0 or n == len(pending):
                        completed = sum(v["status"] == "success" for v in results.values())
                        print("JUDGED", completed, "/", len(requests), flush=True)
                        atomic_json(ROOT / "reports/deepseek_status.json", {
                            "status": "judging", "completed": completed, "expected": len(requests),
                            "updated_at": datetime.now().astimezone().isoformat()})
        # A format-only retry avoids repeatedly quoting the question's spelling
        # instead of the response's spelling. The answer validation is unchanged.
        for h in requests:
            if results[h]["status"] == "success":
                continue
            value = results[h]
            request = requests[h]
            text = json.loads(request["messages"][1]["content"])["candidate_response"]
            for _ in range(2):
                payload = {**request, "messages": [
                    {"role": "system", "content": request["messages"][0]["content"] +
                     "\nFor this formatting retry, evidence must be at most 12 characters. "
                     "Use a single Yes or No if it occurs in the candidate and supports the extracted answer. "
                     "Do not change the answer extraction rules. Never copy evidence from the question."},
                    *request["messages"][1:]]}
                entry = {"requested_at": datetime.now().astimezone().isoformat(),
                         "request": payload, "retry_kind": "short_verbatim_evidence"}
                try:
                    response = _post(BASE_URL, payload, key)
                    entry["response"] = response
                    parsed = pope_llm_judge.parse_response(
                        response["choices"][0]["message"]["content"], text)
                    value.update(status="success", parsed=parsed,
                                 returned_model=response.get("model"), usage=response.get("usage"))
                except Exception as error:
                    entry["error"] = str(error)[:500]
                value["attempts"].append(entry)
                atomic_json(CACHE / f"{h}.json", value)
                if value["status"] == "success":
                    break
    errors = [h for h in requests if results.get(h, {}).get("status") != "success"]
    assert not errors, f"{len(errors)} API/format failures remain pending; no final score published"
    report = {
        "status": "complete", "created_at": datetime.now().astimezone().isoformat(),
        "judge_spec": pope_llm_judge.spec(), "judge_spec_hash": digest(pope_llm_judge.spec()),
        "unique_requests": len(requests), "cache_hits": hits, "api_errors": 0,
        "format_retry_extension": "After standard retries, constrain verbatim evidence to at most 12 characters; unchanged answer rules and validation. Actual payloads retained.",
        "inputs_per_arm": len(rows), "image_pairs_per_arm": 1187,
        "failure_policy": "Fixed denominator; generation failure, missing and ambiguous count incorrect. API errors block finalization.",
        "score": "MME category-wise 100*(Acc+Acc+), summed into perception/cognition/total",
        "source_reports": {a: {"path": str(p / "reports/result.json"),
                                "sha256": file_hash(p / "reports/result.json")} for a, p in ARMS.items()},
        "rows_sha256": file_hash(rows_path), "arms": {}, "per_sample": {}, "paired": {},
    }
    answers = {}
    for arm, source in sources.items():
        answers[arm], per, statuses, transitions = {}, {}, Counter(), Counter()
        for row in rows:
            sid = row["sample_id"]
            h = keys[arm][sid]
            original = source["per_sample"][sid]
            parsed = results[h]["parsed"] if h else {
                "answer": None, "status": original["generation_status"], "evidence": ""}
            prediction = parsed["answer"] or "other"
            answers[arm][sid] = prediction
            statuses[parsed["status"]] += 1
            old_correct = original["prediction"] == row["answer"]
            correct = prediction == row["answer"]
            transitions[f"{int(old_correct)}_to_{int(correct)}"] += 1
            per[sid] = {"request_sha256": h, "prediction": prediction, "judge": parsed,
                        "generation_status": original["generation_status"], "correct": correct,
                        "official_prediction": original["prediction"], "official_correct": old_correct}
        original_answers = {sid: v["prediction"] for sid, v in source["per_sample"].items()}
        report["arms"][arm] = {
            "metrics": run.metrics(rows, answers[arm], categories), "official_metrics": source["metrics"],
            "judge_statuses": dict(statuses), "official_to_judge_correctness": dict(transitions),
            "judge_minus_official": run.paired_bootstrap(rows, answers[arm], original_answers, categories),
            "official_unparseable_recovered": sum(
                v["official_prediction"] == "other" and v["correct"] for v in per.values()),
        }
        report["per_sample"][arm] = per
    report["paired"]["think_minus_direct"] = run.paired_bootstrap(
        rows, answers["think_k4"], answers["direct_k4"], categories)
    atomic_json(ROOT / "reports/deepseek_requests.json", results)
    atomic_json(ROOT / "reports/deepseek.json", report)
    lines = ["# MME K4：DeepSeek 答案提取评分", "",
             "两组各2374题、1187个图像对，L256/I128、r70/W16/K4。"
             "沿用pope_text_llm_judge_v1（deepseek-flash），API只看题目和完整的首EOS归一化回答，不提供图像或标准答案。"
             "提取Yes/No后在本地计算MME Acc+Acc+；缺答、歧义、生成失败均计错。此为补充评分，不替代官方解析结果。",
             "", "| 组别 | Perception /2000 | Cognition /800 | 总分 /2800 | 官方总分 | 缺答 | 歧义 | 生成失败 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, v in report["arms"].items():
        m, s = v["metrics"], v["judge_statuses"]
        failures = sum(n for name, n in s.items() if name not in ("answered", "missing", "ambiguous"))
        lines.append(f"| {LABELS[arm]} | {m['perception']:.2f} | {m['cognition']:.2f} | "
                     f"{m['total']:.2f} | {v['official_metrics']['total']:.2f} | "
                     f"{s.get('missing', 0)} | {s.get('ambiguous', 0)} | {failures} |")
    lines += ["", "按类别分层、图像对配对bootstrap，2000次，seed=42：", ""]
    comparisons = {"think − 直答": report["paired"]["think_minus_direct"],
                   **{f"{LABELS[a]}：DeepSeek − 官方": v["judge_minus_official"] for a, v in report["arms"].items()}}
    for label, v in comparisons.items():
        lines.append(f"- {label}：总分 {v['delta_total']:+.2f}，95% CI "
                     f"[{v['ci95_total'][0]:+.2f}, {v['ci95_total'][1]:+.2f}]。")
    lines += ["", "| 类别 | think官方 | think DeepSeek | 直答官方 | 直答 DeepSeek |",
              "|---|---:|---:|---:|---:|"]
    for category in sorted(categories["Perception"] + categories["Cognition"]):
        values = [report["arms"][a][key]["per_category"][category]["score"]
                  for a in ARMS for key in ("official_metrics", "metrics")]
        lines.append(f"| {category} | " + " | ".join(f"{v:.2f}" for v in values) + " |")
    lines += ["", f"唯一请求 {len(requests)}；缓存命中 {hits}；最终API错误 0。",
              "deepseek.json包含逐题结果、证据、请求哈希和配对区间；deepseek_requests.json保存请求及API响应。"]
    (ROOT / "reports/deepseek.md").write_text("\n".join(lines) + "\n")
    files = [Path(__file__), Path(run.__file__), Path(pope_llm_judge.__file__),
             ROOT / "scripts/reference_score_mme.py", ROOT / "upstream/mme/calculation.py", rows_path,
             *[p / "reports/result.json" for p in ARMS.values()],
             *[ROOT / "reports" / n for n in ("deepseek.json", "deepseek_requests.json", "deepseek.md")]]
    (ROOT / "reports/deepseek.sha256").write_text("".join(f"{file_hash(p)}  {p}\n" for p in files))
    atomic_json(ROOT / "reports/deepseek_status.json", {
        "status": "complete", "completed": len(requests), "expected": len(requests)})
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
