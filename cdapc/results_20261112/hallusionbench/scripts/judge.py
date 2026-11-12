"""DeepSeek answer-extraction judge (pope_text_llm_judge_v1) for all HallusionBench arms + CD baselines."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import sys
from pathlib import Path

sys.argv = sys.argv[:1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run
from cdapc_lab.evaluation.llm_judge import judge_key
from cdapc_lab.evaluation.pope_llm_judge import judge_request, request_for, spec
from cdapc_lab.io.artifacts import atomic_json, digest, read_json

ROOT = run.ROOT
CACHE = ROOT.parent / "cdapc-experiments/reports/pope_think_llm_judge/cache"
LABEL = {"cdapc_direct": "CDAPC 直答", "native_direct": "Native 直答（B64）", "cdapc_think": "CDAPC think",
         "native_think": "Native think（B64）", "baseline_native": "基线 Native（B32）",
         "baseline_vcd": "基线 VCD（B32）", "baseline_vigor": "基线 VIGOR（B32）"}


def main():
    rows = read_json(ROOT / "data/hallusion/rows.json")
    result = read_json(ROOT / "reports/result.json")
    texts = {arm: {sid: (v["text"], v["status"]) for sid, v in result["per_row"][arm].items()} for arm in run.ARMS}
    for m in ("native", "vcd", "vigor"):
        texts[f"baseline_{m}"] = {a["index"]: (a["text"], "success") for a in read_json(ROOT / f"baseline/{m}/answers.json")}
    question = {r["index"]: r["question"] for r in rows}
    requests, keys = {}, {}
    for arm, items in texts.items():
        keys[arm] = {}
        for sid, (text, status) in items.items():
            if status != "success":
                keys[arm][sid] = None
                continue
            request = request_for(question[sid], text)
            keys[arm][sid] = digest(request)
            requests[keys[arm][sid]] = request
    print("UNIQUE_REQUESTS", len(requests), flush=True)
    with (CACHE.parent / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        results = {h: read_json(CACHE / f"{h}.json") for h in requests if (CACHE / f"{h}.json").exists()}
        hits = sum(v["status"] == "success" for v in results.values())
        key = judge_key()
        for attempt in range(4):
            pending = [h for h in requests if results.get(h, {}).get("status") != "success"]
            print("ATTEMPT", attempt, "PENDING", len(pending), flush=True)
            if not pending:
                break
            for offset in range(0, len(pending), 64):
                batch = pending[offset:offset + 64]
                with ThreadPoolExecutor(max_workers=8) as pool:
                    futures = {pool.submit(judge_request, requests[h], CACHE, key): h for h in batch}
                    for future in as_completed(futures):
                        results[futures[future]] = future.result()
                done = sum(v["status"] == "success" for v in results.values())
                print("JUDGED", done, "/", len(requests), flush=True)
    errors = [h for h in requests if results.get(h, {}).get("status") != "success"]
    assert not errors, f"{len(errors)} judge requests still failing"
    extract_fn, rating = run.official_functions()
    image_rows = [r for r in rows if r["input_mode"] == "image"]
    report = {"judge_spec": spec(), "judge_spec_hash": digest(spec()), "unique_requests": len(requests),
              "cache_hits": hits, "api_errors": 0, "arms": {}, "per_row": {}}
    scores = {}
    for arm in texts:
        pred, status = {}, Counter()
        for sid, h in keys[arm].items():
            parsed = results[h]["parsed"] if h else {"answer": None, "status": "budget_exhausted"}
            status[parsed["status"]] += 1
            # Map the extracted answer to a single word so the unchanged VLMEvalKit rating applies.
            pred[sid] = {"yes": "Yes", "no": "No"}.get(parsed["answer"], "")
        metrics, scores[arm] = run.rate(rows, pred, extract_fn, rating)
        image, _ = run.rate(image_rows, pred, extract_fn, rating)
        report["arms"][arm] = {"metrics": metrics, "image": image, "judge_statuses": dict(status)}
        report["per_row"][arm] = {sid: {"request_sha256": keys[arm][sid], "prediction": pred[sid],
                                        "correct": bool(scores[arm][sid])} for sid in keys[arm]}
    pairs = [("cdapc_direct", "native_direct"), ("cdapc_think", "native_think"), ("cdapc_direct", "baseline_native"),
             ("cdapc_direct", "baseline_vcd"), ("cdapc_direct", "baseline_vigor"),
             ("cdapc_think", "cdapc_direct"), ("native_think", "native_direct")]
    report["paired"] = {f"{a} - {b}": {"all": run.set_bootstrap(rows, scores[a], scores[b]),
                                       "image": run.set_bootstrap(image_rows, scores[a], scores[b])} for a, b in pairs}
    atomic_json(ROOT / "reports/deepseek.json", report)
    lines = ["# HallusionBench：DeepSeek judge（pope_text_llm_judge_v1，deepseek-flash）", "",
             "API只看题目与完整回答（不给图像和标准答案），提取回答实际表达的Yes/No；本地比对。缺答、歧义、预算耗尽计错。",
             "提取后的Yes/No交给VLMEvalKit原版Hallusion_rating计算aAcc/fAcc/qAcc。", "",
             "| 组别 | aAcc | fAcc | qAcc | 图像题aAcc | 官方规则aAcc | 缺答/歧义 |", "|---|---:|---:|---:|---:|---:|---:|"]
    official = {**{a: b["metrics"]["aAcc"] for a, b in result["arms"].items()},
                **{f"baseline_{m}": b["aAcc"] for m, b in result["baseline"].items()}}
    for arm, b in report["arms"].items():
        m, s = b["metrics"], b["judge_statuses"]
        lines.append(f"| {LABEL[arm]} | {m['aAcc']:.2f} | {m['fAcc']:.2f} | {m['qAcc']:.2f} | {b['image']['aAcc']:.2f} | "
                     f"{official[arm]:.2f} | {s.get('missing', 0)}/{s.get('ambiguous', 0)} |")
    lines += ["", "| 比较 | 范围 | ΔaAcc [95% CI] | ΔfAcc [95% CI] | ΔqAcc [95% CI] |", "|---|---|---:|---:|---:|"]
    for name, scopes in report["paired"].items():
        for scope, v in scopes.items():
            lines.append(f"| {name} | {'全部' if scope == 'all' else '图像题'} | " + " | ".join(
                f"{v[k]['delta']:+.2f} [{v[k]['ci95'][0]:+.2f}, {v[k]['ci95'][1]:+.2f}]" for k in ("aAcc", "fAcc", "qAcc")) + " |")
    lines.append(f"\n请求 {len(requests)} 个（缓存命中 {hits}），API错误 0。")
    (ROOT / "reports/deepseek.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
