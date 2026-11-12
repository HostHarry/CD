"""Official plus DeepSeek evaluation of the fast EOS-prefix pilot only."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import fcntl
import hashlib
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FAST = ROOT
CONTROL = ROOT.parent / "cdapc-pope-eos-full/reports/judge/pope_coco_think_eos_prefix_r70_k4_L256_I128.json"
CACHE = ROOT.parent / "cdapc-experiments/reports/pope_think_llm_judge/cache"
OUTPUT = ROOT / "reports/judge"
sys.path.insert(0, str(FAST / "src"))
from cdapc_lab.evaluation.llm_judge import judge_key
from cdapc_lab.evaluation.pope_llm_judge import judge_request, request_for, spec
from cdapc_lab.experiments.score import pope_metrics
from cdapc_lab.io.artifacts import atomic_json, digest, file_hash, read_json


def save(state):
    state["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    atomic_json(OUTPUT / "status.json", state)


def metrics(data, rows):
    splits = {split: pope_metrics((data[r["input_key"]]["prediction"], r["label"])
                                  for r in rows if r["split"] == split)
              for split in ("random", "popular", "adversarial")}
    return {"splits": splits,
            "macro": {k: sum(v[k] for v in splits.values()) / 3
                      for k in ("accuracy", "f1", "precision", "recall", "yes_ratio")},
            "statuses_unique": dict(Counter(v["status"] for v in data.values())),
            "unresolved_rows": sum(data[r["input_key"]]["prediction"] is None for r in rows)}


def main():
    OUTPUT.mkdir(exist_ok=True)
    state = {"status": "waiting_for_fast_official", "controller_pid": os.getpid(), "workers": 8}
    with (OUTPUT / "runner.lock").open("a") as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            source = read_json(CONTROL)
            judge_spec = spec()
            assert source["status"] == "complete" and source["api_error_inputs"] == 0
            assert source["judge_spec_hash"] == digest(judge_spec)
            atomic_json(OUTPUT / "judge_manifest.json",
                        {"spec": judge_spec, "spec_hash": digest(judge_spec),
                         "control_report": str(CONTROL), "control_report_sha256": file_hash(CONTROL),
                         "scope": "Only fast-prefix new outputs submitted; control results reused",
                         "failure_policy": "Generation failures incorrect; API failures pending, not model mistakes"})
            save(state)
            while True:
                progress = read_json(FAST / "reports/status.json")
                if progress["status"] == "complete":
                    break
                if progress["status"] in ("interrupted_or_failed", "paused"):
                    raise RuntimeError(f"Fast experiment is {progress['status']}")
                time.sleep(15)
            official_path = FAST / "reports/comparison.json"
            official = read_json(official_path)
            rows = read_json(ROOT / "manifests/rows.json")
            assert official["inputs"] == 5127 and official["rows"] == len(rows) == 9000
            questions = {r["input_key"]: r["question"] for r in rows}
            judged, requests, b_keys = {"A": {}, "B": {}}, {}, {}
            for sid in questions:
                a = official["per_input"]["A"][sid]
                old = source["per_input"][sid]
                assert hashlib.sha256(a["text"].encode()).hexdigest() == old["text_sha256"]
                assert a["status"] == old["generation_status"]
                if a["status"] == "success":
                    sha = digest(request_for(questions[sid], a["text"]))
                    assert sha == old["request_sha256"]
                    cached = read_json(CACHE / (sha + ".json"))
                    assert cached["status"] == "success"
                    assert cached["parsed"]["answer"] == old["judge_prediction"]
                    assert cached["parsed"]["status"] == old["judge_status"]
                judged["A"][sid] = {
                    "prediction": old["judge_prediction"], "status": old["judge_status"],
                    "evidence": old["evidence"], "request_sha256": old["request_sha256"],
                    "text_sha256": old["text_sha256"], "reused": True}
                b = official["per_input"]["B"][sid]
                if b["status"] != "success":
                    judged["B"][sid] = {"prediction": None, "status": b["status"], "evidence": "",
                                         "request_sha256": None,
                                         "text_sha256": hashlib.sha256(b["text"].encode()).hexdigest()}
                    continue
                request = request_for(questions[sid], b["text"])
                sha = digest(request)
                requests[sha], b_keys[sid] = request, sha
            state.update(status="waiting_for_cache_lock", unique_requests=len(requests),
                         control_inputs_reused=len(judged["A"]), failed_generation_inputs=len(judged["B"]))
            save(state)
            with (CACHE.parent / "runner.lock").open("a") as cache_lock:
                fcntl.flock(cache_lock, fcntl.LOCK_EX)
                key = judge_key()
                results = {sha: read_json(CACHE / (sha + ".json"))
                           for sha in requests if (CACHE / (sha + ".json")).exists()}
                initial_attempts = sum(len(v.get("attempts", [])) for v in results.values())
                cache_hits = sum(v["status"] == "success" for v in results.values())
                state.update(status="judging", cache_hits=cache_hits)
                save(state)
                while True:
                    pending = [sha for sha in requests if results.get(sha, {}).get("status") != "success"]
                    if not pending:
                        break
                    if not key:
                        raise RuntimeError("DeepSeek API credential unavailable")
                    if any(len(results.get(sha, {}).get("attempts", [])) >= 9 for sha in pending):
                        raise RuntimeError("A Judge request failed nine attempts; evaluation remains pending")
                    for offset in range(0, len(pending), 64):
                        batch = pending[offset:offset + 64]
                        failures = 0
                        with ThreadPoolExecutor(max_workers=8) as pool:
                            futures = {pool.submit(judge_request, requests[sha], CACHE, key): sha for sha in batch}
                            for future in as_completed(futures):
                                sha = futures[future]
                                results[sha] = future.result()
                                failures += results[sha]["status"] != "success"
                        state.update(requests_succeeded=sum(v["status"] == "success" for v in results.values()),
                                     requests_pending=sum(results.get(sha, {}).get("status") != "success" for sha in requests),
                                     api_errors=sum(v["status"] == "judge_error" for v in results.values()))
                        save(state)
                        print("JUDGE_PROGRESS", state["requests_succeeded"], "/", len(requests), flush=True)
                        if failures >= max(4, len(batch) // 2):
                            raise RuntimeError("High API/format failure rate; inspect cached errors")
                for sid, sha in b_keys.items():
                    result = results[sha]
                    assert result["request_sha256"] == sha
                    assert digest(result["request"]) == sha
                    parsed = result["parsed"]
                    judged["B"][sid] = {
                        "prediction": parsed["answer"], "status": parsed["status"], "evidence": parsed["evidence"],
                        "request_sha256": sha,
                        "text_sha256": hashlib.sha256(official["per_input"]["B"][sid]["text"].encode()).hexdigest()}
                attempts_sent = sum(len(v.get("attempts", [])) for v in results.values()) - initial_attempts
            assert set(judged["A"]) == set(judged["B"]) == set(questions)
            summary = {arm: metrics(data, rows) for arm, data in judged.items()}
            transitions = {}
            for arm in ("A", "B"):
                transitions[arm] = dict(Counter(
                    f"official_{int(official['per_input'][arm][r['input_key']]['prediction'] == r['label'])}"
                    f"_deepseek_{int(judged[arm][r['input_key']]['prediction'] == r['label'])}"
                    for r in rows))
            images = sorted({r["image_sha256"] for r in rows})
            cells = ("TP", "FP", "TN", "FN", "unresolved")
            cube = np.zeros((len(images), 3, 2, 5), dtype=np.int64)
            for i, image in enumerate(images):
                for j, split in enumerate(("random", "popular", "adversarial")):
                    selected = [r for r in rows if r["image_sha256"] == image and r["split"] == split]
                    for k, arm in enumerate(("A", "B")):
                        values = pope_metrics((judged[arm][r["input_key"]]["prediction"], r["label"]) for r in selected)
                        cube[i, j, k] = [values[c] for c in cells]
            draws = np.random.default_rng(42).integers(0, len(images), size=(2000, len(images)))
            counts = cube[draws].sum(axis=1)
            accuracy = ((counts[..., 0] + counts[..., 2]) / counts.sum(axis=-1)).mean(axis=1)
            f1 = (2 * counts[..., 0] / (2 * counts[..., 0] + counts[..., 1] + counts[..., 3])).mean(axis=1)
            ci = {name: np.quantile(value[:, 1] - value[:, 0], [.025, .975]).tolist()
                  for name, value in (("accuracy", accuracy), ("f1", f1))}
            result = {
                "status": "complete", "images": 500, "unique_inputs_per_condition": 5127, "evaluation_rows_per_condition": 9000,
                "conditions": {"A": "EOS-prefix with Final-answer suffix", "B": "EOS-prefix without suffix"},
                "judge_spec": judge_spec, "judge_spec_hash": digest(judge_spec),
                "official": official["metrics"], "deepseek": summary,
                "official_paired_ci95": official["paired_image_bootstrap_B_minus_A_95ci"],
                "deepseek_paired_ci95": ci, "bootstrap": {"unit": "image", "repeats": 2000, "seed": 42},
                "official_vs_deepseek_transitions": transitions, "per_input": judged,
                "api": {"unique_B_requests": len(requests), "cache_hits": cache_hits,
                        "http_attempts_this_invocation": attempts_sent, "errors": 0,
                        "returned_models": dict(Counter(v.get("returned_model") or "unreported" for v in results.values()))},
                "sources": {str(official_path): file_hash(official_path), str(CONTROL): file_hash(CONTROL),
                            str(ROOT / "manifests/rows.json"): file_hash(ROOT / "manifests/rows.json")},
            }
            from report_scopes import scopes
            result["scope_metrics"] = scopes(judged)
            atomic_json(ROOT / "reports/fast_official_deepseek.json", result)
            lines = ["# 快速EOS前缀版：官方与DeepSeek配对评分", "",
                     "全量500张图、5127输入、每条件9000评测行。A带Final answer后缀，B仅删除该后缀。",
                     "官方：最后一个</think>后的答案，未闭合回退全文，再套固定POPE规则。",
                     "DeepSeek：沿用deepseek-flash文本提取协议；不给图像、不提供gold，提取表达的Yes/No后在本地与标签比较。",
                     "缺答/歧义、生成预算失败均保留分母计错；API错误保持待处理。本报告API错误为0。",
                     "", "| 条件 | 官方 Acc / F1 | DeepSeek Acc / F1 |",
                     "|---|---:|---:|"]
            for arm, label in (("A", "带后缀"), ("B", "去后缀")):
                a, b = official["metrics"][arm]["macro"], summary[arm]["macro"]
                lines.append(f"| {label} | {100*a['accuracy']:.2f}% / {100*a['f1']:.2f}% | "
                             f"{100*b['accuracy']:.2f}% / {100*b['f1']:.2f}% |")
            lines += ["", "去后缀−带后缀；同图像配对bootstrap 2000次，seed=42："]
            for name, values, intervals in (
                ("官方", official["metrics"], official["paired_image_bootstrap_B_minus_A_95ci"]),
                ("DeepSeek", summary, ci)):
                for metric in ("accuracy", "f1"):
                    delta = 100 * (values["B"]["macro"][metric] - values["A"]["macro"][metric])
                    bounds = intervals[metric]
                    lines.append(f"- {name} {metric}: {delta:+.2f}个百分点，95%区间[{100*bounds[0]:+.2f}, {100*bounds[1]:+.2f}]。")
            lines += ["", "A组DeepSeek结果全部复用已验证缓存；B组仅对未命中缓存的成功生成请求API。",
                      "提示协议根据64图探索选择；新增436图另列，不宣称未经测试集分析。",
                      "", "| DeepSeek范围 | 图像/评测行 | 带后缀 Acc/F1 | 新prompt Acc/F1 |",
                      "|---|---:|---:|---:|"]
            for name, item in result["scope_metrics"].items():
                a, b = item["methods"]["A"]["macro"], item["methods"]["B"]["macro"]
                lines.append(f"| {name} | {item['images']}/{item['rows']} | {100*a['accuracy']:.2f}%/{100*a['f1']:.2f}% | "
                             f"{100*b['accuracy']:.2f}%/{100*b['f1']:.2f}% |")
            (ROOT / "reports/fast_official_deepseek.md").write_text("\n".join(lines) + "\n")
            state.update(status="complete", judged_inputs_A=5127, judged_inputs_B=5127, api_errors=0,
                         report=str(ROOT / "reports/fast_official_deepseek.md"))
            save(state)
            print("FAST_DUAL_EVALUATION_COMPLETE", state["report"], flush=True)
        except BaseException as error:
            state.update(status="needs_attention", error=repr(error))
            save(state)
            raise


if __name__ == "__main__":
    main()
