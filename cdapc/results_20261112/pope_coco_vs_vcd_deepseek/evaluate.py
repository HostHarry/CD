"""Frozen matched CDAPC/VCD comparison using the existing DeepSeek protocol."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import fcntl
from pathlib import Path
import os
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "cdapc-pope-coco-prefix-no-final"
CACHE = ROOT.parent / "cdapc-experiments/reports/pope_think_llm_judge/cache"
sys.path.insert(0, str(SOURCE / "src"))
from cdapc_lab.evaluation.llm_judge import judge_key
from cdapc_lab.evaluation.normalize import normalize_output
from cdapc_lab.evaluation.pope_llm_judge import judge_request, request_for, spec
from cdapc_lab.experiments.score import pope_metrics
from cdapc_lab.io.artifacts import atomic_json, digest, file_hash, read_json
from cdapc_lab.models.mmada import OFFICIAL_THINKING_INSTRUCTION

ARMS = ("CDAPC", "VCD")
SPLITS = ("random", "popular", "adversarial")


def save(state):
    state["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    atomic_json(ROOT / "reports/status.json", state)


def prepare():
    from transformers import AutoTokenizer
    plan = read_json(ROOT / "reports/plan.json")
    rows = read_json(ROOT / "inputs/rows.json")
    allrows = read_json(SOURCE / "manifests/rows.json")
    rowmap = {r["row_id"]: r for r in allrows}
    baseline = ROOT / "inputs/vcd_answers.json"
    assert file_hash(baseline) == "f6d6d3dbd367fa21da8a54cfe05c5fca1799bcd1610ce9d095944ab29f2b4b51"
    vcd_rows = read_json(baseline)
    assert len(vcd_rows) == len(rowmap) == 9000
    assert {r["id"] for r in vcd_rows} == set(rowmap)
    vcd = {}
    for answer in vcd_rows:
        row = rowmap[answer["id"]]
        assert answer["split"] == row["split"] and answer["label"] == row["label"]
        assert answer["image_sha256"] == row["image_sha256"]
        assert answer["question"] == OFFICIAL_THINKING_INSTRUCTION + "\n" + row["question"]
        assert answer["remaining_masks"] == 0, "VCD has unresolved visible masks"
        sid = row["input_key"]
        if sid in vcd:
            assert vcd[sid]["text"] == answer["text"], "VCD duplicate-input responses differ"
        vcd[sid] = answer
    assert len(vcd) == 5127 and set(plan["predictions"]) <= set(vcd)
    questions = {r["input_key"]: r["question"] for r in rows}
    gold = {r["input_key"]: r["label"] for r in rows}
    tokenizer = AutoTokenizer.from_pretrained(SOURCE / "models/mmada", local_files_only=True)
    text_tokens, records = {}, {arm: {} for arm in ARMS}
    for sid, item in plan["predictions"].items():
        assert file_hash(item["path"]) == item["sha256"]
        raw = read_json(item["path"])
        assert raw["sample_id"] == sid and raw["status"] == item["status"]
        question = questions[sid]
        if question not in text_tokens:
            text_tokens[question] = tokenizer.apply_chat_template(
                [{"role": "user", "content": OFFICIAL_THINKING_INSTRUCTION + "\n" + question}],
                tokenize=True, add_generation_prompt=True)
        assert raw["prompt_ids"][0][1027:] == text_tokens[question]
        text = normalize_output(raw["raw_response_ids"], tokenizer, raw["eos_ids"])["text"]
        records["CDAPC"][sid] = {
            "question": question, "text": text, "gold": gold[sid],
            "generation_status": raw["status"], "source_path": item["path"],
            "source_sha256": item["sha256"], "seconds": raw["generation_seconds"],
            "iterations": raw["decode_iterations"]}
        other = vcd[sid]
        records["VCD"][sid] = {
            "question": question, "text": other["text"], "gold": gold[sid],
            "generation_status": "success", "source_row_id": other["id"],
            "source_path": str(baseline), "seconds": other["decode_seconds"],
            "iterations": other["executed_steps"]}
    for arm in ARMS:
        assert set(records[arm]) == set(questions)
        for value in records[arm].values():
            value["request_sha256"] = digest(request_for(value["question"], value["text"])) \
                if value["generation_status"] == "success" else None
            value["text_sha256"] = digest(value["text"])
    pilot = read_json(ROOT.parent / "cdapc-pope-prompt-2x2/reports/fast_official_deepseek.json")
    assert pilot["judge_spec_hash"] == digest(spec())
    for sid, item in pilot["per_input"]["B"].items():
        assert records["CDAPC"][sid]["request_sha256"] == item["request_sha256"]
    checkpoint = read_json(ROOT / "inputs/vcd_manifest.json")["checkpoints"]
    metadata_match = {}
    for field, path in (("checkpoint", SOURCE / "models/mmada"), ("vq_checkpoint", SOURCE / "models/magvitv2")):
        checks = {name: file_hash(path / name) == sha
                  for name, sha in checkpoint[field]["metadata_sha256"].items()}
        assert all(checks.values()), checks
        metadata_match[field] = checks
    atomic_json(ROOT / "inputs/matched_inputs.json", records)
    atomic_json(ROOT / "reports/verification.json", {
        "all_9000_baseline_rows_match_image_hash_question_label": True,
        "baseline_unique_inputs": len(vcd), "matched_inputs": len(questions), "matched_rows": len(rows),
        "all_cdapc_prompt_text_tokens_verified": True, "model_metadata_match": metadata_match,
        "model_weight_hash_limit": "VCD manifest publishes weight sizes/mtime, not weight content hashes",
        "prompt_alignment": "Official MMaDA thinking instruction plus original POPE question; no extra suffix",
        "seed_difference": {"CDAPC": "20261105, derived per sample", "VCD": "42, reset per unique input/method"},
        "comparison_scope": "Reported configurations, not a seed-matched isolated decoder intervention",
        "baseline_commit": "fcc76d6", "baseline_sha256": file_hash(baseline)})
    paths = [ROOT / "reports/plan.json", ROOT / "reports/verification.json", Path(__file__),
             *sorted((ROOT / "inputs").glob("*"))]
    atomic_json(ROOT / "reports/source_snapshot.json",
                {str(p.relative_to(ROOT)): file_hash(p) for p in paths if p.is_file()})
    print("MATCHED_INPUTS_VERIFIED", len(questions), len(rows), flush=True)


def metrics(records, rows):
    per_split = {s: pope_metrics((records[r["input_key"]]["prediction"], r["label"])
                                for r in rows if r["split"] == s) for s in SPLITS}
    pooled = pope_metrics((records[r["input_key"]]["prediction"], r["label"]) for r in rows)
    return {
        "splits": per_split, "pooled": pooled,
        "macro": {m: sum(v[m] for v in per_split.values()) / 3
                  for m in ("accuracy", "f1", "precision", "recall", "yes_ratio")},
        "judge_statuses_unique": dict(Counter(v["judge_status"] for v in records.values())),
        "judge_unresolved_rows": sum(records[r["input_key"]]["prediction"] is None for r in rows)}


def write_report(inputs, judged, api):
    rows = read_json(ROOT / "inputs/rows.json")
    plan = read_json(ROOT / "reports/plan.json")
    methods = {arm: metrics(judged[arm], rows) for arm in ARMS}
    transitions = Counter()
    for row in rows:
        a, b = [int(judged[arm][row["input_key"]]["prediction"] == row["label"]) for arm in ARMS]
        transitions[f"CDAPC_{a}_VCD_{b}"] += 1
    images = sorted({r["image_sha256"] for r in rows})
    cube = np.zeros((len(images), 3, 2, 5), dtype=np.int64)
    for i, image in enumerate(images):
        for j, split in enumerate(SPLITS):
            selected = [r for r in rows if r["image_sha256"] == image and r["split"] == split]
            for k, arm in enumerate(ARMS):
                value = pope_metrics((judged[arm][r["input_key"]]["prediction"], r["label"]) for r in selected)
                cube[i, j, k] = [value[c] for c in ("TP", "FP", "TN", "FN", "unresolved")]
    draws = np.random.default_rng(42).integers(0, len(images), size=(2000, len(images)))
    counts = cube[draws].sum(axis=1)
    acc = (counts[..., 0] + counts[..., 2]) / counts.sum(axis=-1)
    f1 = 2 * counts[..., 0] / (2 * counts[..., 0] + counts[..., 1] + counts[..., 3])
    intervals = {}
    for name, values in (("accuracy", acc), ("f1", f1)):
        intervals[name] = {"macro": np.quantile((values[..., 0] - values[..., 1]).mean(axis=1), [.025, .975]).tolist(),
                           **{s: np.quantile(values[:, j, 0] - values[:, j, 1], [.025, .975]).tolist()
                              for j, s in enumerate(SPLITS)}}
    formatting = {arm: {
        "closed_unique": sum("</think>" in v["text"] for v in inputs[arm].values()),
        "generation_statuses_unique": dict(Counter(v["generation_status"] for v in inputs[arm].values())),
        "mean_seconds": statistics.mean(v["seconds"] for v in inputs[arm].values()),
        "timing_note": "Different runs; no controlled hardware timing claim"}
        for arm in ARMS}
    result = {
        "status": "complete", "frozen_at": plan["frozen_at"], "inputs": plan["inputs"], "rows": len(rows),
        "images": len(images), "split_rows": plan["split_rows"], "gold_labels": plan["gold_labels"],
        "scope": "Full matched COCO, 5127 inputs and 9000 rows, all generation and answer failures retained",
        "judge_spec": spec(), "judge_spec_hash": digest(spec()), "metrics": methods,
        "paired_ci95_CDAPC_minus_VCD": intervals, "paired_correctness_transitions_rows": dict(transitions),
        "bootstrap": {"unit": "image", "repeats": 2000, "seed": 42},
        "formatting": formatting, "api": api, "per_input": judged,
        "sources": {str(ROOT / "reports/source_snapshot.json"): file_hash(ROOT / "reports/source_snapshot.json"),
                    str(ROOT / "reports/verification.json"): file_hash(ROOT / "reports/verification.json")},
        "selection_note": "All three splits complete and balanced. Prompt selected after the 64-image pilot; this is not an untouched benchmark."}
    atomic_json(ROOT / "reports/comparison.json", result)
    lines = ["# COCO全量：CDAPC新prompt vs VCD，统一DeepSeek评分", "",
             f"冻结时间：{plan['frozen_at']}。{plan['inputs']}个唯一输入，{len(rows)}行，{len(images)}张图。",
             "两种方法在同题、同图像哈希、同标签上比较；均为think指令＋问题，无额外Final answer要求。",
             "DeepSeek只提取实际表达的Yes/No，不看图、不给gold；本地与标签比较。生成失败/缺答/歧义均计错。",
             "", "| 子集 | 行数 | Yes标签比例 | CDAPC Acc/F1 | VCD Acc/F1 |",
             "|---|---:|---:|---:|---:|"]
    for split in SPLITS:
        subset = [r for r in rows if r["split"] == split]
        a, b = [methods[arm]["splits"][split] for arm in ARMS]
        lines.append(f"| {split} | {len(subset)} | {100*sum(r['label']=='yes' for r in subset)/len(subset):.2f}% | "
                     f"{100*a['accuracy']:.2f}%/{100*a['f1']:.2f}% | {100*b['accuracy']:.2f}%/{100*b['f1']:.2f}% |")
    for name in ("macro", "pooled"):
        a, b = [methods[arm][name] for arm in ARMS]
        lines.append(f"| {name} | — | — | {100*a['accuracy']:.2f}%/{100*a['f1']:.2f}% | "
                     f"{100*b['accuracy']:.2f}%/{100*b['f1']:.2f}% |")
    lines += ["", "图像配对bootstrap 2000次，seed=42；差值=CDAPC−VCD：", ""]
    for metric in ("accuracy", "f1"):
        delta = 100*(methods["CDAPC"]["macro"][metric]-methods["VCD"]["macro"][metric])
        lo, hi = intervals[metric]["macro"]
        lines.append(f"- macro {metric}: {delta:+.2f}个百分点，95%区间[{100*lo:+.2f}, {100*hi:+.2f}]。")
    lines += ["", "三个split各完整3000行；每组1500个Yes和1500个No。提示协议在64图探索之后选择。",
              "两方法使用各自已记录配置和seed（CDAPC派生自20261105；VCD每题固定42），不是仅改变解码算法的单变量干预。",
              "此前的部分报告保留不变；本报告覆盖全部500张图和9000行。"]
    (ROOT / "reports/comparison.md").write_text("\n".join(lines) + "\n")


def main():
    if "--prepare" in sys.argv:
        prepare()
        return
    state = {"status": "waiting_for_cache_lock", "controller_pid": os.getpid(), "workers": 8}
    with (ROOT / "logs/runner.lock").open("a") as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            snapshot = read_json(ROOT / "reports/source_snapshot.json")
            assert all(file_hash(ROOT / path) == sha for path, sha in snapshot.items())
            inputs = read_json(ROOT / "inputs/matched_inputs.json")
            requests = {v["request_sha256"]: request_for(v["question"], v["text"])
                        for arm in ARMS for v in inputs[arm].values() if v["request_sha256"]}
            state["unique_requests"] = len(requests)
            save(state)
            with (CACHE.parent / "runner.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                results = {sha: read_json(CACHE / (sha+".json"))
                           for sha in requests if (CACHE / (sha+".json")).exists()}
                cache_hits = sum(v["status"] == "success" for v in results.values())
                before = sum(len(v.get("attempts", [])) for v in results.values())
                key = judge_key()
                state.update(status="judging", cache_hits=cache_hits)
                save(state)
                while True:
                    pending = [sha for sha in requests if results.get(sha, {}).get("status") != "success"]
                    if not pending:
                        break
                    if not key:
                        raise RuntimeError("DeepSeek credential unavailable")
                    if any(len(results.get(sha, {}).get("attempts", [])) >= 9 for sha in pending):
                        raise RuntimeError("Judge request failed nine attempts; remains pending")
                    for offset in range(0, len(pending), 64):
                        batch = pending[offset:offset+64]
                        failures = 0
                        with ThreadPoolExecutor(max_workers=8) as pool:
                            futures = {pool.submit(judge_request, requests[sha], CACHE, key): sha for sha in batch}
                            for future in as_completed(futures):
                                sha = futures[future]
                                results[sha] = future.result()
                                failures += results[sha]["status"] != "success"
                        state.update(succeeded=sum(v["status"] == "success" for v in results.values()),
                                     pending=sum(results.get(sha, {}).get("status") != "success" for sha in requests),
                                     api_errors=sum(v["status"] == "judge_error" for v in results.values()))
                        save(state)
                        print("JUDGE_PROGRESS", state["succeeded"], "/", len(requests), flush=True)
                        if failures >= max(4, len(batch)//2):
                            raise RuntimeError("High API/format failure rate")
                judged = {arm: {} for arm in ARMS}
                for arm in ARMS:
                    for sid, item in inputs[arm].items():
                        sha = item["request_sha256"]
                        parsed = {"answer": None, "status": item["generation_status"], "evidence": ""}
                        if sha:
                            result = results[sha]
                            assert result["request_sha256"] == digest(result["request"]) == sha
                            parsed = result["parsed"]
                        judged[arm][sid] = {
                            "prediction": parsed["answer"], "judge_status": parsed["status"],
                            "evidence": parsed["evidence"], "request_sha256": sha,
                            "text_sha256": item["text_sha256"], "generation_status": item["generation_status"]}
                api = {"unique_requests": len(requests), "cache_hits": cache_hits,
                       "http_attempts_this_invocation": sum(len(v.get("attempts", [])) for v in results.values())-before,
                       "errors": 0, "returned_models": dict(Counter(v.get("returned_model") or "unreported"
                                                                  for v in results.values()))}
            state["status"] = "summarizing"
            save(state)
            write_report(inputs, judged, api)
            state.update(status="complete", report="reports/comparison.md", api_errors=0)
            save(state)
            print("MATCHED_DEEPSEEK_COMPARISON_COMPLETE", flush=True)
        except BaseException as error:
            state.update(status="needs_attention", error=repr(error))
            save(state)
            raise


if __name__ == "__main__":
    main()
