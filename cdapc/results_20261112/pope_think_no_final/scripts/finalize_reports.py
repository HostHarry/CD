"""Wait for complete grading, independently audit counts, and write the final summary."""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(records, rows):
    result = {}
    for split in ("random", "popular", "adversarial"):
        selected = [r for r in rows if r["split"] == split]
        assert len(selected) == 3000
        assert Counter(r["label"] for r in selected) == {"yes": 1500, "no": 1500}
        pairs = [(records[r["input_key"]]["prediction"], r["label"]) for r in selected]
        tp = sum(p == g == "yes" for p, g in pairs)
        fp = sum(p == "yes" and g == "no" for p, g in pairs)
        fn = sum(p != "yes" and g == "yes" for p, g in pairs)
        result[split] = {"accuracy": sum(p == g for p, g in pairs) / len(pairs),
                         "f1": 2 * tp / (2 * tp + fp + fn)}
    return result


deadline = time.monotonic() + 2700
previous = None
while time.monotonic() < deadline:
    state = read(ROOT / "reports/status.json")
    if state["status"] != previous:
        print("SUITE_PHASE", state["status"], datetime.now().astimezone().isoformat(timespec="seconds"), flush=True)
        previous = state["status"]
    if state["status"] == "complete":
        break
    if state["status"] == "needs_attention":
        raise RuntimeError(state.get("error"))
    time.sleep(15)
else:
    raise TimeoutError("Generation/scoring still running; inspect status before finalizing")

snapshot = read(ROOT / "reports/source_snapshot.json")
assert all(sha(ROOT / p) == value for p, value in snapshot.items())
plan = read(ROOT / "reports/plan.json")
coco_root = Path(plan["source_reference"])
coco_official = read(coco_root / "reports/comparison.json")
coco_judge = read(coco_root / "reports/fast_official_deepseek.json")
summary = {"COCO": {"official": coco_official["metrics"]["B"]["macro"],
                    "deepseek": coco_judge["deepseek"]["B"]["macro"],
                    "inputs": 5127, "rows": 9000}}
receipts = {}
sources = [coco_root / "reports/comparison.json", coco_root / "reports/fast_official_deepseek.json",
           ROOT / "reports/source_snapshot.json", Path(__file__)]
for dataset in plan["order"]:
    folder = ROOT / "reports" / dataset
    rows = read(ROOT / f"manifests/{dataset}_rows.json")
    ids = {r["input_key"] for r in rows}
    assert len(rows) == 9000 and len(ids) == plan["stages"][dataset]["inputs"]
    inventory = read(folder / "inventory.json")
    official = read(folder / "official.json")
    judge = read(folder / "deepseek.json")
    assert set(inventory) == set(official["per_input"]) == set(judge["per_input"]) == ids
    assert official["status"] == judge["status"] == "complete"
    assert judge["official_sha256"] == sha(folder / "official.json")
    assert judge["judge_spec_hash"] == coco_judge["judge_spec_hash"]
    assert judge["api"]["errors"] == 0
    active = read(folder / "active_preflight.json")
    allowed = {v["generation_fingerprint"] for v in active.values()}
    generation = Counter()
    for sid, record in inventory.items():
        path = Path(record["path"])
        raw = read(path)
        assert raw["sample_id"] == sid
        assert raw["generation_fingerprint"] in allowed
        assert sha(path) == record["sha256"] == official["per_input"][sid]["source_sha256"]
        assert raw["status"] == official["per_input"][sid]["generation_status"]
        generation[raw["status"]] += 1
        if raw["status"] != "success":
            assert official["per_input"][sid]["prediction"] is None
            assert judge["per_input"][sid]["prediction"] is None
        if judge["per_input"][sid]["status"] != "answered":
            assert judge["per_input"][sid]["prediction"] is None
    for result in (official, judge):
        recounted = metrics(result["per_input"], rows)
        for split, values in recounted.items():
            for name, value in values.items():
                assert abs(value - result["metrics"]["splits"][split][name]) < 1e-12
        for name in ("accuracy", "f1"):
            assert abs(sum(v[name] for v in recounted.values()) / 3 - result["metrics"]["macro"][name]) < 1e-12
    label = "A-OKVQA" if dataset == "aokvqa" else "GQA"
    summary[label] = {"official": official["metrics"]["macro"], "deepseek": judge["metrics"]["macro"],
                      "inputs": len(ids), "rows": len(rows)}
    receipts[dataset] = {"inputs": len(ids), "rows": len(rows), "images": len({r["image_sha256"] for r in rows}),
                         "generation_statuses": dict(generation), "judge_statuses": judge["statuses"],
                         "mean_generation_seconds": official["mean_seconds"],
                         "closed_unique": official["closed_unique"], "api": judge["api"],
                         "all_prediction_hashes_verified": True, "fixed_denominator_metrics_recounted": True}
    sources += [folder / name for name in ("official.json", "deepseek.json", "inventory.json", "summary.md", "active_preflight.json")]

receipt = {"status": "complete", "verified_at": datetime.now().astimezone().isoformat(timespec="seconds"),
           "source_snapshot_verified": True, "datasets": receipts,
           "sources": {str(p): sha(p) for p in sources}}
(ROOT / "reports/completion_verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
(ROOT / "reports/three_dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
lines = ["# 新prompt快速EOS：COCO / A-OKVQA / GQA", "",
         "官方MMaDA think指令＋原问题，不加Final answer要求；r70/W16/K4/L256/I128。",
         "每个数据集500张图、9000行；Acc/F1均为三个split等权平均。", "",
         "| 数据集 | 唯一输入 | 官方Acc / F1 | DeepSeek Acc / F1 |",
         "|---|---:|---:|---:|"]
for dataset, value in summary.items():
    a, b = value["official"], value["deepseek"]
    lines.append(f"| {dataset} | {value['inputs']} | {100*a['accuracy']:.2f}% / {100*a['f1']:.2f}% | "
                 f"{100*b['accuracy']:.2f}% / {100*b['f1']:.2f}% |")
lines += ["", "A-OKVQA和GQA全量输出哈希、生成指纹、分母、逐split及平均Acc/F1已核验。",
          "预算耗尽、缺答和歧义始终保留分母。API错误为0；请求或格式重试次数保留在各组JSON中。",
          "COCO沿用其冻结seed 20261105；A-OKVQA/GQA沿用各自原配置seed 42和数据版本。",
          "旧prompt输出保持不变；本队列不自动重启其他实验。", ""]
for dataset, value in receipts.items():
    lines.append(f"- {dataset}: generation={value['generation_statuses']}; judge={value['judge_statuses']}; "
                 f"平均生成{value['mean_generation_seconds']:.2f}秒/输入。")
lines += ["", "详情：各数据集目录下official.json、deepseek.json、inventory.json和summary.md；",
          "审计：completion_verification.json与final.sha256。"]
(ROOT / "reports/final_summary.md").write_text("\n".join(lines) + "\n")
files = sources + [ROOT / "reports/completion_verification.json", ROOT / "reports/three_dataset_summary.json",
                   ROOT / "reports/final_summary.md"]
(ROOT / "reports/final.sha256").write_text("".join(f"{sha(p)}  {p}\n" for p in files))
print("FINAL_REPORTS_VERIFIED", json.dumps(summary), flush=True)
