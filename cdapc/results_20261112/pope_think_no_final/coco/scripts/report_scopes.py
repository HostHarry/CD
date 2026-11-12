"""Report full COCO and the additional images separately after prompt selection."""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cdapc_lab.experiments.score import pope_metrics
from cdapc_lab.io.artifacts import atomic_json, read_json


def scopes(records):
    rows = read_json(ROOT / "manifests/rows.json")
    pilot = set(read_json(ROOT / "reports/plan.json")["pilot_images"])
    result = {}
    for name, selected in (("full", rows), ("additional", [r for r in rows if r["image_sha256"] not in pilot])):
        images = sorted({r["image_sha256"] for r in selected})
        methods = {}
        cube = np.zeros((len(images), 3, 2, 5), dtype=np.int64)
        for k, arm in enumerate(("A", "B")):
            splits = {s: pope_metrics((records[arm][r["input_key"]]["prediction"], r["label"])
                                      for r in selected if r["split"] == s)
                      for s in ("random", "popular", "adversarial")}
            methods[arm] = {"splits": splits, "macro": {
                m: sum(v[m] for v in splits.values()) / 3
                for m in ("accuracy", "f1", "precision", "recall", "yes_ratio")}}
            for i, image in enumerate(images):
                for j, split in enumerate(("random", "popular", "adversarial")):
                    values = pope_metrics((records[arm][r["input_key"]]["prediction"], r["label"])
                                          for r in selected if r["image_sha256"] == image and r["split"] == split)
                    cube[i, j, k] = [values[c] for c in ("TP", "FP", "TN", "FN", "unresolved")]
        draws = np.random.default_rng(42).integers(0, len(images), size=(2000, len(images)))
        counts = cube[draws].sum(axis=1)
        acc = ((counts[..., 0] + counts[..., 2]) / counts.sum(axis=-1)).mean(axis=1)
        f1 = (2 * counts[..., 0] / (2 * counts[..., 0] + counts[..., 1] + counts[..., 3])).mean(axis=1)
        result[name] = {
            "images": len(images), "inputs": len({r["input_key"] for r in selected}), "rows": len(selected),
            "methods": methods, "paired_ci95_B_minus_A": {
                m: np.quantile(v[:, 1] - v[:, 0], [.025, .975]).tolist() for m, v in (("accuracy", acc), ("f1", f1))}}
    assert result["full"]["rows"] == 9000 and result["additional"]["rows"] == 7848
    return result


def write_official_scopes():
    source = read_json(ROOT / "reports/comparison.json")
    result = scopes(source["per_input"])
    atomic_json(ROOT / "reports/official_scopes.json", result)
    lines = ["# COCO官方评分：全量与新增部分", "",
             "提示协议根据64图探索选择。新增436图未参与该小样本效应估计，但不宣称整个研究未经测试集分析。",
             "", "| 范围 | 图像/评测行 | 旧后缀 Acc/F1 | 新prompt Acc/F1 |",
             "|---|---:|---:|---:|"]
    for name, item in result.items():
        a, b = item["methods"]["A"]["macro"], item["methods"]["B"]["macro"]
        lines.append(f"| {name} | {item['images']}/{item['rows']} | {100*a['accuracy']:.2f}%/{100*a['f1']:.2f}% | "
                     f"{100*b['accuracy']:.2f}%/{100*b['f1']:.2f}% |")
    (ROOT / "reports/official_scopes.md").write_text("\n".join(lines) + "\n")
