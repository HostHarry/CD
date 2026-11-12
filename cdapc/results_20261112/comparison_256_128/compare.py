"""256/128 comparison: cloud Native/VCD/VIGOR records (HostHarry/CD) vs local CDAPC, re-scored row by row.

Every benchmark is scored with one rule for all methods, rows are aligned by ID with image/label checks,
and paired bootstrap intervals are computed on identical rows. Nothing here calls a model or an API.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
T = Path("/root/autodl-tmp")
CLOUD = HERE / "cloud"
sys.path.insert(0, str(T / "cdapc-hallusion/src"))
from cdapc_lab.evaluation.normalize import normalize_output
from cdapc_lab.evaluation.scorers import load_pope_official_rule, load_mmmu_official, mmmu_random_for, _StrictRandom
from cdapc_lab.experiments.score import pope_metrics
from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained(T / "cdapc-experiments/models/mmada", local_files_only=True)
POPE_RULE = load_pope_official_rule(T / "cdapc-experiments/upstream/pope/evaluate.py")
SPLITS = ("random", "popular", "adversarial")
RNG_SEED, REPEATS = 42, 2000


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def after_think(text):
    return text.rsplit("</think>", 1)[-1].strip() if "</think>" in text else text


# ----------------------------------------------------------------------------- POPE
def pope_rows(dataset):
    name = {"coco": "pope_yesno", "aokvqa": "pope_aokvqa", "gqa": "pope_gqa"}[dataset]
    return load(T / f"cdapc-pope-eos-full/data/{name}/rows.json")


def pope_cloud(record, dataset, method, rows, think):
    by_id = {r["row_id"]: r for r in rows}
    rid = lambda value: value.split(":", 1)[1] if value.startswith(dataset + ":") else value
    answers = load(CLOUD / record / dataset / method / "answers.json")
    assert len(answers) == len(rows) == 9000 and {rid(a["id"]) for a in answers} == set(by_id)
    out = {}
    for a in answers:
        a = {**a, "id": rid(a["id"])}
        r = by_id[a["id"]]
        assert (a["label"], a["image_sha256"], a["split"]) == (r["label"], r["image_sha256"], r["split"])
        assert a["question"].endswith(r["question"] + ("" if think else " Please answer yes or no."))
        out[a["id"]] = {"official": POPE_RULE(after_think(a["text"]) if think else a["text"]),
                        "seconds": float(a["end_to_end_seconds"]), "closed": "</think>" in a["text"]}
    if think:
        judged = {rid(j["id"]): j for j in load(CLOUD / record / dataset / "judge" / f"{method}_judged.json")}
        assert set(judged) == set(out)
        for rid, j in judged.items():
            assert j["label"] == by_id[rid]["label"]
            out[rid]["judge"] = j["judge_prediction"] if j["judge_status"] == "answered" else None
    return out


def pope_local_direct(dataset, rows):
    report = load(T / f"cdapc-pope-eos-full/reports/pope_{dataset}_direct_eos_prefix_r70_k4_L256_I128.json")
    preds = {}
    for run in report["runs"]:
        for p in (T / "cdapc-pope-eos-full" / run / "predictions").glob("*.json"):
            raw = load(p)
            assert raw["sample_id"] not in preds
            text = normalize_output(raw["raw_response_ids"], TOKENIZER, raw["eos_ids"])["text"]
            preds[raw["sample_id"]] = {"official": POPE_RULE(text) if raw["status"] == "success" else None,
                                       "seconds": raw["generation_seconds"]}
    assert set(preds) == {r["input_key"] for r in rows}
    return {r["row_id"]: preds[r["input_key"]] for r in rows}


def pope_local_think(dataset, rows):
    if dataset == "coco":
        root = T / "cdapc-pope-coco-prefix-no-final/reports"
        official = load(root / "comparison.json")["per_input"]["B"]
        judge = load(root / "fast_official_deepseek.json")
        assert judge["judge_spec_hash"] == "7c5198337359efe544788233f9c1ab3a40fc6569dd34b453bd79e33f1e80a9fd"
        judge = judge["per_input"]["B"]
        text = {k: (v["text"], v["status"], v["generation_seconds"]) for k, v in official.items()}
    else:
        root = T / f"cdapc-pope-no-final-suite/reports/{dataset}"
        official = load(root / "official.json")["per_input"]
        judge_report = load(root / "deepseek.json")
        assert judge_report["judge_spec_hash"] == "7c5198337359efe544788233f9c1ab3a40fc6569dd34b453bd79e33f1e80a9fd"
        judge = judge_report["per_input"]
        text = {k: (v["text"], v["generation_status"], v["generation_seconds"]) for k, v in official.items()}
    assert set(text) == set(judge) == {r["input_key"] for r in rows}
    out = {}
    for r in rows:
        t, status, seconds = text[r["input_key"]]
        ok = status == "success"
        out[r["row_id"]] = {"official": POPE_RULE(after_think(t)) if ok else None,
                            "judge": judge[r["input_key"]]["prediction"] if ok else None,
                            "seconds": seconds, "closed": "</think>" in t}
    return out


def pope_score(rows, preds, key):
    splits = {s: pope_metrics((preds[r["row_id"]][key], r["label"]) for r in rows if r["split"] == s) for s in SPLITS}
    return {"splits": splits, **{m: float(np.mean([splits[s][m] for s in SPLITS])) for m in ("accuracy", "f1", "yes_ratio")}}


def pope_bootstrap(rows, a, b, key):
    """Image-level bootstrap of split-mean Acc/F1 (A - B). Unresolved predictions are wrong:
    a Yes-label unresolved row is a FN, a No-label unresolved row is neither TP/FP/FN (as in pope_metrics)."""
    images = sorted({r["image_sha256"] for r in rows})
    index = {img: i for i, img in enumerate(images)}
    cube = np.zeros((len(images), 3, 2, 5))  # correct, n, TP, FP, FN
    for r in rows:
        for k, preds in enumerate((a, b)):
            p, g = preds[r["row_id"]][key], r["label"]
            cube[index[r["image_sha256"]], SPLITS.index(r["split"]), k] += [
                p == g, 1, p == "yes" and g == "yes", p == "yes" and g == "no", p != "yes" and g == "yes"]
    draws = np.random.default_rng(RNG_SEED).integers(0, len(images), size=(REPEATS, len(images)))
    c = cube[draws].sum(1)
    acc = c[..., 0] / c[..., 1]
    tp, fp, fn = c[..., 2], c[..., 3], c[..., 4]
    f1 = 2 * tp / (2 * tp + fp + fn)
    out = {}
    for name, value in (("accuracy", acc), ("f1", f1)):
        out[name] = np.quantile((value[:, :, 0] - value[:, :, 1]).mean(1), [0.025, 0.975]).tolist()
    return out


def pope_section():
    result = {}
    for record, think, label in (("pope_yesno_256_128_32", False, "POPE 直答（yes/no后缀）"),
                                 ("pope_think_256_128_64", True, "POPE think（无后缀，</think>后作答）")):
        result[label] = {}
        for dataset in ("coco", "aokvqa", "gqa"):
            if not (CLOUD / record / dataset).exists():
                result[label][dataset] = {"cloud": "not available (still running in cloud)"}
                continue
            rows = pope_rows(dataset)
            methods = {m: pope_cloud(record, dataset, m, rows, think) for m in ("native", "vcd", "vigor")}
            methods["cdapc"] = pope_local_think(dataset, rows) if think else pope_local_direct(dataset, rows)
            keys = ("official", "judge") if think else ("official",)
            block = {"rows": len(rows), "metrics": {}, "paired_vs_cdapc": {}, "seconds": {}}
            for key in keys:
                block["metrics"][key] = {m: pope_score(rows, p, key) for m, p in methods.items()}
                block["paired_vs_cdapc"][key] = {m: pope_bootstrap(rows, methods["cdapc"], methods[m], key)
                                                 for m in ("native", "vcd", "vigor")}
            if think:
                block["closed_think"] = {m: sum(v["closed"] for v in p.values()) / len(p) for m, p in methods.items()}
            block["seconds"] = {m: float(np.mean([v["seconds"] for v in p.values()])) for m, p in methods.items()}
            published = load(CLOUD / record / dataset / ("primary_metrics.json" if think else "metrics.json"))
            for m in ("native", "vcd", "vigor"):
                splits = published[m]["splits"] if think else published[m]["pope"]
                for s in SPLITS:
                    for metric in ("accuracy", "f1"):
                        ours = block["metrics"]["official"][m]["splits"][s][metric]
                        assert abs(ours - splits[s][metric]) < 1e-9, (record, dataset, m, s, metric, ours, splits[s][metric])
            block["published_check"] = "cloud Native/VCD/VIGOR official Acc/F1 reproduced exactly per split"
            result[label][dataset] = block
    return result


# ----------------------------------------------------------------------------- MMMU
def mmmu_section():
    gold = {g["sample_id"]: g for g in load(T / "cdapc-experiments/data/mmmu_val/gold.json")}
    samples = {s["sample_id"]: s for s in load(T / "cdapc-experiments/data/mmmu_val/samples.json")}
    make = load_mmmu_official(T / "cdapc-experiments/upstream/mmmu/eval/eval_utils.py",
                              T / "cdapc-experiments/upstream/mmmu/eval/data_utils.py")
    methods = {}
    hist = load(T / "cdapc-experiments/reports/mmmu_cdapc_r70_L256_I128.json")
    k4 = load(T / "cdapc-mmmu-prefix-k4-l256-i128/reports/result.json")
    for name, block in (("native(本地)", hist["native"]), ("cdapc_k16", hist["cdapc"]), ("cdapc_k4", k4["cdapc"])):
        methods[name] = {sid: {m: float(v[m]) for m in ("official", "strict", "judge")} for sid, v in block["per_sample"].items()}
    cloud_judge = {m: {j["id"]: float(j["score"]) for j in load(CLOUD / f"mmmu_wino_v1/mmmu_val/judge/{m}_judged.json")}
                   for m in ("native", "vcd", "vigor")}
    for m in ("vcd", "vigor"):
        rows = {}
        for a in load(CLOUD / f"mmmu_wino_v1/mmmu_val/{m}/answers.json"):
            sid = a["id"]
            g = gold[sid]
            options = samples[sid]["options"]
            qtype = g["metric_metadata"]["question_type"]
            official = make(mmmu_random_for(sid))(a["text"], options, g["answer"], qtype)
            strict = make(_StrictRandom())(a["text"], options, g["answer"], qtype)
            rows[sid] = {"official": float(official["correct"]), "strict": float(strict["correct"]),
                         "judge": cloud_judge[m][sid]}
        methods[m] = rows
    methods["native(云端judge)"] = {sid: {**methods["native(本地)"][sid], "judge": cloud_judge["native"][sid]}
                                   for sid in methods["native(本地)"]}
    ids = sorted(gold)
    assert all(set(v) == set(ids) for v in methods.values())
    metrics = {m: {k: float(np.mean([v[s][k] for s in ids])) for k in ("official", "strict", "judge")} for m, v in methods.items()}
    rng = np.random.default_rng(RNG_SEED)
    draws = rng.integers(0, len(ids), size=(REPEATS, len(ids)))
    paired = {}
    for m in ("native(本地)", "vcd", "vigor", "cdapc_k16"):
        paired[m] = {}
        for k in ("official", "strict", "judge"):
            a = np.array([methods["cdapc_k4"][s][k] for s in ids]); b = np.array([methods[m][s][k] for s in ids])
            paired[m][k] = {"delta": float(a.mean() - b.mean()), "ci95": np.quantile((a[draws] - b[draws]).mean(1), [0.025, 0.975]).tolist()}
    return {"rows": len(ids), "metrics": metrics, "paired_cdapc_k4_minus": paired,
            "note": "Official/strict re-scored with the local pinned MMMU scorer (per-question random fallback) for all methods; "
                    "judge = identical DeepSeek spec ffda0193; cloud VCD/VIGOR/native judged in one batch, local CDAPC/native separately"}


# ----------------------------------------------------------------------------- HallusionBench / MME
def hallusion_section():
    root = T / "cdapc-hallusion/reports"
    if (root / "result.json").exists():
        r = load(root / "result.json")
        return {"status": "complete", "arms": {a: {**b["metrics"], "image": b["by_input_mode"]["image"]["aAcc"]} for a, b in r["arms"].items()},
                "baseline": r["baseline"], "paired": r["paired"]}
    r = load(root / "interim_direct.json")
    return {"status": "interim: direct arms only (think arms still generating)",
            "arms": {a: {**{k: b["metrics"][k] for k in ("aAcc", "fAcc", "qAcc", "Avg", "unknown")}, "image": b["by_input_mode"]["image"]["aAcc"]}
                     for a, b in r["arms"].items()},
            "baseline": {m: {**{k: b[k] for k in ("aAcc", "fAcc", "qAcc", "Avg", "unknown")}, "image": b["by_input_mode"]["image"]["aAcc"]}
                         for m, b in r["baseline"].items()},
            "paired": r["paired"]}


def mme_section():
    out = {}
    for name, path in (("CDAPC 直答 K4", T / "cdapc-mme-direct-prefix-k4/reports/result.json"),
                       ("CDAPC think K4", T / "cdapc-mme-think-prefix-k4/reports/result.json"),
                       ("CDAPC 直答 K16（历史）", T / "cdapc-mme/reports/mme_cdapc_r70_L256_I128.json")):
        m = load(path)["metrics"]
        out[name] = {k: m[k] for k in ("perception", "cognition", "total")}
    out["paired"] = load(T / "cdapc-mme-direct-prefix-k4/reports/result.json")["paired_image_bootstrap"]
    return out


def main():
    report = {"cloud_commit": (CLOUD / "COMMIT").read_text().strip(), "bootstrap": {"repeats": REPEATS, "seed": RNG_SEED},
              "pope": pope_section(), "mmmu": mmmu_section(), "hallusion": hallusion_section(), "mme": mme_section()}
    (HERE / "comparison_256_128.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k in ("cloud_commit",)}))


if __name__ == "__main__":
    main()
