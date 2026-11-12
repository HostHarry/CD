"""Interim score of the completed direct arms, reusing the frozen run.py scoring functions."""
import json
import sys
from collections import Counter
from pathlib import Path
import statistics

sys.argv = [sys.argv[0]]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run
from transformers import AutoTokenizer
from cdapc_lab.evaluation.normalize import normalize_output
from cdapc_lab.io.artifacts import atomic_json, file_hash, read_json

ROOT = run.ROOT
ARMS = ("cdapc_direct", "native_direct")


def main():
    assert all(file_hash(ROOT / p) == sha for p, sha in read_json(ROOT / "reports/source_snapshot.json").items())
    rows = read_json(ROOT / "data/hallusion/rows.json")
    by_index = {r["index"]: r for r in rows}
    extract, rating = run.official_functions()
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/mmada", local_files_only=True)
    texts, info = {}, {}
    for arm in ARMS:
        inventory = read_json(ROOT / f"reports/{arm}_inventory.json")
        assert len(inventory) == run.read_json(ROOT / "reports/preflight.json")["arms"][arm]["inputs"]
        texts[arm], info[arm] = {}, {}
        for sid, item in inventory.items():
            assert file_hash(item["path"]) == item["sha256"]
            raw = read_json(item["path"])
            expected = tokenizer.apply_chat_template([{"role": "user", "content": by_index[sid]["question"]}],
                                                     tokenize=True, add_generation_prompt=True)
            ids = raw["prompt_ids"][0]
            assert ids[-len(expected):] == expected and len(ids) == len(expected) + (1027 if by_index[sid]["input_mode"] == "image" else 0)
            texts[arm][sid] = normalize_output(raw["raw_response_ids"], tokenizer, raw["eos_ids"])["text"]
            info[arm][sid] = {"status": raw["status"], "seconds": raw["generation_seconds"], "iterations": raw["decode_iterations"]}
    predictions, scores, report = {}, {}, {"status": "interim_direct_only", "rows": len(rows), "arms": {}}
    for arm in ARMS:
        pred, own = {}, []
        for r in rows:
            sid = r["index"]
            source = arm if sid in texts[arm] else "native_direct"
            assert source == arm or (arm == "cdapc_direct" and r["input_mode"] == "text_only")
            own += [info[arm][sid]] if source == arm else []
            pred[sid] = texts[source][sid] if info[source][sid]["status"] == "success" else ""
        predictions[arm] = pred
        metrics, scores[arm] = run.rate(rows, pred, extract, rating)
        report["arms"][arm] = {
            "metrics": metrics,
            "by_input_mode": {m: run.rate([r for r in rows if r["input_mode"] == m], pred, extract, rating)[0] for m in ("image", "text_only")},
            "generation_status_own_rows": dict(Counter(v["status"] for v in own)), "own_rows": len(own),
            "mean_seconds": statistics.mean(v["seconds"] for v in own), "mean_iterations": statistics.mean(v["iterations"] for v in own)}
    published = read_json(ROOT / "baseline/metrics.json")["methods"]
    baseline = {}
    for m in ("native", "vcd", "vigor"):
        pred = {a["index"]: a["text"] for a in read_json(ROOT / f"baseline/{m}/answers.json")}
        baseline[m], scores[f"baseline_{m}"] = run.rate(rows, pred, extract, rating)
        assert abs(baseline[m]["aAcc"] - published[m]["accuracy_percent"]) < 1e-9
        baseline[m]["by_input_mode"] = {mode: run.rate([r for r in rows if r["input_mode"] == mode], pred, extract, rating)[0]
                                        for mode in ("image", "text_only")}
    report["baseline"] = baseline
    image_rows = [r for r in rows if r["input_mode"] == "image"]
    pairs = {"CDAPC直答 − Native直答(B64)": ("cdapc_direct", "native_direct"),
             "CDAPC直答 − 基线Native(B32)": ("cdapc_direct", "baseline_native"),
             "CDAPC直答 − 基线VCD(B32)": ("cdapc_direct", "baseline_vcd"),
             "CDAPC直答 − 基线VIGOR(B32)": ("cdapc_direct", "baseline_vigor"),
             "Native直答B64 − 基线Native B32": ("native_direct", "baseline_native")}
    report["paired"] = {n: {"all_rows": run.set_bootstrap(rows, scores[a], scores[b]),
                            "image_rows": run.set_bootstrap(image_rows, scores[a], scores[b])} for n, (a, b) in pairs.items()}
    base_native = {a["index"]: a["text"] for a in read_json(ROOT / "baseline/native/answers.json")}
    report["native_b64_text_identical_to_baseline_b32"] = sum(texts["native_direct"][s] == base_native[s] for s in by_index)
    atomic_json(ROOT / "reports/interim_direct.json", report)
    lines = ["# HallusionBench 中期评分：直答两组（CDAPC think / Native think 尚未完成）", "",
             "1129题（951图像、178纯文本）；VLMEvalKit原版评分，Unknown计错。CDAPC纯文本题沿用Native直答输出。", "",
             "| 组别 | aAcc | fAcc | qAcc | Avg | 图像题aAcc | 纯文本aAcc | Unknown | Yes比例 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    rows_out = [("CDAPC 直答（K4）", report["arms"]["cdapc_direct"]), ("Native 直答（256/128/64）", report["arms"]["native_direct"])]
    rows_out += [(f"基线 {m}（256/128/32）", {"metrics": baseline[m], "by_input_mode": baseline[m]["by_input_mode"]}) for m in ("native", "vcd", "vigor")]
    for name, b in rows_out:
        m, i, t = b["metrics"], b["by_input_mode"]["image"], b["by_input_mode"]["text_only"]
        lines.append(f"| {name} | {m['aAcc']:.2f} | {m['fAcc']:.2f} | {m['qAcc']:.2f} | {m['Avg']:.2f} | {i['aAcc']:.2f} | "
                     f"{t['aAcc']:.2f} | {m['unknown']} | {m['yes_share']:.1f}% |")
    lines += ["", "题组整组配对bootstrap（2000次，seed 42），单位百分点：", "",
              "| 比较 | 范围 | ΔaAcc [95% CI] | ΔfAcc [95% CI] | ΔqAcc [95% CI] |", "|---|---|---:|---:|---:|"]
    for name, scopes in report["paired"].items():
        for scope, v in scopes.items():
            cells = " | ".join(f"{v[k]['delta']:+.2f} [{v[k]['ci95'][0]:+.2f}, {v[k]['ci95'][1]:+.2f}]" for k in ("aAcc", "fAcc", "qAcc"))
            lines.append(f"| {name} | {'全部' if scope == 'all_rows' else '图像题'} | {cells} |")
    for arm, label in (("cdapc_direct", "CDAPC直答"), ("native_direct", "Native直答")):
        b = report["arms"][arm]
        lines.append(f"\n- {label}：本组生成 {b['own_rows']} 题，{b['generation_status_own_rows']}，平均 {b['mean_seconds']:.2f} 秒、{b['mean_iterations']:.2f} 轮。")
    lines.append(f"- Native直答B64与基线B32逐题文本相同：{report['native_b64_text_identical_to_baseline_b32']}/1129。")
    (ROOT / "reports/interim_direct.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
