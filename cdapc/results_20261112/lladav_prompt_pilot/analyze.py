"""LLaDA-V prompt pilot (24 POPE-COCO inputs): output length, early stop, parseability and accuracy per arm."""
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cdapc_lab.evaluation.normalize import normalize_output
from cdapc_lab.evaluation.scorers import load_pope_official_rule
from transformers import AutoTokenizer

ARMS = {"A": "当前：/think＋Final answer后缀", "B": "/think，无后缀", "C": "原题，无标记无后缀",
        "D": "/no_think＋Please answer yes or no."}
FINAL = re.compile(r"Final answer:\s*\**\s*(Yes|No)\b", re.I)


def answer(text, rule):
    found = FINAL.findall(text)
    if found:
        return found[-1].lower(), "final_answer"
    tail = text.rsplit("</think>", 1)[-1].strip()
    head = re.match(r"^\W*(yes|no)\b", tail, re.I)
    if head:
        return head.group(1).lower(), "leading_yes_no"
    return rule(tail), "pope_official_first_sentence"


def records(arm, ids):
    if arm == "A":
        paths = json.loads((ROOT / "manifests/pilot24_armA_native_predictions.json").read_text())
        return {sid: json.loads(Path(paths[sid]).read_text()) for sid in ids}
    out = {}
    for run in (ROOT / "runs").iterdir():
        if json.loads((run / "resolved_config.json").read_text())["run"]["name"] == arm:
            for p in (run / "predictions").glob("*.json"):
                r = json.loads(p.read_text())
                out[r["sample_id"]] = r
    return out


def main():
    rows = json.loads((ROOT / "manifests/pilot24_rows.json").read_text())
    ids = [r["input_key"] for r in rows]
    label = {r["input_key"]: r["label"] for r in rows}
    tok = AutoTokenizer.from_pretrained(ROOT / "models/lladav", local_files_only=True)
    rule = load_pope_official_rule(ROOT / "upstream/pope/evaluate.py")
    arms = dict(ARMS)
    arms.update({k: v for k, v in (("E", "CDAPC I128：" + "最佳提示"),) if any(
        json.loads((r / "resolved_config.json").read_text())["run"]["name"].startswith("pilot24_cdapc")
        for r in (ROOT / "runs").iterdir())})
    report, lines = {}, ["# LLaDA-V 提示词小样本（POPE-COCO 24题，Native 256/128/64）", "",
                         "| 组 | 提示 | 完成 | 画布 | EOS位置中位数 | 提前停止 | 预算耗尽 | 平均步数 | 平均秒 | 答案来源 | 准确率 |",
                         "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
    names = {a: f"pilot24_native_{a}" for a in "BCD"}
    for run in (ROOT / "runs").iterdir():
        name = json.loads((run / "resolved_config.json").read_text())["run"]["name"]
        if name.startswith("pilot24_L64_"):
            arm = name.replace("pilot24_", "")
            names[arm] = name
            arms[arm] = {"L64_A64": "Native 64/32，/think＋Final answer后缀", "L64_B64": "Native 64/32，/think无后缀",
                         "L64_E64": "CDAPC L64/I32 K4，/think＋后缀"}[arm]
        elif name.endswith("_endoftext"):
            arm = name.split("_")[2] + "+eot"
            names[arm] = name
            arms[arm] = ARMS[name.split("_")[2]] + "；允许<|endoftext|>收尾"
        elif name.startswith("pilot24_cdapc_"):
            names[name.replace("pilot24_", "")] = name
            arms[name.replace("pilot24_", "")] = "CDAPC r70/W16/K4/L256/I128，提示" + name.split("_")[-1]
    for arm in arms:
        recs = records(arm, ids) if arm == "A" else records(names[arm], ids)
        done = [sid for sid in ids if sid in recs]
        if not done:
            continue
        stats, sources, correct, examples = [], Counter(), 0, []
        for sid in done:
            r = recs[sid]
            n = normalize_output(r["raw_response_ids"], tok, r["eos_ids"])
            pred, src = answer(n["text"], rule) if r["status"] == "success" else (None, "generation_failed")
            sources[src] += 1
            correct += pred == label[sid]
            stats.append((n["first_eos_position"], r["decode_iterations"], r["generation_seconds"], r["finish_reason"], n["full_buffer_length"]))
            if len(examples) < 2:
                examples.append(n["text"][:220].replace("\n", " "))
        eos = [s[0] if s[0] is not None else s[4] for s in stats]
        early = sum(s[3] == "eos_prefix_complete" for s in stats)
        block = {"done": len(done), "eos_median": statistics.median(eos), "eos_mean": statistics.mean(eos),
                 "early_stop": early, "budget_exhausted": sum(s[3] == "budget_exhausted" for s in stats), "canvas": stats[0][4],
                 "mean_steps": statistics.mean(s[1] for s in stats), "mean_seconds": statistics.mean(s[2] for s in stats),
                 "answer_sources": dict(sources), "accuracy": correct / len(done), "examples": examples}
        report[arm] = block
        src = "，".join(f"{k}:{v}" for k, v in sources.items())
        lines.append(f"| {arm} | {arms[arm]} | {len(done)}/24 | {block['canvas']} | {block['eos_median']:.0f} | {early} | {block['budget_exhausted']} | "
                     f"{block['mean_steps']:.1f} | {block['mean_seconds']:.1f} | {src} | {100*block['accuracy']:.1f}% |")
    lines += ["", "示例（每组前两题开头）："] + [f"- {a}: {e}" for a, b in report.items() for e in b["examples"]]
    (ROOT / "reports/pilot24.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    (ROOT / "reports/pilot24.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
