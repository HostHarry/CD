"""Apply the user-quoted final-region/fallback protocol to a frozen POPE snapshot."""
from collections import Counter
from datetime import datetime
from pathlib import Path
import statistics

from cdapc_lab.evaluation.scorers import load_pope_official_rule
from cdapc_lab.experiments.score import pope_metrics
from cdapc_lab.io.artifacts import atomic_json, digest, file_hash, read_json

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("random", "popular", "adversarial")
METRICS = ("accuracy", "precision", "recall", "f1", "yes_ratio")


def select_answer(text):
    """Use the last closing tag, or the full text when no closing tag exists."""
    return text.rsplit("</think>", 1)[-1].strip()


def main():
    source_path = ROOT / "reports/pope_completed_official_raw.json"
    snapshot = read_json(source_path)
    official_path = ROOT / "upstream/pope/evaluate.py"
    assert file_hash(official_path) == snapshot["official_source_sha256"]
    rule = load_pope_official_rule(official_path)
    spec = {
        "protocol_id": "pope_after_last_think_else_full_official_v1",
        "protocol_source": "User-provided description of the Native/VCD/VIGOR evaluation",
        "external_implementation_verified": False,
        "input_snapshot": str(source_path),
        "input_snapshot_sha256": file_hash(source_path),
        "official_source": str(official_path),
        "official_source_sha256": file_hash(official_path),
        "analysis_source_sha256": file_hash(Path(__file__)),
        "metrics_source_sha256": file_hash(ROOT / "src/cdapc_lab/experiments/score.py"),
        "answer_scope": "text.rsplit('</think>', 1)[-1].strip(); full visible text if unclosed",
        "empty_answer_policy": "Official rule defaults to yes, including empty extracted text.",
        "failure_policy": "Non-successful generation counts incorrect, with all rows retained.",
        "aggregation": "Three 3000-row splits; unweighted mean of per-split metrics.",
    }
    result = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input_snapshot_at": snapshot["snapshot_at"],
        "spec": spec, "evaluation_spec_sha256": digest(spec), "stages": {},
    }
    for name, old in snapshot["stages"].items():
        root = Path(old["workspace"])
        prior = read_json(old["source_report"])
        assert file_hash(old["source_report"]) == old["source_report_sha256"]
        config = read_json(root / prior["runs"][0] / "resolved_config.json")
        rows_path = root / config["protocol"]["data_root"] / "rows.json"
        assert file_hash(rows_path) == old["rows_sha256"]
        rows = read_json(rows_path)
        assert len(rows) == old["evaluation_rows"] == 9000
        assert {r["input_key"] for r in rows} == set(old["per_input"])
        records = {}
        for sid, record in old["per_input"].items():
            text = record["text"]
            answer = select_answer(text)
            prediction = rule(answer)
            closed = "</think>" in text
            effective = prediction if record["generation_status"] == "success" else None
            records[sid] = {
                "gold": record["gold"], "generation_status": record["generation_status"],
                "selected_text": answer, "official_prediction": prediction,
                "effective_prediction": effective,
                "diagnostics": {
                    "has_closing_tag": closed,
                    "full_text_fallback": not closed,
                    "empty_after_closing": closed and not answer,
                    "empty_selected_text": not answer,
                    "multiple_closing_tags": text.count("</think>") > 1,
                    "changed_vs_raw_official": prediction != record["raw_official_prediction"],
                    "changed_vs_prior_final_parser": prediction != record["prior_final_prediction"],
                },
            }
            if old["mode"] == "direct":
                assert effective == record["effective_prediction"]
        assert all(records[r["input_key"]]["gold"] == r["label"] for r in rows)
        splits = {}
        for split in SPLITS:
            selected = [r for r in rows if r["split"] == split]
            assert len(selected) == 3000
            splits[split] = {
                "n": len(selected),
                **pope_metrics((records[r["input_key"]]["effective_prediction"], r["label"])
                               for r in selected),
            }
            assert sum(splits[split][k] for k in ("TP", "FP", "TN", "FN", "unresolved")) == 3000
        macro = {metric: statistics.mean(s[metric] for s in splits.values())
                 for metric in METRICS}
        if old["mode"] == "direct":
            assert macro == old["macro"]
        diagnostic_keys = next(iter(records.values()))["diagnostics"].keys()
        item = {
            "dataset": old["dataset"], "mode": old["mode"],
            "unique_inputs": len(records), "evaluation_rows": len(rows),
            "rows_path": str(rows_path), "rows_sha256": file_hash(rows_path),
            "generation_statuses": dict(Counter(r["generation_status"] for r in records.values())),
            "splits": splits, "macro": macro,
            "pooled": pope_metrics((records[r["input_key"]]["effective_prediction"], r["label"])
                                   for r in rows),
            "raw_official_macro": old["macro"], "judge_macro": old["judge_macro"],
            "diagnostics_unique": {k: sum(r["diagnostics"][k] for r in records.values())
                                   for k in diagnostic_keys},
            "diagnostics_rows": {k: sum(records[r["input_key"]]["diagnostics"][k] for r in rows)
                                for k in diagnostic_keys},
            "per_input": records,
        }
        result["stages"][name] = item
        print(name, macro, item["diagnostics_unique"], flush=True)
    target = ROOT / "reports/pope_after_think_official.json"
    atomic_json(target, result)
    lines = [
        "# POPE：think 后答案／未闭合回退全文，再套官方规则", "",
        f"计算时间：{result['created_at']}；输入快照：{result['input_snapshot_at']}。",
        "本报告按用户提供的 Native/VCD/VIGOR 评分描述实现，未获取该外部实现源码。",
        "有 </think> 时取最后一个标签之后的文本并去除首尾空白；没有闭合标签时使用全文。",
        "对截取结果执行固定版本 POPE 官方规则：取第一个英文句点前的文本、"
        "删除英文逗号、按普通空格分词；出现 No/not/no 判 no，否则判 yes。",
        "**闭合后为空仍默认判 yes；未闭合也不直接判错。** 生成失败仍保留分母计错。",
        "每组9000评测行，三个split各3000。表中所有指标为split等权平均。",
        "", "| 数据集 | 模式 | Acc % | F1 % | Precision % | Recall % | Yes ratio % |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("coco", "aokvqa", "gqa"):
        for item in result["stages"].values():
            if item["dataset"] == dataset:
                m = item["macro"]
                lines.append(f"| {dataset} | {item['mode']} | " +
                             " | ".join(f"{100*m[k]:.2f}" for k in
                                        ("accuracy", "f1", "precision", "recall", "yes_ratio")) + " |")
    lines += ["", "| think 数据集 | 唯一输入 | 已闭合 | 未闭合、回退全文 | 闭合后为空 |",
              "|---|---:|---:|---:|---:|"]
    for item in result["stages"].values():
        if item["mode"] == "think":
            d = item["diagnostics_unique"]
            lines.append(f"| {item['dataset']} | {item['unique_inputs']} | "
                         f"{d['has_closing_tag']} | {d['full_text_fallback']} | "
                         f"{d['empty_after_closing']} |")
    lines += [
        "", "原文官方评分、最终答案严格解析和 Judge 是不同口径，已有结果全部保留。",
        "本报告仅对齐所描述的评分规则；模型、提示词、解码预算、EOS 策略等仍需另行核对，"
        "不能把不同配置的分数差归因于某个机制。",
        "逐输入截取文本、判定、数据映射和来源哈希见同名 JSON。",
    ]
    markdown = target.with_suffix(".md")
    markdown.write_text("\n".join(lines) + "\n")
    target.with_suffix(".sha256").write_text("".join(
        f"{file_hash(p)}  {p.relative_to(ROOT)}\n"
        for p in (Path(__file__), source_path, target, markdown)))


if __name__ == "__main__":
    main()
