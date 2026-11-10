"""Score one completed POPE stage; optionally wait for its existing queue."""
import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
import statistics
import time

from cdapc_lab.evaluation.scorers import load_pope_official_rule
from cdapc_lab.experiments.score import pope_metrics
from cdapc_lab.io.artifacts import atomic_json, file_hash, read_json
import pope_after_think_official as answer_protocol

ROOT = Path(__file__).resolve().parents[1]


def score_stage(root, name, stage):
    source_report = root / stage["report"]
    prior = read_json(source_report)
    cfg = read_json(root / prior["runs"][0] / "resolved_config.json")
    assert cfg["protocol"]["task"] == "pope"
    rows_path = root / cfg["protocol"]["data_root"] / "rows.json"
    rows = read_json(rows_path)
    official = root / "upstream/pope/evaluate.py"
    rule = load_pope_official_rule(official)
    records = {}
    evaluations = []
    for run in prior["cdapc"]["per_run"]:
        directory = root / run["run"] / "scores" / run["evaluation_fingerprint"]
        evaluations.append({"run": run["run"], "evaluation_fingerprint": run["evaluation_fingerprint"]})
        for path in sorted(directory.glob("*.json")):
            if path.name == "evaluation_manifest.json":
                continue
            saved = read_json(path)
            sid = saved["sample_id"]
            assert sid not in records
            text = saved.get("normalized", {}).get("text", "")
            if saved["generation_status"] == "success":
                assert "normalized" in saved
            selected = answer_protocol.select_answer(text)
            prediction = rule(selected)
            records[sid] = {
                "gold": saved["gold"], "generation_status": saved["generation_status"],
                "selected_text": selected, "official_prediction": prediction,
                "effective_prediction": prediction if saved["generation_status"] == "success" else None,
                "has_closing_tag": "</think>" in text,
                "empty_after_closing": "</think>" in text and not selected,
                "source_score": str(path), "source_score_sha256": file_hash(path),
            }
    assert len(records) == stage["expected_inputs"]
    assert set(records) == {r["input_key"] for r in rows}
    assert len(rows) == 9000
    assert all(records[r["input_key"]]["gold"] == r["label"] for r in rows)
    splits = {}
    for split in ("random", "popular", "adversarial"):
        selected = [r for r in rows if r["split"] == split]
        assert len(selected) == 3000
        splits[split] = {
            "n": len(selected),
            **pope_metrics((records[r["input_key"]]["effective_prediction"], r["label"])
                           for r in selected),
        }
    metrics = {k: statistics.mean(s[k] for s in splits.values())
               for k in ("accuracy", "f1", "precision", "recall", "yes_ratio")}
    return {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "workspace": str(root), "stage": name,
        "protocol": "last </think> suffix, or full text if unclosed; strip; pinned POPE rule",
        "empty_answer_policy": "Successful empty answers default to yes under official rule.",
        "failure_policy": "Non-successful generations count incorrect; fixed 9000-row denominator.",
        "official_source": str(official), "official_source_sha256": file_hash(official),
        "script_sha256": file_hash(Path(__file__)),
        "answer_protocol_sha256": file_hash(Path(answer_protocol.__file__)),
        "metrics_source_sha256": file_hash(ROOT / "src/cdapc_lab/experiments/score.py"),
        "source_report": str(source_report), "source_report_sha256": file_hash(source_report),
        "rows_sha256": file_hash(rows_path), "evaluations": evaluations,
        "unique_inputs": len(records), "evaluation_rows": len(rows),
        "generation_statuses": dict(Counter(r["generation_status"] for r in records.values())),
        "failed_evaluation_rows": sum(records[r["input_key"]]["generation_status"] != "success" for r in rows),
        "closed_unique": sum(r["has_closing_tag"] for r in records.values()),
        "empty_after_closing_unique": sum(r["empty_after_closing"] for r in records.values()),
        "macro": metrics, "splits": splits, "cost": prior["cdapc"]["cost"], "per_input": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    root = args.workspace.resolve()
    output = root / "reports" / f"{args.stage}_after_think_official.json"
    status = output.with_suffix(".status.json")
    last_state = None
    try:
        while True:
            index = read_json(root / "reports/index.json")
            stage = index["stages"][args.stage]
            state = stage["status"]
            if state == "complete":
                break
            if state in ("failed", "cancelled") or not args.wait:
                raise RuntimeError(f"Stage is not complete: {state}")
            atomic_json(status, {"status": "waiting_for_generation_and_scores", "stage": args.stage,
                                 "observed_stage_status": state,
                                 "updated_at": datetime.now().astimezone().isoformat(timespec="seconds")})
            if state != last_state:
                print(f"WAITING {args.stage}: {state}", flush=True)
                last_state = state
            time.sleep(60)
        result = score_stage(root, args.stage, stage)
        atomic_json(output, result)
        lines = [f"# {args.stage}：答案区域＋官方规则", "",
                 f"时间：{result['created_at']}；{result['unique_inputs']}个唯一输入，9000评测行。",
                 "取最后一个 </think> 后的文本；未闭合回退全文。成功生成但答案区域为空时，"
                 "依官方规则判 yes。生成失败／预算耗尽计错，保留分母。",
                 "", "| split | Accuracy % | F1 % | Precision % | Recall % | Yes ratio % |",
                 "|---|---:|---:|---:|---:|---:|"]
        for key, values in [*result["splits"].items(), ("mean", result["macro"])]:
            lines.append(f"| {key} | " + " | ".join(
                f"{100*values[m]:.2f}" for m in ("accuracy", "f1", "precision", "recall", "yes_ratio")) + " |")
        lines += ["", f"生成状态（唯一输入）：{result['generation_statuses']}；"
                  f"失败评测行：{result['failed_evaluation_rows']}。"]
        output.with_suffix(".md").write_text("\n".join(lines) + "\n")
        output.with_suffix(".sha256").write_text("".join(
            f"{file_hash(p)}  {p}\n" for p in (Path(__file__), output, output.with_suffix(".md"))))
        atomic_json(status, {"status": "complete", "report": str(output),
                             "report_sha256": file_hash(output), "completed_at": result["created_at"]})
        print("COMPLETE", output, result["macro"], flush=True)
    except Exception as exc:
        atomic_json(status, {"status": "failed", "error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
