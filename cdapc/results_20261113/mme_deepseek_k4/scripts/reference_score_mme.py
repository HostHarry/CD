"""Official MME Acc/Acc+ with fixed question and image-pair denominators."""
import ast
from collections import Counter, defaultdict
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / "cdapc-experiments"
sys.path.insert(0, str(ROOT / "src"))
from cdapc_lab.evaluation.normalize import normalize_output
from cdapc_lab.io.artifacts import atomic_json, file_hash, read_json, safe_id


def official_protocol():
    source = ROOT / "upstream/mme/calculation.py"
    tree = ast.parse(source.read_text())
    categories = next(ast.literal_eval(n.value) for n in tree.body
                      if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "eval_type_dict" for t in n.targets))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "calculate_metrics")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "parse_pred_ans")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return categories, lambda text: namespace["parse_pred_ans"](None, text.lower())


def metrics(rows, answers, categories):
    by_category = defaultdict(list)
    seen = set()
    for row in rows:
        sid = row["sample_id"]
        if sid in seen:
            raise ValueError(f"Duplicate sample: {sid}")
        seen.add(sid)
        by_category[row["category"]].append(row)
    expected = {name for values in categories.values() for name in values}
    if set(by_category) != expected:
        raise ValueError("Missing or unexpected MME category")
    result = {"per_category": {}}
    for category, items in by_category.items():
        pairs = defaultdict(list)
        for row in items:
            pairs[row["group_id"]].append(row)
        if any(len(pair) != 2 or {r["answer"] for r in pair} != {"yes", "no"}
               for pair in pairs.values()):
            raise ValueError("MME requires one Yes and one No question per image")
        correct = {r["sample_id"]: answers.get(r["sample_id"], "other") == r["answer"] for r in items}
        acc = sum(correct.values()) / len(items)
        pair_correct = sum(all(correct[r["sample_id"]] for r in pair) for pair in pairs.values())
        acc_plus = pair_correct / len(pairs)
        result["per_category"][category] = {
            "questions": len(items), "image_pairs": len(pairs),
            "correct": sum(correct.values()), "pairs_both_correct": pair_correct,
            "other": sum(answers.get(r["sample_id"], "other") == "other" for r in items),
            "acc": acc, "acc_plus": acc_plus, "score": 100 * (acc + acc_plus),
        }
    for section, tasks in categories.items():
        result[section.lower()] = sum(result["per_category"][name]["score"] for name in tasks)
    result["total"] = result["perception"] + result["cognition"]
    return result


def score(runs):
    from transformers import AutoTokenizer
    rows = read_json(ROOT / "data/mme/rows.json")
    ids = {r["sample_id"] for r in rows}
    provenance = read_json(ROOT / "data/mme/provenance.json")
    assert file_hash(ROOT / "upstream/mme/calculation.py") == provenance["scorer_sha256"]
    categories, parse = official_protocol()
    first = Path(runs[0])
    cfg = read_json(first / "resolved_config.json")
    assets = read_json(first / "run_manifest.json")["assets"]["model_files"]
    for name, expected in assets.items():
        if not name.endswith(".safetensors"):
            assert file_hash(ROOT / cfg["model"]["checkpoint"] / name) == expected
    tokenizer = AutoTokenizer.from_pretrained(ROOT / cfg["model"]["checkpoint"], local_files_only=True)
    answers, records, exported = {}, {}, {}
    statuses, finish = Counter(), Counter()
    seconds = iterations = 0
    for run in map(Path, runs):
        manifest = read_json(run / "run_manifest.json")
        for entry in read_json(run / "dataset_manifest.json")["samples"]:
            sid = entry["sample_id"]
            assert sid in ids and sid not in records
            prediction = read_json(run / "predictions" / (safe_id(sid) + ".json"))
            assert prediction["sample_id"] == sid
            assert prediction["generation_fingerprint"] == manifest["generation_fingerprint"]
            assert prediction["status"] in ("success", "budget_exhausted")
            normalized = normalize_output(prediction["raw_response_ids"], tokenizer, prediction["eos_ids"])
            text = normalized["text"]
            answer = parse(text) if prediction["status"] == "success" else "other"
            answers[sid] = answer
            exported[sid] = (text.replace("\t", " ").replace("\r", " ").replace("\n", " ")
                             if prediction["status"] == "success" else "")
            records[sid] = {
                "text": text, "prediction": answer, "generation_status": prediction["status"],
                "finish_reason": prediction["finish_reason"], "normalized": normalized,
                "generation_fingerprint": prediction["generation_fingerprint"], "run": str(run),
            }
            statuses[prediction["status"]] += 1
            finish[prediction["finish_reason"]] += 1
            seconds += prediction["generation_seconds"]
            iterations += prediction["decode_iterations"]
    assert set(records) == ids and len(ids) == 2374
    result = metrics(rows, answers, categories)
    export = ROOT / "reports/mme_official_text"
    export.mkdir(exist_ok=True)
    for category in result["per_category"]:
        selected = sorted((r for r in rows if r["category"] == category),
                          key=lambda r: (r["group_id"], r["sample_id"]))
        lines = [f"{r['image_name']}\t{r['question']}\t{r['answer']}\t{exported[r['sample_id']]}\n"
                 for r in selected]
        (export / f"{category}.txt").write_text("".join(lines))
    # Run the unmodified upstream scorer as an independent full-data check.
    checked = subprocess.run([sys.executable, str(ROOT / "upstream/mme/calculation.py"),
                              "--results_dir", str(export)], capture_output=True, text=True, check=True,
                             env={**os.environ, "PYTHONPATH": os.pathsep.join(
                                 [str(ROOT / "vendor"), str(ROOT / "src")])})
    (ROOT / "reports/mme_official_scorer.log").write_text(checked.stdout + checked.stderr)
    official_tasks = dict(re.findall(r"^[ \t]+(\w+)[ \t]+score:\s+([\d.eE+-]+)", checked.stdout, re.M))
    assert set(official_tasks) == set(result["per_category"])
    for name, value in official_tasks.items():
        assert abs(float(value) - result["per_category"][name]["score"]) < 1e-8
    report = {
        "status": "complete", "method": "cdapc_r70", "runs": [str(r) for r in runs],
        "canvas_length": 256, "iteration_limit": 128, "r_gate": .7, "window": 16, "max_commit": 16,
        "prompt_id": "mme_yesno_v1", "question_count": len(rows), "image_pairs": 1187,
        "metrics": result, "maximum_scores": {"perception": 2000, "cognition": 800, "total": 2800},
        "generation_status_counts": dict(statuses), "finish_reason_counts": dict(finish),
        "cost": {"count": len(rows), "generation_seconds": seconds, "decode_iterations": iterations,
                 "mean_seconds": seconds / len(rows), "mean_iterations": iterations / len(rows)},
        "protocol": provenance, "source_snapshot": read_json(ROOT / "reports/source_snapshot.json"),
        "official_scorer_verified": True, "report_code_sha256": file_hash(__file__),
        "per_sample": records,
        "notes": ["Budget-exhausted outputs count as incorrect with fixed denominators.",
                  "Standard MME Yes/No prompt; no additional reasoning instruction.",
                  "No MME Native baseline in this queued condition."],
    }
    for path in (ROOT / "reports/mme_cdapc_r70_L256_I128.json",
                 BASE / "reports/mme_cdapc_r70_L256_I128.json"):
        atomic_json(path, report)
    text = ["# MME CD-APC r70，L256/I128", "",
            f"全部 {len(rows)} 题、1187 个图像问题对；官方 Yes/No 提示词。",
            f"Perception：{result['perception']:.2f}/2000；Cognition：{result['cognition']:.2f}/800；"
            f"总分：{result['total']:.2f}/2800。", "",
            "| 类别 | 题数 | Acc % | Acc+ % | 得分 / 200 |",
            "|---|---:|---:|---:|---:|"]
    for name, values in result["per_category"].items():
        text.append(f"| {name} | {values['questions']} | {100*values['acc']:.2f} | "
                    f"{100*values['acc_plus']:.2f} | {values['score']:.2f} |")
    text += ["", "已与未修改的官方评分脚本逐类核对。预算耗尽计错且保留在分母中。"]
    for path in (ROOT / "reports/mme_cdapc_r70_L256_I128.md",
                 BASE / "reports/mme_cdapc_r70_L256_I128.md"):
        path.write_text("\n".join(text) + "\n")
    return report


if __name__ == "__main__":
    score(sys.argv[1:])
