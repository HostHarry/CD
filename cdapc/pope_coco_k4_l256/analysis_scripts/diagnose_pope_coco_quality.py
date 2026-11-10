"""Offline diagnosis of POPE quality, using frozen outputs and existing Judge results."""
import ast
from collections import Counter, defaultdict
from datetime import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from transformers import AutoTokenizer
from cdapc_lab.evaluation.scorers import load_pope_official_rule
from cdapc_lab.experiments.score import pope_metrics
from cdapc_lab.io.artifacts import atomic_json, file_hash, read_json
from pope_after_think_official import select_answer

ROOT = Path(__file__).resolve().parents[1]
PREFIX = ROOT.parent / "cdapc-pope-eos-full"
SAFE = ROOT.parent / "cdapc-pope-safe-eos"
BASELINE = ROOT.parent / "cd-baselines-review/baselines"
NEW_NAME = "pope_coco_think_safe_eos_r70_k4_L256_I128"
OLD_NAME = "pope_coco_think_eos_prefix_r70_k4_L256_I128"
DIRECT_NAME = "pope_coco_direct_eos_prefix_r70_k4_L256_I128"
LABEL = re.compile(r"\s*(?:(?:Final\s+answer|Answer)\s*:\s*)?(?P<label>yes|no)\b", re.I)


def differences(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        return [d for k in sorted(a.keys() | b.keys())
                for d in differences(a.get(k), b.get(k), path + "." + k)]
    return [] if a == b else [{"path": path, "direct": a, "think": b}]


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def main():
    report_paths = [PREFIX / "reports" / f"{DIRECT_NAME}_after_think_official.json",
                    PREFIX / "reports" / f"{OLD_NAME}_after_think_official.json",
                    SAFE / "reports" / f"{NEW_NAME}_after_think_official.json"]
    reports = [read_json(p) for p in report_paths]
    judge_path = SAFE / "reports/judge" / f"{NEW_NAME}.json"
    judge = read_json(judge_path)["per_input"]
    rows_path = SAFE / "data/pope_yesno/rows.json"
    rows = read_json(rows_path)
    current = reports[-1]["per_input"]
    assert all(set(r["per_input"]) == set(current) for r in reports)
    assert set(judge) == set(current)
    tokenizer = AutoTokenizer.from_pretrained(SAFE / "models/mmada", local_files_only=True)
    token_text = {}

    def decode(ids):
        return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    groups = defaultdict(list)
    per_input = {}
    for sid, record in current.items():
        score_path = Path(record["source_score"])
        saved = read_json(score_path)
        text, selected = saved["normalized"]["text"], record["selected_text"]
        j = judge[sid]
        category = ("failed" if record["generation_status"] != "success" else
                    "unclosed" if not record["has_closing_tag"] else
                    "closed_empty" if not selected else
                    "closed_explicit_label" if LABEL.match(selected) else "closed_other")
        pred_path = score_path.parents[2] / "predictions" / score_path.name
        pred = read_json(pred_path)
        assert pred["sample_id"] == sid
        events = {e["position"]: e for e in pred["commit_log"]}
        assert len(events) == len(pred["commit_log"])
        end = pred["first_eos_position"]
        end = end if end is not None else len(pred["raw_response_ids"])
        visible = [e for pos, e in events.items() if pos < end]
        eos = events.get(pred["first_eos_position"])
        eos_entries = [e for e in events.values() if e["written_id"] in pred["eos_ids"]]
        all_ids = pred["raw_response_ids"][:end]
        raw = decode(all_ids)
        assert raw.strip() == text
        tail = raw.rsplit("</think>", 1)[-1]
        match, offset, label_source = LABEL.match(tail), len(raw) - len(tail), "final_region"
        if not match and j["judge_status"] == "answered" and j["evidence"]:
            evidence = j["evidence"]
            m = re.search(r"\b" + j["judge_prediction"] + r"\b", evidence, re.I)
            if m:
                offset = raw.rfind(evidence)
                if offset >= 0:
                    span = (offset + m.start(), offset + m.end())
                    label_source = "judge_evidence_literal"
                else:
                    span = None
            else:
                span = None
        elif match:
            span = tuple(offset + n for n in match.span("label"))
        else:
            span = None
        answer = None
        if span is not None:
            for token in all_ids:
                if token not in token_text:
                    token_text[token] = decode([token])
            pieces = [token_text[token] for token in all_ids]
            if "".join(pieces) == raw:
                ends, length = [], 0
                for piece in pieces:
                    length += len(piece)
                    ends.append(length)
            else:
                ends = [len(decode(all_ids[:i + 1])) for i in range(len(all_ids))]
            starts = [0] + ends[:-1]
            positions = [i for i, (start, stop) in enumerate(zip(starts, ends))
                         if start < span[1] and stop > span[0]]
            label = raw[span[0]:span[1]].lower()
            if (positions and all(p in events for p in positions) and
                    j["judge_status"] == "answered" and label == j["judge_prediction"]):
                chosen = [events[p] for p in positions]
                iteration = max(e["iteration"] for e in chosen)
                answer = {
                    "label": label, "source": label_source, "positions": positions,
                    "iteration": iteration, "all_written_equal_base": all(e["written_id"] == e["base_id"] for e in chosen),
                    "fallback": any(e["reason"] == "fallback" for e in chosen),
                    "minimum_r": min(e["r_contrast"] for e in chosen),
                    "minimum_u": min(math.exp(e["logp_visual_contrast"]) for e in chosen),
                    "mean_gain": mean([e["gain_contrast"] for e in chosen]),
                    "all_pool_singleton": all(e["pool_size"] == 1 for e in chosen),
                    "visible_context_committed_before": sum(e["iteration"] < iteration for e in visible),
                    "visible_context_fraction_before": sum(e["iteration"] < iteration for e in visible) / max(end, 1),
                    "eos_already_committed": any(e["iteration"] < iteration for e in eos_entries),
                    "base_text_at_answer": decode([e["base_id"] for e in chosen]),
                }
        per_input[sid] = {
            "gold": record["gold"], "category": category, "text": text, "selected_text": selected,
            "rule_prediction": record["effective_prediction"], "judge_prediction": j["judge_prediction"],
            "judge_status": j["judge_status"], "judge_evidence": j["evidence"],
            "generation_status": record["generation_status"], "visible_tokens": end,
            "decode_iterations": pred["decode_iterations"],
            "visible_commits": len(visible),
            "visible_nonbase_writes": sum(e["written_id"] != e["base_id"] for e in visible),
            "visible_fallback_writes": sum(e["reason"] == "fallback" for e in visible),
            "first_any_eos_iteration": min((e["iteration"] for e in eos_entries), default=None),
            "final_eos_iteration": eos["iteration"] if eos else None,
            "final_eos_r": eos["r_contrast"] if eos else None,
            "final_eos_gain": eos["gain_contrast"] if eos else None,
            "final_eos_with_prefix_holes": any(e["iteration"] > eos["iteration"] for e in visible) if eos else None,
            "answer_commit": answer, "source_prediction": str(pred_path),
            "source_prediction_sha256": file_hash(pred_path),
        }
    for row in rows:
        groups[per_input[row["input_key"]]["category"]].append(row)
    group_metrics = {}
    for category, selected_rows in groups.items():
        ids = {r["input_key"] for r in selected_rows}
        group_metrics[category] = {
            "unique_inputs": len(ids), "evaluation_rows": len(selected_rows),
            "gold_yes": sum(r["label"] == "yes" for r in selected_rows),
            "rule": pope_metrics((per_input[r["input_key"]]["rule_prediction"], r["label"]) for r in selected_rows),
            "judge": pope_metrics((per_input[r["input_key"]]["judge_prediction"], r["label"]) for r in selected_rows),
            "mean_visible_tokens": mean([per_input[s]["visible_tokens"] for s in ids]),
            "mean_first_eos_iteration": mean([per_input[s]["first_any_eos_iteration"] for s in ids]),
            "eos_with_prefix_holes_unique": sum(per_input[s]["final_eos_with_prefix_holes"] is True for s in ids),
        }
    transitions = Counter()
    semantic_errors = Counter()
    group_error_rows = Counter()
    for row in rows:
        p = per_input[row["input_key"]]
        a, b = p["rule_prediction"] == row["label"], p["judge_prediction"] == row["label"]
        transitions[f"rule_{int(a)}_judge_{int(b)}"] += 1
        if not a:
            group_error_rows[p["category"]] += 1
        if not b:
            semantic_errors[p["judge_status"] if p["judge_prediction"] is None else
                            ("explicit_false_yes" if p["judge_prediction"] == "yes" else "explicit_false_no")] += 1
    trace_groups = defaultdict(list)
    for p in per_input.values():
        if p["answer_commit"]:
            key = f"{p['answer_commit']['label']}_{'correct' if p['judge_prediction'] == p['gold'] else 'wrong'}"
            trace_groups[key].append(p)
    trace_summary = {}
    for name, values in trace_groups.items():
        answers = [p["answer_commit"] for p in values]
        trace_summary[name] = {
            "unique_inputs_with_mapped_label": len(values),
            "all_written_equal_base": sum(a["all_written_equal_base"] for a in answers),
            "fallback": sum(a["fallback"] for a in answers),
            "all_pool_singleton": sum(a["all_pool_singleton"] for a in answers),
            "mean_r": mean([a["minimum_r"] for a in answers]),
            "r_at_least_07": sum(a["minimum_r"] >= .7 for a in answers),
            "mean_gain": mean([a["mean_gain"] for a in answers]),
            "gain_nonpositive": sum(a["mean_gain"] <= 0 for a in answers),
            "mean_answer_iteration": mean([a["iteration"] for a in answers]),
            "mean_context_fraction_before_answer": mean([a["visible_context_fraction_before"] for a in answers]),
            "eos_already_committed": sum(a["eos_already_committed"] for a in answers),
        }
    index = read_json(PREFIX / "reports/index.json")
    config_audit = {}
    for shard in ("0", "1", "2"):
        runs = [PREFIX / index["stages"][name]["shards"][shard]["run"] for name in (DIRECT_NAME, OLD_NAME)]
        cfgs = [read_json(r / "resolved_config.json") for r in runs]
        manifests = [read_json(r / "run_manifest.json") for r in runs]
        config_audit[shard] = {
            "differences": differences(*cfgs),
            "same_generation_source": manifests[0]["generation_source_hash"] == manifests[1]["generation_source_hash"],
            "same_assets": manifests[0]["assets"] == manifests[1]["assets"],
            "same_environment": manifests[0]["environment"] == manifests[1]["environment"],
        }
        assert all(config_audit[shard][k] for k in ("same_generation_source", "same_assets", "same_environment"))
        assert {d["path"] for d in config_audit[shard]["differences"]} == {
            ".run.name", ".protocol.prompt_id", ".protocol.strict_parser_id"}
    diagnostic = BASELINE / "records/pope_prompt_diagnostic/coco_192"
    samples = [json.loads(line) for line in (diagnostic / "samples.jsonl").read_text().splitlines()]
    module = ast.parse((BASELINE / "methods/common/mmmu_protocol.py").read_text())
    instruction = next(ast.literal_eval(n.value) for n in module.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "INSTRUCTION" for t in n.targets))
    rule = load_pope_official_rule(SAFE / "upstream/pope/evaluate.py")
    lookup = {(r["image_sha256"], r["question"]): r["input_key"] for r in rows}
    diagnostic_pairs = []
    native = {}
    for variant in ("B", "D"):
        native[variant] = {}
        for path in (diagnostic / variant).glob("predictions-*.jsonl.gz"):
            with gzip.open(path, "rt") as stream:
                for line in stream:
                    obj = json.loads(line)
                    native[variant][obj["input_key"]] = obj["payload"]
    for sample in samples:
        sid = lookup[(sample["image_sha256"], sample["q0"])]
        record = {"row_id": sample["id"], "input_key": sid, "gold": sample["label"]}
        assert record["gold"] == current[sid]["gold"]
        for variant in ("B", "D"):
            q = instruction + "\n" + sample["q0"] if variant == "B" else sample["q0"] + " Please answer yes or no."
            key = hashlib.sha256(json.dumps([sample["image_sha256"], q], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            p = native[variant][key]
            record["native_" + variant] = rule(select_answer(p["text"]) if variant == "B" else p["text"])
        for name, report in zip(("cdapc_direct_prefix", "cdapc_think_prefix", "cdapc_think_original"), reports):
            record[name] = report["per_input"][sid]["effective_prediction"]
        diagnostic_pairs.append(record)
    small = {name: pope_metrics((r[name], r["gold"]) for r in diagnostic_pairs) for name in
             ("native_B", "native_D", "cdapc_direct_prefix", "cdapc_think_prefix", "cdapc_think_original")}
    result = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": "5127 unique inputs / 9000 rows; group metrics are pooled, headline metrics are split means",
        "sources": {str(p): file_hash(p) for p in [*report_paths, judge_path, rows_path, Path(__file__)]},
        "config_audit_direct_vs_think_same_eos": config_audit,
        "headline": {r["stage"]: r["macro"] for r in reports},
        "format_groups": group_metrics, "rule_errors_by_group_rows": dict(group_error_rows),
        "rule_judge_transitions_rows": dict(transitions), "judge_error_categories_rows": dict(semantic_errors),
        "answer_commit_groups": trace_summary,
        "visible_token_writes": {
            "total": sum(p["visible_commits"] for p in per_input.values()),
            "nonbase": sum(p["visible_nonbase_writes"] for p in per_input.values()),
            "fallback": sum(p["visible_fallback_writes"] for p in per_input.values()),
        },
        "native_192_diagnostic": {
            "metrics": small, "rows": diagnostic_pairs,
            "limits": "Native B has no Final answer suffix and uses L256/T256/B32; CDAPC uses L256/I128/W16/K4. Not an isolated method comparison.",
            "inputs": {str(p): file_hash(p) for p in [diagnostic / "samples.jsonl",
                       *sorted((diagnostic / "B").glob("predictions-*.jsonl.gz")),
                       *sorted((diagnostic / "D").glob("predictions-*.jsonl.gz"))]},
        },
        "per_input": per_input,
    }
    target = ROOT / "reports/pope_coco_quality_diagnosis.json"
    atomic_json(target, result)
    target.with_suffix(".sha256").write_text("".join(
        f"{file_hash(p)}  {p}\n" for p in (Path(__file__), target)))
    for key in ("headline", "format_groups", "rule_judge_transitions_rows",
                "judge_error_categories_rows", "answer_commit_groups", "visible_token_writes"):
        print(key, json.dumps(result[key], ensure_ascii=False))
    print("native_192_diagnostic", json.dumps(small))
    print("REPORT", target)


if __name__ == "__main__":
    main()
