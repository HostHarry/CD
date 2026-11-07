"""Audit existing MMMU predictions without model inference or judge requests."""
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from cdapc_lab.experiments.mmmu_report import question_bootstrap


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def describe(values):
    if not values:
        return {"n": 0}
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                p90=float(np.quantile(a, .9)), min=float(a.min()), max=float(a.max()))


def transitions(native, other, ids):
    cells = Counter()
    for sid in ids:
        a, b = native[sid]["judge"], other[sid]["judge"]
        cells["both_correct" if a and b else "fixed" if b else
              "broken" if a else "both_wrong"] += 1
    return {"n": len(ids), **cells,
            "net_correct": cells["fixed"] - cells["broken"]}


def audit(method, index, tokenizer):
    totals, counts, checks = Counter(), Counter(), Counter()
    values, histograms = defaultdict(list), defaultdict(Counter)
    samples, switches = {}, []
    for shard in range(3):
        run = ROOT / index[f"shard{shard}"]
        config = read(run / "resolved_config.json")
        cd = config.get("cdapc")
        judge_dirs = list((run / "judge").iterdir())
        assert len(judge_dirs) == 1
        for path in sorted((run / "predictions").glob("*.json")):
            p = read(path)
            sid = p["sample_id"]
            assert p["status"] == "success" and sid not in samples
            score = read(run / "scores" / index[f"eval{shard}"] / path.name)
            judge = read(judge_dirs[0] / path.name)
            assert judge["score"] in (0, 1)
            text = score["normalized"]["text"]
            raw = p["raw_response_ids"]
            first = p["first_eos_position"]
            visible = p["emitted_tokens_before_eos"]
            counts["samples"] += 1
            counts["official_correct"] += int(score["official"])
            counts["strict_correct"] += int(score["strict_accuracy"])
            counts["judge_correct"] += judge["score"]
            counts["has_think_close"] += "</think>" in text
            counts["empty_after_think_close"] += (
                "</think>" in text and not text.rsplit("</think>", 1)[1].strip())
            counts["length_le_5"] += visible <= 5
            counts["no_eos"] += first is None
            histograms["strict_status"][score["strict_status"]] += 1
            histograms["finish_reason"][p["finish_reason"]] += 1
            for key in ("decode_iterations", "emitted_tokens_before_eos",
                        "committed_buffer_tokens", "generation_seconds"):
                values[key].append(p[key])
            samples[sid] = {"judge": judge["score"], "text": text,
                            "gold": score["gold"], "switches": 0,
                            "visible_switches": 0, "opening_switch": False,
                            "steps": p["decode_iterations"], "length": visible}
            if cd is None:
                continue
            totals.update(p["metrics"])
            totals["iterations"] += p["decode_iterations"]
            log = p["commit_log"]
            by_iteration = defaultdict(list)
            pending = set(range(len(raw)))
            reconstructed = [p["mask_id"]] * len(raw)
            observed = 0
            first_eos_iteration = None
            for e in log:
                by_iteration[e["iteration"]].append(e)
            assert sorted(by_iteration) == list(range(p["decode_iterations"]))
            for iteration, entries in by_iteration.items():
                candidates = sorted(pending)[:cd["window"]["width"]]
                observed += len(candidates)
                positions = [e["position"] for e in entries]
                reason = entries[0]["reason"]
                assert all(e["reason"] == reason for e in entries)
                assert len(positions) == len(set(positions))
                assert set(positions) <= set(candidates)
                assert 1 <= len(entries) <= cd["max_commit_per_iteration"]
                assert reason != "fallback" or len(entries) == 1
                checks["iterations_with_valid_window_and_commit"] += 1
                counts["iterations_leaving_leftmost_pending"] += min(pending) not in positions
                values["physical_window_span"].append(candidates[-1] - candidates[0] + 1)
                histograms["commits_per_iteration"][len(entries)] += 1
                histograms["iteration_reason"][reason] += 1
                if reason == "qualified":
                    assert positions == [e["position"] for e in sorted(
                        entries, key=lambda x: (-x["r_contrast"],
                                               -x["logp_visual_contrast"], x["position"]))]
                    values["qualified_batch_size"].append(len(entries))
                for e in entries:
                    pos = e["position"]
                    u = math.exp(e["logp_visual_contrast"])
                    u_base = math.exp(e["logp_visual_base"])
                    r = e["r_contrast"]
                    pass_u, pass_r = u >= cd["u_gate"]["threshold"], r >= cd["r_gate"]["threshold"]
                    assert u + 1e-6 >= cd["beta"] * u_base
                    assert abs(r - e["pool_mass"] * e["q_contrast"]) < 1e-5
                    if reason == "qualified":
                        assert pass_u and pass_r and e["written_id"] == e["contrast_id"]
                    else:
                        assert not (pass_u and pass_r) and e["written_id"] == e["base_id"]
                        blocker = "both" if not pass_u and not pass_r else "u_only" if not pass_u else "r_only"
                        histograms["fallback_selected_blocker"][blocker] += 1
                        counts["fallback_rejected_contrast_flip"] += e["contrast_id"] != e["base_id"]
                    histograms["pool_size_" + reason][int(e["pool_size"])] += 1
                    for name in ("pool_mass", "pool_size", "r_contrast", "gain_contrast", "gain_base"):
                        values[reason + "_" + name].append(e[name])
                    values[reason + "_u_contrast"].append(u)
                    region = "visible" if pos < visible else "eos" if pos == first else "after_eos"
                    histograms["committed_region"][region] += 1
                    if region == "after_eos":
                        counts["after_eos_written_eos"] += e["written_id"] in p["eos_ids"]
                    reconstructed[pos] = e["written_id"]
                    if e["written_id"] != e["base_id"]:
                        base = tokenizer.decode([e["base_id"]], skip_special_tokens=False)
                        written = tokenizer.decode([e["written_id"]], skip_special_tokens=False)
                        histograms["switch_region"][region] += 1
                        histograms["switch_pair"][json.dumps([base, written], ensure_ascii=False)] += 1
                        histograms["switch_position"][pos] += 1
                        histograms["switch_iteration"][iteration] += 1
                        samples[sid]["switches"] += 1
                        samples[sid]["visible_switches"] += region == "visible"
                        opening = pos == 3 and base == "\n" and written == " The"
                        samples[sid]["opening_switch"] |= opening
                        prefix = tokenizer.decode(raw[:pos], skip_special_tokens=True)
                        after_think = region == "visible" and "</think>" in prefix
                        counts["visible_switch_after_think_close"] += after_think
                        values["switch_gain_advantage"].append(e["gain_contrast"] - e["gain_base"])
                        if method == "cdapc_r90":
                            switches.append({"sample_id": sid, "region": region, "position": pos,
                                             "iteration": iteration, "base": base, "written": written,
                                             "u_contrast": u, "r_contrast": r,
                                             "gain_base": e["gain_base"], "gain_contrast": e["gain_contrast"],
                                             "after_think_close": after_think})
                pending.difference_update(positions)
                eos_positions = [i for i, token in enumerate(reconstructed) if token in p["eos_ids"]]
                if eos_positions and first_eos_iteration is None:
                    first_eos_iteration = iteration
                    counts["first_eos_with_prefix_pending"] += any(i < min(eos_positions) for i in pending)
                can_stop = bool(eos_positions) and not any(i < min(eos_positions) for i in pending)
                assert not can_stop or iteration == p["decode_iterations"] - 1
            if first_eos_iteration is not None:
                values["iterations_after_first_committed_eos"].append(
                    p["decode_iterations"] - 1 - first_eos_iteration)
            assert reconstructed == raw
            assert observed == p["metrics"]["observed"]
            assert len(log) == p["metrics"]["committed"] == p["committed_buffer_tokens"]
            assert sum(e["reason"] == "fallback" for e in log) == p["metrics"]["fallback_iterations"]
            assert sum(e["reason"] == "qualified" for e in log) == p["metrics"]["eligible"]
            assert samples[sid]["switches"] == p["metrics"]["actual_switches"] == p["metrics"]["qualified_flips"]
            assert p["cost"]["model_forward_calls"] == 2 * p["decode_iterations"]
            snapshot_path = run / "snapshots" / path.name
            assert snapshot_path.exists() == bool(samples[sid]["switches"])
            if snapshot_path.exists():
                event = read(snapshot_path)
                entry = next(e for e in log if e["written_id"] != e["base_id"])
                assert event["target_position"] == entry["position"]
                assert event["state"]["iteration"] == entry["iteration"]
            checks["predictions_consistent_with_log_metrics_cost_and_snapshot"] += 1
    counts["samples_with_switch"] = sum(bool(s["switches"]) for s in samples.values())
    counts["samples_with_visible_switch"] = sum(bool(s["visible_switches"]) for s in samples.values())
    counts["samples_with_opening_switch"] = sum(s["opening_switch"] for s in samples.values())
    return {"counts": dict(counts), "mechanism_totals": dict(totals),
            "distributions": {k: describe(v) for k, v in values.items()},
            "histograms": {k: dict(v) for k, v in histograms.items()},
            "validation": dict(checks)}, samples, switches


def main():
    index = read(ROOT / "reports/mmmu_runs.json")
    tokenizer = Tokenizer.from_file(str(ROOT / "models/mmada/tokenizer.json"))
    report, records = {}, {}
    events = []
    for method in ("native", "cdapc_r90", "cdapc_r50"):
        report[method], records[method], switches = audit(method, index[method], tokenizer)
        events.extend(switches)
    ids = set(records["native"])
    assert len(ids) == 900 and all(set(rows) == ids for rows in records.values())
    for method in ("cdapc_r90", "cdapc_r50"):
        rows = records[method]
        subsets = {
            "all": ids,
            "no_switch": {s for s in ids if not rows[s]["switches"]},
            "any_switch": {s for s in ids if rows[s]["switches"]},
            "visible_switch": {s for s in ids if rows[s]["visible_switches"]},
            "opening_switch": {s for s in ids if rows[s]["opening_switch"]},
        }
        report[method]["judge_transitions_vs_native"] = {
            name: transitions(records["native"], rows, selected)
            for name, selected in subsets.items()}
        report[method]["judge_delta_bootstrap"] = question_bootstrap(
            {s: records["native"][s]["judge"] for s in ids},
            {s: rows[s]["judge"] for s in ids})
        report[method]["text_equal_to_native"] = sum(rows[s]["text"] == records["native"][s]["text"] for s in ids)
    for e in events:
        sid = e["sample_id"]
        e["judge_native"] = records["native"][sid]["judge"]
        e["judge_r90"] = records["cdapc_r90"][sid]["judge"]
    report["r90_switch_events"] = events
    report["r90_switched_samples"] = {sid: row for sid, row in records["cdapc_r90"].items() if row["switches"]}
    report["scope"] = {
        "runs": index,
        "method": "Offline audit of saved logs; no additional model inference or judge calls.",
        "limitations": [
            "Proposal observations can count the same position on several iterations.",
            "Full traces are off: independent gate passes and rejected candidate distributions are unavailable.",
            "Fallback blockers describe the selected fallback position, not every pending position.",
            "Native and r50 comparisons are not ablations isolating each mechanism.",
            "Switch means a token differs from base in the same state, not an answer differs from native.",
            "Post-EOS committed tokens are invisible but can influence earlier tokens via bidirectional attention.",
        ],
    }
    output = ROOT / "reports/mmmu_mechanism_audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for method in ("native", "cdapc_r90", "cdapc_r50"):
        r = report[method]
        print(method, json.dumps({k: r[k] for k in (
            "counts", "mechanism_totals", "validation", "judge_transitions_vs_native",
            "judge_delta_bootstrap") if k in r}, ensure_ascii=False))
    print(output)


if __name__ == "__main__":
    main()
