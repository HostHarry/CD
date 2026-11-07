"""Reconstruct MMMU generation order from saved commits, without inference."""
import json
import math
import re
from collections import Counter, defaultdict

from tokenizers import Tokenizer

from mmmu_mechanism_audit import ROOT, describe, question_bootstrap, read, transitions


CASE_IDS = {
    "validation_Basic_Medical_Science_5",
    "validation_Art_Theory_2",
    "validation_Public_Health_27",
}


def token_at_character(ids, offset, tokenizer):
    """Find the token spanning an ASCII marker/answer character in decoded text."""
    low, high = 1, len(ids)
    while low < high:
        middle = (low + high) // 2
        if len(tokenizer.decode(ids[:middle], skip_special_tokens=True)) > offset:
            high = middle
        else:
            low = middle + 1
    return low - 1


def analyze(method, index, tokenizer):
    counts, regions, hist = Counter(), defaultdict(Counter), defaultdict(Counter)
    values, rows, cases = defaultdict(list), {}, {}
    for shard in range(3):
        run = ROOT / index[f"shard{shard}"]
        judges = list((run / "judge").iterdir())
        assert len(judges) == 1
        for path in sorted((run / "predictions").glob("*.json")):
            p = read(path)
            score = read(run / "scores" / index[f"eval{shard}"] / path.name)
            judge = read(judges[0] / path.name)["score"]
            assert p["status"] == "success" and judge in (0, 1)
            sid, n = p["sample_id"], p["emitted_tokens_before_eos"]
            ids = p["raw_response_ids"][:n]
            text = tokenizer.decode(ids, skip_special_tokens=True)
            assert text.strip() == score["normalized"]["text"]
            close = text.rfind("</think>")
            suffix = text[close + len("</think>"):] if close >= 0 else ""
            has_open = "<think>" in text
            structure = ("closed_with_suffix" if close >= 0 and suffix.strip() else
                         "closed_without_suffix" if close >= 0 else
                         "open_without_close" if has_open else "no_complete_think_tags")
            hist["structure"][structure] += 1
            hist["structure_correct"][structure] += judge
            grams = Counter(tuple(ids[i:i + 8]) for i in range(max(0, len(ids) - 7)))
            repeated = max(grams.values(), default=0) >= 3
            counts["repeated_8gram_at_least_3_times"] += repeated
            counts["adjacent_TheThe"] += "TheThe" in text
            counts["contains_thinkthink"] += "thinkthink" in text
            counts["think_only"] += text.strip() == "<think>"
            counts["think_only_judge_correct"] += (text.strip() == "<think>") * judge
            counts["samples"] += 1
            counts["judge_correct"] += judge
            summary = {"judge": judge, "steps": p["decode_iterations"], "length": n,
                       "structure": structure, "repeated_8gram": repeated,
                       "think_only": text.strip() == "<think>", "gold": score["gold"]}
            rows[sid] = summary
            if sid in CASE_IDS:
                cases[sid] = {"text": score["normalized"]["text"], "gold": score["gold"],
                              "judge": judge, "steps": p["decode_iterations"], "length": n}
            if method == "native":
                continue
            threshold = .9 if method == "cdapc_r90" else .5
            entries = p["commit_log"]
            iterations = defaultdict(list)
            position_entry = {}
            for e in entries:
                iterations[e["iteration"]].append(e)
                position_entry[e["position"]] = e
                region = ("visible" if e["position"] < n else
                          "first_eos" if e["position"] == p["first_eos_position"] else "after_eos")
                r = regions[region]
                r["committed"] += 1
                r[e["reason"]] += 1
                r["switch"] += e["written_id"] != e["base_id"]
                if e["reason"] == "qualified":
                    r["qualified_pool1"] += e["pool_size"] == 1
                    r["qualified_visual_base_below_r_threshold"] += math.exp(e["logp_visual_base"]) < threshold
                    values[region + "_qualified_visual_probability"].append(math.exp(e["logp_visual_contrast"]))
                else:
                    r["fallback_visual_base_above_r_threshold"] += math.exp(e["logp_visual_base"]) >= threshold
                if e["pool_size"] == 1:
                    assert e["base_id"] == e["contrast_id"]
                    assert abs(e["r_contrast"] - math.exp(e["logp_visual_base"])) < 1e-5
                if e["written_id"] != e["base_id"]:
                    base = tokenizer.decode([e["base_id"]], skip_special_tokens=False)
                    written = tokenizer.decode([e["written_id"]], skip_special_tokens=False)
                    category = "opening_newline_to_The" if e["position"] == 3 and base == "\n" and written == " The" else "other_switch"
                    for key, value in {
                        "visual_probability": math.exp(e["logp_visual_contrast"]),
                        "ablated_probability": math.exp(e["logp_visual_contrast"] - e["gain_contrast"]),
                        "r": e["r_contrast"], "gain": e["gain_contrast"],
                    }.items():
                        values[category + "_" + key].append(value)
            if text.strip() == "<think>":
                eos_entry = position_entry[p["first_eos_position"]]
                hist["think_only_eos_reason"][eos_entry["reason"]] += 1
                counts["think_only_eos_switch"] += eos_entry["written_id"] != eos_entry["base_id"]
                values["think_only_steps"].append(p["decode_iterations"])
                values["think_only_eos_visual_probability"].append(math.exp(eos_entry["logp_visual_contrast"]))
                values["think_only_eos_r"].append(eos_entry["r_contrast"])

            pending, known_eos = set(range(p["full_buffer_length"])), None
            tail_steps = tail_tokens = longest = streak = 0
            timeline = []
            for iteration, es in iterations.items():
                old_eos = known_eos
                only_tail = old_eos is not None and all(e["position"] > old_eos for e in es)
                holes_before_eos = sum(pos < old_eos for pos in pending) if old_eos is not None else None
                if only_tail:
                    tail_steps += 1
                    tail_tokens += len(es)
                    streak += 1
                    longest = max(longest, streak)
                    counts["known_eos_tail_only_iterations"] += 1
                    counts["known_eos_tail_only_qualified_iterations"] += es[0]["reason"] == "qualified"
                    counts["known_eos_tail_only_commits"] += len(es)
                    counts["known_eos_tail_only_eos_commits"] += sum(e["written_id"] in p["eos_ids"] for e in es)
                    hist["prefix_holes_in_tail_only_iteration"][holes_before_eos] += 1
                    hist["tail_only_commit_batch_size"][len(es)] += 1
                else:
                    streak = 0
                for e in es:
                    pending.remove(e["position"])
                    if e["written_id"] in p["eos_ids"]:
                        known_eos = e["position"] if known_eos is None else min(known_eos, e["position"])
                if sid in CASE_IDS:
                    timeline.append({"iteration": iteration, "known_eos_before": old_eos,
                                     "prefix_holes_before": holes_before_eos, "tail_only": only_tail,
                                     "reason": es[0]["reason"], "positions": [e["position"] for e in es],
                                     "tokens": [tokenizer.decode([e["written_id"]], skip_special_tokens=False) for e in es]})
            counts["samples_with_known_eos_tail_only_iterations"] += tail_steps > 0
            counts["total_iterations"] += p["decode_iterations"]
            values["tail_only_steps_per_sample"].append(tail_steps)
            values["longest_tail_only_streak"].append(longest)
            last_entries = iterations[p["decode_iterations"] - 1]
            hist["last_iteration_reason"][last_entries[0]["reason"]] += 1
            for e in last_entries:
                if e["position"] < n:
                    hist["last_visible_tokens"][tokenizer.decode([e["written_id"]], skip_special_tokens=False)] += 1
            summary.update(tail_only_steps=tail_steps, tail_only_tokens=tail_tokens,
                           longest_tail_only_streak=longest,
                           visible_fallback=sum(e["reason"] == "fallback" and e["position"] < n for e in entries))

            # This probe covers only an explicit option letter after the final closing tag.
            match = re.match(r"\s*(?:\(([A-Z])\)|([A-Z])(?=\s*(?:[.)!:]|$)))", suffix)
            if close >= 0 and match and (match.group(1) or match.group(2)) in p["options"]:
                group = 1 if match.group(1) else 2
                answer_offset = close + len("</think>") + match.start(group)
                answer_pos = token_at_character(ids, answer_offset, tokenizer)
                assert tokenizer.decode(ids[:answer_pos + 1], skip_special_tokens=True)[answer_offset] == match.group(group)
                close_pos = token_at_character(ids, close, tokenizer)
                answer_iteration = position_entry[answer_pos]["iteration"]
                late_body = [pos for pos in range(close_pos)
                             if position_entry[pos]["iteration"] > answer_iteration]
                letter = match.group(group)
                counts["explicit_suffix_option_samples"] += 1
                counts["suffix_option_before_body_complete"] += bool(late_body)
                counts["suffix_option_written_by_fallback"] += position_entry[answer_pos]["reason"] == "fallback"
                counts["suffix_option_written_by_switch"] += position_entry[answer_pos]["written_id"] != position_entry[answer_pos]["base_id"]
                hist["suffix_option_judge_correct"]["body_incomplete" if late_body else "body_complete"] += judge
                hist["suffix_option_group_n"]["body_incomplete" if late_body else "body_complete"] += 1
                values["body_holes_after_suffix_option_commit"].append(len(late_body))
                summary.update(suffix_option=letter, suffix_option_iteration=answer_iteration,
                               body_holes_after_suffix_option_commit=len(late_body),
                               suffix_option_switch=position_entry[answer_pos]["written_id"] != position_entry[answer_pos]["base_id"],
                               suffix_option_base=tokenizer.decode([position_entry[answer_pos]["base_id"]], skip_special_tokens=False),
                               suffix_option_written=tokenizer.decode([position_entry[answer_pos]["written_id"]], skip_special_tokens=False),
                               suffix_option_choices=list(p["options"]))
                if sid in CASE_IDS:
                    cases[sid]["suffix_option_probe"] = {
                        "letter": letter, "token_position": answer_pos, "iteration": answer_iteration,
                        "body_holes_after_commit": late_body,
                        "reason": position_entry[answer_pos]["reason"]}
            if sid in CASE_IDS:
                cases[sid]["timeline"] = timeline
                cases[sid]["last_commits"] = [
                    {**e, "token": tokenizer.decode([e["written_id"]], skip_special_tokens=False)}
                    for e in entries[-8:]]
    return {"counts": dict(counts), "regions": {k: dict(v) for k, v in regions.items()},
            "histograms": {k: dict(v) for k, v in hist.items()},
            "distributions": {k: describe(v) for k, v in values.items()},
            "samples": rows, "cases": cases}


def main():
    index = read(ROOT / "reports/mmmu_runs.json")
    tokenizer = Tokenizer.from_file(str(ROOT / "models/mmada/tokenizer.json"))
    report = {method: analyze(method, index[method], tokenizer)
              for method in ("native", "cdapc_r90", "cdapc_r50")}
    sets = [set(value["samples"]) for value in report.values()]
    assert len(sets[0]) == 900 and all(s == sets[0] for s in sets)
    r90, r50 = report["cdapc_r90"]["samples"], report["cdapc_r50"]["samples"]
    ids = sorted(r90)
    subsets = {
        "all": ids,
        "became_think_only": [s for s in ids if r50[s]["think_only"] and not r90[s]["think_only"]],
        "recovered_from_think_only": [s for s in ids if r90[s]["think_only"] and not r50[s]["think_only"]],
        "lost_closed_suffix": [s for s in ids if r90[s]["structure"] == "closed_with_suffix"
                               and r50[s]["structure"] != "closed_with_suffix"],
        "gained_closed_suffix": [s for s in ids if r50[s]["structure"] == "closed_with_suffix"
                                 and r90[s]["structure"] != "closed_with_suffix"],
        "r50_explicit_suffix_switch": [s for s in ids if r50[s].get("suffix_option_switch")],
    }
    local_letter_flips = []
    for sid in subsets["r50_explicit_suffix_switch"]:
        row = r50[sid]
        base, written = row["suffix_option_base"].strip(), row["suffix_option_written"].strip()
        if base in row["suffix_option_choices"] and written in row["suffix_option_choices"]:
            local_letter_flips.append({"sample_id": sid, "base": base, "written": written,
                                      "gold": row["gold"], "judge_r50": row["judge"],
                                      "judge_r90": r90[sid]["judge"]})
    report["r50_vs_r90"] = {
        "judge_transitions": {name: transitions(r90, r50, selected) for name, selected in subsets.items()},
        "judge_delta_bootstrap": question_bootstrap({s: r90[s]["judge"] for s in ids},
                                                   {s: r50[s]["judge"] for s in ids}),
        "r50_suffix_switch_base_tokens": dict(Counter(r50[s]["suffix_option_base"]
                                                      for s in subsets["r50_explicit_suffix_switch"])),
        "r50_suffix_local_letter_flips": local_letter_flips,
    }
    metadata = {r["sample_id"]: r for r in read(ROOT / "data/mmmu_val/samples.json")}
    report["case_questions"] = {sid: {k: metadata[sid][k] for k in ("question", "options")} for sid in sorted(CASE_IDS)}
    report["limitations"] = [
        "Offline reconstruction; no new model inference and no counterfactual continuation.",
        "Known-EOS suffix-only iterations are identified using state available before that iteration.",
        "Skipping suffix writes can change preceding tokens through bidirectional attention.",
        "Suffix-option ordering uses only explicit letters after a complete closing think tag.",
        "Repetition is a token-level proxy, not an independent assessment of reasoning correctness.",
    ]
    out = ROOT / "reports/mmmu_generation_patterns.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for method in ("native", "cdapc_r90", "cdapc_r50"):
        r = report[method]
        print(method, json.dumps({k: r[k] for k in ("counts", "regions", "histograms") if k != "histograms"}))
        print("structure", r["histograms"]["structure"], "correct", r["histograms"]["structure_correct"])
    print(out)


if __name__ == "__main__":
    main()
