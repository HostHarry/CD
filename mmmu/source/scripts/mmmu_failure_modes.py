"""Classify saved MMMU output structure and preserve reviewed failure examples."""
import itertools
import json
import re
from collections import Counter

from export_mmmu_answers import ROOT, read


KINDS = ("empty_output", "think_tags_only", "no_think_tags", "closing_without_open",
         "unclosed_think", "closed_without_suffix", "closed_with_suffix")
REVIEWS = [
    ("cdapc_r90", "validation_Psychology_22", ["direct_answer"],
     "输出 B 与 gold B 相同，没有 think 标签本身不构成回答失败。"),
    ("cdapc_r50", "validation_Accounting_2", ["empty_think"],
     "输出恰好为 <think>，没有推理正文或答案。"),
    ("native", "validation_Accounting_11", ["incomplete_reasoning", "no_final_answer"],
     "正文结束于未完成的 The，随后闭合 think，没有给出损失数值或选项；gold 为 A / $9.00。"),
    ("cdapc_r90", "validation_Accounting_29", ["logical_error", "answer_inside_think"],
     "正文同时称 A、B 波动率为 5%，随后说 A 的 5% 高于 B，内部逻辑矛盾；答案 A 写在 think 内，gold B。"),
    ("cdapc_r90", "validation_Accounting_1", ["arithmetic_error"],
     "正文明确写 10 × 30 = 30，算术错误；末尾 D 与 gold B 不同。"),
    ("cdapc_r50", "validation_Accounting_1", ["arithmetic_error", "malformed_think_close"],
     "正文把 10,000 / 25,000 写为每单位 $4，按其给出的数字应为 $0.4；闭合标记为 </thinkthink>，没有最终选项。"),
    ("native", "validation_Diagnostics_and_Laboratory_Medicine_2", ["body_final_conflict"],
     "正文明确说 The answer is D，think 外却写 B；gold B、judge=1，说明答案评分没有排除正文矛盾。"),
    ("cdapc_r50", "validation_Clinical_Medicine_21", ["repetition"],
     "正文大量机械重复 abdominal，末尾 B 与 gold 相同且 judge=1；正确选项不等于有效推理。"),
]


def classify(text):
    text = text.strip()
    opening, closing = text.find("<think>"), text.rfind("</think>")
    if not text:
        return "empty_output"
    if not re.sub(r"</?think>", "", text).strip():
        return "think_tags_only"
    if opening < 0 and closing < 0:
        return "no_think_tags"
    if opening < 0:
        return "closing_without_open"
    if closing < opening:
        return "unclosed_think"
    return "closed_with_suffix" if text[closing + 8:].strip() else "closed_without_suffix"


def main():
    metadata = {row["sample_id"]: row for row in read(ROOT / "data/mmmu_val/samples.json")}
    all_rows, report = {}, {}
    for method in ("native", "cdapc_r90", "cdapc_r50"):
        rows = [json.loads(line) for line in (ROOT / f"reports/mmmu_answers/{method}.jsonl").read_text().splitlines()]
        assert len(rows) == len({r["sample_id"] for r in rows}) == 900
        all_rows[method] = {r["sample_id"]: r for r in rows}
        counts, correct, examples, candidates = Counter(), Counter(), {k: [] for k in KINDS}, []
        assignments, conflicts = {}, []
        for row in rows:
            sid, text = row["sample_id"], row["text"]
            kind = classify(text)
            counts[kind] += 1
            correct[kind] += row["judge_correct"]
            assignments[sid] = {"kind": kind, "judge_correct": row["judge_correct"]}
            if len(examples[kind]) < 5:
                examples[kind].append(sid)
            words = re.findall(r"[a-z]+", text.lower())
            runs = [(len(list(group)), word) for word, group in itertools.groupby(words) if len(word) >= 2]
            longest = max(runs, default=(0, ""))
            if longest[0] >= 8:
                candidates.append({"sample_id": sid, "repeated_word": longest[1],
                                   "consecutive_count": longest[0], "judge_correct": row["judge_correct"]})
            opening, closing = text.find("<think>"), text.rfind("</think>")
            if 0 <= opening < closing:
                body, suffix = text[opening + 7:closing], text[closing + 8:]
                matches = list(re.finditer(
                    r"\b(?:correct\s+(?:answer|option)|(?:final\s+)?answer)\s*(?:is|:)\s*"
                    r"(?:option\s*)?[*\s]*(?:\(([A-Z])\)|([A-Z])\b)", body))
                final = re.match(r"\s*(?:\(([A-Z])\)|([A-Z])(?=\s*(?:[.)!:]|$)))", suffix)
                if matches and final:
                    a, b = matches[-1].group(1) or matches[-1].group(2), final.group(1) or final.group(2)
                    if a != b:
                        conflicts.append({"sample_id": sid, "body_letter": a, "final_letter": b,
                                          "gold": row["gold"], "judge_correct": row["judge_correct"], "text": text})
        assert sum(counts.values()) == 900
        report[method] = {
            "structure": {k: {"n": counts[k], "judge_correct": correct[k],
                              "judge_wrong": counts[k] - correct[k]} for k in KINDS},
            "sample_assignments": assignments, "example_ids": examples,
            "consecutive_word_repeat_candidates": candidates,
            "explicit_letter_conflict_candidates": conflicts,
        }
        print(method, json.dumps(report[method]["structure"]))
        print("repeat candidates:", len(candidates), "explicit letter conflicts:", len(conflicts))
    reviewed = []
    for method, sid, labels, note in REVIEWS:
        row = all_rows[method][sid]
        reviewed.append({**row, "question": metadata[sid]["question"], "options": metadata[sid]["options"],
                         "labels": labels, "review_note": note})
    report["reviewed_examples"] = reviewed
    report["limitations"] = [
        "Structural classes are mutually exclusive, using empty-output and tags-only classes first.",
        "No text after a closing tag does not establish that no answer exists inside the body.",
        "Judge scores assess answers; they are not labels for reasoning correctness.",
        "Semantic error examples were manually checked; no prevalence estimate for all reasoning errors is claimed.",
        "Repetition and explicit-letter-conflict scans are conservative candidate detectors, not exhaustive semantic checks.",
    ]
    output = ROOT / "reports/mmmu_failure_modes.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
