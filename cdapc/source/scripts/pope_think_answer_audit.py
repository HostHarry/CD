"""Separate missing answers from rejected answer formats in saved POPE outputs.

This is an offline audit: production parsers and scored runs are never modified.
Gold labels are used only after extracting an answer, to measure correctness.
"""
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from cdapc_lab.evaluation.parsers import reasoned_yes_no

ROOT = Path(__file__).resolve().parents[1]
STAGES = (
    "pope_aokvqa_reasoning_r90",
    "pope_gqa_reasoning_r90",
    "pope_gqa_reasoning_r70",
)
SOFT_TAG = re.compile(r"</?(?:think)+\s*>|(?<![a-z])(?:think)+\s*>", re.I)
POLARITY = re.compile(r"\b(?:yes|no)\b", re.I)
ANSWER_MARKER = re.compile(r"\b(?:final\s+answer\s*:|(?:the\s+)?answer\s+is\s*:?)", re.I)
FINAL_MARKER = re.compile(r"\bfinal\s+answer\s*:", re.I)
MANUAL_PATH = ROOT / "reports/pope_think_answer_manual.json"
CATEGORIES = {
    "original_parsed": "原规则已解析",
    "empty_shell": "只有 think 标签/空文本，确实无答案",
    "bare_polarity_bad_structure": "仅 Yes/No 加标签，但格式不完整",
    "outside_explicit": "正常闭合后明确 Yes/No，因附加文本被拒绝",
    "malformed_final": "标签或多余符号异常，但可辨最终 Yes/No",
    "body_explicit": "明确 Yes/No 结论仅在 think 正文中",
    "outside_semantic": "最终句用自然语言明确回答，未用独立 Yes/No",
    "body_semantic": "正文明确表达目标物体有/无",
    "fragment_no_reliable_answer": "只有无明确判断的残片",
    "no_reliable_answer": "有内容，但未对所问物体给出可靠可提取的判断",
    "conflicting_answer": "答案存在冲突，无法唯一提取",
    "needs_review": "待逐例核对",
}


def read(path):
    return json.loads(path.read_text())


def leading_polarity(region):
    """Require a leading Yes/No and no opposite polarity in that region."""
    match = re.match(r"\s*(yes|no)\b", region, re.I)
    if match and len({m.group().lower() for m in POLARITY.finditer(region)}) == 1:
        return match.group(1).lower()
    return None


def classify(text):
    parsed = reasoned_yes_no(text)
    if parsed.prediction is not None:
        return "original_parsed", parsed.prediction
    content = SOFT_TAG.sub("", text).strip(" \t\r\n>")
    if not content:
        return "empty_shell", None
    bare = re.fullmatch(r"(yes|no)[.!]?", content, re.I)
    if bare:
        return "bare_polarity_bad_structure", bare.group(1).lower()

    opening, closing = text.rfind("<think>"), text.rfind("</think>")
    if closing >= 0 and closing >= opening:
        suffix = text[closing + len("</think>"):].strip()
        markers = list(FINAL_MARKER.finditer(suffix))
        region = suffix[markers[-1].end():] if markers else suffix
        answer = leading_polarity(region)
        if answer:
            return "outside_explicit", answer
        answer = leading_polarity(region.lstrip(" \t\r\n>"))
        if answer:
            return "malformed_final", answer

    # Repair recognizable closing-tag typos for audit only, e.g. </thinkthink>.
    tags = list(SOFT_TAG.finditer(text))
    if tags and tags[-1].start() > 0 and tags[-1].group().lower() != "<think>":
        suffix = text[tags[-1].end():].strip(" \t\r\n>")
        markers = list(FINAL_MARKER.finditer(suffix))
        region = suffix[markers[-1].end():] if markers else suffix
        answer = leading_polarity(region)
        if answer:
            return "malformed_final", answer

    markers = list(ANSWER_MARKER.finditer(content))
    if markers:
        answer = leading_polarity(content[markers[-1].end():])
        if answer:
            return "body_explicit", answer
    answer = leading_polarity(content)
    if answer:
        return "body_explicit", answer
    # A separate concluding paragraph can carry an answer inside an unclosed block.
    starts = list(re.finditer(r"(?:^|\n)\s*(?:yes|no)\b", content, re.I))
    if starts:
        answer = leading_polarity(content[starts[-1].start():])
        if answer:
            return "body_explicit", answer
    if len(re.findall(r"\w+", content)) <= 3 and not POLARITY.search(content):
        return "fragment_no_reliable_answer", None
    return "needs_review", None


def write_report(output):
    names = ("A-OKVQA r90", "GQA r90", "GQA r70")
    values = [output["stages"][stage] for stage in STAGES]
    lines = [
        "# POPE think：缺答案与解析拒绝审计",
        "",
        "审计覆盖已完成的 A-OKVQA think r90、GQA think r90/r70，共 15,487 个唯一输入。"
        "使用 first_eos_v1 实际送评分的完整文本；不把 EOS 后未展示的画布算作答案。"
        "先用固定规则分类，再逐例核对 35 条特殊输出。提取规则不使用 gold；"
        "提取后才与 gold 比较。没有改动原始解析器、评分文件或实验队列。",
        "",
        "结论：未解析不等于没有答案。多数未解析记录中有可辨认的 Yes/No，"
        "但标签未闭合、答案带解释、标签损坏或答案只出现在 think 内。"
        "同时也有大量只有标签的空壳，必须继续记为无答案。",
        "",
        "以下按每组 9,000 条评测记录计数，与运行报告的未解析率同口径。"
        "跨子集复用输入按原权重计数；完整 JSON 同时保存唯一输入计数。",
        "",
        "| 未解析类型 | " + " | ".join(names) + " |",
        "|---|---:|---:|---:|",
    ]
    groups = [
        ("标签空壳，确实无答案", ("empty_shell",)),
        ("仅 Yes/No 加标签，结构不完整", ("bare_polarity_bad_structure",)),
        ("正常闭合后的 Yes/No 带解释", ("outside_explicit",)),
        ("其他明确答案：标签错误、正文结论、自然语言", (
            "malformed_final", "body_explicit", "body_semantic", "outside_semantic")),
        ("残片、无法可靠确定或冲突", (
            "fragment_no_reliable_answer", "no_reliable_answer", "conflicting_answer")),
    ]
    for label, categories in groups:
        lines.append("| " + label + " | " + " | ".join(
            str(sum(v["weighted_categories"].get(c, 0) for c in categories)) for v in values) + " |")
    lines += [
        "| 原规则未解析合计 | " + " | ".join(str(v["unresolved_rows"]) for v in values) + " |",
        "",
        "| 口径 | " + " | ".join(names) + " |",
        "|---|---:|---:|---:|",
    ]
    for key, label in (
        ("strict", "原规则 Accuracy %"),
        ("final_sentence", "仅允许正常闭合后 Yes/No 带解释 %"),
        ("plus_bare", "再接受只有标签加 Yes/No 的明确回答 %"),
        ("explicit_text", "再接受明确正文答案/可辨标签错误 %"),
        ("semantic_text", "再加入逐例核对的自然语言答案 %"),
    ):
        lines.append("| " + label + " | " + " | ".join(
            f"{100*v['metrics'][key]['accuracy']:.2f}" for v in values) + " |")
    direct = []
    for stage in STAGES:
        r = read(ROOT / "reports" / (stage.replace("_reasoning", "") + ".json"))
        direct.append(sum(s["accuracy"] for s in r["cdapc"]["splits"].values()) / 3)
    lines += [
        "| 对应直答 Accuracy % | " + " | ".join(f"{100*a:.2f}" for a in direct) + " |",
        "",
        "后四行是离线审计口径，不替代原来的最终答案格式评分。"
        "接受 think 内回答后，评测目标从“合规的最终回答”扩展为“文本里可辨认的回答”。"
        "无答案、无法确定及冲突仍计错，不随机猜测，也不把没有 Yes 自动当作 No。"
        "即使采用最宽的审计口径，结果仍低于对应直答；格式限制解释了部分降幅，"
        "但不能把全部问题归给解析器。",
        "",
        "解析器的两个关键限制位于 `src/cdapc_lab/evaluation/parsers.py::reasoned_yes_no`："
        "打开 think 后没有正确关闭，直接返回 unclosed_think；"
        "最终区域必须完整匹配 `yes|no` 加可选句号/感叹号，附加解释会匹配失败。"
        "因此 `</think> Yes, there is a toilet in the image.` 也被标成未解析。",
        "",
        "全部标签空壳都有 EOS；可见文本中确实没有答案词。"
        "另有 GQA r70 的一个唯一输入（2 条评测记录）输出"
        "`Yes, there is no person in the image.`，含相反判断，保留为冲突未解析。",
        "",
        "## 唯一输入计数",
        "",
        "| 类别 | " + " | ".join(names) + " |",
        "|---|---:|---:|---:|",
    ]
    for category, label in CATEGORIES.items():
        if category in ("original_parsed", "needs_review"):
            continue
        lines.append("| " + label + " | " + " | ".join(
            str(v["unique_categories"].get(category, 0)) for v in values) + " |")
    lines += [
        "| 未解析唯一输入合计 | " + " | ".join(str(v["unresolved_unique"]) for v in values) + " |",
        "| 全部唯一输入 | " + " | ".join(str(v["unique_inputs"]) for v in values) + " |",
        "",
        "## 原文案例",
        "",
    ]
    cases = [
        ("pope_aokvqa_reasoning_r90", "1c09f4a195dbc4e56d014ced165512caefd995e06722ea1f5d6ed2f747d06a89"),
        ("pope_gqa_reasoning_r70", "b60638be3419447db6615efd854def053b907ea89fc81399c514bdc224a96bc6"),
        ("pope_gqa_reasoning_r70", "112fee29f9d24cf8952cff8c794122c427dbca3e1076a5c9f8379d9e3ad8be17"),
        ("pope_gqa_reasoning_r70", "3b640d37bfd5a6ce2a018e10e15d7b0082c856600644c58b00c64502590270a7"),
    ]
    for stage, sid in cases:
        record = output["stages"][stage]["per_input"][sid]
        lines += [
            f"### {stage} / {sid}", "",
            f"Question: {record['question']}",
            f"Gold: {record['gold']}; 分类: {CATEGORIES[record['category']]}; "
            f"审计提取: {record['answer']}; 原解析原因: {record['strict_rule']}",
            "", "```text", record["text"], "```", "",
        ]
    lines += [
        "## 复现与文件", "",
        "运行 `.venv/bin/python scripts/pope_think_answer_audit.py`。",
        "脚本逐条验证原解析结果和评分，并核对 9,000 条加权准确率/未解析率与原报告一致。",
        "",
        "- `reports/pope_think_answer_audit.json`：所有输出、逐条分类、权重、原文案例和不同口径结果。",
        "- `reports/pope_think_answer_manual.json`：35 条特殊输出的判定、理由及原文 SHA256。",
        "- `reports/pope_think_answer_review.json`：尚待核对的记录；本轮为空。",
        "- `reports/pope_think_answer_audit.sha256`：文件校验清单。",
    ]
    path = ROOT / "reports/pope_think_answer_audit.md"
    path.write_text("\n".join(lines) + "\n")
    files = [
        Path(__file__), path, ROOT / "reports/pope_think_answer_audit.json",
        MANUAL_PATH, ROOT / "reports/pope_think_answer_review.json",
        ROOT / "src/cdapc_lab/evaluation/parsers.py",
    ]
    (ROOT / "reports/pope_think_answer_audit.sha256").write_text("".join(
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT)}\n" for p in files))


def audit():
    index = read(ROOT / "reports/chair_pope_extensions_runs.json")
    manual = read(MANUAL_PATH) if MANUAL_PATH.exists() else {}
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "categories": CATEGORIES,
        "definitions": {
            "units": "Unique generated inputs and original 9,000 evaluation rows are both reported",
            "text_scope": "Complete first_eos_v1 normalized output; excludes the canvas after EOS",
            "strict": "Unmodified pope_final_yes_no_v1",
            "final_sentence": "Strict plus clear leading Yes/No after a properly closed think block",
            "plus_bare": "final_sentence plus bare Yes/No with malformed/missing think closure",
            "explicit_text": "Also accept explicit body answers and recognizable tag repairs",
            "semantic_text": "Also accept individually reviewed clear natural-language answers",
            "unresolved": "Always scores zero; no random guess and no absent-word No heuristic",
        },
        "sources": {}, "stages": {},
    }
    review = []
    for stage_name in STAGES:
        stage = index["stages"][stage_name]
        assert stage["status"] == "complete"
        report_path = ROOT / stage["report"]
        report = read(report_path)
        config = read(ROOT / report["runs"][0] / "resolved_config.json")
        data = read(ROOT / config["protocol"]["data_root"] / "rows.json")
        weights = Counter(row["input_key"] for row in data)
        questions = {row["input_key"]: row["question"] for row in data}
        counts, weighted = Counter(), Counter()
        correctness, weighted_correctness = Counter(), Counter()
        reasons = Counter()
        records = {}
        examples = defaultdict(list)
        for item in report["cdapc"]["per_run"]:
            score_root = ROOT / item["run"] / "scores" / item["evaluation_fingerprint"]
            for path in sorted(score_root.glob("*.json")):
                if path.name == "evaluation_manifest.json":
                    continue
                row = read(path)
                sid, text = row["sample_id"], row["normalized"]["text"]
                assert sid not in records and row["generation_status"] == "success"
                parsed = reasoned_yes_no(text)
                assert parsed.prediction == row["parsed"]["prediction"]
                assert row["official"] == int(parsed.prediction == row["gold"])
                category, answer = classify(text)
                decision = manual.get(stage_name, {}).get(sid)
                if decision is not None:
                    assert category == "needs_review", (stage_name, sid, category)
                    assert decision["text_sha256"] == hashlib.sha256(text.encode()).hexdigest()
                    category, answer = decision["category"], decision["answer"]
                    assert category in CATEGORIES and category != "needs_review"
                    assert answer in (None, "yes", "no")
                counts[category] += 1
                weighted[category] += weights[sid]
                correctness[category] += int(answer == row["gold"])
                weighted_correctness[category] += weights[sid] * int(answer == row["gold"])
                if parsed.prediction is None:
                    reasons[parsed.rule_id] += 1
                record = {
                    "sample_id": sid, "question": questions[sid], "text": text,
                    "category": category, "answer": answer, "gold": row["gold"],
                    "strict_answer": parsed.prediction, "strict_rule": parsed.rule_id,
                    "evaluation_row_weight": weights[sid],
                    "output_tokens": row["normalized"]["emitted_tokens_before_eos"],
                    "has_eos": row["normalized"]["first_eos_position"] is not None,
                    "score_path": str(path.relative_to(ROOT)),
                }
                if decision:
                    record["review"] = decision["reason"]
                records[sid] = record
                if len(examples[category]) < 4:
                    examples[category].append(record)
                if category == "needs_review":
                    review.append({
                        "stage": stage_name, "sample_id": sid, "question": questions[sid],
                        "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    })
        assert set(records) == set(weights)
        assert sum(weighted.values()) == len(data) == 9000
        n_strict = sum(r["strict_answer"] is not None for r in records.values())
        strict_correct = sum(
            weights[sid] * int(r["strict_answer"] == r["gold"]) for sid, r in records.items())
        assert abs(strict_correct / 9000 -
                   sum(s["accuracy"] for s in report["cdapc"]["splits"].values()) / 3) < 1e-10
        strict_invalid = sum(weights[sid] for sid, r in records.items() if r["strict_answer"] is None)
        assert abs(strict_invalid / 9000 -
                   sum(s["invalid_ratio"] for s in report["cdapc"]["splits"].values()) / 3) < 1e-10
        variants = {
            "strict": {"original_parsed"},
            "final_sentence": {"original_parsed", "outside_explicit"},
            "plus_bare": {"original_parsed", "outside_explicit", "bare_polarity_bad_structure"},
            "explicit_text": {"original_parsed", "outside_explicit", "bare_polarity_bad_structure",
                              "malformed_final", "body_explicit"},
            "semantic_text": set(CATEGORIES) - {"needs_review", "empty_shell",
                "fragment_no_reliable_answer", "no_reliable_answer", "conflicting_answer"},
        }
        metrics = {}
        for variant, accepted in variants.items():
            by_split = {}
            for split in ("random", "popular", "adversarial"):
                pairs = [(records[r["input_key"]]["answer"] if
                          records[r["input_key"]]["category"] in accepted else None, r["label"])
                         for r in data if r["split"] == split]
                by_split[split] = {
                    "n": len(pairs), "correct": sum(a == g for a, g in pairs),
                    "unresolved": sum(a is None for a, _ in pairs),
                    "accuracy": sum(a == g for a, g in pairs) / len(pairs),
                }
            metrics[variant] = {
                "splits": by_split,
                "accuracy": sum(s["correct"] for s in by_split.values()) / 9000,
                "unresolved": sum(s["unresolved"] for s in by_split.values()),
            }
        output["stages"][stage_name] = {
            "unique_inputs": len(records), "evaluation_rows": len(data),
            "unresolved_unique": len(records) - n_strict, "unresolved_rows": strict_invalid,
            "unique_categories": counts, "weighted_categories": weighted,
            "unique_answer_correct": correctness, "weighted_answer_correct": weighted_correctness,
            "parser_reasons": reasons, "metrics": metrics, "examples": examples,
            "per_input": records,
        }
        output["sources"][stage["report"]] = hashlib.sha256(report_path.read_bytes()).hexdigest()
        print(stage_name, "unique", dict(counts), "weighted", dict(weighted),
              "accuracies", {k: round(100*v["accuracy"], 3) for k, v in metrics.items()}, flush=True)
    (ROOT / "reports/pope_think_answer_audit.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    (ROOT / "reports/pope_think_answer_review.json").write_text(
        json.dumps(review, indent=2, ensure_ascii=False) + "\n")
    if not review:
        write_report(output)
    print("REVIEW_REMAINING", len(review), flush=True)
    return output


if __name__ == "__main__":
    audit()
