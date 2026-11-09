"""Audit POPE-COCO r90--r50 saved commits without model inference.

Run from any working directory with the experiment repository's Python.
Counts of commits and switches use the 5,127 unique inputs. Official accuracy
uses all 9,000 rows, including repeated inputs across the three POPE splits.
Local polarity corrections compare base and written tokens at the same saved
state; they are not counterfactual generation experiments.
"""
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
TAUS = ("0.9", "0.8", "0.7", "0.6", "0.5")


def read(path):
    return json.loads(path.read_text())


def describe(values):
    return {"count": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "min": min(values),
            "max": max(values)} if values else {}


def write_report(audit, sweep):
    native = read(ROOT / "reports/pope_yesno_mmada_cdapc_L128.json")["native"]["splits"]
    macro = lambda rows, key: 100 * statistics.mean(row[key] for row in rows.values())
    false_positive_rate = lambda rows: 100 * sum(
        row["confusion"]["fp"] for row in rows.values()) / sum(
        row["confusion"]["fp"] + row["confusion"]["tn"] for row in rows.values())
    lines = [
        "# POPE-COCO r90–r50 行为审计",
        "",
        "覆盖每组 5,127 个唯一输入、9,000 条评测记录；三种采样各 3,000 条。"
        "质量指标按官方评测记录计算，F1/Precision/Recall 为三子集等权平均。"
        "机制统计按唯一输入的提交日志计算，不能直接把两种计数相减。",
        "",
        "五组保存配置已核对：模型、输入协议、seed 均相同，仅 r 阈值不同。"
        "固定 alpha=0.5、beta=0.1、u=0.1、W=K=16、画布/迭代上限=128；"
        "使用 pope_yesno_v1 直答协议、sequential 双分支和 eos_prefix_complete 停止。",
        "",
        "| 方法 | 平均步数 | 平均秒数 | Accuracy % | F1 % |",
        "|---|---:|---:|---:|---:|",
        f"| Native | — | — | {macro(native, 'accuracy'):.2f} | {macro(native, 'f1'):.2f} |",
    ]
    for tau in TAUS:
        x, s = audit["thresholds"][tau], sweep[tau]
        lines.append(f"| r{int(float(tau)*100)} | {x['statistics']['steps']['mean']:.2f} | "
                     f"{s['cost']['mean_generation_seconds']:.2f} | "
                     f"{macro(s['splits'], 'accuracy'):.2f} | {macro(s['splits'], 'f1'):.2f} |")
    lines += [
        "",
        "| 方法 | 答案位置 fallback % | 只填已知 EOS 尾部的迭代 % | 实际替换 | 大小写替换 | Yes/No 极性替换 | EOS 后替换 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for tau in TAUS:
        x = audit["thresholds"][tau]
        c, h = x["counts"], x["histograms"]
        categories = h.get("switch_category", {})
        lines.append(
            f"| r{int(float(tau)*100)} | "
            f"{100*h['position_zero_reason'].get('fallback', 0)/c['samples']:.2f} | "
            f"{100*c['tail_only_iterations']/c['decode_iterations']:.2f} | "
            f"{c.get('switches', 0)} | {categories.get('yesno_spelling', 0)} | "
            f"{categories.get('polarity', 0)} | {x['regions']['after_eos'].get('switches', 0)} |")
    lines += [
        "",
        "所有组均无空输出。除 r80 的 3 条回答为 10 tokens 外，全部回答为 1 token。"
        "r90 平均提交 121.17 个位置才输出一个 Yes/No；4,648/5,127 条填满画布后停止，"
        "这些样本全部有 EOS，canvas_filled 不表示缺少 EOS。"
        "r90 中 4,558 次 fallback 全部写在答案位置：占全部迭代 9.46%，"
        "却占全部答案 88.90%。门控和 fallback 的分母必须区分。",
        "",
        "EOS 后的位置仍进入窗口，尾部高置信度 EOS 通过门控时会阻止 fallback。"
        "r90 有 38,060 轮只提交已知 EOS 之后的位置，写入 535,129 个 EOS。"
        "降低阈值使答案位置更早通过门控，缩短等待；r50 仍有 30.59% 的迭代只填尾部。",
        "",
        "r90 无任何非 base token 写入；r80 唯一替换为 Yes→yes。"
        "它们与 Native 的最终答案差异不能归因为直接改写 token，"
        "但对比统计仍可通过门控、提交顺序和停止影响生成轨迹。"
        "r90 通过门控的答案 token 中，只有 40.77% 的候选池为单 token，"
        "不能直接套用 MMMU 的 singleton 比例。",
        "",
        "r70 的 69 次替换分为 63 次大小写、5 次 Yes→No 和 1 次 EOS 后替换。"
        "5 次极性替换在当时状态下均改对。r60 的 39 次极性替换为 30 次改对、9 次改错；"
        "r50 的 169 次为 115 次改对、54 次改错。"
        "这里比较同一保存状态下的 base 与 written token，"
        "不是完整关闭替换的反事实实验，不能当作端到端净收益。"
        "EOS 后替换不直接显示，但可能通过双向注意力影响后续答案提交。",
        "",
        "| 方法 | Precision % | Recall % | Yes ratio % | 负样本误报率 % | 对 Native 改对/改错（9,000 条） |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Native | {macro(native, 'precision'):.2f} | {macro(native, 'recall'):.2f} | "
        f"{macro(native, 'yes_ratio'):.2f} | {false_positive_rate(native):.2f} | — |",
    ]
    for tau in TAUS:
        s = sweep[tau]
        fixed = sum(v["fixed"] for v in s["transitions_per_split"].values())
        broken = sum(v["broken"] for v in s["transitions_per_split"].values())
        lines.append(f"| r{int(float(tau)*100)} | {macro(s['splits'], 'precision'):.2f} | "
                     f"{macro(s['splits'], 'recall'):.2f} | {macro(s['splits'], 'yes_ratio'):.2f} | "
                     f"{false_positive_rate(s['splits']):.2f} | "
                     f"{fixed}/{broken} |")
    lines += [
        "",
        "相对 Native，CDAPC 的 Yes 比例和 Recall 增高，Precision 降低，"
        "负样本误报率上升；F1 提高不能直接解释为物体误报减少。"
        "r70 的平均 Accuracy 比 r90 高 0.078 个百分点，即 9,000 条中净多对 7 条；"
        "原报告按图像重采样的三个子集配对 Accuracy 差值 95% 区间均包含零，"
        "不能宣称 r70 显著优于 r90。r50 最快，平均 Accuracy 最低。",
        "",
        "候选参数可优先保留 r70（本次成本与分数折中）和 r50（速度）。"
        "更明确的因果检验是分别测试 EOS 前缀窗口与 timing_only_crossed；"
        "本审计没有运行这些干预，任何窗口修改也不能先假定为无损。",
        "",
        "复现：在实验仓库运行 `.venv/bin/python scripts/pope_rgate_behavior.py`。"
        "JSON 保存每个输入的文本、标签、步数、尾部等待和全部替换事件。"
        "审计断言覆盖样本集合、配置一致性、commit 与最终画布一致性、"
        "实际写入的门控条件、fallback 写 base、计数与原汇总一致性。",
        "",
        "- 机器可读审计：`reports/pope_rgate_behavior.json`",
        "- 原始扫参报告：`reports/pope_rgate_sweep.json`",
        "- 校验清单：`reports/pope_rgate_behavior.sha256`",
    ]
    report = ROOT / "reports/pope_rgate_behavior.md"
    report.write_text("\n".join(lines) + "\n")
    files = [Path(__file__), report, ROOT / "reports/pope_rgate_behavior.json",
             ROOT / "reports/pope_rgate_sweep.json"]
    (ROOT / "reports/pope_rgate_behavior.sha256").write_text("".join(
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT)}\n"
        for p in files))


def main():
    source = ROOT / "reports/pope_rgate_sweep.json"
    sweep = read(source)
    data = read(ROOT / "data/pope_yesno/rows.json")
    questions = {r["input_key"]: r for r in data}
    weights = Counter(r["input_key"] for r in data)
    tokenizer = Tokenizer.from_file(str(ROOT / "models/mmada/tokenizer.json"))
    decode = lambda token: tokenizer.decode([token], skip_special_tokens=False)
    out = {"source": str(source.relative_to(ROOT)),
           "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
           "unique_inputs": len(questions), "evaluation_rows": len(data),
           "definitions": {
               "tail_only_iteration": "EOS known before iteration; every new commit is after it",
               "visible": "Position strictly before the final first EOS",
               "switch": "Written token differs from base at that same saved state",
               "local_polarity_outcome": "Gold comparison of base/written yes/no tokens; not causal",
               "iteration_number": "One-based in derived per-input records",
           }, "thresholds": {}}
    common_config = None
    for tau in TAUS:
        source_row = sweep[tau]
        counts, hist, regions = Counter(), defaultdict(Counter), defaultdict(Counter)
        values, records, switches = defaultdict(list), {}, []
        examples = []
        for run_name in source_row["runs"]:
            run = ROOT / run_name
            config = read(run / "resolved_config.json")
            comparison = {k: config[k] for k in ("model", "protocol", "cdapc")}
            comparison["seed"] = config["run"]["seed"]
            assert comparison["cdapc"]["r_gate"].pop("threshold") == float(tau)
            if common_config is None:
                common_config = comparison
            assert comparison == common_config
            assert config["recording"]["commit_log"]
            for path in sorted((run / "predictions").glob("*.json")):
                p = read(path)
                assert p["status"] == "success"
                sid, n = p["sample_id"], p["emitted_tokens_before_eos"]
                assert sid in questions and sid not in records
                first_eos = p["first_eos_position"]
                ids = p["raw_response_ids"]
                eos_ids = set(p["eos_ids"])
                found_eos = next((i for i, t in enumerate(ids) if t in eos_ids), None)
                assert first_eos == found_eos
                assert n == (first_eos if first_eos is not None else len(ids))
                text = tokenizer.decode(ids[:n], skip_special_tokens=True).strip()
                counts["samples"] += 1
                counts["empty"] += not text
                counts["one_token_outputs"] += n == 1
                counts["full_canvas_with_eos"] += (
                    p["finish_reason"] == "canvas_filled" and first_eos is not None)
                hist["finish_reason"][p["finish_reason"]] += 1
                hist["text"][text] += 1
                iterations, position_entries = defaultdict(list), {}
                sample_switches = 0
                sample_fallbacks = set()
                for e in p["commit_log"]:
                    position = e["position"]
                    assert position not in position_entries
                    assert ids[position] == e["written_id"]
                    position_entries[position] = e
                    iterations[e["iteration"]].append(e)
                    region = ("visible" if position < n else
                              "first_eos" if position == first_eos else "after_eos")
                    c = regions[region]
                    c["commits"] += 1
                    c[e["reason"]] += 1
                    c["eos_tokens"] += e["written_id"] in eos_ids
                    counts["commits"] += 1
                    if e["reason"] == "qualified":
                        assert e["written_id"] == e["contrast_id"]
                        assert e["r_contrast"] >= float(tau) - 1e-6
                        assert math.exp(e["logp_visual_contrast"]) >= 0.1 - 1e-6
                        c["qualified_pool_one"] += e["pool_size"] == 1
                        if region == "visible":
                            values["qualified_visible_visual_probability"].append(
                                math.exp(e["logp_visual_contrast"]))
                    else:
                        assert e["reason"] == "fallback"
                        assert e["written_id"] == e["base_id"]
                        sample_fallbacks.add(e["iteration"])
                    if e["written_id"] != e["base_id"]:
                        counts["switches"] += 1
                        c["switches"] += 1
                        sample_switches += 1
                        base, written = decode(e["base_id"]), decode(e["written_id"])
                        b, w = base.strip().lower(), written.strip().lower()
                        yesno = b in ("yes", "no") and w in ("yes", "no")
                        category = ("polarity" if yesno and b != w else
                                    "yesno_spelling" if yesno else
                                    "to_eos" if e["written_id"] in eos_ids else
                                    "from_eos" if e["base_id"] in eos_ids else "other")
                        hist["switch_category"][category] += 1
                        hist["switch_pairs"][f"{base!r} -> {written!r}"] += 1
                        outcome = None
                        if category == "polarity":
                            outcome = "fixed" if w == questions[sid]["label"] else "broken"
                            hist["local_polarity_outcome"][outcome] += 1
                            hist["local_polarity_outcome_row_weighted"][outcome] += weights[sid]
                        switches.append({
                            "sample_id": sid, "region": region, "position": position,
                            "iteration": e["iteration"] + 1, "base": base, "written": written,
                            "category": category, "local_outcome": outcome,
                            "gold": questions[sid]["label"], "text": text,
                            "u": math.exp(e["logp_visual_contrast"]), "r": e["r_contrast"],
                            "pool_size": e["pool_size"], "run": run_name,
                        })
                assert len(position_entries) == p["committed_buffer_tokens"]
                assert set(iterations) == set(range(p["decode_iterations"]))
                assert sample_switches == p["metrics"]["actual_switches"]
                assert len(sample_fallbacks) == p["metrics"]["fallback_iterations"]
                counts["fallback_iterations"] += len(sample_fallbacks)
                counts["decode_iterations"] += p["decode_iterations"]
                counts["samples_with_switches"] += sample_switches > 0
                known_eos, pending = None, set(range(p["full_buffer_length"]))
                tail_steps = tail_tokens = longest = streak = 0
                for i in sorted(iterations):
                    batch = iterations[i]
                    tail_only = (known_eos is not None and
                                 all(e["position"] > known_eos for e in batch))
                    if tail_only:
                        assert any(pos < known_eos for pos in pending)
                        tail_steps += 1
                        tail_tokens += len(batch)
                        streak += 1
                        longest = max(longest, streak)
                        counts["tail_only_eos_commits"] += sum(
                            e["written_id"] in eos_ids for e in batch)
                    else:
                        streak = 0
                    for e in batch:
                        pending.remove(e["position"])
                        if e["written_id"] in eos_ids:
                            known_eos = (e["position"] if known_eos is None else
                                         min(known_eos, e["position"]))
                counts["tail_only_iterations"] += tail_steps
                counts["tail_only_commits"] += tail_tokens
                counts["samples_with_tail_only"] += tail_steps > 0
                first = position_entries[0]
                hist["position_zero_reason"][first["reason"]] += 1
                hist["position_zero_iteration"][first["iteration"] + 1] += 1
                for key, value in {
                    "steps": p["decode_iterations"], "output_tokens": n,
                    "commits_per_sample": len(position_entries),
                    "position_zero_iteration": first["iteration"] + 1,
                    "tail_only_iterations": tail_steps,
                    "longest_tail_only_streak": longest,
                }.items():
                    values[key].append(value)
                record = {
                    "text": text, "gold": questions[sid]["label"],
                    "steps": p["decode_iterations"], "output_tokens": n,
                    "tail_only_iterations": tail_steps, "tail_only_commits": tail_tokens,
                    "position_zero_iteration": first["iteration"] + 1,
                    "position_zero_reason": first["reason"], "switches": sample_switches,
                }
                records[sid] = record
                examples.append({**record, "sample_id": sid, "run": run_name,
                                 "question": questions[sid]["question"],
                                 "image": questions[sid]["image"]})
        assert set(records) == set(questions)
        for key in ("decode_iterations", "fallback_iterations"):
            assert counts[key] == source_row["mechanism_totals"][key]
        assert counts["switches"] == source_row["mechanism_totals"]["actual_switches"]
        assert dict(hist["finish_reason"]) == source_row["finish_reason_counts"]
        out["thresholds"][tau] = {
            "counts": counts, "regions": regions, "histograms": hist,
            "statistics": {key: describe(v) for key, v in values.items()},
            "examples": sorted(examples, key=lambda r: (-r["tail_only_iterations"],
                                                        r["sample_id"]))[:3],
            "switches": switches, "per_input": records,
        }
        print(tau, dict(counts), flush=True)
    out["common_config"] = common_config
    target = ROOT / "reports/pope_rgate_behavior.json"
    target.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    write_report(out, sweep)
    print(target)


if __name__ == "__main__":
    main()
