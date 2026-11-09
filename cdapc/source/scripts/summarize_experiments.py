"""Build a timestamped summary from saved results; no inference or API calls."""
import hashlib
import fcntl
import json
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {}


def load(relative):
    raw = (ROOT / relative).read_bytes()
    SOURCES[relative] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def macro(splits):
    return {
        key: statistics.mean(row[key] for row in splits.values())
        for key in ("accuracy", "f1", "precision", "recall", "yes_ratio", "invalid_ratio")
        if all(row.get(key) is not None for row in splits.values())
    }


def pope_result(splits, cost=None):
    result = {"macro": macro(splits), "splits": splits, "evaluation_rows": sum(
        row["n"] for row in splits.values())}
    assert result["evaluation_rows"] == 9000
    if cost:
        result.update(unique_inputs=cost["count"],
                      mean_iterations=cost["decode_iterations"] / cost["count"],
                      mean_seconds=cost["mean_generation_seconds"])
    return result


def percent(value):
    return f"{100*value:.2f}" if value is not None else "—"


def number(value):
    return f"{value:.2f}" if value is not None else "—"


def render(out):
    lines = [
        "# MMaDA / CDAPC 实验结果汇总",
        "",
        f"快照时间：{out['snapshot_at']}。后续重跑汇总脚本可刷新完成状态。",
        "",
        "## 评分与范围",
        "",
        "- 百分比指标以 % 显示。POPE 的 Accuracy/F1/Precision/Recall 均为"
        " Random、Popular、Adversarial 三个子集的等权平均；每个子集 3,000 条。",
        "- 生成成本按唯一输入计算：COCO 5,127、A-OKVQA 5,145、GQA 5,171；"
        "各数据集均映射回 9,000 条评测记录。平均秒数不含模型加载和输入预处理。",
        "- MMMU：900 题，think 协议，256-token 画布。Official 有随机 fallback，"
        "Strict 不猜答案，Judge 是 DeepSeek API 的 deepseek-flash 答案判分。",
        "- CHAIR：COCO val2014 的 500 张图，64-token 描述，不加 think。",
        "- POPE：128-token 画布。直答采用官方 Yes/No 规则；think 主分数为"
        "最终答案区域解析，未解析计错。文本审计允许可辨认的正文答案等，单独保存。",
        "- A-OKVQA/GQA 没有 Native 基线。新增 MMMU r70 为 L256/I128；"
        "旧 r90/r50 为 L256/I256，因此并非只改变 r 阈值的对照。"
        "已有 POPE-COCO 扫参为直答，不包含 COCO think 实验。",
        "",
        "## MMMU",
        "",
        "| 方法 | Official % | Strict % | Judge % | 平均步数 | 平均秒数 | 平均输出 tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in out["mmmu"].items():
        lines.append(f"| {name} | {percent(row['official'])} | {percent(row['strict'])} | "
                     f"{percent(row['judge'])} | {number(row['mean_iterations'])} | "
                     f"{number(row['mean_seconds'])} | {number(row['mean_output_tokens'])} |")
    lines += [
        "",
        "r90 的 Judge 比 Native 高 2.22 个百分点，但按题配对 bootstrap 的"
        "95% 差值区间为 [-0.78, 5.22] 个百分点，包含零。"
        "r50 的 Official 比 r90 高一题，实际是 Strict 少对 41 题与"
        "随机 fallback 多猜对 42 题相抵，不能当作更强推理的证据。",
        "",
        "## POPE 直答",
        "",
        "| 方法 | COCO Accuracy / F1 % | A-OKVQA Accuracy / F1 % | GQA Accuracy / F1 % |",
        "|---|---:|---:|---:|",
    ]
    for method in ("Native", "r90", "r80", "r70", "r60", "r50"):
        cells = []
        for dataset in ("coco", "aokvqa", "gqa"):
            row = out["pope_direct"][dataset].get(method)
            cells.append(f"{percent(row['macro']['accuracy'])} / {percent(row['macro']['f1'])}"
                         if row else "—")
        lines.append("| " + method + " | " + " | ".join(cells) + " |")
    lines += [
        "",
        "— 表示该条件没有已完成结果；不能把 COCO Native 用作其他数据集的基线。",
        "",
        "| 数据集 | 方法 | 平均步数 | 平均秒数 |",
        "|---|---|---:|---:|",
    ]
    for dataset, methods in out["pope_direct"].items():
        for method, row in methods.items():
            if "mean_seconds" in row:
                lines.append(f"| {dataset} | {method} | {number(row['mean_iterations'])} | "
                             f"{number(row['mean_seconds'])} |")
    lines += [
        "",
        "COCO r70 的平均 Accuracy 比 r90 高 0.078 个百分点（净 7/9,000 条）；"
        "各子集配对 Accuracy 差值的 95% 区间均包含零。r70 并非跨任务的通用最优参数。",
        "",
        "## CHAIR",
        "",
        "| 方法 | CHAIRs ↓ % | CHAIRi ↓ % | Recall ↑ % | 平均步数 | 平均秒数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in out["chair"].items():
        m = row["metrics"]
        lines.append(f"| {name} | {percent(m['CHAIRs'])} | {percent(m['CHAIRi'])} | "
                     f"{percent(m['Recall'])} | {number(row.get('mean_iterations'))} | "
                     f"{number(row.get('mean_seconds'))} |")
    lines += [
        "",
        "r50 的 CHAIRi 比 Native 低 2.34 个百分点，但 CHAIRs 高 0.60 个百分点。"
        "因此物体提及级与描述级幻觉需要分别报告。",
        "",
        "## POPE think",
        "",
        "| 条件 | 状态 | 严格最终答案 Acc % | 原未解析率 % | 文本审计 Acc % | LLM Judge Acc % | 平均秒数 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in out["pope_think"].items():
        status = row["status"]
        if status != "complete":
            progress = f" {row['generated']}/{row['expected_inputs']}" if "generated" in row else ""
            lines.append(f"| {name} | {status}{progress} | — | — | — | — | — |")
        else:
            lines.append(f"| {name} | complete | {percent(row['macro']['accuracy'])} | "
                         f"{percent(row['macro']['invalid_ratio'])} | "
                         f"{percent(row.get('text_audit_accuracy'))} | "
                         f"{percent(row.get('llm_judge', {}).get('macro', {}).get('accuracy'))} | "
                         f"{number(row['mean_seconds'])} |")
    lines += [
        "",
        "文本审计不替代原严格评分。原未解析记录中，A-OKVQA r90、GQA r90/r70"
        "分别有约 67%、65%、59% 存在可辨认的答案；放宽格式后仍有标签空壳和实际答错。"
        "未审计的完成组在该列保持 —，不根据其他阈值外推。",
        "",
        "LLM Judge 是独立的 pope_text_llm_judge_v1：DeepSeek 仅接收题目和完整回答，"
        "提取 Yes/No/缺答/冲突，再本地与 gold 比较。只在整组判分完成后填入主表。"
        "同一题目与完整文本的相同请求共享缓存，不包含图片或 gold。",
        "",
        "## 解码行为证据",
        "",
        "- POPE-COCO r90 没有实际非 base token 写入，但答案仍与 Native 有差异；"
        "对比统计可以通过门控与提交调度影响结果。",
        "- COCO r90 约 78.97% 的迭代仅填已知 EOS 后的位置，"
        "r50 仍有 30.59%；降低阈值显著减少等待。",
        "- COCO r70 的 69 次替换中，63 次只是大小写、5 次改变 Yes/No、1 次在 EOS 后；"
        "r50 的 896 次中有 694 次大小写变化。不能将 token 替换次数当作语义纠正次数。",
        "- MMMU 的推理文本存在空壳、未闭合、重复、算术错误等；"
        "Judge 判对也不保证推理过程正确。全部输出与逐例审计已保存。",
        "",
        "## 运行状态",
        "",
        f"队列：{out['queue']['status']}；当前：{out['queue']['active']}。"
        f"阶段统计：{out['queue']['status_counts']}。",
        "",
        "| 阶段 | 状态 |",
        "|---|---|",
    ]
    for name, row in out["queue"]["stages"].items():
        lines.append(f"| {name} | {row['status']} |")
    if out.get("mmmu_r70_run"):
        lines += ["", "新增 MMMU r70 L256/I128：" + out["mmmu_r70_run"]["status"] + "。"]
    if out.get("pope_llm_judge"):
        lines += ["", "| POPE LLM Judge 条件 | 状态 | 已判 / 唯一输入 |",
                  "|---|---|---:|"]
        for name, stage in out["pope_llm_judge"]["stages"].items():
            lines.append(f"| {name} | {stage['status']} | "
                         f"{stage.get('judged_unique', 0)} / {stage.get('expected_unique', '—')} |")
    if out.get("k_commit_pilot"):
        pilot = out["k_commit_pilot"]
        lines += ["", "## GQA think r70 提交上限小规模对照", "",
                  f"状态：{pilot['status']}。固定 30 张图、{pilot['unique_inputs']} 个唯一输入、"
                  "540 条记录，比较 K16/K4/K2；W16、r0.7、L128/I128 不变。"
                  "这部分只比较相同子集，不与上面的全量分数直接比较。"]
        if out.get("k_commit_pilot_results"):
            lines += ["", "| K | Judge Acc % | Strict Acc % | 秒/条 | Judge缺答 % |",
                      "|---|---:|---:|---:|---:|"]
            for k in ("16", "4", "2"):
                row = out["k_commit_pilot_results"]["methods"][k]
                lines.append(f"| {k} | {percent(row['judge_accuracy'])} | "
                             f"{percent(row['strict_accuracy'])} | {number(row['mean_seconds'])} | "
                             f"{percent(row['judge_missing_rate'])} |")
            lines += ["", "图像级配对差值区间与逐题输出见 `reports/k_commit_pilot/results.md` 和 JSON。"]
    if out.get("pope_k4_256_128"):
        rerun = out["pope_k4_256_128"]
        lines += ["", "## POPE 全量 think：K4，L256/I128", "",
                  f"生成队列：{rerun['status']}；当前：{rerun['active']}。",
                  "固定 r=0.7、W=16；COCO/A-OKVQA/GQA 各 9,000 条评测记录。"
                  "主分数使用最终答案严格解析与独立 LLM Judge；预算耗尽计错并保留分母。"
                  "本轮同时改变 K 与画布长度，不能仅归因于 K。",
                  "", "| 数据集 | 生成/严格评分 | 已生成/输入 | Strict Acc / F1 % | Judge Acc % | Judge状态 | 秒/条 |",
                  "|---|---|---:|---:|---:|---|---:|"]
        for row in rerun["datasets"].values():
            judged = row.get("llm_judge", {})
            scores = row.get("macro", {})
            lines.append(
                f"| {row['dataset']} | {row['status']} | {row['generated']}/{row['expected_inputs']} | "
                f"{percent(scores.get('accuracy'))} / {percent(scores.get('f1'))} | "
                f"{percent(judged.get('macro', {}).get('accuracy'))} | "
                f"{judged.get('status', 'waiting_for_generation')} | {number(row.get('mean_seconds'))} |")
    if out.get("eos_prefix_trial"):
        trial = out["eos_prefix_trial"]
        lines += ["", "## EOS 前缀候选配对试验", "",
                  f"状态：{trial['status']}；变体已生成 {trial['generated']}/{trial['expected_inputs']} 个输入。",
                  "暂停全量 POPE 后，冻结 187 个已完成 COCO 输入、385 条评测记录、32 张图。"
                  "仅改变已提交 EOS 之后的候选范围，r70/W16/K4/L256/I128 等设置相同。"
                  "该子集是完成顺序的前缀，Yes/No 为 282/103，不作为全量 POPE 成绩。"]
        if out.get("eos_prefix_trial_results"):
            results = out["eos_prefix_trial_results"]
            lines += ["", "| 方法 | 秒/条 | 步/条 | Strict Acc % | Judge Acc % | Judge F1 % |",
                      "|---|---:|---:|---:|---:|---:|"]
            for name, row in results["methods"].items():
                lines.append(f"| {name} | {number(row['mean_seconds'])} | {number(row['mean_iterations'])} | "
                             f"{percent(row['strict']['accuracy'])} | {percent(row['judge']['accuracy'])} | "
                             f"{percent(row['judge']['f1'])} |")
            lines += ["", "配对区间和逐输入变化见 `reports/eos_prefix_pilot_results.md` / JSON。"]
    if out.get("mme_run"):
        mme = out["mme_run"]
        generated = sum(item.get("generated", 0) for item in mme["shards"].values())
        lines += ["", "## MME r70 L256/I128", "",
                  f"状态：{mme['status']}；已生成 {generated}/{mme['expected_inputs']} 题。",
                  "原版 MME 全部 14 类、1,187 个图像问题对；使用官方 Yes/No 提示词。"
                  "每类按 100 × (Acc + Acc+) 评分；Perception 满分 2,000，Cognition 满分 800。",
                  "在 K4/K2 生成后领取同一 GPU 队列锁，使用独立源码快照。"]
        if out.get("mme_results"):
            result = out["mme_results"]
            lines += ["", f"Perception：{result['metrics']['perception']:.2f}；"
                      f"Cognition：{result['metrics']['cognition']:.2f}；"
                      f"总分：{result['metrics']['total']:.2f}。",
                      "逐类指标和官方脚本校验见 `reports/mme_cdapc_r70_L256_I128.md`。"]
    lines += [
        "",
        "## 复现",
        "",
        "运行 `.venv/bin/python scripts/summarize_experiments.py`。"
        "脚本从已保存报告汇总，核对 MMMU 导出 SHA256 和计数，不发起模型/API 请求。",
        "",
        "- `reports/experiment_summary.json`：精确指标、各 POPE 子集分数、"
        "当前进度和来源文件 SHA256。",
        "- `reports/experiment_summary.sha256`：汇总产物与脚本校验。",
        "- 运行索引的来源哈希表示快照时的状态；运行中的索引会继续更新。",
    ]
    return "\n".join(lines) + "\n"


def main():
    SOURCES.clear()
    index = load("reports/chair_pope_extensions_runs.json")
    export = load("reports/mmmu_answers/manifest.json")
    mechanism = load("reports/mmmu_mechanism_audit.json")
    out = {
        "snapshot_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mmmu": {}, "chair": {}, "pope_direct": {"coco": {}, "aokvqa": {}, "gqa": {}},
        "pope_think": {},
        "queue": {"status": index["status"], "active": index["active"],
                  "status_counts": dict(Counter(s["status"] for s in index["stages"].values())),
                  "stages": {key: {"status": s["status"], "report": s.get("report")}
                             for key, s in index["stages"].items()}},
    }
    for method, label in (("native", "Native"), ("cdapc_r90", "r90"), ("cdapc_r50", "r50")):
        path = f"reports/mmmu_answers/{method}.jsonl"
        raw = (ROOT / path).read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        assert sha == export["files"][f"{method}.jsonl"]["sha256"]
        SOURCES[path] = sha
        rows = [json.loads(line) for line in raw.splitlines()]
        assert len(rows) == len({row["sample_id"] for row in rows}) == 900
        result = {"samples": 900}
        for metric in ("official", "strict", "judge"):
            correct = sum(row[f"{metric}_correct"] for row in rows)
            assert correct == mechanism[method]["counts"][f"{metric}_correct"]
            result[metric] = correct / 900
            result[f"{metric}_correct"] = correct
        result.update(
            mean_iterations=statistics.mean(row["decode_iterations"] for row in rows),
            mean_output_tokens=statistics.mean(row["emitted_tokens_before_eos"] for row in rows),
            mean_seconds=mechanism[method]["distributions"]["generation_seconds"]["mean"],
        )
        assert result["mean_iterations"] == mechanism[method]["distributions"]["decode_iterations"]["mean"]
        if "judge_delta_bootstrap" in mechanism[method]:
            result["judge_delta_bootstrap_vs_native"] = mechanism[method]["judge_delta_bootstrap"]
        out["mmmu"][label] = result
    coco = load("reports/pope_yesno_mmada_cdapc_L128.json")
    out["pope_direct"]["coco"]["Native"] = pope_result(coco["native"]["splits"])
    sweep = load("reports/pope_rgate_sweep.json")
    for tau in ("0.9", "0.8", "0.7", "0.6", "0.5"):
        out["pope_direct"]["coco"][f"r{int(float(tau)*100)}"] = pope_result(
            sweep[tau]["splits"], sweep[tau]["cost"])
    chair_native = load("reports/chair_reference_verified.json")
    out["chair"]["Native"] = {"metrics": chair_native["native_reproduced"], "images": 500}
    for tau in (90, 70, 50):
        stage = index["stages"][f"chair_r{tau}"]
        if stage["status"] == "complete":
            r = load(stage["report"])
            out["chair"][f"r{tau}"] = {
                "metrics": r["overall_metrics"], "images": r["coverage"]["images"],
                "mean_iterations": r["cost"]["mean_iterations"],
                "mean_seconds": r["cost"]["mean_seconds"],
                "mean_output_tokens": r["mean_answer_tokens"],
            }
    audit = load("reports/pope_think_answer_audit.json")
    llm_path = "reports/pope_think_llm_judge/index.json"
    if (ROOT / llm_path).is_file():
        out["pope_llm_judge"] = load(llm_path)
    new_mmmu = "reports/mmmu_r70_256_128_runs.json"
    if (ROOT / new_mmmu).is_file():
        out["mmmu_r70_run"] = load(new_mmmu)
        state = out["mmmu_r70_run"]
        if state["status"] == "complete":
            result = load(state["report"])["cdapc"]
            totals, cost = result["blocks"]["all"], result["cost"]
            assert result["covered"] == result["judged"] == 900
            out["mmmu"]["r70 (256/128)"] = {
                "samples": 900, "canvas_length": 256, "iteration_limit": 128,
                **{metric: totals[metric]["accuracy"] for metric in ("official", "strict", "judge")},
                "mean_iterations": cost["decode_iterations"] / cost["count"],
                "mean_seconds": cost["mean_generation_seconds"], "mean_output_tokens": None,
                "finish_reason_counts": result["finish_reason_counts"],
            }
    for dataset in ("aokvqa", "gqa"):
        expected = len(load(f"manifests/pope_{dataset}_all_ids.json"))
        for tau in (90, 70, 50):
            for reasoning in (False, True):
                key = f"pope_{dataset}_{'reasoning_' if reasoning else ''}r{tau}"
                stage = index["stages"][key]
                row = {"status": stage["status"], "expected_inputs": expected}
                if stage["status"] == "complete":
                    r = load(stage["report"])
                    row.update(pope_result(r["cdapc"]["splits"], r["cdapc"]["cost"]))
                    row["answer_scope"] = r["answer_scope"]
                    assert row["unique_inputs"] == expected
                elif stage.get("shards"):
                    row["generated"] = sum(
                        len(list((ROOT / shard["run"] / "predictions").glob("*.json")))
                        for shard in stage["shards"].values() if shard.get("run"))
                if reasoning:
                    llm = out.get("pope_llm_judge", {}).get("stages", {}).get(key)
                    if llm:
                        row["llm_judge"] = llm
                    if key in audit["stages"]:
                        a = audit["stages"][key]
                        assert row["status"] == "complete"
                        assert abs(a["metrics"]["strict"]["accuracy"] - row["macro"]["accuracy"]) < 1e-10
                        row["text_audit_accuracy"] = a["metrics"]["semantic_text"]["accuracy"]
                        row["text_audit_variants"] = a["metrics"]
                    out["pope_think"][key] = row
                elif stage["status"] == "complete":
                    out["pope_direct"][dataset][f"r{tau}"] = row
    if (ROOT / "reports/k_commit_pilot/index.json").is_file():
        out["k_commit_pilot"] = load("reports/k_commit_pilot/index.json")
        if out["k_commit_pilot"]["status"] == "complete":
            pilot = load("reports/k_commit_pilot/results.json")
            out["k_commit_pilot_results"] = {
                k: pilot[k] for k in ("protocol", "methods", "paired_judge_accuracy")}
    if (ROOT / "reports/pope_k4_256_128_runs.json").is_file():
        rerun = load("reports/pope_k4_256_128_runs.json")
        judge = (load(rerun["judge_index"]) if (ROOT / rerun["judge_index"]).is_file() else {"stages": {}})
        out["pope_k4_256_128"] = {"status": rerun["status"], "active": rerun["active"], "datasets": {}}
        for name, stage in rerun["stages"].items():
            row = {"dataset": stage["dataset"], "status": stage["status"],
                   "expected_inputs": stage["expected_inputs"],
                   "generated": sum(len(list((ROOT / shard["run"] / "predictions").glob("*.json")))
                                    for shard in stage["shards"].values() if shard.get("run")),
                   "llm_judge": judge["stages"].get(name, {})}
            if stage["status"] == "complete":
                result = load(stage["report"])["cdapc"]
                row.update(pope_result(result["splits"], result["cost"]),
                           generation_status_counts=result["generation_status_counts"],
                           finish_reason_counts=result["finish_reason_counts"])
            out["pope_k4_256_128"]["datasets"][name] = row
    trial_root = ROOT.parent / "cdapc-eos-prefix"
    if (trial_root / "reports/index.json").is_file():
        trial = load(str(trial_root / "reports/index.json"))
        out["eos_prefix_trial"] = {
            "status": trial["status"], "expected_inputs": trial["expected_inputs"],
            "generated": sum(len(list((trial_root / item["run"] / "predictions").glob("*.json")))
                             for item in trial["shards"].values() if item.get("run"))}
        if (ROOT / "reports/eos_prefix_pilot_results.json").is_file():
            results = load("reports/eos_prefix_pilot_results.json")
            out["eos_prefix_trial_results"] = {k: results[k] for k in ("methods", "paired", "judge_api_errors")}
    if (ROOT / "reports/mme_r70_256_128_runs.json").is_file():
        out["mme_run"] = load("reports/mme_r70_256_128_runs.json")
        if out["mme_run"]["status"] == "complete":
            mme = load(out["mme_run"]["report"])
            out["mme_results"] = {
                k: mme[k] for k in ("metrics", "cost", "generation_status_counts",
                                   "finish_reason_counts", "official_scorer_verified")}
    out["source_sha256"] = dict(SOURCES)
    target = ROOT / "reports/experiment_summary.json"
    target.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    markdown = ROOT / "reports/experiment_summary.md"
    markdown.write_text(render(out))
    (ROOT / "reports/experiment_summary.sha256").write_text("".join(
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT)}\n"
        for p in (Path(__file__), target, markdown)))
    print(out["snapshot_at"], out["queue"]["status_counts"], out["queue"]["active"])
    print(markdown)


if __name__ == "__main__":
    with (ROOT / "logs/experiment_summary.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        main()
