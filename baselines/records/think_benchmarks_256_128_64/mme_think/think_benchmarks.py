"""HallusionBench (thinking prompt; VCD, VIGOR) and MME (thinking prompt and
original prompt; native, VCD, VIGOR) at 256/128/64 with safe EOS stopping.

Stages run sequentially on two GPU shards: hallusion_think -> mme_think ->
mme_original. Each stage: preflight (3 inputs, GPU 0) -> two-shard generation
via the unchanged `common.run` -> audit -> scoring. Resume with the same
command; manifests are write-once and `common.run` rejects changed inputs.

    .venv/bin/python experiments/think_benchmarks.py prepare
    setsid nohup .venv/bin/python -u experiments/think_benchmarks.py run --gpus 0 1 \
        >> results/think_benchmarks_256_128_64/pipeline.log 2>&1 &
"""
import argparse
import ast
from collections import Counter, defaultdict
from contextlib import ExitStack
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import sqlite3
import statistics
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.mmmu_protocol import INSTRUCTION  # noqa: E402
from common.run import input_key  # noqa: E402
from common.runtime import save_json, sha256, source_hashes  # noqa: E402

CONFIG = ROOT / "configs/think_256_128_64.yaml"
RUN = ROOT / "results/think_benchmarks_256_128_64"
HB_SOURCE = ROOT / "data/hallusion_full_original/samples.jsonl"
HB_DATA = ROOT / "data/hallusion_think"
MME = ROOT / "data/mme"
MME_REVISION = "d6c9023f017b564f7b3ccccf5348166bce8fdbcd"
MME_FILES = {
    "test-00000-of-00004-a25dbe3b44c4fda6.parquet": "1563fd59c8049dbbabe211a049c38b2e20a6d58da94a75c90cfc937d1f6c52e3",
    "test-00001-of-00004-7d22c7f1aba6fca4.parquet": "3623728c0c98807cc7eef9cbd10dfa18c21654321c4b02b7b19bc8e6fc86d830",
    "test-00002-of-00004-594798fd3f5b029c.parquet": "c169d85bdf84e21b005170698217e96380bef172e8bd7432fc75ddce51dcaabe",
    "test-00003-of-00004-53ae1794f93b1e35.parquet": "aec84886b2bb6dd7bffc94eb3bbca9319e0671f3ee989a08e0758bc2de6c67f6",
}
MME_SCORER_COMMIT = "dd2950902889cd614d4edf606827240166d29381"
MME_SCORER_BLOB = "e02681d44e5a5e7f81023e50d3ce4cfce34c07d0"
SUFFIX = "Please answer yes or no."
STAGES = {
    "hallusion_think": {"benchmark": "hallusion", "methods": ["vcd", "vigor"],
                        "data": HB_DATA, "prompt_id": "hallusion_think_v1", "think": True},
    "mme_think": {"benchmark": "mme", "methods": ["native", "vcd", "vigor"],
                  "data": MME / "think", "prompt_id": "mme_think_v1", "think": True},
    "mme_original": {"benchmark": "mme", "methods": ["native", "vcd", "vigor"],
                     "data": MME / "original", "prompt_id": "mme_yesno_v1", "think": False},
}


def write_locked(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"refusing changed artifact: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def jsonl(rows):
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def think_question(question):
    return INSTRUCTION + "\n" + question


def write_split(directory, rows, shard_of, preflight, extra):
    shards = [[r for r in rows if shard_of(r) == s] for s in (0, 1)]
    write_locked(directory / "samples.jsonl", jsonl(rows))
    for s, selected in enumerate(shards):
        write_locked(directory / f"shard_{s}.jsonl", jsonl(selected))
    write_locked(directory / "preflight.jsonl", jsonl(preflight))
    manifest = {**extra, "rows": len(rows), "samples_sha256": sha256(directory / "samples.jsonl"),
                "shards": [{"rows": len(x), "sha256": sha256(directory / f"shard_{s}.jsonl")}
                           for s, x in enumerate(shards)],
                "preflight_sha256": sha256(directory / "preflight.jsonl"),
                "preparer_sha256": sha256(__file__)}
    write_locked(directory / "manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
    return manifest


def prepare_hallusion():
    source = read_rows(HB_SOURCE)
    assert len(source) == 1129
    rows = [{**r, "question": think_question(r["original_question"]),
             "prompt_id": STAGES["hallusion_think"]["prompt_id"]} for r in source]
    position = {r["id"]: i for i, r in enumerate(rows)}
    image_rows = [r for r in rows if r["input_mode"] == "image"]
    preflight = [max(image_rows, key=lambda r: len(r["original_question"])),
                 next(r for r in image_rows if r["answer"] == "No"),
                 next(r for r in rows if r["input_mode"] == "text_only")]
    return write_split(HB_DATA, rows, lambda r: position[r["id"]] % 2, preflight, {
        "source": str(HB_SOURCE.relative_to(ROOT)), "source_sha256": sha256(HB_SOURCE),
        "prompt": "MMaDA thinking instruction + '\\n' + unchanged TSV question; text-only rows "
                  "use the pure chat template and native decoding for every method",
        "shard_rule": "same as hallusion_full_original (row index % 2)",
        "category_counts": dict(Counter(r["category"] for r in rows)),
        "input_mode_counts": dict(Counter(r["input_mode"] for r in rows))})


def official_mme_questions():
    archive = MME / "upstream/eval_tool.zip"
    raw = archive.read_bytes()
    if hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() != MME_SCORER_BLOB:
        raise ValueError("MME eval_tool.zip differs from the pinned git blob")
    records = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for name in ("calculation.py", "readme.txt"):
            write_locked(MME / "upstream" / name, z.read("eval_tool/" + name))
        for name in sorted(z.namelist()):
            if not (name.startswith("eval_tool/Your_Results/") and name.endswith(".txt")):
                continue
            category = Path(name).stem
            lines = z.read(name).decode("utf-8").splitlines()
            assert len(lines) % 2 == 0
            for index, line in enumerate(lines):
                image, question, answer, *_ = line.split("\t")
                key = (category, Path(image).name, question.strip())
                assert key not in records
                records[key] = {"answer": answer.lower(), "pair_index": index % 2}
    return records


def prepare_mme():
    import pyarrow.parquet as pq
    from PIL import Image

    official = official_mme_questions()
    rows, seen = [], set()
    (MME / "images").mkdir(parents=True, exist_ok=True)
    for name, expected in MME_FILES.items():
        path = MME / "downloads" / name
        if sha256(path) != expected:
            raise ValueError(f"MME parquet SHA256 mismatch: {name}")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
            for record in batch.to_pylist():
                category = record["category"]
                image_name = Path(record["question_id"]).name
                question = record["question"].strip()
                key = (category, image_name, question)
                assert key in official and key not in seen, key
                assert record["answer"].lower() == official[key]["answer"], key
                assert question.endswith(SUFFIX), key
                seen.add(key)
                payload = record["image"]["bytes"]
                digest = hashlib.sha256(payload).hexdigest()
                with Image.open(io.BytesIO(payload)) as image:
                    extension = {"PNG": "png", "JPEG": "jpg"}.get(image.format, image.format.lower())
                    image.verify()
                target = MME / "images" / f"{digest}.{extension}"
                write_locked(target, payload)
                group = f"{category}/{image_name}"
                rows.append({
                    "id": f"mme:{group}:{official[key]['pair_index']}", "group_id": group,
                    "category": category, "image_name": image_name, "original_question": question,
                    "answer": official[key]["answer"], "image": str(target), "image_sha256": digest})
    assert seen == set(official) and len(rows) == 2374
    rows.sort(key=lambda r: r["id"])
    groups = sorted({r["group_id"] for r in rows},
                    key=lambda g: hashlib.sha256(("mme_shard_v1:" + g).encode()).hexdigest())
    assert len(groups) == 1187
    shard = {g: i % 2 for i, g in enumerate(groups)}
    extra = {"repo": "lmms-lab/MME", "revision": MME_REVISION, "files": MME_FILES,
             "scorer_commit": MME_SCORER_COMMIT, "scorer_archive_git_blob": MME_SCORER_BLOB,
             "scorer_sha256": sha256(MME / "upstream/calculation.py"),
             "official_questions_and_labels_verified": True, "image_pairs": len(groups),
             "categories": dict(Counter(r["category"] for r in rows)),
             "shard_rule": "image pairs sorted by sha256('mme_shard_v1:'+group), alternating"}
    out = {}
    for variant in ("original", "think"):
        stage = STAGES[f"mme_{variant}"]
        derived = []
        for r in rows:
            q = r["original_question"]
            if variant == "think":
                q = think_question(q[:-len(SUFFIX)].rstrip())
            derived.append({**r, "question": q, "prompt_id": stage["prompt_id"]})
        by_cat = {}
        for r in derived:
            by_cat.setdefault(r["category"], r)
        preflight = [by_cat["existence"], by_cat["OCR"], by_cat["code_reasoning"]]
        prompt = ("official question verbatim (includes 'Please answer yes or no.')" if variant == "original"
                  else "MMaDA thinking instruction + '\\n' + official question without "
                       "'Please answer yes or no.'")
        out[variant] = write_split(stage["data"], derived, lambda r: shard[r["group_id"]], preflight,
                                   {**extra, "prompt": prompt, "prompt_id": stage["prompt_id"]})
    return out


def command(stage, samples, output):
    return [sys.executable, "-u", "-m", "common.run", "--config", str(CONFIG),
            "--samples", str(samples), "--output", str(output),
            "--methods", *STAGES[stage]["methods"], "--benchmark", "custom",
            "--warmups", "3", "--early-eos"]


def gpu_inventory(gpus):
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.free",
                                    "--format=csv,noheader,nounits"], text=True).splitlines()
    free = {r.split(",")[0].strip(): int(r.split(",")[2]) for r in rows}
    for gpu in gpus:
        if free.get(gpu, 0) < 22 * 1024:
            raise ValueError(f"GPU {gpu} has less than 22 GiB free")
    return rows


def parallel(jobs, lock_fd):
    with ExitStack() as stack:
        children = []
        try:
            for argv, gpu, log_path in jobs:
                log = stack.enter_context(log_path.open("a"))
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "OMP_NUM_THREADS": "4",
                       "TOKENIZERS_PARALLELISM": "false", "PYTHONHASHSEED": "42"}
                children.append(subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log,
                                                 stderr=subprocess.STDOUT, start_new_session=True,
                                                 pass_fds=(lock_fd,)))
            while any(c.poll() is None for c in children):
                if any(c.poll() not in (None, 0) for c in children):
                    raise RuntimeError("worker failed; see shard logs")
                time.sleep(5)
            if any(c.returncode for c in children):
                raise RuntimeError("worker failed; committed rows remain resumable")
        finally:
            for c in children:
                if c.poll() is None:
                    os.killpg(c.pid, signal.SIGTERM)
                    c.wait()


def load_records(run_dir, samples, methods):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if (manifest.get("early_eos") != "safe_eos_v1" or set(manifest["methods"]) != set(methods)
            or manifest["generation"]["max_new_tokens"] != 256 or manifest["generation"]["steps"] != 128
            or manifest["generation"]["block_length"] != 64 or manifest["limit"] != 0):
        raise ValueError(f"unexpected run manifest: {run_dir}")
    with sqlite3.connect(f"file:{run_dir}/predictions.sqlite?mode=ro", uri=True) as db:
        if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("prediction database is corrupt")
        records = {(m, k): json.loads(p) for m, k, p in db.execute(
            "SELECT method,input_key,payload FROM predictions")}
    expected = {(m, input_key(s)) for m in methods for s in samples}
    if set(records) != expected:
        raise ValueError(f"incomplete or unexpected coverage: {run_dir}")
    return manifest, records


def audit(samples, records, methods):
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from training.prompting_utils import UniversalPrompting
    from verify_eos import audit_record

    config = OmegaConf.load(CONFIG)
    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint, local_files_only=True)
    special = UniversalPrompting(tokenizer, use_reserved_token=True).sptids_dict
    mask, eos = tokenizer.convert_tokens_to_ids("<|mdm_mask|>"), tokenizer.eos_token_id
    mmu, soi, eoi = (int(special[t]) for t in ("<|mmu|>", "<|soi|>", "<|eoi|>"))
    for sample in samples:
        chat = tokenizer.apply_chat_template([{"role": "user", "content": sample["question"]}],
                                             tokenize=True, add_generation_prompt=True)
        inputs = []
        for method in methods:
            record = records[method, input_key(sample)]
            ids = record["input"]["prompt_ids"]
            if sample.get("input_mode") == "text_only":
                if (ids != chat or record["input"]["visual_positions"]
                        or record.get("effective_method") != "native"):
                    raise ValueError("text-only prompt or decoder policy mismatch")
            elif (ids[:2] != [mmu, soi] or ids[1026] != eoi or ids[1027:] != chat
                  or record["input"]["visual_positions"] != list(range(2, 1026))
                  or record["input"]["image_sha256"] != sample["image_sha256"]):
                raise ValueError(f"visual prompt differs from prepared sample: {sample['id']}")
            steps = record["executed_steps"]
            if (record["planned_steps"] != 128 or record["planned_generated_positions"] != 256
                    or steps != len(record["trace"]) or not 0 < steps <= 128
                    or record["remaining_masks"]):
                raise ValueError("invalid generation schedule or output")
            first_closed = audit_record(record, mask, eos)
            if first_closed is not None and first_closed != steps:
                raise ValueError("generation did not stop at its first committed EOS prefix")
            inputs.append(ids)
        if any(x != inputs[0] for x in inputs):
            raise ValueError("normal input differs across methods")


def answer_text(text, think):
    """Primary: text after the last </think> (full text if unclosed); original: full text."""
    if think and "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip(), True
    return text, (False if think else None)


def mme_official():
    source = MME / "upstream/calculation.py"
    tree = ast.parse(source.read_text())
    categories = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "eval_type_dict" for t in n.targets))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "calculate_metrics")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "parse_pred_ans")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return categories, lambda text: namespace["parse_pred_ans"](None, text.lower())


def mme_metrics(rows, categories):
    result = {"per_category": {}}
    for category in [c for group in categories.values() for c in group]:
        items = [r for r in rows if r["category"] == category]
        pairs = defaultdict(list)
        for r in items:
            pairs[r["group_id"]].append(r)
        assert all(len(p) == 2 for p in pairs.values())
        acc = sum(r["correct"] for r in items) / len(items)
        plus = sum(all(r["correct"] for r in p) for p in pairs.values()) / len(pairs)
        result["per_category"][category] = {
            "questions": len(items), "image_pairs": len(pairs), "acc": acc, "acc_plus": plus,
            "other": sum(r["parsed"] == "other" for r in items), "score": 100 * (acc + plus)}
    for section, names in categories.items():
        result[section.lower()] = sum(result["per_category"][n]["score"] for n in names)
    result["total"] = result["perception"] + result["cognition"]
    result["accuracy"] = sum(r["correct"] for r in rows) / len(rows)
    result["yes_ratio"] = sum(r["parsed"] == "yes" for r in rows) / len(rows)
    return result


def paired_bootstrap(left, right, unit, repeats=2000, seed=42):
    import numpy as np
    groups = defaultdict(list)
    for a, b in zip(left, right):
        assert a["id"] == b["id"]
        groups[a[unit]].append(b["correct"] - a["correct"])
    values = list(groups.values())
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(repeats):
        picked = [d for i in rng.integers(0, len(values), len(values)) for d in values[i]]
        deltas.append(sum(picked) / len(picked))
    total = [d for v in values for d in v]
    return {"unit": unit, "groups": len(values), "repeats": repeats, "seed": seed,
            "delta_accuracy": sum(total) / len(total),
            "delta_accuracy_ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]}


def diagnostics(rows, think):
    out = {"rows": len(rows),
           "mean_answer_tokens": statistics.mean(r["answer_token_length"] for r in rows),
           "mean_executed_steps": statistics.mean(r["executed_steps"] for r in rows),
           "mean_end_to_end_seconds": statistics.mean(r["end_to_end_seconds"] for r in rows),
           "eos_found_rate": sum(bool(r["eos_found"]) for r in rows) / len(rows),
           "rows_with_generated_image_code_tokens": sum(bool(r["image_token_count"]) for r in rows),
           "answer_not_starting_with_yes_or_no": sum(
               not r["scored_text"].strip().lower().startswith(("yes", "no")) for r in rows)}
    if think:
        out["think_close_rate"] = sum("</think>" in r["text"] for r in rows) / len(rows)
        out["think_close_missing"] = sum(r["think_closed"] is False for r in rows)
    return out


def score(stage):
    spec = STAGES[stage]
    stage_dir = RUN / stage
    samples = read_rows(spec["data"] / "samples.jsonl")
    records = {}
    for s in (0, 1):
        shard_samples = read_rows(spec["data"] / f"shard_{s}.jsonl")
        _, part = load_records(stage_dir / f"shard{s}", shard_samples, spec["methods"])
        records.update(part)
    audit(samples, records, spec["methods"])
    report = {"stage": stage, "benchmark": spec["benchmark"], "prompt_id": spec["prompt_id"],
              "methods": spec["methods"], "rows": len(samples),
              "generation": {"max_new_tokens": 256, "steps": 128, "block_length": 64},
              "early_eos": "safe_eos_v1", "scoring_text": (
                  "text after the last </think> (full text if unclosed)" if spec["think"]
                  else "complete decoded response"), "results": {}}
    method_rows = {}
    for method in spec["methods"]:
        rows = []
        for sample in samples:
            r = records[method, input_key(sample)]
            answer, closed = answer_text(r["text"], spec["think"])
            rows.append({**{k: v for k, v in sample.items() if k != "question"},
                         "text": r["text"], "scored_text": answer, "think_closed": closed,
                         **{k: r[k] for k in ("answer_token_length", "executed_steps",
                                              "end_to_end_seconds", "eos_found", "early_stopped",
                                              "image_token_count")}})
        method_rows[method] = rows
    if spec["benchmark"] == "hallusion":
        from common.hallusion import rate
        for method, rows in method_rows.items():
            scored, primary = rate([{**r, "prediction": r["scored_text"]} for r in rows])
            _, full = rate([{**r, "prediction": r["text"]} for r in rows])
            primary["by_input_mode"] = {m: rate([{**r, "prediction": r["scored_text"]} for r in rows
                                                 if r["input_mode"] == m])[1]
                                        for m in ("image", "text_only")}
            report["results"][method] = {"primary": primary, "full_text_reference": {
                k: full[k] for k in ("accuracy_percent", "figure_accuracy_percent",
                                     "question_accuracy_percent", "unknown_answers")},
                "diagnostics": diagnostics(rows, True)}
            save_json(stage_dir / f"{method}_scores.json", scored)
    else:
        categories, parse = mme_official()
        for method, rows in method_rows.items():
            for r in rows:
                r["parsed"] = parse(r["scored_text"])
                r["correct"] = r["parsed"] == r["answer"]
        for method, rows in method_rows.items():
            block = {"primary": mme_metrics(rows, categories), "diagnostics": diagnostics(rows, spec["think"])}
            if spec["think"]:
                full = [{**r, "parsed": parse(r["text"])} for r in rows]
                for r in full:
                    r["correct"] = r["parsed"] == r["answer"]
                block["full_text_reference"] = {k: v for k, v in mme_metrics(full, categories).items()
                                                if k != "per_category"}
            if method != "native" and "native" in method_rows:
                block["paired_vs_native"] = paired_bootstrap(method_rows["native"], rows, "group_id")
            report["results"][method] = block
            save_json(stage_dir / f"{method}_scores.json", rows)
        export_official_text(stage_dir, method_rows)
        import re
        for method in method_rows:
            checked = subprocess.run(
                [sys.executable, str(MME / "upstream/calculation.py"),
                 "--results_dir", str(stage_dir / "official_text" / method)],
                capture_output=True, text=True, check=True,
                env={**os.environ, "PYTHONPATH": str(MME / "vendor")})
            (stage_dir / f"official_scorer_{method}.log").write_text(checked.stdout + checked.stderr)
            official = dict(re.findall(r"^[ \t]+(\w+)[ \t]+score:\s+([\d.eE+-]+)", checked.stdout, re.M))
            mine = report["results"][method]["primary"]["per_category"]
            if set(official) != set(mine) or any(
                    abs(float(v) - mine[k]["score"]) > 1e-8 for k, v in official.items()):
                raise ValueError(f"unmodified MME calculation.py disagrees for {method}")
            report["results"][method]["official_scorer_verified"] = True
    for method, rows in method_rows.items():
        save_json(stage_dir / f"{method}_answers.json", rows)
    save_json(stage_dir / "metrics.json", report)
    write_results(stage_dir, report)
    return report


def export_official_text(stage_dir, method_rows):
    """MME eval_tool format, so the unmodified calculation.py can re-check scores."""
    for method, rows in method_rows.items():
        folder = stage_dir / "official_text" / method
        folder.mkdir(parents=True, exist_ok=True)
        for category in sorted({r["category"] for r in rows}):
            selected = sorted((r for r in rows if r["category"] == category),
                              key=lambda r: (r["group_id"], r["id"]))
            lines = [f"{r['image_name']}\t{r['original_question']}\t{r['answer']}\t"
                     f"{' '.join(r['scored_text'].split())}\n" for r in selected]
            (folder / f"{category}.txt").write_text("".join(lines))


def write_results(stage_dir, report):
    lines = [f"# {report['stage']} (256/128/64, safe EOS)", "",
             f"Prompt `{report['prompt_id']}`; scoring on {report['scoring_text']}.", ""]
    if report["benchmark"] == "hallusion":
        lines += ["| Method | aAcc % | fAcc % | qAcc % | Unknown | think closed | full-text aAcc % |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for m, b in report["results"].items():
            p, d = b["primary"], b["diagnostics"]
            lines.append(f"| {m} | {p['accuracy_percent']:.2f} | {p['figure_accuracy_percent']:.2f} | "
                         f"{p['question_accuracy_percent']:.2f} | {p['unknown_answers']} | "
                         f"{100 * d['think_close_rate']:.1f}% | "
                         f"{b['full_text_reference']['accuracy_percent']:.2f} |")
    else:
        lines += ["| Method | Perception /2000 | Cognition /800 | Total /2800 | Acc % | Yes % | other |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for m, b in report["results"].items():
            p = b["primary"]
            other = sum(c["other"] for c in p["per_category"].values())
            lines.append(f"| {m} | {p['perception']:.2f} | {p['cognition']:.2f} | {p['total']:.2f} | "
                         f"{100 * p['accuracy']:.2f} | {100 * p['yes_ratio']:.1f} | {other} |")
    (stage_dir / "RESULTS.md").write_text("\n".join(lines) + "\n")


def run_stage(stage, gpus, lock_fd, status):
    spec = STAGES[stage]
    stage_dir = RUN / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    if (stage_dir / "metrics.json").exists():
        return
    preflight = stage_dir / "preflight"
    if not (stage_dir / "preflight_validation.json").exists():
        status["stages"][stage] = {"state": "preflight", "started": time.time()}
        save_json(RUN / "queue_status.json", status)
        parallel([(command(stage, spec["data"] / "preflight.jsonl", preflight), gpus[0],
                   stage_dir / "preflight.log")], lock_fd)
        samples = read_rows(spec["data"] / "preflight.jsonl")
        _, records = load_records(preflight, samples, spec["methods"])
        audit(samples, records, spec["methods"])
        save_json(stage_dir / "preflight_validation.json", {"passed": True, "checks": [
            {"method": m, "id": s["id"], "text": records[m, input_key(s)]["text"],
             "executed_steps": records[m, input_key(s)]["executed_steps"],
             "end_to_end_seconds": records[m, input_key(s)]["end_to_end_seconds"]}
            for s in samples for m in spec["methods"]]})
    status["stages"][stage] = {"state": "generating", "started": time.time()}
    save_json(RUN / "queue_status.json", status)
    gpu_inventory(gpus)
    parallel([(command(stage, spec["data"] / f"shard_{s}.jsonl", stage_dir / f"shard{s}"), gpu,
               stage_dir / f"shard{s}.log") for s, gpu in enumerate(gpus)], lock_fd)
    status["stages"][stage]["state"] = "scoring"
    save_json(RUN / "queue_status.json", status)
    score(stage)
    status["stages"][stage].update(state="generation_and_scoring_complete", finished=time.time())
    save_json(RUN / "queue_status.json", status)


def run(args):
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = {"stages": {k: {**v, "data": str(v["data"].relative_to(ROOT)),
                               "data_manifest_sha256": sha256(v["data"] / "manifest.json")}
                           for k, v in STAGES.items()},
                "config_sha256": sha256(CONFIG), "launcher_sha256": sha256(__file__),
                "sources": source_hashes(), "gpus": args.gpus, "early_eos": "safe_eos_v1"}
        write_locked(RUN / "plan.json", (json.dumps(plan, indent=2) + "\n").encode())
        status = {"state": "running", "pid": os.getpid(), "started": time.time(), "stages": {},
                  "complete": False}
        if (RUN / "queue_status.json").exists():
            status["stages"] = json.loads((RUN / "queue_status.json").read_text()).get("stages", {})
        save_json(RUN / "queue_status.json", status)
        try:
            for stage in STAGES:
                if source_hashes() != plan["sources"] or sha256(__file__) != plan["launcher_sha256"]:
                    raise ValueError("launcher or inference sources changed")
                status["current"] = stage
                run_stage(stage, args.gpus, lock.fileno(), status)
            status.update(state="complete", complete=True)
        except BaseException as exc:
            status.update(state="failed", error=repr(exc))
            raise
        finally:
            status["updated"] = time.time()
            save_json(RUN / "queue_status.json", status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    r = sub.add_parser("run")
    r.add_argument("--gpus", nargs=2, default=["0", "1"])
    s = sub.add_parser("score")
    s.add_argument("stage", choices=list(STAGES))
    args = parser.parse_args()
    if args.cmd == "prepare":
        print(json.dumps({"hallusion": prepare_hallusion()["rows"],
                          **{k: v["rows"] for k, v in prepare_mme().items()}}))
    elif args.cmd == "score":
        print(json.dumps(score(args.stage)["results"], default=str)[:3000])
    else:
        def terminate(signum, frame):
            raise KeyboardInterrupt(f"received signal {signum}")
        signal.signal(signal.SIGTERM, terminate)
        run(args)


if __name__ == "__main__":
    main()
