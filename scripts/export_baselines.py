"""Export completed MMaDA-Baselines VCD/VIGOR experiments into baselines/records/.

Run from the records repo checkout. Reads the source project read-only and
writes verbatim prediction payloads (gzip JSONL, <=45 MB each), answers,
manifests, configs, samples, scoring summaries, per-method SHA256SUMS and a
baselines/records_index.json. Paused/incomplete/running experiments are not
touched. Mirrors scripts/export_from_server.py conventions.
"""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys

REPO = Path(__file__).resolve().parents[1]
SRC = Path("/root/autodl-tmp/MMaDA-Baselines")
OUT = REPO / "baselines"
GZ_LIMIT = 45 * 1024 * 1024          # hard cap per predictions-*.jsonl.gz
CHUNK_UNCOMPRESSED = 200 * 1024 * 1024  # initial budget; shrinks if gz exceeds cap
METHODS = ("native", "vcd", "vigor")

INSTRUCTION = (
    "You should first think about the reasoning process in the mind and then "
    "provide the user with the answer. The reasoning process is enclosed within "
    "<think> </think> tags, i.e. <think> reasoning process here </think> answer here"
)
SUFFIX = " Please answer yes or no."
DIAG_SAMPLES = SRC / "results/pope_prompt_diagnostic/samples.jsonl"


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    return sha_bytes(Path(path).read_bytes())


def input_key(sample):
    # Identical to common.run.input_key / scripts/verify.py.
    return hashlib.sha256(json.dumps(
        [sample["image_sha256"], sample.get("messages", sample.get("question"))],
        ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1, allow_nan=False) + "\n")


def jsonl_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_db(path):
    """Return {method: {input_key: payload_str}}; method "" when the table has none."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        if con.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError(f"corrupt predictions database: {path}")
        cols = [r[1] for r in con.execute("PRAGMA table_info(predictions)")]
        rows = {}
        if "method" in cols:
            for method, key, payload in con.execute(
                    "SELECT method,input_key,payload FROM predictions ORDER BY method,input_key"):
                rows.setdefault(method, {})[key] = payload
        else:
            for key, payload in con.execute(
                    "SELECT input_key,payload FROM predictions ORDER BY input_key"):
                rows.setdefault("", {})[key] = payload
        return rows
    finally:
        con.close()


def merge_method_rows(db_specs):
    """db_specs: {method: [sqlite paths]}. Returns {method: {key: payload}}."""
    merged = {}
    for method, paths in db_specs.items():
        combined = {}
        sources = []
        for rel in paths:
            path = SRC / rel
            db = load_db(path)
            if method not in db:
                raise ValueError(f"{path} has no method {method}")
            for key, payload in db[method].items():
                if key in combined:
                    # Shards can regenerate the same input; wall-clock/resource
                    # fields legitimately differ. Require identical content.
                    volatile = ("decode_seconds", "end_to_end_seconds",
                                "preprocessing_seconds", "peak_allocated_bytes",
                                "peak_reserved_bytes")
                    scrub = lambda raw: {k: v for k, v in json.loads(raw).items()
                                         if k not in volatile}
                    if scrub(combined[key]) != scrub(payload):
                        raise ValueError(f"{method}: conflicting duplicate input key {key} in {rel}")
                    continue
                combined[key] = payload
            sources.append({"path": rel, "sha256": sha_file(path)})
        merged[method] = {"rows": combined, "sources": sources}
    return merged


def transform_payload(raw):
    """Payload format B: replace input.prompt_ids with a sha256 + length and
    compact contiguous visual_positions to [start, end).

    Asserts image-code integrity before transforming: every position in
    visual_positions must index a code >= image_code_offset, and image inputs
    must carry exactly 1024 visual codes. All other fields are kept verbatim
    (the payload is re-serialized with the same json.dumps options the
    runtime used, so untouched fields keep their values and ordering).
    """
    payload = json.loads(raw)
    inp = payload.get("input")
    if not isinstance(inp, dict) or "prompt_ids" not in inp:
        return raw
    prompt_ids = inp["prompt_ids"]
    visual = inp.get("visual_positions") or []
    offset = inp.get("image_code_offset")
    if visual:
        if len(visual) != 1024:
            raise ValueError(f"visual_positions length {len(visual)} != 1024")
        if offset is None or not all(prompt_ids[p] >= offset for p in visual):
            raise ValueError("visual positions do not all index image codes")
    new_input = {}
    for key, value in inp.items():
        if key == "prompt_ids":
            continue
        if key == "image_sha256":
            new_input[key] = value
            new_input["prompt_ids_sha256"] = sha_bytes(
                json.dumps(prompt_ids, separators=(",", ":")).encode())
            new_input["prompt_ids_length"] = len(prompt_ids)
            continue
        if (key == "visual_positions" and visual
                and visual == list(range(visual[0], visual[0] + len(visual)))):
            new_input[key] = [visual[0], visual[0] + len(visual)]
            continue
        new_input[key] = value
    payload["input"] = new_input
    return json.dumps(payload, ensure_ascii=False, allow_nan=False)


def export_predictions(method_dir, rows):
    """Write predictions-NNN.jsonl.gz chunks (<=GZ_LIMIT compressed) + index entries."""
    method_dir.mkdir(parents=True, exist_ok=True)
    items = sorted((key, transform_payload(raw)) for key, raw in rows.items())
    entries, shard, pos, budget = [], 0, 0, CHUNK_UNCOMPRESSED
    while pos < len(items):
        end, nbytes = pos, 0
        while end < len(items) and nbytes < budget:
            nbytes += len(items[end][0]) + len(items[end][1]) + 64
            end += 1
        chunk = items[pos:end]
        name = f"predictions-{shard:03d}.jsonl.gz"
        tmp = method_dir / (name + ".tmp")
        with gzip.GzipFile(tmp, "wb", mtime=0) as stream:
            for key, raw in chunk:
                stream.write(b'{"input_key": ' + json.dumps(key).encode()
                             + b', "payload": ' + raw.encode() + b"}\n")
        size = tmp.stat().st_size
        if size > GZ_LIMIT:
            tmp.unlink()
            budget = max(4 * 1024 * 1024, int(budget * GZ_LIMIT / size * 0.85))
            continue  # retry the same span with a smaller uncompressed budget
        tmp.replace(method_dir / name)
        for key, raw in chunk:
            entries.append({"input_key": key, "shard": name,
                            "payload_sha256": sha_bytes(raw.encode())})
        shard += 1
        pos = end
    return entries, shard


def index_entries_from_disk(method_dir):
    """Rebuild index entries by streaming existing predictions-*.jsonl.gz."""
    entries = []
    for shard in sorted(method_dir.glob("predictions-*.jsonl.gz")):
        with gzip.open(shard, "rb") as stream:
            for line in stream:
                prefix, raw = line.split(b', "payload": ', 1)
                key = json.loads(prefix + b"}")["input_key"]
                entries.append({"input_key": key, "shard": shard.name,
                                "payload_sha256": sha_bytes(raw[:-2])})
    return entries


def method_sums(method_dir):
    files = sorted(p for p in method_dir.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    lines = [f"{sha_file(p)}  {p.relative_to(method_dir)}\n" for p in files]
    (method_dir / "SHA256SUMS").write_text("".join(lines))


def dataset_dir(exp, ds):
    return OUT / "records" / exp / ds


def prepare_dataset_dir(exp, ds, spec):
    """Copy samples/exclude/manifests/status/configs/metrics sources; return info."""
    target = dataset_dir(exp, ds)
    target.mkdir(parents=True, exist_ok=True)
    samples_manifest = {"files": [], "source_files": spec.get("samples", [])}
    for rel in spec.get("samples", []):
        src = SRC / rel
        dst = target / Path(rel).name
        if dst.exists():
            if sha_file(dst) != sha_file(src):
                raise ValueError(f"conflicting sample file {dst}")
        else:
            copy(src, dst)
        samples_manifest["files"].append({
            "name": dst.name, "sha256": sha_file(dst),
            "rows": sum(1 for _ in dst.open())})
    for rel in spec.get("exclude", []):
        copy(SRC / rel, target / "exclude_samples.jsonl")
        samples_manifest["exclude_sha256"] = sha_file(target / "exclude_samples.jsonl")
    write_json(target / "samples_manifest.json", samples_manifest)
    for tag, rel in spec.get("manifests", []):
        copy(SRC / rel, target / f"manifest__{tag}.json")
    for tag, rel in spec.get("status", []):
        copy(SRC / rel, target / f"status__{tag}.json")
    for tag, rel in spec.get("metrics_sources", []):
        copy(SRC / rel, target / f"source_metrics__{tag}.json")
    for tag, rel in spec.get("configs", []):
        copy(SRC / rel, target / f"config__{tag}.yaml")
    for rel in spec.get("extra", []):
        copy(SRC / rel, target / Path(rel).name)
    return target


def merged_metrics(spec):
    """Build {method: method_metrics} merged from the primary source metrics files."""
    merged = {}
    for tag, rel in spec.get("metrics_merge", []):
        data = json.loads((SRC / rel).read_text())
        for method, value in data.items():
            if method in METHODS and method not in merged:
                merged[method] = value
    return merged


def export_common_run_dataset(exp, ds, spec):
    target = prepare_dataset_dir(exp, ds, spec)
    method_rows = merge_method_rows(spec["method_dbs"])
    metrics = (json.loads((SRC / spec["metrics_verbatim"]).read_text())
               if spec.get("metrics_verbatim") else merged_metrics(spec))
    methods_index = {}
    for method in METHODS:
        method_dir = target / method
        if (method_dir / "SHA256SUMS").exists():
            # resume: method already exported; rebuild index entry from disk
            entries = index_entries_from_disk(method_dir)
            answers = json.loads((method_dir / "answers.json").read_text())
            preds = [e for e in entries]
            methods_index[method] = {
                "predictions": len(preds), "shards": len({e["shard"] for e in preds}),
                "entries": preds, "answers_rows": len(answers),
                "sources": method_rows[method]["sources"], "resumed": True}
            continue
        rows = method_rows[method]["rows"]
        if not rows:
            raise ValueError(f"{exp}/{ds}/{method}: no predictions")
        entries, shards = export_predictions(method_dir, rows)
        # answers: concatenate per-method answers files (or single merged file)
        answers = []
        for rel in spec["answers"][method]:
            answers.extend(json.loads((SRC / rel).read_text()))
        write_json(method_dir / "answers.json", answers)
        # per-method scores slice from merged metrics
        if not spec.get("metrics_verbatim") and method in metrics:
            write_json(method_dir / "scores.json", metrics[method])
        method_sums(method_dir)
        methods_index[method] = {
            "predictions": len(entries), "shards": shards, "entries": entries,
            "sources": method_rows[method]["sources"],
            "answers_rows": len(answers),
        }
    if spec.get("metrics_verbatim"):
        copy(SRC / spec["metrics_verbatim"], target / "metrics.json")
    else:
        write_json(target / "metrics.json", metrics)
    return methods_index


def export_diagnostic():
    exp, ds = "pope_prompt_diagnostic", "coco_192"
    spec = {
        "samples": ["results/pope_prompt_diagnostic/samples.jsonl"],
        "manifests": [("run", "results/pope_prompt_diagnostic/manifest.json")],
        "status": [], "metrics_sources": [("run", "results/pope_prompt_diagnostic/metrics.json")],
        "configs": [],
        "extra": ["results/pope_prompt_diagnostic/run_diag.py",
                  "results/pope_prompt_diagnostic/score.py",
                  "results/pope_prompt_diagnostic/RESULTS.md"],
    }
    target = prepare_dataset_dir(exp, ds, spec)
    samples = jsonl_rows(DIAG_SAMPLES)
    methods_index = {}
    for variant in ("A", "B", "C", "D"):
        method_dir = target / variant
        db_path = SRC / f"results/pope_prompt_diagnostic/{variant}/predictions.sqlite"
        if (method_dir / "SHA256SUMS").exists():
            entries = index_entries_from_disk(method_dir)
            answers = json.loads((method_dir / "answers.json").read_text())
            methods_index[variant] = {
                "predictions": len(entries), "shards": len({e["shard"] for e in entries}),
                "entries": entries, "answers_rows": len(answers),
                "sources": [{"path": f"results/pope_prompt_diagnostic/{variant}/predictions.sqlite",
                             "sha256": sha_file(db_path)}], "resumed": True}
            continue
        db = load_db(db_path)
        # diagnostic DBs have no method column: single pseudo-method ""
        rows = db.get("", {})
        if not rows:
            raise ValueError(f"{variant}: no predictions")
        entries, shards = export_predictions(method_dir, rows)
        # generate scored answers rows (same fields score.py used)
        answers = []
        for row in samples:
            question = {"A": row["q0"], "B": INSTRUCTION + "\n" + row["q0"],
                        "C": row["q0"], "D": row["q0"] + SUFFIX}[variant]
            sample = {"id": row["id"], "split": row["split"], "label": row["label"],
                      "image": row["image"], "image_sha256": row["image_sha256"],
                      "question": question}
            key = input_key(sample)
            record = json.loads(rows[key])
            text = record["text"]
            if variant == "B" and "</think>" in text:
                answer, closed = text.rsplit("</think>", 1)[1].strip(), True
            elif variant == "B":
                answer, closed = text, False
            else:
                answer, closed = text, None
            answers.append({
                "id": row["id"], "split": row["split"], "label": row["label"], "q0": row["q0"],
                "answer": answer, "generated_text": text,
                "parsed": "no" if any(w in text.split(".")[0].replace(",", "").split(" ")
                                      for w in ("No", "not", "no")) else "yes",
                "think_closed": closed,
                "answer_token_length": record["answer_token_length"],
                "eos_found": record["eos_found"],
                "executed_steps": record["executed_steps"],
                "end_to_end_seconds": record["end_to_end_seconds"],
                "early_stopped": record.get("early_stopped"),
                "stop_reason": record.get("stop_reason"),
                "image_sha256": row["image_sha256"],
            })
        write_json(method_dir / "answers.json", answers)
        # per-variant scores slice
        metrics = json.loads((SRC / "results/pope_prompt_diagnostic/metrics.json").read_text())
        write_json(method_dir / "scores.json", metrics["variants"][variant])
        method_sums(method_dir)
        methods_index[variant] = {
            "predictions": len(entries), "shards": shards, "entries": entries,
            "sources": [{"path": f"results/pope_prompt_diagnostic/{variant}/predictions.sqlite",
                         "sha256": sha_file(db_path)}],
            "answers_rows": len(answers),
        }
    write_json(target / "metrics.json", json.loads(
        (SRC / "results/pope_prompt_diagnostic/metrics.json").read_text()))
    return methods_index


def main():
    # records_index.json uses the flat {"<experiment>/<dataset>": {method: {...}}}
    # convention of the existing repo index; per-method SHA256SUMS make the
    # export resumable (completed method dirs are skipped and re-indexed).
    index = {}

    # ---- POPE yes/no suffix, 128/64/32 ----
    exp = "pope_yesno_128_64_32"
    vcd_pope = "vcd/results/early_eos/pope_full"
    vig_pope = "vigor/results/early_eos/pope_full"
    ext = "results/pope_extended"
    specs = {
        "coco": {
            "method_dbs": {
                "native": [f"{vcd_pope}/predictions.sqlite"],
                "vcd": [f"{vcd_pope}/predictions.sqlite"],
                "vigor": [f"{vig_pope}/predictions.sqlite"],
            },
            "answers": {
                "native": [f"{vcd_pope}/native_answers.json"],
                "vcd": [f"{vcd_pope}/vcd_answers.json"],
                "vigor": [f"{vig_pope}/vigor_answers.json"],
            },
            "samples": ["data/pope_coco_yesno.jsonl"],
            "exclude": ["data/calibration.jsonl"],
            "manifests": [("vcd_run", f"{vcd_pope}/manifest.json"),
                          ("vigor_run", f"{vig_pope}/manifest.json")],
            "status": [("vcd_run", f"{vcd_pope}/status.json"),
                       ("vigor_run", f"{vig_pope}/status.json")],
            "metrics_sources": [("vcd_run", f"{vcd_pope}/metrics.json"),
                                ("vigor_run", f"{vig_pope}/metrics.json")],
            "metrics_merge": [("vcd_run", f"{vcd_pope}/metrics.json"),
                              ("vigor_run", f"{vig_pope}/metrics.json")],
            "configs": [("vcd", "vcd/config.yaml"), ("vigor", "vigor/config.yaml")],
        },
        "aokvqa": {
            "method_dbs": {m: [f"{ext}/aokvqa/predictions.sqlite"] for m in METHODS},
            "answers": {m: [f"{ext}/aokvqa/{m}_answers.json"] for m in METHODS},
            "samples": ["data/pope_extended/aokvqa_yesno.jsonl"],
            "exclude": ["data/pope_extended/excluded_samples.jsonl"],
            "manifests": [("run", f"{ext}/aokvqa/manifest.json")],
            "status": [("run", f"{ext}/aokvqa/status.json")],
            "metrics_sources": [("run", f"{ext}/aokvqa/metrics.json")],
            "metrics_merge": [("run", f"{ext}/aokvqa/metrics.json")],
            "configs": [("vcd", "vcd/config.yaml")],
        },
        "gqa": {
            "method_dbs": {m: [f"{ext}/gqa/predictions.sqlite"] for m in METHODS},
            "answers": {m: [f"{ext}/gqa/{m}_answers.json"] for m in METHODS},
            "samples": ["data/pope_extended/gqa_yesno.jsonl"],
            "exclude": ["data/pope_extended/excluded_samples.jsonl"],
            "manifests": [("run", f"{ext}/gqa/manifest.json")],
            "status": [("run", f"{ext}/gqa/status.json")],
            "metrics_sources": [("run", f"{ext}/gqa/metrics.json")],
            "metrics_merge": [("run", f"{ext}/gqa/metrics.json")],
            "configs": [("vcd", "vcd/config.yaml")],
        },
    }
    for ds, spec in specs.items():
        index[f"{exp}/{ds}"] = export_common_run_dataset(exp, ds, spec)

    # ---- POPE yes/no suffix, 256/128/32 ----
    exp = "pope_yesno_256_128_32"
    base = "results/pope_256_128_yesno"
    specs = {}
    for ds, srcname in (("coco", "coco"), ("aokvqa", "aokvqa"), ("gqa", "gqa")):
        specs[ds] = {
            "method_dbs": {m: [f"{base}/{srcname}/predictions.sqlite"] for m in METHODS},
            "answers": {m: [f"{base}/{srcname}/{m}_answers.json"] for m in METHODS},
            "samples": [f"data/pope_coco_yesno.jsonl" if ds == "coco"
                        else f"data/pope_extended/{ds}_yesno.jsonl"],
            "exclude": ["data/calibration.jsonl" if ds == "coco"
                        else "data/pope_extended/excluded_samples.jsonl"],
            "manifests": [("run", f"{base}/{srcname}/manifest.json")],
            "status": [("run", f"{base}/{srcname}/status.json")],
            "metrics_sources": [("run", f"{base}/{srcname}/metrics.json")],
            "metrics_merge": [("run", f"{base}/{srcname}/metrics.json")],
            "configs": [("run", f"{base}/config.yaml")],
        }
    for ds, spec in specs.items():
        index[f"{exp}/{ds}"] = export_common_run_dataset(exp, ds, spec)
    # experiment-level summary files
    expdir = OUT / "records" / exp
    copy(SRC / f"{base}/RESULTS.md", expdir / "RESULTS.md")
    copy(SRC / f"{base}/metrics.json", expdir / "metrics_experiment.json")

    # ---- CHAIR 500, 64/32/16 ----
    exp = "chair_64_32_16"
    vcd_ch = "vcd/results/early_eos/chair_full"
    vig_ch = "vigor/results/early_eos/chair_full"
    spec = {
        "method_dbs": {
            "native": [f"{vcd_ch}/predictions.sqlite"],
            "vcd": [f"{vcd_ch}/predictions.sqlite"],
            "vigor": [f"{vig_ch}/predictions.sqlite"],
        },
        "answers": {
            "native": [f"{vcd_ch}/native_answers.json"],
            "vcd": [f"{vcd_ch}/vcd_answers.json"],
            "vigor": [f"{vig_ch}/vigor_answers.json"],
        },
        "samples": ["data/chair500.jsonl"],
        "exclude": ["data/calibration.jsonl"],
        "manifests": [("vcd_run", f"{vcd_ch}/manifest.json"),
                      ("vigor_run", f"{vig_ch}/manifest.json")],
        "status": [("vcd_run", f"{vcd_ch}/status.json"),
                   ("vigor_run", f"{vig_ch}/status.json")],
        "metrics_sources": [("vcd_run", f"{vcd_ch}/metrics.json"),
                            ("vigor_run", f"{vig_ch}/metrics.json"),
                            ("vcd_chair", f"{vcd_ch}/chair_metrics.json"),
                            ("vigor_chair", f"{vig_ch}/chair_metrics.json")],
        "metrics_merge": [("vcd_run", f"{vcd_ch}/metrics.json"),
                          ("vigor_run", f"{vig_ch}/metrics.json")],
        "configs": [("vcd", "vcd/config.yaml"), ("vigor", "vigor/config.yaml")],
        "extra": ["REPRODUCTION_AUDIT.md"],
    }
    part = export_common_run_dataset(exp, "chair500", spec)
    index[f"{exp}/chair500"] = part
    # per-method chair scorer outputs -> method dirs
    dsdir = OUT / "records" / exp / "chair500"
    chair_metrics = {}
    for tag, rel in (("vcd_run", f"{vcd_ch}"), ("vigor_run", f"{vig_ch}")):
        data = json.loads((SRC / rel / "chair_metrics.json").read_text())
        chair_metrics[tag] = data
        for method in data["methods"]:
            src_chair = SRC / rel / f"{method}_chair.json"
            if src_chair.exists():
                copy(src_chair, dsdir / method / "chair.json")
    write_json(dsdir / "chair_metrics.json", {
        "vcd_run": chair_metrics["vcd_run"], "vigor_run": chair_metrics["vigor_run"]})
    # regenerate method SHA256SUMS now that chair.json was added
    for method in METHODS:
        method_sums(dsdir / method)

    # ---- MMMU wino v1 (think 256/128/64), VCD + VIGOR ----
    exp = "mmmu_wino_v1"
    spec = {
        "method_dbs": {
            "vcd": ["vcd/results/mmmu_wino_v1/shard0/predictions.sqlite",
                    "vcd/results/mmmu_wino_v1/shard1/predictions.sqlite"],
            "vigor": ["vigor/results/mmmu_wino_v1/shard0/predictions.sqlite",
                      "vigor/results/mmmu_wino_v1/shard1/predictions.sqlite"],
        },
        "answers": {
            "vcd": ["vcd/results/mmmu_wino_v1/shard0/vcd_answers.json",
                    "vcd/results/mmmu_wino_v1/shard1/vcd_answers.json"],
            "vigor": ["vigor/results/mmmu_wino_v1/shard0/vigor_answers.json",
                      "vigor/results/mmmu_wino_v1/shard1/vigor_answers.json"],
        },
        "samples": ["data/mmmu_wino_v1/shard_0.jsonl", "data/mmmu_wino_v1/shard_1.jsonl"],
        "manifests": [("vcd_shard0", "vcd/results/mmmu_wino_v1/shard0/manifest.json"),
                      ("vcd_shard1", "vcd/results/mmmu_wino_v1/shard1/manifest.json"),
                      ("vigor_shard0", "vigor/results/mmmu_wino_v1/shard0/manifest.json"),
                      ("vigor_shard1", "vigor/results/mmmu_wino_v1/shard1/manifest.json")],
        "status": [("vcd_shard0", "vcd/results/mmmu_wino_v1/shard0/status.json"),
                   ("vcd_shard1", "vcd/results/mmmu_wino_v1/shard1/status.json"),
                   ("vigor_shard0", "vigor/results/mmmu_wino_v1/shard0/status.json"),
                   ("vigor_shard1", "vigor/results/mmmu_wino_v1/shard1/status.json")],
        "metrics_sources": [("run", "results/mmmu_wino_v1/metrics.json"),
                            ("vcd_scores", "results/mmmu_wino_v1/vcd_scores.json"),
                            ("vigor_scores", "results/mmmu_wino_v1/vigor_scores.json")],
        "configs": [("run", "configs/mmmu_wino_v1.yaml")],
        "extra": ["MMMU.md"],
    }
    # custom two-method export (no native)
    target = prepare_dataset_dir(exp, "mmmu_val", spec)
    method_rows = merge_method_rows(spec["method_dbs"])
    mmmu_methods = {}
    for method in ("vcd", "vigor"):
        method_dir = target / method
        if (method_dir / "SHA256SUMS").exists():
            entries = index_entries_from_disk(method_dir)
            answers = json.loads((method_dir / "answers.json").read_text())
            mmmu_methods[method] = {
                "predictions": len(entries), "shards": len({e["shard"] for e in entries}),
                "entries": entries, "answers_rows": len(answers),
                "sources": method_rows[method]["sources"], "resumed": True}
            continue
        rows = method_rows[method]["rows"]
        entries, shards = export_predictions(method_dir, rows)
        answers = []
        for rel in spec["answers"][method]:
            answers.extend(json.loads((SRC / rel).read_text()))
        write_json(method_dir / "answers.json", answers)
        score_rel = f"results/mmmu_wino_v1/{method}_scores.json"
        write_json(method_dir / "scores.json", json.loads((SRC / score_rel).read_text()))
        method_sums(method_dir)
        mmmu_methods[method] = {
            "predictions": len(entries), "shards": shards, "entries": entries,
            "sources": method_rows[method]["sources"], "answers_rows": len(answers)}
    write_json(target / "metrics.json", json.loads(
        (SRC / "results/mmmu_wino_v1/metrics.json").read_text()))
    index[f"{exp}/mmmu_val"] = mmmu_methods
    expdir = OUT / "records" / exp
    copy(SRC / "results/mmmu_wino_v1/RESULTS.md", expdir / "RESULTS.md")

    # ---- HallusionBench 256/128/32 ----
    exp = "hallusion_full_original_256_128"
    hb = "results/hallusion_full_original_256_128"
    spec = {
        "method_dbs": {m: [f"{hb}/shard0/predictions.sqlite",
                           f"{hb}/shard1/predictions.sqlite"] for m in METHODS},
        "answers": {m: [f"{hb}/shard0/{m}_answers.json",
                        f"{hb}/shard1/{m}_answers.json"] for m in METHODS},
        "samples": ["data/hallusion_full_original/shard_0.jsonl",
                    "data/hallusion_full_original/shard_1.jsonl"],
        "manifests": [("shard0", f"{hb}/shard0/manifest.json"),
                      ("shard1", f"{hb}/shard1/manifest.json")],
        "status": [("shard0", f"{hb}/shard0/status.json"),
                   ("shard1", f"{hb}/shard1/status.json")],
        "metrics_sources": [("run", f"{hb}/metrics.json"),
                            ("native_scores", f"{hb}/native_scores.json"),
                            ("vcd_scores", f"{hb}/vcd_scores.json"),
                            ("vigor_scores", f"{hb}/vigor_scores.json")],
        "metrics_verbatim": f"{hb}/metrics.json",
        "configs": [("run", "configs/hallusion_256_128.yaml")],
        "extra": ["results/hallusion_full_original_256_128/RESULTS.md", "HALLUSION.md",
                  "data/hallusion_full_original/manifest.json",
                  "data/hallusion_full_original/original_prompt_validation.json"],
    }
    part = export_common_run_dataset(exp, "hallusionbench", spec)
    index[f"{exp}/hallusionbench"] = part
    # per-method scores slices
    dsdir = OUT / "records" / exp / "hallusionbench"
    for method in METHODS:
        rel = f"results/hallusion_full_original_256_128/{method}_scores.json"
        if (SRC / rel).exists():
            write_json(dsdir / method / "scores.json",
                       json.loads((SRC / rel).read_text()))
            method_sums(dsdir / method)
    expdir = OUT / "records" / exp
    copy(SRC / f"{hb}/RESULTS.md", expdir / "RESULTS.md")

    # ---- POPE prompt diagnostic A/B/C/D ----
    index["pope_prompt_diagnostic/coco_192"] = export_diagnostic()

    write_json(OUT / "records_index.json", index)
    print(json.dumps({key: {m: v["predictions"] for m, v in methods.items()}
                      for key, methods in sorted(index.items())}, indent=1))


THINK_EXP = "pope_think_256_128_64"
THINK_BASE = f"results/{THINK_EXP}"


def export_think(datasets):
    """Thinking-prompt POPE 256/128/64: export completed datasets only and
    merge them into the existing records_index.json."""
    sys.path.insert(0, str(SRC))
    sys.path.insert(0, str(SRC / THINK_BASE))
    import run_queue as rq
    index_path = OUT / "records_index.json"
    index = json.loads(index_path.read_text())
    for ds in datasets:
        status = json.loads((SRC / THINK_BASE / "queue_status.json").read_text())
        if status["stages"].get(ds, {}).get("state") != "generation_and_scoring_complete":
            raise ValueError(f"{ds}: not complete")
        run = f"{THINK_BASE}/{ds}"
        spec = {
            "method_dbs": {m: [f"{run}/predictions.sqlite"] for m in METHODS},
            "answers": {m: [f"{run}/{m}_answers.json"] for m in METHODS},
            "samples": [f"{THINK_BASE}/inputs/{ds}_think.jsonl"],
            "manifests": [("run", f"{run}/manifest.json")],
            "status": [("run", f"{run}/status.json")],
            "metrics_sources": [("run", f"{run}/metrics.json")],
            "metrics_merge": [("run", f"{run}/metrics.json")],
            "configs": [("run", f"{THINK_BASE}/config.yaml")],
            "extra": [f"{run}/merge_receipt.json", f"{THINK_BASE}/plan.json",
                      f"{THINK_BASE}/run_queue.py", f"{THINK_BASE}/README.md"],
        }
        index[f"{THINK_EXP}/{ds}"] = export_common_run_dataset(THINK_EXP, ds, spec)
        answers = {m: json.loads((SRC / run / f"{m}_answers.json").read_text())
                   for m in METHODS}
        primary = {m: rq.primary_block(answers[m], answers["native"], m) for m in METHODS}
        write_json(dataset_dir(THINK_EXP, ds) / "primary_metrics.json", primary)
    write_json(index_path, index)
    print(json.dumps({k: {m: v["predictions"] for m, v in index[k].items()}
                      for k in index if k.startswith(THINK_EXP)}, indent=1))


def write_judge_dir(target, src, judged_files, cache):
    """Copy judge summaries/per-row files and bundle every referenced cached
    API request/response into api_records.jsonl.gz."""
    target.mkdir(parents=True, exist_ok=True)
    needed = set()
    for name in judged_files:
        rows = json.loads((src / name).read_text())
        needed |= {r["request_sha256"] for r in rows}
        copy(src / name, target / name)
    for name in ("summary.json",):
        copy(src / name, target / name)
    with gzip.open(target / "api_records.jsonl.gz", "wt") as handle:
        for sha in sorted(needed):
            record = json.loads((cache / f"{sha}.json").read_text())
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    names = sorted(p.name for p in target.iterdir() if p.name != "SHA256SUMS")
    (target / "SHA256SUMS").write_text("".join(f"{sha_file(target / n)}  {n}\n" for n in names))
    return len(needed)


def export_think_judge(datasets):
    judge = SRC / THINK_BASE / "judge"
    for ds in datasets:
        target = dataset_dir(THINK_EXP, ds) / "judge"
        copy(judge / "judge_manifest.json", target / "judge_manifest.json")
        n = write_judge_dir(target, judge / ds, [f"{m}_judged.json" for m in METHODS], judge / "cache")
        print(ds, "judge records", n)


def export_mmmu_judge():
    judge = SRC / "results/mmmu_wino_v1/judge"
    target = dataset_dir("mmmu_wino_v1", "mmmu_val") / "judge"
    copy(judge / "judge_manifest.json", target / "judge_manifest.json")
    n = write_judge_dir(target, judge, [f"{m}_judged.json" for m in METHODS], judge / "cache")
    print("mmmu judge records", n)


TB_EXP = "think_benchmarks_256_128_64"
TB_STAGES = {"hallusion_think": ("data/hallusion_think", ("vcd", "vigor")),
             "mme_think": ("data/mme/think", ("native", "vcd", "vigor")),
             "mme_original": ("data/mme/original", ("native", "vcd", "vigor"))}


def export_think_benchmark(stage):
    """HallusionBench/MME 256/128/64 stages from experiments/think_benchmarks.py."""
    data, methods = TB_STAGES[stage]
    run = f"results/{TB_EXP}/{stage}"
    if not (SRC / run / "metrics.json").is_file():
        raise ValueError(f"{stage}: not scored")
    spec = {
        "method_dbs": {m: [f"{run}/shard0/predictions.sqlite", f"{run}/shard1/predictions.sqlite"]
                       for m in methods},
        "samples": [f"{data}/samples.jsonl"],
        "manifests": [("data", f"{data}/manifest.json"), ("shard0", f"{run}/shard0/manifest.json"),
                      ("shard1", f"{run}/shard1/manifest.json")],
        "status": [], "metrics_sources": [],
        "configs": [("run", "configs/think_256_128_64.yaml")],
        "extra": [f"{run}/RESULTS.md", f"{run}/preflight_validation.json",
                  f"results/{TB_EXP}/plan.json", f"results/{TB_EXP}/plan_v1_overstrict_audit.json",
                  "experiments/think_benchmarks.py"],
    }
    target = prepare_dataset_dir(TB_EXP, stage, spec)
    method_rows = merge_method_rows(spec["method_dbs"])
    part = {}
    for method in methods:
        method_dir = target / method
        rows = method_rows[method]["rows"]
        entries, shards = export_predictions(method_dir, rows)
        answers = json.loads((SRC / run / f"{method}_answers.json").read_text())
        write_json(method_dir / "answers.json", answers)
        write_json(method_dir / "scores.json", json.loads((SRC / run / f"{method}_scores.json").read_text()))
        method_sums(method_dir)
        part[method] = {"predictions": len(entries), "shards": shards, "entries": entries,
                        "sources": method_rows[method]["sources"], "answers_rows": len(answers)}
    copy(SRC / run / "metrics.json", target / "metrics.json")
    judge = SRC / run / "judge"
    if judge.is_dir():
        copy(judge / "judge_manifest.json", target / "judge" / "judge_manifest.json")
        n = write_judge_dir(target / "judge", judge, [f"{m}_judged.json" for m in methods],
                            SRC / "results/pope_think_256_128_64/judge/cache")
        print(stage, "judge records", n)
    index_path = OUT / "records_index.json"
    index = json.loads(index_path.read_text())
    index[f"{TB_EXP}/{stage}"] = part
    write_json(index_path, index)
    print(stage, {m: v["predictions"] for m, v in part.items()})


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--think-benchmark":
        export_think_benchmark(sys.argv[2])
    elif len(sys.argv) > 2 and sys.argv[1] == "--think":
        export_think(sys.argv[2:])
    elif len(sys.argv) > 2 and sys.argv[1] == "--think-judge":
        export_think_judge(sys.argv[2:])
    elif sys.argv[1:] == ["--mmmu-judge"]:
        export_mmmu_judge()
    else:
        main()
