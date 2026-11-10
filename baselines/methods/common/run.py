import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import shutil

import numpy as np
from omegaconf import OmegaConf
import torch

from .metrics import paired_bootstrap, pope_metrics
from .runtime import Runtime, REVISION, environment, method_config, save_json, sha256, source_hashes


def input_key(sample):
    return hashlib.sha256(json.dumps(
        [sample["image_sha256"], sample.get("messages", sample.get("question"))],
        ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def summarize(samples, database, methods, target_count):
    results = {}
    for method in methods:
        values = {}
        for key, payload in database.execute("SELECT input_key,payload FROM predictions WHERE method=?", (method,)):
            result = json.loads(payload)
            trace = result.pop("trace")
            result["denoising_steps"] = len(trace)
            result["active_decisions"] = sum(len(step.get("active_positions", [])) for step in trace)
            result["candidate_flips"] = sum(a != b for step in trace for a, b in zip(
                step.get("normal_top1", []), step.get("contrast_top1", [])))
            result.pop("input")
            result.pop("generated_ids")
            result.pop("raw_text")
            result.pop("generation_state_ids", None)
            values[key] = result
        rows = [{**sample, **values[input_key(sample)]} for sample in samples if input_key(sample) in values]
        results[method] = {"rows": rows}
        results[method]["metrics"] = {
            "evaluated_rows": len(rows), "expected_rows": target_count,
            "complete": len(rows) == target_count,
            "unique_predictions": len(values),
            "pope": {split: pope_metrics([r for r in rows if r.get("split") == split])
                     for split in sorted({r["split"] for r in rows if r.get("label") in {"yes", "no"}})},
        }
        if values:
            metrics = results[method]["metrics"]
            for key in ("decode_seconds", "end_to_end_seconds", "preprocessing_seconds",
                        "answer_token_length", "denoising_steps"):
                numbers = [r[key] for r in values.values()]
                metrics[key] = {"mean": float(np.mean(numbers)), "std": float(np.std(numbers))}
            metrics["peak_allocated_bytes"] = max(r["peak_allocated_bytes"] or 0 for r in values.values())
            metrics["remaining_masks"] = sum(r["remaining_masks"] for r in values.values())
            metrics["image_tokens"] = sum(r["image_token_count"] for r in values.values())
            total = sum(r["active_decisions"] for r in values.values())
            metrics["candidate_flip_ratio"] = sum(r["candidate_flips"] for r in values.values()) / total if total else None
            profiles = {}
            for row in values.values():
                profile = row.get("execution_profile", "full_schedule_v1")
                profiles.setdefault(profile, []).append(row)
            metrics["timing_by_execution_profile"] = {
                name: {"samples": len(group),
                       "decode_seconds_mean": float(np.mean([r["decode_seconds"] for r in group])),
                       "end_to_end_seconds_mean": float(np.mean([r["end_to_end_seconds"] for r in group])),
                       "executed_steps_mean": float(np.mean([r["denoising_steps"] for r in group]))}
                for name, group in profiles.items()
            }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=["native", "vcd", "vigor"], required=True)
    parser.add_argument("--benchmark", choices=["POPE", "CHAIR", "custom"], default="custom")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--exclude-samples", type=Path)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--early-eos", action="store_true")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--block-length", type=int)
    args = parser.parse_args()
    if args.limit < 0 or args.warmups < 0 or len(set(args.methods)) != len(args.methods):
        parser.error("invalid limit, warmups or duplicate methods")
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    samples = [json.loads(line) for line in args.samples.read_text().splitlines() if line.strip()]
    if not samples or len({s["id"] for s in samples}) != len(samples):
        raise ValueError("sample IDs must be nonempty and unique")
    image_hashes = {}
    for sample in samples:
        if sample.get("input_mode") == "text_only":
            if sample.get("image") is not None or sample.get("image_sha256") is not None:
                raise ValueError("text-only sample contains an image or image hash")
            continue
        path = str(Path(sample["image"]).resolve())
        if path not in image_hashes:
            image_hashes[path] = sha256(path)
        if sample.get("image_sha256") != image_hashes[path]:
            raise ValueError(f"missing or incorrect image SHA256: {path}")
    if args.exclude_samples:
        excluded = {json.loads(line)["image_sha256"] for line in args.exclude_samples.read_text().splitlines() if line.strip()}
        if excluded & {s["image_sha256"] for s in samples}:
            raise ValueError("calibration and evaluation images overlap")
    expected = len(samples)
    if args.limit:
        samples = samples[:args.limit]
    generation = dict(config["generation"])
    if args.benchmark != "custom":
        generation.update(config["benchmarks"][args.benchmark])
    for name in ("max_new_tokens", "steps", "block_length"):
        if getattr(args, name) is not None:
            generation[name] = getattr(args, name)
    length, steps, block = (generation[k] for k in ("max_new_tokens", "steps", "block_length"))
    if min(length, steps, block) < 1 or length % block or steps % (length // block):
        raise ValueError("invalid generation schedule")
    methods = {name: method_config(name, config.get("vcd") if name == "vcd" else None)
               for name in args.methods}
    manifest = {
        "upstream_revision": REVISION, "environment": environment(), "sources": source_hashes(),
        "config": config, "generation": generation, "methods": methods,
        "samples_sha256": sha256(args.samples), "limit": args.limit,
        "expected_rows": expected, "selected_rows": len(samples),
        "benchmark": args.benchmark, "warmups_per_method_per_session": args.warmups,
        "excluded_samples_sha256": sha256(args.exclude_samples) if args.exclude_samples else None,
        "seed_protocol": "same fixed seed reset for every unique input and method",
        "checkpoints": {},
    }
    if args.early_eos:
        manifest["early_eos"] = "safe_eos_v1"
    for name in ("checkpoint", "vq_checkpoint"):
        path = Path(config[name])
        manifest["checkpoints"][name] = {
            "path": str(path.resolve()),
            "metadata_sha256": {p.name: sha256(p) for p in sorted(path.glob("*.json"))},
            "weights": {p.name: {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                        for p in sorted(path.iterdir()) if p.suffix in {".safetensors", ".bin"}},
        }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = args.output / "manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError("resume rejected: source, environment, data or configuration changed")
        else:
            if (args.output / "predictions.sqlite").exists():
                raise ValueError("predictions database exists without manifest")
            save_json(manifest_path, manifest)
            from .runtime import ROOT
            for relative in manifest["sources"]:
                destination = args.output / "source" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, destination)
        with sqlite3.connect(args.output / "predictions.sqlite") as database:
            database.execute("CREATE TABLE IF NOT EXISTS predictions "
                             "(method TEXT,input_key TEXT,payload TEXT,PRIMARY KEY(method,input_key))")
            completed = set(database.execute("SELECT method,input_key FROM predictions"))
            unique = {input_key(s): s for s in samples}
            total = len(unique) * len(methods)
            if not args.score_only and len(completed) < total:
                runtime = Runtime(config["checkpoint"], config["vq_checkpoint"], args.device,
                                  config["resolution"], config["resize"], early_eos=args.early_eos)
                first = next(iter(unique.values()))
                for name, method in methods.items():
                    for _ in range(args.warmups):
                        runtime.generate(first, method, generation, config["seed"])
                    for key, sample in unique.items():
                        if (name, key) in completed:
                            continue
                        result = runtime.generate(sample, method, generation, config["seed"])
                        with database:
                            database.execute("INSERT INTO predictions VALUES (?,?,?)",
                                             (name, key, json.dumps(result, ensure_ascii=False, allow_nan=False)))
                        completed.add((name, key))
                        save_json(args.output / "progress.json", {
                            "completed_unique_predictions": len(completed), "target_unique_predictions": total,
                            "last_method": name, "last_id": sample["id"], "updated": time.time()})
                        print(f"{len(completed)}/{total} {name} {sample['id']}: {result['text'][:100]}", flush=True)
            summary = summarize(samples, database, methods, expected)
            metrics = {name: value["metrics"] for name, value in summary.items()}
            for name, value in summary.items():
                save_json(args.output / f"{name}_answers.json", value["rows"])
                if name != "native" and "native" in summary and len(value["rows"]) == expected:
                    for split in value["metrics"]["pope"]:
                        left = [r for r in summary["native"]["rows"] if r.get("split") == split]
                        right = [r for r in value["rows"] if r.get("split") == split]
                        if len(left) == len(right):
                            value["metrics"]["pope"][split]["paired_vs_native"] = paired_bootstrap(left, right)
            save_json(args.output / "metrics.json", metrics)
            save_json(args.output / "status.json", {
                "selected_run_complete": len(completed) == total,
                "full_dataset_complete": len(completed) == total and len(samples) == expected,
                "chair_scored": False if args.benchmark == "CHAIR" else None,
            })


if __name__ == "__main__":
    main()
