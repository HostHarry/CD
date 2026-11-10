"""Freeze a paired runtime audit from completed predictions without running a model."""
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import statistics
import subprocess

from cdapc_lab.io.artifacts import atomic_json, file_hash, read_json

ROOT = Path(__file__).resolve().parents[1]


def read_runtime(path):
    p = read_json(path)
    groups = defaultdict(list)
    for event in p["commit_log"]:
        groups[event["iteration"]].append(event)
    assert len(groups) == p["decode_iterations"]
    known_eos = None
    tail_only = 0
    for _, events in sorted(groups.items()):
        if known_eos is not None and all(e["position"] > known_eos for e in events):
            tail_only += 1
        new_eos = [e["position"] for e in events if e["written_id"] in p["eos_ids"]]
        if new_eos:
            known_eos = min(new_eos + ([] if known_eos is None else [known_eos]))
    assert p["cost"]["model_forward_calls"] == p["decode_iterations"] * 2
    return {
        "sample_id": p["sample_id"], "path": str(path), "sha256": file_hash(path),
        "generation_fingerprint": p["generation_fingerprint"], "status": p["status"],
        "finish_reason": p["finish_reason"], "seconds": p["generation_seconds"],
        "preprocess_seconds": p["preprocess_seconds"], "iterations": p["decode_iterations"],
        "forward_calls": p["cost"]["model_forward_calls"], "tail_only_iterations": tail_only,
        "committed": p["committed_buffer_tokens"], "visible_tokens": p["emitted_tokens_before_eos"],
        "sequence_length": p["actual_sequence_length"],
        "fallback_iterations": p["metrics"]["fallback_iterations"],
    }


def summarize(records):
    keys = ("seconds", "preprocess_seconds", "iterations", "forward_calls", "tail_only_iterations",
            "committed", "visible_tokens", "sequence_length", "fallback_iterations")
    steps = sum(r["iterations"] for r in records)
    return {
        "n": len(records), "means": {k: statistics.mean(r[k] for r in records) for k in keys},
        "generation_statuses": dict(Counter(r["status"] for r in records)),
        "finish_reasons": dict(Counter(r["finish_reason"] for r in records)),
        "seconds_per_iteration": sum(r["seconds"] for r in records) / steps,
        "tail_only_iteration_fraction": sum(r["tail_only_iterations"] for r in records) / steps,
    }


def main():
    new_root = ROOT.parent / "cdapc-pope-safe-eos"
    old_root = ROOT.parent / "cdapc-pope-eos-full"
    new_name = "pope_coco_think_safe_eos_r70_k4_L256_I128"
    old_name = "pope_coco_think_eos_prefix_r70_k4_L256_I128"
    new_stage = read_json(new_root / "reports/index.json")["stages"][new_name]
    old_stage = read_json(old_root / "reports/index.json")["stages"][old_name]
    assert old_stage["status"] == "complete"
    old_paths = {
        path.name: path for shard in old_stage["shards"].values()
        for path in (old_root / shard["run"] / "predictions").glob("*.json")
    }
    new_paths = [
        path for shard in new_stage["shards"].values()
        for path in sorted((new_root / shard["run"] / "predictions").glob("*.json"))
    ]
    records = []
    for path in new_paths:
        new = read_runtime(path)
        old = read_runtime(old_paths[path.name])
        assert new["sample_id"] == old["sample_id"]
        records.append({"original_candidates": new, "eos_prefix_candidates": old})
    new_stats = summarize([r["original_candidates"] for r in records])
    old_stats = summarize([r["eos_prefix_candidates"] for r in records])
    code_paths = [
        new_root / "src/cdapc_lab" / suffix for suffix in
        ("models/mmada.py", "models/base.py", "decoding/engine.py",
         "decoding/proposal.py", "decoding/selection.py", "experiments/infer.py")
    ]
    repo = ROOT.parent / "cd-mmmu-upload"
    git = lambda *args: subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    result = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "stage_status_at_capture": new_stage["status"], "expected_inputs": new_stage["expected_inputs"],
        "source_note": "Paired completed input subset; current stage not assumed complete.",
        "cd_origin_main": git("rev-parse", "origin/main"),
        "cd_origin_main_committed_at": git("show", "-s", "--format=%cI", "origin/main"),
        "external_vcd_vigor": {
            "implementation_available_in_checked_origin_main": False,
            "quoted_seconds_per_input": 11.9,
            "timings_source": "User-provided report; configuration and timing scope unverified",
        },
        "analysis_source_sha256": file_hash(Path(__file__)),
        "source_hashes": {str(p): file_hash(p) for p in code_paths},
        "original_candidates": new_stats, "eos_prefix_candidates": old_stats,
        "paired_mean_speedup": new_stats["means"]["seconds"] / old_stats["means"]["seconds"],
        "records": records,
    }
    target = ROOT / "reports/pope_runtime_audit.json"
    atomic_json(target, result)
    target.with_suffix(".sha256").write_text("".join(
        f"{file_hash(p)}  {p}\n" for p in (Path(__file__), target)))
    print({k: v for k, v in result.items() if k not in ("records", "source_hashes")})


if __name__ == "__main__":
    main()
