"""Audit immutable prefixes and test safe EOS against preserved real-model runs."""
import argparse
import json
from pathlib import Path
import sqlite3
import statistics
import torch

from common.eos import answer_ids, closed_prefix_end
from common.run import input_key
from common.runtime import ROOT, Runtime, save_json, sha256, source_hashes
from migrate_results import verify_snapshot


def audit_record(record, mask, eos):
    final = record.get("generation_state_ids", record["generated_ids"])
    state = [mask] * len(final)
    prompt = len(record["input"]["prompt_ids"])
    stop, prefix = None, None
    for index, step in enumerate(record["trace"], 1):
        for pos, token in zip(step["committed_positions"][0], step["committed_tokens"][0]):
            offset = pos - prompt
            if not 0 <= offset < len(state) or state[offset] != mask:
                raise ValueError("generator overwrote an already committed token")
            state[offset] = token
        end = closed_prefix_end(state, eos, mask)
        if stop is None and end is not None:
            stop, prefix = index, state[:end]
    if state != final:
        raise ValueError("trace does not reconstruct the recorded generation state")
    if prefix is not None and prefix != answer_ids(record):
        raise ValueError("committed prefix differs from the final answer")
    return stop


def assert_equal(reference, actual, mask, eos):
    if answer_ids(reference) != answer_ids(actual) or reference["text"] != actual["text"]:
        raise ValueError("early stopping changed the answer")
    if reference["input"] != actual["input"]:
        raise ValueError("early stopping changed input preparation or noise")
    audit_record(actual, mask, eos)
    steps = actual["executed_steps"]
    for before, after in zip(reference["trace"][:steps], actual["trace"]):
        for key in ("block", "step", "committed_positions", "committed_tokens",
                    "committed_confidence", "remaining_masks"):
            if before[key] != after[key]:
                raise ValueError(f"commit trajectory changed: {key}")
    if steps != len(actual["trace"]) or steps > len(reference["trace"]):
        raise ValueError("invalid executed step count")
    if actual["early_stopped"]:
        if mask in actual["generated_ids"] or actual["generated_ids"][-1] != eos:
            raise ValueError("stopped output is not a complete EOS-terminated prefix")
        suffix = actual["generation_state_ids"][len(actual["generated_ids"]):]
        if actual["uncomputed_suffix_masks"] != suffix.count(mask):
            raise ValueError("uncomputed suffix was not recorded accurately")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["vcd", "vigor"], required=True)
    args = parser.parse_args()
    base = ROOT / args.method / "results"
    manifest = verify_snapshot(base / "pope_full")
    config = manifest["config"]
    runtime = Runtime(config["checkpoint"], config["vq_checkpoint"],
                      resolution=config["resolution"], resize=config["resize"], early_eos=True)
    mask, eos = runtime.model.config.mask_token_id, runtime.tokenizer.eos_token_id
    report = {"method": args.method, "sources": source_hashes(), "historical_manifests": {},
              "trace_audits": {}, "samples": [], "paired_timing": [],
              "mask_id": mask, "eos_id": eos, "memory_checks": []}
    for benchmark, filename, directory in (
        ("POPE", "pope_coco_yesno.jsonl", "pope_full"), ("CHAIR", "chair500.jsonl", "chair_full"),
    ):
        manifest = verify_snapshot(base / directory)
        report["historical_manifests"][directory] = sha256(base / directory / "manifest.json")
        unique = {}
        for line in (ROOT / "data" / filename).read_text().splitlines():
            sample = json.loads(line)
            unique.setdefault(input_key(sample), sample)
        with sqlite3.connect((base / directory / "predictions.sqlite").as_uri() + "?mode=ro", uri=True) as db:
            stored = {method: {key: json.loads(payload) for key, payload in db.execute(
                "SELECT input_key,payload FROM predictions WHERE method=?", (method,))}
                      for method in ("native", args.method)}
        stops = {}
        for method, records in stored.items():
            values = {}
            for key, record in records.items():
                values[key] = audit_record(record, mask, eos)
            report["trace_audits"][f"{directory}/{method}"] = {
                "rows": len(values), "all_committed_prefixes_immutable": True,
                "safe_eos_rows": sum(v is not None for v in values.values()),
            }
            if method == args.method:
                stops = values
        keys = [key for key in unique if key in stored[args.method]]
        if benchmark == "POPE":
            proposed = keys[:4] + keys[-2:]
            proposed += [min(keys, key=lambda k: stops[k] or manifest["generation"]["steps"]),
                         max(keys, key=lambda k: stops[k] or manifest["generation"]["steps"])]
            selected = list(dict.fromkeys(proposed))
        else:
            selected = keys[:2]
        if benchmark == "POPE":
            for _ in range(3):
                runtime.generate(unique[selected[0]], manifest["methods"][args.method],
                                 manifest["generation"], config["seed"])
        for index, key in enumerate(selected):
            methods = ("native", args.method) if index == 0 else (args.method,)
            for method in methods:
                reference = stored[method][key]
                result = runtime.generate(unique[key], manifest["methods"][method], manifest["generation"], config["seed"])
                assert_equal(reference, result, mask, eos)
                item = {"benchmark": benchmark, "id": unique[key]["id"], "method": method,
                        "all_answer_tokens_and_commits_equal": True, "steps": result["executed_steps"],
                        "end_to_end_seconds": result["end_to_end_seconds"],
                        "decode_seconds": result["decode_seconds"]}
                report["samples"].append(item)
                print(item, flush=True)
                if benchmark == "POPE" and method == args.method and index < 2:
                    runtime.early_eos = False
                    full = runtime.generate(unique[key], manifest["methods"][method], manifest["generation"], config["seed"])
                    runtime.early_eos = True
                    if full["generated_ids"] != reference["generated_ids"]:
                        raise ValueError("unoptimized runtime differs from historical generation")
                    assert_equal(full, result, mask, eos)
                    report["paired_timing"].append({
                        "id": unique[key]["id"], "full_seconds": full["end_to_end_seconds"],
                        "early_seconds": result["end_to_end_seconds"], "executed_steps": result["executed_steps"],
                    })
        if benchmark == "POPE":
            baseline = torch.cuda.memory_allocated(runtime.device)
            # Test consecutive early exits without empty_cache or forced GC.
            # This catches traceback cycles that isolated output checks miss.
            for index in range(8):
                key = selected[index % len(selected)]
                result = runtime.generate(unique[key], manifest["methods"][args.method],
                                          manifest["generation"], config["seed"])
                assert_equal(stored[args.method][key], result, mask, eos)
                allocated = torch.cuda.memory_allocated(runtime.device)
                if allocated > baseline + 64 * 1024 ** 2:
                    raise ValueError("live GPU allocation grew after repeated early exits")
                report["memory_checks"].append({"id": unique[key]["id"], "allocated_bytes": allocated})
            print("Consecutive early-stop memory checks passed", flush=True)
    report["all_equal"] = True
    report["timing_summary"] = {
        "paired_samples": len(report["paired_timing"]),
        "full_seconds_mean": statistics.mean(r["full_seconds"] for r in report["paired_timing"]),
        "early_seconds_mean": statistics.mean(r["early_seconds"] for r in report["paired_timing"]),
    }
    save_json(base / "early_eos_validation.json", report)
    print(json.dumps(report["timing_summary"], indent=2))


if __name__ == "__main__":
    main()
