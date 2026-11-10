"""Verify real-model outputs and every token-commit against imported predictions."""
import argparse
import json
from pathlib import Path
import sqlite3

from omegaconf import OmegaConf

from common.runtime import ROOT, Runtime, method_config, save_json, sha256, source_hashes
from common.run import input_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["vcd", "vigor"], required=True)
    args = parser.parse_args()
    config = OmegaConf.to_container(OmegaConf.load(ROOT / args.method / "config.yaml"), resolve=True)
    runtime = Runtime(config["checkpoint"], config["vq_checkpoint"], resolution=config["resolution"], resize=config["resize"])
    report = {"method": args.method, "sources": source_hashes(), "samples": [],
              "verifier_sha256": sha256(__file__)}
    for benchmark, sample_name, directory in (
        ("POPE", "pope_coco_yesno.jsonl", "pope_full"), ("CHAIR", "chair500.jsonl", "chair_full"),
    ):
        samples = [json.loads(line) for line in (ROOT / "data" / sample_name).read_text().splitlines()][:2]
        generation = {**config["generation"], **config["benchmarks"][benchmark]}
        with sqlite3.connect(ROOT / args.method / "results" / directory / "predictions.sqlite") as db:
            for sample in samples:
                for method in ("native", args.method):
                    preserved = json.loads(db.execute(
                        "SELECT payload FROM predictions WHERE method=? AND input_key=?",
                        (method, input_key(sample))).fetchone()[0])
                    result = runtime.generate(sample, method_config(method, config.get("vcd") if method == "vcd" else None),
                                              generation, config["seed"])
                    for key in ("generated_ids", "text", "raw_text"):
                        if result[key] != preserved[key]:
                            raise ValueError(f"{benchmark}/{method}/{sample['id']} differs: {key}")
                    if result["input"]["prompt_ids"] != preserved["input"]["prompt_ids"]:
                        raise ValueError("prompt differs")
                    if len(result["trace"]) != len(preserved["trace"]):
                        raise ValueError("step count differs")
                    for left, right in zip(result["trace"], preserved["trace"]):
                        for key in ("committed_positions", "committed_tokens", "committed_confidence", "remaining_masks"):
                            if left[key] != right[key]:
                                raise ValueError(f"{benchmark}/{method}/{sample['id']} trajectory differs: {key}")
                    report["samples"].append({"benchmark": benchmark, "id": sample["id"], "method": method,
                                              "tokens_prompt_text_and_commits_equal": True})
                    print(report["samples"][-1], flush=True)
    report["all_equal"] = True
    save_json(ROOT / args.method / "results/runtime_validation.json", report)


if __name__ == "__main__":
    main()
