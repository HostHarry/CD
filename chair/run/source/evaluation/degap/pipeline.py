import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from .runtime import save_json, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", default="configs/mmada_degap.yaml")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--pope", type=Path, required=True)
    parser.add_argument("--chair", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--nltk-data", type=Path, required=True)
    parser.add_argument("--calibration-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    calibration = {json.loads(l)["image_sha256"] for l in args.calibration.read_text().splitlines() if l.strip()}
    for dataset in (args.pope, args.chair):
        images = {json.loads(l)["image_sha256"] for l in dataset.read_text().splitlines() if l.strip()}
        if images & calibration:
            raise ValueError("calibration images overlap with evaluation")
    status = {"stages": {}, "started": time.time()}

    def execute(stage, arguments):
        status["stages"][stage] = {"state": "in_progress", "started": time.time(), "argv": arguments}
        save_json(args.output / "pipeline_status.json", status)
        with (args.output / f"{stage}.log").open("a") as log:
            result = subprocess.run([sys.executable, "-u", "-m", *arguments], stdout=log, stderr=subprocess.STDOUT)
        status["stages"][stage].update(state="completed" if result.returncode == 0 else "blocked",
                                       exit_code=result.returncode, finished=time.time())
        save_json(args.output / "pipeline_status.json", status)
        if result.returncode:
            raise RuntimeError(f"{stage} failed; see {args.output / (stage + '.log')}")

    def run(stage, samples, benchmark, methods, extra=()):
        execute(stage, ["evaluation.degap.run", "--config", args.config, "--samples", str(samples),
                        "--output", str(args.output / stage), "--benchmark", benchmark,
                        "--methods", *methods, *extra])

    def score(stage):
        execute(stage + "_score", [
            "evaluation.degap.score_chair", "--run-dir", str(args.output / stage),
            "--annotations", str(args.annotations), "--evaluator", str(args.evaluator),
            "--nltk-data", str(args.nltk_data)])

    static = [f"fixed{b}" for b in (0, 5, 10, 15, 20, 25, 30)]
    run("calibration", args.calibration, "CHAIR", ["native", *static, "degap_token", "degap_step_mean"])
    score("calibration")
    scores = json.loads((args.output / "calibration/chair_metrics.json").read_text())["methods"]
    best = min(static, key=lambda name: (scores[name]["CHAIRs"], scores[name]["CHAIRi"],
                                        -scores[name]["Recall"], int(name[5:])))
    protocol = {
        "fixed_baseline": best,
        "selection": "calibration CHAIRs, then CHAIRi, then negative Recall, then earliest boundary",
        "calibration_sha256": sha256(args.calibration),
        "calibration_metrics_sha256": sha256(args.output / "calibration/chair_metrics.json"),
        "pope_sha256": sha256(args.pope), "chair_sha256": sha256(args.chair),
        "config_sha256": sha256(args.config),
        "main_alpha": 1.0, "main_lens": "native", "main_query_scope": "all",
        "second_stage_benchmarks": {
            "MME": "pending_dataset_and_full_VLMEvalKit_environment",
            "HallusionBench": "pending_dataset_and_full_VLMEvalKit_environment",
            "MMBench": "pending_dataset_and_full_VLMEvalKit_environment",
            "LLaVABench": "pending_dataset_and_judge_access",
            "MMVet": "pending_dataset_and_judge_access",
        },
    }
    target = args.output / "locked_protocol.json"
    if target.exists() and json.loads(target.read_text()) != protocol:
        raise ValueError("locked protocol differs")
    save_json(target, protocol)
    for alpha in (0.25, 0.5, 0.75):
        stage = f"calibration_alpha_{alpha}"
        run(stage, args.calibration, "CHAIR", ["degap_token"], ["--limit", "8", "--alpha", str(alpha)])
        score(stage)
    for label, arguments in (
        ("raw_lens", ["--lens", "raw_linear"]),
        ("nonvisual_query", ["--query-scope", "nonvisual"]),
    ):
        stage = "calibration_" + label
        run(stage, args.calibration, "CHAIR", ["degap_token"], ["--limit", "8", *arguments])
        score(stage)
    if args.calibration_only:
        return
    extra = ["--exclude-samples", str(args.calibration)]
    run("pope_full", args.pope, "POPE", ["native", best, "degap_token", "degap_step_mean"], extra)
    run("chair_full", args.chair, "CHAIR", ["native", best, "degap_token", "degap_step_mean"], extra)
    score("chair_full")
    run("pope_length2_diagnostic", args.pope, "POPE", ["native", "degap_token"], [
        "--limit", "60", "--max-new-tokens", "2", "--steps", "2", "--block-length", "2", *extra])


if __name__ == "__main__":
    main()
