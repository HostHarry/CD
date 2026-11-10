"""Independent method entry point; execute modules from the project root."""
import argparse
from pathlib import Path
import subprocess
import sys

from .runtime import ROOT


def main(method):
    parser = argparse.ArgumentParser(description=f"Run the independent {method} baseline")
    parser.add_argument("--benchmark", choices=["POPE", "CHAIR"], default="POPE")
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--early-eos", action="store_true")
    args = parser.parse_args()
    directory = "pope_full" if args.benchmark == "POPE" else "chair_full"
    base = ROOT / method / "results"
    if args.early_eos:
        base /= "early_eos"
    output = args.output or base / directory
    samples = ROOT / "data" / ("pope_coco_yesno.jsonl" if args.benchmark == "POPE" else "chair500.jsonl")
    command = [sys.executable, "-u"]
    if args.score_only:
        command += ["-m", "common.run", "--output", str(output), "--score-only", "--methods", "native", method]
    else:
        command += [str(ROOT / "multigpu.py"), "--run-dir", str(output), "--method", method, "--gpus", *args.gpus]
    command += ["--config", str(ROOT / method / "config.yaml"), "--samples", str(samples),
                "--benchmark", args.benchmark, "--exclude-samples", str(ROOT / "data/calibration.jsonl")]
    if args.early_eos:
        command += ["--early-eos"]
    subprocess.run(command, check=True, cwd=ROOT)
