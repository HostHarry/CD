"""Copy committed predictions into a new, explicitly validated early-EOS profile."""
import argparse
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

from common.run import input_key
from common.runtime import ROOT, save_json, sha256, source_hashes
from migrate_results import import_rows, verify_snapshot
from verify_eos import audit_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["vcd", "vigor"], required=True)
    parser.add_argument("--additional-run", type=Path)
    args = parser.parse_args()
    base = ROOT / args.method / "results"
    validation_path = base / "early_eos_validation.json"
    validation = json.loads(validation_path.read_text())
    if not validation["all_equal"] or validation["sources"] != source_hashes():
        raise ValueError("early EOS has not passed verification with the current source")
    changes = {"common/eos.py", "common/runtime.py", "common/run.py", "common/pipeline.py",
               "common/entrypoint.py", "multigpu.py", "verify_eos.py", "resume_eos.py"}
    for benchmark, sample_name, directory in (
        ("POPE", "pope_coco_yesno.jsonl", "pope_full"), ("CHAIR", "chair500.jsonl", "chair_full"),
    ):
        old = base / directory
        previous = verify_snapshot(old)
        if sha256(old / "manifest.json") != validation["historical_manifests"][directory]:
            raise ValueError("historical manifest changed after validation")
        for name, digest in source_hashes().items():
            if name not in changes and previous["sources"].get(name) != digest:
                raise ValueError(f"unverified algorithm/source change: {name}")
        output = base / "early_eos" / directory
        subprocess.run([
            sys.executable, "-m", "common.run", "--config", str(ROOT / args.method / "config.yaml"),
            "--samples", str(ROOT / "data" / sample_name), "--output", str(output),
            "--benchmark", benchmark, "--methods", "native", args.method,
            "--exclude-samples", str(ROOT / "data/calibration.jsonl"), "--score-only", "--early-eos",
        ], check=True, cwd=ROOT)
        target = json.loads((output / "manifest.json").read_text())
        target_contract = {k: v for k, v in target.items() if k not in {"sources", "early_eos"}}
        previous_contract = {k: v for k, v in previous.items() if k != "sources"}
        if target_contract != previous_contract:
            raise ValueError("input, generation, environment or checkpoint protocol changed")
        keys = {input_key(json.loads(line)) for line in (ROOT / "data" / sample_name).read_text().splitlines()}
        audit = {method: import_rows(old, output, method, keys) for method in ("native", args.method)}
        if args.additional_run:
            extra = args.additional_run.resolve() / directory
            extra_manifest = verify_snapshot(extra)
            extra_contract = {k: v for k, v in extra_manifest.items() if k not in {"sources", "early_eos"}}
            if extra_contract != target_contract or extra_manifest.get("early_eos") != "safe_eos_v1":
                raise ValueError("additional completed rows use a different protocol")
            for name, digest in source_hashes().items():
                if name not in changes and extra_manifest["sources"].get(name) != digest:
                    raise ValueError(f"additional run has an unverified algorithm: {name}")
            with sqlite3.connect((extra / "predictions.sqlite").as_uri() + "?mode=ro", uri=True) as db:
                for (payload,) in db.execute("SELECT payload FROM predictions"):
                    record = json.loads(payload)
                    if record.get("execution_profile") == "safe_eos_v1":
                        audit_record(record, validation["mask_id"], validation["eos_id"])
            audit["additional_run"] = {
                method: import_rows(extra, output, method, keys) for method in ("native", args.method)}
        audit.update(
            validation_sha256=sha256(validation_path),
            source_profile="full_schedule_v1", target_profile="safe_eos_v1",
            comparison_scope="exact answer tokens through EOS, not uncomputed suffix or full trace length",
        )
        save_json(output / "resume_import.json", audit)
        print(directory, {m: audit[m]["count"] for m in ("native", args.method)}, flush=True)


if __name__ == "__main__":
    main()
