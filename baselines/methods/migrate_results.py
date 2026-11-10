"""One-time audited import from a previous run; no old project code is imported."""
import argparse
import ast
import fcntl
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

from common.run import input_key
from common.runtime import ROOT, save_json, sha256


def definitions(path):
    tree = ast.parse(path.read_text())
    result = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            result[node.name] = ast.dump(node, include_attributes=False)
            if isinstance(node, ast.ClassDef):
                for method in node.body:
                    if isinstance(method, ast.FunctionDef):
                        result[node.name + "." + method.name] = ast.dump(method, include_attributes=False)
    return result


def verify_snapshot(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["sources"].items():
        path = directory / "source" / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"historical source snapshot differs: {path}")
    return manifest


def verify_relocation_ast(previous):
    old = definitions(previous / "source/open_source_baselines.py")
    pairs = [
        ("common/decoding.py", name, name)
        for name in ("validate_generation", "checked_logits", "record_step")
    ] + [
        ("vcd/decode.py", "generate_vcd", "generate_vcd"),
        ("vcd/decode.py", "distorted_prompt", "BaselineRuntime.distorted_prompt"),
        ("vigor/decode.py", "generate_vigor", "generate_vigor"),
        ("vigor/decode.py", "_VigorObserver", "_VigorObserver"),
        ("vigor/decode.py", "_ObservedTorch", "_ObservedTorch"),
    ]
    checked = []
    for relative, target, source in pairs:
        if definitions(ROOT / relative)[target] != old[source]:
            raise ValueError(f"method algorithm changed during relocation: {relative}:{target}")
        checked.append(f"{relative}:{target}")
    old_runtime = definitions(previous / "source/evaluation/degap/runtime.py")
    new_runtime = definitions(ROOT / "common/runtime.py")
    for name in ("Runtime.__init__", "Runtime.prepare", "Runtime.decode"):
        if old_runtime[name] != new_runtime[name]:
            raise ValueError(f"input/model/output protocol changed: {name}")
        checked.append(name)
    pins = json.loads((ROOT / "configs/mmada-source.json").read_text())
    for name, digest in pins["files"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"official MMaDA file was modified: {name}")
    return checked


def validate_contract(source, target, method, sample_name):
    for key in ("upstream_revision", "environment", "generation", "expected_rows", "selected_rows",
                "limit", "benchmark", "checkpoints", "seed_protocol"):
        if source[key] != target[key]:
            raise ValueError(f"prediction reuse rejected: {key}")
    for key in ("checkpoint", "vq_checkpoint", "resolution", "resize", "seed"):
        if source["config"][key] != target["config"][key]:
            raise ValueError(f"prediction reuse rejected: {key}")
    if source["methods"][method] != target["methods"][method]:
        raise ValueError(f"method parameters changed: {method}")
    moves = json.loads((ROOT / "data/relocation.json").read_text())
    if (source["samples_sha256"] != moves[sample_name]["old_sha256"]
            or target["samples_sha256"] != moves[sample_name]["new_sha256"]
            or sha256(ROOT / "data" / sample_name) != moves[sample_name]["new_sha256"]):
        raise ValueError("dataset relocation does not match the locked source")
    calibration = moves["calibration.jsonl"]
    if (source["excluded_samples_sha256"] != calibration["old_sha256"]
            or target["excluded_samples_sha256"] != calibration["new_sha256"]):
        raise ValueError("calibration exclusion changed")
    # Model forward, VQ encoding, tokenizer framing, and image transforms must
    # match the prior sources. The native generator is now the unmodified,
    # pinned official source; its outputs/commits are checked on real inputs.
    for name in ("models/modeling_llada.py", "models/configuration_llada.py",
                 "models/modeling_magvitv2.py", "training/prompting_utils.py", "training/utils.py"):
        if source["sources"][name] != target["sources"][name]:
            raise ValueError(f"shared inference dependency changed: {name}")


def import_rows(source_dir, target_dir, method, keys):
    source = sqlite3.connect((source_dir / "predictions.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
    count, digest = 0, hashlib.sha256()
    try:
        with (target_dir / "run.lock").open("a") as lock, sqlite3.connect(target_dir / "predictions.sqlite") as target:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            seen = set()
            for key, payload in source.execute(
                    "SELECT input_key,payload FROM predictions WHERE method=? ORDER BY input_key", (method,)):
                if key not in keys:
                    raise ValueError("source contains unknown input")
                existing = target.execute(
                    "SELECT payload FROM predictions WHERE method=? AND input_key=?", (method, key)).fetchone()
                if existing and existing[0] != payload:
                    raise ValueError("destination has a different prediction")
                target.execute("INSERT OR IGNORE INTO predictions VALUES (?,?,?)", (method, key, payload))
                digest.update(json.dumps([method, key, payload], ensure_ascii=False).encode())
                seen.add(key)
                count += 1
            if method == "native" and seen != keys:
                raise ValueError("native source is incomplete")
    finally:
        source.close()
    return {"count": count, "ordered_rows_sha256": digest.hexdigest(),
            "reference": str(source_dir.resolve()), "manifest_sha256": sha256(source_dir / "manifest.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--native-reference", type=Path, required=True)
    args = parser.parse_args()
    proof = verify_relocation_ast(args.previous / "pope_full")
    report = {"unchanged_algorithm_and_protocol_definitions": proof, "imports": {}}
    for benchmark, sample_name, directory in (
        ("POPE", "pope_coco_yesno.jsonl", "pope_full"), ("CHAIR", "chair500.jsonl", "chair_full"),
    ):
        old = verify_snapshot(args.previous / directory)
        native = verify_snapshot(args.native_reference / directory)
        keys = {input_key(json.loads(line)) for line in (ROOT / "data" / sample_name).read_text().splitlines()}
        for method in ("vcd", "vigor"):
            output = ROOT / method / "results" / directory
            subprocess.run([
                sys.executable, "-m", "common.run", "--config", str(ROOT / method / "config.yaml"),
                "--samples", str(ROOT / "data" / sample_name), "--output", str(output),
                "--benchmark", benchmark, "--methods", "native", method,
                "--exclude-samples", str(ROOT / "data/calibration.jsonl"), "--score-only",
            ], check=True, cwd=ROOT)
            target = json.loads((output / "manifest.json").read_text())
            validate_contract(native, target, "native", sample_name)
            validate_contract(old, target, method, sample_name)
            audit = {
                "native": import_rows(args.native_reference / directory, output, "native", keys),
                method: import_rows(args.previous / directory, output, method, keys),
                "algorithm_proof": proof,
                "scope": "byte-for-byte prediction reuse; source paths and framework layout changed",
            }
            save_json(output / "migration.json", audit)
            report["imports"][f"{method}/{directory}"] = audit
    save_json(ROOT / "results/migration.json", report)
    print(json.dumps({name: {m: v["count"] for m, v in data.items() if isinstance(v, dict) and "count" in v}
                      for name, data in report["imports"].items()}, indent=2))


if __name__ == "__main__":
    main()
