"""Export the three completed COCO K4/L256 conditions without altering source runs."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import io
import json
from pathlib import Path
import shutil
import tarfile

from cdapc_lab.io.artifacts import atomic_json, file_hash, read_json, source_hash

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = (
    ("direct_eos_prefix", "cdapc-pope-eos-full", "pope_coco_direct_eos_prefix_r70_k4_L256_I128", "eos_prefix"),
    ("think_eos_prefix", "cdapc-pope-eos-full", "pope_coco_think_eos_prefix_r70_k4_L256_I128", "eos_prefix"),
    ("think_original_safe_eos", "cdapc-pope-safe-eos", "pope_coco_think_safe_eos_r70_k4_L256_I128", "original"),
)
DEPENDENCIES = ["types.py", "config.py", "models/base.py", "data", "io",
                "experiments/infer.py", "models/mmada.py", "decoding"]


def export(out):
    inventory = {}

    def copy(src, relative):
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        inventory[relative] = {"source": str(src), "sha256": file_hash(target)}

    rows_source = ROOT.parent / "cdapc-pope-safe-eos/data/pope_yesno/rows.json"
    rows = read_json(rows_source)
    assert len(rows) == 9000 and len({r["row_id"] for r in rows}) == 9000
    expected_ids = {r["input_key"] for r in rows}
    assert len(expected_ids) == 5127
    copy(rows_source, "rows.json")
    sources = {}
    for name, workspace, _, source in CONDITIONS:
        if source in sources:
            continue
        root = ROOT.parent / workspace
        for path in sorted((root / "src").rglob("*.py")):
            copy(path, f"source/{source}/" + str(path.relative_to(root)))
        for relative in ("pyproject.toml", "upstream.lock.json"):
            copy(root / relative, f"source/{source}/{relative}")
        for filename in ("samples.json", "gold.json", "provenance.json", "rows.json"):
            copy(root / "data/pope_yesno" / filename, f"source/{source}/data/pope_yesno/{filename}")
        sources[source] = source_hash(root / "src/cdapc_lab", DEPENDENCIES)
        assert sources[source] == source_hash(out / f"source/{source}/src/cdapc_lab", DEPENDENCIES)
    conditions = {}
    for name, workspace, stage, source in CONDITIONS:
        root = ROOT.parent / workspace
        report_path = root / "reports" / f"{stage}_after_think_official.json"
        report = read_json(report_path)
        judged = read_json(root / "reports/judge" / f"{stage}.json")
        assert report["unique_inputs"] == 5127 and report["evaluation_rows"] == 9000
        assert set(report["per_input"]) == expected_ids == set(judged["per_input"])
        assert judged["status"] == "complete" and judged["api_error_inputs"] == 0
        assert report["rows_sha256"] == file_hash(rows_source)
        for suffix in (".json", ".md"):
            copy(report_path.with_suffix(suffix), f"{name}/official{suffix}")
        copy(root / "reports" / f"{stage}.json", f"{name}/generation_report.json")
        copy(root / "reports/judge" / f"{stage}.json", f"{name}/judge.json")
        entries, answers, run_ids, statuses = {}, [], set(), Counter()
        items = sorted(report["per_input"].items())
        target = out / name
        (target / "records").mkdir(exist_ok=True)
        for offset in range(0, len(items), 256):
            archive_path = target / "records" / f"part-{offset // 256:03d}.tar.gz"
            with archive_path.open("wb") as raw_stream:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw_stream, mtime=0, compresslevel=6) as compressed:
                    with tarfile.open(fileobj=compressed, mode="w|") as archive:
                        for sid, record in items[offset:offset + 256]:
                            score_path = Path(record["source_score"])
                            assert file_hash(score_path) == record["source_score_sha256"]
                            score = read_json(score_path)
                            run = score_path.parents[2]
                            pred_path = run / "predictions" / score_path.name
                            pred = read_json(pred_path)
                            manifest = read_json(run / "run_manifest.json")
                            cfg = read_json(run / "resolved_config.json")
                            assert manifest["status"] in ("complete", "incomplete")
                            assert set(manifest["sample_status_counts"]) <= {"success", "budget_exhausted"}
                            assert sum(manifest["sample_status_counts"].values()) == manifest["sample_count"]
                            assert manifest["status"] == (
                                "incomplete" if manifest["sample_status_counts"].get("budget_exhausted") else "complete")
                            assert manifest["generation_source_hash"] == sources[source]
                            assert pred["sample_id"] == sid == score["sample_id"]
                            assert pred["generation_fingerprint"] == manifest["generation_fingerprint"]
                            assert pred["status"] == record["generation_status"] == score["generation_status"]
                            assert cfg["protocol"]["canvas_length"] == 256
                            assert cfg["cdapc"]["iteration_limit"] == 128
                            assert cfg["cdapc"]["max_commit_per_iteration"] == 4
                            statuses[pred["status"]] += 1
                            if run.name not in run_ids:
                                run_ids.add(run.name)
                                ids = cfg["run"]["sample_ids_file"]
                                copy(root / ids, f"source/{source}/{ids}")
                                yaml = f"configs/experiments/{cfg['run']['name']}.yaml"
                                copy(root / yaml, f"source/{source}/{yaml}")
                                for filename in ("run_manifest.json", "resolved_config.json", "dataset_manifest.json"):
                                    copy(run / filename, f"{name}/runs/{run.name}/{filename}")
                                copy(score_path.parent / "evaluation_manifest.json",
                                     f"{name}/runs/{run.name}/evaluation_manifest.json")
                            members = {}
                            for kind, path in (("prediction", pred_path), ("score", score_path)):
                                content = path.read_bytes()
                                member_name = f"{run.name}/{kind}/{path.name}"
                                member = tarfile.TarInfo(member_name)
                                member.size, member.mode, member.mtime = len(content), 0o644, 0
                                archive.addfile(member, io.BytesIO(content))
                                members[kind] = {"member": member_name, "sha256": file_hash(path),
                                                 "source_path": str(path)}
                            entries[sid] = {"archive": str(archive_path.relative_to(out)),
                                            "run": run.name, **members}
                            answers.append({
                                "input_key": sid, "gold": record["gold"],
                                "text": score["normalized"]["text"], "answer_region": record["selected_text"],
                                "official_prediction": record["official_prediction"],
                                "effective_prediction": record["effective_prediction"],
                                "generation_status": pred["status"], "finish_reason": pred["finish_reason"],
                                "decode_iterations": pred["decode_iterations"],
                                "generation_seconds": pred["generation_seconds"],
                                "judge_prediction": judged["per_input"][sid]["judge_prediction"],
                                "judge_status": judged["per_input"][sid]["judge_status"],
                            })
            print(name, archive_path.name, archive_path.stat().st_size, flush=True)
        with (target / "answers.jsonl").open("w") as stream:
            for answer in answers:
                stream.write(json.dumps(answer, ensure_ascii=False, sort_keys=True) + "\n")
        with (target / "answers.txt").open("w") as stream:
            for i, answer in enumerate(answers):
                if i:
                    stream.write("\n")
                stream.write(f"[{answer['input_key']}] gold={answer['gold']} status={answer['generation_status']}\n"
                             + answer["text"] + "\n")
        atomic_json(target / "records_index.json", entries)
        assert dict(statuses) == report["generation_statuses"]
        conditions[name] = {"stage": stage, "source_snapshot": source,
                            "generation_source_hash": sources[source], "runs": sorted(run_ids),
                            "unique_inputs": len(entries), "evaluation_rows": len(rows),
                            "generation_statuses": dict(statuses),
                            "official_macro": report["macro"], "judge_macro": judged["macro"]}
    reports = (
        "pope_coco_safe_vs_prefix_full_official.json",
        "pope_coco_quality_diagnosis.json", "pope_coco_quality_diagnosis.md",
        "pope_runtime_audit.json", "pope_runtime_audit.md",
        "cd_baselines_63b9d74_comparison.json", "cd_baselines_63b9d74_comparison.md",
    )
    for filename in reports:
        copy(ROOT / "reports" / filename, f"reports/{filename}")
    for filename in ("export_pope_coco_k4.py", "compare_pope_coco_safe_eos.py",
                     "diagnose_pope_coco_quality.py", "audit_pope_runtime.py",
                     "score_pope_answer_region.py", "pope_after_think_official.py"):
        copy(ROOT / "scripts" / filename, f"analysis_scripts/{filename}")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scope": "Three completed COCO CDAPC K4/L256 conditions; no A-OKVQA/GQA run outputs",
        "conditions": conditions, "source_dependencies": DEPENDENCIES,
        "copied_file_provenance": inventory,
        "raw_payloads": "Prediction and score files preserved byte-for-byte inside deterministic tar.gz parts.",
        "run_completion": "Original run manifests mark budget-exhausted runs incomplete. All planned inputs have "
                          "terminal records and all 9000 evaluation rows are scored; these manifests are not rewritten.",
        "paths": "Absolute source paths are historical provenance; archive/member paths are portable.",
        "exclusions": ["model weights", "images", "Judge HTTP request/response caches", "credentials",
                       "third-party author source", "intermediate inference snapshots"],
    }
    atomic_json(out / "manifest.json", manifest)
    print("EXPORTED", json.dumps(conditions, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.output.is_dir()
    export(args.output)
