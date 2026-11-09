"""Export existing MMMU answer text; no inference or external requests."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def main():
    index = read(ROOT / "reports/mmmu_runs.json")
    ids = read(ROOT / "manifests/mmmu_val_all_ids.json")
    assert len(ids) == len(set(ids)) == 900
    output = ROOT / "reports/mmmu_answers"
    output.mkdir(exist_ok=True)
    manifest = {
        "description": "Exact normalized.text from saved score files, including reasoning and final answer.",
        "normalization": "Response tokens before the first EOS, decoded with special tokens skipped and outer whitespace stripped by the original scorer.",
        "sample_count_per_method": len(ids),
        "run_index": index,
        "files": {},
    }
    for method in ("native", "cdapc_r90", "cdapc_r50"):
        records = {}
        for shard in range(3):
            run = ROOT / index[method][f"shard{shard}"]
            assert read(run / "run_manifest.json")["status"] == "complete"
            judge_dirs = [p for p in (run / "judge").iterdir() if p.is_dir()]
            assert len(judge_dirs) == 1
            for path in sorted((run / "predictions").glob("*.json")):
                prediction = read(path)
                score = read(run / "scores" / index[method][f"eval{shard}"] / path.name)
                judge = read(judge_dirs[0] / path.name)
                sid = prediction["sample_id"]
                assert prediction["status"] == score["generation_status"] == "success"
                assert sid == score["sample_id"] == judge["sample_id"] and sid not in records
                assert judge["score"] in (0, 1)
                records[sid] = {
                    "sample_id": sid, "method": method, "gold": score["gold"],
                    "text": score["normalized"]["text"],
                    "judge_correct": judge["score"], "official_correct": bool(score["official"]),
                    "strict_correct": int(score["strict_accuracy"]), "strict_status": score["strict_status"],
                    "decode_iterations": prediction["decode_iterations"],
                    "emitted_tokens_before_eos": prediction["emitted_tokens_before_eos"],
                    "finish_reason": prediction["finish_reason"],
                    "source_run": str(run.relative_to(ROOT)),
                }
        assert set(records) == set(ids)
        jsonl = output / f"{method}.jsonl"
        jsonl.write_text("".join(json.dumps(records[sid], ensure_ascii=False) + "\n" for sid in ids))
        exported = [json.loads(line) for line in jsonl.read_text().splitlines()]
        assert exported == [records[sid] for sid in ids]
        readable = []
        for row in exported:
            readable.append(
                f"# {row['sample_id']}\n"
                f"Gold: {row['gold']} | Judge: {row['judge_correct']} | "
                f"Official: {int(row['official_correct'])} | Strict: {row['strict_correct']}\n"
                f"Steps: {row['decode_iterations']} | Tokens before EOS: {row['emitted_tokens_before_eos']}\n\n"
                f"{row['text']}\n\n")
        textfile = output / f"{method}.txt"
        textfile.write_text("".join(readable))
        for path in (jsonl, textfile):
            manifest["files"][path.name] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
                "samples": len(exported),
            }
        print(method, len(exported), "answers;", textfile.stat().st_size, "text bytes")
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
