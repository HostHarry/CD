"""Derive a POPE samples file with an explicit yes/no response instruction."""
import argparse
import json
from pathlib import Path

from evaluation.degap.prepare import write_samples
from evaluation.degap.runtime import sha256

SUFFIX = " Please answer yes or no."


def derive(source: Path, output: Path) -> dict:
    source_manifest = source.with_suffix(".manifest.json")
    if sha256(source) != json.loads(source_manifest.read_text())["samples_sha256"]:
        raise ValueError("source bytes do not match its manifest")
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    derived = []
    for row in rows:
        if "question" not in row or "messages" in row or row["question"].endswith(SUFFIX):
            raise ValueError("row must carry an unsuffixed question and no messages")
        derived.append({**row, "question": row["question"] + SUFFIX})
    write_samples(output, derived, {
        "dataset": "POPE-COCO", "derived_from": str(source),
        "parent_sha256": sha256(source), "parent_manifest_sha256": sha256(source_manifest),
        "prompt_suffix": SUFFIX,
        "transformation": "question + prompt_suffix; all other fields unchanged",
        "answer_parser": "official_no_not_no_else_yes",
    })
    return json.loads(output.with_suffix(".manifest.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/pope_coco.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(derive(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
