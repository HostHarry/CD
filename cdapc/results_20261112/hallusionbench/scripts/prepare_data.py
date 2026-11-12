"""Prepare HallusionBench (VLMEvalKit TSV) in the cdapc_lab format and match it to the CD baseline rows."""
import base64
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TSV = ROOT.parent / "hallusion-assets/HallusionBench.tsv"
TSV_MD5 = "0c23ac0dc9ef46832d7a24504f2a0c7c"
TSV_SHA256 = "8f92be804abf146f24bfa1a98d8fcb1cba335f6b0dc69fc27ea698fa446df3b8"
SCORER_HASHES = {"yorn.py": "692d72ee0c8fe64c7dd93b466f1b47c432303d8fba57503c56aae770b67ef59e",
                 "misc.py": "407ec5eaaf84227368365c6c7c93725166becfe755ed6a24eb93260d65464258"}
SCORER_REVISION = "3c192143007e4eb3f5c1a36558a4f8c1451980cb"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n")


def main():
    data = TSV.read_bytes()
    assert hashlib.md5(data).hexdigest() == TSV_MD5 and sha(data) == TSV_SHA256
    for name, expected in SCORER_HASHES.items():
        assert sha((ROOT / "evaluator" / name).read_bytes()) == expected, name
    out = ROOT / "data/hallusion"
    (out / "images").mkdir(parents=True, exist_ok=True)
    csv.field_size_limit(sys.maxsize)
    source = list(csv.DictReader(io.StringIO(data.decode()), delimiter="\t"))
    assert len(source) == 1129 and len({r["index"] for r in source}) == 1129
    samples, gold, rows = [], [], []
    for row in source:
        assert row["answer"] in ("Yes", "No")
        image_sha, paths = None, []
        if row["image"]:
            payload = base64.b64decode(row["image"], validate=True)
            image_sha = sha(payload)
            with Image.open(io.BytesIO(payload)) as image:
                fmt = image.format
                image.verify()
            suffix = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}[fmt]
            target = out / "images" / (image_sha + suffix)
            if not target.exists():
                target.write_bytes(payload)
            assert sha(target.read_bytes()) == image_sha
            paths = [f"images/{image_sha}{suffix}"]
        parts = row["index"].split("_")
        mode = "image" if paths else "text_only"
        samples.append({"sample_id": row["index"], "question": row["question"], "options": {},
                        "image_paths": paths, "task_id": "hallusion",
                        "group_id": f"{row['l2-category']}_{parts[3]}_{parts[4]}", "public_context": None})
        gold.append({"sample_id": row["index"], "answer": row["answer"],
                     "metric_metadata": {"category": row["category"], "l2-category": row["l2-category"],
                                         "input_mode": mode}})
        rows.append({"index": row["index"], "question": row["question"], "answer": row["answer"],
                     "category": row["category"], "l2-category": row["l2-category"],
                     "input_mode": mode, "image_sha256": image_sha})
    baseline = json.loads((ROOT / "baseline/native/answers.json").read_text())
    by_index = {r["index"]: r for r in baseline}
    assert set(by_index) == {r["index"] for r in rows}
    for r in rows:
        b = by_index[r["index"]]
        assert (b["question"], b["answer"], b["category"], b["l2-category"], b["input_mode"], b["image_sha256"]) == \
            (r["question"], r["answer"], r["category"], r["l2-category"], r["input_mode"], r["image_sha256"]), r["index"]
    write_json(out / "samples.json", samples)
    write_json(out / "gold.json", gold)
    write_json(out / "rows.json", rows)
    provenance = {"revision": TSV_SHA256, "split": "test", "repo": "VLMEvalKit HallusionBench.tsv",
                  "dataset_url": "https://opencompass.openxlab.space/utils/VLMEval/HallusionBench.tsv",
                  "tsv_md5": TSV_MD5, "tsv_sha256": TSV_SHA256, "rows": len(rows),
                  "input_modes": dict(Counter(r["input_mode"] for r in rows)),
                  "answers": dict(Counter(r["answer"] for r in rows)),
                  "categories": dict(Counter(r["category"] for r in rows)),
                  "scorer": {"repo": "open-compass/VLMEvalKit", "revision": SCORER_REVISION, "files": SCORER_HASHES},
                  "baseline_rows_matched": "all 1129 index/question/answer/category/l2-category/input_mode/image_sha256",
                  "baseline_commit": (ROOT / "baseline/COMMIT").read_text().strip(),
                  "preparer_sha256": sha(Path(__file__).read_bytes())}
    write_json(out / "provenance.json", provenance)
    ids = [r["index"] for r in rows]
    image_ids = [r["index"] for r in rows if r["input_mode"] == "image"]
    write_json(ROOT / "manifests/all_ids.json", ids)
    for k in range(3):
        write_json(ROOT / f"manifests/all_shard{k}.json", ids[k::3])
        write_json(ROOT / f"manifests/image_shard{k}.json", image_ids[k::3])
    print("PREPARED", provenance["rows"], provenance["input_modes"], provenance["answers"])


if __name__ == "__main__":
    main()
