"""Fetch and pin POPE inputs aligned with the HostHarry/CD native MMaDA protocol."""
from collections import Counter
from io import BytesIO
from pathlib import Path
import hashlib
import json
import re
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cdapc_lab.io.artifacts import atomic_json, file_hash, source_hash
from cdapc_lab.data.pope import SUFFIX, pope_input_key
from huggingface_hub import hf_hub_download

REPO = "lmms-lab/POPE"
REVISION = "4db1276663dfa5eb8ad16a52d24c31a09e470896"
CACHE = "/root/autodl-fs/hf_home/hub"
FILES = {"data/test-00000-of-00003.parquet": "38851606d0e9f473c69dd45e3b018286842070473f168f7fa1c82e4c5086f116",
         "data/test-00001-of-00003.parquet": "6800d7e0d43c7f7eb3e560faee8dd247ac35003f2d43955a29f66ef0dfb2aaa2",
         "data/test-00002-of-00003.parquet": "f973337fbd2d114aaeeb9b96499b554df17d46654be3ff25aa14d3282c2a9d9a"}
RUCAIBOX_URL = "https://github.com/RUCAIBox/POPE"
RUCAIBOX_COMMIT = "08d957b917e5a378a2f99d35b6293c536a66298b"
SPLIT_ORDER = ("random", "popular", "adversarial")
RUCAIBOX_FILES = ["evaluate.py"] + [f"output/coco/coco_pope_{c}.json" for c in ("adversarial", "popular", "random")]
HOSTHARRY = Path("/root/autodl-tmp/HostHarry-CD")


def fetch_rucaibox():
    upstream = ROOT / "upstream/pope"
    hashes = {}
    for relative in RUCAIBOX_FILES:
        url = f"https://raw.githubusercontent.com/RUCAIBox/POPE/{RUCAIBOX_COMMIT}/{relative}"
        target = upstream / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as response:
            target.write_bytes(response.read())
        hashes[relative] = file_hash(target)
    return hashes


def load_lmms_rows():
    rows = []
    source_hashes = {}
    for relative, expected in FILES.items():
        path = Path(hf_hub_download(REPO, relative, repo_type="dataset", revision=REVISION, cache_dir=CACHE))
        actual = file_hash(path)
        if actual != expected:
            raise ValueError(f"Official dataset hash mismatch: {path} ({actual})")
        source_hashes[relative] = actual
    import pyarrow.parquet as pq
    for relative in FILES:
        path = Path(hf_hub_download(REPO, relative, repo_type="dataset", revision=REVISION, cache_dir=CACHE))
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            rows.extend(batch.to_pylist())
    if len(rows) != 9000:
        raise ValueError(f"Expected 9000 POPE rows, got {len(rows)}")
    return rows, source_hashes


def main():
    from PIL import Image
    rucaibox_hashes = fetch_rucaibox()
    manifest = json.loads((HOSTHARRY / "protocol/pope_coco.manifest.json").read_text())
    for split, expected in manifest["questions_sha256"].items():
        actual = rucaibox_hashes[f"output/coco/coco_pope_{split}.json"]
        if actual != expected:
            raise ValueError(f"RUCAIBox {split} hash differs from HostHarry manifest: {actual}")
    lmms_rows, source_hashes = load_lmms_rows()
    image_bytes = {}
    for row in lmms_rows:
        source = row["image_source"]
        name = source if source.endswith(".jpg") else source + ".jpg"
        payload = row["image"]
        if not isinstance(payload, dict) or not payload.get("bytes"):
            raise ValueError("Expected embedded image bytes")
        if name in image_bytes and image_bytes[name] != payload["bytes"]:
            raise ValueError(f"Conflicting bytes for image {name}")
        image_bytes[name] = payload["bytes"]
    tsv = {}
    for line in (HOSTHARRY / "protocol/images.tsv").read_text().splitlines()[1:]:
        if line.strip():
            name, sha = line.split("\t")
            tsv[name] = sha
    official = {split: [json.loads(line) for line in
                        (ROOT / f"upstream/pope/output/coco/coco_pope_{split}.json").read_text().splitlines() if line.strip()]
                for split in SPLIT_ORDER}
    for split, values in official.items():
        if len(values) != 3000 or {v["question_id"] for v in values} != set(range(1, 3001)):
            raise ValueError(f"invalid official POPE IDs: {split}")
        if sum(v["label"] == "yes" for v in values) != 1500 or any(v["label"] not in ("yes", "no") for v in values):
            raise ValueError(f"invalid POPE labels: {split}")
    rows, image_sha, typo_counts = [], {}, {}
    lmms_check = {}
    for split in SPLIT_ORDER:
        typo_counts[split] = 0
        ours = Counter()
        typo_counts[split] = sum(re.sub(r"\bimange\b", "image", r["question"].strip()) != r["question"].strip()
                                 for r in lmms_rows if r["category"] == split)
        lmms_by_key = {(r["image_source"], re.sub(r"\bimange\b", "image", r["question"].strip()),
                        str(r["answer"]).lower()) for r in lmms_rows if r["category"] == split}
        for entry in official[split]:
            name = entry["image"]
            if name not in image_bytes:
                raise ValueError(f"POPE image absent from lmms-lab parquet: {name}")
            if name not in tsv:
                raise ValueError(f"POPE image absent from images.tsv: {name}")
            sha = hashlib.sha256(image_bytes[name]).hexdigest()
            if sha != tsv[name]:
                raise ValueError(f"Image sha256 differs from images.tsv: {name}")
            image_sha[name] = sha
            question = entry["text"]
            key = (Path(name).stem, question.strip(), entry["label"].lower())
            ours[key] += 1
            rows.append({"row_id": f"{split}:{entry['question_id']}", "split": split,
                         "question_id": entry["question_id"], "image": name, "image_sha256": sha,
                         "question": question, "label": entry["label"].lower(),
                         "input_key": pope_input_key(sha, question + SUFFIX)})
        lmms_check[split] = {"entries": sum(ours.values()),
                             "equal": all(k in lmms_by_key for k in ours)}
        missing = [k for k in ours if k not in lmms_by_key][:5]
        if missing:
            raise ValueError(f"lmms-lab cross-check failed for {split}: missing={missing}")
    lmms_check["typo_normalized_rows"] = typo_counts
    lmms_check["typo_rule"] = "imange->image, cross-check only; prompts keep RUCAIBox text verbatim"
    if len({r["row_id"] for r in rows}) != 9000:
        raise ValueError("Duplicate POPE row IDs")
    row_ids_by_key, labels = {}, {}
    for row in rows:
        key = row["input_key"]
        if key in labels and labels[key] != row["label"]:
            raise ValueError(f"Conflicting labels for input {key}")
        labels[key] = row["label"]
        row_ids_by_key.setdefault(key, []).append(row["row_id"])
    unique = []
    seen = set()
    for row in rows:
        key = row["input_key"]
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    if len(unique) != 5127:
        raise ValueError(f"Expected 5127 unique inputs, got {len(unique)}")
    output = ROOT / "data/pope_yesno"
    (output / "images").mkdir(parents=True, exist_ok=True)
    names = {}
    for row in unique:
        name = row["image"]
        if name not in names:
            payload = image_bytes[name]
            with Image.open(BytesIO(payload)) as image:
                extension = {"JPEG": ".jpg", "PNG": ".png"}.get(image.format, f".{image.format.lower()}")
                image.verify()
            names[name] = "images/" + hashlib.sha256(payload).hexdigest() + extension
            (output / names[name]).write_bytes(payload)
    samples = [dict(sample_id=row["input_key"], question=row["question"], options={},
                    image_paths=[names[row["image"]]], task_id="pope",
                    group_id=row["image"], public_context=None) for row in unique]
    gold = [dict(sample_id=row["input_key"], answer=labels[row["input_key"]],
                 metric_metadata={"image": row["image"], "image_sha256": row["image_sha256"],
                                  "row_ids": row_ids_by_key[row["input_key"]]}) for row in unique]
    atomic_json(output / "samples.json", samples)
    atomic_json(output / "gold.json", gold)
    atomic_json(output / "rows.json", rows)
    provenance = dict(repo=REPO, revision=REVISION, split="test", source_files=source_hashes,
                      source_count=len(rows), selected_count=len(unique),
                      unique_inputs=len(unique), unique_images=len(names),
                      row_order="RUCAIBox file order: random, popular, adversarial",
                      questions="RUCAIBox text verbatim; prompt suffix appended by the task template",
                      prompt_suffix=SUFFIX,
                      rucaibox={"repo": RUCAIBOX_URL, "commit": RUCAIBOX_COMMIT, "files": rucaibox_hashes,
                                "questions_sha256_match": manifest["questions_sha256"]},
                      hostharry={"path": str(HOSTHARRY), "images_tsv_entries_checked": len(image_sha),
                                 "images_tsv_sha256_match": True},
                      lmms_vs_rucaibox=lmms_check,
                      selection="all rows; deduplicated by input_key in first-appearance order")
    atomic_json(output / "provenance.json", provenance)
    all_ids = [row["input_key"] for row in unique]
    atomic_json(ROOT / "manifests/pope_yesno_all_ids.json", all_ids)
    for shard in range(3):
        atomic_json(ROOT / f"manifests/pope_yesno_shard{shard}_ids.json", all_ids[shard::3])
    atomic_json(ROOT / "manifests/pope_yesno_parity16_ids.json", all_ids[:16])
    lock_path = ROOT / "upstream.lock.json"
    lock = json.loads(lock_path.read_text())
    lock["sources"]["pope"] = {"url": RUCAIBOX_URL, "commit": RUCAIBOX_COMMIT,
                               "source_hash": source_hash(ROOT / "upstream/pope", ["."])}
    lock["pope_dataset"] = {"repo": REPO, "revision": REVISION, "files": dict(FILES)}
    atomic_json(lock_path, lock)
    print(json.dumps({"rows": len(rows), "unique_inputs": len(unique), "unique_images": len(names),
                      "per_split": {s: sum(r["split"] == s for r in rows) for s in SPLIT_ORDER},
                      "typo_normalized_rows": typo_counts,
                      "images_tsv_match": True, "questions_sha256_match": True}, indent=2), flush=True)
    for row in rows[:3]:
        print(json.dumps({"row_id": row["row_id"], "question": row["question"], "label": row["label"],
                          "input_key": row["input_key"][:12]}), flush=True)


if __name__ == "__main__":
    main()
