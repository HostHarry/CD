"""Prepare MMMU-val data for MMaDA native + CD-APC runs (WINO-aligned)."""
import ast
import hashlib
import json
import sys
from io import BytesIO
from pathlib import Path
from string import ascii_uppercase

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cdapc_lab.io.artifacts import atomic_json, file_hash, source_hash
from cdapc_lab.data.mmmu import choose_image, mmmu_context, strip_image_placeholders
from huggingface_hub import HfApi

REPO = "lmms-lab/MMMU"
REVISION = "364f2e2eb107b36e07ff4c5a15f5947a759cef47"
PARQUET = f"data/validation-00000-of-00001.parquet"
LOCAL = Path("/root/autodl-fs/hf_home/hub/datasets--lmms-lab--MMMU/snapshots") / REVISION / PARQUET
MMMU_COMMIT = "51ce7f3e829c16bb44bc5445782686b4c3508794"


def extract_subset_name(sample_id):
    import re
    split = sample_id.split("_")[0]
    return re.search(rf"^{split}_(.+?)_\d+$", sample_id).group(1)


def image_ext(data):
    with Image.open(BytesIO(data)) as im:
        return {"PNG": "png", "JPEG": "jpg", "GIF": "gif", "WEBP": "webp"}.get(im.format, "bin")


def main():
    api = HfApi(endpoint="https://hf-mirror.com")
    info = api.dataset_info(REPO, revision=REVISION, files_metadata=True)
    expected = {s.rfilename: s.lfs.sha256 for s in info.siblings if s.lfs}
    assert PARQUET in expected, "parquet metadata missing"
    actual = file_hash(LOCAL)
    assert actual == expected[PARQUET], f"parquet sha256 mismatch {actual}"
    df = pd.read_parquet(LOCAL)
    assert len(df) == 900

    out = ROOT / "data/mmmu_val"
    (out / "images").mkdir(parents=True, exist_ok=True)
    samples, gold = [], []
    n_multi_image = n_no_placeholder = 0
    for _, row in df.iterrows():
        options_list = ast.literal_eval(row["options"]) if row["question_type"] == "multiple-choice" else []
        options = {ascii_uppercase[i]: str(o) for i, o in enumerate(options_list)}
        context_src = row["question"] + "\n" + "\n".join(f"{k}. {v}" for k, v in options.items())
        used = choose_image(context_src)
        if "<image" not in context_src:
            n_no_placeholder += 1
        images = [c for c in ("image_1", "image_2", "image_3", "image_4", "image_5", "image_6", "image_7")
                  if isinstance(row[c], dict) and row[c].get("bytes")]
        if len(images) > 1:
            n_multi_image += 1
        data = row[used]["bytes"] if isinstance(row[used], dict) else row["image_1"]["bytes"]
        sha = hashlib.sha256(data).hexdigest()
        rel = f"images/{sha}.{image_ext(data)}"
        path = out / rel
        if not path.is_file():
            path.write_bytes(data)
        samples.append({"sample_id": row["id"], "question": row["question"], "options": options,
                        "image_paths": (rel,), "task_id": "mmmu", "group_id": extract_subset_name(row["id"]),
                        "public_context": None})
        gold.append({"sample_id": row["id"], "answer": str(row["answer"]),
                     "metric_metadata": {"question_type": row["question_type"],
                                         "subject": extract_subset_name(row["id"]),
                                         "subfield": row["subfield"], "topic_difficulty": row["topic_difficulty"],
                                         "img_type": row["img_type"], "n_images": len(images),
                                         "used_image": used}})
    atomic_json(out / "samples.json", samples)
    atomic_json(out / "gold.json", gold)
    provenance = {"repo": REPO, "revision": REVISION, "split": "validation",
                  "parquet_sha256": actual, "parquet_sha256_matches_hf_metadata": True,
                  "rows": len(df), "question_types": {k: int(v) for k, v in df["question_type"].value_counts().items()},
                  "multi_image_rows": n_multi_image, "no_placeholder_rows": n_no_placeholder,
                  "image_policy": "first distinct <image n> placeholder in question+options (lmms-eval order), else image_1",
                  "prompt": "WINO mmmu_val_mmada construct_prompt with wrapper placeholder removal",
                  "scorer": {"repo": "https://github.com/MMMU-Benchmark/MMMU", "commit": MMMU_COMMIT,
                             "files": {"eval/eval_utils.py": file_hash(ROOT / "upstream/mmmu/eval/eval_utils.py"),
                                       "eval/data_utils.py": file_hash(ROOT / "upstream/mmmu/eval/data_utils.py")}}}
    atomic_json(out / "provenance.json", provenance)
    ids = [s["sample_id"] for s in samples]
    atomic_json(ROOT / "manifests/mmmu_val_all_ids.json", ids)
    for shard in range(3):
        atomic_json(ROOT / f"manifests/mmmu_val_shard{shard}_ids.json", ids[shard::3])
    smoke = sorted(ids, key=lambda i: hashlib.sha256(
        json.dumps(["mmmu_smoke_v1", i]).encode()).hexdigest())[:8]
    atomic_json(ROOT / "manifests/mmmu_val_smoke8_ids.json", smoke)
    lock_path = ROOT / "upstream.lock.json"
    lock = json.loads(lock_path.read_text())
    lock["sources"]["mmmu"] = {"url": "https://github.com/MMMU-Benchmark/MMMU",
                               "commit": MMMU_COMMIT,
                               "source_hash": source_hash(ROOT / "upstream/mmmu", ["."])}
    lock["mmmu_dataset"] = {"repo": REPO, "revision": REVISION, "files": {PARQUET: actual}}
    atomic_json(lock_path, lock)
    print(json.dumps({"rows": len(df), "question_types": {k: int(v) for k, v in df["question_type"].value_counts().items()},
                      "multi_image_rows": n_multi_image, "no_placeholder_rows": n_no_placeholder,
                      "parquet_sha256_match": True, "images": len(list((out / "images").iterdir())),
                      "manifests": {"all": len(ids), "smoke8": smoke}}, indent=2))


if __name__ == "__main__":
    main()
