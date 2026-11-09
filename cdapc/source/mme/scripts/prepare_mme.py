"""Freeze the public MME data and original scorer before queueing inference."""
import copy
import hashlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
import shutil
import sys
import time
import zipfile

import pyarrow.parquet as pq
from PIL import Image
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / "cdapc-experiments"
sys.path.insert(0, str(ROOT / "src"))
from cdapc_lab.config import resolve_config
from cdapc_lab.io.artifacts import atomic_json, file_hash, read_json, source_hash

REVISION = "d6c9023f017b564f7b3ccccf5348166bce8fdbcd"
SCORER_COMMIT = "dd2950902889cd614d4edf606827240166d29381"
SCORER_BLOB = "e02681d44e5a5e7f81023e50d3ce4cfce34c07d0"
FILES = {
    "test-00000-of-00004-a25dbe3b44c4fda6.parquet": "1563fd59c8049dbbabe211a049c38b2e20a6d58da94a75c90cfc937d1f6c52e3",
    "test-00001-of-00004-7d22c7f1aba6fca4.parquet": "3623728c0c98807cc7eef9cbd10dfa18c21654321c4b02b7b19bc8e6fc86d830",
    "test-00002-of-00004-594798fd3f5b029c.parquet": "c169d85bdf84e21b005170698217e96380bef172e8bd7432fc75ddce51dcaabe",
    "test-00003-of-00004-53ae1794f93b1e35.parquet": "aec84886b2bb6dd7bffc94eb3bbca9319e0671f3ee989a08e0758bc2de6c67f6",
}


def fetch_large_ranges(url, temporary, size):
    """Use bounded parallel ranges, retaining a previously downloaded prefix."""
    offset = temporary.stat().st_size if temporary.exists() else 0
    if offset > size:
        raise ValueError("Partial dataset exceeds pinned size")
    chunks = [(start, min(start + 32 * 1024 * 1024, size) - 1)
              for start in range(offset, size, 32 * 1024 * 1024)]

    def fetch(bounds):
        start, end = bounds
        part = temporary.with_name(f"{temporary.name}.range.{start}.{end}")
        if part.exists() and part.stat().st_size == end - start + 1:
            return part
        for attempt in range(4):
            try:
                with requests.get(url, headers={"Range": f"bytes={start}-{end}"},
                                  stream=True, timeout=(20, 120)) as response:
                    response.raise_for_status()
                    if (response.status_code != 206
                            or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}"):
                        raise ValueError("Server returned a different dataset range")
                    with part.open("wb") as handle:
                        for chunk in response.iter_content(1024 * 1024):
                            handle.write(chunk)
                if part.stat().st_size != end - start + 1:
                    raise ValueError("Truncated dataset range")
                print("RANGE VERIFIED LENGTH", start, end, flush=True)
                return part
            except (requests.RequestException, ValueError):
                if attempt == 3:
                    raise
                time.sleep(5)

    with ThreadPoolExecutor(max_workers=6) as pool:
        parts = list(pool.map(fetch, chunks))
    with temporary.open("ab") as handle:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, handle)
    for part in parts:
        part.unlink()


def fetch_parquet(item):
    name, expected = item
    path = ROOT / "data/mme/downloads" / name
    if path.exists() and file_hash(path) == expected:
        return path
    temporary = path.with_suffix(".partial")
    url = f"https://hf-mirror.com/datasets/lmms-lab/MME/resolve/{REVISION}/data/{name}"
    if name == "test-00002-of-00004-594798fd3f5b029c.parquet":
        fetch_large_ranges(url, temporary, 694888210)
        if file_hash(temporary) != expected:
            raise ValueError(f"Dataset SHA256 mismatch: {name}")
        temporary.replace(path)
        print("VERIFIED", name, path.stat().st_size, flush=True)
        return path
    for attempt in range(4):
        try:
            offset = temporary.stat().st_size if temporary.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with requests.get(url, headers=headers, stream=True, timeout=(20, 120)) as response:
                response.raise_for_status()
                append = offset and response.status_code == 206
                if append:
                    assert response.headers["Content-Range"].startswith(f"bytes {offset}-")
                with temporary.open("ab" if append else "wb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        handle.write(chunk)
            if file_hash(temporary) != expected:
                temporary.unlink()
                raise ValueError(f"Dataset SHA256 mismatch: {name}")
            temporary.replace(path)
            print("VERIFIED", name, path.stat().st_size, flush=True)
            return path
        except (requests.RequestException, ValueError):
            if attempt == 3:
                raise
            time.sleep(5)


def official_questions():
    archive = ROOT / "upstream/mme/eval_tool.zip"
    raw = archive.read_bytes()
    assert hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == SCORER_BLOB
    records = {}
    with zipfile.ZipFile(BytesIO(raw)) as z:
        for filename in ("calculation.py", "readme.txt"):
            (archive.parent / filename).write_bytes(z.read("eval_tool/" + filename))
        for name in sorted(z.namelist()):
            if not name.startswith("eval_tool/Your_Results/") or not name.endswith(".txt"):
                continue
            category = Path(name).stem
            lines = z.read(name).decode("utf-8").splitlines()
            assert len(lines) % 2 == 0
            for index, line in enumerate(lines):
                image, question, answer, *_ = line.split("\t")
                key = (category, Path(image).name, question.strip())
                assert key not in records
                records[key] = {"answer": answer.lower(), "pair_index": index % 2}
    return records


def validate_pairs(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    for key, pair in groups.items():
        if (len(pair) != 2 or {r["answer"] for r in pair} != {"yes", "no"}
                or len({r["question"] for r in pair}) != 2
                or len({r["image_sha256"] for r in pair}) != 1):
            raise ValueError(f"Invalid MME question pair: {key}")
    return groups


def main():
    official = official_questions()
    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = list(pool.map(fetch_parquet, FILES.items()))
    image_root = ROOT / "data/mme/images"
    image_root.mkdir(exist_ok=True)
    rows = []
    seen = set()
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
            for record in batch.to_pylist():
                category = record["category"]
                image_name = Path(record["question_id"]).name
                question = record["question"].strip()
                key = (category, image_name, question)
                assert key in official and key not in seen, key
                assert record["answer"].lower() == official[key]["answer"], key
                assert question.endswith("Please answer yes or no."), key
                seen.add(key)
                image_bytes = record["image"]["bytes"]
                sha = hashlib.sha256(image_bytes).hexdigest()
                with Image.open(BytesIO(image_bytes)) as image:
                    extension = {"PNG": "png", "JPEG": "jpg"}.get(image.format, image.format.lower())
                    image.verify()
                relative = f"images/{sha}.{extension}"
                target = ROOT / "data/mme" / relative
                if not target.exists():
                    target.write_bytes(image_bytes)
                else:
                    assert file_hash(target) == sha, target
                group = f"{category}/{image_name}"
                rows.append({
                    "sample_id": f"mme:{group}:{official[key]['pair_index']}",
                    "group_id": group, "category": category, "image_name": image_name,
                    "question": question, "answer": official[key]["answer"],
                    "image_sha256": sha, "image_path": relative,
                })
    assert seen == set(official), (len(seen), len(official))
    assert len(rows) == 2374 and len({r["category"] for r in rows}) == 14
    groups = validate_pairs(rows)
    assert len(groups) == 1187
    rows.sort(key=lambda r: r["sample_id"])
    samples = [{"sample_id": r["sample_id"], "question": r["question"], "options": {},
                "image_paths": [r["image_path"]], "task_id": "mme",
                "group_id": r["group_id"], "public_context": None} for r in rows]
    gold = [{"sample_id": r["sample_id"], "answer": r["answer"],
             "metric_metadata": {"category": r["category"], "image": r["group_id"]}} for r in rows]
    out = ROOT / "data/mme"
    for name, payload in (("samples", samples), ("gold", gold), ("rows", rows)):
        atomic_json(out / f"{name}.json", payload)
    provenance = {
        "repo": "lmms-lab/MME", "revision": REVISION, "split": "test",
        "files": FILES, "sha256_verified": True, "rows": len(rows), "image_pairs": len(groups),
        "categories": dict(Counter(r["category"] for r in rows)),
        "official_questions_and_labels_verified": True,
        "prompt": "Original question including Please answer yes or no.; no additional reasoning instruction",
        "images": "Original bytes, including MPO; existing PIL/MMaDA preprocessing uses the first frame",
        "scorer_commit": SCORER_COMMIT, "scorer_archive_git_blob": SCORER_BLOB,
        "scorer_sha256": file_hash(ROOT / "upstream/mme/calculation.py"),
    }
    atomic_json(out / "provenance.json", provenance)
    ids = [s["sample_id"] for s in samples]
    atomic_json(ROOT / "manifests/mme_all_ids.json", ids)
    ordered_groups = sorted(groups, key=lambda g: hashlib.sha256(("mme_shard_v1:" + g).encode()).hexdigest())
    source = yaml.safe_load((BASE / "configs/experiments/mmada_mmmu_cdapc_r70_L256_I128_shard0.yaml").read_text())
    shard_counts = []
    for shard in range(3):
        selected = {g for g in ordered_groups[shard::3]}
        shard_ids = [r["sample_id"] for r in rows if r["group_id"] in selected]
        manifest = f"manifests/mme_shard{shard}_ids.json"
        atomic_json(ROOT / manifest, shard_ids)
        config = copy.deepcopy(source)
        config["run"].update(name=f"mmada_mme_cdapc_r70_L256_I128_shard{shard}",
                             sample_ids_file=manifest, fail_on_error=True)
        config["protocol"].update(
            task="mme", dataset_revision=REVISION, data_root="data/mme", split="test",
            prompt_id="mme_yesno_v1", strict_parser_id="mme_official_v1",
            official_scorer_revision=SCORER_COMMIT)
        path = ROOT / f"configs/experiments/{config['run']['name']}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        resolve_config(path)
        shard_counts.append(len(shard_ids))
    lock = read_json(ROOT / "upstream.lock.json")
    lock["sources"]["mme"] = {
        "url": "https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models",
        "commit": SCORER_COMMIT, "source_hash": source_hash(ROOT / "upstream/mme", ["."])}
    lock["mme_dataset"] = provenance
    atomic_json(ROOT / "upstream.lock.json", lock)
    print("PREPARED", len(rows), "questions", len(groups), "image pairs", shard_counts, flush=True)


if __name__ == "__main__":
    main()
