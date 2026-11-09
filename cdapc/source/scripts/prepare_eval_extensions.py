"""Prepare pinned CHAIR-500 and official POPE-SEEM AOKVQA/GQA inputs."""
import argparse
import base64
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from io import BytesIO
import json
from pathlib import Path
import time

from PIL import Image
import requests
import yaml

from cdapc_lab.data.pope import pope_input_key, SUFFIX
from cdapc_lab.io.artifacts import atomic_json, file_hash, source_hash

ROOT = Path(__file__).resolve().parents[1]
EXPORT = Path("/root/autodl-tmp/cd-mmmu-upload")
POPE_REV = "08d957b917e5a378a2f99d35b6293c536a66298b"
CHAIR_REV = "4087a26211aa2339b9a76307cb8f0321ef691d0a"
CHAIR_HASH = "98728cd2b7fb8b97420b5f3bf14789dcad1d7d980de9c089b82856217387770d"
SPLITS = ("random", "popular", "adversarial")


def get_bytes(urls, expected=None):
    error = None
    for attempt in range(4):
        for url in urls:
            try:
                r = requests.get(url, timeout=(15, 50))
                r.raise_for_status()
                if expected and hashlib.sha256(r.content).hexdigest() != expected:
                    raise ValueError(f"Hash mismatch: {url}")
                return r.content, url
            except (requests.RequestException, ValueError) as exc:
                error = exc
        time.sleep(attempt + 1)
    raise RuntimeError(f"Could not fetch {urls}: {error}")


def github_file(repo, path, revision):
    url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={revision}"
    payload, _ = get_bytes([url])
    record = json.loads(payload)
    if record.get("content"):
        return base64.b64decode(record["content"])
    return get_bytes([f"https://raw.githubusercontent.com/{repo}/{revision}/{path}"])[0]


def image_file(name, folder, expected=None, gqa=False):
    path = folder / name
    if path.exists():
        with Image.open(path) as image:
            image.verify()
        sha = file_hash(path)
        if expected and sha != expected:
            raise ValueError(f"Cached image hash mismatch: {path}")
        return name, path, sha, "cached"
    if gqa:
        urls = [f"https://cs.stanford.edu/people/rak248/{part}/{name}" for part in ("VG_100K", "VG_100K_2")]
    else:
        split = "val2014" if "val2014" in name else "train2014" if "train2014" in name else "val2017"
        urls = [f"https://s3.amazonaws.com/images.cocodataset.org/{split}/{name}",
                f"http://images.cocodataset.org/{split}/{name}"]
    payload, url = get_bytes(urls, expected)
    with Image.open(BytesIO(payload)) as image:
        image.verify()
    path.write_bytes(payload)
    return name, path, hashlib.sha256(payload).hexdigest(), url


def prepare(name, workers):
    root = ROOT / "data" / name
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "source").mkdir(exist_ok=True)
    image_hashes = {}
    rows = []
    if name == "chair":
        origin = EXPORT / "protocol/chair500.jsonl"
        manifest = json.loads((EXPORT / "protocol/chair500.manifest.json").read_text())
        assert file_hash(origin) == manifest["samples_sha256"]
        rows = [json.loads(line) for line in origin.read_text().splitlines()]
        assert len(rows) == len({r["image_id"] for r in rows}) == 500
        image_hashes = {Path(r["image"]).name: r["image_sha256"] for r in rows}
        revision = file_hash(origin)
        (root / "source/chair500.jsonl").write_bytes(origin.read_bytes())
        scorer_root = ROOT / "upstream/chair"
        scorer_root.mkdir(exist_ok=True)
        scorer = scorer_root / "chair.py"
        if not scorer.exists():
            scorer.write_bytes(github_file("Maxlinn/CHAIR-metric-standalone", "chair.py", CHAIR_REV))
        assert file_hash(scorer) == CHAIR_HASH
        lock = json.loads((ROOT / "upstream.lock.json").read_text())
        lock["sources"]["chair"] = {"commit": CHAIR_REV, "source_hash": source_hash(scorer_root, ["."]),
                                    "url": "https://github.com/Maxlinn/CHAIR-metric-standalone"}
        atomic_json(ROOT / "upstream.lock.json", lock)
        provenance = {"revision": revision, "split": "validation", "dataset": "CHAIR-COCO-val2014-500",
                      "protocol": manifest, "evaluator_revision": CHAIR_REV, "evaluator_sha256": CHAIR_HASH}
    else:
        dataset = name.removeprefix("pope_")
        revision = POPE_REV
        hashes = {}
        for split in SPLITS:
            relative = f"output/seem/{dataset}/{dataset}_pope_seem_{split}.json"
            target = root / "source" / Path(relative).name
            if not target.exists():
                target.write_bytes(github_file("RUCAIBox/POPE", relative, POPE_REV))
            official = json.loads(target.read_text())
            assert len(official) == 3000
            assert len({r["question_id"] for r in official}) == 3000
            hashes[relative] = file_hash(target)
            for r in official:
                assert r["label"] in ("yes", "no")
                assert Path(r["image"]).name == r["image"]
                rows.append({"row_id": f"{split}:{r['question_id']}", "split": split,
                             "question_id": r["question_id"], "image": r["image"],
                             "question": r["text"], "label": r["label"]})
                image_hashes[r["image"]] = None
        provenance = {"revision": revision, "split": "test", "dataset": f"POPE-{dataset.upper()}",
                      "question_source": "RUCAIBox/POPE output/seem", "question_files_sha256": hashes}
    print(name, "images", len(image_hashes), "records", len(rows), flush=True)
    images = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(image_file, filename, root / "images", expected, name == "pope_gqa")
                for filename, expected in image_hashes.items()]
        for n, future in enumerate(as_completed(jobs), 1):
            filename, path, sha, url = future.result()
            images[filename] = {"sha256": sha, "source": url, "path": str(path.relative_to(root))}
            if n % 25 == 0 or n == len(jobs):
                print(name, "images ready", n, "/", len(jobs), flush=True)
    samples, gold, labels, mapping = {}, [], {}, defaultdict(list)
    for row in rows:
        filename = Path(row["image"]).name
        image = images[filename]
        if name == "chair":
            key, question, task, group = row["id"], row["question"], "chair", str(row["image_id"])
        else:
            key = pope_input_key(image["sha256"], row["question"] + SUFFIX)
            question, task, group = row["question"], "pope", filename
            row.update(input_key=key, image_sha256=image["sha256"])
            assert key not in labels or labels[key] == row["label"]
            labels[key] = row["label"]
            mapping[key].append(row["row_id"])
        samples[key] = {"sample_id": key, "question": question, "options": {},
                        "image_paths": [image["path"]], "public_context": None, "group_id": group, "task_id": task}
    for key, label in labels.items():
        gold.append({"sample_id": key, "answer": label, "metric_metadata": {"row_ids": mapping[key]}})
    atomic_json(root / "samples.json", list(samples.values()))
    atomic_json(root / "gold.json", gold)
    atomic_json(root / "rows.json", rows)
    atomic_json(root / "image_manifest.json", images)
    provenance.update(image_manifest_sha256=file_hash(root / "image_manifest.json"),
                      unique_inputs=len(samples), records=len(rows), images=len(images))
    atomic_json(root / "provenance.json", provenance)
    ids = list(samples)
    atomic_json(ROOT / f"manifests/{name}_all_ids.json", ids)
    atomic_json(ROOT / f"manifests/{name}_smoke_ids.json", ids[:6])
    for shard in range(3):
        atomic_json(ROOT / f"manifests/{name}_shard{shard}_ids.json", ids[shard::3])
    write_configs(name, revision)
    print(name, "PREPARED", provenance, flush=True)


def write_configs(name, revision, taus=(90, 70, 50)):
    """Generate configs independently of data preparation for added thresholds."""
    base = yaml.safe_load((ROOT / "configs/experiments/mmada_pope_yesno_shard0.yaml").read_text())
    base["run"]["seed"] = 42
    base["run"]["fail_on_error"] = False
    base["protocol"].update(data_root=f"data/{name}", dataset_revision=revision)
    if name == "chair":
        base["protocol"].update(task="chair", split="validation", prompt_id="chair_caption_v1",
                                strict_parser_id="chair_caption_v1", canvas_length=64,
                                official_scorer_revision=CHAIR_REV)
        base["cdapc"]["iteration_limit"] = 64
    for tau in taus:
        base["cdapc"]["r_gate"]["threshold"] = tau / 100
        for shard in range(3):
            tag = f"mmada_{name}_r{tau}_shard{shard}"
            base["run"].update(name=tag, sample_ids_file=f"manifests/{name}_shard{shard}_ids.json")
            base["runtime"]["device"] = "cuda:0"
            (ROOT / f"configs/experiments/{tag}.yaml").write_text(yaml.safe_dump(base, sort_keys=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("chair", "pope_aokvqa", "pope_gqa"))
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    prepare(args.dataset, args.workers)
