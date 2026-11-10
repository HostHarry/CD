"""Prepare pinned VCD-release POPE A-OKVQA/GQA with the current yes/no protocol."""
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from io import BytesIO
import json
from pathlib import Path
import shutil
import threading
import time

from PIL import Image
import requests

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "data/pope_extended"
PIN = "d6568ff81b8fd306a49e630df44f2db5c2300191"
LOCAL = threading.local()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def write_locked(path, text):
    if path.exists() and path.read_text() != text:
        raise ValueError(f"refusing to change an existing dataset: {path}")
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)


def image_info(path):
    with Image.open(path) as image:
        size = image.size
        image.verify()
    return {"sha256": digest(path), "bytes": path.stat().st_size, "size": list(size)}


def acquire(dataset, name, existing, cached):
    if Path(name).name != name or not name.endswith(".jpg"):
        raise ValueError(f"unexpected image name: {name}")
    destination = BASE / f"{dataset}_images" / name
    sidecar = destination.with_suffix(".source.json")
    temporary = destination.with_suffix(".download")
    if not destination.exists() and temporary.exists() and sidecar.exists():
        info = image_info(temporary)
        if json.loads(sidecar.read_text())["sha256"] == info["sha256"]:
            temporary.replace(destination)
    if destination.exists():
        info = image_info(destination)
        provenance = cached.get(name) or (json.loads(sidecar.read_text()) if sidecar.exists() else None)
        if provenance is None or provenance["sha256"] != info["sha256"]:
            raise ValueError(f"image exists without matching provenance: {destination}")
        return name, provenance
    if name in existing:
        source = existing[name]
        info = image_info(source)
        temporary = destination.with_suffix(".download")
        shutil.copyfile(source, temporary)
        provenance = {**info, "source": str(source)}
        save(sidecar, provenance)
        temporary.replace(destination)
        return name, provenance
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
    urls = ([f"http://images.cocodataset.org/val2014/{name}",
             f"https://s3.amazonaws.com/images.cocodataset.org/val2014/{name}"]
            if dataset == "aokvqa" else
            [f"https://cs.stanford.edu/people/rak248/VG_100K_2/{name}",
             f"https://cs.stanford.edu/people/rak248/VG_100K/{name}"])
    errors = []
    for attempt in range(3):
        for url in urls:
            try:
                response = LOCAL.session.get(url, timeout=(10, 35))
                response.raise_for_status()
                with Image.open(BytesIO(response.content)) as image:
                    image.verify()
                temporary = destination.with_suffix(".download")
                temporary.write_bytes(response.content)
                info = image_info(temporary)
                provenance = {**info, "source": url}
                save(sidecar, provenance)
                temporary.replace(destination)
                return name, provenance
            except (requests.RequestException, OSError, ValueError) as error:
                errors.append(f"{url}: {error}")
        time.sleep(attempt + 1)
    raise RuntimeError(f"could not acquire {name}: {errors}")


def prepare(dataset, workers):
    tree = json.loads((BASE / "upstream_tree.json").read_text())
    assert tree["sha"] == PIN and not tree.get("truncated", False)
    blobs = {entry["path"]: entry["sha"] for entry in tree["tree"] if entry["type"] == "blob"}
    all_rows, sources, images_by_split = [], {}, {}
    for split in ("random", "popular", "adversarial"):
        name = f"{dataset}_pope_{split}.json"
        path = BASE / "raw" / name
        data = path.read_bytes()
        relative = f"experiments/data/POPE/{dataset}/{name}"
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if blob != blobs[relative]:
            raise ValueError(f"question file differs from pinned upstream Git blob: {name}")
        rows = [json.loads(line) for line in data.decode().splitlines()]
        assert len(rows) == 3000 and len({r["question_id"] for r in rows}) == 3000
        assert collections.Counter(r["label"] for r in rows) == {"yes": 1500, "no": 1500}
        images_by_split[split] = {r["image"] for r in rows}
        assert len(images_by_split[split]) == 500
        sources[split] = {"repository": "https://github.com/DAMO-NLP-SG/VCD", "revision": PIN,
                          "path": relative, "git_blob": blob, "sha256": hashlib.sha256(data).hexdigest()}
        all_rows.extend((split, row) for row in rows)
    assert all(s == images_by_split["random"] for s in images_by_split.values())
    names = sorted(images_by_split["random"])
    image_directory = BASE / f"{dataset}_images"
    image_directory.mkdir(exist_ok=True)
    cache_path = BASE / f"{dataset}_images.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    existing = {}
    if dataset == "aokvqa":
        for directory in ("pope_images", "chair_images", "calibration_images"):
            existing.update({p.name: p for p in (ROOT / "data" / directory).glob("*.jpg")})
    errors = []
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(acquire, dataset, name, existing, cache): name for name in names}
        for index, future in enumerate(as_completed(futures), 1):
            try:
                name, info = future.result()
                cache[name] = info
                save(cache_path, cache)
            except Exception as error:
                errors.append(str(error))
                if len(errors) <= 3:
                    print(f"{dataset}: {error}", flush=True)
            if index % 25 == 0:
                print(f"{dataset}: checked {index}/500 images; failures {len(errors)}", flush=True)
    if errors:
        save(BASE / f"{dataset}_image_errors.json", errors)
        raise RuntimeError(f"missing/invalid images; refusing partial benchmark: {len(errors)}")
    calibration = {json.loads(line)["image_sha256"] for line in
                   (ROOT / "data/calibration.jsonl").read_text().splitlines()}
    rows = []
    for split, source in all_rows:
        image = image_directory / source["image"]
        question = source["text"].rstrip() + " Please answer yes or no."
        rows.append({
            "id": f"{dataset}:{split}:{source['question_id']}", "dataset": dataset, "split": split,
            "image": str(image.resolve()), "image_sha256": cache[source["image"]]["sha256"],
            "question": question, "label": source["label"], "original_question": source["text"],
            "image_id": int(image.stem.split("_")[-1]),
        })
    labels = collections.defaultdict(set)
    for row in rows:
        labels[row["image_sha256"], row["question"]].add(row["label"])
    assert all(len(value) == 1 for value in labels.values())
    output = BASE / f"{dataset}_yesno.jsonl"
    write_locked(output, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    overlap = calibration & {r["image_sha256"] for r in rows}
    save(BASE / f"{dataset}_manifest.json", {
        "sources": sources, "rows": len(rows), "unique_inputs": len(labels),
        "unique_images": len(names), "samples_sha256": digest(output),
        "suffix": "Please answer yes or no.", "image_manifest_sha256": digest(cache_path),
        "calibration_overlap_image_hashes": sorted(overlap),
        "calibration_policy": "These native/VCD/VIGOR baselines were not fitted or selected on DEGAP calibration; retain every official benchmark row.",
    })
    print(f"{dataset}: ready, 9000 rows / {len(labels)} unique inputs / 500 images", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=["aokvqa", "gqa"], default=["aokvqa", "gqa"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    # Explicitly record that the fixed author baselines use no fitted calibration.
    write_locked(BASE / "excluded_samples.jsonl", "")
    for dataset in args.datasets:
        prepare(dataset, args.workers)


if __name__ == "__main__":
    main()
