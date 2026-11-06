import argparse
import hashlib
import json
from pathlib import Path
import random
import io
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

from .runtime import save_json, sha256


def write_samples(path, samples, provenance):
    if path.exists() or path.with_suffix(".manifest.json").exists():
        raise FileExistsError("sample output must be new")
    with path.open("x") as stream:
        for sample in samples:
            stream.write(json.dumps(sample, ensure_ascii=False) + "\n")
    save_json(path.with_suffix(".manifest.json"), {
        **provenance, "rows": len(samples), "images": len({r["image_sha256"] for r in samples}),
        "samples_sha256": sha256(path),
    })


def prepare_pope(args):
    import pyarrow.parquet as pq
    rows = []
    hashes = {}
    for split in ("random", "popular", "adversarial"):
        path = args.questions / f"coco_pope_{split}.json"
        values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(values) != 3000 or {v["question_id"] for v in values} != set(range(1, 3001)):
            raise ValueError(f"invalid official POPE IDs: {split}")
        if sum(v["label"] == "yes" for v in values) != 1500 or any(v["label"] not in {"yes", "no"} for v in values):
            raise ValueError(f"invalid POPE labels: {split}")
        hashes[split] = sha256(path)
        for value in values:
            if Path(value["image"]).name != value["image"]:
                raise ValueError("unsafe image filename")
            rows.append({
                "id": f"{split}:{value['question_id']}", "split": split,
                "image": value["image"], "question": value["text"], "label": value["label"],
            })
    needed = {r["image"] for r in rows}
    found = {}
    args.images.mkdir(parents=True, exist_ok=True)
    for batch in pq.ParquetFile(args.parquet).iter_batches(batch_size=64):
        for row in batch.to_pylist():
            name = row["image_source"]
            if not name.endswith(".jpg"):
                name += ".jpg"
            if name not in needed:
                continue
            data = row["image"]["bytes"]
            digest = hashlib.sha256(data).hexdigest()
            target = args.images / name
            if target.exists():
                if sha256(target) != digest:
                    raise ValueError(f"existing image differs: {target}")
            else:
                with target.open("xb") as stream:
                    stream.write(data)
            if name in found and found[name] != digest:
                raise ValueError("conflicting image copies")
            found[name] = digest
    if needed != found.keys():
        raise ValueError(f"missing {len(needed - found.keys())} POPE images")
    for row in rows:
        row["image_sha256"] = found[row["image"]]
        row["image"] = str((args.images / row["image"]).resolve())
    write_samples(args.output, rows, {
        "dataset": "POPE-COCO", "questions_sha256": hashes,
        "parquet_sha256": sha256(args.parquet), "prompt_suffix": "",
        "answer_parser": "official_no_not_no_else_yes",
    })


def prepare_chair(args):
    annotations = json.loads(args.instances.read_text())
    excluded = set()
    if args.exclude:
        excluded = {Path(json.loads(line)["image"]).name for line in args.exclude.read_text().splitlines() if line.strip()}
    images = sorted([im for im in annotations["images"] if im["file_name"] not in excluded],
                    key=lambda im: im["id"])
    if args.count > len(images):
        raise ValueError("not enough distinct candidate images")
    selected = random.Random(args.seed).sample(images, args.count)
    if any(Path(im["file_name"]).name != im["file_name"] for im in selected):
        raise ValueError("unsafe COCO image filename")
    missing = [im for im in selected if not (args.images / im["file_name"]).is_file()]
    if missing and args.download:
        args.images.mkdir(parents=True, exist_ok=True)
        def download(im):
            url = im["coco_url"]
            if not url.startswith(("http://images.cocodataset.org/", "https://images.cocodataset.org/")):
                raise ValueError("unexpected COCO image host")
            data = urllib.request.urlopen(url, timeout=45).read()
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
            with (args.images / im["file_name"]).open("xb") as stream:
                stream.write(data)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(download, missing))
        missing = []
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} sampled COCO images missing; no replacement sampling. "
            f"First files: {[im['file_name'] for im in missing[:10]]}")
    rows = [{
        "id": f"chair:{im['id']}", "image_id": im["id"], "split": "chair",
        "image": str((args.images / im["file_name"]).resolve()),
        "image_sha256": sha256(args.images / im["file_name"]),
        "question": "Please describe this image in detail.",
    } for im in selected]
    write_samples(args.output, rows, {
        "dataset": "COCO-val2014", "seed": args.seed, "candidate_images": len(images),
        "instances_sha256": sha256(args.instances),
        "excluded_samples_sha256": sha256(args.exclude) if args.exclude else None,
    })


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    pope = commands.add_parser("pope")
    pope.add_argument("--questions", type=Path, required=True)
    pope.add_argument("--parquet", type=Path, required=True)
    chair = commands.add_parser("chair")
    chair.add_argument("--instances", type=Path, required=True)
    chair.add_argument("--count", type=int, default=500)
    chair.add_argument("--seed", type=int, default=42)
    chair.add_argument("--exclude", type=Path)
    chair.add_argument("--download", action="store_true")
    demo = commands.add_parser("demo")
    demo.add_argument("--images", type=Path, default=Path("mmu_validation"))
    demo.add_argument("--output", type=Path, required=True)
    for command in (pope, chair):
        command.add_argument("--images", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.parent.is_dir():
        parser.error("output parent must exist")
    if args.command == "demo":
        rows = [{
            "id": name, "image": str((args.images / name).resolve()),
            "image_sha256": sha256(args.images / name),
            "question": "Please describe this image in detail.",
        } for name in ("dog.png", "llava1.jpg")]
        write_samples(args.output, rows, {"dataset": "official_mmada_demo"})
    else:
        (prepare_pope if args.command == "pope" else prepare_chair)(args)


if __name__ == "__main__":
    main()
