"""Fetch official metadata/configs; verify cached binary assets against official SHA-256."""
from pathlib import Path
import sys
import hashlib
import json
import os
from dataclasses import asdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cdapc_lab.io.artifacts import atomic_json, file_hash, source_hash, digest
from huggingface_hub import HfApi, hf_hub_download

SOURCES = {
    "mmada": ("https://github.com/Gen-Verse/MMaDA", "3cdb8709635beb29f66e4e23cd66060872ac3996"),
    "lavida": ("https://github.com/jacklishufan/LaViDa", "24220c0077f403d24ca52968b6a7c55b687d5c2f"),
    "m3cot": ("https://github.com/LightChen233/M3CoT", "ee50d5ac69661f81731d33647dd307d1f84eada5"),
}
ASSETS = {
    "mmada": ("Gen-Verse/MMaDA-8B-MixCoT", "3ee0085f0c42541f1134aae30482954451952406", "/root/autodl-fs/models/MMaDA-8B-MixCoT"),
    "lavida": ("hbXNov/lavida-llada-reason", "246b218b0444bc9edfd09ab0e05b072db96e5e2a", "/root/autodl-fs/lavida-ckpts/lavida-llada-reason"),
    "magvitv2": ("showlab/magvitv2", None, "/root/autodl-fs/models/magvitv2"),
    "siglip": ("google/siglip-so400m-patch14-384", None, "/root/autodl-fs/lavida-ckpts/siglip-so400m-patch14-384"),
}


def materialize_model(name, spec, api):
    repo, revision, cache = spec
    info = api.model_info(repo, revision=revision, files_metadata=True)
    output = ROOT / "models" / name
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for entry in info.siblings:
        filename = entry.rfilename
        if "/" in filename or not (filename.endswith((".safetensors", ".json", ".model"))):
            continue
        target = output / filename
        if filename.endswith(".safetensors"):
            cached = Path(cache) / filename
            if not cached.is_file() or entry.lfs is None:
                raise ValueError(f"No verifiable official binary cache: {cached}")
            print(f"Verifying {cached}", flush=True)
            actual = file_hash(cached)
            if actual != entry.lfs.sha256:
                raise ValueError(f"Official binary hash mismatch: {cached}")
            if target.is_symlink():
                if target.resolve() != cached.resolve():
                    raise ValueError(f"Unexpected asset link: {target}")
            elif not target.exists():
                target.symlink_to(cached)
        else:
            hf_hub_download(repo, filename, revision=info.sha, local_dir=output)
            actual = file_hash(target)
        hashes[filename] = actual
    return {"repo": repo, "revision": info.sha, "path": str(output), "files": hashes,
            "verified_stats": {f: {"size": (output / f).stat().st_size, "mtime_ns": (output / f).stat().st_mtime_ns} for f in hashes},
            "binary_policy": "official_lfs_sha256_verified_external_weights; fresh official metadata/tokenizer; no cached code"}


def prepare_data(api, count=2, name="m3cot_smoke"):
    import pyarrow.parquet as pq
    from PIL import Image
    from io import BytesIO
    repo, revision = "LightChen2333/M3CoT", "48cf35001d595a6b0290c82c897a4b4563390821"
    info = api.dataset_info(repo, revision=revision, files_metadata=True)
    cached_root = Path("/root/autodl-fs/hf_home/hub/datasets--LightChen2333--M3CoT/snapshots") / revision
    files, hashes = [], {}
    for entry in info.siblings:
        if entry.rfilename.startswith("data/test-") and entry.rfilename.endswith(".parquet"):
            path = cached_root / entry.rfilename
            actual = file_hash(path)
            if entry.lfs is None or actual != entry.lfs.sha256:
                raise ValueError(f"Official dataset hash mismatch: {path}")
            files.append(path)
            hashes[entry.rfilename] = actual
    if not files:
        raise ValueError("No official test split files")
    files.sort()
    ids = [str(x) for path in files for x in pq.read_table(path, columns=["id"])["id"].to_pylist()]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source IDs")
    selected = sorted(ids, key=lambda x: digest(["smoke_ids_v1", 20261105, x]))[:count]
    output = ROOT / "data" / name
    (output / "images").mkdir(parents=True, exist_ok=True)
    samples, gold = {}, {}
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                sid = str(row["id"])
                if sid not in selected:
                    continue
                print({"selected_id": sid, "source_fields": list(row)}, flush=True)
                image_data = row["image"]
                if not isinstance(image_data, dict) or not image_data.get("bytes"):
                    raise ValueError("Expected official embedded image bytes")
                filename = hashlib.sha256(sid.encode()).hexdigest() + ".png"
                with Image.open(BytesIO(image_data["bytes"])) as image:
                    image.convert("RGB").save(output / "images" / filename)
                options = {chr(65+i): choice for i, choice in enumerate(row["choices"])}
                answer = row["answer"]
                if isinstance(answer, int):
                    answer = chr(65 + answer)
                if answer not in options:
                    raise ValueError("Gold outside answer domain")
                samples[sid] = dict(sample_id=sid, question=row["question"], options=options,
                                    image_paths=["images/" + filename], public_context=row.get("context") or None,
                                    group_id=sid, task_id="m3cot")
                gold[sid] = dict(sample_id=sid, answer=answer,
                                 metric_metadata={k: row.get(k) for k in ("domain", "topic", "category")})
    if set(samples) != set(selected):
        raise ValueError("Missing selected sample")
    atomic_json(output / "samples.json", [samples[x] for x in selected])
    atomic_json(output / "gold.json", [gold[x] for x in selected])
    provenance = dict(repo=repo, revision=revision, split="test", source_files=hashes,
                      source_count=len(ids), selected_count=len(selected),
                      selection=f"lowest sha256(['smoke_ids_v1',20261105,id]), n={count}, no outcome selection",
                      filter="none; fixed IDs; require one valid image and MCQ; errors fail",
                      public_context="official context field only; rationale excluded")
    atomic_json(output / "provenance.json", provenance)
    atomic_json(ROOT / "manifests" / f"{name}_ids.json", selected)
    if name == "m3cot_smoke":
        atomic_json(ROOT / "manifests/m3cot_trace_ids.json", selected)
    return provenance


def main():
    api = HfApi()
    lock_path = ROOT / "upstream.lock.json"
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else {"schema_version": 1, "models": {}}
    lock["sources"] = {name: {"url": url, "commit": commit,
                                "source_hash": source_hash(ROOT / "upstream" / name, ["."])}
                        for name, (url, commit) in SOURCES.items()}
    for name, spec in ASSETS.items():
        if name not in lock["models"]:
            lock["models"][name] = materialize_model(name, spec, api)
            atomic_json(lock_path, lock)
    lock["dataset"] = prepare_data(api)
    lock["lavida_patch_sha256"] = file_hash(ROOT / "patches/lavida-sdpa-mask.patch")
    lock["patches"] = {path.name: file_hash(path) for path in sorted((ROOT / "patches").glob("*.patch"))}
    atomic_json(lock_path, lock)
    print("Official assets verified; lock saved", flush=True)


if __name__ == "__main__":
    main()
