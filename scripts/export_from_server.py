"""Export MMaDA-DEGAP yes/no POPE and CHAIR records verbatim (run on the original server)."""
import gzip
import hashlib
import json
import shutil
import sqlite3
import sys
from pathlib import Path

SRC = Path("/root/autodl-tmp/MMaDA-DEGAP")
RUN = SRC / "results/experiment_128_64_32_yesno"
OUT = Path(__file__).resolve().parents[1]
SHARD_BYTES = 200 * 1024 * 1024  # uncompressed; keeps each gzip shard well below GitHub's 100 MB limit
METHODS = ("native", "degap_token")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst) if src.is_file() else shutil.copytree(src, dst)


def export_predictions(benchmark, database):
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    index = {}
    for method in METHODS:
        target = OUT / benchmark / method
        target.mkdir(parents=True)
        shard, written, stream, entries = 0, 0, None, []
        for key, payload in connection.execute(
                "SELECT input_key,payload FROM predictions WHERE method=? ORDER BY input_key", (method,)):
            raw = payload.encode()
            if stream is None or written >= SHARD_BYTES:
                if stream:
                    stream.close()
                name = f"predictions-{shard:03d}.jsonl.gz"
                stream = gzip.GzipFile(target / name, "wb", mtime=0)
                shard, written = shard + 1, 0
            line = b'{"input_key": ' + json.dumps(key).encode() + b', "payload": ' + raw + b"}\n"
            stream.write(line)
            written += len(line)
            entries.append({"input_key": key, "shard": name, "payload_sha256": sha(raw)})
        stream.close()
        index[method] = {"predictions": len(entries), "shards": shard, "entries": entries}
    return index


def main():
    if any(OUT.glob("pope")) or any(OUT.glob("chair")):
        raise SystemExit("export already exists")
    copy(SRC / "data/pope_coco.jsonl", OUT / "protocol/pope_coco.jsonl")
    for name in ("pope_coco.manifest.json", "pope_coco_yesno.jsonl", "pope_coco_yesno.manifest.json",
                 "chair500.jsonl", "chair500.manifest.json"):
        copy(SRC / "data" / name, OUT / "protocol" / name)
    copy(SRC / "derive_pope_prompt.py", OUT / "protocol/derive_pope_prompt.py")
    copy(RUN / "protocol.json", OUT / "protocol/run_protocol.json")
    copy(SRC / "configs/mmada_degap_128_64_32.yaml", OUT / "protocol/mmada_degap_128_64_32.yaml")
    for name in ("checkpoint-sha256.json", "environment-lock.txt"):
        copy(SRC / "results/00_setup" / name, OUT / "environment" / name)
    images = {}
    for file, folder in (("pope_coco_yesno.jsonl", "pope_images"), ("chair500.jsonl", "chair_images")):
        for line in (SRC / "data" / file).read_text().splitlines():
            row = json.loads(line)
            images[Path(row["image"]).name] = row["image_sha256"]
    (OUT / "protocol/images.tsv").write_text(
        "file\tsha256\n" + "".join(f"{k}\t{v}\n" for k, v in sorted(images.items())))
    index = {}
    for benchmark, files in (("pope", ("manifest.json", "metrics.json", "status.json", "progress.json",
                                        "native_answers.json", "degap_token_answers.json", "source")),
                             ("chair", ("manifest.json", "metrics.json", "status.json", "progress.json",
                                        "chair_metrics.json", "native_answers.json", "degap_token_answers.json",
                                        "native_chair.json", "degap_token_chair.json", "source"))):
        run = RUN / f"{benchmark}_full"
        index[benchmark] = export_predictions(benchmark, run / "predictions.sqlite")
        index[benchmark]["sqlite_sha256"] = sha((run / "predictions.sqlite").read_bytes())
        for name in files:
            copy(run / name, OUT / benchmark / "run" / name)
    (OUT / "records_index.json").write_text(json.dumps(index, indent=1) + "\n")
    write_sums()


def write_sums():
    sums = [f"{sha(p.read_bytes())}  {p.relative_to(OUT)}\n" for p in sorted(OUT.rglob("*"))
            if p.is_file() and ".git" not in p.parts and p.name != "SHA256SUMS"]
    (OUT / "SHA256SUMS").write_text("".join(sums))


if __name__ == "__main__":
    write_sums() if sys.argv[1:] == ["--sums-only"] else main()
