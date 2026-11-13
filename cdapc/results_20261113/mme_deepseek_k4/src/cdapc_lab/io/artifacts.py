from __future__ import annotations
from pathlib import Path
import hashlib
import json
import os
import tempfile
import copy
import subprocess


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path):
    return json.loads(Path(path).read_text())


def safe_id(sample_id):
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id must be a nonempty string")
    return hashlib.sha256(sample_id.encode()).hexdigest()


def sample_seed(seed, sample_id, replicate_id=0):
    return int(digest(["sample_seed_v1", seed, sample_id, replicate_id])[:8], 16)


def generation_projection(config):
    result = copy.deepcopy(config)
    result.pop("recording", None)
    result.pop("derived", None)
    result["run"] = {k: config["run"][k] for k in ("seed", "replicate_id")}
    for key in ("normalizer_id", "strict_parser_id", "official_scorer_revision", "scorer_profile", "primary_metric", "data_root"):
        result["protocol"].pop(key, None)
    # Paths are provenance; content and revisions are separately fingerprinted.
    for key in ("checkpoint", "upstream_root", "vision_checkpoint"):
        result["model"].pop(key, None)
    return result


def source_hash(root, folders):
    root = Path(root)
    paths = set()
    for folder in folders:
        item = root / folder
        paths.update(item.rglob("*.py") if item.is_dir() else [item])
    return digest({str(p.relative_to(root)): file_hash(p) for p in sorted(paths)})


def git_provenance(root):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL, text=True).strip()
    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except subprocess.CalledProcessError:
        return {"commit": None, "dirty": None}


def reusable_prediction(path, sample_id, generation_fingerprint):
    if not Path(path).exists():
        return False
    value = read_json(path)
    if value.get("sample_id") != sample_id or value.get("generation_fingerprint") != generation_fingerprint:
        raise ValueError("Prediction identity/fingerprint mismatch")
    return value.get("status") == "success"
