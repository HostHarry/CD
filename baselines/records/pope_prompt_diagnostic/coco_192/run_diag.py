"""POPE prompt-format diagnostic: native MMaDA, 4 prompt variants, 192 rows.

Variants (all native, generation 256/256/32, temperature 0, CFG 0,
low_confidence, seed 42, crop 512, safe EOS):
  A  chat template, user content = q0 (POPE question without yes/no suffix)
  B  chat template, user content = common.mmmu_protocol.INSTRUCTION + "\\n" + q0
  C  MMaDA training text format: literal "<eot_id>" (not the special token),
     explicit bos before the header text, prompt_protocol
     "mmada_training_text_literal_eot"
  D  chat template, user content = q0 + " Please answer yes or no." (original)

GPU0 runs A then C then D; GPU1 runs B. Each variant keeps its own resumable
SQLite under <variant>/predictions.sqlite.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import sys
import time

import torch
from PIL import Image

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(ROOT))

from common.mmmu_protocol import INSTRUCTION
from common.run import input_key
from common.runtime import Runtime, save_json, sha256, source_hashes
from training.utils import image_transform, image_transform_squash

SOURCE = ROOT / "data/pope_coco_yesno.jsonl"
SAMPLES_PATH = BASE / "samples.jsonl"
MANIFEST_PATH = BASE / "manifest.json"
CHECKPOINT = "/root/autodl-fs/models/MMaDA-8B-MixCoT"
VQ_CHECKPOINT = "/root/autodl-fs/models/magvitv2"
SUFFIX = " Please answer yes or no."
SELECT_KEY = "pope-prompt-diag-v1:"
VARIANTS = ("A", "B", "C", "D")
RESOLUTION, RESIZE = 512, "crop"
SEED = 42
WARMUPS = 3
GENERATION = {"max_new_tokens": 256, "steps": 256, "block_length": 32,
              "temperature": 0.0, "cfg_scale": 0.0, "remasking": "low_confidence"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_locked(path, content):
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"refusing changed artifact: {path}")
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(path)


def read_samples():
    return [json.loads(line) for line in SAMPLES_PATH.read_text().splitlines() if line.strip()]


def variant_question(variant, q0):
    if variant == "A":
        return q0
    if variant == "B":
        return INSTRUCTION + "\n" + q0
    if variant == "C":
        return q0
    if variant == "D":
        return q0 + SUFFIX
    raise ValueError(f"unknown variant {variant}")


def variant_sample(row, variant):
    sample = {"id": row["id"], "split": row["split"], "label": row["label"],
              "image": row["image"], "image_sha256": row["image_sha256"],
              "question": variant_question(variant, row["q0"])}
    if variant == "C":
        sample["prompt_variant"] = "c_training_text"
    return sample


def prepare():
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]
    if len(rows) != 9000:
        raise ValueError("expected 9000 POPE COCO rows")
    groups = {}
    for row in rows:
        groups.setdefault((row["split"], row["label"]), []).append(row)
    expected = {(s, l) for s in ("random", "popular", "adversarial") for l in ("yes", "no")}
    if set(groups) != expected or any(len(g) != 1500 for g in groups.values()):
        raise ValueError("unexpected POPE split/label groups")
    selected = []
    for key in sorted(groups):
        ranked = sorted(groups[key], key=lambda r: hashlib.sha256(
            (SELECT_KEY + r["id"]).encode()).hexdigest())
        selected.extend(ranked[:32])
    if len(selected) != 192 or len({r["id"] for r in selected}) != 192:
        raise ValueError("expected 192 unique diagnostic rows")
    out, images = [], {}
    for row in selected:
        question = row["question"]
        if not question.endswith(SUFFIX):
            raise ValueError(f"{row['id']}: question lacks the exact yes/no suffix")
        q0 = question[:-len(SUFFIX)]
        if not q0:
            raise ValueError(f"{row['id']}: empty question after suffix removal")
        path_image = row["image"]
        if path_image not in images:
            images[path_image] = sha256(path_image)
        if images[path_image] != row["image_sha256"]:
            raise ValueError(f"image hash differs: {path_image}")
        out.append({"id": row["id"], "q0": q0, "split": row["split"], "label": row["label"],
                    "image": path_image, "image_sha256": row["image_sha256"]})
    write_locked(SAMPLES_PATH, "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in out).encode())
    unique_inputs = {variant: len({input_key(variant_sample(row, variant)) for row in out})
                     for variant in VARIANTS}
    score_path = BASE / "score.py"
    manifest = {
        "experiment": "pope_prompt_diagnostic",
        "created": time.time(),
        "source": str(SOURCE), "source_sha256": sha256(SOURCE),
        "samples": str(SAMPLES_PATH), "samples_sha256": sha256(SAMPLES_PATH),
        "selection": "per (split,label) sort by sha256('pope-prompt-diag-v1:'+id), first 32",
        "q0": "question minus trailing ' Please answer yes or no.'",
        "variants": {
            "A": "chat template; user = q0",
            "B": "chat template; user = common.mmmu_protocol.INSTRUCTION + '\\n' + q0",
            "C": "MMaDA training text format, literal <eot_id>, explicit bos; "
                 "prompt_protocol=mmada_training_text_literal_eot",
            "D": "chat template; user = q0 + ' Please answer yes or no.'",
        },
        "generation": GENERATION, "seed": SEED,
        "warmups_per_variant_per_session": WARMUPS, "early_eos": True, "method": "native",
        "checkpoint": CHECKPOINT, "vq_checkpoint": VQ_CHECKPOINT,
        "resolution": RESOLUTION, "resize": RESIZE,
        "unique_inputs_per_variant": unique_inputs,
        "sources": source_hashes(), "launcher_sha256": sha256(__file__),
        "score_sha256": sha256(score_path) if score_path.exists() else None,
    }
    write_locked(MANIFEST_PATH, (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode())
    print(json.dumps({"rows": len(out), "unique_inputs_per_variant": unique_inputs}, indent=2))


def verify_manifest():
    manifest = json.loads(MANIFEST_PATH.read_text())
    if (manifest["launcher_sha256"] != sha256(__file__)
            or manifest["samples_sha256"] != sha256(SAMPLES_PATH)
            or manifest["sources"] != source_hashes()):
        raise ValueError("diagnostic launcher, samples or inference sources changed")
    return manifest


class DiagnosticRuntime(Runtime):
    """Runtime whose prepare() implements the variant-C training text format."""

    def prepare(self, sample):
        if sample.get("prompt_variant") != "c_training_text":
            return super().prepare(sample)
        image_path = Path(sample["image"])
        actual_sha = sha256(image_path)
        if sample.get("image_sha256") and sample["image_sha256"] != actual_sha:
            raise ValueError(f"image changed: {image_path}")
        with Image.open(image_path) as image:
            transform = image_transform if self.resize == "crop" else image_transform_squash
            pixels = transform(image.convert("RGB"), resolution=self.resolution).unsqueeze(0).to(self.device)
        codes = self.vq.get_code(pixels) + len(self.tokenizer)
        text = ('<|start_header_id|>user<|end_header_id|>\n' + sample["question"]
                + '<eot_id><|start_header_id|>assistant<|end_header_id|>\n')
        text_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        special = self.prompting.sptids_dict
        ids = [int(special["<|mmu|>"]), int(special["<|soi|>"])]
        ids += codes[0].tolist() + [int(special["<|eoi|>"])]
        ids += [int(self.tokenizer.bos_token_id)] + [int(t) for t in text_ids]
        tokens = torch.tensor([ids], device=self.device, dtype=torch.long)
        visual = list(range(2, 2 + codes.shape[1]))
        provenance = {
            "image_sha256": actual_sha, "prompt_ids": ids,
            "visual_positions": visual, "image_code_offset": len(self.tokenizer),
            "resolution": self.resolution, "resize": self.resize,
            "prompt_protocol": "mmada_training_text_literal_eot",
            "prompt_variant": "c_training_text",
        }
        return tokens, visual, provenance


def run_variant(runtime, variant, generation):
    samples = read_samples()
    unique = {}
    for row in samples:
        sample = variant_sample(row, variant)
        unique.setdefault(input_key(sample), sample)
    outdir = BASE / variant
    outdir.mkdir(exist_ok=True)
    database = sqlite3.connect(outdir / "predictions.sqlite")
    database.execute("CREATE TABLE IF NOT EXISTS predictions "
                     "(input_key TEXT PRIMARY KEY, payload TEXT)")
    done = {key for (key,) in database.execute("SELECT input_key FROM predictions")}
    total = len(unique)
    if len(done) < total:
        first = next(iter(unique.values()))
        for _ in range(WARMUPS):
            runtime.generate(first, None, generation, SEED)
        for key, sample in unique.items():
            if key in done:
                continue
            result = runtime.generate(sample, None, generation, SEED)
            with database:
                database.execute("INSERT INTO predictions VALUES (?,?)",
                                 (key, json.dumps(result, ensure_ascii=False, allow_nan=False)))
            done.add(key)
            save_json(outdir / "progress.json", {
                "variant": variant, "completed_unique_predictions": len(done),
                "target_unique_predictions": total, "last_id": sample["id"],
                "updated": time.time()})
            print(f"{variant} {len(done)}/{total} {sample['id']}: {result['text'][:100]}", flush=True)
    database.close()
    save_json(outdir / "status.json", {
        "variant": variant, "unique_inputs": total, "committed": len(done),
        "complete": len(done) == total, "finished": time.time()})
    print(f"{variant} complete: {len(done)}/{total}", flush=True)


def sanity():
    verify_manifest()
    runtime = DiagnosticRuntime(CHECKPOINT, VQ_CHECKPOINT, device="cuda",
                                resolution=RESOLUTION, resize=RESIZE, early_eos=True)
    tokenizer = runtime.tokenizer
    print(json.dumps({
        "bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "model_mask_token_id": runtime.model.config.mask_token_id,
        "model_eos_token_id": runtime.model.config.eos_token_id,
        "instruction": INSTRUCTION,
    }, ensure_ascii=False))
    row = read_samples()[0]
    print("sample:", json.dumps({k: row[k] for k in ("id", "split", "label", "q0")}, ensure_ascii=False))
    decoded = {}
    for variant in VARIANTS:
        tokens, visual, provenance = runtime.prepare(variant_sample(row, variant))
        ids = provenance["prompt_ids"]
        start = visual[-1] + 2 if visual else 0
        tail_ids = ids[start:]
        decoded[variant] = tail_ids
        print(f"=== variant {variant} protocol={provenance['prompt_protocol']} "
              f"prompt_len={len(ids)} visual={len(visual)} ===")
        print("tail_ids:", tail_ids)
        print("decoded :", repr(tokenizer.decode(tail_ids, skip_special_tokens=False)))
        print("plain  :", repr(tokenizer.decode(tail_ids, skip_special_tokens=True)))
    ac_diff = [i for i, (a, c) in enumerate(zip(decoded["A"], decoded["C"])) if a != c]
    print("A/C tail id positions differing (first 40):", ac_diff[:40],
          "lenA", len(decoded["A"]), "lenC", len(decoded["C"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--sanity", action="store_true")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.check:
        verify_manifest()
        print("Diagnostic manifest hashes match. No model was loaded.")
    elif args.sanity:
        sanity()
    elif args.variants:
        manifest = verify_manifest()
        runtime = DiagnosticRuntime(CHECKPOINT, VQ_CHECKPOINT, device="cuda",
                                    resolution=RESOLUTION, resize=RESIZE, early_eos=True)
        for variant in args.variants:
            run_variant(runtime, variant, manifest["generation"])
    else:
        parser.error("choose --prepare, --check, --sanity or --variants")


if __name__ == "__main__":
    def terminate(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")
    signal.signal(signal.SIGTERM, terminate)
    main()
