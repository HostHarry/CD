"""Prepare the pinned HostHarry/CD MMMU input protocol, without its decoder."""
import ast
import collections
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
from string import ascii_uppercase

from PIL import Image

from .runtime import ROOT, save_json, sha256

REFERENCE_COMMIT = "e48814d6994bd3a8ed1647f5e3906758135bfc6d"
DATASET_REVISION = "364f2e2eb107b36e07ff4c5a15f5947a759cef47"
DATASET_SHA256 = "18e0e97776217e7b5fc992dc9594014e7adc0cc0473f624ea957c18702e16681"
REFERENCE = ROOT / "data/mmmu_reference/e48814d/mmmu"
DATA = ROOT / "data/mmmu_wino_v1"
CHECKPOINT = Path("/root/autodl-fs/models/MMaDA-8B-MixCoT")
INSTRUCTION = (
    "You should first think about the reasoning process in the mind and then "
    "provide the user with the answer. The reasoning process is enclosed within "
    "<think> </think> tags, i.e. <think> reasoning process here </think> answer here"
)


def image_key(question, options):
    source = question + "\n" + "\n".join(f"{k}. {v}" for k, v in options.items())
    placeholders = sorted(set(re.findall(r"<image \d+>", source)))
    return placeholders[0].strip("<>").replace(" ", "_") if placeholders else "image_1"


def user_content(question, options):
    context = question + "\n"
    if options:
        context += "\n".join(f"{k}. {v}" for k, v in options.items()) + "\n"
    context = context.replace("<image>", "")
    context = re.sub(r"<image \d+>", "", context)
    context = re.sub(r"<image\d+>", "", context)
    return INSTRUCTION + "\n" + context


def write_locked(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"refusing changed prepared artifact: {path}")
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)


def jsonl_bytes(rows):
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()


def prepare():
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from training.prompting_utils import UniversalPrompting

    parquet = ROOT / "data/mmmu_reference/hf/data/validation-00000-of-00001.parquet"
    if sha256(parquet) != DATASET_SHA256:
        raise ValueError("pinned parquet hash mismatch")
    metadata = json.loads((ROOT / "data/mmmu_reference/hf_metadata.json").read_text())
    if metadata["revision"] != DATASET_REVISION or metadata["sha256"] != DATASET_SHA256:
        raise ValueError("HF metadata differs from pinned data")
    remote_tokenizer = REFERENCE / "source/models/mmada/tokenizer_config.json"
    if sha256(CHECKPOINT / "tokenizer_config.json") != sha256(remote_tokenizer):
        raise ValueError("tokenizer differs from reference")
    remote_transform = REFERENCE / "source/upstream/mmada/training/utils.py"
    if sha256(ROOT / "training/utils.py") != sha256(remote_transform):
        raise ValueError("official preprocessing source differs")
    model_source = REFERENCE / "source/src/cdapc_lab/models/mmada.py"
    assignments = [node for node in ast.parse(model_source.read_text()).body
                   if isinstance(node, ast.Assign) and any(
                       isinstance(t, ast.Name) and t.id == "OFFICIAL_THINKING_INSTRUCTION"
                       for t in node.targets)]
    assert ast.literal_eval(assignments[0].value) == INSTRUCTION
    # Only the supplied pure text/selection helpers are loaded for comparison.
    helper_path = REFERENCE / "source/src/cdapc_lab/data/mmmu.py"
    spec = importlib.util.spec_from_file_location("_reference_mmmu_input", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True)
    UniversalPrompting(tokenizer, use_reserved_token=True)
    rows = pq.read_table(parquet).to_pylist()
    assert len(rows) == 900 and len({r["id"] for r in rows}) == 900
    samples, gold = [], []
    counts = collections.Counter()
    for row in rows:
        assert row["id"].startswith("validation_")
        choices = ast.literal_eval(row["options"]) if row["question_type"] == "multiple-choice" else []
        options = {ascii_uppercase[i]: str(v) for i, v in enumerate(choices)}
        source_text = row["question"] + "\n" + "\n".join(f"{k}. {v}" for k, v in options.items())
        chosen = image_key(row["question"], options)
        assert chosen == helper.choose_image(source_text)
        content = user_content(row["question"], options)
        assert content == INSTRUCTION + "\n" + helper.mmmu_context(row["question"], options)
        actual = chosen if isinstance(row[chosen], dict) else "image_1"
        payload = row[actual]["bytes"]
        image_sha = hashlib.sha256(payload).hexdigest()
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            extension = {"JPEG": "jpg", "PNG": "png", "GIF": "gif", "WEBP": "webp"}.get(image.format, "bin")
        image_path = DATA / "images" / f"{image_sha}.{extension}"
        write_locked(image_path, payload)
        n_images = sum(isinstance(row[f"image_{i}"], dict) and bool(row[f"image_{i}"].get("bytes"))
                       for i in range(1, 8))
        subject = re.fullmatch(r"validation_(.+)_\d+", row["id"]).group(1)
        text_ids = tokenizer.apply_chat_template([{"role": "user", "content": content}],
                                                tokenize=True, add_generation_prompt=True)
        samples.append({
            "id": row["id"], "dataset": "MMMU", "split": "validation",
            "question": content, "original_question": row["question"], "options": options,
            "question_type": row["question_type"], "subject": subject,
            "image": str(image_path), "image_sha256": image_sha,
            "source_image_count": n_images, "selected_image": actual,
            "prompt_id": "mmmu_wino_reasoning_v1", "text_token_count": len(text_ids),
            "expected_prompt_tokens": 1027 + len(text_ids),
        })
        gold.append({"id": row["id"], "answer": str(row["answer"]),
                     "question_type": row["question_type"], "subject": subject})
        counts[row["question_type"]] += 1
        counts["multi_image_rows"] += n_images > 1
        counts[f"selected:{actual}"] += 1
    assert counts["multi_image_rows"] == 43 and counts["selected:image_1"] == 900
    write_locked(DATA / "samples.jsonl", jsonl_bytes(samples))
    write_locked(DATA / "gold.jsonl", jsonl_bytes(gold))
    shards = []
    for index in range(2):
        selected = samples[index::2]
        assert len(selected) == 450
        path = DATA / f"shard_{index}.jsonl"
        write_locked(path, jsonl_bytes(selected))
        shards.append({"shard": index, "rows": 450, "path": str(path), "sha256": sha256(path)})
    manifest = {
        "reference_repo": "HostHarry/CD", "reference_commit": REFERENCE_COMMIT,
        "reference_readme_sha256": sha256(REFERENCE / "README.md"),
        "dataset_repo": "lmms-lab/MMMU", "dataset_revision": DATASET_REVISION,
        "parquet_sha256": DATASET_SHA256, "selected_rows": 900, "counts": dict(counts),
        "prompt_id": "mmmu_wino_reasoning_v1", "instruction": INSTRUCTION,
        "input_policy": "reference single selected image; all 43 multi-image rows retained",
        "reference_text_and_image_selection_matches": 900,
        "tokenizer_config_sha256": sha256(remote_tokenizer),
        "image_transform_sha256": sha256(remote_transform),
        "samples_sha256": sha256(DATA / "samples.jsonl"), "gold_sha256": sha256(DATA / "gold.jsonl"),
        "sharding": "original parquet row order, indices shard::2", "shards": shards,
        "max_planned_sequence_tokens": max(s["expected_prompt_tokens"] for s in samples) + 256,
        "longest_sample": max(samples, key=lambda s: s["expected_prompt_tokens"])["id"],
        "preparer_sha256": sha256(__file__),
    }
    write_locked(DATA / "manifest.json", (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode())
    print(json.dumps({k: manifest[k] for k in ("selected_rows", "counts", "max_planned_sequence_tokens",
                                             "longest_sample", "reference_text_and_image_selection_matches")}))


if __name__ == "__main__":
    prepare()
