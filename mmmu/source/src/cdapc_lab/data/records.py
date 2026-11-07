from dataclasses import asdict
from pathlib import Path
from string import ascii_uppercase
from PIL import Image
from cdapc_lab.types import Sample, GoldRecord
from cdapc_lab.io.artifacts import read_json, file_hash, digest

PROMPT_TEMPLATE = "{context}{question}\n{options}\nThink step by step, then end with Final answer: (X), using exactly one option letter."

TASKS = {
    "m3cot": {"template": PROMPT_TEMPLATE, "prompt_id": "m3cot_cot_final_v1", "model_reasoning_instruction": True},
    "pope": {"template": "{question} Please answer yes or no.", "prompt_id": "pope_yesno_v1", "model_reasoning_instruction": False},
    "mmmu": {"template": "mmmu_wino_v1", "prompt_id": "mmmu_wino_reasoning_v1", "model_reasoning_instruction": True},
}


def build_question(sample: Sample):
    if not isinstance(sample, Sample):
        raise TypeError("Model input builder accepts only Sample")
    if sample.task_id not in TASKS:
        raise ValueError(f"Unknown task: {sample.task_id}")
    if sample.task_id == "mmmu":
        from .mmmu import mmmu_context
        return mmmu_context(sample.question, sample.options)
    return TASKS[sample.task_id]["template"].format(
        context=(sample.public_context + "\n") if sample.public_context else "",
        question=sample.question,
        options="\n".join(f"({k}) {v}" for k, v in sample.options.items()))


def uses_model_reasoning_instruction(sample: Sample):
    if not isinstance(sample, Sample):
        raise TypeError("Model input builder accepts only Sample")
    if sample.task_id not in TASKS:
        raise ValueError(f"Unknown task: {sample.task_id}")
    return TASKS[sample.task_id]["model_reasoning_instruction"]


def load_manifest(data_root, ids_file, revision, split, task="m3cot"):
    root = Path(data_root)
    provenance = read_json(root / "provenance.json")
    if provenance["revision"] != revision or provenance["split"] != split:
        raise ValueError("Dataset revision/split mismatch")
    ids = read_json(ids_file)
    if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)):
        raise ValueError("Manifest IDs must be nonempty and unique")
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    records = read_json(root / "samples.json")
    if len({r["sample_id"] for r in records}) != len(records):
        raise ValueError("Duplicate dataset IDs")
    by_id = {r["sample_id"]: r for r in records}
    samples, entries = [], []
    for sample_id in ids:
        record = dict(by_id[sample_id])
        record["image_paths"] = tuple(str(root / p) for p in record["image_paths"])
        sample = Sample(**record)
        if sample.task_id != task:
            raise ValueError(f"Sample task {sample.task_id} does not match configured task {task}")
        if len(sample.image_paths) != 1:
            raise ValueError("Initial model protocol requires exactly one image")
        if task == "m3cot":
            if not sample.options or list(sample.options) != list("ABCDEF"[:len(sample.options)]):
                raise ValueError("Expected ordered contiguous MCQ options")
        elif task == "mmmu":
            if sample.options and list(sample.options) != list(ascii_uppercase[:len(sample.options)]):
                raise ValueError("MMMU options must be {} or contiguous letters A..")
        elif sample.options != {}:
            raise ValueError("POPE samples must have empty options")
        for path in sample.image_paths:
            with Image.open(path) as image:
                image.verify()
        content = asdict(sample)
        content["image_paths"] = [file_hash(p) for p in sample.image_paths]
        entries.append({"sample_id": sample_id, "group_id": sample.group_id, "content_hash": digest(content)})
        samples.append(sample)
    return samples, {"revision": revision, "split": split, "samples": entries,
                     "prompt_hash": digest(TASKS[task]["template"]), "source_provenance": provenance}


def load_gold(data_root, ids):
    rows = read_json(Path(data_root) / "gold.json")
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate gold IDs")
    by_id = {row["sample_id"]: GoldRecord(**row) for row in rows}
    return {sample_id: by_id[sample_id] for sample_id in ids}
