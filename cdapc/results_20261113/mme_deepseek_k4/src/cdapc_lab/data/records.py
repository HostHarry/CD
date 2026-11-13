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
    "chair": {"template": "{question}", "prompt_id": "chair_caption_v1", "model_reasoning_instruction": False},
    "mme": {"template": "{question}", "prompt_id": "mme_yesno_v1", "model_reasoning_instruction": False},
}

MME_THINK = {
    "template": "{question}",
    "prompt_id": "mme_think_v1", "model_reasoning_instruction": True,
}


POPE_REASONING = {
    "template": "{question}\nAfter </think>, end with exactly Final answer: Yes or Final answer: No.",
    "prompt_id": "pope_yesno_reasoning_v1", "model_reasoning_instruction": True,
}

POPE_THINK_NO_FINAL = {
    "template": "{question}",
    "prompt_id": "pope_think_no_final_v1", "model_reasoning_instruction": True,
}


def prompt_spec(task, prompt_id=None):
    spec = POPE_REASONING if task == "pope" and prompt_id == POPE_REASONING["prompt_id"] else TASKS[task]
    if task == "pope" and prompt_id == POPE_THINK_NO_FINAL["prompt_id"]:
        spec = POPE_THINK_NO_FINAL
    if task == "mme" and prompt_id == MME_THINK["prompt_id"]:
        spec = MME_THINK
    if prompt_id is not None and prompt_id != spec["prompt_id"]:
        raise ValueError(f"Unsupported prompt {prompt_id} for {task}")
    return spec


def build_question(sample: Sample, prompt_id=None):
    if not isinstance(sample, Sample):
        raise TypeError("Model input builder accepts only Sample")
    if sample.task_id not in TASKS:
        raise ValueError(f"Unknown task: {sample.task_id}")
    if sample.task_id == "mmmu":
        from .mmmu import mmmu_context
        return mmmu_context(sample.question, sample.options)
    return prompt_spec(sample.task_id, prompt_id)["template"].format(
        context=(sample.public_context + "\n") if sample.public_context else "",
        question=sample.question,
        options="\n".join(f"({k}) {v}" for k, v in sample.options.items()))


def uses_model_reasoning_instruction(sample: Sample, prompt_id=None):
    if not isinstance(sample, Sample):
        raise TypeError("Model input builder accepts only Sample")
    if sample.task_id not in TASKS:
        raise ValueError(f"Unknown task: {sample.task_id}")
    return prompt_spec(sample.task_id, prompt_id)["model_reasoning_instruction"]


def load_manifest(data_root, ids_file, revision, split, task="m3cot", prompt_id=None):
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
            raise ValueError("POPE/CHAIR samples must have empty options")
        for path in sample.image_paths:
            with Image.open(path) as image:
                image.verify()
        content = asdict(sample)
        content["image_paths"] = [file_hash(p) for p in sample.image_paths]
        entries.append({"sample_id": sample_id, "group_id": sample.group_id, "content_hash": digest(content)})
        samples.append(sample)
    return samples, {"revision": revision, "split": split, "samples": entries,
                     "prompt_hash": digest(prompt_spec(task, prompt_id)["template"]), "source_provenance": provenance}


def load_gold(data_root, ids):
    rows = read_json(Path(data_root) / "gold.json")
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate gold IDs")
    by_id = {row["sample_id"]: GoldRecord(**row) for row in rows}
    return {sample_id: by_id[sample_id] for sample_id in ids}
