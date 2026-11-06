from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

from PIL import Image
import torch
from transformers import AutoTokenizer

from models import MAGVITv2, MMadaModelLM
from models.degap import DEGAPConfig, synchronize
from training.prompting_utils import UniversalPrompting
from training.utils import image_transform, image_transform_squash


REVISION = "3cdb8709635beb29f66e4e23cd66060872ac3996"
ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_hashes():
    paths = [*ROOT.joinpath("models").glob("*.py"),
             *ROOT.joinpath("evaluation/degap").glob("*.py"),
             ROOT / "training/prompting_utils.py", ROOT / "training/utils.py"]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def environment():
    packages = {}
    for name in ("torch", "torchvision", "transformers", "diffusers", "accelerate",
                 "numpy", "safetensors", "tokenizers", "omegaconf"):
        packages[name] = importlib.metadata.version(name)
    return {"python": platform.python_version(), "packages": packages,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}


def method_config(name, alpha=1.0, lens="native", query_scope="all", record_entropy=False):
    if name == "native":
        return None
    options = dict(alpha=alpha, lens=lens, query_scope=query_scope, record_entropy=record_entropy)
    if name.startswith("fixed"):
        return asdict(DEGAPConfig(mode="fixed_b", fixed_boundary=int(name[5:]), **options))
    return asdict(DEGAPConfig(mode=name, **options))


class Runtime:
    def __init__(self, checkpoint, vq_checkpoint, device="cuda", resolution=512, resize="crop"):
        self.device = torch.device(device)
        if self.device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(self.device)
            if free < 22 * 1024 ** 3:
                raise RuntimeError("less than 22 GiB GPU memory free; refusing to contend with another job")
        self.checkpoint = Path(checkpoint)
        self.vq_checkpoint = Path(vq_checkpoint)
        self.resolution, self.resize = resolution, resize
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        self.prompting = UniversalPrompting(self.tokenizer, use_reserved_token=True)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = MMadaModelLM.from_pretrained(
            checkpoint, local_files_only=True, torch_dtype=dtype).to(self.device).eval()
        self.vq = MAGVITv2.from_pretrained(vq_checkpoint, local_files_only=True).to(self.device).eval()
        self.model.requires_grad_(False)
        self.vq.requires_grad_(False)
        if len(self.tokenizer) + self.model.config.codebook_size > self.model.config.vocab_size:
            raise ValueError("VQ code range exceeds model vocabulary")

    @torch.inference_mode()
    def prepare(self, sample):
        image_path = Path(sample["image"])
        actual_sha = sha256(image_path)
        if sample.get("image_sha256") and sample["image_sha256"] != actual_sha:
            raise ValueError(f"image changed: {image_path}")
        with Image.open(image_path) as image:
            transform = image_transform if self.resize == "crop" else image_transform_squash
            pixels = transform(image.convert("RGB"), resolution=self.resolution).unsqueeze(0).to(self.device)
        codes = self.vq.get_code(pixels) + len(self.tokenizer)
        messages = sample["messages"] if "messages" in sample else [{"role": "user", "content": sample["question"]}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        special = self.prompting.sptids_dict
        ids = [int(special["<|mmu|>"]), int(special["<|soi|>"])]
        ids += codes[0].tolist() + [int(special["<|eoi|>"])] + text
        tokens = torch.tensor([ids], device=self.device, dtype=torch.long)
        visual = list(range(2, 2 + codes.shape[1]))
        provenance = {
            "image_sha256": actual_sha, "prompt_ids": ids,
            "visual_positions": visual, "image_code_offset": len(self.tokenizer),
            "resolution": self.resolution, "resize": self.resize,
            "prompt_protocol": "official_inference_mmu_chat_template",
        }
        return tokens, visual, provenance

    def decode(self, generated):
        ids = generated.tolist()
        eos = self.model.config.eos_token_id
        stop = ids.index(eos) if eos in ids else len(ids)
        return {
            "generated_ids": ids,
            "raw_text": self.tokenizer.decode(ids, skip_special_tokens=False),
            "text": self.tokenizer.decode(ids[:stop], skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False).strip(),
            "answer_token_length": stop,
            "eos_found": stop < len(ids),
            "remaining_masks": ids.count(self.model.config.mask_token_id),
            "image_token_count": sum(len(self.tokenizer) <= t < len(self.tokenizer) +
                                     self.model.config.codebook_size for t in ids),
            "special_token_count": sum(t in self.tokenizer.all_special_ids for t in ids),
        }

    @torch.inference_mode()
    def generate(self, sample, config, generation, seed=42):
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
            torch.cuda.reset_peak_memory_stats(self.device)
        synchronize(self.device)
        start = time.perf_counter()
        tokens, visual, provenance = self.prepare(sample)
        synchronize(self.device)
        prepared = time.perf_counter()
        trace = []
        output = self.model.mmu_generate(
            tokens, **generation, mask_id=self.model.config.mask_token_id,
            visual_positions=visual, degap_config=config, trace=trace)
        synchronize(self.device)
        decoded = time.perf_counter()
        result = self.decode(output[0, tokens.shape[1]:])
        result.update({
            "input": provenance, "trace": trace,
            "preprocessing_seconds": prepared - start,
            "decode_seconds": decoded - prepared,
            "end_to_end_seconds": time.perf_counter() - start,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device) if self.device.type == "cuda" else None,
        })
        return result
