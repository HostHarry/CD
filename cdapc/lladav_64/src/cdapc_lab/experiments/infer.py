from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
import random
import platform
import time
import traceback
import fcntl
import numpy as np
import torch
from cdapc_lab.config import parse_cd
from cdapc_lab.data.records import load_manifest
from cdapc_lab.decoding.engine import decode_cdapc
from cdapc_lab.models.base import load_adapter
from cdapc_lab.io.artifacts import (atomic_json, read_json, digest, file_hash, source_hash,
                                   generation_projection, git_provenance, safe_id,
                                   sample_seed, reusable_prediction)

ROOT = Path(__file__).resolve().parents[3]
GENERATION_SOURCES = ["types.py", "config.py", "models/base.py", "data", "io", "experiments/infer.py"]
VISION_ASSET = {"mmada": "magvitv2", "lavida": "siglip", "lladav": "siglip2"}


def environment(device):
    import transformers
    result = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
              "transformers": transformers.__version__, "platform": platform.platform(), "device": device}
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA device unavailable")
        props = torch.cuda.get_device_properties(device)
        result.update(gpu=props.name, gpu_memory=props.total_memory, capability=list(torch.cuda.get_device_capability(device)))
    return result


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_lock(config, lock):
    if "PIN_REQUIRED" in json.dumps(config) or "PIN_REQUIRED" in json.dumps(lock):
        raise ValueError("Unresolved PIN_REQUIRED")
    name = config["model"]["backend"]
    model = lock["models"][name]
    vision = lock["models"][VISION_ASSET[name]]
    expected = {"checkpoint_revision": model["revision"], "tokenizer_revision": model["revision"],
                "vision_component_revision": vision["revision"], "upstream_source_revision": lock["sources"][name]["commit"]}
    for key, value in expected.items():
        if config["model"][key] != value:
            raise ValueError(f"Version lock mismatch: model.{key}")
    for key, asset in (("checkpoint", model), ("vision_checkpoint", vision)):
        folder = Path(config["model"][key])
        for filename, expected_hash in asset["files"].items():
            path = folder / filename
            if not path.is_file():
                raise ValueError(f"Missing model file: {path}")
            # Large binaries were hashed against official LFS in prepare_assets.
            # Re-hash on metadata change; small config/tokenizer files always hash.
            stamp = asset.get("verified_stats", {}).get(filename)
            current = {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            if not filename.endswith(".safetensors") or stamp != current:
                if file_hash(path) != expected_hash:
                    raise ValueError(f"Model content hash mismatch: {path}")
                asset.setdefault("verified_stats", {})[filename] = current
    scorer_key = config["protocol"]["task"]
    for key in (name, scorer_key):
        actual = source_hash(ROOT / "upstream" / key, ["."])
        if actual != lock["sources"][key]["source_hash"]:
            raise ValueError(f"Official source/patch changed: {key}")
    if config["protocol"]["official_scorer_revision"] != lock["sources"][scorer_key]["commit"]:
        raise ValueError("Official scorer revision mismatch")


def preflight(config):
    lock = read_json(ROOT / "upstream.lock.json")
    validate_lock(config, lock)
    atomic_json(ROOT / "upstream.lock.json", lock)
    p, r = config["protocol"], config["run"]
    samples, manifest = load_manifest(p["data_root"], r["sample_ids_file"], p["dataset_revision"], p["split"], p["task"], p["prompt_id"])
    env = environment(config["runtime"]["device"])
    output = Path(r["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    if not os.access(output, os.W_OK):
        raise ValueError("Output root not writable")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["checkpoint"], local_files_only=True)
    if tokenizer.eos_token_id is None or tokenizer.convert_tokens_to_ids("<|mdm_mask|>") in (None, tokenizer.unk_token_id):
        raise ValueError("Official EOS/MASK token IDs missing")
    dependencies = GENERATION_SOURCES + [f"models/{config['model']['backend']}.py"]
    if config["method"] == "cdapc":
        dependencies.append("decoding")
    code_hash = source_hash(ROOT / "src/cdapc_lab", dependencies)
    selected_lock = {"source": lock["sources"][config["model"]["backend"]],
                     "model_files": lock["models"][config["model"]["backend"]]["files"],
                     "vision_files": lock["models"][VISION_ASSET[config["model"]["backend"]]]["files"]}
    fingerprint = digest({"generation_config": generation_projection(config), "manifest": manifest,
                          "sources": code_hash, "assets": selected_lock, "runtime": env})
    return samples, manifest, {"generation_fingerprint": fingerprint, "generation_source_hash": code_hash,
                                "environment": env, "assets": selected_lock, "git": git_provenance(ROOT)}


def infer(config, resume_run=None, adapter=None, max_samples=None):
    cd_config = parse_cd(config["cdapc"]) if config["method"] == "cdapc" else None
    samples, manifest, provenance = preflight(config)
    fingerprint = provenance["generation_fingerprint"]
    if resume_run:
        run = Path(resume_run)
        previous = read_json(run / "run_manifest.json")
        if previous["generation_fingerprint"] != fingerprint:
            raise ValueError("Resume refused: generation fingerprint changed")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run = Path(config["run"]["output_root"]) / f"{stamp}_{fingerprint[:12]}"
        run.mkdir(parents=True, exist_ok=False)
        atomic_json(run / "resolved_config.json", config)
        atomic_json(run / "dataset_manifest.json", manifest)
    run_manifest = provenance | {"framework": "new_cdapc_lab_v1", "status": "running", "sample_count": len(samples),
                                 "timing": "preprocess includes image encode; generation excludes model load, I/O, scoring; synchronized CUDA"}
    lock_fd = os.open(run / ".writer.lock", os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(lock_fd)
        raise ValueError("Another writer holds this run")
    os.ftruncate(lock_fd, 0)
    os.write(lock_fd, str(os.getpid()).encode())
    atomic_json(run / "run_manifest.json", run_manifest)
    generated = 0
    try:
        for sample in samples:
            path = run / "predictions" / (safe_id(sample.sample_id) + ".json")
            if reusable_prediction(path, sample.sample_id, fingerprint):
                continue
            if max_samples is not None and generated >= max_samples:
                break
            seed = sample_seed(config["run"]["seed"], sample.sample_id, config["run"]["replicate_id"])
            record = {"sample_id": sample.sample_id, "group_id": sample.group_id, "generation_fingerprint": fingerprint,
                      "sample_seed": seed, "options": sample.options}
            try:
                if adapter is None:
                    adapter = load_adapter(config)
                set_seed(seed)
                cuda = str(config["runtime"]["device"]).startswith("cuda")
                sync = lambda: torch.cuda.synchronize(config["runtime"]["device"]) if cuda else None
                sync()
                start = time.perf_counter()
                prepared = adapter.prepare(sample)
                sync()
                preprocess = time.perf_counter() - start
                if cuda:
                    torch.cuda.reset_peak_memory_stats(config["runtime"]["device"])
                trace_config = config["recording"]
                trace_ids = read_json(trace_config["trace_sample_ids_file"]) if trace_config["trace_level"] == "sampled" else []
                trace = trace_config["trace_level"] == "full" or sample.sample_id in trace_ids
                start = time.perf_counter()
                if config["method"] == "native":
                    raw = adapter.native(prepared, config["native"])
                else:
                    raw = decode_cdapc(adapter, prepared, cd_config, trace=trace,
                                        snapshot=trace_config["snapshot_first_eligible_flip"],
                                        commit_log=trace_config.get("commit_log", False))
                sync()
                elapsed = time.perf_counter() - start
                ids = raw.response_ids.cpu().tolist()
                first_eos = next((i for i, token in enumerate(ids) if token in prepared.eos_ids), None)
                record.update(status="success" if raw.finish_reason in ("canvas_filled", "eos_prefix_complete") else "budget_exhausted",
                              raw_response_ids=ids, response_mapping="response_only", eos_ids=list(prepared.eos_ids),
                              mask_id=prepared.mask_id, finish_reason=raw.finish_reason, cost=asdict(raw.cost),
                              decode_iterations=raw.decode_iterations, metrics=raw.metrics,
                              preprocess_seconds=preprocess, generation_seconds=elapsed,
                              peak_memory_bytes=torch.cuda.max_memory_allocated(config["runtime"]["device"]) if cuda else 0,
                              input_fingerprint=prepared.input_fingerprint, preprocessing_version=prepared.preprocessing_version,
                              prompt_ids=prepared.prompt_ids.cpu().tolist(), visual_key_positions=prepared.visual_keys.nonzero().flatten().cpu().tolist(),
                              actual_sequence_length=prepared.prompt_ids.shape[1] + len(ids),
                              committed_buffer_tokens=sum(x != prepared.mask_id for x in ids),
                              commit_log=raw.commit_log,
                              emitted_tokens_before_eos=len(ids) if first_eos is None else first_eos,
                              first_eos_position=first_eos, full_buffer_length=len(ids),
                              event_status=("triggered" if raw.first_flip else "no_eligible_flip")
                              if cd_config is not None and cd_config.variant == "full" and cd_config.window.kind == "fixed" and trace_config["snapshot_first_eligible_flip"]
                              else "not_applicable")
                if raw.first_flip is not None:
                    atomic_json(run / "snapshots" / (safe_id(sample.sample_id) + ".json"), raw.first_flip)
                if raw.traces:
                    atomic_json(run / "traces" / (safe_id(sample.sample_id) + ".json"), raw.traces)
            except Exception as error:
                record.update(status="error", error=repr(error), traceback=traceback.format_exc())
                atomic_json(path, record)
                if config["run"]["fail_on_error"]:
                    raise
            atomic_json(path, record)
            generated += 1
            print(json.dumps({"run": str(run), "sample_id": sample.sample_id, "status": record["status"]}), flush=True)
        states = [read_json(run / "predictions" / (safe_id(s.sample_id) + ".json"))["status"]
                  if (run / "predictions" / (safe_id(s.sample_id) + ".json")).exists() else "missing" for s in samples]
        run_manifest.update(status="complete" if all(s == "success" for s in states) else "incomplete",
                            sample_status_counts={s: states.count(s) for s in sorted(set(states))})
    except BaseException:
        run_manifest["status"] = "interrupted"
        raise
    finally:
        atomic_json(run / "run_manifest.json", run_manifest)
        os.close(lock_fd)
    return run
