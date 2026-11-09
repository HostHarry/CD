"""Create frozen small-sample configurations after official assets are verified."""
from pathlib import Path
import json
import yaml

root = Path(__file__).resolve().parents[1]
lock = json.loads((root / "upstream.lock.json").read_text())
destination = root / "configs/experiments"
destination.mkdir(parents=True, exist_ok=True)
for model in ("mmada", "lavida"):
    vision = "magvitv2" if model == "mmada" else "siglip"
    for variant in ("native", "full", "gain_zero", "timing_only_crossed", "base_candidate_visual"):
        cfg = {
            "schema_version": 1,
            "run": {"name": f"{model}_m3cot_{variant}_smoke_v1", "seed": 20261105, "replicate_id": 0,
                    "sample_ids_file": "manifests/m3cot_smoke_ids.json", "output_root": "runs", "fail_on_error": True},
            "model": {"backend": model, "checkpoint": f"models/{model}",
                      "checkpoint_revision": lock["models"][model]["revision"],
                      "tokenizer_revision": lock["models"][model]["revision"],
                      "upstream_source_revision": lock["sources"][model]["commit"],
                      "vision_component_revision": lock["models"][vision]["revision"],
                      "vision_checkpoint": f"models/{vision}", "upstream_root": f"upstream/{model}",
                      "dtype": "bfloat16", "attention_backend": "sdpa", "attention_semantics": "bidirectional"},
            "runtime": {"device": "cuda:0", "sample_batch_size": 1},
            "protocol": {"task": "m3cot", "dataset_revision": lock["dataset"]["revision"], "data_root": "data/m3cot_smoke",
                         "split": "test", "prompt_id": "m3cot_cot_final_v1", "canvas_length": 128,
                         "normalizer_id": "first_eos_v1", "strict_parser_id": "strict_final_v1",
                         "official_scorer_revision": lock["sources"]["m3cot"]["commit"],
                         "scorer_profile": "official_and_strict", "primary_metric": "official"},
            "method": "native" if variant == "native" else "cdapc",
            "recording": {"trace_level": "off", "snapshot_first_eligible_flip": True},
        }
        if variant == "native":
            cfg["native"] = {"steps": 128, "block_length": 32, "temperature": 0.0, "remasking": "low_confidence", "cfg_scale": 0.0, "cache": "disabled"}
        else:
            cfg["cdapc"] = {"variant": variant, "alpha": .5 if model == "mmada" else .25, "beta": .1,
                             "u_gate": {"enabled": True, "threshold": .1}, "r_gate": {"enabled": True, "threshold": .9},
                             "window": {"kind": "fixed", "width": 16}, "max_commit_per_iteration": 16,
                             "fallback": {"position_rule": "normalized_gate_readiness", "token_rule": "base_top1"},
                             "branch_execution": "sequential", "cache": "disabled", "stop_rule": "fill_canvas", "iteration_limit": 128}
        name = f"{model}_native_smoke" if variant == "native" else f"{model}_m3cot_{'fixed' if variant == 'full' else variant}_smoke"
        (destination / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
print(destination)
