"""Preregister sixteen hash-selected cases before mechanism generation."""
from prepare_assets import prepare_data, ROOT
from huggingface_hub import HfApi
import yaml

prepare_data(HfApi(), count=16, name="m3cot_mechanism16")
path = ROOT / "configs/experiments/mmada_m3cot_fixed_smoke.yaml"
config = yaml.safe_load(path.read_text())
config["run"].update(name="mmada_mechanism16_L512_v1", sample_ids_file="manifests/m3cot_mechanism16_ids.json")
config["protocol"].update(canvas_length=512, data_root="data/m3cot_mechanism16")
config["cdapc"]["iteration_limit"] = 512
config["recording"].update(trace_level="sampled", trace_sample_ids_file="manifests/m3cot_trace_ids.json")
(path.parent / "mmada_mechanism16.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
