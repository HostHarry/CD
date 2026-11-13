from pathlib import Path
import copy
import yaml
import pytest
from cdapc_lab.config import resolve_config, parse_cd

ROOT = Path(__file__).parents[2]


def write(tmp_path, value):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(value))
    return path


def config():
    return yaml.safe_load((ROOT / "configs/experiments/mmada_m3cot_fixed_smoke.yaml").read_text())


def test_resolved_bindings_and_defaults(tmp_path):
    value = config()
    resolved = resolve_config(write(tmp_path, value))
    assert resolved["derived"] == {"gain_used": "measured", "gate_target": "contrast", "write_target": "contrast"}
    assert parse_cd({"r_gate": {"enabled": False}}).r_gate.threshold == .9


@pytest.mark.parametrize("mutation", [
    lambda c: c.update(method="unknown"),
    lambda c: c.update(native={"steps": 32}),
    lambda c: c["cdapc"].update(temperature=0),
    lambda c: c["model"].update(hidden_cfg=2),
    lambda c: c["cdapc"].update(iteration_limit=1),
    lambda c: c["cdapc"].update(branch_execution="paired_batch"),
    lambda c: c["recording"].update(trace_level="sampled"),
    lambda c: c["protocol"].update(canvas_length=True),
])
def test_unused_and_invalid_fields_fail(tmp_path, mutation):
    value = config()
    mutation(value)
    with pytest.raises(ValueError):
        resolve_config(write(tmp_path, value))
