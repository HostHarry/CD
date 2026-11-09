"""Run only full+fixed and the preregistered single content swap, without tuning."""
from pathlib import Path
import json
from cdapc_lab.config import resolve_config
from cdapc_lab.models.base import load_adapter
from cdapc_lab.experiments.infer import infer
from cdapc_lab.experiments.score import score_run, summarize
from cdapc_lab.experiments.replay import replay
from cdapc_lab.io.artifacts import atomic_json

root = Path(__file__).resolve().parents[1]
config = resolve_config(root / "configs/experiments/mmada_mechanism16.yaml")
adapter = load_adapter(config)
run = infer(config, adapter=adapter)
evaluation = score_run(run)
report = {"run": str(run), **evaluation, "summary": summarize(run, evaluation["evaluation_fingerprint"])}
atomic_json(root / "reports/mechanism16.json", report)
report["replay"] = replay(run, adapter=adapter)
atomic_json(root / "reports/mechanism16.json", report)
print(json.dumps(report["replay"], indent=2), flush=True)
