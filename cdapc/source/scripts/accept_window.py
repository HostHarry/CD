"""Independent fixed/ccaw_v1 comparison after the fixed framework gate."""
from pathlib import Path
from dataclasses import replace, asdict
import argparse
import copy
import json
import yaml
from cdapc_lab.config import resolve_config, parse_cd, AdaptiveWindow
from cdapc_lab.models.base import load_adapter
from cdapc_lab.experiments.infer import infer, preflight
from cdapc_lab.experiments.score import score_run, summarize
from cdapc_lab.analysis.paired import compare
from cdapc_lab.decoding.engine import decode_cdapc
from cdapc_lab.io.artifacts import atomic_json, read_json, safe_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["mmada", "lavida"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if not read_json(root / "reports" / f"{args.model}_acceptance.json")["fixed_acceptance_passed"]:
        raise ValueError("Fixed-window acceptance gate is not satisfied")
    fixed_path = root / "configs/experiments" / f"{args.model}_m3cot_fixed_smoke.yaml"
    raw = yaml.safe_load(fixed_path.read_text())
    adaptive_raw = copy.deepcopy(raw)
    adaptive_raw["run"]["name"] = f"{args.model}_m3cot_ccaw_v1_smoke"
    adaptive_raw["cdapc"]["window"] = dict(kind="ccaw_v1", initial=16, minimum=16, maximum=64, ema_decay=.8, expand_step=4, shrink_step=8)
    adaptive_path = fixed_path.with_name(f"{args.model}_m3cot_ccaw_v1_smoke.yaml")
    adaptive_path.write_text(yaml.safe_dump(adaptive_raw, sort_keys=False))
    configs = {"fixed": resolve_config(fixed_path), "ccaw_v1": resolve_config(adaptive_path)}
    adapter = load_adapter(configs["fixed"])
    samples, _, _ = preflight(configs["fixed"])
    prepared = adapter.prepare(samples[0])
    # Warm-up is separate from the timed sample runs.
    from cdapc_lab.decoding.state import initialize_response
    adapter.forward_pair(prepared, initialize_response(prepared, parse_cd(configs["fixed"]["cdapc"])))
    result = {"model": args.model, "scope": "two fixed smoke IDs, L=128, only window controller differs", "runs": {}}
    for kind, cfg in configs.items():
        run = infer(cfg, adapter=adapter)
        evaluation = score_run(run)
        result["runs"][kind] = {"run": str(run), **evaluation, "summary": summarize(run, evaluation["evaluation_fingerprint"])}
    fixed, adaptive = (result["runs"][key] for key in ("fixed", "ccaw_v1"))
    result["paired"] = compare(fixed["run"], fixed["evaluation_fingerprint"], adaptive["run"], adaptive["evaluation_fingerprint"])
    degeneration = replace(parse_cd(configs["fixed"]["cdapc"]), window=AdaptiveWindow(initial=16, minimum=16, maximum=16))
    raw_degenerate = decode_cdapc(adapter, prepared, degeneration, snapshot=False)
    reference = read_json(Path(fixed["run"]) / "predictions" / (safe_id(samples[0].sample_id) + ".json"))
    result["real_fixed_degeneracy"] = {"sample_id": samples[0].sample_id,
        "tokens_equal": reference["raw_response_ids"] == raw_degenerate.response_ids.tolist(),
        "cost_equal": reference["cost"] == asdict(raw_degenerate.cost)}
    if not all(result["real_fixed_degeneracy"][k] for k in ("tokens_equal", "cost_equal")):
        raise RuntimeError("Real fixed-range CCAW degeneration failed")
    updates = [row for sample in samples for row in read_json(Path(adaptive["run"]) / "predictions" / (safe_id(sample.sample_id) + ".json"))["metrics"]["window_updates"]]
    result["observed_window_range"] = [min(r["previous_width"] for r in updates), max(r["previous_width"] for r in updates)]
    result["pressure_range"] = [min(r["pressure"] for r in updates), max(r["pressure"] for r in updates)]
    result["passed"] = True
    destination = root / "reports" / f"{args.model}_ccaw_v1.json"
    atomic_json(destination, result)
    print(json.dumps({"report": str(destination), "passed": True}), flush=True)


if __name__ == "__main__":
    main()
