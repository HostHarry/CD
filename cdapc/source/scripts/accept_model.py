"""Bounded two-sample real-model acceptance, one model in GPU memory at a time."""
from pathlib import Path
import argparse
import json
from cdapc_lab.config import resolve_config
from cdapc_lab.models.base import load_adapter
from cdapc_lab.experiments.verify import verify
from cdapc_lab.experiments.infer import infer
from cdapc_lab.experiments.score import score_run, summarize
from cdapc_lab.experiments.replay import replay
from cdapc_lab.analysis.paired import compare
from cdapc_lab.io.artifacts import read_json, atomic_json, safe_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["mmada", "lavida"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    def config(variant):
        name = f"{args.model}_native_smoke" if variant == "native" else f"{args.model}_m3cot_{'fixed' if variant == 'full' else variant}_smoke"
        return resolve_config(root / "configs/experiments" / (name + ".yaml"))
    cfg = config("full")
    adapter = load_adapter(cfg)
    report = {"model": args.model, "scope": "two preregistered M3CoT IDs, L=128 engineering acceptance; no benchmark claim", "runs": {}}
    destination = root / "reports" / f"{args.model}_acceptance.json"
    for kind in ("adapter", "native"):
        result = verify(config("native") if kind == "native" else cfg, kind, adapter)
        report[kind] = result
        atomic_json(destination, report)
        if not result["passed"]:
            raise RuntimeError(f"{kind} verification failed: {result['report_path']}")
    for variant in ("native", "full", "gain_zero", "timing_only_crossed", "base_candidate_visual"):
        c = config(variant)
        run = infer(c, adapter=adapter, max_samples=1 if variant == "full" else None)
        if variant == "full":
            run = infer(c, run, adapter=adapter)
        evaluation = score_run(run)
        summary = summarize(run, evaluation["evaluation_fingerprint"])
        report["runs"][variant] = {"run": str(run), **evaluation, "summary": summary}
        atomic_json(destination, report)
    full, crossed = (report["runs"][v] for v in ("full", "timing_only_crossed"))
    ids = [s["sample_id"] for s in read_json(Path(full["run"]) / "dataset_manifest.json")["samples"]]
    no_switch = []
    for sid in ids:
        a = read_json(Path(full["run"]) / "predictions" / (safe_id(sid) + ".json"))
        b = read_json(Path(crossed["run"]) / "predictions" / (safe_id(sid) + ".json"))
        if a["metrics"]["actual_switches"] == 0:
            if a["raw_response_ids"] != b["raw_response_ids"] or a["cost"] != b["cost"]:
                raise RuntimeError("Full-defined no-switch subset parity failed")
            no_switch.append(sid)
    report["no_switch_subset"] = {"selection": "full actual_switches == 0", "sample_ids": no_switch,
                                   "passed": True, "nonvacuous": bool(no_switch)}
    report["full_vs_crossed"] = compare(full["run"], full["evaluation_fingerprint"], crossed["run"], crossed["evaluation_fingerprint"])
    report["replay"] = replay(full["run"], adapter=adapter)
    report["fixed_acceptance_passed"] = True
    atomic_json(destination, report)
    print(json.dumps({"report": str(destination), "passed": True}), flush=True)


if __name__ == "__main__":
    main()
