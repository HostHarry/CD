from dataclasses import replace, asdict
from pathlib import Path
import torch
from cdapc_lab.types import DecodeState
from cdapc_lab.config import parse_cd
from cdapc_lab.decoding.engine import decode_cdapc
from cdapc_lab.decoding.proposal import make_proposals
from cdapc_lab.decoding.gates import evaluate_gates
from cdapc_lab.decoding.selection import choose_commits
from cdapc_lab.decoding.state import apply_commit, update_observation_history
from cdapc_lab.decoding.window import choose_window
from cdapc_lab.models.base import load_adapter
from cdapc_lab.io.artifacts import read_json, atomic_json, safe_id, digest, source_hash, file_hash
from cdapc_lab.data.records import load_gold
from cdapc_lab.evaluation.normalize import normalize_output
from cdapc_lab.evaluation.scorers import load_official_scorer, score_text
from cdapc_lab.analysis.paired import paired_statistics
from .infer import preflight, ROOT


def restore_state(record, device):
    values = dict(record)
    for key in ("response_ids", "pending", "previous_candidate_ids", "previous_observed"):
        values[key] = torch.tensor(values[key], device=device, dtype=torch.bool if key in ("pending", "previous_observed") else torch.long)
    return DecodeState(**values)


@torch.inference_mode()
def fork_once(adapter, prepared, cfg, event):
    state = restore_state(event["state"], prepared.prompt_ids.device)
    pair = adapter.forward_pair(prepared, state, execution=cfg.branch_execution)
    stats = make_proposals(pair.visual.response_logits, pair.ablated.response_logits, prepared.allowed_text_vocab, cfg)
    positions = choose_window(state)
    gates = evaluate_gates(stats, positions, cfg)
    decision = choose_commits(stats, gates, positions, cfg)
    if prepared.input_fingerprint != event["input_fingerprint"] or decision.reason != "qualified" or decision.positions.tolist() != event["decision"]["positions"] or decision.written_ids.tolist() != event["decision"]["written_ids"]:
        return {"status": "replay_mismatch"}
    target = event["target_position"]
    where = (decision.positions == target).nonzero().flatten()
    if where.numel() != 1 or int(stats.base_ids[target]) != event["base_id"] or int(stats.contrast_ids[target]) != event["contrast_id"]:
        return {"status": "replay_mismatch"}
    outputs, initial = {}, {}
    for name, token in (("A", event["base_id"]), ("B", event["contrast_id"])):
        written = decision.written_ids.clone()
        written[int(where[0])] = token
        fork_decision = replace(decision, written_ids=written)
        fork_state = apply_commit(state, fork_decision, prepared)
        fork_state = update_observation_history(fork_state, stats, positions)
        fork_state.iteration += 1
        initial[name] = fork_state.response_ids.tolist()
        outputs[name] = decode_cdapc(adapter, prepared, cfg, initial_state=fork_state, snapshot=False)
    differences = [i for i, (a, b) in enumerate(zip(initial["A"], initial["B"])) if a != b]
    if differences != [target]:
        raise RuntimeError("Content swap changed more than its target")
    return {"status": "valid_pair", "outputs": outputs, "initial_states": initial,
            "verification_cost": asdict(pair.execution_cost), "prefix_cost": event["prefix_cost"]}


def replay(run, event="first_eligible_flip", design="content_swap", adapter=None):
    if event != "first_eligible_flip" or design != "content_swap":
        raise ValueError("Only the preregistered first flip content-swap design is implemented")
    run = Path(run)
    config = read_json(run / "resolved_config.json")
    if config["protocol"].get("task", "m3cot") != "m3cot":
        raise ValueError("Replay is implemented for the M3CoT task only")
    if config["method"] != "cdapc" or config["cdapc"]["variant"] != "full" or config["cdapc"]["window"]["kind"] != "fixed":
        raise ValueError("Replay requires full+fixed source run")
    samples, manifest, provenance = preflight(config)
    if provenance["generation_fingerprint"] != read_json(run / "run_manifest.json")["generation_fingerprint"]:
        raise ValueError("Generation environment/source changed; replay refused")
    cfg = parse_cd(config["cdapc"])
    gold = load_gold(config["protocol"]["data_root"], [s.sample_id for s in samples])
    official_path = ROOT / "upstream/m3cot/utils/metric.py"
    official = load_official_scorer(official_path)
    study = {"event": event, "design": design, "generation_fingerprint": provenance["generation_fingerprint"],
             "replay_source_hash": source_hash(ROOT / "src/cdapc_lab", ["experiments/replay.py", "analysis/paired.py", "evaluation"]),
             "gold_hash": digest({k: asdict(v) for k, v in gold.items()}), "official_source_hash": file_hash(official_path)}
    output = run / "replays" / digest(study)
    records, answers_a, answers_b, groups = [], [], [], []
    for sample in samples:
        snapshot = run / "snapshots" / (safe_id(sample.sample_id) + ".json")
        record = {"sample_id": sample.sample_id, "status": "no_eligible_flip"}
        prediction = run / "predictions" / (safe_id(sample.sample_id) + ".json")
        if not prediction.exists() or read_json(prediction).get("status") != "success":
            record["status"] = "source_failure"
        elif snapshot.exists():
            try:
                adapter = adapter or load_adapter(config)
                prepared = adapter.prepare(sample)
                result = fork_once(adapter, prepared, cfg, read_json(snapshot))
                record.update({k: v for k, v in result.items() if k != "outputs"})
                if result["status"] == "valid_pair":
                    branches = {}
                    for name, raw in result["outputs"].items():
                        if raw.finish_reason != "canvas_filled":
                            raise RuntimeError("Replay continuation did not complete")
                        normalized = normalize_output(raw.response_ids.tolist(), adapter.tokenizer, prepared.eos_ids)
                        scores = score_text(normalized["text"], sample.options, gold[sample.sample_id].answer, official)
                        branches[name] = {"raw_response_ids": raw.response_ids.tolist(), "normalized": normalized,
                                          "scores": scores, "continuation_cost": asdict(raw.cost)}
                    record["branches"] = branches
                    answers_a.append(branches["A"]["scores"]["strict_accuracy"])
                    answers_b.append(branches["B"]["scores"]["strict_accuracy"])
                    groups.append(sample.group_id or sample.sample_id)
            except Exception as error:
                record.update(status="execution_failure", error=repr(error))
        records.append(record)
        atomic_json(output / (safe_id(sample.sample_id) + ".json"), record)
    counts = {s: sum(r["status"] == s for r in records) for s in ("valid_pair", "replay_mismatch", "execution_failure", "no_eligible_flip", "source_failure")}
    triggered = sum(counts[s] for s in ("valid_pair", "replay_mismatch", "execution_failure"))
    summary = {**study, "counts": counts, "manifest_count": len(samples),
               "triggered_events": triggered, "valid_pairs": counts["valid_pair"],
               "coverage": {"numerator": triggered, "denominator": len(samples)},
               "valid_fraction": {"numerator": counts["valid_pair"], "denominator": triggered},
               "paired_strict": paired_statistics(answers_a, answers_b, groups) if answers_a else None,
               "cost_scope": "research replay; kept separate from deployment inference"}
    atomic_json(output / "summary.json", summary)
    return summary | {"output": str(output)}
