from dataclasses import asdict
import math
import torch
from cdapc_lab.types import ComputeCost, RawGeneration
from .proposal import make_proposals
from .gates import evaluate_gates
from .selection import choose_commits
from .state import initialize_response, apply_commit, update_observation_history
from .window import choose_window, compute_next_window_state


def tensor_record(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: tensor_record(v) for k, v in value.items()}
    return value


def commit_entries(iteration, decision, stats, pair, prepared, config):
    allowed = prepared.allowed_text_vocab.sort().values
    entries = []
    for j in range(decision.positions.numel()):
        p = int(decision.positions[j])
        lp = pair.visual.response_logits[0, p, allowed].float().log_softmax(-1)
        la = pair.ablated.response_logits[0, p, allowed].float().log_softmax(-1)
        pool = lp >= lp.max() + math.log(config.beta)
        top5 = [{"id": int(allowed[i]), "logp_visual": float(lp[i]), "logp_ablated": float(la[i]),
                 "in_pool": bool(pool[i])} for i in lp.argsort(descending=True)[:5]]
        entries.append({"iteration": iteration, "position": p, "reason": decision.reason,
                        "written_id": int(decision.written_ids[j]), "base_id": int(decision.base_ids[j]),
                        "contrast_id": int(decision.contrast_ids[j]),
                        "logp_visual_base": float(stats.logp_b[p]), "logp_visual_contrast": float(stats.logp_c[p]),
                        "gain_base": float(stats.gain_b[p]), "gain_contrast": float(stats.gain_c[p]),
                        "pool_mass": float(stats.pool_mass[p]), "pool_size": float(stats.pool_size[p]),
                        "q_base": float(stats.q_b[p]), "q_contrast": float(stats.q_c[p]),
                        "r_base": float(stats.r_b[p]), "r_contrast": float(stats.r_c[p]),
                        "h_margin": float(stats.h_margin[p]), "top5": top5})
    return entries


@torch.inference_mode()
def decode_cdapc(adapter, prepared, config, *, trace=False, snapshot=True, initial_state=None, commit_log=False):
    if prepared.mask_id in prepared.allowed_text_vocab or not set(prepared.eos_ids).issubset(set(prepared.allowed_text_vocab.tolist())):
        raise ValueError("Legal text vocabulary must exclude MASK and retain EOS")
    state = initialize_response(prepared, config) if initial_state is None else initial_state.clone()
    cost, traces, first_flip, commits_log = ComputeCost(), [], None, []
    metrics = dict(observed=0, proposal_flips=0, eligible=0, qualified_flips=0,
                   committed=0, actual_switches=0, fallback_iterations=0)
    if config.window.kind == "ccaw_v1":
        metrics["window_updates"] = []
    while state.pending.any():
        if state.iteration >= min(config.iteration_limit, prepared.canvas_length):
            state.finish_reason = "budget_exhausted"
            break
        pair = adapter.forward_pair(prepared, state.clone(), execution=config.branch_execution, cache="disabled")
        stats = make_proposals(pair.visual.response_logits, pair.ablated.response_logits, prepared.allowed_text_vocab, config)
        positions = choose_window(state)
        gates = evaluate_gates(stats, positions, config)
        decision = choose_commits(stats, gates, positions, config)
        flips = stats.base_ids != stats.contrast_ids
        selected_flips = (decision.contrast_ids != decision.base_ids).nonzero(as_tuple=True)[0]
        if snapshot and first_flip is None and config.variant == "full" and config.window.kind == "fixed" and decision.reason == "qualified" and selected_flips.numel():
            j = int(selected_flips[0])
            first_flip = dict(state=tensor_record(asdict(state)), decision=tensor_record(asdict(decision)),
                              target_position=int(decision.positions[j]), base_id=int(decision.base_ids[j]),
                              contrast_id=int(decision.contrast_ids[j]), prefix_cost=asdict(cost),
                              input_fingerprint=prepared.input_fingerprint, continuation_policy="full+fixed",
                              rng_policy="deterministic_eval_argmax_no_rng")
        metrics["observed"] += positions.numel()
        metrics["proposal_flips"] += int(flips[positions].sum())
        metrics["eligible"] += decision.eligible_count
        metrics["qualified_flips"] += int(flips[positions[gates.eligible]].sum())
        metrics["committed"] += decision.positions.numel()
        metrics["actual_switches"] += int((decision.written_ids != decision.base_ids).sum())
        metrics["fallback_iterations"] += decision.reason == "fallback"
        cost += pair.execution_cost
        next_window = compute_next_window_state(state, stats, gates, positions, config.window)
        if config.window.kind == "ccaw_v1":
            metrics["window_updates"].append({"iteration": state.iteration, **next_window})
        if trace:
            traces.append(dict(iteration=state.iteration, response_ids=state.response_ids.tolist(),
                               pending_count=int(state.pending.sum()), positions=positions.tolist(), requested_width=state.window_width,
                               proposals={k: tensor_record(v[positions] if isinstance(v, torch.Tensor) else v) for k, v in vars(stats).items()},
                               gates=tensor_record(asdict(gates)), decision=tensor_record(asdict(decision)),
                               cost=asdict(pair.execution_cost), cumulative_cost=asdict(cost), window_update=next_window))
        if commit_log:
            commits_log.extend(commit_entries(state.iteration, decision, stats, pair, prepared, config))
        state = apply_commit(state, decision, prepared)
        state = update_observation_history(state, stats, positions)
        state.window_width, state.pressure_ema = next_window["next_width"], next_window["ema"]
        state.iteration += 1
        if config.stop_rule == "eos_prefix_complete":
            eos_positions = (~state.pending) & torch.isin(
                state.response_ids, torch.tensor(list(prepared.eos_ids), device=state.response_ids.device))
            if eos_positions.any():
                first_eos = int(eos_positions.nonzero().flatten()[0])
                if not state.pending[:first_eos].any():
                    state.finish_reason = "eos_prefix_complete"
                    break
    if not state.pending.any():
        state.finish_reason = "canvas_filled"
    return RawGeneration(state.response_ids, state.finish_reason, state.iteration, cost, metrics, first_flip, traces, commits_log)
