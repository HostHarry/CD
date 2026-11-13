from dataclasses import replace
from pathlib import Path
import torch
from cdapc_lab.models.base import load_adapter, visual_ablation_bias
from cdapc_lab.config import parse_cd, CDConfig
from cdapc_lab.decoding.state import initialize_response
from cdapc_lab.decoding.proposal import make_proposals
from cdapc_lab.decoding.gates import evaluate_gates
from cdapc_lab.decoding.selection import choose_commits
from cdapc_lab.io.artifacts import atomic_json, digest
from .infer import preflight, set_seed, ROOT


def error_summary(a, b):
    delta = (a.float() - b.float()).abs()
    return {"max_logit_error": float(delta.max()), "mean_logit_error": float(delta.mean()),
            "argmax_disagreements": int((a.argmax(-1) != b.argmax(-1)).sum())}


@torch.inference_mode()
def verify(config, kind, adapter=None):
    samples, manifest, provenance = preflight(config)
    adapter = adapter or load_adapter(config)
    prepared = adapter.prepare(samples[0])
    result = {"kind": kind, "generation_fingerprint": provenance["generation_fingerprint"],
              "environment": provenance["environment"], "sample_id": samples[0].sample_id}
    if kind == "native":
        if config["method"] != "native":
            raise ValueError("verify-native requires native config")
        set_seed(20261105)
        wrapped = adapter.native(prepared, config["native"])
        set_seed(20261105)
        native = config["native"]
        if config["model"]["backend"] == "mmada":
            official_output = adapter.model.mmu_generate(prepared.prompt_ids,
                max_new_tokens=prepared.canvas_length, steps=native["steps"], block_length=native["block_length"],
                temperature=native["temperature"], cfg_scale=native["cfg_scale"], remasking=native["remasking"], mask_id=prepared.mask_id)
        else:
            official_output = adapter.model.generate(prepared.model_inputs["original_prompt_ids"],
                images=prepared.model_inputs["pixels"], image_sizes=prepared.model_inputs["image_sizes"],
                max_new_tokens=prepared.canvas_length, block_length=native["block_length"],
                step_per_block=native["steps"] // (prepared.canvas_length // native["block_length"]),
                temperature=native["temperature"], cfg_scale=native["cfg_scale"], remasking=native["remasking"],
                mask_id=prepared.mask_id, prefix_lm=False, verbose=False)
        direct = official_output[0, prepared.prompt_ids.shape[1]:]
        result.update(tokens_equal=torch.equal(wrapped.response_ids, direct),
                      response_tokens=direct.tolist(), cost=vars(wrapped.cost),
                      prompt_ids=prepared.prompt_ids.tolist(),
                      native_mapping={"total_steps": config["native"]["steps"],
                                      "block_length": config["native"]["block_length"],
                                      "step_per_block": config["native"]["steps"] // (prepared.canvas_length // config["native"]["block_length"])})
        result["passed"] = result["tokens_equal"] and wrapped.finish_reason == "canvas_filled"
    elif kind == "adapter":
        cfg = parse_cd(config["cdapc"]) if config["method"] == "cdapc" else CDConfig()
        state = initialize_response(prepared, cfg)
        with adapter.count_forwards() as measured:
            visual = adapter.forward_logits(prepared, state)
        if config["model"]["backend"] == "mmada":
            full_ids = torch.cat([prepared.prompt_ids, state.response_ids[None]], dim=1)
            official = adapter.model(full_ids).logits[:, -prepared.canvas_length:]
        else:
            full_embeddings = torch.cat([prepared.model_inputs["prompt_embeddings"], adapter.denoiser.transformer.wte(state.response_ids[None])], dim=1)
            official = adapter.model.get_model()(None, input_embeddings=full_embeddings).logits[:, -prepared.canvas_length:]
        result["normal_parity"] = error_summary(visual, official)
        del official
        seq = adapter.forward_pair(prepared, state)
        result["standalone_vs_sequential"] = error_summary(visual, seq.visual.response_logits)
        pair = adapter.forward_pair(prepared, state, execution="paired_batch")
        result["paired_visual"] = error_summary(seq.visual.response_logits, pair.visual.response_logits)
        result["paired_ablated"] = error_summary(seq.ablated.response_logits, pair.ablated.response_logits)
        positions = torch.arange(min(state.window_width, prepared.canvas_length), device=visual.device)
        seq_stats = make_proposals(seq.visual.response_logits, seq.ablated.response_logits, prepared.allowed_text_vocab, cfg)
        pair_stats = make_proposals(pair.visual.response_logits, pair.ablated.response_logits, prepared.allowed_text_vocab, cfg)
        seq_gate, pair_gate = evaluate_gates(seq_stats, positions, cfg), evaluate_gates(pair_stats, positions, cfg)
        a, b = choose_commits(seq_stats, seq_gate, positions, cfg), choose_commits(pair_stats, pair_gate, positions, cfg)
        result["paired_decisions"] = {"gate_disagreements": int((seq_gate.eligible != pair_gate.eligible).sum()),
                                      "positions_equal": torch.equal(a.positions, b.positions),
                                      "writes_equal": torch.equal(a.written_ids, b.written_ids),
                                      "min_u_threshold_distance": float((seq_stats.u_c[positions] - cfg.u_gate.threshold).abs().min()),
                                      "min_r_threshold_distance": float((seq_stats.r_c[positions] - cfg.r_gate.threshold).abs().min())}
        if config["model"]["backend"] == "mmada":
            ids = prepared.prompt_ids.clone()
            ids[:, prepared.visual_keys] = ids[:, prepared.visual_keys].roll(1, 1)
            changed = replace(prepared, prompt_ids=ids)
        else:
            inputs = dict(prepared.model_inputs)
            embeddings = inputs["prompt_embeddings"].clone()
            embeddings[:, prepared.visual_keys] = embeddings[:, prepared.visual_keys].roll(1, 1) + 0.25
            inputs["prompt_embeddings"] = embeddings
            changed = replace(prepared, model_inputs=inputs)
        mask_seen = []
        original_sdpa = torch.nn.functional.scaled_dot_product_attention
        def capture(*args, **kwargs):
            bias = kwargs.get("attn_mask")
            mask_seen.append(bias is not None and bool(torch.isneginf(bias).any() or (bias < -1e4).any()))
            return original_sdpa(*args, **kwargs)
        torch.nn.functional.scaled_dot_product_attention = capture
        try:
            changed_pair = adapter.forward_pair(changed, state)
        finally:
            torch.nn.functional.scaled_dot_product_attention = original_sdpa
        result["visual_content_change"] = error_summary(seq.visual.response_logits, changed_pair.visual.response_logits)
        result["ablated_invariance"] = error_summary(seq.ablated.response_logits, changed_pair.ablated.response_logits)
        result["kernel_mask_calls"] = sum(mask_seen)
        again = adapter.forward_logits(prepared, state)
        result["order_independence"] = error_summary(visual, again)
        result["counts"] = {"sequential": vars(seq.execution_cost), "paired": vars(pair.execution_cost)}
        result["tolerances"] = {"sequential_max_abs": 0.0, "invariance_max_abs": 0.0,
                                "paired_max_abs": 0.5, "paired_gate_disagreements": 0}
        result["sequential_passed"] = all(result[k]["max_logit_error"] == 0 for k in
                    ("normal_parity", "standalone_vs_sequential", "ablated_invariance", "order_independence")) and result["kernel_mask_calls"] > 0 and result["visual_content_change"]["max_logit_error"] > 0
        result["paired_passed"] = max(result["paired_visual"]["max_logit_error"], result["paired_ablated"]["max_logit_error"]) <= .5 and result["paired_decisions"]["gate_disagreements"] == 0 and result["paired_decisions"]["positions_equal"] and result["paired_decisions"]["writes_equal"]
        result["passed"] = result["sequential_passed"] and (cfg.branch_execution == "sequential" or result["paired_passed"])
    else:
        raise ValueError("Unknown verification")
    path = ROOT / "reports" / f"{config['model']['backend']}_{kind}_{digest(result)[:12]}.json"
    atomic_json(path, result)
    result["report_path"] = str(path)
    return result
