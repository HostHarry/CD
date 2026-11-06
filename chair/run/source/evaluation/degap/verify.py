import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf
import torch

from models.degap import DEGAP, DEGAPConfig, visual_bias
from .runtime import Runtime, environment, save_json, source_hashes


def compare(actual, expected):
    lp, lq = actual.float().log_softmax(-1), expected.float().log_softmax(-1)
    return {
        "max_abs_error": (actual.float() - expected.float()).abs().max().item(),
        "mean_kl": (lp.exp() * (lp - lq)).sum(-1).mean().item(),
        "top1_agreement": (actual.argmax(-1) == expected.argmax(-1)).float().mean().item(),
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/mmada_degap.yaml"))
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    runtime = Runtime(config["checkpoint"], config["vq_checkpoint"], args.device,
                      config["resolution"], config["resize"])
    model = runtime.model
    engine = DEGAP(model, DEGAPConfig(record_entropy=True))
    samples = [json.loads(line) for line in args.samples.read_text().splitlines() if line.strip()]
    records = []
    for sample in samples[:2]:
        prompt, visual, provenance = runtime.prepare(sample)
        tokens = torch.cat((prompt, torch.full((1, 4), model.config.mask_token_id, device=prompt.device)), dim=1)
        positions = torch.arange(prompt.shape[1], tokens.shape[1], device=prompt.device)
        native = model(tokens).logits[:, positions].clone()
        normal, entropy, cache = engine.observe(tokens, positions)
        record = {"id": sample["id"], "input": provenance, "entropy": entropy.tolist(),
                  "normal": compare(normal[:, positions], native), "suffix": {}}
        record["last_readout"] = compare(engine.reference(cache[len(engine.blocks) - 1], positions,
                                                        len(engine.blocks) - 1, None), native[0])
        del normal
        bias = visual_bias(tokens.shape[1], visual, "all", tokens.device)
        for b in (0, len(engine.blocks) // 2, len(engine.blocks) - 1):
            shared = engine.reference(cache[b], positions, b, bias)
            h = model.model.transformer.emb_drop(model.model.transformer.wte(tokens))
            for l, block in enumerate(engine.blocks):
                h, _ = block(h, attention_bias=bias if l >= b else None, use_cache=False)
            full = engine.readout(h[:, positions])[0]
            record["suffix"][str(b)] = compare(shared, full)
        del cache
        kwargs = dict(max_new_tokens=8, steps=4, block_length=4, mask_id=model.config.mask_token_id)
        left_trace, right_trace = [], []
        left = model.mmu_generate(prompt, trace=left_trace, **kwargs)
        right = model.mmu_generate(prompt, degap_config={"alpha": 0}, trace=right_trace, **kwargs)
        record["alpha_zero_equal"] = torch.equal(left, right) and left_trace == right_trace
        record["native_trace"] = left_trace
        checks = [record["normal"], record["last_readout"], *record["suffix"].values()]
        record["passed"] = record["alpha_zero_equal"] and all(
            c["max_abs_error"] <= 0.125 and abs(c["mean_kl"]) <= 1e-4 and c["top1_agreement"] == 1
            for c in checks)
        records.append(record)
        print(sample["id"], record["passed"], flush=True)
    passed = bool(records) and all(r["passed"] for r in records)
    save_json(args.output, {"passed": passed, "environment": environment(), "sources": source_hashes(),
                           "bf16_tolerance": {"max_abs": 0.125, "abs_mean_kl": 1e-4, "top1": 1.0},
                           "records": records})
    if not passed:
        raise RuntimeError("real-model equivalence verification failed")


if __name__ == "__main__":
    main()
