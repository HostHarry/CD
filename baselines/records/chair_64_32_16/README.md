# chair_64_32_16 — CHAIR-500, schedule 64/32/16

One dataset (`chair500`), three methods (`native`, `vcd`, `vigor`), 500 images.

## Protocol

- Prompt: CHAIR captioning prompt, image `<|mmu|><|soi|>[1024 codes]<|eoi|>`,
  crop 512.
- Generation: `max_new_tokens=64, steps=32, block_length=16`, temperature 0,
  CFG 0, `low_confidence`, seed 42, safe-EOS `safe_eos_v1`.
- Scorer: pinned CHAIR evaluator (Maxlinn/CHAIR-metric-standalone @
  `4087a26211aa2339b9a76307cb8f0321ef691d0a`, sha256 in
  `chair_metrics.json.provenance`) restricted to the 500 evaluated image IDs;
  NLKT tokenization via `patternfork-nosql` (provenance in same file).

## Headline (from `chair_metrics.json`, project `RESULTS.md`)

| Method | CHAIRs ↓ | CHAIRi ↓ | Object Recall ↑ | Mean answer tokens |
|---|---:|---:|---:|---:|
| native | 21.80 | 9.90 | 54.92 | 61.994 |
| vcd | 25.20 | 10.10 | 57.55 | 60.464 |
| vigor | 22.80 | 9.38 | 55.35 | 59.708 |

Mixed outcome — not a uniform hallucination reduction; no CHAIR significance
claim was made upstream.

## Sources

- `vcd/results/early_eos/chair_full/` (native+vcd) and
  `vigor/results/early_eos/chair_full/` (native+vigor) — early-EOS outputs the
  reported numbers come from. Per-method scorer outputs are `*/chair.json`;
  aggregated `chair_metrics.json` at dataset level.
- Samples: `data/chair500.jsonl`; exclude `data/calibration.jsonl`.

## Caveats

- **Inherited scorer defect**: the evaluator's inflection step maps `bus` →
  `bu` (documented in `methods/REPRODUCTION_AUDIT.md`); scores are the formal
  unchanged-scorer values.
- `verify.py` re-aggregates CHAIRs/CHAIRi/Recall from the exported
  `chair.json` sentences (formulas verified against `data/evaluators/chair.py`).
- Payloads use format B (see `../README.md`).
