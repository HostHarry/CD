# pope_yesno_128_64_32 — POPE yes/no, schedule 128/64/32

Three datasets (`coco`, `aokvqa`, `gqa`), three methods (`native`, `vcd`,
`vigor`), 9000 evaluated rows per dataset-method (unique model inputs fewer —
see `records_index.json`).

## Protocol

- Prompt: original POPE question **with** the trailing
  `" Please answer yes or no."` suffix, chat template, no system message;
  image as `<|mmu|><|soi|>[1024 codes]<|eoi|>`, crop 512.
- Generation: `max_new_tokens=128, steps=64, block_length=32`, temperature 0,
  CFG scale 0, remasking `low_confidence`, seed 42, safe-EOS profile
  `safe_eos_v1` (`--early-eos`).
- VCD: `cd_alpha=1.0, cd_beta=0.1, noise_step=500` (+APC). VIGOR: author
  token-ID loop, MASK image ablation, FP64, clamp eps 1e-6… (see
  `methods/vigor/`).
- Scorer: unchanged official POPE parser (`common.metrics.pope_metrics`),
  per split (random/popular/adversarial) + image-level paired bootstrap vs
  native (500 unique COCO images / dataset-specific image counts, 2000
  resamples, seed 42).

## Headline (from stored `metrics.json` / project `RESULTS.md`)

Mean accuracy over the three splits, percent:

| Dataset | native | vcd | vigor |
|---|---:|---:|---:|
| coco | 80.33 | 78.76 | 80.73 |
| aokvqa | 78.84 | 78.72 | 78.50 |
| gqa | 79.47 | 79.00 | 78.89 |

Per-split detail and paired bootstrap CIs are in each dataset's `metrics.json`
(and `scores.json` per method). Full project-level discussion:
`methods/RESULTS.md`.

## Sources (in `/root/autodl-tmp/MMaDA-Baselines`)

- coco: `vcd/results/early_eos/pope_full/` (native+vcd) and
  `vigor/results/early_eos/pope_full/` (native+vigor) — the **early-EOS**
  outputs the reported numbers come from; native predictions are identical
  across both runs (verified matching input protocol) and are exported once.
- aokvqa/gqa: `results/pope_extended/{aokvqa,gqa}/` (all three methods in one
  run each).
- Samples: `data/pope_coco_yesno.jsonl`,
  `data/pope_extended/{aokvqa,gqa}_yesno.jsonl`; excludes
  `data/calibration.jsonl` / `data/pope_extended/excluded_samples.jsonl`
  (both verified no-ops: 0 overlap / empty).

## Caveats

- Mixed execution profiles: early-EOS records coexist with retained
  full-schedule records; do not compare mixed averages as simultaneous
  throughput (see project `RESULTS.md` / `REPRODUCTION_AUDIT.md`).
- VCD position-ranking equivalence vs the author implementation is an
  unresolved audit item (`methods/REPRODUCTION_AUDIT.md`).
- Prediction payloads use format B (`prompt_ids_sha256` +
  `prompt_ids_length` + compact `visual_positions`); see `../README.md`.
