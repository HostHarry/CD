# pope_yesno_256_128_32 — POPE yes/no, schedule 256/128/32

Three datasets (`coco`, `aokvqa`, `gqa`), three methods (`native`, `vcd`,
`vigor`), 9000 evaluated rows per dataset-method.

## Protocol

- Prompt: original POPE question **with** the yes/no suffix, chat template,
  no system message; image `<|mmu|><|soi|>[1024 codes]<|eoi|>`, crop 512.
- Generation: `max_new_tokens=256, steps=128, block_length=32`, temperature 0,
  CFG 0, `low_confidence`, seed 42, safe-EOS `safe_eos_v1`.
- VCD `{cd_alpha:1.0, cd_beta:0.1, noise_step:500}`; VIGOR author loop.
- Scorer: official POPE parser per split + paired bootstrap vs native.

## Headline (from `results/pope_256_128_yesno/RESULTS.md` / `metrics.json`)

Mean accuracy over three splits, percent:

| Dataset | Method | Accuracy | F1 | Precision | Recall | Yes ratio |
|---|---|---:|---:|---:|---:|---:|
| coco | native | 80.61 | 79.79 | 83.80 | 76.27 | 45.66 |
| coco | vcd | 79.33 | 80.34 | 76.98 | 84.13 | 54.80 |
| coco | vigor | 80.96 | 80.93 | 81.55 | 80.47 | 49.51 |
| aokvqa | native | 80.09 | 80.71 | 78.82 | 82.87 | 52.78 |
| aokvqa | vcd | 79.39 | 80.89 | 75.91 | 86.73 | 57.34 |
| aokvqa | vigor | 79.61 | 81.02 | 76.37 | 86.47 | 56.86 |
| gqa | native | 79.58 | 80.09 | 78.58 | 81.80 | 52.22 |
| gqa | vcd | 79.61 | 81.15 | 75.84 | 87.40 | 57.79 |
| gqa | vigor | 79.66 | 81.08 | 76.30 | 86.67 | 57.01 |

## Sources

- `results/pope_256_128_yesno/{coco,aokvqa,gqa}/` — one `common.run`
  predictions DB + per-method answers per dataset; experiment-level
  `RESULTS.md` / `metrics.json` copied to this dir (`RESULTS.md`,
  `metrics_experiment.json`). Config: `config__run.yaml` per dataset.
- Samples/excludes identical to `pope_yesno_128_64_32`.

## Caveats

- Same mixed early-EOS/full-schedule profile note as the 128/64/32 records.
- Payloads use format B (see `../README.md`).
