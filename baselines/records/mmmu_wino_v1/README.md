# mmmu_wino_v1 — MMMU validation, thinking prompt, VCD + VIGOR

One dataset (`mmmu_val`), two methods (`vcd`, `vigor`; no native arm), 900
scored rows (two fixed 450-row shards).

## Protocol

- Benchmark: `mmmu_wino_reasoning_v1` — MMMU validation questions in the
  reference single-image policy with the MMaDA thinking instruction.
- Generation: `max_new_tokens=256, steps=128, block_length=64`, seed
  `20261105`, crop 512, safe-EOS profile.
- Scorer: local rule-based extraction (`common.mmmu_score` /
  `results/mmmu_wino_v1/{vcd,vigor}_scores.json`); **local rule scores are not
  directly comparable to the reference report's LLM-judge accuracy**.

## Headline (from `results/mmmu_wino_v1/metrics.json` / `RESULTS.md`)

| Method | Scored | Official accuracy | No-guess accuracy | Random fallbacks | Mean answer tokens | Mean steps |
|---|---:|---:|---:|---:|---:|---:|
| vcd | 900/900 | 27.33% | 18.78% | 312 | 102.33 | 65.36 |
| vigor | 900/900 | 28.33% | 19.78% | 322 | 91.57 | 54.67 |

Paired comparison (`metrics.json.paired`): official-correct vcd-only 92 /
vigor-only 101; no-guess vcd-only 84 / vigor-only 93.

## Sources

- `vcd/results/mmmu_wino_v1/shard{0,1}/` and
  `vigor/results/mmmu_wino_v1/shard{0,1}/` — per-shard prediction DBs, merged
  here per method; shard manifests/statuses kept as `manifest__*`/`status__*`.
- Samples: `data/mmmu_wino_v1/shard_{0,1}.jsonl`; config
  `config__run.yaml` (= `configs/mmmu_wino_v1.yaml`); experiment docs
  `methods/MMMU.md`.

## Caveats

- External-judge scores are **not recomputed** by `verify.py` (integrity +
  coverage only).
- Payloads use format B (see `../README.md`).
