# hallusion_full_original_256_128 — HallusionBench, schedule 256/128/32

One dataset (`hallusionbench`), three methods (`native`, `vcd`, `vigor`),
1129 evaluated rows.

## Protocol

- Benchmark: HallusionBench full original split — image + original question
  prompts (not the yes/no variant), per `configs/hallusion_256_128.yaml`.
- Generation: `max_new_tokens=256, steps=128, block_length=32`, seed 42,
  crop 512, safe-EOS profile.
- Scorer: HallusionBench field metrics — `aAcc` (overall answer accuracy),
  `fAcc` (figure), `qAcc` (question), `Avg`; per-split breakdowns in
  `metrics.json` / `{method}_scores.json`.

## Headline (from `results/hallusion_full_original_256_128/metrics.json`)

Overall split, percent:

| Method | aAcc | fAcc | qAcc | Avg | unknown answers |
|---|---:|---:|---:|---:|---:|
| native | 42.60 | 14.64 | 9.67 | 22.30 | 146 |
| vcd | 40.48 | 12.90 | 9.01 | 20.80 | 160 |
| vigor | 40.12 | 12.16 | 9.23 | 20.50 | 188 |

## Sources

- `results/hallusion_full_original_256_128/shard{0,1}/` — two-shard run,
  merged here per method; shard manifests/statuses as `manifest__*`/`status__*`.
- Samples: `data/hallusion_full_original/shard_{0,1}.jsonl` +
  `manifest.json`, `original_prompt_validation.json`;
  docs `methods/HALLUSION.md`.
- `HallusionBench.tsv` (94 MB, third-party dataset with embedded images) is
  not redistributed. Download it from `dataset_url` in `manifest.json` and
  check `tsv_sha256`
  (`8f92be804abf146f24bfa1a98d8fcb1cba335f6b0dc69fc27ea698fa446df3b8`).

## Caveats

- Scoring used the benchmark's own judge pipeline; `verify.py` performs
  integrity + coverage checks only (scores not recomputed).
- `unknown_answers` = rows the scorer could not classify; kept verbatim.
- Payloads use format B (see `../README.md`).
