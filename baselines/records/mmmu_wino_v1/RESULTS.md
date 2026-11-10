# MMMU VCD / VIGOR

Protocol: mmmu_wino_reasoning_v1, 256/128/64, seed 20261105.
Reference single-image policy; two fixed 450-row shards.

| Method | Scored | Official accuracy | No-guess accuracy | Random fallbacks | Mean answer tokens | Mean steps |
|---|---:|---:|---:|---:|---:|---:|
| vcd | 900/900 | 27.33% | 18.78% | 312 | 102.33 | 65.36 |
| vigor | 900/900 | 28.33% | 19.78% | 322 | 91.57 | 54.67 |

Local rule scores are not directly comparable to the reference report's LLM-judge accuracy.

Detailed scoring, subject breakdowns and compute costs: metrics.json.
Raw IDs, full generation states, input IDs and commitment traces remain in the per-shard SQLite databases.
