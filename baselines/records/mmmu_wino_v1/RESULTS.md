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


## DeepSeek LLM judge (`mmmu_val/judge/`)

Reuses the cloud CD-APC MMMU judge verbatim (`cdapc_lab.evaluation.llm_judge`,
WINO `JUDGE_RULES` from LaViDa@24220c0 `eval/lmms_eval/tasks/mmmu/utils.py`,
sha256 `cb865ea4…85de`, not redistributed), runner
`methods/judge/judge_mmmu.py`: `deepseek-flash`, temperature 0, max_tokens 20,
score = 1 if the reply contains "1". Input: question + options, gold, and the
`<answer>` span if present else the complete response. 2,522 unique requests,
0 errors; every request/response is in `api_records.jsonl.gz`. As a
same-setup control the cloud native outputs (`mmmu/answers/native.jsonl`) were
re-judged: 895/900 agree with the cloud judge (24.89% vs 25.22%).

| Method | Official | Random guess counted wrong | DeepSeek judge | Judge delta vs native (95% CI) |
|---|---:|---:|---:|---|
| native (cloud outputs) | 27.78 | 18.89 | 24.89 | — |
| VCD | 27.33 | 18.78 | 23.00 | −1.89 [−4.78, +0.89] |
| VIGOR | 28.33 | 19.78 | 24.22 | −0.67 [−3.67, +2.33] |

Question-level paired bootstrap, 2000 repeats, seed 42.
