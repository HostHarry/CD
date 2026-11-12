# A-OKVQA → GQA: official think prompt, no Final-answer suffix

Authorized continuation after the full COCO validation. Three GPUs run each
dataset's three shards. A-OKVQA generates first, then GQA. Official scoring
runs at each generation boundary; its DeepSeek worker runs independently
while the next dataset generates.

| Dataset | Unique inputs | Evaluation rows | Images |
|---|---:|---:|---:|
| A-OKVQA | 5145 | 9000 | 500 |
| GQA | 5171 | 9000 | 500 |

The decoder source, environment and model assets match the completed
`cdapc-pope-coco-prefix-no-final` run. Settings are r70, W16/K4, L256/I128,
fast EOS-prefix candidates, and official MMaDA thinking instruction plus
the question, without the extra Final-answer requirement.

Each dataset retains its existing data revision and **seed 42** from its
original configuration. COCO's separately frozen seed was 20261105.
Old-prompt generation is not reused. The canceled old queue and its 1448
partial A-OKVQA outputs remain untouched.

`reports/plan.json`, `reports/preflight.json`, and
`reports/source_snapshot.json` freeze the setup. All 10316 selected inputs
passed prompt/source/asset checks. The scorer reproduces the completed COCO
metrics, the DeepSeek protocol hash matches, and credential availability
was checked without exposing its value.

The controller preserves successful and budget-exhausted terminal records.
On continuation it creates fresh waves containing missing IDs only. It
never uses the inference runner's retry behavior to overwrite budget
failures.

Official scoring selects the answer after the last `</think>` (or the
whole visible response when unclosed), then applies the original POPE rule.
DeepSeek uses the existing `pope_text_llm_judge_v1` / `deepseek-flash`
protocol and shared exact-request cache. Gold is compared locally.
Generation failures, missing and ambiguous answers remain in the denominator.
API failures remain pending and do not become model mistakes.

Progress: `reports/status.json`, `logs/controller.log`.
Per dataset: `reports/<dataset>/official.json`, `judge_status.json`,
`deepseek.json`, and `summary.md`. All split metrics and equal-split means
are retained.

Run from this directory:

```sh
.venv/bin/python scripts/suite.py prepare
.venv/bin/python scripts/suite.py generate
```

`prepare` is for freezing the setup before launch, not for changing a running
experiment. An interrupted judge can be restarted separately with
`.venv/bin/python scripts/suite.py judge aokvqa` (or `gqa`).
