# MME think: r70 / W16 / K4 / L256 / I128, EOS-prefix candidates

All 2374 MME questions (1187 image pairs, 14 categories), three GPU shards
using the historical MME shard assignment.

- Prompt `mme_think_v1`: MMaDA official thinking instruction + the verbatim
  MME question (which itself ends in `Please answer yes or no.`).
- Decoder: alpha 0.5, beta 0.1, u 0.1, r 0.7, rolling window 16, at most four
  commits per iteration, 256 positions, 128 iterations, candidates restricted
  before the earliest committed EOS. Decoder/model/IO sources are byte-identical
  to `cdapc-pope-coco-prefix-no-final`; only MME task/prompt registration was
  added to `config.py` and `data/records.py`.
- Seed 20261105 with the existing per-sample derivation; per-sample seeds and
  image tokens match the historical direct MME run.

Primary score: answer after the last `</think>` (whole visible text when
unclosed), parsed by the official MME first-four-character Yes/No rule; each
category scores `100 * (Acc + Acc+)` (Perception /2000, Cognition /800).
Budget exhaustion counts as `other` (incorrect) with fixed denominators.
The unmodified upstream `calculation.py` independently re-checks every category.
A whole-text parse is kept only as a diagnostic. No API judge.

The scorer was checked before launch by reproducing the historical direct-answer
total (1225.95) exactly from its saved outputs. The historical run differs in
prompt, K (16) and EOS candidate policy, so it is context, not a single-variable
control.

Commands (from this directory):

```sh
.venv/bin/python scripts/run.py --prepare   # freeze + preflight (done)
.venv/bin/python scripts/run.py             # generate missing IDs, verify, score
.venv/bin/python scripts/run.py --score     # rescore saved inventory
```

Status: `reports/status.json`; results: `reports/summary.md`, `result.json`.
