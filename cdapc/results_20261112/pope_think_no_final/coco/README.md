# Full COCO: fast EOS-prefix, official thinking prompt, no Final-answer suffix

Authorized on 2026-11-11 after the 64-image pilot. The old-prompt A-OKVQA/GQA
queue is cancelled; its historical records remain intact. This run waits for
the already-running original-candidate pilot and its 2x2 report to finish.

The prompt is the official MMaDA thinking instruction + newline + original
POPE question. There is no additional Final-answer requirement or yes/no
suffix. Candidates are restricted before the earliest committed EOS, with
safe prefix-complete stopping. Other settings remain r70/W16/K4/L256/I128,
seed 20261105, BF16, SDPA, image crop 512.

The 661 completed new-prompt pilot records are reused byte-for-byte.
Only 4466 missing inputs are generated, preserving original GPU shard
assignment and per-sample seeds. New runs have selection-specific generation
fingerprints but identical generation source, model assets and parameters.
Resume generates missing IDs in fresh runs; terminal budget failures are
retained and never overwritten.

Official and DeepSeek reports cover:

- Full COCO: 500 images / 5127 unique inputs / 9000 evaluation rows.
- Additional part excluding the pilot: 436 images / 4466 inputs / 7848 rows.

The prompt was selected after inspecting the pilot. The additional part is
reported separately; this is not described as a benchmark with no prior
test-set analysis. The existing with-suffix EOS-prefix results are the
matched control. Failures remain in each fixed denominator.

Official scoring uses the last `</think>` suffix, full-text fallback, and
the pinned POPE rule. DeepSeek uses the existing `deepseek-flash` text-only
answer-extraction protocol, without images or gold sent to the API. Cached
pilot judgments are reused; API errors remain pending, distinct from model
errors. Both reports include image-paired bootstrap intervals.

`reports/plan.json` freezes the reused outputs and full cohort.
`reports/preflight.json` records the 3 continuation configurations.
`reports/source_snapshot.json` freezes source, scripts, configs and manifests.

Progress: `reports/status.json`; DeepSeek: `reports/judge/status.json`.
Outputs: `reports/comparison.md`, `reports/official_scopes.md`,
`reports/fast_official_deepseek.md` (the latter contains both full and
additional-part DeepSeek results).

```bash
PYTHONPATH=src OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 .venv/bin/python scripts/run_full.py --prepare
PYTHONPATH=src OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 .venv/bin/python -u scripts/run_full.py
CUDA_VISIBLE_DEVICES='' .venv/bin/python -u scripts/judge_full.py
```
