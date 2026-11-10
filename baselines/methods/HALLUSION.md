# HallusionBench: 256/128/32

The user requested HallusionBench accuracy with 256 generated positions and
128 denoising steps on 2026-11-08. Native, VCD and VIGOR share block length 32,
crop 512, temperature=0, CFG=0, seed=42 and the existing safe EOS stopping rule.
Visual method algorithms and official model code are unchanged. The native
observer also accepts an empty visual span for the newly included text-only
questions.

The VLMEvalKit TSV has MD5 `0c23ac0dc9ef46832d7a24504f2a0c7c`.
The user explicitly requested inclusion of no-image questions: all **1129 rows**
are evaluated, comprising 951 image-bearing and 178 text-only questions. Images
and data are copied into this project, with hashes and all text-only IDs recorded
in `data/hallusion_full_original/manifest.json`.
At the user's request, each original TSV question is passed unchanged into the
standalone MMaDA chat template, without adding a suffix, system prompt or
reasoning instruction. Any yes/no wording already present in a source question
is retained verbatim. The prompt ID is `hallusion_original_question_v1`.

Text-only questions use the tokenizer's chat template directly, without image
tokens, MMU/SOI/EOI delimiters, or fabricated blank images. VCD and VIGOR have no
visual branch on these inputs; they use the identical native decoder and record
both requested and effective methods. All three methods' text-only tokens and
traces must match. Full-dataset accuracy is primary; image/text-only breakdowns
are also saved.

Scoring uses unchanged `YOrN_Extraction`, `process_punctuation` and
`Hallusion_rating` function bodies from VLMEvalKit revision
`3c192143007e4eb3f5c1a36558a4f8c1451980cb`. Sources and hashes are under
`data/evaluators/hallusion`. No external judge is used. Both yes and no, neither
word, and empty responses yield Unknown and count as incorrect. The primary
requested accuracy is per-answer accuracy **aAcc (%)**. Figure accuracy fAcc,
question consistency accuracy qAcc and their Avg are saved separately.

Run from this project with its own environment (`pandas==2.2.3` for scoring):

```bash
.venv/bin/python -m pip install -r requirements-hallusion.txt
.venv/bin/python -m common.hallusion --prepare /path/to/HallusionBench.tsv
OMP_NUM_THREADS=2 .venv/bin/python -m unittest discover -s tests -p test_baselines.py -v
OMP_NUM_THREADS=4 .venv/bin/python -u run_hallusion.py --gpus 0 1
```

The coordinator runs two fixed shards (565 and 564 rows), with one model replica
per GPU. Each replica runs Native, VCD, then VIGOR. Before full generation, both
GPUs run the same three samples (including a text-only question) with every method and verify identical inputs,
returned tokens and commitment traces, prompt assembly, the 256/128 schedule,
and closed-prefix EOS stopping. The preflight is not a performance result.

Resume with the identical command; source, config, data and scorer hashes are
locked. The queue lock is inherited by workers, so a surviving worker prevents
an overlapping coordinator. Each committed prediction is stored transactionally
with raw tokens, input IDs and traces. Do not edit sources during a run.

Progress:

- `results/hallusion_full_original_256_128/queue_status.json`
- `results/hallusion_full_original_256_128/shard{0,1}/progress.json`
- `results/hallusion_full_original_256_128/shard{0,1}.log`

After both shards finish, the coordinator audits every prediction, checks full
coverage and writes `metrics.json`, `{native,vcd,vigor}_scores.json` and
`RESULTS.md` in `results/hallusion_full_original_256_128/`. Completion requires both generation
and scoring. The older paused DEGAP queue is separate.

The initial image-only preflight was stopped when the user expanded the scope;
`results/hallusion_256_128` and `data/hallusion_yesno` retain that superseded setup.
The subsequent full-data run with an appended yes/no instruction was also
stopped at the user's request. Its 209 committed native predictions remain in
`results/hallusion_full_256_128`; its data remain in `data/hallusion_full_yesno`.
The original-question run starts fresh and does not import those predictions.
