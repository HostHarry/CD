# MMMU validation: r70 / W16 / K4 / L256 / I128

One fresh 900-question run (847 multiple-choice, 53 open), using the
recently validated EOS-prefix CDAPC implementation. Three GPUs each receive
the same 300-question shard used by the historical local MMMU run.

- Decoder: alpha 0.5, beta 0.1, u gate 0.1, r gate 0.7, rolling window 16,
  maximum four commits per iteration, 256 response positions, 128 iterations.
- Candidates are restricted by the earliest committed EOS; stopping requires
  its entire prefix to be committed.
- Prompt remains `mmmu_wino_reasoning_v1`: official MMaDA thinking instruction
  plus the existing question/options construction. The POPE prompt is not used.
- Images retain the frozen WINO/lmms-eval selection policy and 512 center crop.
  The 43 originally multi-image questions use the selected first image under
  that policy; they are retained in the 900-question denominator and reported
  separately from the 857 single-image questions.
- Base seed 20261105 with the existing CDAPC per-sample derivation.
  The external baseline's per-input reset implementation has not been
  independently verified to be identical.

The supplied external baseline uses 256/128/64. Its fixed block length 64
is a baseline decoder parameter; CDAPC retains its rolling W16/K4 mechanism.
The existing local r70 run was K16 with the original candidate range.
Thus comparisons against that historical run change both K and EOS candidate
policy. They are not single-variable K ablations.

Preflight checks all 900 question/image/option records against the historical
dataset manifests and verifies identical generation sources/assets/environment
to the recent COCO EOS-prefix workspace. Actual saved model input IDs,
including image codes, and per-sample seeds are checked against the old MMMU
outputs before scoring.

Scoring is automatic:

1. Pinned MMMU official scorer, including its deterministic per-question
   random fallback.
2. Strict scoring without guessing when unparsed.
3. Existing WINO-style DeepSeek correctness judge (`deepseek-flash`), with
   its existing answer extraction and gold-answer protocol. This differs
   from POPE's answer-only extraction judge.

Exact matching historical judge requests can be reused. API failures remain
pending; generation failures count incorrect in all three metrics. All 900
questions remain in the denominator. Interrupted generation resumes in new
missing-only waves, preserving successful, budget-exhausted and error records.

Status: `reports/status.json`; log: `logs/controller.log`.
Frozen setup: `reports/plan.json`, `preflight.json`, `source_snapshot.json`.
Final output: `reports/result.json`, `summary.md`, `answers.jsonl`, and
`final.sha256`, including question-paired intervals versus historical K16.
