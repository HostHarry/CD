# HallusionBench / MME, 256/128/64 — thinking and original prompts

Launcher `methods/experiments/think_benchmarks.py` (stages run sequentially on
two GPU shards; preflight -> generation via unchanged `common.run` -> audit ->
scoring). Config `methods/configs/think_256_128_64.yaml`: MMaDA-8B-MixCoT,
crop 512, seed 42, temperature 0, CFG 0, 256 tokens / 128 steps / block 64,
safe EOS `safe_eos_v1`. VCD alpha 1, beta 0.1, noise_step 500 (APC); VIGOR
author loop with MASK image ablation.

| Stage | Methods | Prompt | Status |
|---|---|---|---|
| `hallusion_think` | VCD, VIGOR | MMaDA thinking instruction + `\n` + unchanged TSV question | complete |
| `mme_think` | native, VCD, VIGOR | thinking instruction + official question without `Please answer yes or no.` | running |
| `mme_original` | native, VCD, VIGOR | official question verbatim | queued |

## HallusionBench (thinking prompt), 1129 rows (951 image + 178 text-only)

Text-only rows use native decoding for both methods (identical outputs). No
native run at this setting was requested. Primary scoring: pinned VLMEvalKit
`YOrN_Extraction` + `Hallusion_rating` on the text after the last `</think>`
(full text if unclosed); Unknown counts as wrong. DeepSeek judge
(`judge/`, protocol `pope_text_llm_judge_v1`, runner
`methods/judge/judge_hallusion.py`): the API receives only the original
question and the complete response (no image, no gold); the extracted yes/no
is fed to the same `Hallusion_rating`. 1,965 unique requests, 0 errors.

| Metric | VCD official | VCD judge | VIGOR official | VIGOR judge |
|---|---:|---:|---:|---:|
| aAcc % | 34.01 | 39.42 | 37.64 | 42.60 |
| fAcc % | 8.19 | 9.93 | 11.91 | 12.90 |
| qAcc % | 7.47 | 8.35 | 10.11 | 11.65 |
| image aAcc % (951) | 36.49 | 41.11 | 40.80 | 44.90 |
| text-only aAcc % (178) | 20.79 | 30.34 | 20.79 | 30.34 |
| Unknown | 276 | 159 | 254 | 137 |

`</think>` closed: VCD 79.5%, VIGOR 79.0%; EOS found: 91.9% / 92.4%; mean
answer tokens 98 / 103. The 256 canvas is often too short for this benchmark.

Audit note: the first audit treated generated image-code tokens as a schedule
error; two outputs (one per method) contain one such token. They are real model
outputs and are now only counted (`rows_with_generated_image_code_tokens`).
The superseded queue plan is kept as `plan_v1_overstrict_audit.json`; no
prediction changed.

Payloads use format B (see `../../README.md`).
