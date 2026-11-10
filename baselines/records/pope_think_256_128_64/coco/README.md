# POPE think prompt — 256/128/64, native + VCD + VIGOR, 3 datasets

Full POPE evaluation on three datasets — COCO (`data/pope_coco_yesno.jsonl`),
A-OKVQA and GQA (`data/pope_extended/{aokvqa,gqa}_yesno.jsonl`) — 9000 rows per
dataset, methods `native`, `vcd`, `vigor`, generation schedule
`max_new_tokens=256, steps=128, block_length=64`, seed 42, crop 512,
temperature 0, CFG 0, `low_confidence` remasking, VCD `cd_alpha=1.0,
cd_beta=0.1, noise_step=500`, safe EOS (`safe_eos_v1`, `--early-eos`).

## Prompt (prompt_id `pope_think_v1`)

User-approved VISAGE-style thinking prompt, identical to diagnostic variant B
of `results/pope_prompt_diagnostic` and to the paused
`results/pope_coco_think_256_256` inputs:

```
question = common.mmmu_protocol.INSTRUCTION + "\n" + q0
```

where `q0` is the original POPE question with the exact trailing
`" Please answer yes or no."` removed (asserted per row). For A-OKVQA/GQA,
`q0` must equal the row's `original_question`. The derived string is stored in
`sample["question"]`, so the **unchanged** `common.runtime.Runtime.prepare`
chat template is used: no system message, user message = thinking instruction
+ newline + suffix-free question, image = `<|mmu|><|soi|>[1024 codes]<|eoi|>`
+ chat text. Rows also carry `removed_suffix` and `prompt_id`.

Excludes: COCO → `data/calibration.jsonl`; A-OKVQA/GQA →
`data/pope_extended/excluded_samples.jsonl` (empty). Each dataset is split
across GPU 0/1 by alternating unique input keys (first-occurrence order;
duplicate rows stay together); each worker runs all three methods
(native → vcd → vigor) on its shard via one `common.run` invocation.

## Scoring — two tracks

- **PRIMARY `primary_post_think`**: answer = text after the last `</think>`
  (stripped); if absent, the full text is used and counted as
  `think_close_missing`. Parsed by the unchanged `common.metrics
  .official_pope_answer`. Reports per split + pooled + equal 3-split mean of
  accuracy/F1/precision/recall/yes_ratio/invalid_ratio, think-tag/close
  rates, post-think non-yes/no starts, yes-start-parsed-no and
  no-start-parsed-yes contradictions, EOS rate, 256-position-limit hits,
  answer-token length, executed steps and seconds. For vcd/vigor, a paired
  image-resampled bootstrap vs native on this track is reported per split
  (`common.metrics.paired_bootstrap`, 2000 repeats, seed 42).
- **SECONDARY `secondary_raw_full_text`**: the unchanged `common.run`
  metrics — official parser on the full generated text (dominated by the
  `<think>` block, kept for provenance only).

`metrics.json` holds per-dataset per-method blocks plus `nine_subset_mean`
(equal mean over the 3 datasets × 3 splits). `RESULTS.md` renders the tables.

## Layout / resume

`plan.json` pins source/config/data/launcher hashes; `inputs/` holds the
derived `{dataset}_think.jsonl` files and `{dataset}_{0,1}.jsonl` shards.
Per dataset: `{dataset}_shard{0,1}/` worker outputs, `{dataset}/` merged
output, `{dataset}_shard{0,1}.log`, `{dataset}_score.log`. `queue_status.json`
tracks stage state; `pipeline.log` is the launcher log. Resume = rerun the
same command; `write_locked` refuses changed artifacts and `common.run`
rejects changed manifests.

Preflight (3 fixed COCO inputs × 3 methods) lives in `preflight/` and is not
merged into the dataset outputs.
