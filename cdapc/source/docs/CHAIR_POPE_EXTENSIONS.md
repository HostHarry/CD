# CHAIR and POPE-AOKVQA/GQA evaluation

The queue runs CHAIR first, followed by POPE-GQA direct, POPE-AOKVQA direct,
POPE-GQA reasoning, and POPE-AOKVQA reasoning. Each phase runs CDAPC full at
r=0.9, r=0.7, and r=0.5, using three shards on GPUs 0/1/2.
New direct-answer stages precede new reasoning stages. Restarting the queue
controller first adopts and finishes the active stage, before starting any
newly inserted stage, so existing workers keep exclusive use of their GPUs.
Completed stages with reports are skipped.

The r70 addition was requested while POPE-AOKVQA reasoning r90 was running.
That stage continues first. With GQA prioritized, the remaining order is CHAIR
r70, POPE-GQA direct r70, POPE-AOKVQA direct r70, POPE-GQA reasoning
r90/r70/r50, then POPE-AOKVQA reasoning r70/r50.
Each new r70 condition uses the same inputs, seed,
prompt, parser, and generation settings as its corresponding r90/r50 runs;
only `cdapc.r_gate.threshold` and the identifying run name change.

On 2026-11-08 the user cancelled the pending POPE-AOKVQA reasoning r50 stage.
Its generation and Judge status are `cancelled`; it is skipped on resume.
The active AOKVQA reasoning r70 workers continue, and existing completed r50
artifacts remain available. After r70, the next GPU job is MMMU r70, followed
by the separately queued K4/K2 pilot.

## Fixed generation parameters

- MMaDA and MAGVITv2 checkpoints, revisions, preprocessing, and attention semantics
  match the existing POPE-COCO runs.
- BF16, SDPA, batch size 1, sequential visual/ablated branches, cache disabled.
- alpha=0.5, beta=0.1, u gate=0.1, fixed W=K=16.
- CHAIR: L=64, iteration limit 64; POPE: L=128, iteration limit 128.
- Stop rule: `eos_prefix_complete`; seed 42.
- A change of prompt does not add a two-phase decoding constraint: final-answer
  tokens can still be committed before all reasoning tokens.

## Inputs and provenance

CHAIR uses the exact 500 COCO val2014 image IDs and JPEG hashes from
`HostHarry/CD/protocol/chair500.jsonl`. Every image hash is checked before
generation. Its question is exactly `Please describe this image in detail.`,
without a thinking instruction.

POPE questions come from `RUCAIBox/POPE` revision
`08d957b917e5a378a2f99d35b6293c536a66298b`, under `output/seem/aokvqa` and
`output/seem/gqa`. These are the official SEEM variants for these datasets.
Each has random/popular/adversarial splits of 3,000 rows, balanced 1,500 yes and
1,500 no, over 500 images. Original question strings and labels are preserved.
Identical image/question inputs are deduplicated across splits:

| Dataset | Evaluation rows | Unique generations per method |
| --- | ---: | ---: |
| CHAIR | 500 | 500 |
| POPE-AOKVQA | 9,000 | 5,145 |
| POPE-GQA | 9,000 | 5,171 |

Images are obtained from the original COCO image bucket or Stanford Visual
Genome image hosting, with local SHA256 manifests. Prepared data and source
question snapshots are under `data/chair`, `data/pope_aokvqa`, and `data/pope_gqa`.
Ground truth is kept outside the model's `Sample` inputs.

## POPE reasoning protocol

The existing `pope_yesno_v1` remains unchanged: original question followed by
one space and `Please answer yes or no.`.

The new `pope_yesno_reasoning_v1` prepends the exact
`OFFICIAL_THINKING_INSTRUCTION` used in MMMU:

```text
You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here
{original question}
After </think>, end with exactly Final answer: Yes or Final answer: No.
```

Its paired parser is `pope_final_yes_no_v1`. It considers only the region after
the final closing `</think>`; if a `Final answer:` marker exists there, it reads
after the last such marker. It accepts a single Yes/No with optional terminal
punctuation. An unclosed think, missing final answer, or conflicting final
polarities is unresolved and scores zero for either ground-truth label.
Bare Yes/No or `Final answer: Yes/No` is also accepted when no think block was
generated, keeping answer correctness distinct from adherence to the reasoning
format.

The final-answer score applies the official POPE rule to the extracted polarity.
The original official prediction on the full raw text is retained separately.
Reports explicitly record `answer_scope=extracted_final_answer` and
`raw_official_splits`; these should not be conflated with raw-text official scores.
This prevents a `no/not` in the reasoning body from determining the final answer.

Canvas length, CDAPC settings, image selection, and sample IDs are the same for
direct and reasoning conditions. The generation queue does not call an LLM Judge.
The separate `run_pope_think_judge.py` controller judges completed think stages
and skips cancelled stages.

## CHAIR scoring verification

The evaluator is `Maxlinn/CHAIR-metric-standalone` at
`4087a26211aa2339b9a76307cb8f0321ef691d0a`, SHA256
`98728cd2b7fb8b97420b5f3bf14789dcad1d7d980de9c089b82856217387770d`.
It uses NLTK 3.9.1 and patternfork-nosql 3.6, as in the previous reference.

To avoid downloading the complete annotation archives again, the annotation
loader consumes the complete saved `mscoco_gt_words` sets for these 500 images.
The object sets agree between the exported Native and DEGAP records. The word
matching and metric calculation code is unchanged. Before scoring new outputs,
the scorer reproduces all 500 Native per-image metrics and aggregate
CHAIRs/CHAIRi/Recall exactly, and verifies the inflection implementation hash.
See `reports/chair_reference_verified.json`.

The completed CDAPC CHAIR reports also require all prompt token IDs to match
the corresponding old Native inputs.

## Execution and results

```bash
.venv/bin/python scripts/prepare_eval_extensions.py chair
.venv/bin/python scripts/prepare_eval_extensions.py pope_aokvqa
.venv/bin/python scripts/prepare_eval_extensions.py pope_gqa
.venv/bin/python scripts/run_chair_pope_extensions.py
```

The queue refuses simultaneous instances, records exact run directories, resumes
incomplete runs, retries generation once, and stops on any remaining failure or
scoring error. It does not substitute another shard or report partial coverage as
complete.

- Queue index: `reports/chair_pope_extensions_runs.json`
- Queue log: `logs/chair_pope_extensions_queue.log`
- Per-shard logs: `logs/benchmark_extensions/`
- CHAIR reports: `reports/chair_r{90,70,50}.json`
- POPE direct reports: `reports/pope_{aokvqa,gqa}_r{90,70,50}.json`
- POPE reasoning reports: `reports/pope_{aokvqa,gqa}_reasoning_r{90,70,50}.json`

Only completed stages have final reports. The run index records the active
stage and completion state; the file naming convention alone does not imply
that an experiment has completed.
