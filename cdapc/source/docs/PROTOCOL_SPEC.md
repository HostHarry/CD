# Input, output and evaluation protocol v1

Source: [official M³CoT dataset](https://huggingface.co/datasets/LightChen2333/M3CoT),
revision `48cf35001d595a6b0290c82c897a4b4563390821`, official `test` parquet shards.
The shard SHA-256 values are verified against official LFS metadata before use.
The full split count and exact source files are stored in `provenance.json`.

Smoke IDs are the first two IDs sorted by
`sha256(JSON(['smoke_ids_v1',20261105,id]))`, independent of model outcomes:
`physical-commonsense-855`, `physical-commonsense-386`.
No row is silently removed. The initial model protocol requires exactly one
decodable image and ordered multiple-choice options. Errors stop preflight.
Images are converted once to RGB PNG without resizing; models then apply their
own official image preprocessing.

`samples.json` contains question, choices, public context and image references.
`gold.json` contains answer and evaluation-only domain/topic/category. Gold
rationale is not exported to Sample. The model builder accepts only Sample.

The same question is used by all methods:

```
{public context if present}
{question}
(A) {option A}
(B) {option B}
...
Think step by step, then end with Final answer: (X), using exactly one option letter.
```

For MMaDA the user message is the official thinking instruction (as in the
README demo prompts file and `app.py` thinking mode), a newline, then the
shared question above. LaViDa receives the shared question unchanged. Native
and CD-APC of the same model always receive identical prompts.

MMaDA uses the official chat template, MAGVITv2 code generation and the
preprocessing of the README MMU entry (`inference_mmu.py` with
`configs/mmada_demo.yaml`: resize and 512-pixel center-crop, yielding 1024
MAGVITv2 codes), applied uniformly to all filenames (the script's filename-based
squash switch is not used, guide §9.2). The prompt layout matches that script,
without the sot marker that the VLMEvalKit wrapper adds. LaViDa uses the
official `llada` conversation template and official image processing/expansion.
Normal attention is explicitly bidirectional with cache disabled in both Native
and CD-APC. Native is labelled `native_no_cache`. Model variants are
MMaDA-8B-MixCoT and LaViDa-LLaDA-reason, with exact official revisions in the lock.

Native keeps the official generator. MMaDA consumes total steps directly.
LaViDa's official generator overwrites its `steps` argument; the wrapper converts
the declared total to `step_per_block = total_steps / number_of_blocks`. Length
and step divisibility are checked. CD-APC accepts none of those native fields.

Raw prediction artifacts preserve the entire response canvas. The offline
normalizer extracts only response IDs, finds the first EOS there, truncates before
it, decodes with `skip_special_tokens=True, clean_up_tokenization_spaces=False`,
then strips outer whitespace. It records EOS position, emitted token count and
full canvas length. Prompt EOS and tokens following response EOS cannot affect
answer extraction. Incomplete generations remain incomplete and score zero.

## POPE task

POPE is aligned byte-for-byte with the published MMaDA native baseline
(HostHarry/CD @6848cbf577). Rows come from the RUCAIBox/POPE output lists at
commit `08d957b917e5a378a2f99d35b6293c536a66298b` in order random, popular,
adversarial (file line order); `row_id` is `{split}:{question_id}`; image bytes
come from the `lmms-lab/POPE` parquet, revision
`4db1276663dfa5eb8ad16a52d24c31a09e470896`, and every image sha256 is verified
against the baseline's `protocol/images.tsv`. The RUCAIBox file hashes are
verified against `protocol/pope_coco.manifest.json`. A consistency check maps
every RUCAIBox row to the lmms-lab row (image stem, question, label); 57
lmms-lab adversarial questions read "imange" and are normalized for the check
only — prompts keep the RUCAIBox text verbatim.

The prompt is the RUCAIBox question plus `" Please answer yes or no."` (one
ASCII space, no newline; `pope_yesno_v1`). There is no MMaDA thinking
instruction — that instruction is part of the M3CoT CoT protocol, not POPE.
Identical inputs are deduplicated by
`input_key = sha256(json.dumps([image_sha256, question_with_suffix]))`,
yielding 5,127 unique inputs from 9,000 rows; sample_id is the input_key and
`group_id` is the image file name. The canvas is 128 tokens. POPE shard runs
use `stop_rule: eos_prefix_complete`: once the earliest committed EOS has no
pending position before it, decoding stops and the remaining positions stay
MASK; normalized text and scores are identical by construction, only cost and
mechanism counters stop early.

The official score is the answer-rewriting rule of the pinned RUCAIBox
`evaluate.py`: take the first sentence (split at `.`), strip commas, and if the
words contain `No`, `not`, or `no` the prediction is `no`, otherwise `yes` —
so it defaults to `yes` whenever the first sentence has no negation word. The
strict parser `strict_yes_no_v1` lowercases into `[a-z]` words: only `yes`
parses yes, only `no` parses no, both is ambiguous, neither is unparsed.
The headline POPE metrics are the `pope-report` port of the baseline's
`pope_metrics`: tp/fp/fn/tn, accuracy, precision, recall,
F1 `= 2tp/(2tp+fp+fn)`, yes-ratio, invalid-ratio and empty-answer count per
split (3000 rows each), plus the image-clustered paired bootstrap versus
native. The per-run `summarize` yes/no metrics with `unresolved` are a
diagnostic only, not the headline.

Shard predictions may carry a `commit_log`: one entry per committed position
per iteration with the decision fields and a five-token visual/ablated
log-probability snapshot. `pope-report` uses it to attribute changed answers
to actual contrast replacements (`answer_switch`) and to report the
log-odds confidence shift (LO_I, Δ, LO_h, sign flips and their blocking
cause) at the answer token versus the native export.

## MMMU-val task (MMaDA, WINO-aligned)

MMMU uses `lmms-lab/MMMU` revision `364f2e2eb107b36e07ff4c5a15f5947a759cef47`
(`data/validation-00000-of-00001.parquet`, sha256 verified against HF LFS
metadata): 900 rows — 847 multiple-choice, 53 open; 43 rows have >1 image.
One image per question: the first distinct `<image n>` placeholder found in
question+options, sorted by name (lmms-eval `mmmu_doc_to_visual` order), else
`image_1`; original image bytes are kept. `sample_id` is the row `id`;
`group_id` is the subject.

The prompt is WINO's `mmmu_val_mmada` context (Feng-Hong/WINO-DLLM
@9549a1d0c3cbd9dc9b8f7cd09be8a7aac322d6fb): MC `question\noptions\n` with
`"A. opt"` lines, open `question\n`, then every `<image>`, `<image n>` and
`<imagen>` occurrence removed, and our adapter prepends the official thinking
instruction (equal to WINO's `reasoning_prompt`). Input layout stays our
README `inference_mmu` form — `<|mmu|><|soi|>codes<|eoi|>` + chat template, no
`<|sot|>` — which differs from WINO's hand-written template. Canvas 256.
Native runs steps 128 / block_length 64 / low_confidence / cfg 0.

Three score tracks: PRIMARY is a DeepSeek `deepseek-flash` LLM judge (WINO's
`JUDGE_RULES`, system prompt, score = 1 if the reply contains "1"), official
is the MMMU rule parser (MMMU-Benchmark @51ce7f3e, per-sample seeded random
fallback `sha256("mmmu_rule_v1:"+id)`), and strict is the same parser with the
random-fallback case counting as unparsed/0. `mmmu-report` aggregates per
method over all 900, the 857 single-image subset, and per question_type, plus
paired bootstrap vs native (question-level, 2000 repeats, seed 42), judge
transitions, mechanism totals, cost, finish_reason and truncation counts.

## Shared score tracks

Two independent score tracks receive the same normalized text:

- `official`: the exact `ALPHA_MAP` and `judge_answer` AST from the pinned
  [official scorer](https://github.com/LightChen233/M3CoT/blob/ee50d5ac69661f81731d33647dd307d1f84eada5/utils/metric.py).
  Its extraction behavior is preserved, including permissive text matches. The
  function is deterministic; importing unrelated inference/training tools is
  unnecessary.
- `strict_accuracy`: `strict_final_v1` extracts explicit final-region labels,
  trailing parenthesized/boxed labels, or a last line consisting solely of one
  valid label (bare, parenthesized or boxed), even after reasoning; this covers
  MMaDA's `</think>` newline letter ending. A same-line trailing label conflicts
  with any other parenthesized/boxed label on that line. At most one terminal
  `.。!！` is allowed. Conflicts are
  ambiguous, illegal labels invalid, incomplete markers invalid, and absent
  answers unparsed. There is no random fallback and the parser never sees gold.

The preregistered primary metric is official. The smoke suite cannot establish
benchmark accuracy: two cases and a 128-token engineering canvas are deliberately
small and may truncate reasoning. The general 512-token CoT configuration and
additional datasets are not implied by a smoke pass.

Generation fingerprints include only effective generation settings, source,
official assets, exact input/image hashes, prompt, sample IDs, RNG and runtime.
Evaluation fingerprints include raw predictions, actual evaluation source, gold,
answer domains, tokenizer asset hashes and metric protocol. Offline scoring
verifies tokenizer/config content against the generation receipt before loading;
changed assets are rejected without model loading or overwriting old scores.
The separate evaluation spec hash excludes
prediction contents and must match for comparisons. Scores never overwrite raw
predictions. Missing/error/budget-exhausted cases stay in the fixed denominator.

Seed derivation is SHA-256 over `['sample_seed_v1',run_seed,sample_id,replicate_id]`,
using the first eight hex digits as a uint32 seed; Python, NumPy and Torch/CUDA
are reset per sample. Resume skips only successful matching predictions.
Paired confidence intervals resample questions, clustering by group_id when
questions share a statistical unit. The small sample interval is descriptive.
