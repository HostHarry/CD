# MMaDA-8B-MixCoT POPE (yes/no) and CHAIR decoding records

Verbatim generation records from an independent MMaDA adaptation of DEGAP
(not the DEGAP authors' code). Upstream MMaDA revision
`3cdb8709635beb29f66e4e23cd66060872ac3996`; checkpoints `MMaDA-8B-MixCoT` and
`magvitv2` (full weight hashes in `environment/checkpoint-sha256.json`).
Methods: `native` (official `mmu_generate` behaviour) and `degap_token`.

Run `python scripts/verify.py` (needs only `numpy`) to check every file hash and
payload hash, confirm the records cover exactly the protocol inputs, and
recompute `pope/run/metrics.json` exactly from the records.

## Input protocol (identical for both methods)

- Questions: `protocol/pope_coco_yesno.jsonl` = original POPE-COCO question +
  `" Please answer yes or no."` (one ASCII space, no newline; see
  `protocol/derive_pope_prompt.py`). Parent file `protocol/pope_coco.jsonl`.
  CHAIR: `protocol/chair500.jsonl`, question `"Please describe this image in detail."`.
- Images: COCO val2014, not redistributed; file names and SHA256 in `protocol/images.tsv`.
- Image preprocessing: RGB, `Resize(512, BICUBIC)` on the short side,
  `CenterCrop(512)`, `ToTensor`, `Normalize(0.5, 0.5)` (MMaDA
  `training/utils.py:image_transform`) → 1024 MAGVIT codes, each offset by
  `len(tokenizer) = 126349`.
- Token layout (1052 prompt tokens for POPE):
  `<|mmu|>(126089) <|soi|>(126084) [1024 image codes] <|eoi|>(126085)` +
  `tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=True, add_generation_prompt=True)`,
  i.e. text
  `<|startoftext|><|start_header_id|>user<|end_header_id|>\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n`.
  No extra `<|sot|>` is inserted; the template's `<|startoftext|>` *is* id 126080
  (= `<|sot|>`), appearing once. No reasoning/CoT instruction. This is the official
  `inference_mmu.py` protocol, not the VLMEvalKit wrapper (256 px, extra `<|sot|>`).
  Every record stores its exact `input.prompt_ids`.
- Generation: POPE `max_new_tokens=128, steps=64, block_length=32`; CHAIR
  `64/32/16`; temperature 0, CFG 0, `low_confidence` remasking, seed 42, BF16,
  RTX 5090, PyTorch 2.8.0+cu128, transformers 4.46.0.
- Answer text: `decode(ids[:first EOS], skip_special_tokens=True,
  clean_up_tokenization_spaces=False).strip()`; `raw_text`/`generated_ids` keep
  all 128 positions.

## Scoring and de-duplication

- De-duplication key (identical inputs share one deterministic prediction):
  `sha256(json.dumps([image_sha256, question], ensure_ascii=False, sort_keys=True).encode())`,
  where `question` already contains the suffix. POPE: 9000 rows → 5127 unique inputs.
- POPE parser (original POPE rule): take text before the first `.`, remove `,`,
  split on spaces; `no` if any word is `No`, `not` or `no`, otherwise `yes`.
  Per split (3000 rows): positive class `yes`, accuracy `(tp+tn)/n`,
  F1 `2tp/(2tp+fp+fn)`. Answers not starting with yes/no are counted as
  `invalid` but kept. Paired intervals resample the 500 images (2000 repeats, seed 42).
- CHAIR: `Maxlinn/CHAIR-metric-standalone@4087a262` (`chair.py` SHA256
  `98728cd2…387770d`), see `chair/run/chair_metrics.json` for provenance.

## Results (from the stored metrics)

| POPE split | native Acc / F1 / yes% | degap_token Acc / F1 / yes% |
|---|---|---|
| random | 83.0 / 82.5 / 46.9 | 79.3 / 80.9 / 58.1 |
| popular | 82.0 / 81.6 / 48.0 | 80.1 / 81.4 / 57.3 |
| adversarial | 76.0 / 76.9 / 53.9 | 72.9 / 76.3 / 64.5 |

CHAIR (500 images): native CHAIRs 21.8 / CHAIRi 9.9 / Recall 54.9;
degap_token 26.8 / 10.6 / 56.1.

## Layout

- `{pope,chair}/{native,degap_token}/predictions-*.jsonl.gz`: one line per unique
  input, `{"input_key": ..., "payload": <verbatim SQLite payload>}`. Payload fields:
  `generated_ids`, `raw_text`, `text`, `input` (prompt ids, visual positions,
  preprocessing), per-step `trace` (committed positions/tokens/confidences; DEGAP
  also boundaries and normal/contrast top-1), timings.
- `records_index.json`: per-record payload SHA256 and the source SQLite SHA256.
- `{pope,chair}/run/`: run manifest, metrics, exported answers, and the exact
  source snapshot used (MMaDA code under its MIT licence, see `LICENSE-MMaDA`).
- `protocol/`, `environment/`: sample files, manifests, run config, package lock.
- `scripts/export_from_server.py` produced this export; `scripts/verify.py` checks it.

## MMMU input protocol and CDAPC audit

The separate [MMMU protocol](mmmu/README.md) documents the exact
`mmmu_wino_reasoning_v1` user prompt, tokenizer template, image preprocessing,
and single-image selection for multi-image questions. It includes the input
construction source excerpts and Native / CDAPC r90 / CDAPC r50 configurations.

The [r90 mechanism audit](mmmu/reports/r90_mechanism_audit.md) summarizes all
900 questions per method, including gate activity, contrast replacements,
fallback, stopping behavior, and paired judge results. The
[machine-readable audit](mmmu/reports/mechanism_audit.json) contains the detailed
statistics and r90 replacement events. This package contains source excerpts,
normalized answer text, and summaries; full raw-token/commit-log prediction
artifacts and model weights are not included.

The [generation-pattern follow-up](mmmu/reports/r90_generation_patterns.md)
reconstructs how suffix EOS commits delay unfinished prefixes, when answer
letters are committed relative to reasoning text, and how empty `<think>`
outputs arise. Its [data](mmmu/reports/generation_patterns.json) includes
per-question summaries and three complete example timelines.

The [r50 comparison](mmmu/reports/r50_generation_patterns.md) examines increased
contrast replacements, early EOS, output structure, and paired answer changes.
It also separates official random-fallback hits from strict scoring results.

[All 2,700 MMMU answer texts](mmmu/answers/README.md) are available as readable
TXT and JSONL files, with 900 responses each for Native, r90, and r50. The text
is copied exactly from the saved scoring records, including incomplete or
repetitive reasoning and the final answer.

The [failure-mode analysis](mmmu/reports/failure_modes.md) separates missing
reasoning tags, empty reasoning, unclosed reasoning, missing suffix answers,
and reviewed arithmetic or consistency errors, with per-question assignments.

To verify the MMMU package files, run `cd mmmu && sha256sum -c SHA256SUMS`.

## CD-APC experiments

The [cdapc/](cdapc/README.md) package archives all completed CD-APC experiments
from `cdapc-experiments` and the `cdapc-mme` snapshot: MMMU r70 (L256/I128),
POPE direct/think on COCO/A-OKVQA/GQA, CHAIR, the GQA think K-commit pilot,
and MME. It includes source excerpts, configs, stored reports, and normalized
per-sample answer text (JSONL/TXT); raw token artifacts, model weights, and
the incomplete POPE think K4 L256/I128 experiment are excluded. Verify with
`cd cdapc && sha256sum -c SHA256SUMS`.
