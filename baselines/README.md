# baselines/ — MMaDA VCD/VIGOR reproduction records

This directory packages the local MMaDA-Baselines work for review and
reproduction:

- `methods/` — snapshot of the project-authored code (runtime, VCD/VIGOR
  method implementations, configs, tests, docs). See `methods/README.md` for
  the pinned revisions and what was excluded (official MMaDA sources,
  datasets, unlicensed VIGOR author files).
- `records/` — completed experiments only: every generated prediction
  payload, per-method answers, scoring summaries, run manifests/configs and
  sample files, with per-directory `SHA256SUMS` and a top-level
  `records_index.json`.
- `verify.py` — re-checks every `SHA256SUMS`, verifies every exported payload
  hash against `records_index.json`, and re-scores POPE and CHAIR from the
  exported data, comparing to the stored summaries.

## Experiments included

| Record key | Benchmark | Schedule (tokens/steps/block) | Prompt | Methods |
|---|---|---|---|---|
| `pope_yesno_128_64_32/{coco,aokvqa,gqa}` | POPE yes/no | 128/64/32 | yes/no suffix | native, vcd, vigor |
| `pope_yesno_256_128_32/{coco,aokvqa,gqa}` | POPE yes/no | 256/128/32 | yes/no suffix | native, vcd, vigor |
| `chair_64_32_16/chair500` | CHAIR-500 | 64/32/16 | caption | native, vcd, vigor |
| `mmmu_wino_v1/mmmu_val` | MMMU validation | 256/128/64 | thinking | vcd, vigor |
| `hallusion_full_original_256_128/hallusionbench` | HallusionBench | 256/128/32 | original | native, vcd, vigor |
| `pope_prompt_diagnostic/coco_192` | POPE prompt diagnostic | 256/256/32 | 4 prompt variants A–D | A, B, C, D (native) |

Paused/incomplete runs (`pope_coco_nosuffix_256_256`, `pope_coco_think_256_256`,
`pope_think_256_128_64`) are intentionally **not** included.

## Prediction payload format

Each `records/<experiment>/<dataset>/<method>/predictions-NNN.jsonl.gz` is a
gzip JSONL stream of `{"input_key": ..., "payload": {...}}` lines, one per
unique model input, sorted by input key and chunked to <=45 MB compressed.
`input_key` is `sha256(json.dumps([image_sha256, messages-or-question]))` —
the same key the runtime used for dedup and resume.

Inside each payload, the `input` object records prompt provenance in compact
form ("payload format B"):

- `prompt_ids_sha256` — `sha256` of `json.dumps(prompt_ids,
  separators=(",", ":"))` (the exact token-id list fed to the model);
- `prompt_ids_length` — number of prompt tokens;
- `visual_positions` — `[start, end)` half-open range when the image-code
  positions are contiguous (always the case for these runs: `[2, 1026)`),
  otherwise the explicit position list;
- `image_code_offset`, `image_sha256`, `resolution`, `resize`,
  `prompt_protocol` — unchanged.

Every exported payload was asserted during export: the positions in
`visual_positions` index exactly 1024 image codes, all `>= image_code_offset`.
All other payload fields (`generated_ids`, `generation_state_ids`, `trace`,
`raw_text`, `text`, timings, EOS/early-stop fields, method-specific decision
records) are verbatim from the run databases.

To rebuild a prompt from `prompt_ids_sha256` you need the original token-id
list; `verify.py --rebuild-prompts` demonstrates this for a sample of rows by
re-running `common.runtime` on the image + question.

## Verifying

```bash
python3 baselines/verify.py                    # hashes + POPE/CHAIR re-score
python3 baselines/verify.py --rebuild-prompts  # + prompt rebuild spot-check
```

`verify.py` needs `numpy`. `--rebuild-prompts` additionally needs the
MMaDA-Baselines checkout (runtime + images) and is optional.

## Per-experiment notes

Each `records/<experiment>/` has a `README.md` with the protocol (prompt,
schedule, seed, crop, EOS profile, scorer), headline metrics copied from the
stored summaries, caveats, and the source directories the records came from.
