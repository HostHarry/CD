# pope_prompt_diagnostic — POPE prompt-format diagnostic, 4 variants

One dataset (`coco_192`), four native variants (`A`, `B`, `C`, `D`), 192 fixed
POPE-COCO rows (32 per split × label).

## Protocol

- Purpose: diagnose the ~98% yes bias of suffix-free native POPE prompts.
- Schedule: `max_new_tokens=256, steps=256, block_length=32`, seed 42, crop
  512, safe-EOS, native only.
- Variants (user content): **A** = q0 (suffix-stripped question);
  **B** = `common.mmmu_protocol.INSTRUCTION + "\n" + q0` (thinking prompt);
  **C** = MMaDA training-text format with literal `<eot_id>`;
  **D** = q0 + yes/no suffix (control).
- Scorer: for B the answer is the text after the last `</think>` (missing →
  full text, counted `think_close_missing`); all variants parsed with the
  unchanged official POPE parser. Secondary full-text track kept for B.
  Paired parsed-answer flips vs A included.

## Headline (from `metrics.json`, project diagnostic `RESULTS.md`)

| Variant | Accuracy | F1 | Yes ratio | Invalid ratio |
|---|---:|---:|---:|---:|
| A q0 chat | 50.52 | 66.67 | 98.44 | 0.00 |
| B think+q0 | 75.00 | 75.00 | 50.00 | 1.04 |
| C training text | 50.00 | 66.67 | 100.00 | 100.00 |
| D q0+suffix | 76.56 | 74.29 | 41.15 | 0.00 |

Conclusion recorded upstream: the yes bias is not a chat-template artifact —
answer-format pressure (suffix or thinking instruction) restores
discrimination.

## Sources

- `results/pope_prompt_diagnostic/{A,B,C,D}/predictions.sqlite` — per-variant
  DBs (no method column); `run_diag.py` / `score.py` copied alongside;
  `samples.jsonl` fixed 192-row list; `manifest.json`.
- Determinism vs the paused no-suffix run: 138/138 overlapping inputs produced
  identical generated ids (recorded in `metrics.json`).

## Caveats

- `verify.py` fully re-scores this experiment from the exported payloads +
  samples (metrics, diagnostics, think stats, paired flips vs A).
- Payloads use format B (see `../README.md`).
