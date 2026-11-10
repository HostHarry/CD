# POPE, thinking prompt, 256/128/64 — native / VCD / VIGOR

Status: **COCO complete** (generation + scoring). A-OKVQA and GQA are still
running and will be added when finished.

## Protocol

- Checkpoint MMaDA-8B-MixCoT, MAGVITv2, crop 512, seed 42, temperature 0, CFG 0,
  `low_confidence` remasking.
- Generation 256 tokens / 128 steps / block 64 (MMaDA official `L/2`, `L/4`).
- Safe EOS early stopping `safe_eos_v1` (stop only when the generated prefix
  through the first EOS is fully committed).
- Prompt `pope_think_v1`: chat-template user message = MMaDA official thinking
  instruction + `\n` + POPE question with ` Please answer yes or no.` removed;
  no system message; image as `<|mmu|><|soi|>[1024 codes]<|eoi|>`.
- VCD: author APC formula, alpha 1, beta 0.1, noise_step 500. VIGOR: author
  token-ID loop with MASK image ablation, FP64, epsilon 1e-8, probe_interval 1.
- Two GPU shards, merged verbatim (`merge_receipt.json`).

## Scoring

- **Primary** (`primary_metrics.json`): official POPE rule applied to the text
  after the last `</think>`; rows without `</think>` fall back to the full
  text. Paired image-level bootstrap vs native (2000 repeats, seed 42).
- `metrics.json`: the runner's full-text official scoring (reference only; the
  reasoning text starts with "The image shows ..." and biases it to Yes).

## COCO results (primary, Acc / F1 %)

| Split | native | VCD | VIGOR |
|---|---:|---:|---:|
| Random | 75.70 / 77.15 | 78.37 / 78.99 | 76.13 / 77.91 |
| Popular | 75.37 / 76.91 | 75.87 / 77.12 | 74.60 / 76.82 |
| Adversarial | 71.90 / 74.49 | 71.87 / 74.30 | 71.20 / 74.51 |
| **Mean** | 74.32 / 76.19 | 75.37 / 76.80 | 73.98 / 76.42 |
| Yes ratio | 57.7% | 56.0% | 60.2% |
| `</think>` closed | 96.9% | 98.0% | 93.9% |
| Answer not starting with yes/no | 10.1% | 4.4% | 18.1% |
| Mean answer tokens | 47.6 | 51.3 | 66.0 |
| Mean executed steps (of 128) | 39.4 | 37.1 | 36.7 |
| Mean seconds / input | 6.5 | 11.9 | 11.8 |

Paired 95% CI of delta accuracy vs native: VCD Random [+1.0, +4.4] points;
all other splits for VCD and VIGOR include zero.

## DeepSeek text judge (`coco/judge/`)

Protocol `pope_text_llm_judge_v1`, reused verbatim from the cloud CD-APC
package (`cdapc/source/src/cdapc_lab/evaluation/pope_llm_judge.py`, sha256
`1ab1a8ef…fa57`), runner `methods/judge_pope_think.py`. Model `deepseek-flash`,
temperature 0, thinking disabled, max_tokens 192. The API receives only the
plain POPE question (thinking instruction and yes/no suffix removed) and the
complete decoded response; no image and no gold. The judge returns
answered / missing / ambiguous with a verbatim evidence quote; scoring is
local and missing/ambiguous count as wrong. 15,132 unique requests, 0 errors;
every request and raw response is in `api_records.jsonl.gz`.

| Split (judge, Acc / F1 %) | native | VCD | VIGOR |
|---|---:|---:|---:|
| Random | 76.37 / 77.54 | 78.70 / 79.36 | 77.40 / 79.42 |
| Popular | 75.37 / 76.73 | 76.33 / 77.61 | 74.67 / 77.04 |
| Adversarial | 72.10 / 74.38 | 72.43 / 74.75 | 71.40 / 74.80 |
| **Mean** | 74.61 / 76.22 | 75.82 / 77.24 | 74.49 / 77.09 |
| missing / ambiguous rows | 43 / 16 | 12 / 19 | 4 / 117 |

Paired image-level bootstrap of delta accuracy vs native (95% CI, points):
VCD Random +2.33 [+0.73, +4.07], Popular +0.97 [−0.50, +2.57], Adversarial
+0.33 [−1.33, +2.07]; VIGOR Random +1.03 [−0.53, +2.63], Popular −0.70
[−2.27, +0.87], Adversarial −0.70 [−2.30, +0.93].

Payloads use format B (see `../../README.md`).
