# POPE, thinking prompt, 256/128/64 — native / VCD / VIGOR

Status: **COCO and A-OKVQA complete** (generation, scoring and DeepSeek judge).
GQA is still running and will be added when finished.

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

## DeepSeek text judge (`<dataset>/judge/`)

Protocol `pope_text_llm_judge_v1`, reused verbatim from the cloud CD-APC
package (`cdapc/source/src/cdapc_lab/evaluation/pope_llm_judge.py`, sha256
`1ab1a8ef…fa57`), runner `methods/judge/judge_pope_think.py`. Model `deepseek-flash`,
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

## A-OKVQA results

Primary (official rule after `</think>`), Acc / F1 %:

| Split | native | VCD | VIGOR |
|---|---:|---:|---:|
| Random | 74.57 / 76.87 | 77.13 / 78.63 | 75.43 / 77.95 |
| Popular | 72.97 / 75.77 | 76.20 / 77.95 | 74.30 / 77.17 |
| Adversarial | 67.50 / 72.23 | 70.97 / 74.34 | 68.33 / 73.28 |
| **Mean** | 71.68 / 74.96 | 74.77 / 76.97 | 72.69 / 76.14 |
| Yes ratio | 62.9% | 59.4% | 64.2% |
| `</think>` closed | 96.9% | 98.0% | 94.4% |

DeepSeek judge, Acc / F1 %:

| Split | native | VCD | VIGOR |
|---|---:|---:|---:|
| Random | 75.60 / 77.74 | 77.53 / 78.99 | 76.80 / 79.70 |
| Popular | 73.30 / 76.16 | 76.37 / 78.23 | 74.93 / 78.26 |
| Adversarial | 67.70 / 72.35 | 71.07 / 74.58 | 68.83 / 74.02 |
| **Mean** | 72.20 / 75.41 | 74.99 / 77.27 | 73.52 / 77.33 |
| missing / ambiguous rows | 36 / 26 | 10 / 21 | 6 / 139 |

Paired image-level bootstrap of delta accuracy vs native (judge, 95% CI,
points): VCD Random +1.93 [+0.33, +3.53], Popular +3.07 [+1.47, +4.70],
Adversarial +3.37 [+1.83, +4.90]; VIGOR Random +1.20 [−0.43, +2.90], Popular
+1.63 [+0.10, +3.17], Adversarial +1.13 [−0.37, +2.67].

Payloads use format B (see `../../README.md`).
