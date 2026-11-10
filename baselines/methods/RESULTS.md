# Completed VCD / VIGOR comparison on MMaDA

The subsequent [reproduction audit](REPRODUCTION_AUDIT.md) verified all stored
trajectories and recomputed the scores. It also documents unresolved VCD
position-ranking equivalence, protocol differences and the inherited CHAIR
`bus` inflection defect. A separate one-word scoring correction did not reverse
the main method ordering; the formal scores below are unchanged.

Both independent pipelines reached `generation_and_scoring_complete`.
Each method has 5127 unique POPE predictions covering all 9000 POPE-COCO rows,
plus 500 CHAIR captions and completed CHAIR scoring. Native predictions were
reused under the verified matching input protocol.

## POPE-COCO

Each split contains 3000 questions. Values are percentages.

| Method | Random Accuracy / F1 | Popular Accuracy / F1 | Adversarial Accuracy / F1 |
|---|---:|---:|---:|
| Native | 83.03 / 82.49 | 81.97 / 81.59 | 76.00 / 76.91 |
| VCD | 81.43 / 81.94 | 80.30 / 81.05 | 74.53 / 76.79 |
| VIGOR | 82.73 / 82.76 | 83.57 / 83.45 | 75.90 / 77.47 |

VCD accuracy decreases by 1.60, 1.67 and 1.47 percentage points respectively.
Its recall increases from native's 79.93% to 84.27%, while false positives
increase in every split. The early 419-question partial result did not predict
this full-dataset outcome.

VIGOR improves Popular accuracy by 1.60 points and F1 by 1.86 points. Its Random
and Adversarial accuracy changes are -0.30 and -0.10 points.
Image-level paired bootstrap intervals (500 images, 2000 bootstrap resamples,
seed 42) support the Popular improvement:

- Accuracy change 95% interval: [+0.90, +2.33] percentage points.
- F1 change 95% interval: [+1.16, +2.61] percentage points.

VIGOR's Random/Adversarial accuracy and F1 intervals include zero. VCD's
accuracy intervals are below zero on all three splits, while its F1 intervals
include zero. Bootstrap resamples are not independent model-seed repetitions.

All POPE methods have zero empty/invalid answers and a mean effective answer
length of one token. The prompt retains `Please answer yes or no.`.

## CHAIR: 500 images

CHAIRs and CHAIRi are lower-is-better; object Recall is higher-is-better.
Values are percentages except for answer length.

| Method | CHAIRs | CHAIRi | Object Recall | Mean answer tokens |
|---|---:|---:|---:|---:|
| Native | 21.80 | 9.90 | 54.92 | 61.994 |
| VCD | 25.20 | 10.10 | 57.55 | 60.464 |
| VIGOR | 22.80 | 9.38 | 55.35 | 59.708 |

VCD increases both hallucination indicators, despite higher object recall.
VIGOR lowers CHAIRi by about 0.52 points, but increases CHAIRs by 1.00 point.
Its captions are also somewhat shorter. These are mixed outcomes, not evidence
of a uniform hallucination reduction; no CHAIR significance claim is made.

## Execution and scope

Planned schedules remain POPE 128/64/32 and CHAIR 64/32/16, with crop 512,
temperature zero and seed 42. Safe EOS stopping was enabled after preserving
the initial full-schedule predictions. Scores use the same completed answer
prefix; uncomputed suffixes are explicitly represented in the raw records.

Across the optimized POPE records only:

| Method | Optimized records | Mean end-to-end seconds | Mean executed steps |
|---|---:|---:|---:|
| VCD | 4491 | 3.430 | 14.633 |
| VIGOR | 4490 | 0.716 | 2.820 |

The remaining 636/637 method records and all native predictions retain their
historical full-schedule execution. Do not compare mixed-profile averages as
if they were simultaneous optimized throughput measurements.

These are MMaDA adaptation results under the chosen matched protocol, not the
papers' original configuration or backbone results. See `PAPER_PROTOCOL_AUDIT.md`
for differences, including VCD noise_step=500 versus the original paper's
POPE T=999. CHAIR uses the pinned Python3 derivative documented in README.md.

## Result files

- `vcd/results/early_eos/pope_full/metrics.json`
- `vigor/results/early_eos/pope_full/metrics.json`
- `vcd/results/early_eos/chair_full/chair_metrics.json`
- `vigor/results/early_eos/chair_full/chair_metrics.json`

Each run retains its manifest, source snapshot, raw prediction database,
commit traces and reuse receipts. Both baseline `queue_status.json` files
report completion; both CHAIR `status.json` files set `chair_scored=true`.
