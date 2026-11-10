# POPE prompt diagnostic — native, 256/256/32

192 rows (32 per split x label), native only, seed 42, crop 512, temperature 0, CFG 0, low_confidence, safe EOS stopping.
Image: <|mmu|><|soi|>[1024 codes]<|eoi|> + chat/training text as in run_diag.py.
Scorer: unchanged official POPE parser (common.metrics.pope_metrics).
Values are percentages; per-split rows plus an all-rows row per variant.

## Overall metrics

| Variant | Rows | Accuracy | F1 | Precision | Recall | Yes ratio | Invalid ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 192 | 50.52 | 66.67 | 50.26 | 98.96 | 98.44 | 0.00 |
| B | 192 | 75.00 | 75.00 | 75.00 | 75.00 | 50.00 | 1.04 |
| C | 192 | 50.00 | 66.67 | 50.00 | 100.00 | 100.00 | 100.00 |
| D | 192 | 76.56 | 74.29 | 82.28 | 67.71 | 41.15 | 0.00 |

## Per split

| Variant | Split | Accuracy | F1 | Precision | Recall | Yes ratio | Invalid ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| A | random | 50.00 | 65.96 | 50.00 | 96.88 | 96.88 | 0.00 |
| A | popular | 50.00 | 66.67 | 50.00 | 100.00 | 100.00 | 0.00 |
| A | adversarial | 51.56 | 67.37 | 50.79 | 100.00 | 98.44 | 0.00 |
| A | all | 50.52 | 66.67 | 50.26 | 98.96 | 98.44 | 0.00 |
| B | random | 64.06 | 64.62 | 63.64 | 65.62 | 51.56 | 0.00 |
| B | popular | 81.25 | 81.25 | 81.25 | 81.25 | 50.00 | 0.00 |
| B | adversarial | 79.69 | 79.37 | 80.65 | 78.12 | 48.44 | 3.12 |
| B | all | 75.00 | 75.00 | 75.00 | 75.00 | 50.00 | 1.04 |
| C | random | 50.00 | 66.67 | 50.00 | 100.00 | 100.00 | 100.00 |
| C | popular | 50.00 | 66.67 | 50.00 | 100.00 | 100.00 | 100.00 |
| C | adversarial | 50.00 | 66.67 | 50.00 | 100.00 | 100.00 | 100.00 |
| C | all | 50.00 | 66.67 | 50.00 | 100.00 | 100.00 | 100.00 |
| D | random | 71.88 | 67.86 | 79.17 | 59.38 | 37.50 | 0.00 |
| D | popular | 85.94 | 85.25 | 89.66 | 81.25 | 45.31 | 0.00 |
| D | adversarial | 71.88 | 68.97 | 76.92 | 62.50 | 40.62 | 0.00 |
| D | all | 76.56 | 74.29 | 82.28 | 67.71 | 41.15 | 0.00 |

## Answer diagnostics

| Variant | yes-start parsed no | no-start parsed yes | neither start | tok len mean | tok len median | EOS rate | steps mean | e2e mean s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0 | 0 | 0 | 10.23 | 10.0 | 1.000 | 32.0 | 5.03 |
| B | 0 | 0 | 2 | 67.72 | 69.5 | 1.000 | 83.2 | 13.50 |
| C | 0 | 0 | 192 | 77.24 | 87.5 | 1.000 | 96.0 | 15.26 |
| D | 0 | 0 | 0 | 1.00 | 1.0 | 1.000 | 32.0 | 4.97 |

## Variant B thinking format

- rows with `<think>`: 192/192 (100.00%)
- rows with `</think>`: 191/192 (99.48%)
- think_close_missing (full text used as answer): 1
- secondary parse of FULL B text — accuracy 51.04, F1 66.19, yes ratio 94.79

## Paired parsed-answer changes vs A

| Variant | Changed | Fixed | Broken | yes-label fixed | yes-label broken | no-label fixed | no-label broken |
|---|---:|---:|---:|---:|---:|---:|---:|
| B | 93 | 70 | 23 | 0 | 23 | 70 | 0 |
| C | 3 | 1 | 2 | 1 | 0 | 0 | 2 |
| D | 110 | 80 | 30 | 0 | 30 | 80 | 0 |

## Example outputs (first 3 label=no rows per variant)

### A
- `adversarial:2274` (adversarial, label=no) q0: Is there a chair in the image?
  - answer: 'Yes, there is a chair in the image.' parsed=yes eos=True steps=32 e2e=4.98s
- `adversarial:1208` (adversarial, label=no) q0: Is there a car in the image?
  - answer: 'Yes, there is a car in the image.' parsed=yes eos=True steps=32 e2e=4.97s
- `adversarial:1314` (adversarial, label=no) q0: Is there a cup in the image?
  - answer: 'Yes, there is a cup in the image.' parsed=yes eos=True steps=32 e2e=4.98s

### B
- `adversarial:2274` (adversarial, label=no) q0: Is there a chair in the image?
  - answer: 'Yes, there is a chair in the image.' parsed=yes eos=True steps=96 e2e=15.62s
  - generated: '<think> The image shows a close-up of a chair with a white cushion and a backrest. The chair appears to be part of a larger structure, possibly a chair or a piece of furniture, and the presence of a shadow on the wall suggests that it is a chair. The angle and the shape of the cushion are consistent with that of a chair.</think> Yes, there is a chair in the image.'
- `adversarial:1208` (adversarial, label=no) q0: Is there a car in the image?
  - answer: 'No, there is no car in the image.' parsed=no eos=True steps=64 e2e=10.46s
  - generated: '<think> The image shows a group of people in a swimming pool, with some floating on inflatable rafts and others in the water. There is no car visible in the image. The focus is on the people and the pool setting.</think> No, there is no car in the image.'
- `adversarial:1314` (adversarial, label=no) q0: Is there a cup in the image?
  - answer: 'No, there is not a cup in the image.' parsed=no eos=True steps=96 e2e=15.49s
  - generated: '<think> The image shows a bedroom with a bed, a chair, a desk, and a mirror. There is a small object on the desk that appears to be a cup, but it is not clearly identifiable as a cup. The cup is not in the image.</think> No, there is not a cup in the image.'

### C
- `adversarial:2274` (adversarial, label=no) q0: Is there a chair in the image?
  - answer: 'A black and white photograph of a chair with a shadow cast on the wall.' parsed=yes eos=True steps=32 e2e=5.08s
- `adversarial:1208` (adversarial, label=no) q0: Is there a car in the image?
  - answer: 'The image depicts a lively scene at a swimming pool. Several people are enjoying themselves, with some floating on inflatable rafts and others swimming in the water. The pool is filled with individuals, and the atmosphere appears relaxed and social. The setting includes a brick wall with framed pictures and a staircase leading to an upper level. The overall ambiance suggests a fun and casual poolside gathering.' parsed=yes eos=True steps=96 e2e=15.28s
- `adversarial:1314` (adversarial, label=no) q0: Is there a cup in the image?
  - answer: 'A vintage photograph of a bedroom with a large bed, a desk, and a chair. the room is decorated with floral wallpaper and has a vintage aesthetic. there is a mirror on the wall, and a window with a white frame. the bed is neatly made with a white comforter and pillows. the room has a dated appearance, suggesting it may be from the mid-20th century. the text in the image indicates that the room is part of a hotel room, and it is described as being dated.' parsed=yes eos=True steps=128 e2e=20.34s

### D
- `adversarial:2274` (adversarial, label=no) q0: Is there a chair in the image?
  - answer: 'no' parsed=no eos=True steps=32 e2e=4.95s
- `adversarial:1208` (adversarial, label=no) q0: Is there a car in the image?
  - answer: 'No' parsed=no eos=True steps=32 e2e=4.95s
- `adversarial:1314` (adversarial, label=no) q0: Is there a cup in the image?
  - answer: 'no' parsed=no eos=True steps=32 e2e=4.96s

## Determinism vs paused no-suffix run

- A committed unique inputs: 192
- compared: 138, identical: 138, mismatched: 0

