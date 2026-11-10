# POPE 256/128/32

COCO, A-OKVQA and GQA: 9000 rows per dataset and method.
Existing yes/no prompts, seed 42, crop 512, temperature 0, safe EOS stopping.
Values below are percentages, averaged across the three splits. F1 is not pooled.

| Dataset | Method | Accuracy | F1 | Precision | Recall | Yes ratio |
|---|---|---:|---:|---:|---:|---:|
| coco | native | 80.61 | 79.79 | 83.80 | 76.27 | 45.66 |
| coco | vcd | 79.33 | 80.34 | 76.98 | 84.13 | 54.80 |
| coco | vigor | 80.96 | 80.93 | 81.55 | 80.47 | 49.51 |
| aokvqa | native | 80.09 | 80.71 | 78.82 | 82.87 | 52.78 |
| aokvqa | vcd | 79.39 | 80.89 | 75.91 | 86.73 | 57.34 |
| aokvqa | vigor | 79.61 | 81.02 | 76.37 | 86.47 | 56.86 |
| gqa | native | 79.58 | 80.09 | 78.58 | 81.80 | 52.22 |
| gqa | vcd | 79.61 | 81.15 | 75.84 | 87.40 | 57.79 |
| gqa | vigor | 79.66 | 81.08 | 76.30 | 86.67 | 57.01 |
