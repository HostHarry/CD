# HallusionBench: 256/128/32

Full dataset: 1129 rows, including 951 image and 178 text-only questions.
Original TSV questions in the MMaDA chat template; safe EOS stopping.
Text-only rows use identical native decoding for all three methods.
Pinned VLMEvalKit exact matching; Unknown answers count as incorrect.

| Method | Scored | Accuracy (aAcc %) | fAcc % | qAcc % | Unknown | Mean steps |
|---|---:|---:|---:|---:|---:|---:|
| native | 1129/1129 | 42.60 | 14.64 | 9.67 | 146 | 17.71 |
| vcd | 1129/1129 | 40.48 | 12.90 | 9.01 | 160 | 14.67 |
| vigor | 1129/1129 | 40.12 | 12.16 | 9.23 | 188 | 7.36 |
