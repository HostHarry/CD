# mme_think: DeepSeek answer extraction

Protocol: pope_text_llm_judge_v1; deepseek-flash; full response and original question.
No gold or image sent. Missing/ambiguous count incorrect; API errors block final scoring.
MME category Acc + Acc+ is rechecked using the unmodified official scorer.
Paired intervals concern question accuracy, not MME total score.

| Method | Perception /2000 | Cognition /800 | Total /2800 | Acc % | Unknown |
|---|---:|---:|---:|---:|---:|
| native | 1022.72 | 246.43 | 1269.14 | 58.26 | 208 |
| vcd | 996.78 | 211.79 | 1208.56 | 59.65 | 153 |
| vigor | 1028.08 | 264.29 | 1292.37 | 59.73 | 98 |
