# mme_original: DeepSeek answer extraction

Protocol: pope_text_llm_judge_v1; deepseek-flash; full response and original question.
No gold or image sent. Missing/ambiguous count incorrect; API errors block final scoring.
MME category Acc + Acc+ is rechecked using the unmodified official scorer.
Paired intervals concern question accuracy, not MME total score.

| Method | Perception /2000 | Cognition /800 | Total /2800 | Acc % | Unknown |
|---|---:|---:|---:|---:|---:|
| native | 987.95 | 272.50 | 1260.45 | 60.11 | 0 |
| vcd | 1034.90 | 306.07 | 1340.97 | 62.55 | 0 |
| vigor | 1017.38 | 281.43 | 1298.81 | 61.25 | 2 |
