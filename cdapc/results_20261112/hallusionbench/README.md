# HallusionBench: CDAPC and Native, direct and think

All 1129 VLMEvalKit HallusionBench rows (951 image, 178 text-only), matched row by row
to the CD baseline record `hallusion_full_original_256_128` (commit in `baseline/COMMIT`).
TSV MD5/SHA256 and pinned VLMEvalKit scorer files (rev 3c19214) are verified.

| Arm | Decoder | Prompt |
|---|---|---|
| cdapc_direct | r70/W16/K4, L256/I128, EOS-prefix candidates | original TSV question |
| native_direct | official mmu_generate 256/128/64, safe EOS | original TSV question |
| cdapc_think | as cdapc_direct | MMaDA thinking instruction + question |
| native_think | as native_direct | MMaDA thinking instruction + question |

Seed 42. Native uses block 64 at the user's request; the CD baseline Native/VCD/VIGOR
use block 32 (direct only). Text-only rows have no visual branch: CDAPC arms use the
Native output with the same prompt (the baseline's VCD/VIGOR policy). Native safe EOS is
the baseline's StopAtClosedPrefix rule (stop before a forward once the prefix through
the first EOS is mask-free); at block 32 it reproduced baseline Native text, prompt-ID
hashes and executed steps on 6 smoke rows.

Scoring: unchanged VLMEvalKit `YOrN_Extraction` / `Hallusion_rating`; aAcc, fAcc, qAcc,
Avg; Unknown counts incorrect; no API judge. Think arms score the text after the last
`</think>` (whole text when unclosed); a raw whole-text diagnostic is kept. The scorer
reproduces the published baseline Native/VCD/VIGOR metrics exactly. Paired deltas use a
question-set cluster bootstrap (l2-category + set_id; 2000 draws, seed 42).

```sh
.venv/bin/python scripts/prepare_data.py     # data + manifests (done)
.venv/bin/python scripts/run.py --prepare    # freeze + preflight (done)
.venv/bin/python scripts/run.py              # generate missing rows per arm, then score
.venv/bin/python scripts/run.py --score      # rescore saved inventories
```

Status `reports/status.json`; results `reports/summary.md`, `reports/result.json`.
