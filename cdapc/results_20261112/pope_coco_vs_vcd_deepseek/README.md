# Full COCO CDAPC/VCD DeepSeek comparison

Continuation of the user-authorized matched VCD comparison after generation
of all COCO inputs. The previous frozen partial report remains unchanged.
This directory covers all 5127 unique inputs and 9000 evaluation rows:
three complete, balanced splits over 500 images.

The coordinator waits for full CDAPC official/DeepSeek grading, then reuses
those cached judgments and the previously judged VCD subset. Only remaining
VCD requests need new API calls. It repeats row/image/question/label matching
against the complete, hash-verified VCD export at CD commit `fcc76d6`.

Protocol, failure handling and paired image bootstrap are identical to the
earlier partial comparison. The official thinking instruction and question
match; the methods retain their recorded configurations and different seed
protocols. CDAPC's prompt was selected after the 64-image pilot.

Progress: `reports/status.json`; log: `logs/evaluation.log`.
Final output: `reports/comparison.json` and `reports/comparison.md`.
