# Independent baseline project

User explicitly requested an independent framework outside MMaDA-DEGAP.
Work in this project; do not import runtime/method code from that project.
Historical source/manifest reads are confined to the explicit one-time
`migrate_results.py` command. Both method algorithms, author sources, configs,
entry points and results live separately in `vcd/` and `vigor/`.

Official MMaDA is pinned at `3cdb8709635beb29f66e4e23cd66060872ac3996`.
Keep `models/` and the used official training utilities unchanged; their hashes
are in `configs/mmada-source.json`. Shared checkpoints are immutable. There
is no training requirement. Only `training/utils.py` and
`training/prompting_utils.py` are used by the standalone inference runtime.

Always run modules from the project root. Use this project's own `.venv`.
Read README.md and the relevant method README before changing generation.
VCD uses author parser defaults alpha=1, beta=0.1, noise_step=500, with APC.
VIGOR uses the complete author token-ID loop with MASK image ablation,
FP64 probabilities, clamp epsilon=1e-8 and probe_interval=1. Never substitute
contrastive token logits for VIGOR's position ranking.

POPE is the yes/no-suffix protocol at 128/64/32. CHAIR is the fixed 500 images
at 64/32/16. Both use crop 512, temperature=0, CFG=0, seed=42.
Do not tune parameters on the evaluation results.

Do not change source/config/data during active runs. Each pipeline owns its
queue lock; each benchmark owns a run lock and SQLite atomic task queue.
One model replica per GPU. Never start independent queues on the same GPU
without checking capacity; never kill another user's process. Restart using
the same command. Resume rejects changed manifests.

Full completion requires actual generation plus scoring, not startup or smoke
status. Preserve raw output and nonfinite failures. No paid external judge is
authorized or needed by these POPE/CHAIR queues.

Checks:

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q common vcd/decode.py vcd/run.py vcd/pipeline.py \
  vigor/decode.py vigor/run.py vigor/pipeline.py baseline_sources.py migrate_results.py \
  multigpu.py verify_runtime.py tests
git diff --check
```

Real-model migration verification: `verify_runtime.py --method vcd` or
`--method vigor`; each checks 2 POPE and 2 CHAIR samples for native plus the
selected method against preserved outputs and all commits. Do not repeat
large optimization experiments for directory-only changes.

## Authorized safe EOS profile (2026-11-06)

The user authorized stopping once the generated prefix through its first EOS
is fully committed, and resuming existing progress. Active commands now use
`--early-eos`; active outputs are `vcd/results/early_eos/` and
`vigor/results/early_eos/`. Read `EOS_OPTIMIZATION.md` before editing this path.
Original full-schedule outputs/manifests remain archives. Do not mix their
timings with optimized timings. All historical prediction payloads are retained.

`common/eos.py` is an observational model proxy. Do not alter author loops or
native model code to implement stopping. Never stop from candidate logits or
an EOS with preceding MASK holes. Keep the planned span and schedule intact.
For stopped rows, reconstruct traces against `generation_state_ids`; the
returned `generated_ids` contains only the completed prefix through EOS.
Record uncomputed suffix MASKs rather than inventing EOS padding.

The approved importer is `resume_eos.py`, gated by `verify_eos.py` reports and
explicit old/new manifests. A resume command does not require repeating GPU
validation if sources have not changed.
