# Safe EOS stopping and resumed progress

Authorized by the user on 2026-11-06 after inspecting the one-token POPE answers.
The yes/no question suffix, images, preprocessing, checkpoints, seed and method
parameters remain unchanged. Planned POPE generation stays **128/64/32**;
planned CHAIR generation stays **64/32/16**.

## Safety condition

Stop only if the actual generated state contains an EOS and **every generated
position up to and including the first EOS is already non-MASK**.
Prompt tokens are excluded from this check. A predicted EOS in logits is not
enough, and a committed EOS with earlier MASK holes is not enough.

The supported generators update MASK positions only. Once this contiguous
prefix is committed, subsequent forwards cannot change its tokens or its EOS.
The answer decoder already ends at that EOS. Therefore those later forwards
cannot change the scored answer. No claim is made that the uncomputed suffix
or the number of recorded steps remains identical.

The full 128/64-position generation context is retained during all executed
steps. This is not a reduction of the initial generation span or a change to
block boundaries, candidate probabilities, confidence ordering or commit counts.

`common/eos.py` adds a model-call proxy that checks actual state before a
forward and raises a private completion exception. `Runtime` catches only
that exception and extracts the complete prefix. MMaDA model files and the
VCD/VIGOR decoding functions and author sources remain unchanged.
If no safe prefix exists, the original full schedule executes.

## Preservation and provenance

The old queues were stopped with these committed predictions:

| Dataset | VCD folder | VIGOR folder |
|---|---|---|
| POPE | native 5127, VCD 636 | native 5127, VIGOR 637 |
| CHAIR | native 500, VCD 8 | native 500, VIGOR 8 |

`results/eos_resume_before.json` records ordered payload hashes and manifest
hashes. The original `results/{pope_full,chair_full}` directories are preserved.

`resume_eos.py` imports every committed payload byte-for-byte into
`results/early_eos/{pope_full,chair_full}`, with new manifests locking the
validated source and `early_eos: safe_eos_v1`. It checks all non-execution
protocol fields and permits only the explicitly reviewed wrapper/runner changes.
The old source snapshots and manifests are not rewritten.
Each destination has a `resume_import.json` receipt with source paths,
row hashes/counts and the validation report hash.

The new queues skip all imported predictions and recover only unfinished work.
Native full-dataset predictions are reused. The same flag applies to the
subsequent CHAIR queue, with fallback to the full schedule when necessary.

## Output semantics

New records carry:

- `execution_profile`: `safe_eos_v1` or `full_schedule_v1`.
- `planned_steps`, `executed_steps`, `planned_generated_positions`.
- `early_stopped` and `stop_reason`.

For an actual early stop:

- `generated_ids` contains the committed answer prefix **including EOS**.
- `text` is the same decoded answer as the full-schedule reference.
- `raw_text` decodes that actually returned prefix.
- `generation_state_ids` preserves the entire actual generation state at stop,
  including committed suffix tokens and still-uncomputed MASKs.
- `uncomputed_suffix_masks` records those suffix MASKs explicitly.
- `trace` contains all and only executed steps; it reconstructs
  `generation_state_ids`, not an invented full completion.
- `remaining_masks=0` refers to the returned complete answer prefix.

Historical full-schedule payloads remain untouched. Records without an
execution-profile field are interpreted as `full_schedule_v1`.
Timing summaries are separated by profile; mixed overall timings must not be
reported as the new optimized throughput.

Startup fingerprints compare prompt IDs and answer tokens through EOS for
the optimized profile. Full-schedule runs retain full-token fingerprints.
The weaker suffix scope is deliberate and explicitly recorded in
`multi_gpu/<session>/execution.json`.

## Validation and commands

The unit suite covers an EOS before an unfinished prefix, prompt EOS, empty
answers, absent EOS fallback, every decoder's unchanged commit prefix, answer
fingerprints across full/early records, and immediate release of intermediate
tensors with cyclic GC disabled. All 14 tests pass.

`verify_eos.py` first audits every preserved native/method trajectory for
immutable commits and reconstructible states. It then checks real-model output
on POPE and CHAIR, including fast/slow historical stopping cases and recent
samples. It compares every answer token, input/noise provenance and every
executed committed position/token/confidence against preserved full runs.
Two POPE examples per method are also rerun without optimization in the same
process for paired timing and fresh full-output comparisons.

Reports: `vcd/results/early_eos_validation.json` and
`vigor/results/early_eos_validation.json`. These reports must match the current
source hashes before `resume_eos.py` admits imported rows.

From the project root, the resume workflow is:

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 .venv/bin/python verify_eos.py --method vcd
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 .venv/bin/python verify_eos.py --method vigor
.venv/bin/python resume_eos.py --method vcd
.venv/bin/python resume_eos.py --method vigor
OMP_NUM_THREADS=4 .venv/bin/python -m vcd.pipeline --gpus 0 --early-eos
OMP_NUM_THREADS=4 .venv/bin/python -m vigor.pipeline --gpus 1 --early-eos
```

The two GPU validation commands and the two queue commands may run in parallel
on their assigned GPUs. Do not run validation against a GPU occupied by a queue.
Do not rerun verification merely to resume an already validated queue.

## Measured verification results

After the traceback-lifetime fix below, all 14 unit tests and 23 real-model
comparison cases passed. The GPU cases
include native and both methods, POPE and CHAIR, early-stop cases and full-budget
fallback cases. Every historical trajectory was audited before import.

Two paired POPE examples per method, measured in the same process after warmup:

| Method | Full schedule, mean end-to-end | Safe EOS, mean end-to-end | Ratio |
|---|---:|---:|---:|
| VCD | 15.240 s | 3.563 s | 4.28× |
| VIGOR | 15.289 s | 0.525 s | 29.10× |

VCD executed 16 and 15 steps in these pairs; VIGOR executed 2 and 2 steps.
These are two-example measurements, not the full-dataset mean or a guaranteed
wall-time speedup. Later inputs can require more steps; CHAIR often reaches the
full budget. Use `timing_by_execution_profile` or recent optimized database
records for updated throughput.

`results/eos_resume_preservation.json` confirms that every imported payload and
every historical manifest is unchanged, with the same ordered row hashes as
the archived source run. The first import reused 636 VCD and 637 VIGOR POPE
predictions; the final import also retains the additional completed rows
described below. Eight CHAIR predictions per method and all native predictions
are preserved throughout.

## Continuous-run memory fix

The first resumed execution passed answer checks but later ran out of GPU
memory. Retaining the completion exception also retained its traceback and
the interrupted generator's large logits/probability tensors through a cycle
back to the Runtime frame. Cyclic collection was not timely enough for a
continuous worker.

The catch block now immediately clears `completion.__traceback__` and retains
only the small token-state tensor and EOS index, not the exception. This does
not change any numerical operation. A regression test disables cyclic GC and
uses weak references to prove that intermediate tensors are freed on every
early return. GPU verification additionally runs eight consecutive early
exits per method, checking live allocated memory without calling `empty_cache`
or forcing collection.

The first attempt is archived under `results/early_eos_attempt1/`, together with
its original source snapshots and manifests. It committed **13 additional VCD**
and **8 additional VIGOR** POPE predictions before stopping. Those rows are valid
completed outputs and are preserved too; the resumed totals start at
**VCD 649 / VIGOR 645**, plus all native and prior CHAIR predictions.
`results/eos_attempt1_preservation.json` locks those additional rows.

The second import uses, respectively:

```bash
.venv/bin/python resume_eos.py --method vcd \
  --additional-run vcd/results/early_eos_attempt1
.venv/bin/python resume_eos.py --method vigor \
  --additional-run vigor/results/early_eos_attempt1
```

The importer verifies both source manifests and contracts, audits all completed
early-stop states, and copies only committed prediction rows. Failed/uncommitted
claims are retried by the new queue.

## Confirmed live continuation

Both resumed workers passed their preserved-answer fingerprints and continued
past the first attempt's failure point. VCD processed at least 25 consecutive
new inputs and VIGOR more than 25 without memory errors before the resumption
was declared healthy. GPU validation's eight consecutive checks per method
had identical live allocations of 16,561,906,176 bytes after every completion.

At the recorded live snapshot, POPE counts were **VCD 705 / 5127** and
**VIGOR 915 / 5127**. Each baseline still contains all 5127 native predictions.
The last 25 newly executed inputs averaged:

| Method | End-to-end seconds/input | Executed steps/input |
|---|---:|---:|
| VCD | 3.476 | 14.80 |
| VIGOR | 0.927 | 3.76 |

These moving-window means differ from the two-example paired measurements
because the stopping point varies by input. Full sample keys, commit times,
step counts and timing values for this snapshot are saved in
`results/eos_live_validation.json`. Both queues continue automatically into
CHAIR and scoring.
