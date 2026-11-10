# Paper versus current configuration audit (2026-11-06)

The later full-run investigation, including actual MMaDA mechanism probes,
author-loop equivalence and CHAIR scoring sensitivity, is recorded in
`REPRODUCTION_AUDIT.md`.

This audit describes the full-schedule runs before the user's subsequent
safe-EOS optimization request. The resumed profile retains all planned
parameters below but can stop after a fully committed EOS prefix; see
`EOS_OPTIMIZATION.md`. Historical statements about always running the complete
schedule apply to the archived pre-optimization runs.

This is a read-only protocol audit requested by the user after the independent
queues started. No running source, configuration, manifest or prediction was
changed. The current run is a **matched-budget MMaDA adaptation comparison**,
not a reproduction of the original papers' complete experimental configurations.

## Current locked runs

Both VCD and VIGOR use:

| Benchmark | Generation positions | Total denoising steps | Block length | Temperature |
|---|---:|---:|---:|---:|
| POPE-COCO yes/no | 128 | 64 | 32 | 0 |
| CHAIR | 64 | 32 | 16 | 0 |

These are the previously selected MMaDA comparison budgets. The POPE budget
was explicitly selected by the user in the earlier experiment; it was carried
over when adding the two baselines. It is not sourced from either new paper.
The length is the fixed diffusion generation span; EOS truncates the decoded
answer but does not terminate the denoising loop.

## VIGOR

Primary paper: https://arxiv.org/html/2611.05066v1 .

Section 4.1 and Appendix B require native and VIGOR to share generation length
and the block schedule, and use temperature zero. Current runs follow those
comparison principles. The paper evaluates LLaDA-V and Lumina-DiMOO, not MMaDA,
and reports CHAIR at lengths 64 and 512. It does not prescribe one fixed
generation schedule for MMaDA.

The author-linked anonymous evaluation release was additionally inspected
online on 2026-11-06. Its values below come from command-line defaults and the
explicit `decoding_schedule` function, not from a recovered record of the
authors' exact launch commands:

| Author evaluation entry point | Length / steps / block | Current MMaDA run |
|---|---|---|
| POPE LLaDA-V and Lumina scripts, CLI defaults | 16 / 8 / 16 | 128 / 64 / 32 |
| CHAIR Lumina, 64-token branch | 64 / 32 / 64 | 64 / 32 / 16 |
| CHAIR Lumina, 512-token branch | 512 / 256 / 128 | Not scheduled |

Sources:

- https://anonymous.4open.science/r/519_VIGOR-4051/scripts/run_pope_lladav.py
- https://anonymous.4open.science/r/519_VIGOR-4051/scripts/run_pope_lumina.py
- https://anonymous.4open.science/r/519_VIGOR-4051/scripts/run_chair_lumina.py

Use the POPE CLI `--steps=8` default, not the different defaults in internal
helper signatures: the CLI explicitly passes its arguments into those helpers.
The current POPE denoising step count is eight times the author script's CLI
default; this is not a measured eightfold wall-time ratio.

The runtime's pinned GitHub source `THU-BPM/Vigor@f907674...` contains AMBER
scripts. We reuse its generation function, not its AMBER task budgets. Its
discriminative default length 4 must not be described as the POPE default.

Numerical convention: the current runtime follows the published code's
probability clamping at `1e-8`; the paper pseudocode writes epsilon addition.
That source-level difference was already documented in `vigor/README.md`.

## VCD

Primary paper, including Appendix A:
https://arxiv.org/pdf/2311.16922 .

The original method evaluates autoregressive LVLMs and has no masked-text
denoising-step or block-length setting. Its `T` / `noise_step` refers to image
corruption, not the number of MMaDA text-denoising steps.

| Setting | Paper / released code | Current run |
|---|---|---|
| Contrast alpha | Paper 1 | 1 |
| APC beta | Paper 0.1; Python parser 0.1; shell example 0.2 | 0.1 |
| POPE image-noise setting | Paper Appendix A: T=999; Python parser defaults to 500 | 500 |
| Main decoding strategy | Paper directly samples the softmax distribution | Greedy, temperature 0 |
| LLaVA script max_new_tokens | Hardcoded 1024, with autoregressive EOS stopping | POPE fixed span 128 |
| Text denoising steps / block | Not defined in the original AR method | POPE 64 / 32 |

The paper also studies greedy decoding in its sampling-strategy ablation;
greedy is not its principal reported sampling protocol. The paper specifies
noise T=500 for MME and LLaVA-Bench, whereas its POPE setting is 999.
Thus choosing the Python parser defaults was not sufficient to reproduce the
paper's POPE noise configuration.

The author's LLaVA script is pinned locally under
`vcd/upstream/experiments/eval/object_hallucination_vqa_llava.py`.
Its 1024 autoregressive maximum is not an instruction to run 1024 diffusion
steps or to use a 1024-position MMaDA span.

There is also an unresolved adaptation detail beyond numeric budgets:
VIGOR Appendix C.1 describes a diffusion VCD baseline whose commit decision is
“driven by the clean side alone.” Our VCD adapter ranks positions using candidate
probabilities from the contrastive distribution, makes one commit decision, and
shares the resulting text/MASK state between the clean and distorted passes.
The appendix does not give an explicit confidence formula for that commitment
decision. “Clean side alone” must not be overinterpreted as proof that it ranks
with the original, pre-contrast clean probabilities: it may instead describe
which branch controls the shared update. The inspected pinned public GitHub
revision exposes native/VIGOR AMBER scripts, not the diffusion VCD baseline.
Therefore equivalence to that baseline remains unverified, rather than a
demonstrated mismatch. The original VCD AR code itself does not specify how to
rank multiple masked positions.

## Interpretation

Current results can compare native, VCD and VIGOR under the same chosen MMaDA
input/length/step/block protocol. They cannot be labelled as matching the
original papers' full experimental configurations or reported scores.
POPE currently covers COCO only, whereas the papers also evaluate other POPE
source datasets. A change to the shorter author-script schedule requires
new matching native predictions and a separate run manifest; old 128/64/32
predictions cannot be mixed into it.
