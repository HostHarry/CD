# VIGOR baseline

This directory owns Visual Gain Ordering for multimodal discrete diffusion
models. It is not one of the other similarly named visual-grounding methods.

- Author code: https://github.com/THU-BPM/Vigor
- Commit: `f907674eec79bb2e30597b22564ba4f644e6256e`
- Lock: `source.json`; configuration: `config.yaml`; implementation: `decode.py`.
- Results and logs: `results/`, separate from VCD.
- The pinned public repository has no license file; public availability does
  not establish an open-source license.

The complete author `generate_text_understanding_visual_ablation` function
from `scripts/run_amber_lumina.py` executes through an MMaDA call adapter.
At every step it masks image codes in the ablation branch and computes FP64
probabilities of the **normal branch's same candidate token**:

```text
score = (c_img - c_noimg)
      + log(clamp(c_img, min=1e-8))
      - log(clamp(c_noimg, min=1e-8))
```

The released code uses clamping, not addition of epsilon. The candidate token
stays from the normal branch; only the positions' commitment order changes.
`probe_interval=1` recomputes the ablation pass every step. MMaDA SOI/EOI tokens
remain outside the masked visual-code span.

The later full-run audit also compared this function with the author's
POPE/CHAIR release. Under baseline=False and the supported valid schedule,
the specialized function bodies have identical ASTs. The remaining protocol
differences include the generation/block budgets and prompts, documented in
`../REPRODUCTION_AUDIT.md`.

A local torch proxy records author topk decisions and returns the original
tensors. It does not globally patch torch. Trace `committed_confidence` is a
visual-gain score and can be negative; it is not a probability.

From the **MMaDA-Baselines project root**:

```bash
OMP_NUM_THREADS=4 .venv/bin/python -m vigor.pipeline --gpus 1 --early-eos
OMP_NUM_THREADS=4 .venv/bin/python -m vigor.run --benchmark POPE --gpus 1 --early-eos
```

Use the pipeline or the single-benchmark command, not both on the same output
directory at once. Resume with the identical command.
Shared environment/data/scorer instructions are in `../README.md`.
The active resumed profile writes to `results/early_eos/`; old full-schedule
results remain archived. See `../EOS_OPTIMIZATION.md` for the verified stop rule.
