# VCD baseline

This directory owns the MMaDA adaptation of Visual Contrastive Decoding.

- Author code: https://github.com/DAMO-NLP-SG/VCD
- Commit: `d6568ff81b8fd306a49e630df44f2db5c2300191`
- License: Apache-2.0, retained under `upstream/LICENSE`.
- Lock: `source.json`; configuration: `config.yaml`; implementation: `decode.py`.
- Results and logs: `results/`, separate from VIGOR.

The loader executes the author's unchanged diffusion-noise function and the
three original contrast/APC assignments from `vcd_sample.py`. It avoids the
upstream command-line imports and global Transformers sampling monkeypatch.
Noise is added to the preprocessed CPU image, then the noisy image is encoded
with MAGVIT. Only image-code tokens change; text and image delimiters are retained.
Both forwards see the same current text/MASK state at every denoising step.

Fixed parameters are the author Python evaluation parser defaults:
`cd_alpha=1`, `cd_beta=0.1`, `noise_step=500`.
The author's shell example instead defaults beta to 0.2; this run does not
silently mix those configurations. VCD includes its original adaptive
plausibility constraint (APC).

MMaDA adaptations: discrete image codes after distortion, MMaDA preprocessing,
temperature zero, native fixed denoising transfer counts, and position ranking
by the VCD candidate probability. These are not the author's autoregressive
model or sampling protocol. Noise is seeded with the common per-input seed 42.

The probability used for position ranking is computed **after APC**. Even at
alpha=0, APC can therefore change the commit order relative to native MMaDA.
A four-input real-model audit confirmed this mechanism while preserving the
four answers; disabling APC in that diagnostic recovered native commits.
This is an explicit adaptation choice, whose equivalence to VIGOR Appendix C's
VCD baseline remains unverified. See `../REPRODUCTION_AUDIT.md`.

From the **MMaDA-Baselines project root**:

```bash
OMP_NUM_THREADS=4 .venv/bin/python -m vcd.pipeline --gpus 0 --early-eos
OMP_NUM_THREADS=4 .venv/bin/python -m vcd.run --benchmark POPE --gpus 0 --early-eos
```

Use the pipeline or the single-benchmark command, not both on the same output
directory at once. Resume with the identical command.
Shared environment/data/scorer instructions are in `../README.md`.
The active resumed profile writes to `results/early_eos/`; old full-schedule
results remain archived. See `../EOS_OPTIMIZATION.md` for the verified stop rule.
