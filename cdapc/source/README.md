# CD-APC experiments — new framework v1

Fresh source built from the supplied 2026-11-05 implementation guide. The only
methods are `native` and `cdapc`; diagnostic variants share one CD-APC engine.
This repository contains no historical experiment imports or bytecode loaders.

Remote workspace: `/root/autodl-tmp/cdapc-experiments`.
Run commands from this directory. Use `.venv/bin/python` on the supplied server.

```bash
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q tests
python -m cdapc_lab doctor --config configs/experiments/mmada_m3cot_fixed_smoke.yaml
python -m cdapc_lab verify-native --config configs/experiments/mmada_native_smoke.yaml
python -m cdapc_lab verify-adapter --config configs/experiments/mmada_m3cot_fixed_smoke.yaml
python -m cdapc_lab infer --config configs/experiments/mmada_m3cot_fixed_smoke.yaml
python -m cdapc_lab score --run runs/RUN_ID --profile official_and_strict
python -m cdapc_lab summarize --run runs/RUN_ID --evaluation EVALUATION_FINGERPRINT
python -m cdapc_lab infer --config configs/experiments/mmada_m3cot_fixed_smoke.yaml --resume-run runs/RUN_ID
python -m cdapc_lab replay --run runs/RUN_ID --event first_eligible_flip --design content_swap
```

Use the `run` and full `evaluation_fingerprint` returned by the commands. A
summary may omit the evaluation only when exactly one exists. Scores do not
load a model, and cannot mutate raw predictions. An interrupted run retains its
fixed denominator. The POSIX writer lock releases automatically on process exit.

```bash
python -m cdapc_lab compare --run-a runs/A --evaluation-a EVAL_A \
  --run-b runs/B --evaluation-b EVAL_B --unit sample_id --metric strict_accuracy
```

`scripts/accept_model.py mmada` and `scripts/accept_model.py lavida` run the bounded
real acceptance suite with one model resident at a time. They use two fixed
M³CoT IDs and a 128-token canvas. This is engineering acceptance, not a benchmark
result or a parameter-selection experiment. The separate preregistered
`scripts/accept_replay.py` study uses 16 IDs and L=512; zero eligible flip events
were observed, so it provides no valid real-model content-swap pairs.

After fixed acceptance, `scripts/accept_window.py mmada` (or `lavida`) compares
fixed with `ccaw_v1` on the same two smoke IDs, changing only the controller.
It also checks real-model degeneration when minimum=maximum=initial=16.
The generated `*_m3cot_ccaw_v1_smoke.yaml` configs use the guide's starting
parameters, without parameter selection. See `docs/ACCEPTANCE.md` for executed
checks, run IDs and interpretation limits.

Official dependencies are under ignored `upstream/` and `models/` directories.
`upstream.lock.json` records source commits and every model asset SHA-256.
Fresh official tokenizer/config files are downloaded. Existing **binary weights
only** may be external symlinks after SHA-256 equality with the official LFS
metadata is established. No cached Python files are imported. For unchanged
verified large files, preflight uses their size/mtime receipt; changed files are
re-hashed. Small tokenizer/config files are always re-hashed.

The server environment uses Python 3.12.3, PyTorch 2.8.0+cu128, Transformers
4.50.3. The small local CPU test environment uses Python 3.9.6 and PyTorch 2.8.0;
the supported project runtime is Python >=3.10. The local environment is not a
model-validation environment.
For another GPU host, install the matching CUDA-enabled PyTorch first, then
`DS_BUILD_OPS=0 python -m pip install -r requirements-inference.txt` and the
editable package. `reports/runtime.freeze.txt` records the tested server
environment. The supplied asset paths are machine-specific and are checked by
`doctor`; the local mirror is for source, tests and results, without GPU weights.

Read `docs/ALGORITHM_SPEC.md`, `docs/PROTOCOL_SPEC.md`, `docs/DECISIONS.md` and the
actual JSON reports under `reports/` before interpreting an experiment. A report
marked failed is evidence, not a successful acceptance. No large benchmark grid
is launched by installation or preflight.
