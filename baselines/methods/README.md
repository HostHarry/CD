# methods/ — MMaDA-Baselines code snapshot

Project-authored code for the VCD/VIGOR MMaDA baseline experiments recorded
under `../records/`. This is a snapshot of `/root/autodl-tmp/MMaDA-Baselines`
(the independent baseline project), taken at export time; the live project is
the source of truth for any later changes.

## What is here

- `common/` — shared runtime, metrics, POPE/CHAIR/MMMU/HallusionBench protocol
  and scoring code (`runtime.py`, `run.py`, `metrics.py`, `eos.py`, ...).
- `vcd/` — VCD (visual contrastive decoding) method: config, decode/run/
  pipeline entry points, `source.json` pin manifest, and `upstream/` — the
  pinned author sources **included verbatim** because the upstream repository
  carries an Apache-2.0 `LICENSE` (see `vcd/source.json`,
  `vcd/upstream/provenance.json`).
- `vigor/` — VIGOR method: config, decode/run/pipeline entry points,
  `source.json` pin manifest, `upstream/provenance.json`. The VIGOR author
  sources themselves are **not** redistributed: the pinned public repository
  (THU-BPM/Vigor) contains no license file (see `vigor/source.json`). Fetch
  them with `python baseline_sources.py --fetch` (URLs, revision and
  per-file sha256 are pinned there and in `vigor/upstream/provenance.json`).
- `configs/` — experiment configs incl. `mmada-source.json` (official MMaDA
  pin manifest).
- `tests/` — unit tests.
- Top level — entry points (`multigpu.py`, `run_mmmu.py`, `run_hallusion.py`,
  `resume_eos.py`, `verify_eos.py`, `verify_runtime.py`, `migrate_results.py`,
  `prepare_pope_extended.py`, `baseline_sources.py`), docs (`README.md`,
  `AGENTS.md`, `EOS_OPTIMIZATION.md`, `REPRODUCTION_AUDIT.md`, `RESULTS.md`,
  `MMMU.md`, `HALLUSION.md`, `PAPER_PROTOCOL_AUDIT.md`), `LICENSE`,
  `requirements*.txt`, `.gitignore`.

## Not included

- `results/` — all run outputs; the completed ones are exported under
  `../records/` instead.
- `models/`, `training/` — official MMaDA sources (see pin below).
- `data/` — datasets, images, annotations and evaluators (large; each record
  dir ships the exact sample files + hashes instead).
- `.venv/`, `__pycache__/`, checkpoints.
- `CD_APC_Implementation_Guide_v1_No_DCD.md` (excluded per project decision).
- `vigor/upstream/{README.md,scripts/}` — unlicensed author files (pin
  recorded in `vigor/source.json`).

## Pinned revisions

- Official MMaDA: commit `3cdb8709635beb29f66e4e23cd66060872ac3996`
  (`configs/mmada-source.json` lists per-file sha256).
- VCD author sources: github.com/DAMO-NLP-SG/VCD @
  `d6568ff81b8fd306a49e630df44f2db5c2300191` (Apache-2.0, vendored under
  `vcd/upstream/`).
- VIGOR author sources: github.com/THU-BPM/Vigor @
  `f907674eec79bb2e30597b22564ba4f644e6256e` (no license; fetch via
  `baseline_sources.py --fetch`).

## Reproducing a run

1. Fetch official MMaDA at the pinned commit and verify against
   `configs/mmada-source.json` (the project keeps `models/` and the used
   `training/` utilities unchanged).
2. Checkpoints used: MMaDA-8B-MixCoT and magvitv2 (paths in each run's
   `manifest__*.json` under `../records/`).
3. `pip install -r requirements.txt` (+ `requirements-chair.txt` /
   `requirements-hallusion.txt` for those benchmarks), then run the relevant
   entry point from the project root, e.g.
   `python -m common.run --benchmark POPE --methods native vcd vigor --early-eos ...`
   (exact commands are in each experiment README and the copied `config*.yaml`).
