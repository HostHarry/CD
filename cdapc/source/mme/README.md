# MME CD-APC r70, L256/I128

This isolated extension snapshots the local experiment framework so adding MME
does not change the generation fingerprint of active MMMU or queued K4/K2 runs.
The original source hashes are in `reports/source_snapshot.json`.

Protocol: MMaDA-8B-MixCoT, all 2,374 MME questions / 1,187 image pairs / 14 tasks.
Use each official question verbatim, including `Please answer yes or no.`,
without an additional reasoning instruction. Keep the current CD-APC settings:
r=0.7, alpha=0.5, beta=0.1, u=0.1, W=K=16, canvas 256, iteration limit 128,
sequential branches, no cache, and eos_prefix_complete. This queued condition
does not include a new Native baseline.

The data is pinned to lmms-lab/MME revision
`d6c9023f017b564f7b3ccccf5348166bce8fdbcd`. Every parquet is SHA256-checked;
all questions and labels are matched to the original official evaluation ZIP.
Each image pair stays on one GPU shard. Images retain their original bytes.

Scoring uses the original first-four-character Yes/No parser at official
evaluation commit `dd2950902889cd614d4edf606827240166d29381`.
Each category scores `100 * (Acc + Acc+)`, where Acc+ requires both questions
for an image to be correct. Perception has a maximum of 2,000, Cognition 800.
Budget exhaustion remains incorrect with fixed denominators; its generation
cost is included. The unmodified upstream scorer independently verifies all
14 category scores from exported text files. No API judge is required.

Preparation and CPU checks:

```sh
.venv/bin/python scripts/prepare_mme.py
PYTHONPATH=src .venv/bin/python -m pytest -q tests/integration/test_mme.py
```

The controller is `scripts/run_mme_256_128.py`. It waits for both K4/K2
generation stages and acquires the existing shared GPU queue lock before
launching three workers. Completed predictions are reused only with the
matching generation fingerprint.

Live state is written to this workspace's `reports/index.json` and the parent
experiment workspace's `reports/mme_r70_256_128_runs.json`. Final JSON/Markdown
reports are copied into both workspaces as `mme_cdapc_r70_L256_I128`.
Worker logs are `logs/shard{0,1,2}.log`; the controller log is `logs/driver.log`.

The official scorer's extra dependencies (`scikit-learn==1.7.2` and
`threadpoolctl==3.6.0`) are isolated in `vendor/`; the shared inference
environment is unchanged. The scorer subprocess explicitly adds this directory
to its Python path.
