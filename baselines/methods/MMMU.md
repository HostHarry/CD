# MMMU: VCD and VIGOR

This run follows the input protocol supplied by the user in
`HostHarry/CD@e48814d6994bd3a8ed1647f5e3906758135bfc6d`, including its
`mmmu_wino_reasoning_v1` instruction and **single selected image** policy.
The downloaded reference files have been verified against their Git blob hashes
and are archived under `data/mmmu_reference/e48814d/`.
Only pure input-formatting helpers were executed to verify the local preparation;
neither the reference decoder nor the DEGAP runtime is imported.

## Fixed protocol

| Setting | Value |
|---|---|
| Methods | Existing MMaDA VCD and VIGOR adaptations |
| Dataset | `lmms-lab/MMMU@364f2e2eb107b36e07ff4c5a15f5947a759cef47`, validation |
| Parquet SHA256 | `18e0e97776217e7b5fc992dc9594014e7adc0cc0473f624ea957c18702e16681` |
| Coverage | All 900 IDs: 847 multiple-choice, 53 open questions |
| Shards | Two fixed shards of 450, original parquet indices `shard::2` |
| GPUs / batch | GPU 0 and GPU 1, one model replica and batch 1 on each |
| MMaDA checkpoint/tokenizer | `3ee0085f0c42541f1134aae30482954451952406` |
| MAGVITv2 | `5c3fa78f8b3523347c5cd1a4c97f3c4e96f33d5d` |
| MMaDA source | `3cdb8709635beb29f66e4e23cd66060872ac3996`, unchanged |
| Model precision / attention | BF16 / SDPA, bidirectional, no cross-round cache |
| Image preprocessing | Original bytes, RGB, bicubic short-side resize 512, center crop 512 |
| Generation | 256 response positions, 128 steps, block 64, temperature 0, CFG 0 |
| Stop | Existing verified `safe_eos_v1`, only after a committed prefix through EOS |
| Seed | 20261105, reset for each unique input and method |
| VCD | Alpha 1, beta 0.1, noise step 500, APC confidence ordering |
| VIGOR | Unchanged author loop, image-code MASK ablation, probe interval 1 |

The requested three-GPU plan was changed by the user to **two shards** after
the current machine was verified to expose only two RTX 5090 GPUs.
Every GPU processes its 450 inputs with VCD, then with VIGOR.
Each method has separate per-shard databases, manifests, source snapshots and locks.

The initial VLMEval TSV inventory in `data/mmmu_validation/` is **not** the
generation input. The final input comes from the pinned original parquet,
whose SHA256 was checked against the dataset's LFS metadata. It preserves
the original image bytes used by the reference experiment.

## Input equivalence

Preparation verifies all 900 user-message strings and image-selection decisions
against the supplied reference helpers. The checkpoint tokenizer configuration
and official image transform source match the uploaded files byte for byte.

The user content begins with the exact `<think> ... </think>` reasoning
instruction, then the question and `A. ...` options, including the trailing
newline before the tokenizer's own trimming. There is no added final-letter
instruction or system message. All three image-placeholder forms are removed.
The actual input is MMU, SOI, 1024 image codes, EOI, then the tokenizer's chat
template. Its BOS is retained once.

The 43 questions containing multiple images remain in the 900-item evaluation.
As in the reference, only the first sorted distinct `<image n>` selected from
question plus options is used, with the documented image_1 fallback.
All 900 selections in this dataset resolve to image_1. This is not an evaluation
in which all images are supplied. The 900 selected image occurrences contain
879 distinct original byte streams.

Gold answers are in a separate file and are absent from model input records.
The preparation manifest records input hashes and both disjoint shard hashes.

## Run and resume

Run from this project root with its `.venv`:

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m common.mmmu_protocol
OMP_NUM_THREADS=2 .venv/bin/python -m unittest discover -s tests -v
OMP_NUM_THREADS=4 .venv/bin/python -u run_mmmu.py --gpus 0 1
```

The launcher first runs four fixed acceptance inputs for each method: the
longest prompt, the first sample, an open question, and a multiple-image
question. It verifies actual input IDs across methods, reconstructs every
commit, validates EOS stopping, and records peak memory and costs. Passing
acceptance automatically starts both full shard workers and then scoring.
`--preflight-only` stops after acceptance.

Repeating the full command resumes committed SQLite predictions. Source,
configuration or data changes are rejected. Acceptance outputs are separate
from the formal 900 predictions and are not mixed into formal throughput.
There is one excluded warmup per method, shard and execution session.

- Global status and logs: `results/mmmu_wino_v1/`
- VCD predictions: `vcd/results/mmmu_wino_v1/shard{0,1}/predictions.sqlite`
- VIGOR predictions: `vigor/results/mmmu_wino_v1/shard{0,1}/predictions.sqlite`
- Input data and hashes: `data/mmmu_wino_v1/`
- Config: `configs/mmmu_wino_v1.yaml`

The generic existing runner uses `--benchmark custom`; the dataset is explicitly
MMMU in its locked sample records and the top-level protocol. Per-shard
completion represents 450 inputs. Only the top-level status plus successful
scoring establish completion of both 900-item methods.

## Scoring and cost

The official functions are pinned to
`MMMU-Benchmark/MMMU@51ce7f3e829c16bb44bc5445782686b4c3508794`.
They receive the entire normalized text, including reasoning. For example,
the official parser can prioritize a bracketed option inside reasoning over
a plain final letter. Its random fallback is retained in the official track
and explicitly counted.

Fallback choices use a deterministic per-question seed: the first 32 bits of
SHA256(`mmmu-official-choice-v1:20261105:<sample_id>`), shared across methods.
A second diagnostic uses the same rules but counts random fallbacks as
incorrect. This is a local no-guess diagnostic, not a claim to reproduce the
reference's unpublished `mmmu_rule_noguess_v1` wrapper. The three stringified
lists of acceptable open answers are decoded into lists before `eval_open`.

The reference mechanism report's LLM-judge accuracy uses a different scorer.
It must not be directly mixed with these local official/no-guess accuracies.

Scoring also audits every full-run commitment trajectory and checks identical
normal prompt IDs across both methods. It reports overall/type/subject
accuracy, paired correct/incorrect changes, effective answer length, EOS and
early stopping counts, actual steps, denoiser forwards, vision encodings,
measured inference time and peak allocated memory. Each step performs two full
denoiser forwards. VCD has two vision encodings per sample; VIGOR has one.
Raw tokens, full stopped generation states, prompt IDs and traces remain in
SQLite. The top-level wall time includes initialization, acceptance and scoring;
per-sample inference times do not.
