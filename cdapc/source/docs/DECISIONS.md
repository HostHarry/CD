# Engineering decisions and acceptance boundaries

1. Work began in a new directory. The provided historical repository was not
   copied, imported or linked. Only freshly fetched official source and
   SHA-256-verified official binary assets are used.
2. Official source commits are frozen in `upstream.lock.json`. Source tree hashes
   include the exact deployed Python files. All local code remains readable.
3. `patches/lavida-sdpa-mask.patch` contains only two attention-mask forwarding
   fixes. Normal calls with no bias remain unchanged. Actual-kernel receipt and
   visual-content invariance are checked on the real model.
4. LaViDa imports DeepSpeed in its official model module; a pinned dependency is
   required even for this inference-only task. No training workflow is added.
   Activation checkpointing is disabled for inference using the official API.
5. On the RTX 5090 BF16 MMaDA fixture, paired versus sequential logits differed
   by up to 0.5 and the fallback commit differed although argmax and gate booleans
   matched. Therefore configuration accepts sequential execution only. Paired
   execution remains an internal verification path and is not claimed equivalent.
6. Smoke uses two outcome-independent test IDs and L=128. It does not select
   parameters, establish accuracy or substitute for the guide's L=512 CoT run.
   Native steps/block are kept distinct from CD-APC W/K.
7. Official score wraps the exact pinned upstream pure function. Strict scoring
   is a separate track. The official permissive parser is not silently corrected.
8. Writer exclusion uses a process-scoped POSIX advisory lock, so abrupt process
   termination does not strand a run behind an unrecoverable lock file.
9. Weight binaries on shared storage are hash-verified against official LFS and
   referenced externally to avoid copying 34 GB. Model code/config/tokenizer
   comes from fresh official sources. Unchanged large-file stat receipts avoid
   re-reading every weight shard at every preflight; changes trigger SHA-256.
10. CCAW is an independent subsequent change after fixed-window software and
    controlled replay acceptance. Real replay coverage remains zero among the
    16 preregistered L=512 samples: no empirical causal estimate is available.
    Other datasets and new research mechanisms remain deferred. No unimplemented
    method files or registry placeholders are created.
11. The official conversation module eagerly downloads three global tokenizers,
    including an unused Dream model. `lavida-offline-tokenizer.patch` defers those
    three loads. The adapter binds the exact locked checkpoint tokenizer to the
    unchanged official LLaDA conversation template. Other model templates are
    outside this adapter's scope and require explicit tokenizer binding.
12. Generation source hashes include only the selected adapter and, for CD-APC,
    the decoder. Other adapters and scoring-only changes do not invalidate raw
    predictions. The initial MMaDA acceptance is preserved at commit `54c046b`;
    its pre-refinement generation fingerprints remain historical evidence.
13. Two official MMaDA MMU paths differ: the README entry (`inference_mmu.py`
    with `configs/mmada_demo.yaml`) uses resize plus 512-pixel center-crop and
    no sot marker, while the VLMEvalKit wrapper defaults to 256, adds the sot
    marker and a white-image fallback. We adopt the README path because it is
    the documented MixCoT MMU entry, its config names MMaDA-8B-MixCoT, and the
    stage-3/4 CoT training configs use 512. MMaDA runs before this change used
    256 (source commit `df31f79`); they remain historical evidence and are never
    resumed or mixed with new runs.
14. With 512 input and only the shared question, 11/16 mechanism IDs hit the
    first EOS within 1-2 response tokens (answer only), leaving CD-APC almost
    nothing to act on. MMaDA's official MMU demo prefixes the thinking
    instruction, so it is treated as part of the model's official input
    protocol, like its chat template; the shared question text stays
    method-independent. VLMEvalKit's USE_COT mode appends the same instruction;
    we follow the README demo prefix. The intermediate 512 runs without the
    instruction (run IDs starting `20261105T2242` to `20261105T2251`) were
    produced from an uncommitted working tree and remain historical.
15. POPE follows the published MMaDA native baseline (HostHarry/CD
    @6848cbf577) byte-for-byte: RUCAIBox questions (random, popular,
    adversarial order) plus the " Please answer yes or no." suffix, lmms-lab
    parquet image bytes verified against the baseline `images.tsv`, the
    official RUCAIBox answer rule, per-split metrics and the image-clustered
    paired bootstrap (2,000 repeats, seed 42, deltas B−A with A = native).
    Inputs are deduplicated to 5,127 keys; L=128 (native baseline runs
    128/64/32). An interim question-only, L=256 design produced three partial
    runs (315 MMaDA adversarial / 311 MMaDA popular / 232 LaViDa adversarial
    predictions); those were stopped and remain historical — native MMaDA
    under question-only prompts answers yes ~96%, and the formal baseline
    uses the yes/no suffix. 57 lmms-lab adversarial questions read "imange";
    the lmms/RUCAIBox consistency check normalizes only that typo and prompts
    keep RUCAIBox text. MMaDA only; no LaViDa POPE configs.
16. `stop_rule: eos_prefix_complete` deviates from guide v1 `fill_canvas` at
    the user's request for evaluation speed: decoding stops once the earliest
    committed EOS has no pending position before it. Normalized text and
    scores are identical by construction — committed tokens are final and the
    trajectory is unchanged up to the stop; post-EOS positions remain MASK.
    Cost and mechanism statistics cover only iterations up to the stop. M3CoT
    configs stay on `fill_canvas`, and no equal-compute claim is made versus
    native (native runs its full 64 steps). On the 5,127-input POPE run the
    rule fired for 479 inputs (`eos_prefix_complete`) while 4,648 ended
    `canvas_filled`, at a mean 2.54 generation seconds per input.
17. An r-gate threshold sweep on POPE was preregistered by the user before any
    sweep result existed: τ_r ∈ {0.8, 0.7, 0.6, 0.5} plus the 0.9 run already
    done, everything else fixed (MMaDA, same 5,127 inputs and shard manifests,
    L=128, iteration_limit 128, alpha 0.5, beta 0.1, u gate 0.1, W=K=16,
    sequential, cache disabled, eos_prefix_complete, commit_log on). The τ_r=0.9
    POPE run had 0 qualified flips and 0 actual switches — CD-APC acted only
    through commit order — so the sweep measures when contrast replacements
    start and what they do. It runs on the POPE evaluation set itself (no dev
    split), all grid points are reported, and no grid point may later be
    presented as a default or tuned setting without stating it was selected on
    POPE test. Fixed metrics in advance: per-split accuracy, precision,
    recall, F1, yes-ratio, invalid-ratio; paired bootstrap vs native and vs
    τ_r=0.9; answer transitions; answer_switch attribution; confidence-shift
    summary; mechanism totals; cost; finish_reason counts.
18. MMMU-val follows a WINO/VISAGE-style protocol chosen by the user: WINO's
    `mmmu_val_mmada` prompt construction and judge rules
    (@9549a1d0c3cbd9dc9b8f7cd09be8a7aac322d6fb) with the official MMMU rule
    parser (@51ce7f3e) as a second track. Deviations from WINO: our README
    `inference_mmu` input layout (no `<|sot|>`, chat template instead of the
    hand-written header template), first image only per question, first-EOS
    normalization, and a DeepSeek `deepseek-flash` judge instead of GPT-4o.
    VISAGE used Qwen3-8B and WINO/VISAGE different judges, so absolute numbers
    are comparable to neither paper. Provider aliases such as `deepseek-flash`
    may be remapped server-side, so every raw judge request/response and the
    returned model id are stored per sample under `runs/<run>/judge/`. The
    primary metric — judge accuracy — was fixed before any MMMU result.

Reports distinguish executed passes, failures and unrun checks. CPU tests do not
prove real-adapter correctness. Small-sample negative results remain in the run
artifacts. No historical scores are carried into new results.
