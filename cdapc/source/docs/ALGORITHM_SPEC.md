# CD-APC algorithm contract v1

Both branches run from the same complete response state. Each forward recomputes
all transformer layers; there is no inter-iteration KV/prefix cache. Normal
attention is bidirectional. The reference masks every nonvisual query → actual
visual key edge, in every self-attention layer/head. Visual key positions are
derived from official expansion, including noncontiguous positions. Previously
committed text may still carry visual information.

On sorted legal text IDs, compute FP32 `logp_I` and `logp_A`. MASK, visual codes
and non-output special tokens are excluded; EOS remains legal. Illegal logits
are never subtracted. Nonfinite legal logits cause an explicit failure.

For each response position:

- `b = argmax(logp_I)`, ties choose the smallest legal token ID.
- `pool = logp_I >= max(logp_I) + log(beta)`, `0 < beta <= 1`.
- `m = sum(exp(logp_I)[pool])`, using the original probability mass.
- `gain = logp_I - logp_A`; `h = logp_I + alpha * gain_used`, `alpha >= 0`.
- `q = softmax(h on pool)`, `c = argmax(h on pool)`.
- `u(z) = p_I(z)`, `r(z) = m*q(z)`.

The two independent eligibility conditions are `(not enabled_u or u >= tau_u)`
and `(not enabled_r or r >= tau_r)`. Disabling a gate changes no proposal or
ordering statistic. Neither score is a calibrated correctness probability.

The window is the leftmost **W pending positions**, not a physical span.
Qualified positions sort by descending `r`, descending `u`, then ascending
response index. Commit at most K, with `1 <= K <= W`. If no position qualifies,
rank `rho = min(score/threshold over enabled gates)`, then r, then index; commit
exactly one **base token b** at the winning position. Fallback location selection
still uses the selected target's statistics.

| Variant | Gain used | Gate/order target | Qualified write | Fallback write |
| --- | --- | --- | --- | --- |
| full | measured | c | c | b |
| gain_zero | zero, both forwards still run | b=c | b | b |
| timing_only_crossed | measured | c | b | b |
| base_candidate_visual | measured | b, including q(b) | b | b |

The immutable prompt is never writable. Only pending positions can be committed.
Every step strictly reduces MASK count. `fill_canvas` stops only after all
response slots are committed, at most L steps. A smaller explicit iteration
budget returns `budget_exhausted` and retains the remaining MASK IDs. EOS is
handled by the common offline normalizer. History marks only this iteration's
observed positions that remain pending; older out-of-window observations expire.

Every branch call is counted at the denoiser execution point. Sequential pairs
cost two calls/two branch evaluations. A paired batch costs one call/two branch
evaluations. Child records are not added again. `logical_query_positions` is a
size measure, not FLOPs.

The first planned qualified actual flip in a full+fixed trajectory is the sole
eligible replay event per sample. Branch A writes b at its target, branch B writes
c; all other planned writes are identical. Both continue with full+fixed. A
recomputed event mismatch is recorded and never silently repaired. Replay costs
are separate research costs. No-event and failed events remain in coverage
denominators; ordinary strict parse failures count wrong within valid pairs.

`ccaw_v1` is a separate controller added after fixed-window software acceptance.
Before each commit, compute `f_def = ineligible / |C|`,
`f_flip = eligible proposal flips / |C|`, and `f_inst` as candidate changes among
positions in C observed in the immediately previous iteration and still pending.
Empty history has `f_inst=0` with support explicitly recorded as zero.
Pressure is `(f_def + f_flip + f_inst) / 2`; EMA is
`gamma * previous_ema + (1-gamma) * pressure`.
The target is `maximum - floor((maximum-minimum)*ema + 0.5)`.
Clamp the change to `[-shrink_step, expand_step]`, then the resulting width to
`[minimum, maximum]`. This update occurs once per iteration and affects only
the next window. K, model inputs, candidates and gates remain unchanged.
Initial W is `initial`, initial EMA is zero. Setting all three width parameters
equal degenerates to fixed. All pressure components, support, EMA, target and
widths are saved even with detailed trace disabled. This reconstructed version
does not claim equivalence to historical controllers or a measured benefit.
