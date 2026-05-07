# CoreX Next Stable Optimization Report

Date: 2026-05-05

## Summary

This round implemented three opt-in CoreX candidates and validated several existing
GPU fallback switches. None passed the `wrecking_ball400` stability gate, so no new
optimization was promoted to the default path.

The only source changes are env-gated:

- `UIPC_COREX_MATCONV_BLOCK_REDUCE=1`: block-per-segment reductions for matrix
  converter 3x3 and 3x1 segment reduce.
- `UIPC_COREX_SPMV_SEGMENTED_ROW=1`: CoreX-safe symmetric SpMV variant with
  row-block segmented accumulation plus off-diagonal atomics.
- `UIPC_COREX_PCG_FUSED_SPMV_DOT=1`: fused PCG `spmv_dot()` use for `Ap` and
  `pAp`, skipping the explicit post-SpMV synchronization.

## Validation Data

All correctness gates listed below passed without NaN, exception, or max-iteration
failure markers:

- Matrix converter block reduce: `simple90`, `simple300`, `stack120`
- SpMV segmented row-block: `simple90`, `simple300`, `stack120`
- PCG fused SpMV-dot: `simple90`, `simple300`, `stack120`
- Contact early-active candidate: `simple90`, `simple300`, `stack120`
- ABD split screening: `simple90` for vertex, BDF1 energy, BDF1 gradient/hessian,
  tolerance, and line-search light reduction

Short and long performance gates:

- Matrix converter block reduce:
  - `wb150`: default `64s`, candidate `53s`; PCG sum `54899 -> 46183`
  - `wb400`: default `172s`, candidate `175s`; PCG sum `165306 -> 163625`
  - Decision: keep opt-in. Long run regressed because matrix conversion time rose
    from about `3.42s` to `5.18s`.
- SpMV segmented row-block:
  - `wb150`: default `64s`, candidate `53s`; `pcg.spmv` `1654ms -> 1345ms`
  - `wb400`: default `172s`, candidate `194s`; PCG sum `165306 -> 170840`
  - Decision: keep opt-in. Short-run gain did not survive the long gate.
- PCG fused SpMV-dot:
  - `wb150`: default `64s`, candidate `61s`; PCG sum `54899 -> 56200`
  - Decision: keep opt-in and do not run `wb400`; PCG count worsened and SpMV
    bucket increased despite removing nearly all `spmv_sync` time.
- ABD fallback split:
  - `wb80` screening showed no stable winner. Every tested switch was wall-neutral
    or slower and increased PCG sum versus default.
  - Decision: no default changes.
- Contact early-active candidate:
  - `wb150`: default `64s`, candidate `67s`; PCG sum `54899 -> 67119`
  - Decision: keep opt-in only.

Artifacts are in `/tmp/corex_next_opt_20260505/`.

## Defaulting Decision

No new flag should be enabled by default in this round.

The dominant pattern is that several candidates improve a short run, but either
increase algorithm bucket time directly or perturb PCG/Newton behavior enough to
fail the `wb400` stability rule. The current default remains the safer baseline.

## Next Recommendations

The most useful next step is not another env sweep. The better target is direct
conditioning/assembly quality work that reduces PCG iteration count without relying
on nondeterministic execution-order changes:

- Revisit contact Hessian scaling and SPD projection by contact type using the
  existing SPD/PCG diagnostics.
- Add selected-set equivalence diagnostics before changing contact filtering again.
- Profile high-PCG frames from `wb400` and inspect which contact class and matrix
  assembly phases correlate with iteration spikes.
