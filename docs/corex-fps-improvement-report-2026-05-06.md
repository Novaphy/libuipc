# CoreX FPS Improvement Report

Date: 2026-05-06

## Summary

This round implemented NVIDIA-aligned CoreX candidates for matrix conversion,
SpMV, contact Hessian diagnostics, and selected-set equivalent-cost handling.

No candidate is promoted to the default path. The only clear short-run FPS win is
the opt-in MatrixConverter segmented reduce candidate, but it slightly increases
`wb150` PCG max, so it does not pass the defaulting rule. The default path remains
unchanged.

## Added Opt-In Controls

- Default change: CoreX now skips the device-wide sync after triangle AABB build
  in `StacklessBVHSimplexTrajectoryFilter`. This follows the NVIDIA path's stream
  ordering model while keeping edge/point AABB syncs in place.
- `UIPC_COREX_FILTER_AABB_ASYNC_MASK=0`: rollback switch for the new default.
- `UIPC_COREX_FILTER_AABB_ASYNC_MASK=N`: controls AABB sync removal by bit:
  `0x1` codim point, `0x2` surf point, `0x4` edge, `0x8` triangle.
- `UIPC_COREX_FILTER_AABB_ASYNC=1`: enables all AABB sync removals for diagnosis
  only; this is not defaulted because it caused a high `wb400` PCG max outlier.
- `UIPC_COREX_MATCONV_NVIDIA_SEG_REDUCE=1`: enables a CoreX-safe segmented reduce
  path that keeps the NVIDIA `FastSegmentalReduce` segment/count/offset semantics.
- `UIPC_COREX_MATCONV_SEG_THRESHOLD=N`: threshold for sending long segments to
  block reduction. The tested useful value is `16`.
- `UIPC_COREX_MATCONV_SEG_HIST=1`: logs segment length histograms as
  `[corex_matconv_seg_hist]`.
- `UIPC_COREX_SPMV_WARP_SEGMENTED=1`: enables a manual warp segmented SpMV row
  reduction that avoids CoreX's CUB `HeadSegmentedReduce` risk.
- `UIPC_COREX_CONTACT_SPD_DIAG=1`: now also logs `[corex_contact_input]` records
  with per-type `kt2`, `d_hat`, and `thickness` input distributions.
- `UIPC_COREX_FILTER_VIEW_SLICE=1`: keeps selected-set output buffers at their
  reserved max size and passes exact selected subviews, avoiding shrink/expand
  churn without changing selected PP/PE/PT/EE sets.

## Validation Baseline

All runs used GPU1 via `CUDA_VISIBLE_DEVICES=1`, with `--gpu 0` inside
`corex_demo`.

- Build: `cmake --build build_corex --target corex_demo --config Release -j8`
  passed.
- Default diagnostic baseline:
  - `wb80`: `17s`, FPS `4.706`, PCG sum `10899`, PCG max `72`.
  - `wb150`: `63s`, FPS `2.381`, PCG sum `53387`, PCG max `173`.

## Candidate Results

Triangle AABB build sync removal:

- Correctness gates passed: `simple90`, `simple300`, `stack120`.
- Current same-source default baseline:
  - `wb400`: `196s`, FPS `2.041`, PCG sum `168525`, PCG max `178`,
    contact detail `55776.290ms`.
- Defaulted triangle-only AABB async:
  - `wb150`: repeat run `53s`, FPS `2.830`, PCG sum `46491`, PCG max `179`.
  - `wb400`: `186s`, FPS `2.151`, PCG sum `172834`, PCG max `176`,
    contact detail `53302.786ms`.
  - Failure scan found no `NaN`, exception, assert, abort, or max-iteration
    markers.
  - Decision: default. It gives a measured `wb400` wall-time improvement while
    keeping PCG max non-regressive in the validating long run.

AABB async variants that were not defaulted:

- Full AABB async (`UIPC_COREX_FILTER_AABB_ASYNC=1`):
  - `wb400`: `162s`, FPS `2.469`, PCG sum `155158`, PCG max `291`.
  - Decision: reject for default because of the large PCG max outlier.
- Edge-only mask `0x4`:
  - `wb400`: `199s`, FPS `2.010`, PCG sum `186821`, PCG max `174`.
  - Decision: reject for wall-time regression.
- Triangle-only mask `0x8` was the only AABB async subset promoted to default.

MatrixConverter NVIDIA-aligned segmented reduce:

- Correctness gates passed: `simple90`, `simple300`, `stack120`.
- `threshold=16`:
  - `wb80`: `17s`, FPS `4.706`, PCG sum `10867`, PCG max `72`.
  - `wb150`: `52s`, FPS `2.885`, PCG sum `48277`, PCG max `175`.
  - Decision: keep opt-in. This is a visible `wb150` FPS gain, but PCG max rises
    from `173` to `175`, so it is not defaulted.
- `threshold=32`:
  - `wb150`: `65s`, FPS `2.308`, PCG sum `55127`, PCG max `172`.
  - Decision: rejected for performance.
- `threshold=64`:
  - `wb150`: `66s`, FPS `2.273`, PCG sum `56175`, PCG max `185`.
  - Decision: rejected.

SpMV manual warp segmented row reduction:

- Correctness gates passed: `simple90`, `simple300`, `stack120`.
- `wb80`: `17s`, FPS `4.706`, PCG sum `10934`, PCG max `68`.
- `wb150`: `63s`, FPS `2.381`, PCG sum `56385`, PCG max `176`.
- Decision: keep opt-in only. It does not improve wall time and regresses PCG on
  `wb150`.

PE/EE Hessian/SPD NVIDIA-diff diagnostics:

- Build passed.
- `simple90` with `UIPC_COREX_CONTACT_SPD_DIAG=1` passed.
- The run emitted `1028` `[corex_contact_input]` records.
- Decision: diagnostic only. No contact stiffness, active-set, barrier, or PCG
  convergence semantics were changed.

Selected-set equivalent-cost view slicing:

- Correctness gates passed: `simple90`, `simple300`, `stack120`.
- `wb80`: `19s`, FPS `4.211`, PCG sum `10976`, PCG max `68`.
- `wb150`: `64s`, FPS `2.344`, PCG sum `57175`, PCG max `169`.
- Contact filter detail on `wb150` decreased from `7681.849ms` to `7374.495ms`,
  but total wall time did not improve.
- Decision: keep opt-in only.

## Defaulting Decision

One new default is enabled: triangle AABB build sync removal in the CoreX
stackless BVH simplex trajectory filter.

The MatrixConverter candidate remains opt-in. It provides visible short-run FPS
improvement, but its long-run/defaulting behavior is weaker than the AABB sync
cleanup.

## Next Recommendation

The next high-value direction is to continue removing synchronization boundaries
that are not present in the NVIDIA path, but only one dependency edge at a time.
The full AABB async experiment shows there is more wall-time headroom, but the
large PCG max outlier means the remaining sync removals need finer dependency
isolation before defaulting.
