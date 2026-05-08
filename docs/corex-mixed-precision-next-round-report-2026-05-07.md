# CoreX Mixed-Precision Next Round Report

Date: 2026-05-07

## Summary

This round tested the hypothesis from the NVIDIA/CoreX-compat comparison:
because native NVIDIA float is unstable while CoreX float now completes with
`~88k` PCG iterations, local mixed precision might reduce the remaining gap to
the native NVIDIA high-precision baseline (`38k` PCG iterations).

Two opt-in candidates were implemented:

- `UIPC_COREX_ABD_PRECOND_LDLT_DOUBLE=1`: use `double` as the internal work type
  for the 12x12 ABD block-inverse LDLT factorization and explicit inverse, then
  cast the final `diag_inv` back to `Float`.
- `UIPC_COREX_PCG_REDUCE_DOUBLE=1`: use `double` accumulators for CoreX PCG
  `dot`, `dot/norm`, `pAp`, `rz`, `norm_b`, and `norm_r` scalar reductions.
  Host-side `alpha`, `beta`, and residual tests are computed in `double`, then
  update kernels receive `Float` casts.

Both candidates are **rejected**. The mixed-precision ladder fails at the smoke
stage, so no `wb150`/`wb400` gate was run.

## Build

Build command:

```bash
cmake --build build_corex --target corex_demo --config Release -j8
```

Result: passed. Only existing CoreX/muda warnings were emitted.

## Smoke Results

All smoke runs used `CUDA_VISIBLE_DEVICES=1`, `corex_demo`, `--backend cuda`,
`--gpu 0`, and `UIPC_COREX_PCG_PINNED_SCALAR=0` to keep the result aligned with
the 146s block-inverse baseline.

| Candidate | `simple90` | `wb80` | Decision |
|---|---:|---:|---|
| `LDLT_DOUBLE=1` | abort at frame 1 (`r^T z = nan`) | abort before useful PCG stats | reject |
| `PCG_REDUCE_DOUBLE=1` | pass, `3s` | abort at frame 1 (`r^T z = inf`) | reject |
| both enabled | pass but `simple90` hits repeated `max_iter=48`; `12s` | completes `10s`, PCG sum `13992`, max `13800` | reject |
| default rollback (`LDLT_DOUBLE=0`, `PCG_REDUCE_DOUBLE=0`) | not rerun | pass, `16s`, PCG sum `4901`, max `48` | unchanged |

Failure snippets:

```text
LDLT_DOUBLE simple90:
Frame 1, Newton 0, PCG Iter 18: r^T*z = nan, norm(r) = nan, norm(z) = nan.
Hint: PCG iteration diverged.
```

```text
PCG_REDUCE_DOUBLE wb80:
Frame 1, Newton 0, PCG Iter 1: r^T*z = inf, norm(r) = 74.583725,
norm(z) = 0.023457639. Hint: PCG iteration diverged.
```

```text
Mixed wb80:
PCG_calls=158 PCG_iter_sum=13992 PCG_iter_max=13800
LinearPCG: reached max_iter = 13800
```

## Interpretation

The result suggests that simply introducing `double` into CoreX device kernels is
not a safe route on the current ILUVATAR/CoreX toolchain. The failure mode is not
a slow performance regression; it is immediate scalar/solver invalidity:

- `LDLT_DOUBLE` corrupts the preconditioner enough that even `simple90` diverges
  at the first PCG solve.
- `PCG_REDUCE_DOUBLE` passes `simple90` but produces `inf` in the first `wb80`
  solve, which means the double reduction path is not numerically trustworthy
  under this compiler/runtime combination.
- Combining both does not cancel the issue. It avoids the early abort in `wb80`
  but creates a catastrophic PCG max-iter hit (`13800`), so it is unusable.

This is consistent with the earlier observation that native NVIDIA float is
itself unstable, but it changes the expected remediation: CoreX cannot rely on a
naive "use device double locally" patch to recover native high-precision PCG
quality.

## Code State

The code keeps both candidates as explicit opt-ins only:

- `UIPC_COREX_ABD_PRECOND_LDLT_DOUBLE=1`
- `UIPC_COREX_PCG_REDUCE_DOUBLE=1`

Default behavior is unchanged from the block-inverse baseline. A rollback smoke
run confirms the default path still completes `wb80` cleanly:

```text
wb80 default after mixed patch:
exit=0 wall=16s PCG_calls=315 PCG_iter_sum=4901 PCG_iter_max=48
```

These opt-ins should not be defaulted or used for production runs.

## Next Step Recommendation

Do not continue the "device double everywhere" mixed-precision ladder on CoreX.
The next useful direction is a **numerical equivalence audit** that does not rely
on CoreX double execution:

1. Dump representative `wb80/wb150` high-PCG linear systems from CoreX float:
   `diag_hessian`, `diag_inv`, selected `r/z/p/Ap` scalar traces, and matrix
   triplet checksums.
2. Analyze those dumps offline on CPU double or on NVIDIA, where double is
   reliable.
3. Use the offline comparison to identify whether the remaining `88k` PCG count
   comes from:
   - preconditioner quality (`H * H^{-1}` residual),
   - matrix assembly/reduction order,
   - contact Hessian/SPD projection,
   - or PCG stopping/scalar trajectory.

If a specific float operation is identified as the source, implement a targeted
float-safe correction such as scaling, compensated summation, pivot threshold
tuning, or reordered accumulation. Avoid adding more broad CoreX `double` device
kernels until the toolchain behavior is better characterized.
