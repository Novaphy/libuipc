# CoreX DyTopo Parallel Optimization Report 2026-05-02

## Summary

This round continued past the PE Hessian conditioning experiments and found a clear win in ABD DyTopo assembly.

The effective optimization is to replace the CoreX serial DyTopo gradient/hessian assembly kernels with parallel kernels:

- `kernel_abd_dytopo_gradients_parallel`
- `kernel_abd_dytopo_hessians_parallel`

The parallel path is now default. A rollback switch is available:

- `UIPC_COREX_ABD_DYTOPO_SERIAL=1`

`UIPC_COREX_ABD_DYTOPO_PARALLEL=0` also disables the new path.

## Failed Candidates Before The Win

PE diagonal regularization did not produce a reliable improvement:

- `UIPC_COREX_PE_DIAG_REG=0.001`
- `wb150`: wall 102s -> 104s, PCG sum 56073 -> 58444, PCG max 174 -> 180.

PE stiffness scaling produced a small `wb80` signal but failed on `wb150`:

- `UIPC_COREX_PE_KAPPA_SCALE=0.5`
- `wb80`: PCG sum 10912 -> 10584, PCG max 70 -> 64, wall 29s -> 30s.
- `wb150`: wall 104s -> 136s, PCG sum 56843 -> 74257.

Both remain opt-in only and are not defaulted.

## Why DyTopo Was The Right Target

The previous phase profiling showed `abd_assemble.dytopo_effect` as the largest assembly component. The CoreX path still used one-thread serial kernels for DyTopo projection into ABD gradients and Hessians. That made the phase scale poorly in `wrecking_ball` once contact topology grew.

The new parallel kernels preserve the same output layout:

- Gradient contributions use `atomicAdd` into ABD body gradients.
- Same-body Hessian contributions use `atomicAdd` into `diag_hessian`.
- Triplet output remains one source Hessian entry to 16 destination 3x3 blocks.
- Same-body terms still zero lower-triangle blocks before triplet write, matching the serial semantics.

## Validation

Build:

- `cmake --build build_corex --target corex_demo --config Release -j8`
- Result: passed.

Correctness gates on `gpu1` with the parallel DyTopo path:

- `simple90`: passed, 4s.
- `simple300`: passed, 11s.
- `stack120`: passed, 6s.
- Failure scan found no NaN, exception, assert, abort, or reached-max-iter marker.

Default-on verification with no DyTopo env var:

- `simple90`: passed, 4s.
- `simple300`: passed, 11s.
- `stack120`: passed, 6s.
- `wb80`: passed, 17s, PCG sum 10662, PCG max 68.

## A/B Results

With `UIPC_COREX_PHASE_PROFILE=1`:

- `wb80 default serial`: 30s, PCG sum 10882, PCG max 69.
- `wb80 parallel`: 18s, PCG sum 10562, PCG max 72.
- `wb150 default serial`: 118s, PCG sum 68724, PCG max 177.
- `wb150 parallel`: 65s, PCG sum 55892, PCG max 171.

Phase impact:

- `wb150 serial`: `abd_assemble.dytopo_effect` was 43643.84 ms and dominated `linear.assemble_linear_system`.
- `wb150 parallel`: DyTopo assembly no longer appeared among the top phase buckets; the leading costs shifted back to PCG and contact detection.

Long gate:

- `wb400 parallel`: passed, 180s, PCG sum 173559, PCG max 198, solves 2725.
- Failure scan: no NaN, exception, assert, abort, or reached-max-iter marker.

## Defaulting Decision

Default the parallel DyTopo path.

This is the first optimization in this round with a clear wall-time win across `wb80`, `wb150`, and `wb400`, without correctness-gate failures or PCG regression. The remaining bottlenecks after this change are PCG dot/norm reductions, contact detection/filtering, and matrix conversion.
