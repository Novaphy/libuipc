# CoreX Float-Only Numerics Report

Date: 2026-05-07

## Summary

After confirming that CoreX hardware/runtime should not rely on device `double`,
this round removed the previous mixed-precision runtime branches and tested two
float-only numerical candidates:

- `UIPC_COREX_ABD_PRECOND_SCALED_LDLT=1`: diagonal equilibration for the 12x12
  ABD block-inverse LDLT preconditioner.
- `UIPC_COREX_PCG_RESIDUAL_REPLACE_INTERVAL=N`: periodic PCG residual
  replacement, recomputing `r = b - A*x` every `N` iterations.

Neither candidate is promoted. Both are stable, but neither gives a meaningful
PCG iteration reduction beyond normal run-to-run noise.

## Removed Double Runtime Code

The rejected device-double runtime paths from the prior mixed-precision smoke
were deleted:

- Removed `UIPC_COREX_ABD_PRECOND_LDLT_DOUBLE`.
- Removed `UIPC_COREX_PCG_REDUCE_DOUBLE`.
- Removed `DeviceVar<double>` / `DeviceBuffer<double>` PCG scalar buffers.
- Removed double dot/dotnorm kernels.

Default CoreX remains pure `Float` for runtime kernels.

Validation after deletion:

```text
wb80 default after double deletion:
exit=0 wall=16s PCG_calls=316 PCG_iter_sum=4946 PCG_iter_max=48
```

This matches the existing block-inverse baseline range.

## Candidate 1: Scaled LDLT

Implementation:

For each 12x12 block, before LDLT factorization:

$$
A_s = S A S,\quad S_i = \frac{1}{\sqrt{\max(|A_{ii}|, \epsilon)}}
$$

After inverting the scaled block:

$$
A^{-1} = S A_s^{-1} S
$$

All operations remain `Float`; there is no device double.

Results:

| Run | Wall | PCG calls | PCG iter sum | PCG max | Status |
|---|---:|---:|---:|---:|---|
| `simple90` scaled LDLT | `4s` | - | - | - | pass |
| `wb80` default | `16s` | `316` | `4946` | `48` | pass |
| `wb80` scaled LDLT | `16s` | `311` | `4839` | `48` | pass |
| `wb150` baseline range | `49-50s` | - | `~27467` | `~124` | pass |
| `wb150` scaled LDLT | `50s` | `744` | `27568` | `125` | pass |

Conclusion: scaled LDLT is safe but not useful. It slightly improves `wb80`, but
`wb150` is effectively unchanged.

## Candidate 2: PCG Residual Replacement

Implementation:

When `UIPC_COREX_PCG_RESIDUAL_REPLACE_INTERVAL=N`, after the normal PCG update

```text
x = x + alpha * p
r = r - alpha * Ap
```

the solver periodically recomputes:

```text
Ap = A * x
r  = b - Ap
```

This is intended to reduce float residual drift without changing PCG tolerance
or using double scalar reductions.

Results:

| Run | Wall | PCG calls | PCG iter sum | PCG max | Status |
|---|---:|---:|---:|---:|---|
| `wb80` default | `16s` | `316` | `4946` | `48` | pass |
| `wb80` interval 20 | `16s` | `312` | `4848` | `48` | pass |
| `wb80` interval 50 | `16s` | `311` | `4842` | `48` | pass |
| `wb80` interval 100 | `16s` | `310` | `4828` | `48` | pass |
| `wb150` baseline range | `49-50s` | - | `~27467` | `~124` | pass |
| `wb150` interval 50 | `51s` | `749` | `29034` | `122` | pass, worse |
| `wb150` interval 100 | `49s` | `745` | `27410` | `124` | pass, flat |

Conclusion: residual replacement is stable but not helpful at the tested
intervals. The best short result (`interval=100`) is effectively identical to
the existing baseline, and `interval=50` regresses `wb150`.

## Decision

No default changes from this float-only numerics round.

Keep as opt-in diagnostic candidates only:

- `UIPC_COREX_ABD_PRECOND_SCALED_LDLT=1`
- `UIPC_COREX_PCG_RESIDUAL_REPLACE_INTERVAL=N`

Do not default either candidate.

## Next Recommendation

The remaining `88k` vs native NVIDIA high-precision `38k` PCG gap is unlikely to
be fixed by small local PCG/preconditioner tweaks. The next productive step is
to dump and audit numerical data offline:

1. Dump selected high-PCG frames (`wb150` around frames 140-149, then `wb400`
   if needed):
   - ABD `diag_hessian`
   - ABD `diag_inv`
   - BCOO/BSR matrix checksums
   - PCG scalar trace (`rz`, `pAp`, `norm_r`, `r_tol`, iter count)
2. Analyze dumps on CPU/NVIDIA using reliable high precision.
3. Identify whether the remaining gap comes from preconditioner quality,
   matrix assembly/reduction order, contact SPD projection, or stopping/scalar
   trajectory.

Runtime CoreX kernels should stay float-only unless a very narrow operation is
proven safe and beneficial on the target hardware/toolchain.
