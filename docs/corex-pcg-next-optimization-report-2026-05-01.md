# CoreX PCG Next Optimization Report

## Summary

This round focused on the CoreX/NVIDIA PCG gap observed in `wrecking_ball400`.
The main production change is to make CoreX compute `rz_new = dot(r, z)` and
`norm(r)` in one device reduction kernel. This keeps the per-iteration
`norm(r)` convergence semantics, but removes one separate scalar reduction path
per PCG iteration.

Default CoreX results improved from the v15 baseline:

- `wrecking_ball400`: `319s` -> `296s`
- `wrecking_ball800`: `621s` -> `587s`

The improvement is smaller than the remaining NVIDIA gap because CoreX still
needs about 4x more PCG iterations than NVIDIA, but this round reduces part of
the per-iteration scalar reduction overhead.

## Implemented Changes

### PCG diagnostics

Added opt-in `UIPC_COREX_PCG_DIAG=1` logging in `LinearPCG`:

- frame/newton/dof/unique triplets
- local/no-preconditioner counts
- `norm(b)`, initial/final `norm(r)`
- `rz0`, final `rz`, tolerance
- min/max/last `pAp`
- accumulated `pcg.spmv`, `pcg.spmv_sync`, `pcg.preconditioner`, `pcg.dotnorm`

This is diagnostic-only and off by default.

### Device-side `rz/norm` reduction

Added a CoreX kernel that computes:

```text
dot(r, z)
norm2(r)
```

in a single pass. CoreX now uses this path by default. The old separated
`dot(r,z)` plus `norm(r)` path is retained through:

```bash
UIPC_COREX_PCG_SEPARATE_RZ_NORM=1
```

### ABD Jacobi preconditioner diagnostics and clamp experiment

Added:

```bash
UIPC_COREX_ABD_PRECOND_DIAG_STATS=1
UIPC_COREX_ABD_PRECOND_DIAG_CLAMP=min,max
```

Early `wrecking_ball` diagnostics showed ABD diagonal entries roughly in
`[1, 3.8e5]` with no zero/tiny diagonal entries in the sampled frames.

The tested clamp:

```bash
UIPC_COREX_ABD_PRECOND_DIAG_CLAMP=1,100000
```

passed `simple/stack`, but worsened `wrecking_ball50` PCG iterations
(`5661` -> `6095`), so it was not selected.

## Validation Results

All runs used GPU1.

| Case | Config | Result | Wall Time | PCG Calls | PCG Iter Sum | Max Newton Exits |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `simple90` | default | exit 0 | 4s | - | - | - |
| `simple300` | default | exit 0 | 7s | - | - | - |
| `stack120` | default | exit 0 | 5s | - | - | - |
| `wrecking_ball150` | default | exit 0 | 97s | 841 | 54639 | 2 |
| `wrecking_ball400` | default | exit 0 | 296s | 2576 | 161852 | 2 |
| `domino600` | default | exit 0 | 59s | 1825 | 82856 | 0 |
| `wrecking_ball800` | default | exit 0 | 587s | 5425 | 330439 | 1 |

Opt-in A/B runs:

| Case | Config | Result | Wall Time | PCG Calls | PCG Iter Sum | Note |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `wrecking_ball50` | diagnostics only | exit 0 | 16s | 194 | 5661 | `pcg.dotnorm` sum about `1103ms` |
| `wrecking_ball50` | fused `rz/norm` | exit 0 | 16s | 196 | 5721 | `pcg.dotnorm` sum about `903ms` |
| `wrecking_ball150` | fused `rz/norm` opt-in | exit 0 | 85s | 759 | 47953 | before defaulting |
| `wrecking_ball400` | fused `rz/norm` opt-in | exit 0 | 292s | 2560 | 163436 | before defaulting |
| `domino600` | fused `rz/norm` opt-in | exit 0 | 63s | 1858 | 83885 | before defaulting |
| `wrecking_ball800` | fused `rz/norm` opt-in | exit 0 | 609s | 5515 | 338923 | before defaulting |
| `wrecking_ball50` | ABD clamp `1,100000` | exit 0 | 16s | 192 | 6095 | iteration regression |

## Contact Scale Follow-up

Contact was not changed in this round. Existing `wb400` logs still show large
remaining candidate/output scale:

- v15 default `wb400`: sampled sums `PT=9006149`, `EE=58511635`, `PE=86555946`
- fused default `wb400`: sampled sums `PT=8011499`, `EE=45727511`, `PE=53340505`

The remaining contact work is still significant, especially EE/PE. This should
be treated as a later, correctness-sensitive optimization line, not mixed into
the PCG scalar reduction change.

## Conclusion

The selected change is stable enough to keep as the CoreX default:

- preserves per-iteration `norm(r)` convergence semantics
- improves `wb400` by about `23s` versus the v15 baseline
- improves `wb800` by about `34s` versus the v15 baseline
- passes `simple`, `stack`, `domino600`, and `wb800`

The remaining gap to NVIDIA is still dominated by PCG iteration count and
large contact/linear-system work. The next optimization round should focus on
why CoreX needs many more PCG iterations, rather than further relaxing
convergence checks.
