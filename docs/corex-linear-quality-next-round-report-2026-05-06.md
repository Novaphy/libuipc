# CoreX Linear Quality Next Round Report 2026-05-06

## Summary

This round investigated the remaining high-PCG frames on the stable default path:

- Default AABB sync (`UIPC_COREX_FILTER_AABB_ASYNC_MASK=0`)
- Stackless BVH query reserve ratio default `2.0`

No new optimization was promoted to default in this round. The current stable default
remains the BVH reserve `2.0` path from the previous report.

## Diagnostics

Stable default diagnostic `wb400`:

- Wall: `170s`
- PCG sum: `159855`
- PCG max: `178`
- PCG solve count: `2508`

Top high-PCG rows remain concentrated around frames `145-149` and `151`, mostly at
newton `1`. The leader is PE contact:

- Frame `146`, newton `1`: PCG `178`, selected PE `9990`,
  PE correction/diag avg `0.1998`, max `6.45`
- Frame `148`, newton `1`: PCG `176`, selected PE `10207`,
  PE correction/diag avg `0.1786`, max `4.79`
- Frame `147`, newton `1`: PCG `173`, selected PE `10127`,
  PE correction/diag avg `0.1918`, max `4.62`
- Frame `145`, newton `1`: PCG `172`, selected PE `9794`,
  PE correction/diag avg `0.2694`, max `6.10`

Global weighted contact correction ratios in this diagnostic run:

- PE: `0.1203`
- PP: `0.0977`
- EE: `0.0477`
- PT: `0.0115`

This confirms that the remaining high-PCG issue is primarily a PE-heavy linear
system quality problem rather than an AABB/filter buffer problem.

## Candidate Results

### PE-only SPD conditioning

Tested existing opt-in contact SPD conditioning only for PE, leaving PT/EE/PP
unchanged.

`wb150`:

- Default: `63s`, PCG sum `55385`, PCG max `170`
- Trigger `0.2`, scale `1e-5`: `61s`, PCG sum `55188`, PCG max `176`
- Trigger `0.2`, scale `3e-6`: `49s`, PCG sum `46507`, PCG max `182`
- Trigger `0.5`, scale `1e-5`: `51s`, PCG sum `46685`, PCG max `176`

The best-looking conservative short-run candidate, trigger `0.5` and scale `1e-5`,
failed the long run:

- Default `wb400`: `169s`, PCG sum `166117`, PCG max `181`
- PE conditioning `wb400`: `171s`, PCG sum `170258`, PCG max `183`

Decision: do not default PE SPD conditioning. It improves short-run totals but hurts
long-run wall time and PCG metrics.

### BVH query reserve ratio sweep

Tested whether reserve ratio above the current default `2.0` improves the contact
filter hot path.

`wb150`:

- Reserve `2.0`: `63s`, PCG sum `55385`, PCG max `170`
- Reserve `3.0`: `51s`, PCG sum `47459`, PCG max `176`
- Reserve `4.0`: `63s`, PCG sum `56422`, PCG max `174`

`wb400`:

- Reserve `2.0`: `169s`, PCG sum `166117`, PCG max `181`
- Reserve `3.0`: `178s`, PCG sum `167098`, PCG max `177`

Decision: keep default reserve ratio at `2.0`.

## Next Direction

The next meaningful direction is not scalar PE diagonal boosting. The high-PCG rows
need a more structural comparison of the PE-heavy matrix contribution:

- Compare PE selected-set identities and Hessian contribution distribution in
  frames `145-149` against NVIDIA if available.
- Add a diagnostic that maps high PE correction/diag contacts to vertex/row IDs and
  matrix row imbalance contributors.
- Investigate whether PE Hessian assembly order or duplicate/near-duplicate PE
  contacts inflate specific rows before SPD projection.

Artifacts are in `/tmp/corex_linear_quality_20260506/`.
