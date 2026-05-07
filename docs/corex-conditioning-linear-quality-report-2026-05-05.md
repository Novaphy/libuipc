# CoreX Conditioning And Linear Quality Report

Date: 2026-05-05

## Summary

This round shifted from local kernel replacement to PCG conditioning diagnostics.
The implementation added opt-in diagnostics for selected-set distribution and matrix
quality, enhanced the SPD/PCG aggregation script, and added an opt-in contact SPD
conditioning candidate.

No new optimization was promoted to default. The SPD conditioning candidates can
reduce short-run wall time and PCG total, but they increase PCG max or correction
outliers, so they do not satisfy the long-run stability policy.

## Added Controls

- `UIPC_COREX_SELECTED_SET_DIAG=1`: logs candidate/temp/selected PP, PE, PT, EE
  counts per frame/newton.
- `UIPC_COREX_MATRIX_QUALITY_DIAG=1`: logs BCOO matrix block count, diagonal and
  off-diagonal absolute sums, near-zero diagonal blocks, and row imbalance.
- `UIPC_COREX_CONTACT_SPD_CONDITION=1`: enables conservative post-SPD diagonal
  conditioning for PE/EE/PP by default. PT remains disabled unless
  `UIPC_COREX_CONTACT_SPD_CONDITION_PT=1`.
- `UIPC_COREX_CONTACT_SPD_RATIO_TRIGGER`: correction ratio threshold, default
  `0.25`.
- `UIPC_COREX_CONTACT_SPD_DIAG_SCALE`: default PE/EE/PP diagonal scale, default
  `1e-4`.
- Per-type scale overrides:
  `UIPC_COREX_CONTACT_SPD_PE_DIAG_SCALE`,
  `UIPC_COREX_CONTACT_SPD_EE_DIAG_SCALE`,
  `UIPC_COREX_CONTACT_SPD_PP_DIAG_SCALE`,
  `UIPC_COREX_CONTACT_SPD_PT_DIAG_SCALE`.

## Validation

Build passed for `corex_demo`.

Correctness gates passed without NaN, exception, or max-iteration failure markers:

- `simple90`, `simple300`, `stack120` with full SPD conditioning diagnostics.
- `simple90` with PE-only low-scale conditioning.
- `simple90` smoke test for aligned selected-set and matrix-quality diagnostics.

Short A/B:

- Default diagnostic baseline:
  - `wb80`: `18s`, PCG sum `10679`, PCG max `70`
  - `wb150`: `65s`, PCG sum `54171`, PCG max `169`
- Full PE/EE/PP conditioning, scale `1e-4`:
  - `wb80`: `17s`, PCG sum `10275`, PCG max `68`
  - `wb150`: `53s`, PCG sum `46296`, PCG max `178`
  - Decision: reject for default; PCG max regressed and EE correction-ratio
    outliers became very large.
- PE-only conditioning, PE scale `1e-5`:
  - `wb80`: `17s`, PCG sum `10767`, PCG max `67`
  - `wb150`: `65s`, PCG sum `57420`, PCG max `179`
  - Decision: reject; PCG total and max regressed on `wb150`.
- Full PE/EE/PP conditioning, scale `1e-5`:
  - `wb80`: `17s`, PCG sum `10738`, PCG max `69`
  - `wb150`: `54s`, PCG sum `47146`, PCG max `178`
  - Decision: reject for default; short-run wall improves, but PCG max regresses.

No conditioning candidate entered `wb400` candidate validation because none satisfied
the `wb150` PCG max gate.

## wb400 Diagnostic Findings

Aligned default diagnostic run:

- Wall: `175s`
- PCG sum: `164626`
- PCG max: `175`
- Diagnostic rows: `2571`

Weighted correction-diag ratios:

- PT: `0.011299`
- EE: `0.047452`
- PE: `0.101112`
- PP: `0.100344`

Max correction-diag ratios:

- PT: `0.727962`
- EE: `9.862394`
- PE: `56.630623`
- PP: `7.075102`

Top high-PCG rows cluster around frames `145-149`, newton `1`. A representative
row is frame `149`, newton `1`, PCG iter `175`, selected PE `8899`, selected PT
`493`, matrix row imbalance `34.56`, PE ratio max `4.48`, EE ratio max `1.25`,
and dotnorm around `56.8%`.

This suggests the long-run bottleneck is not a single bad kernel. The critical
frames combine:

- High PE selected count.
- Significant PE/EE SPD projection corrections.
- Very imbalanced matrix rows.
- PCG dominated by dot/norm and scalar-boundary cost.

## Defaulting Decision

No new default changes.

The new SPD conditioning path is useful as an opt-in diagnostic/candidate, but the
tested parameters do not meet the stability rule. The current safest default remains
the previous baseline.

## Next Direction

The next optimization should target the frame `145-149` pattern directly:

- Investigate why PE selected count is very high in those frames.
- Compare CoreX and NVIDIA selected-set composition for the same frames if a NVIDIA
  diagnostic run is available.
- Explore PE/EE Hessian conditioning that reduces extreme projection without adding
  broad diagonal stiffness, because diagonal boosts improved total PCG but worsened
  long-tail max iteration.

Artifacts are in `/tmp/corex_conditioning_20260505/`.
