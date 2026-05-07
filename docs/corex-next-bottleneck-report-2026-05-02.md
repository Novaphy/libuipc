# CoreX Next Bottleneck Report

Date: 2026-05-02

## Summary

This round followed `corex-next-bottleneck` and shifted the focus from PCG reduction micro-optimization to the higher-level conditioning problem behind CoreX's high PCG iteration count.

Default behavior is unchanged. No new optimization is defaulted.

Main outcomes:

- Added a reusable SPD/PCG log aggregator: `tools/corex_spd_pcg_aggregate.py`.
- Extended `UIPC_COREX_CONTACT_SPD_DIAG=1` with Hessian diagonal scale and correction/diagonal ratio statistics.
- Evaluated `UIPC_COREX_ABD_PRECOND_STRUCT_BLOCK=1` through `simple90/simple300/stack120/wb80/wb150/wb400`.
- Concluded that structured block preconditioning reduces PCG iterations, but its current preconditioner overhead prevents defaulting.

## Implemented Diagnostics

### SPD/PCG aggregation

File:

- `tools/corex_spd_pcg_aggregate.py`

The script parses:

- `[corex_spd_contact_begin]`
- `[corex_spd_contact]`
- `[corex_pcg_cost]`
- `LinearPCG: frame=... newton_iter=... -> iters=...`
- `[corex_precond_struct_block]`

Outputs:

- `artifacts/corex_next_bottleneck_spd_pcg.csv`
- `artifacts/corex_next_bottleneck_spd_pcg.json`

### Contact SPD scale ratios

File:

- `src/backends/cuda/contact_system/contact_models/ipc_simplex_normal_contact.cu`

The opt-in SPD diagnostic now also reports:

- `diag_abs_avg`
- `correction_diag_ratio_avg`
- `correction_diag_ratio_max`

These fields are diagnostic-only and are emitted only when `UIPC_COREX_CONTACT_SPD_DIAG=1`.

## SPD/PCG Findings

Extended `wb80` and `wb150` diagnostic runs passed without failure markers.

Combined `wb80 + wb150` aggregation:

| Metric | Value |
| --- | ---: |
| PCG calls | 1177 |
| PCG iter sum | 67684 |
| Max PCG iter | 185 |
| SPD nonzero rows | 1143 |
| PCG `dotnorm` bucket | 57.143% |
| PCG `spmv_sync` bucket | 17.060% |

SPD correction/diagonal ratios show that PE/PP/EE remain the strongest conditioning suspects:

| Type | Weighted correction/diag avg | Max correction/diag | Weighted correction avg | Max correction |
| --- | ---: | ---: | ---: | ---: |
| PT | 0.010456 | 0.651048 | 169.982 | 17798.574 |
| EE | 0.067234 | 9.309665 | 29.019 | 391946.875 |
| PE | 0.109795 | 24.975420 | 19.557 | 125626.836 |
| PP | 0.142291 | 5.076192 | 286.500 | 168359.062 |

Interpretation: high PCG iterations are not explained by reduction overhead alone. Contact Hessian SPD projection is widespread, and PE/PP/EE have much larger relative correction signals than PT. This supports continuing with conditioning-oriented diagnostics rather than more reduction kernels.

## Structured Preconditioner A/B

Config:

```bash
UIPC_COREX_ABD_PRECOND_STRUCT_BLOCK=1
```

Validation on GPU 1:

| Case | Result | Wall Time | PCG Calls | PCG Iter Sum | Max PCG Iter | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `simple90` | PASS | 4.1s | 247 | 1597 | 19 | no failure markers |
| `simple300` | PASS | 10.1s | 798 | 5068 | 19 | no failure markers |
| `stack120` | PASS | 5.6s | 255 | 2794 | 44 | no failure markers |
| `wb80` | PASS | 30.1s | 320 | 10088 | 71 | iter sum improved, max slightly higher |
| `wb150` | PASS | 85.3s | 760 | 44618 | 150 | iter sum improved |
| `wb400` | PASS | 325.9s | 2666 | 158981 | 148 | iter sum improved, wall time not improved |

`wb150` cost split with structured preconditioner:

| spmv | spmv_sync | preconditioner | dotnorm |
| ---: | ---: | ---: | ---: |
| 11.057% | 16.513% | 17.805% | 54.624% |

`wb400` cost split with structured preconditioner:

| spmv | spmv_sync | preconditioner | dotnorm |
| ---: | ---: | ---: | ---: |
| 11.228% | 16.900% | 17.467% | 54.404% |

Compared with the current default reports, structured block preconditioning reduces total PCG iterations and max PCG iterations on `wb400`, but the preconditioner bucket grows enough that wall time regresses. It therefore does not satisfy the defaulting bar.

## Decision

Do not default `UIPC_COREX_ABD_PRECOND_STRUCT_BLOCK=1` in its current form.

Do not run `domino600` or `wb800` for default validation in this round, because `wb400` already failed the wall-time requirement.

Recommended next work:

- Keep structured block preconditioning as an opt-in diagnostic because it proves stronger preconditioning can reduce iterations.
- Optimize the structured preconditioner overhead before considering defaulting, especially extract/apply synchronization and block mask handling.
- Continue contact conditioning analysis using the new correction/diagonal ratios, focusing on PE/PP/EE cases with high ratio spikes.
- Avoid more PCG reduction changes until conditioning and preconditioner overhead are better understood.
