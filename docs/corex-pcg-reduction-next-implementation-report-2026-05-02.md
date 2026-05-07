# CoreX PCG Reduction Next Implementation Report

Date: 2026-05-02

## Summary

This round implemented the next low-risk diagnostics and PCG reduction experiments from `corex-pcg-reduction-next`.

Default behavior is unchanged. The new PCG two-level reduction is opt-in only:

```bash
UIPC_COREX_PCG_REDUCE2=1
```

Initial validation shows the contact SPD diagnostic path is now usable on `wrecking_ball80`, while the two-level PCG reduction does not yet improve `wrecking_ball150` and should not be defaulted.

## Implemented Changes

### Contact SPD diagnostics

File:

- `src/backends/cuda/contact_system/contact_models/ipc_simplex_normal_contact.cu`

Fixes:

- Added `[corex_spd_contact_begin]` logging to the CoreX explicit `__global__` assembly branch.
- Always emits PT/EE/PE/PP `[corex_spd_contact]` rows when diagnostics are enabled, even when a type has zero blocks.
- Replaced `unsigned long long` diagnostic counters with `unsigned int` counters. On CoreX, the 64-bit counter atomics did not update reliably even though Float atomics did, producing impossible rows such as `count=0` with nonzero max correction.

### PCG two-level reduction experiment

File:

- `src/backends/cuda/linear_system/linear_pcg_corex.cu.inc`
- `src/backends/cuda/linear_system/linear_pcg_corex.h`

Added opt-in two-level reduction kernels:

- `dot(r,z)` + `norm(r)` writes block partials, then a single small finalize kernel reduces partials.
- `pAp = dot(p, Ap)` uses the same two-level pattern when `UIPC_COREX_PCG_REDUCE2=1`.
- Existing atomic fused `rz/norm` remains the default baseline.

Diagnostics now include `reduce2=0/1` in `[corex_pcg_cost]` and `[corex_pcg_diag]` rows.

## Validation

Build:

- `cmake --build build_corex --config Release -j 8`: PASS

Runtime smoke tests on GPU 1:

| Case | Config | Result | Wall Time | PCG Calls | PCG Iter Sum | Max PCG Iter | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `simple90` | `UIPC_COREX_PCG_REDUCE2=1 UIPC_COREX_PCG_COST_DIAG=1` | PASS | 4.4s | 258 | 1970 | 18 | no failure markers |
| `wrecking_ball80` | `UIPC_COREX_CONTACT_SPD_DIAG=1` | PASS | 30.6s | 322 | 10684 | 67 | nonzero PT/EE/PE/PP SPD rows emitted |
| `wrecking_ball150` | `UIPC_COREX_PCG_REDUCE2=1 UIPC_COREX_PCG_COST_DIAG=1` | PASS | 102.3s | 849 | 55476 | 178 | no failure markers |

Aggregated `wrecking_ball150` cost split with `UIPC_COREX_PCG_REDUCE2=1`:

| spmv | spmv_sync | preconditioner | dotnorm |
| ---: | ---: | ---: | ---: |
| 11.166% | 16.706% | 13.863% | 58.264% |

For comparison, the previous baseline report measured `dotnorm` at about `56.771%` on `wrecking_ball150`. The first reduce2 result is therefore not a win.

## Recommendation

Keep `UIPC_COREX_PCG_REDUCE2` opt-in and do not default it. It removes global scalar atomics but adds another kernel launch per reduction, and on `wrecking_ball150` that tradeoff is not favorable.

The useful outcome of this round is the repaired contact SPD diagnostic path. The `wrecking_ball80` diagnostic now shows very high projection rates, including large PE/PP correction magnitudes once those contact types become active. The next optimization step should use these SPD records to investigate why CoreX has higher PCG iteration counts before adding more reduction kernels.
