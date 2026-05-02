# CoreX Structured Preconditioner Overhead Report

Date: 2026-05-02

## Summary

This round implemented the low-risk overhead reduction planned for the opt-in CoreX ABD structured block preconditioner.

Default behavior is unchanged. The structured block path remains opt-in:

```bash
UIPC_COREX_ABD_PRECOND_STRUCT_BLOCK=1
```

The implementation now removes one extraction kernel launch from the structured preconditioner setup and avoids the default host-side block-mask copy used only for acceptance statistics. Validation passed the simple/stack correctness gates, but `wrecking_ball150` did not meet the wall-time and PCG-iteration gate for `wb400` validation. The path should stay opt-in.

## Implemented Changes

File:

- `src/backends/cuda/affine_body/abd_diag_preconditioner.cu`

Changes:

- Replaced separate Jacobi reciprocal extraction and structured 3x3 block extraction kernels with a single `kernel_abd_precond_extract` launch.
- Kept the existing Jacobi default path unchanged when `UIPC_COREX_ABD_PRECOND_STRUCT_BLOCK` is off.
- Added explicit statistics switch:

```bash
UIPC_COREX_ABD_PRECOND_STRUCT_BLOCK_STATS=1
```

`[corex_precond_struct_block]` now requires either this switch or `UIPC_COREX_TRACE_LINEAR_SYSTEM=1`. This avoids paying the host `cudaMemcpy` cost during normal structured-preconditioner A/B runs.

## Validation

Build:

- `cmake --build /root/libuipc-v11-restored/libuipc-v11-extracted/build_corex --target corex_demo --config Release -j8`: PASS

Correctness gates on GPU 1:

| Case | Result | Wall Time | Failure Markers |
| --- | --- | ---: | --- |
| `simple90` | PASS | 4s | none |
| `simple300` | PASS | 10s | none |
| `stack120` | PASS | 6s | none |

The logs were scanned for `NaN`, `nan`, `exception`, `reached max_iter`, `Assertion false`, and abort markers. No matches were found.

Artifacts:

- `/tmp/corex_next_opt_20260502/`
- `/tmp/corex_next_opt_20260502/wb_ab_spd_pcg.json`
- `/tmp/corex_next_opt_20260502/wb_ab_spd_pcg.csv`

## Wrecking Ball A/B

All runs used:

```bash
CUDA_VISIBLE_DEVICES=1
UIPC_COREX_PCG_COST_DIAG=1
UIPC_COREX_CONTACT_SPD_DIAG=1
```

Structured runs also used:

```bash
UIPC_COREX_ABD_PRECOND_STRUCT_BLOCK=1
UIPC_COREX_ABD_PRECOND_STRUCT_BLOCK_STATS=1
```

| Case | Config | Wall Time | PCG Iter Sum | Max PCG Iter | Newton Sum | Precond Bucket | DotNorm Bucket |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `wb80` | default | 30s | 10796 | 69 | 239 | 14.459% | 57.287% |
| `wb80` | struct block | 30s | 10292 | 79 | 244 | 17.864% | 55.118% |
| `wb150` | default | 84s | 46647 | 179 | 497 | 14.415% | 56.881% |
| `wb150` | struct block | 101s | 54144 | 154 | 511 | 17.818% | 54.665% |

Contact SPD correction/diagonal weighted averages:

| Case | Config | PE | PP | EE |
| --- | --- | ---: | ---: | ---: |
| `wb80` | default | 0.221317 | 0.144121 | 0.070946 |
| `wb80` | struct block | 0.251230 | 0.151563 | 0.071304 |
| `wb150` | default | 0.118944 | 0.143783 | 0.065667 |
| `wb150` | struct block | 0.086184 | 0.093764 | 0.066001 |

## Decision

Do not default `UIPC_COREX_ABD_PRECOND_STRUCT_BLOCK=1`.

Although the implementation removes setup overhead and `wb80` shows a small PCG iteration reduction, `wb150` regresses both wall time and total PCG iterations. This fails the planned gate for running `wb400`, so no `wb400` default validation was run in this round.

The structured block path remains useful as a diagnostic/experimental preconditioner because it changes iteration shape and lowers max PCG iteration on `wb150`, but it is not yet a production improvement.

## Next Direction

- Keep the fused extraction and opt-in stats switch; they reduce avoidable diagnostic overhead without changing default behavior.
- Continue conditioning analysis with `UIPC_COREX_CONTACT_SPD_DIAG=1`, especially PE/PP/EE rows where correction/diagonal ratios shift materially between A/B runs.
- Do not add more PCG reduction kernels before identifying why structured preconditioning reduces some peaks but worsens `wb150` total iterations and wall time.
- If preconditioner work continues, prefer cheaper apply-time designs or adaptive activation over always applying four 3x3 blocks per ABD body.
