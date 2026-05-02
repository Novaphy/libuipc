# CoreX Contact Conditioning Next Report

Date: 2026-05-02

## Summary

This round implemented the diagnostic-first roadmap for the next CoreX optimization pass. Default simulation behavior is unchanged.

Main outcomes:

- Extended the SPD/PCG aggregator with contact outlier views by contact type and frame.
- Added opt-in PCG scalar round-trip audit logging.
- Split CoreX linear assembly profiling into clearer subphases.
- Extended profile summary parsing for contact candidate/selected statistics.
- Re-evaluated grouped-row SpMV on the current baseline.

No new optimization path is defaulted. `UIPC_COREX_SPMV_GROUPED_ROW=1` is not recommended for defaulting based on the current `wb150` result.

## Implemented Diagnostics

### Contact SPD outlier aggregation

File:

- `tools/corex_spd_pcg_aggregate.py`

The aggregator now adds:

- `outliers_by_type`: top PT/EE/PE/PP rows ranked by `correction_diag_ratio_max`, PCG iter, and correction max.
- `high_iter_contact_leaders`: for the highest-PCG-iteration rows, the contact type with the largest correction/diagonal spike.
- `top_frames_by_pcg_iter`: frame-level PCG iteration hotspots.
- `top_frames_by_contact_ratio`: frame-level contact correction/diagonal hotspots.

This is post-processing only and does not change runtime behavior.

### PCG scalar audit

File:

- `src/backends/cuda/linear_system/linear_pcg_corex.cu.inc`

New opt-in switch:

```bash
UIPC_COREX_PCG_SCALAR_AUDIT=1
```

When enabled, `LinearPCG` emits:

```text
[corex_pcg_scalar_audit] frame=... newton=... iter=... dot_calls=... norm_calls=... fused_dotnorm_calls=... reduce2_calls=... host_scalar_results=...
```

The audit is intentionally lightweight and logs counts after each PCG solve. It does not alter convergence semantics.

### Assembly phase split

Files:

- `src/backends/cuda/linear_system/global_linear_system.cu`
- `src/backends/cuda/affine_body/abd_linear_subsystem.cu`

New `UIPC_COREX_PHASE_PROFILE=1` phase names include:

- `linear.assemble_clear`
- `linear.assemble_diag_subsystems`
- `linear.assemble_offdiag_subsystems`
- `abd_assemble.prepare_reporter_buffers`
- `abd_assemble.kinetic_shape`
- `abd_assemble.reporters`
- `abd_assemble.contact_zero_gradients`
- `abd_assemble.dytopo_effect`

These phase splits are diagnostic-only.

### Contact selected-rate parsing

File:

- `tools/simple_physics_audit/profile_summary.py`

The profile summary now parses `[corex_contact_early_stats]` and reports candidate totals, selected totals, selected rates, and per-type candidate/selected counts.

## Validation

Build:

- `cmake --build /root/libuipc-v11-restored/libuipc-v11-extracted/build_corex --target corex_demo --config Release -j8`: PASS

Correctness gates on GPU 1:

| Case | Result | Failure markers |
| --- | --- | --- |
| `simple90` | PASS | none |
| `simple300` | PASS | none |
| `stack120` | PASS | none |

Failure scan covered `NaN`, `nan`, `exception`, `reached max_iter`, assertion, abort, and core-dump markers.

Artifacts:

- `/tmp/corex_roadmap_20260502/`
- `/tmp/corex_roadmap_20260502/spd_pcg_outliers.json`
- `/tmp/corex_roadmap_20260502/spd_pcg_outliers.csv`

## Wrecking Ball A/B

All runs used:

```bash
CUDA_VISIBLE_DEVICES=1
UIPC_COREX_PHASE_PROFILE=1
UIPC_COREX_PCG_COST_DIAG=1
UIPC_COREX_PCG_SCALAR_AUDIT=1
UIPC_COREX_CONTACT_SPD_DIAG=1
UIPC_COREX_CONTACT_EARLY_ACTIVE_STATS=1
```

Grouped-row runs additionally used:

```bash
UIPC_COREX_SPMV_GROUPED_ROW=1
```

| Case | Config | Wall Time | PCG Iter Sum | Max PCG Iter | Newton Sum |
| --- | --- | ---: | ---: | ---: | ---: |
| `wb80` | default | 31s | 10911 | 69 | 243 |
| `wb80` | grouped row | 30s | 10875 | 70 | 245 |
| `wb150` | default | 87s | 47028 | 180 | 506 |
| `wb150` | grouped row | 100s | 54773 | 174 | 495 |

PCG cost buckets:

| Case | Config | SpMV | SpMV Sync | Preconditioner | DotNorm |
| --- | --- | ---: | ---: | ---: | ---: |
| `wb80` | default | 11.614% | 16.488% | 14.532% | 57.366% |
| `wb80` | grouped row | 11.737% | 17.197% | 14.378% | 56.689% |
| `wb150` | default | 11.698% | 16.813% | 14.498% | 56.990% |
| `wb150` | grouped row | 11.588% | 17.673% | 14.324% | 56.416% |

Decision: grouped-row SpMV does not pass the `wb150` gate. It slightly lowers max PCG iter and dotnorm percentage, but wall time and total PCG iterations regress. It remains opt-in only.

## Conditioning Findings

The high-PCG-iteration leader rows are PE-dominated in this run:

| Case | Top High-Iter Row | Leader Type | Ratio Max | PCG Iter | Frame | Newton |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `wb80` default | highest PCG row | PE | 3.723232 | 69 | 60 | 1 |
| `wb80` grouped | highest PCG row | PE | 4.571087 | 70 | 60 | 1 |
| `wb150` default | highest PCG row | PE | 4.546151 | 180 | 148 | 1 |
| `wb150` grouped | highest PCG row | PE | 4.294203 | 174 | 144 | 1 |

Interpretation:

- PE remains the clearest first target for conditioning diagnostics in these `wb80/wb150` runs.
- The grouped-row SpMV experiment changes solver trajectory and contact candidate totals enough that a small bucket improvement is not useful by itself.
- Future preconditioner work should not globally strengthen ABD blocks. It should be adaptive and driven by frame/contact outliers.

## PCG Scalar Audit

Scalar audit totals:

| Case | Config | PCG Iter Sum | Dot Calls | Norm Calls | Fused DotNorm Calls | Host Scalar Results | Host Results / Iter |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `wb80` | default | 10911 | 10911 | 322 | 11233 | 22466 | 2.059 |
| `wb80` | grouped row | 10875 | 10875 | 324 | 11199 | 22398 | 2.060 |
| `wb150` | default | 47028 | 47028 | 754 | 47782 | 95564 | 2.032 |
| `wb150` | grouped row | 54773 | 54773 | 842 | 55615 | 111230 | 2.031 |

Interpretation:

- With fused `rz/norm`, CoreX still has about two scalar-producing reductions per PCG iteration: `pAp` plus fused `rz/norm`.
- `norm_b` is one extra scalar per solve and is not the dominant count.
- Reducing dotnorm cost likely requires removing or delaying a per-iteration scalar boundary, not adding another finalize kernel.

## Assembly and Contact Filter Findings

The new phase split shows that CoreX linear assembly time is dominated by diag subsystem assembly, and within ABD assembly by dytopo effect conversion/assembly:

| Case | Config | Linear Diag Sum | Linear OffDiag Sum | ABD Kinetic Sum | ABD DyTopo Sum |
| --- | --- | ---: | ---: | ---: | ---: |
| `wb80` | default | 12876.878ms | 0ms | 454.586ms | 12416.052ms |
| `wb80` | grouped row | 12966.436ms | 0ms | 459.490ms | 12500.553ms |
| `wb150` | default | 35022.125ms | 0ms | 1125.686ms | 33881.602ms |
| `wb150` | grouped row | 39457.791ms | 0ms | 1255.274ms | 38186.690ms |

Contact selected-rate parsing shows high candidate volume and low selected ratio:

| Case | Config | Candidate Sum | Selected Sum | Mean Selected Rate |
| --- | --- | ---: | ---: | ---: |
| `wb80` | default | 726818 | 125291 | 0.099306 |
| `wb80` | grouped row | 759392 | 123468 | 0.094554 |
| `wb150` | default | 4515477 | 1066230 | 0.204105 |
| `wb150` | grouped row | 6120311 | 1411793 | 0.207322 |

Interpretation:

- Assembly optimization should focus on `abd_assemble.dytopo_effect` first, not generic sync removal.
- Contact filtering still has high candidate waste, but early filtering has a known convergence risk. Any future contact optimization must prove selected-set equivalence or remain diagnostic-only.

## Adaptive Preconditioner Design

Based on the outlier data and the previous structured-block regression, the next preconditioner candidate should be adaptive:

- Keep Jacobi as the default baseline.
- Do not globally apply all four 3x3 blocks per ABD body.
- Use diagnostics to gate structured blocks only around frames/contact regimes where PE/PP/EE correction/diagonal outliers predict benefit.
- Prefer cheap apply-time masks or coarse solve-level activation over per-body host-side decisions.
- Keep full 12x12 inverse out of scope because it already failed `simple90` with NaN.

Suggested opt-in shape for a later implementation:

```bash
UIPC_COREX_ABD_PRECOND_ADAPTIVE_STRUCT=1
UIPC_COREX_ABD_PRECOND_ADAPTIVE_RATIO=<threshold>
```

The threshold should be derived from `spd_pcg_outliers.json`, not hard-coded from a single frame.

## Decision

Do not run `wb400` for this round. The only behavior-changing experiment, grouped-row SpMV, failed the `wb150` gate.

Keep all new changes as diagnostics:

- SPD/PCG outlier aggregation
- PCG scalar audit
- Assembly phase split
- Contact selected-rate parsing

Recommended next implementation target:

1. Use the new outlier map to design a PE-focused conditioning experiment.
2. Separately investigate `abd_assemble.dytopo_effect`, because it dominates the assembly split.
3. Avoid more PCG reduction experiments unless they remove a scalar boundary without adding per-iteration launches.
