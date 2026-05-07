# CoreX SPD/PCG Next Optimization Report

Date: 2026-05-01

## Summary

This round followed the `corex-spd-pcg-next` plan and kept the default simulation path conservative. The implemented default change is limited to removing unnecessary device-wide synchronizations between equivalent `filter_active` kernels. All other numerical experiments and diagnostics are opt-in.

Default regression on GPU 1 passed:

- `simple90`: PASS, 5s
- `simple300`: PASS, 11s
- `stack120`: PASS, 6s
- `wrecking_ball80`: PASS, 31s
- `wrecking_ball150`: PASS, 99s
- `wrecking_ball400`: PASS, 288s
- `wrecking_ball800`: PASS, 610s
- `domino600`: PASS, 64s

No default regression log contained `NaN`, `nan`, `exception`, or `reached max_iter`.

## Implemented Changes

### Contact Hessian/SPD diagnostics

File:

- `src/backends/cuda/contact_system/contact_models/ipc_simplex_normal_contact.cu`

Added opt-in instrumentation behind:

- `UIPC_COREX_CONTACT_SPD_DIAG=1`

The diagnostic records, per contact type (`PT`, `EE`, `PE`, `PP`):

- number of Hessian blocks passed through `make_spd`
- number and rate of blocks changed by projection
- average Gershgorin lower bound before projection
- average and max absolute Hessian correction magnitude

This instrumentation is intended for conditioning diagnosis only and is not enabled by default.

Diagnostic note: after fixing an initial managed-memory crash in this opt-in path, the `wrecking_ball80` diagnostic run completed cleanly but did not emit `[corex_spd_contact]` records. The default path is unaffected. The next debugging step is to confirm which `IPCSimplexNormalContact` implementation branch is active at runtime and why the opt-in log hook is not reached even when simplex contacts are assembled.

### PCG single-iteration cost audit

File:

- `src/backends/cuda/linear_system/linear_pcg_corex.cu.inc`

Added opt-in logging behind:

- `UIPC_COREX_PCG_COST_DIAG=1`

The new `[corex_pcg_cost]` line reports total and percentage cost for:

- `spmv`
- `spmv_sync`
- `preconditioner`
- `dotnorm`

Measured on GPU 1:

- `wrecking_ball150`: 83.481s, 747 PCG cost records, no failure markers.
- `wrecking_ball400`: 310.792s, 2606 PCG cost records, no failure markers.

Aggregate cost split:

| scene | spmv | spmv_sync | preconditioner | dotnorm |
|---|---:|---:|---:|---:|
| `wb150` | 11.631% | 16.870% | 14.729% | 56.771% |
| `wb400` | 11.570% | 17.627% | 14.348% | 56.454% |

Conclusion: PCG is still dominated by reduction/dot-norm style work. Synchronization after SpMV remains non-trivial at about 17%, but `dotnorm` is the largest measured single bucket.

### SPD-safe structured block Jacobi experiment

File:

- `src/backends/cuda/affine_body/abd_diag_preconditioner.cu`

Added an opt-in structured preconditioner behind:

- `UIPC_COREX_ABD_PRECOND_STRUCT_BLOCK=1`

It does not attempt a full 12x12 inverse. Instead, each ABD block is split into four conservative 3x3 blocks:

- translation block
- three affine row blocks

Each 3x3 block is inverted only if a strict diagonal-dominance/SPD safety check passes. Failed blocks fall back to the existing Jacobi reciprocal path. This keeps the experiment SPD-safe by construction and avoids the unstable full block inverse that previously failed `simple90`.

This remains opt-in and was not defaulted in this round.

### Equivalent contact filter cost optimization

File:

- `src/backends/cuda/collision_detection/filters/stackless_bvh_simplex_trajectory_filter.cu`

The `filter_active` kernels for PP, CodimPE, PT, and EE write to disjoint temporary ranges and are followed by selection on the same stream. The previous implementation synchronized the whole device after each kernel launch. This round removes those default device-wide synchronizations and keeps only launch error probing.

Fallback switch:

- `UIPC_COREX_FILTER_ACTIVE_SYNC=1`

This change preserves active-set equivalence: no new BVH predicate pruning was added, and selected contact semantics are unchanged.

## Validation

Build:

- `cmake --build build_corex --config Release -j 8`: PASS after fixes.

Correctness:

- Full default regression passed on GPU 1: `simple90/simple300/stack120/wb80/wb150/wb400/wb800/domino600`.
- After the final diagnostic-path fixes, `simple90` was rerun and passed.

Diagnostic runs:

- `UIPC_COREX_PCG_COST_DIAG=1 wb150`: PASS, no failure markers.
- `UIPC_COREX_PCG_COST_DIAG=1 wb400`: PASS, no failure markers.
- `UIPC_COREX_CONTACT_SPD_DIAG=1 wb80`: PASS after managed-memory fix, no failure markers, but no SPD records emitted.

## Recommendation

Default only the synchronization removal in `filter_active`; keep the structured preconditioner and SPD/PCG diagnostics opt-in. The next useful optimization target is PCG reduction cost, because `dotnorm` accounts for about 56% of measured PCG bucket time on both `wb150` and `wb400`.
