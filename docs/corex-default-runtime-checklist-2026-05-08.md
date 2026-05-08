# CoreX Default Runtime Checklist

This document records the push-ready CoreX default runtime path after pruning rejected experiment branches. Normal production runs should use the code defaults directly, without extra performance environment variables.

Related summaries:

- [Stable effective optimizations](./corex-stable-effective-optimizations-summary-2026-05-05.md)
- [Current NVIDIA performance gap analysis](./corex-current-nvidia-performance-gap-analysis-2026-05-07.md)

## Default Path

The current CoreX default path includes:

- ABD `12x12` block-inverse preconditioner, default on.
- PCG pinned scalar readback, default on.
- PCG fused `rz/norm` reduction, default on.
- Device GE2SYM / matrix conversion path, default on.
- Parallel ABD DyTopo assembly, default on.
- Conservative AABB/filter synchronization policy, with only proven default behavior enabled.

Do not set additional performance env vars for a normal run. The historical best default reference is:

```text
wb400 wall = 143.956s
PCG calls = 2457
PCG iter sum = 88441
PCG iter max = 122
failure markers = 0
```

## Canonical Commands

Build:

```bash
cmake --build build_corex --target corex_demo --config Release -j8
```

Smoke:

```bash
CUDA_VISIBLE_DEVICES=1 ./build_corex/Release/bin/corex_demo --backend cuda --scene simple --frames 90 --gpu 0
```

Reference performance run:

```bash
CUDA_VISIBLE_DEVICES=1 ./build_corex/Release/bin/corex_demo --backend cuda --scene wrecking_ball --frames 400 --gpu 0
```

Representative non-wrecking regression:

```bash
CUDA_VISIBLE_DEVICES=1 ./build_corex/Release/bin/corex_demo --backend cuda --scene stack --frames 200 --gpu 0
```

If the binary is located under `./Release/bin/corex_demo` in a local build layout, keep the same arguments and only replace the executable path.

## Rollback Controls To Keep

These env vars are intentionally preserved because they protect the current default path or help triage regressions:

- `UIPC_COREX_ABD_PRECOND_BLOCK_INVERSE=0`
- `UIPC_COREX_ABD_PRECOND_DIAG_JACOBI=1`
- `UIPC_COREX_PCG_PINNED_SCALAR=0`
- `UIPC_COREX_PCG_SEPARATE_RZ_NORM=1`
- `UIPC_COREX_ABD_DYTOPO_SERIAL=1`
- `UIPC_COREX_FORCE_HOST_GE2SYM=1`
- `UIPC_COREX_FILTER_ACTIVE_SYNC=1`
- `UIPC_COREX_FILTER_AABB_ASYNC_MASK=0`

Diagnostics such as trace, phase profile, matrix quality, PCG cost/diag, contact SPD/input/outlier diagnostics, and selected-set diagnostics remain valid for investigations.

## Rejected Envs Removed From Production Code

The following documented no-benefit or regressing paths have been pruned from the active CoreX production code. Do not reintroduce them unless a new report shows a stable `wb400` win without PCG or correctness regression:

- `UIPC_COREX_ORTHO_POTENTIAL_GPU`
- `UIPC_COREX_ABD_ENERGY_GPU`
- `UIPC_COREX_ABD_ENERGY_REDUCTION_GPU`
- `UIPC_COREX_ABD_LIGHT_REDUCTION_GPU`
- `UIPC_COREX_ABD_FUSED_ENERGY_GPU`
- `UIPC_COREX_TOI_GPU`
- `UIPC_COREX_TOI_DEVICE_MIN`
- `UIPC_COREX_FILTER_TOI_SKIP_PRE_SYNC`
- `UIPC_COREX_DYTOPO_UPPER_BOUND_COMPACTION`
- `UIPC_COREX_PCG_REDUCE2`
- `UIPC_COREX_PCG_FUSED_SPMV_DOT`
- `UIPC_COREX_PCG_RESIDUAL_REPLACE_INTERVAL`
- `UIPC_COREX_SPMV_SEGMENTED_ROW`
- `UIPC_COREX_SPMV_GROUPED_ROW`
- `UIPC_COREX_SPMV_WARP_SEGMENTED`
- `UIPC_COREX_MATCONV_ASYNC`
- `UIPC_COREX_MATCONV_BLOCK_REDUCE`
- `UIPC_COREX_MATCONV_NVIDIA_SEG_REDUCE`
- `UIPC_COREX_CONTACT_EARLY_ACTIVE_FILTER`
- `UIPC_COREX_CONTACT_EARLY_ACTIVE_SCALE`
- `UIPC_COREX_CONTACT_EARLY_ACTIVE_STATS`
- `UIPC_COREX_PE_DIAG_REG`
- `UIPC_COREX_PE_KAPPA_SCALE`
- `UIPC_COREX_CONTACT_SPD_CONDITION`
- `UIPC_COREX_CONTACT_SPD_CONDITION_PT`
- `UIPC_COREX_ABD_PRECOND_STRUCT_BLOCK`
- `UIPC_COREX_ABD_PRECOND_STRUCT_BLOCK_STATS`
- `UIPC_COREX_ABD_PRECOND_SCALED_LDLT`

## Acceptance Gates

Before treating future changes as production defaults, require:

- `simple90` smoke passes without NaN, exception, assert, abort, or `reached max_iter`.
- `wb80` stays in the same PCG stability band as the default path.
- One representative non-wrecking scene, such as `stack200` or `domino300`, completes.
- `wb400` remains near `143-146s`, PCG sum near `88k`, and PCG max near `122`.
