# CoreX wb400 Historical Best Reproduction 2026-05-06

## Goal

Reproduce the previous `wrecking_ball400` historical best near `175s` before trying
new optimization directions.

## Key Finding

The historical `175s` artifact was
`/tmp/corex_conditioning_20260505/wb400_default_diag_aligned.log`, not the ordinary
default diagnostic run from the same directory.

Same historical batch:

- `wb400_default_diag.log`: `185s`
- `wb400_default_diag_aligned.log`: `175s`, PCG sum `164626`, PCG max `175`

The current reproduction used the aligned/default diagnostic stack with triangle AABB
async explicitly selected:

```bash
CUDA_VISIBLE_DEVICES=1 \
UIPC_COREX_FILTER_AABB_ASYNC_MASK=8 \
UIPC_COREX_CONTACT_INPUT_DIAG=0 \
UIPC_COREX_PHASE_PROFILE=1 \
UIPC_COREX_PCG_COST_DIAG=1 \
UIPC_COREX_SELECTED_SET_DIAG=1 \
UIPC_COREX_MATRIX_QUALITY_DIAG=1 \
UIPC_COREX_CONTACT_SPD_DIAG=1 \
./Release/bin/corex_demo --backend cuda --scene wrecking_ball --frames 400 --gpu 0
```

Result:

- Wall: `171s`
- PCG sum: `166992`
- PCG max: `178`
- PCG solve count: `2567`
- Status: `0`

Same cleaned historical diagnostic scope without triangle AABB async
(`UIPC_COREX_FILTER_AABB_ASYNC_MASK=0`) was also run for comparison:

- Wall: `172s`
- PCG sum: `162866`
- PCG max: `181`
- PCG solve count: `2544`
- Status: `0`

Under this matched scope, triangle-only AABB async is a small wall-time improvement
(`172s -> 171s`) and slightly lowers PCG max (`181 -> 178`), while PCG total is
higher. This is a much narrower delta than the earlier mixed-baseline comparison.

Full AABB async (`UIPC_COREX_FILTER_AABB_ASYNC=1`) was also rerun under the same
matched scope:

- Wall: `181s`
- PCG sum: `171346`
- PCG max: `179`
- PCG solve count: `2610`
- Status: `0`

In this cleaned comparison, full async does not reproduce the earlier `162s` result.
It is slower than both `MASK=0` and triangle-only `MASK=8`, with more PCG work.

Artifacts:

- Log: `/tmp/corex_reproduce_175_20260506/wb400_aligned_mask8.log`
- Profile: `/tmp/corex_reproduce_175_20260506/wb400_aligned_mask8_profile.json`
- SPD/PCG summary: `/tmp/corex_reproduce_175_20260506/wb400_aligned_mask8_pcg_summary.json`
- No-triangle comparison log: `/tmp/corex_reproduce_175_20260506/wb400_aligned_mask0.log`
- Full-async comparison log: `/tmp/corex_reproduce_175_20260506/wb400_aligned_full_async.log`

## Interpretation

The historical wall-time target was reproduced and exceeded. It is not an exact
bitwise/trace reproduction: the current run has slightly different PCG totals and
high-PCG contact counts from the old `175s` log, so the old and current code states
should not be treated as identical numerical traces.

One source of baseline confusion was introduced in the latest FPS round:
`UIPC_COREX_CONTACT_SPD_DIAG=1` also emitted `[corex_contact_input]` diagnostics.
Those input diagnostics are now separated behind `UIPC_COREX_CONTACT_INPUT_DIAG=1`
so the main SPD diagnostic baseline is closer to the historical measurement scope.
