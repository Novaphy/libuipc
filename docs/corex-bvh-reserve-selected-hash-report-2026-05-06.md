# CoreX BVH Reserve and Selected-Set Hash Report 2026-05-06

## Summary

This round moved away from frame-count AABB sync guards and focused on two
diagnostic/optimization points in the contact detection/filter path:

- Added opt-in selected-set hash diagnostics with
  `UIPC_COREX_SELECTED_SET_HASH_DIAG=1`.
- Increased CoreX `StacklessBVH::Config::reserve_ratio` default from `1.2` to `2.0`,
  still overrideable through `UIPC_COREX_BVH_QUERY_RESERVE_RATIO`.

The default AABB async mask was restored to `0`. Triangle-only AABB async remains
available via `UIPC_COREX_FILTER_AABB_ASYNC_MASK=8`, but is not default because it
can alter selected-set evolution on CoreX.

## Findings

Selected-set count/hash diagnostics showed that `MASK=0` and `MASK=8` can produce
different late-frame selected-set trajectories. That confirms AABB async is not a
pure timing-only optimization on the current CoreX path.

The meaningful stable optimization was BVH query reserve sizing. A larger reserve
reduces the chance of overflow-triggered resize/requery in the Stackless BVH query
buffers without changing contact filtering semantics.

## Lightweight Validation

No SPD/phase/selected/matrix heavy diagnostics.

Reference synced AABB baseline:

- `MASK=0`, old reserve `1.2`: `wb400 187s`, PCG sum `192785`, PCG max `181`

Reserve candidate:

- `MASK=0`, reserve `2.0`: `wb400 169s`, PCG sum `161046`, PCG max `176`

Final default after promoting reserve `2.0` and restoring AABB default sync:

- `wb150`: `63s`, PCG sum `55385`, PCG max `170`
- `wb400`: `169s`, PCG sum `166117`, PCG max `181`

Triangle-only AABB async plus reserve `2.0` reached `167s` in one run, but repeated
default/explicit-mask runs showed substantial trajectory and wall-time variability.
It is therefore kept as opt-in rather than treated as the stable path.

## Decision

Default changes:

- Promote BVH query reserve ratio `2.0`.
- Restore default AABB sync mask to `0`.

Opt-in diagnostics/candidates:

- `UIPC_COREX_SELECTED_SET_HASH_DIAG=1`
- `UIPC_COREX_BVH_QUERY_RESERVE_RATIO=<ratio>`
- `UIPC_COREX_FILTER_AABB_ASYNC_MASK=8`

Artifacts are in `/tmp/corex_selected_hash_20260506/`.
