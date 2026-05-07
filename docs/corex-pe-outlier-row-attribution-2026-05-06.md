# CoreX PE Outlier Row Attribution 2026-05-06

## Summary

This round added an opt-in PE outlier diagnostic:

- `UIPC_COREX_PE_OUTLIER_DIAG=1`
- `UIPC_COREX_PE_OUTLIER_RATIO_TRIGGER=<ratio>`, default `0.5`

The diagnostic records, per contact assembly, the PE contact with the maximum
`correction / diag_abs` ratio after SPD projection, plus a count of PE contacts above
the trigger. It reports:

- PE tuple `(point, edge_v0, edge_v1)`
- contact index
- max correction/diag ratio
- correction magnitude
- diagonal magnitude
- Gershgorin lower estimate

Default behavior is unchanged.

## Diagnostic Run

Run:

- Stable default path: AABB sync, BVH reserve `2.0`
- `wrecking_ball160`
- `UIPC_COREX_PCG_COST_DIAG=1`
- `UIPC_COREX_SELECTED_SET_DIAG=1`
- `UIPC_COREX_PE_OUTLIER_DIAG=1`
- `UIPC_COREX_PE_OUTLIER_RATIO_TRIGGER=0.5`

Artifact:

- `/tmp/corex_pe_outlier_20260506/pe_outlier_wb160.log`

## Findings

The high-PCG interval is dominated by a small number of repeated PE outlier tuples.

Most frequent PE outliers in frames `140-152`:

- `(9327, 9411, 9413)`: 123 observations, max ratio `4.626`
- `(11103, 11187, 11189)`: 16 observations, max ratio `4.346`
- `(6531, 5990, 6000)`: 5 observations, max ratio `4.617`
- `(4527, 5020, 5031)`: 4 observations, max ratio `6.715`
- `(3544, 4027, 4038)`: 4 observations, max ratio `6.661`
- `(4039, 3546, 3557)`: 4 observations, max ratio `4.644`

Top PCG rows in the same interval:

- Frame `148`, newton `1`: PCG `179`, PE `(6531,5990,6000)`, ratio `4.617`
- Frame `147`, newton `1`: PCG `178`, PE `(6531,5990,6000)`, ratio `4.615`
- Frame `149`, newton `1`: PCG `176`, PE `(11103,11187,11189)`, ratio `4.331`
- Frame `146`, newton `1`: PCG `169`, PE `(6531,5990,6000)`, ratio `4.540`
- Frame `141`, newton `1`: PCG `168`, PE `(9327,9411,9413)`, ratio `4.615`

This confirms that the remaining high-PCG behavior is not a broad all-PE issue. It
is concentrated around a few persistent PE contacts/vertices with high SPD
projection correction.

## Decision

No optimization was promoted in this round. The previous PE-only diagonal
conditioning improved short runs but failed `wb400`, and this diagnostic explains
why: a global PE diagonal shift is too broad for a problem concentrated in a few
PE row/contact hotspots.

## Row Hotspot Follow-up

Added two more opt-in attribution outputs:

- `UIPC_COREX_MATRIX_ROW_HOTSPOT_DIAG=1` logs the top matrix block rows by
  absolute row sum when `UIPC_COREX_MATRIX_QUALITY_DIAG=1` is enabled.
- `UIPC_COREX_PE_OUTLIER_DIAG=1` now also logs the PE tuple's mapped
  `body=(...)`, using `GlobalVertexManager::body_ids()`.

Artifact:

- `/tmp/corex_row_hotspot_20260506/row_hotspot_body_wb160.log`

In frames `140-152`, the rank-0 matrix row is overwhelmingly stable:

- rank-0 row `53`: `143` observations
- rank-0 row `48`: `1` observation
- rank-0 row `4`: `1` observation

The dominant PE outlier bodies in the same window are different:

- body `256`: `230` touches
- body `246`: `115` touches
- body `2`: `32` touches
- body `3`: `23` touches
- body `4`: `11` touches

Only `7` PE outlier solves overlap the top-8 matrix rows. The clearest high-PCG
overlap is frame `149`, newton `1`: PCG `179`, PE `(6531,5990,6000)`,
body `(13,12,12)`, with body `12` present in the top rows. However, most high-PCG
solves still have the matrix row hotspot led by rows `53/54/48`, not by the
current PE outlier body.

This changes the attribution: PE outliers are real and sometimes enter the hot
rows, but they are not the dominant source of global row imbalance.

## Row Owner Follow-up

Extended `UIPC_COREX_MATRIX_ROW_HOTSPOT_DIAG=1` to report row ownership:

- `owner_index`
- `local_block`
- `dof_offset`
- `dof_count`
- `subsystem`

Artifact:

- `/tmp/corex_row_owner_20260506/row_owner_wb160.log`

In frames `140-152`, every rank-0 row belongs to `ABDLinearSubsystem`:

- rank-0 row `53`: `145` observations, local ABD block `53`
- rank-0 row `48`: `6` observations, local ABD block `48`
- rank-0 owner: `ABDLinearSubsystem`, `151` observations

`ABDLinearSubsystem` uses `12` DoFs per affine body, i.e. four 3x3 block rows
per body. Therefore:

- row `48` maps to body `12`, affine block `0`
- row `53` maps to body `13`, affine block `1`
- row `54` maps to body `13`, affine block `2`

This is a better attribution than the original PE-only hypothesis: the persistent
linear-system hotspot is an ABD affine-body local block issue around bodies `12`
and `13`. PE outliers sometimes touch body `12/13`, but the row imbalance is
owned by ABD rows even in solves where the current PE max outlier maps elsewhere.

## ABD Preconditioner Probe

Re-tested the existing opt-in ABD structured 3x3 block preconditioner:

- `UIPC_COREX_ABD_PRECOND_STRUCT_BLOCK=1`
- `UIPC_COREX_ABD_PRECOND_STRUCT_BLOCK_STATS=1`

Artifact:

- `/tmp/corex_abd_struct_20260506/struct_wb160.log`

Result versus default `wb160`:

- default: `55s`, PCG solves `795`, PCG sum `51436`, max `169`, mean `64.70`
- structured block: `70s`, PCG solves `912`, PCG sum `58606`, max `151`, mean `64.26`

The structured block preconditioner does reduce the worst PCG solve, but it
increases solve count, total PCG work, and wall time. It should remain opt-in and
not be promoted.

## Next Direction

The next useful diagnostic should aggregate PE outlier contribution by vertex/row:

- Count how many high-ratio PE outliers touch each vertex.
- Track each vertex's max correction/diag ratio and accumulated correction.
- Join those vertex IDs with matrix row imbalance diagnostics.

Only after identifying row-local concentration should we try a targeted remedy.
Given the owner follow-up, the immediate next diagnostic should attribute ABD
body `12/13` row contributions by source family: body kinetic/shape, ABD reporter
hessians, and DyTopo/contact-transformed Hessians.
