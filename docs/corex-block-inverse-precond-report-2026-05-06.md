# CoreX 12x12 ABD Block-Inverse Preconditioner Report

Date: 2026-05-06

## Summary

This round restored the NVIDIA-equivalent algorithmic choice for the ABD diagonal
preconditioner on CoreX: a full SPD 12x12 block inverse per body. The
implementation uses an SPD-safe LDLT factorization with an explicit pivot test,
followed by an explicit dense inverse construction, and is dispatched through
an unconditional 12x12 mat-vec apply kernel. A per-body Jacobi inverse is folded
into the same `diag_inv` buffer for any body that fails the LDLT pivot test, so
the apply path stays branch-free.

The new path is **promoted to the CoreX default** because every gate from the
plan passed:

- `simple90` / `simple300` / `stack120` correctness gates: clean (no NaN, no
  exception, no `reached max_iter`).
- `wb400` long-run A/B (same build): wall `167s -> 146s` (-12.6%), PCG iter
  sum `168717 -> 88478` (-47.6%), PCG iter max `183 -> 122` (-33.3%).
- Regression: `domino600` `85s -> 51s` (-40%), `wb800` finishes 800 frames in
  `326s` with `0` rejected bodies and no failure markers.

Two rollback environment variables are kept in place:

- `UIPC_COREX_ABD_PRECOND_BLOCK_INVERSE=0` disables the new path and selects
  the legacy Jacobi extract.
- `UIPC_COREX_ABD_PRECOND_DIAG_JACOBI=1` is a hard rollback that forces the
  legacy Jacobi extract regardless of the block-inverse flag.

The legacy `UIPC_COREX_ABD_PRECOND_STRUCT_BLOCK=1` switch still takes
precedence over both, since it allocates and uses a different buffer layout.

## Background

`docs/corex-stable-effective-optimizations-summary-2026-05-05.md` and
`docs/corex-pe-outlier-row-attribution-2026-05-06.md` traced the residual CoreX
gap to PCG iteration count rather than per-iteration cost. NVIDIA's wb400 runs
in `45.6s` with `~38066` total PCG iterations; CoreX (pre-this-round) was at
`~167s` and `~168717` PCG iterations, a ~4.4x iteration-count gap.

The NVIDIA path
`src/backends/cuda/affine_body/abd_diag_preconditioner.cu#L406-L412` uses
`muda::eigen::inverse(diag_hessian(i))` to produce a full 12x12 block inverse
per body. The CoreX `UIPC_COREX_CUDA10_COMPAT` branch had to fall back to a
diagonal-only Jacobi inverse because the muda `GaussEliminationInverse` used
inside `muda::eigen::inverse` does not implement partial pivoting and produces
catastrophic cancellation on CoreX's float-precision pipeline for ill-conditioned
12x12 ABD Hessians (PE-heavy frames around bodies 12 and 13).

A previously tested 4-block 3x3 structured inverse only reduced PCG iter sum by
~4% because it discarded inter-block coupling.

## Implementation

The new path is gated by `block_inverse_precond_enabled()` in
`src/backends/cuda/affine_body/abd_diag_preconditioner.cu`. When enabled, the
class allocates `diag_inv: DeviceBuffer<Matrix12x12>` and
`block_inv_status: DeviceBuffer<int>`, and dispatches the new extract kernel.

`kernel_abd_precond_extract_block_inverse` performs, per ABD body:

1. Symmetrize the input Hessian: $A = \tfrac{1}{2}(H + H^\top)$.
2. Always extract a per-DoF Jacobi reciprocal $1/A_{kk}$ for fallback.
3. In-register LDLT factorization of $A$ via `corex_ldlt_factorize_inplace<12>`.
   Pivot rejection: if any $D_k < \varepsilon$ where
   $\varepsilon = \max(\max_k |A_{kk}|\cdot 10^{-10}, 10^{-30})$, the body is
   marked rejected.
4. On accept, compute $A^{-1}$ column by column via
   `corex_ldlt_explicit_inverse<12>` (which solves $A x_j = e_j$ using the
   in-place LDLT factors). The result is written column-major into
   `diag_inv[i]`.
5. On reject, write a diagonal-only inverse into `diag_inv[i]` whose nonzero
   entries are the per-DoF Jacobi reciprocals, so that the apply path is
   identical for accepted and rejected bodies.

`kernel_abd_block_inverse_apply` is a branch-free 12x12 mat-vec per body that
mirrors NVIDIA's apply semantics
`z.segment<12>(i*12) = diag_inv(i) * r.segment<12>(i*12)`.

The new extract kernel uses a smaller block size (`64`) than the legacy Jacobi
extract (`256`) to leave room for the in-register 12x12 working matrix and LDLT
temporaries.

The runtime stat line `[corex_abd_precond_block_inv]` reports per-assembly
`bodies / accepted / rejected` when `UIPC_COREX_ABD_PRECOND_BLOCK_INVERSE_STATS=1`
or `UIPC_COREX_TRACE_LINEAR_SYSTEM=1` is set.

## Validation

All runs used GPU1 via `CUDA_VISIBLE_DEVICES=1`, with `--gpu 0` inside
`corex_demo`. Build: `cmake --build build_corex --target corex_demo --config
Release -j8` passed.

### Correctness gates (with `UIPC_COREX_ABD_PRECOND_BLOCK_INVERSE=1`)

| Scene      | Wall | LDLT accepted/rejected | Failure markers       |
|------------|------|------------------------|-----------------------|
| simple90   | `4s` | `2 / 0`                | none                  |
| simple300  | `10s`| `2 / 0`                | none                  |
| stack120   | `6s` | `4 / 0`                | none                  |

### Short A/B (`wb80`, `wb150`)

| Scene  | Path        | Wall | PCG calls | PCG iter sum | PCG iter max |
|--------|-------------|------|-----------|--------------|--------------|
| wb80   | Jacobi      | `18s`| `322`     | `10960`      | `69`         |
| wb80   | Block-inv   | `16s`| `310`     | `4762`       | `48`         |
| wb150  | Jacobi      | `50s`| `749`     | `46896`      | `177`        |
| wb150  | Block-inv   | `49s`| `743`     | `27467`      | `124`        |

`wb150` `[corex_pcg_cost]` shows the precond bucket rises from ~14% (Jacobi) to
~26% (block inverse) as expected, but per-iter time only rises ~14%
(`0.254 ms -> 0.290 ms`). The 41% iter-sum reduction more than compensates.

### Long run (`wb400`, same build)

| Path      | Wall  | PCG calls | PCG iter sum | PCG iter max |
|-----------|-------|-----------|--------------|--------------|
| Jacobi    | `167s`| `2558`    | `168717`     | `183`        |
| Block-inv | `146s`| `2428`    | `88478`      | `122`        |
| **Delta** | **-12.6%** | -5.1% | **-47.6%** | **-33.3%**   |

All three plan thresholds pass: wall not worse, PCG iter sum drops, PCG iter
max not regressed. We did not hit the `<= 130s` ideal target; the leftover gap
is the doubled per-iter precond apply cost (12x12 mat-vec vs 12 reciprocal
multiplies) plus the heavier `dotnorm`/`spmv_sync` buckets that this round did
not touch.

### Regression

| Scene      | Path        | Wall | PCG iter sum | PCG iter max | Status     |
|------------|-------------|------|--------------|--------------|------------|
| domino600  | Jacobi      | `85s`| `79894`      | `122`        | finished   |
| domino600  | Block-inv   | `51s`| `17560`      | `22`         | finished   |
| wb800      | Block-inv   | `326s`| `170430`    | `123`        | finished   |

`wb800` ran with block-inverse default-on and reached `End Frame: 799` cleanly
with zero LDLT rejections.

### Default-on and rollback verification

Re-built with the default flipped to on, then `wb150` was run three ways:

| Configuration                                  | PCG iter sum | Notes                          |
|------------------------------------------------|--------------|--------------------------------|
| (no env)                                       | `28989`      | block-inverse active (default) |
| `UIPC_COREX_ABD_PRECOND_BLOCK_INVERSE=0`       | `48342`      | rollback to Jacobi             |
| `UIPC_COREX_ABD_PRECOND_DIAG_JACOBI=1`         | `66608`      | hard rollback to Jacobi        |

Both rollback envvars produce iteration counts in the legacy Jacobi range,
confirming they are wired correctly.

## Decisions

- Default behavior: block-inverse is enabled by default in CoreX.
- LDLT pivot tolerance: `eps = max(max(|A_kk|) * 1e-10, 1e-30)`. All measured
  scenes hit `accepted=N, rejected=0` so the fallback path is currently exercised
  only by stress tests; this tolerance can be revisited if any scene starts
  showing nonzero `rejected`.
- Apply uses an explicit-inverse mat-vec rather than a per-iteration LDLT solve.
  Empirically the explicit-inverse path is stable across all measured scenes,
  so the more expensive solve fallback was not needed.

## Not Done (Intentionally Out of Scope)

- PCG convergence criterion (`norm_r vs r_tol`) was not changed.
- `dot/norm` and SpMV kernels were not changed.
- Contact / matrix converter / DyTopo were not changed.
- The NVIDIA `#else` branch of `abd_diag_preconditioner.cu` was not touched;
  only the `UIPC_COREX_CUDA10_COMPAT` branch was modified.

## Risks and Follow-ups

- The new extract kernel's working matrix is held in per-thread local memory.
  At block size `64`, the `[corex_pcg_cost]` precond bucket is ~26% on wb150,
  which is acceptable. If a future scene shows the precond bucket dominating,
  the extract can be split into two kernels (LDLT factor -> explicit inverse)
  with `__launch_bounds__`, or moved to an LDLT-solve apply that trades
  apply-time for precond storage.
- Bodies that fail the LDLT pivot test currently fall back to a diagonal-only
  inverse for the entire body. If a future scene shows nonzero `rejected` and
  worse PCG behavior, a per-3x3-block fallback (mixing accepted 3x3 inverses
  with Jacobi columns) is the next step.
- `dotnorm` is still the largest per-iter PCG bucket (~49% on wb150 with
  block-inverse). A future round can target the PCG scalar boundary
  (D2H/sync) compression now that iter-sum has dropped enough to make
  per-iter cost the next bottleneck.
