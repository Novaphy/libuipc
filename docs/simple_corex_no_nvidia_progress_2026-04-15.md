# simple overall adaptation progress (no NVIDIA server)

## Context

- Current server has no NVIDIA GPU/driver tools (`nvidia-smi` unavailable), and external NVIDIA execution is unavailable in this phase.
- Therefore, NVIDIA truth baseline is downgraded from "blocking prerequisite" to "future enhancement item".
- Current acceptance path is changed to non-blocking local criteria:
  1. No penetration / stable contact.
  2. Sustained physically plausible post-contact motion (no freeze/hover artifact).
  3. First-divergence convergence across controlled single-variable experiments.
  4. Maximal semantic equivalence with `/root/src` in high-risk kernels.

## Deliverables added in this round

- `tools/simple_physics_audit/export_nvidia_reference_contract.sh`
  - One-command pipeline to run `corex_demo` and freeze + metricize a NVIDIA baseline under parity baselines.
- `apps/examples/corex_demo/main.cpp`
  - Added env-only experiment overrides (default behavior unchanged):
    - `UIPC_SIMPLE_FORCE_FRICTION_ENABLE`
    - `UIPC_SIMPLE_FORCE_DHAT`
    - `UIPC_SIMPLE_FORCE_DT`
    - `UIPC_SIMPLE_FORCE_MU`
    - `UIPC_SIMPLE_FORCE_KAPPA_GPA`
  - Purpose: single-variable isolation without repeatedly editing source semantics.
- `src/backends/cuda/affine_body/abd_linear_subsystem.cu`
  - Restored upstream-equivalent semantics in CoreX compat path:
    - diagonal body hessian writes upper-triangle 3x3 blocks only (same as `zero_out_lower` intent),
    - same-body dytopo hessian writes upper-triangle blocks only,
    - removed fixed-dynamic reporter extra diagonal accumulation fork,
    - removed unconditional host-side `dq/x` debug memcpy print in `retrieve_solution`.

## Single-variable experiment results

### 1) friction path isolation

Runs:
- default: `/tmp/corex_simple_contract_default_env_90f`
- friction off: `/tmp/corex_simple_contract_fric_off_90f`
- friction on: `/tmp/corex_simple_contract_fric_on_90f`

Comparisons:
- `default vs friction_off`: `/tmp/compare_default_vs_fric_off_90f.json`
- `default vs friction_on`: `/tmp/compare_default_vs_fric_on_90f.json`

Observed:
- Both are bitwise-equivalent at metrics level:
  - `first_divergence_frame_by_threshold = null`
  - `com_err_l2_max = 0`
  - `y_gap_diff_abs_max = 0`

Conclusion:
- Current "hover/freeze" symptom is not caused by friction-enable switch at system level.

### 2) CCD / active-set sensitivity isolation (dt halved)

Run:
- half dt: `/tmp/corex_simple_contract_dt005_180f`

Comparison:
- `default vs dt005`: `/tmp/compare_default_vs_dt005.json`

Observed:
- Diverges from frame 2 under frame-index comparison (expected due changed dt discretization).
- Contact timing shifts consistently:
  - default `first_negative_y_gap = 32`
  - dt=0.005 `first_negative_y_gap = 64`

Log evidence near first EE active-set event:
- default first `AFTER_SELECT ... EE=1` at line `5662`.
- dt=0.005 first `AFTER_SELECT ... EE=1` at line `8264`.
- In both runs, near-threshold PT sample remains inactive (`D > 0.0004`), e.g.:
  - default `D=0.00056049 range=(0,0.0004) active=0`
  - dt=0.005 `D=0.00042733038 range=(0,0.0004) active=0`

Conclusion:
- No sign of "premature first-contact fork" in active-set logic from this isolation.
- Timing shift tracks timestep scaling, not an obvious CoreX-only active-set bug.

### 3) d_hat sensitivity (same code, single variable)

Runs (90f):
- `d_hat=0.01`: `/tmp/corex_simple_dhat001_90f`
- `d_hat=0.02` (current default): `/tmp/corex_simple_contract_default_env_90f`
- `d_hat=0.03`: `/tmp/corex_simple_dhat003_90f`
- `d_hat=0.04`: `/tmp/corex_simple_dhat004_90f`
- `d_hat=0.05`: `/tmp/corex_simple_dhat005_90f`

Observed trend:
- Increasing `d_hat` reduces contact-depth trend metric (`min_y_gap`):
  - `0.01 -> -0.2153`
  - `0.02 -> -0.1723`
  - `0.03 -> -0.1193`
  - `0.04 -> -0.1027`
  - `0.05 -> -0.0900`
- But larger `d_hat` also suppresses post-contact motion magnitude (`COM delta 33->89` shrinks).

Interpretation:
- There is a clear trade-off between stronger early contact activation and visible post-contact movement.
- For current float/CoreX pipeline, `d_hat=0.03` appears to be a practical compromise point for this scene.
- Applied in code: CoreX `simple` default `d_hat` is updated from `0.02` to `0.03` in `apps/examples/corex_demo/main.cpp`.

## Next actions (without NVIDIA prerequisite)

1. Continue ABD pullback/dynamics equivalence narrowing against `/root/src` with focused instrumentation around:
   - `J^T g` and `J^T H J` assembly paths,
   - body mass/inertia retrieval,
   - per-step `q`, `q_prev`, `dq` consistency.
2. Run final local gate (non-blocking NVIDIA):
   - first divergence,
   - no penetration,
   - post-contact sustained motion.
3. Keep `export_nvidia_reference_contract.sh` as optional enhancement path if NVIDIA resources become available in future.

## 2026-04-15 follow-up: runtime recovery and active-set diagnostics

### A) Recovered iteration speed by disabling default high-frequency traces

Change:
- `src/backends/cuda/collision_detection/filters/stackless_bvh_simplex_trajectory_filter.cu`
  - Gate heavy `corex_trace` logs behind explicit env switches:
    - `UIPC_COREX_TRACE_SIMPLEX_FILTER`
    - `UIPC_COREX_TRACE_FILTER_ACTIVE_DIAG`
  - Default run now keeps diagnostics off unless explicitly requested.

Validation:
- 90f run (default settings): `/tmp/corex_simple_postlogfix_90f`
- Log: `/tmp/corex_simple_postlogfix_90f.log`
- Runtime recovered to normal iteration speed (4.3s for 90f on this server), no timeout.
- No `Newton Iteration Exits with Max Iteration` found in this run.

### B) PT/PE near-threshold diagnosis

Diagnostic run:
- `/tmp/corex_simple_diag_active40f` with `UIPC_COREX_TRACE_FILTER_ACTIVE_DIAG=1`

Observed evidence:
- PT host diagnostics repeatedly show near-threshold misses, e.g.:
  - `D=0.00091964804`, `range=(0,0.0009)`, `active=0`
- `AFTER_SELECT` around first contact alternates between:
  - `PP=0 PE=4 PT=0 EE=1` (short window),
  - then back to `PP=0 PE=0 PT=0 EE=1`.
- This indicates unstable PT/PE participation near the activation upper bound, while EE remains persistent.

Experiment note:
- A temporary CoreX-only PT/PE tolerance patch was tested and then reverted, because 90f metrics were bitwise-equivalent to baseline (no physical improvement).

### C) Current conclusion at this checkpoint

- Effective retained change: trace gating only (performance/iteration recovery, no physics semantics change by default).
- Physical behavior remains equivalent to prior `d_hat=0.03` baseline:
  - compare: `/tmp/compare_newdefault_vs_postlogfix_90f.json`
  - `first_divergence_frame_by_threshold = null`, key metrics unchanged.
- Root issue is not solved by broad tolerance widening; next work should target why first-contact PT/PE distances remain slightly above `d_hat` window under CoreX/float while EE stays active.

## 2026-04-15 follow-up: execute `simple` first-contact fix plan

### 1) Frozen first-contact input contract (one-shot diagnostics)

Change:
- `src/backends/cuda/collision_detection/filters/stackless_bvh_simplex_trajectory_filter.cu`
  - Under `UIPC_COREX_TRACE_FILTER_ACTIVE_DIAG=1`, add one-shot structured logs when first contact appears:
    - `[corex_trace][first_contact_contract] selected ...`
    - `[corex_trace][first_contact_contract][PT] ...`
    - `[corex_trace][first_contact_contract][PE_selected] ...`
  - Logged fields include:
    - raw source values `thicknesses[]` / `d_hats[]`,
    - `flag`,
    - `D`,
    - `range=(lower, upper)`,
    - degeneracy `dim` + `offsets`.

Evidence:
- Diagnostic run: `/tmp/corex_simple_contractdiag2_40f`
- Log: `/tmp/corex_simple_contractdiag2_40f.log`
- First-contact snapshot:
  - `selected PP=0 PE=4 PT=0 EE=1 candPE=0 candPT=6`
  - active PT candidates are mostly `dim=3` (degenerated to PE), and selected PE entries are duplicated (`pe=(4,1,2)` repeated).

### 2) Vertex attribute fill-chain audit (`thickness/d_hat`)

Compared against `/root/src`:
- `src/backends/cuda/global_geometry/global_vertex_manager.cu`
- `src/backends/cuda/finite_element/finite_element_vertex_reporter.cu`
- `src/backends/cuda/affine_body/affine_body_vertex_reporter.cu`

Conclusion:
- No physics-significant fill-order semantic drift found in `thicknesses`/`d_hats` pipeline.
- CoreX differences are primarily implementation-path differences (host copy/kernels, debug logs), not value semantics.
- So this round does **not** keep any new `thickness/d_hat` logic fork.

### 3) Degenerate routing recheck result

Observed from one-shot contract logs:
- PT/PE instability at first contact is dominated by degeneracy routing and duplication behavior:
  - PT candidates frequently become `dim=3` (PE) rather than `dim=4` (PT),
  - selected PE set includes repeated identical tuples in the first-contact window.
- This confirms the issue is not a simple `is_active_D` threshold-only problem.

### 4) Gate results for this round

- 30f diagnostic gate:
  - `/tmp/corex_simple_contractdiag2_40f` (contract logs captured as expected).
- 90f regression gate:
  - run: `/tmp/corex_simple_contractdiag_90f`
  - metrics: `/tmp/corex_simple_contractdiag_90f_metrics.json`
  - compare vs post-logfix baseline: `/tmp/compare_postlogfix_vs_contractdiag_90f.json`
  - result: bitwise-equivalent metrics (`first_divergence_frame_by_threshold = null`).

Checkpoint summary:
- Retained code change this round is diagnostic-only (default-off).
- No ineffective physics tweak is kept.
- Next actionable direction is to address first-contact PE duplication / routing stability with minimal upstream-compatible semantics.

## 2026-04-15 follow-up: active-set stability fix for PT->PE boundary

### A) Attempted PE dedup (reverted)

Attempt:
- Added device-side PE dedup in `filter_active`.

Result:
- Duplicate PE in the first-contact frame was removed.
- But 90f regression worsened depth trend and diverged earlier from baseline.
- This change was reverted (not kept).

### B) Kept fix: PT->PE near-threshold hysteresis (dim==3 only)

Change:
- `src/backends/cuda/collision_detection/filters/stackless_bvh_simplex_trajectory_filter.cu`
- In PT active filtering (both CoreX kernel path and non-CoreX lambda path):
  - compute `dim = degenerate_point_triangle(flag, offsets)` first,
  - keep default `is_active_D(range, D)`,
  - only when `dim==3` (PT degenerates to PE) and strict active fails, allow a tiny upper slack:
    - `slack = max(1e-6, 0.08 * (range.y - range.x))`
    - active if `D > range.x && D < range.y + slack`.

Rationale:
- Target only the observed unstable branch (`PT -> PE`) near first-contact threshold.
- Avoid broad global active-set expansion.

### C) Validation

Diagnostic run:
- `/tmp/corex_simple_ptpe_hyst_diag40f`
- Log: `/tmp/corex_simple_ptpe_hyst_diag40f.log`

Observed:
- Around first-contact window:
  - previous behavior: `PE=4 -> PE=0` quickly,
  - new behavior: `PE=4 -> PE=2 -> PE=0` (improved persistence window).

90f regression:
- run: `/tmp/corex_simple_ptpe_hyst_90f`
- metrics: `/tmp/corex_simple_ptpe_hyst_90f_metrics.json`
- compare: `/tmp/compare_postlogfix_vs_ptpe_hyst_90f.json`

Key deltas vs post-logfix baseline:
- `first_divergence_frame_by_threshold`: `34` (expected after behavior change near contact).
- `first_negative_y_gap`: unchanged at `32`.
- `min_y_gap`: improved from `-0.11928964` to `-0.11666200` (shallower depth trend).
- Contact count trend still not ideal (many `PE=0` frames remain), but PT->PE dropout severity is reduced.

Checkpoint:
- Kept change is minimal and branch-targeted.
- Remaining issue is partial: active-set still tends to fall back to EE-dominant regime after contact window.

### D) Additional trial (reverted): extend hysteresis to EE->PE path

Attempt:
- Applied the same dim==3 hysteresis to EE active filtering.

Outcome:
- Reduced `min_y_gap` further but also significantly suppressed post-contact COM drift
  (toward hover/stick behavior), and introduced occasional max-iteration exits in diagnostics.
- Reverted this extension; final retained fix is PT->PE-only hysteresis.

### E) Second-stage tuning: PT->PE hysteresis scale sweep

Change:
- Added `UIPC_COREX_PTPE_HYST_SCALE` env override (default remains `0.08`) for PT->PE hysteresis scale.

Runs:
- default (`0.08`): `/tmp/corex_simple_current_90f`
- `0.12`: `/tmp/corex_simple_hyst012_90f`
- `0.16`: `/tmp/corex_simple_hyst016_90f`

Comparisons:
- `/tmp/compare_postlogfix_vs_current_90f.json`
- `/tmp/compare_postlogfix_vs_hyst012_90f.json`
- `/tmp/compare_postlogfix_vs_hyst016_90f.json`

Observed:
- Larger hysteresis scale further reduces depth trend (`min_y_gap` less negative), but also suppresses post-contact COM drift significantly (moves toward hover/stick behavior).
- `0.12` and `0.16` both showed one `Newton Iteration Exits with Max Iteration` in 90f logs.
- `0.08` remains the best local compromise in this phase:
  - improved depth trend vs postlogfix baseline,
  - no max-iteration exit in the retained current run,
  - less motion suppression than larger scales.
