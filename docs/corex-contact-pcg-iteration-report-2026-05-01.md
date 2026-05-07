# CoreX Contact/PCG Iteration Optimization Report

Date: 2026-05-01

## Summary

This round followed the `corex-contact-pcg-iteration` plan and focused on the remaining contact candidate scale and the high CoreX PCG iteration count.

The final default path remains conservative:

- `UIPC_COREX_CONTACT_EARLY_ACTIVE_FILTER` was added as an opt-in experiment for PT/EE BVH query filtering, but it is not recommended as a default.
- The damped ABD 12x12 block-inverse preconditioner experiment failed the `simple90` correctness gate and was removed from the source path.
- The validated default path completed `simple90`, `simple300`, `stack120`, `wb80`, `wb150`, `wb400`, `domino600`, and `wb800` without NaN or process failure.

## Implemented Changes

### Opt-in PT/EE early active filter

File:

- `src/backends/cuda/collision_detection/filters/stackless_bvh_simplex_trajectory_filter.cu`

New opt-in switches:

- `UIPC_COREX_CONTACT_EARLY_ACTIVE_FILTER=1`
- `UIPC_COREX_CONTACT_EARLY_ACTIVE_SCALE=<float>`, default `4.0`
- `UIPC_COREX_CONTACT_EARLY_ACTIVE_STATS=1`

The filter is only applied to PT and EE query predicates. It first keeps the existing CCD broadphase test, then optionally computes current and predicted squared distances. A candidate is dropped only when both endpoints are still farther than `D_range.y() * scale`.

The default path is unchanged because `UIPC_COREX_CONTACT_EARLY_ACTIVE_FILTER` is off by default.

### Contact candidate statistics

`filter_active` now logs an opt-in summary:

```text
[corex_contact_early_stats] PP_cands=... CodimPE_cands=... PT_cands=... EE_cands=... selected_PP=... selected_PE=... selected_PT=... selected_EE=...
```

This provides direct comparison between candidate count and selected contact output.

### Damped block inverse experiment

The plan allowed a small ABD 12x12 block-inverse preconditioner experiment only if diagnostics justified it. A local opt-in implementation was tested with `UIPC_COREX_ABD_PRECOND_BLOCK_INV_DAMPING=0.001`.

Result: failed `simple90` at frame 43 with PCG NaN:

```text
Assertion false failed. Frame 43, Newton 2, PCG Iter 35:
r^T*z = nan, norm(r) = nan, norm(z) = nan.
```

Before NaN, the solver repeatedly hit `max_iter=48` on a 24-DoF simple system. This means the block inverse is unstable even on the smallest correctness gate. The experiment was abandoned and removed from the source path.

## Experiment Results

### Early active filter A/B

`wb80` default:

- Wall time: `30s`
- PCG total iterations: `10734`
- PCG max per solve: `69`
- Newton total: `243`
- Max unique triplets: `14998`
- Max simplex candidates: `22102`

`wb80`, early filter with default scale `4`:

- Wall time: `33s`
- PCG total iterations: `14453`
- PCG max per solve: `469`
- Newton total: `259`
- Max unique triplets: `15014`
- Max simplex candidates: `20462`

`wb80`, early filter with scale `1`:

- Wall time: `32s`
- PCG total iterations: `17732`
- PCG max per solve: `181`
- Newton total: `243`
- Max unique triplets: `14934`
- Max simplex candidates: `20424`

Conclusion:

- The filter reduces raw PT/EE candidates in several windows.
- It does not reduce final matrix size enough to help.
- It changes contact selection timing and worsens PCG convergence.
- It should remain an opt-in diagnostic switch, not a default.

## PCG Diagnosis

NVIDIA `wb400` reference from `/root/SUMMARY.md`:

- Wall time: `45.61s`
- PCG total iterations: `38066`
- PCG calls: `2647`
- Mean PCG iterations: `14.38`
- Newton total: `2248`

CoreX default `wb400` in this round:

- Wall time: `323.08s`
- PCG total iterations: `172204`
- PCG calls: `2678`
- Mean PCG iterations: `64.30`
- Newton total: `1982`
- Max unique triplets: `30214`
- Max simplex candidates: `22388`

Ratios:

- Wall time ratio vs NVIDIA: `323.08 / 45.61 = 7.08x`
- PCG total iteration ratio: `172204 / 38066 = 4.52x`
- PCG calls are nearly equal: `2678 / 2647 = 1.01x`
- Newton total is lower on CoreX in this run: `1982 / 2248 = 0.88x`

Interpretation:

- The gap is not caused by more Newton iterations.
- PCG is called about the same number of times as NVIDIA, but each solve takes far more iterations.
- Since CoreX's time ratio is still higher than the PCG iteration ratio, both solver conditioning and per-iteration GPU cost still matter.
- Contact filtering alone is risky: reducing candidates can make convergence worse if it changes active-set timing.

## Regression Results

Short validation:

- `simple90`: pass, `4.67s`
- `simple300`: pass, `11.28s`
- `stack120`: pass, `6.07s`, PCG total `2835`, max PCG `44`
- `wb80`: pass, `29.71s`, PCG total `10795`, max PCG `70`
- `wb150`: pass, `83.85s`, PCG total `47001`, max PCG `177`

Long validation:

- `wb400`: pass, `323.08s`, PCG total `172204`, max PCG `182`
- `domino600`: pass, `65.07s`, PCG total `80291`, max PCG `129`
- `wb800`: pass, `629.96s`, PCG total `333041`, max PCG `201`

No `NaN`, `Exception`, process error, or `reached max_iter` warning was found in the default regression logs.

## Defaulting Recommendation

Do not default the early active filter.

It reduces some PT/EE candidates but worsens PCG iteration count and wall time on `wb80`. This indicates contact timing/active-set changes can harm the linear system even when candidate count goes down.

Do not pursue the damped ABD block inverse in its current form.

It fails the smallest simple gate with PCG NaN, so it is not a safe optimization path.

Keep the current default path:

- AllPE remains removed/off.
- Matrix converter linear reduce remains default.
- PCG `rz/norm` fused reduction remains default.
- ABD preconditioner remains Jacobi with diagnostics only.

## Next Optimization Direction

The remaining performance gap is more likely in:

- A safer preconditioner design than direct 12x12 inverse, possibly structured block Jacobi that preserves positive definiteness.
- PCG per-iteration cost (`spmv`, `spmv_sync`, `preconditioner`, `dotnorm`) after the active-set/conditioning issue is understood.
- Contact generation strategies that preserve final active-set equivalence, rather than simply dropping candidates before `filter_active`.

Raw logs and profile summaries for this round are under:

```text
/tmp/corex_contact_pcg_regress_20260501/
```
