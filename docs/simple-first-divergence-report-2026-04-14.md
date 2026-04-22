# Simple First-Divergence Report (2026-04-14)

## Purpose

Record the first observable divergence between the current CoreX float run and the available frozen reference baseline using standardized artifacts.

## Inputs

- Reference baseline (provisional):  
  `output/examples/corex_demo/parity_baselines/ref_corex_legacy_verify_simple_120f_v2`
- Current baseline:  
  `output/examples/corex_demo/parity_baselines/cur_corex_float_simple_final_90f`
- Comparison report:  
  `output/examples/corex_demo/parity_baselines/compare_ref_legacy_vs_cur90.json`

## Result

The first divergence appears at **frame 0** (`first_divergence_frame_by_threshold = 0`).

Primary evidence:

- Large COM mismatch from the first frame.
- Large orientation mismatch from the first frame.
- Contact timing mismatch (`reference_first_negative_y_gap = 65`, `current_first_negative_y_gap = 32`).

## Interpretation

This is not a late-stage numeric drift. The two runs are initialized under materially different scene/config states, so parity is already broken before contact dynamics.

Likely earliest divergence stage: **Stage 0/1 (run identity + initialization)**:

- Config profile mismatch between reference and current run.
- Different historical scene setup in legacy baseline.

## Action Taken

- Standard baseline protocol and tooling were added:
  - `tools/simple_physics_audit/freeze_baseline.py`
  - `tools/simple_physics_audit/simple_metrics.py`
  - `tools/simple_physics_audit/compare_runs.py`
- Checklist documented in:
  - `docs/simple-stage-audit-checklist.md`
  - `docs/simple-physics-baseline-protocol.md`

## Next Required Step for True Parity

Export a **NVIDIA/upstream simple baseline** with the same scenario contract and artifact layout, then re-run comparison.  
Current legacy baseline is useful for regression tracking but remains **provisional**, not final parity truth.
