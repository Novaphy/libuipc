# Simple Physics Baseline Protocol

This protocol standardizes how `simple` reference runs are frozen and compared.

## Scope

- Primary scene: `simple`
- Quick regression: `30` frames
- Non-quick regression: `90` frames
- Truth priority: NVIDIA/upstream run (when available)
- Temporary fallback: legacy CoreX run with explicit `provisional` label

## Baseline Artifact Layout

Each baseline lives under:

- `output/examples/corex_demo/parity_baselines/<baseline-name>/`

Required files:

- `baseline_manifest.json`
- `config.json`
- `systems.json`
- `run.log`
- `frames/scene_surface_*.obj`
- `metrics.json`

## Freeze Commands

Freeze run directory:

```bash
python3 tools/simple_physics_audit/freeze_baseline.py \
  --run-dir <run-dir> \
  --baseline-root output/examples/corex_demo/parity_baselines \
  --baseline-name <name> \
  --label "<label>" \
  --source-type nvidia|upstream|corex-legacy|corex-current \
  --notes "<notes>"
```

Extract metrics:

```bash
python3 tools/simple_physics_audit/simple_metrics.py \
  --frames-dir output/examples/corex_demo/parity_baselines/<name>/frames \
  --output output/examples/corex_demo/parity_baselines/<name>/metrics.json
```

Compare two runs:

```bash
python3 tools/simple_physics_audit/compare_runs.py \
  --reference <ref-metrics.json> \
  --current <cur-metrics.json> \
  --output output/examples/corex_demo/parity_baselines/<report>.json
```

## Interpretation Rules

- `y_gap` is not a sole penetration criterion.
- Penetration proxy uses tetra containment (`tetra_overlap_proxy`).
- First divergence is reported by multi-signal threshold:
  - COM error
  - normal-angle error
  - `y_gap` difference
- `provisional` references must not be used as final parity sign-off.
