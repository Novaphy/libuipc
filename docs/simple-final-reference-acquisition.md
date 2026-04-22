# Simple Final Reference Acquisition

This checklist is for generating the final same-contract reference baseline on an NVIDIA/upstream environment.

## Prerequisites

- Build and runtime environment that can execute the CUDA/upstream path with NVIDIA GPU.
- Same repository state for scene contract definition:
  - [`/root/libuipc/apps/examples/corex_demo/main.cpp`](/root/libuipc/apps/examples/corex_demo/main.cpp)
  - [`/root/libuipc/docs/simple-scene-contract-spec.md`](/root/libuipc/docs/simple-scene-contract-spec.md)

## Required Run Commands

Use the same invocation style as current CoreX runs and export both 30f and 90f:

```bash
timeout 300s ./Release/bin/corex_demo --backend cuda --scene simple --frames 30 --gpu 0 --output_dir /tmp/simple_ref_nvidia_30f > /tmp/simple_ref_nvidia_30f.log 2>&1
timeout 300s ./Release/bin/corex_demo --backend cuda --scene simple --frames 90 --gpu 0 --output_dir /tmp/simple_ref_nvidia_90f > /tmp/simple_ref_nvidia_90f.log 2>&1
```

## Freeze and Metrics

```bash
python3 tools/simple_physics_audit/freeze_baseline.py \
  --run-dir /tmp/simple_ref_nvidia_90f \
  --baseline-root output/examples/corex_demo/parity_baselines \
  --baseline-name ref_nvidia_simple_contract_90f \
  --label "NVIDIA simple contract 90f" \
  --source-type nvidia \
  --notes "same-contract final reference for signoff"

python3 tools/simple_physics_audit/simple_metrics.py \
  --frames-dir output/examples/corex_demo/parity_baselines/ref_nvidia_simple_contract_90f/frames \
  --output output/examples/corex_demo/parity_baselines/ref_nvidia_simple_contract_90f/metrics.json
```

## Contract Validation Before Signoff

Verify the reference artifact has:

- `baseline_manifest.json` with `source_type = nvidia` or `upstream`
- `config.json` and `systems.json` matching the canonical contract
- full `frames/scene_surface_*.obj` for 90 frames
- extracted `metrics.json`

## Final Compare Target

All CoreX candidate runs must be compared against:

- `output/examples/corex_demo/parity_baselines/ref_nvidia_simple_contract_90f/metrics.json`

No `corex-legacy` or provisional baseline is allowed for final physical-correctness signoff.
