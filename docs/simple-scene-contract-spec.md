# Simple Scene Contract Spec

This document defines the canonical `simple` scene contract for final parity signoff.

## Canonical Contract (current target)

Source: [`/root/libuipc/apps/examples/corex_demo/main.cpp`](/root/libuipc/apps/examples/corex_demo/main.cpp)

- **Scene name:** `simple`
- **Time step:** `dt = 0.01`
- **Gravity:** `(0, -9.8, 0)`
- **Contact:** enabled
- **Friction system:** enabled
- **`d_hat` (simple override):** `0.02`
- **Linear solver:** `linear_pcg`
- **Linear tolerance rate:** `1e-3`
- **Sanity check:** enabled
- **Contact model (simple):** `default_model(mu=0.0, kappa=30.0_GPa)`
- **Object grouping:** two objects
  - `tets_falling`
  - `tets_fixed`
- **Contact element propagation:** explicit `contact_element_id` propagation to vertices.

## Legacy Contract Differences (non-signoff)

Source: [`/root/libuipc4.13/apps/examples/corex_demo/main.cpp`](/root/libuipc4.13/apps/examples/corex_demo/main.cpp)

- `simple` friction disabled
- `simple` sanity check disabled
- contact model `default_model(mu=0.5, kappa=1.0_GPa)`
- single object `tets` containing both geometries
- older linear tolerance `1e-2`

These differences change the physical contract and make legacy baselines unsuitable as final truth for current signoff.

## Baseline Validity Rules

- `corex-current` baselines are valid for **A/B diagnosis** under the same contract.
- `corex-legacy` baselines are **provisional only**.
- Final signoff requires a same-contract `nvidia` or `upstream` baseline artifact set:
  - `baseline_manifest.json`
  - `config.json`
  - `systems.json`
  - `run.log`
  - `frames/scene_surface_*.obj`
  - `metrics.json`
