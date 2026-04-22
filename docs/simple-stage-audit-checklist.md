# Simple Stage Audit Checklist

Use this checklist to find the **first** mismatch against reference without relying on final visualization.

## Stage 0: Run Identity

- [ ] `config.json` matches expected profile for this test
- [ ] `systems.json` exists and includes expected systems
- [ ] `run.log` captured for the full run

## Stage 1: Geometry / State Initialization

- [ ] Frame 0 OBJ exists and vertex count is valid
- [ ] Initial COM/normal from `metrics.json` matches reference trend
- [ ] No NaN/Inf in frame 0 positions

## Stage 2: Candidate / Active Set

- [ ] `detect` has non-zero contact candidates in contact window
- [ ] `filter_active` remains non-zero during contact phases
- [ ] No regression to all-zero mask behavior

## Stage 3: Contact / DyTopo Assembly

- [ ] Contact gradient/hessian path runs without NaN/Inf
- [ ] Fixed-dynamic pairs do not erase all dynamic curvature
- [ ] Contact types do not collapse to a single degenerate mode

## Stage 4: Linear System / Solve

- [ ] Solver converges without repeated hard line-search fallback
- [ ] Rotational increments are non-trivial post-contact
- [ ] No persistent max-iteration failure pattern

## Stage 5: Line Search / Step Acceptance

- [ ] Acceptance criterion is stable under float noise
- [ ] `alpha` is not consistently collapsed to near-zero post-contact
- [ ] Energy log shows monotone-safe behavior without runaway

## Stage 6: State Update / Physical Motion

- [ ] Quick run (`30f`): no crash, no obvious instability
- [ ] Non-quick run (`90f`): sustained post-contact motion
- [ ] COM and attitude trend remain physically plausible across `60~90`

## Final Acceptance (Simple 90f)

- [ ] `frame_count == 90`
- [ ] `tetra_overlap_frame_count == 0`
- [ ] Active-set/candidate chain remains alive
- [ ] COM + orientation continue evolving after contact
- [ ] Departure trajectory does not show frozen-angle artifact
