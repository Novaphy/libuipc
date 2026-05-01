# CoreX v15 Conservative Prune Report

## Summary

This round keeps the validated v14 default path as the production path and removes stale experimental branches that were either slower, unsafe, or no longer used by the 308s execution route.

- MatrixConverter now uses the linear segment reduce path only. The old `UIPC_COREX_MATCONV_SCAN_REDUCE` scan fallback was removed.
- Contact detection/filtering no longer contains the independent AllP-AllE / AllPE channel or its full/fallback/post-processing modes.
- `LinearPCG` no longer contains the failed fused `pAp` opt-in or norm-check interval experiment. It keeps the stable per-iteration `norm(r)` convergence check.
- `corex_demo` no longer exposes the solver/check-interval override switches; it keeps `linear_pcg` and the natural domino defaults.

The following diagnostic or compatibility paths were intentionally kept: `UIPC_COREX_PHASE_PROFILE`, CoreX trace switches, ABD fallback/trace paths, dot/norm device fallback behavior, TOI device-min opt-in, SpMV grouped-row opt-in, and `UIPC_COREX_MATCONV_ASYNC`.

## Removed Switches

- `UIPC_COREX_MATCONV_SCAN_REDUCE`
- `UIPC_COREX_CONTACT_ALLPE_MODE`
- `UIPC_COREX_ALLPE_FALLBACK_ONLY`
- `UIPC_COREX_ALLPE_POINTWISE_DEDUP`
- `UIPC_COREX_ALLPE_FILL_MISSING_ONLY`
- `UIPC_COREX_ALLPE_DETECT_DIM3_ONLY`
- `UIPC_COREX_ALLPE_SUPPRESS_DEGENERATE_PE`
- `UIPC_COREX_PCG_FUSED_PAP`
- `UIPC_COREX_PCG_NORM_CHECK_INTERVAL`
- `UIPC_COREX_LINEAR_SOLVER`
- `UIPC_COREX_LINEAR_CHECK_INTERVAL`

## Validation

All runs used GPU1 (`CUDA_VISIBLE_DEVICES=1`) with the rebuilt `corex_demo`.

| Scene | Frames | Result | Wall Time |
| --- | ---: | --- | ---: |
| `simple` | 90 | exit 0 | smoke gate |
| `simple` | 300 | exit 0 | smoke gate |
| `stack` | 120 | exit 0 | smoke gate |
| `simple` | 200 | exit 0 | 7s |
| `slope` | 200 | exit 0 | 12s |
| `stack` | 200 | exit 0 | 9s |
| `domino` | 300 | exit 0 | 44s |
| `wrecking_ball` | 400 | exit 0 | 319s |
| `wrecking_ball` | 800 | exit 0 | 621s |

`wb800` reached frame 799 and shut down cleanly:

```text
[cuda] <<< End Frame: 799
[corex_demo] frame 799 timings: advance=649ms sync=0ms retrieve=0ms write_obj=31ms
Cuda Backend Shutdown Success.
```

The `wb400/wb800` wall times are close to the v14 default range while removing unused branches. No IDE linter diagnostics were reported for the edited files.

## Bundle Notes

The v15 bundle should be generated from `/root/corex-bundle-v13-work-base`, which already contains the v14 commits. Suggested split:

- `corex(prune): remove stale matconv/contact experiments`
- `corex(pcg): drop unsafe scalar reduction experiments`
- `corex(docs): report v15 conservative prune validation`
