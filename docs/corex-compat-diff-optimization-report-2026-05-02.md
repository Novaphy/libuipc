# CoreX Compatibility-Diff Optimization Report 2026-05-02

## Scope

This round followed the DyTopo lesson: look for CoreX compatibility paths that diverge from NVIDIA by using serial kernels, host fallback, strong synchronization, or lower-throughput replacement algorithms.

The audit found several candidates:

- SpMV CoreX workaround in `linear_system/spmv.cu`.
- PCG scalar/reduction synchronization in `linear_system/linear_pcg_corex.cu.inc`.
- Matrix converter synchronization in `dytopo_effect_system/global_dytopo_effect_manager.cu`.
- ABD host fallback opt-ins in vertex/BDF1/tolerance paths.

## Baseline After DyTopo Parallel

Using the DyTopo-parallel default path on `gpu1`:

- `wb80`: 18s, PCG sum 10562, PCG max 72.
- `wb150`: 65s, PCG sum 55892, PCG max 171.
- Main hotspots shifted to PCG, contact detect/filter, and matrix conversion. `abd_assemble.dytopo_effect` was no longer dominant.

## Candidate Results

### SpMV grouped-row candidate

Existing opt-in:

- `UIPC_COREX_SPMV_GROUPED_ROW=1`

Result:

- `wb80`: default 18s, grouped 17s.
- `wb150`: default 50s, grouped 50s.
- `wb150` PCG sum regressed from 45280 to 46013.

Decision: keep opt-in only. Do not default.

### PCG reduce2 / SpMV sync options

Tested:

- `UIPC_COREX_PCG_REDUCE2=1`
- `UIPC_COREX_PCG_SKIP_SPMV_SYNC=1`
- both combined

Result:

- `wb150 default`: 63s, PCG sum 55759, max 174.
- `wb150 reduce2`: 50s, PCG sum 46676, max 174.
- `wb150 skip`: 49s, PCG sum 46001, max 180.
- `wb150 both`: 52s, PCG sum 47427, max 174.

Short gate showed a promising signal for `reduce2`, while `skip_spmv_sync` looked riskier because max iteration increased. However, when `reduce2` was temporarily defaulted and run through `wb400`, the long gate was slower than the DyTopo-only default.

Decision: keep `UIPC_COREX_PCG_REDUCE2` opt-in. Do not default.

### Matrix converter async

Tested:

- `UIPC_COREX_MATCONV_ASYNC=1`

Result:

- `wb80`: default 18s, async 18s.
- `wb150`: default 61s, async 50s.
- `dytopo.convert_matrix` dropped from 7359.0 ms to 5590.5 ms in the `wb150` diagnostic run.

This was also promising on short gate, but the combined long-gate default did not beat the DyTopo-only `wb400` result.

Decision: keep `UIPC_COREX_MATCONV_ASYNC` opt-in. Do not default.

### ABD host fallback opt-ins

Tested combined opt-in:

- `UIPC_COREX_ABD_VERTEX_GPU=1`
- `UIPC_COREX_ABD_BDF1_ENERGY_GPU=1`
- `UIPC_COREX_ABD_BDF1_GRADIENT_HESSIAN_GPU=1`
- `UIPC_COREX_ABD_TOLERANCE_GPU=1`
- `UIPC_COREX_TOI_DEVICE_MIN=1`

Result:

- `wb80`: default 18s, opt-in 17s.
- `wb150`: default 63s, opt-in 62s.
- `wb150` PCG sum increased from 55896 to 56543.

Decision: signal is too small and not clean. Keep opt-in only.

## Long Gate Outcome

Temporary defaulting of `PCG_REDUCE2` and `MATCONV_ASYNC` produced:

- `wb400`: passed, 188s, PCG sum 181413, max 177, solves 2755.

This is valid but worse than the previous DyTopo-parallel `wb400` result:

- DyTopo-parallel default: 180s, PCG sum 173559, max 198, solves 2725.

Therefore the temporary defaults were rolled back. The only default optimization retained from these compatibility-diff rounds is the DyTopo parallel assembly path.

Rollback sanity after reverting those defaults:

- `simple90`: passed, 4s.
- `wb80`: passed, 18s.
- Failure scan found no NaN, exception, assert, abort, or reached-max-iter marker.

## Conclusion

The DyTopo serial-kernel issue was a strong structural mismatch and produced a clear default win. The next compatibility differences are more nuanced:

- SpMV grouped-row is not a clean win.
- PCG reduce2 and MATCONV async can improve short runs, but did not pass the `wb400` defaulting bar in this round.
- ABD host fallback opt-ins are too small to justify defaulting.

Recommended next work:

- Treat `UIPC_COREX_PCG_REDUCE2=1` and `UIPC_COREX_MATCONV_ASYNC=1` as useful experiment knobs, not defaults.
- For another real default candidate, build a better matrix converter segment-reduction kernel rather than only removing syncs.
- Continue comparing against NVIDIA algorithm shape, but default only after `wb400` improves.
