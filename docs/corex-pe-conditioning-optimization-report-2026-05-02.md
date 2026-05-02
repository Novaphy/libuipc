# CoreX PE Conditioning Optimization Report 2026-05-02

## Scope

This round targeted the PE contact conditioning path only. The default runtime path is unchanged.

Implemented artifacts:

- Extended `tools/corex_spd_pcg_aggregate.py` with PE-specific rankings by max ratio, weighted ratio signal, PCG iteration, and 10-frame windows.
- Added opt-in PE-only diagonal regularization before `make_spd(H)` in `ipc_simplex_normal_contact.cu`.
- New switch: `UIPC_COREX_PE_DIAG_REG=<float>`.
- Aggregated artifacts:
  - `/tmp/corex_pe_conditioning_20260502/pe_outliers.json`
  - `/tmp/corex_pe_conditioning_20260502/pe_outliers.csv`

## Candidate

The implemented numerical candidate is PE-only diagonal regularization:

- PE Hessian path only.
- Applied before `make_spd(H)`.
- Only active when `UIPC_COREX_PE_DIAG_REG` is positive.
- Tested value: `0.001`.

The intent was to reduce PE SPD projection spikes without weakening the SPD projection correction itself.

## Validation Notes

`gpu0` stalled during `simple90` initialization, stopping near backend system construction. Per operator request, all validation below was rerun on `gpu1`.

Build:

- `cmake --build build_corex --target corex_demo --config Release -j8`
- Result: passed.

Correctness gates on `gpu1` with `UIPC_COREX_PE_DIAG_REG=0.001`:

- `simple90`: passed, 4s.
- `simple300`: passed, 11s.
- `stack120`: passed, 6s.
- Failure scan found no NaN, exception, assert, abort, or reached-max-iter marker. The only matches were benign `no error` CUDA probe lines and `SanityCheck Summary: 0 errors`.

Short A/B on `gpu1` used:

- `UIPC_COREX_CONTACT_SPD_DIAG=1`
- `UIPC_COREX_PCG_COST_DIAG=1`
- `UIPC_COREX_PCG_SCALAR_AUDIT=1`
- `UIPC_COREX_PHASE_PROFILE=1`

## Short A/B Results

`wb80`:

- Default: wall 31s, PCG iter sum 10859, PCG iter max 72, PE weighted ratio 0.225715, PE ratio max 8.793927.
- `PE_DIAG_REG=0.001`: wall 30s, PCG iter sum 10922, PCG iter max 68, PE weighted ratio 0.216302, PE ratio max 31.358198.
- Result: mixed. Wall and max iter improved slightly, but total PCG iter regressed and PE max ratio produced a larger spike.

`wb150`:

- Default: wall 102s, PCG iter sum 56073, PCG iter max 174, PE weighted ratio 0.124707, PE ratio max 13.268469.
- `PE_DIAG_REG=0.001`: wall 104s, PCG iter sum 58444, PCG iter max 180, PE weighted ratio 0.137758, PE ratio max 8.219514.
- Result: failed. Wall time, total PCG iterations, and max PCG iterations all regressed.

Because `wb150` failed the gate, `wb400` was not run.

## PE Outlier Map Findings

The enhanced aggregation confirms the important distinction between isolated ratio spikes and broad late-frame PE pressure:

- In `wb150 default`, the highest PE ratio row was frame 53/newton 1 with PE ratio max 13.268469, but it only had PCG iter 37 and PE count 96. This is a narrow spike, not the dominant high-iteration source.
- The dominant high-iteration region is the late frame window `140-149`. In `wb150 default`, this window had PCG iter sum 23081, PCG iter max 174, PE count sum 2480358, PE weighted ratio 0.148346, and PE ratio max 7.060968.
- With `PE_DIAG_REG=0.001`, the same late window worsened to PCG iter sum 25152 and max 180, with PE weighted ratio 0.180032.
- High PCG rows in `wb150` remain PE-led after regularization. Examples include frame 148/newton 1 with 180 iterations and frame 149/newton 1 with 177 iterations in the diag-reg run.

Conclusion: the major cost is not just a few extreme PE SPD projection spikes. It is more consistent with a broad late-frame PE contact population with medium correction pressure. The tested diagonal regularization does not improve that spectrum; it can reduce some max-ratio values while still worsening solver iterations.

## Defaulting Decision

Do not default `UIPC_COREX_PE_DIAG_REG`.

Keep it as an opt-in diagnostic/numerical experiment only. The default path remains unchanged.

## Next Step

The next PE-focused candidate should avoid uniformly perturbing all PE Hessians. A better next experiment is a gated PE correction clamp or adaptive path that triggers only on the late-frame PE pressure signature:

- Trigger on PE count and weighted correction ratio, not only `PE_corr_diag_ratio_max`.
- Apply only when the PE population is large enough to match the high-iteration windows.
- A direct Hessian clamp remains risky because it may break SPD, so the safer next candidate is an adaptive precondition trigger keyed by PE weighted ratio windows.

If that fails as well, the evidence supports moving the next optimization round to `abd_assemble.dytopo_effect`, while keeping PE outlier mapping as a diagnostic signal.
