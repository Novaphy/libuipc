# CoreX 稳定有效优化汇总

日期：2026-05-05

## 文档目的

本文档汇总 `docs/` 中已经记录、并且具备稳定时间收益或已进入 CoreX 默认路径的有效优化。

这里有意排除只在短跑中变快、但没有通过 `wb400` 长跑稳定门槛的候选，例如后续的 block-level matrix reduce、segmented SpMV、PCG fused SpMV-dot、contact early-active filtering，以及大范围 SPD diagonal boosting。

## 稳定有效优化

### 1. PCG、SpMV、BVH、Matrix 和 Copy 基础热路径清障

来源报告：

- `corex-performance-optimization-work-report-2026-04-25.md`

有效改动：

- 缓存 PCG 的 `norm(b)`，避免每轮迭代重复计算。
- 将 legacy convergence 观察日志放到 trace 开关之后，避免默认运行刷屏。
- 将 CoreX 旧的 row-scan SpMV fallback 替换为 triplet-parallel atomic SpMV。
- 移除 BVH build 多个阶段后的不必要全设备同步。
- 将部分 matrix converter 清零从同步 `cudaMemset` 改为 `cudaMemsetAsync`。
- 恢复低风险 Device-to-Device async copy，H2D/D2H 仍保持保守同步。

验证与收益：

- `simple90/simple300`、`slope200`、`stack200`、`domino300`、`wrecking_ball400` 均完成。
- `wrecking_ball400` 完整跑通，耗时 `526.41s`。
- 报告中说明优化前 `wrecking_ball400` 完整 baseline 没有干净跑完，因此这一轮更适合作为“稳定跑通 + 基础热路径清障”的收益，而不是严格单点加速比。

回退/诊断开关：

- `UIPC_COREX_SPMV_ROW_SCAN=1`
- `UIPC_COREX_TRACE_LINEAR_SYSTEM=1`

### 2. Dot/Norm fallback 与 bbox device reduction 清障

来源报告：

- `corex-host-fallback-kernel-work-report-2026-04-27.md`

有效改动：

- 为 CoreX contiguous float dot/norm 增加 device fallback kernel。
- 将 vertex bounding box 的全量 positions D2H 拷贝替换为设备端 reduction。
- 继续保持 H2D/D2H 同步保守策略，只保留 D2D async。

验证与收益：

- `simple90/simple300`、`stack120` 和短帧 `wrecking_ball` 检查通过。
- `wrecking_ball400` 从 `526.41s` 到 `517.45s`，提升约 `1.7%`。
- 报告判断这属于稳定的清障型收益，不是 `wrecking_ball400` 的主要性能突破。

回退/诊断开关：

- `UIPC_COREX_DOTNORM_HOST_FALLBACK=1`
- `UIPC_COREX_DOTNORM_FORCE_DEVICE_FALLBACK=1`
- `UIPC_COREX_BBOX_HOST_FALLBACK=1`
- `UIPC_COREX_MEMCPY_STATS=1`

### 3. ABD line-search `step_forward` 默认 GPU 化

来源报告：

- `corex-host-fallback-gpu-restoration-rerun-report-2026-04-28.md`

有效改动：

- `ABDLineSearchReporter::step_forward` 从 D2H/CPU/H2D 更新切换为默认 GPU kernel。

验证与收益：

- 其它 ABD GPU fallback 候选虽然通过小场景门禁，但在 `wb400` 中导致 wall、PCG 或 Newton 回退，因此继续保持 opt-in。
- 最终默认组合：
  - `wrecking_ball400`：参考上一轮 `526.41s`，优化后 `502s`
  - 约 `4.6%` 提升
  - PCG sum `166063`，PCG max `174`，Newton sum `2012`

保持 opt-in 的路径：

- `UIPC_COREX_ABD_BDF1_GRADIENT_HESSIAN_GPU=1`
- `UIPC_COREX_ABD_BDF1_ENERGY_GPU=1`
- `UIPC_COREX_ABD_ENERGY_REDUCTION_GPU=1`
- `UIPC_COREX_ABD_TOLERANCE_GPU=1`
- `UIPC_COREX_ABD_VERTEX_GPU=1`
- `UIPC_COREX_ABD_BODY_IOTA_GPU=1`

### 4. MatrixConverter linear reduce

来源报告：

- `corex-matconv-next-opt-implementation-report-2026-04-29.md`
- `corex-pcg-contact-next-opt-implementation-report-2026-04-29.md`
- `corex-v14-default-and-next-bottleneck-report-2026-04-29.md`

有效改动：

- 将 CoreX matrix converter 的扫描式 segment reduce 替换为 input-linear atomic accumulation，用于 `3x3` 和 `3x1` reduce。

验证与收益：

- 早期 `wb150` profiling 显示旧 scan reduce 是主要瓶颈：
  - `matconv_kernel.segmental_reduce_3x3_scan` 约 `35.70s`
- `UIPC_COREX_MATCONV_LINEAR_REDUCE=1` 后：
  - `linear.converter_convert`：`17361.30ms -> 1031.37ms`
  - `dytopo.convert_matrix`：`32986.25ms -> 6743.38ms`
- 后续无 phase profile 的 `wb400` 复验：
  - 默认约 `553s`
  - linear reduce 单独约 `433s`
- v14 将 linear reduce 行为默认化。

回退历史：

- 早期报告保留 `UIPC_COREX_MATCONV_SCAN_REDUCE=1` 作为旧扫描路径回退。
- v15 后续清理中删除了过时 scan fallback 分支。

### 5. 默认关闭独立 AllPE contact 通道

来源报告：

- `corex-pcg-contact-next-opt-implementation-report-2026-04-29.md`
- `corex-v14-default-and-next-bottleneck-report-2026-04-29.md`
- `corex-v15-conservative-prune-report-2026-05-01.md`

有效改动：

- 默认移除 CoreX 额外增加的独立 AllP-AllE / AllPE candidate 通道，使 CoreX contact generation 更接近 NVIDIA 路径。

验证与收益：

- 与 `MATCONV_LINEAR_REDUCE=1` 组合时，`UIPC_COREX_CONTACT_ALLPE_MODE=off` 结果：
  - `wrecking_ball400`：约 `298s`
  - 原默认约 `553s`
  - linear reduce 单独约 `433s`
- correctness gates 通过：
  - `simple90`
  - `simple300`
  - `stack120`
- v14 默认路径：
  - `wrecking_ball400`：`308s`
  - `wrecking_ball800`：`595s`
- `domino600` 也通过，最后仍有 PE 接触，降低了 AllPE-off 漏掉必要 PE 行为的风险。

回退历史：

- v14 曾保留 `UIPC_COREX_CONTACT_ALLPE_MODE=full` 等回退模式。
- v15 在确认保守默认路径后，删除了过时 AllPE 实验分支。

### 6. PCG fused `rz/norm` reduction

来源报告：

- `corex-pcg-next-optimization-report-2026-05-01.md`
- `corex-contact-pcg-iteration-report-2026-05-01.md`

有效改动：

- CoreX 默认使用一个 device reduction kernel 同时计算：
  - `dot(r, z)`
  - `norm2(r)`
- 该改动保留每迭代 `norm(r)` 收敛检查语义，只减少一次独立 scalar reduction 路径。

验证与收益：

- 相比 v15 baseline：
  - `wrecking_ball400`：`319s -> 296s`
  - `wrecking_ball800`：`621s -> 587s`
- 额外默认验证：
  - `simple90/simple300/stack120` 通过
  - `domino600`：`59s`
  - `wrecking_ball800` 完成
- 后续 contact/PCG 报告将其记录为保守默认路径的一部分：
  - AllPE off
  - Matrix converter linear reduce
  - PCG fused `rz/norm`

回退开关：

- `UIPC_COREX_PCG_SEPARATE_RZ_NORM=1`

### 7. `filter_active` 等价去同步

来源报告：

- `corex-spd-pcg-next-report-2026-05-01.md`

有效改动：

- 移除 `StacklessBVHSimplexTrajectoryFilter` 中 PP、CodimPE、PT、EE active filter kernel 之间默认的全设备同步。
- 这些 kernel 写入互不重叠的临时区间，后续同 stream 做 selection，因此 active-set 语义保持等价。

验证与收益：

- 默认完整回归通过：
  - `simple90`：`5s`
  - `simple300`：`11s`
  - `stack120`：`6s`
  - `wrecking_ball80`：`31s`
  - `wrecking_ball150`：`99s`
  - `wrecking_ball400`：`288s`
  - `wrecking_ball800`：`610s`
  - `domino600`：`64s`
- 该轮报告建议只默认这一项同步移除；structured preconditioner 和 SPD/PCG 诊断继续保持 opt-in。

回退开关：

- `UIPC_COREX_FILTER_ACTIVE_SYNC=1`

### 8. ABD DyTopo 并行 assembly

来源报告：

- `corex-dytopo-parallel-optimization-report-2026-05-02.md`
- `corex-compat-diff-optimization-report-2026-05-02.md`

有效改动：

- 将 CoreX ABD DyTopo gradient/hessian assembly 从串行 kernel 替换为并行 kernel：
  - `kernel_abd_dytopo_gradients_parallel`
  - `kernel_abd_dytopo_hessians_parallel`

验证与收益：

- correctness gates 通过：
  - `simple90`：`4s`
  - `simple300`：`11s`
  - `stack120`：`6s`
- phase profile A/B：
  - `wb80 serial`：`30s`
  - `wb80 parallel`：`18s`
  - `wb150 serial`：`118s`
  - `wb150 parallel`：`65s`
- 阶段影响：
  - `wb150 serial` 中 `abd_assemble.dytopo_effect = 43643.84ms`
  - 并行化后该阶段不再是主导 bucket
- 长跑门槛：
  - `wb400 parallel`：`180s`
  - 日志未发现 NaN、exception、assert、abort 或 reached-max-iter marker
- 报告明确将 DyTopo parallel path 默认化。

回退开关：

- `UIPC_COREX_ABD_DYTOPO_SERIAL=1`
- `UIPC_COREX_ABD_DYTOPO_PARALLEL=0`

## 明确排除的非稳定默认候选

以下路径是有价值的实验或诊断，但根据当前报告不能计入“稳定有效默认优化”：

- `UIPC_COREX_MATCONV_ASYNC=1`：短跑有效，但未通过 `wb400`。
- `UIPC_COREX_MATCONV_BLOCK_REDUCE=1`：`wb150` 有收益，`wb400` 回退。
- `UIPC_COREX_SPMV_SEGMENTED_ROW=1`：`wb150` 有收益，`wb400` 回退。
- `UIPC_COREX_PCG_FUSED_SPMV_DOT=1`：减少 sync，但 PCG count 变差。
- `UIPC_COREX_PCG_REDUCE2=1`：短跑信号好，但长跑门槛失败。
- `UIPC_COREX_PCG_SKIP_SPMV_SYNC=1`：短跑有信号，长跑不稳定。
- `UIPC_COREX_TOI_DEVICE_MIN=1`：小规模 reduction 的 kernel launch 成本大于收益。
- ABD BDF1 energy/GH、tolerance、vertex、body-iota GPU 路径：正确性通过，但长跑 wall 或 PCG/Newton 指标不支持默认化。
- Contact early-active filtering：减少部分 raw candidates，但 wall 和 PCG 变差。
- PE diagonal regularization、PE kappa scaling、后续 SPD diagonal boosting：局部或短跑信号不满足稳定 `wb400` / PCG-max 标准。
- Structured/block preconditioner：诊断价值保留，但当前没有稳定默认收益记录。

## 稳定默认路径演进时间线

根据已有报告，大致演进如下：

1. 基础热路径清障后，`wrecking_ball400` 稳定完整跑到约 `526s`。
2. Dot/norm fallback 与 bbox 清障后，`wrecking_ball400` 到约 `517s`。
3. ABD line-search `step_forward` GPU 默认后，`wrecking_ball400` 到约 `502s`。
4. Matrix linear reduce + AllPE off 后，`wrecking_ball400` 到约 `298-308s`。
5. PCG fused `rz/norm` 后，v15 default 从 `319s` 到 `296s`。
6. `filter_active` 等价去同步后，默认回归中 `wrecking_ball400` 约 `288s`。
7. ABD DyTopo 并行 assembly 是后续最明确的大收益，`wrecking_ball400` 约 `180s`。

## 当前稳定优化的共同特征

这些稳定收益主要来自几个模式：

- 消除 CoreX-only 的串行或近似二次复杂度热路径。
- 在 same-stream 顺序足够保证语义时，移除冗余全设备同步。
- 保留 PCG 收敛语义，只减少每迭代 scalar reduction 固定成本。
- 当 CoreX 兼容适配额外引入 contact 或 conversion 工作时，尽量恢复到 NVIDIA 更接近的算法形态。

后续报告已经显示，继续做大范围环境变量 sweep 的收益有限。剩余差距主要与高 PE/EE contact load、matrix row imbalance 和 PCG iteration count 相关，新的默认候选需要围绕这些长跑指标验证，而不能只看 `wb80/wb150` 短跑 wall time。
# CoreX Stable Effective Optimizations Summary

Date: 2026-05-05

## Purpose

This document summarizes the optimization work recorded in `docs/` that has shown
stable time improvement or has been promoted to the CoreX default path after
correctness and long-run validation.

It intentionally excludes candidates that only improved short runs but failed the
`wb400` stability gate, such as later block-level matrix reduce, segmented SpMV,
PCG fused SpMV-dot, contact early-active filtering, and broad SPD diagonal boosting.

## Stable Optimizations

### 1. Core PCG, SpMV, BVH, Matrix, And Copy Cleanup

Source report:

- `corex-performance-optimization-work-report-2026-04-25.md`

Effective changes:

- Cached PCG `norm(b)` and moved legacy convergence observation logs behind trace.
- Replaced the CoreX row-scan SpMV fallback with triplet-parallel atomic SpMV.
- Removed unnecessary device-wide synchronizations from BVH build stages.
- Changed selected matrix-converter clears from synchronous `cudaMemset` to
  `cudaMemsetAsync`.
- Restored low-risk Device-to-Device async copy while keeping H2D/D2H conservative.

Validation and impact:

- `simple90/simple300`, `slope200`, `stack200`, `domino300`, and `wrecking_ball400`
  all completed.
- `wrecking_ball400` completed in `526.41s`.
- The report notes that the pre-optimization full `wrecking_ball400` baseline did
  not complete cleanly, so this is best treated as a stable foundational speedup
  and path cleanup rather than a precise single-run speedup ratio.

Rollback/diagnostic controls:

- `UIPC_COREX_SPMV_ROW_SCAN=1`
- `UIPC_COREX_TRACE_LINEAR_SYSTEM=1`

### 2. Dot/Norm Fallback And BBox Device Reduction Cleanup

Source report:

- `corex-host-fallback-kernel-work-report-2026-04-27.md`

Effective changes:

- Added CoreX device fallback kernels for contiguous float dot/norm.
- Replaced full-position host copy in vertex bounding-box calculation with device
  reduction.
- Kept H2D/D2H copies conservative while preserving D2D async copy.

Validation and impact:

- `simple90/simple300`, `stack120`, and short `wrecking_ball` checks passed.
- `wrecking_ball400`: `526.41s -> 517.45s`, about `1.7%` improvement.
- The report characterizes this as stable cleanup rather than the main
  `wrecking_ball` bottleneck fix.

Rollback/diagnostic controls:

- `UIPC_COREX_DOTNORM_HOST_FALLBACK=1`
- `UIPC_COREX_DOTNORM_FORCE_DEVICE_FALLBACK=1`
- `UIPC_COREX_BBOX_HOST_FALLBACK=1`
- `UIPC_COREX_MEMCPY_STATS=1`

### 3. ABD Line Search `step_forward` GPU Default

Source report:

- `corex-host-fallback-gpu-restoration-rerun-report-2026-04-28.md`

Effective change:

- `ABDLineSearchReporter::step_forward` was moved from D2H/CPU/H2D update to a
  GPU kernel by default.

Validation and impact:

- Other ABD GPU fallback candidates passed small gates but regressed `wb400`, so
  they remained opt-in.
- Final default combination:
  - `wrecking_ball400`: previous `526.41s` reference -> `502s`
  - About `4.6%` improvement.
  - PCG sum `166063`, PCG max `174`, Newton sum `2012`.

Kept opt-in only:

- `UIPC_COREX_ABD_BDF1_GRADIENT_HESSIAN_GPU=1`
- `UIPC_COREX_ABD_BDF1_ENERGY_GPU=1`
- `UIPC_COREX_ABD_ENERGY_REDUCTION_GPU=1`
- `UIPC_COREX_ABD_TOLERANCE_GPU=1`
- `UIPC_COREX_ABD_VERTEX_GPU=1`
- `UIPC_COREX_ABD_BODY_IOTA_GPU=1`

### 4. MatrixConverter Linear Reduce

Source reports:

- `corex-matconv-next-opt-implementation-report-2026-04-29.md`
- `corex-pcg-contact-next-opt-implementation-report-2026-04-29.md`
- `corex-v14-default-and-next-bottleneck-report-2026-04-29.md`

Effective change:

- Replaced the CoreX matrix converter scan-style segment reduce with an
  input-linear atomic accumulation path for `3x3` and `3x1` reductions.

Validation and impact:

- Early `wb150` profiling showed the old scan reduce dominated matrix conversion:
  `matconv_kernel.segmental_reduce_3x3_scan` was about `35.70s`.
- `UIPC_COREX_MATCONV_LINEAR_REDUCE=1` reduced:
  - `linear.converter_convert`: `17361.30ms -> 1031.37ms` on `wb150`
  - `dytopo.convert_matrix`: `32986.25ms -> 6743.38ms` on `wb150`
- Later no-profile `wb400` recheck:
  - default around `553s`
  - linear reduce alone around `433s`
- v14 promoted the linear reduce behavior into the CoreX default path.

Rollback:

- Older reports mention `UIPC_COREX_MATCONV_SCAN_REDUCE=1` as the legacy rollback.
- v15 later pruned the stale scan fallback from the production branch.

### 5. Disable Independent AllPE Contact Channel

Source reports:

- `corex-pcg-contact-next-opt-implementation-report-2026-04-29.md`
- `corex-v14-default-and-next-bottleneck-report-2026-04-29.md`
- `corex-v15-conservative-prune-report-2026-05-01.md`

Effective change:

- Removed the CoreX-only independent AllP-AllE / AllPE candidate channel from the
  default path, matching NVIDIA-style contact generation more closely.

Validation and impact:

- With `MATCONV_LINEAR_REDUCE=1`, setting `UIPC_COREX_CONTACT_ALLPE_MODE=off`
  produced:
  - `wrecking_ball400`: about `298s`
  - default about `553s`
  - linear reduce alone about `433s`
- Correctness gates passed:
  - `simple90`
  - `simple300`
  - `stack120`
- v14 default path:
  - `wrecking_ball400`: `308s`
  - `wrecking_ball800`: `595s`
- `domino600` also passed with PE contacts still present, reducing concern that
  AllPE-off removed required PE behavior.

Rollback/history:

- v14 kept rollback modes such as `UIPC_COREX_CONTACT_ALLPE_MODE=full`.
- v15 removed stale experimental AllPE branches after validating the conservative
  default path.

### 6. PCG Fused `rz/norm` Reduction

Source reports:

- `corex-pcg-next-optimization-report-2026-05-01.md`
- `corex-contact-pcg-iteration-report-2026-05-01.md`

Effective change:

- CoreX now computes `dot(r, z)` and `norm2(r)` in one device reduction kernel.
- This preserves the per-iteration `norm(r)` convergence check while removing one
  separate scalar reduction path per PCG iteration.

Validation and impact:

- Default CoreX improved from the v15 baseline:
  - `wrecking_ball400`: `319s -> 296s`
  - `wrecking_ball800`: `621s -> 587s`
- Additional default validation:
  - `simple90/simple300/stack120` passed.
  - `domino600`: `59s`
  - `wrecking_ball800`: completed.
- Later contact/PCG report records this as part of the conservative default path:
  - AllPE off
  - Matrix converter linear reduce
  - PCG fused `rz/norm`

Rollback:

- `UIPC_COREX_PCG_SEPARATE_RZ_NORM=1`

### 7. Equivalent `filter_active` Synchronization Removal

Source report:

- `corex-spd-pcg-next-report-2026-05-01.md`

Effective change:

- Removed default device-wide synchronizations between equivalent `filter_active`
  kernels in `StacklessBVHSimplexTrajectoryFilter`.
- The kernels write to disjoint temporary ranges and are followed by same-stream
  selection, so active-set semantics are preserved.

Validation and impact:

- Full default regression passed:
  - `simple90`: `5s`
  - `simple300`: `11s`
  - `stack120`: `6s`
  - `wrecking_ball80`: `31s`
  - `wrecking_ball150`: `99s`
  - `wrecking_ball400`: `288s`
  - `wrecking_ball800`: `610s`
  - `domino600`: `64s`
- The report recommends defaulting only this synchronization removal from that
  round, while keeping structured preconditioner and SPD diagnostics opt-in.

Rollback:

- `UIPC_COREX_FILTER_ACTIVE_SYNC=1`

### 8. ABD DyTopo Parallel Assembly

Source reports:

- `corex-dytopo-parallel-optimization-report-2026-05-02.md`
- `corex-compat-diff-optimization-report-2026-05-02.md`

Effective change:

- Replaced CoreX serial ABD DyTopo gradient/hessian assembly kernels with parallel
  kernels:
  - `kernel_abd_dytopo_gradients_parallel`
  - `kernel_abd_dytopo_hessians_parallel`

Validation and impact:

- Correctness gates passed:
  - `simple90`: `4s`
  - `simple300`: `11s`
  - `stack120`: `6s`
- A/B with phase profile:
  - `wb80 serial`: `30s`
  - `wb80 parallel`: `18s`
  - `wb150 serial`: `118s`
  - `wb150 parallel`: `65s`
- Phase impact:
  - `wb150 serial` had `abd_assemble.dytopo_effect = 43643.84ms`.
  - Parallel DyTopo removed this phase from the dominant buckets.
- Long gate:
  - `wb400 parallel`: `180s`
  - no NaN, exception, assert, abort, or reached-max-iter marker.
- The report explicitly defaulted the parallel DyTopo path.

Rollback:

- `UIPC_COREX_ABD_DYTOPO_SERIAL=1`
- `UIPC_COREX_ABD_DYTOPO_PARALLEL=0`

## Explicitly Excluded From Stable-Effective List

The following were useful experiments or diagnostics but should not be counted as
stable effective default optimizations based on the current reports:

- `UIPC_COREX_MATCONV_ASYNC=1`: short-run improvement, did not pass `wb400`.
- `UIPC_COREX_MATCONV_BLOCK_REDUCE=1`: `wb150` improved, `wb400` regressed.
- `UIPC_COREX_SPMV_SEGMENTED_ROW=1`: `wb150` improved, `wb400` regressed.
- `UIPC_COREX_PCG_FUSED_SPMV_DOT=1`: reduced sync but worsened PCG count.
- `UIPC_COREX_PCG_REDUCE2=1`: promising short-run signal, failed long gate.
- `UIPC_COREX_PCG_SKIP_SPMV_SYNC=1`: short-run signal, unstable long-run behavior.
- `UIPC_COREX_TOI_DEVICE_MIN=1`: kernel launch overhead exceeded benefit.
- ABD BDF1 energy/GH, tolerance, vertex, body-iota GPU paths: correctness passed
  but long-run wall or PCG/Newton metrics did not justify defaulting.
- Contact early-active filtering: reduced some raw candidates but worsened PCG and
  wall time.
- PE diagonal regularization, PE kappa scaling, and later SPD diagonal boosting:
  short-run or local signals did not satisfy stable `wb400`/PCG-max criteria.
- Structured/block preconditioner experiments: opt-in diagnostics remain useful,
  but no stable default win is recorded.

## Timeline Of Stable Defaults

Approximate progression from the recorded reports:

1. Foundational cleanup made `wrecking_ball400` complete at about `526s`.
2. Dot/norm fallback and bbox cleanup improved it to about `517s`.
3. ABD line-search `step_forward` GPU default improved it to about `502s`.
4. Matrix linear reduce plus AllPE-off reduced `wb400` to about `298-308s`.
5. PCG fused `rz/norm` reduced the v15 default from `319s` to `296s`.
6. `filter_active` synchronization removal produced a `wb400` result around `288s`
   in its default regression.
7. ABD DyTopo parallel assembly produced the strongest later win, with `wb400`
   around `180s`.

## Current Stable Baseline Takeaway

The stable performance gains came from a few recurring patterns:

- Removing CoreX-only serial or near-quadratic hot paths.
- Avoiding redundant full-device synchronization where same-stream ordering is
  sufficient.
- Preserving solver convergence semantics while reducing per-iteration scalar
  reduction work.
- Matching NVIDIA algorithm structure when CoreX-only compatibility additions
  created extra contact or conversion work.

The later reports show that further progress is unlikely to come from broad
environment-variable sweeps. The remaining gap is tied to high PE/EE contact load,
matrix row imbalance, and PCG iteration count, so future default candidates should
be validated against those long-run metrics before promotion.
