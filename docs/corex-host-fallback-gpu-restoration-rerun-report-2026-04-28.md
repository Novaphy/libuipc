# CoreX ABD Host Fallback GPU 化恢复复跑报告（2026-04-28）

## 执行结论

本轮按 `host-fallback-gpu-restoration_2a5b7216.plan.md` 重新逐块执行并验证。所有实现均保留 GPU 版本与回退开关；`wrecking_ball400` 逐块验证后，默认路径只保留收益明确的 `ABDLineSearchReporter::step_forward` GPU kernel，其余 GPU 化实现默认关闭、作为实验开关保留。

最终默认组合：

- `ABDLineSearchReporter::step_forward`：默认 GPU。
- `AffineBodyBDF1Kinetic::compute_gradient_hessian`：默认 host fallback，`UIPC_COREX_ABD_BDF1_GRADIENT_HESSIAN_GPU=1` 启用实验 GPU。
- `AffineBodyBDF1Kinetic::compute_energy`：默认 host fallback，`UIPC_COREX_ABD_BDF1_ENERGY_GPU=1` 启用实验 GPU。
- `ABDLineSearchReporter` energy reductions：默认 host sum，`UIPC_COREX_ABD_ENERGY_REDUCTION_GPU=1` 启用实验 GPU reduction。
- `ABDToleranceChecker::do_check`：默认 host checker，`UIPC_COREX_ABD_TOLERANCE_GPU=1` 启用实验 GPU checker。
- ABD vertex init/update：默认 host fallback，`UIPC_COREX_ABD_VERTEX_GPU=1` 启用实验 GPU。
- ABD body iota：默认 host fallback，`UIPC_COREX_ABD_BODY_IOTA_GPU=1` 启用实验 GPU。

## 代码变更

- `src/backends/cuda/affine_body/bdf/affine_body_bdf1_kinetic.cu`
  - 新增显式 `kernel_abd_bdf1_energy` 和 `kernel_abd_bdf1_gradient_hessian`。
  - 将质量矩阵乘法与 Hessian 写入展开为显式 12 维 / 12x12 写入。
  - 增加分项 GPU 实验开关。
- `src/backends/cuda/affine_body/abd_jacobi_matrix_corex.h`
  - 增加 `ABDJacobiDyadicMass` 的只读 accessor，供显式 kernel 读取质量展开数据。
- `src/backends/cuda/affine_body/abd_line_search_reporter.cu`
  - `step_forward` 默认切到已有显式 `kernel_step_forward`。
  - 三处 energy sum 增加 device reduction 实验路径。
- `src/backends/cuda/affine_body/abd_tolerance_checker.cu`
  - 增加 device flag checker 实验路径。
- `src/backends/cuda/affine_body/affine_body_vertex_reporter.cu`
  - 增加 vertex init/update 显式 kernel 实验路径。
  - `INIT_ATTR` 逐点日志改为 `UIPC_COREX_TRACE_VERTEX_INIT` trace-only。
- `src/backends/cuda/affine_body/affine_body_body_reporter.cu`
  - 增加 body coindices iota kernel 实验路径。
- `docs/corex-abd-host-fallback-gpu-restoration-inventory-2026-04-29.md`
  - 新增 ABD host fallback 状态清单。

## 逐块验证结果

基线参考：用户给出的上一轮 `wrecking_ball400 = 526.41s`。

| 修改块 | simple | 额外场景 | wb400 wall | PCG sum/max | Newton sum/max | 默认决策 |
|---|---:|---:|---:|---:|---:|---|
| BDF1 gradient/hessian GPU | 90/300 PASS | - | 557s | 184386 / 177 | 1993 / 35 | 默认关闭 |
| BDF1 energy GPU | 90/300 PASS | - | 567s | 184689 / 175 | 1989 / 25 | 默认关闭 |
| line search step_forward GPU | 90/300 PASS | - | 516s | 169608 / 175 | 2043 / 34 | 默认开启 |
| energy reductions GPU | 90/300 PASS | slope120 PASS | 546s | 176130 / 172 | 1993 / 22 | 默认关闭 |
| tolerance checker GPU | 90/300 PASS | slope120 PASS | 546s | 175690 / 179 | 2072 / 35 | 默认关闭 |
| vertex init/update + iota GPU | 90/300 PASS | slope120/stack120 PASS | 522s | 173540 / 177 | 2053 / 20 | 默认关闭 |
| final default组合 | 300 PASS | - | 502s | 166063 / 174 | 2012 / 22 | 保留 |

## 最终性能

最终默认 `wrecking_ball400`：

- 总耗时：`502s`
- 相对上一轮 `526.41s`：约 `4.6%` 提升
- PCG：mean `63.65`，max `174`，sum `166063`
- Newton：mean `5.07`，max `22`，sum `2012`
- Unique triplets：mean `21211.33`，max `29174`
- Simplex candidates：mean `15970.01`，max `21385`

## 判断

这轮最有效的默认改动是 `ABDLineSearchReporter::step_forward` 从 D2H/CPU/H2D 切到显式 GPU kernel。其它 GPU 化路径虽然能通过 correctness gate，但在 `wrecking_ball400` 长程统计中会引入更高 PCG/Newton 或候选波动，导致整体 wall time 变差，因此默认关闭，仅保留为后续分析和 A/B 的实验路径。

下一步建议继续按非 ABD fallback 批次推进：TOI reduction、DyTopo scalar count、strided dot/norm，以及更主要的 DyTopo/linear converter/PCG 热点。
