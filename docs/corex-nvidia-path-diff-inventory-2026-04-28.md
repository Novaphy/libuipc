# CoreX / NVIDIA GPU 路径对照清单（2026-04-28）

## 目的

本清单对照 NVIDIA `#else` 原始 GPU 路径与 CoreX `UIPC_COREX_CUDA10_COMPAT` 路径，解释 host fallback 的原因、当前回 GPU 的规避方式，以及可能导致性能收益不高的实现差异。

## ABD Line Search

文件：`src/backends/cuda/affine_body/abd_line_search_reporter.cu`

| 项目 | NVIDIA 路径 | CoreX 当前路径 | fallback / 规避原因 | 性能风险 |
|---|---|---|---|---|
| `step_forward` | `ParallelFor` 更新 `q` | 手写 `kernel_step_forward`，默认 GPU | 避免复杂 viewer/lambda 路径 | 基本等价，是少数默认有收益路径 |
| kinetic energy sum | `compute_energy` + `DeviceReduce` | 默认 host sum；可选 `DeviceReduce`、light reduction、fused kinetic | `DeviceReduce`/CUB 在 CoreX 上成本和兼容性不确定 | host sum 有 D2H/H2D；light reduction 多 kernel + atomic；fused kinetic 仍未赢过 `wb400` |
| shape/reporter sum | `DeviceReduce` | 默认 host sum；可选 GPU reduction | 同上 | 小数组 reduction 单独 GPU 化容易被 launch 成本抵消 |

结论：单独恢复 reduction 不够，下一步只在能和 line-search 主流程融合时继续推进。

## ABD BDF1 与 Assembly

文件：`src/backends/cuda/affine_body/bdf/affine_body_bdf1_kinetic.cu`、`src/backends/cuda/affine_body/abd_linear_subsystem.cu`

| 项目 | NVIDIA 路径 | CoreX 当前路径 | fallback / 规避原因 | 性能风险 |
|---|---|---|---|---|
| BDF1 energy | `ParallelFor`，device Eigen 表达式 | 默认 host fallback；可选手写 kernel | 避免 `ABDJacobiDyadicMass::operator*` device 表达式 | 手写 kernel 正确但小 launch；默认 host fallback 会产生全量 copy |
| BDF1 gradient/hessian | `ParallelFor`，`M * dq` 与 `M.to_mat()` | 默认 host fallback；可选手写 `corex_abd_mass_mul/to_mat` | 避免 Eigen/muda device codegen 风险 | 仍写 `Matrix12x12[N]`，随后 assembly 再读，未减少中间态 |
| kinetic/shape assembly | `ParallelFor` 异步装配 | CoreX 手写 kernel，存在显式 `cudaDeviceSynchronize()` | 早期用于定位 CoreX kernel 错误 | 全局同步会打断流水，是优先优化点 |

结论：BDF1 单 kernel 不是瓶颈核心，assembly 数据流和同步更值得优先处理。

## ABD Vertex Reporter

文件：`src/backends/cuda/affine_body/affine_body_vertex_reporter.cu`

| 项目 | NVIDIA 路径 | CoreX 当前路径 | fallback / 规避原因 | 性能风险 |
|---|---|---|---|---|
| init/update attributes | `ParallelFor` + `ABDJacobi::point_x` | 默认 host fallback；可选手写 `corex_abd_point_x` kernel | 避免 `point_x` device 表达式 | host fallback 全量 D2H/H2D，GPU kernel launch 低频 |
| report displacements | `ParallelFor` 异步 | 手写 kernel 后强制 `cudaDeviceSynchronize()` | 可能是早期 debug/正确性保险 | 每次报告位移都全设备同步，和 NVIDIA 异步路径差异明显 |

结论：强同步应改为 debug-only；若消费者在同一 stream 上，默认不需要全设备同步。

## Linear System / SpMV / DotNorm

文件：`src/backends/cuda/linear_system/spmv.cu`、`external/muda/src/muda/ext/linear_system/details/routines/dot.inl`、`norm.inl`

| 项目 | NVIDIA 路径 | CoreX 当前路径 | fallback / 规避原因 | 性能风险 |
|---|---|---|---|---|
| `sym_spmv` | triplet 并行 atomic | 默认 triplet atomic；可回退 row-scan | 已从病态 row-scan 回到 triplet 并行 | atomic 冲突仍可能高，但比 row-scan 好 |
| `rbk_spmv` | CUB warp segmented reduce | CoreX workaround 强制转 `sym_spmv` | CUB HeadSegmentedReduce / shuffle 曾在 CoreX 崩溃 | 放弃 RBK 优化路径，PCG 高频放大 |
| dot/norm | cublas 优先 | cublas 优先；NOT_SUPPORTED 时自定义两阶段 reduction | 避免 float fallback D2H | 两阶段 reduction + host-pointer overload 同步；force/trace 环境变量不能用于性能跑 |

结论：SpMV 下一步应减少 triplet atomic 冲突，或在可控条件下重新评估 RBK segmented reduce。

## TOI / DyTopo

文件：`src/backends/cuda/collision_detection/global_trajectory_filter.cu`、`src/backends/cuda/dytopo_effect_system/global_dytopo_effect_manager.cu`

| 项目 | NVIDIA 路径 | CoreX 当前路径 | fallback / 规避原因 | 性能风险 |
|---|---|---|---|---|
| TOI min | D2H `tois` + host min | 默认同 NVIDIA；可选 device min | 尝试减少 D2H | `wb400` 已证明单独 device min 退化，保持 opt-in |
| DyTopo distribute count | scan 后 copy scalar | 与 NVIDIA 类似 | 需要 host resize 决策 | 单独 GPU 化意义有限 |
| CoreX matrix converter | NVIDIA 无此特化 | CoreX `corex_matconv` 多个 launch 后强同步；segmental reduce 每 segment 扫全量输入 | 避免原 converter/库路径兼容问题 | 非 trace 同步和 `O(N*out_count)` 是明显热点风险 |

结论：DyTopo 优化重点不是 scalar count，而是 matrix converter 的同步和 segmental reduce 算法复杂度。

## 优先级

1. 移除 ABD assembly / vertex reporter 中非必要强同步。
2. 对 BDF1 kinetic Hessian 做 direct assembly 实验，减少 `Matrix12x12[N]` 中间写读。
3. 设计 CoreX grouped SpMV，降低 triplet atomic 冲突。
4. 分析并替换 DyTopo CoreX segmental reduce。
5. Energy / TOI / dot-norm 仅在能并入热路径或真实触发高频时继续默认候选验证。
