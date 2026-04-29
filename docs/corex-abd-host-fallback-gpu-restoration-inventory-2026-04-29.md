# ABD Host Fallback GPU 恢复 — 清单与状态（2026-04-29）

对照 `Corex 兼容版本libuipc的性能与影响性能的主要原因.md` 中 8 项 ABD 相关 host fallback，在 **CoreX (`UIPC_COREX_CUDA10_COMPAT`) 默认路径** 下的结论如下。

| # | 位置 | 频率 / 触发 | 默认路径（改动前） | GPU 恢复 / 开关 |
|---|------|-------------|-------------------|----------------|
| 1 | `ABDLineSearchReporter::step_forward` | 每 Newton × line search | D2H `is_fixed,q_temp,dq` + CPU + H2D | `kernel_step_forward`；`UIPC_COREX_ABD_LINE_SEARCH_HOST_STEP=1` 回退 |
| 2 | `ABDLineSearchReporter::compute_energy` 三处 sum | 每能量评估 | D2H 数组 + CPU sum + H2D scalar | `DeviceReduce::Sum`；`UIPC_COREX_ABD_ENERGY_HOST_SUM=1` 回退 |
| 3 | `ABDToleranceChecker::do_check` | 每 Newton | D2H 全量 `dqs` | device check + D2H 标量 `success`；`UIPC_COREX_ABD_TOLERANCE_HOST_FALLBACK=1` 回退 |
| 4 | `AffineBodyVertexReporter::init/update_attributes` | 首帧 init / 按需 update | D2H `J,v2b,q` + `point_x`/`x_bar` | `kernel_abd_init_vertex_attributes` / `kernel_abd_update_vertex_positions`；`UIPC_COREX_ABD_VERTEX_HOST_FALLBACK=1` |
| 5 | `report_displacements` | 每帧 | 已在 CoreX 用 `kernel_abd_report_displacements` | 无变更 |
| 6 | `AffineBodyBodyReporter::report_attributes` coindices iota | 低频 | Host iota + H2D | `kernel_abd_coindices_iota` |
| 7 | `AffineBodyBDF1Kinetic::compute_energy` | Newton × line search | D2H + CPU | `kernel_abd_bdf1_energy`；`UIPC_COREX_ABD_BDF1_HOST_FALLBACK=1` |
| 8 | `AffineBodyBDF1Kinetic::compute_gradient_hessian` | 每 Newton | D2H + CPU | `kernel_abd_bdf1_gradient_hessian`；`UIPC_COREX_ABD_BDF1_HOST_FALLBACK=1` |

## 验证场景建议

- **正确性**：`simple` gate（GPU1，`simple_gate.py`）。
- **性能**：`wrecking_ball400`，记录总时长与 `profile_summary.py` 中的 PCG / Newton / triplets / candidates。
- **敏感场景**：`slope`（摩擦 / line search），`wrecking_ball150`（长程 Newton/PCG 对比）。

## P4 非 ABD 项（本轮仅记录，不修改代码）

- `GlobalTrajectoryFilter::filter_toi` min-TOI D2H。
- `GlobalDyTopoEffectManager::_distribute` scalar count。
- dot/norm `float` strided (`inc!=1`)。
- host `ge2sym` / CPU SpMV debug 路径、trace 全量 D2H、常量 host fill。

后续批次单独跟踪。
