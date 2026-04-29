# CoreX 下一轮优化执行报告（2026-04-28）

## 修改概览

本轮围绕 `corex-next-opt` 计划实现了以下内容：

- ABD line-search energy 新增 `UIPC_COREX_ABD_FUSED_ENERGY_GPU=1` 实验路径，把 BDF1 kinetic energy 计算与求和合并为单个 kernel。
- ABD shape/reporter energy 新增 `UIPC_COREX_ABD_LIGHT_REDUCTION_GPU=1` 轻量 block reduction，避免走通用 `muda::DeviceReduce`。
- 新增 `UIPC_COREX_ABD_ENERGY_AB_COMPARE=1` A/B 误差日志，并扩展 `tools/simple_physics_audit/profile_summary.py`，输出 frame/newton 级 PCG 统计与 energy A/B 误差统计。
- TOI min 新增 `UIPC_COREX_TOI_DEVICE_MIN=1` device reduction 实验路径，但默认保持 host min。
- CoreX `muda::LinearSystemContext` 的 dot/norm device fallback 支持 `inc!=1` strided vector，避免该类 fallback 只能回到 host。
- BDF1 gradient/hessian 新增 `UIPC_COREX_ABD_BDF1_TRACE_GH=1` 调用频率诊断，用于评估 assembly 融合机会。

## 验证结果

### Fused ABD Energy

- `simple90`：PASS
- `simple300`：PASS
- `slope120`：PASS
- `simple90` A/B 误差：
  - kinetic 最大绝对误差约 `1.19e-7`
  - shape/reporter 误差为 `0`
- `wrecking_ball400`：`508s`
  - 当前默认基线约 `502s`
  - PCG sum `168730`
  - Newton sum `1992`

结论：数值等价性较好，但长跑 wall time 未优于默认配置，因此保持实验开关，不默认启用。

### 非 ABD Scalar Fallback

- TOI device min 初版默认启用后：
  - `simple90/simple300`：PASS
  - `wrecking_ball400`：`536s`
  - PCG sum `171225`
  - Newton sum `1989`
- 因极小规模 reduction 的 kernel launch 成本高于收益，TOI device min 改为 opt-in：`UIPC_COREX_TOI_DEVICE_MIN=1`。
- strided dot/norm 的 device fallback 已支持 stride。`simple10` trace 未观察到 host fallback 触发，说明默认场景当前仍主要走 cublas 成功路径。

结论：TOI device min 不默认启用；dot/norm stride 支持保留，作为 cublas 不支持 strided 场景时的安全替代路径。

### BDF1 Gradient/Hessian 与 Assembly 融合评估

- `stack120` trace：
  - BDF1 GH calls：`253`
  - `gradient_only=0`：`253`
  - `gradient_only=1`：`0`
- `wrecking_ball150` phase profile：
  - BDF1 GH calls：`869`
  - `gradient_only=0`：`869`
  - `gradient_only=1`：`0`
  - `linear.assemble_linear_system` sum：`41078.080ms`
  - `dytopo.convert_matrix` sum：`38836.376ms`
  - `linear.converter_convert` sum：`20056.693ms`
  - `pcg.solve_total` sum：`15021.708ms`
  - `pcg.dotnorm` sum：`9204.926ms`

结论：当前 ABD BDF1 路径没有可利用的 `gradient_only` 跳过 Hessian 机会。下一步若继续优化，应改造 ABD kinetic/constitution 与 ABDLinearSubsystem 的接口，让 kinetic 直接写入 assembly 目标结构，避免先写 `Matrix12x12[N]` 再被 assembly/converter 处理；这属于跨接口改造，不应在当前已验证候选中默认启用。

## 默认启用决策

- 默认启用：无新增性能开关。
- 最终默认 `wrecking_ball400` 复测：`511s`
  - PCG sum `165153`
  - Newton sum `2074`
  - 该结果接近前一轮默认 `502s` 基线，未引入已知实验路径默认退化；差异主要作为长跑波动/迭代路径变化继续观察。
- 保持默认关闭：
  - `UIPC_COREX_ABD_FUSED_ENERGY_GPU`
  - `UIPC_COREX_ABD_LIGHT_REDUCTION_GPU`
  - `UIPC_COREX_TOI_DEVICE_MIN`
  - 既有 `UIPC_COREX_ABD_BDF1_*_GPU` 实验路径

默认策略继续遵守 `wb400` 优先：只有 wall time 下降且 PCG/Newton 不恶化的候选才默认启用。
