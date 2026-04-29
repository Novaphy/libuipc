# CoreX Wrecking Ball 下一轮优化与短跑判断报告

日期：2026-04-28

项目：`libuipc-v11-restored/libuipc-v11-extracted`

## 1. 本轮目标

本轮按照 `wrecking-next-optimization` 计划推进，目标是从上一轮 host fallback 清障转向 `wrecking_ball400` 的真实主瓶颈，重点判断以下路径是否值得继续优化：

- PCG / SpMV 每迭代固定成本。
- `GlobalLinearSystem` 同步链。
- preconditioner apply 成本。
- contact filter、DyTopo 装配与 matrix conversion。
- 是否能用 150 帧短跑判断改动收益，而不是每次都跑完整 400 帧。

本轮没有修改计划文件。

## 2. 代码修改概览

### 2.1 新增 CoreX Phase Profiling

新增文件：

- `src/backends/cuda/utils/corex_phase_profile.h`

修改文件：

- `apps/examples/corex_demo/main.cpp`
- `tools/simple_physics_audit/profile_summary.py`
- `src/backends/cuda/linear_system/linear_pcg_corex.cu.inc`
- `src/backends/cuda/linear_system/global_linear_system.cu`
- `src/backends/cuda/collision_detection/global_trajectory_filter.cu`
- `src/backends/cuda/dytopo_effect_system/global_dytopo_effect_manager.cu`

新增环境变量：

```bash
UIPC_COREX_PHASE_PROFILE=1
```

打开后会输出统一格式：

```text
[corex_phase] category=<category> name=<phase> frame=<frame> newton=<newton> iter=<iter> elapsed_ms=<ms>
```

已接入的阶段包括：

- `pcg.solve_total`
- `pcg.spmv`
- `pcg.spmv_sync`
- `pcg.preconditioner`
- `pcg.dotnorm`
- `linear.update_subsystem_extent`
- `linear.assemble_linear_system`
- `linear.converter_ge2sym`
- `linear.converter_convert`
- `linear.pre_preconditioner_sync`
- `linear.assemble_preconditioner`
- `linear.post_preconditioner_sync`
- `linear.pre_pcg_sync`
- `contact.detect`
- `contact.filter_active`
- `contact.filter_toi`
- `dytopo.assemble_total`
- `dytopo.report_extent`
- `dytopo.scan_allocate`
- `dytopo.reporter_assemble`
- `dytopo.convert_matrix`
- `dytopo.distribute_total`

`profile_summary.py` 已扩展 `corex_phase_ms` 聚合字段，可按 phase 输出 count/min/max/mean/sum。

### 2.2 CoreX Demo 每帧 Timing

修改文件：

- `apps/examples/corex_demo/main.cpp`

行为：

- 默认仍只打印前 3 帧和最后一帧 timing。
- 设置 `UIPC_COREX_PHASE_PROFILE=1` 后，每帧都打印：

```text
[corex_demo] frame <i> timings: advance=<ms> sync=<ms> retrieve=<ms> write_obj=<ms>
```

这解决了上一轮 `profile_summary.py` 只能解析少量帧 timing，无法稳定判断 p50/p95 的问题。

### 2.3 PCG SpMV 后同步实验

修改文件：

- `src/backends/cuda/linear_system/linear_pcg_corex.cu.inc`

实验内容：

- 曾尝试默认移除每次 SpMV 后的 `cudaDeviceSynchronize()`。
- `simple` 90/300 gate 通过。
- `wrecking_ball150` 短跑显示有一定收益。
- 但 `wrecking_ball400` 当前路径结果变差，说明长程稳定性和阶段负载波动不能用短跑完全替代。

最终处理：

- 默认恢复保守同步。
- 保留显式实验开关：

```bash
UIPC_COREX_PCG_SKIP_SPMV_SYNC=1
```

### 2.4 GlobalLinearSystem PCG 前同步实验

修改文件：

- `src/backends/cuda/linear_system/global_linear_system.cu`

实验内容：

- 曾尝试默认跳过 `solve_linear_system()` 进入 PCG 前的全设备同步。
- `simple90` gate 通过。
- 但 profile 显示该同步耗时极小，不是主瓶颈。

最终处理：

- 默认保留同步。
- 保留显式实验开关：

```bash
UIPC_COREX_SKIP_PRE_PCG_SYNC=1
```

### 2.5 ABD Diag Preconditioner 同步实验

修改文件：

- `src/backends/cuda/affine_body/abd_diag_preconditioner.cu`

实验内容：

- 将 `kernel_abd_jacobi_extract` / `kernel_abd_jacobi_apply` 后的同步改为可跳过。
- `simple90` gate 通过。
- 但短跑 profile 显示 preconditioner apply 不是本轮最大热点。

最终处理：

- 默认保留同步。
- 保留显式实验开关：

```bash
UIPC_COREX_ABD_PRECOND_SKIP_SYNC=1
```

## 3. 构建与基本验证

构建命令：

```bash
cmake --build /root/libuipc-v11-restored/libuipc-v11-extracted/build_corex --target corex_demo -j112
```

结果：

- 构建通过。
- 仍有既有 CoreX/Clang 警告，但无阻塞错误。
- `ReadLints` 对本轮主要修改文件未发现新增诊断。

Correctness gate：

- phase profiling 冒烟：`simple30 + UIPC_COREX_PHASE_PROFILE=1` 通过。
- 移除 SpMV 同步实验后：`simple90` / `simple300` 通过。
- 移除 pre-PCG 同步实验后：`simple90` 通过。
- ABD preconditioner 同步实验后：`simple90` 通过。
- 最终保守默认路径：`simple90` 通过。

## 4. A/B 与长跑结果

### 4.1 SpMV 后同步短跑 A/B

场景：`wrecking_ball150`

对比：

- 默认新路径（当时跳过 SpMV 后同步）：约 135.5s。
- 强制旧同步：约 163.1s。

初步判断：

- 150 帧短跑显示跳过 SpMV 后同步有收益。
- 但后续 400 帧验证不稳定，不能直接作为默认策略。

### 4.2 最终 150 帧 A/B

当前保守默认路径：

- 起止：11:51:49.767 到 11:54:05.409。
- 总耗时：约 135.64s。
- PCG mean: 63.00
- PCG max: 179
- PCG sum: 48324
- Newton mean: 3.51
- Newton sum: 519
- unique triplets mean: 14317.55
- candidate mean: 17830.10

强制旧同步路径：

- 起止：11:54:14.801 到 11:56:34.338。
- 总耗时：约 139.54s。
- PCG mean: 61.59
- PCG max: 174
- PCG sum: 47607
- Newton mean: 3.55
- Newton sum: 525
- unique triplets mean: 14382.32
- candidate mean: 18279.59

判断：

- 当前路径略快，约 2.8%。
- 但 PCG/Newton/candidate 差异并不完全同向，短跑 A/B 仍有明显噪声。
- 150 帧适合作为筛选，不适合作为最终收益证明。

### 4.3 400 帧当前路径验证

场景：`wrecking_ball400`

结果：

- 起止：11:56:51.731 到 12:06:16.672。
- 总耗时：约 564.94s。
- PCG count: 2844
- PCG mean: 63.33
- PCG max: 168
- PCG sum: 180098
- Newton count: 395
- Newton mean: 5.19
- Newton max: 22
- Newton sum: 2049
- unique triplets mean: 20602.78
- unique triplets max: 29990
- candidate mean: 17605.59
- candidate max: 21852

与上一轮 400 帧结果对比：

- 上一轮：517.45s
- 本轮当前路径：564.94s
- 结果：变慢约 47.49s

判断：

- 本轮同步实验不能作为默认优化合入。
- 400 帧后半程负载更重，PCG count/sum、candidate 记录数等与 150 帧短跑差异明显。
- 短跑可用于“判断方向”，但最终仍必须用 400 帧确认组合效果。

## 5. 短跑判断结果

本轮专门执行了一次 `wrecking_ball150 + UIPC_COREX_PHASE_PROFILE=1` 作为短跑判断。

产物：

- `artifacts/wrecking_judgement/wb150_current_profile.log`
- `artifacts/wrecking_judgement/wb150_current_profile.json`

总耗时：

- 起止：09:35:55.175 到 09:38:12.313。
- 约 137.14s。

基础指标：

- PCG count: 760
- PCG mean: 62.43
- PCG max: 175
- PCG sum: 47443
- Newton count: 148
- Newton mean: 3.46
- Newton max: 11
- unique triplets mean: 14191.26
- simplex candidates mean: 19305.90
- simplex candidates max: 21379

阶段耗时排序（按 sum）：

1. `linear.assemble_linear_system`: 35.33s
2. `dytopo.convert_matrix`: 35.26s
3. `linear.converter_convert`: 17.28s
4. `pcg.solve_total`: 12.82s
5. `contact.detect`: 10.23s
6. `contact.filter_active`: 8.93s
7. `pcg.dotnorm`: 7.86s
8. `contact.filter_toi`: 2.16s
9. `pcg.spmv_sync`: 1.90s
10. `pcg.preconditioner`: 1.59s
11. `pcg.spmv`: 1.27s

关键判断：

- 最大热点不是同步，而是 matrix conversion / assembly 链路。
- `dytopo.convert_matrix` 与 `linear.converter_convert` 是最值得继续细化的路径。
- `pcg.dotnorm` 比 `pcg.spmv` 更值得继续看。
- preconditioner apply 和 pre-PCG sync 不是主要瓶颈。
- contact detect/filter_active 有明显成本，但仍低于 conversion/assembly。

## 6. 对短跑判断方法的结论

短跑判断可用于回答：

- 某个改动是否命中目标阶段。
- 是否值得进入 400 帧长跑。
- 当前主要热点在哪个子系统。

短跑判断不能单独回答：

- 400 帧端到端一定会变快。
- 后半程候选、Newton、PCG 尖峰是否会恶化。
- 改动是否在长时间接触密集阶段稳定。

建议后续采用三级验证：

1. `simple` gate：正确性快速门禁。
2. `wrecking_ball150 + UIPC_COREX_PHASE_PROFILE=1`：判断目标阶段收益。
3. `wrecking_ball400`：只对候选优化组合做最终验证。

## 7. 下一步优化建议

本轮判断后，下一步不建议继续优先删同步，而应转向以下方向：

### P0：DyTopo / Linear Matrix Conversion

重点文件：

- `src/backends/cuda/dytopo_effect_system/global_dytopo_effect_manager.cu`
- `src/backends/cuda/algorithm/details/matrix_converter.inl`
- `src/backends/cuda/linear_system/global_linear_system.cu`

原因：

- `dytopo.convert_matrix`: 35.26s
- `linear.converter_convert`: 17.28s
- `linear.assemble_linear_system`: 35.33s

下一步应细化：

- `matrix_converter.convert` 内部 sort/reduce/scan 各阶段耗时。
- DyTopo hessian/gradient conversion 是否重复排序或重复分配。
- `GlobalLinearSystem` triplet 到 BCOO 的转换是否可以复用 workspace、减少中间 buffer 清零或同步。

### P1：PCG Dot/Norm

原因：

- `pcg.dotnorm`: 7.86s
- 明显高于 `pcg.spmv`: 1.27s

下一步应检查：

- 每次 PCG 迭代中的 dot/norm 次数。
- 是否能融合 `dot(r,z)` 与 `norm(r)`，或减少 host scalar synchronization。
- 是否能保留正确收敛判据的同时降低 reduction 次数。

### P2：Contact Detect / Filter Active

原因：

- `contact.detect`: 10.23s
- `contact.filter_active`: 8.93s

下一步应检查：

- stackless BVH detect 内部各 query 通道耗时。
- `filter_active` 是否存在重复候选处理。
- 是否有高频小同步或可合并的 temp buffer pass。

### P3：同步实验暂缓默认启用

保留实验开关：

- `UIPC_COREX_PCG_SKIP_SPMV_SYNC=1`
- `UIPC_COREX_SKIP_PRE_PCG_SYNC=1`
- `UIPC_COREX_ABD_PRECOND_SKIP_SYNC=1`

默认保持保守同步，直到有稳定 400 帧收益证据。

## 8. 产物索引

本轮主要 profile：

- `artifacts/wrecking_judgement/wb150_current_profile.json`
- `artifacts/wrecking_next/wb150_current_profile.json`
- `artifacts/wrecking_next/wb150_force_legacy_sync_profile.json`
- `artifacts/wrecking_next/wb400_current_profile.json`
- `artifacts/wrecking_next/wb150_phase_profile.json`

本轮主要代码修改：

- `src/backends/cuda/utils/corex_phase_profile.h`
- `apps/examples/corex_demo/main.cpp`
- `tools/simple_physics_audit/profile_summary.py`
- `src/backends/cuda/linear_system/linear_pcg_corex.cu.inc`
- `src/backends/cuda/linear_system/global_linear_system.cu`
- `src/backends/cuda/collision_detection/global_trajectory_filter.cu`
- `src/backends/cuda/dytopo_effect_system/global_dytopo_effect_manager.cu`
- `src/backends/cuda/affine_body/abd_diag_preconditioner.cu`

最终状态：

- profiling 功能保留。
- 同步实验保留为显式 opt-in。
- 默认路径保持保守同步。
- `simple90` gate 通过。

