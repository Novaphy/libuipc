# CoreX Host Fallback 替代 Kernel 工作报告

日期：2026-04-27

项目：`libuipc-v11-restored/libuipc-v11-extracted`

## 1. 工作目标

本轮工作目标是针对 CoreX 兼容路径中的 host fallback 和 H2D/D2H 热点，优先实现功能等价的设备端替代路径，减少全量 D2H/H2D 搬运和全设备同步，同时维持仿真结果稳定。

计划中的重点包括：

- 为 PCG hot loop 中的 dot/norm host fallback 增加计数与 trace。
- 实现 `float && inc == 1` 的 CoreX device dot/norm fallback kernel。
- 将 `GlobalVertexManager` 中 bbox 全量 D2H 改为设备端 reduction。
- 审计 `GlobalLinearSystem` 默认热路径同步点。
- 统计 H2D/D2H 调用量，判断是否适合继续恢复 async copy。
- 每次关键路径修改后立即运行 GPU1 `simple` gate。

## 2. 代码修改概览

### 2.1 Dot/Norm Host Fallback 替代

修改文件：

- `external/muda/src/muda/ext/linear_system/details/routines/dot.inl`
- `external/muda/src/muda/ext/linear_system/details/routines/norm.inl`

主要变更：

- 新增 CoreX dot/norm fallback 计数。
- 新增可选 trace：
  - `UIPC_COREX_TRACE_DOTNORM_FALLBACK=1`
  - 也兼容 `UIPC_COREX_TRACE_LINEAR_SYSTEM=1`
- 保留旧 host fallback 回退开关：
  - `UIPC_COREX_DOTNORM_HOST_FALLBACK=1`
- 新增 `float && inc == 1` 的 device reduction fallback：
  - `dot`: block partial reduction + final partial reduction。
  - `norm`: block sum-of-squares reduction + final sqrt。
- 新增测试/诊断专用强制 device fallback 开关：
  - `UIPC_COREX_DOTNORM_FORCE_DEVICE_FALLBACK=1`

设计取舍：

- 第一版只覆盖 PCG 主要使用的 dense contiguous vector。
- `inc != 1`、`double`、其他非主路径仍保留原 fallback。
- `VarView<T>` 结果路径直接写 device scalar，避免 host round-trip。
- Host scalar 结果路径只 D2H 一个 scalar。

### 2.2 Vertex Bounding Box 替代

修改文件：

- `src/backends/cuda/global_geometry/global_vertex_manager.cu`

主要变更：

- CoreX 分支 `compute_vertex_bounding_box()` 不再默认将全部 `positions` copy 到 host。
- 默认复用 NVIDIA 分支已有的 `muda::DeviceReduce<Vector3>` 逻辑，分别计算 min/max。
- 最终只读取 `min_pos` / `max_pos` 两个 `Vector3`。
- 保留旧 CPU fallback：
  - `UIPC_COREX_BBOX_HOST_FALLBACK=1`

收益预期：

- 消除 bbox 阶段的全量 D2H。
- 对 vertex 数较多的场景减少 host 端阻塞。
- 对 `wrecking_ball400` 的总体贡献有限，因为主耗时并不在 bbox。

### 2.3 Memory Copy 统计

修改文件：

- `external/muda/src/muda/launch/details/memory.inl`

主要变更：

- 保持当前策略：CoreX 下 D2D 使用 `cudaMemcpyAsync`，H2D/D2H 默认仍使用同步 `cudaMemcpy`。
- 新增可选统计：
  - `UIPC_COREX_MEMCPY_STATS=1`
- 新增可选轻量 trace：
  - `UIPC_COREX_TRACE_MEMCPY=1`
- 默认不设置环境变量时，不进行 atomic 计数，避免引入默认路径开销。

结论：

- 当前不建议全局恢复 H2D/D2H async。
- 后续如需 async，应按 call site 证明 host 生命周期安全后逐点恢复。

### 2.4 审计文档

新增审计文件：

- `artifacts/host_fallback_dotnorm/global_linear_system_sync_audit.md`
- `artifacts/host_fallback_memory/selective_h2d_d2h_async_audit.md`

审计结论：

- `GlobalLinearSystem` 默认热路径仍有 3 个关键同步点：
  - pre-preconditioner sync
  - post-preconditioner sync
  - solve 前 pre-PCG sync
- 不建议一次性删除所有同步。
- 下一轮应先 instrument stream ownership，再单点替换为 stream/event 依赖。
- H2D/D2H 是高频小字节，不是当前 `wrecking_ball400` 的主瓶颈。

## 3. 构建与验证结果

### 3.1 构建

使用 `-j112` 构建：

```bash
cmake --build /root/libuipc-v11-restored/libuipc-v11-extracted/build_corex --target corex_demo -j112
```

结果：通过。

说明：

- 初始 `-j2` 构建耗时过长，后按要求改为 `-j112`。
- 构建过程中仍有既有 CoreX/Clang 警告，但未出现阻塞错误。
- `ReadLints` 对本轮主要修改文件未发现新增诊断。

### 3.2 Dot/Norm Micro 与 Gate

`corex_cublas_inventory`：

- `cublasSdot` stride1/strided：通过。
- `cublasSnrm2` stride1/strided：通过。
- `double` dot/norm 仍返回 `NOT_SUPPORTED`，符合 CoreX 当前兼容现状。

GPU1 `simple` gate：

- `simple` 90 帧：PASS。
- `simple` 300 帧：PASS。
- forced device dot/norm fallback 的 `simple` 90 帧：PASS。

forced device fallback 覆盖：

- `simple90_forced_device_run.log` 中记录到 6648 次 dot/norm device fallback trace。
- 说明新增 reduction kernel 被真实 PCG 路径覆盖。

`simple` 90 帧 profile：

- PCG count: 258
- PCG mean: 7.63
- PCG max: 18
- Newton mean: 1.90
- candidate sum: 273

`simple` 300 帧 profile：

- PCG count: 847
- PCG mean: 8.14
- PCG max: 18
- Newton mean: 1.83
- candidate sum: 273

### 3.3 Stack / Wrecking Ball 短帧 A/B

`stack30` default 与 forced device fallback：

- PCG count: 58
- PCG mean: 0.5
- PCG sum: 29
- Newton mean: 1.0
- candidate sum: 0
- 两者统计一致。

`wrecking_ball5` default 与 forced device fallback：

- PCG count: 8
- PCG mean: 9.0
- PCG sum: 72
- Newton mean: 1.0
- triplets mean: 5750
- 两者统计一致。

结论：

- 新 device dot/norm fallback 对收敛行为没有可见扰动。
- 对短帧场景没有引入 PCG/Newton 回归。

### 3.4 BBox 验证

GPU1 `simple` 90 帧 gate：

- PASS。
- frame_count: 90
- first_negative_y_gap: 32
- tetra_overlap_frame_count: 0
- gravity_estimate_y: -9.80025
- post_contact_com_delta: 1.11637
- post_contact_normal_angle_deg: 45.5142

`slope30`：

- candidate sum: 613
- candidate max: 7
- PCG mean: 7.62
- Newton mean: 1.66

`stack120`：

- candidate sum: 22799
- candidate max: 77
- PCG mean: 10.86
- PCG max: 47
- Newton mean: 1.13

结论：

- bbox device reduction 未导致 broadphase 候选归零。
- `stack` 延长到 120 帧后候选正常出现，说明 30 帧候选为 0 是接触窗口未到，不是 bbox 回归。

### 3.5 H2D/D2H 统计

`simple` 90 帧，`UIPC_COREX_MEMCPY_STATS=1`：

- H2D: 2102 calls, 331024 bytes
- D2H: 10959 calls, 133512 bytes
- D2D: 150 calls, 24180 bytes

`stack` 120 帧，`UIPC_COREX_MEMCPY_STATS=1`：

- H2D: 2098 calls, 645828 bytes
- D2H: 11169 calls, 235276 bytes
- D2D: 277 calls, 470024 bytes

结论：

- H2D/D2H 调用频率高，但总字节量小。
- 全局恢复 H2D/D2H async 风险大于收益。
- 更合理的策略是后续给 call site 加标签或局部 wrapper，逐点证明生命周期安全后再恢复 async。

## 4. Wrecking Ball 400 帧结果

运行命令：

```bash
./corex_demo --backend cuda --scene wrecking_ball --frames 400 --gpu 1 --output_dir artifacts/wrecking_ball400/frames
```

输出：

- 日志：`artifacts/wrecking_ball400/run.log`
- OBJ：`artifacts/wrecking_ball400/frames`
- profile：`artifacts/wrecking_ball400/profile.json`

结果：

- 运行成功，退出码 0。
- 起始时间：11:05:32.887
- 结束时间：11:14:10.337
- 总耗时：517.45s，约 8 分 37.5 秒。

与上一轮结果对比：

- 上一轮：526.41s
- 本轮：517.45s
- 改善：8.96s
- 相对提升：约 1.7%

Profile 统计：

- PCG count: 2633
- PCG mean: 64.33
- PCG max: 178
- PCG sum: 169383
- Newton count: 397
- Newton mean: 5.13
- Newton max: 22
- unique triplets mean: 20870.52
- unique triplets max: 29254
- simplex candidates mean: 17088.31
- simplex candidates max: 22016
- 最后一帧 `399` 正常收敛，advance 为 1208ms，write OBJ 为 32ms。

## 5. 性能结论

本轮改动对 `wrecking_ball400` 的端到端提升有限，约 1.7%。这说明：

- dot/norm host fallback 替代是正确的稳定性清障，但不是 `wrecking_ball400` 主瓶颈。
- bbox 全量 D2H 替代消除了不合理路径，但对整体耗时贡献较小。
- H2D/D2H 当前是高频小字节，不能解释 400 帧秒级耗时。
- `wrecking_ball400` 真正被放大的路径是：
  - PCG 总迭代数和每迭代成本。
  - SpMV。
  - preconditioner apply。
  - GlobalLinearSystem 同步链。
  - 高候选接触与 DyTopo assembly。

因此，本轮改动属于“清障与风险消除”，不是主要性能突破。

## 6. 风险与稳定性评估

### 数值稳定性

- dot/norm device fallback 使用 tree reduction，不保证与 CPU 顺序累加 bitwise 一致。
- forced device fallback 下 `simple`、`stack`、`wrecking_ball` 短帧统计与 default 基本一致，未观察到 PCG/Newton 行为异常。
- `simple` 90/300 gate 均通过，说明物理门禁未回归。

### 回退能力

保留以下回退/诊断开关：

- `UIPC_COREX_DOTNORM_HOST_FALLBACK=1`
- `UIPC_COREX_DOTNORM_FORCE_DEVICE_FALLBACK=1`
- `UIPC_COREX_TRACE_DOTNORM_FALLBACK=1`
- `UIPC_COREX_BBOX_HOST_FALLBACK=1`
- `UIPC_COREX_MEMCPY_STATS=1`
- `UIPC_COREX_TRACE_MEMCPY=1`

### 未处理风险

- `GlobalLinearSystem` 中仍有同步点，没有在本轮直接删除。
- `linear_pcg_corex.cu.inc` 中 SpMV 后仍存在 CoreX 分支同步。
- 接触候选与 DyTopo assembly 尚未做阶段化计时。

## 7. 下一轮建议

下一轮不建议继续把主要精力放在 host fallback 替代上，应转向 `wrecking_ball400` 的真实主瓶颈：

1. 补充 per-frame/per-phase profiling。
2. 优先处理 PCG/SpMV 同步与每迭代固定成本。
3. 审计并逐点替换 `GlobalLinearSystem` 中的全设备同步为 stream/event 依赖。
4. 统计并优化 `apply_preconditioner()` 成本。
5. 细分 stackless BVH、filter_active、filter_toi、DyTopo extent/assemble/convert/distribute 的耗时。
6. 对 `wrecking_ball` 做 150 帧 A/B，再做 400 帧最终验证。

验收指标应包括：

- 总耗时。
- per-frame p50/p95/max。
- PCG sum/mean/max。
- Newton mean/max。
- unique triplets 分布。
- simplex candidates 分布。
- `simple` gate 是否通过。
- `wrecking_ball` 最终是否正常收敛。

## 8. 产物索引

主要代码改动：

- `external/muda/src/muda/ext/linear_system/details/routines/dot.inl`
- `external/muda/src/muda/ext/linear_system/details/routines/norm.inl`
- `src/backends/cuda/global_geometry/global_vertex_manager.cu`
- `external/muda/src/muda/launch/details/memory.inl`

主要验证与审计产物：

- `artifacts/host_fallback_dotnorm/simple90_profile.json`
- `artifacts/host_fallback_dotnorm/simple300_profile.json`
- `artifacts/host_fallback_dotnorm/simple90_forced_device_profile.json`
- `artifacts/host_fallback_bbox/simple90_report.json`
- `artifacts/host_fallback_bbox/stack120_profile.json`
- `artifacts/host_fallback_memory/simple90_profile.json`
- `artifacts/host_fallback_memory/stack120_profile.json`
- `artifacts/host_fallback_dotnorm/global_linear_system_sync_audit.md`
- `artifacts/host_fallback_memory/selective_h2d_d2h_async_audit.md`
- `artifacts/wrecking_ball400/run.log`
- `artifacts/wrecking_ball400/profile.json`

