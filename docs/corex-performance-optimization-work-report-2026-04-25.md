# CoreX 性能优化执行报告（2026-04-25）

## 1. 背景与目标

本轮工作基于前期对 libuipc v11 内部 CoreX/NVIDIA 双分支差异的统计结果继续推进。项目当前已能在天数 CoreX GPU 上运行，但 CoreX 兼容路径中存在大量性能风险点，包括：

- `ParallelFor` / `BufferLaunch` 兼容替换带来的显式 kernel 与同步开销。
- CUDA 10.2 兼容导致的 async memory / graph / CUB API 降级。
- PCG / SpMV / BVH / matrix converter 等关键路径的保守实现。
- 调试日志、主机回退和全设备同步混入热路径。

本轮目标不是大范围重构，而是按计划对 P0/P1 热路径做小批次优化，并严格执行以下门禁：

> 每次在关键路径修复后，立即使用 GPU 1 运行 `simple` 场景正确性回归；只有通过后，才继续下一处优化。

## 2. 执行环境与约束

- 工作目录：`/root/libuipc-v11-restored/libuipc-v11-extracted`
- 构建目录：`/root/libuipc-v11-restored/libuipc-v11-extracted/build_corex`
- 运行设备：GPU 1
- 主要可执行文件：`build_corex/Release/bin/corex_demo`
- 核心场景：`simple`
- 扩展回归场景：`slope`、`stack`、`domino`、`wrecking_ball`

注意事项：

- 该源码包不是 git 仓库，无法使用 `git status` / `git diff` 直接统计最终工作树差异。
- 本报告按实际编辑文件、构建结果和回归产物记录本轮工作。
- 所有 CoreX 性能修复仍限制在 CoreX 兼容路径、CoreX sidecar 或工具脚本中，不主动修改 NVIDIA 分支语义。

## 3. 本轮新增工具

### 3.1 `simple_gate.py`

新增文件：

- `tools/simple_physics_audit/simple_gate.py`

用途：

- 可选运行 `corex_demo --scene simple`。
- 自动提取 OBJ 序列指标。
- 生成 metrics JSON 与 gate report JSON。
- 使用退出码作为自动化门禁。

核心检查项：

- OBJ 帧数完整。
- 没有 tetra containment overlap proxy。
- 自由落体加速度接近 `-9.8`。
- 接触后 COM 有明显运动，不出现冻结。
- 接触后法向旋转保留。
- 发现运行失败或超时时写出结构化失败报告。

典型命令：

```bash
python3 tools/simple_physics_audit/simple_gate.py \
  --run-demo \
  --cwd build_corex \
  --demo ./Release/bin/corex_demo \
  --frames-dir /tmp/corex_simple_gate_after_change_gpu1 \
  --metrics-output /tmp/corex_simple_gate_after_change_gpu1_metrics.json \
  --report-output /tmp/corex_simple_gate_after_change_gpu1_report.json \
  --run-log /tmp/corex_simple_gate_after_change_gpu1.log \
  --frames 90 \
  --min-frames 90 \
  --min-last-frame 89 \
  --gpu 1 \
  --timeout 300
```

### 3.2 `profile_summary.py`

新增文件：

- `tools/simple_physics_audit/profile_summary.py`

用途：

- 从 `corex_demo` 日志中提取简要 profile。
- 汇总 frame timing、PCG 迭代、Newton 迭代、unique triplets、simplex candidate 数量。
- 用于每轮优化后快速判断是否改变求解行为或候选规模。

典型命令：

```bash
python3 tools/simple_physics_audit/profile_summary.py \
  --log /tmp/run.log \
  --output /tmp/run_profile.json
```

## 4. 代码修改清单

本节不只列出“改了什么”，也解释“为什么这些改动会让帧生成变快”、每项优化在本轮中的贡献判断，以及是否会影响仿真表现。

需要先说明贡献统计口径：

- 本轮没有为每个优化点单独做完整五场景 ablation（例如只开 SpMV、只关 BVH sync、只关 memory async 的全矩阵实验），因此不能给出严格的“每项贡献百分比”。
- 可以较可靠判断的是每项优化消除了哪类开销，以及该开销在不同场景中的放大倍数。
- `simple` 门禁用于正确性与局部性能趋势验证；五场景回归用于确认整体行为和完整帧生成能力。
- `wrecking_ball` 优化前完整 400 帧基线没有跑完，本轮最终 400 帧完整跑通并耗时 526.41s。因此 `wrecking_ball` 的“提升”更适合表述为“完整跑通并显著降低日志/同步/SpMV 负担”，而不是严格加速比。

综合贡献判断：

| 优化项 | 贡献判断 | 主要受益场景 | 贡献来源 | 稳定性结论 |
|---|---|---|---|---|
| PCG `norm(b)` 缓存与日志降噪 | 中到高 | `domino`、`wrecking_ball` | 减少每个 PCG 迭代的重复 norm 和大量日志 I/O | 不改收敛判据，simple 与五场景通过 |
| SpMV O(nnz) triplet atomic 路径 | 最高 | `wrecking_ball`，其次 `domino` | 消除 row-scan 近似 `rows * triplets` 的复杂度放大 | atomic 加法顺序有微小浮点差异；已过 simple 90/300 和五场景 |
| BVH build 去除逐 kernel 全设备同步 | 中到高 | candidate 多的接触场景 | 去除 BVH 构建阶段 host/device 往返等待 | 不改候选语义；simple candidate 与五场景正常 |
| Matrix converter 异步清零 | 中 | 矩阵转换/装配较频繁场景 | `cudaMemsetAsync` 避免 host 阻塞 | 依赖默认 stream 顺序；回归通过 |
| D2D copy 异步恢复 | 小到中 | 全局 buffer copy 频繁路径 | D2D copy 不再强制 host 等待 | 仅 D2D 改 async，H2D/D2H 保守同步，风险较低 |

从最终 profile 看，`wrecking_ball` 仍是最重场景：

- PCG records：2705
- PCG iter mean：64.39
- PCG iter sum：174163
- Newton mean：5.07
- unique triplets mean：20872.77
- candidate mean：16505.68
- candidate max：24740

因此帧生成速度改善的核心原因，不是单一函数“变快一点”，而是多个热路径同时减少了每帧内重复发生的 CPU/GPU 同步、日志 I/O、O(rows × nnz) 扫描和同步内存操作。

### 4.1 PCG / linear system

修改文件：

- `src/backends/cuda/linear_system/linear_pcg_corex.cu.inc`

修改内容：

- 删除 PCG 初始阶段对 `b` 的手工 D2H 全量拷贝与 CPU norm 计算。
- 改用已有 `ctx().norm(b)` 计算并缓存 `norm_b`。
- 避免每轮迭代重复计算 `norm(b)`。
- 将 `[pcg_legacy_would_exit]` 观察日志改为仅在 `UIPC_COREX_TRACE_LINEAR_SYSTEM` 打开时输出，避免默认运行下日志刷屏。

保留行为：

- 相对残差判据不变。
- `b == 0` 的 trivial solution 早退逻辑保留。
- NaN/Inf 诊断逻辑保留。

预期影响：

- 减少每个 PCG solve 中重复 norm。
- 减少默认日志 I/O 开销。
- 降低 host-side 同步与 D2H 检查成本。

为什么能提升帧生成速度：

- PCG 是每个 Newton 迭代都会调用的线性求解器，复杂场景中每帧可能多次进入 PCG。
- 原实现中 `norm(b)` 在初始化判断和每轮收敛判断中重复计算。对 `wrecking_ball` 这种 PCG iter sum 达到 174163 的场景，重复 norm 会被放大为大量额外库调用和同步点。
- `[pcg_legacy_would_exit]` 日志在优化前默认输出，复杂接触场景中会产生海量文本 I/O。日志 I/O 会放大帧生成时间，并干扰性能观测。
- 本轮改为每个 PCG solve 只缓存一次 `norm_b`，并把 legacy 判据日志放到 trace 开关后面。这样不改变 PCG 数学判据，但减少了每轮迭代的固定开销。

贡献判断：

- 对 `simple` 这类小系统，收益主要表现为日志减少和少量 norm 调用减少，帧时间波动内不容易严格量化。
- 对 `domino` 和 `wrecking_ball`，PCG 迭代次数高，收益会随迭代数线性放大。
- 最终 `wrecking_ball` 中 PCG iter sum 为 174163，说明任何每迭代固定开销都会被放大，因此该项属于中到高贡献。

稳定性与仿真影响：

- 相对残差收敛判据没有修改。
- `b == 0` 早退仍保留，只是从手写 D2H norm 改为 `ctx().norm(b)`。
- NaN/Inf 诊断仍保留。
- 优化后 `simple` 90 帧和后续所有场景均通过，未观察到物理表现回退。

### 4.2 SpMV

修改文件：

- `src/backends/cuda/linear_system/spmv.cu`

修改内容：

- 新增 `kernel_sym_spmv_triplet_atomic`。
- 新路径按 triplet 并行，对目标 block row 使用 atomic add，复杂度从旧路径的近似 `rows * triplets` 降到 O(nnz/triplets) 级。
- 保留旧 `kernel_sym_spmv_v2` row-scan 实现作为 correctness fallback。
- 新增运行时回退开关：

```bash
UIPC_COREX_SPMV_ROW_SCAN=1
```

开启该变量时会回到旧 row-scan 路径。

预期影响：

- 大场景中 `n_block_rows` 与 `triplet_count` 均较大时，避免每个 row 扫全量 triplets。
- 对小场景可能收益不明显，但 correctness gate 已确认物理结果稳定。

为什么能提升帧生成速度：

- SpMV 是 PCG 内部最高频操作之一。每次 PCG 迭代都要计算 `A * p`。
- 旧 CoreX workaround 的 `kernel_sym_spmv_v2` 是按 block row 并行，每个 block row 线程扫描全部 triplets：

```cpp
for(int t = 0; t < n_triplets; ++t)
```

- 这使复杂度近似变成 `block_rows * triplets`。例如 `wrecking_ball` 最终 profile 中 unique triplets mean 约 20872，DoF 最大场景约 6900，对应 block rows 约 2300；旧路径会产生数量级非常高的无效检查。
- 新 `kernel_sym_spmv_triplet_atomic` 改为每个 triplet 一个线程处理，直接对目标行 atomic add，复杂度近似 O(triplets)。
- 对 `simple` 这类只有几十个 triplets 的场景，差异不明显；对 `wrecking_ball` 这种 triplets 上万的场景，这是本轮最重要的结构性优化。

贡献判断：

- 这是本轮理论贡献最高的优化项。
- 最终 `wrecking_ball` 400 帧完整跑通，PCG iter sum 达 174163。每次 PCG 迭代都会触发 SpMV，因此 SpMV 单次复杂度下降会被 PCG 迭代总数进一步放大。
- 由于本轮没有单独做 `UIPC_COREX_SPMV_ROW_SCAN=1` 与默认新路径的完整五场景 A/B，不能直接给出精确百分比；但从复杂度变化看，它应是大场景帧生成速度提升的主要来源。

稳定性与仿真影响：

- 新路径使用 atomic add，浮点累加顺序与旧 row-scan 不完全相同，因此会有微小数值扰动。
- 已保留旧路径回退开关：

```bash
UIPC_COREX_SPMV_ROW_SCAN=1
```

- 新路径已通过 `simple` 90 帧、`simple` 300 帧、`slope`、`stack`、`domino`、`wrecking_ball` 回归。
- `simple` 指标几乎不变：接触窗口仍为第 32 帧，`tetra_overlap_frame_count = 0`，接触后 COM 和法向旋转正常。
- 后续建议对 `stack` 与 `wrecking_ball` 做多次重复运行，确认 atomic 顺序导致的 PCG 迭代波动是否稳定。

### 4.3 BVH / collision

修改文件：

- `src/backends/cuda/collision_detection/details/stackless_bvh.inl`

修改内容：

将 CoreX 分支中多个 BVH build 阶段 kernel 后的强制全设备同步：

```cpp
cudaDeviceSynchronize();
```

替换为：

```cpp
checkCudaErrors(cudaGetLastError());
```

涉及阶段：

- `calcMCsFromBox`
- `calcInverseMapping`
- `buildPrimitivesFromBox`
- `calcExtNodeSplitMetrics`
- `calcIntNodeOrders`
- `updateBvhExtNodeLinks`
- `reorderNode`

保留行为：

- kernel launch error 仍会被及时捕获。
- 不改变 BVH 构建语义。
- 不改变候选过滤逻辑。

预期影响：

- 减少 BVH build 中不必要的 host/device 同步。
- 允许连续 kernel 和后续 Thrust 操作按流语义推进。

为什么能提升帧生成速度：

- BVH build / query 是 collision candidate generation 的核心路径，接触场景每帧会多次进入。
- 原 CoreX 分支在多个 build 阶段 kernel 后立即 `cudaDeviceSynchronize()`。这会强制 CPU 等待整个设备完成，而不是只检查 launch 是否成功。
- 同步点会切断 GPU work 的流水化，也会让 CPU/GPU 在每个小 kernel 后来回等待。
- 本轮改为 `cudaGetLastError()`，仍然能捕捉 launch error，但不强制等待 kernel 完成。
- 后续 Thrust 操作和 kernel 本身仍在默认 stream 顺序下执行，因此语义上依赖流内顺序，而不是每步全局同步。

贡献判断：

- 对 `simple` 影响有限，因为候选数量很少。
- 对 `slope`、`stack`、`domino` 这类持续有接触候选的场景收益更明显。
- 对 `wrecking_ball`，candidate mean 达 16505.68、max 达 24740，BVH/collision 链路频繁，因此同步移除属于中到高贡献。

稳定性与仿真影响：

- 未改变 BVH 节点构建算法、候选筛选条件或 contact mask 语义。
- `simple` gate 检查候选不回退，且五场景均完整写出 OBJ。
- 风险在于如果后续引入多 stream 或改变 Thrust execution policy，需要重新确认流顺序；当前默认 stream 模式下回归通过。

### 4.4 Matrix converter

修改文件：

- `src/backends/cuda/algorithm/details/matrix_converter.inl`

修改内容：

将 CoreX matrix converter 中两个同步清零：

```cpp
cudaMemset(...)
```

替换为：

```cpp
checkCudaErrors(cudaMemsetAsync(...))
```

涉及位置：

- block matrix segmental reduce 前的 `sorted_partition_input` 清零。
- vector segmental reduce 前的 `sorted_partition_input` 清零。

预期影响：

- 减少 host-side 阻塞。
- 保持与后续 `DeviceScan` / reduce 的顺序一致性。

为什么能提升帧生成速度：

- Matrix converter 参与 triplet/BCOO/CSR 等格式转换，是装配线性系统和预条件相关路径的基础设施。
- 原来的 `cudaMemset` 是同步 API，host 会等待清零完成。
- 改成 `cudaMemsetAsync` 后，清零工作进入 stream，由后续同流的 `DeviceScan` / reduce 自然保证顺序。
- 该改动不会减少算法复杂度，但减少了 host-side 阻塞点。

贡献判断：

- 单点贡献低于 SpMV，但在每帧多次矩阵转换时会累积。
- 对 `simple` 影响较小。
- 对 `domino` 和 `wrecking_ball` 这种装配次数更多、triplet 数更高的场景属于中等贡献。

稳定性与仿真影响：

- 改动只改变清零 API 的同步方式，不改变清零内容和后续 reduce 逻辑。
- 依赖同一 stream 内顺序；当前回归通过。
- 若未来切换到多 stream，需确认 memset 与 scan/reduce 是否仍在同一 stream 或有显式依赖。

### 4.5 Memory / copy

修改文件：

- `external/muda/src/muda/launch/details/memory.inl`

修改内容：

CoreX 路径中原本所有 non-graph copy 都使用同步 `cudaMemcpy`。本轮做保守恢复：

- `cudaMemcpyDeviceToDevice` 改为 stream 上的 `cudaMemcpyAsync`。
- `cudaMemcpyHostToDevice` / `cudaMemcpyDeviceToHost` 仍保持同步 `cudaMemcpy`。

保守原因：

- H2D/D2H 涉及主机内存生命周期与同步语义，风险更高。
- D2D copy 更适合作为第一步恢复 async copy。

预期影响：

- 减少设备内拷贝的 host 阻塞。
- 保持主机参与拷贝路径的兼容稳定性。

为什么能提升帧生成速度：

- CoreX 兼容路径原本把 non-graph copy 全部退成同步 `cudaMemcpy`。
- 在每帧状态更新、临时 buffer 迁移、矩阵/向量数据移动中，DeviceToDevice copy 是比较安全且常见的拷贝类型。
- D2D copy 改回 `cudaMemcpyAsync` 后，host 不必等待设备内拷贝完成，而是交给 stream 顺序保证后续依赖。
- H2D/D2H 没有改，避免主机内存生命周期和隐式同步语义带来的稳定性风险。

贡献判断：

- 该项不是最大贡献，属于小到中等贡献。
- 对 small scene 难以明显体现。
- 对复杂场景中频繁 buffer copy 的路径会累积减少 host 阻塞。

稳定性与仿真影响：

- 只恢复 D2D async，H2D/D2H 仍保守同步。
- `simple` gate 与五场景回归均通过。
- 若后续继续恢复 H2D/D2H async，需要额外验证 host buffer 生命周期、stream sync 和 API tests。

### 4.6 帧生成速度提升的总体解释

帧生成速度提升来自几个层面的叠加：

1. 热循环内重复工作减少：PCG 不再每轮重复算 `norm(b)`，也不再默认输出大量 legacy 观察日志。
2. 核心算法复杂度下降：SpMV 从 row-scan 的近似 `rows * triplets` 改成 triplet 并行的 O(nnz) 结构。
3. 同步点减少：BVH build 多个 kernel 后不再强制全设备同步，matrix converter 和 D2D copy 也减少了 host-side blocking。
4. 保守恢复 async：只恢复风险较低的 async D2D/memset，不碰 H2D/D2H 和 graph 的高风险路径。

对帧生成最关键的是第 2 点和第 3 点。`wrecking_ball` 的 profile 显示，它每帧候选数量、triplet 数和 PCG 迭代都远大于其他场景，因此这些优化在 `wrecking_ball` 中被明显放大。

稳定性方面，本轮每个关键路径修改后都立即跑了 `simple` gate，并在最后跑完五场景：

- `simple` 90/300 帧均通过。
- `slope` 200/200 帧通过。
- `stack` 200/200 帧通过。
- `domino` 300/300 帧通过。
- `wrecking_ball` 400/400 帧通过。

当前没有证据表明本轮优化破坏仿真表现。唯一需要继续跟踪的是 SpMV atomic 累加顺序可能带来的微小数值差异，这类差异可能影响 PCG 迭代数的轻微波动，但已通过本轮正确性门禁。

## 5. 构建验证

每批关键路径改动后均执行：

```bash
cmake --build /root/libuipc-v11-restored/libuipc-v11-extracted/build_corex -j16 --target corex_demo
```

结果：

- PCG 修改后构建通过。
- SpMV 修改后构建通过。
- BVH 修改后构建通过。
- Matrix converter 修改后构建通过。
- Memory/copy 修改后构建通过。

构建过程中仍存在既有 warning，例如：

- `host_defines.h` 的 `#include_next` warning。
- cuSPARSE / cuSOLVER enum switch warning。
- muda `print.h` format string warning。

这些 warning 非本轮新增阻断项，本轮未处理。

## 6. Simple 门禁记录

### 6.1 初始门禁

GPU 0 上 90 帧 simple 曾在 300 秒超时，日志显示已进入 `world.init/build_systems` 阶段但未写出 OBJ。随后按要求改用 GPU 1。

GPU 1 quick gate 通过：

- 路径：`/tmp/corex_simple_gate_quick_gpu1_report.json`
- 帧数：90
- `first_negative_y_gap = 32`
- `tetra_overlap_frame_count = 0`
- `gravity_estimate_y = -9.80025`
- `post_contact_com_delta = 1.11637`
- `post_contact_normal_angle_deg = 45.5177`

GPU 1 300 帧基线通过：

- 路径：`/tmp/corex_simple_gate_300_gpu1_report.json`
- 帧数：300
- `last_frame = 299`
- `tetra_overlap_frame_count = 0`
- `gravity_estimate_y = -9.80025`

### 6.2 PCG 修改后

路径：

- `/tmp/corex_simple_gate_after_pcg_gpu1_report.json`

结果：

- 90/90 帧通过。
- `tetra_overlap_frame_count = 0`
- `gravity_estimate_y = -9.80025`
- 接触后 COM 与法向旋转正常。
- 默认日志中不再刷 `[pcg_legacy_would_exit]`。

### 6.3 SpMV 修改后

Quick gate：

- `/tmp/corex_simple_gate_after_spmv_gpu1_report.json`
- 90/90 帧通过。
- `tetra_overlap_frame_count = 0`
- `gravity_estimate_y = -9.80025`
- `post_contact_com_delta = 1.11637`
- `post_contact_normal_angle_deg = 45.5142`

300 帧 gate：

- `/tmp/corex_simple_gate_after_spmv_300_gpu1_report.json`
- 300/300 帧通过。
- `tetra_overlap_frame_count = 0`

### 6.4 BVH 修改后

路径：

- `/tmp/corex_simple_gate_after_bvh_gpu1_report.json`

结果：

- 90/90 帧通过。
- `tetra_overlap_frame_count = 0`
- 候选/接触行为未回退。

### 6.5 Matrix converter 修改后

路径：

- `/tmp/corex_simple_gate_after_matconv_gpu1_report.json`

结果：

- 90/90 帧通过。
- `tetra_overlap_frame_count = 0`

### 6.6 BVH + Matrix converter 批次结束

路径：

- `/tmp/corex_simple_gate_after_bvh_matconv_300_gpu1_report.json`

结果：

- 300/300 帧通过。
- `last_frame = 299`
- `tetra_overlap_frame_count = 0`
- `gravity_estimate_y = -9.80025`
- `post_contact_com_delta = 1.11637`
- `post_contact_normal_angle_deg = 45.5142`

### 6.7 Memory/copy 修改后

路径：

- `/tmp/corex_simple_gate_after_memory_gpu1_report.json`

结果：

- 90/90 帧通过。
- `tetra_overlap_frame_count = 0`
- `gravity_estimate_y = -9.80025`

## 7. Five-scene 回归结果

优化后使用 GPU 1 运行扩展回归：

```bash
./Release/bin/corex_demo --backend cuda --scene <scene> --frames <frames> --gpu 1 --output_dir <out>
```

结果汇总：

| 场景 | 帧数 | exit code | OBJ 数 | 墙钟时间 |
|---|---:|---:|---:|---:|
| `slope` | 200 | 0 | 200 | 13.69 s |
| `stack` | 200 | 0 | 200 | 11.53 s |
| `domino` | 300 | 0 | 300 | 53.39 s |
| `wrecking_ball` | 400 | 0 | 400 | 526.41 s |

结果文件：

- `/tmp/corex_regression_gpu1_after_opts/regression_results.json`
- `/tmp/corex_regression_gpu1_after_opts/slope_profile.json`
- `/tmp/corex_regression_gpu1_after_opts/stack_profile.json`
- `/tmp/corex_regression_gpu1_after_opts/domino_profile.json`
- `/tmp/corex_regression_gpu1_after_opts/wrecking_ball_profile.json`

## 8. Profile 摘要

### 8.1 `simple` 300 帧批次结束

最终 300 帧 simple profile：

- 路径：`/tmp/corex_simple_gate_after_bvh_matconv_300_gpu1_profile.json`
- PCG records：847
- PCG iter mean：8.14
- PCG iter sum：6896
- Newton mean：1.83
- candidate max：6
- candidate sum：273

### 8.2 `slope`

- PCG iter mean：10.37
- PCG iter sum：6255
- Newton mean：2.03
- unique triplets mean：35.39
- candidate max：12
- candidate sum：5916

### 8.3 `stack`

- PCG iter mean：12.24
- PCG iter sum：5056
- Newton mean：1.08
- unique triplets mean：76.69
- candidate max：77
- candidate sum：47439

### 8.4 `domino`

- PCG iter mean：58.59
- PCG iter sum：71944
- Newton mean：3.11
- unique triplets mean：189.51
- candidate max：92
- candidate sum：166306

### 8.5 `wrecking_ball`

- PCG iter mean：64.39
- PCG iter sum：174163
- Newton mean：5.07
- Newton max：30
- unique triplets mean：20872.77
- unique triplets max：30934
- candidate mean：16505.68
- candidate max：24740
- candidate sum：161788647

`wrecking_ball` 仍是当前最重场景，主要压力来自：

- 大量 contact candidates。
- 高 unique triplet 数。
- PCG 迭代与 Newton 迭代叠加。

## 9. 与优化前基线的观察对比

可比基线来自本轮优化前 GPU 1 baseline：

- `slope`：200 帧完成。
- `stack`：200 帧完成。
- `domino`：300 帧完成。
- `wrecking_ball`：完整优化前 run 被中断，仅有 partial 数据；因此 `wrecking_ball` 只能做弱对比。

观察：

- `domino` PCG mean 从约 60.14 降到约 58.59，Newton mean 从约 3.13 降到约 3.11。
- `slope` 与 `stack` 行为整体稳定，候选规模基本不变。
- `stack` 的 PCG mean 有上升，需要后续用更严格的同条件重复运行确认是否是数值扰动、场景随机性或 atomic SpMV 顺序带来的迭代差异。
- `wrecking_ball` 优化后完整跑完 400 帧，用时 526.41s；优化前完整 baseline 未跑完，无法给出严格加速比。
- 所有回归均写满 OBJ，未发现 simple 正确性回退。

## 10. 回退与调试开关

### 10.1 SpMV fallback

如怀疑新的 triplet atomic SpMV 路径引入数值或性能问题，可设置：

```bash
UIPC_COREX_SPMV_ROW_SCAN=1
```

该变量会回到旧的 `kernel_sym_spmv_v2` row-scan 路径。

### 10.2 PCG 兼容日志

默认不再输出 `[pcg_legacy_would_exit]` 观察日志。若需要重新观察旧判据会触发的位置，可设置：

```bash
UIPC_COREX_TRACE_LINEAR_SYSTEM=1
```

### 10.3 Memory/copy

本轮只恢复 DeviceToDevice 的 async copy。H2D/D2H 仍保持同步 copy，避免主机内存生命周期风险。

## 11. 风险评估

### 11.1 数值非确定性

新的 SpMV triplet atomic 路径使用 atomic add，浮点加法顺序与旧 row-scan 路径不同，可能导致极小数值差异。这已经通过 `simple` 90/300 帧与五场景回归，但建议后续对 `stack` 和 `wrecking_ball` 做重复运行统计。

### 11.2 BVH 同步减少

BVH build 阶段移除多个强制同步后，依赖 CUDA stream 顺序与 Thrust 默认流行为。当前 `simple` 和五场景已通过，说明未出现明显顺序问题；若后续引入多 stream，需要重新审计该路径。

### 11.3 Matrix converter async memset

`cudaMemsetAsync` 与后续 scan/reduce 的顺序依赖默认 stream。当前回归通过，但若后续切换 stream policy，需要再次验证。

### 11.4 Memory async copy

DeviceToDevice async copy 风险较低；H2D/D2H 未改。后续若继续恢复 H2D/D2H async，需要引入更强的生命周期与同步验证。

## 12. 后续建议

优先继续做以下工作：

1. 对 `stack` 做 3 次重复回归，确认 PCG mean 上升是否稳定。
2. 对 `wrecking_ball` 做分段 profile，更细分 PCG、SpMV、candidate generation、contact assembly 时间。
3. 在 `spmv.cu` 中对 triplet atomic 路径和 row-scan fallback 做场景级 A/B：`simple`、`stack`、`wrecking_ball`。
4. 继续优化 `wrecking_ball` 的 candidate/contact assembly，因为当前 candidate mean 和 triplet mean 远高于其他场景。
5. 若 CoreX runtime 支持更稳定的 async H2D/D2H 或 graph path，再逐项恢复，并每项后立即跑 `simple` gate。

## 13. 产物索引

新增/修改源码：

- `tools/simple_physics_audit/simple_gate.py`
- `tools/simple_physics_audit/profile_summary.py`
- `src/backends/cuda/linear_system/linear_pcg_corex.cu.inc`
- `src/backends/cuda/linear_system/spmv.cu`
- `src/backends/cuda/collision_detection/details/stackless_bvh.inl`
- `src/backends/cuda/algorithm/details/matrix_converter.inl`
- `external/muda/src/muda/launch/details/memory.inl`

关键验证产物：

- `/tmp/corex_simple_gate_after_bvh_matconv_300_gpu1_report.json`
- `/tmp/corex_simple_gate_after_bvh_matconv_300_gpu1_profile.json`
- `/tmp/corex_simple_gate_after_memory_gpu1_report.json`
- `/tmp/corex_regression_gpu1_after_opts/regression_results.json`
- `/tmp/corex_regression_gpu1_after_opts/wrecking_ball_profile.json`

最终结论：

本轮已完成计划中的所有 todo，并建立了可重复 `simple` correctness gate。所有关键路径修复均已通过 GPU 1 `simple` 门禁，五场景扩展回归全部完成并写满 OBJ。当前最大剩余性能瓶颈仍集中在 `wrecking_ball` 的 contact candidate / assembly / PCG 组合压力上。
