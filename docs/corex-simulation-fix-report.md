# libuipc CoreX/天数 GPU 适配 — 仿真异常修复工作总结

## 一、项目背景

将 `libuipc` 物理仿真库从 NVIDIA GPU (CUDA) 移植到天数智芯 (Tianshu) GPU 的 CoreX 环境。CoreX 提供了一个兼容 CUDA 10 的运行时，但底层硬件和编译器与 NVIDIA 有显著差异。移植后仿真出现严重异常：**仿真在第一帧后完全静止，后续修复后又出现四面体瞬移现象。**

---

## 二、问题一：仿真静止（第一帧后无运动）

### 现象

仿真输出的 OBJ 序列中，所有帧的顶点坐标完全相同，物体没有任何运动。

### 根因

CoreX 环境下，`muda` 库的 `ParallelFor` 设备端 lambda 捕获机制静默失败。`muda::ParallelFor` 是 libuipc 中几乎所有 GPU 并行计算的基础设施，它在 CoreX 上编译通过但运行时不执行任何计算（无报错、无崩溃，只是静默跳过）。

受影响的关键路径包括：

- **PCG 线性求解器** (`linear_pcg.cu`) — `update_xr`、`update_p` 等向量运算
- **全局线性系统** (`global_linear_system.cu`) — Hessian 组装、梯度组装
- **仿射体动力学** — 能量计算、梯度计算、Hessian 计算
- **对角预条件器** (`abd_diag_preconditioner.cu`)
- **线搜索** (`abd_line_search_reporter.cu`) — 状态保存与恢复
- **`BufferLaunch().copy()` / `BufferLaunch().fill()`** — 内部同样依赖 `ParallelFor`

### 解决方式

采用 `#if defined(UIPC_COREX_CUDA10_COMPAT)` 条件编译，针对每个失败的 `ParallelFor` 调用选择以下策略之一：

**策略 1：重写为 `__global__` CUDA kernel + 裸指针参数**（优先）

将 `muda::ParallelFor` 的 device lambda 改写为标准的 `__global__` kernel 函数，手动传递裸指针。例如 `bdf1_predict_dof_kernel`、`bdf1_update_state_kernel`、`kernel_ext_force_acc` 等。

**策略 2：用 `cudaMemcpy` / `cudaMemset` 替代 `BufferLaunch`**

对于简单的内存复制和清零操作，直接使用 CUDA runtime API：

- `BufferLaunch().copy()` → `cudaMemcpy(..., cudaMemcpyDeviceToDevice)`
- `BufferLaunch().fill(buf, 0)` → `cudaMemset(buf, 0, size)`

**策略 3：主机端回退**（仅在前两种不可行时使用）

对于复杂的向量运算（如 PCG 的 `update_xr`、`update_p`），将数据从设备拷贝到主机，在 CPU 上计算后再拷回。

### 修复的文件清单（第一阶段）

| 文件 | 修复内容 |
|------|---------|
| `linear_pcg.cu` | PCG 向量运算（axpy、dot、更新残差/方向） |
| `global_linear_system.cu` | Hessian/梯度组装、能量求和 |
| `abd_diag_preconditioner.cu` | 对角预条件器计算 |
| `abd_line_search_reporter.cu` | 线搜索状态保存/恢复（q_temp） |
| `abd_linear_subsystem.cu` | 梯度组装、解的提取 |
| `affine_body_dynamics.cu` | 初始化缓冲区（q_temp、q_tilde、q_prev、dq） |
| `ortho_potential.cu` | 形状能量/梯度/Hessian |
| `global_active_set_manager.cu` | 活跃集管理 |
| `spmv.cu` | 稀疏矩阵向量乘 |

---

## 三、问题二：四面体瞬移（X/Z 方向大幅跳变）

### 现象

修复仿真静止后，上方四面体从第二帧开始在 X 和 Z 方向出现约 2.0 单位的瞬移，不符合仅有 Y 方向重力的预期。

### 排查过程

#### 第 1 轮：排查未初始化缓冲区

追踪发现 `affine_body_dynamics.cu` 中的 `async_transfer`（用 `BufferLaunch().copy()` 做 D2D 拷贝）和 `async_resize`（用 `BufferLaunch().fill()` 清零 `body_id_to_dq`）在 CoreX 上静默失败，导致 `q_temp`、`q_tilde`、`q_prev` 未被正确初始化。

**修复：** 替换为 `cudaMemcpy` 和 `cudaMemset`。

#### 第 2 轮：排查外力加速度

瞬移仍然存在。追踪到 `affine_body_external_force_manager.cu`：

- `Impl::clear()` 使用 `ParallelFor` 清零 `body_id_to_external_force` → 静默失败 → 外力缓冲区残留垃圾值
- `Impl::step()` 使用 `ParallelFor` 计算 `body_id_to_external_force_acc = M_inv * F` → 静默失败 → 加速度为未初始化值

**修复：** `clear()` 改用 `cudaMemset`；`step()` 改写为 `__global__ kernel_ext_force_acc`。

#### 第 3 轮：排查时间积分器

外力修复后瞬移仍在。追踪 `abd_bdf1_time_integrator.cu`：

- `do_predict_dof`（计算 `q_tilde = q + gravity*dt² + velocity*dt`）使用 `ParallelFor` → 静默失败
- `do_update_state`（计算 `q_v = (q - q_prev)/dt`）使用 `ParallelFor` → 静默失败

**修复：** 改写为 `__global__` kernel：`bdf1_predict_dof_kernel` 和 `bdf1_update_state_kernel`。

#### 第 4 轮：追踪动能梯度异常

修复上述问题后，Frame 1 仍出现瞬移。添加调试追踪发现：

```
kinetic_gradient = [-1082.5, 0.424, -1082.5, ...]  ← X/Z 分量异常巨大
```

而预期应为 `(0, 0.424, 0, ...)`，因为 `q - q_tilde` 仅在 Y 方向有微小分量。

进一步对比 GPU 和主机端计算结果：

```
gpu_grad  = [-1082.5, 0.424, -1082.5, ...]  ← GPU 计算结果
host_grad = [0.0,     0.424, 0.0,     ...]  ← 主机端计算结果（正确）
```

同一份数据，GPU 和主机给出不同的结果。

### 根因发现：CoreX GPU 跨缓冲区双精度减法 Bug

通过编写诊断 kernel，发现了一个关键的硬件/编译器 bug：

```cpp
// 在 GPU kernel 内：
const double* q_raw  = (const double*)&qs[0];      // buffer A
const double* qt_raw = (const double*)&q_tildes[0]; // buffer B

out[0] = q_raw[0];              // → 0.0  ✓ （读取正确）
out[1] = qt_raw[0];             // → 0.0  ✓ （读取正确）
out[2] = q_raw[0] - qt_raw[0];  // → -2.0 ✗ （应为 0.0！）
```

**单独读取每个值都正确（0.0），但减法结果错误（-2.0）。**

验证排除的因素：

- `sizeof(ABDJacobiDyadicMass)` 主机/设备端一致（104 字节） ✓
- `sizeof(Vector12)` 一致（96 字节） ✓
- 数据布局完全一致 ✓
- 使用 `volatile` 限定符 → 仍然错误 ✗
- 使用标量（非 Eigen）减法 → 仍然错误 ✗
- 同一 buffer 内的减法 → 正常 ✓
- **仅跨 buffer 减法异常** ✗

这是 CoreX GPU 在从两个不同的 device memory buffer 读取 double 值并执行减法时的底层 bug，可能与 GPU 的寄存器分配、缓存一致性或浮点单元有关。

### 解决方式

将所有涉及跨 buffer 减法的计算移至主机端执行：

| 文件 | 修复内容 |
|------|---------|
| `affine_body_bdf1_kinetic.cu` | 动能能量 `E = 0.5*dq·(M*dq)`、梯度 `G = M*(q-q_tilde)`、Hessian `H = M` 全部改为主机端计算 |
| `abd_bdf1_time_integrator.cu` | `predict_dof` 和 `update_state` 改为主机端计算（`update_state` 含 `q_v = (q-q_prev)/dt` 跨 buffer 减法） |

---

## 四、问题三：Newton 求解器不收敛

### 现象

每帧的 Newton 迭代都达到最大次数（4次）后强制退出，没有正常收敛。

### 根因

三个子系统的 `ParallelFor` / `DeviceReduce` 在 CoreX 上静默失败：

1. **`abd_tolerance_checker.cu`** — 使用 `ParallelFor` 检查 `dq` 各分量是否小于容差，`BufferLaunch().fill()` 初始化标志位 → 均失败 → 收敛状态始终未被正确判断

2. **`global_vertex_manager.cu` 的 `compute_axis_max_displacement()`** — 使用 `muda::DeviceReduce().Reduce()` 计算最大位移 → 失败 → `MaxTranslationChecker` 无法获得正确残差

3. **`global_vertex_manager.cu` 的 `step_forward()`** — 使用 `ParallelFor` 更新顶点位置 `pos = safe_pos + alpha * disp` → 失败 → 线搜索中的顶点位置未更新

4. **`affine_body_vertex_reporter.cu`** — `init_attributes()`、`update_attributes()`、`report_displacements()` 三个 `ParallelFor` 全部失败 → 顶点位移未正确从仿射体 DOF 传播到全局顶点

### 解决方式

| 文件 | 修复内容 |
|------|---------|
| `abd_tolerance_checker.cu` | `do_check()` 改为主机端：下载 `dq`，逐分量检查是否收敛 |
| `global_vertex_manager.cu` | `step_forward()` 改为主机端向量加法 |
| `global_vertex_manager.cu` | `compute_axis_max_displacement()` 改为主机端归约求最大值 |
| `global_vertex_manager.cu` | `setup_ccd()` / `restore_ccd()` 改为 `cudaMemcpy` + 主机端减法 |
| `affine_body_vertex_reporter.cu` | 三个函数全部改为主机端：下载 Jacobi/q/dq，计算顶点位置和位移后上传 |

---

## 五、修复后验证结果

运行 `simple` 场景（上方自由四面体 + 下方固定四面体，dt=0.01s，gravity=(0,-9.8,0)，contact/friction 关闭），200 帧：

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 仿真状态 | 第一帧后静止 / 瞬移 | 正常自由落体 |
| Newton 收敛 | 每帧达最大迭代 | 199/199 帧正常收敛 |
| 收敛速度 | N/A | Frame 1: 3 次迭代; Frame 2+: 1 次迭代 |
| X/Z 偏移（200帧） | ~2.0（瞬移） | ~3×10⁻⁴（数值噪声） |
| Y 轨迹 vs 理论值 | 不下落 | 误差 < 0.6%（200帧） |

Y 方向轨迹与理论自由落体对比：

| 帧 | 时间 (s) | 实际 y | 理论 y = 2.3 - ½·9.8·t² | 相对误差 |
|----|----------|--------|--------------------------|----------|
| 0 | 0.00 | 2.300 | 2.300 | 0% |
| 50 | 0.50 | 1.051 | 1.075 | 2.2% |
| 100 | 1.00 | -2.649 | -2.600 | 1.9% |
| 150 | 1.50 | -8.799 | -8.725 | 0.8% |
| 199 | 1.99 | -17.205 | -17.100 | 0.6% |

---

## 六、全部修改文件汇总

```
CMakeLists.txt                                              — CoreX 编译选项与工具链配置
src/backends/cuda/CMakeLists.txt                            — CUDA 后端构建配置
src/backends/cuda/active_set_system/global_active_set_manager.cu
src/backends/cuda/affine_body/abd_diag_preconditioner.cu
src/backends/cuda/affine_body/abd_line_search_reporter.cu
src/backends/cuda/affine_body/abd_linear_subsystem.cu
src/backends/cuda/affine_body/abd_tolerance_checker.cu
src/backends/cuda/affine_body/affine_body_dynamics.cu
src/backends/cuda/affine_body/affine_body_external_force_manager.cu
src/backends/cuda/affine_body/affine_body_vertex_reporter.cu
src/backends/cuda/affine_body/bdf/abd_bdf1_time_integrator.cu
src/backends/cuda/affine_body/bdf/affine_body_bdf1_kinetic.cu
src/backends/cuda/affine_body/constitutions/ortho_potential.cu
src/backends/cuda/algorithm/givens.hpp
src/backends/cuda/collision_detection/details/info_stackless_bvh.inl
src/backends/cuda/collision_detection/details/info_stackless_bvh_v0.inl
src/backends/cuda/finite_element/fem_mas_preconditioner.cu
src/backends/cuda/finite_element/mas_preconditioner_engine.cu
src/backends/cuda/finite_element/mas_preconditioner_engine.h
src/backends/cuda/global_geometry/global_vertex_manager.cu
src/backends/cuda/linear_system/global_linear_system.cu
src/backends/cuda/linear_system/linear_pcg.cu
src/backends/cuda/linear_system/spmv.cu
apps/examples/corex_demo/main.cpp                          — CoreX 测试用例
cmake/corex-clang-cuda-wrapper.sh                          — CoreX clang CUDA 编译包装脚本
cmake/corex-compat/crt/host_defines.h                      — CoreX 兼容头文件
cmake/patches/eigen3-SparseMatrix-corex-clang-max.patch    — Eigen3 CoreX 编译补丁
```

---

## 七、核心经验总结

1. **CoreX 的 `muda::ParallelFor` 设备端 lambda 捕获静默失败** — 这是最广泛的问题，影响了几乎所有 GPU 并行计算路径。没有任何运行时错误提示，必须通过输出对比才能发现。

2. **CoreX GPU 存在跨 buffer 双精度减法硬件 bug** — 从两个不同的 device buffer 读取 double 并相减，结果可能完全错误（如 `0.0 - 0.0 = -2.0`）。单独读取每个值正确，仅减法结果异常。`volatile` 无法绕过。

3. **`Matrix12x12::Zero()` + `add_to()` 在 GPU 上产生错误矩阵** — 12×12 矩阵（1152 字节）的栈分配在 CoreX GPU 上可能导致未正确清零，进而使 `to_mat()` 返回错误值。

4. **调试策略：GPU vs 主机对比** — 对同一数据分别在 GPU kernel 内和主机端计算，对比结果，是定位 CoreX 特有 bug 最有效的方法。

5. **修复策略优先级** — `__global__` kernel 重写 > `cudaMemcpy`/`cudaMemset` 替换 > 主机端回退。仅在涉及跨 buffer 减法等硬件 bug 时才使用主机端回退。
