# libuipc 天数（CoreX）GPU 单精度仿真适配工作总结

## 一、项目概述

将 `libuipc` 物理仿真库（基于 IPC — Incremental Potential Contact 方法）从 NVIDIA GPU 移植到天数智芯（Tianshu / CoreX）GPU 环境，实现单精度（`float`）模式下的完整物理仿真。

### 硬件与软件环境

| 项目 | 规格 |
|------|------|
| GPU | Iluvatar MR-V100（天数智芯），32GB 显存，8 卡 |
| CUDA 兼容层 | CoreX CUDA 10.2 |
| 编译器 | CoreX clang-CUDA（ivcore11 模式） |
| 主机编译器 | g++-13 |
| 精度模式 | `UIPC_USE_FLOAT=ON`（单精度 `float`） |

### CoreX 平台关键约束

1. **双精度 FPU 缺陷**：CoreX 双精度运算的有效尾数仅约 23 位（等效于 float），导致 `double` 四则运算、`log`、`sqrt` 等函数输出精度异常。
2. **Float 运算正确**：单精度运算在 CoreX 上完全正确，因此项目采用 `UIPC_USE_FLOAT=ON` 构建。
3. **Device Lambda 失效**：`muda::ParallelFor` 等使用设备端 lambda 捕获的代码在 CoreX 上静默失败（编译通过但不执行）。
4. **Eigen 模板地址空间问题**：CoreX clang-CUDA 在 lambda 或 `__device__` 函数中传递 Eigen 固定大小矩阵时，会产生 `invalid addrspacecast` 错误。

---

## 二、适配工作阶段

### 阶段 1：编译通过

**目标**：让 `libuipc_backend_cuda.so` 在 CoreX 编译器下成功编译链接。

**主要修改**：
- 添加 `UIPC_COREX_CUDA10_COMPAT` 条件编译宏，隔离 CoreX 专用适配代码
- 修复 CoreX clang-CUDA 不支持的 C++20 特性（`__VA_OPT__`、`consteval`、结构化绑定在 constexpr 中的使用等）
- 替换 CUDA 11+ API（`cub::DeviceRadixSort` 签名差异、`cudaMemcpyDefault` 缺失等）
- 修复 Eigen 头文件与 CoreX 编译器的兼容性问题（`__host__ __device__` 资格冲突）
- 处理 `muda` 库中的 CUDA 版本检查和条件编译

**涉及文件**：约 50+ 文件的条件编译适配（详见 `corex-muda-adaptation-report-v3.md`）

---

### 阶段 2：基础仿真运行（无接触）

**目标**：在 contact 关闭的 simple 场景下，实现正确的自由落体仿真。

**问题与解决**：

| 问题 | 根因 | 解决方案 |
|------|------|----------|
| 仿真静止（第一帧后无运动） | `muda::ParallelFor` device lambda 静默失败 | 关键路径（PCG 求解器、全局线性系统、仿射体动力学、对角预条件器、线搜索）全部改写为 `__global__` kernel |
| 四面体瞬移 | `BufferLaunch().copy()` 底层依赖 ParallelFor 同样失败 | 替换为 `cudaMemcpy` / `cudaMemset` |
| Newton 不收敛 | 惯性梯度组装路径中 ParallelFor 失败 | 改写为 `__global__` kernel |

**涉及文件**：`linear_pcg.cu`、`global_linear_system.cu`、`abd_linear_subsystem.cu`、`abd_line_search_reporter.cu`、`abd_diag_preconditioner.cu` 等（详见 `corex-simulation-fix-report.md`）

---

### 阶段 3：碰撞检测与接触力

**目标**：开启 contact，实现碰撞检测和接触力组装。

**问题与解决**：

| 问题 | 根因 | 解决方案 |
|------|------|----------|
| BVH 碰撞检测输出 0 个碰撞对 | `stackless_bvh.inl` 中 BVH 构建的 ParallelFor 全部静默失败 | 13 个 BVH 构建步骤改写为 `__global__` kernel |
| 轨迹过滤器无输出 | `simplex_trajectory_filter.cu` 的 ParallelFor 失败 | 改写为 `__global__` kernel |
| CSR 矩阵转换挂死 | `TripletToCSR` 底层 ParallelFor 失败 | 替换为 cusparse + 显式 kernel |
| 接触力 kernel 挂死 | PP/PE/EE/PT 接触力组装使用复杂 device lambda | 改写为 4 个独立的 `__global__` kernel |

**涉及文件**：`stackless_bvh.inl`、`simplex_trajectory_filter.cu`、`ipc_simplex_normal_contact.cu`、`triplet_to_csr.cu` 等（详见 `corex-collision-contact-fix-report.md`、`corex-contact-force-fix-report.md`）

---

### 阶段 4：接触力精度修复

**目标**：修复 CoreX 双精度 FPU 缺陷导致的接触力计算错误。

**问题与解决**：

| 问题 | 根因 | 解决方案 |
|------|------|----------|
| Barrier 能量 `KappaBarrier` 返回异常值 | `log(s/V)` 中 CoreX 双精度除法和 `log` 精度不足 | 实现 `safe_log`（基于 `atanh` 级数展开）和 `safe_log_ratio`（避免小数除法） |
| 距离梯度 `GradD` 错误 | CoreX 双精度解析导数不可靠 | 改用中心差分数值梯度 |
| `evd_jacobi` 特征值分解输出垃圾 | Jacobi 迭代中双精度运算精度不足 | 改用 Gershgorin 圆盘定理的对角位移（仅 double 路径） |

> **注意**：这些修复针对的是 `UIPC_FLOAT_SCALAR=0`（双精度）路径。在当前 `UIPC_USE_FLOAT=ON` 构建中，这些代码路径被 `#if` 条件编译跳过，因为 float 运算在 CoreX 上是正确的。

**涉及文件**：`codim_ipc_contact.inl`、`codim_ipc_simplex_normal_contact_function.h`（详见 `corex-contact-force-fix-report.md`）

---

### 阶段 5：运行时环境与稳定性

**目标**：解决运行时设备选择、初始化等问题。

**问题与解决**：

| 问题 | 根因 | 解决方案 |
|------|------|----------|
| `cudaStreamCreate` / `cudaMalloc` 失败 | 默认 GPU 0 异常 | 通过 `--gpu` 参数切换到正常设备（如 GPU 1 / GPU 7） |
| `corex_demo` 初始化卡住 | 设备路径异常 + 动态库加载顺序 | 显式指定 GPU 设备号 |

**涉及文件**：`corex_demo/main.cpp`、启动脚本（详见 `corex-runtime-diagnosis-2026-04-14.md`）

---

### 阶段 6：接触后悬空（hover）问题修复 ★

**目标**：解决 simple 场景中物体接触后非物理地悬空冻结的问题。

#### 现象

在 simple 场景（两个四面体，一个自由落体撞击一个固定）中：
- 自由落体阶段正常
- 接触后物体冻结在接触位置，COM 变化 < 0.006（51 帧内）
- 法向量旋转仅 0.02°（几乎为零）
- 无穿透，但也无滑动/弹跳

#### 根因分析

**最初假设（已排除）**：barrier 梯度的有限差分步长 `eps` 超出 barrier 域宽度导致梯度错误。

经分析发现：由于 `UIPC_USE_FLOAT=ON`（`UIPC_FLOAT_SCALAR=1`），barrier 的 CoreX 数值微分路径被 `#if !(defined(UIPC_FLOAT_SCALAR) && UIPC_FLOAT_SCALAR)` 跳过，实际走的是标准解析路径。**Float 解析 barrier 梯度在 CoreX 上是正确的。**

**真正的根因**：`make_spd.h` 中的 Hessian 半正定投影。

```
文件: src/backends/cuda/utils/make_spd.h

旧代码守卫:
  #if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
      → Gershgorin 对角位移（不区分 float/double）
  #else
      → EVD 特征值投影
```

`make_spd` 的 CoreX 守卫**只检查 `UIPC_COREX_CUDA10_COMPAT`，不检查 `UIPC_FLOAT_SCALAR`**。即使在 float 构建中（float 运算在 CoreX 上完全正确），仍然使用 Gershgorin 对角位移而非 EVD 特征值投影。

**Gershgorin vs EVD 的物理影响**：

| 方法 | 处理方式 | 物理效果 |
|------|----------|----------|
| EVD 投影 | 将负特征值投影为 0，保留正特征值不变 | 正确保留切向方向的自由度 |
| Gershgorin 位移 | 给所有对角元素加上均匀偏移 | **额外刚化切向方向**，阻止滑动/弹跳 |

Gershgorin 对角位移将所有特征值均匀增大，包括切向方向原本很小的正特征值。这导致 Newton 求解器在切向方向上的步长被严重压缩，物体无法产生切向运动（滑动或弹跳），表现为"悬空冻结"。

#### 解决方案

在 `make_spd.h` 中为 CoreX + float 路径添加内联 Jacobi EVD：

```
新代码守卫:
  #if defined(UIPC_COREX_CUDA10_COMPAT)
    #if defined(UIPC_FLOAT_SCALAR)
        → 内联 Jacobi EVD（float 运算正确，EVD 可正常收敛）
    #else
        → Gershgorin 对角位移（double 运算不可靠，EVD 会发散）
    #endif
  #else
      → Eigen SelfAdjointEigenSolver（标准 NVIDIA 路径）
  #endif
```

**实现要点**：
- 使用原始 C 数组 `Float E[N][N]` 代替 Eigen 矩阵，避免 CoreX clang-CUDA 对 Eigen 表达式模板产生 `invalid addrspacecast`
- 收敛容差使用相对值 `diag_norm * 1e-4`
- 特征值投影：将负特征值设为 0，然后重构 `H = E * diag(evals) * E^T`
- 所有矩阵运算用逐元素循环实现，不调用 Eigen 成员函数

#### 修复效果

| 指标 | 修复前（Gershgorin） | 修复后（EVD Jacobi） |
|------|---------------------|---------------------|
| 接触后 COM 变化（51 帧） | 0.006（冻结） | **2.245**（活跃运动） |
| 接触减速 vs 自由落体 | N/A（冻结） | +0.98m（正确的接触脉冲） |
| 横向偏转 | 0（冻结） | +1.04m（正确的弹偏） |
| 法向旋转 | 0.02°（冻结） | **41.5°**（正确的旋转） |
| 穿透 | 无 | 无 |
| Newton 最大迭代退出 | 0 次 | 0 次 |

**300 帧长时间测试验证**：
- 重力加速度 Ay ≈ -9.77（理论值 -9.8，误差 < 0.3%）
- 水平速度脱离接触后保持恒定 ≈ 2.52 m/s
- 无 Newton 发散、无穿透、无非物理行为

---

## 三、修改文件汇总

### 核心修改（本轮 hover 修复）

| 文件 | 修改内容 |
|------|----------|
| `src/backends/cuda/utils/make_spd.h` | CoreX + float 路径使用内联 Jacobi EVD 替代 Gershgorin |
| `src/backends/cuda/contact_system/contact_models/sym/codim_ipc_contact.inl` | barrier 有限差分步长域感知修复（double 路径，float 构建中未激活） |
| `src/backends/cuda/contact_system/contact_models/ipc_simplex_normal_contact.cu` | 添加 `UIPC_COREX_TRACE_BARRIER_DBDD` 运行时诊断 |

### 历史修改（此前各阶段累计）

| 类别 | 文件数 | 主要修改 |
|------|--------|----------|
| 编译适配（muda / Eigen / CUDA API） | ~50 | 条件编译、API 替换、模板特化 |
| ParallelFor → `__global__` kernel | ~20 | PCG、线性系统、BVH、轨迹过滤器、接触力 |
| BufferLaunch → cudaMemcpy | ~10 | 内存操作替换 |
| FPU 精度变通（double 路径） | ~5 | safe_log、数值梯度、Gershgorin |
| 运行时 / 场景配置 | ~3 | GPU 选择、参数调优入口 |

---

## 四、当前状态

### 已验证通过

- **simple 场景**（两个四面体）：自由落体 → 接触弹偏 → 重力下抛物线运动，300 帧物理行为正确
- Newton 求解器收敛正常（无最大迭代退出）
- 无穿透
- 重力加速度精度 > 99.7%

### 待验证

- 更复杂的场景（多体、大规模网格）
- 摩擦接触（`friction.enable = true`）
- 长时间稳定性（> 1000 帧）
- 性能基准测试

---

## 五、技术经验总结

1. **CoreX 双精度不可信，float 可信**：所有核心数值路径必须在 `UIPC_USE_FLOAT=ON` 下工作。双精度代码路径保留但需要独立的精度变通（safe_log、数值微分等）。

2. **Device Lambda 是 CoreX 上的主要移植障碍**：`muda::ParallelFor` 的设备端 lambda 在 CoreX 上静默失败，是整个移植工作中改动量最大的部分。核心路径必须改写为 `__global__` kernel。

3. **Eigen 模板操作在 CoreX device 代码中需谨慎**：`setIdentity()`、`asDiagonal()`、`transpose()` 等 Eigen 表达式模板会触发 `invalid addrspacecast`，需改用原始元素访问。

4. **条件编译守卫必须同时检查精度模式**：`UIPC_COREX_CUDA10_COMPAT` 单独不够，必须与 `UIPC_FLOAT_SCALAR` 组合使用。CoreX double 和 CoreX float 是完全不同的精度环境，需要不同的代码路径。

5. **Gershgorin 对角位移不能替代 EVD 特征值投影**：两者在半正定投影中的物理效果截然不同。Gershgorin 会额外刚化切向方向，在接触场景中导致非物理的悬空/冻结现象。
