# Libuipc 天数 CoreX 平台适配工作总结

> 本文档对照原版代码（`/root/src/`）与适配后代码（`/root/libuipc/`），系统总结了 libuipc 从 NVIDIA GPU 移植到天数（Tianshu）CoreX 环境过程中所进行的全部代码调整，并按问题类别分类。

---

## 一、构建系统与工具链适配

### 1.1 问题描述

CoreX 使用基于 Clang 的 CUDA 前端（ILUVATAR 编译器），与 NVCC 在编译选项、设备链接、诊断信息等方面存在大量不兼容。

### 1.2 具体改动

#### 项目根 CMakeLists.txt

| 改动项 | 说明 |
|--------|------|
| 新增 `UIPC_MUDA_USE_COREX` 选项 | 总开关，启用 CoreX 兼容编译模式 |
| 新增 `UIPC_COREX_CUDA_STANDARD` | CoreX 下的 CUDA C++ 标准（默认 `"20"`） |
| 新增 `UIPC_COREX_CUDA_ARCHITECTURES` | CoreX 架构标识（默认 `"ivcore11"`） |
| 新增 `UIPC_COREX_CUDA_COMPILER` | 可选的 CoreX CUDA 编译器路径 |
| 新增 `UIPC_USE_FLOAT` 选项 | 单精度构建开关，在 CoreX 双精度不可靠时启用 |
| ILUVATAR depfile 修复 | 当 `CMAKE_CUDA_COMPILER_ID` 为 `ILUVATAR` 时，重写 `CMAKE_CUDA_COMPILE_OBJECT` 注入 `-MD -MT -MF` 以修复 Ninja 下增量编译 |
| PSTL 串行后端 | 定义 `_PSTL_PAR_BACKEND_SERIAL=1`、`_GLIBCXX_USE_TBB_PAR_BACKEND=0`，绕过 GCC10 + oneTBB 的 PSTL 兼容问题 |
| 强制关闭 RDC | CoreX 下设 `CMAKE_CUDA_SEPARABLE_COMPILATION OFF` |
| 架构传播逻辑调整 | CoreX 模式下不将 `UIPC_CUDA_ARCHITECTURES`（默认 `"native"`）传播给 CMake，避免破坏 CoreX clang |

**涉及文件**: `CMakeLists.txt`

#### CUDA 后端 CMakeLists.txt

| 原版 | 适配版 |
|------|--------|
| `cublas`、`cusparse`、`cusolver` 直接链接 | 改为 `CUDA::cublas`、`CUDA::cusparse`、`CUDA::cusolver`（CMake imported targets） |
| 链接 `uipc_io` | 移除 `uipc_io` 链接 |
| 无 `find_package(CUDAToolkit)` | 新增 `find_package(CUDAToolkit REQUIRED)` |
| `CUDA_SEPARABLE_COMPILATION ON`（始终） | 默认 OFF，仅 NVCC 时开启 |
| 无 CoreX 编译定义 | 新增 `UIPC_COREX_CUDA10_COMPAT=1`、`MUDA_FORCE_HD_GENERIC=1`、链接 `dl` |
| 单一编译选项块 | 分 Clang/ILUVATAR（`-x ivcore`、`--diag_suppress`）和 NVCC（`--expt-relaxed-constexpr`）两路 |

**涉及文件**: `src/backends/cuda/CMakeLists.txt`

#### 新增构建辅助文件

| 文件 | 用途 |
|------|------|
| `cmake/corex-clang-cuda-wrapper.sh` | 编译器包装脚本：过滤 `-rdc=true`、`-Xcudafe`，注入 `-x ivcore`、`--gcc-toolchain`、兼容头文件路径 |
| `cmake/corex-compat/crt/host_defines.h` | 覆盖 CoreX `crt/host_defines.h`，修复 `__noinline__` 与 libstdc++ 的冲突 |
| `cmake/patches/eigen3-SparseMatrix-corex-clang-max.patch` | Eigen 补丁：消除 CoreX Clang CUDA 下 `std::min`/`std::max` 歧义 |

---

## 二、CUDA API 版本兼容（CUDA 10.2 vs 11+）

### 2.1 问题描述

CoreX 运行时为 CUDA 10.2 级别，而 libuipc 依赖的 muda 库大量使用了 CUDA 11+ 才有的 API（异步内存分配、CUDA Graph 新接口、CUB ByKey 扫描、cuSolver 新参数 API 等）。

### 2.2 具体改动

#### CUDA Graph 降级

- **`external/muda/src/muda/graph/details/graph_exec.inl`**：`cudaGraphUpload`、`cudaGraphExecMemcpyNodeSetParams1D` 用 `#if CUDART_VERSION >= 11000` 守护；低版本退回 `cudaGraphExecMemcpyNodeSetParams`（3D 参数包装）或空实现。
- **`external/muda/src/muda/graph/details/graph.inl`**：`cudaGraphInstantiateWithFlags`、`cudaGraphAddMemcpyNode1D` 同样用版本守护降级。

#### CUB ByKey Scan 禁用

- **`external/muda/src/muda/cub/device/device_scan.h`**：所有 `*ByKey`（`ExclusiveSumByKey`、`InclusiveSumByKey` 等）入口在 `UIPC_COREX_CUDA10_COMPAT` 下通过条件编译移除。

#### 异步内存分配退化

- **`external/muda/src/muda/launch/details/memory.inl`**：
  - `alloc_1d`：跳过 `cudaMallocAsync`，改用同步 `cudaMalloc`，失败后依次尝试 `cudaMallocPitch` → `cudaMalloc3D`。
  - `copy`：`cudaMemcpyAsync` 替换为同步 `cudaMemcpy`。

#### cuSolver 新参数 API

- **`external/muda/src/muda/ext/linear_system/details/routines/solve/solve_dense.inl`**：`cusolverDnParams_t` + `cusolverDnXgetrf`/`Xgetrs` 等新 API 用 `CUDART_VERSION >= 11000` 守护，低版本报错返回。

---

## 三、CUDA 数学库兼容（cuBLAS / cuSOLVER / cuSPARSE）

### 3.1 问题描述

CoreX 提供的数学库对部分操作返回 `NOT_SUPPORTED`（如双精度 dot/nrm2），稀疏求解句柄 `cusolverSpCreate` 创建失败，`cusparseSpMV` 在某些情况下返回全零。部分库缺少符号（如 `cublasNrm2Ex`）。

### 3.2 具体改动

| 文件 | 改动 |
|------|------|
| `external/muda/.../routines/norm.inl` | `cublasSnrm2`/`cublasDnrm2` 调用后检查 `CUBLAS_STATUS_NOT_SUPPORTED`，回退到主机端计算（`cudaMemcpy` D2H → CPU `std::sqrt(Σx²)`） |
| `external/muda/.../routines/dot.inl` | `cublasSdot`/`cublasDdot` 同样增加 `NOT_SUPPORTED` 检查与主机回退（CPU 内积） |
| `external/muda/.../linear_system_handles.h` | `cusolverSpCreate` 返回 `NOT_SUPPORTED` 时将句柄置空（不阻塞初始化），销毁/设流操作增加空指针保护 |
| `src/backends/cuda/collision_detection/details/stackless_bvh.inl` | `thrust::cuda::par_nosync` 替换为 `thrust::cuda::par.on(nullptr)`（CoreX Thrust 版本不提供 `par_nosync`） |

---

## 四、ParallelFor 设备端 Lambda 捕获失效

### 4.1 问题描述

CoreX 的 clang-CUDA 编译器在处理以匿名 lambda 类型为模板参数的 kernel 实例化时，host 端的符号与 device 端 fatbin 中的符号无法正确匹配，导致 `muda::ParallelFor` 的 kernel 在运行时找不到对应 GPU 代码，**静默不执行任何计算**（编译通过、无运行时报错）。

### 4.2 影响范围

PCG 线性求解器、全局线性系统组装、仿射体动力学（能量/梯度/Hessian）、对角预条件器、线搜索、全局顶点管理器等核心路径。

### 4.3 具体改动

所有受影响路径将 lambda 式 `muda::ParallelFor` 重写为**显式 `__global__` kernel + 裸指针参数**。

| 文件 | 改动说明 |
|------|----------|
| `src/backends/cuda/linear_system/linear_pcg.cu` | PCG 求解器中的向量运算 kernel 化 |
| `src/backends/cuda/linear_system/global_linear_system.cu` | 全局线性系统组装 kernel 化 |
| `src/backends/cuda/affine_body/abd_diag_preconditioner.cu` | 对角预条件器 kernel 化 |
| `src/backends/cuda/affine_body/affine_body_dynamics.cu` | 能量/梯度/Hessian 计算 kernel 化 |
| `src/backends/cuda/affine_body/constitutions/ortho_potential.cu` | 正交势能 kernel 化 |
| `src/backends/cuda/engine/sim_engine.cu` | hello kernel 改为裸 `<<<>>>` 启动 + `cudaDeviceSynchronize` |
| `src/backends/cuda/global_geometry/global_vertex_manager.cu` | 顶点步进、CCD 设置等 kernel 化（`kernel_gvm_step_forward`、`kernel_gvm_setup_ccd`） |
| `src/backends/cuda/algorithm/corex_matrix_converter_kernels.cu` | 矩阵转换器全部操作 kernel 化（详见第七类） |
| `src/backends/cuda/algorithm/details/fast_segmental_reduce.inl` | 分段归约初始化改用 `cudaMemsetAsync` |

### 4.4 验证

修复后，simple 场景（两四面体，contact 关闭）200 帧自由落体正常，Newton 199/199 帧收敛，Y 方向轨迹与理论值误差 < 0.6%。

---

## 五、GPU 双精度运算缺陷

### 5.1 问题描述

CoreX GPU 存在**双精度减法 bug**：两个值单独读取正确，但 `a - b` 返回错误结果。此外，CoreX 双精度有效精度约为 23 位（而非 IEEE 754 的 52 位），导致涉及 `log`、`pow`、除法等运算的数值路径严重失真。

诊断 kernel 证据：
```
out[0] = q_raw[0];              // → 0.0  ✓
out[1] = qt_raw[0];             // → 0.0  ✓
out[2] = q_raw[0] - qt_raw[0];  // → -2.0 ✗（应为 0.0）
```

### 5.2 具体改动

#### 接触屏障函数（codim_ipc_contact.inl）

**原版**：使用 SymEigen 生成的解析公式（`std::pow`、`log`、直接除法）。

**适配版**（`UIPC_COREX_CUDA10_COMPAT && !UIPC_FLOAT_SCALAR` 路径）：
- 新增 `corex_barrier_detail` 命名空间：
  - `safe_log(x)`：手写 `log` 实现，基于 `atanh` 级数 + 范围归约，避免 CoreX 双精度 `log` 的精度问题。
  - `safe_log_ratio(a, b)`：`log(a/b)` 拆分为 `safe_log(a) - safe_log(b)`，避免小数除法的精度灾难。
  - `finite_diff_step(center, rel_scale)`：自适应有限差分步长。
- `KappaBarrier`：使用 `safe_log_ratio` 替代直接 `log((D-ξ²)/V)`，并增加值域钳位。
- `dKappaBarrierdD`：用**中心差分数值微分**替代解析一阶导数。
- `ddKappaBarrierddD`：用**中心差分数值微分**替代解析二阶导数。

**涉及文件**: `src/backends/cuda/contact_system/contact_models/sym/codim_ipc_contact.inl`

#### SPD 矩阵投影（make_spd.h）

**原版**：统一使用 `muda::eigen::evd` 特征值分解 → 负特征值归零 → 重构。

**适配版**三路分支：
- **CoreX + float**：手写 **Jacobi 迭代** 特征分解（避免 Eigen 表达式模板在设备端触发 `addrspacecast` 错误）+ 负特征值钳位 → 逐元素重构 `H = E * diag(λ) * E^T`。
- **CoreX + double**：**Gershgorin 圆盘定理对角偏移**（因 ~23 位精度下 Jacobi 迭代不收敛）：计算最小 Gershgorin 下界，若为负则整体偏移 `H(i,i) += shift`。
- **非 CoreX**：保持原版 EVD 路径不变。

**涉及文件**: `src/backends/cuda/utils/make_spd.h`

#### 单精度构建选项

- `UIPC_USE_FLOAT` / `UIPC_FLOAT_SCALAR` 宏：全局将 `Float` 从 `double` 切换为 `float`，作为规避双精度缺陷的整体方案。
- `UIPC_FLOAT_D_HAT_SCALE` 环境变量：float 构建下接触距离 `d_hat` 可能过紧时，提供运行时缩放乘子。

**涉及文件**: `CMakeLists.txt`、`src/backends/cuda/global_geometry/global_vertex_manager.cu`、`src/backends/cuda/contact_system/global_contact_manager.cu`

---

## 六、设备端代码生成问题

### 6.1 问题描述

CoreX LLVM 后端在某些模板组合下（Eigen 向量类型 + CUB DeviceReduce、复杂设备端表达式模板实例化等）会触发代码生成失败或无效的地址空间转换（`addrspacecast`）。

### 6.2 具体改动

| 文件 | 改动说明 |
|------|----------|
| `src/backends/cuda/global_geometry/global_vertex_manager.cu` | `compute_vertex_bounding_box()` 从设备端 `DeviceReduce` 降级为主机端计算：`cudaMemcpy` D2H → `std::min/std::max` 聚合（原版使用 muda 的设备端归约） |
| `src/backends/cuda/details/corex_device_placement_new.cu` | **新增文件**：为 CoreX 提供 `__device__` 版本的 placement `operator new` / `operator delete`（CoreX 缺少内建实现） |
| `src/backends/cuda/affine_body/details/abd_jacobi_matrix.inl` | 原版仅含简短结构定义；适配版扩展了 `ABDJacobi` 的 `operator*` / `to_mat` 实现，使用显式 `MUDA_HOST MUDA_DEVICE` 修饰（适配无 RDC 环境） |
| `src/backends/cuda/algorithm/givens.hpp` | `#include <Eigen/Eigen>` 改为 `Eigen/Core` + `Eigen/Dense`，避免 CoreX `cuda_wrappers/algorithm` 与 `std::min/max` 冲突 |

---

## 七、矩阵转换器 CoreX Kernel 替换

### 7.1 问题描述

`MatrixConverter` 中基于 `ParallelFor` 的矩阵操作（哈希计算、坐标解码、分段归约、拷贝等）在 CoreX 上因 lambda 捕获问题全部失效。

### 7.2 具体改动

| 文件 | 改动说明 |
|------|----------|
| `src/backends/cuda/algorithm/corex_matrix_converter_kernels.h` | **新增**：声明 `corex_matconv::launch_*` 系列显式 kernel 启动函数 |
| `src/backends/cuda/algorithm/corex_matrix_converter_kernels.cu` | **新增**：实现所有矩阵转换器核函数（hash/ij、decode block、segmental reduce 等） |
| `src/backends/cuda/algorithm/details/matrix_converter.inl` | CoreX 路径通过 `#if UIPC_COREX_CUDA10_COMPAT` 切换到 `corex_matconv::launch_*`；非 CoreX 保持原 `ParallelFor` |
| `src/backends/cuda/algorithm/details/fast_segmental_reduce.inl` | CoreX 路径使用 `cudaMemsetAsync` + `thrust::raw_pointer_cast` 替代 `BufferLaunch().fill`；修复共享内存数组索引的 constexpr 问题（`warp_count` → `kWarpCount`） |

---

## 八、类型系统扩展

### 8.1 具体改动

| 文件 | 改动说明 |
|------|----------|
| `src/backends/cuda/type_define.h` | 新增 `Eigen::AlignedBox<T, Dim>` 的 4 个 trivially 特性特化（`force_trivially_destructible`、`force_trivially_constructible`、`force_trivially_copy_constructible`、`force_trivially_copy_assignable`），与已有的 `Eigen::Matrix` 特化对齐。原版仅有 `Eigen::Matrix` 特化。 |
| `CMakeLists.txt` | `UIPC_USE_FLOAT` → 定义 `UIPC_FLOAT_SCALAR`，全局切换 `Float` 为 `float` |

---

## 九、引擎初始化与运行时诊断

### 9.1 问题描述

CoreX 运行时初始化阶段存在阻塞风险（`cudaSetDevice`、首个 kernel 启动、流创建等），需要更精细的初始化控制和诊断能力。

### 9.2 具体改动（sim_engine.cu）

| 原版 | 适配版 |
|------|--------|
| 使用 `KernelCout` 打印 hello 消息 | 移除 `KernelCout` 依赖（CoreX 不兼容），改用 `uipc_cuda_engine_nop_kernel` 裸 kernel |
| 简洁初始化（cudaGetDeviceCount → cudaSetDevice → hello） | 详细分步初始化，每步带日志输出和错误检查 |
| 无运行时探针 | 新增 `corex_runtime_probe()` 分步诊断函数 |
| 无环境变量控制 | 新增多个运行时环境变量（见下表） |

**新增运行时环境变量**：

| 变量 | 用途 |
|------|------|
| `UIPC_COREX_PROBE_STEPS` | 控制运行时探针启用的步骤（`all` / 逗号分隔列表） |
| `UIPC_SKIP_CUDA_SET_DEVICE` | 跳过 `cudaSetDevice` 调用 |
| `UIPC_FORCE_CUDA_HELLO_KERNEL` | 强制执行 hello kernel |
| `UIPC_SKIP_CUDA_HELLO_KERNEL` | 跳过 hello kernel |
| `UIPC_COREX_WARMUP_ALLOC` | 预热 GPU 内存分配 |
| `UIPC_COREX_GPU_DEVICE` | 指定 GPU 设备 ID（corex_demo 使用） |

**涉及文件**: `src/backends/cuda/engine/sim_engine.cu`、`src/backends/cuda/entrance.cpp`

---

## 十、新增示例与测试基础设施

### 10.1 具体改动

| 目录/文件 | 说明 |
|-----------|------|
| `apps/examples/corex_demo/` | **新增**：CoreX 综合测试 demo，包含 simple、wrecking_ball、slope、stack、domino 等 5+ 场景，支持 OBJ 输出和帧级控制（`main.cpp` 873 行） |
| `tools/corex_api_tests/` | **新增**：CUDA API 测试套件，含 6 个独立测试程序（runtime、graph、cublas、cusparse、cusolver、cross-buffer），配有测试矩阵文档和结果 CSV |
| `tools/corex_cuda_api_probe/` | **新增**：最小化 CUDA API 探针工具，用于定位运行时初始化阻塞 |
| `scripts/cuda_api_inventory.py` | **新增**：静态 CUDA API 使用盘点脚本，生成 API 调用统计和覆盖差异报告 |

---

## 十一、文件级变动汇总

### 新增文件（相对于原版）

| 路径 | 类别 |
|------|------|
| `cmake/corex-clang-cuda-wrapper.sh` | 构建系统 |
| `cmake/corex-compat/crt/host_defines.h` | 构建系统 |
| `cmake/patches/eigen3-SparseMatrix-corex-clang-max.patch` | 构建系统 |
| `src/backends/cuda/algorithm/corex_matrix_converter_kernels.h` | 矩阵转换器 |
| `src/backends/cuda/algorithm/corex_matrix_converter_kernels.cu` | 矩阵转换器 |
| `src/backends/cuda/details/corex_device_placement_new.cu` | 设备代码生成 |
| `apps/examples/corex_demo/` | 测试基础设施 |
| `tools/corex_api_tests/` | 测试基础设施 |
| `tools/corex_cuda_api_probe/` | 测试基础设施 |
| `scripts/cuda_api_inventory.py` | 测试基础设施 |

### 主要修改文件

| 路径 | 涉及类别 |
|------|----------|
| `CMakeLists.txt` | 一、八 |
| `src/backends/cuda/CMakeLists.txt` | 一 |
| `src/backends/cuda/type_define.h` | 八 |
| `src/backends/cuda/engine/sim_engine.cu` | 四、九 |
| `src/backends/cuda/engine/sim_engine_do_init.cu` | 九 |
| `src/backends/cuda/entrance.cpp` | 九 |
| `src/backends/cuda/contact_system/contact_models/sym/codim_ipc_contact.inl` | 五 |
| `src/backends/cuda/utils/make_spd.h` | 五、六 |
| `src/backends/cuda/global_geometry/global_vertex_manager.cu` | 四、五、六 |
| `src/backends/cuda/linear_system/linear_pcg.cu` | 四 |
| `src/backends/cuda/linear_system/global_linear_system.cu` | 四 |
| `src/backends/cuda/affine_body/affine_body_dynamics.cu` | 四 |
| `src/backends/cuda/affine_body/abd_diag_preconditioner.cu` | 四 |
| `src/backends/cuda/affine_body/constitutions/ortho_potential.cu` | 四 |
| `src/backends/cuda/affine_body/details/abd_jacobi_matrix.inl` | 六 |
| `src/backends/cuda/algorithm/details/matrix_converter.inl` | 七 |
| `src/backends/cuda/algorithm/details/fast_segmental_reduce.inl` | 七 |
| `src/backends/cuda/algorithm/givens.hpp` | 六 |
| `src/backends/cuda/collision_detection/details/stackless_bvh.inl` | 三 |
| `external/muda/src/muda/graph/details/graph_exec.inl` | 二 |
| `external/muda/src/muda/graph/details/graph.inl` | 二 |
| `external/muda/src/muda/cub/device/device_scan.h` | 二 |
| `external/muda/src/muda/launch/details/memory.inl` | 二 |
| `external/muda/src/muda/ext/linear_system/linear_system_handles.h` | 三 |
| `external/muda/src/muda/ext/linear_system/details/routines/norm.inl` | 三 |
| `external/muda/src/muda/ext/linear_system/details/routines/dot.inl` | 三 |
| `external/muda/src/muda/ext/linear_system/details/routines/solve/solve_dense.inl` | 二 |

---

## 十二、编译期开关与运行时环境变量汇总

### 编译期宏

| 宏 | 作用 |
|----|------|
| `UIPC_COREX_CUDA10_COMPAT` | CoreX CUDA 10.2 兼容模式总开关，控制全部 CoreX 特有代码路径 |
| `UIPC_FLOAT_SCALAR` | 单精度模式标记，由 `UIPC_USE_FLOAT` CMake 选项控制 |
| `MUDA_FORCE_HD_GENERIC` | 强制 `MUDA_GENERIC` 展开为 `__host__ __device__`，适配 CoreX 不定义 `__CUDA_ARCH__` 的情况 |

### 运行时环境变量

| 变量 | 用途 | 建议保留 |
|------|------|----------|
| `UIPC_COREX_GPU_DEVICE` | 指定 GPU 设备 ID | 是 |
| `UIPC_FLOAT_D_HAT_SCALE` | float 构建下 d_hat 缩放 | 是 |
| `UIPC_COREX_WARMUP_ALLOC` | 预热 GPU 分配 | 是 |
| `UIPC_COREX_PROBE_STEPS` | 运行时诊断探针 | 调试用 |
| `UIPC_SKIP_CUDA_SET_DEVICE` | 跳过 cudaSetDevice | 调试用 |
| `UIPC_FORCE_CUDA_HELLO_KERNEL` | 强制 hello kernel | 调试用 |
| `UIPC_SKIP_CUDA_HELLO_KERNEL` | 跳过 hello kernel | 调试用 |
| `UIPC_COREX_FORCE_HOST_GE2SYM` | 强制主机端 ge2sym | 调试用 |
| `UIPC_COREX_TRACE_LINEAR_SYSTEM` | 线性系统追踪 | 调试用 |
| `UIPC_COREX_TRACE_CONTACT_TYPE_ENERGY` | 接触能量追踪 | 调试用 |
| `UIPC_COREX_TRACE_CONTACT_TYPE_GRAD` | 接触梯度追踪 | 调试用 |
| `UIPC_COREX_TRACE_BARRIER_DBDD` | 屏障二阶导追踪 | 调试用 |
| `UIPC_COREX_TRACE_ABD_ASSEMBLE` | 仿射体组装追踪 | 调试用 |
| `UIPC_COREX_TRACE_SIMPLEX_FILTER` | 单纯形过滤追踪 | 调试用 |
| `UIPC_COREX_TRACE_FILTER_ACTIVE_DIAG` | 过滤活跃对角追踪 | 调试用 |
| `UIPC_COREX_ALLPE_*`（6 个） | 碰撞检测调优参数 | 调试用 |

---

## 十三、当前验证状态

在 CoreX 环境下，以下场景已通过测试：

| 场景 | 状态 |
|------|------|
| simple（两四面体自由落体） | 通过，200 帧收敛 |
| wrecking_ball（摆锤碰撞） | 通过 |
| slope（斜面滑落） | 通过 |
| stack（堆叠） | 通过 |
| domino（多米诺） | 通过 |

主要求解链路（Newton 迭代 + PCG 线性求解）可正常收敛。接触与约束相关的物理结果仍在与 NVIDIA 参考环境对照验证中。
