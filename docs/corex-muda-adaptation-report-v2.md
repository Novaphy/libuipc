# libuipc × muda（Corex）适配报告（重生版）

## 1. 文档目的

这份报告是对本轮 Corex 适配工作的重写版本，目标是：

- 以“适配过程”为主线，重点展开到代码级别；
- 明确补齐编译链（CMake/工具链/子模块）改动；
- 每类改动都给出可落地的代码示例；
- 保留必要的背景、构建、结果与演示建议，便于对外说明。

---

## 2. libuipc 项目概览（结构与主要功能）

### 2.1 libuipc 是什么（主要功能）

`libuipc` 是一个现代 C++20 的统一 IPC（Incremental Potential Contact）仿真库，面向：

- **刚体/软体/布料/细杆/线程** 等多物体与多材料形态
- **强耦合**（例如 Rigid–Deformable / Soft–Deformable 等耦合系统）
- **无穿透、摩擦接触** 处理（IPC barrier 能量 + 优化）
- 面向 GPU 的并行实现（官方后端之一为 `cuda`）
- 同时提供 C++ 与 Python 使用路径（本次适配聚焦于 C++/后端动态库）

### 2.2 前端/后端架构与动态加载

`libuipc` 的核心是“前端构建场景 + 后端执行仿真”的分层：

- **前端（frontend）**：创建 scene/world/objects/geometries，并通过统一接口驱动
- **后端（backend）**：独立模块，通常以动态库形式存在，由前端通过 `engine::Engine` 动态加载并链接

来自官方文档的最小使用形态：

```cpp
engine::Engine engine{"BACKEND_NAME"};
world::World world{engine};
```

### 2.3 代码与产物结构（以本次构建为准）

本次 Corex 构建（`/private/libuipc/build_cuda/Release/bin/`）的主要产物：

- **核心库**
  - `libuipc_core.so`：核心框架与前端/后端桥接（World/Engine/Visitor 等）
  - `libuipc_geometry.so`：几何/拓扑相关能力
  - `libuipc_constitution.so`：本构/约束/材料模型相关
  - `libuipc_io.so`：I/O 相关
  - `libuipc_sanity_check.so`：检查/校验相关
- **后端模块**
  - `libuipc_backend_none.so`：官方 `none` 后端（空实现/模板后端）
  - `libuipc_backend_cuda.so`：官方 `cuda` 后端（GPU 仿真执行）

> 注：后端模块是“可替换”的，实现形态与加载方式相同；差异在于它提供的仿真系统是否实现、以及是否依赖 CUDA/设备侧计算。

### 2.4 `none backend` 是什么，为什么必须写进适配过程

`none backend` 的定位在官方文档里非常明确：**一个“什么都不做”的后端模板**，用于展示如何写一个 backend，以及用于验证后端加载/接口链路本身。

这对 Corex 适配的价值是：

- **先验证 backend 动态加载/符号导出/基础接口** 没问题（与 CUDA 无关）
- 将问题拆解为两层：
  - “框架/构建/加载是否 OK”（`none` 足够）
  - “CUDA 工具链/设备侧代码是否 OK”（`cuda` 才涉及）

也就是说，本次适配并不是“跳过 none”，而是把它作为**基准后端**：

- 只要 `none backend` 能编译并加载，说明 `libuipc_core` 与 backend 模块机制工作正常；
- 后续 `cuda backend` 的失败，才更大概率归因于 Corex CUDA 工具链、muda 或 device 代码路径。

---

## 3. 目标与范围

本轮目标不是仅让 `none backend` 可编译，而是让 `libuipc` 的 Corex CUDA 路径可用，具体包括：

1. **构建链路可复现**：稳定配置并完整编译通过；
2. **运行链路可验证**：关键动态库可成功加载；
3. **CUDA backend 可交付**：`libuipc_backend_cuda.so` 编译并可 `dlopen`。

---

## 4. 构建环境与可复现命令（保留）

项目目录：`/private/libuipc`  
构建目录：`/private/libuipc/build_cuda`

推荐命令：

```bash
rm -rf build_cuda && mkdir build_cuda && cd build_cuda
VCPKG_FORCE_SYSTEM_BINARIES=1 \
CMAKE_TOOLCHAIN_FILE=/private/libuipc/vcpkg/scripts/buildsystems/vcpkg.cmake \
cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-10 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-10 \
  -DUIPC_BUILD_PYBIND=OFF \
  -DUIPC_BUILD_EXAMPLES=OFF \
  -DUIPC_BUILD_TESTS=OFF \
  -DUIPC_BUILD_BENCHMARKS=OFF \
  -DUIPC_MUDA_USE_COREX=ON \
  -DUIPC_COREX_CUDA_ARCHITECTURES=ivcore11 \
  -DUIPC_COREX_CUDA_STANDARD=20 \
  -DUIPC_COREX_CUDA_COMPILER=/private/libuipc/cmake/corex-clang-cuda-wrapper.sh \
  ..

VCPKG_FORCE_SYSTEM_BINARIES=1 cmake --build . -j2
```

---

## 5. 适配方法（保留）

- 按“**首批阻塞错误 -> 最小修复 -> 继续编译**”循环推进；
- 优先修复 **host/device 资格冲突**（Corex clang 最敏感）；
- 对 Corex CUDA 10.2 能力缺失点做**降级兼容**，避免一次性重构；
- 每轮都做“编译 + 加载”双验证，而不是只看编译。

---

## 6. 关键问题总览（保留）

1. **编译链问题**：NVCC 参数直接喂给 Corex clang，触发非法参数；
2. **模板函数资格不一致**：`MUDA_GENERIC/UIPC_GENERIC` 在不同路径被当成 host-only；
3. **Viewer/Distance/Friction 大量调用链在 device 侧失配**；
4. **运行时符号差异**：Corex `libcublas.so.10` 不导出 `cublasNrm2Ex`。

---

## 7. 重点：适配过程与代码示例（核心章节）

> 本章按“类别 -> 修改原因 -> 修改目的 -> 代码示例”组织，重点覆盖你指出的编译链与代码细节。

### 6.1 编译链与构建系统适配（补齐重点）

#### A) 根 CMake 增加 Corex 统一入口

**文件**：`CMakeLists.txt`  
**原因**：原始路径以 NVCC 为中心，Corex 编译器与架构无法统一配置。  
**目的**：让 Corex 成为可显式启用的构建模式。

```cmake
option(UIPC_MUDA_USE_COREX "Enable Corex-specific CUDA build compatibility mode" OFF)
set(UIPC_COREX_CUDA_STANDARD "17" CACHE STRING "CUDA C++ standard used when UIPC_MUDA_USE_COREX=ON")
set(UIPC_COREX_CUDA_ARCHITECTURES "ivcore11" CACHE STRING "CUDA architectures used when UIPC_MUDA_USE_COREX=ON")
set(UIPC_COREX_CUDA_COMPILER "" CACHE FILEPATH "Path to Corex CUDA compiler (optional)")

if(UIPC_MUDA_USE_COREX)
    if(UIPC_COREX_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "${UIPC_COREX_CUDA_ARCHITECTURES}" CACHE STRING "CUDA architectures" FORCE)
    endif()
    if(UIPC_COREX_CUDA_COMPILER)
        set(CMAKE_CUDA_COMPILER "${UIPC_COREX_CUDA_COMPILER}" CACHE FILEPATH "CUDA compiler" FORCE)
    endif()
    set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF CACHE BOOL "Enable CUDA separable compilation" FORCE)
endif()
```

#### B) CUDA backend CMake 针对 Corex 路径分支

**文件**：`src/backends/cuda/CMakeLists.txt`  
**原因**：Corex clang 对 RDC 与某些 CUDA target 属性不兼容。  
**目的**：让 backend 的编译属性与 Corex 工具链匹配。

```cmake
if(UIPC_MUDA_USE_COREX)
    target_link_libraries(cuda PUBLIC dl)
    target_compile_definitions(cuda PUBLIC UIPC_COREX_CUDA10_COMPAT=1)
    set_target_properties(cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    )
endif()
```

#### C) 子模块初始化路径修正

**文件**：`cmake/uipc_utils.cmake`  
**原因**：在子目录执行 submodule 更新会导致 `external/muda` 状态不一致。  
**目的**：固定在项目根目录执行，保证可复现。

```cmake
# Git submodule update must run from repo root; use path relative to PROJECT_SOURCE_DIR
file(RELATIVE_PATH _submod_path "${PROJECT_SOURCE_DIR}" "${_submod_dir}")
execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive "${_submod_path}"
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                RESULT_VARIABLE GIT_SUBMOD_RESULT)
```

#### D) Corex wrapper 过滤 NVCC 参数

**文件**：`cmake/corex-clang-cuda-wrapper.sh`  
**原因**：`-rdc=true`、`--display_error_number`、`-Xcudafe` 等会被 Corex clang 拒绝。  
**目的**：最小侵入兼容原构建流程。

```bash
if [[ "$arg" == "-rdc=true" ]]; then
    continue
fi
if [[ "$arg" == "--display_error_number" ]]; then
    continue
fi
if [[ "$arg" == "-Xcudafe" ]]; then
    skip_next=1
    continue
fi
...
if [[ "$is_cuda_input" -eq 1 && "$has_ivcore_lang" -eq 0 ]]; then
    args=("-x" "ivcore" "${args[@]}")
fi
```

---

### 6.2 Host/Device 资格统一（大规模主线）

#### A) 典型入口函数：`lerp`

**文件**：`src/backends/cuda/animator/utils.h`  
**问题**：device lambda 调用 `lerp` 时被识别为 host-only。  
**改法**：显式 `UIPC_HOST UIPC_DEVICE`。

```cpp
template <typename T, int M, int N>
UIPC_HOST UIPC_DEVICE Eigen::Matrix<T, M, N> lerp(const Eigen::Matrix<T, M, N>& src,
                                                  const Eigen::Matrix<T, M, N>& dst,
                                                  T                             alpha)
```

#### B) Viewer 体系：`Dense2DBase`

**文件**：`external/muda/src/muda/viewer/dense/dense_2d.h`  
**问题**：`operator()(x,y)`/`check()` 在 device 上频繁报 host-only。  
**改法**：构造、访问、检查全套接口统一显式 host+device。

```cpp
MUDA_HOST MUDA_DEVICE auto_const_t<T>& operator()(int x, int y) MUDA_NOEXCEPT
...
MUDA_INLINE MUDA_HOST MUDA_DEVICE void check() const MUDA_NOEXCEPT
```

#### C) 摩擦工具链：point/edge/triangle/edge-edge 全链路

**文件**：`src/backends/cuda/utils/friction_utils.h`  
**问题**：`codim_ipc_simplex_frictional_contact_function.h` 调用大量 no matching。  
**改法**：统一为 `UIPC_HOST UIPC_DEVICE`。

```cpp
inline UIPC_HOST UIPC_DEVICE void point_triangle_closest_point(...);
inline UIPC_HOST UIPC_DEVICE void edge_edge_jacobi(...);
inline UIPC_HOST UIPC_DEVICE void point_edge_tan_rel_dx(...);
inline UIPC_HOST UIPC_DEVICE void point_point_jacobi(...);
```

---

### 6.3 Distance / CCD / Finite Element 调用链收敛

#### A) CCD 声明层先统一资格

**文件**：`src/backends/cuda/utils/distance/ccd.h`  
**问题**：broadphase/ccd 函数在 device 侧不可见。  
**改法**：声明层统一 `MUDA_HOST MUDA_DEVICE`。

```cpp
template <typename T>
MUDA_HOST MUDA_DEVICE bool point_edge_ccd_broadphase(...);

template <typename T>
MUDA_HOST MUDA_DEVICE bool edge_edge_ccd(...);
```

#### B) 离散壳弯曲函数链补齐资格

**文件**：`src/backends/cuda/finite_element/constitutions/discrete_shell_bending_function.h`  
**问题**：`discrete_shell_bending.cu` 中调用 `E/dEdx/ddEddx` 报 host-only。  
**改法**：核心函数显式 host+device。

```cpp
inline UIPC_HOST UIPC_DEVICE Float E(const Vector3& x0,
                                     const Vector3& x1,
                                     const Vector3& x2,
                                     const Vector3& x3,
                                     Float          L0,
                                     Float          h_bar,
                                     Float          theta_bar,
                                     Float          kappa)
```

---

### 6.4 Collision Detection 专项修复

#### A) `linear_bvh` device 日志路径裁剪

**文件**：`src/backends/cuda/collision_detection/details/linear_bvh.inl`  
**问题**：device 检查中依赖 `this->name()/kernel_name()` 导致匹配失败。  
**改法**：简化为纯数值信息，避免 device 不必要字符串路径。

```cpp
MUDA_KERNEL_ASSERT(idx < m_num_objects,
                   "BVHViewer: index out of range, idx=%u, num_objects=%u",
                   idx,
                   m_num_objects);
```

#### B) MortonIndex 声明/定义资格一致

**文件**：`src/backends/cuda/collision_detection/details/linear_bvh.inl`  
**问题**：声明和定义资格不一致引发重载冲突。  
**改法**：统一为 `MUDA_HOST MUDA_DEVICE`。

```cpp
MUDA_INLINE MUDA_HOST MUDA_DEVICE LinearBVHMortonIndex::LinearBVHMortonIndex(uint32_t m, uint32_t idx) noexcept
...
MUDA_INLINE MUDA_HOST MUDA_DEVICE bool operator==(const LinearBVHMortonIndex& lhs,
                                                  const LinearBVHMortonIndex& rhs) noexcept
```

---

### 6.5 运行时符号兼容（`cublasNrm2Ex`）

**文件**：`external/muda/src/muda/ext/linear_system/details/routines/norm.inl`  
**问题**：`dlopen(libuipc_backend_cuda.so)` 报 `undefined symbol: cublasNrm2Ex`。  
**根因**：Corex `libcublas.so.10` 不导出该符号。  
**改法**：对 float/double 走 `Snrm2/Dnrm2`，其它类型保留 `Nrm2Ex`。

```cpp
if constexpr(std::is_same_v<T, float>)
{
    checkCudaErrors(cublasSnrm2(cublas(), size, x.data(), x.inc(), result.data()));
}
else if constexpr(std::is_same_v<T, double>)
{
    checkCudaErrors(cublasDnrm2(cublas(), size, x.data(), x.inc(), result.data()));
}
else
{
    auto type = cuda_data_type<T>();
    checkCudaErrors(cublasNrm2Ex(
        cublas(), size, x.data(), type, x.inc(), result.data(), type, type));
}
```

---

## 8. Corex 不兼容 API/库：调整清单与影响评估

> 本章专门回答“有没有因 Corex 不兼容而放弃/调整功能”的问题。  
> 结论：有，且主要采取了 **版本分支降级**、**兼容替代实现**、**能力显式禁用** 三类策略。

### 7.1 CUDA Graph 新 API 在 Corex CUDA 10.2 不完整

- **不兼容点**
  - `cudaGraphUpload`
  - `cudaGraphInstantiateWithFlags`
  - `cudaGraphAddMemcpyNode1D`
- **原因**
  - 这些 API 在较高 CUDA runtime 才稳定可用，Corex 10.2 路径不完整或签名差异。
- **修改方式**
  - 文件：
    - `external/muda/src/muda/graph/details/graph_exec.inl`
    - `external/muda/src/muda/graph/details/graph.inl`
  - 通过 `#if CUDART_VERSION >= 11000` 分支使用新 API，否则退回旧 API。
- **代码示例**

```cpp
#if CUDART_VERSION >= 11000
    checkCudaErrors(cudaGraphUpload(m_handle, stream));
#else
    (void)stream;
#endif
```

```cpp
#if CUDART_VERSION >= 11000
    checkCudaErrors(cudaGraphAddMemcpyNode1D(...));
#else
    cudaMemcpy3DParms parms = {};
    checkCudaErrors(cudaGraphAddMemcpyNode(..., &parms));
#endif
```

- **影响评估**
  - 低版本路径下 Graph 的某些“1D 便捷接口”与 upload 优化能力会降级；
  - 功能可用性保留，但性能与行为细节更接近旧版 CUDA Graph。

### 7.2 CUB `*ByKey` scan 能力缺失：显式禁用

- **不兼容点**
  - `cub::DeviceScan::ExclusiveSumByKey` / `ExclusiveScanByKey` / `Inclusive*ByKey`
- **原因**
  - Corex 配套 CUB 在该版本不完整，直接实例化会报符号或编译错误。
- **修改方式**
  - 文件：`external/muda/src/muda/cub/device/device_scan.h`
  - 在 `UIPC_COREX_CUDA10_COMPAT` + 低版本 runtime 下直接不暴露对应封装。
- **代码示例**

```cpp
#if !defined(UIPC_COREX_CUDA10_COMPAT) && CUDART_VERSION >= 11000
template <typename KeysInputIteratorT, typename ValuesInputIteratorT, typename ValuesOutputIteratorT, typename EqualityOpT = cub::Equality>
DeviceScan& ExclusiveSumByKey(...){ ... }
#endif
```

- **影响评估**
  - 依赖 `ByKey` scan 的调用在 Corex 10.2 路径不可用（功能收缩）；
  - 其它普通 scan (`ExclusiveSum/InclusiveSum/Scan`) 仍可用。

### 7.3 cuSolver 新参数 API 不可用：保留旧路径并显式报错

- **不兼容点**
  - `cusolverDnParams_t` + `cusolverDnXgetrf/Xgetrs` 相关高级参数接口
- **原因**
  - 该 API 属于较新 cuSolver 体系，Corex 10.2 不具备完整支持。
- **修改方式**
  - 文件：`external/muda/src/muda/ext/linear_system/details/routines/solve/solve_dense.inl`
  - `#if CUDART_VERSION >= 11000` 才走新 API；否则给出明确错误并返回。
- **代码示例**

```cpp
#if CUDART_VERSION >= 11000
    cusolverDnParams_t params;
    cusolverDnCreateParams(&params);
    ...
#else
    MUDA_ERROR_WITH_LOCATION(
        "cusolverDn params API is unavailable on this CUDA runtime; "
        "gesv() is not supported in this backend configuration.");
    return;
#endif
```

- **影响评估**
  - `gesv()` 在 Corex 10.2 下被明确标为不支持；
  - 优点是“失败可解释”，避免静默错误或崩溃。

### 7.4 cuBLAS `cublasNrm2Ex` 缺符号：降级到 `Snrm2/Dnrm2`

- **不兼容点**
  - `cublasNrm2Ex` 运行时缺失（`undefined symbol`）
- **原因**
  - Corex `libcublas.so.10` 不导出该符号。
- **修改方式**
  - 文件：`external/muda/src/muda/ext/linear_system/details/routines/norm.inl`
  - 对 `float/double` 分别走 `cublasSnrm2/cublasDnrm2`；其它类型保留 `Nrm2Ex`。
- **代码示例**

```cpp
if constexpr(std::is_same_v<T, float>)
    checkCudaErrors(cublasSnrm2(...));
else if constexpr(std::is_same_v<T, double>)
    checkCudaErrors(cublasDnrm2(...));
else
    checkCudaErrors(cublasNrm2Ex(...));
```

- **影响评估**
  - 常见 `float/double` 路径恢复可用，且行为与标准 BLAS 一致；
  - 混合精度等特殊类型仍依赖 `Nrm2Ex`，在旧 runtime 可能继续受限。

### 7.5 Thrust 策略对象差异：`par_nosync` 替换为 `par`

- **不兼容点**
  - `thrust::cuda::par_nosync` 不可用
- **原因**
  - Corex 对应 Thrust 版本未提供该执行策略。
- **修改方式**
  - 文件：`src/backends/cuda/collision_detection/details/stackless_bvh.inl`
  - 替换为 `thrust::cuda::par.on(nullptr)`。
- **代码示例**

```cpp
auto null_stream = thrust::cuda::par.on(nullptr);
thrust::sequence(null_stream, sorted_id.begin(), sorted_id.end());
```

- **影响评估**
  - 语义更保守，可能减少部分“无同步优化”收益；
  - 稳定性提升，避免因策略对象缺失导致构建失败。

### 7.6 LLVM/后端代码生成稳定性问题：局部改为主机侧计算

- **不兼容点**
  - 某些 `DeviceReduce + Eigen` 组合触发 Corex LLVM 后端异常（`Cannot select`）
- **原因**
  - 属于工具链后端在特定模板/向量类型组合下的代码生成问题。
- **修改方式**
  - 文件：`src/backends/cuda/global_geometry/global_vertex_manager.cu`
  - `compute_vertex_bounding_box()` 改为拷回 host 后做 `std::min/std::max` 聚合。
- **代码示例**

```cpp
std::vector<Vector3> h_positions(n);
positions.copy_to(h_positions);
for(const auto& p : h_positions)
{
    min_pos_host[0] = std::min(min_pos_host[0], p[0]);
    ...
}
```

- **影响评估**
  - 功能正确性保留；
  - 性能上会引入一次设备到主机的数据拷贝，属于“稳定优先”的临时降级策略。

### 7.7 Corex clang 与 NVCC 参数模型不同：禁用 RDC + wrapper 过滤

- **不兼容点**
  - `-rdc=true`、`--display_error_number`、`-Xcudafe` 等 NVCC 参数
- **原因**
  - Corex 使用 clang CUDA 前端，参数语义与 NVCC 不同。
- **修改方式**
  - 文件：
    - `cmake/corex-clang-cuda-wrapper.sh`
    - `CMakeLists.txt`
    - `src/backends/cuda/CMakeLists.txt`
  - 过滤不兼容参数，并关闭 `CUDA_SEPARABLE_COMPILATION` / `CUDA_RESOLVE_DEVICE_SYMBOLS`。
- **影响评估**
  - 构建稳定性显著提升；
  - 设备端分离编译/RDC 相关能力在 Corex 路径被收缩，需要以后随工具链能力再恢复。

---

## 9. 按时间线进展（保留，简化版）

### 阶段 A：恢复增量编译
- 从网络波动恢复后继续 `cmake --build`，获取新首批错误。

### 阶段 B：集中修 host/device 资格链
- 从 `affine/collision` 扩展到 `distance/friction/finite_element`，成批收敛 no matching。

### 阶段 C：全量重编与运行态校验
- 执行 clean + full rebuild；
- 发现并修复 `cublasNrm2Ex` 运行期符号问题；
- 重新验证全部主要模块可 `dlopen`。

---

## 10. 当前状态与可演示项

### 8.1 当前状态

- 全量重编通过（`exit_code: 0`）；
- 主要库均可加载：
  - `libuipc_core.so`
  - `libuipc_geometry.so`
  - `libuipc_constitution.so`
  - `libuipc_io.so`
  - `libuipc_sanity_check.so`
  - `libuipc_backend_none.so`
  - `libuipc_backend_cuda.so`

### 8.2 可演示内容（建议）

1. **构建演示**：展示 Corex 选项配置与成功构建产物；
2. **加载演示**：逐个 `dlopen` 核心模块与 CUDA backend；
3. **能力说明演示**：解释本次适配重点是“Corex 下 CUDA 路径打通 + 运行时兼容修复”。

---

## 11. 结论

本次适配已经从“能否编译”推进到“编译通过 + 运行可加载”，并且补齐了你关注的两点：

1. **编译链改动**（CMake/子模块/wrapper）已明确列出；
2. **按类别附代码示例**（host-device、collision、distance/ccd、finite element、cublas 运行兼容）已给出。

如果你需要，我可以基于这版再补一页“**按文件分组的提交建议**”（方便你拆 commit 和对外讲述每组改动的价值）。

