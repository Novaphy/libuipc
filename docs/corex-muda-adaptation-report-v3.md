# libuipc × muda（Corex）适配报告 v3（仅到“编译成功”版）

> 本文档是对 `corex-muda-adaptation-report-v2.md` 的**重新整理版**（不是新增更多范围）。  
> **范围严格截止到：`libuipc`（含 `cuda backend`）在 Corex 环境下编译成功**。  
> 不包含 dlopen/运行/演示/推进帧阶段的任何问题与修复。

---

## 1. 文档目的与阅读方式

你希望看懂的不是“我说我修了什么”，而是能在代码层面确认：

- **我做了什么**：改了哪些文件、改了哪类点
- **为什么这样做**：每类改动对应的编译错误/不兼容根因是什么
- **改动的边界**：哪些是 Corex 专用兼容层，哪些是通用的 host/device 资格修复

因此 v3 用统一结构组织每个改动点：

- **现象（编译期错误）**
- **根因（Corex clang-CUDA 的触发机制）**
- **修改策略（最小侵入/可维护）**
- **落点（文件 + 关键代码片段）**
- **影响（对上游/后续维护的含义）**

---

## 2. 适配目标（本报告范围内）

目标只有一个：**让 `libuipc_backend_cuda.so` 能在 Corex clang-CUDA + g++-10 主机编译器组合下编译成功**。

“编译成功”的判定：

- CMake 配置完成
- `cmake --build` 完整结束且没有编译/链接错误
- 产物生成（例如 `Release/bin/libuipc_backend_cuda.so`）

> 注：`dlopen`、运行期缺符号、以及推进帧/接触摩擦等属于“运行验证”，不在本文范围。

---

## 3. 构建链与工具链（为什么必须先改这一层）

### 3.1 关键矛盾：NVCC 参数集 ≠ Corex clang-CUDA 参数集

**现象**：libuipc/muda 的 CUDA 构建默认以 NVCC 为中心，CMake 会把一批“NVCC 习惯参数”（例如 `-rdc=true`、`-Xcudafe ...`、`--display_error_number` 等）传给 CUDA 编译器。  
在 Corex 环境中 CUDA 前端是 clang 变体，直接吞这些参数会在编译阶段报错，导致“连进入源码编译都做不到”。

**策略**：保留项目原有的 CUDA build 组织方式，只在“编译器入口”做**最小过滤/转换**，让 clang-CUDA 能吃下参数并正确识别 Corex 的 device 语言（`-x ivcore`）。

### 3.2 修改点（文件级）

#### A) 根 `CMakeLists.txt` 增加 Corex 构建入口（显式开关）

**目的**：把 Corex 相关配置收口到一个 option 下，避免污染常规 NVCC 路径。

**文件**：`CMakeLists.txt`

关键片段（v2 已给出，这里保留核心）：

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

**为什么要关 `SEPARABLE_COMPILATION`**：Corex clang-CUDA 路径对 RDC/设备符号解析链路更敏感，先以“单 TU/非 RDC”跑通编译是最小风险方案。

#### B) cuda backend 的 CMake 做 Corex 分支

**目的**：避免 CMake 为 CUDA target 加的某些属性/链路在 Corex 下触发不兼容。

**文件**：`src/backends/cuda/CMakeLists.txt`

关键片段：

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

#### C) 子模块初始化工作目录修正

**现象**：在子目录执行 submodule 更新会让 `external/muda` 状态不一致，造成“编译拿到的 muda 版本不稳定”。  
**策略**：强制从 repo 根执行 submodule update，确保可复现。

**文件**：`cmake/uipc_utils.cmake`

关键片段：

```cmake
file(RELATIVE_PATH _submod_path "${PROJECT_SOURCE_DIR}" "${_submod_dir}")
execute_process(
  COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive "${_submod_path}"
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  RESULT_VARIABLE GIT_SUBMOD_RESULT
)
```

#### D) Corex clang wrapper：过滤/转换 NVCC 参数 + 注入 `-x ivcore`

**目的**：让 Corex clang-CUDA 能接管现有 build.ninja 的参数形态。

**文件**：`cmake/corex-clang-cuda-wrapper.sh`

关键片段：

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

**影响**：这类改动属于“构建链兼容层”，主要影响 Corex 构建，不改变 libuipc 的高层算法语义。

---

## 4. 编译主线：host/device 资格不一致（为什么这一类改动最多）

### 4.1 关键矛盾：同一个函数在不同上下文被当成 host-only

**现象**：大量错误形态为：

- “calling a __host__ function from a __device__ function is not allowed”
- “no matching function for call …”（实质是 device 侧看不到候选，因为资格不匹配）

**根因（适配经验总结）**：

- `MUDA_GENERIC` / `UIPC_GENERIC` 这类宏在 Corex clang-CUDA 下更容易因为包含路径、inline/模板实例化位置、以及 lambda/device 语境而被推断为 host-only；
- 一旦某个函数（尤其是模板/inline）被当成 host-only，它在 device code 里的调用会彻底失配；
- libuipc 的 cuda backend 中，distance/friction/collision 等链路调用非常深，任何一个基础函数资格不一致都会引发“错误风暴”。

**策略**：

- 对“会同时在 host 与 device 上调用”的函数，显式改成 `MUDA_HOST MUDA_DEVICE` / `UIPC_HOST UIPC_DEVICE`；
- 同时保证声明与定义资格一致（避免重载/ODR 混乱）。

### 4.2 代表性改动（按类别列出）

#### A) 基础工具函数（最常见的触发点）

**文件**：`src/backends/cuda/animator/utils.h`  
**改动**：`lerp` 显式 `UIPC_HOST UIPC_DEVICE`

```cpp
template <typename T, int M, int N>
UIPC_HOST UIPC_DEVICE Eigen::Matrix<T, M, N> lerp(const Eigen::Matrix<T, M, N>& src,
                                                  const Eigen::Matrix<T, M, N>& dst,
                                                  T                             alpha)
```

**为什么这样做**：`lerp` 被 device lambda 间接调用，宏推断不稳定时会被当成 host-only；显式资格是最直接的消歧方式。

#### B) muda Viewer（Dense2D）接口全套资格统一

**文件**：`external/muda/src/muda/viewer/dense/dense_2d.h`  
**改动**：构造、访问、范围检查等统一 `MUDA_HOST MUDA_DEVICE`

```cpp
MUDA_HOST MUDA_DEVICE auto_const_t<T>& operator()(int x, int y) MUDA_NOEXCEPT
...
MUDA_INLINE MUDA_HOST MUDA_DEVICE void check() const MUDA_NOEXCEPT
```

**为什么这样做**：viewer 是设备端最频繁被调用的“容器访问层”。如果 `operator()` / `check()` 在 device 侧不可见，会导致成片的 “no matching function”。

#### C) 摩擦工具链（点/边/面组合）统一资格

**文件**：`src/backends/cuda/utils/friction_utils.h`  
**改动**：摩擦相关的基函数与雅可比/切向基等统一 `UIPC_HOST UIPC_DEVICE`

```cpp
inline UIPC_HOST UIPC_DEVICE void point_triangle_closest_point(...);
inline UIPC_HOST UIPC_DEVICE void edge_edge_jacobi(...);
inline UIPC_HOST UIPC_DEVICE void point_edge_tan_rel_dx(...);
inline UIPC_HOST UIPC_DEVICE void point_point_jacobi(...);
```

**为什么这样做**：接触/摩擦能量与梯度在 device kernel 中需要大量几何子函数；任何一个环节 host-only 都会把整条链路“切断”。

#### D) Distance / CCD / Finite Element 的声明与实现一致化

**文件**：`src/backends/cuda/utils/distance/ccd.h`（以及对应 `details/*.inl`）  
**改动**：声明层先统一 `MUDA_HOST MUDA_DEVICE`，再同步到实现。

```cpp
template <typename T>
MUDA_HOST MUDA_DEVICE bool point_edge_ccd_broadphase(...);

template <typename T>
MUDA_HOST MUDA_DEVICE bool edge_edge_ccd(...);
```

**文件**：`src/backends/cuda/finite_element/constitutions/discrete_shell_bending_function.h`  
**改动**：`E/dEdx/ddEddx` 统一 `UIPC_HOST UIPC_DEVICE`

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

**为什么这样做**：这类是“被 kernel 调用的数学核心函数”，资格不一致会直接阻断 FE/CCD 的设备端编译。

#### E) Collision Detection（BVH）里“声明/定义一致 + device 日志裁剪”

**文件**：`src/backends/cuda/collision_detection/details/linear_bvh.inl`  
**改动 1**：device 侧断言日志去掉依赖 `this->name()` 等字符串路径（避免 device 不必要符号/资格依赖）

```cpp
MUDA_KERNEL_ASSERT(idx < m_num_objects,
                   "BVHViewer: index out of range, idx=%u, num_objects=%u",
                   idx,
                   m_num_objects);
```

**改动 2**：`LinearBVHMortonIndex` 的 ctor/operator/比较等统一 `MUDA_HOST MUDA_DEVICE`，并在头文件声明保持一致（对应 `linear_bvh.h`）

```cpp
MUDA_INLINE MUDA_HOST MUDA_DEVICE LinearBVHMortonIndex::LinearBVHMortonIndex(uint32_t m, uint32_t idx) noexcept
...
MUDA_INLINE MUDA_HOST MUDA_DEVICE bool operator==(const LinearBVHMortonIndex& lhs,
                                                  const LinearBVHMortonIndex& rhs) noexcept
```

**为什么这样做**：BVH/过滤器/查看器组合非常深，任何“声明/定义资格不一致”都会触发重载冲突与模板推导失败。

---

## 5. Corex CUDA 10.2 不兼容 API（编译期分支降级）

> 这一节只写“为编译通过而必须做的 API 兼容”，不写运行期行为差异。

### 5.1 CUDA Graph 新 API 在 10.2 路径不完整

**现象**：在较新 CUDA runtime 才稳定出现的 Graph API，会在 Corex CUDA 10.2 下缺失或签名不一致，导致编译失败。  
**策略**：用 `CUDART_VERSION` 做编译期分支：新版本走新 API，旧版本退化到旧路径/空实现（保持可编译）。

**文件**：

- `external/muda/src/muda/graph/details/graph_exec.inl`
- `external/muda/src/muda/graph/details/graph.inl`

示例：

```cpp
#if CUDART_VERSION >= 11000
    checkCudaErrors(cudaGraphUpload(m_handle, stream));
#else
    (void)stream;
#endif
```

**影响**：Corex 10.2 下会禁用/退化部分 Graph 能力，但换来的是“后端可被编译出来”，这是本阶段的核心目标。

### 5.2 CUB `*ByKey` scan 能力缺失：编译期显式禁用封装

**现象**：Corex 配套 CUB/Thrust 组合下，`cub::DeviceScan::*ByKey` 相关模板实例化会触发编译失败。  
**策略**：在 `UIPC_COREX_CUDA10_COMPAT`（以及低版本 runtime）下不暴露对应 muda 封装，避免项目在不需要该能力时被阻塞。

**文件**：`external/muda/src/muda/cub/device/device_scan.h`

示例：

```cpp
#if !defined(UIPC_COREX_CUDA10_COMPAT) && CUDART_VERSION >= 11000
template <typename KeysInputIteratorT,
          typename ValuesInputIteratorT,
          typename ValuesOutputIteratorT,
          typename EqualityOpT = cub::Equality>
DeviceScan& ExclusiveSumByKey(...){ ... }
#endif
```

**影响**：Corex 10.2 兼容路径下，ByKey scan 封装不可用；普通 `ExclusiveSum/InclusiveSum` 等仍可用。

### 5.3 cuSolver 新参数 API 不可用：编译期分支并给出明确错误

**现象**：`cusolverDnParams_t` + `cusolverDnXgetrf/Xgetrs` 等“新参数 API”在 Corex CUDA 10.2 体系下不可用，直接走新 API 会编译失败。  
**策略**：用 `CUDART_VERSION` 做编译期分支：高版本启用新 API；低版本路径直接报错并返回（保证编译通过且失败可解释）。

**文件**：`external/muda/src/muda/ext/linear_system/details/routines/solve/solve_dense.inl`

示例：

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

### 5.4 Thrust 策略对象差异：`par_nosync` 替换为可用策略

**现象**：Corex 对应 Thrust 版本不提供 `thrust::cuda::par_nosync`，相关调用会在编译期失败。  
**策略**：替换为更保守但可用的执行策略（保持功能路径可编译）。

**文件**：`src/backends/cuda/collision_detection/details/stackless_bvh.inl`

示例：

```cpp
auto null_stream = thrust::cuda::par.on(nullptr);
thrust::sequence(null_stream, sorted_id.begin(), sorted_id.end());
```

### 5.5 Corex LLVM 后端在特定模板组合下不稳定：局部改为主机侧计算

**现象**：少数 “DeviceReduce + Eigen/向量类型” 组合触发 Corex LLVM 后端代码生成失败（例如 `Cannot select` 一类）。  
**策略**：对这一处热点（顶点包围盒）做“稳定优先”的降级：把数据拷回 host 用 `std::min/std::max` 聚合，绕开后端选择失败。

**文件**：`src/backends/cuda/global_geometry/global_vertex_manager.cu`

示例：

```cpp
std::vector<Vector3> h_positions(n);
positions.copy_to(h_positions);
for(const auto& p : h_positions)
{
    min_pos_host[0] = std::min(min_pos_host[0], p[0]);
    ...
}
```

---

## 6. 构建参数与可复现命令（本阶段的最终形态）

项目目录：`/private/libuipc`  
构建目录：`/private/libuipc/build_cuda`

推荐命令（v2 的可复现命令在此保留，仅强调与“编译成功”相关的关键项）：

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

**为什么主机用 g++-10**：避免更高版本 libstdc++ ABI/诊断差异与项目依赖链产生额外噪音；适配阶段先稳住组合再谈升级。

---

## 7. 编译产物（只到“编译成功”的结果）

编译成功后，主要产物在（示例路径，具体以你的 build 输出为准）：

- `/private/libuipc/build_cuda/Release/bin/libuipc_backend_cuda.so`
- 以及与之配套的 `libuipc_core.so`、`libuipc_geometry.so` 等基础库

---

## 8. 本阶段改动清单（按“编译链/源码适配”分组）

### 8.1 编译链/构建系统（让 Corex clang-CUDA 能参与构建）

- `CMakeLists.txt`
- `src/backends/cuda/CMakeLists.txt`
- `cmake/uipc_utils.cmake`
- `cmake/corex-clang-cuda-wrapper.sh`

### 8.2 源码适配（让 device 侧可见并通过模板/inline 编译）

> 这部分集中体现在“显式 host+device 资格”与“声明/定义一致”两类。

下面是**逐文件的完整覆盖清单（本阶段所有改动，一个不省略）**。为了可读性，我把“同类批量改动”压缩为：**列出每个文件 + 用一句话说明做了什么**，并在每个小类给出至少一个代表性代码片段。你如果需要“每个文件都贴出 diff”，我可以在此清单基础上追加附录（会非常长，但可以做到逐文件逐段落贴出）。

#### A) `UIPC_GENERIC`/`MUDA_GENERIC` → 显式 `HOST + DEVICE`（批量资格统一）

- `src/backends/cuda/animator/utils.h`：`lerp(...)` 等从 `UIPC_GENERIC` 改为 `UIPC_HOST UIPC_DEVICE`
- `src/backends/cuda/affine_body/constraints/soft_transform_constraint.cu`：`compute_constraint_mass(...)` 从 `UIPC_GENERIC` 改为 `UIPC_HOST UIPC_DEVICE`
- `src/backends/cuda/utils/friction_utils.h`：整套 friction 几何/雅可比/切向基工具函数从 `UIPC_GENERIC` 改为 `UIPC_HOST UIPC_DEVICE`
- `src/backends/cuda/finite_element/constitutions/discrete_shell_bending_function.h`：`compute_constants/E/dEdx/ddEddx` 从 `UIPC_GENERIC` 改为 `UIPC_HOST UIPC_DEVICE`

- `src/backends/cuda/utils/codim_thickness.h`：`edge_thickness/PT_thickness/D_range/is_active_D` 等从 `MUDA_GENERIC` 改为 `MUDA_HOST MUDA_DEVICE`
- `src/backends/cuda/utils/primitive_d_hat.h`：`*_dcd_expansion` 与 `PT_d_hat/EE_d_hat/PE_d_hat/PP_d_hat` 从 `MUDA_GENERIC` 改为 `MUDA_HOST MUDA_DEVICE`

- `src/backends/cuda/utils/distance/ccd.h`：所有 broadphase/ccd 声明从 `MUDA_GENERIC` 改为 `MUDA_HOST MUDA_DEVICE`
- `src/backends/cuda/utils/distance/details/ccd.inl`：对应实现同步从 `MUDA_GENERIC` 改为 `MUDA_HOST MUDA_DEVICE`
- `src/backends/cuda/utils/distance/point_point.h` 与 `src/backends/cuda/utils/distance/details/point_point.inl`：distance/grad/hess 资格统一
- `src/backends/cuda/utils/distance/point_edge.h` 与 `src/backends/cuda/utils/distance/details/point_edge.inl`：distance/grad/hess + helper 资格统一
- `src/backends/cuda/utils/distance/point_triangle.h` 与 `src/backends/cuda/utils/distance/details/point_triangle.inl`：distance/grad/hess + helper 资格统一
- `src/backends/cuda/utils/distance/edge_edge.h` 与 `src/backends/cuda/utils/distance/details/edge_edge.inl`：distance/grad/hess + helper 资格统一
- `src/backends/cuda/utils/distance/distance_flagged.h`：flagged distance 组合函数从 `MUDA_GENERIC` 改为 `MUDA_HOST MUDA_DEVICE`
- `src/backends/cuda/utils/distance/edge_edge_mollifier.h` 与 `src/backends/cuda/utils/distance/details/edge_edge_mollifier.inl`：mollifier 全链路从 `MUDA_GENERIC` 改为 `MUDA_HOST MUDA_DEVICE`

- `src/backends/cuda/collision_detection/details/stackless_bvh.inl`：多处几何/排序/拷贝工具函数与结构体方法从 `MUDA_GENERIC` 改为 `MUDA_HOST MUDA_DEVICE`（并包含 Thrust 策略替换，见下方 E）
- `src/backends/cuda/collision_detection/details/linear_bvh.inl`：`LinearBVHMortonIndex` ctor/operator/比较从 `MUDA_GENERIC` 改为 `MUDA_HOST MUDA_DEVICE`
- `src/backends/cuda/collision_detection/linear_bvh.h`：上述 `LinearBVHMortonIndex` 的**声明**同步改为 `MUDA_HOST MUDA_DEVICE`（保证声明/定义一致）

- `external/muda/src/muda/viewer/dense/dense_2d.h`：`Dense2DBase` 的 ctor/operator()/check 等整套接口从 `MUDA_GENERIC` 改为 `MUDA_HOST MUDA_DEVICE`
- `external/muda/src/muda/ext/eigen/evd.h`：`evd/evd_jacobi` 及内部 helper 从 `MUDA_GENERIC` 改为 `MUDA_HOST MUDA_DEVICE`

代表性片段（“资格显式化”的通用形态）：

```cpp
inline MUDA_HOST MUDA_DEVICE Vector2 D_range(Float xi, Float d_hat)
{
    ...
}
```

#### B) Collision Detection：device 侧日志路径裁剪（避免额外资格/符号依赖）

- `src/backends/cuda/collision_detection/details/linear_bvh.inl`：将 `MUDA_KERNEL_ASSERT/WARN` 中携带的 `this->name()/kernel_*()` 等 location 信息裁剪为“纯数值日志”

代表性片段：

```cpp
MUDA_KERNEL_ASSERT(idx < m_num_objects,
                   "BVHViewer: index out of range, idx=%u, num_objects=%u",
                   idx,
                   m_num_objects);
```

#### C) Corex CUDA 10.2：Graph API 编译期分支降级

- `external/muda/src/muda/graph/details/graph_exec.inl`
- `external/muda/src/muda/graph/details/graph.inl`

（见第 5.1 节示例）

#### D) Corex CUDA 10.2：CUB ByKey scan 封装编译期禁用

- `external/muda/src/muda/cub/device/device_scan.h`

（见第 5.2 节示例）

#### E) Corex Thrust：执行策略对象替换

- `src/backends/cuda/collision_detection/details/stackless_bvh.inl`

（见第 5.4 节示例）

#### F) 工具链后端不稳定：局部改为主机侧计算（编译期绕过）

- `src/backends/cuda/global_geometry/global_vertex_manager.cu`

（见第 5.5 节示例）

#### G) Corex 10.2：cuSolver 新参数 API 编译期分支

- `external/muda/src/muda/ext/linear_system/details/routines/solve/solve_dense.inl`

（见第 5.3 节示例）



