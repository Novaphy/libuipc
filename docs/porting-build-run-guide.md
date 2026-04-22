## libuipc 移植与构建运行指南

> 适用范围：将本仓库 `/root/libuipc` 的代码搬到其它机器（NVIDIA 或 Corex/Iluvatar）上重新构建并跑通 `corex_demo` 五个示例场景。
>
> 关联文档：[`corex-compat-policy.md`](corex-compat-policy.md)、[`corex-compat-audit-2026-04-17.md`](corex-compat-audit-2026-04-17.md)。

---

## 1. 总体设计回顾

代码库内同一份源码同时支持两套编译路径，用单一宏 `UIPC_COREX_CUDA10_COMPAT` 区分：

| 路径 | 触发方式 | 标量类型 | dylib API | CUDA 编译器 | PCG 求解 |
|---|---|---|---|---|---|
| **NVIDIA**（上游对齐） | `-DUIPC_MUDA_USE_COREX=OFF`（默认） | `Float = double` | dylib v2（`class dylib`） | nvcc | upstream `linear_pcg.cu` 原版 |
| **Corex / Iluvatar** | `-DUIPC_MUDA_USE_COREX=ON` | `Float = float` | dylib v3（`namespace dylib { class library; }`） | clang-cuda wrapper | `linear_pcg_corex.cu.inc`（带 PCG fix-A：教科书相对残差判据） |

切换由 22 个 **switcher overlay 主文件**承担：每个文件顶部有 `#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT` ... `#else` ...`#endif`，两侧分别 include corex sidecar 或内联 upstream 原版。除此以外另有少量行级 `#if` 守护和 CMake 级文件过滤。

---

## 2. 必须打包的内容

### 2.1 代码本体

仓库整棵工作树都需要带走。**特别注意 20 个 `_corex` sidecar 是 untracked 文件**，仅 `git archive`/`git stash` 会丢。一定要用 `tar` 打包整个工作树：

```bash
cd /root/libuipc
tar --exclude='build*' \
    --exclude='_build*' \
    --exclude='vcpkg_installed' \
    --exclude='logs' \
    --exclude='output' \
    --exclude='libuipc_backend_cuda.so' \
    --exclude='.git' \
    -czf /tmp/libuipc-port.tar.gz .
```

20 个 corex sidecar 完整清单（**两路均必须共存**，缺一编不过 corex 路径，但 NVIDIA 路径不会用到）：

```
src/backends/cuda/affine_body/abd_jacobi_matrix_corex.cu
src/backends/cuda/affine_body/abd_jacobi_matrix_corex.h
src/backends/cuda/affine_body/constitutions/affine_body_revolute_joint_corex.cu
src/backends/cuda/collision_detection/info_stackless_bvh_corex.h
src/backends/cuda/collision_detection/info_stackless_bvh_v0_corex.h
src/backends/cuda/collision_detection/linear_bvh_corex.h
src/backends/cuda/collision_detection/stackless_bvh_corex.h
src/backends/cuda/finite_element/fem_utils_corex.h
src/backends/cuda/finite_element/mas_preconditioner_engine_corex.cu
src/backends/cuda/finite_element/mas_preconditioner_engine_corex.h
src/backends/cuda/finite_element/matrix_utils_corex.h
src/backends/cuda/linear_system/linear_fused_pcg_corex.cu
src/backends/cuda/linear_system/linear_fused_pcg_corex.h
src/backends/cuda/linear_system/linear_pcg_corex.cu.inc
src/backends/cuda/linear_system/linear_pcg_corex.h
src/core/core/i_engine_corex.cpp.inc
src/core/core/internal/engine_corex.cpp.inc
src/core/core/internal/world_corex.cpp.inc
src/core/core/sanity_checker_corex.cpp.inc
src/core/core/world_corex.cpp.inc
```

### 2.2 不需要带走的东西

| 内容 | 说明 |
|---|---|
| `/root/src/` | 仅作为 audit 对照参考；NVIDIA 路径的源码已经 inline 在每个 switcher 的 `#else` 分支里，运行/构建期都不依赖这个目录 |
| `/root/vcpkg1/` | 这是当前机器的离线 vcpkg；NVIDIA 端推荐重新拉一份 vcpkg 用 manifest 模式联网装 |
| `build_*/`、`output/`、`logs/` | 构建产物和运行结果 |
| 仓库根的 `.so` 残留物 | 历史遗留，无意义 |

### 2.3 需要在目标机上准备的依赖

**NVIDIA 端**：

| 软件 | 版本要求 |
|---|---|
| CMake | ≥ 3.26 |
| Ninja | 任意现代版本 |
| GCC / Clang | 支持 C++20（GCC 11+，或 Clang 14+） |
| CUDA Toolkit | 12.x（含 nvcc、cuBLAS、cuSPARSE） |
| vcpkg | 任意能联网的版本，仓库根有 `vcpkg.json` 走 manifest 模式自动拉依赖 |
| 系统库 | `libdl`、`libpthread` 等系统标配 |

**Corex / Iluvatar 端**：

| 软件 | 路径示例 / 说明 |
|---|---|
| Iluvatar Corex SDK | 装在 `/usr/local/corex/`，提供 clang-based CUDA 工具链 |
| Corex CUDA wrapper | `cmake/corex-clang-cuda-wrapper.sh`（已在仓库内） |
| 离线 vcpkg + libdylib v3 | 当前机器对应 `/root/vcpkg1`，含 `libdylib.a` v3 |

---

## 3. 构建命令

### 3.1 NVIDIA 路径（要拿过去测的就是这个）

```bash
cd /path/to/libuipc
rm -rf build_nvidia && mkdir build_nvidia && cd build_nvidia

cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake \
  -DUIPC_USING_LOCAL_VCPKG=OFF \
  -DUIPC_WITH_CUDA_BACKEND=ON \
  -DUIPC_MUDA_USE_COREX=OFF \
  -DUIPC_USE_FLOAT=OFF \
  -DUIPC_CUDA_ARCHITECTURES=native \
  -DUIPC_BUILD_EXAMPLES=ON \
  -DUIPC_BUILD_TESTS=OFF \
  ..

cmake --build . --config Release -j112
```

关键开关：

- `UIPC_MUDA_USE_COREX=OFF` → 不定义 `UIPC_COREX_CUDA10_COMPAT`，所有 switcher overlay 走 `#else` 分支（dylib v2、analytical PCG、原版语义）
- `UIPC_USE_FLOAT=OFF` → `Float = double`（NVIDIA 标准精度）
- `UIPC_CUDA_ARCHITECTURES=native` → 自动探测当前卡，可改 `"75;86;89"` 等显式列表

### 3.2 Corex 路径（当前机器，复现/增量构建）

已经存在的 build 目录直接 ninja 增量：

```bash
cd /root/libuipc/build_corex_current
cmake --build . --config Release -j112
```

如果要完全重新配置：

```bash
cd /root/libuipc
rm -rf build_corex_current && mkdir build_corex_current && cd build_corex_current

cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=/root/vcpkg1/scripts/buildsystems/vcpkg.cmake \
  -DUIPC_USING_LOCAL_VCPKG=ON \
  -DUIPC_WITH_CUDA_BACKEND=ON \
  -DUIPC_MUDA_USE_COREX=ON \
  -DUIPC_USE_FLOAT=ON \
  -DUIPC_COREX_CUDA_ARCHITECTURES=ivcore11 \
  -DCMAKE_CUDA_COMPILER=/root/libuipc/cmake/corex-clang-cuda-wrapper.sh \
  -DUIPC_BUILD_EXAMPLES=ON \
  -DUIPC_BUILD_COREX_API_TESTS=ON \
  ..

cmake --build . --config Release -j112
```

---

## 4. 运行命令

可执行文件：`build_*/Release/bin/corex_demo`，两路通用。

```bash
cd build_*/Release/bin
./corex_demo --backend cuda \
             --scene <scene> \
             --frames <N> \
             --gpu <id> \
             --output_dir <dir>
```

`<scene>` ∈ `simple | slope | stack | domino | wrecking_ball`

### 4.1 五个场景的推荐参数

| 场景 | 推荐命令 | 预期物理 |
|---|---|---|
| **simple** | `UIPC_SIMPLE_FORCE_MU=0.4 ./corex_demo --backend cuda --scene simple --frames 200 --gpu 0 --output_dir /tmp/simple` | 立方体下落、撞 tet 平台、被摩擦留在上面 |
| **slope** | `./corex_demo --backend cuda --scene slope --frames 200 --gpu 0 --output_dir /tmp/slope` | 物体沿斜面持续下滑（默认 μ=0.4）|
| **stack** | `./corex_demo --backend cuda --scene stack --frames 200 --gpu 0 --output_dir /tmp/stack` | 3 块立方体在 60 帧内稳定堆叠并保持静止 |
| **domino** | `UIPC_DOMINO_TILT_DEG=12 UIPC_DOMINO_MU=0.15 UIPC_GROUND_MU=1.0 ./corex_demo --backend cuda --scene domino --frames 300 --gpu 0 --output_dir /tmp/domino` | 5 块完整链式倒下，D5 完全趴平 |
| **wrecking_ball** | `./corex_demo --backend cuda --scene wrecking_ball --frames 200 --gpu 0 --output_dir /tmp/wb` | 链式吊球摆动撞砌块，574 体全场景计算 |

### 4.2 完整 env 变量列表

| 场景 | 环境变量 | 默认 | 含义 |
|---|---|---|---|
| simple | `UIPC_SIMPLE_FORCE_MU` | 0.0 | 摩擦系数 |
|  | `UIPC_SIMPLE_FORCE_KAPPA_GPA` | 30 | 接触刚度 |
| slope | `UIPC_SLOPE_*` | — | 详见 `apps/examples/corex_demo/main.cpp` |
| stack | `UIPC_STACK_MU` | — | 立方体间摩擦 |
| domino | `UIPC_DOMINO_MU` | 0.25 | 骨牌之间摩擦 |
|  | `UIPC_DOMINO_KAPPA_GPA` | 40 | 接触刚度 |
|  | `UIPC_DOMINO_SPACING` | 0.55 | 骨牌中心间距（米） |
|  | `UIPC_DOMINO_TILT_DEG` | 0 | D1 初始倾角（度） |
|  | `UIPC_DOMINO_ABD_MPA` | 1000 | ABD 体杨氏模量（MPa） |
|  | `UIPC_DOMINO_VX` | 3.0 | D1 初始 +X 平动速度 |
|  | `UIPC_GROUND_MU` | =DOMINO_MU | 地面 vs 骨牌摩擦 |
|  | `UIPC_DOMINO_DENSITY` | 1000 | 密度 |
| 通用 | `UIPC_COREX_TRACE_LINEAR_SYSTEM=1` | off | 打印 PCG 每步 trace |

---

## 5. NVIDIA 端字节等价验收

仓库自带审计脚本，把 NVIDIA 路径的 22 个文件**预处理后**与 upstream 源码逐字节比对：

```bash
cd /path/to/libuipc
./tools/audit/verify_nvidia_branch_equiv.sh /path/to/upstream-libuipc-source
```

`/path/to/upstream-libuipc-source` 是参考的 upstream 仓库路径（在本机即 `/root/src`）。

期望输出：22 个 `OK: NVIDIA branch byte-identical`，0 个 `DIVERGE`。

如果某个文件 DIVERGE，说明在 NVIDIA 编译时（即 `UIPC_COREX_CUDA10_COMPAT` 不定义）走的代码与 upstream 不一致，需要修复对应文件的 `#else` 分支或 corex 守护。

---

## 6. 容易踩的坑

1. **千万不要只压缩 `git archive`**：20 个 `_corex` sidecar 是 untracked，会全部丢失。**必须 `tar` 整棵工作树**（如 §2.1）。
2. **NVIDIA 端不要开 `UIPC_MUDA_USE_COREX=ON`**：那会强行走 dylib v3 API + 单精度 + ivcore11，跟 NVIDIA 工具链与 ABI 完全对不上。
3. **NVIDIA 端也不要开 `UIPC_USE_FLOAT=ON`**：NVIDIA 标准是 `Float=double`。Corex 才是 `Float=float`，因为 Corex 的预编译 CUDA 库只有 float 版本。
4. **vcpkg 用 manifest 模式**：仓库根的 `vcpkg.json` + `vcpkg-configuration.json` 锁了 baseline 和依赖列表，别在目标机手动 `vcpkg install` 单个包，会版本错位。
5. **assets 路径**：`apps/examples/asset/` 是仓库自带的几何文件（已 commit），打包时确保没漏。运行时通过 `AssetDir::tetmesh_path()` 找。
6. **`include/` 目录**：本仓库的公共头已经升级到与 upstream 对齐（`ISanityCheckContext`、`IEngine::insert_sanity_checkers`、`create_proxy`、`to_rigid_body` 等新 API）。带过去后不需要再合并 upstream 头文件，否则会 redeclare。
7. **PCG fix-A 在 corex sidecar 里**：`linear_pcg_corex.cu.inc` 行 367/420 用的是教科书相对残差判据 `||r|| ≤ max(tol_rate·||b||, pcg_zero_tol)`。这是上次会话定位的 root cause（preconditioner 把 z 压到 ~1e-8 让 `|r·z|` 假装为零导致 dq=0 永远卡死）。NVIDIA 端不受影响，因为走 upstream 原版判据。**不要把 corex sidecar 的判据回退到 `|r·z|`**。

---

## 7. 验收 checklist

NVIDIA 机器拿到代码后建议按顺序：

1. `cmake -DUIPC_MUDA_USE_COREX=OFF ...` 配置成功
2. `cmake --build . -j` 编译成功
3. `verify_nvidia_branch_equiv.sh` 22/22 OK
4. 跑 `simple`：200 帧 exit=0，立方体落下、停在平台上
5. 跑 `slope`：200 帧 exit=0，物体下滑无穿透
6. 跑 `stack`：200 帧 exit=0，60 帧内堆叠稳定
7. 跑 `domino` 推荐参数：300 帧 exit=0，5 块完整链式倒下，D5 maxY ≈ 0.22
8. 跑 `wrecking_ball`：200 帧 exit=0，零 sanity error

每一步都通过，就说明 dual-branch 的 NVIDIA 路径在新机器上行为正确，可以开始正式 NVIDIA 测试。
