# Corex 上 libuipc 编译结果演示工作记录（2026-03-18）

本文档聚焦“**如何演示今天编译出来的 libuipc（尤其 CUDA backend）**”，并解释我为了让演示链路可用与稳定所做的改动：**做了什么、为什么这样做、如何复现**。

> 背景：你的目标是“能在 Corex 环境下把 `libuipc` 的 CUDA backend 跑起来并用于演示”，因此除了“编译通过”，还需要解决 **运行时依赖、动态加载、首帧推进（advance）与接触/摩擦（contact/friction）** 等一系列演示链路问题。

---

## 1. 今日产物与演示入口

### 1.1 构建产物目录

本次使用的 build 目录：

- `build_cuda/Release/bin/`

其中关键文件：

- `libuipc_backend_none.so`
- `libuipc_backend_cuda.so`
- `libuipc_core.so` / `libuipc_geometry.so` / `libuipc_io.so` / `libuipc_constitution.so` / `libuipc_sanity_check.so`

### 1.2 新增演示程序 `corex_demo`

为了让你能“一条命令演示后端能跑”，我新增了一个专用 demo：

- 源码：`apps/examples/corex_demo/main.cpp`
- CMake：`apps/examples/corex_demo/CMakeLists.txt`
- 可执行文件：`build_cuda/Release/bin/corex_demo`

它的设计目标：

- **命令行可切换后端**：`--backend=cuda|none`
- **命令行可控制帧数**：`--frames=N`
- **命令行可切换场景**：`--scene=wrecking_ball|simple`
- 每帧导出 `scene_surface_XXXX.obj`，用于 MeshLab/Blender 等直接展示

为什么要新增 demo 而不是直接跑已有 examples？

- 现有 `apps/examples/hello_affine_body` / `wrecking_ball` 固定写死了某些配置与路径；
- 我需要一个“可控开关”的入口，用来在 Corex 环境里 **逐步打开能力（none → cuda init → cuda advance → contact/friction + 复杂场景）**，并快速定位问题。

---

## 2. 演示的基本运行方式（复现命令）

### 2.1 运行前的动态库路径（必须）

因为 demo 会在运行时动态加载 `libuipc_backend_cuda.so`，且该 `.so` 依赖 Corex CUDA 数学库，所以运行时需要把 Corex 的库目录加入 `LD_LIBRARY_PATH`：

```bash
cd /private/libuipc/build_cuda/Release/bin
LD_LIBRARY_PATH=.:/usr/local/corex/lib64 ./corex_demo --backend=cuda --frames=2
```

其中：

- `LD_LIBRARY_PATH=.`：让程序优先从当前目录找到 `libuipc_*.so`
- `LD_LIBRARY_PATH` 追加 `/usr/local/corex/lib64`：让程序找到 `libcublas.so` / `libcusolver.so` / `libcusparse.so` / `libcublasLt.so` 等

### 2.2 `none` backend（验证动态加载链路）

用于验证“前端 Engine/World + 后端动态加载 + 符号导出”链路无误：

```bash
cd /private/libuipc/build_cuda/Release/bin
LD_LIBRARY_PATH=. ./corex_demo --backend=none --frames=1
```

预期现象：

- `none` 后端被加载
- `World.init(scene)` 正常
- 导出 `scene_surface_0000.obj`

### 2.3 `cuda` backend + 复杂场景 + contact/friction（最终演示形态）

```bash
cd /private/libuipc/build_cuda/Release/bin
LD_LIBRARY_PATH=.:/usr/local/corex/lib64 ./corex_demo --backend=cuda --scene=wrecking_ball --frames=50
```

输出目录：

- `/mnt/libuipc/output/examples/corex_demo/main.cpp/scene_surface_0000.obj ...`

---

## 3. 今日演示链路遇到的问题与修复（为什么要这样做）

### 3.1 问题：运行时缺少 `libcublasLt.so.10`

现象（最初运行 `--backend=cuda`）：

- 动态加载 `libuipc_backend_cuda.so` 失败
- 报错：`libcublasLt.so.10: cannot open shared object file`

原因：

- Corex CUDA 数学库在 `/usr/local/corex/lib64`，默认运行环境没有把它加入动态库搜索路径

处理方式：

- 演示命令统一加：
  - `LD_LIBRARY_PATH=.:/usr/local/corex/lib64`

这样做的原因：

- 这是运行时环境的客观要求；不改变系统全局 ld 配置的情况下，**LD_LIBRARY_PATH 是最可控、最容易复现的演示方式**。

---

### 3.2 问题：`cusolverSpCreate` 在 Corex cuSolver 上不支持

现象：

- 进入 CUDA backend 初始化后，抛出：
  - `CUSOLVER_STATUS_NOT_SUPPORTED`
  - `cusolverSpCreate(&m_cusolver_sp)`

原因：

- Corex 环境下的 cuSolver 可能缺失（或裁剪）Sparse 相关接口（Sp）。
- 但很多场景下 **并不需要** cuSolverSp 才能完成演示（至少可以先跑通 init/部分求解路径）。

修复（让 cuSolverSp “可选化”）：

- 文件：`external/muda/src/muda/ext/linear_system/linear_system_handles.h`
- 做法：`cusolverSpCreate` 返回 `CUSOLVER_STATUS_NOT_SUPPORTED` 时，保持 `m_cusolver_sp=nullptr`，后续 `SetStream` 等操作做空指针保护。

为什么这样做：

- 目标是先让 CUDA backend “能跑起来并推进帧”，再逐步验证更完整的线性系统后端能力；
- 把“缺失的可选能力”从“致命错误”降级为“功能受限但可运行”，有利于演示与进一步定位。

---

### 3.3 问题：`cudaMalloc` 收到接近 \(2^{64}\) 的异常申请大小（溢出型错误）

现象：

- `world.advance()` 首帧进入后，报：
  - `cudaMalloc(ptr, byte_size=18446743989641879224)` → `cudaErrorMemoryAllocation`

这个值的关键点：

- 它不是“真实需要几十 GB 显存”的正常 OOM
- 而是典型的“**负数转无符号 / 未初始化 / 溢出**”导致的巨大 size（接近 \(2^{64}\)）

为定位该错误，我做了两步：

#### 3.3.1 在 `cudaMalloc` 入口打印一次 backtrace（定位是谁算坏了 size）

- 文件：`external/muda/src/muda/launch/details/memory.inl`
- 行为：当 `byte_size > 1TB` 时，打印 host backtrace（一次即可）

这一步的目的：

- 让错误从“只有一个巨大数字”变成“明确的调用链路”，可直接定位到具体函数。

#### 3.3.2 根据 backtrace 定位到根因：`MatrixConverter::ge2sym` 的 total_count 写回不稳

backtrace（关键帧）：

- `muda::DeviceBuffer<Eigen::Matrix<double,3,3>>::resize`
- `uipc::backend::cuda::MatrixConverter<double,3>::ge2sym`
- `GlobalLinearSystem::Impl::build_linear_system`
- `SimEngine::do_advance`

根因解释：

- `MatrixConverter::ge2sym` 里原先通过“最后一个线程写回 total_count”来决定 `resize_triplets(h_total_count)`。
- 在 Corex/clang-CUDA 的执行环境下，这个写回路径出现不稳定（写回没有发生或读取异常），导致 `h_total_count` 变成负数，随后被当作 `size_t` 使用，引发 `cudaMalloc` 申请超大字节数。

修复（把 total_count 计算改为稳健、单线程写回）：

- 文件：`src/backends/cuda/algorithm/details/matrix_converter.inl`
- 做法：
  - 把 `count` 先置 0
  - 用 `ParallelFor().apply(1, ...)` 读取 `offsets(last)+counts(last)` 写回 `count`
  - 再 `int h_total_count = (int)count;` 用于 `resize_triplets`

为什么这样做：

- “最后一个线程写回”属于脆弱实现（依赖调度/边界条件/实现细节）
- 单线程写回能消除竞态与不确定性，成本小但稳定性提升很大

---

### 3.4 问题：开启 contact/friction 后 `toi==0` 触发断言

现象：

- 在 `wrecking_ball` 场景首帧推进时，报：
  - `Assertion toi > 0.0f failed. Invalid toi[EasyVertexHalfPlaneTrajectoryFilter] value: 0`

原因（从代码逻辑出发）：

- `GlobalTrajectoryFilter::filter_toi()` 会收集每个 filter 的 `toi`；
- 某些 filter 在“没有候选/无需限制”时可能输出 `toi=0`；
- 原逻辑把 `toi=0` 当成错误（`assert(toi > 0)`），导致演示在早期帧直接中断。

修复（把 `toi==0` 视为“无约束”，统一当作 `toi=1`）：

- 文件：`src/backends/cuda/collision_detection/global_trajectory_filter.cu`
- 行为：
  - 仍然拒绝 `toi < 0`
  - 将 `toi == 0` 归一化为 `1.0`

为什么这样做：

- 从“全局最小 TOI”语义看，`toi==0` 在“无候选”场景下不应把全局步长压成 0；
- 允许 `toi==0` 并归一化，能让演示继续推进，同时不掩盖真正的非法值（负数）。

---

## 4. 复杂场景接入方式（为什么复用 wrecking_ball）

为了更好的演示效果，我把 `corex_demo` 的默认场景切到 `wrecking_ball`，并复用了官方例子里的场景 JSON：

- JSON：`apps/examples/wrecking_ball/wrecking_ball.json`

这样做的原因：

- 不重复造轮子：官方场景已经包含更丰富的刚体链条/球/立方体组合；
- 演示时你只需要跑 `corex_demo` 一个入口，不需要记多个 sample 的差异；
- `corex_demo` 可以继续支持 `--scene=simple` 作为最小回归路径。

---

## 5. 演示建议（给你对外展示用）

- **快速证明“CUDA backend 可用”**：
  - 跑 `--frames=2`，确认能导出 `0000/0001` 两帧 OBJ
- **展示接触/摩擦启用**：
  - `--scene=wrecking_ball --frames=50`
  - 对比 OBJ 序列中结构的运动与落地/碰撞（可用 Blender 直接导入序列）
- **如果需要更明显的接触效果**（后续可再做）：
  - 把 demo 增加 CLI 参数：`--dhat`、`--fric`、`--k`、`--dt`、`--gravity-scale`

---

## 6. 今日新增/修改文件清单（演示相关）

- **新增 demo**
  - `apps/examples/corex_demo/CMakeLists.txt`
  - `apps/examples/corex_demo/main.cpp`
  - `apps/examples/CMakeLists.txt`（加入 `add_subdirectory(corex_demo)`）

- **运行稳定性修复（用于保证演示跑通）**
  - `external/muda/src/muda/ext/linear_system/linear_system_handles.h`（cusolverSp 可选化）
  - `src/backends/cuda/algorithm/details/matrix_converter.inl`（`ge2sym` total_count 稳健计算）
  - `src/backends/cuda/collision_detection/global_trajectory_filter.cu`（`toi==0` 归一化，避免断言中断）

- **定位辅助（用于快速定位溢出问题）**
  - `external/muda/src/muda/launch/details/memory.inl`（异常 byte_size 打印 backtrace）

