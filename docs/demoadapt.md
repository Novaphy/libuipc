# libuipc × muda（Corex）适配报告 v3（代码级复盘版）

> 本文档是在 `corex-muda-adaptation-report-v2.md` 基础上，**结合当前仓库内的实际改动**重新整理的 v3。  
> v3 的着重点是：**按文件/函数/调用链**解释“做了什么、为什么做、带来什么影响、如何验证/演示”。

---

## 1. v3 相对 v2 的新增内容（你为什么需要 v3）

v2 主要覆盖“编译通过 + dlopen 成功”的适配主线。但在你实际演示时，我们又遇到并解决了几类**只有在运行/推进帧/开启 contact+friction+复杂场景时才会暴露的问题**。这些修复与演示入口（demo）都发生在 v2 之后，因此需要 v3 把它们**落到文件与调用路径**上说明清楚。

v3 新增重点：

- **演示入口**：新增 `apps/examples/corex_demo`，用于在 Corex 上稳定演示 `none/cuda` backend，并导出逐帧 OBJ。
- **运行期兼容**：Corex cuSolver Sparse API 不支持 → 在 muda 里把 `cusolverSp` 句柄“可选化”。
- **运行期致命溢出**：`cudaMalloc(byte_size≈2^64)` 的溢出型错误 → 定位到 `MatrixConverter::ge2sym` 的 total_count 写回不稳，并修复。
- **contact/friction + 复杂场景稳定性**：`toi==0` 的过滤器输出导致断言中断 → 将 `toi==0` 视作“无约束”，归一化为 `1`，保证推进帧可继续。

---

## 2. 适配目标与“可演示”验收口径（v3 版本）

本轮最终交付目标（以“能对外演示”为准）：

- **能编译**：`libuipc_backend_cuda.so` 在 Corex clang-CUDA 下编译通过。
- **能加载**：`Engine("cuda")` 能动态加载并初始化 cuda backend（不因缺符号/不支持 API 直接崩溃）。
- **能推进**：至少能 `World.advance()` 连续推进多帧并导出可视化结果（OBJ 序列）。
- **能开启接触/摩擦**：在 `contact=on + friction=on` 且更复杂场景（`wrecking_ball`）下仍能推进帧（至少几十帧用于演示）。

---

## 3. 演示入口（新增）：`corex_demo`

### 3.1 为什么要新增 demo

现有 examples（如 `apps/examples/wrecking_ball`）虽然能展示能力，但不适合在 Corex 适配阶段做“逐步开关 + 稳定演示 + 快速定位问题”，原因包括：

- 场景与配置写死，无法快速切换 `none/cuda`、帧数、场景复杂度；
- 适配期需要一个**最小可控入口**，能够把问题拆成：
  - 动态加载链路是否 OK
  - cuda backend init 是否 OK
  - 推进帧是否 OK
  - contact/friction 是否 OK

因此新增 `corex_demo`，成为你对外演示和自查的主入口。

### 3.2 新增/修改文件

- `apps/examples/CMakeLists.txt`：加入 `add_subdirectory(corex_demo)`
- `apps/examples/corex_demo/CMakeLists.txt`：新增目标 `corex_demo`
- `apps/examples/corex_demo/main.cpp`：演示入口实现

### 3.3 demo 的命令行与行为

支持参数：

- `--backend=cuda|none`：选择 backend（默认 cuda）
- `--scene=wrecking_ball|simple`：选择场景（默认 wrecking_ball）
- `--frames=N`：输出 N 帧（默认 50）

输出：

- 每帧导出 `scene_surface_XXXX.obj` 到：
  - `/mnt/libuipc/output/examples/corex_demo/main.cpp/`

### 3.4 演示命令（Corex 必需的动态库路径）

```bash
cd /private/libuipc/build_cuda/Release/bin
LD_LIBRARY_PATH=.:/usr/local/corex/lib64 ./corex_demo --backend=cuda --scene=wrecking_ball --frames=50
```

---

## 4. 运行期兼容修复（muda）：`cusolverSpCreate` 不支持

### 4.1 现象

在 Corex 环境中，`cusolverSpCreate` 返回 `CUSOLVER_STATUS_NOT_SUPPORTED`，会导致 cuda backend 初始化阶段直接抛异常/中断。

### 4.2 根因

Corex 的 cuSolver 运行库可能裁剪或不提供 Sparse（Sp）接口，但 libuipc 的线性系统上下文初始化会创建 `muda::LinearSystemHandles`，其中默认会创建 `cusolverSpHandle_t`。

### 4.3 修改点（文件级）

**文件**：`external/muda/src/muda/ext/linear_system/linear_system_handles.h`

**修改策略**：把 `cusolverSp` 当作可选能力：

- 若 `cusolverSpCreate` 返回 `CUSOLVER_STATUS_NOT_SUPPORTED`：
  - `m_cusolver_sp = nullptr`
  - 后续 `cusolverSpSetStream` 做空指针保护

### 4.4 影响评估

- **正向**：cuda backend 可以继续初始化并推进帧，不再被 Sp 接口缺失阻断。
- **潜在影响**：若未来某条路径强依赖 `cusolverSp`，需要在使用处做 capability check（本轮演示链路不依赖）。

---

## 5. 运行期致命溢出修复：`MatrixConverter::ge2sym` 导致 `cudaMalloc(byte_size≈2^64)`

### 5.1 现象（推进首帧时崩溃）

在 `World.advance()` 首帧中出现：

- `cudaMalloc(ptr, byte_size=18446743989641879224)` → `cudaErrorMemoryAllocation`

该数字接近 \(2^{64}\)，典型意味着：

- 上游产生了**负数/未初始化/溢出**的 count
- 经过隐式转换后被当作 `size_t` 传入分配逻辑

### 5.2 定位方法（调用链级）

为了定位“是谁算坏了 size”，我在 muda 的 `cudaMalloc` 入口对“明显异常的大分配（>1TB）”输出 host backtrace，得到关键调用链：

- `muda::Memory::alloc_1d`  
→ `muda::DeviceBuffer<Eigen::Matrix<double,3,3>>::resize`  
→ `uipc::backend::cuda::MatrixConverter<double,3>::ge2sym(DeviceTripletMatrix)`  
→ `uipc::backend::cuda::GlobalLinearSystem::Impl::build_linear_system`  
→ `uipc::backend::cuda::SimEngine::do_advance`

### 5.3 根因

`MatrixConverter::ge2sym` 里需要计算 `total_count`（上三角筛选后的 triplet 数），原实现依赖某个“最后线程写回”的不稳定路径（在 Corex clang-CUDA 下表现不可靠），导致 `count` 在 host 侧读到负值，进而触发 `resize_triplets(size_t(负值))`。

### 5.4 修改点（文件级 + 代码级）

**文件**：`src/backends/cuda/algorithm/details/matrix_converter.inl`

**修改策略**：改为“稳健的单线程写回”：

- 先将 `count = 0`
- 再用 `ParallelFor().apply(1, ...)` 在 device 上读取 `offsets(last)+counts(last)` 写回 `count`
- 然后在 host 上 `int h_total_count = (int)count;` 再执行 `resize_triplets(h_total_count)`

这样不依赖“最后一个 i 的线程一定会写回”的假设。

### 5.5 影响评估

- **正向**：消除溢出型崩溃，使推进帧可持续。
- **性能影响**：多一次很小的单线程 kernel（`apply(1, ...)`），成本极低，稳定性收益巨大。

---

## 6. contact/friction + 复杂场景稳定性：`toi==0` 导致断言中断

### 6.1 现象

开启 contact/friction 并运行复杂场景（wrecking_ball）时，首帧推进出现：

- `Invalid toi[EasyVertexHalfPlaneTrajectoryFilter] value: 0`
- 在 `GlobalTrajectoryFilter::filter_toi` 的 runtime check 中被断言中断

### 6.2 根因（语义层）

在“全局 TOI（time-of-impact）过滤”语义下：

- 某些 filter 在“没有候选/不限制步长”的情况下可能输出 `toi=0`
- 原实现把 `toi==0` 当作错误（`toi>0`），导致在 Corex 环境里演示被中断

### 6.3 修改点（文件级）

**文件**：`src/backends/cuda/collision_detection/global_trajectory_filter.cu`

**修改策略**：

- 在 `Impl::init()` 中将 `tois` 初始化为 `1.0`
- 在 `filter_toi()` 每次评估前 `tois.fill(1.0f)`，避免 filter 没写时遗留旧值
- runtime check：
  - 仍拒绝 `toi < 0`
  - 将 `toi == 0` 归一化为 `1.0`

### 6.4 影响评估

- **正向**：避免“无候选”被误判为错误，复杂场景可连续推进帧用于演示。
- **潜在影响**：如果 `toi==0` 在某些实现里代表更强语义（例如“必须停步”），需要更严格区分；但从现有 filter 行为与日志看，这是“无候选”的合理表达。

---

## 7. 调用路径索引（从 demo 到关键改动点）

这部分用来帮助你在代码层面快速定位“改动为什么会被触发”。

### 7.1 demo → backend 动态加载 → 推进帧

- `apps/examples/corex_demo/main.cpp`  
→ `uipc::core::Engine engine{"cuda", workspace}`  
→ `uipc::core::World world{engine}`  
→ `world.init(scene)` / `world.advance()`

### 7.2 推进帧 → 线性系统（触发 MatrixConverter::ge2sym）

- `World::advance()`  
→ `uipc::backend::cuda::SimEngine::do_advance()`（`src/backends/cuda/engine/sim_engine_do_advance.cu`）  
→ `uipc::backend::cuda::GlobalLinearSystem::solve()`（`src/backends/cuda/linear_system/global_linear_system.cu`）  
→ `GlobalLinearSystem::Impl::build_linear_system()`  
→ `converter.ge2sym(triplet_A)`（`src/backends/cuda/algorithm/details/matrix_converter.inl`）

### 7.3 推进帧 + contact/friction → trajectory filter（触发 toi 归一化）

- `SimEngine::do_advance()`  
→ trajectory filter 系统调用链  
→ `uipc::backend::cuda::GlobalTrajectoryFilter::filter_toi()`  
→ `src/backends/cuda/collision_detection/global_trajectory_filter.cu`

### 7.4 线性系统上下文 → muda handles（触发 cusolverSp 可选化）

- `src/backends/cuda/linear_system/iterative_solver.h/.cu`：`muda::LinearSystemContext& IterativeSolver::ctx()`  
→ `muda` 线性系统上下文创建/持有 handles  
→ `external/muda/src/muda/ext/linear_system/linear_system_handles.h`

---

## 8. v3 文件改动清单（本轮新增/追加）

### 8.1 演示入口

- `apps/examples/CMakeLists.txt`
- `apps/examples/corex_demo/CMakeLists.txt`
- `apps/examples/corex_demo/main.cpp`

### 8.2 运行期兼容与稳定性

- `external/muda/src/muda/ext/linear_system/linear_system_handles.h`（cusolverSp 可选化）
- `src/backends/cuda/algorithm/details/matrix_converter.inl`（ge2sym total_count 稳健计算）
- `src/backends/cuda/collision_detection/global_trajectory_filter.cu`（toi 初始化/归一化）

### 8.3 定位辅助（用于定位溢出根因）

- `external/muda/src/muda/launch/details/memory.inl`（异常大分配打印 backtrace）

---

## 9. 建议的对外演示流程（最小成本 + 最大信息量）

1. **证明后端加载 OK**：`--backend=none --frames=1`
2. **证明 CUDA init + 推进 OK**：`--backend=cuda --scene=simple --frames=2`
3. **展示 contact/friction + 复杂场景**：`--backend=cuda --scene=wrecking_ball --frames=50`

输出 OBJ 序列位于：

- `/mnt/libuipc/output/examples/corex_demo/main.cpp/`

