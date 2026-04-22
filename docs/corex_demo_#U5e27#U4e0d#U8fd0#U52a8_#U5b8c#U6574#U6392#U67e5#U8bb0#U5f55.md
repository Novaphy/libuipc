# Corex `corex_demo` 帧不运动完整排查记录（代码级）

## 1. 问题起点与症状

最初目标是让 `corex_demo` 在 Corex/天数环境下跑出可用仿真（至少满足：

1. `world.init(scene)` 能完成；
2. 主循环帧能推进；
3. 输出帧文件（如 surface）不是静止不变。

实际早期症状：

- 程序能启动，但在初始化阶段容易长时间卡住；
- 即使偶尔能走到输出，表现为“帧不运动/位移不更新”；
- 30s 冒烟经常超时，无法区分是慢初始化还是真实死锁。

---

## 2. 早期判断：先把“初始化卡住”与“物理不运动”分离

### 2.1 `SimEngine` 构造期 hello kernel 可疑

文件：`/mnt/libuipc/src/backends/cuda/engine/sim_engine.cu`

关键判断：

- Corex 上，构造期首个 kernel + 默认流同步行为可能导致长时间阻塞；
- 当时 hello 路径还可能牵涉 `KernelCout`/Logger 初始化，带来额外 `cudaMalloc` 与同步噪声。

采取的修改方向（已落地）：

1. 构造中先 `cudaSetDevice(device_id)`，避免仅 `cudaGetDeviceProperties` 但设备上下文未显式设定；
2. Corex 下 hello kernel 默认跳过，仅在 `UIPC_FORCE_CUDA_HELLO_KERNEL=1` 时执行；
3. hello 改为最小空 kernel + `cudaDeviceSynchronize`，避免 `KernelCout` 引入复杂初始化路径。

这一步的意义：把“卡在构造期”的噪声移开，让问题尽量暴露在真实 `world.init` 路径中。

---

## 3. 针对“帧不运动”先做场景与开关层面的减负

文件：`/mnt/libuipc/apps/examples/corex_demo/main.cpp`

做过的核心调整：

1. Corex 构建默认场景改为 `simple`（避免 `wrecking_ball` 资产/接触链路复杂度）；
2. `simple` 下默认关闭 `contact` 与 `friction`；
3. 关闭 `sanity_check`（在 Corex 先优先跑通主链路）；
4. 增加 `world.init(scene)` 前后、首尾帧调试输出；
5. 增加 host 侧质量与体积调试打印（确认基础质量参数非零）。

目的：

- 先确认仿真主循环能推进；
- 把接触系统等高复杂度模块暂时从“必经链路”里拿掉；
- 排除“配置导致没有动力学更新”的伪问题。

---

## 4. 关键转折：并非“帧不运动”本身，而是卡在系统构建阶段

后续日志显示，程序大量时间卡在：

- `world.init(scene)` -> `SimEngine::do_init` -> `build_systems()`

而非时间推进循环本身。

为了定位，在 `SimEngine::build_systems()` 加了 creator 序号日志：

文件：`/mnt/libuipc/src/backends/common/sim_engine.cpp`

日志形式：

- `creator #N calling...`
- `creator #N done (ptr=...)`

定位结果是：卡点会向后移动，但反复出现在“某个 creator 构造期”，而不是统一固定在一个地方。

这说明根因更像是：

- 某类系统在构造期触发了 Corex 上不稳定/超慢的 GPU 资源路径（常见是 `DeviceVar` 默认构造内的分配+同步）。

---

## 5. 编译层面的第一批硬错误与修复

### 5.1 `spmv.cu` 的 `atomicAdd(double)` 在 Corex 报错

文件：`/mnt/libuipc/src/backends/cuda/linear_system/spmv.cu`

错误现象：

- `cannot compile this builtin function yet`
- 指向 `__nvvm_atom_add_gen_d`

根因：

- `Float=double` 情况下直接调用 `atomicAdd(double*, double)`；
- Corex clang 路径对该 builtin 支持不完整。

修复：

- 改为 `muda::atomic_add(d_dot.data(), dot_local)`；
- `muda::atomic_add<double>` 走 CAS 退化路径。

### 5.2 `rbk_*` kernel 在 Iluvatar `llc` 崩溃

同文件：`spmv.cu`

现象：

- 后端指令选择阶段 `CannotYetSelect`，`llc` abort；
- 栈信息指向 `rbk_spmv/rbk_sym_spmv` 相关复杂 warp reduce/shuffle 路径。

修复策略（Corex 特化）：

- 增加 `UIPC_SPMV_ILUVATAR_RBK_WORKAROUND` 分支；
- Corex 下 `rbk_spmv`、`rbk_sym_spmv` 退化到 `sym_spmv`；
- `rbk_sym_spmv_dot` 退化为 `sym_spmv + 并行 dot`。

目标是先换稳定性，再考虑性能。

---

## 6. 运行时卡点从 creator #2 向后推进（说明局部修复有效）

在后续多轮日志中可见：

- 早期卡在 creator #2（`ABDLineSearchReporter`）；
- 修复后能过 #2/#3，继续卡在 #4；
- 再修复后能推进到 #30+，说明不是单点，而是多个系统构造路径都可能触发同类问题。

---

## 7. 围绕 `DeviceVar` 构造期分配做的重点改造

### 7.1 `muda::DeviceVar` 默认构造行为

文件：`/mnt/libuipc/external/muda/src/muda/buffer/details/device_var.inl`

原行为：

- `DeviceVar<T>::DeviceVar()` 里 `Memory().alloc(...).wait()`

在 Corex 下风险：

- `wait()` 对默认流会走同步（有时退化成 device 级同步）；
- 在系统构造期（`build_systems`）触发，容易表现为“像死锁”。

做过的调整：

- 将默认构造改为同步 `alloc(..., false)`，避免额外 `.wait()`。

### 7.2 把部分系统里的 `DeviceVar` 改为延迟分配 `DeviceBuffer`

代表文件：

- `/mnt/libuipc/src/backends/cuda/affine_body/abd_line_search_reporter.h`
- `/mnt/libuipc/src/backends/cuda/affine_body/abd_line_search_reporter.cu`
- `/mnt/libuipc/src/backends/cuda/affine_body/abd_tolerance_checker.cu`
- `/mnt/libuipc/src/backends/cuda/collision_detection/filters/easy_vertex_half_plane_trajectory_filter.h`
- `/mnt/libuipc/src/backends/cuda/collision_detection/filters/easy_vertex_half_plane_trajectory_filter.cu`

思路：

- 避免在 SimSystem 构造函数时隐式 `cudaMalloc`；
- 改为在 `do_build()/init` 或首次使用时 `resize(1)`；
- 主机读回改用 `view().copy_to(...)`。

观察结果：

- creator 卡点确实持续后移，说明这类改造在工程上是有效的。

---

## 8. 工具链/构建链路的反复问题（后期主阻塞）

当排查逐步深入后，主阻塞从“业务代码”转为“工具链组合兼容”。

### 8.1 `/usr/bin/g++-10` 缺失

现象：

- `CMAKE_CXX_COMPILER=/usr/bin/g++-10` 在缓存中，但系统无此文件；
- 触发 CMake regenerate 后直接失败。

### 8.2 `g++-9` 不满足项目 C++20 需求

现象：

- `std::span`/`requires` 报错；
- host 侧 `.cpp` 无法通过。

### 8.3 切到 `/private` 的 GCC12 后，新冲突出现

1. Host C++ 基本可编；
2. CUDA 路径先报 `<span>/<cmath>` 寻址问题（wrapper 未正确引入该工具链 C++ 头）；
3. 后续又遇到 Corex `host_defines.h` 中 `__noinline__` 定义与 libstdc++ 宏展开冲突（`__attribute__((__noinline__))` 相关语法破坏）。

对应文件：

- `/private/libuipc/cmake/corex-clang-cuda-wrapper.sh`
- `/usr/local/corex/include/crt/host_defines.h`

这部分已超出单纯业务代码修改，属于“编译器 + CUDA 前端 + 标准库”三者兼容问题。

---

## 9. 当前状态（截至本记录）

1. 已完成大量 Corex 兼容修改（尤其 `spmv` 与多处 `DeviceVar` 构造期风险点）；
2. creator 构建日志显示“可通过的系统数量明显增加”，说明运行时卡点修复方向正确；
3. 但目前最终阻塞在工具链组合稳定性，尚未完成“可用仿真”闭环（`world.init OK` -> 连续帧推进 -> 输出验证）。

---

## 10. 后续建议（按优先级）

### P0：先固定一套稳定的“Host + CUDA”编译组合

建议先保证：

- Host 编译器路径稳定（不再引用被移除目录）；
- CUDA wrapper 对该 host 工具链的 C++ 头与宏行为可控；
- 一次 clean configure + build 能稳定复现。

否则每次都可能在不同编译阶段被工具链噪声打断，无法继续业务排查。

### P1：保留 creator 序号日志直到跑通

文件：`src/backends/common/sim_engine.cpp`

继续保留：

- `creator #N calling/done`

价值：

- 一旦再次卡住，能立刻定位是哪个系统构造路径；
- 可与系统名日志（`creator type`）对齐。

### P2：继续“构造期去 GPU 分配”改造

对 remaining 系统，优先筛查：

- 类成员 `muda::DeviceVar<...>`；
- 构造时触发 `Memory().alloc().wait()` 的路径。

策略：

- 迁移为 `DeviceBuffer` + 延迟 `resize(1)`；
- 或把分配移动到 `do_build/do_init` 并避免默认流全局同步。

### P3：跑通验收标准

最低验收：

1. `world.init(scene)` 打印完成；
2. 帧号连续推进（例如 10~50 帧）；
3. 输出文件存在且相邻帧数据发生变化（非静止复制）。

---

## 11. 本轮涉及的关键文件清单（便于快速索引）

- 引擎初始化与构建：
  - `/mnt/libuipc/src/backends/cuda/engine/sim_engine.cu`
  - `/mnt/libuipc/src/backends/cuda/engine/sim_engine_do_init.cu`
  - `/mnt/libuipc/src/backends/common/sim_engine.cpp`
  - `/mnt/libuipc/src/backends/common/sim_system_auto_register.h`

- demo 场景与调试输出：
  - `/mnt/libuipc/apps/examples/corex_demo/main.cpp`
  - `/mnt/libuipc/apps/examples/corex_demo/CMakeLists.txt`

- 线性系统与 Corex CUDA 编译兼容：
  - `/mnt/libuipc/src/backends/cuda/linear_system/spmv.cu`

- `DeviceVar` 与分配同步路径：
  - `/mnt/libuipc/external/muda/src/muda/buffer/details/device_var.inl`
  - `/mnt/libuipc/src/backends/cuda/affine_body/abd_line_search_reporter.h`
  - `/mnt/libuipc/src/backends/cuda/affine_body/abd_line_search_reporter.cu`
  - `/mnt/libuipc/src/backends/cuda/affine_body/abd_tolerance_checker.cu`
  - `/mnt/libuipc/src/backends/cuda/collision_detection/filters/easy_vertex_half_plane_trajectory_filter.h`
  - `/mnt/libuipc/src/backends/cuda/collision_detection/filters/easy_vertex_half_plane_trajectory_filter.cu`

- 编译器 wrapper / 工具链相关：
  - `/private/libuipc/cmake/corex-clang-cuda-wrapper.sh`
  - `/usr/local/corex/include/crt/host_defines.h`

---

## 12. 结语

“帧不运动”并不单纯是物理参数问题，而是经历了：

1. 初始化卡住（构造期 GPU 路径）；
2. 编译器 builtin 与后端 codegen 问题（`atomicAdd(double)` / `llc`）；
3. 多系统构造阶段的资源分配阻塞；
4. 最终进入工具链兼容层面的硬冲突。

目前最关键是先稳定“可重复编译”的工具链组合，随后继续沿 creator 链路做剩余系统构造期去阻塞，即可恢复到“实际可运动仿真”的路径。

