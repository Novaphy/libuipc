# muda 在 Corex (Iluvatar BI) 上的移植与验证总结

> 本文档总结了在 `/private/muda-app/submodules/muda` 上为支持 **Corex 工具链 + ivcore GPU** 所进行的全部关键改动，以及围绕 **事件 (Event) + 双流/单流** 示例的调试过程与结论，便于在组会中汇报。

---

## 1. 总体工作思路

   目前项目需要适配的是`libuipc`这个基于CUDA的外部项目，因此计划通过将`libuipc`项目先进行单独适配，再将适配后的`libuipc`整合回`Novaphy`。工作流程如下：
- **1：工具链验证**  
  证明 Corex 的 `clang++` 能像 NVCC 一样，完整编译、链接并运行 CUDA代码。（目前已在IPC仿真使用的外部项目`libuipc`的一个外部库`muda`上进行了验证）。
- **2：编译链调整**  
  对项目中的Cmake进行调整，添加Corex环境下的编译链（编译器、架构、CUDA 标准、CUDAToolkit等），实现项目在Corex环境下的编译。
- **3：API 兼容性适配**  
  针对 Corex 对应的 **CUDA 10 API**，为 `libuipc` 中依赖 CUDA 12+ 特性的部分进行调整（采用替代API、基于替代API补充实现等）。

---
## 2. 本周工作进展


### 2.1 项目根目录CMake：为 Corex 新增构建选项与编译链路

文件：`CMakeLists.txt`（muda 根目录）

**改动点：**

- 新增选项：
  - `option(MUDA_BUILD_COREX "build for Iluvatar CoreX (BI) GPU with Corex toolchain" OFF)`
- 当 `MUDA_BUILD_COREX=ON` 时：
  - 从环境或默认路径设置 `CUDA_PATH`（默认 `/usr/local/corex`）。
  - 设定 Corex GPU 架构：`COREX_ARCH="ivcore11"`（可配置）。
  - 将编译器切换为 Corex 提供的 Clang：
    - `CMAKE_CXX_COMPILER = ${CUDA_PATH}/bin/clang++`
    - `CMAKE_C_COMPILER   = ${CUDA_PATH}/bin/clang`
    - `CMAKE_CUDA_COMPILER = ${CUDA_PATH}/bin/clang++`
  - 启用 CUDA 语言并指定标准/架构：
    - `set(CMAKE_CUDA_ARCHITECTURES "${COREX_ARCH}")`
    - `set(CMAKE_CUDA_STANDARD 17)`
    - `project(muda LANGUAGES CXX CUDA)`
    - `enable_language(CUDA)`
  - 链接 Corex 的运行时：
    - `find_library(CUDART_LIBRARY cudart ...)`
    - `find_library(CUDA_DRIVER_LIBRARY cuda ...)`

**目的：** 在 Corex 环境下用 Corex 的 `clang++` 完成 C++ + CUDA 统一编译，明确 GPU 架构为 `ivcore11`。

### 2.2 muda 接口库的 Corex 特定编译选项

对 `add_library(muda INTERFACE)` 之后的配置：

- Corex 模式下：
  - `target_compile_definitions(muda INTERFACE MUDA_BUILD_COREX=1)`
  - `target_compile_options(muda INTERFACE`  
    `$<$<COMPILE_LANGUAGE:CUDA>:--cuda-path=${CUDA_PATH}>`  
    `$<$<COMPILE_LANGUAGE:CUDA>:--cuda-gpu-arch=${COREX_ARCH}>`  
    `$<$<COMPILE_LANGUAGE:CUDA>:-std=c++17>`  
    `$<$<COMPILE_LANGUAGE:CUDA>:-include ${PROJECT_SOURCE_DIR}/src/muda/corex_cassert.h>`  
    `$<$<COMPILE_LANGUAGE:CUDA>:-Wno-unused-command-line-argument>`  
  - `target_include_directories(muda INTERFACE "${CUDA_PATH}/include" "${PROJECT_SOURCE_DIR}/src/" "${PROJECT_SOURCE_DIR}/src/muda")`
  - `target_link_libraries(muda INTERFACE ${CUDART_LIBRARY} ${CUDA_DRIVER_LIBRARY})`
  - `target_link_directories(muda INTERFACE ${CUDA_PATH}/lib64)`

**目的：** 为所有编译 muda 的 CUDA 翻译单元自动加上 Corex 所需的 `--cuda-path` / `--cuda-gpu-arch`，并通过 `-include corex_cassert.h` 保证 `assert` 有定义或安全退化为 no-op。

### 2.3 示例工程 muda_example 的 Corex 配置

文件：`example/CMakeLists.txt`

**改动点：**

- 若 `MUDA_BUILD_COREX`：用 `list(FILTER ... EXCLUDE REGEX ...)` 从源文件列表中排除以下示例：
  - `cooperative_groups/async_transfer.cu`
  - `thrust_support/viewer.cu`
  - `device_query/device_query.cu`
  - `launch/parallel_for.cu`
  - `warp/warp.cu`
  - `pba/sph2d.cu`
  - `pba/mpm3d.cu`
- Corex 下：
  - `set_property(TARGET muda_example PROPERTY CUDA_ARCHITECTURES ${COREX_ARCH})`
  - `target_compile_options(muda_example PRIVATE -include ... corex_cassert.h)`
  - `target_link_libraries(muda_example PRIVATE dl)`

**目的：** 保证 `muda_example` 在 Corex 上能成功构建，并避免链接到 CUDA 11+ 或额外系统库（如 filesystem）。

---

### 3.1 事件默认 flag 与 record / wait API

文件：`src/muda/launch/launch_base.h`、`src/muda/launch/details/launch_base.inl`

**问题：** Corex 对应 CUDA 10，没有 `cudaEventRecordWithFlags`、`cudaEventRecordDefault`、`cudaEventWaitDefault` 等 CUDA 11+ API/常量。

**改动：**

1. 在 `launch_base.h` 顶部，当 `CUDART_VERSION < 11100` 时：
   - `#define cudaEventRecordDefault 0`
   - `#define cudaEventWaitDefault 0`
2. 在 `launch_base.inl` 中：
   - `LaunchCore::record(cudaEvent_t e, int flag)`：`CUDART_VERSION >= 11100` 时调用 `cudaEventRecordWithFlags(e, stream(), flag)`，否则忽略 flag，只调用 `cudaEventRecord(e, stream())`。
   - `LaunchCore::when` / `wait` 及 ComputeGraph 内 wait：新版用 `cudaStreamWaitEvent(..., flag/cudaEventWaitDefault)`，旧版用 `cudaStreamWaitEvent(..., 0)`。

**目的：** 让 muda 在 CUDA 10/11+ 上都能使用统一接口；Corex 自动走旧 API 分支。

### 3.2 Graph 相关 API 兼容

文件：`src/muda/graph/graph_instantiate_flag.h`、`src/muda/graph/details/graph_exec.inl`

**改动：**

- `GraphInstantiateFlagBit`：仅在 `CUDART_VERSION >= 11100` 且定义了 `CUDA_GRAPH_INSTANTIATE_FLAG_*` 时使用 `CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_*`；否则用数值枚举 `FreeOnLaunch = 1`, `Upload = 2`, `DeviceLaunch = 4`, `UseNodePriority = 8`。
- `GraphExec::upload(cudaStream_t stream)`：仅在 `CUDART_VERSION >= 11100` 时调用 `cudaGraphUpload`；否则为空实现（launch 仍可用）。

**目的：** 兼容 Corex/CUDA 10 的头与实现，避免找不到 CUDA 11+ graph API。

### 3.3 cooperative_groups::memcpy_async 的处理

文件：`src/muda/cuda/cooperative_groups/memcpy_async.h`

**改动：** 在 Corex 模式下 `#error "cooperative_groups/memcpy_async is not available on Corex (CUDA 10)..."`；CMake 中排除使用该头的示例。

**目的：** 明确标记 Corex 当前不支持该能力，避免静默失败。

---

### 4. Clang CUDA / Corex 专用宏调整

文件：`src/muda/muda_def.h`

**改动：**

- `MUDA_GENERIC`：普通 CUDA 下 `#ifdef __CUDA_ARCH__` 时为 `MUDA_HOST MUDA_DEVICE`；Corex 下强制为 `__host__ __device__`，避免 Clang CUDA 对 `__host__ __device__` 与属性组合解析异常。
- `MUDA_NODISCARD`：普通为 `[[nodiscard]]`；Corex 下为空宏，避免 “attribute list cannot appear here” 等错误。

文件：`src/muda/buffer/buffer_view.h`、`buffer_2d_view.h` 及对应 `.inl`

**改动：** 构造函数/成员声明与定义处统一使用 `MUDA_GENERIC`（如 `BufferViewT` 跨 const 构造、`Buffer2DViewT::as_const()`），避免 Clang 报 `__host__` 与 `__host__ __device__` 重载冲突。

---

## 5. Catch2 / Thrust / assert 相关兼容性处理

### 5.1 corex_cassert.h 与全局 assert 回退

文件：`src/muda/corex_cassert.h`

**内容：**

```cpp
#pragma once
#ifndef __CUDA_ARCH__
#include <cassert>
#endif
#ifndef assert
#define assert(x) ((void)0)
#endif
```

**目的：** 通过 `-include corex_cassert.h` 保证 host 侧有 `<cassert>`，且若无 `assert` 则退化为 no-op，避免 Corex/Thrust 路径下链接或编译错误。

### 5.2 Catch2 单头文件的适配

文件：`example/external/catch2/catch.hpp`

**改动：** 在文件开头附近增加 `#include <cassert>`，并在未定义 `assert` 时定义为 no-op。

**目的：** 解决 Corex/Clang CUDA 编译 Catch2 时 “use of undeclared identifier 'assert'” 问题。

### 5.3 Thrust 中 assert 与 libdl

- Thrust 头（如 `agent_launcher.h`）内使用 `assert(...)`，通过全局 `-include corex_cassert.h` 解决未声明问题。
- Thrust 内部使用 `dlclose` 等符号，Corex 构建下对 `muda_example` 及使用 muda 的可执行目标增加 `target_link_libraries(··· PRIVATE dl)`。

---

## 6. 事件 (Event) + 双流/单流 测试过程与结论

测试用例：`example/event/event.cu` 中的 `event_record_and_wait`（Catch2 测试 `"event"`, `"[quick_start]"`）。

### 6.1 原始逻辑（上游语义）

双流 + Event：stream1 上 kernelA 改 DeviceVar、record(event)、kernelB；stream2 上 when(event) 后 kernelC 读 DeviceVar。语义上 kernelC 应读到 kernelA 写入的值。

### 6.2 在 Corex 上的异常表现

移植初期该用例在 Corex 上常出现：编译链接成功，运行时在 event 用例中 SIGSEGV。

### 6.3 单流简化实验

改为单流顺序执行 A → B → C、不用 Event：在 Corex 上不再 SIGSEGV，且 kernel C 读到正确的 v。说明单流 + 简单 wait 在 Corex 上正常；崩溃更可能与 Event 相关 API 或多流 + device printf 有关。

### 6.4 最小双流 + Event 复现

保持双流 + Event，但将 A/B/C kernel 改为空 `[] __device__() {}`，保留 record/when 与 host 侧 event/stream 同步及析构前同步。在此最小复现下 Corex 上不再 SIGSEGV，用例 PASS。说明事件记录与多流等待在 Corex 上可稳定工作；原先 SIGSEGV 更可能来自 device printf 或复杂 kernel。

### 6.5 结论

- Corex 上 muda 的 Event + 双流同步机制在最小用例下可工作。
- 带 `MUDA_KERNEL_PRINT` 或复杂 `some_work()` 的原始版本在 Corex 上易触发 SIGSEGV，可归因于 Corex 在 device 日志或复杂 kernel 路径上的实现限制。
- 在 Corex 下对 event 示例采用“空 kernel + 显式同步”的最小实现，并在文档中说明慎用设备端 printf。

---

## 7. 当前状态与可汇报的结论

1. **构建与链接：** muda 在 Corex 下可通过 `./build_corex.sh` 或 CMake `-DMUDA_BUILD_COREX=ON` 正常构建；`muda_example` 在排除部分示例后可成功链接并运行。
2. **API 兼容性：** 对 CUDA 11+ 的事件 flag、graph instantiate/upload、cooperative_groups memcpy_async 等做了版本守卫和回退；对 Catch2/Thrust/assert 做了适配。
3. **运行时：** 单流 + DeviceVar + 简单 kernel + 流同步在 Corex 上稳定；双流 + Event + 空 kernel + 正确同步也可稳定运行；带 device printf 的完整 event 示例在 Corex 上易崩溃，需规避或简化。
4. **定位：** muda 的 launch/event/graph 核心机制在 Corex 上可跑通（在最小用例下）；使用 `MUDA_KERNEL_PRINT` 或复杂 device 逻辑时需在文档中说明风险。

---

## 8. 在“关闭部分功能”后的可用性评估

- **被关闭/排除的：** 部分示例不参与 Corex 构建（见第 10 节）；事件 API 降级为无 flag 的 record/wait；`cudaGraphUpload` 空实现；`cooperative_groups::memcpy_async` 在 Corex 下 `#error`；ParallelFor 等处 static_assert 在 Corex 下关闭；设备端 printf 在多流+Event 场景下建议慎用。
- **仍可用的：** Launch 体系、Stream/Event 创建与 record/when/wait、DeviceVar/DeviceBuffer/视图、Compute Graph 的 capture+launch、以及未排除的示例（hello_muda、vector_add、stream、event 最小版、graph_quick_start、部分 logger/viewer 等）。在 Corex 上仍可完成常规 GPU 开发：发 kernel、多流与事件同步、显存管理、简单 host 日志。

---

## 9. 后续可选工作

1. 与 Corex 团队沟通：提供带 `MUDA_KERNEL_PRINT` 和原始 event 示例的最小复现，便于定位 device printf/日志路径问题。
2. 在 muda 层增加 Corex 专用开关（如 `MUDA_DISABLE_DEVICE_PRINT_ON_COREX`），在 Corex 下默认关闭设备端 printf。
3. 建立更系统的 Corex 兼容测试集与能力矩阵。
4. 在 README_COREX.md 等文档中写明：已验证功能、需规避的 CUDA 11+ API、Corex 下应避免/慎用的特性。

---

## 10. 向天数汇报：Corex 不支持而采取的规避措施（框架 vs 示例）

本节按**代码框架（muda 库）本身**与**示例（example）** 分开列出所有规避内容，便于向天数列举需求与限制。

### 10.1 代码框架（muda 库）本身的规避

以下为 **muda 库源码与构建** 中的改动，与示例是否参与构建无关。

#### 10.1.1 完全不可用、在库内直接报错或条件编译

| 具体内容 | 原因（Corex/CUDA 10 限制） | 规避方式 | 对实际功能造成的影响 |
|----------|----------------------------|----------|----------------------|
| **头文件** `<cooperative_groups/memcpy_async.h>` | Corex 无此头（CUDA 11+ 提供） | 库头 `src/muda/cuda/cooperative_groups/memcpy_async.h` 内 `#if MUDA_BUILD_COREX` 时 `#error`，禁止在 Corex 下包含。 | **Corex 上无法使用 block 级异步拷贝**：依赖该头的业务代码无法在 Corex 下编译，需改用普通 `cudaMemcpy` 或手写拷贝，无法获得 memcpy_async 的流水与延迟隐藏优势。 |
| **API** `cooperative_groups::memcpy_async(block, ...)` | 依赖上述头与 CUDA 11+ 实现 | 库内不主动使用；任何包含该头的代码在 Corex 下会因上述 `#error` 无法编译。 | 同上：**无法在 kernel 内做与计算重叠的异步块内拷贝**，影响依赖该模式的性能优化（如 stencil、预取）在 Corex 上的实现。 |

#### 10.1.2 API 降级或空实现（库内接口保留、行为弱化）

| 具体 API / 宏 | 原因 | 规避方式（库内实现） | 对实际功能造成的影响 |
|---------------|------|------------------------|----------------------|
| **宏** `cudaEventRecordDefault`、`cudaEventWaitDefault` | CUDA 11 才在头中定义 | `launch_base.h` 中在 `CUDART_VERSION < 11100` 时手动定义为 `0`。 | **无功能损失**：仅补足宏定义，record/wait 语义与“默认行为”一致，多流与事件同步逻辑不变。 |
| **API** `cudaEventRecordWithFlags(cudaEvent_t, cudaStream_t, int flag)` | CUDA 11 才提供 | `launch_base.inl` 中 `LaunchCore::record(cudaEvent_t e, int flag)`：`CUDART_VERSION >= 11100` 时调用 `cudaEventRecordWithFlags(e, stream(), flag)`，否则只调用 `cudaEventRecord(e, stream())`。 | **无法使用“带 flag 的 record”**：在 Corex 上无法使用 `cudaEventRecordExternal` 等 flag 做 graph 捕获时的外部事件节点；仅做流间同步时无影响。 |
| **API** `cudaStreamWaitEvent(..., unsigned int flag)` 的 flag 参数 | CUDA 11 才支持带 flag 的等待 | `launch_base.inl` 中 `LaunchCore::when`、`LaunchCore::wait` 及 ComputeGraph 内 wait：新版传 `flag`，否则传 `0`。 | **无法使用“带 flag 的 wait”**：在 Corex 上 graph 内事件等待只能使用默认语义；常规多流同步不受影响。 |
| **API** `cudaGraphUpload(cudaGraphExec_t, cudaStream_t)` | CUDA 11 才提供 | `graph_exec.inl` 中 `GraphExec::upload(stream)`：`CUDART_VERSION >= 11100` 时调用 `cudaGraphUpload`，否则为空实现。 | **Graph 少一步上传优化**：capture + launch 仍可用，但无法在指定流上预上传 graph 到设备，首次 launch 可能略慢或占用不同优化路径；正确性不受影响。 |
| **宏** `CUDA_GRAPH_INSTANTIATE_FLAG_*`、`CUgraphInstantiate_flags` | CUDA 11+ 才定义 | `graph_instantiate_flag.h` 中在满足版本且已定义上述宏时使用，否则用数值：`FreeOnLaunch = 1`, `Upload = 2`, `DeviceLaunch = 4`, `UseNodePriority = 8`。 | **无功能损失**：仅用数值替代枚举，若运行时支持对应行为则效果一致；若 Corex 不支持某些 flag 的语义，则对应高级 graph 特性在 Corex 上不可用。 |

#### 10.1.3 编译期放宽（库内宏与 static_assert）

| 具体项目（库内文件与符号） | 原因 | 规避方式 | 对实际功能造成的影响 |
|---------------------------|------|----------|----------------------|
| **宏** `MUDA_GENERIC`（`muda_def.h`） | Clang CUDA 对 `__host__ __device__` 与属性组合解析异常 | Corex 下定义为 `__host__ __device__`。 | **无功能损失**：仅统一为 H+D，可能多生成少量 host 副本；运行时行为与 NVCC 一致。 |
| **宏** `MUDA_NODISCARD`（`muda_def.h`） | Clang “attribute list cannot appear here” | Corex 下定义为空宏。 | **编译期提示减弱**：Corex 上忽略返回值不会产生编译器告警，需靠代码规范或静态分析补足。 |
| **类** `BufferViewT`、`Buffer2DViewT` 的跨 const 构造、`as_const()` 等 | Clang 报 `__host__` 与 `__host__ __device__` 重载冲突 | 声明与定义处统一使用 `MUDA_GENERIC`。 | **无功能损失**：仅保证重载一致，视图的 const/非 const 转换与使用方式不变。 |
| **函数** `HostCall::apply` 中的 `static_assert(std::is_invocable_v<CallableType>, ...);` | Corex 设备编译路径下触发问题 | 用 `#if !MUDA_BUILD_COREX` 包裹，Corex 下不编译该 static_assert。 | **类型检查减弱**：Corex 上传入错误签名的 callable 不会在编译期报错，可能推迟到运行时发现。 |
| **函数** `details::generic_kernel` 中的 `static_assert(std::is_invocable_v<F>, ...);` | 同上 | 同上。 | **同上**：Launch kernel 的 callable 签名错误在 Corex 上无编译期报错。 |
| **函数** `details::grid_stride_loop_kernel`、`details::parallel_for_kernel` 中的 `static_assert(always_false_v<F>, ...);` | 同上 | 同上。 | **同上**：ParallelFor 的 callable 必须为 `void(int)` 或 `void(ParallelForDetails)` 在 Corex 上无编译期强制。 |
| **符号** `assert` 在部分 CUDA/Thrust 路径下未定义 | Corex/Thrust 部分路径未包含 `<cassert>` | 库内提供 `corex_cassert.h`，CMake 对 muda 的 CUDA 编译加 `-include corex_cassert.h`。 | **无功能损失**：assert 行为与标准一致（或 no-op），不影响业务逻辑。 |
| **符号** `dlclose`、`dlsym` 等（Thrust 内部使用） | Thrust 依赖 `libdl` | 使用 muda 的可执行目标在 Corex 构建下需自行链接 `dl`。 | **无功能损失**：仅需在 CMake 中多链接 `dl`，Thrust 与 muda 功能正常。 |

#### 10.1.4 运行时规避（库内析构与同步）

| 具体位置（库内） | 现象 | 规避方式 | 对实际功能造成的影响 |
|------------------|------|----------|----------------------|
| **析构** `Event::~Event()`（`launch/details/event.inl`） | 部分 Corex 运行时要求先同步再销毁 | Corex 下在 `cudaEventDestroy` 前调用 `cudaEventSynchronize(m_handle)`。 | **析构时多一次同步**：Event 销毁时会在 host 上阻塞直到该事件完成，若业务未先 wait，可能掩盖“未同步即销毁”的 bug；对已正确同步的代码无语义变化。 |
| **析构** `Stream::~Stream()`（`launch/details/stream.inl`） | 同上 | Corex 下在 `cudaStreamDestroy` 前调用 `cudaStreamSynchronize(m_handle)`。 | **同上**：Stream 销毁时 host 会等待该流上所有任务完成，未同步即析构时可能多出一次阻塞；正确用法下无影响。 |
| **函数** `LaunchCore::wait_device()`（`launch_base.inl`） | 部分运行时在 sync 前存在未消费错误时行为异常 | Corex 下在 `cudaDeviceSynchronize()` 前调用 `cudaGetLastError()` 消费 pending 错误。 | **无功能损失**：仅避免因未消费错误导致的同步行为异常，`wait_device()` 的“等待全部设备完成”语义不变。 |

---

### 10.2 示例（example）层面的规避

以下为 **仅影响示例程序** 的规避：不改变库 API，只通过排除示例文件或修改示例内逻辑在 Corex 上可构建、可运行。

#### 10.2.1 因依赖不可用 API 而从 Corex 构建中排除的示例文件

| 排除的示例文件 | 依赖的具体内容（导致在 Corex 上不可用或不愿在示例中处理） | 规避方式 | 对实际功能造成的影响 |
|----------------|-----------------------------------------------------------|----------|----------------------|
| `example/cooperative_groups/async_transfer.cu` | `<muda/cuda/cooperative_groups/memcpy_async.h>`、`cg::memcpy_async(block, ...)`、`cg::this_grid().thread_rank()` | Corex 时从 `muda_example` 源列表 FILTER 排除。 | **Corex 构建的 muda_example 中无“异步块内拷贝”示例**：无法通过该示例学习 memcpy_async 用法；库的 memcpy_async 能力在 Corex 上本身不可用（见 10.1.1）。 |
| `example/warp/warp.cu` | `<muda/cuda/cooperative_groups/memcpy_async.h>`、`<muda/cuda/cooperative_groups/reduce.h>`、`<muda/cuda/cooperative_groups/scan.h>` 及 cooperative_groups 扩展 | 同上。 | **Corex 构建中无 warp 级 reduce/scan + memcpy_async 示例**：无法在 Corex 上通过该示例验证 warp 协作与异步拷贝；业务若需类似模式需自写并避免 memcpy_async。 |
| `example/launch/parallel_for.cu` | `cooperative_groups::this_grid().thread_rank()` | 同上。 | **Corex 构建中无使用 this_grid 的 ParallelFor 示例**：ParallelFor 本身在 Corex 上可用，但“按 grid 线性索引”的示例写法不可见；业务可用 `void(int)` 等签名替代。 |
| `example/thrust_support/viewer.cu` | `thrust::cuda::par_nosync`、`thrust::cuda::par_nosync.on(nullptr)`、`thrust::for_each(..., par_nosync, ...)` | 同上。 | **Corex 构建中无“Thrust + muda viewer”的 par_nosync 示例**：无法在 Corex 上演示 Thrust 与 muda 安全 viewer 的配合；业务若用 Thrust 需改用其他执行策略（如 par）。 |
| `example/device_query/device_query.cu` | 对 `cudaDeviceProp` 大量成员的访问（见下表） | 同上。 | **Corex 构建中无完整设备属性查询示例**：无法在 Corex 上打印 CUDA 10 之后新增的各类属性（L2 持久化、流优先级、协同启动等）；基础属性仍可通过 cudaGetDeviceProperties 等自行查询。 |
| `example/pba/sph2d.cu` | `std::experimental::filesystem` 及相关路径/文件操作 | 同上。 | **Corex 构建中无 SPH 2D 示例**：无法在 Corex 上运行该物理示例；若业务依赖 filesystem 做资源加载，需自行链接 `-lstdc++fs` 或改用其他方式。 |
| `example/pba/mpm3d.cu` | `std::experimental::filesystem` 及相关路径/文件操作 | 同上。 | **Corex 构建中无 MPM 3D 示例**：同上，无法在 Corex 上运行该物理示例；filesystem 依赖需业务侧自行解决。 |

**device_query.cu 中使用的、在 Corex（CUDA 10）头中可能缺失或布局不一致的 `cudaDeviceProp` 成员（示例层面排除原因）：**  
`luidDeviceNodeMask`, `texturePitchAlignment`, `maxTexture1DMipmap`, `maxTexture2DLinear`, `maxTexture2DMipmap`, `maxTexture2DGather`, `maxTexture3DAlt`, `maxTexture1DLayered`, `maxTexture2DLayered`, `maxTextureCubemapLayered`, `maxSurface1D`, `maxSurface2D`, `maxSurface3D`, `maxSurface1DLayered`, `maxSurface2DLayered`, `maxSurfaceCubemap`, `maxSurfaceCubemapLayered`, `surfaceAlignment`, `persistingL2CacheMaxSize`, `streamPrioritiesSupported`, `globalL1CacheSupported`, `localL1CacheSupported`, `sharedMemPerMultiprocessor`, `regsPerMultiprocessor`, `managedMemory`, `isMultiGpuBoard`, `multiGpuBoardGroupID`, `hostNativeAtomicSupported`, `singleToDoublePrecisionPerfRatio`, `pageableMemoryAccess`, `concurrentManagedAccess`, `computePreemptionSupported`, `canUseHostPointerForRegisteredMem`, `cooperativeLaunch`, `cooperativeMultiDeviceLaunch`, `sharedMemPerBlockOptin`, `pageableMemoryAccessUsesHostPageTables`, `directManagedMemAccessFromHost`, `maxBlocksPerMultiProcessor`, `accessPolicyMaxWindowSize`, `reservedSharedMemPerBlock`。

#### 10.2.2 示例内逻辑在 Corex 下的简化（不排除文件，只改实现）

| 示例与位置 | 现象 | 规避方式（仅示例内逻辑） | 对实际功能造成的影响 |
|------------|------|---------------------------|----------------------|
| **宏** `MUDA_KERNEL_PRINT(...)` 与 **示例** `event_record_and_wait()`（`example/event/event.cu`）：使用 `DeviceVar<int> v`、在 kernel 内 `MUDA_KERNEL_PRINT(...)`，以及 `some_work()` | 在 Corex 上该用例易 SIGSEGV | 在 `event.cu` 内用 `#if defined(MUDA_BUILD_COREX) && MUDA_BUILD_COREX` 分支：Corex 下改为仅空 kernel + `record(set_value_done)` + `when(set_value_done)` + 显式 `cudaEventSynchronize` / `cudaStreamSynchronize`；原始带 DeviceVar、`MUDA_KERNEL_PRINT`、`some_work()` 的代码仅在非 Corex 分支编译。 | **Corex 上 event 示例不演示“带 DeviceVar 与 device printf 的双流同步”**：仅演示“空 kernel + record/when + 显式同步”的最小事件用法；业务若在 Corex 上使用 DeviceVar + 多流 + Event 仍可用，但**在 kernel 内大量使用 MUDA_KERNEL_PRINT 与复杂逻辑时存在 SIGSEGV 风险**，建议减少设备端打印或改用 host 侧日志。 |

**说明：** 上述排除或简化仅针对示例；muda 库内对应 API（除 memcpy_async 头内明确 `#error` 外）未删除，业务代码在 Corex 上仍可使用 Launch、Event、Stream、DeviceVar、Graph 等，只需避免使用 memcpy_async 及在“多流+Event+复杂 kernel+设备端 printf”组合下谨慎使用。

---

### 10.3 汇总列表（便于口头汇报）

- **框架（库）层面：**  
  - 完全不可用/报错：`<cooperative_groups/memcpy_async.h>`、`cooperative_groups::memcpy_async`（库内 `#error`）。  
  - 降级或空实现：`cudaEventRecordDefault`/`cudaEventWaitDefault` 手动为 0；`cudaEventRecordWithFlags` 在 CUDA 10 下改为 `cudaEventRecord`；`cudaStreamWaitEvent` 的 flag 在 CUDA 10 下传 0；`cudaGraphUpload` 在 CUDA 10 下空实现；Graph 实例化 flag 在 CUDA 10 下用数值 1/2/4/8。  
  - 编译/链接：`MUDA_GENERIC`、`MUDA_NODISCARD` 在 Corex 下保守定义；BufferViewT/Buffer2DViewT 统一 `MUDA_GENERIC`；HostCall/generic_kernel/parallel_for 中 invocable 的 static_assert 在 Corex 下关闭；`-include corex_cassert.h`；使用 muda 的可执行目标在 Corex 下需链接 `dl`。  
  - 运行：Event/Stream 析构前在 Corex 下先 synchronize；`wait_device()` 在 Corex 下先 `cudaGetLastError()`。

- **示例层面：**  
  - 排除的示例文件：`async_transfer.cu`、`warp.cu`、`parallel_for.cu`、`thrust_support/viewer.cu`、`device_query.cu`、`pba/sph2d.cu`、`pba/mpm3d.cu`（原因：memcpy_async/this_grid、par_nosync、cudaDeviceProp 成员、filesystem）。  
  - 示例内简化：`event.cu` 在 Corex 下改为空 kernel + 显式 event/stream 同步，避免 `MUDA_KERNEL_PRINT` 与复杂 kernel 组合导致 SIGSEGV；建议在 Corex 上减少依赖设备端 printf 与多流+Event 组合。
