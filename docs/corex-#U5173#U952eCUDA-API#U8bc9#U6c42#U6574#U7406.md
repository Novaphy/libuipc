## 1) `cusparseSpMV`

## 功能

- 稀疏矩阵向量乘（SpMV），是迭代线性求解（如 PCG）核心算子。

## 具体调用

- MUDA 通用线性系统路径：
  - `external/muda/src/muda/ext/linear_system/details/routines/spmv.inl`
  - 通过 `LinearSystemContext::generic_spmv()` 调用 `cusparseSpMV_bufferSize` + `cusparseSpMV`。

## `libuipc -> muda -> API` 调用链

- 典型链路：
  - `libuipc/src/backends/cuda/linear_system/iterative_solver.cu`
  - `IterativeSolver::ctx()` -> `GlobalLinearSystem::m_impl.ctx`（`muda::LinearSystemContext`）
  - `muda::LinearSystemContext::generic_spmv(...)`
  - `cusparseSpMV_bufferSize(...)` + `cusparseSpMV(...)`

## 2) `cudaMallocAsync` / `cudaFreeAsync`

## 功能

- 基于 stream 的异步分配/释放，减少 host 同步阻塞，改善高频小块内存操作吞吐。

## 具体调用

- MUDA 内存分配封装：
  - `external/muda/src/muda/launch/details/memory.inl`
  - `Memory::alloc_1d(..., async=true)`、`Memory::free(..., async=true)`。

## `libuipc -> muda -> API` 调用链

- 典型链路：
  - `libuipc/src/backends/cuda/*` 中大量 `muda::DeviceBuffer<T>::resize()/reserve()`
  - `DeviceBuffer` -> `BufferLaunch::resize/reserve`（`external/muda/src/muda/buffer/details/buffer_launch.inl`）
  - `NDReshaper::resize/reserve` -> `reserve_1d/2d/3d`（`external/muda/src/muda/buffer/reshape_nd/reserve.h`）
  - `Memory(stream).alloc_1d(...)`（`external/muda/src/muda/launch/details/memory.inl`）

## `libuipc` 中的作用

- `libuipc` 各系统（碰撞、接触、线性系统、状态更新）都会经过 MUDA 的 Buffer/Launch 分配路径。

## 当前替代实现

- 在 `UIPC_COREX_CUDA10_COMPAT` 路径中，`alloc_1d` 退化到同步分配链：
  - `cudaMalloc` -> 失败回退 `cudaMallocPitch` -> 再回退 `cudaMalloc3D`。
- 对 copy 操作也偏保守（CoreX 兼容路径使用同步 `cudaMemcpy`）。

## 替代实现影响

- 失去 async 分配的并发优势，可能增加帧间抖动和总耗时。
- 更依赖同步点，吞吐上限受限。
- 内存行为与 CUDA 11+ 语义存在差异，不利于统一性能调优。

