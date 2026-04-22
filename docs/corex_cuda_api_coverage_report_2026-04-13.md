# CoreX CUDA API 覆盖审计报告（当前代码树复核）

## 1）当前 API 测试套件覆盖范围（源码层）

- 规范测试源码目录是 `tools/corex_api_tests`。
- `build_api_tests_corex` 仅是构建产物目录（可执行文件 + CMake/Ninja 输出），本身不提供新增测试覆盖。
- `tools/corex_api_tests/CMakeLists.txt` 中定义的测试目标包括：
  - `corex_runtime_extended`
  - `corex_cuda_graph`
  - `corex_cublas_inventory`
  - `corex_cusparse_inventory`
  - `corex_cusolver_inventory`
  - `corex_cross_buffer_subtract`

静态测试矩阵参考文件：
- `tools/corex_api_tests/api_test_matrix.md`
- `tools/corex_api_tests/api_test_results.csv`

## 2）历史基线（已有文档结论）

历史基线来源：`docs/Libuipc在国产平台天数Corex环境上的适配移植（3.27～4.3）.md`。

- A 类（受 CUDART 版本门控，10.x 下未编译/未执行）：如 `cudaMallocAsync`、`cudaFreeAsync`、`cudaGraphUpload`、`cudaEventRecordWithFlags`、`cudaGraphExecMemcpyNodeSetParams1D`、`cusolverDnX*`。
- B 类（运行时 `NOT_SUPPORTED`）：`cublasDdot`、`cublasDnrm2`、`cusolverDnDpotrf`、`cusolverDnDpotrs`、`cusolverDnDgetrs`、`cusolverSpCreate`。
- C 类（兼容性异常）：`cusparseSpMV` 可能返回错误值或全零。

## 3）当前复测结果（实际执行）

复测日期：2026-04-13（当前工作区代码树）。

在 `build_api_tests_corex` 下对以下目标逐个执行（每个 180s 超时）：
- `corex_runtime_extended`
- `corex_cuda_graph`
- `corex_cublas_inventory`
- `corex_cusparse_inventory`
- `corex_cusolver_inventory`
- `corex_cross_buffer_subtract`

观测结果：
- 6 个可执行均在打印任何 `[CASE]` 前超时退出（`exit code 124`）。
- 对应日志：
  - `build_api_tests_corex/retest_logs/corex_runtime_extended.log`
  - `build_api_tests_corex/retest_logs/corex_cuda_graph.log`
  - `build_api_tests_corex/retest_logs/corex_cublas_inventory.log`
  - `build_api_tests_corex/retest_logs/corex_cusparse_inventory.log`
  - `build_api_tests_corex/retest_logs/corex_cusolver_inventory.log`
  - `build_api_tests_corex/retest_logs/corex_cross_buffer_subtract.log`

补充运行时证据：
- 对 `corex_runtime_extended` 做 `strace`（`timeout 20s`）同样超时。
- 追踪文件：`build_api_tests_corex/retest_logs/strace_corex_runtime_extended.txt`。
- 行为表现为：`libcudart.so.10.2` / `libcuda.so.1` 加载成功后，运行时线程进入驱动等待循环，未进入用户测试用例输出阶段。

本次复测结论：
- 由于全部可执行在早期运行时初始化阶段阻塞，历史 A/B/C 结论**无法在本次按单 API 粒度复现 pass/fail**。
- 当前可稳定复现的事实是：**测试套件级启动阻塞**。

## 4）libuipc 仿真链路的实际 CUDA API 使用面

使用以下脚本生成静态盘点：
- `python3 scripts/cuda_api_inventory.py`

输出文件：
- `tools/cuda_api_inventory_out/cuda_api_inventory.csv`
- `tools/cuda_api_inventory_out/cuda_api_inventory.md`

生产路径统计（`src/backends/cuda` + `external/muda/src`）：
- API 引用总数：465
- 唯一 CUDA/库符号数：111
- 其中 `cuda_runtime`：64 个
- 其中 `cublas`：10 个
- 其中 `cusparse`：22 个
- 其中 `cusolver`：15 个
- CUDA Graph 子集：25 个

仿真关键路径热点：
- 运行时/设备初始化：
  - `src/backends/cuda/engine/sim_engine.cu`（`cudaGetDeviceCount`、`cudaGetDeviceProperties`、`cudaSetDevice`，以及可选 `cudaMalloc/cudaFree` 预热探针）。
- 场景构建与首批设备分配：
  - `src/backends/cuda/affine_body/affine_body_dynamics.cu`（`cudaMalloc`、`cudaMallocPitch`、`cudaMalloc3D`、`cudaMemcpy`、`cudaMemset`、`cudaDeviceSynchronize`）。
- 线性求解迭代路径：
  - `src/backends/cuda/linear_system/linear_pcg.cu`（`cudaMemsetAsync`、`cudaMemcpyAsync`、`cudaGetLastError`、`cudaDeviceSynchronize`）。
  - `src/backends/cuda/linear_system/spmv.cu`（显式 kernel + `cub::WarpReduce`/`cub::WarpScan`，以及 `cudaMemset`、`cudaMemsetAsync`）。
- muda 线性系统后端：
  - `external/muda/src/muda/ext/linear_system/linear_system_handles.h`（cuBLAS/cuSPARSE/cuSOLVER 句柄创建、销毁、设流）。
  - `external/muda/src/muda/ext/linear_system/details/routines/dot.inl`（`cublasSdot`、`cublasDdot`、`cublasDotEx`，`NOT_SUPPORTED` 时主机回退）。
  - `external/muda/src/muda/ext/linear_system/details/routines/norm.inl`（`cublasSnrm2`、`cublasDnrm2`、`cublasNrm2Ex`，`NOT_SUPPORTED` 时主机回退）。
  - `external/muda/src/muda/ext/linear_system/details/routines/spmv.inl`（`cusparseSpMV_bufferSize`、`cusparseSpMV`）。
  - `external/muda/src/muda/ext/linear_system/details/routines/solve/solve_dense.inl`（`cusolverDnXpotrf/Xpotrs/Xgetrf/Xgetrs` 家族）。
  - `external/muda/src/muda/ext/linear_system/details/routines/solve/solve_sparse.inl`（`cusolverSpScsrlsvqr`、`cusolverSpDcsrlsvqr`）。
- muda 内存抽象层：
  - `external/muda/src/muda/launch/details/memory.inl`（`cudaMalloc`、`cudaMallocPitch`、`cudaMalloc3D`、`cudaMemcpy*`、`cudaMemset*`、异步变体、Graph 捕获路径）。

## 5）覆盖差异（测试矩阵 vs 实际使用）

已生成差异文件：
- `tools/cuda_api_inventory_out/corex_coverage_diff.csv`

结果：
- 实际使用唯一符号：111
- 被 `corex_api_tests` 矩阵覆盖：106
- 未覆盖：5

当前未覆盖符号（生产代码有使用，但现有 API tests 未直接覆盖）：
- `cudaGetDeviceFlags`
- `cudaGraphAddMemcpyNode1D`
- `cudaGraphInstantiateWithFlags`
- `cusparseCreateBsr`
- `cusparseDestroyMatDescr`

弱覆盖说明：
- 虽然测试矩阵对多数符号已有分类（A/B/C/D），但本次运行由于启动即阻塞，当前对“运行时行为”的置信度仍受限于该阻塞问题。

## 6）面向 CoreX 的优先 API 实现/加固建议

按“当前不稳定风险 + 实现现实性 + 仿真关键性”优先排序：

1. `cudaMalloc` / `cudaMallocPitch` / `cudaMalloc3D`
   - 位置：`src/backends/cuda/affine_body/affine_body_dynamics.cu` 首批分配；`external/muda/src/muda/launch/details/memory.inl` 通用分配路径。
   - 关键性：首次设备分配阻塞会直接卡死场景初始化与全部 API tests。
   - 建议重点：冷启动确定性、小块分配稳定性、异常时返回明确错误而非无限阻塞。

2. `cudaMemcpy` / `cudaMemcpyAsync` 与 `cudaMemset` / `cudaMemsetAsync`（含 2D/3D 变体）
   - 位置：初始化与求解数据搬运全链路（`memory.inl`、`linear_pcg.cu`、`spmv.cu`、几何上传路径）。
   - 关键性：这是每帧和初始化阶段的核心数据通路。
   - 建议重点：CoreX CUDA10 兼容模式下的正确性一致性与同步语义稳定性。

3. `cublasDdot` / `cublasDnrm2`（双精度）及 `DotEx/Nrm2Ex` 回退链路
   - 位置：`external/muda/.../routines/dot.inl`、`norm.inl`。
   - 关键性：PCG 收敛判据和线性求解稳定性强依赖 dot/norm 结果。
   - 当前信号：历史上常见 `NOT_SUPPORTED`；虽有主机回退，但设备端支持对性能和一致性更优。

4. `cusolverDnDpotrf` / `cusolverDnDpotrs` / `cusolverDnDgetrs`
   - 位置：稠密求解路径与对应 API tests。
   - 关键性：影响稠密解算覆盖与回退策略完整性。
   - 当前信号：历史标记 `NOT_SUPPORTED`，建议至少提供能力探测与确定性降级路径。

5. `cusolverSpCreate`（及 `cusolverSpSetStream`、`cusolverSpScsrlsvqr`、`cusolverSpDcsrlsvqr`）
   - 位置：`external/muda/.../solve/solve_sparse.inl` 稀疏求解路径。
   - 关键性：句柄创建失败会使整个稀疏 QR 解算链路失效。
   - 当前信号：历史 `NOT_SUPPORTED`，存在级联禁用。

6. `cusparseSpMV` 正确性加固
   - 位置：`external/muda/.../routines/spmv.inl`，仿真中高频 SpMV 路径。
   - 关键性：若 generic SpMV 返回错误，迭代求解会静默发散或停滞。
   - 当前信号：历史存在 generic SpMV 与 legacy/reference 不一致的兼容异常。

7. `cudaGraphUpload`、`cudaEventRecordWithFlags`、`cudaGraphExecMemcpyNodeSetParams1D`（版本门控路径）
   - 位置：Graph 测试与 muda graph 封装。
   - 关键性：是从 CUDA10 子集向更高版本能力平滑过渡的关键接口。
   - 当前信号：历史主要归于 D 类（版本门控），但建议纳入实现路线图。

## 后续最小验证闭环建议

由于本轮在测试用例执行前即阻塞，建议下一轮按以下闭环推进：
- 先在同环境运行最小 CUDA runtime 探针程序，定位“首个阻塞调用”；
- 再优先重跑 `corex_runtime_extended`，随后重跑其余 4 个主 API 清单目标；
- 一旦可观察到单 API 级结果，立即回写 `tools/corex_api_tests/api_test_results.csv` 与本报告。
