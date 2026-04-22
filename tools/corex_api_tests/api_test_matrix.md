# 生产静态清单 API × Corex 测试结果总表

**数据来源：** [`../cuda_api_inventory_out/cuda_api_inventory.csv`](../cuda_api_inventory_out/cuda_api_inventory.csv)（`src/backends/cuda` + `external/muda/src` 生产路径）。  
**测试来源：** [`tools/corex_api_tests/`](./) 下五个可执行文件（`corex_cublas_inventory`、`corex_cusparse_inventory`、`corex_cusolver_inventory`、`corex_runtime_extended`、`corex_cuda_graph`）。  
**统计输出：** 每个程序结束打印 `=== 名称: ok=N failures=F warnings=W ===`（`ok` 为 `[OK]` 条数）；详见 [`README.md`](README.md)「统计输出」。

**正确性策略（A 类 · 一符号一测）：** 每个 **A** 类符号在对应 `.cu` 中有独立 **`[CASE 符号名]`** 子测；浮点/缓冲区在 **SUCCESS** 路径上与 CPU 参考或回读数据比对。`NOT_SUPPORTED` 记 WARN、不做数值强断言。cuSPARSE **generic `cusparseSpMV`** 在部分栈上曾与 **legacy `cusparseScsrmv`** / CPU 不一致：程序内用 legacy 校验真值，generic 与真值不一致时记 **compat WARN**（非 FAIL）。

**典型环境说明（下列分类基于该环境）：** Corex 工具链、`CUDART_VERSION` 为 **10.x** 量级、`CMAKE_CUDA_ARCHITECTURES=ivcore11` 可完整编译五目标。换用 **CUDA 11+/12+** 后，「版本分支未测」类会收缩，需重新跑二进制并更新本文档。

---

## 分类口径（每条 API 只归入一类，优先级从高到低）

| 代号 | 含义 |
|------|------|
| **A** | **正常**：测试内已调用，返回成功或数值/流程符合预期（含 `[OK]`）。 |
| **B** | **运行时 NOT_SUPPORTED**：测试内已调用，库返回 `NOT_SUPPORTED`（或等价日志，如 ixsolver 提示不支持）。 |
| **C** | **功能/兼容异常（WARN）**：已调用，但结果与 CPU 参考不一致等，测试记 WARN、不记 FAIL（如 SpMV 数值）。 |
| **D** | **版本分支未测**：源码 `#if CUDART_VERSION >= …` 未满足，**当前二进制未执行该 API**（等价于在本环境下**未验证**；部分符号在旧工具链上**不参与链接**）。 |
| **E** | **本套件未调用**：静态清单中有，但 `corex_api_tests` 源码**未出现**该符号调用（需扩测试或依赖其他手段验证）。 |

---

## A — 正常（当前环境下已跑通或成功路径）

**cuBLAS：** `cublasCreate`，`cublasDestroy`，`cublasSetStream`，`cublasSetPointerMode`，`cublasDotEx`（成功分支），`cublasSdot`（含 fallback），`cublasSnrm2`。

**cuSPARSE：** `cusparseCreate`，`cusparseDestroy`，`cusparseSetStream`，`cusparseSetPointerMode`，`cusparseCreateCsr`，`cusparseCreateCoo`，`cusparseCreateDnVec`，`cusparseCreateMatDescr`，`cusparseCreateSpVec`，`cusparseDestroyDnVec`，`cusparseDestroyMatDescr`，`cusparseDestroySpMat`，`cusparseDestroySpVec`，`cusparseSetMatType`，`cusparseSetMatIndexBase`，`cusparseSetMatDiagType`，`cusparseSbsrmv`，`cusparseDbsrmv`，`cusparseSbsr2csr`，`cusparseDbsr2csr`，`cusparseSpMV`（**调用成功**，另见 **C** 中数值问题）。

**cuSOLVER：** `cusolverDnCreate`，`cusolverDnSetStream`，`cusolverDnDestroy`，`cusolverDnDgetrf`（legacy 路径已调用；若你侧日志另有失败再改类）。

**Runtime（非 Graph）：** `cudaSetDevice`，`cudaGetDevice`，`cudaGetDeviceCount`，`cudaGetDeviceProperties`，`cudaGetLastError`，`cudaGetErrorName`，`cudaGetErrorString`，`cudaStreamCreateWithFlags`，`cudaStreamDestroy`，`cudaStreamSynchronize`，`cudaMalloc`，`cudaFree`，`cudaMemcpyAsync`，`cudaMemsetAsync`，`cudaMallocPitch`，`cudaMemcpy2DAsync`，`cudaMemset2DAsync`，`cudaMalloc3D`，`cudaMemcpy3DAsync`，`cudaMemset3DAsync`，`cudaMemcpy`，`cudaMemset`，`cudaEventCreateWithFlags`，`cudaEventRecord`，`cudaEventDestroy`，`cudaEventQuery`，`cudaEventElapsedTime`，`cudaEventSynchronize`，`cudaStreamWaitEvent`，`cudaStreamAddCallback`，`cudaLaunchHostFunc`，`cudaOccupancyMaxPotentialBlockSize`，`cudaProfilerStart`，`cudaProfilerStop`，`cudaDeviceSynchronize`。

**Graph / Capture：** `cudaGraphCreate`，`cudaGraphDestroy`，`cudaGraphAddMemcpyNode`，`cudaGraphAddKernelNode`，`cudaGraphAddMemsetNode`，`cudaGraphAddEventRecordNode`，`cudaGraphAddEventWaitNode`，`cudaGraphAddDependencies`，`cudaGraphAddChildGraphNode`，`cudaGraphAddHostNode`（若返回失败则 WARN，不阻断），`cudaGraphInstantiate`，`cudaGraphLaunch`，`cudaGraphExecKernelNodeSetParams`，`cudaGraphExecMemcpyNodeSetParams`，`cudaGraphExecMemsetNodeSetParams`（可能 WARN），`cudaGraphExecEventRecordNodeSetEvent`，`cudaGraphExecEventWaitNodeSetEvent`，`cudaGraphExecChildGraphNodeSetParams`，`cudaGraphExecDestroy`，`cudaStreamBeginCapture`，`cudaStreamEndCapture`，`cudaMemcpyAsync`（capture 路径），`cudaMalloc` / `cudaFree` / `cudaStreamSynchronize`（graph 程序内）。

---

## B — 运行时 NOT_SUPPORTED（已调用，库/栈声明不支持）

**cuBLAS：** `cublasDdot`，`cublasDnrm2`。

**cuSOLVER：** `cusolverDnDpotrf`（及缓冲查询路径在日志中与之一致），`cusolverDnDpotrs`，`cusolverDnDgetrs`，`cusolverSpCreate`（导致稀疏 QR 句柄路径不可用）。

*说明：`cusolverDnXpotrf` / `cusolverDnXgetrf` 等在 **D** 类未执行时不会出现在此；若在 CUDA 11+ 上运行后仍 NOT_SUPPORTED，应改记入 **B**。*

---

## C — 功能/兼容异常（WARN，非 FAIL）

**cuSPARSE：** 若仅 **generic** `cusparseSpMV` 与 CPU/legacy 不一致而 **legacy `cusparseScsrmv`** 正确，记 **compat WARN**（当前测试逻辑如此；换栈后若 generic 已对齐可收紧为 FAIL）。

---

## D — 版本分支未测（`#if CUDART_VERSION` 未满足，当前二进制未验证该 API）

下列条件以**源码阈值**为准，与 **A/B** 互斥：满足版本并重新编译运行前，下列条目**不得**算作已在 Corex 10.x 环境跑通。

| 条件 | API |
|------|-----|
| `>= 11020` | `cudaMallocAsync`，`cudaFreeAsync` |
| `>= 11010` | `cudaGraphUpload` |
| `>= 11000` | `cublasNrm2Ex`，`cusparseCreateBsr`，`cudaEventRecordWithFlags`，`cudaGraphExecMemcpyNodeSetParams1D`，`cusolverDnXpotrf_bufferSize`，`cusolverDnXpotrf`，`cusolverDnXpotrs`，`cusolverDnCreateParams`，`cusolverDnSetAdvOptions`，`cusolverDnXgetrf_bufferSize`，`cusolverDnXgetrf`，`cusolverDnXgetrs`，`cusolverDnDestroyParams` |

---

## E — 本套件未单独调用（清单中有，测试源码未调用）

| API | 说明 |
|-----|------|
| `cudaGraphAddMemcpyNode1D` | 当前用 `cudaGraphAddMemcpyNode` + `cudaMemcpy3DParms` 兼容 CUDA 10.x，未调用 1D 变体。 |
| `cudaGraphInstantiateWithFlags` | 当前仅用 `cudaGraphInstantiate`（含 capture 与显式图）。 |

*若未来补测，应从 **E** 移除并归入 **A/B/C/D**。*

---

## 稀疏解算未覆盖（由 B 连带）

下列符号在 **`cusolverSpCreate` 为 NOT_SUPPORTED** 时**未执行**：`cusolverSpSetStream`，`cusolverSpDcsrlsvqr`，`cusolverSpScsrlsvqr`，`cusolverSpDestroy`。待稀疏句柄可用后应在同程序内复测并改类。

---

## 维护

- 重新生成静态清单：`python3 scripts/cuda_api_inventory.py`（仓库根目录）。  
- 更新本文档：在目标工具链上编译运行五程序，按日志把各符号调整到 **A–E**。  
- 不要求再维护单独的「分支跳过」文档；**D/E** 即未验证项。  
- 各符号与 **典型测试结果**（含 PASS/WARN/SKIPPED）的表格见 **[`api_test_results.csv`](api_test_results.csv)**（与本文档同步维护）。
