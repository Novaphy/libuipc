# corex_api_tests

Five standalone executables that exercise **CUDA Runtime**, **CUDA Graph**, **cuBLAS**, **cuSPARSE**, and **cuSOLVER** calls matching the production static inventory (`tools/cuda_api_inventory_out/cuda_api_inventory.csv`). Intended for **Corex / Iluvatar** (or any CUDA-compatible stack) bring-up.

**A-class coverage:** Each API listed under **A** in [`api_test_matrix.md`](api_test_matrix.md) is invoked in its own **`[CASE symbol]`** block in the matching binary, with CPU/legacy-buffer checks where applicable (see matrix preamble). Shared helpers live in [`correctness_utils.hpp`](correctness_utils.hpp).

## Binaries

| Executable | Role |
|------------|------|
| `corex_cublas_inventory` | All `cublas*` symbols from CSV (`cublasNrm2Ex` gated on CUDA 11+). |
| `corex_cusparse_inventory` | CSR/COO SpMV, BSR legacy `bsrmv`, `bsr2csr`, descriptors, `cusparseCreateBsr` on CUDA 11+. |
| `corex_cusolver_inventory` | `cusolverDnX*` on CUDA 11+, legacy `cusolverDnD*` on older toolkits; `cusolverSp*csrlsvqr`. |
| `corex_runtime_extended` | Async mem, pitch/3D, events, streams, callbacks, occupancy, profiler; some APIs skipped on older CUDART (see stderr WARN). |
| `corex_cuda_graph` | Graph build/instantiate/launch/exec updates; `cudaGraphUpload` on CUDA 11.1+; stream capture. |

## Build (standalone)

From this directory:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=ivcore11
cmake --build build -j
```

On NVIDIA toolchains, set `CMAKE_CUDA_ARCHITECTURES` appropriately (e.g. `75`).

## Build (with libuipc)

When configuring the main project with `UIPC_WITH_CUDA_BACKEND=ON`, enable `UIPC_BUILD_COREX_API_TESTS` (default ON). Targets appear next to other tools.

## Run (Corex)

Suggested order (math libraries first, graph last):

```bash
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/corex/lib:${LD_LIBRARY_PATH}

./build/corex_cublas_inventory
./build/corex_cusparse_inventory
./build/corex_cusolver_inventory
./build/corex_runtime_extended
./build/corex_cuda_graph
```

Exit code `0` means no `[FAIL]` lines; `[WARN]` may still appear (e.g. `NOT_SUPPORTED`, skipped symbols on older toolkits).

## 统计输出（每程序末尾一行）

每个可执行文件结束时在 **stdout** 打印汇总行，格式为：

```text
=== <binary_name>: ok=N failures=F warnings=W ===
```

| 字段 | 含义 |
|------|------|
| `ok` | 本进程内 **`[OK]`** 行数（每次 `ok()` 记 1；表示通过的检查/断言条数，非「API 符号数」） |
| `failures` | **`[FAIL]`** 条数；任一 `failures>0` 时进程 **exit code 非 0** |
| `warnings` | **`[WARN]`** 条数（如 `NOT_SUPPORTED`、兼容告警、未满足的 `#if CUDART_VERSION` 跳过说明等） |

日志中 **`[CASE 符号名]`** 表示进入该 API 的一符号一测子块；可用 `grep '^\[CASE'` 统计子测段数量。分类与 A–E 口径见 [`api_test_matrix.md`](api_test_matrix.md)。

一次性跑完五程序并只看汇总行：

```bash
for x in corex_cublas_inventory corex_cusparse_inventory corex_cusolver_inventory corex_runtime_extended corex_cuda_graph; do
  ./build/$x 2>&1 | tail -n 1
done
```

## Per-symbol results (CSV)

**[`api_test_results.csv`](api_test_results.csv)** lists each API exercised in the five binaries with **matrix class (A–E)** and **typical outcome** on **Corex + CUDART 10.x** (PASS / WARN_NOT_SUPPORTED / WARN_COMPAT / SKIPPED_VERSION / …). Re-run tests on your toolkit and adjust rows if your logs differ.

## API classification (single document)

All symbols from the production CSV are classified by test outcome (**正常 / NOT_SUPPORTED / 兼容 WARN / 版本未测 / 套件未调用**) in **[`api_test_matrix.md`](api_test_matrix.md)**. Update that file after re-running the five binaries on your toolkit.
