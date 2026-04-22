# corex_cuda_api_probe

Small standalone binary that exercises CUDA Runtime, cuBLAS, cuSPARSE, and cuSOLVER the same way libuipc’s CUDA backend and vendored muda do: create handles, run a few tiny numerical kernels, and compare against CPU references where practical. Intended for **Corex / Iluvatar bring-up** and for cross-checking against the static API inventory.

## Build (standalone, no libuipc / vcpkg)

From this directory:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=ivcore11
cmake --build build -j
```

On NVIDIA toolchains, set `CMAKE_CUDA_ARCHITECTURES` to a suitable value (for example `70`) instead of `ivcore11`.

When libuipc is configured with `UIPC_WITH_CUDA_BACKEND` and `UIPC_BUILD_CUDA_API_PROBE` (default ON), CMake also builds this target as `corex_cuda_api_probe` under the main project.

## Run (Corex)

Point the dynamic linker at the Corex toolkit libraries (typical layout):

```bash
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/corex/lib:${LD_LIBRARY_PATH}
./build/corex_cuda_api_probe
```

## Static API inventory

Regenerate the deduplicated symbol list from the repo root:

```bash
python3 scripts/cuda_api_inventory.py
```

Outputs go to `tools/cuda_api_inventory_out/` (`cuda_api_inventory.csv` and `cuda_api_inventory.md`). That directory is gitignored; use it to see which `cuda*` / `cublas*` / `cusparse*` / `cusolver*` calls appear in `src/backends/cuda` and `external/muda`, then compare with what this probe actually executes at runtime.

## Exit codes and messages

| Exit | Meaning |
|------|--------|
| `0` | No **failures** (`g_failures == 0`). Warnings may still be present. |
| `1` | At least one failure (for example CUDA runtime error, missing device, or cuSOLVER dense handle creation failure). |

- **`CUBLAS_STATUS_NOT_SUPPORTED`** is counted as a **warning**, not a failure, so double-precision paths may warn while a float `cublasDotEx` path still validates.
- **Legacy `cusparseDcsrmv`**: if the call succeeds but results do not match the CPU CSR reference within tolerance, the probe **warns only** (Corex compatibility / bring-up); check stderr for printed host vs GPU values.
- **`cusolverSpCreate`**: `NOT_SUPPORTED` is expected on some stacks and is treated as a warning (optional in muda).

The final line prints `failures` and `warnings` counts.
