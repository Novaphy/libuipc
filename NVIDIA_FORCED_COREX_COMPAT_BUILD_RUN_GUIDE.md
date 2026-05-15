# NVIDIA forced CoreX compat build/run guide

This guide records the exact NVIDIA-side build path validated on the RTX 5070 Ti machine for the current source drop.

## What this build is

This is a forced CoreX compatibility build on NVIDIA:

- `UIPC_MUDA_USE_COREX=ON`
- `UIPC_USE_FLOAT=ON`
- CUDA source is compiled with `UIPC_COREX_CUDA10_COMPAT=1`
- Scalar type is FP32 via `UIPC_FLOAT_SCALAR=1`
- Compiler is NVIDIA nvcc, not the real CoreX clang wrapper

It uses the same CoreX compat source branches as the CoreX server, but the generated device code is NVIDIA code.

## Required source fixes for this source drop

### 1. Non-git source copy

If the source tree is unpacked from a zip and has no `.git`, but `external/muda` and other external directories are already present, skip `git submodule update`.

Patch `cmake/uipc_utils.cmake` inside `uipc_init_submodule()` after the submodule existence check:

```cmake
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/.git")
    uipc_info("Submodule ${target} already present; skip git init in non-git source copy")
    return()
endif()
```

Without this, configure fails with:

```text
fatal: not a git repository
git submodule update --init failed with 128
```

### 2. CUDA 12.x placement new duplicate definition

Patch `src/backends/cuda/details/corex_device_placement_new.cu`:

```cpp
#if defined(UIPC_COREX_CUDA10_COMPAT) \
    && (!defined(__CUDACC_VER_MAJOR__) || __CUDACC_VER_MAJOR__ < 11)
```

Without this, CUDA 12.8 + GCC 13 fails with duplicate definitions of placement `operator new/delete`.

### 3. nvcc forced compat needs CUDA device link

Patch `src/backends/cuda/CMakeLists.txt` in the `if(UIPC_MUDA_USE_COREX)` block. Real CoreX clang should keep RDC off, but NVIDIA nvcc forced compat must enable it:

```cmake
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    set_target_properties(cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
else()
    set_target_properties(cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    )
endif()
```

Without this, build may succeed but runtime `dlopen` can fail with unresolved `__cudaRegisterLinkedBinary_*` symbols.

## Configure

Run from the real source directory. Quote the path if it contains spaces:

```bash
cd "/home/china/桌面/libuipc-main. (1)/libuipc-main"

cmake -S . -B build_nvidia_corex_compat_try \
  -DCMAKE_TOOLCHAIN_FILE=/home/china/vcpkg/scripts/buildsystems/vcpkg.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DUIPC_CUDA_ARCHITECTURES=native \
  -DUIPC_COREX_CUDA_ARCHITECTURES=native \
  -DUIPC_COREX_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DUIPC_BUILD_TESTS=OFF \
  -DUIPC_BUILD_BENCHMARKS=OFF \
  -DUIPC_BUILD_EXAMPLES=ON \
  -DUIPC_BUILD_GUI=OFF \
  -DUIPC_BUILD_PYBIND=OFF \
  -DUIPC_USE_FLOAT=ON \
  -DUIPC_MUDA_USE_COREX=ON
```

## Build

```bash
cmake --build build_nvidia_corex_compat_try --config Release --parallel 8 --target corex_demo
```

A correct nvcc forced compat build should show a CUDA device-link step:

```text
Linking CUDA device code CMakeFiles/cuda.dir/cmake_device_link.o
```

## Verify the build path

```bash
grep -E "UIPC_MUDA_USE_COREX|UIPC_USE_FLOAT|CMAKE_CUDA_COMPILER|UIPC_COREX_CUDA_COMPILER|UIPC_COREX_CUDA_ARCHITECTURES" \
  build_nvidia_corex_compat_try/CMakeCache.txt
```

Expected:

```text
CMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc
UIPC_COREX_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc
UIPC_COREX_CUDA_ARCHITECTURES=native
UIPC_MUDA_USE_COREX=ON
UIPC_USE_FLOAT=ON
```

Check one CUDA translation unit:

```bash
python3 - <<'PY'
import json
cc = json.load(open("build_nvidia_corex_compat_try/compile_commands.json"))
needle = "stackless_bvh_simplex_trajectory_filter.cu"
for row in cc:
    cmd = row.get("command", "")
    if needle in cmd:
        print(cmd.split()[0])
        for token in ["UIPC_COREX_CUDA10_COMPAT", "UIPC_FLOAT_SCALAR", "MUDA_FORCE_HD_GENERIC"]:
            print(token, token in cmd)
        break
PY
```

Expected compiler and tokens:

```text
/usr/local/cuda-12.8/bin/nvcc
UIPC_COREX_CUDA10_COMPAT True
UIPC_FLOAT_SCALAR True
MUDA_FORCE_HD_GENERIC True
```

## Run wb400

Run from `Release/bin` so backend dynamic libraries are found reliably:

```bash
OUT=/tmp/uipc_new_repo_compat_wb400
mkdir -p "$OUT/run"

cd "/home/china/桌面/libuipc-main. (1)/libuipc-main/build_nvidia_corex_compat_try/Release/bin"

/usr/bin/time -p ./corex_demo \
  --backend cuda \
  --scene wrecking_ball \
  --frames 400 \
  --gpu 0 \
  --output_dir "$OUT/run" \
  > "$OUT/wb400.log" \
  2> "$OUT/wb400.time"
```

## Validated result on this machine

Validated output directory:

```text
/tmp/uipc_new_repo_compat_wb400_20260515_153349
```

Results:

```text
real 35.31
user 34.72
sys 0.46
OBJ count: 400
PCG solves: 2525
PCG total iterations: 89297
PCG avg iterations/solve: 35.365149
Max PCG iteration: 122 at frame 149 newton 1
Final frame PCG: n0=40 n1=37 n2=36 n3=41 n4=36 n5=37 n6=39 n7=43
```

The run ended normally with:

```text
Cuda Backend Shutdown Success.
```
