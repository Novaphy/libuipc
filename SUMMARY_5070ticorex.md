# v17 Corex-compat on NVIDIA wb400 Diagnosis

Date: 2026-05-07

## Build

Source root:

```text
/home/china/桌面/libuipc-nvidia-regression-2026-04-20/libuipc-v17-extracted./libuipc-v17-extracted
```

Build directory:

```text
build_nvidia_corex_compat
```

Important CMake settings:

```text
UIPC_MUDA_USE_COREX=ON
UIPC_USE_FLOAT=ON
UIPC_COREX_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc
UIPC_COREX_CUDA_ARCHITECTURES=native
```

Two NVIDIA-only compatibility build fixes were needed:

1. Exclude `src/backends/cuda/details/corex_device_placement_new.cu` when
   `UIPC_MUDA_USE_COREX=ON` but the CUDA compiler is NVIDIA nvcc. nvcc already
   provides device placement new/delete through `<new>`, so the Corex clang shim
   double-defines the operators.
2. Enable CUDA separable compilation/device symbol resolution for the nvcc
   Corex-compat build. Without this, `libuipc_backend_cuda.so` fails to dlopen
   with an undefined `__cudaRegisterLinkedBinary_*` symbol.

These changes are only for "Corex branch compiled by nvcc on NVIDIA"; the real
Corex/ILUVATAR build remains on the original no-RDC path.

## Run Environment

The `wb400` run used the same major switches as the Corex 146s report:

```bash
CUDA_VISIBLE_DEVICES=0 \
UIPC_COREX_ABD_PRECOND_BLOCK_INVERSE=1 \
UIPC_COREX_ABD_PRECOND_BLOCK_INVERSE_STATS=1 \
UIPC_COREX_PCG_PINNED_SCALAR=0 \
UIPC_COREX_MATCONV_NVIDIA_SEG_REDUCE=0 \
UIPC_COREX_FILTER_VIEW_SLICE=0 \
UIPC_COREX_FILTER_AABB_ASYNC=0 \
UIPC_COREX_ABD_ASSEMBLE_ASYNC=0 \
UIPC_COREX_PCG_COST_DIAG=1 \
./Release/bin/corex_demo --backend cuda --scene wrecking_ball --frames 400 --gpu 0
```

## Initial False Start

The first attempt was built with `UIPC_USE_FLOAT=OFF`, while the real Corex
`build_corex` cache uses `UIPC_USE_FLOAT=ON`.

That double-precision/non-matching build was invalid as a Corex comparison. It
showed catastrophic PCG behavior from frame 13 onward:

```text
completed frames: 33
PCG iter sum:     28,222,321
PCG iter max:     13,800
PCG max_iter hits: 2067
line-search max:  1376
```

After rebuilding with `UIPC_USE_FLOAT=ON`, the frame-13 issue disappeared.

## Short Gates After Rebuild

| Run | Wall | PCG calls | PCG iter sum | PCG max | Newton sum | PCG max_iter | Line-search max |
|---|---:|---:|---:|---:|---:|---:|---:|
| `wb14` | 1.18s | 30 | 45 | 7 | 17 | 0 | 0 |
| `wb80` | 9.87s | 310 | 4,819 | 48 | 231 | 0 | 0 |
| `wb150` | 40.92s | 840 | 33,715 | 122 | 493 | 0 | 20 |

The `wb80` result closely matches the Corex report's block-inverse short gate:

```text
Corex wb80 block-inv: 16s, PCG sum 4762, PCG max 48
NVIDIA compat wb80:   9.87s, PCG sum 4819, PCG max 48
```

## wb400 Result

Log:

```text
output/nvidia_corex_float_wb400_20260507_112151/wb400_blockinv_float.log
```

Summary:

| Metric | Value |
|---|---:|
| Exit code | 0 |
| Last frame | 399 |
| Wall time | **102.01s** |
| PCG calls | 2476 |
| PCG iter sum | **88,085** |
| PCG iter max | **122** |
| PCG iter mean | 35.58 |
| Newton records | 398 |
| Newton sum | 1978 |
| Newton max | 22 |
| Newton mean | 4.97 |
| PCG max_iter hits | 0 |
| Line-search max | 0 |
| Failure markers | 0 |
| Block-inverse rejected bodies | 0 |

## Comparison

| Environment | Wall | PCG calls | PCG iter sum | PCG max |
|---|---:|---:|---:|---:|
| Corex 146s report | 146s | 2428 | 88,478 | 122 |
| NVIDIA Corex-compat float | **102.01s** | 2476 | 88,085 | 122 |
| NVIDIA native path reference | 45.61s | 2647 | 38,066 | 83 |

The Corex-compat run on NVIDIA matches the Corex 146s numerical workload very
closely:

```text
PCG iter sum ratio = 88085 / 88478 = 0.996
PCG max ratio      = 122 / 122     = 1.000
wall ratio         = 102.01 / 146  = 0.699
```

This means the 146s Corex result is not primarily due to a different iteration
trajectory versus the NVIDIA-compiled Corex branch. The linear-solver workload
is almost identical; NVIDIA simply executes that Corex branch about 1.43x faster
on this test.

Compared with the native NVIDIA path:

```text
Corex-compat NVIDIA wall / native NVIDIA wall = 102.01 / 45.61 = 2.24x
Corex-compat PCG sum / native NVIDIA PCG sum  = 88085 / 38066  = 2.31x
```

So the remaining difference between the Corex-compatible algorithm and the native
NVIDIA path is mostly algorithmic/iteration-count difference, not just raw GPU
throughput.

## Conclusion

After using the correct `UIPC_USE_FLOAT=ON` build, NVIDIA can run the v17 Corex
compatibility path cleanly. The result is:

```text
wb400 Corex-compat on NVIDIA: 102.01s
PCG iter sum: 88,085
PCG max: 122
```

This is a good cross-hardware comparison point for the Corex 146s result. It
suggests:

1. Corex 146s and NVIDIA Corex-compat 102s do almost the same PCG work.
2. The raw execution/runtime/hardware gap for the same Corex-compatible path is
   about `1.43x`.
3. The gap from native NVIDIA 45.6s to Corex-compatible NVIDIA 102s is mainly
   the Corex-compatible algorithm requiring about `2.31x` more PCG iterations.
