#pragma once
// muda's baseline cuda version  is 11.6
#define MUDA_BASELINE_CUDACC_VER_MAJOR 11
#define MUDA_BASELINE_CUDACC_VER_MINOR 6

#if defined(__CUDACC_VER_MAJOR__) && defined(__CUDACC_VER_MINOR__)
#define MUDA_CUDACC_VER_MAJOR __CUDACC_VER_MAJOR__
#define MUDA_CUDACC_VER_MINOR __CUDACC_VER_MINOR__
#elif defined(CUDART_VERSION)
#define MUDA_CUDACC_VER_MAJOR (CUDART_VERSION / 1000)
#define MUDA_CUDACC_VER_MINOR ((CUDART_VERSION % 1000) / 10)
#else
#define MUDA_CUDACC_VER_MAJOR 0
#define MUDA_CUDACC_VER_MINOR 0
#endif

#if(MUDA_CUDACC_VER_MAJOR >= MUDA_BASELINE_CUDACC_VER_MAJOR)                    \
    && (MUDA_CUDACC_VER_MINOR >= MUDA_BASELINE_CUDACC_VER_MINOR)

#define MUDA_BASELINE_CUDACC_VER_SATISFIED
#define MUDA_WITH_THRUST_UNIVERSAL
#define MUDA_WITH_GRAPH_MEMORY_ALLOC_FREE

#endif


// Iluvatar Corex 4.4+ ships cudaMallocAsync / cudaFreeAsync (verified at runtime
// on Iluvatar MR-V100). The toolchain still reports CUDART_VERSION=10020, so the
// vanilla version gate below would (wrongly) keep the sync-only path. When the
// build sets UIPC_COREX_ASYNC_MEMORY=1 (via CMake option UIPC_COREX_USE_ASYNC_MEMORY),
// force the async path on regardless of the reported CUDART version.
#if defined(UIPC_COREX_ASYNC_MEMORY) && UIPC_COREX_ASYNC_MEMORY

#ifndef MUDA_WITH_ASYNC_MEMORY_ALLOC_FREE
#define MUDA_WITH_ASYNC_MEMORY_ALLOC_FREE
#endif
namespace muda
{
constexpr bool DEFAULT_ASYNC_ALLOC_FREE = true;
}

#elif(MUDA_CUDACC_VER_MAJOR >= 11) && (MUDA_CUDACC_VER_MINOR >= 2)

#define MUDA_WITH_ASYNC_MEMORY_ALLOC_FREE
namespace muda
{
constexpr bool DEFAULT_ASYNC_ALLOC_FREE = true;
}
#else
namespace muda
{
constexpr bool DEFAULT_ASYNC_ALLOC_FREE = false;
}
#endif

#if(MUDA_CUDACC_VER_MAJOR >= 12) && (MUDA_CUDACC_VER_MINOR >= 0)
#define MUDA_WITH_DEVICE_STREAM_MODEL 1
#else
#define MUDA_WITH_DEVICE_STREAM_MODEL 0
#endif