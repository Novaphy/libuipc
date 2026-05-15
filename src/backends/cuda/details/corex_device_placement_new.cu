#include <cstddef>

#if defined(UIPC_COREX_CUDA10_COMPAT) \
    && (!defined(__CUDACC_VER_MAJOR__) || __CUDACC_VER_MAJOR__ < 11)

__device__ void* operator new(std::size_t, void* p) noexcept
{
    return p;
}

__device__ void operator delete(void*, void*) noexcept {}

#endif
