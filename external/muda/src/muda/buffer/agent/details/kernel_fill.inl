#include <muda/type_traits/type_label.h>
#include <muda/launch/memory.h>
#include <muda/launch/parallel_for.h>
#include <muda/buffer/buffer_view.h>
#include <muda/buffer/buffer_2d_view.h>
#include <muda/buffer/buffer_3d_view.h>

namespace muda::details::buffer
{

#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
// CoreX: ParallelFor with device lambdas silently fails.
// Use host-side loop + cudaMemcpy for fill operations.
template <typename T>
MUDA_INLINE MUDA_HOST void corex_host_fill(T* dst, size_t count, const T& val, cudaStream_t stream)
{
    if(count == 0) return;
    std::vector<T> h(count, val);
    cudaMemcpyAsync(dst, h.data(), count * sizeof(T), cudaMemcpyHostToDevice, stream);
}
#endif

// fill 0D
template <typename T>
MUDA_INLINE MUDA_HOST void kernel_fill(cudaStream_t stream, VarView<T> dst, const T& val)
{
    // workaround for nvcc requirement
    auto kernel = [dst, val] __device__(int i) mutable { *dst.data() = val; };

    if constexpr(muda::is_trivially_copy_assignable_v<T>)
    {
        Memory(stream).upload(dst.data(), &val, sizeof(T));
    }
    else
    {
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
        corex_host_fill(dst.data(), 1, val, stream);
#else
        ParallelFor(1, 1, 0, stream).apply(1, kernel);
#endif
    }
}

// fill 1D
template <typename T>
MUDA_INLINE MUDA_HOST void kernel_fill(
    int grid_dim, int block_dim, cudaStream_t stream, BufferView<T> dst, const T& val)
{
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
    corex_host_fill(dst.data(), dst.size(), val, stream);
#else
    ParallelFor(grid_dim, block_dim, 0, stream)
        .apply(dst.size(),
               [dst, val] __device__(int i) mutable { *dst.data(i) = val; });
#endif
}

// fill 2D
template <typename T>
MUDA_INLINE MUDA_HOST void kernel_fill(
    int grid_dim, int block_dim, cudaStream_t stream, Buffer2DView<T> dst, const T& val)
{
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
    corex_host_fill(dst.data(0), dst.total_size(), val, stream);
#else
    ParallelFor(grid_dim, block_dim, 0, stream)
        .apply(dst.total_size(),
               [dst, val] __device__(int i) mutable { *dst.data(i) = val; });
#endif
};

// fill 3D
template <typename T>
MUDA_INLINE MUDA_HOST void kernel_fill(
    int grid_dim, int block_dim, cudaStream_t stream, Buffer3DView<T> dst, const T& val)
{
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
    corex_host_fill(dst.data(0), dst.total_size(), val, stream);
#else
    ParallelFor(grid_dim, block_dim, 0, stream)
        .apply(dst.total_size(),
               [dst, val] __device__(int i) mutable { *dst.data(i) = val; });
#endif
};
}  // namespace muda::details::buffer
