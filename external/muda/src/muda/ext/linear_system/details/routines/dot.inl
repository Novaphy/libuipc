#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace muda
{
namespace details::linear_system
{
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
    MUDA_INLINE bool corex_dotnorm_trace_enabled()
    {
        return std::getenv("UIPC_COREX_TRACE_DOTNORM_FALLBACK") != nullptr
               || std::getenv("UIPC_COREX_TRACE_LINEAR_SYSTEM") != nullptr;
    }

    MUDA_INLINE bool corex_dotnorm_force_host_fallback()
    {
        return std::getenv("UIPC_COREX_DOTNORM_HOST_FALLBACK") != nullptr;
    }

    MUDA_INLINE bool corex_dotnorm_force_device_fallback()
    {
        return std::getenv("UIPC_COREX_DOTNORM_FORCE_DEVICE_FALLBACK") != nullptr;
    }

    MUDA_INLINE std::atomic<unsigned long long>& corex_dot_host_fallback_count()
    {
        static std::atomic<unsigned long long> count{0};
        return count;
    }

    MUDA_INLINE std::atomic<unsigned long long>& corex_dot_device_fallback_count()
    {
        static std::atomic<unsigned long long> count{0};
        return count;
    }

    MUDA_INLINE void corex_trace_dot_fallback(const char* path, int n, int inc_x, int inc_y)
    {
        auto host_count   = corex_dot_host_fallback_count().load(std::memory_order_relaxed);
        auto device_count = corex_dot_device_fallback_count().load(std::memory_order_relaxed);
        if(corex_dotnorm_trace_enabled())
        {
            std::fprintf(stderr,
                         "[corex_trace][dot_fallback] path=%s n=%d inc=(%d,%d) host=%llu device=%llu\n",
                         path,
                         n,
                         inc_x,
                         inc_y,
                         host_count,
                         device_count);
        }
    }

    MUDA_INLINE int corex_reduction_blocks(int n)
    {
        constexpr int block_size = 256;
        constexpr int max_blocks = 1024;
        int           blocks     = (n + block_size - 1) / block_size;
        if(blocks < 1)
            blocks = 1;
        if(blocks > max_blocks)
            blocks = max_blocks;
        return blocks;
    }

    static MUDA_GLOBAL void corex_dot_blocks(int          n,
                                             const float* x,
                                             int          inc_x,
                                             const float* y,
                                             int          inc_y,
                                             float*       partials)
    {
        constexpr int block_size = 256;
        __shared__ float sdata[block_size];

        int   tid    = threadIdx.x;
        int   stride = blockDim.x * gridDim.x;
        int   index  = blockIdx.x * blockDim.x + threadIdx.x;
        float sum    = 0.0f;

        for(int i = index; i < n; i += stride)
            sum += x[i * inc_x] * y[i * inc_y];

        sdata[tid] = sum;
        __syncthreads();

        for(int offset = blockDim.x / 2; offset > 0; offset >>= 1)
        {
            if(tid < offset)
                sdata[tid] += sdata[tid + offset];
            __syncthreads();
        }

        if(tid == 0)
            partials[blockIdx.x] = sdata[0];
    }

    static MUDA_GLOBAL void corex_reduce_dot_partials(int n, const float* partials, float* out)
    {
        constexpr int block_size = 256;
        __shared__ float sdata[block_size];

        int   tid = threadIdx.x;
        float sum = 0.0f;

        for(int i = tid; i < n; i += blockDim.x)
            sum += partials[i];

        sdata[tid] = sum;
        __syncthreads();

        for(int offset = blockDim.x / 2; offset > 0; offset >>= 1)
        {
            if(tid < offset)
                sdata[tid] += sdata[tid + offset];
            __syncthreads();
        }

        if(tid == 0)
            *out = sdata[0];
    }

    MUDA_INLINE bool corex_device_dot_supported(CDenseVectorView<float> x, CDenseVectorView<float> y)
    {
        return !corex_dotnorm_force_host_fallback();
    }

    MUDA_INLINE void corex_device_dot(cudaStream_t           stream,
                                      int                    n,
                                      const float*           x,
                                      int                    inc_x,
                                      const float*           y,
                                      int                    inc_y,
                                      BufferView<float>      scratch,
                                      float*                 out)
    {
        constexpr int block_size = 256;
        int           blocks     = corex_reduction_blocks(n);

        corex_dot_blocks<<<blocks, block_size, 0, stream>>>(
            n, x, inc_x, y, inc_y, scratch.data());
        checkCudaErrors(cudaGetLastError());
        corex_reduce_dot_partials<<<1, block_size, 0, stream>>>(blocks, scratch.data(), out);
        checkCudaErrors(cudaGetLastError());
    }
#endif

    template <typename T>
    MUDA_INLINE void dot_common_check(CDenseVectorView<T> x, CDenseVectorView<T> y)
    {
        MUDA_ASSERT(x.data() && y.data(), "x.data() and y.data() should not be nullptr");
        MUDA_ASSERT(x.size() / x.inc() == y.size() / y.inc(),
                    "x (size=%lld, inc=%d) should be the same as y (size=%lld, inc=%d)",
                    x.size(),
                    x.inc(),
                    y.size(),
                    y.inc());
    }

    template <typename T>
    MUDA_INLINE T host_fallback_dot(LinearSystemContext& ctx, CDenseVectorView<T> x, CDenseVectorView<T> y)
    {
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
        corex_dot_host_fallback_count().fetch_add(1, std::memory_order_relaxed);
        corex_trace_dot_fallback("host", x.size() / x.inc(), x.inc(), y.inc());
#endif
        ctx.sync();

        const auto size = x.size() / x.inc();
        std::vector<T> host_x(x.size());
        std::vector<T> host_y(y.size());

        checkCudaErrors(cudaMemcpy(
            host_x.data(), x.data(), sizeof(T) * x.size(), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(
            host_y.data(), y.data(), sizeof(T) * y.size(), cudaMemcpyDeviceToHost));

        T result = {};
        for(int i = 0; i < size; ++i)
            result += host_x[i * x.inc()] * host_y[i * y.inc()];

        return result;
    }
}  // namespace details::linear_system


template <typename T>
void LinearSystemContext::dot(CDenseVectorView<T> x, CDenseVectorView<T> y, T* result)
{
    set_pointer_mode_host();
    details::linear_system::dot_common_check(x, y);

    auto size = x.size() / x.inc();
    if constexpr(std::is_same_v<T, float>)
    {
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
        if(details::linear_system::corex_dotnorm_force_device_fallback()
           && details::linear_system::corex_device_dot_supported(x, y))
        {
            details::linear_system::corex_dot_device_fallback_count().fetch_add(
                1, std::memory_order_relaxed);
            details::linear_system::corex_trace_dot_fallback(
                "device_forced", size, x.inc(), y.inc());
            const int blocks  = details::linear_system::corex_reduction_blocks(size);
            auto      scratch = temp_buffer<float>(blocks + 1);
            auto*     out     = scratch.data(blocks);
            details::linear_system::corex_device_dot(
                stream(), size, x.data(), x.inc(), y.data(), y.inc(), scratch, out);
            checkCudaErrors(
                cudaMemcpyAsync(result, out, sizeof(float), cudaMemcpyDeviceToHost, stream()));
            checkCudaErrors(cudaStreamSynchronize(stream()));
            return;
        }
#endif
        auto status = cublasSdot(cublas(), size, x.data(), x.inc(), y.data(), y.inc(), result);
        if(status == CUBLAS_STATUS_NOT_SUPPORTED)
        {
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
            if(details::linear_system::corex_device_dot_supported(x, y))
            {
                details::linear_system::corex_dot_device_fallback_count().fetch_add(
                    1, std::memory_order_relaxed);
                details::linear_system::corex_trace_dot_fallback(
                    "device", size, x.inc(), y.inc());
                const int blocks  = details::linear_system::corex_reduction_blocks(size);
                auto      scratch = temp_buffer<float>(blocks + 1);
                auto*     out     = scratch.data(blocks);
                details::linear_system::corex_device_dot(
                    stream(), size, x.data(), x.inc(), y.data(), y.inc(), scratch, out);
                checkCudaErrors(cudaMemcpyAsync(
                    result, out, sizeof(float), cudaMemcpyDeviceToHost, stream()));
                checkCudaErrors(cudaStreamSynchronize(stream()));
            }
            else
#endif
            *result = details::linear_system::host_fallback_dot(*this, x, y);
        }
        else
            checkCudaErrors(status);
    }
    else if constexpr(std::is_same_v<T, double>)
    {
        auto status = cublasDdot(cublas(), size, x.data(), x.inc(), y.data(), y.inc(), result);
        if(status == CUBLAS_STATUS_NOT_SUPPORTED)
            *result = details::linear_system::host_fallback_dot(*this, x, y);
        else
            checkCudaErrors(status);
    }
    else
    {
        auto type = cuda_data_type<T>();
        checkCudaErrors(cublasDotEx(
            cublas(), size, x.data(), type, x.inc(), y.data(), type, y.inc(), result, type, type));
    }
}

template <typename T>
T LinearSystemContext::dot(CDenseVectorView<T> x, CDenseVectorView<T> y)
{
    T result;
    dot(x, y, &result);
    sync();
    return result;
}

template <typename T>
void LinearSystemContext::dot(CDenseVectorView<T> x, CDenseVectorView<T> y, VarView<T> result)
{
    set_pointer_mode_device();
    details::linear_system::dot_common_check(x, y);

    auto size = x.size() / x.inc();
    if constexpr(std::is_same_v<T, float>)
    {
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
        if(details::linear_system::corex_dotnorm_force_device_fallback()
           && details::linear_system::corex_device_dot_supported(x, y))
        {
            details::linear_system::corex_dot_device_fallback_count().fetch_add(
                1, std::memory_order_relaxed);
            details::linear_system::corex_trace_dot_fallback(
                "device_forced", size, x.inc(), y.inc());
            const int blocks  = details::linear_system::corex_reduction_blocks(size);
            auto      scratch = temp_buffer<float>(blocks);
            details::linear_system::corex_device_dot(
                stream(), size, x.data(), x.inc(), y.data(), y.inc(), scratch, result.data());
            return;
        }
#endif
        auto status = cublasSdot(cublas(), size, x.data(), x.inc(), y.data(), y.inc(), result.data());
        if(status == CUBLAS_STATUS_NOT_SUPPORTED)
        {
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
            if(details::linear_system::corex_device_dot_supported(x, y))
            {
                details::linear_system::corex_dot_device_fallback_count().fetch_add(
                    1, std::memory_order_relaxed);
                details::linear_system::corex_trace_dot_fallback(
                    "device", size, x.inc(), y.inc());
                const int blocks  = details::linear_system::corex_reduction_blocks(size);
                auto      scratch = temp_buffer<float>(blocks);
                details::linear_system::corex_device_dot(
                    stream(), size, x.data(), x.inc(), y.data(), y.inc(), scratch, result.data());
            }
            else
            {
#endif
            auto host_result = details::linear_system::host_fallback_dot(*this, x, y);
            checkCudaErrors(cudaMemcpy(result.data(), &host_result, sizeof(T), cudaMemcpyHostToDevice));
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
            }
#endif
        }
        else
            checkCudaErrors(status);
    }
    else if constexpr(std::is_same_v<T, double>)
    {
        auto status = cublasDdot(cublas(), size, x.data(), x.inc(), y.data(), y.inc(), result.data());
        if(status == CUBLAS_STATUS_NOT_SUPPORTED)
        {
            auto host_result = details::linear_system::host_fallback_dot(*this, x, y);
            checkCudaErrors(cudaMemcpy(result.data(), &host_result, sizeof(T), cudaMemcpyHostToDevice));
        }
        else
            checkCudaErrors(status);
    }
    else
    {
        auto type = cuda_data_type<T>();
        checkCudaErrors(cublasDotEx(cublas(),
                                    size,
                                    x.data(),
                                    type,
                                    x.inc(),
                                    y.data(),
                                    type,
                                    y.inc(),
                                    result.data(),
                                    type,
                                    type));
    }
}

}  // namespace muda