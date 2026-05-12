#include <muda/ext/linear_system/type_mapper/data_type_mapper.h>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
namespace muda
{
namespace details::linear_system
{
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
    MUDA_INLINE bool corex_norm_trace_enabled()
    {
        return std::getenv("UIPC_COREX_TRACE_DOTNORM_FALLBACK") != nullptr
               || std::getenv("UIPC_COREX_TRACE_LINEAR_SYSTEM") != nullptr;
    }

    MUDA_INLINE bool corex_norm_force_host_fallback()
    {
        return std::getenv("UIPC_COREX_DOTNORM_HOST_FALLBACK") != nullptr;
    }

    MUDA_INLINE bool corex_norm_force_device_fallback()
    {
        return std::getenv("UIPC_COREX_DOTNORM_FORCE_DEVICE_FALLBACK") != nullptr;
    }

    MUDA_INLINE std::atomic<unsigned long long>& corex_norm_host_fallback_count()
    {
        static std::atomic<unsigned long long> count{0};
        return count;
    }

    MUDA_INLINE std::atomic<unsigned long long>& corex_norm_device_fallback_count()
    {
        static std::atomic<unsigned long long> count{0};
        return count;
    }

    MUDA_INLINE void corex_trace_norm_fallback(const char* path, int n, int inc)
    {
        auto host_count   = corex_norm_host_fallback_count().load(std::memory_order_relaxed);
        auto device_count = corex_norm_device_fallback_count().load(std::memory_order_relaxed);
        if(corex_norm_trace_enabled())
        {
            std::fprintf(stderr,
                         "[corex_trace][norm_fallback] path=%s n=%d inc=%d host=%llu device=%llu\n",
                         path,
                         n,
                         inc,
                         host_count,
                         device_count);
        }
    }

    MUDA_INLINE int corex_norm_reduction_blocks(int n)
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

    static MUDA_GLOBAL void corex_norm2_blocks(int n, const float* x, int inc, float* partials)
    {
        constexpr int block_size = 256;
        __shared__ float sdata[block_size];

        int   tid    = threadIdx.x;
        int   stride = blockDim.x * gridDim.x;
        int   index  = blockIdx.x * blockDim.x + threadIdx.x;
        float sum    = 0.0f;

        for(int i = index; i < n; i += stride)
        {
            float v = x[i * inc];
            sum += v * v;
        }

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

    static MUDA_GLOBAL void corex_reduce_norm2_partials(int n, const float* partials, float* out)
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
            *out = sqrtf(sdata[0]);
    }

    MUDA_INLINE bool corex_device_norm_supported(CDenseVectorView<float> x)
    {
        return !corex_norm_force_host_fallback();
    }

    MUDA_INLINE void corex_device_norm(cudaStream_t      stream,
                                       int               n,
                                       const float*      x,
                                       int               inc,
                                       BufferView<float> scratch,
                                       float*            out)
    {
        constexpr int block_size = 256;
        int           blocks     = corex_norm_reduction_blocks(n);

        corex_norm2_blocks<<<blocks, block_size, 0, stream>>>(n, x, inc, scratch.data());
        checkCudaErrors(cudaGetLastError());
        corex_reduce_norm2_partials<<<1, block_size, 0, stream>>>(blocks, scratch.data(), out);
        checkCudaErrors(cudaGetLastError());
    }
#endif

    template <typename T>
    MUDA_INLINE void norm_common_check(CDenseVectorView<T> x)
    {
        MUDA_ASSERT(x.data(), "Vector x is empty");
    }

    template <typename T>
    MUDA_INLINE T host_fallback_norm(LinearSystemContext& ctx, CDenseVectorView<T> x)
    {
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
        corex_norm_host_fallback_count().fetch_add(1, std::memory_order_relaxed);
        corex_trace_norm_fallback("host", x.size() / x.inc(), x.inc());
#endif
        ctx.sync();

        const auto size = x.size() / x.inc();
        std::vector<T> host_x(x.size());
        checkCudaErrors(cudaMemcpy(
            host_x.data(), x.data(), sizeof(T) * x.size(), cudaMemcpyDeviceToHost));

        T sum_sq = {};
        for(int i = 0; i < size; ++i)
        {
            auto v = host_x[i * x.inc()];
            sum_sq += v * v;
        }
        return std::sqrt(sum_sq);
    }
}  // namespace details::linear_system

template <typename T>
T LinearSystemContext::norm(CDenseVectorView<T> x)
{
    T result;
    norm(x, &result);
    sync();
    return result;
}

template <typename T>
void LinearSystemContext::norm(CDenseVectorView<T> x, VarView<T> result)
{
    set_pointer_mode_device();
    details::linear_system::norm_common_check(x);
    if constexpr(std::is_same_v<T, float>)
    {
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
        if(details::linear_system::corex_norm_force_device_fallback()
           && details::linear_system::corex_device_norm_supported(x))
        {
            details::linear_system::corex_norm_device_fallback_count().fetch_add(
                1, std::memory_order_relaxed);
            details::linear_system::corex_trace_norm_fallback(
                "device_forced", x.size() / x.inc(), x.inc());
            const int blocks =
                details::linear_system::corex_norm_reduction_blocks(x.size() / x.inc());
            auto scratch = temp_buffer<float>(blocks);
            details::linear_system::corex_device_norm(
                stream(), x.size() / x.inc(), x.data(), x.inc(), scratch, result.data());
            return;
        }
#endif
        auto status = cublasSnrm2(cublas(), x.size() / x.inc(), x.data(), x.inc(), result.data());
        if(status == CUBLAS_STATUS_NOT_SUPPORTED)
        {
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
            if(details::linear_system::corex_device_norm_supported(x))
            {
                details::linear_system::corex_norm_device_fallback_count().fetch_add(
                    1, std::memory_order_relaxed);
                details::linear_system::corex_trace_norm_fallback(
                    "device", x.size() / x.inc(), x.inc());
                const int blocks  = details::linear_system::corex_norm_reduction_blocks(
                    x.size() / x.inc());
                auto scratch = temp_buffer<float>(blocks);
                details::linear_system::corex_device_norm(
                    stream(), x.size() / x.inc(), x.data(), x.inc(), scratch, result.data());
            }
            else
            {
#endif
            auto host_result = details::linear_system::host_fallback_norm(*this, x);
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
        auto status = cublasDnrm2(cublas(), x.size() / x.inc(), x.data(), x.inc(), result.data());
        if(status == CUBLAS_STATUS_NOT_SUPPORTED)
        {
            auto host_result = details::linear_system::host_fallback_norm(*this, x);
            checkCudaErrors(cudaMemcpy(result.data(), &host_result, sizeof(T), cudaMemcpyHostToDevice));
        }
        else
            checkCudaErrors(status);
    }
    else
    {
        auto type = cuda_data_type<T>();
        checkCudaErrors(cublasNrm2Ex(
            cublas(), x.size() / x.inc(), x.data(), type, x.inc(), result.data(), type, type));
    }
}

template <typename T>
void LinearSystemContext::norm(CDenseVectorView<T> x, T* result)
{
    set_pointer_mode_host();
    details::linear_system::norm_common_check(x);
    if constexpr(std::is_same_v<T, float>)
    {
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
        if(details::linear_system::corex_norm_force_device_fallback()
           && details::linear_system::corex_device_norm_supported(x))
        {
            details::linear_system::corex_norm_device_fallback_count().fetch_add(
                1, std::memory_order_relaxed);
            details::linear_system::corex_trace_norm_fallback(
                "device_forced", x.size() / x.inc(), x.inc());
            const int blocks =
                details::linear_system::corex_norm_reduction_blocks(x.size() / x.inc());
            auto  scratch = temp_buffer<float>(blocks + 1);
            auto* out     = scratch.data(blocks);
            details::linear_system::corex_device_norm(
                stream(), x.size() / x.inc(), x.data(), x.inc(), scratch, out);
            checkCudaErrors(
                cudaMemcpyAsync(result, out, sizeof(float), cudaMemcpyDeviceToHost, stream()));
            checkCudaErrors(cudaStreamSynchronize(stream()));
            return;
        }
#endif
        auto status = cublasSnrm2(cublas(), x.size() / x.inc(), x.data(), x.inc(), result);
        if(status == CUBLAS_STATUS_NOT_SUPPORTED)
        {
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
            if(details::linear_system::corex_device_norm_supported(x))
            {
                details::linear_system::corex_norm_device_fallback_count().fetch_add(
                    1, std::memory_order_relaxed);
                details::linear_system::corex_trace_norm_fallback(
                    "device", x.size() / x.inc(), x.inc());
                const int blocks = details::linear_system::corex_norm_reduction_blocks(
                    x.size() / x.inc());
                auto  scratch   = temp_buffer<float>(blocks + 1);
                auto* out       = scratch.data(blocks);
                details::linear_system::corex_device_norm(
                    stream(), x.size() / x.inc(), x.data(), x.inc(), scratch, out);
                checkCudaErrors(cudaMemcpyAsync(
                    result, out, sizeof(float), cudaMemcpyDeviceToHost, stream()));
                checkCudaErrors(cudaStreamSynchronize(stream()));
            }
            else
#endif
            *result = details::linear_system::host_fallback_norm(*this, x);
        }
        else
            checkCudaErrors(status);
    }
    else if constexpr(std::is_same_v<T, double>)
    {
        auto status = cublasDnrm2(cublas(), x.size() / x.inc(), x.data(), x.inc(), result);
        if(status == CUBLAS_STATUS_NOT_SUPPORTED)
            *result = details::linear_system::host_fallback_norm(*this, x);
        else
            checkCudaErrors(status);
    }
    else
    {
        auto type = cuda_data_type<T>();
        checkCudaErrors(cublasNrm2Ex(
            cublas(), x.size() / x.inc(), x.data(), type, x.inc(), result, type, type));
    }
}
}  // namespace muda