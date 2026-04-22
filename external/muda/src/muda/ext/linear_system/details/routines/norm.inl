#include <muda/ext/linear_system/type_mapper/data_type_mapper.h>
#include <cmath>
#include <vector>
namespace muda
{
namespace details::linear_system
{
    template <typename T>
    MUDA_INLINE void norm_common_check(CDenseVectorView<T> x)
    {
        MUDA_ASSERT(x.data(), "Vector x is empty");
    }

    template <typename T>
    MUDA_INLINE T host_fallback_norm(LinearSystemContext& ctx, CDenseVectorView<T> x)
    {
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
        auto status = cublasSnrm2(cublas(), x.size() / x.inc(), x.data(), x.inc(), result.data());
        if(status == CUBLAS_STATUS_NOT_SUPPORTED)
        {
            auto host_result = details::linear_system::host_fallback_norm(*this, x);
            checkCudaErrors(cudaMemcpy(result.data(), &host_result, sizeof(T), cudaMemcpyHostToDevice));
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
        auto status = cublasSnrm2(cublas(), x.size() / x.inc(), x.data(), x.inc(), result);
        if(status == CUBLAS_STATUS_NOT_SUPPORTED)
            *result = details::linear_system::host_fallback_norm(*this, x);
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