#include <vector>

namespace muda
{
namespace details::linear_system
{
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
        auto status = cublasSdot(cublas(), size, x.data(), x.inc(), y.data(), y.inc(), result);
        if(status == CUBLAS_STATUS_NOT_SUPPORTED)
            *result = details::linear_system::host_fallback_dot(*this, x, y);
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
        auto status = cublasSdot(cublas(), size, x.data(), x.inc(), y.data(), y.inc(), result.data());
        if(status == CUBLAS_STATUS_NOT_SUPPORTED)
        {
            auto host_result = details::linear_system::host_fallback_dot(*this, x, y);
            checkCudaErrors(cudaMemcpy(result.data(), &host_result, sizeof(T), cudaMemcpyHostToDevice));
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