#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
#include <newton_tolerance/newton_tolerance_checker.h>
#include <affine_body/affine_body_dynamics.h>
#include <muda/atomic.h>
#include <muda/check/check_cuda_errors.h>
#include <cstdlib>

namespace uipc::backend::cuda
{
namespace
{
IndexT* pinned_abd_tolerance_failure_buffer()
{
    static IndexT* buffer = [] {
        IndexT* ptr = nullptr;
        checkCudaErrors(cudaHostAlloc(&ptr, sizeof(IndexT), cudaHostAllocDefault));
        return ptr;
    }();
    return buffer;
}
}  // namespace

__global__ void kernel_abd_tolerance_check(int n,
                                           const Vector12* dqs,
                                           Float abs_tol,
                                           IndexT* failure)
{
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    int local_failure = 0;
    if(I < n)
    {
        const Vector12& dq = dqs[I];
        for(IndexT i = 3; i < 12; ++i)
        {
            const Float abs_dq = dq[i] >= Float{0} ? dq[i] : -dq[i];
            if(abs_dq > abs_tol)
            {
                local_failure = 1;
                break;
            }
        }
    }

    __shared__ int block_failure[256];
    block_failure[threadIdx.x] = local_failure;
    __syncthreads();

    for(int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
        if(threadIdx.x < stride)
            block_failure[threadIdx.x] |= block_failure[threadIdx.x + stride];
        __syncthreads();
    }

    if(threadIdx.x == 0 && block_failure[0])
        atomicExch(failure, 1);
}

class ABDToleranceChecker final : public NewtonToleranceChecker
{
  public:
    using NewtonToleranceChecker::NewtonToleranceChecker;

    SimSystemSlot<AffineBodyDynamics> affine_body_dynamics;
    Float                             abs_tol = 0.0;
    // DeviceBuffer avoids DeviceVar allocation during CoreX SimEngine construction.
    muda::DeviceBuffer<IndexT>        failure;
    IndexT h_success = 1;  // 1 means success, 0 means failure

    // Inherited via NewtonToleranceChecker
    void do_build(BuildInfo& info) override
    {
        affine_body_dynamics     = require<AffineBodyDynamics>();
        auto& config             = world().scene().config();
        auto  dt_attr            = config.find<Float>("dt");
        Float dt                 = dt_attr->view()[0];
        auto  transrate_tol_attr = config.find<Float>("newton/transrate_tol");
        Float transrate_tol      = transrate_tol_attr->view()[0];
        abs_tol                  = transrate_tol * dt;
        failure.resize(1);
    }

    void do_init(InitInfo& info) override {}

    void do_pre_newton(PreNewtonInfo& info) override {}

    void do_check(CheckResultInfo& info) override
    {
        auto dqs = affine_body_dynamics->dqs();
        int n = static_cast<int>(dqs.size());

        h_success = 1;
        if(n > 0)
        {
            checkCudaErrors(cudaMemsetAsync(failure.data(), 0, sizeof(IndexT)));
            constexpr int block = 256;
            int           grid  = (n + block - 1) / block;
            kernel_abd_tolerance_check<<<grid, block>>>(
                n, dqs.data(), abs_tol, failure.data());
            checkCudaErrors(cudaGetLastError());

            IndexT* pinned_failure = pinned_abd_tolerance_failure_buffer();
            checkCudaErrors(cudaMemcpyAsync(pinned_failure,
                                            failure.data(),
                                            sizeof(IndexT),
                                            cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaStreamSynchronize(nullptr));
            h_success = *pinned_failure == 0 ? 1 : 0;
        }

        info.converged(h_success != 0);
    }

    std::string do_report() override
    {
        return fmt::format("Tol: {}{}", (h_success ? "< " : "> "), abs_tol);
    }
};

REGISTER_SIM_SYSTEM(ABDToleranceChecker);
}  // namespace uipc::backend::cuda
#else
#include <newton_tolerance/newton_tolerance_checker.h>
#include <affine_body/affine_body_dynamics.h>

namespace uipc::backend::cuda
{
class ABDToleranceChecker final : public NewtonToleranceChecker
{
  public:
    using NewtonToleranceChecker::NewtonToleranceChecker;

    SimSystemSlot<AffineBodyDynamics> affine_body_dynamics;
    Float                             abs_tol = 0.0;
    muda::DeviceVar<IndexT>           success;
    IndexT h_success = 1;  // 1 means success, 0 means failure

    // Inherited via NewtonToleranceChecker
    void do_build(BuildInfo& info) override
    {
        affine_body_dynamics     = require<AffineBodyDynamics>();
        auto& config             = world().scene().config();
        auto  dt_attr            = config.find<Float>("dt");
        Float dt                 = dt_attr->view()[0];
        auto  transrate_tol_attr = config.find<Float>("newton/transrate_tol");
        Float transrate_tol      = transrate_tol_attr->view()[0];
        abs_tol                  = transrate_tol * dt;
    }

    void do_init(InitInfo& info) override {}

    void do_pre_newton(PreNewtonInfo& info) override {}

    void do_check(CheckResultInfo& info) override
    {
        auto dqs = affine_body_dynamics->dqs();
        using namespace muda;
        BufferLaunch().fill(success.view(), 1);  // reset success flag

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(dqs.size(),
                   [dqs     = dqs.viewer().name("dqs"),
                    success = success.viewer().name("success"),
                    abs_tol = abs_tol] __device__(int I)
                   {
                       const Vector12& dq            = dqs(I);
                       IndexT          success_value = *success;

                       // if success is already marked as failed, skip
                       if(success_value == 0)
                           return;

                       // the first 3 components are translation, ignore
                       // the rest 9 components are rotation/scaling/shear, take
                       for(IndexT i = 3; i < 12; ++i)
                       {
                           if(abs(dq[i]) > abs_tol)
                           {
                               muda::atomic_exch(success.data(), 0);
                               break;  // no need to check further
                           }
                       }
                   });

        // copy from device to host
        bool h_success = success;
        info.converged(h_success);
    }

    std::string do_report() override
    {
        return fmt::format("Tol: {}{}", (h_success ? "< " : "> "), abs_tol);
    }
};

REGISTER_SIM_SYSTEM(ABDToleranceChecker);
}  // namespace uipc::backend::cuda
#endif
