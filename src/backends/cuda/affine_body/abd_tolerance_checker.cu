#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
#include <newton_tolerance/newton_tolerance_checker.h>
#include <affine_body/affine_body_dynamics.h>
#include <muda/atomic.h>
#include <muda/check/check_cuda_errors.h>
#include <cstdlib>

namespace uipc::backend::cuda
{

__global__ void kernel_abd_tolerance_check(int n,
                                            const Vector12* __restrict__ dqs,
                                            Float abs_tol,
                                            IndexT* __restrict__ success_flag)
{
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    if(I >= n)
        return;
    if(*success_flag == 0)
        return;

    const Vector12& dq = dqs[I];
#pragma unroll
    for(IndexT i = 3; i < 12; ++i)
    {
        if(abs(dq(i)) > abs_tol)
        {
            muda::atomic_exch(success_flag, IndexT(0));
            break;
        }
    }
}

class ABDToleranceChecker final : public NewtonToleranceChecker
{
  public:
    using NewtonToleranceChecker::NewtonToleranceChecker;

    SimSystemSlot<AffineBodyDynamics> affine_body_dynamics;
    Float                             abs_tol = 0.0;
    // DeviceBuffer: avoid DeviceVar ctor cudaMalloc during SimEngine::build_systems (Corex/Iluvatar).
    muda::DeviceBuffer<IndexT>        success;
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
        success.resize(1);
    }

    void do_init(InitInfo& info) override {}

    void do_pre_newton(PreNewtonInfo& info) override {}

    void do_check(CheckResultInfo& info) override
    {
        auto dqs = affine_body_dynamics->dqs();
        using namespace muda;

        int n = static_cast<int>(dqs.size());

        if(std::getenv("UIPC_COREX_ABD_TOLERANCE_HOST_FALLBACK")
           || std::getenv("UIPC_COREX_ABD_TOLERANCE_GPU") == nullptr)
        {
            std::vector<Vector12> h_dq(n);
            cudaMemcpy(h_dq.data(), dqs.data(), n * sizeof(Vector12), cudaMemcpyDeviceToHost);
            bool converged = true;
            for(int I = 0; I < n && converged; ++I)
            {
                for(int i = 3; i < 12; ++i)
                {
                    if(std::abs(h_dq[I](i)) > abs_tol)
                    {
                        converged = false;
                        break;
                    }
                }
            }
            h_success = converged ? 1 : 0;
            info.converged(converged);
            return;
        }

        success.fill(IndexT(1));

        if(n > 0)
        {
            constexpr int block = 256;
            int           grid  = (n + block - 1) / block;
            kernel_abd_tolerance_check<<<grid, block>>>(
                n, dqs.data(), abs_tol, success.data());
            checkCudaErrors(cudaGetLastError());
        }

        checkCudaErrors(cudaMemcpy(&h_success, success.data(), sizeof(IndexT), cudaMemcpyDeviceToHost));
        bool converged = h_success != 0;
        info.converged(converged);
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