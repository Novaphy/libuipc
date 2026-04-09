#include <linear_system/local_preconditioner.h>
#include <affine_body/affine_body_dynamics.h>
#include <affine_body/abd_linear_subsystem.h>
#include <linear_system/global_linear_system.h>
#include <muda/ext/eigen/inverse.h>
#include <kernel_cout.h>
#include <vector>

namespace uipc::backend::cuda
{
class ABDDiagPreconditioner final : public LocalPreconditioner
{
  public:
    using LocalPreconditioner::LocalPreconditioner;

    ABDLinearSubsystem* abd_linear_subsystem = nullptr;

    muda::DeviceBuffer<Matrix12x12> diag_inv;

    virtual void do_build(BuildInfo& info) override
    {
        auto& global_linear_system = require<GlobalLinearSystem>();
        abd_linear_subsystem       = &require<ABDLinearSubsystem>();

        info.connect(abd_linear_subsystem);
    }

    virtual void do_init(InitInfo& info) override {}

    virtual void do_assemble(GlobalLinearSystem::LocalPreconditionerAssemblyInfo& info) override
    {
        using namespace muda;

        if(std::getenv("UIPC_COREX_TRACE_LINEAR_SYSTEM"))
            logger::info("[corex_trace][precond] do_assemble: entry");

        auto diag_hessian = abd_linear_subsystem->diag_hessian();

        if(std::getenv("UIPC_COREX_TRACE_LINEAR_SYSTEM"))
            logger::info("[corex_trace][precond] do_assemble: diag_hessian.size()={}, data()={}",
                         diag_hessian.size(), (void*)diag_hessian.data());

        diag_inv.resize(diag_hessian.size());

        if(std::getenv("UIPC_COREX_TRACE_LINEAR_SYSTEM"))
            logger::info("[corex_trace][precond] do_assemble: diag_inv resized to {}", diag_inv.size());

#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
        auto n = static_cast<int>(diag_inv.size());
        if(n > 0)
        {
            std::vector<Matrix12x12> h_hess(n), h_inv(n);
            checkCudaErrors(cudaMemcpy(h_hess.data(),
                                       diag_hessian.data(),
                                       sizeof(Matrix12x12) * n,
                                       cudaMemcpyDeviceToHost));
            for(int i = 0; i < n; ++i)
            {
                Float diag_sum = 0;
                for(int d = 0; d < 12; ++d)
                    diag_sum += std::abs(h_hess[i](d, d));

                if(diag_sum > Float(1e-20))
                {
                    Float max_diag = 0;
                    for(int d = 0; d < 12; ++d)
                        max_diag = std::max(max_diag, std::abs(h_hess[i](d, d)));
                    Float eps = max_diag * Float(1e-6);
                    if(eps < Float(1e-10)) eps = Float(1e-10);

                    Matrix12x12 H_reg = h_hess[i] + eps * Matrix12x12::Identity();
                    h_inv[i] = H_reg.inverse();

                    if(!h_inv[i].allFinite())
                    {
                        h_inv[i] = Matrix12x12::Identity();
                        logger::warn("[corex] body[{}] inverse produced NaN/Inf, fallback to identity", i);
                    }
                }
                else
                {
                    h_inv[i] = Matrix12x12::Identity();
                }

                if(std::getenv("UIPC_COREX_TRACE_LINEAR_SYSTEM"))
                {
                    logger::info("[corex_trace][precond] body[{}] diag_sum={}, diag[0..2]={},{},{}",
                                 i, diag_sum,
                                 h_hess[i](0, 0), h_hess[i](1, 1), h_hess[i](2, 2));
                }
            }
            checkCudaErrors(cudaMemcpy(diag_inv.data(),
                                       h_inv.data(),
                                       sizeof(Matrix12x12) * n,
                                       cudaMemcpyHostToDevice));
        }
#else
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(diag_inv.size(),
                   [diag_hessian = diag_hessian.viewer().name("diag_hessian"),
                    diag_inv = diag_inv.viewer().name("diag_inv")] __device__(int i) mutable
                   { diag_inv(i) = muda::eigen::inverse(diag_hessian(i)); });
#endif
    }

    virtual void do_apply(GlobalLinearSystem::ApplyPreconditionerInfo& info) override
    {
        using namespace muda;
        auto converged = info.converged();

#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
        auto n = static_cast<int>(diag_inv.size());
        if(n > 0)
        {
            auto r_view = info.r();
            auto z_view = info.z();
            auto dof    = static_cast<int>(r_view.size());

            std::vector<Matrix12x12> h_inv(n);
            std::vector<Float>       h_r(dof), h_z(dof, 0.0);

            checkCudaErrors(cudaMemcpy(h_inv.data(),
                                       diag_inv.data(),
                                       sizeof(Matrix12x12) * n,
                                       cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(h_r.data(),
                                       r_view.data(),
                                       sizeof(Float) * dof,
                                       cudaMemcpyDeviceToHost));

            for(int i = 0; i < n; ++i)
            {
                Eigen::Map<const Eigen::Vector<Float, 12>> ri(h_r.data() + i * 12);
                Eigen::Map<Eigen::Vector<Float, 12>>       zi(h_z.data() + i * 12);
                zi = h_inv[i] * ri;
            }

            checkCudaErrors(cudaMemcpy(z_view.data(),
                                       h_z.data(),
                                       sizeof(Float) * dof,
                                       cudaMemcpyHostToDevice));
        }
#else
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(diag_inv.size(),
                   [r = info.r().viewer().name("r"),
                    z = info.z().viewer().name("z"),
                    converged = converged.cviewer().name("converged"),
                    diag_inv = diag_inv.viewer().name("diag_inv")] __device__(int i) mutable
                   {
                       if(*converged != 0)
                           return;
                       z.segment<12>(i * 12).as_eigen() =
                           diag_inv(i) * r.segment<12>(i * 12).as_eigen();
                   });
#endif
    }
};

REGISTER_SIM_SYSTEM(ABDDiagPreconditioner);
}  // namespace uipc::backend::cuda
