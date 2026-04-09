#include <affine_body/affine_body_kinetic.h>
#include <time_integrator/bdf1_flag.h>
#include <muda/ext/eigen/evd.h>
#include <muda/check/check_cuda_errors.h>

namespace uipc::backend::cuda
{

#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
#endif

class AffineBodyBDF1Kinetic final : public AffineBodyKinetic
{
  public:
    using AffineBodyKinetic::AffineBodyKinetic;

    virtual void do_build(BuildInfo& info) override
    {
        require<BDF1Flag>();
    }

    virtual void do_compute_energy(ComputeEnergyInfo& info) override
    {
        using namespace muda;
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
        int n = static_cast<int>(info.qs().size());
        if(n > 0)
        {
            std::vector<Vector12>            h_q(n), h_qt(n);
            std::vector<ABDJacobiDyadicMass>  h_m(n);
            std::vector<IndexT>              h_fixed(n), h_ext(n);
            std::vector<Float>               h_e(n);

            cudaMemcpy(h_q.data(), info.qs().data(), n*sizeof(Vector12), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_qt.data(), info.q_tildes().data(), n*sizeof(Vector12), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_m.data(), info.masses().data(), n*sizeof(ABDJacobiDyadicMass), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_fixed.data(), info.is_fixed().data(), n*sizeof(IndexT), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_ext.data(), info.external_kinetic().data(), n*sizeof(IndexT), cudaMemcpyDeviceToHost);

            for(int i = 0; i < n; ++i)
            {
                if(h_fixed[i] || h_ext[i])
                {
                    h_e[i] = 0.0;
                }
                else
                {
                    Vector12 dq   = h_q[i] - h_qt[i];
                    Vector12 M_dq = h_m[i] * dq;
                    h_e[i] = 0.5 * dq.dot(M_dq);
                }
            }

            cudaMemcpy((void*)info.energies().data(), h_e.data(), n*sizeof(Float), cudaMemcpyHostToDevice);
        }
#else
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.qs().size(),
                   [is_fixed   = info.is_fixed().cviewer().name("is_fixed"),
                    is_dynamic = info.is_dynamic().cviewer().name("is_dynamic"),
                    ext_kinetic = info.external_kinetic().cviewer().name("ext_kinetic"),
                    qs        = info.qs().cviewer().name("qs"),
                    q_prevs   = info.q_prevs().cviewer().name("q_tildes"),
                    q_tildes  = info.q_tildes().cviewer().name("q_tildes"),
                    gravities = info.gravities().cviewer().name("gravities"),
                    masses    = info.masses().cviewer().name("masses"),
                    Ks = info.energies().viewer().name("kinetic_energy")] __device__(int i) mutable
                   {
                       auto& K = Ks(i);
                       if(is_fixed(i) || ext_kinetic(i))
                       {
                           K = 0.0;
                       }
                       else
                       {
                           const auto& q       = qs(i);
                           const auto& q_tilde = q_tildes(i);
                           const auto& M       = masses(i);
                           Vector12    dq      = q - q_tilde;
                           K                   = 0.5 * dq.dot(M * dq);
                       }
                   });
#endif
    }

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        using namespace muda;
        const int n = static_cast<int>(info.qs().size());

#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
        if(n > 0)
        {
            std::vector<Vector12>            h_q(n), h_qt(n);
            std::vector<ABDJacobiDyadicMass>  h_m(n);
            std::vector<IndexT>              h_fixed(n);
            std::vector<Vector12>            h_grad(n);
            std::vector<Matrix12x12>         h_hess(n);

            cudaMemcpy(h_q.data(), info.qs().data(), n*sizeof(Vector12), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_qt.data(), info.q_tildes().data(), n*sizeof(Vector12), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_m.data(), info.masses().data(), n*sizeof(ABDJacobiDyadicMass), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_fixed.data(), info.is_fixed().data(), n*sizeof(IndexT), cudaMemcpyDeviceToHost);

            bool grad_only = info.gradient_only();
            for(int i = 0; i < n; ++i)
            {
                Vector12 dq = h_q[i] - h_qt[i];
                h_grad[i] = h_m[i] * dq;
                if(h_fixed[i])
                    h_grad[i] = Vector12::Zero();
                if(!grad_only)
                    h_hess[i] = h_m[i].to_mat();
            }

            cudaMemcpy((void*)info.gradients().data(), h_grad.data(), n*sizeof(Vector12), cudaMemcpyHostToDevice);
            if(!grad_only)
                cudaMemcpy((void*)info.hessians().data(), h_hess.data(), n*sizeof(Matrix12x12), cudaMemcpyHostToDevice);
        }
#else
        const auto is_fixed = info.is_fixed().cviewer();
        const auto qs       = info.qs().cviewer();
        const auto q_tildes = info.q_tildes().cviewer();
        const auto masses   = info.masses().cviewer();
        const int  grad_only = info.gradient_only() ? 1 : 0;
        auto       gradients = info.gradients().viewer();
        auto       hessians  = info.hessians().viewer();

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(n,
                   [is_fixed, qs, q_tildes, masses, gradients, hessians, grad_only] __device__(int i) mutable
                   {
                       const auto& q       = qs(i);
                       const auto& q_tilde = q_tildes(i);
                       auto&       G       = gradients(i);
                       const auto& M       = masses(i);

                       G = M * (q - q_tilde);
                       if(is_fixed(i))
                       {
                           G = Vector12::Zero();
                       }
                       if(grad_only)
                           return;
                       hessians(i) = M.to_mat();
                   });
#endif
    }
};

REGISTER_SIM_SYSTEM(AffineBodyBDF1Kinetic);
}  // namespace uipc::backend::cuda
