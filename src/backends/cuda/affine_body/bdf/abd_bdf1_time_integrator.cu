#include <affine_body/abd_time_integrator.h>
#include <time_integrator/bdf1_flag.h>
#include <muda/check/check_cuda_errors.h>

namespace uipc::backend::cuda
{

#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT

__global__ void bdf1_predict_dof_kernel(int             n,
                                        Float           dt,
                                        const IndexT*   is_fixed,
                                        const IndexT*   is_dynamic,
                                        const Vector12* qs,
                                        const Vector12* q_vs,
                                        const Vector12* gravities,
                                        const Vector12* ext_force_accs,
                                        Vector12*       q_prevs,
                                        Vector12*       q_tildes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;

    q_prevs[i] = qs[i];

    Vector12 q_tilde = q_prevs[i];

    if(!is_fixed[i])
    {
        q_tilde += (gravities[i] + ext_force_accs[i]) * dt * dt;

        if(is_dynamic[i])
        {
            q_tilde += q_vs[i] * dt;
        }
    }

    q_tildes[i] = q_tilde;
}

__global__ void bdf1_update_state_kernel(int             n,
                                         Float           inv_dt,
                                         const Vector12* qs,
                                         const Vector12* q_prevs,
                                         Vector12*       q_vs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;

    q_vs[i] = (qs[i] - q_prevs[i]) * inv_dt;
}

#endif

class ABDBDF1Integrator final : public ABDTimeIntegrator
{
  public:
    using ABDTimeIntegrator::ABDTimeIntegrator;

    void do_build(BuildInfo& info) override
    {
        require<BDF1Flag>();
    }

    virtual void do_init(InitInfo& info) override {}

    virtual void do_predict_dof(PredictDofInfo& info) override
    {
        using namespace muda;
        const int n = static_cast<int>(info.qs().size());

#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
        if(n > 0)
        {
            Float dt = info.dt();
            std::vector<Vector12> h_q(n), h_qv(n), h_grav(n), h_ext(n), h_qprev(n), h_qt(n);
            std::vector<IndexT>   h_fixed(n), h_dyn(n);

            cudaMemcpy(h_q.data(), info.qs().data(), n*sizeof(Vector12), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_qv.data(), info.q_vs().data(), n*sizeof(Vector12), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_grav.data(), info.gravities().data(), n*sizeof(Vector12), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_ext.data(), info.external_force_accs().data(), n*sizeof(Vector12), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_fixed.data(), info.is_fixed().data(), n*sizeof(IndexT), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_dyn.data(), info.is_dynamic().data(), n*sizeof(IndexT), cudaMemcpyDeviceToHost);

            for(int i = 0; i < n; ++i)
            {
                h_qprev[i] = h_q[i];
                Vector12 q_tilde = h_qprev[i];
                if(!h_fixed[i])
                {
                    q_tilde += (h_grav[i] + h_ext[i]) * dt * dt;
                    if(h_dyn[i])
                        q_tilde += h_qv[i] * dt;
                }
                h_qt[i] = q_tilde;
            }

            cudaMemcpy((void*)info.q_prevs().data(), h_qprev.data(), n*sizeof(Vector12), cudaMemcpyHostToDevice);
            cudaMemcpy((void*)info.q_tildes().data(), h_qt.data(), n*sizeof(Vector12), cudaMemcpyHostToDevice);
        }
#else
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.qs().size(),
                   [is_fixed   = info.is_fixed().cviewer().name("is_fixed"),
                    is_dynamic = info.is_dynamic().cviewer().name("is_dynamic"),
                    qs         = info.qs().cviewer().name("qs"),
                    q_prevs    = info.q_prevs().viewer().name("q_prev"),
                    q_vs       = info.q_vs().cviewer().name("q_velocities"),
                    q_tildes   = info.q_tildes().viewer().name("q_tilde"),
                    affine_gravity = info.gravities().cviewer().name("affine_gravity"),
                    external_force_accs = info.external_force_accs().cviewer().name("external_force_accs"),
                    dt = info.dt()] __device__(int i) mutable
                   {
                       auto& q_prev = q_prevs(i);
                       q_prev       = qs(i);

                       auto& q_v       = q_vs(i);
                       auto& g         = affine_gravity(i);
                       auto& f_ext_acc = external_force_accs(i);

                       Vector12 q_tilde = q_prev;

                       if(!is_fixed(i))
                       {
                           q_tilde += (g + f_ext_acc) * dt * dt;

                           if(is_dynamic(i))
                           {
                               q_tilde += q_v * dt;
                           }
                       }

                       q_tildes(i) = q_tilde;
                   });
#endif
    }

    virtual void do_update_state(UpdateVelocityInfo& info) override
    {
        using namespace muda;
        const int n = static_cast<int>(info.qs().size());

#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
        if(n > 0)
        {
            Float inv_dt = 1.0 / info.dt();
            std::vector<Vector12> h_q(n), h_qprev(n), h_qv(n);

            cudaMemcpy(h_q.data(), info.qs().data(), n*sizeof(Vector12), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_qprev.data(), info.q_prevs().data(), n*sizeof(Vector12), cudaMemcpyDeviceToHost);

            for(int i = 0; i < n; ++i)
                h_qv[i] = (h_q[i] - h_qprev[i]) * inv_dt;

            cudaMemcpy((void*)info.q_vs().data(), h_qv.data(), n*sizeof(Vector12), cudaMemcpyHostToDevice);
        }
#else
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.qs().size(),
                   [qs      = info.qs().cviewer().name("qs"),
                    q_vs    = info.q_vs().viewer().name("q_vs"),
                    q_prevs = info.q_prevs().cviewer().name("q_prevs"),
                    dt      = info.dt()] __device__(int i) mutable
                   {
                       auto& q_v    = q_vs(i);
                       auto& q_prev = q_prevs(i);

                       const auto& q = qs(i);

                       q_v = (q - q_prev) * (1.0 / dt);
                   });
#endif
    }
};

REGISTER_SIM_SYSTEM(ABDBDF1Integrator);
}  // namespace uipc::backend::cuda