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
            constexpr int block = 128;
            int grid            = (n + block - 1) / block;
            bdf1_predict_dof_kernel<<<grid, block>>>(n,
                                                     info.dt(),
                                                     info.is_fixed().data(),
                                                     info.is_dynamic().data(),
                                                     info.qs().data(),
                                                     info.q_vs().data(),
                                                     info.gravities().data(),
                                                     info.external_force_accs().data(),
                                                     info.q_prevs().data(),
                                                     info.q_tildes().data());
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());
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
            constexpr int block = 128;
            int grid            = (n + block - 1) / block;
            bdf1_update_state_kernel<<<grid, block>>>(
                n, Float(1) / info.dt(), info.qs().data(), info.q_prevs().data(), info.q_vs().data());
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());
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