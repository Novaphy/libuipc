#include <affine_body/affine_body_external_force_manager.h>
#include <affine_body/affine_body_dynamics.h>
#include <affine_body/affine_body_external_force_reporter.h>
#include <muda/check/check_cuda_errors.h>

namespace uipc::backend::cuda
{

#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
__global__ void kernel_ext_force_acc(int               n,
                                     const Vector12*   forces,
                                     const Matrix12x12* masses_inv,
                                     Vector12*         force_accs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;
    force_accs[i] = masses_inv[i] * forces[i];
}
#endif

REGISTER_SIM_SYSTEM(AffineBodyExternalForceManager);

void AffineBodyExternalForceManager::do_build(BuildInfo& info)
{
    m_impl.affine_body_dynamics = &require<AffineBodyDynamics>();
}

void AffineBodyExternalForceManager::register_reporter(AffineBodyExternalForceReporter* reporter)
{
    check_state(SimEngineState::BuildSystems, "register_reporter");
    m_impl.m_reporters.register_sim_system(*reporter);
}

void AffineBodyExternalForceManager::Impl::clear()
{
    auto external_forces =
        affine_body_dynamics->m_impl.body_id_to_external_force.view();

#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
    if(external_forces.size() > 0)
    {
        checkCudaErrors(cudaMemset(external_forces.data(),
                                   0,
                                   external_forces.size() * sizeof(Vector12)));
    }
#else
    using namespace muda;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(external_forces.size(),
               [forces = external_forces.viewer().name("forces")] __device__(int i) mutable
               { forces(i).setZero(); });
#endif
}

void AffineBodyExternalForceManager::Impl::step()
{
    ExternalForceInfo info{this};
    for(auto reporter : m_reporters.view())
    {
        reporter->step(info);
    }

    using namespace muda;

    auto& abd = affine_body_dynamics->m_impl;
    auto force_accs = abd.body_id_to_external_force_acc.view();
    auto forces     = affine_body_dynamics->body_external_forces();
    auto masses_inv = affine_body_dynamics->body_mass_invs();

    SizeT body_count = forces.size();

#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
    if(body_count > 0)
    {
        int block = 128;
        int grid  = (static_cast<int>(body_count) + block - 1) / block;
        kernel_ext_force_acc<<<grid, block>>>(
            static_cast<int>(body_count),
            (const Vector12*)forces.data(),
            (const Matrix12x12*)masses_inv.data(),
            (Vector12*)force_accs.data());
        checkCudaErrors(cudaGetLastError());
    }
#else
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(body_count,
               [forces     = forces.cviewer().name("forces"),
                force_accs = force_accs.viewer().name("force_accs"),
                masses_inv = masses_inv.cviewer().name("masses_inv")] __device__(int i)
               {
                   const auto& F     = forces(i);
                   const auto& M_inv = masses_inv(i);

                   force_accs(i) = M_inv * F;
               });
#endif
}

void AffineBodyExternalForceManager::do_init()
{
    // Initialize all sub-reporters
    for(auto reporter : m_impl.m_reporters.view())
    {
        reporter->init();
    }
}

void AffineBodyExternalForceManager::do_clear()
{
    m_impl.clear();
}

void AffineBodyExternalForceManager::do_step()
{
    m_impl.step();
}

muda::BufferView<Vector12> AffineBodyExternalForceManager::ExternalForceInfo::external_forces() noexcept
{
    return m_impl->affine_body_dynamics->m_impl.body_id_to_external_force.view();
}
}  // namespace uipc::backend::cuda
