#include <linear_system/iterative_solver.h>
#include <linear_system/global_linear_system.h>
#include <linear_system/local_preconditioner.h>
namespace uipc::backend::cuda
{
void IterativeSolver::do_build()
{
    m_system = &require<GlobalLinearSystem>();

    BuildInfo info;
    do_build(info);

    m_system->add_solver(this);
}

void IterativeSolver::spmv(Float                         a,
                           muda::CDenseVectorView<Float> x,
                           Float                         b,
                           muda::DenseVectorView<Float>  y)
{
    m_system->m_impl.spmv(a, x, b, y);
}

void IterativeSolver::spmv(muda::CDenseVectorView<Float> x, muda::DenseVectorView<Float> y)
{
    spmv(1.0, x, 0.0, y);
}

void IterativeSolver::spmv_dot(muda::CDenseVectorView<Float> x,
                               muda::DenseVectorView<Float>  y,
                               muda::VarView<Float>          d_dot)
{
    m_system->m_impl.spmv_dot(x, y, d_dot);
}

void IterativeSolver::apply_preconditioner(muda::DenseVectorView<Float>  z,
                                           muda::CDenseVectorView<Float> r,
                                           muda::CVarView<IndexT>        converged)
{
    m_system->m_impl.apply_preconditioner(z, r, converged);
}

bool IterativeSolver::accuracy_statisfied(muda::DenseVectorView<Float> r)
{
    return m_system->m_impl.accuracy_statisfied(r);
}

muda::LinearSystemContext& IterativeSolver::ctx() const
{
    return m_system->m_impl.ctx;
}

SizeT IterativeSolver::linear_system_triplet_count() const
{
    return m_system->m_impl.bcoo_A.triplet_count();
}

SizeT IterativeSolver::linear_system_local_preconditioner_count() const
{
    return m_system->m_impl.local_preconditioners.view().size();
}

SizeT IterativeSolver::linear_system_no_preconditioner_count() const
{
    return m_system->m_impl.no_precond_diag_subsystem_indices.size();
}

void IterativeSolver::solve(GlobalLinearSystem::SolvingInfo& info)
{
    do_solve(info);
}
}  // namespace uipc::backend::cuda
