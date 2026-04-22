#include <affine_body/abd_jacobi_matrix.h>
#include <muda/ext/eigen/atomic.h>

namespace uipc::backend::cuda
{
MUDA_DEVICE ABDJacobiDyadicMass ABDJacobiDyadicMass::atomic_add(ABDJacobiDyadicMass& dst,
                                                                const ABDJacobiDyadicMass& src)
{
    ABDJacobiDyadicMass ret;
    auto                mass = muda::atomic_add(&dst.m_mass, src.m_mass);
    auto                mass_times_x_bar =
        muda::eigen::atomic_add(dst.m_mass_times_x_bar, src.m_mass_times_x_bar);
    auto mass_times_dyadic_x_bar =
        muda::eigen::atomic_add(dst.m_mass_times_dyadic_x_bar, src.m_mass_times_dyadic_x_bar);
    ret.m_mass                    = mass;
    ret.m_mass_times_x_bar        = mass_times_x_bar;
    ret.m_mass_times_dyadic_x_bar = mass_times_dyadic_x_bar;
    return ret;
}
}  // namespace uipc::backend::cuda
