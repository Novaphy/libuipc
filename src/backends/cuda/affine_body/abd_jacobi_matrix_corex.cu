// SPDX-License-Identifier: same-as-project
//
// Iluvatar Corex sidecar implementation. The body of this translation unit
// only makes sense on the Corex compatibility build (where backend headers
// declare extra fields like m_bcoo_triplet_num / build_hessian_connection /
// set_hessian_coupling). On the upstream NVIDIA build path those members do
// not exist and nvcc would fail with "identifier ... is undefined" /
// "no member ..." errors.
//
// CMake's `list(FILTER SOURCES EXCLUDE REGEX "_corex\\.(cu|h|hpp|cpp|inl)$")`
// in src/backends/cuda/CMakeLists.txt is supposed to keep this file out of
// the NVIDIA target, but the regex filter has been observed to silently
// not fire on some GitHub-hosted runners. The guard below is a
// belt-and-suspenders fallback: even when the file is mistakenly fed to
// nvcc on the NVIDIA path, the preprocessor reduces it to an empty
// translation unit and the build keeps working.
#if defined(UIPC_COREX_CUDA10_COMPAT)

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
#endif // UIPC_COREX_CUDA10_COMPAT
