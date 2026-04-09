#include <finite_element/matrix_utils.h>
#include <muda/ext/eigen/svd.h>
#include <muda/ext/eigen/evd.h>
#include <algorithm/qr_svd.hpp>
namespace uipc::backend::cuda
{
// flatten/unflatten are inline in matrix_utils.h for per-TU device codegen.

UIPC_HOST UIPC_DEVICE void polar_decomposition(const Matrix3x3& F, Matrix3x3& R, Matrix3x3& S) noexcept
{
    // this function is already tested in the muda eigen test
    muda::eigen::pd(F, R, S);
}

UIPC_HOST UIPC_DEVICE void evd(const Matrix3x3& A, Vector3& eigen_values, Matrix3x3& eigen_vectors) noexcept
{
    // this function is already tested in the muda eigen test
    muda::eigen::evd(A, eigen_values, eigen_vectors);
}

UIPC_HOST UIPC_DEVICE void evd(const Matrix9x9& A, Vector9& eigen_values, Matrix9x9& eigen_vectors) noexcept
{
    // this function is already tested in the muda eigen test
    muda::eigen::evd(A, eigen_values, eigen_vectors);
}

UIPC_HOST UIPC_DEVICE void evd(const Matrix12x12& A, Vector12& eigen_values, Matrix12x12& eigen_vectors) noexcept
{
    muda::eigen::evd(A, eigen_values, eigen_vectors);
}

UIPC_HOST UIPC_DEVICE Matrix9x9 clamp_to_spd(const Matrix9x9& A) noexcept
{
    // clamp directly
    Matrix9x9 Q;
    Vector9 values;
    muda::eigen::evd(A, values, Q);
    for(int x = 0; x < 9; x++)
        values[x] = (values[x] > 0.0) ? values[x] : 0.0;
    Matrix9x9 B = Q * values.asDiagonal() * Q.transpose();
    return B;
}

UIPC_HOST UIPC_DEVICE Matrix12x12 clamp_to_spd(const Matrix12x12& A) noexcept
{
    // clamp directly
    Matrix12x12 Q;
    Vector12 values;
    muda::eigen::evd(A, values, Q);
    for(int x = 0; x < 12; x++)
        values[x] = (values[x] > 0.0) ? values[x] : 0.0;
    Matrix12x12 B = Q * values.asDiagonal() * Q.transpose();
    return B;
}
}  // namespace uipc::backend::cuda
