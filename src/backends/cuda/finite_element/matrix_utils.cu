#include <finite_element/matrix_utils.h>
#include <muda/ext/eigen/svd.h>
#include <muda/ext/eigen/evd.h>
#include <algorithm/qr_svd.hpp>
namespace uipc::backend::cuda
{
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
// flatten/unflatten/ddot/svd are inline in matrix_utils_corex.h (no-RDC corex codegen).
#else
UIPC_GENERIC Vector9 flatten(const Matrix3x3& A) noexcept
{
    Vector9      column;
    unsigned int index = 0;
    for(unsigned int j = 0; j < A.cols(); j++)
        for(unsigned int i = 0; i < A.rows(); i++, index++)
            column[index] = A(i, j);
    return column;
}

UIPC_GENERIC Matrix3x3 unflatten(const Vector9& v) noexcept
{
    Matrix3x3    A;
    unsigned int index = 0;
    for(unsigned int j = 0; j < A.cols(); j++)
        for(unsigned int i = 0; i < A.rows(); i++, index++)
            A(i, j) = v[index];
    return A;
}

UIPC_GENERIC Float ddot(const Matrix3x3& A, const Matrix3x3& B)
{
    Float result = 0;
    for(int y = 0; y < 3; y++)
        for(int x = 0; x < 3; x++)
            result += A(x, y) * B(x, y);
    return result;
}

UIPC_GENERIC void svd(const Matrix3x3& F, Matrix3x3& U, Vector3& Sigma, Matrix3x3& V) noexcept
{
    math::qr_svd(F, Sigma, U, V);
}
#endif

namespace
{
UIPC_HOST UIPC_DEVICE void copy_mat(const Matrix3x3& src, Matrix3x3& dst) noexcept
{
    for(int j = 0; j < 3; ++j)
        for(int i = 0; i < 3; ++i)
            dst(i, j) = src(i, j);
}

UIPC_HOST UIPC_DEVICE void copy_mat(const Matrix9x9& src, Matrix9x9& dst) noexcept
{
    for(int j = 0; j < 9; ++j)
        for(int i = 0; i < 9; ++i)
            dst(i, j) = src(i, j);
}

UIPC_HOST UIPC_DEVICE void copy_mat(const Matrix12x12& src, Matrix12x12& dst) noexcept
{
    for(int j = 0; j < 12; ++j)
        for(int i = 0; i < 12; ++i)
            dst(i, j) = src(i, j);
}
}  // namespace

UIPC_HOST UIPC_DEVICE void polar_decomposition(const Matrix3x3& F, Matrix3x3& R, Matrix3x3& S) noexcept
{
    // this function is already tested in the muda eigen test
    muda::eigen::pd(F, R, S);
}

UIPC_HOST UIPC_DEVICE void evd(const Matrix3x3& A, Vector3& eigen_values, Matrix3x3& eigen_vectors) noexcept
{
    Matrix3x3 A_local;
    copy_mat(A, A_local);
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
    muda::eigen::evd_jacobi(A_local, eigen_values, eigen_vectors);
#else
    muda::eigen::evd(A_local, eigen_values, eigen_vectors);
#endif
}

UIPC_HOST UIPC_DEVICE void evd(const Matrix9x9& A, Vector9& eigen_values, Matrix9x9& eigen_vectors) noexcept
{
    Matrix9x9 A_local;
    copy_mat(A, A_local);
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
    muda::eigen::evd_jacobi(A_local, eigen_values, eigen_vectors);
#else
    muda::eigen::evd(A_local, eigen_values, eigen_vectors);
#endif
}

UIPC_HOST UIPC_DEVICE void evd(const Matrix12x12& A, Vector12& eigen_values, Matrix12x12& eigen_vectors) noexcept
{
    Matrix12x12 A_local;
    copy_mat(A, A_local);
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
    muda::eigen::evd_jacobi(A_local, eigen_values, eigen_vectors);
#else
    muda::eigen::evd(A_local, eigen_values, eigen_vectors);
#endif
}

namespace{
UIPC_HOST UIPC_DEVICE void spd_from_evd_factors_9(const Matrix9x9& Q,
                                                  const Vector9& values,
                                                  Matrix9x9&       A) noexcept
{
    for(int r = 0; r < 9; ++r)
        for(int c = 0; c < 9; ++c)
        {
            Float s = Float(0);
            for(int k = 0; k < 9; ++k)
                s += Q(r, k) * values[k] * Q(c, k);
            A(r, c) = s;
        }
}

UIPC_HOST UIPC_DEVICE void spd_from_evd_factors_12(const Matrix12x12& Q,
                                                   const Vector12&    values,
                                                   Matrix12x12&       A) noexcept
{
    for(int r = 0; r < 12; ++r)
        for(int c = 0; c < 12; ++c)
        {
            Float s = Float(0);
            for(int k = 0; k < 12; ++k)
                s += Q(r, k) * values[k] * Q(c, k);
            A(r, c) = s;
        }
}
}  // namespace

UIPC_HOST UIPC_DEVICE void clamp_to_spd(Matrix9x9& A) noexcept
{
    Matrix9x9 Q;
    Vector9 values;
    evd(A, values, Q);
    for(int x = 0; x < 9; x++)
        values[x] = (values[x] > Float(0)) ? values[x] : Float(0);
    spd_from_evd_factors_9(Q, values, A);
}

UIPC_HOST UIPC_DEVICE void clamp_to_spd(Matrix12x12& A) noexcept
{
    Matrix12x12 Q;
    Vector12 values;
    evd(A, values, Q);
    for(int x = 0; x < 12; x++)
        values[x] = (values[x] > Float(0)) ? values[x] : Float(0);
    spd_from_evd_factors_12(Q, values, A);
}
}  // namespace uipc::backend::cuda
