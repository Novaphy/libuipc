#pragma once
#include <type_define.h>
#include <muda/muda_def.h>
#include <muda/ext/eigen/svd.h>

// ref: https://github.com/theodorekim/HOBAKv1/blob/main/src/util/MATRIX_UTIL.h

namespace uipc::backend::cuda
{
using Matrix9x12 = Matrix<Float, 9, 12>;

// flatten a matrix3x3 to a vector9 in a consistent way
inline UIPC_HOST UIPC_DEVICE Vector9 flatten(const Matrix3x3& A) noexcept
{
    Vector9 column;
    unsigned int index = 0;
    for(unsigned int j = 0; j < A.cols(); j++)
        for(unsigned int i = 0; i < A.rows(); i++, index++)
            column[index] = A(i, j);
    return column;
}

// unflatten a vector9 to a matrix3x3 in a consistent way
inline UIPC_HOST UIPC_DEVICE Matrix3x3 unflatten(const Vector9& v) noexcept
{
    Matrix3x3    A;
    unsigned int index = 0;
    for(unsigned int j = 0; j < A.cols(); j++)
        for(unsigned int i = 0; i < A.rows(); i++, index++)
            A(i, j) = v[index];
    return A;
}

inline UIPC_HOST UIPC_DEVICE Float ddot(const Matrix3x3& A, const Matrix3x3& B)
{
    Float result = 0;
    for(int y = 0; y < 3; y++)
        for(int x = 0; x < 3; x++)
            result += A(x, y) * B(x, y);
    return result;
}

// compute the singular value decomposition of a matrix3x3
// the U,V are already tested and modified to be a rotation matrices
inline UIPC_HOST UIPC_DEVICE void svd(const Matrix3x3& F,
                                      Matrix3x3&      U,
                                      Vector3&        Sigma,
                                      Matrix3x3&      V) noexcept
{
    muda::eigen::svd(F, U, Sigma, V);
}

// compute the polar decomposition of a matrix3x3
UIPC_HOST UIPC_DEVICE void polar_decomposition(const Matrix3x3& F, Matrix3x3& R, Matrix3x3& S) noexcept;

UIPC_HOST UIPC_DEVICE void evd(const Matrix3x3& A, Vector3& eigen_values, Matrix3x3& eigen_vectors) noexcept;

UIPC_HOST UIPC_DEVICE void evd(const Matrix9x9& A, Vector9& eigen_values, Matrix9x9& eigen_vectors) noexcept;

UIPC_HOST UIPC_DEVICE void evd(const Matrix12x12& A,
                               Vector12&          eigen_values,
                               Matrix12x12&       eigen_vectors) noexcept;

// clamp the eigenvalues to be semi-positive-definite (in-place; avoids Eigen by-value / return on CoreX device when Float=float).
UIPC_HOST UIPC_DEVICE void clamp_to_spd(Matrix9x9& A) noexcept;

UIPC_HOST UIPC_DEVICE void clamp_to_spd(Matrix12x12& A) noexcept;
}  // namespace uipc::backend::cuda