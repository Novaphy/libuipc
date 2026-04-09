#include <finite_element/fem_utils.h>
#include <muda/ext/eigen/inverse.h>
#include <Eigen/Geometry>

namespace uipc::backend::cuda::fem
{
UIPC_GENERIC Float invariant2(const Matrix3x3& F)
{
    return ddot(F, F);
}

UIPC_GENERIC Float invariant2(const Vector3& Sigma)
{
    return Sigma[0] * Sigma[0] + Sigma[1] * Sigma[1] + Sigma[2] * Sigma[2];
}

UIPC_GENERIC Float invariant3(const Matrix3x3& F)
{
    return F.determinant();
}

UIPC_GENERIC Float invariant3(const Vector3& Sigma)
{
    return Sigma[0] * Sigma[1] * Sigma[2];
}

UIPC_GENERIC Float invariant4(const Matrix3x3& F, const Vector3& a)
{
    Matrix3x3 U, V;
    Vector3   Sigma;
    svd(F, U, Sigma, V);
    const Matrix3x3 S = V * Sigma.asDiagonal() * V.transpose();
    return (S * a).dot(a);
}

UIPC_GENERIC Float invariant5(const Matrix3x3& F, const Vector3& a)
{
    return (F * a).squaredNorm();
}

UIPC_GENERIC Matrix3x3 dJdF(const Matrix3x3& F)
{
    Matrix3x3 dJdF;
    //tex:
    //$$
    //\frac{\partial I_{3}}{\partial \mathbf{F}}=\frac{\partial J}{\partial \mathbf{F}}=\left[\begin{array}{l|l|l}
    //\mathbf{f}_{1} \times \mathbf{f}_{2} & \mathbf{f}_{2} \times \mathbf{f}_{0} & \mathbf{f}_{0} \times \mathbf{f}_{1}
    //\end{array}\right]
    //$$
    dJdF.col(0) = F.col(1).cross(F.col(2));
    dJdF.col(1) = F.col(2).cross(F.col(0));
    dJdF.col(2) = F.col(0).cross(F.col(1));
    return dJdF;
}

// Ds / Dm_inv / F / dFdx: inline in fem_utils.h
}  // namespace uipc::backend::cuda::fem
