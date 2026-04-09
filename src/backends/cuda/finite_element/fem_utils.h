#pragma once
#include <finite_element/matrix_utils.h>
#include <muda/ext/eigen/inverse.h>
namespace uipc::backend::cuda::fem
{
UIPC_GENERIC Float invariant2(const Matrix3x3& F);
UIPC_GENERIC Float invariant2(const Vector3& Sigma);
UIPC_GENERIC Float invariant3(const Matrix3x3& F);
UIPC_GENERIC Float invariant3(const Vector3& Sigma);
UIPC_GENERIC Float invariant4(const Matrix3x3& F, const Vector3& a);
UIPC_GENERIC Float invariant5(const Matrix3x3& F, const Vector3& a);

//tex: $\frac{\partial \det(\mathbf{F})}{\partial F}$
UIPC_GENERIC Matrix3x3 dJdF(const Matrix3x3& F);

// inverse material coordinates — defined inline for per-TU device codegen (no RDC on Corex clang).
UIPC_GENERIC inline Matrix3x3 Ds(const Vector3& x0, const Vector3& x1, const Vector3& x2, const Vector3& x3)
{
    Matrix3x3 Ds_mat;
    Ds_mat.col(0) = x1 - x0;
    Ds_mat.col(1) = x2 - x0;
    Ds_mat.col(2) = x3 - x0;
    return Ds_mat;
}

UIPC_GENERIC inline Matrix3x3 Dm_inv(const Vector3& X0,
                                     const Vector3& X1,
                                     const Vector3& X2,
                                     const Vector3& X3)
{
    Matrix3x3 Dm = Ds(X0, X1, X2, X3);
    return muda::eigen::inverse(Dm);
}
// F / dFdx: inline in header for per-TU device codegen (no RDC on Corex clang).
UIPC_GENERIC inline Matrix9x12 dFdx(const Matrix3x3& DmInv)
{
    const Float m = DmInv(0, 0);
    const Float n = DmInv(0, 1);
    const Float o = DmInv(0, 2);
    const Float p = DmInv(1, 0);
    const Float q = DmInv(1, 1);
    const Float r = DmInv(1, 2);
    const Float s = DmInv(2, 0);
    const Float t = DmInv(2, 1);
    const Float u = DmInv(2, 2);

    const Float t1 = -m - p - s;
    const Float t2 = -n - q - t;
    const Float t3 = -o - r - u;

    Matrix9x12 PFPu = Matrix9x12::Zero();
    PFPu(0, 0)      = t1;
    PFPu(0, 3)      = m;
    PFPu(0, 6)      = p;
    PFPu(0, 9)      = s;
    PFPu(1, 1)      = t1;
    PFPu(1, 4)      = m;
    PFPu(1, 7)      = p;
    PFPu(1, 10)     = s;
    PFPu(2, 2)      = t1;
    PFPu(2, 5)      = m;
    PFPu(2, 8)      = p;
    PFPu(2, 11)     = s;
    PFPu(3, 0)      = t2;
    PFPu(3, 3)      = n;
    PFPu(3, 6)      = q;
    PFPu(3, 9)      = t;
    PFPu(4, 1)      = t2;
    PFPu(4, 4)      = n;
    PFPu(4, 7)      = q;
    PFPu(4, 10)     = t;
    PFPu(5, 2)      = t2;
    PFPu(5, 5)      = n;
    PFPu(5, 8)      = q;
    PFPu(5, 11)     = t;
    PFPu(6, 0)      = t3;
    PFPu(6, 3)      = o;
    PFPu(6, 6)      = r;
    PFPu(6, 9)      = u;
    PFPu(7, 1)      = t3;
    PFPu(7, 4)      = o;
    PFPu(7, 7)      = r;
    PFPu(7, 10)     = u;
    PFPu(8, 2)      = t3;
    PFPu(8, 5)      = o;
    PFPu(8, 8)      = r;
    PFPu(8, 11)     = u;

    return PFPu;
}

UIPC_GENERIC inline Matrix3x3 F(const Vector3&   x0,
                                const Vector3&   x1,
                                const Vector3&   x2,
                                const Vector3&   x3,
                                const Matrix3x3& DmInv)
{
    auto ds = Ds(x0, x1, x2, x3);
    return ds * DmInv;
}
}  // namespace uipc::backend::cuda