#pragma once
#include <type_define.h>
#include <muda/ext/eigen/evd.h>

namespace uipc::backend::cuda
{
template <int N>
UIPC_HOST UIPC_DEVICE void make_spd(Matrix<Float, N, N>& H)
{
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
#if defined(UIPC_FLOAT_SCALAR) && UIPC_FLOAT_SCALAR
    // CoreX + float: float operations are correct on CoreX FPU, so Jacobi EVD
    // converges properly.  Use eigenvalue projection (clamp negatives to zero)
    // instead of Gershgorin diagonal shift, which over-stiffens tangential
    // directions and causes post-contact hover/freeze artifacts.
    // All Eigen expression templates (setIdentity, asDiagonal, transpose,
    // operator*) are avoided — CoreX clang-CUDA emits invalid addrspacecast
    // for them inside ParallelFor lambdas.  Raw element access only.
    Float E[N][N];
    for(int i = 0; i < N; ++i)
        for(int j = 0; j < N; ++j)
            E[i][j] = (i == j) ? Float(1) : Float(0);

    constexpr int max_iter = N * N * 10;
    Float diag_norm = Float(0);
    for(int i = 0; i < N; ++i)
    {
        Float v = H(i, i);
        diag_norm += (v < Float(0) ? -v : v);
    }
    Float tol = diag_norm * static_cast<Float>(1e-4);
    if(tol < static_cast<Float>(1e-6))
        tol = static_cast<Float>(1e-6);

    for(int iter = 0; iter < max_iter; ++iter)
    {
        Float off_sum = Float(0);
        for(int i = 0; i < N; ++i)
            for(int j = i + 1; j < N; ++j)
            {
                Float v = H(i, j);
                off_sum += (v < Float(0) ? -v : v);
            }
        if(off_sum <= tol)
            break;

        int   p = 0, q = 1;
        Float max_val = Float(-1);
        for(int i = 0; i < N; ++i)
            for(int j = i + 1; j < N; ++j)
            {
                Float av = H(i, j);
                if(av < Float(0)) av = -av;
                if(av > max_val) { max_val = av; p = i; q = j; }
            }
        if(max_val < static_cast<Float>(1e-12))
            break;

        Float tau = (H(q, q) - H(p, p)) / (Float(2) * H(p, q));
        Float t;
        if(tau >= Float(0))
            t = Float(1) / (tau + sqrt(Float(1) + tau * tau));
        else
            t = -Float(1) / (-tau + sqrt(Float(1) + tau * tau));
        Float c = Float(1) / sqrt(Float(1) + t * t);
        Float s = t * c;

        for(int i = 0; i < N; ++i)
        {
            if(i != p && i != q)
            {
                Float Hip = H(i, p), Hiq = H(i, q);
                H(p, i) = H(i, p) = c * Hip - s * Hiq;
                H(q, i) = H(i, q) = s * Hip + c * Hiq;
            }
            Float Eip = E[i][p], Eiq = E[i][q];
            E[i][p] = c * Eip - s * Eiq;
            E[i][q] = s * Eip + c * Eiq;
        }
        Float Hpp = H(p, p), Hqq = H(q, q), Hpq = H(p, q);
        H(p, p) = c * c * Hpp + s * s * Hqq - Float(2) * c * s * Hpq;
        H(q, q) = s * s * Hpp + c * c * Hqq + Float(2) * c * s * Hpq;
        H(p, q) = H(q, p) = Float(0);
    }

    Float evals[N];
    for(int i = 0; i < N; ++i)
        evals[i] = H(i, i) < Float(0) ? Float(0) : H(i, i);

    // H = E * diag(evals) * E^T, element-by-element
    for(int i = 0; i < N; ++i)
        for(int j = i; j < N; ++j)
        {
            Float sum = Float(0);
            for(int k = 0; k < N; ++k)
                sum += E[i][k] * evals[k] * E[j][k];
            H(i, j) = H(j, i) = sum;
        }
#else
    // CoreX + double: evd_jacobi produces garbage due to ~23-bit double
    // mantissa on CoreX FPU; fall back to Gershgorin-based diagonal shift.
    Float min_gershgorin = H(0, 0);
    for(int i = 0; i < N; ++i)
    {
        Float off_diag_sum = Float(0);
        for(int j = 0; j < N; ++j)
            if(j != i)
            {
                Float v = H(i, j);
                off_diag_sum += (v < Float(0) ? -v : v);
            }
        Float lower = H(i, i) - off_diag_sum;
        if(lower < min_gershgorin)
            min_gershgorin = lower;
    }
    if(min_gershgorin < Float(1e-10))
    {
        Float shift = -min_gershgorin + Float(1e-10);
        for(int i = 0; i < N; ++i)
            H(i, i) += shift;
    }
#endif
#else
    Vector<Float, N>    eigen_values;
    Matrix<Float, N, N> eigen_vectors;
    muda::eigen::template evd<Float, N>(H, eigen_values, eigen_vectors);
    for(int i = 0; i < N; ++i)
    {
        auto& v = eigen_values(i);
        v       = v < 0.0 ? 0.0 : v;
    }
    H = eigen_vectors * eigen_values.asDiagonal() * eigen_vectors.transpose();
#endif
}
}