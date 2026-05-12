#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT

// > Squared Version
// > D := d*d

#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT && \
    !(defined(UIPC_FLOAT_SCALAR) && UIPC_FLOAT_SCALAR)
#include <limits>
namespace corex_barrier_detail {

// Manual log(x) using the identity:
//   log(x) = 2 * atanh((x-1)/(x+1))
// with range reduction x = val * 2^n so that val in [0.5, 2).
template <typename T>
__host__ __device__ T safe_log(T x)
{
    if(x <= T(0)) return T(-1e30);

    T val = x;
    int n = 0;
    while(val > T(2))  { val *= T(0.5); ++n; }
    while(val < T(0.5)) { val *= T(2);  --n; }

    T u = (val - T(1)) / (val + T(1));
    T u2 = u * u;
    T sum = u;
    T uk = u;
    uk *= u2; sum += uk / T(3);
    uk *= u2; sum += uk / T(5);
    uk *= u2; sum += uk / T(7);
    uk *= u2; sum += uk / T(9);
    uk *= u2; sum += uk / T(11);
    uk *= u2; sum += uk / T(13);
    uk *= u2; sum += uk / T(15);

    constexpr T ln2 = T(0.6931471805599453);
    return T(2) * sum + T(n) * ln2;
}

// Compute log(a/b) = safe_log(a) - safe_log(b) without forming a/b,
// avoiding catastrophic precision loss on CoreX for small-number division.
template <typename T>
__host__ __device__ T safe_log_ratio(T a, T b)
{
    return safe_log(a) - safe_log(b);
}

template <typename T>
__host__ __device__ T finite_diff_step(T center, T rel_scale)
{
    // Keep finite difference steps away from float underflow while still
    // scaling with the local magnitude.
    T mag      = center >= T(0) ? center : -center;
    T rel_step = (mag + T(1)) * rel_scale;
    T abs_step = T(64) * std::numeric_limits<T>::epsilon();
    return rel_step > abs_step ? rel_step : abs_step;
}

}  // namespace corex_barrier_detail
#endif

template <typename T>
__host__ __device__ void KappaBarrier(T& R, const T& kappa, const T& D_in, const T& dHat, const T& xi)
{
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT && \
    !(defined(UIPC_FLOAT_SCALAR) && UIPC_FLOAT_SCALAR)
    // Original: B(D) = -kappa * (D - xi^2 - V)^2 * log((D - xi^2) / V)
    // where V = dHat^2 + 2*dHat*xi = (dHat+xi)^2 - xi^2.
    // Use safe_log_ratio to avoid forming the small-number quotient directly.
    T xi2 = xi * xi;
    T s = D_in - xi2;                          // shifted distance
    T V = dHat * dHat + T(2) * dHat * xi;      // = (dHat+xi)^2 - xi^2
    if(s < V * T(1e-3)) s = V * T(1e-3);
    if(s > V * T(0.999)) s = V * T(0.999);
    T diff = s - V;                             // = D - xi^2 - V
    T log_ratio = corex_barrier_detail::safe_log_ratio(s, V);
    R = -kappa * diff * diff * log_ratio;
#else
auto D = D_in;
/* Sub Exprs */
auto x0 = std::pow(xi, 2);
auto x1 = std::pow(dHat, 2) + 2*dHat*xi;
/* Simplified Expr */
R = -kappa*std::pow(D - x0 - x1, 2)*log((D - x0)/x1);
#endif
}
template <typename T>
__host__ __device__ void dKappaBarrierdD(T& R, const T& kappa, const T& D_in, const T& dHat, const T& xi)
{
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT && \
    !(defined(UIPC_FLOAT_SCALAR) && UIPC_FLOAT_SCALAR)
    // Numerical first derivative via central difference of KappaBarrier.
    // eps must be a small fraction of the barrier domain width V so that
    // both D_in +/- eps stay inside the active range and avoid the clamps
    // inside KappaBarrier.  The old formula (|D|+1)*1e-3 produced eps ~1e-3
    // which exceeded the entire domain V = d_hat^2 = 0.0009 for typical
    // d_hat values, yielding completely wrong gradients.
    T xi2 = xi * xi;
    T V   = dHat * dHat + T(2) * dHat * xi;
    T s   = D_in - xi2;
    T eps = s * T(1e-3);
    T eps_min = V * T(1e-6);
    if(eps < eps_min) eps = eps_min;
    T Bp, Bm;
    KappaBarrier(Bp, kappa, D_in + eps, dHat, xi);
    KappaBarrier(Bm, kappa, D_in - eps, dHat, xi);
    R = (Bp - Bm) / (T(2) * eps);
#else
auto D = D_in;
/* Sub Exprs */
auto x0 = std::pow(xi, 2);
auto x1 = D - x0;
auto x2 = std::pow(dHat, 2);
auto x3 = dHat*xi;
auto x4 = x2 + 2*x3;
/* Simplified Expr */
R = -kappa*(2*D - 2*x0 - 2*x2 - 4*x3)*log(x1/x4) - kappa*std::pow(D - x0 - x4, 2)/x1;
#endif
}
template <typename T>
__host__ __device__ void ddKappaBarrierddD(T& R, const T& kappa, const T& D_in, const T& dHat, const T& xi)
{
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT && \
    !(defined(UIPC_FLOAT_SCALAR) && UIPC_FLOAT_SCALAR)
    // Numerical second derivative via central difference of dKappaBarrierdD.
    // Use a wider fraction of s than the first derivative (1e-2 vs 1e-3)
    // to keep the outer FD stable, but still well inside the barrier domain.
    T xi2 = xi * xi;
    T V   = dHat * dHat + T(2) * dHat * xi;
    T s   = D_in - xi2;
    T eps = s * T(1e-2);
    T eps_min = V * T(1e-5);
    if(eps < eps_min) eps = eps_min;
    T gp, gm;
    dKappaBarrierdD(gp, kappa, D_in + eps, dHat, xi);
    dKappaBarrierdD(gm, kappa, D_in - eps, dHat, xi);
    R = (gp - gm) / (T(2) * eps);
#else
auto D = D_in;
/* Sub Exprs */
auto x0 = std::pow(xi, 2);
auto x1 = D - x0;
auto x2 = std::pow(dHat, 2);
auto x3 = dHat*xi;
auto x4 = x2 + 2*x3;
auto x5 = 2*kappa;
/* Simplified Expr */
R = kappa*std::pow(D - x0 - x4, 2)/std::pow(x1, 2) - x5*log(x1/x4) - x5*(2*D - 2*x0 - 2*x2 - 4*x3)/x1;
#endif
}
#else

// > Squared Version
// > D := d*d

template <typename T>
__host__ __device__ void KappaBarrier(T& R, const T& kappa, const T& D, const T& dHat, const T& xi)
{
/*****************************************************************************************************************************
Function generated by SymEigen.py 
Author: MuGdxy
GitHub: https://github.com/MuGdxy/SymEigen
E-Mail: lxy819469559@gmail.com
******************************************************************************************************************************
LaTeX expression:
//tex:$$R = - \kappa \left(D - \hat{d}^{2} - 2 \hat{d} \xi - \xi^{2}\right)^{2} \log{\left(\frac{D - \xi^{2}}{\hat{d}^{2} + 2 \hat{d} \xi} \right)}$$

Symbol Name Mapping:
kappa:
    -> {}
    -> Matrix([[kappa]])
D:
    -> {}
    -> Matrix([[D]])
dHat:
    -> {}
    -> Matrix([[dHat]])
xi:
    -> {}
    -> Matrix([[xi]])
*****************************************************************************************************************************/
/* Sub Exprs */
auto x0 = std::pow(xi, 2);
auto x1 = std::pow(dHat, 2) + 2*dHat*xi;
/* Simplified Expr */
R = -kappa*std::pow(D - x0 - x1, 2)*log((D - x0)/x1);
}
template <typename T>
__host__ __device__ void dKappaBarrierdD(T& R, const T& kappa, const T& D, const T& dHat, const T& xi)
{
/*****************************************************************************************************************************
Function generated by SymEigen.py 
Author: MuGdxy
GitHub: https://github.com/MuGdxy/SymEigen
E-Mail: lxy819469559@gmail.com
******************************************************************************************************************************
LaTeX expression:
//tex:$$R = - \kappa \left(2 D - 2 \hat{d}^{2} - 4 \hat{d} \xi - 2 \xi^{2}\right) \log{\left(\frac{D - \xi^{2}}{\hat{d}^{2} + 2 \hat{d} \xi} \right)} - \frac{\kappa \left(D - \hat{d}^{2} - 2 \hat{d} \xi - \xi^{2}\right)^{2}}{D - \xi^{2}}$$

Symbol Name Mapping:
kappa:
    -> {}
    -> Matrix([[kappa]])
D:
    -> {}
    -> Matrix([[D]])
dHat:
    -> {}
    -> Matrix([[dHat]])
xi:
    -> {}
    -> Matrix([[xi]])
*****************************************************************************************************************************/
/* Sub Exprs */
auto x0 = std::pow(xi, 2);
auto x1 = D - x0;
auto x2 = std::pow(dHat, 2);
auto x3 = dHat*xi;
auto x4 = x2 + 2*x3;
/* Simplified Expr */
R = -kappa*(2*D - 2*x0 - 2*x2 - 4*x3)*log(x1/x4) - kappa*std::pow(D - x0 - x4, 2)/x1;
}
template <typename T>
__host__ __device__ void ddKappaBarrierddD(T& R, const T& kappa, const T& D, const T& dHat, const T& xi)
{
/*****************************************************************************************************************************
Function generated by SymEigen.py 
Author: MuGdxy
GitHub: https://github.com/MuGdxy/SymEigen
E-Mail: lxy819469559@gmail.com
******************************************************************************************************************************
LaTeX expression:
//tex:$$R = - 2 \kappa \log{\left(\frac{D - \xi^{2}}{\hat{d}^{2} + 2 \hat{d} \xi} \right)} - \frac{2 \kappa \left(2 D - 2 \hat{d}^{2} - 4 \hat{d} \xi - 2 \xi^{2}\right)}{D - \xi^{2}} + \frac{\kappa \left(D - \hat{d}^{2} - 2 \hat{d} \xi - \xi^{2}\right)^{2}}{\left(D - \xi^{2}\right)^{2}}$$

Symbol Name Mapping:
kappa:
    -> {}
    -> Matrix([[kappa]])
D:
    -> {}
    -> Matrix([[D]])
dHat:
    -> {}
    -> Matrix([[dHat]])
xi:
    -> {}
    -> Matrix([[xi]])
*****************************************************************************************************************************/
/* Sub Exprs */
auto x0 = std::pow(xi, 2);
auto x1 = D - x0;
auto x2 = std::pow(dHat, 2);
auto x3 = dHat*xi;
auto x4 = x2 + 2*x3;
auto x5 = 2*kappa;
/* Simplified Expr */
R = kappa*std::pow(D - x0 - x4, 2)/std::pow(x1, 2) - x5*log(x1/x4) - x5*(2*D - 2*x0 - 2*x2 - 4*x3)/x1;
}
#endif
