#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
#include <affine_body/affine_body_kinetic.h>
#include <time_integrator/bdf1_flag.h>
#include <muda/ext/eigen/evd.h>
#include <muda/check/check_cuda_errors.h>
#include <affine_body/abd_jacobi_matrix_corex.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>

namespace uipc::backend::cuda
{
namespace
{
std::atomic<unsigned long long>& corex_bdf1_gh_call_count()
{
    static std::atomic<unsigned long long> count{0};
    return count;
}

void corex_trace_bdf1_gh_call(int n, bool gradient_only, bool gpu_path)
{
    static const bool trace_bdf1_gh =
        std::getenv("UIPC_COREX_ABD_BDF1_TRACE_GH") != nullptr;
    if(!trace_bdf1_gh)
        return;
    auto call = corex_bdf1_gh_call_count().fetch_add(1, std::memory_order_relaxed) + 1;
    std::fprintf(stderr,
                 "[corex_abd_bdf1_gh] call=%llu n=%d gradient_only=%d path=%s\n",
                 call,
                 n,
                 gradient_only ? 1 : 0,
                 gpu_path ? "gpu" : "host");
}
}  // namespace

MUDA_DEVICE void corex_abd_mass_mul(const ABDJacobiDyadicMass& mass,
                                    const Float*               p,
                                    Float*                     ret)
{
    const Float     m = mass.mass();
    const Vector3&  x = mass.mass_times_x_bar();
    const Matrix3x3& D = mass.mass_times_dyadic_x_bar();

    ret[0] = x[0] * p[3] + x[1] * p[4] + x[2] * p[5] + m * p[0];
    ret[1] = x[0] * p[6] + x[1] * p[7] + x[2] * p[8] + m * p[1];
    ret[2] = x[0] * p[9] + x[1] * p[10] + x[2] * p[11] + m * p[2];

    for(int r = 0; r < 3; ++r)
    {
        ret[3 + r] = D(r, 0) * p[3] + D(r, 1) * p[4] + D(r, 2) * p[5] + x[r] * p[0];
        ret[6 + r] = D(r, 0) * p[6] + D(r, 1) * p[7] + D(r, 2) * p[8] + x[r] * p[1];
        ret[9 + r] = D(r, 0) * p[9] + D(r, 1) * p[10] + D(r, 2) * p[11] + x[r] * p[2];
    }
}

MUDA_DEVICE void corex_abd_mass_to_mat(const ABDJacobiDyadicMass& mass,
                                       Matrix12x12&               h)
{
    const Float      m = mass.mass();
    const Vector3&   x = mass.mass_times_x_bar();
    const Matrix3x3& D = mass.mass_times_dyadic_x_bar();

    h.setZero();
    h(0, 0) = m;
    h(1, 1) = m;
    h(2, 2) = m;

    for(int k = 0; k < 3; ++k)
    {
        h(0, 3 + k) = x[k];
        h(3 + k, 0) = x[k];
        h(1, 6 + k) = x[k];
        h(6 + k, 1) = x[k];
        h(2, 9 + k) = x[k];
        h(9 + k, 2) = x[k];
    }

    for(int r = 0; r < 3; ++r)
    {
        for(int c = 0; c < 3; ++c)
        {
            h(3 + r, 3 + c) = D(r, c);
            h(6 + r, 6 + c) = D(r, c);
            h(9 + r, 9 + c) = D(r, c);
        }
    }
}

__global__ void kernel_abd_bdf1_energy(int                        n,
                                       const Vector12* __restrict__ qs,
                                       const Vector12* __restrict__ q_tildes,
                                       const ABDJacobiDyadicMass* __restrict__ masses,
                                       const IndexT* __restrict__   is_fixed,
                                       const IndexT* __restrict__   ext_kinetic,
                                       Float* __restrict__          energies)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;
    if(is_fixed[i] || ext_kinetic[i])
    {
        energies[i] = 0.f;
        return;
    }

    Float dq[12];
    Float M_dq[12];
    for(int k = 0; k < 12; ++k)
        dq[k] = qs[i](k) - q_tildes[i](k);

    corex_abd_mass_mul(masses[i], dq, M_dq);

    Float e = 0;
    for(int k = 0; k < 12; ++k)
        e += dq[k] * M_dq[k];
    energies[i] = Float(0.5) * e;
}

__global__ void kernel_abd_bdf1_gradient_hessian(int                        n,
                                                 const Vector12* __restrict__ qs,
                                                 const Vector12* __restrict__ q_tildes,
                                                 const ABDJacobiDyadicMass* __restrict__ masses,
                                                 const IndexT* __restrict__ is_fixed,
                                                 Vector12* __restrict__               gradients,
                                                 Matrix12x12* __restrict__            hessians,
                                                 bool                                 gradient_only)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;

    Float dq[12];
    Float g_values[12];
    for(int k = 0; k < 12; ++k)
        dq[k] = qs[i](k) - q_tildes[i](k);

    corex_abd_mass_mul(masses[i], dq, g_values);

    Vector12 g;
    for(int k = 0; k < 12; ++k)
        g(k) = g_values[k];
    if(is_fixed[i])
        g = Vector12::Zero();
    gradients[i] = g;

    if(gradient_only)
        return;

    Matrix12x12 h;
    corex_abd_mass_to_mat(masses[i], h);
    hessians[i] = h;
}


class AffineBodyBDF1Kinetic final : public AffineBodyKinetic
{
  public:
    using AffineBodyKinetic::AffineBodyKinetic;

    std::vector<ABDJacobiDyadicMass> h_mass_cache;
    std::vector<IndexT>              h_fixed_cache;
    std::vector<Matrix12x12>         h_hessian_cache;
    std::vector<Vector12>            h_q_cache;
    std::vector<Vector12>            h_qtilde_cache;
    std::vector<Vector12>            h_grad_cache;
    muda::DeviceBuffer<Matrix12x12>  cached_kinetic_hessians;
    int                              hessian_cache_size = 0;

    void refresh_hessian_cache_if_needed(int n,
                                         const ABDJacobiDyadicMass* masses,
                                         const IndexT*              is_fixed)
    {
        if(hessian_cache_size == n)
            return;

        h_mass_cache.resize(n);
        h_fixed_cache.resize(n);
        h_hessian_cache.resize(n);
        cudaMemcpy(h_mass_cache.data(),
                   masses,
                   n * sizeof(ABDJacobiDyadicMass),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(h_fixed_cache.data(),
                   is_fixed,
                   n * sizeof(IndexT),
                   cudaMemcpyDeviceToHost);
        for(int i = 0; i < n; ++i)
            h_hessian_cache[i] = h_mass_cache[i].to_mat();

        cached_kinetic_hessians.resize(n);
        cudaMemcpy(cached_kinetic_hessians.data(),
                   h_hessian_cache.data(),
                   n * sizeof(Matrix12x12),
                   cudaMemcpyHostToDevice);
        hessian_cache_size = n;
    }

    virtual void do_build(BuildInfo& info) override
    {
        require<BDF1Flag>();
    }

    virtual void do_compute_energy(ComputeEnergyInfo& info) override
    {
        using namespace muda;
        int n = static_cast<int>(info.qs().size());
        if(n <= 0)
            return;

        constexpr int block = 256;
        int           grid  = (n + block - 1) / block;
        kernel_abd_bdf1_energy<<<grid, block>>>(n,
                                                 info.qs().data(),
                                                 info.q_tildes().data(),
                                                 info.masses().data(),
                                                 info.is_fixed().data(),
                                                 info.external_kinetic().data(),
                                                 info.energies().data());
        checkCudaErrors(cudaGetLastError());
    }

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        using namespace muda;
        const int n = static_cast<int>(info.qs().size());

        if(n <= 0)
            return;

        corex_trace_bdf1_gh_call(n, info.gradient_only(), true);

        constexpr int block = 256;
        int           grid  = (n + block - 1) / block;
        kernel_abd_bdf1_gradient_hessian<<<grid, block>>>(n,
                                                            info.qs().data(),
                                                            info.q_tildes().data(),
                                                            info.masses().data(),
                                                            info.is_fixed().data(),
                                                            info.gradients().data(),
                                                            info.hessians().data(),
                                                            info.gradient_only());
        checkCudaErrors(cudaGetLastError());
    }
};

REGISTER_SIM_SYSTEM(AffineBodyBDF1Kinetic);
}  // namespace uipc::backend::cuda
#else
#include <affine_body/affine_body_kinetic.h>
#include <time_integrator/bdf1_flag.h>
#include <muda/ext/eigen/evd.h>
#include <kernel_cout.h>

namespace uipc::backend::cuda
{
class AffineBodyBDF1Kinetic final : public AffineBodyKinetic
{
  public:
    using AffineBodyKinetic::AffineBodyKinetic;

    virtual void do_build(BuildInfo& info) override
    {
        // need BDF1 flag for BDF1 time integration
        require<BDF1Flag>();
    }

    virtual void do_compute_energy(ComputeEnergyInfo& info) override
    {
        using namespace muda;
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.qs().size(),
                   [is_fixed   = info.is_fixed().cviewer().name("is_fixed"),
                    is_dynamic = info.is_dynamic().cviewer().name("is_dynamic"),
                    ext_kinetic = info.external_kinetic().cviewer().name("ext_kinetic"),
                    qs        = info.qs().cviewer().name("qs"),
                    q_prevs   = info.q_prevs().cviewer().name("q_tildes"),
                    q_tildes  = info.q_tildes().cviewer().name("q_tildes"),
                    gravities = info.gravities().cviewer().name("gravities"),
                    masses    = info.masses().cviewer().name("masses"),
                    Ks = info.energies().viewer().name("kinetic_energy")] __device__(int i) mutable
                   {
                       auto& K = Ks(i);
                       if(is_fixed(i) || ext_kinetic(i))
                       {
                           K = 0.0;
                       }
                       else
                       {
                           const auto& q       = qs(i);
                           const auto& q_tilde = q_tildes(i);
                           const auto& M       = masses(i);
                           Vector12    dq      = q - q_tilde;
                           K                   = 0.5 * dq.dot(M * dq);
                       }
                   });
    }

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        using namespace muda;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.qs().size(),
                   [is_fixed   = info.is_fixed().cviewer().name("is_fixed"),
                    is_dynamic = info.is_dynamic().cviewer().name("is_dynamic"),
                    qs         = info.qs().cviewer().name("qs"),
                    q_prevs    = info.q_prevs().cviewer().name("q_tildes"),
                    q_tildes   = info.q_tildes().cviewer().name("q_tildes"),
                    gravities  = info.gravities().cviewer().name("gravities"),
                    masses     = info.masses().cviewer().name("masses"),
                    hessians   = info.hessians().viewer().name("hessians"),
                    gradients  = info.gradients().viewer().name("gradients"),
                    dt         = info.dt(),
                    gradient_only = info.gradient_only(),
                    cout = KernelCout::viewer()] __device__(int i) mutable
                   {
                       const auto& q       = qs(i);
                       const auto& q_prev  = q_prevs(i);
                       const auto& q_tilde = q_tildes(i);
                       auto&       G       = gradients(i);
                       const auto& M       = masses(i);

                       G = M * (q - q_tilde);


                       if(is_fixed(i))
                       {
                           G = Vector12::Zero();
                       }

                       // cout << "KG(" << i << "): " << G.transpose().eval() << "\n";

                       if(gradient_only)
                           return;

                       hessians(i) = M.to_mat();
                   });
    }
};

REGISTER_SIM_SYSTEM(AffineBodyBDF1Kinetic);
}  // namespace uipc::backend::cuda
#endif
