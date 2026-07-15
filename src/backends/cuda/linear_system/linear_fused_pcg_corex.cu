#include <linear_system/linear_fused_pcg.h>
#include <sim_engine.h>
#include <linear_system/global_linear_system.h>
#include <uipc/common/timer.h>
#include <cub/block/block_reduce.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <algorithm>
#include <cmath>
#include <limits>
namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(LinearFusedPCG);

namespace
{
constexpr Float fused_pcg_zero_tol()
{
    return Float(64) * std::numeric_limits<Float>::epsilon();
}

constexpr Float fused_pcg_tol_floor()
{
#if defined(UIPC_FLOAT_SCALAR) && UIPC_FLOAT_SCALAR
    return Float(1e-3);
#else
    return Float(1e-6);
#endif
}

inline bool fused_pcg_diag_enabled()
{
    static const bool enabled = [] {
        const char* env = std::getenv("UIPC_COREX_FUSED_PCG_DIAG");
        return env && env[0] != '\0' && env[0] != '0';
    }();
    return enabled;
}
}  // namespace

void LinearFusedPCG::do_build(BuildInfo& info)
{
    auto& config = world().scene().config();

    auto        solver_attr = config.find<std::string>("linear_system/solver");
    std::string solver_name =
        solver_attr ? solver_attr->view()[0] : std::string{"fused_pcg"};
    if(solver_name != "fused_pcg")
    {
        throw SimSystemException("LinearFusedPCG unused");
    }

    auto& global_linear_system = require<GlobalLinearSystem>();

    max_iter_ratio = 2;

    auto tol_rate_attr = config.find<Float>("linear_system/tol_rate");
    global_tol_rate    = tol_rate_attr->view()[0];
    if(global_tol_rate < fused_pcg_tol_floor())
    {
        logger::warn("LinearFusedPCG: tol_rate={} is too tight for current scalar type, clamp to {}",
                     global_tol_rate,
                     fused_pcg_tol_floor());
        global_tol_rate = fused_pcg_tol_floor();
    }

    auto check_attr = config.find<IndexT>("linear_system/check_interval");
    if(check_attr)
        check_interval = check_attr->view()[0];

    auto dump_attr = config.find<IndexT>("extras/debug/dump_linear_pcg");
    if(dump_attr && dump_attr->view()[0] != 0)
        logger::warn(
            "LinearFusedPCG: extras/debug/dump_linear_pcg is enabled but "
            "fused_pcg does not support PCG vector dumps. "
            "Set linear_system/solver to \"linear_pcg\" to use this feature.");

    logger::info("LinearFusedPCG: max_iter_ratio = {}, tol_rate = {}, check_interval = {}",
                 max_iter_ratio,
                 global_tol_rate,
                 check_interval);
}

void LinearFusedPCG::do_solve(GlobalLinearSystem::SolvingInfo& info)
{
    auto x = info.x();
    auto b = info.b();

    x.buffer_view().fill(0);

    auto N = x.size();
    if(r.capacity() < N)
    {
        auto M = reserve_ratio * N;
        r.reserve(M);
        z.reserve(M);
        p.reserve(M);
        Ap.reserve(M);
    }

    r.resize(N);
    z.resize(N);
    p.resize(N);
    Ap.resize(N);

    auto iter = fused_pcg(x, b, max_iter_ratio * b.size());

    info.iter_count(iter);
}

void LinearFusedPCG::check_init_rz_nan_inf(Float rz)
{
    if(!std::isfinite(rz)) [[unlikely]]
    {
        auto norm_r = ctx().norm(r.cview());
        auto norm_z = ctx().norm(z.cview());
        bool r_bad  = !std::isfinite(norm_r);
        auto hint = r_bad ? "gradient assembling produced NaN values, likely due to error in formula implementation" :
                            "preconditioner failed, likely due to inverse matrix calculation failure";
        UIPC_ASSERT(false,
                    "Frame {}, Newton {}, FusedPCG Init: r^T*z = {}, norm(r) = {}, norm(z) = {}. "
                    "Hint: {}.",
                    engine().frame(),
                    engine().newton_iter(),
                    rz,
                    norm_r,
                    norm_z,
                    hint);
    }
}

void LinearFusedPCG::check_iter_rz_nan_inf(Float rz, SizeT k)
{
    if(!std::isfinite(rz)) [[unlikely]]
    {
        auto norm_r = ctx().norm(r.cview());
        auto norm_z = ctx().norm(z.cview());
        bool r_ok   = std::isfinite(norm_r);
        bool z_bad  = !std::isfinite(norm_z);
        auto hint = (r_ok && z_bad) ?
                        "preconditioner failed, likely due to inverse matrix calculation failure" :
                        "PCG iteration diverged";
        UIPC_ASSERT(false,
                    "Frame {}, Newton {}, FusedPCG Iter {}: r^T*z = {}, norm(r) = {}, norm(z) = {}. "
                    "Hint: {}.",
                    engine().frame(),
                    engine().newton_iter(),
                    k,
                    rz,
                    norm_r,
                    norm_z,
                    hint);
    }
}

// d_result = x^T * y  (cublas-free, device-only, CUB warp reduction)
void fused_dot(muda::CDenseVectorView<Float> x,
               muda::CDenseVectorView<Float> y,
               muda::VarView<Float>          d_result)
{
    using namespace muda;

    cudaMemsetAsync(d_result.data(), 0, sizeof(Float));

    constexpr int block_dim   = 256;
    constexpr int warp_size   = 32;
    constexpr int num_warps   = block_dim / warp_size;
    int           n           = x.size();
    int           block_count = (n + block_dim - 1) / block_dim;

    Launch(block_count, block_dim)
        .file_line(__FILE__, __LINE__)
        .apply(
            [x        = x.cviewer().name("x"),
             y        = y.cviewer().name("y"),
             d_result = d_result.viewer().name("d_result"),
             n] __device__() mutable
            {
                using WarpReduce = cub::WarpReduce<Float, warp_size>;
                __shared__ typename WarpReduce::TempStorage temp_storage[num_warps];

                int   i   = blockIdx.x * blockDim.x + threadIdx.x;
                Float val = (i < n) ? x(i) * y(i) : Float(0);

                int   warp_id  = threadIdx.x / warp_size;
                int   lane_id  = threadIdx.x & (warp_size - 1);
                Float warp_sum = WarpReduce(temp_storage[warp_id]).Sum(val);

                if(lane_id == 0)
                    muda::atomic_add(d_result.data(), warp_sum);
            });
}

void fused_dot_norm(muda::CDenseVectorView<Float> x,
                    muda::CDenseVectorView<Float> y,
                    muda::VarView<Float>          d_dot,
                    muda::VarView<Float>          d_norm2)
{
    using namespace muda;

    cudaMemsetAsync(d_dot.data(), 0, sizeof(Float));
    cudaMemsetAsync(d_norm2.data(), 0, sizeof(Float));

    constexpr int block_dim = 256;
    int           n         = x.size();
    int           grid      = (n + block_dim - 1) / block_dim;

    Launch(grid, block_dim)
        .file_line(__FILE__, __LINE__)
        .apply(
            [x       = x.cviewer().name("x"),
             y       = y.cviewer().name("y"),
             d_dot   = d_dot.viewer().name("d_dot"),
             d_norm2 = d_norm2.viewer().name("d_norm2"),
            n] __device__() mutable
            {
                using BlockReduce = cub::BlockReduce<Float, block_dim>;
                __shared__ typename BlockReduce::TempStorage dot_storage;
                __shared__ typename BlockReduce::TempStorage norm_storage;

                int tid = threadIdx.x;
                int i   = blockIdx.x * blockDim.x + threadIdx.x;

                Float dot_sum  = 0;
                Float norm_sum = 0;
                if(i < n)
                {
                    Float xv = x(i);
                    dot_sum  = xv * y(i);
                    norm_sum = xv * xv;
                }

                Float block_dot  = BlockReduce(dot_storage).Sum(dot_sum);
                Float block_norm = BlockReduce(norm_storage).Sum(norm_sum);

                if(tid == 0)
                {
                    muda::atomic_add(d_dot.data(), block_dot);
                    muda::atomic_add(d_norm2.data(), block_norm);
                }
            });
}

// Same as linear_pcg update_xr: alpha = rz/pAp, x += alpha*p, r -= alpha*Ap. Alpha computed on device from d_rz, d_pAp.
void fused_update_xr(muda::CVarView<Float>         d_rz,
                     muda::CVarView<Float>         d_pAp,
                     muda::CVarView<IndexT>        d_converged,
                     muda::DenseVectorView<Float>  x,
                     muda::CDenseVectorView<Float> p,
                     muda::DenseVectorView<Float>  r,
                     muda::CDenseVectorView<Float> Ap)
{
    using namespace muda;

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(r.size(),
               [d_rz        = d_rz.cviewer().name("d_rz"),
                d_pAp       = d_pAp.cviewer().name("d_pAp"),
                d_converged = d_converged.cviewer().name("d_converged"),
                x           = x.viewer().name("x"),
                p           = p.cviewer().name("p"),
                r           = r.viewer().name("r"),
                Ap          = Ap.cviewer().name("Ap")] __device__(int i) mutable
               {
                   if(*d_converged != 0)
                       return;
                   Float alpha = *d_rz / *d_pAp;
                   x(i) += alpha * p(i);
                   r(i) -= alpha * Ap(i);
               });
}

// Same as linear_pcg update_p: beta = rz_new/rz, p = z + beta*p.
// Convergence is guarded by d_converged.
void fused_update_p(muda::CVarView<Float>         d_rz_new,
                    muda::CVarView<Float>         d_rz,
                    muda::CVarView<IndexT>        d_converged,
                    muda::DenseVectorView<Float>  p,
                    muda::CDenseVectorView<Float> z)
{
    using namespace muda;

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(p.size(),
               [d_rz_new    = d_rz_new.cviewer().name("d_rz_new"),
                d_rz        = d_rz.cviewer().name("d_rz"),
                d_converged = d_converged.cviewer().name("d_converged"),
                p           = p.viewer().name("p"),
                z           = z.cviewer().name("z")] __device__(int i) mutable
               {
                   if(*d_converged != 0)
                       return;
                   Float beta = *d_rz_new / *d_rz;
                   p(i)       = z(i) + beta * p(i);
               });
}

// d_rz = d_rz_new when not converged (single-thread write).
void fused_swap_rz(muda::CVarView<Float>  d_rz_new,
                   muda::VarView<Float>   d_rz,
                   muda::CVarView<IndexT> d_converged)
{
    using namespace muda;

    Launch()
        .file_line(__FILE__, __LINE__)
        .apply(
            [d_rz_new = d_rz_new.cviewer().name("d_rz_new"),
             d_rz     = d_rz.viewer().name("d_rz"),
             d_converged = d_converged.cviewer().name("d_converged")] __device__() mutable
            {
                if(*d_converged != 0)
                    return;
                *d_rz = *d_rz_new;
            });
}

void fused_update_converged(muda::CVarView<Float> d_rz_new,
                            muda::VarView<IndexT> d_converged,
                            Float                 rz_tol)
{
    using namespace muda;

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(1,
               [d_rz_new    = d_rz_new.cviewer().name("d_rz_new"),
                d_converged = d_converged.viewer().name("d_converged"),
                rz_tol] __device__(int) mutable
               {
                   Float rz_new = *d_rz_new;
                   *d_converged = abs(rz_new) <= rz_tol ? 1 : 0;
               });
}

void fused_update_converged_by_norm(muda::CVarView<Float> d_norm2,
                                    muda::VarView<IndexT> d_converged,
                                    Float                 r_tol)
{
    using namespace muda;

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(1,
               [d_norm2     = d_norm2.cviewer().name("d_norm2"),
                d_converged = d_converged.viewer().name("d_converged"),
                r_tol] __device__(int) mutable
               {
                   Float norm2 = *d_norm2;
                   *d_converged = norm2 <= r_tol * r_tol ? 1 : 0;
               });
}

SizeT LinearFusedPCG::fused_pcg(muda::DenseVectorView<Float>  x,
                                muda::CDenseVectorView<Float> b,
                                SizeT                         max_iter)
{
    Timer pcg_timer{"FusedPCG"};

    SizeT k     = 0;
    d_converged = 0;

    // r = b - A*x, but x0 = 0 so r = b
    r.buffer_view().copy_from(b.buffer_view());

    // z = P^{-1} * r
    {
        Timer timer{"Apply Preconditioner"};
        apply_preconditioner(z, r, d_converged.view());
    }

    // p = z
    p = z;

    // rz = r^T * z
    fused_dot_norm(r.cview(), z.cview(), d_rz.view(), d_norm2.view());
    Float rz_host = d_rz;
    Float norm_r_host =
        std::sqrt(std::max(static_cast<Float>(d_norm2), Float{0}));
    // x is explicitly initialized to zero before entering PCG, so r = b for
    // the initial residual. Reuse the norm already produced by fused_dot_norm
    // instead of launching a second global reduction for ||b||.
    Float norm_b = norm_r_host;
    check_init_rz_nan_inf(rz_host);
    Float abs_rz0 = std::abs(rz_host);
    Float init_norm_r = norm_r_host;

    if(norm_b <= fused_pcg_zero_tol())
        return 0;

    Float r_tol = std::max(global_tol_rate * norm_b, fused_pcg_zero_tol());
    if(norm_r_host <= r_tol)
        return 0;

    SizeT effective_check_interval = check_interval > 0 ? check_interval : SizeT{1};
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
    // Device-side convergence still guards all state updates every iteration.
    // Batch only the host readback that terminates the launch loop.
    effective_check_interval = std::max<SizeT>(effective_check_interval, SizeT{20});
#endif
    const bool diag = fused_pcg_diag_enabled();
    Float final_rz = rz_host;
    Float final_norm_r = norm_r_host;

    for(k = 1; k < max_iter; ++k)
    {
        // Ap = A * p,  pAp = p^T * Ap
        {
            Timer timer{"SpMV"};
            spmv_dot(p.cview(), Ap.view(), d_pAp.view());
        }

        // alpha = rz / pAp,  x += alpha * p,  r -= alpha * Ap
        fused_update_xr(
            d_rz.view(), d_pAp.view(), d_converged.view(), x, p.cview(), r.view(), Ap.cview());

        // z = P^{-1} * r
        {
            Timer timer{"Apply Preconditioner"};
            apply_preconditioner(z, r, d_converged.view());
        }

        // rz_new = r^T * z and norm(r).  Use the same residual-norm
        // convergence criterion as LinearPCG; |r^T z| is not equivalent when
        // the preconditioner strongly scales contact rows.
        fused_dot_norm(r.cview(), z.cview(), d_rz_new.view(), d_norm2.view());
        fused_update_converged_by_norm(d_norm2.view(), d_converged.view(), r_tol);

        // Check error ratio periodically to avoid per-iteration D2H synchronization.
        bool do_check = (k % effective_check_interval == 0) || (k + 1 == max_iter);
        if(do_check)
        {
            Float rz_new_host = d_rz_new;
            norm_r_host =
                std::sqrt(std::max(static_cast<Float>(d_norm2), Float{0}));
            check_iter_rz_nan_inf(rz_new_host, k);
            final_rz = rz_new_host;
            final_norm_r = norm_r_host;
            if(norm_r_host <= r_tol)
                break;
        }

        // p = z + beta * p (skip when abs(rz_new) <= rz_tol), then rz = rz_new.
        fused_update_p(d_rz_new.view(), d_rz.view(), d_converged.view(), p.view(), z.cview());
        fused_swap_rz(d_rz_new.view(), d_rz.view(), d_converged.view());
    }

    if(diag)
    {
        logger::info("[corex_fused_pcg_diag] frame={} newton={} dof={} iter={} "
                     "norm_b={} init_norm_r={} final_norm_r={} rz0={} final_rz={} r_tol={}",
                     engine().frame(),
                     engine().newton_iter(),
                     b.size(),
                     k,
                     norm_b,
                     init_norm_r,
                     final_norm_r,
                     abs_rz0,
                     final_rz,
                     r_tol);
    }

    return k;
}
}  // namespace uipc::backend::cuda
