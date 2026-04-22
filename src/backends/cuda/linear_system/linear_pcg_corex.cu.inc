#include <linear_system/linear_pcg.h>
#include <sim_engine.h>
#include <linear_system/global_linear_system.h>
#include <cuda_device/builtin.h>
#include <utils/matrix_market.h>
#include <backends/common/backend_path_tool.h>
#include <uipc/common/timer.h>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(LinearPCG);

namespace
{
constexpr Float pcg_zero_tol()
{
    return Float(64) * std::numeric_limits<Float>::epsilon();
}

constexpr Float pcg_tol_floor()
{
#if defined(UIPC_FLOAT_SCALAR) && UIPC_FLOAT_SCALAR
    return Float(1e-3);
#else
    return Float(1e-6);
#endif
}
}  // namespace

void LinearPCG::do_build(BuildInfo& info)
{
    auto& config = world().scene().config();

    auto solver_attr = config.find<std::string>("linear_system/solver");
    UIPC_ASSERT(solver_attr, "linear_system/solver not found");
    if(solver_attr->view()[0] != "linear_pcg")
    {
        throw SimSystemException("LinearPCG unused");
    }

    auto& global_linear_system = require<GlobalLinearSystem>();

    // TODO: get info from the scene, now we just use the default value
    max_iter_ratio = 2;

    auto tol_rate_attr = config.find<Float>("linear_system/tol_rate");
    UIPC_ASSERT(tol_rate_attr, "linear_system/tol_rate not found");
    global_tol_rate = tol_rate_attr->view()[0];
    if(global_tol_rate < pcg_tol_floor())
    {
        logger::warn("LinearPCG: tol_rate={} is too tight for current scalar type, clamp to {}",
                     global_tol_rate,
                     pcg_tol_floor());
        global_tol_rate = pcg_tol_floor();
    }

    auto dump_attr = config.find<IndexT>("extras/debug/dump_linear_pcg");
    UIPC_ASSERT(dump_attr, "extras/debug/dump_linear_pcg not found");
    need_debug_dump = dump_attr->view()[0];

    logger::info("LinearPCG: max_iter_ratio = {}, tol_rate = {}, debug_dump = {}",
                 max_iter_ratio,
                 global_tol_rate,
                 need_debug_dump);
}

void LinearPCG::do_solve(GlobalLinearSystem::SolvingInfo& info)
{
    auto x = info.x();
    auto b = info.b();
    auto stream = ctx().stream();

    checkCudaErrors(cudaMemsetAsync(x.data(), 0, sizeof(Float) * x.size(), stream));

    auto N = x.size();
    if(z.capacity() < N)
    {
        auto M = reserve_ratio * N;
        z.reserve(M);
        p.reserve(M);
        r.reserve(M);
        Ap.reserve(M);
    }

    z.resize(N);
    p.resize(N);
    r.resize(N);
    Ap.resize(N);
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
    muda::wait_device();
#endif
    checkCudaErrors(cudaMemsetAsync(d_converged_false.data(), 0, sizeof(IndexT), stream));

#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
    const bool corex_trace_pcg = (std::getenv("UIPC_COREX_TRACE_LINEAR_SYSTEM") != nullptr);
#else
    constexpr bool corex_trace_pcg = false;
#endif

    auto max_iter = static_cast<SizeT>(max_iter_ratio * static_cast<Float>(b.size()));
    max_iter  = std::max(max_iter, SizeT{1});
    if(corex_trace_pcg)
        logger::info("[corex_trace][PCG] do_solve: entering pcg(), dof={} max_iter={}", x.size(), max_iter);
    auto iter = pcg(x, b, max_iter);
    if(corex_trace_pcg)
        logger::info("[corex_trace][PCG] do_solve: pcg() returned iters={}", iter);

    logger::info("LinearPCG: frame={} newton_iter={} dof={} max_iter={} -> iters={}",
                 engine().frame(),
                 engine().newton_iter(),
                 N,
                 max_iter,
                 iter);

    if(iter >= max_iter)
        logger::warn(
            "LinearPCG: reached max_iter = {} (no early convergence); "
            "check preconditioner or linear_system/tol_rate.",
            max_iter);

    info.iter_count(iter);
}

void LinearPCG::dump_r_z(SizeT k)
{

    auto path_tool   = BackendPathTool(workspace());
    auto output_path = path_tool.workspace(UIPC_RELATIVE_SOURCE_FILE, "debug");
    auto output_path_r = fmt::format(
        "{}r.{}.{}.{}.mtx", output_path.string(), engine().frame(), engine().newton_iter(), k);

    export_vector_market(output_path_r, r.cview());
    logger::info("Dumped PCG r to {}", output_path_r);

    auto output_path_z = fmt::format(
        "{}z.{}.{}.{}.mtx", output_path.string(), engine().frame(), engine().newton_iter(), k);

    export_vector_market(fmt::format("{}z.{}.{}.{}.mtx",
                                     output_path.string(),
                                     engine().frame(),
                                     engine().newton_iter(),
                                     k),
                         z.cview());

    logger::info("Dumped PCG z to {}", output_path_z);
}

void LinearPCG::dump_p_Ap(SizeT k)
{
    auto path_tool = BackendPathTool(workspace());
    auto output_folder = path_tool.workspace(UIPC_RELATIVE_SOURCE_FILE, "debug");

    auto output_path_p = fmt::format("{}p.{}.{}.{}.mtx",
                                     output_folder.string(),
                                     engine().frame(),
                                     engine().newton_iter(),
                                     k);

    export_vector_market(output_path_p, p.cview());
    logger::info("Dumped PCG p to {}", output_path_p);

    auto output_path_Ap = fmt::format("{}Ap.{}.{}.{}.mtx",
                                      output_folder.string(),
                                      engine().frame(),
                                      engine().newton_iter(),
                                      k);
    export_vector_market(output_path_Ap, Ap.cview());
    logger::info("Dumped PCG Ap to {}", output_path_Ap);
}

void LinearPCG::check_init_rz_nan_inf(Float rz)
{
    if(!std::isfinite(rz)) [[unlikely]]
    {
        auto norm_r = ctx().norm(r.cview());
        auto norm_z = ctx().norm(z.cview());
        bool r_bad  = !std::isfinite(norm_r);
        auto hint = r_bad ? "gradient assembling produced NaN values, likely due to error in formula implementation" :
                            "preconditioner failed, likely due to inverse matrix calculation failure";
        UIPC_ASSERT(false,
                    "Frame {}, Newton {}, PCG Init: r^T*z = {}, norm(r) = {}, norm(z) = {}. "
                    "Hint: {}.",
                    engine().frame(),
                    engine().newton_iter(),
                    rz,
                    norm_r,
                    norm_z,
                    hint);
    }
}

void LinearPCG::check_iter_rz_nan_inf(Float rz, SizeT k)
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
                    "Frame {}, Newton {}, PCG Iter {}: r^T*z = {}, norm(r) = {}, norm(z) = {}. "
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

#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
__global__ void kernel_update_xr(int n, Float alpha, Float* x, const Float* p, Float* r, const Float* Ap)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    x[i] += alpha * p[i];
    r[i] -= alpha * Ap[i];
}

__global__ void kernel_update_p(int n, Float beta, Float* p, const Float* z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    p[i] = z[i] + beta * p[i];
}
#endif

void update_xr(cudaStream_t                  stream,
               Float                         alpha,
               muda::DenseVectorView<Float>  x,
               muda::CDenseVectorView<Float> p,
               muda::DenseVectorView<Float>  r,
               muda::CDenseVectorView<Float> Ap)
{
    using namespace muda;

#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
    int n = static_cast<int>(r.size());
    int block = 256;
    int grid  = (n + block - 1) / block;
    kernel_update_xr<<<grid, block, 0, stream>>>(n,
                                                  alpha,
                                                  x.buffer_view().data(),
                                                  p.buffer_view().data(),
                                                  r.buffer_view().data(),
                                                  Ap.buffer_view().data());
    checkCudaErrors(cudaGetLastError());
#else
    ParallelFor(0, stream)
        .file_line(__FILE__, __LINE__)
        .apply(r.size(),
               [alpha = alpha,
                x     = x.viewer().name("x"),
                p     = p.cviewer().name("p"),
                r     = r.viewer().name("r"),
                Ap    = Ap.cviewer().name("Ap")] __device__(int i) mutable
               {
                   x(i) += alpha * p(i);
                   r(i) -= alpha * Ap(i);
               });
#endif
}

void update_p(cudaStream_t                  stream,
              muda::DenseVectorView<Float> p,
              muda::CDenseVectorView<Float> z,
              Float                         beta)
{
    using namespace muda;

#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
    int n = static_cast<int>(p.size());
    int block = 256;
    int grid  = (n + block - 1) / block;
    kernel_update_p<<<grid, block, 0, stream>>>(
        n, beta, p.buffer_view().data(), z.buffer_view().data());
    checkCudaErrors(cudaGetLastError());
#else
    ParallelFor(0, stream)
        .file_line(__FILE__, __LINE__)
        .apply(p.size(),
               [p = p.viewer().name("p"), z = z.cviewer().name("z"), beta = beta] __device__(
                   int i) mutable { p(i) = z(i) + beta * p(i); });
#endif
}

SizeT LinearPCG::pcg(muda::DenseVectorView<Float> x, muda::CDenseVectorView<Float> b, SizeT max_iter)
{
    Timer pcg_timer{"PCG"};
    auto  stream = ctx().stream();

    SizeT k = 0;
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
    const bool corex_trace_pcg = (std::getenv("UIPC_COREX_TRACE_LINEAR_SYSTEM") != nullptr);
#else
    constexpr bool corex_trace_pcg = false;
#endif

    // r = b - A * x
    {
        // r = b;
        checkCudaErrors(cudaMemcpyAsync(
            r.buffer_view().data(), b.data(), sizeof(Float) * b.size(), cudaMemcpyDeviceToDevice, stream));

        // x == 0, so we don't need to do the following
        // r = - A * x + r
        //spmv(-1.0, x.as_const(), 1.0, r.view());
    }

#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
    {
        checkCudaErrors(cudaDeviceSynchronize());
        auto n = static_cast<int>(b.size());
        std::vector<Float> hb(n);
        checkCudaErrors(cudaMemcpy(hb.data(), b.data(), sizeof(Float) * n, cudaMemcpyDeviceToHost));
        Float hnorm_b = 0;
        for(int i = 0; i < n; ++i)
            hnorm_b += hb[i] * hb[i];
        if(hnorm_b <= pcg_zero_tol())
        {
            logger::info("[corex] PCG: b==0, trivial solution x=0");
            return 0;
        }
    }
#endif

    Float alpha, beta, rz, abs_rz0;

    if(corex_trace_pcg)
        logger::info("[corex_trace][PCG] apply_preconditioner begin");
    // z = P * r (apply preconditioner)
    {
        Timer timer{"Apply Preconditioner"};
        apply_preconditioner(z, r, d_converged_false.view());
    }
    if(corex_trace_pcg)
        logger::info("[corex_trace][PCG] apply_preconditioner done");

    if(need_debug_dump) [[unlikely]]
        dump_r_z(k);

    // p = z
    checkCudaErrors(cudaMemcpyAsync(p.buffer_view().data(),
                                    z.buffer_view().data(),
                                    sizeof(Float) * z.size(),
                                    cudaMemcpyDeviceToDevice,
                                    stream));

    if(corex_trace_pcg)
        logger::info("[corex_trace][PCG] dot(r,z) begin");
    rz = ctx().dot(r.cview(), z.cview());
    if(corex_trace_pcg)
        logger::info("[corex_trace][PCG] dot(r,z) done, rz={}", rz);
    check_init_rz_nan_inf(rz);

    abs_rz0 = std::abs(rz);
    // Textbook PCG relative-residual convergence: ||r|| <= max(tol_rate*||b||, pcg_zero_tol).
    // The legacy `|r.z| <= pcg_zero_tol` early-exit is unsafe on Corex because the
    // diagonal preconditioner P=diag(H)^-1 can compress z to ~1e-8 when the IPC
    // barrier inflates H_ii to ~1e6, so |r.z| ≈ |r|·|z| satisfies the zero tol
    // even though |r| is far from zero. That false convergence returns dq=0 and
    // the simulation freezes. Below we keep the legacy criterion only as a
    // shadow log so we can observe how often the old bug would have fired.
    {
        Float norm_b   = ctx().norm(b);
        Float norm_r   = ctx().norm(r.cview());
        Float r_tol    = std::max(global_tol_rate * norm_b, pcg_zero_tol());
        bool  legacy_exit = (abs_rz0 <= pcg_zero_tol());
        if(legacy_exit && norm_r > r_tol)
        {
            logger::info("[pcg_legacy_would_exit] init: norm(b)={}, norm(r)={}, norm(z)={}, |r.z|={} (kept iterating)",
                         norm_b, norm_r, ctx().norm(z.cview()), abs_rz0);
        }
        if(norm_r <= r_tol)
        {
            logger::info("LinearPCG: early exit (||r|| <= tol), norm(b)={}, norm(r)={}, norm(z)={}, |r.z|={}",
                         norm_b, norm_r, ctx().norm(z.cview()), abs_rz0);
            return 0;
        }
    }

    for(k = 1; k < max_iter; ++k)
    {
        if(corex_trace_pcg && (k <= 3 || k == max_iter - 1))
            logger::info("[corex_trace][PCG] iter {} begin (spmv)", k);
        {
            Timer timer{"SpMV"};
            spmv(p.cview(), Ap.view());
        }

#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
        checkCudaErrors(cudaDeviceSynchronize());
#endif
        if(corex_trace_pcg && (k <= 3 || k == max_iter - 1))
            logger::info("[corex_trace][PCG] iter {} spmv done", k);

        if(need_debug_dump) [[unlikely]]
            dump_p_Ap(k);

        // alpha = rz / p^T * Ap
        Float pAp = ctx().dot(p.cview(), Ap.cview());
        alpha = rz / pAp;

        if(corex_trace_pcg && (k <= 3 || k == max_iter - 1))
            logger::info("[corex_trace][PCG] iter {} alpha={} (rz={} pAp={})", k, alpha, rz, pAp);

        // x = x + alpha * p
        // r = r - alpha * Ap
        update_xr(stream, alpha, x, p.cview(), r.view(), Ap.cview());

        // z = P * r (apply preconditioner)
        {
            Timer timer{"Apply Preconditioner"};
            apply_preconditioner(z, r, d_converged_false.view());
        }

        if(need_debug_dump) [[unlikely]]
            dump_r_z(k);

        // rz_new = r^T * z
        Float rz_new = ctx().dot(r.cview(), z.cview());
        check_iter_rz_nan_inf(rz_new, k);

        // Textbook PCG iter convergence: ||r|| <= max(tol_rate*||b||, pcg_zero_tol).
        // See the init-block comment above for why we cannot trust |r.z| on Corex.
        {
            Float norm_b      = ctx().norm(b);
            Float norm_r      = ctx().norm(r.cview());
            Float r_tol       = std::max(global_tol_rate * norm_b, pcg_zero_tol());
            auto  rz_tol      = std::max(global_tol_rate * abs_rz0, pcg_zero_tol());
            bool  legacy_exit = (std::abs(rz_new) <= rz_tol);
            if(legacy_exit && norm_r > r_tol)
            {
                logger::info("[pcg_legacy_would_exit] iter {}: norm(r)={}, |r.z_new|={} (kept iterating)",
                             k, norm_r, std::abs(rz_new));
            }
            if(norm_r <= r_tol)
                break;
        }

        // beta = rz_new / rz
        beta = rz_new / rz;

        // p = z + beta * p
        update_p(stream, p.view(), z.cview(), beta);

        rz = rz_new;
    }

    return k;
}
}  // namespace uipc::backend::cuda
