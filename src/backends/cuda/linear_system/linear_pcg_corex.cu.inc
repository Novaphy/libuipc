#include <linear_system/linear_pcg.h>
#include <sim_engine.h>
#include <linear_system/global_linear_system.h>
#include <cuda_device/builtin.h>
#include <utils/matrix_market.h>
#include <utils/corex_phase_profile.h>
#include <backends/common/backend_path_tool.h>
#include <uipc/common/timer.h>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
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

bool corex_pcg_diag_enabled()
{
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
    static const bool enabled = std::getenv("UIPC_COREX_PCG_DIAG") != nullptr;
    return enabled;
#else
    return false;
#endif
}

#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
// Replace the dotnorm finalize sequence
//   cudaDeviceSynchronize() + (DeviceVar<Float>::operator Float)*2
// with a single explicit cudaMemcpyAsync per scalar to a lazily-allocated
// pinned host buffer, followed by one cudaStreamSynchronize.
//
// The default DeviceVar<Float>::operator Float() implementation issues a
// synchronous cudaMemcpy that blocks on the default stream, so the legacy
// path effectively pays for three sync points per dotnorm call (one device
// drain plus two implicit per-scalar drains). Pinned host memory makes the
// async memcpys cheap, so the critical path collapses to one stream sync.
//
// Toggle:
//   - default: ON
//   - UIPC_COREX_PCG_PINNED_SCALAR=0: rollback to the legacy
//     cudaDeviceSynchronize + implicit memcpy path.
inline bool corex_pcg_pinned_scalar_enabled()
{
    const char* env = std::getenv("UIPC_COREX_PCG_PINNED_SCALAR");
    if(!env) return true;
    return env[0] != '\0' && env[0] != '0';
}

inline bool corex_pcg_env_enabled(const char* name)
{
    const char* env = std::getenv(name);
    return env && env[0] != '\0' && env[0] != '0';
}

// Two-slot pinned host buffer. Lazily allocated on first use; kept alive
// for the lifetime of the process. The PCG solver is single-threaded on
// host so a process-wide singleton is safe for the existing call sites.
inline Float* corex_pcg_pinned_scalar_slot()
{
    static Float* g_buf = nullptr;
    if(!g_buf)
    {
        Float* tmp = nullptr;
        if(cudaMallocHost(reinterpret_cast<void**>(&tmp), sizeof(Float) * 2)
           == cudaSuccess)
            g_buf = tmp;
    }
    return g_buf;
}
#endif

bool corex_pcg_fused_rz_norm_enabled()
{
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
    return std::getenv("UIPC_COREX_PCG_SEPARATE_RZ_NORM") == nullptr;
#else
    return false;
#endif
}

bool corex_pcg_cost_diag_enabled()
{
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
    static const bool enabled = std::getenv("UIPC_COREX_PCG_COST_DIAG") != nullptr;
    return enabled;
#else
    return false;
#endif
}

bool corex_pcg_scalar_audit_enabled()
{
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
    static const bool enabled = [] {
        const char* env = std::getenv("UIPC_COREX_PCG_SCALAR_AUDIT");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }();
    return enabled;
#else
    return false;
#endif
}

bool corex_pcg_linear_trace_enabled()
{
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
    static const bool enabled =
        std::getenv("UIPC_COREX_TRACE_LINEAR_SYSTEM") != nullptr;
    return enabled;
#else
    return false;
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
    const bool corex_trace_pcg = corex_pcg_linear_trace_enabled();
#else
    constexpr bool corex_trace_pcg = false;
#endif

    auto max_iter = static_cast<SizeT>(max_iter_ratio * static_cast<Float>(b.size()));
    max_iter  = std::max(max_iter, SizeT{1});
    if(corex_trace_pcg)
        logger::info("[corex_trace][PCG] do_solve: entering pcg(), dof={} max_iter={}", x.size(), max_iter);
    auto solve_profile_start = corex_profile::now_ms();
    auto iter = pcg(x, b, max_iter);
    corex_profile::log_phase("pcg",
                             "solve_total",
                             engine().frame(),
                             engine().newton_iter(),
                             iter,
                             corex_profile::now_ms() - solve_profile_start);
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

__global__ void kernel_dot_norm_blocks(
    int n, const Float* a, const Float* b, Float* out_dot, Float* out_norm2)
{
    __shared__ Float dot_cache[256];
    __shared__ Float norm_cache[256];

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    Float dot_sum  = 0;
    Float norm_sum = 0;
    if(i < n)
    {
        Float av = a[i];
        dot_sum  = av * b[i];
        norm_sum = av * av;
    }

    dot_cache[tid]  = dot_sum;
    norm_cache[tid] = norm_sum;
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if(tid < stride)
        {
            dot_cache[tid] += dot_cache[tid + stride];
            norm_cache[tid] += norm_cache[tid + stride];
        }
        __syncthreads();
    }

    if(tid == 0)
    {
        atomicAdd(out_dot, dot_cache[0]);
        atomicAdd(out_norm2, norm_cache[0]);
    }
}

#endif

std::pair<Float, Float> corex_dot_norm(cudaStream_t                   stream,
                                       muda::CDenseVectorView<Float> a,
                                       muda::CDenseVectorView<Float> b,
                                       muda::DeviceVar<Float>&       d_dot,
                                       muda::DeviceVar<Float>&       d_norm2,
                                       muda::DeviceBuffer<Float>&    partials)
{
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
    int n = static_cast<int>(a.size());
    checkCudaErrors(cudaMemsetAsync(d_dot.data(), 0, sizeof(Float), stream));
    checkCudaErrors(cudaMemsetAsync(d_norm2.data(), 0, sizeof(Float), stream));
    if(n > 0)
    {
        int block = 256;
        int grid  = (n + block - 1) / block;
        kernel_dot_norm_blocks<<<grid, block, 0, stream>>>(
            n, a.buffer_view().data(), b.buffer_view().data(), d_dot.data(), d_norm2.data());
        checkCudaErrors(cudaGetLastError());
    }
    Float dot;
    Float norm2;
    Float* pinned = corex_pcg_pinned_scalar_enabled()
                        ? corex_pcg_pinned_scalar_slot()
                        : nullptr;
    if(pinned)
    {
        checkCudaErrors(cudaMemcpyAsync(
            pinned, d_dot.data(), sizeof(Float), cudaMemcpyDeviceToHost, stream));
        checkCudaErrors(cudaMemcpyAsync(
            pinned + 1, d_norm2.data(), sizeof(Float), cudaMemcpyDeviceToHost, stream));
        checkCudaErrors(cudaStreamSynchronize(stream));
        dot   = pinned[0];
        norm2 = pinned[1];
    }
    else
    {
        checkCudaErrors(cudaDeviceSynchronize());
        dot   = d_dot;
        norm2 = d_norm2;
    }
    return {dot, std::sqrt(std::max(norm2, Float{0}))};
#else
    (void)stream;
    (void)a;
    (void)b;
    (void)d_dot;
    (void)d_norm2;
    (void)partials;
    return {0, 0};
#endif
}

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
    const auto profile_frame  = static_cast<long long>(engine().frame());
    const auto profile_newton = static_cast<long long>(engine().newton_iter());
    double profile_spmv_ms = 0.0;
    double profile_spmv_sync_ms = 0.0;
    double profile_precond_ms = 0.0;
    double profile_dotnorm_ms = 0.0;

    SizeT k = 0;
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
    const bool corex_trace_pcg = corex_pcg_linear_trace_enabled();
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

    Float alpha, beta, rz, abs_rz0;
    Float norm_b = 0;
    Float init_norm_r = 0;
    Float init_norm_z = 0;
    Float final_norm_r = 0;
    Float final_rz = 0;
    Float min_abs_pAp = std::numeric_limits<Float>::max();
    Float max_abs_pAp = 0;
    Float last_pAp = 0;
    const bool pcg_diag = corex_pcg_diag_enabled();
    const bool pcg_cost_diag = corex_pcg_cost_diag_enabled();
    const bool fused_rz_norm = corex_pcg_fused_rz_norm_enabled();
    const bool scalar_audit = corex_pcg_scalar_audit_enabled();
    int scalar_dot_calls = 0;
    int scalar_norm_calls = 0;
    int scalar_fused_dotnorm_calls = 0;

    if(corex_trace_pcg)
        logger::info("[corex_trace][PCG] apply_preconditioner begin");
    // z = P * r (apply preconditioner)
    {
        Timer timer{"Apply Preconditioner"};
        auto profile_t0 = corex_profile::now_ms();
        apply_preconditioner(z, r, d_converged_false.view());
        profile_precond_ms += corex_profile::now_ms() - profile_t0;
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
    auto profile_dot_t0 = corex_profile::now_ms();
    if(fused_rz_norm)
    {
        auto dot_norm = corex_dot_norm(
            stream, r.cview(), z.cview(), d_corex_dot, d_corex_norm2, corex_reduce_partials);
        rz = dot_norm.first;
        init_norm_r = dot_norm.second;
        ++scalar_fused_dotnorm_calls;
    }
    else
    {
        rz = ctx().dot(r.cview(), z.cview());
        ++scalar_dot_calls;
    }
    profile_dotnorm_ms += corex_profile::now_ms() - profile_dot_t0;
    if(corex_trace_pcg)
        logger::info("[corex_trace][PCG] dot(r,z) done, rz={}", rz);
    check_init_rz_nan_inf(rz);

    profile_dot_t0 = corex_profile::now_ms();
    norm_b = ctx().norm(b);
    ++scalar_norm_calls;
    profile_dotnorm_ms += corex_profile::now_ms() - profile_dot_t0;
    if(norm_b <= pcg_zero_tol())
    {
        logger::info("[corex] PCG: b==0, trivial solution x=0");
        return 0;
    }

    abs_rz0 = std::abs(rz);
    // Textbook PCG relative-residual convergence: ||r|| <= max(tol_rate*||b||, pcg_zero_tol).
    // The legacy `|r.z| <= pcg_zero_tol` early-exit is unsafe on Corex because the
    // diagonal preconditioner P=diag(H)^-1 can compress z to ~1e-8 when the IPC
    // barrier inflates H_ii to ~1e6, so |r.z| ≈ |r|·|z| satisfies the zero tol
    // even though |r| is far from zero. That false convergence returns dq=0 and
    // the simulation freezes. Below we keep the legacy criterion only as a
    // shadow log so we can observe how often the old bug would have fired.
    {
        if(!fused_rz_norm)
        {
            profile_dot_t0 = corex_profile::now_ms();
            init_norm_r = ctx().norm(r.cview());
            ++scalar_norm_calls;
            profile_dotnorm_ms += corex_profile::now_ms() - profile_dot_t0;
        }
        if(pcg_diag)
        {
            profile_dot_t0 = corex_profile::now_ms();
            init_norm_z = ctx().norm(z.cview());
            ++scalar_norm_calls;
            profile_dotnorm_ms += corex_profile::now_ms() - profile_dot_t0;
        }
        Float r_tol    = std::max(global_tol_rate * norm_b, pcg_zero_tol());
        bool  legacy_exit = (abs_rz0 <= pcg_zero_tol());
        if(corex_trace_pcg && legacy_exit && init_norm_r > r_tol)
        {
            logger::info("[pcg_legacy_would_exit] init: norm(b)={}, norm(r)={}, norm(z)={}, |r.z|={} (kept iterating)",
                         norm_b, init_norm_r, ctx().norm(z.cview()), abs_rz0);
        }
        final_norm_r = init_norm_r;
        final_rz = rz;
        if(init_norm_r <= r_tol)
        {
            logger::info("LinearPCG: early exit (||r|| <= tol), norm(b)={}, norm(r)={}, norm(z)={}, |r.z|={}",
                         norm_b, init_norm_r, ctx().norm(z.cview()), abs_rz0);
            if(pcg_diag)
            {
                logger::info("[corex_pcg_diag] frame={} newton={} dof={} triplets={} local_precond={} no_precond={} iter=0 norm_b={} init_norm_r={} init_norm_z={} final_norm_r={} rz0={} final_rz={} r_tol={} fused_rz_norm={}",
                             profile_frame,
                             profile_newton,
                             b.size(),
                             linear_system_triplet_count(),
                             linear_system_local_preconditioner_count(),
                             linear_system_no_preconditioner_count(),
                             norm_b,
                             init_norm_r,
                             init_norm_z,
                             final_norm_r,
                             abs_rz0,
                             final_rz,
                             r_tol,
                             fused_rz_norm ? 1 : 0);
            }
            return 0;
        }
    }

    for(k = 1; k < max_iter; ++k)
    {
        if(corex_trace_pcg && (k <= 3 || k == max_iter - 1))
            logger::info("[corex_trace][PCG] iter {} begin (spmv)", k);
        Float pAp = 0;
        {
            Timer timer{"SpMV"};
            auto profile_t0 = corex_profile::now_ms();
            spmv(p.cview(), Ap.view());
            profile_spmv_ms += corex_profile::now_ms() - profile_t0;
        }

#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
        auto profile_sync_t0 = corex_profile::now_ms();
        if(!corex_pcg_env_enabled("UIPC_COREX_PCG_SKIP_SPMV_SYNC")
           && !corex_pcg_env_enabled("UIPC_COREX_LINEAR_SKIP_SYNC"))
            checkCudaErrors(cudaDeviceSynchronize());
        profile_spmv_sync_ms += corex_profile::now_ms() - profile_sync_t0;
#endif
        if(corex_trace_pcg && (k <= 3 || k == max_iter - 1))
            logger::info("[corex_trace][PCG] iter {} spmv done", k);

        if(need_debug_dump) [[unlikely]]
            dump_p_Ap(k);

        // alpha = rz / p^T * Ap
        profile_dot_t0 = corex_profile::now_ms();
        pAp = ctx().dot(p.cview(), Ap.cview());
        ++scalar_dot_calls;
        profile_dotnorm_ms += corex_profile::now_ms() - profile_dot_t0;
        last_pAp = pAp;
        min_abs_pAp = std::min(min_abs_pAp, std::abs(pAp));
        max_abs_pAp = std::max(max_abs_pAp, std::abs(pAp));
        alpha = rz / pAp;

        if(corex_trace_pcg && (k <= 3 || k == max_iter - 1))
            logger::info("[corex_trace][PCG] iter {} alpha={} (rz={} pAp={})", k, alpha, rz, pAp);

        // x = x + alpha * p
        // r = r - alpha * Ap
        update_xr(stream, alpha, x, p.cview(), r.view(), Ap.cview());

        // z = P * r (apply preconditioner)
        {
            Timer timer{"Apply Preconditioner"};
            auto profile_t0 = corex_profile::now_ms();
            apply_preconditioner(z, r, d_converged_false.view());
            profile_precond_ms += corex_profile::now_ms() - profile_t0;
        }

        if(need_debug_dump) [[unlikely]]
            dump_r_z(k);

        // rz_new = r^T * z
        profile_dot_t0 = corex_profile::now_ms();
        Float rz_new = 0;
        Float norm_r = 0;
        if(fused_rz_norm)
        {
            auto dot_norm = corex_dot_norm(
                stream, r.cview(), z.cview(), d_corex_dot, d_corex_norm2, corex_reduce_partials);
            rz_new = dot_norm.first;
            norm_r = dot_norm.second;
            ++scalar_fused_dotnorm_calls;
        }
        else
        {
            rz_new = ctx().dot(r.cview(), z.cview());
            ++scalar_dot_calls;
        }
        profile_dotnorm_ms += corex_profile::now_ms() - profile_dot_t0;
        check_iter_rz_nan_inf(rz_new, k);

        // Textbook PCG iter convergence: ||r|| <= max(tol_rate*||b||, pcg_zero_tol).
        // See the init-block comment above for why we cannot trust |r.z| on Corex.
        {
            if(!fused_rz_norm)
            {
                profile_dot_t0 = corex_profile::now_ms();
                norm_r = ctx().norm(r.cview());
                ++scalar_norm_calls;
                profile_dotnorm_ms += corex_profile::now_ms() - profile_dot_t0;
            }
            Float r_tol       = std::max(global_tol_rate * norm_b, pcg_zero_tol());
            auto  rz_tol      = std::max(global_tol_rate * abs_rz0, pcg_zero_tol());
            bool  legacy_exit = (std::abs(rz_new) <= rz_tol);
            if(corex_trace_pcg && legacy_exit && norm_r > r_tol)
            {
                logger::info("[pcg_legacy_would_exit] iter {}: norm(r)={}, |r.z_new|={} (kept iterating)",
                             k, norm_r, std::abs(rz_new));
            }
            if(norm_r <= r_tol)
            {
                final_norm_r = norm_r;
                final_rz = rz_new;
                break;
            }
        }

        // beta = rz_new / rz
        beta = rz_new / rz;

        // p = z + beta * p
        update_p(stream, p.view(), z.cview(), beta);

        rz = rz_new;
        final_norm_r = norm_r;
        final_rz = rz;
    }

    corex_profile::log_phase("pcg", "spmv", profile_frame, profile_newton, k, profile_spmv_ms);
    corex_profile::log_phase(
        "pcg", "spmv_sync", profile_frame, profile_newton, k, profile_spmv_sync_ms);
    corex_profile::log_phase(
        "pcg", "preconditioner", profile_frame, profile_newton, k, profile_precond_ms);
    corex_profile::log_phase(
        "pcg", "dotnorm", profile_frame, profile_newton, k, profile_dotnorm_ms);
    if(pcg_cost_diag)
    {
        double iter_count = std::max<double>(1.0, static_cast<double>(k));
        double measured = profile_spmv_ms + profile_spmv_sync_ms + profile_precond_ms
                          + profile_dotnorm_ms;
        double inv_measured = measured > 0.0 ? 100.0 / measured : 0.0;
        logger::info("[corex_pcg_cost] frame={} newton={} iter={} "
                     "spmv_ms={:.6g} spmv_sync_ms={:.6g} precond_ms={:.6g} dotnorm_ms={:.6g} "
                     "spmv_pct={:.3f} spmv_sync_pct={:.3f} precond_pct={:.3f} dotnorm_pct={:.3f} "
                     "per_iter_ms={:.6g} skip_spmv_sync={} fused_rz_norm={}",
                     profile_frame,
                     profile_newton,
                     k,
                     profile_spmv_ms,
                     profile_spmv_sync_ms,
                     profile_precond_ms,
                     profile_dotnorm_ms,
                     profile_spmv_ms * inv_measured,
                     profile_spmv_sync_ms * inv_measured,
                     profile_precond_ms * inv_measured,
                     profile_dotnorm_ms * inv_measured,
                     measured / iter_count,
                     std::getenv("UIPC_COREX_PCG_SKIP_SPMV_SYNC") != nullptr ? 1 : 0,
                     fused_rz_norm ? 1 : 0);
    }
    if(scalar_audit)
    {
        logger::info("[corex_pcg_scalar_audit] frame={} newton={} iter={} "
                     "dot_calls={} norm_calls={} fused_dotnorm_calls={} "
                     "host_scalar_results={} fused_rz_norm={}",
                     profile_frame,
                     profile_newton,
                     k,
                     scalar_dot_calls,
                     scalar_norm_calls,
                     scalar_fused_dotnorm_calls,
                     scalar_dot_calls + scalar_norm_calls + scalar_fused_dotnorm_calls,
                     fused_rz_norm ? 1 : 0);
    }
    if(pcg_diag)
    {
        Float r_tol = std::max(global_tol_rate * norm_b, pcg_zero_tol());
        logger::info("[corex_pcg_diag] frame={} newton={} dof={} triplets={} local_precond={} no_precond={} iter={} norm_b={} init_norm_r={} init_norm_z={} final_norm_r={} rz0={} final_rz={} r_tol={} min_abs_pAp={} max_abs_pAp={} last_pAp={} pcg_spmv_ms={} pcg_spmv_sync_ms={} pcg_precond_ms={} pcg_dotnorm_ms={} fused_rz_norm={}",
                     profile_frame,
                     profile_newton,
                     b.size(),
                     linear_system_triplet_count(),
                     linear_system_local_preconditioner_count(),
                     linear_system_no_preconditioner_count(),
                     k,
                     norm_b,
                     init_norm_r,
                     init_norm_z,
                     final_norm_r,
                     abs_rz0,
                     final_rz,
                     r_tol,
                     min_abs_pAp == std::numeric_limits<Float>::max() ? Float{0} : min_abs_pAp,
                     max_abs_pAp,
                     last_pAp,
                     profile_spmv_ms,
                     profile_spmv_sync_ms,
                     profile_precond_ms,
                     profile_dotnorm_ms,
                     fused_rz_norm ? 1 : 0);
    }
    return k;
}
}  // namespace uipc::backend::cuda
