#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
#include <linear_system/local_preconditioner.h>
#include <affine_body/affine_body_dynamics.h>
#include <affine_body/abd_linear_subsystem.h>
#include <linear_system/global_linear_system.h>
#include <muda/ext/eigen/inverse.h>
#include <kernel_cout.h>
#include <muda/check/check_cuda_errors.h>
#include <cstdlib>
#include <limits>
#include <vector>
#include <algorithm>

namespace uipc::backend::cuda
{
namespace
{
// Jacobi (diagonal-only) preconditioner for CoreX: z_k = r_k / H_{kk}.
// Full block-inverse suffers catastrophic cancellation at CoreX's float-level
// double precision when off-diagonal values are close to diagonal values.
bool parse_precond_diag_clamp(Float& min_abs_diag, Float& max_abs_diag)
{
    struct ClampConfig
    {
        bool  enabled;
        Float min_abs_diag;
        Float max_abs_diag;
    };

    static const ClampConfig config = [] {
        ClampConfig cfg{false, Float{0}, std::numeric_limits<Float>::max()};
        const char* env = std::getenv("UIPC_COREX_ABD_PRECOND_DIAG_CLAMP");
        if(!env || env[0] == '\0')
            return cfg;

        char*  end   = nullptr;
        double min_v = std::strtod(env, &end);
        if(end == env || min_v < 0)
            return cfg;

        double max_v = 0.0;
        if(*end == ',')
        {
            const char* max_start = end + 1;
            max_v = std::strtod(max_start, &end);
            if(end == max_start || max_v <= 0)
                max_v = 0.0;
        }

        cfg.enabled      = true;
        cfg.min_abs_diag = static_cast<Float>(min_v);
        cfg.max_abs_diag = max_v > 0.0 ? static_cast<Float>(max_v) :
                                         std::numeric_limits<Float>::max();
        return cfg;
    }();

    if(!config.enabled)
        return false;

    min_abs_diag = config.min_abs_diag;
    max_abs_diag = config.max_abs_diag;
    return true;
}

// Default-on switch for the numerically-stable 12x12 LDLT block-inverse
// preconditioner that mirrors the NVIDIA path semantically. When enabled,
// the diagonal Jacobi reciprocal is still computed as a per-DoF safety
// fallback, which is selected only for bodies whose 12x12 block fails the
// LDLT SPD pivot test.
//
// Rollback knobs:
// - `UIPC_COREX_ABD_PRECOND_BLOCK_INVERSE=0` disables the block-inverse
//   path and keeps the legacy Jacobi extract.
// - `UIPC_COREX_ABD_PRECOND_DIAG_JACOBI=1` is a hard rollback that forces
//   the legacy Jacobi extract regardless of the block-inverse flag.
bool block_inverse_precond_enabled()
{
    static const bool force_jacobi =
        std::getenv("UIPC_COREX_ABD_PRECOND_DIAG_JACOBI") != nullptr;
    if(force_jacobi)
        return false;
    static const int env_enabled = [] {
        const char* env = std::getenv("UIPC_COREX_ABD_PRECOND_BLOCK_INVERSE");
        if(!env)
            return -1;
        return (env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }();
    if(env_enabled >= 0)
        return env_enabled != 0;

#if defined(UIPC_ENABLE_GIPC_CONTACT_MATRIX_FREE) && UIPC_ENABLE_GIPC_CONTACT_MATRIX_FREE
    // The matrix-free contact path removes the same contact Hessian blocks from
    // explicit BCOO assembly but still contributes their diagonal estimate. A
    // pure 12x12 LDLT block inverse is too aggressive on dense CoreX ABD contact
    // frames; a damped block/Jacobi blend keeps the stronger cross-DoF
    // preconditioning while preserving the long-run stability of Jacobi.
    return true;
#else
    return true;
#endif
}

bool block_inverse_precond_stats_enabled()
{
    static const bool enabled = [] {
        const char* env = std::getenv("UIPC_COREX_ABD_PRECOND_BLOCK_INVERSE_STATS");
        return env && env[0] != '\0' && env[0] != '0';
    }();
    return enabled;
}

Float block_inverse_precond_mix()
{
    static const Float mix = [] {
        const char* env = std::getenv("UIPC_COREX_ABD_PRECOND_BLOCK_MIX");
        if(!env || env[0] == '\0')
#if defined(UIPC_ENABLE_GIPC_CONTACT_MATRIX_FREE) && UIPC_ENABLE_GIPC_CONTACT_MATRIX_FREE
            return Float{0.4};
#else
            return Float{1};
#endif

        char*  end = nullptr;
        double v   = std::strtod(env, &end);
        if(end == env)
            return Float{1};
        if(v < 0.0)
            v = 0.0;
        if(v > 1.0)
            v = 1.0;
        return static_cast<Float>(v);
    }();
    return mix;
}

bool corex_abd_precond_sync_enabled()
{
    static const bool enabled =
        std::getenv("UIPC_COREX_ABD_PRECOND_SYNC") != nullptr
        || std::getenv("UIPC_COREX_TRACE_LINEAR_SYSTEM") != nullptr;
    return enabled;
}

bool corex_abd_trace_linear_system_enabled()
{
    static const bool enabled =
        std::getenv("UIPC_COREX_TRACE_LINEAR_SYSTEM") != nullptr;
    return enabled;
}

bool corex_abd_precond_diag_stats_enabled()
{
    static const bool enabled =
        std::getenv("UIPC_COREX_ABD_PRECOND_DIAG_STATS") != nullptr
        || std::getenv("UIPC_COREX_PCG_DIAG") != nullptr;
    return enabled;
}

__device__ inline Float corex_abs(Float v)
{
    return v < 0 ? -v : v;
}

// In-place LDLT factorization for an SPD matrix stored column-major in `A`.
// On exit, A holds the LDLT factors:
//   - the lower triangle of A (i > j) holds L[i][j] (unit diagonal of L is implicit)
//   - the diagonal of A holds D[k]
//   - the upper triangle (i < j) is left untouched
//
// Returns true on success. Returns false if any pivot D[k] is below the
// SPD safety threshold (relative to the largest input diagonal). Callers
// should treat false as "fall back to plain Jacobi for this body".
//
// The threshold uses the input matrix diagonal as the scale reference, and
// also bounds the relative threshold from below by `1e-30` so that bodies
// with extremely large Hessian magnitudes still get a non-degenerate test.
template<int N>
__device__ bool corex_ldlt_factorize_inplace(Float* A)
{
    Float max_diag_abs = static_cast<Float>(0);
    for(int k = 0; k < N; ++k)
    {
        Float d  = A[k * N + k];
        Float ad = d < static_cast<Float>(0) ? -d : d;
        if(ad > max_diag_abs) max_diag_abs = ad;
    }

    Float eps = max_diag_abs * static_cast<Float>(1e-10);
    if(eps < static_cast<Float>(1e-30)) eps = static_cast<Float>(1e-30);

    for(int k = 0; k < N; ++k)
    {
        Float Dk = A[k * N + k];
        for(int j = 0; j < k; ++j)
        {
            Float Lkj = A[j * N + k];
            Float Dj  = A[j * N + j];
            Dk -= Lkj * Lkj * Dj;
        }
        if(Dk < eps) return false;
        A[k * N + k] = Dk;

        for(int i = k + 1; i < N; ++i)
        {
            Float Lik = A[k * N + i];
            for(int j = 0; j < k; ++j)
            {
                Float Lij = A[j * N + i];
                Float Lkj = A[j * N + k];
                Float Dj  = A[j * N + j];
                Lik -= Lij * Lkj * Dj;
            }
            A[k * N + i] = Lik / Dk;
        }
    }
    return true;
}

// Solve A*x = b given the in-place LDLT factors of A. Internally uses two
// length-N stack vectors for the forward/backward substitution intermediates.
template<int N>
__device__ void corex_ldlt_solve(const Float* A, const Float* b, Float* x)
{
    Float w[N];
    for(int i = 0; i < N; ++i)
    {
        Float s = b[i];
        for(int j = 0; j < i; ++j)
            s -= A[j * N + i] * w[j];
        w[i] = s;
    }

    Float y[N];
    for(int i = 0; i < N; ++i)
        y[i] = w[i] / A[i * N + i];

    for(int i = N - 1; i >= 0; --i)
    {
        Float s = y[i];
        for(int j = i + 1; j < N; ++j)
            s -= A[i * N + j] * x[j];
        x[i] = s;
    }
}

// Compute A^{-1} as a column-major dense matrix from the in-place LDLT
// factors of A. Each column of the inverse is recovered by solving against
// a unit basis vector; this avoids constructing L^{-1} explicitly and
// keeps the solve numerically stable.
template<int N>
__device__ void corex_ldlt_explicit_inverse(const Float* A, Float* invA)
{
    for(int j = 0; j < N; ++j)
    {
        Float ej[N];
        Float xj[N];
        for(int i = 0; i < N; ++i)
            ej[i] = (i == j) ? static_cast<Float>(1) : static_cast<Float>(0);
        corex_ldlt_solve<N>(A, ej, xj);
        for(int i = 0; i < N; ++i)
            invA[j * N + i] = xj[i];
    }
}

// Build a 12x12 SPD block inverse for each ABD body using LDLT factorization.
// Mirrors the algorithmic intent of the NVIDIA path
// (`muda::eigen::inverse(diag_hessian(i))`), but uses an SPD-safe LDLT with
// an explicit pivot test instead of the unpivoted Gauss elimination that
// proved unstable on CoreX.
//
// For bodies that fail the SPD pivot test we still write a usable inverse:
// a diagonal matrix containing the per-DoF Jacobi reciprocals, so the apply
// path remains an unconditional 12x12 mat-vec.
//
// `block_status` records 1 for LDLT-accepted bodies and 0 for Jacobi
// fallback, so the host can log accepted/rejected ratios in a diagnostic
// pass without re-reading the Hessian.
__global__ void kernel_abd_precond_extract_block_inverse(
    int n,
    const Float* diag_hessian,
    Float*       diag_inv,
    Float*       diag_recip,
    int*         block_status,
    Float        min_abs_diag,
    Float        max_abs_diag)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    constexpr int N = 12;
    Float A[N * N];

    for(int c = 0; c < N; ++c)
    {
        for(int r = 0; r < N; ++r)
        {
            Float h_rc = diag_hessian[i * 144 + c * N + r];
            Float h_cr = diag_hessian[i * 144 + r * N + c];
            A[c * N + r] = static_cast<Float>(0.5) * (h_rc + h_cr);
        }
    }

    for(int k = 0; k < N; ++k)
    {
        Float d     = A[k * N + k];
        Float abs_d = d < 0 ? -d : d;
        if(min_abs_diag > 0 && abs_d > 0 && abs_d < min_abs_diag)
            d = d < 0 ? -min_abs_diag : min_abs_diag;
        if(max_abs_diag > 0 && abs_d > max_abs_diag)
            d = d < 0 ? -max_abs_diag : max_abs_diag;
        diag_recip[i * N + k] = (d != static_cast<Float>(0))
                                    ? (static_cast<Float>(1) / d)
                                    : static_cast<Float>(0);
    }

    bool ok = corex_ldlt_factorize_inplace<N>(A);

    if(ok)
    {
        Float invA[N * N];
        corex_ldlt_explicit_inverse<N>(A, invA);
        for(int idx = 0; idx < N * N; ++idx)
            diag_inv[i * 144 + idx] = invA[idx];
        if(block_status) block_status[i] = 1;
    }
    else
    {
        for(int idx = 0; idx < N * N; ++idx)
            diag_inv[i * 144 + idx] = static_cast<Float>(0);
        for(int k = 0; k < N; ++k)
            diag_inv[i * 144 + k * N + k] = diag_recip[i * N + k];
        if(block_status) block_status[i] = 0;
    }
}

// Apply z = A^{-1} * r as an unconditional 12x12 mat-vec per body. This is
// the runtime-hot kernel that runs once per PCG iteration, so it must stay
// branch-free (the SPD pivot fallback is folded into `diag_inv` at extract
// time so the apply path is identical for accepted and fallback bodies).
__global__ void kernel_abd_block_inverse_apply(int           n,
                                               const Float*  diag_inv,
                                               const Float*  diag_recip,
                                               Float         block_mix,
                                               const Float*  r,
                                               Float*        z,
                                               const IndexT* converged)
{
    if(*converged != 0) return;
    constexpr int N = 12;
    constexpr int LanesPerBody = 16;
    constexpr int BodiesPerBlock = 4;

    int local_body = threadIdx.x / LanesPerBody;
    int row        = threadIdx.x - local_body * LanesPerBody;
    int i          = blockIdx.x * BodiesPerBlock + local_body;
    if(i >= n) return;

    Float r_lane = row < N ? r[i * N + row] : static_cast<Float>(0);

    if(row < N)
    {
        Float s = static_cast<Float>(0);
        for(int col = 0; col < N; ++col)
        {
            Float r_col = __shfl_sync(0xffffffffu, r_lane, col, LanesPerBody);
            s += diag_inv[i * 144 + col * N + row] * r_col;
        }
        if(block_mix < static_cast<Float>(1))
        {
            const Float j = diag_recip[i * N + row] * r_lane;
            s = block_mix * s + (static_cast<Float>(1) - block_mix) * j;
        }
        z[i * N + row] = s;
    }
}

__global__ void kernel_abd_precond_extract(int          n,
                                           const Float* diag_hessian,
                                           Float*       diag_recip,
                                           Float        min_abs_diag,
                                           Float        max_abs_diag)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    for(int k = 0; k < 12; ++k)
    {
        Float d = diag_hessian[i * 144 + k * 12 + k]; // column-major: element (k,k)
        Float abs_d = d < 0 ? -d : d;
        if(min_abs_diag > 0 && abs_d > 0 && abs_d < min_abs_diag)
            d = d < 0 ? -min_abs_diag : min_abs_diag;
        if(max_abs_diag > 0 && abs_d > max_abs_diag)
            d = d < 0 ? -max_abs_diag : max_abs_diag;
        diag_recip[i * 12 + k] = (d != 0.0) ? (1.0 / d) : 0.0;
    }
}

__global__ void kernel_abd_jacobi_apply(
    int n, const Float* diag_recip, const Float* r, Float* z, const IndexT* converged)
{
    if(*converged != 0) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    for(int k = 0; k < 12; ++k)
        z[i * 12 + k] = diag_recip[i * 12 + k] * r[i * 12 + k];
}

}  // namespace

class ABDDiagPreconditioner final : public LocalPreconditioner
{
  public:
    using LocalPreconditioner::LocalPreconditioner;

    ABDLinearSubsystem* abd_linear_subsystem = nullptr;

    muda::DeviceBuffer<Matrix12x12> diag_inv;
    muda::DeviceBuffer<Float> jacobi_recip; // 12 reciprocals per body
    muda::DeviceBuffer<int> block_inv_status; // 1 = LDLT accepted, 0 = Jacobi fallback
    bool block_inverse_enabled = false;
    Float block_inverse_mix = Float{1};

    virtual void do_build(BuildInfo& info) override
    {
        auto& global_linear_system = require<GlobalLinearSystem>();
        abd_linear_subsystem       = &require<ABDLinearSubsystem>();

        info.connect(abd_linear_subsystem);
    }

    virtual void do_init(InitInfo& info) override {}

    virtual void do_assemble(GlobalLinearSystem::LocalPreconditionerAssemblyInfo& info) override
    {
        using namespace muda;

        if(corex_abd_trace_linear_system_enabled())
            logger::info("[corex_trace][precond] do_assemble: entry");

        auto diag_hessian = abd_linear_subsystem->diag_hessian();

        if(corex_abd_trace_linear_system_enabled())
            logger::info("[corex_trace][precond] do_assemble: diag_hessian.size()={}, data()={}",
                         diag_hessian.size(), (void*)diag_hessian.data());

        diag_inv.resize(diag_hessian.size());

        if(corex_abd_trace_linear_system_enabled())
            logger::info("[corex_trace][precond] do_assemble: diag_inv resized to {}", diag_inv.size());

        {
            auto n = static_cast<int>(diag_hessian.size());
            if(n > 0)
            {
                jacobi_recip.resize(n * 12);
                Float min_abs_diag = 0;
                Float max_abs_diag = std::numeric_limits<Float>::max();
                bool  clamp_enabled = parse_precond_diag_clamp(min_abs_diag, max_abs_diag);
                block_inverse_enabled = block_inverse_precond_enabled();
                block_inverse_mix = block_inverse_precond_mix();
                int blocks = (n + 255) / 256;
                if(block_inverse_enabled)
                {
                    diag_inv.resize(n);
                    block_inv_status.resize(n);
                    // Smaller block size: each thread holds a 12x12 working matrix
                    // plus LDLT temporaries. Keeping the block at
                    // 64 threads avoids excessive local-memory spilling versus the
                    // 256-thread Jacobi extract above.
                    constexpr int kBlk = 64;
                    int  ldlt_blocks   = (n + kBlk - 1) / kBlk;
                    kernel_abd_precond_extract_block_inverse<<<ldlt_blocks, kBlk>>>(
                        n,
                        (const Float*)diag_hessian.data(),
                        (Float*)diag_inv.data(),
                        (Float*)jacobi_recip.data(),
                        (int*)block_inv_status.data(),
                        min_abs_diag,
                        max_abs_diag);
                }
                else
                {
                    block_inv_status.resize(0);
                    kernel_abd_precond_extract<<<blocks, 256>>>(
                        n,
                        (const Float*)diag_hessian.data(),
                        (Float*)jacobi_recip.data(),
                        min_abs_diag,
                        max_abs_diag);
                }
                checkCudaErrors(cudaGetLastError());
                if(corex_abd_precond_sync_enabled())
                    checkCudaErrors(cudaDeviceSynchronize());
                if(block_inverse_enabled
                   && (block_inverse_precond_stats_enabled()
                       || corex_abd_trace_linear_system_enabled()))
                {
                    std::vector<int> h_status(n);
                    cudaMemcpy(h_status.data(),
                               block_inv_status.data(),
                               sizeof(int) * h_status.size(),
                               cudaMemcpyDeviceToHost);
                    SizeT accepted = 0;
                    for(int v : h_status)
                        accepted += v ? 1 : 0;
                    SizeT rejected = static_cast<SizeT>(n) - accepted;
                    logger::info("[corex_abd_precond_block_inv] bodies={} accepted={} rejected={}",
                                 n,
                                 accepted,
                                 rejected);
                }
                if(clamp_enabled || corex_abd_trace_linear_system_enabled())
                    logger::info("[corex_precond_diag] bodies={} clamp={} min_abs_diag={} max_abs_diag={}",
                                 n,
                                 clamp_enabled ? 1 : 0,
                                 min_abs_diag,
                                 max_abs_diag);
                if(corex_abd_precond_diag_stats_enabled())
                {
                    std::vector<Matrix12x12> h_diag(n);
                    cudaMemcpy(h_diag.data(),
                               diag_hessian.data(),
                               sizeof(Matrix12x12) * n,
                               cudaMemcpyDeviceToHost);
                    Float min_nonzero_abs = std::numeric_limits<Float>::max();
                    Float max_abs = 0;
                    SizeT zero_count = 0;
                    SizeT tiny_count = 0;
                    for(const auto& H : h_diag)
                    {
                        for(int k = 0; k < 12; ++k)
                        {
                            Float d = H(k, k);
                            Float a = d < 0 ? -d : d;
                            if(a == 0)
                            {
                                ++zero_count;
                                continue;
                            }
                            if(a < static_cast<Float>(1e-6))
                                ++tiny_count;
                            min_nonzero_abs = std::min(min_nonzero_abs, a);
                            max_abs = std::max(max_abs, a);
                        }
                    }
                    if(min_nonzero_abs == std::numeric_limits<Float>::max())
                        min_nonzero_abs = 0;
                    logger::info("[corex_precond_diag_stats] bodies={} diag_entries={} min_nonzero_abs={} max_abs={} zero_count={} tiny_count={}",
                                 n,
                                 n * 12,
                                 min_nonzero_abs,
                                 max_abs,
                                 zero_count,
                                 tiny_count);
                }
            }
        }
    }

    virtual void do_apply(GlobalLinearSystem::ApplyPreconditionerInfo& info) override
    {
        using namespace muda;
        auto converged = info.converged();

        {
            auto n = static_cast<int>(jacobi_recip.size() / 12);
            if(n > 0)
            {
                if(block_inverse_enabled && diag_inv.size() > 0)
                {
                    constexpr int kBodiesPerBlock = 4;
                    constexpr int kThreads = kBodiesPerBlock * 16;
                    int blocks = (n + kBodiesPerBlock - 1) / kBodiesPerBlock;
                    kernel_abd_block_inverse_apply<<<blocks, kThreads>>>(
                        n,
                        (const Float*)diag_inv.data(),
                        (const Float*)jacobi_recip.data(),
                        block_inverse_mix,
                        (const Float*)info.r().data(),
                        (Float*)info.z().data(),
                        (const IndexT*)converged.data());
                }
                else
                {
                    int blocks = (n + 255) / 256;
                    kernel_abd_jacobi_apply<<<blocks, 256>>>(
                        n,
                        (const Float*)jacobi_recip.data(),
                        (const Float*)info.r().data(),
                        (Float*)info.z().data(),
                        (const IndexT*)converged.data());
                }
                checkCudaErrors(cudaGetLastError());
                if(corex_abd_precond_sync_enabled())
                    checkCudaErrors(cudaDeviceSynchronize());
            }
        }
    }
};

REGISTER_SIM_SYSTEM(ABDDiagPreconditioner);
}  // namespace uipc::backend::cuda
#else
#include <linear_system/local_preconditioner.h>
#include <affine_body/affine_body_dynamics.h>
#include <affine_body/abd_linear_subsystem.h>
#include <linear_system/global_linear_system.h>
#include <muda/ext/eigen/inverse.h>
#include <kernel_cout.h>

namespace uipc::backend::cuda
{
class ABDDiagPreconditioner final : public LocalPreconditioner
{
  public:
    using LocalPreconditioner::LocalPreconditioner;

    ABDLinearSubsystem* abd_linear_subsystem = nullptr;

    muda::DeviceBuffer<Matrix12x12> diag_inv;

    virtual void do_build(BuildInfo& info) override
    {
        auto& global_linear_system = require<GlobalLinearSystem>();
        abd_linear_subsystem       = &require<ABDLinearSubsystem>();

        info.connect(abd_linear_subsystem);
    }

    virtual void do_init(InitInfo& info) override {}

    virtual void do_assemble(GlobalLinearSystem::LocalPreconditionerAssemblyInfo& info) override
    {
        using namespace muda;

        auto diag_hessian = abd_linear_subsystem->diag_hessian();
        diag_inv.resize(diag_hessian.size());

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(diag_inv.size(),
                   [diag_hessian = diag_hessian.viewer().name("diag_hessian"),
                    diag_inv = diag_inv.viewer().name("diag_inv")] __device__(int i) mutable
                   { diag_inv(i) = muda::eigen::inverse(diag_hessian(i)); });
    }

    virtual void do_apply(GlobalLinearSystem::ApplyPreconditionerInfo& info) override
    {
        using namespace muda;
        auto converged = info.converged();

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(diag_inv.size(),
                   [r = info.r().viewer().name("r"),
                    z = info.z().viewer().name("z"),
                    converged = converged.cviewer().name("converged"),
                    diag_inv = diag_inv.viewer().name("diag_inv")] __device__(int i) mutable
                   {
                       if(*converged != 0)
                           return;
                       z.segment<12>(i * 12).as_eigen() =
                           diag_inv(i) * r.segment<12>(i * 12).as_eigen();
                   });
    }
};

REGISTER_SIM_SYSTEM(ABDDiagPreconditioner);
}  // namespace uipc::backend::cuda
#endif
