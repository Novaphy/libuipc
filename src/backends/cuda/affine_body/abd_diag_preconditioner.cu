#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
#include <linear_system/local_preconditioner.h>
#include <affine_body/affine_body_dynamics.h>
#include <affine_body/abd_linear_subsystem.h>
#include <linear_system/global_linear_system.h>
#include <muda/ext/eigen/inverse.h>
#include <kernel_cout.h>
#include <muda/check/check_cuda_errors.h>
#include <cub/block/block_reduce.cuh>
#include <cstdlib>
#include <limits>
#include <vector>
#include <algorithm>

namespace uipc::backend::cuda
{
namespace
{
// A damped block/Jacobi blend keeps cross-DoF preconditioning while remaining
// stable on dense single-precision ABD contact frames.
bool block_inverse_precond_stats_enabled()
{
    static const bool enabled = []
    {
        const char* env = std::getenv("UIPC_COREX_ABD_PRECOND_STATS");
        return env && env[0] != '\0' && env[0] != '0';
    }();
    return enabled;
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
    static const bool enabled = std::getenv("UIPC_COREX_TRACE_LINEAR_SYSTEM") != nullptr;
    return enabled;
}

bool corex_abd_precond_diag_stats_enabled()
{
    static const bool enabled =
        std::getenv("UIPC_COREX_ABD_PRECOND_DIAG_STATS") != nullptr
        || std::getenv("UIPC_COREX_PCG_DIAG") != nullptr;
    return enabled;
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
        if(ad > max_diag_abs)
            max_diag_abs = ad;
    }

    Float eps = max_diag_abs * static_cast<Float>(1e-10);
    if(eps < static_cast<Float>(1e-30))
        eps = static_cast<Float>(1e-30);

    for(int k = 0; k < N; ++k)
    {
        Float Dk = A[k * N + k];
        for(int j = 0; j < k; ++j)
        {
            Float Lkj = A[j * N + k];
            Float Dj  = A[j * N + j];
            Dk -= Lkj * Lkj * Dj;
        }
        if(Dk < eps)
            return false;
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
    int n, const Float* diag_hessian, Float* diag_inv, Float* diag_recip, int* block_status)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;

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
        diag_recip[i * N + k] = (d != static_cast<Float>(0)) ?
                                    (static_cast<Float>(1) / d) :
                                    static_cast<Float>(0);
    }

    bool ok = corex_ldlt_factorize_inplace<N>(A);

    if(ok)
    {
        Float invA[N * N];
        corex_ldlt_explicit_inverse<N>(A, invA);
        for(int idx = 0; idx < N * N; ++idx)
            diag_inv[i * 144 + idx] = invA[idx];
        if(block_status)
            block_status[i] = 1;
    }
    else
    {
        for(int idx = 0; idx < N * N; ++idx)
            diag_inv[i * 144 + idx] = static_cast<Float>(0);
        for(int k = 0; k < N; ++k)
            diag_inv[i * 144 + k * N + k] = diag_recip[i * N + k];
        if(block_status)
            block_status[i] = 0;
    }
}

// Apply z = A^{-1} * r as an unconditional 12x12 mat-vec per body. This is
// the runtime-hot kernel that runs once per PCG iteration, so it must stay
// branch-free (the SPD pivot fallback is folded into `diag_inv` at extract
// time so the apply path is identical for accepted and fallback bodies).
__global__ void kernel_abd_block_inverse_apply(int           n,
                                               const Float*  diag_inv,
                                               const Float*  diag_recip,
                                               const Float*  r,
                                               Float*        z,
                                               const IndexT* converged,
                                               Float*        dot)
{
    if(*converged != 0)
        return;
    constexpr int N = 12;
    constexpr int LanesPerBody = 16;
    constexpr int BodiesPerBlock = 4;
    constexpr int ThreadsPerBlock = LanesPerBody * BodiesPerBlock;

    int local_body = threadIdx.x / LanesPerBody;
    int row        = threadIdx.x - local_body * LanesPerBody;
    int i          = blockIdx.x * BodiesPerBlock + local_body;
    const bool active = i < n && row < N;

    Float r_lane = active ? r[i * N + row] : static_cast<Float>(0);
    Float dot_term = 0;

    if(active)
    {
        Float s = static_cast<Float>(0);
        for(int col = 0; col < N; ++col)
        {
            Float r_col = __shfl_sync(0xffffffffu, r_lane, col, LanesPerBody);
            s += diag_inv[i * 144 + col * N + row] * r_col;
        }
        constexpr Float block_mix = Float{0.99};
        const Float     j = diag_recip[i * N + row] * r_lane;
        s = block_mix * s + (static_cast<Float>(1) - block_mix) * j;
        z[i * N + row] = s;
        dot_term = r_lane * s;
    }

    if(dot)
    {
        using BlockReduce = cub::BlockReduce<Float, ThreadsPerBlock>;
        __shared__ typename BlockReduce::TempStorage storage;
        Float block_dot = BlockReduce(storage).Sum(dot_term);
        if(threadIdx.x == 0)
            muda::atomic_add(dot, block_dot);
    }
}

}  // namespace

class ABDDiagPreconditioner final : public LocalPreconditioner
{
  public:
    using LocalPreconditioner::LocalPreconditioner;

    ABDLinearSubsystem* abd_linear_subsystem = nullptr;

    muda::DeviceBuffer<Matrix12x12> diag_inv;
    muda::DeviceBuffer<Float>       jacobi_recip;  // 12 reciprocals per body
    muda::DeviceBuffer<int>         block_inv_status;

    virtual void do_build(BuildInfo& info) override
    {
        auto& global_linear_system = require<GlobalLinearSystem>();
        abd_linear_subsystem       = &require<ABDLinearSubsystem>();

        info.connect(abd_linear_subsystem);
    }

    virtual void do_init(InitInfo& info) override {}

    virtual bool do_supports_apply_dot() const override { return true; }

    virtual void do_assemble(GlobalLinearSystem::LocalPreconditionerAssemblyInfo& info) override
    {
        using namespace muda;

        if(corex_abd_trace_linear_system_enabled())
            logger::info("[corex_trace][precond] do_assemble: entry");

        auto diag_hessian = abd_linear_subsystem->diag_hessian();

        if(corex_abd_trace_linear_system_enabled())
            logger::info("[corex_trace][precond] do_assemble: diag_hessian.size()={}, data()={}",
                         diag_hessian.size(),
                         (void*)diag_hessian.data());

        auto n = static_cast<int>(diag_hessian.size());
        diag_inv.resize(n);
        jacobi_recip.resize(n * 12);
        block_inv_status.resize(n);

        if(corex_abd_trace_linear_system_enabled())
            logger::info("[corex_trace][precond] do_assemble: diag_inv resized to {}",
                         diag_inv.size());

        if(n > 0)
        {
            // Each thread holds a 12x12 working matrix plus LDLT temporaries.
            // Four warps per block limit local-memory pressure on CoreX.
            constexpr int kBlk        = 64;
            int           ldlt_blocks = (n + kBlk - 1) / kBlk;
            kernel_abd_precond_extract_block_inverse<<<ldlt_blocks, kBlk>>>(
                n,
                (const Float*)diag_hessian.data(),
                (Float*)diag_inv.data(),
                (Float*)jacobi_recip.data(),
                (int*)block_inv_status.data());
            checkCudaErrors(cudaGetLastError());
            if(corex_abd_precond_sync_enabled())
                checkCudaErrors(cudaDeviceSynchronize());

            if(block_inverse_precond_stats_enabled()
               || corex_abd_trace_linear_system_enabled())
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
            if(corex_abd_precond_diag_stats_enabled())
            {
                std::vector<Matrix12x12> h_diag(n);
                cudaMemcpy(h_diag.data(),
                           diag_hessian.data(),
                           sizeof(Matrix12x12) * n,
                           cudaMemcpyDeviceToHost);
                Float min_nonzero_abs = std::numeric_limits<Float>::max();
                Float max_abs         = 0;
                SizeT zero_count      = 0;
                SizeT tiny_count      = 0;
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
                        max_abs         = std::max(max_abs, a);
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

    virtual void do_apply(GlobalLinearSystem::ApplyPreconditionerInfo& info) override
    {
        using namespace muda;
        auto converged = info.converged();
        Float* dot = info.compute_dot() ? info.dot().data() : nullptr;

        auto n = static_cast<int>(jacobi_recip.size() / 12);
        if(n == 0)
            return;

        constexpr int kBodiesPerBlock = 4;
        constexpr int kThreads        = kBodiesPerBlock * 16;
        int blocks = (n + kBodiesPerBlock - 1) / kBodiesPerBlock;
        kernel_abd_block_inverse_apply<<<blocks, kThreads>>>(
            n,
            (const Float*)diag_inv.data(),
            (const Float*)jacobi_recip.data(),
            (const Float*)info.r().data(),
            (Float*)info.z().data(),
            (const IndexT*)converged.data(),
            dot);
        checkCudaErrors(cudaGetLastError());
        if(corex_abd_precond_sync_enabled())
            checkCudaErrors(cudaDeviceSynchronize());
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
