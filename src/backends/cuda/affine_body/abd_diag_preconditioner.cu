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
    const char* env = std::getenv("UIPC_COREX_ABD_PRECOND_DIAG_CLAMP");
    if(!env || env[0] == '\0')
        return false;

    char* end = nullptr;
    double min_v = std::strtod(env, &end);
    if(end == env || min_v < 0)
        return false;

    double max_v = 0.0;
    if(*end == ',')
    {
        const char* max_start = end + 1;
        max_v = std::strtod(max_start, &end);
        if(end == max_start || max_v <= 0)
            max_v = 0.0;
    }

    min_abs_diag = static_cast<Float>(min_v);
    max_abs_diag = max_v > 0.0 ? static_cast<Float>(max_v) :
                                 std::numeric_limits<Float>::max();
    return true;
}

bool struct_block_precond_enabled()
{
    const char* env = std::getenv("UIPC_COREX_ABD_PRECOND_STRUCT_BLOCK");
    if(!env) return false;
    return env[0] != '\0' && env[0] != '0';
}

bool struct_block_precond_stats_enabled()
{
    const char* env = std::getenv("UIPC_COREX_ABD_PRECOND_STRUCT_BLOCK_STATS");
    if(!env) return false;
    return env[0] != '\0' && env[0] != '0';
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
// - `UIPC_COREX_ABD_PRECOND_STRUCT_BLOCK=1` (legacy) takes precedence over
//   both, since it allocates and uses different buffers.
bool block_inverse_precond_enabled()
{
    if(std::getenv("UIPC_COREX_ABD_PRECOND_DIAG_JACOBI"))
        return false;
    const char* env = std::getenv("UIPC_COREX_ABD_PRECOND_BLOCK_INVERSE");
    if(!env) return true;
    return env[0] != '\0' && env[0] != '0';
}

bool block_inverse_precond_stats_enabled()
{
    const char* env = std::getenv("UIPC_COREX_ABD_PRECOND_BLOCK_INVERSE_STATS");
    if(!env) return false;
    return env[0] != '\0' && env[0] != '0';
}

__device__ inline Float corex_abs(Float v)
{
    return v < 0 ? -v : v;
}

__device__ bool invert_spd_safe_3x3(const Float* A, Float* inv)
{
    constexpr Float eps = static_cast<Float>(1e-10);
    for(int r = 0; r < 3; ++r)
    {
        Float off = static_cast<Float>(0);
        for(int c = 0; c < 3; ++c)
            if(c != r)
                off += corex_abs(A[r * 3 + c]);
        if(A[r * 3 + r] <= eps || A[r * 3 + r] <= off)
            return false;
    }

    Float a = A[0], b = A[1], c = A[2];
    Float d = A[3], e = A[4], f = A[5];
    Float g = A[6], h = A[7], i = A[8];
    Float det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if(det <= eps)
        return false;
    Float inv_det = static_cast<Float>(1) / det;
    inv[0] = (e * i - f * h) * inv_det;
    inv[1] = (c * h - b * i) * inv_det;
    inv[2] = (b * f - c * e) * inv_det;
    inv[3] = (f * g - d * i) * inv_det;
    inv[4] = (a * i - c * g) * inv_det;
    inv[5] = (c * d - a * f) * inv_det;
    inv[6] = (d * h - e * g) * inv_det;
    inv[7] = (b * g - a * h) * inv_det;
    inv[8] = (a * e - b * d) * inv_det;
    return true;
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
__global__ void kernel_abd_block_inverse_apply(
    int n, const Float* diag_inv, const Float* r, Float* z, const IndexT* converged)
{
    if(*converged != 0) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    constexpr int N = 12;
    Float ri[N];
    for(int k = 0; k < N; ++k)
        ri[k] = r[i * N + k];

    for(int row = 0; row < N; ++row)
    {
        Float s = static_cast<Float>(0);
        for(int col = 0; col < N; ++col)
            s += diag_inv[i * 144 + col * N + row] * ri[col];
        z[i * N + row] = s;
    }
}

__global__ void kernel_abd_precond_extract(int          n,
                                           const Float* diag_hessian,
                                           Float*       diag_recip,
                                           Float        min_abs_diag,
                                           Float        max_abs_diag,
                                           int          enable_struct_block,
                                           Float*       block_inv,
                                           int*         block_mask)
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

    if(!enable_struct_block)
        return;

    const int block_offsets[4] = {0, 3, 6, 9};
    for(int b = 0; b < 4; ++b)
    {
        int offset = block_offsets[b];
        Float A[9];
        for(int r = 0; r < 3; ++r)
            for(int c = 0; c < 3; ++c)
                A[r * 3 + c] = diag_hessian[i * 144 + (offset + c) * 12 + offset + r];

        Float inv[9];
        bool ok = invert_spd_safe_3x3(A, inv);
        block_mask[i * 4 + b] = ok ? 1 : 0;
        for(int k = 0; k < 9; ++k)
            block_inv[(i * 4 + b) * 9 + k] = ok ? inv[k] : static_cast<Float>(0);
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

__global__ void kernel_abd_struct_block_apply(int n,
                                              const Float* diag_recip,
                                              const Float* block_inv,
                                              const int*   block_mask,
                                              const Float* r,
                                              Float*       z,
                                              const IndexT* converged)
{
    if(*converged != 0) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    const int block_offsets[4] = {0, 3, 6, 9};
    for(int b = 0; b < 4; ++b)
    {
        int offset = block_offsets[b];
        if(block_mask[i * 4 + b])
        {
            const Float* inv = block_inv + (i * 4 + b) * 9;
            for(int row = 0; row < 3; ++row)
            {
                Float sum = static_cast<Float>(0);
                for(int col = 0; col < 3; ++col)
                    sum += inv[row * 3 + col] * r[i * 12 + offset + col];
                z[i * 12 + offset + row] = sum;
            }
        }
        else
        {
            for(int k = 0; k < 3; ++k)
                z[i * 12 + offset + k] =
                    diag_recip[i * 12 + offset + k] * r[i * 12 + offset + k];
        }
    }
}

}  // namespace

class ABDDiagPreconditioner final : public LocalPreconditioner
{
  public:
    using LocalPreconditioner::LocalPreconditioner;

    ABDLinearSubsystem* abd_linear_subsystem = nullptr;

    muda::DeviceBuffer<Matrix12x12> diag_inv;
    muda::DeviceBuffer<Float> jacobi_recip; // 12 reciprocals per body
    muda::DeviceBuffer<Float> struct_block_inv; // 4 conservative 3x3 inverses per body
    muda::DeviceBuffer<int> struct_block_mask;
    bool struct_block_enabled = false;
    muda::DeviceBuffer<int> block_inv_status; // 1 = LDLT accepted, 0 = Jacobi fallback
    bool block_inverse_enabled = false;

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

        if(std::getenv("UIPC_COREX_TRACE_LINEAR_SYSTEM"))
            logger::info("[corex_trace][precond] do_assemble: entry");

        auto diag_hessian = abd_linear_subsystem->diag_hessian();

        if(std::getenv("UIPC_COREX_TRACE_LINEAR_SYSTEM"))
            logger::info("[corex_trace][precond] do_assemble: diag_hessian.size()={}, data()={}",
                         diag_hessian.size(), (void*)diag_hessian.data());

        diag_inv.resize(diag_hessian.size());

        if(std::getenv("UIPC_COREX_TRACE_LINEAR_SYSTEM"))
            logger::info("[corex_trace][precond] do_assemble: diag_inv resized to {}", diag_inv.size());

        {
            auto n = static_cast<int>(diag_hessian.size());
            if(n > 0)
            {
                jacobi_recip.resize(n * 12);
                Float min_abs_diag = 0;
                Float max_abs_diag = std::numeric_limits<Float>::max();
                bool  clamp_enabled = parse_precond_diag_clamp(min_abs_diag, max_abs_diag);
                struct_block_enabled = struct_block_precond_enabled();
                block_inverse_enabled =
                    !struct_block_enabled && block_inverse_precond_enabled();
                int blocks = (n + 255) / 256;
                if(struct_block_enabled)
                {
                    struct_block_inv.resize(n * 4 * 9);
                    struct_block_mask.resize(n * 4);
                }
                else
                {
                    struct_block_inv.resize(0);
                    struct_block_mask.resize(0);
                }
                if(block_inverse_enabled)
                {
                    diag_inv.resize(n);
                    block_inv_status.resize(n);
                    // Smaller block size: each thread holds a 12x12 working matrix
                    // (>= 144 doubles) plus LDLT temporaries. Keeping the block at
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
                        max_abs_diag,
                        struct_block_enabled ? 1 : 0,
                        struct_block_enabled ? (Float*)struct_block_inv.data() : nullptr,
                        struct_block_enabled ? (int*)struct_block_mask.data() : nullptr);
                }
                checkCudaErrors(cudaGetLastError());
                if(std::getenv("UIPC_COREX_ABD_PRECOND_SKIP_SYNC") == nullptr)
                    checkCudaErrors(cudaDeviceSynchronize());
                if(block_inverse_enabled
                   && (block_inverse_precond_stats_enabled()
                       || std::getenv("UIPC_COREX_TRACE_LINEAR_SYSTEM")))
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
                if(clamp_enabled || std::getenv("UIPC_COREX_TRACE_LINEAR_SYSTEM"))
                    logger::info("[corex_precond_diag] bodies={} clamp={} min_abs_diag={} max_abs_diag={}",
                                 n,
                                 clamp_enabled ? 1 : 0,
                                 min_abs_diag,
                                 max_abs_diag);
                if(struct_block_enabled
                   && (struct_block_precond_stats_enabled()
                       || std::getenv("UIPC_COREX_TRACE_LINEAR_SYSTEM")))
                {
                    std::vector<int> h_mask(n * 4);
                    cudaMemcpy(h_mask.data(),
                               struct_block_mask.data(),
                               sizeof(int) * h_mask.size(),
                               cudaMemcpyDeviceToHost);
                    SizeT accepted[4] = {0, 0, 0, 0};
                    for(int i = 0; i < n; ++i)
                        for(int b = 0; b < 4; ++b)
                            accepted[b] += h_mask[i * 4 + b] ? 1 : 0;
                    logger::info("[corex_precond_struct_block] bodies={} accepted_t={} accepted_a0={} accepted_a1={} accepted_a2={} total_blocks={}",
                                 n,
                                 accepted[0],
                                 accepted[1],
                                 accepted[2],
                                 accepted[3],
                                 n * 4);
                }
                if(std::getenv("UIPC_COREX_ABD_PRECOND_DIAG_STATS")
                   || std::getenv("UIPC_COREX_PCG_DIAG"))
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
                int blocks = (n + 255) / 256;
                if(block_inverse_enabled && diag_inv.size() > 0)
                {
                    kernel_abd_block_inverse_apply<<<blocks, 256>>>(
                        n,
                        (const Float*)diag_inv.data(),
                        (const Float*)info.r().data(),
                        (Float*)info.z().data(),
                        (const IndexT*)converged.data());
                }
                else if(struct_block_enabled && struct_block_inv.size() > 0)
                {
                    kernel_abd_struct_block_apply<<<blocks, 256>>>(
                        n,
                        (const Float*)jacobi_recip.data(),
                        (const Float*)struct_block_inv.data(),
                        (const int*)struct_block_mask.data(),
                        (const Float*)info.r().data(),
                        (Float*)info.z().data(),
                        (const IndexT*)converged.data());
                }
                else
                {
                    kernel_abd_jacobi_apply<<<blocks, 256>>>(
                        n,
                        (const Float*)jacobi_recip.data(),
                        (const Float*)info.r().data(),
                        (Float*)info.z().data(),
                        (const IndexT*)converged.data());
                }
                checkCudaErrors(cudaGetLastError());
                if(std::getenv("UIPC_COREX_ABD_PRECOND_SKIP_SYNC") == nullptr)
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
