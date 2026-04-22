#include <linear_system/local_preconditioner.h>
#include <affine_body/affine_body_dynamics.h>
#include <affine_body/abd_linear_subsystem.h>
#include <linear_system/global_linear_system.h>
#include <muda/ext/eigen/inverse.h>
#include <kernel_cout.h>
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
#include <muda/check/check_cuda_errors.h>
#endif

namespace uipc::backend::cuda
{
namespace
{
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
// Jacobi (diagonal-only) preconditioner for CoreX: z_k = r_k / H_{kk}.
// Full block-inverse suffers catastrophic cancellation at CoreX's float-level
// double precision when off-diagonal values are close to diagonal values.
__global__ void kernel_abd_jacobi_extract(int n, const Float* diag_hessian, Float* diag_recip)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    for(int k = 0; k < 12; ++k)
    {
        Float d = diag_hessian[i * 144 + k * 12 + k]; // column-major: element (k,k)
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

#else
__global__ void kernel_abd_diag_inverse(int n, const Matrix12x12* diag_hessian, Matrix12x12* diag_inv)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;

    diag_inv[i] = muda::eigen::inverse(diag_hessian[i]);
}

__global__ void kernel_abd_apply_diag_inverse(
    int n,
    const Matrix12x12* diag_inv,
    const Float* r,
    Float* z,
    const IndexT* converged)
{
    if(*converged != 0)
        return;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;

    Eigen::Map<const Eigen::Vector<Float, 12>> ri(r + i * 12);
    Eigen::Map<Eigen::Vector<Float, 12>>       zi(z + i * 12);
    zi = diag_inv[i] * ri;
}
#endif
}  // namespace

class ABDDiagPreconditioner final : public LocalPreconditioner
{
  public:
    using LocalPreconditioner::LocalPreconditioner;

    ABDLinearSubsystem* abd_linear_subsystem = nullptr;

    muda::DeviceBuffer<Matrix12x12> diag_inv;
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
    muda::DeviceBuffer<Float> jacobi_recip; // 12 reciprocals per body
#endif

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

#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
        {
            auto n = static_cast<int>(diag_hessian.size());
            if(n > 0)
            {
                jacobi_recip.resize(n * 12);
                int blocks = (n + 255) / 256;
                kernel_abd_jacobi_extract<<<blocks, 256>>>(
                    n,
                    (const Float*)diag_hessian.data(),
                    (Float*)jacobi_recip.data());
                checkCudaErrors(cudaDeviceSynchronize());
            }
        }
#else
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(diag_inv.size(),
                   [diag_hessian = diag_hessian.viewer().name("diag_hessian"),
                    diag_inv = diag_inv.viewer().name("diag_inv")] __device__(int i) mutable
                   { diag_inv(i) = muda::eigen::inverse(diag_hessian(i)); });
#endif
    }

    virtual void do_apply(GlobalLinearSystem::ApplyPreconditionerInfo& info) override
    {
        using namespace muda;
        auto converged = info.converged();

#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
        {
            auto n = static_cast<int>(jacobi_recip.size() / 12);
            if(n > 0)
            {
                int blocks = (n + 255) / 256;
                kernel_abd_jacobi_apply<<<blocks, 256>>>(
                    n,
                    (const Float*)jacobi_recip.data(),
                    (const Float*)info.r().data(),
                    (Float*)info.z().data(),
                    (const IndexT*)converged.data());
                checkCudaErrors(cudaDeviceSynchronize());
            }
        }
#else
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
#endif
    }
};

REGISTER_SIM_SYSTEM(ABDDiagPreconditioner);
}  // namespace uipc::backend::cuda
