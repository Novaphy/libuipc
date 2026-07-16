#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
#include <affine_body/abd_linear_subsystem.h>
#include <sim_engine.h>
#include <kernel_cout.h>
#include <muda/ext/eigen.h>
#include <utils/matrix_assembler.h>
#include <utils/matrix_unpacker.h>
#include <uipc/builtin/attribute_name.h>
#include <affine_body/inter_affine_body_constitution_manager.h>
#include <affine_body/abd_linear_subsystem_reporter.h>
#include <affine_body/affine_body_kinetic.h>
#include <affine_body/affine_body_constitution.h>
#include <affine_body/abd_jacobi_matrix_corex.h>
#include <utils/report_extent_check.h>
#include <utils/corex_phase_profile.h>
#include <cstdlib>
#include <vector>

namespace uipc::backend::cuda
{
UIPC_HOST UIPC_DEVICE void zero_out_lower(Matrix12x12& H)
{
    for(int bi = 0; bi < 4; ++bi)
    {
        for(int bj = 0; bj < bi; ++bj)
        {
            H.template block<3, 3>(bi * 3, bj * 3).setZero();
        }
    }
}
}  // namespace uipc::backend::cuda

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(ABDLinearSubsystem);

// ref: https://github.com/spiriMirror/libuipc/issues/271
constexpr U64 ABDLinearSubsystemUID = 0ull;

static bool corex_abd_assemble_sync_enabled()
{
    return std::getenv("UIPC_COREX_ABD_ASSEMBLE_ASYNC") == nullptr
           || std::getenv("UIPC_COREX_ABD_ASSEMBLE_SYNC") != nullptr
           || std::getenv("UIPC_COREX_TRACE_ABD_ASSEMBLE") != nullptr
           || std::getenv("UIPC_COREX_TRACE_LINEAR_SYSTEM") != nullptr;
}

static void corex_abd_assemble_sync_if_requested(const char* where)
{
    if(!corex_abd_assemble_sync_enabled())
        return;
    checkCudaErrors(cudaDeviceSynchronize());
    if(std::getenv("UIPC_COREX_TRACE_ABD_ASSEMBLE")
       || std::getenv("UIPC_COREX_TRACE_LINEAR_SYSTEM"))
        logger::info("[corex_trace][abd] sync ok at {}", where);
}

static bool corex_abd_dytopo_parallel_enabled()
{
    const char* serial = std::getenv("UIPC_COREX_ABD_DYTOPO_SERIAL");
    if(serial && serial[0] != '\0' && serial[0] != '0')
        return false;
    const char* env = std::getenv("UIPC_COREX_ABD_DYTOPO_PARALLEL");
    if(env && env[0] != '\0')
        return env[0] != '0';
    return true;
}

static __global__ void kernel_abd_assemble_gradients(int n,
                                                     const IndexT* is_fixed,
                                                     const IndexT* is_external_kinetic,
                                                     const Vector12* shape_gradient,
                                                     const Vector12* kinetic_gradient,
                                                     Float* gradients)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    auto base = i * 12;
    if(is_fixed[i])
    {
        for(int k = 0; k < 12; ++k)
            gradients[base + k] = 0.0;
        return;
    }

    const auto& shape = shape_gradient[i];
    if(!is_external_kinetic[i])
    {
        const auto& kin = kinetic_gradient[i];
        for(int k = 0; k < 12; ++k)
            gradients[base + k] = shape(k) + kin(k);
    }
    else
    {
        for(int k = 0; k < 12; ++k)
            gradients[base + k] = shape(k);
    }
}

static __global__ void kernel_abd_assemble_hessians(int n,
                                                    const IndexT* is_fixed,
                                                    const IndexT* is_external_kinetic,
                                                    const Matrix12x12* shape_hessian,
                                                    const Matrix12x12* kinetic_hessian,
                                                    Matrix12x12* diag_hessian,
                                                    int* dst_rows,
                                                    int* dst_cols,
                                                    Matrix3x3* dst_vals)
{
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    if(I >= n) return;

    const Float* shape_ptr = reinterpret_cast<const Float*>(shape_hessian + I);
    const Float* kin_ptr   = reinterpret_cast<const Float*>(kinetic_hessian + I);

    // Eigen stores column-major: element (r,c) at col*12+row
    // local_h also uses column-major: local_h[c*12+r] = H(r,c)
    Float local_h[12 * 12];
    for(int c = 0; c < 12; ++c)
    {
        for(int r = 0; r < 12; ++r)
        {
            int cm_idx = c * 12 + r;
            if(is_fixed[I])
                local_h[cm_idx] = (r == c) ? Float(1) : Float(0);
            else
                local_h[cm_idx] =
                    shape_ptr[cm_idx] + (!is_external_kinetic[I] ? kin_ptr[cm_idx] : Float(0));
        }
    }

    Float* diag_ptr = reinterpret_cast<Float*>(diag_hessian + I);
    for(int idx = 0; idx < 12 * 12; ++idx)
        diag_ptr[idx] = local_h[idx];

    // Match upstream semantics: keep only upper-triangle 3x3 blocks for diagonal body terms.
    for(int jj = 0; jj < 4; ++jj)
    {
        for(int ii = jj + 1; ii < 4; ++ii)
        {
            for(int c = 0; c < 3; ++c)
            {
                for(int r = 0; r < 3; ++r)
                {
                    int row = ii * 3 + r;
                    int col = jj * 3 + c;
                    local_h[col * 12 + row] = Float(0);
                }
            }
        }
    }

    constexpr int BLK = 3;
    int base_triplet = I * 16;
    for(int ii = 0; ii < 4; ++ii)
    {
        for(int jj = 0; jj < 4; ++jj)
        {
            int idx     = base_triplet + ii * 4 + jj;
            dst_rows[idx] = I * 4 + ii;
            dst_cols[idx] = I * 4 + jj;
            // dst_vals[idx] is column-major Matrix3x3: element (r,c) at c*3+r
            // local_h is column-major 12x12: element (row,col) at col*12+row
            // Block (ii,jj) row r, col c => global row = ii*3+r, global col = jj*3+c
            Float* block_ptr = reinterpret_cast<Float*>(dst_vals + idx);
            for(int r = 0; r < BLK; ++r)
                for(int c = 0; c < BLK; ++c)
                    block_ptr[c * BLK + r] = local_h[(jj * BLK + c) * 12 + (ii * BLK + r)];
        }
    }
}

static __device__ void corex_abd_mass_mul_values(const ABDJacobiDyadicMass& mass,
                                                 const Float*               p,
                                                 Float*                     ret)
{
    const Float      m = mass.mass();
    const Vector3&   x = mass.mass_times_x_bar();
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

static __device__ void corex_abd_add_mass_to_local_h(const ABDJacobiDyadicMass& mass,
                                                     Float*                     local_h)
{
    const Float      m = mass.mass();
    const Vector3&   x = mass.mass_times_x_bar();
    const Matrix3x3& D = mass.mass_times_dyadic_x_bar();

    local_h[0 * 12 + 0] += m;
    local_h[1 * 12 + 1] += m;
    local_h[2 * 12 + 2] += m;

    for(int k = 0; k < 3; ++k)
    {
        local_h[(3 + k) * 12 + 0] += x[k];
        local_h[0 * 12 + (3 + k)] += x[k];
        local_h[(6 + k) * 12 + 1] += x[k];
        local_h[1 * 12 + (6 + k)] += x[k];
        local_h[(9 + k) * 12 + 2] += x[k];
        local_h[2 * 12 + (9 + k)] += x[k];
    }

    for(int r = 0; r < 3; ++r)
    {
        for(int c = 0; c < 3; ++c)
        {
            local_h[(3 + c) * 12 + (3 + r)] += D(r, c);
            local_h[(6 + c) * 12 + (6 + r)] += D(r, c);
            local_h[(9 + c) * 12 + (9 + r)] += D(r, c);
        }
    }
}

static __global__ void kernel_abd_assemble_gradients_direct_kinetic(
    int n,
    const IndexT* is_fixed,
    const IndexT* is_external_kinetic,
    const Vector12* shape_gradient,
    const Vector12* qs,
    const Vector12* q_tildes,
    const ABDJacobiDyadicMass* masses,
    Float* gradients)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    int base = i * 12;
    if(is_fixed[i])
    {
        for(int k = 0; k < 12; ++k)
            gradients[base + k] = 0;
        return;
    }

    Float kin[12] = {};
    if(!is_external_kinetic[i])
    {
        Float dq[12];
        for(int k = 0; k < 12; ++k)
            dq[k] = qs[i](k) - q_tildes[i](k);
        corex_abd_mass_mul_values(masses[i], dq, kin);
    }

    const auto& shape = shape_gradient[i];
    for(int k = 0; k < 12; ++k)
        gradients[base + k] = shape(k) + kin[k];
}

static __global__ void kernel_abd_assemble_hessians_direct_kinetic(
    int n,
    const IndexT* is_fixed,
    const IndexT* is_external_kinetic,
    const Matrix12x12* shape_hessian,
    const ABDJacobiDyadicMass* masses,
    Matrix12x12* diag_hessian,
    int* dst_rows,
    int* dst_cols,
    Matrix3x3* dst_vals)
{
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    if(I >= n) return;

    const Float* shape_ptr = reinterpret_cast<const Float*>(shape_hessian + I);
    Float local_h[12 * 12];
    for(int c = 0; c < 12; ++c)
    {
        for(int r = 0; r < 12; ++r)
        {
            int cm_idx = c * 12 + r;
            if(is_fixed[I])
                local_h[cm_idx] = (r == c) ? Float(1) : Float(0);
            else
                local_h[cm_idx] = shape_ptr[cm_idx];
        }
    }

    if(!is_fixed[I] && !is_external_kinetic[I])
        corex_abd_add_mass_to_local_h(masses[I], local_h);

    Float* diag_ptr = reinterpret_cast<Float*>(diag_hessian + I);
    for(int idx = 0; idx < 12 * 12; ++idx)
        diag_ptr[idx] = local_h[idx];

    for(int jj = 0; jj < 4; ++jj)
        for(int ii = jj + 1; ii < 4; ++ii)
            for(int c = 0; c < 3; ++c)
                for(int r = 0; r < 3; ++r)
                    local_h[(jj * 3 + c) * 12 + (ii * 3 + r)] = Float(0);

    int base_triplet = I * 16;
    for(int ii = 0; ii < 4; ++ii)
    {
        for(int jj = 0; jj < 4; ++jj)
        {
            int idx       = base_triplet + ii * 4 + jj;
            dst_rows[idx] = I * 4 + ii;
            dst_cols[idx] = I * 4 + jj;
            Float* block_ptr = reinterpret_cast<Float*>(dst_vals + idx);
            for(int r = 0; r < 3; ++r)
                for(int c = 0; c < 3; ++c)
                    block_ptr[c * 3 + r] =
                        local_h[(jj * 3 + c) * 12 + (ii * 3 + r)];
        }
    }
}

static __global__ void kernel_abd_assemble_bdf1_direct(int n,
                                                       const IndexT* is_fixed,
                                                       const IndexT* is_external_kinetic,
                                                       const Vector12* qs,
                                                       const Vector12* q_tildes,
                                                       const ABDJacobiDyadicMass* masses,
                                                       const Vector12* shape_gradient,
                                                       const Matrix12x12* shape_hessian,
                                                       Float* gradients,
                                                       Matrix12x12* diag_hessian,
                                                       int* dst_rows,
                                                       int* dst_cols,
                                                       Matrix3x3* dst_vals)
{
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    if(I >= n) return;

    const bool fixed            = is_fixed[I] != 0;
    const bool external_kinetic = is_external_kinetic[I] != 0;
    const int  grad_base        = I * 12;

    Float dq[12];
    Float kinetic_g[12];
    for(int k = 0; k < 12; ++k)
    {
        dq[k]        = qs[I](k) - q_tildes[I](k);
        kinetic_g[k] = 0;
    }

    if(!fixed && !external_kinetic)
        corex_abd_mass_mul_values(masses[I], dq, kinetic_g);

    const Vector12& shape_g = shape_gradient[I];
    for(int k = 0; k < 12; ++k)
        gradients[grad_base + k] = fixed ? Float(0) : shape_g(k) + kinetic_g[k];

    Float        local_h[12 * 12];
    const Float* shape_ptr = reinterpret_cast<const Float*>(shape_hessian + I);
    for(int c = 0; c < 12; ++c)
    {
        for(int r = 0; r < 12; ++r)
        {
            int cm_idx = c * 12 + r;
            if(fixed)
                local_h[cm_idx] = (r == c) ? Float(1) : Float(0);
            else
                local_h[cm_idx] = shape_ptr[cm_idx];
        }
    }

    if(!fixed && !external_kinetic)
        corex_abd_add_mass_to_local_h(masses[I], local_h);

    Float* diag_ptr = reinterpret_cast<Float*>(diag_hessian + I);
    for(int idx = 0; idx < 12 * 12; ++idx)
        diag_ptr[idx] = local_h[idx];

    for(int jj = 0; jj < 4; ++jj)
    {
        for(int ii = jj + 1; ii < 4; ++ii)
        {
            for(int c = 0; c < 3; ++c)
            {
                for(int r = 0; r < 3; ++r)
                {
                    int row = ii * 3 + r;
                    int col = jj * 3 + c;
                    local_h[col * 12 + row] = Float(0);
                }
            }
        }
    }

    constexpr int BLK = 3;
    int           base_triplet = I * 16;
    for(int ii = 0; ii < 4; ++ii)
    {
        for(int jj = 0; jj < 4; ++jj)
        {
            int idx = base_triplet + ii * 4 + jj;
            dst_rows[idx] = I * 4 + ii;
            dst_cols[idx] = I * 4 + jj;
            Float* block_ptr = reinterpret_cast<Float*>(dst_vals + idx);
            for(int r = 0; r < BLK; ++r)
                for(int c = 0; c < BLK; ++c)
                    block_ptr[c * BLK + r] =
                        local_h[(jj * BLK + c) * 12 + (ii * BLK + r)];
        }
    }
}

static __global__ void kernel_abd_dytopo_gradients_serial(int grad_count,
                                                          int vertex_offset,
                                                          const int* grad_indices,
                                                          const Vector3* grad_values,
                                                          const IndexT* v2b,
                                                          const ABDJacobi* Js,
                                                          const IndexT* is_fixed,
                                                          Float* gradients)
{
    if(blockIdx.x != 0 || threadIdx.x != 0) return;

    for(int I = 0; I < grad_count; ++I)
    {
        int g_i    = grad_indices[I];
        int i      = g_i - vertex_offset;
        int body_i = v2b[i];
        if(is_fixed[body_i])
            continue;

        Vector12 G12 = Js[i].T() * grad_values[I];
        int base      = body_i * 12;
        for(int d = 0; d < 12; ++d)
            gradients[base + d] += G12(d);
    }
}

static __global__ void kernel_abd_dytopo_gradients_parallel(int grad_count,
                                                            int vertex_offset,
                                                            const int* grad_indices,
                                                            const Vector3* grad_values,
                                                            const IndexT* v2b,
                                                            const ABDJacobi* Js,
                                                            const IndexT* is_fixed,
                                                            Float* gradients)
{
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    if(I >= grad_count) return;

    int g_i    = grad_indices[I];
    int i      = g_i - vertex_offset;
    int body_i = v2b[i];
    if(is_fixed[body_i])
        return;

    Vector12 G12 = Js[i].T() * grad_values[I];
    int      base = body_i * 12;
    for(int d = 0; d < 12; ++d)
        atomicAdd(&gradients[base + d], G12(d));
}

static __global__ void kernel_abd_dytopo_hessians_serial(int hess_count,
                                                         int vertex_offset,
                                                         const int* row_indices,
                                                         const int* col_indices,
                                                         const Matrix3x3* values,
                                                         const IndexT* v2b,
                                                         const ABDJacobi* Js,
                                                         const IndexT* is_fixed,
                                                         Matrix12x12* diag_hessian,
                                                         int* dst_rows,
                                                         int* dst_cols,
                                                         Matrix3x3* dst_vals)
{
    if(blockIdx.x != 0 || threadIdx.x != 0) return;

    constexpr int BLK = 3;
    for(int I = 0; I < hess_count; ++I)
    {
        int g_i = row_indices[I];
        int g_j = col_indices[I];
        int i   = g_i - vertex_offset;
        int j   = g_j - vertex_offset;

        int body_i = v2b[i];
        int body_j = v2b[j];

        const auto& J_i  = Js[i];
        const auto& J_j  = Js[j];
        const auto& H3x3 = values[I];

        Matrix12x12 H12x12;
        IndexT      L = body_i;
        IndexT      R = body_j;
        if(body_i > body_j)
        {
            L = body_j;
            R = body_i;
        }

        if(is_fixed[body_i] || is_fixed[body_j])
        {
            H12x12.setZero();
        }
        else if(body_i < body_j)
        {
            H12x12 = ABDJacobi::JT_H_J(J_i.T(), H3x3, J_j);
        }
        else if(body_i > body_j)
        {
            H12x12 = ABDJacobi::JT_H_J(J_j.T(), H3x3.transpose(), J_i);
        }
        else
        {
            if(i != j)
            {
                H12x12 = ABDJacobi::JT_H_J(J_i.T(), H3x3, J_j)
                       + ABDJacobi::JT_H_J(J_j.T(), H3x3.transpose(), J_i);
            }
            else
            {
                H12x12 = ABDJacobi::JT_H_J(J_i.T(), H3x3, J_j);
            }

            Float* diag_ptr  = reinterpret_cast<Float*>(diag_hessian + body_i);
            const Float* src = reinterpret_cast<const Float*>(&H12x12);
            for(int k = 0; k < 12 * 12; ++k)
                diag_ptr[k] += src[k];

        }

        // Match upstream BCOO semantics: same-body terms write upper-triangle blocks only.
        if(body_i == body_j)
            zero_out_lower(H12x12);

        if(dst_rows && dst_cols && dst_vals)
        {
            int base_triplet = I * 16;
            for(int ii = 0; ii < 4; ++ii)
            {
                for(int jj = 0; jj < 4; ++jj)
                {
                    int idx       = base_triplet + ii * 4 + jj;
                    dst_rows[idx] = L * 4 + ii;
                    dst_cols[idx] = R * 4 + jj;
                    dst_vals[idx] =
                        H12x12.template block<BLK, BLK>(ii * BLK, jj * BLK);
                }
            }
        }
    }
}

static __global__ void kernel_abd_dytopo_hessians_parallel(int hess_count,
                                                           int vertex_offset,
                                                           const int* row_indices,
                                                           const int* col_indices,
                                                           const Matrix3x3* values,
                                                           const IndexT* v2b,
                                                           const ABDJacobi* Js,
                                                           const IndexT* is_fixed,
                                                           Matrix12x12* diag_hessian,
                                                           int* dst_rows,
                                                           int* dst_cols,
                                                           Matrix3x3* dst_vals)
{
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    if(I >= hess_count) return;

    constexpr int BLK = 3;
    int           g_i = row_indices[I];
    int           g_j = col_indices[I];
    int           i   = g_i - vertex_offset;
    int           j   = g_j - vertex_offset;

    int body_i = v2b[i];
    int body_j = v2b[j];

    const auto& J_i  = Js[i];
    const auto& J_j  = Js[j];
    const auto& H3x3 = values[I];

    Matrix12x12 H12x12;
    IndexT      L = body_i;
    IndexT      R = body_j;
    if(body_i > body_j)
    {
        L = body_j;
        R = body_i;
    }

    if(is_fixed[body_i] || is_fixed[body_j])
    {
        H12x12.setZero();
    }
    else if(body_i < body_j)
    {
        H12x12 = ABDJacobi::JT_H_J(J_i.T(), H3x3, J_j);
    }
    else if(body_i > body_j)
    {
        H12x12 = ABDJacobi::JT_H_J(J_j.T(), H3x3.transpose(), J_i);
    }
    else
    {
        if(i != j)
        {
            H12x12 = ABDJacobi::JT_H_J(J_i.T(), H3x3, J_j)
                   + ABDJacobi::JT_H_J(J_j.T(), H3x3.transpose(), J_i);
        }
        else
        {
            H12x12 = ABDJacobi::JT_H_J(J_i.T(), H3x3, J_j);
        }

        Float*      diag_ptr = reinterpret_cast<Float*>(diag_hessian + body_i);
        const Float* src     = reinterpret_cast<const Float*>(&H12x12);
        for(int k = 0; k < 12 * 12; ++k)
            atomicAdd(&diag_ptr[k], src[k]);

    }

    // Match upstream BCOO semantics: same-body terms write upper-triangle blocks only.
    if(body_i == body_j)
        zero_out_lower(H12x12);

    if(dst_rows && dst_cols && dst_vals)
    {
        int base_triplet = I * 16;
        for(int ii = 0; ii < 4; ++ii)
        {
            for(int jj = 0; jj < 4; ++jj)
            {
                int idx       = base_triplet + ii * 4 + jj;
                dst_rows[idx] = L * 4 + ii;
                dst_cols[idx] = R * 4 + jj;
                dst_vals[idx] =
                    H12x12.template block<BLK, BLK>(ii * BLK, jj * BLK);
            }
        }
    }
}

static __device__ Vector12 corex_abd_load_vector12(const Float* x, int body)
{
    Vector12 ret;
    int      base = body * 12;
    for(int k = 0; k < 12; ++k)
        ret(k) = x[base + k];
    return ret;
}

static __device__ void corex_abd_atomic_add_vector12(Float* y,
                                                     int    body,
                                                     const Vector12& value,
                                                     Float  a)
{
    int base = body * 12;
    for(int k = 0; k < 12; ++k)
        atomicAdd(&y[base + k], a * value(k));
}

static __global__ void kernel_abd_dytopo_matrix_free_spmv(int hess_count,
                                                          int vertex_offset,
                                                          const int* row_indices,
                                                          const int* col_indices,
                                                          const Matrix3x3* values,
                                                          const IndexT* v2b,
                                                          const ABDJacobi* Js,
                                                          const IndexT* is_fixed,
                                                          const Float* x,
                                                          Float* y,
                                                          Float a)
{
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    if(I >= hess_count) return;

    int g_i = row_indices[I];
    int g_j = col_indices[I];
    int i   = g_i - vertex_offset;
    int j   = g_j - vertex_offset;

    int body_i = v2b[i];
    int body_j = v2b[j];

    if(is_fixed[body_i] || is_fixed[body_j])
        return;

    const auto& J_i  = Js[i];
    const auto& J_j  = Js[j];
    const auto& H3x3 = values[I];

    const Vector12 x_body_i = corex_abd_load_vector12(x, body_i);
    const Vector12 x_body_j =
        body_i == body_j ? x_body_i : corex_abd_load_vector12(x, body_j);

    const Vector3 x_i = J_i * x_body_i;
    const Vector3 x_j = J_j * x_body_j;

    corex_abd_atomic_add_vector12(y, body_i, J_i.T() * (H3x3 * x_j), a);
    if(i != j)
        corex_abd_atomic_add_vector12(y, body_j, J_j.T() * (H3x3.transpose() * x_i), a);
}

void ABDLinearSubsystem::do_build(DiagLinearSubsystem::BuildInfo& info)
{
    m_impl.affine_body_dynamics        = require<AffineBodyDynamics>();
    m_impl.affine_body_vertex_reporter = require<AffineBodyVertexReporter>();
    auto attr = world().scene().config().find<Float>("dt");
    m_impl.dt = attr->view()[0];

    m_impl.dytopo_effect_receiver = find<ABDDyTopoEffectReceiver>();
}

void ABDLinearSubsystem::Impl::init()
{
    auto reporter_view = reporters.view();
    for(auto&& [i, r] : enumerate(reporter_view))
        r->m_index = i;
    for(auto& r : reporter_view)
        r->init();

    reporter_gradient_offsets_counts.resize(reporter_view.size());
    reporter_hessian_offsets_counts.resize(reporter_view.size());

    SizeT body_count = abd().body_count();
    body_id_to_shape_hessian.resize(body_count);
    body_id_to_shape_gradient.resize(body_count);
    body_id_to_kinetic_hessian.resize(body_count);
    body_id_to_kinetic_gradient.resize(body_count);
    diag_hessian.resize(body_count);
    reduced_norm.resize(1);
}

void ABDLinearSubsystem::Impl::report_init_extent(GlobalLinearSystem::InitDofExtentInfo& info)
{
    info.extent(abd().body_count() * 12);
}

void ABDLinearSubsystem::Impl::receive_init_dof_info(WorldVisitor& w,
                                                     GlobalLinearSystem::InitDofInfo& info)
{
    auto& geo_infos = abd().geo_infos;
    auto  geo_slots = w.scene().geometries();

    IndexT offset = info.dof_offset();

    // fill the dof_offset and dof_count for each geometry
    affine_body_dynamics->for_each(
        geo_slots,
        [&](const AffineBodyDynamics::ForEachInfo& foreach_info, geometry::SimplicialComplex& sc)
        {
            auto I          = foreach_info.global_index();
            auto dof_offset = sc.meta().find<IndexT>(builtin::dof_offset);
            UIPC_ASSERT(dof_offset, "dof_offset not found on ABD mesh why can it happen?");
            auto dof_count = sc.meta().find<IndexT>(builtin::dof_count);
            UIPC_ASSERT(dof_count, "dof_count not found on ABD mesh why can it happen?");

            IndexT this_dof_count = 12 * sc.instances().size();
            view(*dof_offset)[0]  = offset;
            view(*dof_count)[0]   = this_dof_count;

            offset += this_dof_count;
        });

    UIPC_ASSERT(offset == info.dof_offset() + info.dof_count(), "dof size mismatch");
}

void ABDLinearSubsystem::Impl::report_extent(GlobalLinearSystem::DiagExtentInfo& info)
{
    // 1. Gradient Count
    constexpr SizeT G12_to_dof = 12;
    SizeT           body_count = abd().body_count();
    auto            dof_count  = body_count * G12_to_dof;

    auto has_complement =
        has_flags(info.component_flags(), GlobalLinearSystem::ComponentFlags::Complement);

    SizeT H12x12_count = 0;

    if(has_complement)
    {
        // 1) Body hessian: kinetic + shape
        if(!info.gradient_only())
            H12x12_count += abd().body_count();

        // 2) Reporters
        auto reporter_view = reporters.view();
        auto grad_counts   = reporter_gradient_offsets_counts.counts();
        auto hess_counts   = reporter_hessian_offsets_counts.counts();

        for(auto&& R : reporter_view)
        {
            ReportExtentInfo extent_info;
            extent_info.m_gradient_only = info.gradient_only();
            R->report_extent(extent_info);

            grad_counts[R->m_index] = extent_info.m_gradient_count;
            hess_counts[R->m_index] = extent_info.m_hessian_count;
        }

        reporter_gradient_offsets_counts.scan();
        reporter_hessian_offsets_counts.scan();

        if(!info.gradient_only())
            H12x12_count += reporter_hessian_offsets_counts.total_count();
    }


    if(dytopo_effect_receiver && !info.gradient_only())
    {
#if !(defined(UIPC_ENABLE_GIPC_CONTACT_MATRIX_FREE) && UIPC_ENABLE_GIPC_CONTACT_MATRIX_FREE)
        H12x12_count += dytopo_effect_receiver->hessians().triplet_count();
#endif
    }


    auto H3x3_count = H12x12_count * (4 * 4);

    // Debug: check whether kinetic+shape (Complement part) is considered.
    // Print only once per process to avoid log spam.
    static bool printed = false;
    if(!printed)
    {
        logger::info("[debug] ABDLinearSubsystem::report_extent "
                     "component_flags={}, gradient_only={}, has_complement={}, "
                     "H12x12_count={}, H3x3_count={}, dof_count={}",
                     enum_flags_name(info.component_flags()),
                     info.gradient_only(),
                     has_complement,
                     H12x12_count,
                     H3x3_count,
                     dof_count);
        printed = true;
    }

    if(info.gradient_only())
    {
        UIPC_ASSERT(H3x3_count == 0,
                    "Hessian block count should be zero (got {}) when gradient_only is true",
                    H3x3_count);
    }

    info.extent(H3x3_count, dof_count);
}

void ABDLinearSubsystem::Impl::assemble(GlobalLinearSystem::DiagInfo& info)
{
    using namespace muda;
    const bool corex_trace = (std::getenv("UIPC_COREX_TRACE_ABD_ASSEMBLE") != nullptr);
    auto trace = [&](const char* msg)
    {
        if(corex_trace)
            logger::info("[corex_trace][abd] {}", msg);
    };
    double profile_prepare_ms = 0.0;
    double profile_kinetic_shape_ms = 0.0;
    double profile_reporters_ms = 0.0;
    double profile_contact_zero_ms = 0.0;
    double profile_dytopo_ms = 0.0;
    auto profile_t0 = corex_profile::now_ms();

    // 0) Prepare buffers for reporters
    trace("assemble: prepare reporter buffers begin");
    {
        auto N = abd().body_count();

        reporter_gradients.reshape(N);
        reporter_gradients.resize_doublets(reporter_gradient_offsets_counts.total_count());

        reporter_hessians.reshape(N, N);
        reporter_hessians.resize_triplets(reporter_hessian_offsets_counts.total_count());
    }
    profile_prepare_ms += corex_profile::now_ms() - profile_t0;
    trace("assemble: prepare reporter buffers end");

    bool has_complement =
        has_flags(info.component_flags(), GlobalLinearSystem::ComponentFlags::Complement);

    IndexT hess_offset = 0;

    // 1) Static Topo Effect: Kinetic + Shape + Other Reporters
    if(has_complement)
    {
        trace("assemble: kinetic_shape begin");
        profile_t0 = corex_profile::now_ms();
        _assemble_kinetic_shape(hess_offset, info);
        profile_kinetic_shape_ms += corex_profile::now_ms() - profile_t0;
        trace("assemble: kinetic_shape end");
        trace("assemble: reporters begin");
        profile_t0 = corex_profile::now_ms();
        _assemble_reporters(hess_offset, info);
        profile_reporters_ms += corex_profile::now_ms() - profile_t0;
        trace("assemble: reporters end");
    }
    else  // contact only
    {
        profile_t0 = corex_profile::now_ms();
        checkCudaErrors(cudaMemset(info.gradients().buffer_view().data(),
                                   0,
                                   sizeof(Float) * info.gradients().size()));
        profile_contact_zero_ms += corex_profile::now_ms() - profile_t0;
    }

    // 2) Dynamic Topology Effect
    trace("assemble: dytopo begin");
    profile_t0 = corex_profile::now_ms();
    _assemble_dytopo_effect(hess_offset, info);
    profile_dytopo_ms += corex_profile::now_ms() - profile_t0;
    trace("assemble: dytopo end");

    UIPC_ASSERT(hess_offset == info.hessians().triplet_count(),
                "Hessian size mismatch: expected {}, got {}",
                info.hessians().triplet_count(),
                hess_offset);
    corex_profile::log_phase("abd_assemble", "prepare_reporter_buffers", -1, -1, -1, profile_prepare_ms);
    corex_profile::log_phase("abd_assemble", "kinetic_shape", -1, -1, -1, profile_kinetic_shape_ms);
    corex_profile::log_phase("abd_assemble", "reporters", -1, -1, -1, profile_reporters_ms);
    corex_profile::log_phase("abd_assemble", "contact_zero_gradients", -1, -1, -1, profile_contact_zero_ms);
    corex_profile::log_phase("abd_assemble", "dytopo_effect", -1, -1, -1, profile_dytopo_ms);
}

void ABDLinearSubsystem::Impl::_assemble_kinetic_shape(IndexT& hess_offset,
                                                       GlobalLinearSystem::DiagInfo& info)
{
    using namespace muda;
    const bool corex_trace = (std::getenv("UIPC_COREX_TRACE_ABD_ASSEMBLE") != nullptr);
    auto trace = [&](const char* msg)
    {
        if(corex_trace)
            logger::info("[corex_trace][abd] kinetic_shape: {}", msg);
    };
    auto sync_dbg = [&](const char* where)
    {
        if(!corex_trace)
            return;
        auto err = cudaDeviceSynchronize();
        UIPC_ASSERT(err == cudaSuccess,
                    "cudaDeviceSynchronize failed at {}: {}",
                    where,
                    cudaGetErrorString(err));
        logger::info("[corex_trace][abd] kinetic_shape: sync ok at {}", where);
    };
    const bool direct_bdf1_assembly =
        std::getenv("UIPC_COREX_ABD_BDF1_DIRECT_ASSEMBLY") != nullptr
        && !info.gradient_only();

    // Collect Kinetic
    if(!direct_bdf1_assembly)
    {
        trace("collect kinetic begin");
        ABDLinearSubsystem::ComputeGradientHessianInfo this_info{
            info.gradient_only(), body_id_to_kinetic_gradient, body_id_to_kinetic_hessian, dt};
        abd().kinetic->compute_gradient_hessian(this_info);
        sync_dbg("after kinetic");
        trace("collect kinetic end");
    }
    else
    {
        trace("collect kinetic skipped for direct BDF1 assembly");
    }

    // Collect Shape
    trace("collect constitutions begin");
    {
        auto body_count = abd().body_count();
        auto err0 = cudaMemset(body_id_to_shape_gradient.data(),
                               0,
                               sizeof(Vector12) * body_count);
        UIPC_ASSERT(err0 == cudaSuccess,
                    "cudaMemset(shape_gradient) failed: {}",
                    cudaGetErrorString(err0));
        auto err1 = cudaMemset(body_id_to_shape_hessian.data(),
                               0,
                               sizeof(Matrix12x12) * body_count);
        UIPC_ASSERT(err1 == cudaSuccess,
                    "cudaMemset(shape_hessian) failed: {}",
                    cudaGetErrorString(err1));

        for(auto&& [i, cst] : enumerate(abd().constitutions.view()))
        {
            if(corex_trace)
                logger::info("[corex_trace][abd] kinetic_shape: constitution {} uid={} begin",
                             i,
                             cst->uid());

            ABDLinearSubsystem::ComputeGradientHessianInfo this_info{
                info.gradient_only(),
                abd().subview(body_id_to_shape_gradient, cst->m_index),
                abd().subview(body_id_to_shape_hessian, cst->m_index),
                dt};

            cst->compute_gradient_hessian(this_info);

            if(corex_trace)
            {
                auto where = fmt::format("after constitution {} uid={}", i, cst->uid());
                sync_dbg(where.c_str());
                logger::info("[corex_trace][abd] kinetic_shape: constitution {} uid={} end",
                             i,
                             cst->uid());
            }
        }
    }
    trace("collect constitutions end");

    if(direct_bdf1_assembly)
    {
        auto body_count = body_id_to_shape_hessian.size();
        auto H3x3_count = body_count * (4 * 4);
        auto body_H3x3  = info.hessians().subview(hess_offset, H3x3_count);

        int n = static_cast<int>(body_count);
        if(n > 0)
        {
            int block = 128;
            int grid  = (n + block - 1) / block;
            kernel_abd_assemble_bdf1_direct<<<grid, block>>>(
                n,
                abd().body_id_to_is_fixed.data(),
                abd().body_id_to_external_kinetic.data(),
                abd().body_id_to_q.data(),
                abd().body_id_to_q_tilde.data(),
                abd().body_id_to_abd_mass.data(),
                body_id_to_shape_gradient.data(),
                body_id_to_shape_hessian.data(),
                info.gradients().data(),
                this->diag_hessian.data(),
                body_H3x3.row_indices().data(),
                body_H3x3.col_indices().data(),
                body_H3x3.values().data());
            checkCudaErrors(cudaGetLastError());
            corex_abd_assemble_sync_if_requested("after direct BDF1 assembly");
        }

        hess_offset += H3x3_count;
        trace("direct BDF1 assembly done");
        return;
    }

    trace("assemble gradients kernel begin");
    {
        int n = static_cast<int>(abd().body_count());
        if(n > 0)
        {
            int block = 128;
            int grid  = (n + block - 1) / block;
            kernel_abd_assemble_gradients<<<grid, block>>>(
                n,
                abd().body_id_to_is_fixed.data(),
                abd().body_id_to_external_kinetic.data(),
                body_id_to_shape_gradient.data(),
                body_id_to_kinetic_gradient.data(),
                info.gradients().data());
            checkCudaErrors(cudaGetLastError());
            corex_abd_assemble_sync_if_requested("after assemble gradients");
        }
    }
    trace("assemble gradients kernel end");

    if(info.gradient_only())
        return;

    auto body_count = body_id_to_shape_hessian.size();
    auto H3x3_count = body_count * (4 * 4);
    auto body_H3x3  = info.hessians().subview(hess_offset, H3x3_count);

    trace("assemble hessians kernel begin");
    {
        int n = static_cast<int>(body_count);
        if(n > 0)
        {
            auto dst_rows = body_H3x3.row_indices();
            auto dst_cols = body_H3x3.col_indices();
            auto dst_vals = body_H3x3.values();

            if(std::getenv("UIPC_COREX_TRACE_LINEAR_SYSTEM"))
            {
                logger::info("[corex_trace][abd] hess host: n={}, triplets={}, "
                             "dst_rows.data()={}, dst_cols.data()={}, dst_vals.data()={}",
                             n, n * 16,
                             (void*)dst_rows.data(), (void*)dst_cols.data(), (void*)dst_vals.data());
            }

            int block = 128;
            int grid  = (n + block - 1) / block;
            if(std::getenv("UIPC_COREX_TRACE_LINEAR_SYSTEM"))
                logger::info("[corex_trace][abd] hess kernel launch: grid={}, block={}", grid, block);
            kernel_abd_assemble_hessians<<<grid, block>>>(
                n,
                abd().body_id_to_is_fixed.data(),
                abd().body_id_to_external_kinetic.data(),
                body_id_to_shape_hessian.data(),
                body_id_to_kinetic_hessian.data(),
                this->diag_hessian.data(),
                dst_rows.data(),
                dst_cols.data(),
                dst_vals.data());
            checkCudaErrors(cudaGetLastError());
            if(std::getenv("UIPC_COREX_TRACE_LINEAR_SYSTEM"))
                logger::info("[corex_trace][abd] hess kernel launch ok, syncing...");
            corex_abd_assemble_sync_if_requested("after assemble hessians");
            if(std::getenv("UIPC_COREX_TRACE_LINEAR_SYSTEM"))
                logger::info("[corex_trace][abd] hess kernel sync done");
        }
    }
    trace("assemble hessians kernel end");

    hess_offset += H3x3_count;
    trace("kinetic_shape done");
}

void ABDLinearSubsystem::Impl::_assemble_reporters(IndexT& offset,
                                                   GlobalLinearSystem::DiagInfo& info)
{
    using namespace muda;

    // Fill TripletMatrix and DoubletVector
    for(auto& R : reporters.view())
    {
        AssembleInfo assemble_info{this, R->m_index, info.gradient_only()};
        R->assemble(assemble_info);
    }

    if(reporter_gradients.doublet_count())
    {
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(reporter_gradients.doublet_count(),
                   [dst = info.gradients().viewer().name("dst_gradient"),
                    src = reporter_gradients.cviewer().name("src_gradient"),
                    is_fixed = abd().body_id_to_is_fixed.cviewer().name(
                        "is_fixed")] __device__(int I) mutable
                   {
                       auto&& [body_i, G12] = src(I);

                       if(is_fixed(body_i))
                       {
                           // Do nothing
                       }
                       else
                       {
                           dst.segment<12>(body_i * 12).atomic_add(G12);
                       }
                   });
    }

    if(!info.gradient_only() && reporter_hessians.triplet_count())
    {
        // get rest
        auto H3x3s = info.hessians().subview(offset);

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(reporter_hessians.triplet_count(),
                   [dst = H3x3s.viewer().name("dst_hessian"),
                    src = reporter_hessians.cviewer().name("src_hessian"),
                    diag_hessian = this->diag_hessian.viewer().name("diag_hessian"),
                    is_fixed = abd().body_id_to_is_fixed.cviewer().name(
                        "is_fixed")] __device__(int I) mutable
                   {
                       TripletMatrixUnpacker MU{dst};
                       Matrix12x12           H12x12;
                       auto&& [body_i, body_j, Value] = src(I);
                       H12x12                         = Value;

                       bool has_fixed = (is_fixed(body_i) || is_fixed(body_j));

                       // Fill diagonal hessian for diag-inv preconditioner
                       if(body_i == body_j && !has_fixed)
                       {
                           eigen::atomic_add(diag_hessian(body_i), H12x12);
                       }

                       if(has_fixed)
                       {
                           // Zero out hessian for fixed bodies.
                           H12x12.setZero();
                       }
                       else
                       {
                           if(body_i == body_j)
                           {
                               // Since body_i == body_j, we only fill the upper triangle part
                               zero_out_lower(H12x12);
                           }
                           else if(body_i > body_j)
                           {
                               // If all the reporters only report upper triangle part, this branch should not be hit
                               H12x12.setZero();
                           }
                       }

                       MU.block<4, 4>(I * 4 * 4)  // triplet range of [I*4*4, (I+1)*4*4)
                           .write(body_i * 4,  // begin row
                                  body_j * 4,  // begin col
                                  H12x12);
                   });

        offset += reporter_hessians.triplet_count() * (4 * 4);
    }
}

void ABDLinearSubsystem::Impl::_assemble_dytopo_effect(IndexT& offset,
                                                       GlobalLinearSystem::DiagInfo& info)
{
    using namespace muda;

    auto  vertex_offset = affine_body_vertex_reporter->vertex_offset();
    SizeT dytopo_effect_gradient_count = 0;
    if(dytopo_effect_receiver)
    {
        dytopo_effect_gradient_count =
            dytopo_effect_receiver->gradients().doublet_count();
    }

    if(dytopo_effect_gradient_count)
    {
        auto src_grad = dytopo_effect_receiver->gradients();
        if(corex_abd_dytopo_parallel_enabled())
        {
            constexpr int kBlk = 256;
            int           n    = static_cast<int>(dytopo_effect_gradient_count);
            kernel_abd_dytopo_gradients_parallel<<<(n + kBlk - 1) / kBlk, kBlk>>>(
                n,
                static_cast<int>(vertex_offset),
                src_grad.indices().data(),
                src_grad.values().data(),
                abd().vertex_id_to_body_id.data(),
                abd().vertex_id_to_J.data(),
                abd().body_id_to_is_fixed.data(),
                info.gradients().data());
        }
        else
        {
            kernel_abd_dytopo_gradients_serial<<<1, 1>>>(
                static_cast<int>(dytopo_effect_gradient_count),
                static_cast<int>(vertex_offset),
                src_grad.indices().data(),
                src_grad.values().data(),
                abd().vertex_id_to_body_id.data(),
                abd().vertex_id_to_J.data(),
                abd().body_id_to_is_fixed.data(),
                info.gradients().data());
        }
        checkCudaErrors(cudaGetLastError());
        corex_abd_assemble_sync_if_requested("after dytopo gradients");

    }

    if(info.gradient_only())
        return;

    SizeT dytopo_effect_hessian_count = 0;
    if(dytopo_effect_receiver)
        dytopo_effect_hessian_count = dytopo_effect_receiver->hessians().triplet_count();

#if defined(UIPC_ENABLE_GIPC_CONTACT_MATRIX_FREE) && UIPC_ENABLE_GIPC_CONTACT_MATRIX_FREE
    auto H3x3_count = SizeT{0};
#else
    auto H3x3_count         = dytopo_effect_hessian_count * (4 * 4);
    auto dytopo_effect_H3x3 = info.hessians().subview(offset, H3x3_count);
#endif

    if(dytopo_effect_hessian_count)
    {
        // Half Contact Hessian
        // ref: https://github.com/spiriMirror/libuipc/issues/272
        auto src_hess = dytopo_effect_receiver->hessians();
#if defined(UIPC_ENABLE_GIPC_CONTACT_MATRIX_FREE) && UIPC_ENABLE_GIPC_CONTACT_MATRIX_FREE
        int*      dst_rows = nullptr;
        int*      dst_cols = nullptr;
        Matrix3x3* dst_vals = nullptr;
#else
        auto      dst_rows = dytopo_effect_H3x3.row_indices().data();
        auto      dst_cols = dytopo_effect_H3x3.col_indices().data();
        auto      dst_vals = dytopo_effect_H3x3.values().data();
#endif
        if(corex_abd_dytopo_parallel_enabled())
        {
            constexpr int kBlk = 256;
            int           n    = static_cast<int>(dytopo_effect_hessian_count);
            kernel_abd_dytopo_hessians_parallel<<<(n + kBlk - 1) / kBlk, kBlk>>>(
                n,
                static_cast<int>(vertex_offset),
                src_hess.row_indices().data(),
                src_hess.col_indices().data(),
                src_hess.values().data(),
                abd().vertex_id_to_body_id.data(),
                abd().vertex_id_to_J.data(),
                abd().body_id_to_is_fixed.data(),
                this->diag_hessian.data(),
                dst_rows,
                dst_cols,
                dst_vals);
        }
        else
        {
            kernel_abd_dytopo_hessians_serial<<<1, 1>>>(
                static_cast<int>(dytopo_effect_hessian_count),
                static_cast<int>(vertex_offset),
                src_hess.row_indices().data(),
                src_hess.col_indices().data(),
                src_hess.values().data(),
                abd().vertex_id_to_body_id.data(),
                abd().vertex_id_to_J.data(),
                abd().body_id_to_is_fixed.data(),
                this->diag_hessian.data(),
                dst_rows,
                dst_cols,
                dst_vals);
        }
        checkCudaErrors(cudaGetLastError());
        corex_abd_assemble_sync_if_requested("after dytopo hessians");

    }

    offset += H3x3_count;
}

void ABDLinearSubsystem::Impl::matrix_free_spmv(GlobalLinearSystem::MatrixFreeSpMVInfo& info)
{
#if defined(UIPC_ENABLE_GIPC_CONTACT_MATRIX_FREE) && UIPC_ENABLE_GIPC_CONTACT_MATRIX_FREE
    if(!dytopo_effect_receiver)
        return;

    auto hess_count = dytopo_effect_receiver->hessians().triplet_count();
    if(!hess_count)
        return;

    auto          vertex_offset = affine_body_vertex_reporter->vertex_offset();
    auto          src_hess      = dytopo_effect_receiver->hessians();
    constexpr int kBlk          = 256;
    int           n             = static_cast<int>(hess_count);
    kernel_abd_dytopo_matrix_free_spmv<<<(n + kBlk - 1) / kBlk, kBlk>>>(
        n,
        static_cast<int>(vertex_offset),
        src_hess.row_indices().data(),
        src_hess.col_indices().data(),
        src_hess.values().data(),
        abd().vertex_id_to_body_id.data(),
        abd().vertex_id_to_J.data(),
        abd().body_id_to_is_fixed.data(),
        info.x().data(),
        info.y().data(),
        info.a());
    checkCudaErrors(cudaGetLastError());
#else
    (void)info;
#endif
}

void ABDLinearSubsystem::Impl::accuracy_check(GlobalLinearSystem::AccuracyInfo& info)
{
    info.satisfied(true);
}

__global__ void kernel_retrieve_solution(int n, Vector12* dq, const Float* x)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    for(int d = 0; d < 12; ++d)
        dq[i](d) = -x[i * 12 + d];
}

void ABDLinearSubsystem::Impl::retrieve_solution(GlobalLinearSystem::SolutionInfo& info)
{
    using namespace muda;

    auto dq = abd().body_id_to_dq.view();
    int n = static_cast<int>(abd().body_count());
    if(n > 0)
    {
        int block = 128;
        int grid  = (n + block - 1) / block;
        kernel_retrieve_solution<<<grid, block>>>(
            n, (Vector12*)dq.data(), (const Float*)info.solution().data());
        checkCudaErrors(cudaGetLastError());
        corex_abd_assemble_sync_if_requested("after retrieve solution");
    }
}

Float ABDLinearSubsystem::Impl::diag_norm()
{
    auto diag_hess = diag_hessian.view();
    block_norm.resize(diag_hess.size() * 12);
    muda::ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(diag_hess.size(),
               [diag_hess        = diag_hess.cviewer().name("diag_hess"),
                diag_blocks_norm = block_norm.viewer().name("diag_blocks_norm"),
                is_fixed = abd().body_id_to_is_fixed.cviewer().name("is_fixed")] __device__(int idx) mutable
               {
                   for(int i = 0; i < 12; i++)
                       diag_blocks_norm(idx * 12 + i) =
                           is_fixed(idx) ? 0 : abs(diag_hess(idx)(i, i));
               });

    muda::DeviceReduce().Max(block_norm.data(), reduced_norm.data(), block_norm.size());
    Float h_reduced_norm = 0;
    reduced_norm.view().copy_to(&h_reduced_norm);
    return h_reduced_norm;
}

Float ABDLinearSubsystem::Impl::mass_norm()
{
    auto mass = abd().body_id_to_abd_mass.view();
    block_norm.resize(mass.size());
    muda::ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(mass.size(),
               [mass       = mass.cviewer().name("diag_hess"),
                block_norm = block_norm.viewer().name("diag_blocks_norm"),
                is_fixed = abd().body_id_to_is_fixed.cviewer().name("is_fixed")] __device__(int idx) mutable
               { block_norm(idx) = is_fixed(idx) ? 0 : mass(idx).mass(); });

    muda::DeviceReduce().Max(block_norm.data(), reduced_norm.data(), block_norm.size());
    Float h_reduced_norm = 0;
    reduced_norm.view().copy_to(&h_reduced_norm);
    return h_reduced_norm;
}
}  // namespace uipc::backend::cuda

namespace uipc::backend::cuda
{
void ABDLinearSubsystem::do_init(InitInfo& info)
{
    m_impl.init();
}

void ABDLinearSubsystem::do_report_extent(GlobalLinearSystem::DiagExtentInfo& info)
{
    m_impl.report_extent(info);
}

void ABDLinearSubsystem::do_assemble(GlobalLinearSystem::DiagInfo& info)
{
    m_impl.assemble(info);
}

void ABDLinearSubsystem::do_matrix_free_spmv(GlobalLinearSystem::MatrixFreeSpMVInfo& info)
{
    m_impl.matrix_free_spmv(info);
}

void ABDLinearSubsystem::do_accuracy_check(GlobalLinearSystem::AccuracyInfo& info)
{
    m_impl.accuracy_check(info);
}

void ABDLinearSubsystem::do_retrieve_solution(GlobalLinearSystem::SolutionInfo& info)
{
    m_impl.retrieve_solution(info);
}

Float ABDLinearSubsystem::do_diag_norm(GlobalLinearSystem::DiagNormInfo& info)
{
    return m_impl.diag_norm();
}

Float ABDLinearSubsystem::do_mass_norm(GlobalLinearSystem::DiagNormInfo& info)
{
    return m_impl.mass_norm();
}

U64 ABDLinearSubsystem::get_uid() const noexcept
{
    return ABDLinearSubsystemUID;
}

void ABDLinearSubsystem::add_reporter(ABDLinearSubsystemReporter* reporter)
{
    UIPC_ASSERT(reporter, "reporter cannot be null");
    check_state(SimEngineState::BuildSystems, "add_reporter");
    m_impl.reporters.register_sim_system(*reporter);
}

void ABDLinearSubsystem::do_report_init_extent(GlobalLinearSystem::InitDofExtentInfo& info)
{
    m_impl.report_init_extent(info);
}

void ABDLinearSubsystem::do_receive_init_dof_info(GlobalLinearSystem::InitDofInfo& info)
{
    m_impl.receive_init_dof_info(world(), info);
}

ABDLinearSubsystem::AssembleInfo::AssembleInfo(Impl* impl, IndexT index, bool gradient_only) noexcept
    : m_impl(impl)
    , m_index(index)
    , m_gradient_only(gradient_only)
{
}

muda::DoubletVectorView<Float, 12> ABDLinearSubsystem::AssembleInfo::gradients() const
{
    auto [offset, count] = m_impl->reporter_gradient_offsets_counts[m_index];
    return m_impl->reporter_gradients.view().subview(offset, count);
}

muda::TripletMatrixView<Float, 12, 12> ABDLinearSubsystem::AssembleInfo::hessians() const
{
    auto [offset, count] = m_impl->reporter_hessian_offsets_counts[m_index];
    return m_impl->reporter_hessians.view().subview(offset, count);
}

bool ABDLinearSubsystem::AssembleInfo::gradient_only() const noexcept
{
    return m_gradient_only;
}

void ABDLinearSubsystem::ReportExtentInfo::gradient_count(SizeT size)
{
    m_gradient_count = size;
}

void ABDLinearSubsystem::ReportExtentInfo::hessian_count(SizeT size)
{
    m_hessian_count = size;
}

void ABDLinearSubsystem::ReportExtentInfo::check(std::string_view name) const
{
    check_report_extent(m_gradient_only_checked, m_gradient_only, m_hessian_count, name);
}

AffineBodyDynamics::Impl& ABDLinearSubsystem::Impl::abd() const noexcept
{
    return affine_body_dynamics->m_impl;
}
}  // namespace uipc::backend::cuda
#else
#include <affine_body/abd_linear_subsystem.h>
#include <sim_engine.h>
#include <kernel_cout.h>
#include <muda/ext/eigen.h>
#include <utils/matrix_assembler.h>
#include <utils/matrix_unpacker.h>
#include <uipc/builtin/attribute_name.h>
#include <affine_body/inter_affine_body_constitution_manager.h>
#include <affine_body/abd_linear_subsystem_reporter.h>
#include <affine_body/affine_body_kinetic.h>
#include <affine_body/affine_body_constitution.h>
#include <utils/report_extent_check.h>

namespace uipc::backend::cuda
{
UIPC_GENERIC void zero_out_lower(Matrix12x12& H)
{
    // clear lower triangle (3x3 block based)
    for(IndexT jj = 0; jj < 4; ++jj)
    {
        for(IndexT ii = jj + 1; ii < 4; ++ii)
        {
            H.block<3, 3>(ii * 3, jj * 3).setZero();
        }
    }
}
}  // namespace uipc::backend::cuda


namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(ABDLinearSubsystem);

// ref: https://github.com/spiriMirror/libuipc/issues/271
constexpr U64 ABDLinearSubsystemUID = 0ull;

void ABDLinearSubsystem::do_build(DiagLinearSubsystem::BuildInfo& info)
{
    m_impl.affine_body_dynamics        = require<AffineBodyDynamics>();
    m_impl.affine_body_vertex_reporter = require<AffineBodyVertexReporter>();
    auto attr = world().scene().config().find<Float>("dt");
    m_impl.dt = attr->view()[0];

    m_impl.dytopo_effect_receiver = find<ABDDyTopoEffectReceiver>();
}

void ABDLinearSubsystem::Impl::init()
{
    auto reporter_view = reporters.view();
    for(auto&& [i, r] : enumerate(reporter_view))
        r->m_index = i;
    for(auto& r : reporter_view)
        r->init();

    reporter_gradient_offsets_counts.resize(reporter_view.size());
    reporter_hessian_offsets_counts.resize(reporter_view.size());

    SizeT body_count = abd().body_count();
    body_id_to_shape_hessian.resize(body_count);
    body_id_to_shape_gradient.resize(body_count);
    body_id_to_kinetic_hessian.resize(body_count);
    body_id_to_kinetic_gradient.resize(body_count);
    diag_hessian.resize(body_count);
}

void ABDLinearSubsystem::Impl::report_init_extent(GlobalLinearSystem::InitDofExtentInfo& info)
{
    info.extent(abd().body_count() * 12);
}

void ABDLinearSubsystem::Impl::receive_init_dof_info(WorldVisitor& w,
                                                     GlobalLinearSystem::InitDofInfo& info)
{
    auto& geo_infos = abd().geo_infos;
    auto  geo_slots = w.scene().geometries();

    IndexT offset = info.dof_offset();

    // fill the dof_offset and dof_count for each geometry
    affine_body_dynamics->for_each(
        geo_slots,
        [&](const AffineBodyDynamics::ForEachInfo& foreach_info, geometry::SimplicialComplex& sc)
        {
            auto I          = foreach_info.global_index();
            auto dof_offset = sc.meta().find<IndexT>(builtin::dof_offset);
            UIPC_ASSERT(dof_offset, "dof_offset not found on ABD mesh why can it happen?");
            auto dof_count = sc.meta().find<IndexT>(builtin::dof_count);
            UIPC_ASSERT(dof_count, "dof_count not found on ABD mesh why can it happen?");

            IndexT this_dof_count = 12 * sc.instances().size();
            view(*dof_offset)[0]  = offset;
            view(*dof_count)[0]   = this_dof_count;

            offset += this_dof_count;
        });

    UIPC_ASSERT(offset == info.dof_offset() + info.dof_count(), "dof size mismatch");
}

void ABDLinearSubsystem::Impl::report_extent(GlobalLinearSystem::DiagExtentInfo& info)
{
    // 1. Gradient Count
    constexpr SizeT G12_to_dof = 12;
    SizeT           body_count = abd().body_count();
    auto            dof_count  = body_count * G12_to_dof;

    auto has_complement =
        has_flags(info.component_flags(), GlobalLinearSystem::ComponentFlags::Complement);

    SizeT H12x12_count = 0;

    if(has_complement)
    {
        // 1) Body hessian: kinetic + shape
        if(!info.gradient_only())
            H12x12_count += abd().body_count();

        // 2) Reporters
        auto reporter_view = reporters.view();
        auto grad_counts   = reporter_gradient_offsets_counts.counts();
        auto hess_counts   = reporter_hessian_offsets_counts.counts();

        for(auto&& R : reporter_view)
        {
            ReportExtentInfo extent_info;
            extent_info.m_gradient_only = info.gradient_only();
            R->report_extent(extent_info);

            grad_counts[R->m_index] = extent_info.m_gradient_count;
            hess_counts[R->m_index] = extent_info.m_hessian_count;
        }

        reporter_gradient_offsets_counts.scan();
        reporter_hessian_offsets_counts.scan();

        if(!info.gradient_only())
            H12x12_count += reporter_hessian_offsets_counts.total_count();
    }


    if(dytopo_effect_receiver && !info.gradient_only())
    {
#if !(defined(UIPC_ENABLE_GIPC_CONTACT_MATRIX_FREE) && UIPC_ENABLE_GIPC_CONTACT_MATRIX_FREE)
        H12x12_count += dytopo_effect_receiver->hessians().triplet_count();
#endif
    }


    auto H3x3_count = H12x12_count * (4 * 4);

    if(info.gradient_only())
    {
        UIPC_ASSERT(H3x3_count == 0,
                    "Hessian block count should be zero (got {}) when gradient_only is true",
                    H3x3_count);
    }

    info.extent(H3x3_count, dof_count);
}

void ABDLinearSubsystem::Impl::assemble(GlobalLinearSystem::DiagInfo& info)
{
    using namespace muda;

    // 0) Prepare buffers for reporters
    {
        auto N = abd().body_count();

        reporter_gradients.reshape(N);
        reporter_gradients.resize_doublets(reporter_gradient_offsets_counts.total_count());

        reporter_hessians.reshape(N, N);
        reporter_hessians.resize_triplets(reporter_hessian_offsets_counts.total_count());
    }

    bool has_complement =
        has_flags(info.component_flags(), GlobalLinearSystem::ComponentFlags::Complement);

    IndexT hess_offset = 0;

    // 1) Static Topo Effect: Kinetic + Shape + Other Reporters
    if(has_complement)
    {
        _assemble_kinetic_shape(hess_offset, info);
        _assemble_reporters(hess_offset, info);
    }
    else  // contact only
    {
        info.gradients().buffer_view().fill(0);
    }

    // 2) Dynamic Topology Effect
    _assemble_dytopo_effect(hess_offset, info);

    UIPC_ASSERT(hess_offset == info.hessians().triplet_count(),
                "Hessian size mismatch: expected {}, got {}",
                info.hessians().triplet_count(),
                hess_offset);
}

void ABDLinearSubsystem::Impl::_assemble_kinetic_shape(IndexT& hess_offset,
                                                       GlobalLinearSystem::DiagInfo& info)
{
    using namespace muda;

    // Collect Kinetic
    ABDLinearSubsystem::ComputeGradientHessianInfo this_info{
        info.gradient_only(), body_id_to_kinetic_gradient, body_id_to_kinetic_hessian, dt};
    abd().kinetic->compute_gradient_hessian(this_info);

    // Collect Shape
    for(auto&& [i, cst] : enumerate(abd().constitutions.view()))
    {
        ABDLinearSubsystem::ComputeGradientHessianInfo this_info{
            info.gradient_only(),
            abd().subview(body_id_to_shape_gradient, cst->m_index),
            abd().subview(body_id_to_shape_hessian, cst->m_index),
            dt};

        cst->compute_gradient_hessian(this_info);
    }

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(abd().body_count(),
               [is_fixed = abd().body_id_to_is_fixed.cviewer().name("is_fixed"),
                is_external_kinetic =
                    abd().body_id_to_external_kinetic.cviewer().name("external_kinetic"),
                shape_gradient = body_id_to_shape_gradient.cviewer().name("shape_gradient"),
                kinetic_gradient = body_id_to_kinetic_gradient.cviewer().name("kinetic_gradient"),
                gradients = info.gradients().viewer().name("gradients"),
                cout      = KernelCout::viewer()] __device__(int i) mutable
               {
                   Vector12 src;

                   if(is_fixed(i))
                   {
                       src.setZero();  // if fixed, set to zero
                   }
                   else
                   {
                       src = shape_gradient(i);

                       // if not external kinetic, add kinetic gradient
                       if(!is_external_kinetic(i)) [[likely]]
                       {
                           src += kinetic_gradient(i);
                       }
                   }

                   gradients.segment<12>(i * 12) = src;

                   // cout << "EKG(" << i << "): " << src.transpose().eval() << "\n";
               });

    if(info.gradient_only())
        return;

    auto body_count = body_id_to_shape_hessian.size();
    auto H3x3_count = body_count * (4 * 4);
    auto body_H3x3  = info.hessians().subview(hess_offset, H3x3_count);

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(body_count,
               [dst      = body_H3x3.viewer().name("dst_hessian"),
                is_fixed = abd().body_id_to_is_fixed.cviewer().name("is_fixed"),
                is_external_kinetic =
                    abd().body_id_to_external_kinetic.cviewer().name("external_kinetic"),
                shape_hessian = body_id_to_shape_hessian.cviewer().name("src_hessian"),
                kinetic_hessian = body_id_to_kinetic_hessian.cviewer().name("kinetic_hessian"),
                diag_hessian = this->diag_hessian.viewer().name("diag_hessian")] __device__(int I) mutable
               {
                   TripletMatrixUnpacker MA{dst};
                   Matrix12x12           H12x12;

                   if(is_fixed(I))
                   {
                       // Fill kinetic hessian to identity to avoid singularity
                       H12x12.setIdentity();
                   }
                   else
                   {
                       // if not fixed, fill shape hessian
                       H12x12 = shape_hessian(I);

                       // if not external kinetic, add kinetic gradient
                       if(!is_external_kinetic(I)) [[likely]]
                       {
                           H12x12 += kinetic_hessian(I);
                       }
                   }

                   // record diagonal hessian for diag-inv preconditioner
                   diag_hessian(I) = H12x12;

                   // set the lower triangle blocks to zero for robustness
                   zero_out_lower(H12x12);

                   MA.block<4, 4>(I * 4 * 4)  // triplet range of [I*4*4, (I+1)*4*4)
                       .write(I * 4,          // begin row
                              I * 4,          // begin col
                              H12x12);
               });

    hess_offset += H3x3_count;
}

void ABDLinearSubsystem::Impl::_assemble_reporters(IndexT& offset,
                                                   GlobalLinearSystem::DiagInfo& info)
{
    using namespace muda;

    // Fill TripletMatrix and DoubletVector
    for(auto& R : reporters.view())
    {
        AssembleInfo assemble_info{this, R->m_index, info.gradient_only()};
        R->assemble(assemble_info);
    }

    if(reporter_gradients.doublet_count())
    {
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(reporter_gradients.doublet_count(),
                   [dst = info.gradients().viewer().name("dst_gradient"),
                    src = reporter_gradients.cviewer().name("src_gradient"),
                    is_fixed = abd().body_id_to_is_fixed.cviewer().name(
                        "is_fixed")] __device__(int I) mutable
                   {
                       auto&& [body_i, G12] = src(I);

                       if(is_fixed(body_i))
                       {
                           // Do nothing
                       }
                       else
                       {
                           dst.segment<12>(body_i * 12).atomic_add(G12);
                       }
                   });
    }

    if(!info.gradient_only() && reporter_hessians.triplet_count())
    {
        // get rest
        auto H3x3s = info.hessians().subview(offset);

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(reporter_hessians.triplet_count(),
                   [dst = H3x3s.viewer().name("dst_hessian"),
                    src = reporter_hessians.cviewer().name("src_hessian"),
                    diag_hessian = this->diag_hessian.viewer().name("diag_hessian"),
                    is_fixed = abd().body_id_to_is_fixed.cviewer().name(
                        "is_fixed")] __device__(int I) mutable
                   {
                       TripletMatrixUnpacker MU{dst};
                       Matrix12x12           H12x12;
                       auto&& [body_i, body_j, Value] = src(I);
                       H12x12                         = Value;

                       bool has_fixed = (is_fixed(body_i) || is_fixed(body_j));

                       // Fill diagonal hessian for diag-inv preconditioner
                       if(body_i == body_j && !has_fixed)
                       {
                           eigen::atomic_add(diag_hessian(body_i), H12x12);
                       }

                       if(has_fixed)
                       {
                           // Zero out hessian for fixed bodies
                           H12x12.setZero();
                       }
                       else
                       {
                           if(body_i == body_j)
                           {
                               // Since body_i == body_j, we only fill the upper triangle part
                               zero_out_lower(H12x12);
                           }
                           else if(body_i > body_j)
                           {
                               // If all the reporters only report upper triangle part, this branch should not be hit
                               H12x12.setZero();
                           }
                       }

                       MU.block<4, 4>(I * 4 * 4)  // triplet range of [I*4*4, (I+1)*4*4)
                           .write(body_i * 4,  // begin row
                                  body_j * 4,  // begin col
                                  H12x12);
                   });

        offset += reporter_hessians.triplet_count() * (4 * 4);
    }
}

void ABDLinearSubsystem::Impl::_assemble_dytopo_effect(IndexT& offset,
                                                       GlobalLinearSystem::DiagInfo& info)
{
    using namespace muda;

    auto  vertex_offset = affine_body_vertex_reporter->vertex_offset();
    SizeT dytopo_effect_gradient_count = 0;
    if(dytopo_effect_receiver)
    {
        dytopo_effect_gradient_count =
            dytopo_effect_receiver->gradients().doublet_count();
    }

    if(dytopo_effect_gradient_count)
    {
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(dytopo_effect_gradient_count,
                   [dytopo_effect_gradient =
                        dytopo_effect_receiver->gradients().cviewer().name("dytopo_effect_gradient"),
                    gradients = info.gradients().viewer().name("gradients"),
                    v2b = abd().vertex_id_to_body_id.cviewer().name("v2b"),
                    Js  = abd().vertex_id_to_J.cviewer().name("Js"),
                    is_fixed = abd().body_id_to_is_fixed.cviewer().name("is_fixed"),
                    vertex_offset = vertex_offset,
                    cout = KernelCout::viewer()] __device__(int I) mutable
                   {
                       const auto& [g_i, G3] = dytopo_effect_gradient(I);

                       auto  i      = g_i - vertex_offset;
                       auto  body_i = v2b(i);
                       auto& J_i    = Js(i);

                       if(is_fixed(body_i))
                       {
                           // Do nothing
                       }
                       else
                       {
                           Vector12 G12 = J_i.T() * G3;
                           gradients.segment<12>(body_i * 12).atomic_add(G12);

                           // cout << "DG(" << I << "): " << G12.transpose().eval() << "\n";
                       }
                   });
    }

    if(info.gradient_only())
        return;

    SizeT dytopo_effect_hessian_count = 0;
    if(dytopo_effect_receiver)
        dytopo_effect_hessian_count = dytopo_effect_receiver->hessians().triplet_count();

    auto H3x3_count = dytopo_effect_hessian_count * (4 * 4);
#if !(defined(UIPC_ENABLE_GIPC_CONTACT_MATRIX_FREE) && UIPC_ENABLE_GIPC_CONTACT_MATRIX_FREE)
    auto dytopo_effect_H3x3 = info.hessians().subview(offset, H3x3_count);
#else
    H3x3_count = 0;
#endif

    if(dytopo_effect_hessian_count)
    {
        // Half Contact Hessian
        // ref: https://github.com/spiriMirror/libuipc/issues/272
#if defined(UIPC_ENABLE_GIPC_CONTACT_MATRIX_FREE) && UIPC_ENABLE_GIPC_CONTACT_MATRIX_FREE
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(dytopo_effect_hessian_count,
                   [dytopo_effect_hessian =
                        dytopo_effect_receiver->hessians().cviewer().name("dytopo_effect_hessian"),
                    v2b = abd().vertex_id_to_body_id.cviewer().name("v2b"),
                    Js  = abd().vertex_id_to_J.cviewer().name("Js"),
                    is_fixed = abd().body_id_to_is_fixed.cviewer().name("is_fixed"),
                    diag_hessian = this->diag_hessian.viewer().name("diag_hessian"),
                    vertex_offset = vertex_offset] __device__(int I) mutable
                   {
                       const auto& [g_i, g_j, H3x3] = dytopo_effect_hessian(I);

                       auto i = g_i - vertex_offset;
                       auto j = g_j - vertex_offset;

                       auto body_i = v2b(i);
                       auto body_j = v2b(j);

                       if(is_fixed(body_i) || is_fixed(body_j) || body_i != body_j)
                           return;

                       auto& J_i = Js(i);
                       auto& J_j = Js(j);

                       Matrix12x12 H12x12;
                       if(i != j)
                       {
                           H12x12 = ABDJacobi::JT_H_J(J_i.T(), H3x3, J_j)
                                    + ABDJacobi::JT_H_J(J_j.T(), H3x3.transpose(), J_i);
                       }
                       else
                       {
                           H12x12 = ABDJacobi::JT_H_J(J_i.T(), H3x3, J_j);
                       }

                       eigen::atomic_add(diag_hessian(body_i), H12x12);
                   });
#else
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(dytopo_effect_hessian_count,
                   [dytopo_effect_hessian =
                        dytopo_effect_receiver->hessians().cviewer().name("dytopo_effect_hessian"),
                    dst = dytopo_effect_H3x3.viewer().name("dst_hessian"),
                    v2b = abd().vertex_id_to_body_id.cviewer().name("v2b"),
                    Js  = abd().vertex_id_to_J.cviewer().name("Js"),
                    is_fixed = abd().body_id_to_is_fixed.cviewer().name("is_fixed"),
                    diag_hessian = this->diag_hessian.viewer().name("diag_hessian"),
                    vertex_offset = vertex_offset] __device__(int I) mutable
                   {
                       const auto& [g_i, g_j, H3x3] = dytopo_effect_hessian(I);

                       auto i = g_i - vertex_offset;
                       auto j = g_j - vertex_offset;

                       auto body_i = v2b(i);
                       auto body_j = v2b(j);

                       auto& J_i = Js(i);
                       auto& J_j = Js(j);

                       Matrix12x12 H12x12;

                       // We know half contact hessian i <= j
                       // but we don't know body_i and body_j order
                       // so test and swap if necessary
                       IndexT L = body_i;
                       IndexT R = body_j;
                       if(body_i > body_j)
                       {
                           L = body_j;
                           R = body_i;
                       }

                       if(is_fixed(body_i) || is_fixed(body_j))
                       {
                           H12x12.setZero();
                       }
                       else
                       {
                           if(body_i < body_j)
                           {
                               H12x12 = ABDJacobi::JT_H_J(J_i.T(), H3x3, J_j);
                           }
                           else if(body_i > body_j)
                           {
                               H12x12 = ABDJacobi::JT_H_J(J_j.T(), H3x3.transpose(), J_i);
                           }
                           else  // body_i == body_j
                           {
                               // Two vertices from the same body
                               if(i != j)
                               {
                                   H12x12 = ABDJacobi::JT_H_J(J_i.T(), H3x3, J_j)
                                            + ABDJacobi::JT_H_J(J_j.T(), H3x3.transpose(), J_i);
                               }
                               else  // i == j
                               {
                                   H12x12 = ABDJacobi::JT_H_J(J_i.T(), H3x3, J_j);
                               }

                               // Fill diagonal hessian for diag-inv preconditioner
                               eigen::atomic_add(diag_hessian(body_i), H12x12);

                               // Since body_i == body_j, we only fill the upper triangle part
                               zero_out_lower(H12x12);
                           }
                       }

                       TripletMatrixUnpacker MU{dst};
                       MU.block<4, 4>(I * 4 * 4)  // triplet range of [I*16, (I+1)*16)
                           .write(L * 4,          // begin row
                                  R * 4,          // begin col
                                  H12x12);
                   });
#endif
    }

    offset += H3x3_count;
}

void ABDLinearSubsystem::Impl::matrix_free_spmv(GlobalLinearSystem::MatrixFreeSpMVInfo& info)
{
#if defined(UIPC_ENABLE_GIPC_CONTACT_MATRIX_FREE) && UIPC_ENABLE_GIPC_CONTACT_MATRIX_FREE
    using namespace muda;

    if(!dytopo_effect_receiver)
        return;

    auto hess_count = dytopo_effect_receiver->hessians().triplet_count();
    if(!hess_count)
        return;

    auto vertex_offset = affine_body_vertex_reporter->vertex_offset();
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(hess_count,
               [dytopo_effect_hessian =
                    dytopo_effect_receiver->hessians().cviewer().name("dytopo_effect_hessian"),
                x = info.x().cviewer().name("x"),
                y = info.y().viewer().name("y"),
                a = info.a(),
                v2b = abd().vertex_id_to_body_id.cviewer().name("v2b"),
                Js  = abd().vertex_id_to_J.cviewer().name("Js"),
                is_fixed = abd().body_id_to_is_fixed.cviewer().name("is_fixed"),
                vertex_offset = vertex_offset] __device__(int I) mutable
               {
                   const auto& [g_i, g_j, H3x3] = dytopo_effect_hessian(I);

                   auto i = g_i - vertex_offset;
                   auto j = g_j - vertex_offset;

                   auto body_i = v2b(i);
                   auto body_j = v2b(j);

                   if(is_fixed(body_i) || is_fixed(body_j))
                       return;

                   auto& J_i = Js(i);
                   auto& J_j = Js(j);

                   const Vector12 x_body_i = x.segment<12>(body_i * 12).as_eigen();
                   const Vector12 x_body_j =
                       body_i == body_j ? x_body_i : x.segment<12>(body_j * 12).as_eigen();
                   const Vector3 x_i = J_i * x_body_i;
                   const Vector3 x_j = J_j * x_body_j;

                   const Vector12 Hx_j = a * (J_i.T() * (H3x3 * x_j));
                   y.segment<12>(body_i * 12).atomic_add(Hx_j);
                   if(i != j)
                   {
                       const Vector12 Ht_x_i =
                           a * (J_j.T() * (H3x3.transpose() * x_i));
                       y.segment<12>(body_j * 12).atomic_add(Ht_x_i);
                   }
               });
#else
    (void)info;
#endif
}

void ABDLinearSubsystem::Impl::accuracy_check(GlobalLinearSystem::AccuracyInfo& info)
{
    info.satisfied(true);
}

void ABDLinearSubsystem::Impl::retrieve_solution(GlobalLinearSystem::SolutionInfo& info)
{
    using namespace muda;

    auto dq = abd().body_id_to_dq.view();
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(abd().body_count(),
               [dq = dq.viewer().name("dq"),
                x = info.solution().viewer().name("x")] __device__(int i) mutable
               {
                   // retrieve solution for each body
                   dq(i) = -x.segment<12>(i * 12).as_eigen();
               });
}

Float ABDLinearSubsystem::Impl::diag_norm()
{
    auto diag_hess = diag_hessian.view();
    block_norm.resize(diag_hess.size() * 12);
    muda::ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(diag_hess.size(),
               [diag_hess        = diag_hess.cviewer().name("diag_hess"),
                diag_blocks_norm = block_norm.viewer().name("diag_blocks_norm"),
                is_fixed = abd().body_id_to_is_fixed.cviewer().name("is_fixed")] __device__(int idx) mutable
               {
                   for(int i = 0; i < 12; i++)
                       diag_blocks_norm(idx * 12 + i) =
                           is_fixed(idx) ? 0 : abs(diag_hess(idx)(i, i));
               });

    muda::DeviceReduce().Max(block_norm.data(), reduced_norm.data(), block_norm.size());

    return reduced_norm;
}

Float ABDLinearSubsystem::Impl::mass_norm()
{
    auto mass = abd().body_id_to_abd_mass.view();
    block_norm.resize(mass.size());
    muda::ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(mass.size(),
               [mass       = mass.cviewer().name("diag_hess"),
                block_norm = block_norm.viewer().name("diag_blocks_norm"),
                is_fixed = abd().body_id_to_is_fixed.cviewer().name("is_fixed")] __device__(int idx) mutable
               { block_norm(idx) = is_fixed(idx) ? 0 : mass(idx).mass(); });

    muda::DeviceReduce().Max(block_norm.data(), reduced_norm.data(), block_norm.size());

    return reduced_norm;
}
}  // namespace uipc::backend::cuda

namespace uipc::backend::cuda
{
void ABDLinearSubsystem::do_init(InitInfo& info)
{
    m_impl.init();
}

void ABDLinearSubsystem::do_report_extent(GlobalLinearSystem::DiagExtentInfo& info)
{
    m_impl.report_extent(info);
}

void ABDLinearSubsystem::do_assemble(GlobalLinearSystem::DiagInfo& info)
{
    m_impl.assemble(info);
}

void ABDLinearSubsystem::do_matrix_free_spmv(GlobalLinearSystem::MatrixFreeSpMVInfo& info)
{
    m_impl.matrix_free_spmv(info);
}

void ABDLinearSubsystem::do_accuracy_check(GlobalLinearSystem::AccuracyInfo& info)
{
    m_impl.accuracy_check(info);
}

void ABDLinearSubsystem::do_retrieve_solution(GlobalLinearSystem::SolutionInfo& info)
{
    m_impl.retrieve_solution(info);
}

Float ABDLinearSubsystem::do_diag_norm(GlobalLinearSystem::DiagNormInfo& info)
{
    return m_impl.diag_norm();
}

Float ABDLinearSubsystem::do_mass_norm(GlobalLinearSystem::DiagNormInfo& info)
{
    return m_impl.mass_norm();
}

U64 ABDLinearSubsystem::get_uid() const noexcept
{
    return ABDLinearSubsystemUID;
}

void ABDLinearSubsystem::add_reporter(ABDLinearSubsystemReporter* reporter)
{
    UIPC_ASSERT(reporter, "reporter cannot be null");
    check_state(SimEngineState::BuildSystems, "add_reporter");
    m_impl.reporters.register_sim_system(*reporter);
}

void ABDLinearSubsystem::do_report_init_extent(GlobalLinearSystem::InitDofExtentInfo& info)
{
    m_impl.report_init_extent(info);
}

void ABDLinearSubsystem::do_receive_init_dof_info(GlobalLinearSystem::InitDofInfo& info)
{
    m_impl.receive_init_dof_info(world(), info);
}

ABDLinearSubsystem::AssembleInfo::AssembleInfo(Impl* impl, IndexT index, bool gradient_only) noexcept
    : m_impl(impl)
    , m_index(index)
    , m_gradient_only(gradient_only)
{
}

muda::DoubletVectorView<Float, 12> ABDLinearSubsystem::AssembleInfo::gradients() const
{
    auto [offset, count] = m_impl->reporter_gradient_offsets_counts[m_index];
    return m_impl->reporter_gradients.view().subview(offset, count);
}

muda::TripletMatrixView<Float, 12, 12> ABDLinearSubsystem::AssembleInfo::hessians() const
{
    auto [offset, count] = m_impl->reporter_hessian_offsets_counts[m_index];
    return m_impl->reporter_hessians.view().subview(offset, count);
}

bool ABDLinearSubsystem::AssembleInfo::gradient_only() const noexcept
{
    return m_gradient_only;
}

void ABDLinearSubsystem::ReportExtentInfo::gradient_count(SizeT size)
{
    m_gradient_count = size;
}

void ABDLinearSubsystem::ReportExtentInfo::hessian_count(SizeT size)
{
    m_hessian_count = size;
}

void ABDLinearSubsystem::ReportExtentInfo::check(std::string_view name) const
{
    check_report_extent(m_gradient_only_checked, m_gradient_only, m_hessian_count, name);
}

AffineBodyDynamics::Impl& ABDLinearSubsystem::Impl::abd() const noexcept
{
    return affine_body_dynamics->m_impl;
}
}  // namespace uipc::backend::cuda
#endif
