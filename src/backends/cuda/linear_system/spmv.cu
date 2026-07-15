#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
#include <linear_system/spmv.h>
#include <muda/atomic.h>
#include <muda/launch/launch.h>
#include <muda/cub/device/device_reduce.h>
#include <cub/warp/warp_reduce.cuh>
#include <cub/warp/warp_scan.cuh>
#include <cub/util_math.cuh>
#include <cuda_device/bit_operation.h>
#include <cuda_device/builtin.h>
#include <Eigen/Sparse>
#include <vector>
#include <cstdlib>

// Iluvatar llc may crash on CUB HeadSegmentedReduce / shuffle paths in rbk_* kernels.
#define UIPC_SPMV_ILUVATAR_RBK_WORKAROUND 1

namespace uipc::backend::cuda
{
namespace
{
__global__ void kernel_pointwise_mul(int n, const Float* x, const Float* y, Float* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
        out[i] = x[i] * y[i];
}

__global__ void kernel_scale_y(int n, Float b, Float* y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
        y[i] = b * y[i];
}

// Simplified per-row kernel: no __restrict__, scalar accumulators, unrolled
// inner products, no Float a parameter (caller handles scaling separately).
// The original kernel_sym_spmv_by_row with array accumulators + __restrict__
// produces corrupt results on CoreX (suspected compiler codegen / vectorization bug).
__global__ void kernel_sym_spmv_v2(int          n_block_rows,
                                   int          n_triplets,
                                   const int*   rows,
                                   const int*   cols,
                                   const Float* blocks,
                                   const Float* x,
                                   Float*       y)
{
    int br = blockIdx.x * blockDim.x + threadIdx.x;
    if(br >= n_block_rows)
        return;

    Float a0 = 0.0, a1 = 0.0, a2 = 0.0;

    for(int t = 0; t < n_triplets; ++t)
    {
        int bi = rows[t];
        int bj = cols[t];
        const Float* B = blocks + t * 9;

        if(bi == br)
        {
            Float x0 = x[bj * 3], x1 = x[bj * 3 + 1], x2 = x[bj * 3 + 2];
            a0 += B[0] * x0 + B[3] * x1 + B[6] * x2;
            a1 += B[1] * x0 + B[4] * x1 + B[7] * x2;
            a2 += B[2] * x0 + B[5] * x1 + B[8] * x2;
        }

        if(bj == br && bi != bj)
        {
            Float x0 = x[bi * 3], x1 = x[bi * 3 + 1], x2 = x[bi * 3 + 2];
            a0 += B[0] * x0 + B[1] * x1 + B[2] * x2;
            a1 += B[3] * x0 + B[4] * x1 + B[5] * x2;
            a2 += B[6] * x0 + B[7] * x1 + B[8] * x2;
        }
    }

    y[br * 3 + 0] = a0;
    y[br * 3 + 1] = a1;
    y[br * 3 + 2] = a2;
}

__global__ void kernel_sym_spmv_triplet_atomic(int          n_triplets,
                                               const int*   rows,
                                               const int*   cols,
                                               const Float* blocks,
                                               const Float* x,
                                               Float        a,
                                               Float*       y)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if(t >= n_triplets)
        return;

    int bi = rows[t];
    int bj = cols[t];
    const Float* B = blocks + t * 9;

    Float xj0 = x[bj * 3 + 0];
    Float xj1 = x[bj * 3 + 1];
    Float xj2 = x[bj * 3 + 2];

    Float yi0 = B[0] * xj0 + B[3] * xj1 + B[6] * xj2;
    Float yi1 = B[1] * xj0 + B[4] * xj1 + B[7] * xj2;
    Float yi2 = B[2] * xj0 + B[5] * xj1 + B[8] * xj2;

    atomicAdd(y + bi * 3 + 0, a * yi0);
    atomicAdd(y + bi * 3 + 1, a * yi1);
    atomicAdd(y + bi * 3 + 2, a * yi2);

    if(bi != bj)
    {
        Float xi0 = x[bi * 3 + 0];
        Float xi1 = x[bi * 3 + 1];
        Float xi2 = x[bi * 3 + 2];

        Float yj0 = B[0] * xi0 + B[1] * xi1 + B[2] * xi2;
        Float yj1 = B[3] * xi0 + B[4] * xi1 + B[5] * xi2;
        Float yj2 = B[6] * xi0 + B[7] * xi1 + B[8] * xi2;

        atomicAdd(y + bj * 3 + 0, a * yj0);
        atomicAdd(y + bj * 3 + 1, a * yj1);
        atomicAdd(y + bj * 3 + 2, a * yj2);
    }
}

}  // namespace

void Spmv::sym_spmv(Float                           a,
                    muda::CBCOOMatrixView<Float, 3> A,
                    muda::CDenseVectorView<Float>   x,
                    Float                           b,
                    muda::DenseVectorView<Float>    y)
{
    constexpr int N = 3;
    using T         = Float;

    int ny = static_cast<int>(y.size());
    int nt = static_cast<int>(A.triplet_count());
    constexpr int kBlk = 256;

    if(b != 0)
    {
        int grid = (ny + kBlk - 1) / kBlk;
        kernel_scale_y<<<grid, kBlk>>>(ny, b, y.buffer_view().data());
    }
    else
    {
        checkCudaErrors(cudaMemsetAsync(y.buffer_view().data(), 0, sizeof(Float) * ny));
    }
    if(nt > 0)
    {
        const bool force_row_scan = std::getenv("UIPC_COREX_SPMV_ROW_SCAN") != nullptr;
        if(force_row_scan)
        {
            int n_block_rows = ny / 3;
            int grid = (n_block_rows + kBlk - 1) / kBlk;
            kernel_sym_spmv_v2<<<grid, kBlk>>>(
                n_block_rows,
                nt,
                A.row_indices().data(),
                A.col_indices().data(),
                reinterpret_cast<const Float*>(A.values().data()),
                x.data(),
                y.buffer_view().data());
            if(a != 1.0)
            {
                int sgrid = (ny + kBlk - 1) / kBlk;
                kernel_scale_y<<<sgrid, kBlk>>>(ny, a, y.buffer_view().data());
            }
        }
        else
        {
            int grid = (nt + kBlk - 1) / kBlk;
            kernel_sym_spmv_triplet_atomic<<<grid, kBlk>>>(
                nt,
                A.row_indices().data(),
                A.col_indices().data(),
                reinterpret_cast<const Float*>(A.values().data()),
                x.data(),
                a,
                y.buffer_view().data());
        }
    }
}

__host__ __device__ constexpr int b2i(bool b)
{
    return b ? 1 : 0;
}

struct Flags
{
    union
    {
        struct
        {
            unsigned char is_head;
            unsigned char is_cross_warp;
            unsigned char is_valid;
        };
        unsigned int flags;
    };

    __host__ __device__ void b2i()
    {
        is_head       = is_head ? 1 : 0;
        is_cross_warp = is_cross_warp ? 1 : 0;
        is_valid      = is_valid ? 1 : 0;
    }
};

void Spmv::rbk_spmv(Float                           a,
                    muda::CBCOOMatrixView<Float, 3> A,
                    muda::CDenseVectorView<Float>   x,
                    Float                           b,
                    muda::DenseVectorView<Float>    y)
{
#if UIPC_SPMV_ILUVATAR_RBK_WORKAROUND
    sym_spmv(a, A, x, b, y);
#else
    using namespace muda;
    constexpr int N = 3;
    using T         = Float;

    if(b != 0)
    {
        muda::ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(y.size(),
                   [b = b, y = y.viewer().name("y")] __device__(int i) mutable
                   { y(i) = b * y(i); });
    }
    else
    {
        muda::BufferLaunch().fill<Float>(y.buffer_view(), 0);
    }

    constexpr int          warp_size = 32;
    constexpr unsigned int warp_mask = ~0u;
    constexpr int          block_dim = 128;
    int block_count = (A.triplet_count() + block_dim - 1) / block_dim;

    muda::Launch(block_count, block_dim)
        .file_line(__FILE__, __LINE__)
        .apply(
            [a = a,
             A = A.viewer().name("A"),
             x = x.viewer().name("x"),
             b = b,
             y = y.viewer().name("y")] __device__() mutable
            {
                using WarpReduceInt   = cub::WarpReduce<int, warp_size>;
                using WarpReduceFloat = cub::WarpReduce<Float, warp_size>;
                using WarpScanInt     = cub::WarpScan<int>;

                auto global_thread_id   = blockDim.x * blockIdx.x + threadIdx.x;
                auto thread_id_in_block = threadIdx.x;
                auto warp_id            = thread_id_in_block / warp_size;
                auto lane_id            = thread_id_in_block & (warp_size - 1);

                __shared__ union
                {
                    typename WarpReduceInt::TempStorage temp_storage_int[block_dim / warp_size];
                    typename WarpReduceFloat::TempStorage temp_storage_float[block_dim / warp_size];
                };

                int prev_i = -1;
                int next_i = -1;
                int i      = -1;

                Flags   flags;
                Vector3 vec;
                flags.is_cross_warp = 0;


                if(global_thread_id > 0 && global_thread_id < A.triplet_count())
                {
                    auto prev_triplet = A(global_thread_id - 1);
                    prev_i            = prev_triplet.row_index;
                }

                if(global_thread_id < A.triplet_count() - 1)
                {
                    auto next_triplet = A(global_thread_id + 1);
                    next_i            = next_triplet.row_index;
                }

                if(global_thread_id < A.triplet_count())
                {
                    auto Triplet = A(global_thread_id);
                    i            = Triplet.row_index;
                    auto j       = Triplet.col_index;

                    vec = Triplet.value * x.segment<N>(j * N).as_eigen();

                    flags.is_valid = 1;
                }
                else
                {
                    i = -1;
                    vec.setZero();
                    flags.is_valid      = 0;
                    flags.is_cross_warp = 0;
                }

                if(lane_id == 0)
                {
                    flags.is_head = 1;
                    // if this thread is the first thread in the warp
                    // check if the previous triplet is in the same row
                    // if so, this row crosses the warp boundary, we need use atomic add
                    flags.is_cross_warp = b2i(prev_i == i);
                }
                else
                {
                    flags.is_head = b2i(prev_i != i);  // must be 1 or 0, or the result is undefined

                    if(lane_id == warp_size - 1)
                    {
                        // if this thread is the last thread in the warp
                        // check if the next triplet is in the same row
                        // if so, this row crosses the warp boundary, we need use atomic add
                        flags.is_cross_warp = b2i(next_i == i);
                    }
                }

                flags.flags = WarpReduceInt(temp_storage_int[warp_id])
                                  .HeadSegmentedReduce(flags.flags,
                                                       flags.is_head,
                                                       [](uint32_t a, uint32_t b)
                                                       { return a + b; });

                vec.x() = WarpReduceFloat(temp_storage_float[warp_id])
                              .HeadSegmentedReduce(vec.x(),
                                                   flags.is_head,
                                                   [](Float a, Float b)
                                                   { return a + b; });

                vec.y() = WarpReduceFloat(temp_storage_float[warp_id])
                              .HeadSegmentedReduce(vec.y(),
                                                   flags.is_head,
                                                   [](Float a, Float b)
                                                   { return a + b; });

                vec.z() = WarpReduceFloat(temp_storage_float[warp_id])
                              .HeadSegmentedReduce(vec.z(),
                                                   flags.is_head,
                                                   [](Float a, Float b)
                                                   { return a + b; });


                // cub::WARP_SYNC(warp_mask);

                flags.is_head = b2i(flags.is_head && flags.is_valid);

                flags.b2i();
                int is_head_mask =
                    detail::bit_operation::WARP_BALLOT(flags.is_head, warp_mask);
                uint32_t offset = __fns(is_head_mask, 0, lane_id + 1);

                int valid_bit = (offset != ~0u);
                int shuffle_mask = detail::bit_operation::WARP_BALLOT(valid_bit, warp_mask);

                i = cub::ShuffleIndex<32>(i, offset, shuffle_mask);
                flags.flags = cub::ShuffleIndex<32>(flags.flags, offset, shuffle_mask);
                vec.x() = cub::ShuffleIndex<32>(vec.x(), offset, shuffle_mask);
                vec.y() = cub::ShuffleIndex<32>(vec.y(), offset, shuffle_mask);
                vec.z() = cub::ShuffleIndex<32>(vec.z(), offset, shuffle_mask);

                if(valid_bit && flags.is_head && flags.is_valid)
                {
                    auto seg_y  = y.segment<N>(i * N);
                    auto result = a * vec;

                    if(flags.is_cross_warp)
                    {
                        seg_y.atomic_add(result.eval());
                    }
                    else
                    {
                        seg_y.as_eigen() += result.eval();
                    }
                }
            });
#endif
}

void Spmv::rbk_sym_spmv(Float                           a,
                        muda::CBCOOMatrixView<Float, 3> A,
                        muda::CDenseVectorView<Float>   x,
                        Float                           b,
                        muda::DenseVectorView<Float>    y)

{
#if UIPC_SPMV_ILUVATAR_RBK_WORKAROUND
    sym_spmv(a, A, x, b, y);
#else
    using namespace muda;
    constexpr int N = 3;
    using T         = Float;

    if(b != 0)
    {
        muda::ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(y.size(),
                   [b = b, y = y.viewer().name("y")] __device__(int i) mutable
                   { y(i) = b * y(i); });
    }
    else
    {
        muda::BufferLaunch().fill<Float>(y.buffer_view(), 0);
    }

    constexpr int warp_size   = 32;
    constexpr int block_dim   = 256;
    int           block_count = (A.triplet_count() + block_dim - 1) / block_dim;

    muda::ParallelFor(block_count, block_dim)
        .file_line(__FILE__, __LINE__)
        .apply(A.triplet_count(),
               [a = a,
                A = A.viewer().name("A"),
                x = x.viewer().name("x"),
                b = b,
                y = y.viewer().name("y")] __device__(int idx) mutable
               {
                   using WarpReduceInt   = cub::WarpReduce<int, warp_size>;
                   using WarpReduceFloat = cub::WarpReduce<Float, warp_size>;
                   using WarpScanInt     = cub::WarpScan<int>;

                   auto global_thread_id   = idx;
                   auto thread_id_in_block = threadIdx.x;
                   auto warp_id            = thread_id_in_block / warp_size;
                   auto lane_id = thread_id_in_block & (warp_size - 1);

                   __shared__ union
                   {
                       typename WarpReduceInt::TempStorage temp_storage_int[block_dim / warp_size];
                       typename WarpReduceFloat::TempStorage temp_storage_float[block_dim / warp_size];
                   };

                   int     prev_i = -1;
                   int     i      = -1;
                   Flags   flags;
                   Vector3 vec;

                   // In symmtric version, we don't need to check the cross warp
                   flags.is_cross_warp = 0;

                   // set the previous row index
                   if(global_thread_id > 0)
                   {
                       auto prev_triplet = A(global_thread_id - 1);
                       prev_i            = prev_triplet.row_index;
                   }

                   {
                       auto Triplet = A(global_thread_id);
                       i            = Triplet.row_index;
                       auto j       = Triplet.col_index;

                       vec = Triplet.value * x.segment<N>(j * N).as_eigen();

                       flags.is_valid = 1;

                       if(i != j)  // process lower triangle
                       {
                           Vector3 vec_ = a * Triplet.value.transpose()
                                          * x.segment<N>(i * N).as_eigen();

                           y.segment<N>(j * N).atomic_add(vec_);
                       }
                   }

                   if(lane_id == 0)
                   {
                       flags.is_head = 1;
                   }
                   else
                   {
                       flags.is_head = b2i(prev_i != i);  // must be 1 or 0, or the result is undefined
                   }


                   // ----------------------------------- warp reduce ----------------------------------------------
                   vec.x() = WarpReduceFloat(temp_storage_float[warp_id])
                                 .HeadSegmentedReduce(vec.x(),
                                                      flags.is_head,
                                                      [](Float a, Float b)
                                                      { return a + b; });

                   vec.y() = WarpReduceFloat(temp_storage_float[warp_id])
                                 .HeadSegmentedReduce(vec.y(),
                                                      flags.is_head,
                                                      [](Float a, Float b)
                                                      { return a + b; });

                   vec.z() = WarpReduceFloat(temp_storage_float[warp_id])
                                 .HeadSegmentedReduce(vec.z(),
                                                      flags.is_head,
                                                      [](Float a, Float b)
                                                      { return a + b; });
                   // ----------------------------------- warp reduce -----------------------------------------------


                   if(flags.is_head)
                   {
                       auto seg_y  = y.segment<N>(i * N);
                       auto result = a * vec;

                       // Must use atomic add!
                       // Because the same row may be processed by different warps
                       seg_y.atomic_add(result.eval());
                   }
               });
#endif
}

void Spmv::rbk_sym_spmv_dot(Float                           a,
                            muda::CBCOOMatrixView<Float, 3> A,
                            muda::CDenseVectorView<Float>   x,
                            Float                           b,
                            muda::DenseVectorView<Float>    y,
                            muda::VarView<Float>            d_dot)
{
#if UIPC_SPMV_ILUVATAR_RBK_WORKAROUND
    sym_spmv(a, A, x, b, y);
    auto n = static_cast<int>(x.size());
    if(n > 0)
    {
        dot_buffer.resize(n);
        constexpr int kBlk = 256;
        int grid = (n + kBlk - 1) / kBlk;
        kernel_pointwise_mul<<<grid, kBlk>>>(n, x.data(), y.data(), dot_buffer.data());
        muda::DeviceReduce().Sum(dot_buffer.data(), d_dot.data(), n);
    }
    else
    {
        checkCudaErrors(cudaMemsetAsync(d_dot.data(), 0, sizeof(Float)));
    }
#else
    using namespace muda;
    constexpr int N = 3;
    using T         = Float;

    if(b != 0)
    {
        muda::ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(y.size(),
                   [b = b, y = y.viewer().name("y")] __device__(int i) mutable
                   { y(i) = b * y(i); });
    }
    else
    {
        muda::BufferLaunch().fill<Float>(y.buffer_view(), 0);
    }

    cudaMemsetAsync(d_dot.data(), 0, sizeof(Float));

    constexpr int warp_size   = 32;
    constexpr int block_dim   = 256;
    int           block_count = (A.triplet_count() + block_dim - 1) / block_dim;

    int triplet_count = A.triplet_count();

    muda::Launch(block_count, block_dim)
        .file_line(__FILE__, __LINE__)
        .apply(
               [a     = a,
                A     = A.viewer().name("A"),
                x     = x.viewer().name("x"),
                b     = b,
                y     = y.viewer().name("y"),
                d_dot = d_dot.viewer().name("d_dot"),
                triplet_count] __device__() mutable
               {
                   using WarpReduceInt   = cub::WarpReduce<int, warp_size>;
                   using WarpReduceFloat = cub::WarpReduce<Float, warp_size>;

                   auto global_thread_id   = blockDim.x * blockIdx.x + threadIdx.x;
                   auto thread_id_in_block = threadIdx.x;
                   auto warp_id            = thread_id_in_block / warp_size;
                   auto lane_id = thread_id_in_block & (warp_size - 1);

                   __shared__ union
                   {
                       typename WarpReduceInt::TempStorage temp_storage_int[block_dim / warp_size];
                       typename WarpReduceFloat::TempStorage temp_storage_float[block_dim / warp_size];
                   };

                   int     prev_i = -1;
                   int     i      = -1;
                   Flags   flags;
                   Vector3 vec;
                   vec.setZero();
                   Float   dot_local = 0;

                   flags.is_cross_warp = 0;
                   flags.is_valid      = 0;
                   flags.is_head       = 0;

                   if(global_thread_id < triplet_count)
                   {
                       if(global_thread_id > 0)
                       {
                           auto prev_triplet = A(global_thread_id - 1);
                           prev_i            = prev_triplet.row_index;
                       }

                       auto Triplet = A(global_thread_id);
                       i            = Triplet.row_index;
                       auto j       = Triplet.col_index;

                       Eigen::Vector<T, N> x_j = x.segment<N>(j * N).as_eigen();
                       vec = Triplet.value * x_j;

                       flags.is_valid = 1;

                       if(i == j)
                       {
                           dot_local = a * x_j.dot(vec);
                       }
                       else
                       {
                           Eigen::Vector<T, N> x_i = x.segment<N>(i * N).as_eigen();
                           dot_local = 2.0 * a * x_i.dot(vec);

                           Vector3 vec_ = a * Triplet.value.transpose() * x_i;
                           y.segment<N>(j * N).atomic_add(vec_);
                       }

                       if(lane_id == 0)
                           flags.is_head = 1;
                       else
                           flags.is_head = b2i(prev_i != i);
                   }

                   vec.x() = WarpReduceFloat(temp_storage_float[warp_id])
                                 .HeadSegmentedReduce(vec.x(),
                                                      flags.is_head,
                                                      [](Float a, Float b)
                                                      { return a + b; });

                   vec.y() = WarpReduceFloat(temp_storage_float[warp_id])
                                 .HeadSegmentedReduce(vec.y(),
                                                      flags.is_head,
                                                      [](Float a, Float b)
                                                      { return a + b; });

                   vec.z() = WarpReduceFloat(temp_storage_float[warp_id])
                                 .HeadSegmentedReduce(vec.z(),
                                                      flags.is_head,
                                                      [](Float a, Float b)
                                                      { return a + b; });

                   if(flags.is_head && flags.is_valid)
                   {
                       auto seg_y  = y.segment<N>(i * N);
                       auto result = a * vec;
                       seg_y.atomic_add(result.eval());
                   }

                   dot_local = WarpReduceFloat(temp_storage_float[warp_id])
                                   .Sum(dot_local);
                   if(lane_id == 0)
                       muda::atomic_add(d_dot.data(), dot_local);
               });
#endif
}

void Spmv::cpu_sym_spmv(Float                           a,
                        muda::CBCOOMatrixView<Float, 3> A_view,
                        muda::CDenseVectorView<Float>   x,
                        Float                           b,
                        muda::DenseVectorView<Float>    y)
{
    using BlockMatrix      = Matrix3x3;
    constexpr int BlockDim = 3;

    // Get matrix dimensions
    int total_block_rows  = A_view.total_rows();
    int total_block_cols  = A_view.total_cols();
    int total_scalar_rows = total_block_rows * BlockDim;
    int total_scalar_cols = total_block_cols * BlockDim;

    // Copy triplet data from device to host
    std::vector<int>         row_indices_host(A_view.triplet_count());
    std::vector<int>         col_indices_host(A_view.triplet_count());
    std::vector<BlockMatrix> values_host(A_view.triplet_count());

    A_view.row_indices().copy_to(row_indices_host.data());
    A_view.col_indices().copy_to(col_indices_host.data());
    A_view.values().copy_to(values_host.data());

    // Build Eigen SparseMatrix from symmetric block-sparse representation
    std::vector<Eigen::Triplet<Float>> triplets;
    triplets.reserve(A_view.triplet_count() * BlockDim * BlockDim * 2);  // Reserve for symmetric expansion

    for(size_t t = 0; t < A_view.triplet_count(); ++t)
    {
        int                block_i = row_indices_host[t];
        int                block_j = col_indices_host[t];
        const BlockMatrix& block   = values_host[t];

        int scalar_i_base = block_i * BlockDim;
        int scalar_j_base = block_j * BlockDim;

        if(block_i == block_j)
        {
            // Diagonal block: symmetrize within the 3x3 block
            // (zero_out_lower may have zeroed the lower triangle)
            BlockMatrix sym_block = (block + block.transpose()) * Float(0.5);
            for(int bi = 0; bi < BlockDim; ++bi)
                for(int bj = 0; bj < BlockDim; ++bj)
                    triplets.emplace_back(scalar_i_base + bi, scalar_j_base + bj, sym_block(bi, bj));
        }
        else
        {
            // Off-diagonal: add block at (i,j) and transpose at (j,i)
            for(int bi = 0; bi < BlockDim; ++bi)
                for(int bj = 0; bj < BlockDim; ++bj)
                    triplets.emplace_back(scalar_i_base + bi, scalar_j_base + bj, block(bi, bj));

            BlockMatrix block_transpose = block.transpose();
            for(int bi = 0; bi < BlockDim; ++bi)
                for(int bj = 0; bj < BlockDim; ++bj)
                    triplets.emplace_back(scalar_j_base + bi, scalar_i_base + bj, block_transpose(bi, bj));
        }
    }

    // Build Eigen SparseMatrix
    Eigen::SparseMatrix<Float> A_sparse(total_scalar_rows, total_scalar_cols);
    A_sparse.setFromTriplets(triplets.begin(), triplets.end());
    A_sparse.makeCompressed();


    // Copy vectors from device to host
    Eigen::VectorX<Float> x_host(x.size());
    Eigen::VectorX<Float> y_host(y.size());

    x.buffer_view().copy_to(x_host.data());
    y.buffer_view().copy_to(y_host.data());

    // Compute y = a * A * x + b * y using Eigen sparse matrix-vector multiplication
    y_host = a * A_sparse * x_host + b * y_host;

    // Copy result back to device
    y.buffer_view().copy_from(y_host.data());
}
}  // namespace uipc::backend::cuda
#else
#include <linear_system/spmv.h>
#include <muda/launch/launch.h>
#include <cub/warp/warp_reduce.cuh>
#include <cub/warp/warp_scan.cuh>
#include <cub/util_math.cuh>
#include <cuda_device/bit_operation.h>
#include <cuda_device/builtin.h>
#include <Eigen/Sparse>

namespace uipc::backend::cuda
{
void Spmv::sym_spmv(Float                           a,
                    muda::CBCOOMatrixView<Float, 3> A,
                    muda::CDenseVectorView<Float>   x,
                    Float                           b,
                    muda::DenseVectorView<Float>    y)
{

    constexpr int N = 3;
    using T         = Float;

    if(b != 0)
    {
        muda::ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(y.size(),
                   [b = b, y = y.viewer().name("y")] __device__(int i) mutable
                   { y(i) = b * y(i); });
    }
    else
    {
        muda::BufferLaunch().fill<Float>(y.buffer_view(), 0);
    }

    muda::ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(A.triplet_count(),
               [a = a,
                A = A.viewer().name("A"),
                x = x.viewer().name("x"),
                b = b,
                y = y.viewer().name("y")] __device__(int index) mutable
               {
                   auto&& [i, j, block] = A(index);

                   if(i == j)  // diagonal block
                   {
                       auto seg_x = x.segment<N>(j * N);

                       Eigen::Vector<T, N> vec_x  = seg_x.as_eigen();
                       auto                result = a * block * vec_x;

                       auto seg_y = y.segment<N>(i * N);
                       seg_y.atomic_add(result.eval());
                   }
                   else  // off-diagonal block
                   {
                       // ij-th block
                       {
                           auto seg_x = x.segment<N>(j * N);

                           Eigen::Vector<T, N> vec_x  = seg_x.as_eigen();
                           auto                result = a * block * vec_x;

                           auto seg_y = y.segment<N>(i * N);
                           seg_y.atomic_add(result.eval());
                       }

                       // ji-th block
                       {
                           auto seg_x = x.segment<N>(i * N);

                           Eigen::Vector<T, N> vec_x = seg_x.as_eigen();
                           auto result = a * block.transpose() * vec_x;

                           auto seg_y = y.segment<N>(j * N);
                           seg_y.atomic_add(result.eval());
                       }
                   }
               });
}

__host__ __device__ constexpr int b2i(bool b)
{
    return b ? 1 : 0;
}

struct Flags
{
    union
    {
        struct
        {
            unsigned char is_head;
            unsigned char is_cross_warp;
            unsigned char is_valid;
        };
        unsigned int flags;
    };

    __host__ __device__ void b2i()
    {
        is_head       = is_head ? 1 : 0;
        is_cross_warp = is_cross_warp ? 1 : 0;
        is_valid      = is_valid ? 1 : 0;
    }
};

void Spmv::rbk_spmv(Float                           a,
                    muda::CBCOOMatrixView<Float, 3> A,
                    muda::CDenseVectorView<Float>   x,
                    Float                           b,
                    muda::DenseVectorView<Float>    y)
{
    using namespace muda;
    constexpr int N = 3;
    using T         = Float;

    if(b != 0)
    {
        muda::ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(y.size(),
                   [b = b, y = y.viewer().name("y")] __device__(int i) mutable
                   { y(i) = b * y(i); });
    }
    else
    {
        muda::BufferLaunch().fill<Float>(y.buffer_view(), 0);
    }

    constexpr int          warp_size = 32;
    constexpr unsigned int warp_mask = ~0u;
    constexpr int          block_dim = 128;
    int block_count = (A.triplet_count() + block_dim - 1) / block_dim;

    muda::Launch(block_count, block_dim)
        .file_line(__FILE__, __LINE__)
        .apply(
            [a = a,
             A = A.viewer().name("A"),
             x = x.viewer().name("x"),
             b = b,
             y = y.viewer().name("y")] __device__() mutable
            {
                using WarpReduceInt   = cub::WarpReduce<int, warp_size>;
                using WarpReduceFloat = cub::WarpReduce<Float, warp_size>;
                using WarpScanInt     = cub::WarpScan<int>;

                auto global_thread_id   = blockDim.x * blockIdx.x + threadIdx.x;
                auto thread_id_in_block = threadIdx.x;
                auto warp_id            = thread_id_in_block / warp_size;
                auto lane_id            = thread_id_in_block & (warp_size - 1);

                __shared__ union
                {
                    typename WarpReduceInt::TempStorage temp_storage_int[block_dim / warp_size];
                    typename WarpReduceFloat::TempStorage temp_storage_float[block_dim / warp_size];
                };

                int prev_i = -1;
                int next_i = -1;
                int i      = -1;

                Flags   flags;
                Vector3 vec;
                flags.is_cross_warp = 0;


                if(global_thread_id > 0 && global_thread_id < A.triplet_count())
                {
                    auto prev_triplet = A(global_thread_id - 1);
                    prev_i            = prev_triplet.row_index;
                }

                if(global_thread_id < A.triplet_count() - 1)
                {
                    auto next_triplet = A(global_thread_id + 1);
                    next_i            = next_triplet.row_index;
                }

                if(global_thread_id < A.triplet_count())
                {
                    auto Triplet = A(global_thread_id);
                    i            = Triplet.row_index;
                    auto j       = Triplet.col_index;

                    vec = Triplet.value * x.segment<N>(j * N).as_eigen();

                    flags.is_valid = 1;
                }
                else
                {
                    i = -1;
                    vec.setZero();
                    flags.is_valid      = 0;
                    flags.is_cross_warp = 0;
                }

                if(lane_id == 0)
                {
                    flags.is_head = 1;
                    // if this thread is the first thread in the warp
                    // check if the previous triplet is in the same row
                    // if so, this row crosses the warp boundary, we need use atomic add
                    flags.is_cross_warp = b2i(prev_i == i);
                }
                else
                {
                    flags.is_head = b2i(prev_i != i);  // must be 1 or 0, or the result is undefined

                    if(lane_id == warp_size - 1)
                    {
                        // if this thread is the last thread in the warp
                        // check if the next triplet is in the same row
                        // if so, this row crosses the warp boundary, we need use atomic add
                        flags.is_cross_warp = b2i(next_i == i);
                    }
                }

                flags.flags = WarpReduceInt(temp_storage_int[warp_id])
                                  .HeadSegmentedReduce(flags.flags,
                                                       flags.is_head,
                                                       [](uint32_t a, uint32_t b)
                                                       { return a + b; });

                vec.x() = WarpReduceFloat(temp_storage_float[warp_id])
                              .HeadSegmentedReduce(vec.x(),
                                                   flags.is_head,
                                                   [](Float a, Float b)
                                                   { return a + b; });

                vec.y() = WarpReduceFloat(temp_storage_float[warp_id])
                              .HeadSegmentedReduce(vec.y(),
                                                   flags.is_head,
                                                   [](Float a, Float b)
                                                   { return a + b; });

                vec.z() = WarpReduceFloat(temp_storage_float[warp_id])
                              .HeadSegmentedReduce(vec.z(),
                                                   flags.is_head,
                                                   [](Float a, Float b)
                                                   { return a + b; });


                // cub::WARP_SYNC(warp_mask);

                flags.is_head = b2i(flags.is_head && flags.is_valid);

                flags.b2i();
                int is_head_mask =
                    detail::bit_operation::WARP_BALLOT(flags.is_head, warp_mask);
                uint32_t offset = __fns(is_head_mask, 0, lane_id + 1);

                int valid_bit = (offset != ~0u);
                int shuffle_mask = detail::bit_operation::WARP_BALLOT(valid_bit, warp_mask);

                i = cub::ShuffleIndex<32>(i, offset, shuffle_mask);
                flags.flags = cub::ShuffleIndex<32>(flags.flags, offset, shuffle_mask);
                vec.x() = cub::ShuffleIndex<32>(vec.x(), offset, shuffle_mask);
                vec.y() = cub::ShuffleIndex<32>(vec.y(), offset, shuffle_mask);
                vec.z() = cub::ShuffleIndex<32>(vec.z(), offset, shuffle_mask);

                if(valid_bit && flags.is_head && flags.is_valid)
                {
                    auto seg_y  = y.segment<N>(i * N);
                    auto result = a * vec;

                    if(flags.is_cross_warp)
                    {
                        seg_y.atomic_add(result.eval());
                    }
                    else
                    {
                        seg_y.as_eigen() += result.eval();
                    }
                }
            });
}

void Spmv::rbk_sym_spmv(Float                           a,
                        muda::CBCOOMatrixView<Float, 3> A,
                        muda::CDenseVectorView<Float>   x,
                        Float                           b,
                        muda::DenseVectorView<Float>    y)

{
    using namespace muda;
    constexpr int N = 3;
    using T         = Float;

    if(b != 0)
    {
        muda::ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(y.size(),
                   [b = b, y = y.viewer().name("y")] __device__(int i) mutable
                   { y(i) = b * y(i); });
    }
    else
    {
        muda::BufferLaunch().fill<Float>(y.buffer_view(), 0);
    }

    constexpr int warp_size   = 32;
    constexpr int block_dim   = 256;
    int           block_count = (A.triplet_count() + block_dim - 1) / block_dim;

    muda::ParallelFor(block_count, block_dim)
        .file_line(__FILE__, __LINE__)
        .apply(A.triplet_count(),
               [a = a,
                A = A.viewer().name("A"),
                x = x.viewer().name("x"),
                b = b,
                y = y.viewer().name("y")] __device__(int idx) mutable
               {
                   using WarpReduceInt   = cub::WarpReduce<int, warp_size>;
                   using WarpReduceFloat = cub::WarpReduce<Float, warp_size>;
                   using WarpScanInt     = cub::WarpScan<int>;

                   auto global_thread_id   = idx;
                   auto thread_id_in_block = threadIdx.x;
                   auto warp_id            = thread_id_in_block / warp_size;
                   auto lane_id = thread_id_in_block & (warp_size - 1);

                   __shared__ union
                   {
                       typename WarpReduceInt::TempStorage temp_storage_int[block_dim / warp_size];
                       typename WarpReduceFloat::TempStorage temp_storage_float[block_dim / warp_size];
                   };

                   int     prev_i = -1;
                   int     i      = -1;
                   Flags   flags;
                   Vector3 vec;

                   // In symmtric version, we don't need to check the cross warp
                   flags.is_cross_warp = 0;

                   // set the previous row index
                   if(global_thread_id > 0)
                   {
                       auto prev_triplet = A(global_thread_id - 1);
                       prev_i            = prev_triplet.row_index;
                   }

                   {
                       auto Triplet = A(global_thread_id);
                       i            = Triplet.row_index;
                       auto j       = Triplet.col_index;

                       vec = Triplet.value * x.segment<N>(j * N).as_eigen();

                       flags.is_valid = 1;

                       if(i != j)  // process lower triangle
                       {
                           Vector3 vec_ = a * Triplet.value.transpose()
                                          * x.segment<N>(i * N).as_eigen();

                           y.segment<N>(j * N).atomic_add(vec_);
                       }
                   }

                   if(lane_id == 0)
                   {
                       flags.is_head = 1;
                   }
                   else
                   {
                       flags.is_head = b2i(prev_i != i);  // must be 1 or 0, or the result is undefined
                   }


                   // ----------------------------------- warp reduce ----------------------------------------------
                   vec.x() = WarpReduceFloat(temp_storage_float[warp_id])
                                 .HeadSegmentedReduce(vec.x(),
                                                      flags.is_head,
                                                      [](Float a, Float b)
                                                      { return a + b; });

                   vec.y() = WarpReduceFloat(temp_storage_float[warp_id])
                                 .HeadSegmentedReduce(vec.y(),
                                                      flags.is_head,
                                                      [](Float a, Float b)
                                                      { return a + b; });

                   vec.z() = WarpReduceFloat(temp_storage_float[warp_id])
                                 .HeadSegmentedReduce(vec.z(),
                                                      flags.is_head,
                                                      [](Float a, Float b)
                                                      { return a + b; });
                   // ----------------------------------- warp reduce -----------------------------------------------


                   if(flags.is_head)
                   {
                       auto seg_y  = y.segment<N>(i * N);
                       auto result = a * vec;

                       // Must use atomic add!
                       // Because the same row may be processed by different warps
                       seg_y.atomic_add(result.eval());
                   }
               });
}

void Spmv::rbk_sym_spmv_dot(Float                           a,
                            muda::CBCOOMatrixView<Float, 3> A,
                            muda::CDenseVectorView<Float>   x,
                            Float                           b,
                            muda::DenseVectorView<Float>    y,
                            muda::VarView<Float>            d_dot)
{
    using namespace muda;
    constexpr int N = 3;
    using T         = Float;

    if(b != 0)
    {
        muda::ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(y.size(),
                   [b = b, y = y.viewer().name("y")] __device__(int i) mutable
                   { y(i) = b * y(i); });
    }
    else
    {
        muda::BufferLaunch().fill<Float>(y.buffer_view(), 0);
    }

    cudaMemsetAsync(d_dot.data(), 0, sizeof(Float));

    constexpr int warp_size   = 32;
    constexpr int block_dim   = 256;
    int           block_count = (A.triplet_count() + block_dim - 1) / block_dim;

    int triplet_count = A.triplet_count();

    muda::Launch(block_count, block_dim)
        .file_line(__FILE__, __LINE__)
        .apply(
               [a     = a,
                A     = A.viewer().name("A"),
                x     = x.viewer().name("x"),
                b     = b,
                y     = y.viewer().name("y"),
                d_dot = d_dot.viewer().name("d_dot"),
                triplet_count] __device__() mutable
               {
                   using WarpReduceInt   = cub::WarpReduce<int, warp_size>;
                   using WarpReduceFloat = cub::WarpReduce<Float, warp_size>;

                   auto global_thread_id   = blockDim.x * blockIdx.x + threadIdx.x;
                   auto thread_id_in_block = threadIdx.x;
                   auto warp_id            = thread_id_in_block / warp_size;
                   auto lane_id = thread_id_in_block & (warp_size - 1);

                   __shared__ union
                   {
                       typename WarpReduceInt::TempStorage temp_storage_int[block_dim / warp_size];
                       typename WarpReduceFloat::TempStorage temp_storage_float[block_dim / warp_size];
                   };

                   int     prev_i = -1;
                   int     i      = -1;
                   Flags   flags;
                   Vector3 vec;
                   vec.setZero();
                   Float   dot_local = 0;

                   flags.is_cross_warp = 0;
                   flags.is_valid      = 0;
                   flags.is_head       = 0;

                   if(global_thread_id < triplet_count)
                   {
                       if(global_thread_id > 0)
                       {
                           auto prev_triplet = A(global_thread_id - 1);
                           prev_i            = prev_triplet.row_index;
                       }

                       auto Triplet = A(global_thread_id);
                       i            = Triplet.row_index;
                       auto j       = Triplet.col_index;

                       Eigen::Vector<T, N> x_j = x.segment<N>(j * N).as_eigen();
                       vec = Triplet.value * x_j;

                       flags.is_valid = 1;

                       if(i == j)
                       {
                           dot_local = a * x_j.dot(vec);
                       }
                       else
                       {
                           Eigen::Vector<T, N> x_i = x.segment<N>(i * N).as_eigen();
                           dot_local = 2.0 * a * x_i.dot(vec);

                           Vector3 vec_ = a * Triplet.value.transpose() * x_i;
                           y.segment<N>(j * N).atomic_add(vec_);
                       }

                       if(lane_id == 0)
                           flags.is_head = 1;
                       else
                           flags.is_head = b2i(prev_i != i);
                   }

                   vec.x() = WarpReduceFloat(temp_storage_float[warp_id])
                                 .HeadSegmentedReduce(vec.x(),
                                                      flags.is_head,
                                                      [](Float a, Float b)
                                                      { return a + b; });

                   vec.y() = WarpReduceFloat(temp_storage_float[warp_id])
                                 .HeadSegmentedReduce(vec.y(),
                                                      flags.is_head,
                                                      [](Float a, Float b)
                                                      { return a + b; });

                   vec.z() = WarpReduceFloat(temp_storage_float[warp_id])
                                 .HeadSegmentedReduce(vec.z(),
                                                      flags.is_head,
                                                      [](Float a, Float b)
                                                      { return a + b; });

                   if(flags.is_head && flags.is_valid)
                   {
                       auto seg_y  = y.segment<N>(i * N);
                       auto result = a * vec;
                       seg_y.atomic_add(result.eval());
                   }

                   dot_local = WarpReduceFloat(temp_storage_float[warp_id])
                                   .Sum(dot_local);
                   if(lane_id == 0)
                       atomicAdd(d_dot.data(), dot_local);
               });
}

void Spmv::cpu_sym_spmv(Float                           a,
                        muda::CBCOOMatrixView<Float, 3> A_view,
                        muda::CDenseVectorView<Float>   x,
                        Float                           b,
                        muda::DenseVectorView<Float>    y)
{
    using BlockMatrix      = Matrix3x3;
    constexpr int BlockDim = 3;

    // Get matrix dimensions
    int total_block_rows  = A_view.total_rows();
    int total_block_cols  = A_view.total_cols();
    int total_scalar_rows = total_block_rows * BlockDim;
    int total_scalar_cols = total_block_cols * BlockDim;

    // Copy triplet data from device to host
    std::vector<int>         row_indices_host(A_view.triplet_count());
    std::vector<int>         col_indices_host(A_view.triplet_count());
    std::vector<BlockMatrix> values_host(A_view.triplet_count());

    A_view.row_indices().copy_to(row_indices_host.data());
    A_view.col_indices().copy_to(col_indices_host.data());
    A_view.values().copy_to(values_host.data());

    // Build Eigen SparseMatrix from symmetric block-sparse representation
    std::vector<Eigen::Triplet<Float>> triplets;
    triplets.reserve(A_view.triplet_count() * BlockDim * BlockDim * 2);  // Reserve for symmetric expansion

    for(size_t t = 0; t < A_view.triplet_count(); ++t)
    {
        int                block_i = row_indices_host[t];
        int                block_j = col_indices_host[t];
        const BlockMatrix& block   = values_host[t];

        // Convert block indices to scalar indices
        int scalar_i_base = block_i * BlockDim;
        int scalar_j_base = block_j * BlockDim;

        // Add all entries from the block (upper triangular part)
        for(int bi = 0; bi < BlockDim; ++bi)
        {
            for(int bj = 0; bj < BlockDim; ++bj)
            {
                Float value    = block(bi, bj);
                int   scalar_i = scalar_i_base + bi;
                int   scalar_j = scalar_j_base + bj;

                triplets.emplace_back(scalar_i, scalar_j, value);
            }
        }

        // Since matrix is symmetric at block level, also add transpose block (lower triangular part)
        if(block_i != block_j)
        {
            BlockMatrix block_transpose = block.transpose();
            for(int bi = 0; bi < BlockDim; ++bi)
            {
                for(int bj = 0; bj < BlockDim; ++bj)
                {
                    Float value  = block_transpose(bi, bj);
                    int scalar_i = scalar_j_base + bi;  // Swapped base indices
                    int scalar_j = scalar_i_base + bj;

                    triplets.emplace_back(scalar_i, scalar_j, value);
                }
            }
        }
    }

    // Build Eigen SparseMatrix
    Eigen::SparseMatrix<Float> A_sparse(total_scalar_rows, total_scalar_cols);
    A_sparse.setFromTriplets(triplets.begin(), triplets.end());
    A_sparse.makeCompressed();

    // Copy vectors from device to host
    Eigen::VectorX<Float> x_host(x.size());
    Eigen::VectorX<Float> y_host(y.size());

    x.buffer_view().copy_to(x_host.data());
    y.buffer_view().copy_to(y_host.data());

    // Compute y = a * A * x + b * y using Eigen sparse matrix-vector multiplication
    y_host = a * A_sparse * x_host + b * y_host;

    // Copy result back to device
    y.buffer_view().copy_from(y_host.data());
}
}  // namespace uipc::backend::cuda
#endif
