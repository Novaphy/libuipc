#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT

#include <affine_body/abd_dytopo_hessian_reducer.h>
#include <muda/muda.h>
#include <muda/cub/device/device_radix_sort.h>
#include <muda/cub/device/device_run_length_encode.h>
#include <muda/cub/device/device_scan.h>
#include <muda/ext/eigen/atomic.h>
#include <muda/buffer/buffer_launch.h>
#include <utils/corex_phase_profile.h>
#include <cstdlib>
#include <algorithm>
#include <limits>

namespace uipc::backend::cuda
{
namespace
{
bool corex_abd_dytopo_direct_body_enabled()
{
    static const bool enabled = []
    {
        const char* env = std::getenv("UIPC_COREX_ABD_DYTOPO_DIRECT_BODY");
        return env && env[0] != '\0' && env[0] != '0';
    }();
    return enabled;
}

inline int corex_abd_hash_sort_end_bit(SizeT rows, SizeT cols)
{
    auto bits_needed = [](SizeT n) -> int
    {
        if(n <= 1)
            return 1;
        --n;
        int bits = 0;
        while(n != 0)
        {
            ++bits;
            n >>= 1;
        }
        return bits;
    };

    const int row_bits = bits_needed(rows);
    const int col_bits = bits_needed(cols);
    int       end_bit  = rows > 1 ? 32 + row_bits : col_bits;
    end_bit            = std::max(end_bit, col_bits);
    return std::min(64, std::max(1, end_bit));
}

inline int corex_abd_compact_hash_sort_end_bit(SizeT rows, SizeT cols)
{
    if(rows == 0 || cols == 0)
        return 1;
    if(rows > std::numeric_limits<SizeT>::max() / cols)
        return 64;
    const SizeT key_count = rows * cols;
    if(key_count <= (SizeT{1} << 32))
    {
        if(key_count <= 1)
            return 1;
        SizeT n = key_count - 1;
        int   bits = 0;
        while(n != 0)
        {
            ++bits;
            n >>= 1;
        }
        return std::min(64, std::max(1, bits));
    }
    return corex_abd_hash_sort_end_bit(rows, cols);
}

inline int corex_abd_readback_int(const muda::DeviceVar<int>& value)
{
    static int* pinned = nullptr;
    if(!pinned)
    {
        int* tmp = nullptr;
        if(cudaMallocHost(reinterpret_cast<void**>(&tmp), sizeof(int)) == cudaSuccess)
            pinned = tmp;
    }

    if(!pinned)
        return value;

    checkCudaErrors(cudaMemcpyAsync(
        pinned, value.data(), sizeof(int), cudaMemcpyDeviceToHost, 0));
    checkCudaErrors(cudaStreamSynchronize(0));
    return *pinned;
}

UIPC_HOST UIPC_DEVICE Matrix12x12 make_abd_contact_block(const ABDJacobi& Ji,
                                                         const Matrix3x3& H,
                                                         const ABDJacobi& Jj)
{
    return ABDJacobi::JT_H_J(Ji.T(), H, Jj);
}

__global__ void kernel_pack_node_triplets(int                               n,
                                          int                               vertex_offset,
                                          const int*                        src_rows,
                                          const int*                        src_cols,
                                          const Matrix3x3*                  src_vals,
                                          int*                              dst_rows,
                                          int*                              dst_cols,
                                          Matrix3x3*                        dst_vals)
{
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    if(I >= n) return;

    int i = src_rows[I] - vertex_offset;
    int j = src_cols[I] - vertex_offset;
    Matrix3x3 H = src_vals[I];

    if(i <= j)
    {
        dst_rows[I] = i;
        dst_cols[I] = j;
        dst_vals[I] = H;
    }
    else
    {
        dst_rows[I] = j;
        dst_cols[I] = i;
        dst_vals[I] = H.transpose();
    }
}

__global__ void kernel_map_node_to_body_triplets(int               n,
                                                 const int*        node_rows,
                                                 const int*        node_cols,
                                                 const Matrix3x3*  node_vals,
                                                 const IndexT*     vertex_to_body,
                                                 const ABDJacobi*  vertex_to_jacobi,
                                                 const IndexT*     body_is_fixed,
                                                 int*              body_rows,
                                                 int*              body_cols,
                                                 Matrix12x12*      body_vals,
                                                 Matrix12x12*      diag_hessian)
{
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    if(I >= n) return;

    int i = node_rows[I];
    int j = node_cols[I];

    IndexT body_i = vertex_to_body[i];
    IndexT body_j = vertex_to_body[j];

    Matrix12x12 H12;
    H12.setZero();

    if(body_is_fixed[body_i] || body_is_fixed[body_j])
    {
        body_rows[I] = 0;
        body_cols[I] = 0;
        body_vals[I] = H12;
        return;
    }

    const auto& Ji = vertex_to_jacobi[i];
    const auto& Jj = vertex_to_jacobi[j];
    const auto& H3 = node_vals[I];

    if(body_i == body_j)
    {
        if(i == j)
        {
            H12 = make_abd_contact_block(Ji, H3, Jj);
        }
        else
        {
            H12 = make_abd_contact_block(Ji, H3, Jj)
                  + make_abd_contact_block(Jj, H3.transpose(), Ji);
        }

        muda::eigen::atomic_add(diag_hessian[body_i], H12);
        body_rows[I] = body_i;
        body_cols[I] = body_i;
        body_vals[I] = H12;
    }
    else if(body_i < body_j)
    {
        H12 = make_abd_contact_block(Ji, H3, Jj);
        body_rows[I] = body_i;
        body_cols[I] = body_j;
        body_vals[I] = H12;
    }
    else
    {
        H12 = make_abd_contact_block(Jj, H3.transpose(), Ji);
        body_rows[I] = body_j;
        body_cols[I] = body_i;
        body_vals[I] = H12;
    }
}

__global__ void kernel_map_raw_to_body_triplets_compact(int               n,
                                                        int               vertex_offset,
                                                        const int*        raw_rows,
                                                        const int*        raw_cols,
                                                        const Matrix3x3*  raw_vals,
                                                        const IndexT*     vertex_to_body,
                                                        const ABDJacobi*  vertex_to_jacobi,
                                                        const IndexT*     body_is_fixed,
                                                        int*              valid_count,
                                                        int*              body_rows,
                                                        int*              body_cols,
                                                        Matrix12x12*      body_vals,
                                                        Matrix12x12*      diag_hessian)
{
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    if(I >= n) return;

    int i = raw_rows[I] - vertex_offset;
    int j = raw_cols[I] - vertex_offset;

    IndexT body_i = vertex_to_body[i];
    IndexT body_j = vertex_to_body[j];

    if(body_is_fixed[body_i] || body_is_fixed[body_j])
        return;

    const auto& Ji = vertex_to_jacobi[i];
    const auto& Jj = vertex_to_jacobi[j];
    const auto& H3 = raw_vals[I];

    Matrix12x12 H12;
    int         row;
    int         col;

    if(body_i == body_j)
    {
        if(i == j)
            H12 = make_abd_contact_block(Ji, H3, Jj);
        else
            H12 = make_abd_contact_block(Ji, H3, Jj)
                  + make_abd_contact_block(Jj, H3.transpose(), Ji);

        muda::eigen::atomic_add(diag_hessian[body_i], H12);
        row = body_i;
        col = body_i;
    }
    else if(body_i < body_j)
    {
        H12 = make_abd_contact_block(Ji, H3, Jj);
        row = body_i;
        col = body_j;
    }
    else
    {
        H12 = make_abd_contact_block(Jj, H3.transpose(), Ji);
        row = body_j;
        col = body_i;
    }

    int out = atomicAdd(valid_count, 1);
    body_rows[out] = row;
    body_cols[out] = col;
    body_vals[out] = H12;
}

__global__ void kernel_copy_sorted_blocks_12x12(int                n,
                                                const Matrix12x12* src_vals,
                                                const int*         sort_index,
                                                Matrix12x12*       dst_vals)
{
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    if(I >= n) return;
    dst_vals[I] = src_vals[sort_index[I]];
}

__global__ void kernel_reduce_sorted_blocks_12x12(
    int                           n,
    const uint64_t*               unique_hashes,
    int                           col_count,
    const int*                    unique_counts,
    const int*                    offsets,
    const Matrix12x12*            src_vals,
    const int*                    sort_index,
    int*                          dst_rows,
    int*                          dst_cols,
    Matrix12x12*                  dst_vals)
{
    int seg = blockIdx.x;
    if(seg >= n) return;

    if(threadIdx.x == 0)
    {
        const uint64_t hash = unique_hashes[seg];
        dst_rows[seg] = static_cast<int>(hash / static_cast<uint64_t>(col_count));
        dst_cols[seg] = static_cast<int>(hash % static_cast<uint64_t>(col_count));
    }

    if(threadIdx.x >= 12 * 12)
        return;

    const int begin = offsets[seg];
    const int end   = begin + unique_counts[seg];
    const int k     = threadIdx.x;

    Float sum = 0;
    for(int I = begin; I < end; ++I)
    {
        const Float* src = reinterpret_cast<const Float*>(src_vals + sort_index[I]);
        sum += src[k];
    }

    Float* dst = reinterpret_cast<Float*>(dst_vals + seg);
    dst[k] = sum;
}

__global__ void kernel_abd_srbk_spmv(int               n,
                                     Float             a,
                                     const int*        rows,
                                     const int*        cols,
                                     const Matrix12x12* vals,
                                     const Float*      x,
                                     Float*            y)
{
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    if(I >= n) return;

    const int row = rows[I];
    const int col = cols[I];
    const auto& H = vals[I];

    Vector12 x_col;
    const int col_base = col * 12;
    const int row_base = row * 12;

    for(int k = 0; k < 12; ++k)
    {
        x_col(k) = x[col_base + k];
    }

    const Vector12 y_row = a * (H * x_col);
    for(int k = 0; k < 12; ++k)
        atomicAdd(&y[row_base + k], y_row(k));

    if(row != col)
    {
        Vector12 x_row;
        for(int k = 0; k < 12; ++k)
            x_row(k) = x[row_base + k];
        const Vector12 y_col = a * (H.transpose() * x_row);
        for(int k = 0; k < 12; ++k)
            atomicAdd(&y[col_base + k], y_col(k));
    }
}

__global__ void kernel_abd_srbk_spmv_rows(int               n,
                                          Float             a,
                                          const int*        rows,
                                          const int*        cols,
                                          const Matrix12x12* vals,
                                          const Float*      x,
                                          Float*            y)
{
    int tid       = blockIdx.x * blockDim.x + threadIdx.x;
    int block_id  = tid / 12;
    int local_row = tid - block_id * 12;
    if(block_id >= n)
        return;

    const int row = rows[block_id];
    const int col = cols[block_id];
    const auto& H = vals[block_id];

    const int row_base = row * 12;
    const int col_base = col * 12;

    Float y_row = 0;
    for(int k = 0; k < 12; ++k)
        y_row += H(local_row, k) * x[col_base + k];
    atomicAdd(&y[row_base + local_row], a * y_row);

    if(row != col)
    {
        Float y_col = 0;
        for(int k = 0; k < 12; ++k)
            y_col += H(k, local_row) * x[row_base + k];
        atomicAdd(&y[col_base + local_row], a * y_col);
    }
}
}  // namespace

void ABDDyTopoHessianReducer::reduce_body_triplets(IndexT body_count)
{
    using namespace muda;

    const int body_triplet_count = static_cast<int>(m_body_triplets.triplet_count());
    m_body_pair_count = 0;
    m_body_blocks.resize(body_count, body_count, 0);

    if(body_triplet_count == 0)
        return;

    loose_resize(m_body_hash_input, body_triplet_count);
    loose_resize(m_body_hash, body_triplet_count);
    loose_resize(m_body_sort_index_input, body_triplet_count);
    loose_resize(m_body_sort_index, body_triplet_count);

    {
        corex_profile::ScopedPhase phase("abd_dytopo_reducer", "body_hash_pairs");
        corex_matconv::launch_hash_ij_compact(body_triplet_count,
                                              m_body_triplets.row_indices().data(),
                                              m_body_triplets.col_indices().data(),
                                              static_cast<int>(body_count),
                                              m_body_hash_input.data(),
                                              m_body_sort_index_input.data());
    }

    {
        corex_profile::ScopedPhase phase("abd_dytopo_reducer", "body_sort_pairs");
        DeviceRadixSort().SortPairs(m_body_hash_input.data(),
                                    m_body_hash.data(),
                                    m_body_sort_index_input.data(),
                                    m_body_sort_index.data(),
                                    body_triplet_count,
                                    0,
                                    corex_abd_compact_hash_sort_end_bit(body_count, body_count));
    }

    loose_resize(m_body_unique_hashes, body_triplet_count);
    loose_resize(m_body_unique_counts, body_triplet_count);

    {
        corex_profile::ScopedPhase phase("abd_dytopo_reducer", "body_rle_pairs");
        DeviceRunLengthEncode().Encode(m_body_hash.data(),
                                       m_body_unique_hashes.data(),
                                       m_body_unique_counts.data(),
                                       m_body_unique_count_var.data(),
                                       body_triplet_count);
    }

    const int unique_count = corex_abd_readback_int(m_body_unique_count_var);
    m_body_pair_count = unique_count;
    if(unique_count == 0)
        return;

    m_body_unique_hashes.unsafe_resize_no_construct(unique_count);
    m_body_unique_counts.unsafe_resize_no_construct(unique_count);
    m_body_offsets.unsafe_resize_no_construct(unique_count);

    {
        corex_profile::ScopedPhase phase("abd_dytopo_reducer", "body_unique_counts_scan");
        DeviceScan().ExclusiveSum(m_body_unique_counts.data(),
                                  m_body_offsets.data(),
                                  unique_count);
    }

    m_body_blocks.reshape(body_count, body_count);
    m_body_blocks.unsafe_resize_triplets_no_construct(unique_count);

    {
        corex_profile::ScopedPhase phase("abd_dytopo_reducer", "body_reduce_12x12");
        constexpr int kBlockEntries = 256;
        kernel_reduce_sorted_blocks_12x12<<<unique_count, kBlockEntries>>>(
            unique_count,
            m_body_unique_hashes.data(),
            static_cast<int>(body_count),
            m_body_unique_counts.data(),
            m_body_offsets.data(),
            m_body_triplets.values().data(),
            m_body_sort_index.data(),
            m_body_blocks.row_indices().data(),
            m_body_blocks.col_indices().data(),
            m_body_blocks.values().data());
        checkCudaErrors(cudaGetLastError());
    }
}

void ABDDyTopoHessianReducer::build(muda::CTripletMatrixView<Float, 3> raw_hessians,
                                    IndexT                            vertex_offset,
                                    IndexT                            body_count,
                                    muda::CBufferView<IndexT>         vertex_to_body,
                                    muda::CBufferView<ABDJacobi>      vertex_to_jacobi,
                                    muda::CBufferView<IndexT>         body_is_fixed,
                                    muda::BufferView<Matrix12x12>     diag_hessian)
{
    const int raw_count = raw_hessians.triplet_count();
    m_raw_hessian_count = raw_count;
    m_node_pair_count   = 0;
    m_body_pair_count   = 0;

    m_node_triplets.reshape(raw_hessians.total_rows(), raw_hessians.total_cols());
    m_node_triplets.unsafe_resize_triplets_no_construct(raw_count);
    m_node_blocks.reshape(raw_hessians.total_rows(), raw_hessians.total_cols());
    m_node_blocks.unsafe_resize_triplets_no_construct(0);
    m_body_triplets.reshape(body_count, body_count);
    m_body_triplets.unsafe_resize_triplets_no_construct(0);
    m_body_blocks.reshape(body_count, body_count);
    m_body_blocks.unsafe_resize_triplets_no_construct(0);

    if(raw_count == 0)
        return;

    if(corex_abd_dytopo_direct_body_enabled())
    {
        m_body_triplets.reshape(body_count, body_count);
        m_body_triplets.unsafe_resize_triplets_no_construct(raw_count);

        {
            corex_profile::ScopedPhase phase("abd_dytopo_reducer", "map_raw_to_body_pairs");
            checkCudaErrors(cudaMemsetAsync(m_body_triplet_count_var.data(), 0, sizeof(int)));
            constexpr int kBlk = 256;
            kernel_map_raw_to_body_triplets_compact<<<(raw_count + kBlk - 1) / kBlk, kBlk>>>(
                raw_count,
                static_cast<int>(vertex_offset),
                raw_hessians.row_indices().data(),
                raw_hessians.col_indices().data(),
                raw_hessians.values().data(),
                vertex_to_body.data(),
                vertex_to_jacobi.data(),
                body_is_fixed.data(),
                m_body_triplet_count_var.data(),
                m_body_triplets.row_indices().data(),
                m_body_triplets.col_indices().data(),
                m_body_triplets.values().data(),
                diag_hessian.data());
            checkCudaErrors(cudaGetLastError());
        }

        const int body_triplet_count = corex_abd_readback_int(m_body_triplet_count_var);
        if(body_triplet_count == 0)
        {
            m_body_triplets.resize(body_count, body_count, 0);
            return;
        }

        m_body_triplets.reshape(body_count, body_count);
        m_body_triplets.unsafe_resize_triplets_no_construct(body_triplet_count);

        {
            corex_profile::ScopedPhase phase("abd_dytopo_reducer", "reduce_body_pairs");
            reduce_body_triplets(body_count);
        }
        return;
    }

    {
        corex_profile::ScopedPhase phase("abd_dytopo_reducer", "pack_node_pairs");
        constexpr int kBlk = 256;
        kernel_pack_node_triplets<<<(raw_count + kBlk - 1) / kBlk, kBlk>>>(
            raw_count,
            static_cast<int>(vertex_offset),
            raw_hessians.row_indices().data(),
            raw_hessians.col_indices().data(),
            raw_hessians.values().data(),
            m_node_triplets.row_indices().data(),
            m_node_triplets.col_indices().data(),
            m_node_triplets.values().data());
        checkCudaErrors(cudaGetLastError());
    }

    {
        corex_profile::ScopedPhase phase("abd_dytopo_reducer", "reduce_node_pairs");
        m_node_converter.convert(m_node_triplets, m_node_blocks);
        m_node_pair_count = m_node_blocks.triplet_count();
    }

    const int node_pair_count = static_cast<int>(m_node_pair_count);
    if(node_pair_count == 0)
        return;

    m_body_triplets.reshape(body_count, body_count);
    m_body_triplets.unsafe_resize_triplets_no_construct(node_pair_count);

    {
        corex_profile::ScopedPhase phase("abd_dytopo_reducer", "map_node_to_body_pairs");
        constexpr int kBlk = 256;
        kernel_map_node_to_body_triplets<<<(node_pair_count + kBlk - 1) / kBlk, kBlk>>>(
            node_pair_count,
            m_node_blocks.row_indices().data(),
            m_node_blocks.col_indices().data(),
            m_node_blocks.values().data(),
            vertex_to_body.data(),
            vertex_to_jacobi.data(),
            body_is_fixed.data(),
            m_body_triplets.row_indices().data(),
            m_body_triplets.col_indices().data(),
            m_body_triplets.values().data(),
            diag_hessian.data());
        checkCudaErrors(cudaGetLastError());
    }

    {
        corex_profile::ScopedPhase phase("abd_dytopo_reducer", "reduce_body_pairs");
        reduce_body_triplets(body_count);
    }
}

void ABDDyTopoHessianReducer::spmv(Float                         a,
                                   muda::CDenseVectorView<Float> x,
                                   muda::DenseVectorView<Float>  y) const
{
    const int block_count = static_cast<int>(m_body_blocks.triplet_count());
    if(block_count == 0)
        return;

    corex_profile::ScopedPhase phase("abd_dytopo_reducer", "srbk_body_spmv");
    constexpr int kBlk = 256;
    const int     work = block_count * 12;
    kernel_abd_srbk_spmv_rows<<<(work + kBlk - 1) / kBlk, kBlk>>>(
        block_count,
        a,
        m_body_blocks.row_indices().data(),
        m_body_blocks.col_indices().data(),
        m_body_blocks.values().data(),
        x.data(),
        y.data());
    checkCudaErrors(cudaGetLastError());
}
}  // namespace uipc::backend::cuda

#endif
