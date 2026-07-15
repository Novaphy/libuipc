#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
#include <muda/cub/device/device_merge_sort.h>
#include <muda/cub/device/device_scan.h>
#include <muda/cub/device/device_radix_sort.h>
#include <muda/cub/device/device_reduce.h>
#include <muda/cub/device/device_select.h>
#include <cub/warp/warp_reduce.cuh>
#include <muda/ext/eigen/atomic.h>
#include <uipc/common/timer.h>
#include <uipc/common/type_define.h>
#include <algorithm/fast_segmental_reduce.h>
#include <muda/cub/device/device_partition.h>
#include <muda/cub/device/device_run_length_encode.h>
#include <fmt/core.h>
#include <algorithm>
#include <cstdio>
#include <limits>
#include <vector>
#include <utils/corex_phase_profile.h>

#include <thrust/device_ptr.h>
#include <algorithm/corex_matrix_converter_kernels.h>

namespace uipc::backend::cuda
{
namespace
{
inline int corex_index_sort_end_bit(SizeT count)
{
    if(count <= 1)
        return 1;
    --count;
    int bits = 0;
    while(count != 0)
    {
        ++bits;
        count >>= 1;
    }
    return std::min(32, std::max(1, bits));
}

inline int corex_hash_sort_end_bit(SizeT rows, SizeT cols)
{
    const int row_bits = corex_index_sort_end_bit(rows);
    const int col_bits = corex_index_sort_end_bit(cols);
    int       end_bit  = rows > 1 ? 32 + row_bits : col_bits;
    end_bit            = std::max(end_bit, col_bits);
    return std::min(64, std::max(1, end_bit));
}

inline int corex_compact_hash_sort_end_bit(SizeT rows, SizeT cols)
{
    if(rows == 0 || cols == 0)
        return 1;
    if(rows > std::numeric_limits<SizeT>::max() / cols)
        return 64;
    const SizeT key_count = rows * cols;
    if(key_count > (SizeT{1} << 32))
        return corex_hash_sort_end_bit(rows, cols);
    return std::min(64, corex_index_sort_end_bit(key_count));
}

struct CorexIntReadback
{
    int*         pinned = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t  ready = nullptr;

    CorexIntReadback()
    {
        int* tmp = nullptr;
        if(cudaMallocHost(reinterpret_cast<void**>(&tmp), sizeof(int)) == cudaSuccess)
            pinned = tmp;
        checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        checkCudaErrors(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    }
};

inline CorexIntReadback& corex_int_readback()
{
    static CorexIntReadback readback;
    return readback;
}

inline int corex_readback_int(const muda::DeviceVar<int>& value)
{
    auto& readback = corex_int_readback();

    if(!readback.pinned)
        return value;

    checkCudaErrors(cudaEventRecord(readback.ready, 0));
    checkCudaErrors(cudaStreamWaitEvent(readback.stream, readback.ready, 0));
    checkCudaErrors(cudaMemcpyAsync(readback.pinned,
                                    value.data(),
                                    sizeof(int),
                                    cudaMemcpyDeviceToHost,
                                    readback.stream));
    checkCudaErrors(cudaStreamSynchronize(readback.stream));
    return *readback.pinned;
}
}  // namespace

template <typename T, int N>
void MatrixConverter<T, N>::convert(const muda::DeviceTripletMatrix<T, N>& from,
                                    muda::DeviceBCOOMatrix<T, N>&          to)
{
    to.reshape(from.rows(), from.cols());
    to.unsafe_resize_triplets_no_construct(from.triplet_count());

    if(to.triplet_count() == 0)
        return;

    {
        corex_profile::ScopedPhase phase("matconv", "triplet_radix_sort_indices_blocks");
        _radix_sort_indices_and_blocks(from, to);
    }
    {
        corex_profile::ScopedPhase phase("matconv", "triplet_reduce_by_key_blocks");
        _make_unique_indices_and_blocks_reduce_by_key(from, to);
    }
}

template <typename T, int N>
void MatrixConverter<T, N>::_radix_sort_indices_and_blocks(
    const muda::DeviceTripletMatrix<T, N>& from, muda::DeviceBCOOMatrix<T, N>& to)
{
    using namespace muda;

    auto src_row_indices = from.row_indices();
    auto src_col_indices = from.col_indices();
    auto src_blocks      = from.values();

    loose_resize_no_construct(ij_hash_input, src_row_indices.size());
    loose_resize_no_construct(sort_index_input, src_row_indices.size());

    loose_resize_no_construct(ij_hash, src_row_indices.size());
    loose_resize_no_construct(sort_index, src_row_indices.size());
    loose_resize_no_construct(ij_pairs, src_row_indices.size());

    auto dst_row_indices = to.row_indices();
    auto dst_col_indices = to.col_indices();
    int n = static_cast<int>(src_row_indices.size());

    corex_matconv::launch_hash_ij_compact(
        n,
        thrust::raw_pointer_cast(src_row_indices.data()),
        thrust::raw_pointer_cast(src_col_indices.data()),
        static_cast<int>(from.cols()),
        thrust::raw_pointer_cast(ij_hash_input.data()),
        thrust::raw_pointer_cast(sort_index_input.data()));

    {
        corex_profile::ScopedPhase phase("matconv", "triplet_sort_pairs");
        DeviceRadixSort().SortPairs(ij_hash_input.data(),
                                    ij_hash.data(),
                                    sort_index_input.data(),
                                    sort_index.data(),
                                    ij_hash.size(),
                                    0,
                                    corex_compact_hash_sort_end_bit(from.rows(), from.cols()));
    }

    corex_matconv::launch_decode_hash_compact(
        n,
        thrust::raw_pointer_cast(ij_hash.data()),
        static_cast<int>(from.cols()),
        reinterpret_cast<int*>(thrust::raw_pointer_cast(ij_pairs.data())));

    // sort the block values
    {
        using BlockT = Eigen::Matrix<T, N, N>;
        loose_resize_no_construct(blocks_sorted, from.values().size());
        corex_matconv::launch_copy_sorted_blocks_3x3(
            n,
            reinterpret_cast<const corex_matconv::BlockT3*>(thrust::raw_pointer_cast(src_blocks.data())),
            thrust::raw_pointer_cast(sort_index.data()),
            reinterpret_cast<corex_matconv::BlockT3*>(thrust::raw_pointer_cast(blocks_sorted.data())));
    }
}

template <typename T, int N>
void MatrixConverter<T, N>::_radix_sort_indices_and_blocks(muda::DeviceBCOOMatrix<T, N>& to)
{
    using namespace muda;

    auto src_row_indices = to.row_indices();
    auto src_col_indices = to.col_indices();
    auto src_blocks      = to.values();

    loose_resize_no_construct(ij_hash_input, src_row_indices.size());
    loose_resize_no_construct(sort_index_input, src_row_indices.size());

    loose_resize_no_construct(ij_hash, src_row_indices.size());
    loose_resize_no_construct(sort_index, src_row_indices.size());
    loose_resize_no_construct(ij_pairs, src_row_indices.size());


    int n = static_cast<int>(src_row_indices.size());

    corex_matconv::launch_hash_ij_compact(
        n,
        thrust::raw_pointer_cast(src_row_indices.data()),
        thrust::raw_pointer_cast(src_col_indices.data()),
        static_cast<int>(to.cols()),
        thrust::raw_pointer_cast(ij_hash_input.data()),
        thrust::raw_pointer_cast(sort_index_input.data()));

    {
        corex_profile::ScopedPhase phase("matconv", "bcoo_sort_pairs");
        DeviceRadixSort().SortPairs(ij_hash_input.data(),
                                    ij_hash.data(),
                                    sort_index_input.data(),
                                    sort_index.data(),
                                    ij_hash.size(),
                                    0,
                                    corex_compact_hash_sort_end_bit(to.rows(), to.cols()));
    }

    auto dst_row_indices = to.row_indices();
    auto dst_col_indices = to.col_indices();

    corex_matconv::launch_decode_hash_compact(
        n,
        thrust::raw_pointer_cast(ij_hash.data()),
        static_cast<int>(to.cols()),
        reinterpret_cast<int*>(thrust::raw_pointer_cast(ij_pairs.data())));

    // sort the block values
    {
        using BlockT = Eigen::Matrix<T, N, N>;
        loose_resize_no_construct(blocks_sorted, to.values().size());
        corex_matconv::launch_copy_sorted_blocks_with_ij_3x3(
            n,
            reinterpret_cast<const corex_matconv::BlockT3*>(thrust::raw_pointer_cast(src_blocks.data())),
            thrust::raw_pointer_cast(sort_index.data()),
            reinterpret_cast<const int*>(thrust::raw_pointer_cast(ij_pairs.data())),
            reinterpret_cast<corex_matconv::BlockT3*>(thrust::raw_pointer_cast(blocks_sorted.data())),
            thrust::raw_pointer_cast(to.row_indices().data()),
            thrust::raw_pointer_cast(to.col_indices().data()));

        to.values().copy_from(blocks_sorted);
    }
}

template <typename T, int N>
void MatrixConverter<T, N>::_make_unique_indices_and_blocks_reduce_by_key(
    const muda::DeviceTripletMatrix<T, N>& from, muda::DeviceBCOOMatrix<T, N>& to)
{
    using namespace muda;

    static_assert(N == 3, "CoreX matrix ReduceByKey only supports 3x3 blocks");
    static_assert(std::is_same_v<T, Float>,
                  "CoreX matrix ReduceByKey block type must match uipc::Float");

    loose_resize_no_construct(unique_ij_pairs, ij_pairs.size());

    DeviceReduce().ReduceByKey(
        ij_pairs.data(),
        unique_ij_pairs.data(),
        blocks_sorted.data(),
        to.values().data(),
        count.data(),
        [] CUB_RUNTIME_FUNCTION(const BlockMatrix& l,
                                const BlockMatrix& r) -> BlockMatrix
        { return l + r; },
        ij_pairs.size());

    const int h_count = corex_readback_int(count);
    unique_ij_pairs.unsafe_resize_no_construct(h_count);
    to.unsafe_resize_triplets_no_construct(h_count);

    corex_matconv::launch_write_unique_ij(
        h_count,
        reinterpret_cast<const int*>(thrust::raw_pointer_cast(unique_ij_pairs.data())),
        thrust::raw_pointer_cast(to.row_indices().data()),
        thrust::raw_pointer_cast(to.col_indices().data()));
}

template <typename T, int N>
void MatrixConverter<T, N>::_make_unique_indices(const muda::DeviceTripletMatrix<T, N>& from,
                                                 muda::DeviceBCOOMatrix<T, N>& to)
{
    using namespace muda;

    auto row_indices = to.row_indices();
    auto col_indices = to.col_indices();

    loose_resize_no_construct(unique_ij_pairs, ij_pairs.size());
    loose_resize_no_construct(unique_counts, ij_pairs.size());


    {
        corex_profile::ScopedPhase phase("matconv", "triplet_rle_ij");
        DeviceRunLengthEncode().Encode(ij_pairs.data(),
                                       unique_ij_pairs.data(),
                                       unique_counts.data(),
                                       count.data(),
                                       ij_pairs.size());
    }

    int h_count = corex_readback_int(count);

    unique_ij_pairs.unsafe_resize_no_construct(h_count);
    unique_counts.unsafe_resize_no_construct(h_count);

    offsets.unsafe_resize_no_construct(unique_counts.size());

    {
        corex_profile::ScopedPhase phase("matconv", "triplet_unique_counts_scan");
        DeviceScan().ExclusiveSum(
            unique_counts.data(), offsets.data(), unique_counts.size());
    }


    corex_matconv::launch_write_unique_ij(
        static_cast<int>(unique_counts.size()),
        reinterpret_cast<const int*>(thrust::raw_pointer_cast(unique_ij_pairs.data())),
        thrust::raw_pointer_cast(row_indices.data()),
        thrust::raw_pointer_cast(col_indices.data()));

    to.unsafe_resize_triplets_no_construct(h_count);
}

template <typename T, int N>
void MatrixConverter<T, N>::_make_unique_block_warp_reduction(
    const muda::DeviceTripletMatrix<T, N>& from, muda::DeviceBCOOMatrix<T, N>& to)
{
    using namespace muda;

    loose_resize_no_construct(sorted_partition_output, ij_pairs.size());

    {
        corex_profile::ScopedPhase phase("matconv", "triplet_fill_segment_ids");
        corex_matconv::launch_fill_segment_ids_from_offsets(
            static_cast<int>(unique_counts.size()),
            thrust::raw_pointer_cast(unique_counts.data()),
            thrust::raw_pointer_cast(offsets.data()),
            thrust::raw_pointer_cast(sorted_partition_output.data()));
    }

    auto blocks = to.values();

    static_assert(N == 3, "CoreX matrix segmental reduce only supports 3x3 blocks");
    static_assert(std::is_same_v<T, Float>,
                  "CoreX matrix segmental reduce block type must match uipc::Float");
    corex_matconv::launch_segmental_reduce_3x3_blocked(
        static_cast<int>(blocks_sorted.size()),
        thrust::raw_pointer_cast(sorted_partition_output.data()),
        thrust::raw_pointer_cast(unique_counts.data()),
        thrust::raw_pointer_cast(offsets.data()),
        reinterpret_cast<const corex_matconv::BlockT3*>(
            thrust::raw_pointer_cast(blocks_sorted.data())),
        reinterpret_cast<corex_matconv::BlockT3*>(
            thrust::raw_pointer_cast(blocks.data())),
        static_cast<int>(blocks.size()));
}

template <typename T, int N>
void MatrixConverter<T, N>::convert(const muda::DeviceBCOOMatrix<T, N>& from,
                                    muda::DeviceBSRMatrix<T, N>&        to)
{
    // calculate the row offsets
    {
        corex_profile::ScopedPhase phase("matconv", "bcoo_calculate_block_offsets");
        _calculate_block_offsets(from, to);
    }

    to.resize(from.non_zeros());

    auto vals        = to.values();
    auto col_indices = to.col_indices();

    {
        corex_profile::ScopedPhase phase("matconv", "bcoo_to_bsr_copy");
        vals.copy_from(from.values());  // BCOO and BSR have the same block values
        col_indices.copy_from(from.col_indices());  // BCOO and BSR have the same block col indices
    }
}

template <typename T, int N>
void MatrixConverter<T, N>::_calculate_block_offsets(const muda::DeviceBCOOMatrix<T, N>& from,
                                                     muda::DeviceBSRMatrix<T, N>& to)
{
    using namespace muda;
    to.reshape(from.rows(), from.cols());


    auto dst_row_offsets = to.row_offsets();

    col_counts_per_row.resize(dst_row_offsets.size());
    col_counts_per_row.fill(0);

    unique_indices.resize(from.non_zeros());
    unique_counts.resize(from.non_zeros());


    // run length encode the row
    {
        corex_profile::ScopedPhase phase("matconv", "bcoo_row_rle");
        DeviceRunLengthEncode().Encode(from.row_indices().data(),
                                       unique_indices.data(),
                                       unique_counts.data(),
                                       count.data(),
                                       from.non_zeros());
    }
    int h_count = corex_readback_int(count);

    unique_indices.unsafe_resize_no_construct(h_count);
    unique_counts.unsafe_resize_no_construct(h_count);

    corex_matconv::launch_scatter_col_counts(
        static_cast<int>(unique_counts.size()),
        thrust::raw_pointer_cast(unique_indices.data()),
        thrust::raw_pointer_cast(unique_counts.data()),
        thrust::raw_pointer_cast(col_counts_per_row.data()));

    // calculate the offsets
    {
        corex_profile::ScopedPhase phase("matconv", "bcoo_row_offsets_scan");
        DeviceScan().ExclusiveSum(col_counts_per_row.data(),
                                  dst_row_offsets.data(),
                                  col_counts_per_row.size());
    }
}

template <typename T, int N>
void MatrixConverter<T, N>::convert(const muda::DeviceDoubletVector<T, N>& from,
                                    muda::DeviceBCOOVector<T, N>&          to)
{
    to.reshape(from.count());
    to.resize_doublets(from.doublet_count());

    if(to.doublet_count() == 0)
        return;

    {
        corex_profile::ScopedPhase phase("matconv", "doublet_radix_sort_indices_segments");
        _radix_sort_indices_and_segments(from, to);
    }
    {
        corex_profile::ScopedPhase phase("matconv", "doublet_make_unique_indices");
        _make_unique_indices(from, to);
    }
    {
        corex_profile::ScopedPhase phase("matconv", "doublet_segmental_reduce");
        _make_unique_segment_warp_reduction(from, to);
    }
}

template <typename T, int N>
void MatrixConverter<T, N>::_radix_sort_indices_and_segments(
    const muda::DeviceDoubletVector<T, N>& from, muda::DeviceBCOOVector<T, N>& to)
{
    using namespace muda;

    auto src_indices  = from.indices();
    auto src_segments = from.values();

    loose_resize_no_construct(indices_sorted, src_indices.size());
    loose_resize_no_construct(segments_sorted, src_segments.size());

    {
        corex_profile::ScopedPhase phase("matconv", "doublet_sort_pairs");
        DeviceRadixSort().SortPairs(src_indices.data(),
                                    indices_sorted.data(),
                                    src_segments.data(),
                                    segments_sorted.data(),
                                    src_indices.size());
    }
}

template <typename T, int N>
void MatrixConverter<T, N>::_make_unique_indices(const muda::DeviceDoubletVector<T, N>& from,
                                                 muda::DeviceBCOOVector<T, N>& to)
{
    using namespace muda;

    auto dst_indices  = to.indices();
    auto dst_segments = to.values();
    loose_resize_no_construct(unique_indices, indices_sorted.size());
    loose_resize_no_construct(unique_counts, indices_sorted.size());

    {
        corex_profile::ScopedPhase phase("matconv", "doublet_rle_indices");
        DeviceRunLengthEncode().Encode(indices_sorted.data(),
                                       unique_indices.data(),
                                       unique_counts.data(),
                                       count.data(),
                                       indices_sorted.size());
    }

    int h_count = corex_readback_int(count);

    unique_indices.unsafe_resize_no_construct(h_count);
    unique_counts.unsafe_resize_no_construct(h_count);

    offsets.unsafe_resize_no_construct(unique_counts.size());

    {
        corex_profile::ScopedPhase phase("matconv", "doublet_unique_counts_scan");
        DeviceScan().ExclusiveSum(
            unique_counts.data(), offsets.data(), unique_counts.size());
    }

    corex_matconv::launch_write_unique_indices(
        static_cast<int>(unique_counts.size()),
        thrust::raw_pointer_cast(unique_indices.data()),
        thrust::raw_pointer_cast(dst_indices.data()));

    to.resize_doublets(h_count);
}

template <typename T, int N>
void MatrixConverter<T, N>::_make_unique_segment_warp_reduction(
    const muda::DeviceDoubletVector<T, N>& from, muda::DeviceBCOOVector<T, N>& to)
{
    using namespace muda;

    loose_resize_no_construct(sorted_partition_output, indices_sorted.size());

    {
        corex_profile::ScopedPhase phase("matconv", "doublet_fill_segment_ids");
        corex_matconv::launch_fill_segment_ids_from_offsets(
            static_cast<int>(unique_counts.size()),
            thrust::raw_pointer_cast(unique_counts.data()),
            thrust::raw_pointer_cast(offsets.data()),
            thrust::raw_pointer_cast(sorted_partition_output.data()));
    }

    auto segments = to.values();

    static_assert(N == 3, "CoreX vector segmental reduce only supports 3x1 blocks");
    static_assert(std::is_same_v<T, Float>,
                  "CoreX vector segmental reduce vector type must match uipc::Float");
    corex_matconv::launch_segmental_reduce_3x1_blocked(
        static_cast<int>(segments_sorted.size()),
        thrust::raw_pointer_cast(sorted_partition_output.data()),
        thrust::raw_pointer_cast(unique_counts.data()),
        thrust::raw_pointer_cast(offsets.data()),
        reinterpret_cast<const corex_matconv::VecT3*>(
            thrust::raw_pointer_cast(segments_sorted.data())),
        reinterpret_cast<corex_matconv::VecT3*>(
            thrust::raw_pointer_cast(segments.data())),
        static_cast<int>(segments.size()));
}

template <typename T, int N>
void MatrixConverter<T, N>::ge2sym(muda::DeviceBCOOMatrix<T, N>& to)
{
    using namespace muda;

    // alias to reuse the memory
    auto& counts     = unique_counts;
    auto& block_temp = blocks_sorted;

    loose_resize_no_construct(counts, to.non_zeros());
    loose_resize_no_construct(offsets, to.non_zeros());
    loose_resize_no_construct(ij_pairs, to.non_zeros());
    loose_resize_no_construct(block_temp, to.values().size());

    // 0. find the upper triangular part (where i <= j)
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(to.non_zeros(),
               [row_indices = to.row_indices().cviewer().name("row_indices"),
                col_indices = to.col_indices().cviewer().name("col_indices"),
                ij_pairs    = ij_pairs.viewer().name("ij_pairs"),
                blocks      = to.values().cviewer().name("block_temp"),
                block_temp  = block_temp.viewer().name("block_temp"),
                counts = counts.viewer().name("counts")] __device__(int i) mutable
               {
                   counts(i)     = row_indices(i) <= col_indices(i) ? 1 : 0;
                   ij_pairs(i).x = row_indices(i);
                   ij_pairs(i).y = col_indices(i);
                   block_temp(i) = blocks(i);
               });

    // exclusive sum
    DeviceScan().ExclusiveSum(counts.data(), offsets.data(), counts.size());

    // set the values
    auto dst_block = to.values();

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(dst_block.size(),
               [dst_blocks  = dst_block.viewer().name("blocks"),
                src_blocks  = block_temp.cviewer().name("src_blocks"),
                ij_pairs    = ij_pairs.cviewer().name("ij_pairs"),
                row_indices = to.row_indices().viewer().name("row_indices"),
                col_indices = to.col_indices().viewer().name("col_indices"),
                counts      = counts.cviewer().name("counts"),
                offsets     = offsets.cviewer().name("offsets")] __device__(int i) mutable
               {
                   auto count  = counts(i);
                   auto offset = offsets(i);

                   if(count != 0)
                   {
                       dst_blocks(offset)  = src_blocks(i);
                       auto ij             = ij_pairs(i);
                       row_indices(offset) = ij.x;
                       col_indices(offset) = ij.y;
                   }
               });

    // Compute total_count robustly (avoid relying on "last thread writes" / viewer total_size).
    count = 0;
    if(counts.size() > 0)
    {
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(1,
                   [counts = counts.cviewer().name("counts"),
                    offsets = offsets.cviewer().name("offsets"),
                    total_count = count.viewer().name("total_count")] __device__(int) mutable
                   {
                       int last = (int)counts.total_size() - 1;
                       total_count = offsets(last) + counts(last);
                   });
    }

    int h_total_count = (int)count;

    to.resize_triplets(h_total_count);
}

template <typename T, int N>
void MatrixConverter<T, N>::ge2sym(muda::DeviceTripletMatrix<T, N>& to)
{
    using namespace muda;

    // alias to reuse the memory
    auto& counts     = unique_counts;
    auto& block_temp = blocks_sorted;

    loose_resize_no_construct(counts, to.triplet_count());
    loose_resize_no_construct(offsets, to.triplet_count());
    loose_resize_no_construct(ij_pairs, to.triplet_count());
    loose_resize_no_construct(block_temp, to.values().size());

    // 0. find the upper triangular part (where i <= j)
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(to.triplet_count(),
               [row_indices = to.row_indices().cviewer().name("row_indices"),
                col_indices = to.col_indices().cviewer().name("col_indices"),
                ij_pairs    = ij_pairs.viewer().name("ij_pairs"),
                blocks      = to.values().cviewer().name("block_temp"),
                block_temp  = block_temp.viewer().name("block_temp"),
                counts = counts.viewer().name("counts")] __device__(int i) mutable
               {
                   counts(i)     = row_indices(i) <= col_indices(i) ? 1 : 0;
                   ij_pairs(i).x = row_indices(i);
                   ij_pairs(i).y = col_indices(i);
                   block_temp(i) = blocks(i);
               });

    // exclusive sum
    DeviceScan().ExclusiveSum(counts.data(), offsets.data(), counts.size());

    // set the values
    auto dst_block = to.values();

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(dst_block.size(),
               [dst_blocks  = dst_block.viewer().name("blocks"),
                src_blocks  = block_temp.cviewer().name("src_blocks"),
                ij_pairs    = ij_pairs.cviewer().name("ij_pairs"),
                row_indices = to.row_indices().viewer().name("row_indices"),
                col_indices = to.col_indices().viewer().name("col_indices"),
                counts      = counts.cviewer().name("counts"),
                offsets     = offsets.cviewer().name("offsets")] __device__(int i) mutable
               {
                   auto count  = counts(i);
                   auto offset = offsets(i);

                   if(count != 0)
                   {
                       dst_blocks(offset)  = src_blocks(i);
                       auto ij             = ij_pairs(i);
                       row_indices(offset) = ij.x;
                       col_indices(offset) = ij.y;
                   }
               });

    // Compute total_count robustly (avoid relying on "last thread writes" / viewer total_size).
    count = 0;
    if(counts.size() > 0)
    {
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(1,
                   [counts = counts.cviewer().name("counts"),
                    offsets = offsets.cviewer().name("offsets"),
                    total_count = count.viewer().name("total_count")] __device__(int) mutable
                   {
                       int last = (int)counts.total_size() - 1;
                       total_count = offsets(last) + counts(last);
                   });
    }

    int h_total_count = (int)count;

    to.resize_triplets(h_total_count);
}


template <typename T, int N>
void MatrixConverter<T, N>::sym2ge(const muda::DeviceBCOOMatrix<T, N>& from,
                                   muda::DeviceBCOOMatrix<T, N>&       to)
{
    using namespace muda;

    auto sym_size = from.non_zeros();

    // alias to reuse the memory
    auto& flags                 = offsets;
    auto& partitioned           = blocks_sorted;
    auto& partition_index_input = sort_index_input;
    auto& partition_index       = sort_index;
    auto& selected_count        = count;
    auto  diag_count            = from.rows();


    loose_resize_no_construct(flags, sym_size);
    loose_resize_no_construct(partitioned, sym_size);
    loose_resize_no_construct(partition_index, sym_size);

    // setup select flag
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(sym_size,
               [flags       = flags.viewer().name("flags"),
                row_indices = from.row_indices().cviewer().name("row_indices"),
                col_indices = from.col_indices().cviewer().name("col_indices"),
                partition_index = partition_index_input.viewer().name(
                    "partitioned")] __device__(int i) mutable
               {
                   flags(i) = (row_indices(i) == col_indices(i)) ? 1 : 0;
                   partition_index(i) = i;
               });


    muda::DevicePartition().Flagged(partition_index_input.data(),
                                    flags.data(),
                                    partition_index.data(),
                                    selected_count.data(),
                                    sym_size);


    auto general_bcoo_size = 2 * (sym_size - diag_count) + diag_count;

    to.resize(from.rows(), from.cols(), general_bcoo_size);

    // copy blocks and ij
    // in this sequence:
    // [ Diag | Upper | Lower ]
    //
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(sym_size,
               [to   = to.viewer().name("to"),
                from = from.cviewer().name("from"),
                partition_index = partition_index.cviewer().name("partition_index"),
                diag_count = diag_count,
                sym_size   = sym_size] __device__(int i) mutable
               {
                   auto index = partition_index(i);
                   auto f     = from(index);
                   // diag + upper
                   to(i).write(f.row_index, f.col_index, f.value);
                   if(i >= diag_count)
                   {
                       // lower
                       to(i + sym_size - diag_count)
                           .write(f.col_index, f.row_index, f.value.transpose());
                   }
               });

    _radix_sort_indices_and_blocks(to);
}
}  // namespace uipc::backend::cuda
#else
#include <muda/cub/device/device_merge_sort.h>
#include <muda/cub/device/device_scan.h>
#include <muda/cub/device/device_radix_sort.h>
#include <muda/cub/device/device_select.h>
#include <cub/warp/warp_reduce.cuh>
#include <muda/ext/eigen/atomic.h>
#include <uipc/common/timer.h>
#include <algorithm/fast_segmental_reduce.h>
#include <muda/cub/device/device_partition.h>
#include <muda/cub/device/device_run_length_encode.h>

namespace uipc::backend::cuda
{
namespace
{
inline int corex_readback_int(const muda::DeviceVar<int>& value)
{
    return value;
}
}  // namespace

template <typename T, int N>
void MatrixConverter<T, N>::convert(const muda::DeviceTripletMatrix<T, N>& from,
                                    muda::DeviceBCOOMatrix<T, N>&          to)
{
    to.reshape(from.rows(), from.cols());
    to.resize_triplets(from.triplet_count());


    if(to.triplet_count() == 0)
        return;

    _radix_sort_indices_and_blocks(from, to);

    _make_unique_indices(from, to);
    _make_unique_block_warp_reduction(from, to);
}

template <typename T, int N>
void MatrixConverter<T, N>::_radix_sort_indices_and_blocks(
    const muda::DeviceTripletMatrix<T, N>& from, muda::DeviceBCOOMatrix<T, N>& to)
{
    using namespace muda;

    auto src_row_indices = from.row_indices();
    auto src_col_indices = from.col_indices();
    auto src_blocks      = from.values();

    loose_resize(ij_hash_input, src_row_indices.size());
    loose_resize(sort_index_input, src_row_indices.size());

    loose_resize(ij_hash, src_row_indices.size());
    loose_resize(sort_index, src_row_indices.size());
    ij_pairs.resize(src_row_indices.size());


    // hash ij
    ParallelFor(256)
        .file_line(__FILE__, __LINE__)
        .apply(src_row_indices.size(),
               [row_indices = src_row_indices.cviewer().name("row_indices"),
                col_indices = src_col_indices.cviewer().name("col_indices"),
                ij_hash     = ij_hash_input.viewer().name("ij_hash"),
                sort_index = sort_index_input.viewer().name("sort_index")] __device__(int i) mutable
               {
                   ij_hash(i) = (static_cast<uint64_t>(row_indices(i)) << 32)
                                + static_cast<uint64_t>(col_indices(i));
                   sort_index(i) = i;
               });

    DeviceRadixSort().SortPairs(ij_hash_input.data(),
                                ij_hash.data(),
                                sort_index_input.data(),
                                sort_index.data(),
                                ij_hash.size());

    // set ij_hash back to row_indices and col_indices

    auto dst_row_indices = to.row_indices();
    auto dst_col_indices = to.col_indices();

    ParallelFor(256)
        .file_line(__FILE__, __LINE__)
        .apply(dst_row_indices.size(),
               [ij_hash = ij_hash.viewer().name("ij_hash"),
                ij_pairs = ij_pairs.viewer().name("ij_pairs")] __device__(int i) mutable
               {
                   auto hash      = ij_hash(i);
                   auto row_index = static_cast<int>(hash >> 32);
                   auto col_index = static_cast<int>(hash & 0xFFFFFFFF);
                   ij_pairs(i).x  = row_index;
                   ij_pairs(i).y  = col_index;
               });

    // sort the block values

    {
        loose_resize(blocks_sorted, from.values().size());
        ParallelFor(256)
            .file_line(__FILE__, __LINE__)
            .apply(src_blocks.size(),
                   [src_blocks = src_blocks.cviewer().name("blocks"),
                    sort_index = sort_index.cviewer().name("sort_index"),
                    dst_blocks = blocks_sorted.viewer().name("values")] __device__(int i) mutable
                   { dst_blocks(i) = src_blocks(sort_index(i)); });
    }
}

template <typename T, int N>
void MatrixConverter<T, N>::_radix_sort_indices_and_blocks(muda::DeviceBCOOMatrix<T, N>& to)
{
    using namespace muda;

    auto src_row_indices = to.row_indices();
    auto src_col_indices = to.col_indices();
    auto src_blocks      = to.values();

    loose_resize(ij_hash_input, src_row_indices.size());
    loose_resize(sort_index_input, src_row_indices.size());

    loose_resize(ij_hash, src_row_indices.size());
    loose_resize(sort_index, src_row_indices.size());
    ij_pairs.resize(src_row_indices.size());


    // hash ij
    ParallelFor(256)
        .file_line(__FILE__, __LINE__)
        .apply(src_row_indices.size(),
               [row_indices = src_row_indices.cviewer().name("row_indices"),
                col_indices = src_col_indices.cviewer().name("col_indices"),
                ij_hash     = ij_hash_input.viewer().name("ij_hash"),
                sort_index = sort_index_input.viewer().name("sort_index")] __device__(int i) mutable
               {
                   ij_hash(i) =
                       (uint64_t{row_indices(i)} << 32) + uint64_t{col_indices(i)};
                   sort_index(i) = i;
               });

    DeviceRadixSort().SortPairs(ij_hash_input.data(),
                                ij_hash.data(),
                                sort_index_input.data(),
                                sort_index.data(),
                                ij_hash.size());

    // set ij_hash back to row_indices and col_indices

    auto dst_row_indices = to.row_indices();
    auto dst_col_indices = to.col_indices();

    ParallelFor(256)
        .file_line(__FILE__, __LINE__)
        .apply(dst_row_indices.size(),
               [ij_hash = ij_hash.viewer().name("ij_hash"),
                ij_pairs = ij_pairs.viewer().name("ij_pairs")] __device__(int i) mutable
               {
                   auto hash      = ij_hash(i);
                   auto row_index = int{hash >> 32};
                   auto col_index = int{hash & 0xFFFFFFFF};
                   ij_pairs(i).x  = row_index;
                   ij_pairs(i).y  = col_index;
               });

    // sort the block values

    {
        loose_resize(blocks_sorted, to.values().size());
        ParallelFor(256)
            .file_line(__FILE__, __LINE__)
            .apply(src_blocks.size(),
                   [src_blocks = src_blocks.cviewer().name("blocks"),
                    sort_index = sort_index.cviewer().name("sort_index"),
                    ij_pairs   = ij_pairs.cviewer().name("ij_pairs"),
                    dst_row    = to.row_indices().viewer().name("row_indices"),
                    dst_col    = to.col_indices().viewer().name("col_indices"),

                    dst_blocks = blocks_sorted.viewer().name("values")] __device__(int i) mutable
                   {
                       dst_blocks(i) = src_blocks(sort_index(i));
                       dst_row(i)    = ij_pairs(i).x;
                       dst_col(i)    = ij_pairs(i).y;
                   });

        to.values().copy_from(blocks_sorted);
    }
}

template <typename T, int N>
void MatrixConverter<T, N>::_make_unique_indices(const muda::DeviceTripletMatrix<T, N>& from,
                                                 muda::DeviceBCOOMatrix<T, N>& to)
{
    using namespace muda;

    auto row_indices = to.row_indices();
    auto col_indices = to.col_indices();

    loose_resize(unique_ij_pairs, ij_pairs.size());
    loose_resize(unique_counts, ij_pairs.size());


    DeviceRunLengthEncode().Encode(ij_pairs.data(),
                                   unique_ij_pairs.data(),
                                   unique_counts.data(),
                                   count.data(),
                                   ij_pairs.size());

    int h_count = corex_readback_int(count);

    unique_ij_pairs.resize(h_count);
    unique_counts.resize(h_count);

    offsets.resize(unique_counts.size());

    DeviceScan().ExclusiveSum(
        unique_counts.data(), offsets.data(), unique_counts.size());


    muda::ParallelFor(256)
        .file_line(__FILE__, __LINE__)
        .apply(unique_counts.size(),
               [unique_ij_pairs = unique_ij_pairs.viewer().name("unique_ij_pairs"),
                row_indices = row_indices.viewer().name("row_indices"),
                col_indices = col_indices.viewer().name("col_indices")] __device__(int i) mutable
               {
                   row_indices(i) = unique_ij_pairs(i).x;
                   col_indices(i) = unique_ij_pairs(i).y;
               });

    to.resize_triplets(h_count);
}

template <typename T, int N>
void MatrixConverter<T, N>::_make_unique_block_warp_reduction(
    const muda::DeviceTripletMatrix<T, N>& from, muda::DeviceBCOOMatrix<T, N>& to)
{
    using namespace muda;

    loose_resize(sorted_partition_input, ij_pairs.size());
    loose_resize(sorted_partition_output, ij_pairs.size());


    BufferLaunch().fill<int>(sorted_partition_input, 0);

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(unique_counts.size(),
               [sorted_partition = sorted_partition_input.viewer().name("sorted_partition"),
                unique_counts = unique_counts.viewer().name("unique_counts"),
                offsets = offsets.viewer().name("offsets")] __device__(int i) mutable
               {
                   auto offset = offsets(i);
                   auto count  = unique_counts(i);

                   sorted_partition(offset + count - 1) = 1;
               });

    // scatter
    DeviceScan().ExclusiveSum(sorted_partition_input.data(),
                              sorted_partition_output.data(),
                              sorted_partition_input.size());

    auto blocks = to.values();

    FastSegmentalReduce<>()
        .file_line(__FILE__, __LINE__)
        .reduce(std::as_const(sorted_partition_output).view(),
                std::as_const(blocks_sorted).view(),
                blocks);
}

template <typename T, int N>
void MatrixConverter<T, N>::convert(const muda::DeviceBCOOMatrix<T, N>& from,
                                    muda::DeviceBSRMatrix<T, N>&        to)
{
    // calculate the row offsets
    _calculate_block_offsets(from, to);

    to.resize(from.non_zeros());

    auto vals        = to.values();
    auto col_indices = to.col_indices();

    vals.copy_from(from.values());  // BCOO and BSR have the same block values
    col_indices.copy_from(from.col_indices());  // BCOO and BSR have the same block col indices
}

template <typename T, int N>
void MatrixConverter<T, N>::_calculate_block_offsets(const muda::DeviceBCOOMatrix<T, N>& from,
                                                     muda::DeviceBSRMatrix<T, N>& to)
{
    //Timer timer{__FUNCTION__};

    using namespace muda;
    to.reshape(from.rows(), from.cols());


    auto dst_row_offsets = to.row_offsets();

    col_counts_per_row.resize(dst_row_offsets.size());
    col_counts_per_row.fill(0);

    unique_indices.resize(from.non_zeros());
    unique_counts.resize(from.non_zeros());


    // run length encode the row
    DeviceRunLengthEncode().Encode(from.row_indices().data(),
                                   unique_indices.data(),
                                   unique_counts.data(),
                                   count.data(),
                                   from.non_zeros());
    int h_count = corex_readback_int(count);

    unique_indices.resize(h_count);
    unique_counts.resize(h_count);

    ParallelFor(256)
        .file_line(__FILE__, __LINE__)
        .apply(unique_counts.size(),
               [unique_indices     = unique_indices.cviewer().name("offset"),
                counts             = unique_counts.viewer().name("counts"),
                col_counts_per_row = col_counts_per_row.viewer().name(
                    "col_counts_per_row")] __device__(int i) mutable
               {
                   auto row                = unique_indices(i);
                   col_counts_per_row(row) = counts(i);
               });

    // calculate the offsets
    DeviceScan().ExclusiveSum(col_counts_per_row.data(),
                              dst_row_offsets.data(),
                              col_counts_per_row.size());
}

//using T         = Float;
//constexpr int N = 3;

template <typename T, int N>
void MatrixConverter<T, N>::convert(const muda::DeviceDoubletVector<T, N>& from,
                                    muda::DeviceBCOOVector<T, N>&          to)
{
    to.reshape(from.count());
    to.resize_doublets(from.doublet_count());

    if(to.doublet_count() == 0)
        return;

    _radix_sort_indices_and_segments(from, to);
    _make_unique_indices(from, to);
    _make_unique_segment_warp_reduction(from, to);
}

template <typename T, int N>
void MatrixConverter<T, N>::_radix_sort_indices_and_segments(
    const muda::DeviceDoubletVector<T, N>& from, muda::DeviceBCOOVector<T, N>& to)
{
    using namespace muda;

    auto src_indices  = from.indices();
    auto src_segments = from.values();

    loose_resize(indices_sorted, src_indices.size());
    loose_resize(segments_sorted, src_segments.size());

    DeviceRadixSort().SortPairs(src_indices.data(),
                                indices_sorted.data(),
                                src_segments.data(),
                                segments_sorted.data(),
                                src_indices.size());
}

template <typename T, int N>
void MatrixConverter<T, N>::_make_unique_indices(const muda::DeviceDoubletVector<T, N>& from,
                                                 muda::DeviceBCOOVector<T, N>& to)
{
    using namespace muda;

    auto dst_indices  = to.indices();
    auto dst_segments = to.values();
    loose_resize(unique_indices, indices_sorted.size());
    loose_resize(unique_counts, indices_sorted.size());

    DeviceRunLengthEncode().Encode(indices_sorted.data(),
                                   unique_indices.data(),
                                   unique_counts.data(),
                                   count.data(),
                                   indices_sorted.size());

    int h_count = corex_readback_int(count);

    unique_indices.resize(h_count);
    unique_counts.resize(h_count);

    offsets.resize(unique_counts.size());

    DeviceScan().ExclusiveSum(
        unique_counts.data(), offsets.data(), unique_counts.size());

    muda::ParallelFor(256)
        .file_line(__FILE__, __LINE__)
        .apply(unique_counts.size(),
               [unique_indices = unique_indices.viewer().name("unique_indices"),
                dst_indices = dst_indices.viewer().name("indices_sorted")] __device__(int i) mutable
               { dst_indices(i) = unique_indices(i); });

    to.resize_doublets(h_count);
}

template <typename T, int N>
void MatrixConverter<T, N>::_make_unique_segment_warp_reduction(
    const muda::DeviceDoubletVector<T, N>& from, muda::DeviceBCOOVector<T, N>& to)
{
    using namespace muda;

    loose_resize(sorted_partition_input, indices_sorted.size());
    loose_resize(sorted_partition_output, indices_sorted.size());

    BufferLaunch().fill<int>(sorted_partition_input, 0);

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(unique_counts.size(),
               [sorted_partition = sorted_partition_input.viewer().name("sorted_partition"),
                unique_counts = unique_counts.viewer().name("unique_counts"),
                offsets = offsets.viewer().name("offsets")] __device__(int i) mutable
               {
                   auto offset = offsets(i);
                   auto count  = unique_counts(i);

                   sorted_partition(offset + count - 1) = 1;
               });

    // scatter
    DeviceScan().ExclusiveSum(sorted_partition_input.data(),
                              sorted_partition_output.data(),
                              sorted_partition_input.size());

    auto segments = to.values();

    FastSegmentalReduce<64, 32>()
        .file_line(__FILE__, __LINE__)
        .reduce(std::as_const(sorted_partition_output).view(),
                std::as_const(segments_sorted).view(),
                segments);
}

template <typename T, int N>
void MatrixConverter<T, N>::ge2sym(muda::DeviceBCOOMatrix<T, N>& to)
{
    using namespace muda;

    // alias to reuse the memory
    auto& counts     = unique_counts;
    auto& block_temp = blocks_sorted;

    loose_resize(counts, to.non_zeros());
    loose_resize(offsets, to.non_zeros());
    loose_resize(ij_pairs, to.non_zeros());
    loose_resize(block_temp, to.values().size());

    // 0. find the upper triangular part (where i <= j)
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(to.non_zeros(),
               [row_indices = to.row_indices().cviewer().name("row_indices"),
                col_indices = to.col_indices().cviewer().name("col_indices"),
                ij_pairs    = ij_pairs.viewer().name("ij_pairs"),
                blocks      = to.values().cviewer().name("block_temp"),
                block_temp  = block_temp.viewer().name("block_temp"),
                counts = counts.viewer().name("counts")] __device__(int i) mutable
               {
                   counts(i)     = row_indices(i) <= col_indices(i) ? 1 : 0;
                   ij_pairs(i).x = row_indices(i);
                   ij_pairs(i).y = col_indices(i);
                   block_temp(i) = blocks(i);
               });

    // exclusive sum
    DeviceScan().ExclusiveSum(counts.data(), offsets.data(), counts.size());

    // set the values
    auto dst_block = to.values();

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(dst_block.size(),
               [dst_blocks  = dst_block.viewer().name("blocks"),
                src_blocks  = block_temp.cviewer().name("src_blocks"),
                ij_pairs    = ij_pairs.cviewer().name("ij_pairs"),
                row_indices = to.row_indices().viewer().name("row_indices"),
                col_indices = to.col_indices().viewer().name("col_indices"),
                counts      = counts.cviewer().name("counts"),
                offsets     = offsets.cviewer().name("offsets"),
                total_count = count.viewer().name("total_count")] __device__(int i) mutable
               {
                   auto count  = counts(i);
                   auto offset = offsets(i);

                   if(count != 0)
                   {
                       dst_blocks(offset)  = src_blocks(i);
                       auto ij             = ij_pairs(i);
                       row_indices(offset) = ij.x;
                       col_indices(offset) = ij.y;
                   }

                   if(i == offsets.total_size() - 1)
                   {
                       total_count = offsets(i) + counts(i);
                   }
               });

    int h_total_count = count;

    to.resize_triplets(h_total_count);
}

template <typename T, int N>
void MatrixConverter<T, N>::ge2sym(muda::DeviceTripletMatrix<T, N>& to)
{
    using namespace muda;

    // alias to reuse the memory
    auto& counts     = unique_counts;
    auto& block_temp = blocks_sorted;

    loose_resize(counts, to.triplet_count());
    loose_resize(offsets, to.triplet_count());
    loose_resize(ij_pairs, to.triplet_count());
    loose_resize(block_temp, to.values().size());

    // 0. find the upper triangular part (where i <= j)
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(to.triplet_count(),
               [row_indices = to.row_indices().cviewer().name("row_indices"),
                col_indices = to.col_indices().cviewer().name("col_indices"),
                ij_pairs    = ij_pairs.viewer().name("ij_pairs"),
                blocks      = to.values().cviewer().name("block_temp"),
                block_temp  = block_temp.viewer().name("block_temp"),
                counts = counts.viewer().name("counts")] __device__(int i) mutable
               {
                   counts(i)     = row_indices(i) <= col_indices(i) ? 1 : 0;
                   ij_pairs(i).x = row_indices(i);
                   ij_pairs(i).y = col_indices(i);
                   block_temp(i) = blocks(i);
               });

    // exclusive sum
    DeviceScan().ExclusiveSum(counts.data(), offsets.data(), counts.size());

    // set the values
    auto dst_block = to.values();

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(dst_block.size(),
               [dst_blocks  = dst_block.viewer().name("blocks"),
                src_blocks  = block_temp.cviewer().name("src_blocks"),
                ij_pairs    = ij_pairs.cviewer().name("ij_pairs"),
                row_indices = to.row_indices().viewer().name("row_indices"),
                col_indices = to.col_indices().viewer().name("col_indices"),
                counts      = counts.cviewer().name("counts"),
                offsets     = offsets.cviewer().name("offsets"),
                total_count = count.viewer().name("total_count")] __device__(int i) mutable
               {
                   auto count  = counts(i);
                   auto offset = offsets(i);

                   if(count != 0)
                   {
                       dst_blocks(offset)  = src_blocks(i);
                       auto ij             = ij_pairs(i);
                       row_indices(offset) = ij.x;
                       col_indices(offset) = ij.y;
                   }

                   if(i == offsets.total_size() - 1)
                   {
                       total_count = offsets(i) + counts(i);
                   }
               });

    int h_total_count = count;

    to.resize_triplets(h_total_count);
}


template <typename T, int N>
void MatrixConverter<T, N>::sym2ge(const muda::DeviceBCOOMatrix<T, N>& from,
                                   muda::DeviceBCOOMatrix<T, N>&       to)
{
    using namespace muda;

    auto sym_size = from.non_zeros();

    // alias to reuse the memory
    auto& flags                 = offsets;
    auto& partitioned           = blocks_sorted;
    auto& partition_index_input = sort_index_input;
    auto& partition_index       = sort_index;
    auto& selected_count        = count;
    auto  diag_count            = from.rows();


    loose_resize(flags, sym_size);
    loose_resize(partitioned, sym_size);
    loose_resize(partition_index, sym_size);

    // setup select flag
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(sym_size,
               [flags       = flags.viewer().name("flags"),
                row_indices = from.row_indices().cviewer().name("row_indices"),
                col_indices = from.col_indices().cviewer().name("col_indices"),
                partition_index = partition_index_input.viewer().name(
                    "partitioned")] __device__(int i) mutable
               {
                   flags(i) = (row_indices(i) == col_indices(i)) ? 1 : 0;
                   partition_index(i) = i;
               });


    muda::DevicePartition().Flagged(partition_index_input.data(),
                                    flags.data(),
                                    partition_index.data(),
                                    selected_count.data(),
                                    sym_size);


    auto general_bcoo_size = 2 * (sym_size - diag_count) + diag_count;

    to.resize(from.rows(), from.cols(), general_bcoo_size);

    // copy blocks and ij
    // in this sequence:
    // [ Diag | Upper | Lower ]
    //
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(sym_size,
               [to   = to.viewer().name("to"),
                from = from.cviewer().name("from"),
                partition_index = partition_index.cviewer().name("partition_index"),
                diag_count = diag_count,
                sym_size   = sym_size] __device__(int i) mutable
               {
                   auto index = partition_index(i);
                   auto f     = from(index);
                   // diag + upper
                   to(i).write(f.row_index, f.col_index, f.value);
                   if(i >= diag_count)
                   {
                       // lower
                       to(i + sym_size - diag_count)
                           .write(f.col_index, f.row_index, f.value.transpose());
                   }
               });

    _radix_sort_indices_and_blocks(to);
}
}  // namespace uipc::backend::cuda
#endif
