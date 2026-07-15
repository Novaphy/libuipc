#pragma once
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT

#include <cstdint>
#include <Eigen/Core>
#include <uipc/common/type_define.h>

namespace uipc::backend::cuda::corex_matconv
{
void launch_hash_ij(int N, const int* row_indices, const int* col_indices,
                    uint64_t* ij_hash, int* sort_index);

void launch_decode_hash(int N, const uint64_t* ij_hash, int* ij_pairs_xy);

void launch_hash_ij_compact(int N, const int* row_indices, const int* col_indices,
                            int col_count, uint64_t* ij_hash, int* sort_index);

void launch_decode_hash_compact(int N, const uint64_t* ij_hash, int col_count,
                                int* ij_pairs_xy);

void launch_write_unique_ij(int N, const int* unique_ij_pairs_xy,
                            int* row_indices, int* col_indices);

void launch_write_unique_ij_compact(int N, const uint64_t* unique_hashes,
                                    int col_count, int* row_indices,
                                    int* col_indices);

void launch_mark_partition(int N, const int* unique_counts,
                           const int* offsets, int* sorted_partition);

void launch_fill_segment_ids_from_offsets(int N, const int* unique_counts,
                                          const int* offsets, int* segment_ids);

void launch_write_unique_indices(int N, const int* unique_indices, int* dst_indices);

void launch_scatter_col_counts(int N, const int* unique_indices,
                               const int* counts, int* col_counts_per_row);

using BlockT3 = Eigen::Matrix<Float, 3, 3>;

void launch_copy_sorted_blocks_3x3(int N, const BlockT3* src_blocks,
                                    const int* sort_index, BlockT3* dst_blocks);

void launch_copy_sorted_blocks_with_ij_3x3(int N, const BlockT3* src_blocks,
                                            const int* sort_index,
                                            const int* ij_pairs_xy,
                                            BlockT3* dst_blocks,
                                            int* dst_row, int* dst_col);

void launch_segmental_reduce_3x3(int N, const int* segment_ids,
                                  const BlockT3* in_blocks, BlockT3* out_blocks,
                                  int out_count);

void launch_segmental_reduce_3x3_blocked(int N, const int* segment_ids,
                                         const int* unique_counts,
                                         const int* offsets,
                                         const BlockT3* in_blocks,
                                         BlockT3* out_blocks,
                                         int out_count);

using VecT3 = Eigen::Matrix<Float, 3, 1>;

void launch_segmental_reduce_3x1(int N, const int* segment_ids,
                                  const VecT3* in_vecs, VecT3* out_vecs,
                                  int out_count);

void launch_segmental_reduce_3x1_blocked(int N, const int* segment_ids,
                                         const int* unique_counts,
                                         const int* offsets,
                                         const VecT3* in_vecs,
                                         VecT3* out_vecs,
                                         int out_count);

}  // namespace uipc::backend::cuda::corex_matconv

#endif
