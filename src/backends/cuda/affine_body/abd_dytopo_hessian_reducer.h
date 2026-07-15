#pragma once

#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT

#include <type_define.h>
#include <algorithm/matrix_converter.h>
#include <affine_body/abd_jacobi_matrix_corex.h>
#include <muda/buffer/device_var.h>
#include <muda/ext/linear_system/device_bcoo_matrix.h>
#include <muda/ext/linear_system/device_triplet_matrix.h>
#include <muda/ext/linear_system/dense_vector_view.h>

namespace uipc::backend::cuda
{
class ABDDyTopoHessianReducer
{
  public:
    void build(muda::CTripletMatrixView<Float, 3> raw_hessians,
               IndexT                            vertex_offset,
               IndexT                            body_count,
               muda::CBufferView<IndexT>         vertex_to_body,
               muda::CBufferView<ABDJacobi>      vertex_to_jacobi,
               muda::CBufferView<IndexT>         body_is_fixed,
               muda::BufferView<Matrix12x12>     diag_hessian);

    void spmv(Float                        a,
              muda::CDenseVectorView<Float> x,
              muda::DenseVectorView<Float>  y) const;

    SizeT raw_hessian_count() const noexcept { return m_raw_hessian_count; }
    SizeT node_pair_count() const noexcept { return m_node_pair_count; }
    SizeT body_pair_count() const noexcept { return m_body_pair_count; }

    muda::CBCOOMatrixView<Float, 12> body_blocks() const noexcept
    {
        return m_body_blocks.cview();
    }

  private:
    MatrixConverter<Float, 3>  m_node_converter;

    muda::DeviceTripletMatrix<Float, 3>  m_node_triplets;
    muda::DeviceBCOOMatrix<Float, 3>     m_node_blocks;
    muda::DeviceTripletMatrix<Float, 12> m_body_triplets;
    muda::DeviceBCOOMatrix<Float, 12>    m_body_blocks;

    muda::DeviceBuffer<uint64_t> m_body_hash_input;
    muda::DeviceBuffer<uint64_t> m_body_hash;
    muda::DeviceBuffer<int>      m_body_sort_index_input;
    muda::DeviceBuffer<int>      m_body_sort_index;

	    muda::DeviceBuffer<MatrixConverterIntPair> m_body_pairs;
	    muda::DeviceBuffer<MatrixConverterIntPair> m_body_unique_pairs;
	    muda::DeviceBuffer<int>                    m_body_unique_counts;
	    muda::DeviceBuffer<int>                    m_body_offsets;
	    muda::DeviceBuffer<Matrix12x12>            m_body_blocks_sorted;
	    muda::DeviceVar<int>                       m_body_unique_count_var;
	    muda::DeviceVar<int>                       m_body_triplet_count_var;

	    SizeT m_raw_hessian_count = 0;
	    SizeT m_node_pair_count   = 0;
	    SizeT m_body_pair_count   = 0;

	    void reduce_body_triplets(IndexT body_count);

    template <typename T>
    void loose_resize(muda::DeviceBuffer<T>& buffer, size_t size)
    {
        buffer.resize(size);
    }
};
}  // namespace uipc::backend::cuda

#endif
