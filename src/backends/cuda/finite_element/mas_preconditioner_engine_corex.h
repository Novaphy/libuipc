#pragma once
#include <type_define.h>
#include <muda/buffer/device_buffer.h>
#include <muda/buffer/device_var.h>
#include <muda/ext/linear_system.h>
#include <uipc/common/span.h>
#include <cstdint>
#include <filesystem>
#include <string_view>

namespace uipc::backend::cuda
{
/**
 * @brief MAS (MultiLevel Additive Schwarz) preconditioner engine.
 *
 * Multi-level domain-decomposition preconditioner from StiffGIPC.
 * Operates on 3x3 block matrices (BCOO format) and produces z = M^{-1} r.
 *
 * Key parameters:
 * - BANKSIZE = 16: each cluster has at most 16 nodes (48 DOFs).
 * - Mixed precision (Float=double): Hessian in double, inverse blocks in float.
 * - Float=float: Hessian and inverse blocks use float throughout.
 */
class MASPreconditionerEngine
{
  public:
    static constexpr int BANKSIZE          = 16;
    static constexpr int DEFAULT_BLOCKSIZE = 256;
    static constexpr int DEFAULT_WARPNUM   = 16;
    static constexpr int MAX_LEVELS        = 6;

    // Symmetric upper-triangle block count: BANKSIZE*(BANKSIZE+1)/2
    static constexpr int SYM_BLOCK_COUNT = BANKSIZE * (BANKSIZE + 1) / 2;

    template <typename Scalar>
    struct alignas(16) ClusterMatrixSymT
    {
        Eigen::Matrix<Scalar, 3, 3> M[SYM_BLOCK_COUNT];
        MUDA_GENERIC                ClusterMatrixSymT()
        {
            for(auto& m : M)
                m.setZero();
        }
    };

    using ClusterMatrixSym = ClusterMatrixSymT<Float>;  // Hessian assembly
#if defined(UIPC_FLOAT_SCALAR)
    using ClusterMatrixSymF = ClusterMatrixSymT<Float>;  // inverted blocks
#else
    using ClusterMatrixSymF = ClusterMatrixSymT<float>;  // compact inverse storage
#endif

#if defined(UIPC_FLOAT_SCALAR)
    using MasVec3 = float3;
#else
    using MasVec3 = double3;
#endif

    // Level traversal table per node
    struct LevelTable
    {
        int index[MAX_LEVELS];
    };

    // Host-side level size cache (replaces CUDA int2 to avoid host-side CUDA types)
    struct Int2
    {
        int x = 0;
        int y = 0;
    };

    MASPreconditionerEngine()  = default;
    ~MASPreconditionerEngine() = default;

    // ---- Phase 1: Initialize neighbor structures (called once) ----

    void init_neighbor(int                           vert_num,
                       int                           total_neighbor_num,
                       int                           part_map_size,
                       uipc::span<const unsigned int> h_neighbor_list,
                       uipc::span<const unsigned int> h_neighbor_start,
                       uipc::span<const unsigned int> h_neighbor_num,
                       uipc::span<const int>          h_part_to_real,
                       uipc::span<const int>          h_real_to_part);

    // ---- Phase 1b: Allocate matrix-level buffers (called once) ----

    void init_matrix();

    /**
     * @brief Optional: register global BCOO Hessian triplets (device pointers, not owned)
     *        so reorder_realtime can inject off-diagonal connectivity for contact.
     */
    void set_hessian_coupling(const int* d_row_ids,
                              const int* d_col_ids,
                              int        triplet_num,
                              int        dof_offset);

    // ---- Phase 2: Assemble preconditioner (per Newton iteration) ----

    void set_preconditioner(const Matrix3x3*  d_triplet_values,
                            const int*        d_row_ids,
                            const int*        d_col_ids,
                            const uint32_t*   d_indices,
                            int               dof_offset,
                            int               triplet_num,
                            int               cp_num);

    // ---- Phase 3: Apply preconditioning z = M^{-1} r (per PCG iteration) ----

    void apply(muda::CDenseVectorView<Float> r,
               muda::DenseVectorView<Float>  z,
               muda::CVarView<IndexT>        converged);

    bool is_initialized() const { return m_initialized; }

    /** Dump cluster matrices in Matrix Market (.mtx) format and metadata as JSON for debug. */
    void dump_cluster_matrices_debug(const std::filesystem::path& output_dir,
                                     SizeT                        frame,
                                     SizeT                        newton_iter);

    // ===========================================================================
    // All methods below are public because NVCC on Windows requires
    // extended __device__ lambdas to reside in methods with public access.
    // ===========================================================================

    // Hierarchy building steps
    void compute_num_levels(int vert_num);
    int  reorder_realtime(int cp_num);
    void build_connect_mask_L0();
    void prepare_prefix_sum_L0();
    void build_level1();
    void build_connect_mask_Lx(int level);
    void next_level_cluster(int level);
    void prefix_sum_Lx(int level);
    void compute_next_level(int level);
    void aggregation_kernel();

    void build_hessian_connection(unsigned int* connection_mask,
                                  const int*    coarse_table,
                                  int           level);

    // Hessian assembly + inversion
    void scatter_hessian_to_clusters(const Matrix3x3* d_triplet_values,
                                     const int*       d_row_ids,
                                     const int*       d_col_ids,
                                     const uint32_t*  d_indices,
                                     int              dof_offset,
                                     int              triplet_num);
    void invert_cluster_matrices();

    // Preconditioning steps
    void build_multi_level_R(const MasVec3*           R,
                             muda::CVarView<IndexT> converged);
    void schwarz_local_solve(muda::CVarView<IndexT> converged);
    void collect_final_Z(MasVec3* Z, muda::CVarView<IndexT> converged);

  private:
    // ---- State ----
    bool m_initialized        = false;
    int  m_total_nodes        = 0;
    int  m_total_map_nodes    = 0;
    int  m_level_num          = 0;
    int  m_total_num_clusters = 0;
    Int2 m_h_level_size;

    // ---- Optional BCOO coupling (contact): device pointers, not owned ----
    const int* m_bcoo_row_ids     = nullptr;
    const int* m_bcoo_col_ids     = nullptr;
    int        m_bcoo_triplet_num = 0;
    int        m_bcoo_dof_offset  = 0;

    // ---- GPU buffers: hierarchy ----
    muda::DeviceBuffer<Int2>         level_sizes;
    muda::DeviceBuffer<int>          coarse_space_tables;
    muda::DeviceBuffer<int>          prefix_original;
    muda::DeviceBuffer<int>          prefix_sum_original;
    muda::DeviceBuffer<int>          going_next;
    muda::DeviceBuffer<int>          dense_level;
    muda::DeviceBuffer<LevelTable>   coarse_tables;
    muda::DeviceBuffer<unsigned int> fine_connect_masks;
    muda::DeviceBuffer<unsigned int> next_connect_masks;
    muda::DeviceBuffer<unsigned int> next_prefixes;
    muda::DeviceBuffer<unsigned int> next_prefix_sums;

    // ---- GPU buffers: neighbor graph ----
    int                              m_neighbor_list_size = 0;
    muda::DeviceBuffer<unsigned int> neighbor_lists;
    muda::DeviceBuffer<unsigned int> neighbor_starts;
    muda::DeviceBuffer<unsigned int> neighbor_nums;
    muda::DeviceBuffer<unsigned int> neighbor_lists_init;
    muda::DeviceBuffer<unsigned int> neighbor_nums_init;

    // ---- GPU buffers: partition mappings ----
    muda::DeviceBuffer<int> part_to_real;   // partition-ordered index -> real vertex index
    muda::DeviceBuffer<int> real_to_part;   // real vertex index -> partition-ordered index

    // ---- GPU buffers: cluster matrices ----
    muda::DeviceBuffer<ClusterMatrixSym>  cluster_hessians;   // assembled Hessian blocks
    muda::DeviceBuffer<ClusterMatrixSymF> cluster_inverses;  // inverted preconditioner blocks

    // ---- GPU buffers: multi-level residual / solution ----
    muda::DeviceBuffer<Eigen::Vector3f> multi_level_R;
    muda::DeviceBuffer<float3>          multi_level_Z;
};
}  // namespace uipc::backend::cuda

namespace muda
{
template <>
struct force_trivially_destructible<uipc::backend::cuda::MASPreconditionerEngine::Int2>
{
    constexpr static bool value = true;
};

template <>
struct force_trivially_constructible<uipc::backend::cuda::MASPreconditionerEngine::Int2>
{
    constexpr static bool value = true;
};

template <>
struct force_trivially_copy_constructible<uipc::backend::cuda::MASPreconditionerEngine::Int2>
{
    constexpr static bool value = true;
};

template <>
struct force_trivially_copy_assignable<uipc::backend::cuda::MASPreconditionerEngine::Int2>
{
    constexpr static bool value = true;
};

template <typename Scalar>
struct force_trivially_destructible<uipc::backend::cuda::MASPreconditionerEngine::ClusterMatrixSymT<Scalar>>
{
    constexpr static bool value = true;
};

template <typename Scalar>
struct force_trivially_constructible<uipc::backend::cuda::MASPreconditionerEngine::ClusterMatrixSymT<Scalar>>
{
    constexpr static bool value = true;
};

template <typename Scalar>
struct force_trivially_copy_constructible<uipc::backend::cuda::MASPreconditionerEngine::ClusterMatrixSymT<Scalar>>
{
    constexpr static bool value = true;
};

template <typename Scalar>
struct force_trivially_copy_assignable<uipc::backend::cuda::MASPreconditionerEngine::ClusterMatrixSymT<Scalar>>
{
    constexpr static bool value = true;
};
}  // namespace muda
