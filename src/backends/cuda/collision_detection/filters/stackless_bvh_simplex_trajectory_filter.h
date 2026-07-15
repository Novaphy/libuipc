#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
#pragma once
#include <sim_system.h>
#include <global_geometry/global_vertex_manager.h>
#include <global_geometry/global_simplicial_surface_manager.h>
#include <contact_system/global_contact_manager.h>
#include <collision_detection/stackless_bvh.h>
#include <collision_detection/atomic_counting_lbvh.h>
#include <collision_detection/simplex_trajectory_filter.h>

namespace uipc::backend::cuda
{
class StacklessBVHSimplexTrajectoryFilter final : public SimplexTrajectoryFilter
{
  public:
    using SimplexTrajectoryFilter::SimplexTrajectoryFilter;

    class Impl
    {
      public:
        void detect(DetectInfo& info, SizeT frame, SizeT newton_iter);
        void filter_active(FilterActiveInfo& info, int frame, int newton_iter);
        void filter_toi(FilterTOIInfo& info);

        /****************************************************
        *                   Broad Phase
        ****************************************************/

        muda::DeviceBuffer<AABB> codim_point_aabbs;
        muda::DeviceBuffer<AABB> point_aabbs;
        muda::DeviceBuffer<AABB> edge_aabbs;
        muda::DeviceBuffer<AABB> triangle_aabbs;
        muda::DeviceBuffer<Float> edge_thicknesses;
        muda::DeviceBuffer<Float> edge_d_hats;
        muda::DeviceBuffer<Float> triangle_thicknesses;
        muda::DeviceBuffer<Float> triangle_d_hats;

        using ThisBVH = StacklessBVH;

        // CodimP count always less or equal to AllP count.
        ThisBVH              lbvh_CodimP;
        ThisBVH::QueryBuffer candidate_AllP_CodimP_pairs;

        // Used to detect CodimP-AllE, and AllE-AllE pairs.
        ThisBVH              lbvh_E;
        ThisBVH::QueryBuffer candidate_CodimP_AllE_pairs;
        ThisBVH::QueryBuffer candidate_AllE_AllE_pairs;

        // Used to detect AllP-AllT pairs.
        ThisBVH              lbvh_T;
        ThisBVH::QueryBuffer candidate_AllP_AllT_pairs;

        muda::DeviceVar<IndexT> selected_PT_count;
        muda::DeviceVar<IndexT> selected_EE_count;
        muda::DeviceVar<IndexT> selected_PE_count;
        muda::DeviceVar<IndexT> selected_PP_count;
        muda::DeviceBuffer<IndexT> selected_counts;

        muda::DeviceBuffer<Vector4i> temp_PTs;
        muda::DeviceBuffer<Vector4i> temp_EEs;
        muda::DeviceBuffer<Vector3i> temp_PEs;
        muda::DeviceBuffer<Vector2i> temp_PPs;

        muda::DeviceBuffer<std::byte> select_temp_PP;
        muda::DeviceBuffer<std::byte> select_temp_PE;
        muda::DeviceBuffer<std::byte> select_temp_PT;
        muda::DeviceBuffer<std::byte> select_temp_EE;
        size_t                        select_temp_PP_bytes = 0;
        size_t                        select_temp_PE_bytes = 0;
        size_t                        select_temp_PT_bytes = 0;
        size_t                        select_temp_EE_bytes = 0;
        int                           select_temp_PP_capacity = 0;
        int                           select_temp_PE_capacity = 0;
        int                           select_temp_PT_capacity = 0;
        int                           select_temp_EE_capacity = 0;
        muda::DeviceBuffer<int>       compact_flags;
        muda::DeviceBuffer<int>       compact_offsets;
        muda::DeviceBuffer<int>       block_select_counts;
        muda::DeviceBuffer<int>       block_select_offsets;
        muda::DeviceBuffer<int>       block_select_counts_PP;
        muda::DeviceBuffer<int>       block_select_offsets_PP;
        muda::DeviceBuffer<int>       block_select_counts_PE;
        muda::DeviceBuffer<int>       block_select_offsets_PE;
        muda::DeviceBuffer<int>       block_select_counts_PT;
        muda::DeviceBuffer<int>       block_select_offsets_PT;
        muda::DeviceBuffer<int>       block_select_counts_EE;
        muda::DeviceBuffer<int>       block_select_offsets_EE;

        muda::DeviceBuffer<Vector4i> PTs;
        muda::DeviceBuffer<Vector4i> EEs;
        muda::DeviceBuffer<Vector3i> PEs;
        muda::DeviceBuffer<Vector2i> PPs;

        bool          mask_cache_valid = false;
        const IndexT* cached_contact_mask_ptr = nullptr;
        const IndexT* cached_subscene_mask_ptr = nullptr;
        int           cached_contact_mask_w = 0;
        int           cached_contact_mask_h = 0;
        int           cached_subscene_mask_w = 0;
        int           cached_subscene_mask_h = 0;
        int           contact_mask_fast_mode = 0;
        int           subscene_mask_fast_mode = 0;
        bool          edge_tri_bvh_valid = false;
        SizeT         edge_bvh_size = 0;
        SizeT         tri_bvh_size = 0;
        bool          direct_active_valid = false;
        IndexT        direct_active_counts[4] = {0, 0, 0, 0};

        /****************************************************
        *                   CCD TOI
        ****************************************************/

        muda::DeviceBuffer<Float> tois;  // PP, PE, PT, EE
    };

    virtual muda::CBufferView<Vector2i> candidate_PTs() const noexcept override;
    virtual muda::CBufferView<Vector2i> candidate_EEs() const noexcept override;
    virtual muda::CBufferView<Float>    toi_PTs() const noexcept override;
    virtual muda::CBufferView<Float>    toi_EEs() const noexcept override;

  private:
    Impl m_impl;

    virtual void do_build(BuildInfo& info) override final;
    virtual void do_detect(DetectInfo& info) override final;
    virtual void do_filter_active(FilterActiveInfo& info) override final;
    virtual void do_filter_toi(FilterTOIInfo& info) override final;
};
}  // namespace uipc::backend::cuda
#else
#pragma once
#include <sim_system.h>
#include <global_geometry/global_vertex_manager.h>
#include <global_geometry/global_simplicial_surface_manager.h>
#include <contact_system/global_contact_manager.h>
#include <collision_detection/stackless_bvh.h>
#include <collision_detection/atomic_counting_lbvh.h>
#include <collision_detection/simplex_trajectory_filter.h>

namespace uipc::backend::cuda
{
class StacklessBVHSimplexTrajectoryFilter final : public SimplexTrajectoryFilter
{
  public:
    using SimplexTrajectoryFilter::SimplexTrajectoryFilter;

    class Impl
    {
      public:
        void detect(DetectInfo& info);
        void filter_active(FilterActiveInfo& info);
        void filter_toi(FilterTOIInfo& info);

        /****************************************************
        *                   Broad Phase
        ****************************************************/

        muda::DeviceBuffer<AABB> codim_point_aabbs;
        muda::DeviceBuffer<AABB> point_aabbs;
        muda::DeviceBuffer<AABB> edge_aabbs;
        muda::DeviceBuffer<AABB> triangle_aabbs;

        using ThisBVH = StacklessBVH;

        // CodimP count always less or equal to AllP count.
        ThisBVH              lbvh_CodimP;
        ThisBVH::QueryBuffer candidate_AllP_CodimP_pairs;

        // Used to detect CodimP-AllE, and AllE-AllE pairs.
        ThisBVH              lbvh_E;
        ThisBVH::QueryBuffer candidate_CodimP_AllE_pairs;
        ThisBVH::QueryBuffer candidate_AllE_AllE_pairs;

        // Used to detect AllP-AllT pairs.
        ThisBVH              lbvh_T;
        ThisBVH::QueryBuffer candidate_AllP_AllT_pairs;

        muda::DeviceVar<IndexT> selected_PT_count;
        muda::DeviceVar<IndexT> selected_EE_count;
        muda::DeviceVar<IndexT> selected_PE_count;
        muda::DeviceVar<IndexT> selected_PP_count;
        muda::DeviceBuffer<IndexT> selected_counts;
        muda::DeviceBuffer<int>    compact_flags;
        muda::DeviceBuffer<int>    compact_offsets;

        muda::DeviceBuffer<Vector4i> temp_PTs;
        muda::DeviceBuffer<Vector4i> temp_EEs;
        muda::DeviceBuffer<Vector3i> temp_PEs;
        muda::DeviceBuffer<Vector2i> temp_PPs;

        muda::DeviceBuffer<Vector4i> PTs;
        muda::DeviceBuffer<Vector4i> EEs;
        muda::DeviceBuffer<Vector3i> PEs;
        muda::DeviceBuffer<Vector2i> PPs;


        /****************************************************
        *                   CCD TOI
        ****************************************************/

        muda::DeviceBuffer<Float> tois;  // PP, PE, PT, EE
    };

    virtual muda::CBufferView<Vector2i> candidate_PTs() const noexcept override;
    virtual muda::CBufferView<Vector2i> candidate_EEs() const noexcept override;
    virtual muda::CBufferView<Float>    toi_PTs() const noexcept override;
    virtual muda::CBufferView<Float>    toi_EEs() const noexcept override;

  private:
    Impl m_impl;

    virtual void do_build(BuildInfo& info) override final;
    virtual void do_detect(DetectInfo& info) override final;
    virtual void do_filter_active(FilterActiveInfo& info) override final;
    virtual void do_filter_toi(FilterTOIInfo& info) override final;
};
}  // namespace uipc::backend::cuda
#endif
