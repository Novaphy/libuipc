/**
 * @file stackless_bvh.h
 * 
 * @brief UIPC Compatible Version of Stackless BVH for AABB overlap detection (safe muda-style)
 * 
 * References:
 * 
 * Thanks to the original authors of the following repositories for their excellent implementations of Stackless BVH!
 * 
 * - https://github.com/ZiXuanVickyLu/culbvh
 * - https://github.com/jerry060599/KittenGpuLBVH
 * 
 */

#pragma once
#include <uipc/common/logger.h>
#include <type_define.h>
#include <collision_detection/aabb.h>
#include <utils/corex_phase_profile.h>
#include <utils/codim_thickness.h>
#include <utils/distance.h>
#include <utils/primitive_d_hat.h>
#include <muda/buffer.h>

#include <thrust/device_vector.h>
#include <thrust/swap.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <cstdlib>

namespace uipc::backend::cuda
{
/**
 * @brief Friend class to access private members of StacklessBVH (for internal use only)
 */
template <typename T>
class StacklessBVHFriend;

/**
 * @brief Stackless Bounding Volume Hierarchy for AABB overlap detection
 */
class StacklessBVH
{
  public:
    template <typename T>
    friend class StacklessBVHFriend;

    class Config
    {
      public:
        Float reserve_ratio;
        Config()
            : reserve_ratio(2.0)
        {
            const char* env = std::getenv("UIPC_COREX_BVH_QUERY_RESERVE_RATIO");
            if(env && env[0] != '\0')
            {
                char*  end = nullptr;
                double v   = std::strtod(env, &end);
                if(end != env && v >= 1.0)
                    reserve_ratio = static_cast<Float>(v);
            }
        }
    };

    class QueryBuffer
    {
      public:
        QueryBuffer() = default;

        auto  view() const noexcept { return m_pairs.view(0, m_size); }
        void  reserve(size_t size) { m_pairs.resize(size); }
        SizeT size() const noexcept { return m_size; }
        auto  viewer() const noexcept { return view().viewer(); }

      public:
        friend class StacklessBVH;
        SizeT                        m_size = 0;
        muda::DeviceBuffer<Vector2i> m_pairs;

        muda::DeviceBuffer<unsigned int> m_queryMtCode;
        muda::DeviceVar<AABB>            m_querySceneBox;
        muda::DeviceBuffer<int>          m_querySortedId;
        muda::DeviceVar<int>             m_cpNum;

        void  build(muda::CBufferView<AABB> aabbs);
        SizeT query_count() { return m_queryMtCode.size(); }
    };

    struct /*__align__(16) */ Node
    {
        IndexT lc;
        IndexT escape;
        AABB   bound;
    };

    StacklessBVH(Config config = Config{}) { m_impl.config = config; }

    ~StacklessBVH() = default;

    struct DefaultQueryCallback
    {
        MUDA_GENERIC bool operator()(IndexT i, IndexT j) const { return true; }
    };

    /**
     * @brief Build the Stackless BVH from given AABBs
     * 
     * @param aabbs Input AABBs, aabbs must be kept valid during the lifetime of this BVH
     */
    void build(muda::CBufferView<AABB> aabbs);

    /**
     * @brief Refit the Stackless BVH from given AABBs, reusing the topology
     * created by the previous build().
     *
     * The primitive count must match the previous build.
     */
    void refit(muda::CBufferView<AABB> aabbs);

    /**
     * @brief Detect overlapping AABB pairs in the BVH
     * 
     * @param callback f: (int i, int j) -> bool Callback predicate to filter overlapping pairs
     * @param qbuffer Output buffer to store detected overlapping pairs
     */
    template <typename Pred = DefaultQueryCallback>
    void detect(Pred callback, QueryBuffer& qbuffer);

    void detect_edges_no_mask(muda::CBufferView<Vector2i> edges,
                              muda::CBufferView<IndexT>   vertex_to_body,
                              muda::CBufferView<IndexT>   body_self_collision,
                              QueryBuffer&                qbuffer);

    bool detect_edges_no_mask_launch(muda::CBufferView<Vector2i> edges,
                                     muda::CBufferView<IndexT>   vertex_to_body,
                                     muda::CBufferView<IndexT>   body_self_collision,
                                     QueryBuffer&                qbuffer);


    /*
    * @brief Query overlapping AABBs from external AABBs
    * 
    * @param aabbs Input external AABBs to query, aabbs must be kept valid during the lifetime of this BVH
    * @param callback f: (int i, int j) -> bool Callback predicate to filter overlapping pairs
    * @param qbuffer Output buffer to store detected overlapping pairs
    */
    template <typename Pred = DefaultQueryCallback>
    void query(muda::CBufferView<AABB> aabbs, Pred callback, QueryBuffer& qbuffer);

    void query_points_triangles_no_mask(muda::CBufferView<AABB>     point_aabbs,
                                        muda::CBufferView<IndexT>   surf_vertices,
                                        muda::CBufferView<Vector3i> surf_triangles,
                                        muda::CBufferView<Vector3>  positions,
                                        muda::CBufferView<Vector3>  displacements,
                                        muda::CBufferView<Float>    thicknesses,
                                        muda::CBufferView<Float>    d_hats,
                                        muda::CBufferView<Float>    triangle_thicknesses,
                                        muda::CBufferView<Float>    triangle_d_hats,
                                        Float                       alpha,
                                        muda::CBufferView<IndexT>   vertex_to_body,
                                        muda::CBufferView<IndexT>   body_self_collision,
                                        QueryBuffer&                qbuffer);

    bool query_points_triangles_no_mask_launch(muda::CBufferView<AABB>     point_aabbs,
                                               muda::CBufferView<IndexT>   surf_vertices,
                                               muda::CBufferView<Vector3i> surf_triangles,
                                               muda::CBufferView<Vector3>  positions,
                                               muda::CBufferView<Vector3>  displacements,
                                               muda::CBufferView<Float>    thicknesses,
                                               muda::CBufferView<Float>    d_hats,
                                               muda::CBufferView<Float>    triangle_thicknesses,
                                               muda::CBufferView<Float>    triangle_d_hats,
                                               Float                       alpha,
                                               muda::CBufferView<IndexT>   vertex_to_body,
                                     muda::CBufferView<IndexT>   body_self_collision,
                                     QueryBuffer&                qbuffer);

    void detect_edges_active_no_mask(muda::CBufferView<Vector2i> edges,
                                     muda::CBufferView<Vector3>  positions,
                                     muda::CBufferView<Vector3>  displacements,
                                     muda::CBufferView<Vector3>  rest_positions,
                                     muda::CBufferView<Float>    edge_thicknesses,
                                     muda::CBufferView<Float>    edge_d_hats,
                                     Float                       alpha,
                                     muda::CBufferView<IndexT>   vertex_to_body,
                                     muda::CBufferView<IndexT>   body_self_collision,
                                     muda::BufferView<Vector2i>  out_PPs,
                                     muda::BufferView<Vector3i>  out_PEs,
                                     muda::BufferView<Vector4i>  out_EEs,
                                     muda::BufferView<IndexT>    selected_counts);

    void query_points_triangles_active_no_mask(muda::CBufferView<AABB>     point_aabbs,
                                               muda::CBufferView<IndexT>   surf_vertices,
                                               muda::CBufferView<Vector3i> surf_triangles,
                                               muda::CBufferView<Vector3>  positions,
                                               muda::CBufferView<Vector3>  displacements,
                                               muda::CBufferView<Float>    thicknesses,
                                               muda::CBufferView<Float>    d_hats,
                                               muda::CBufferView<Float>    triangle_thicknesses,
                                               muda::CBufferView<Float>    triangle_d_hats,
                                               Float                       alpha,
                                               muda::CBufferView<IndexT>   vertex_to_body,
                                               muda::CBufferView<IndexT>   body_self_collision,
                                               muda::BufferView<Vector2i>  out_PPs,
                                               muda::BufferView<Vector3i>  out_PEs,
                                               muda::BufferView<Vector4i>  out_PTs,
                                               muda::BufferView<IndexT>    selected_counts);


  public:
    class Impl
    {
      public:
        void build(muda::CBufferView<AABB> aabbs);
        void refit(muda::CBufferView<AABB> aabbs);
        bool can_refit(muda::CBufferView<AABB> aabbs) const;
        template <typename Pred>
        void StacklessCDSharedSelf(Pred                       pred,
                                   muda::VarView<int>         cpNum,
                                   muda::BufferView<Vector2i> buffer);
        void StacklessCDSharedSelfEdgesNoMask(
            muda::CBufferView<Vector2i> edges,
            muda::CBufferView<IndexT>   vertex_to_body,
            muda::CBufferView<IndexT>   body_self_collision,
            muda::VarView<int>          cpNum,
            muda::BufferView<Vector2i>  buffer);
        void StacklessCDSharedSelfEdgesActiveNoMask(
            muda::CBufferView<Vector2i> edges,
            muda::CBufferView<Vector3>  positions,
            muda::CBufferView<Vector3>  displacements,
            muda::CBufferView<Vector3>  rest_positions,
            muda::CBufferView<Float>    edge_thicknesses,
            muda::CBufferView<Float>    edge_d_hats,
            Float                       alpha,
            muda::CBufferView<IndexT>   vertex_to_body,
            muda::CBufferView<IndexT>   body_self_collision,
            muda::BufferView<Vector2i>  out_PPs,
            muda::BufferView<Vector3i>  out_PEs,
            muda::BufferView<Vector4i>  out_EEs,
            muda::BufferView<IndexT>    selected_counts);
        void StacklessCDSharedOtherPointsTrianglesNoMask(
            muda::CBufferView<AABB>     point_aabbs,
            muda::CBufferView<IndexT>   surf_vertices,
            muda::CBufferView<Vector3i> surf_triangles,
            muda::CBufferView<Vector3>  positions,
            muda::CBufferView<Vector3>  displacements,
            muda::CBufferView<Float>    thicknesses,
            muda::CBufferView<Float>    d_hats,
            muda::CBufferView<Float>    triangle_thicknesses,
            muda::CBufferView<Float>    triangle_d_hats,
            Float                       alpha,
            muda::CBufferView<IndexT>   vertex_to_body,
            muda::CBufferView<IndexT>   body_self_collision,
            muda::VarView<int>          cpNum,
            muda::BufferView<Vector2i>  buffer);
        void StacklessCDSharedOtherPointsTrianglesActiveNoMask(
            muda::CBufferView<AABB>     point_aabbs,
            muda::CBufferView<IndexT>   surf_vertices,
            muda::CBufferView<Vector3i> surf_triangles,
            muda::CBufferView<Vector3>  positions,
            muda::CBufferView<Vector3>  displacements,
            muda::CBufferView<Float>    thicknesses,
            muda::CBufferView<Float>    d_hats,
            muda::CBufferView<Float>    triangle_thicknesses,
            muda::CBufferView<Float>    triangle_d_hats,
            Float                       alpha,
            muda::CBufferView<IndexT>   vertex_to_body,
            muda::CBufferView<IndexT>   body_self_collision,
            muda::BufferView<Vector2i>  out_PPs,
            muda::BufferView<Vector3i>  out_PEs,
            muda::BufferView<Vector4i>  out_PTs,
            muda::BufferView<IndexT>    selected_counts);
        template <typename Pred>
        void StacklessCDSharedOther(Pred                       pred,
                                    muda::CBufferView<AABB>    query_aabbs,
                                    muda::CBufferView<int>     query_sorted_id,
                                    muda::VarView<int>         cpNum,
                                    muda::BufferView<Vector2i> buffer);


        static void calcMaxBVFromBox(muda::CBufferView<AABB> aabbs,
                                     muda::VarView<AABB>     scene_box);
        static void calcMCsFromBox(muda::CBufferView<AABB>    aabbs,
                                   muda::CVarView<AABB>       scene_box,
                                   muda::BufferView<uint32_t> codes);
        void        calcInverseMapping();
        void        buildPrimitivesFromBox(muda::CBufferView<AABB> aabbs);
        void        calcExtNodeSplitMetrics();
        void        buildIntNodes(int size);
        void        calcIntNodeOrders(int size);
        void        updateBvhExtNodeLinks(int size);
        void        reorderNode(int intSize);
        void        updateRefitNodeBounds(int intSize);
        muda::CBufferView<AABB> objs;  // external AABBs, should be kept valid
        muda::DeviceVar<AABB>      scene_box;  // external bounding boxes
        muda::DeviceBuffer<uint32_t> flags;
        muda::DeviceBuffer<uint32_t> mtcode;  // external morton codes
        muda::DeviceBuffer<uint32_t> mtcode_sorted;
        muda::DeviceBuffer<int32_t>  sorted_id_input;
        muda::DeviceBuffer<int32_t>  sorted_id;
        muda::DeviceBuffer<int32_t>  primMap;
        muda::DeviceBuffer<int>      metric;
        muda::DeviceBuffer<uint32_t> count;
        muda::DeviceBuffer<int>      tkMap;
        muda::DeviceBuffer<uint32_t> offsetTable;

        muda::DeviceBuffer<AABB>     ext_aabb;
        muda::DeviceBuffer<int>      ext_idx;
        muda::DeviceBuffer<int>      ext_lca;
        muda::DeviceBuffer<uint32_t> ext_mark;
        muda::DeviceBuffer<uint32_t> ext_par;

        muda::DeviceBuffer<int>      int_lc;
        muda::DeviceBuffer<int>      int_rc;
        muda::DeviceBuffer<int>      int_par;
        muda::DeviceBuffer<int>      int_range_x;
        muda::DeviceBuffer<int>      int_range_y;
        muda::DeviceBuffer<uint32_t> int_mark;
        muda::DeviceBuffer<AABB>     int_aabb;

        muda::DeviceBuffer<ulonglong2> quantNode;
        muda::DeviceBuffer<Node>       nodes;
        muda::DeviceBuffer<int>        node_range_y;

        Config config;
    };

  private:
    Impl m_impl;
};

}  // namespace uipc::backend::cuda

namespace muda
{
template <>
struct force_trivially_destructible<uipc::backend::cuda::StacklessBVH::Node>
{
    constexpr static bool value = true;
};

template <>
struct force_trivially_constructible<uipc::backend::cuda::StacklessBVH::Node>
{
    constexpr static bool value = true;
};

template <>
struct force_trivially_copy_constructible<uipc::backend::cuda::StacklessBVH::Node>
{
    constexpr static bool value = true;
};

template <>
struct force_trivially_copy_assignable<uipc::backend::cuda::StacklessBVH::Node>
{
    constexpr static bool value = true;
};
}  // namespace muda

#include "details/stackless_bvh.inl"
