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
#include <muda/buffer.h>

#include <thrust/device_vector.h>
#include <thrust/swap.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

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
            : reserve_ratio(1.2)
        {
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
     * @brief Detect overlapping AABB pairs in the BVH
     * 
     * @param callback f: (int i, int j) -> bool Callback predicate to filter overlapping pairs
     * @param qbuffer Output buffer to store detected overlapping pairs
     */
    template <typename Pred = DefaultQueryCallback>
    void detect(Pred callback, QueryBuffer& qbuffer);


    /*
    * @brief Query overlapping AABBs from external AABBs
    * 
    * @param aabbs Input external AABBs to query, aabbs must be kept valid during the lifetime of this BVH
    * @param callback f: (int i, int j) -> bool Callback predicate to filter overlapping pairs
    * @param qbuffer Output buffer to store detected overlapping pairs
    */
    template <typename Pred = DefaultQueryCallback>
    void query(muda::CBufferView<AABB> aabbs, Pred callback, QueryBuffer& qbuffer);


  public:
    class Impl
    {
      public:
        void build(muda::CBufferView<AABB> aabbs);
        template <typename Pred>
        void StacklessCDSharedSelf(Pred                       pred,
                                   muda::VarView<int>         cpNum,
                                   muda::BufferView<Vector2i> buffer);
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


        muda::CBufferView<AABB> objs;  // external AABBs, should be kept valid
        muda::DeviceVar<AABB>      scene_box;  // external bounding boxes
        muda::DeviceBuffer<uint32_t> flags;
        muda::DeviceBuffer<uint32_t> mtcode;  // external morton codes
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