#pragma once
#include <type_define.h>
#include <collision_detection/aabb.h>
#include <muda/buffer.h>
#include <uipc/common/log.h>
#include <concepts>
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
// InfoStacklessBVHV0: LBVH with per-node body/contact ID metadata.
// V0 (baseline): node_cull receives only (query_id, node_bid, node_cid);
// the user's NodePred must read query_bid/query_cid from global memory via query_id.
// Compare with InfoStacklessBVH which pre-loads query bid/cid into shared memory.
class InfoStacklessBVHV0
{
  public:
    class NodePredInfo
    {
      public:
        IndexT query_id = -1;
        IndexT node_bid = -1;
        IndexT node_cid = -1;

        NodePredInfo() = default;
        MUDA_GENERIC NodePredInfo(IndexT query_id, IndexT node_bid, IndexT node_cid)
            : query_id(query_id)
            , node_bid(node_bid)
            , node_cid(node_cid)
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
        friend class InfoStacklessBVHV0;
        SizeT                            m_size = 0;
        muda::DeviceBuffer<Vector2i>     m_pairs;
        muda::DeviceBuffer<unsigned int> m_queryMtCode;
        muda::DeviceVar<AABB>            m_querySceneBox;
        muda::DeviceBuffer<int>          m_querySortedId;
        muda::DeviceVar<int>             m_cpNum;

        void build(muda::CBufferView<AABB> aabbs);
    };

    struct Node
    {
        IndexT lc     = -1;
        IndexT escape = -1;
        AABB   bound;
        IndexT bid = -1;
        IndexT cid = -1;
    };

    class Config
    {
      public:
        Float reserve_ratio = 1.2;
    };

    InfoStacklessBVHV0(muda::Stream& stream = muda::Stream::Default()) noexcept;

    void build(muda::CBufferView<AABB>   aabbs,
               muda::CBufferView<IndexT> BIDs,
               muda::CBufferView<IndexT> CIDs);
    void build(muda::CBufferView<AABB> aabbs);

    template <typename NodePred, typename LeafPred>
    void detect(muda::CBuffer2DView<IndexT> cmts, NodePred np, LeafPred lp, QueryBuffer& qbuffer);

    template <typename NodePred, typename LeafPred>
    void query(muda::CBufferView<AABB>     query_aabbs,
               muda::CBufferView<IndexT>   query_BIDs,
               muda::CBufferView<IndexT>   query_CIDs,
               muda::CBuffer2DView<IndexT> cmts,
               NodePred                    np,
               LeafPred                    lp,
               QueryBuffer&                qbuffer);

    Config&       config() noexcept { return m_impl.config; }
    const Config& config() const noexcept { return m_impl.config; }

  public:
    class Impl
    {
      public:
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
        void        propagateInformativeMetadata(int intSize);
        void        build(muda::CBufferView<AABB>   aabbs,
                          muda::CBufferView<IndexT> bids,
                          muda::CBufferView<IndexT> cids);

        template <typename NodeCull, typename PairPred>
        void stacklessSelf(NodeCull                   node_cull,
                           PairPred                   pair_pred,
                           muda::VarView<int>         cpNum,
                           muda::BufferView<Vector2i> buffer);

        template <typename NodeCull, typename PairPred>
        void stacklessOther(NodeCull                   node_cull,
                            PairPred                   pair_pred,
                            muda::CBufferView<AABB>    query_aabbs,
                            muda::CBufferView<int>     query_sorted_id,
                            muda::VarView<int>         cpNum,
                            muda::BufferView<Vector2i> buffer);

        muda::CBufferView<AABB>      objs;
        muda::CBufferView<IndexT>    bids;
        muda::CBufferView<IndexT>    cids;
        muda::DeviceVar<AABB>        scene_box;
        muda::DeviceBuffer<uint32_t> flags;
        muda::DeviceBuffer<uint32_t> mtcode;
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
        muda::DeviceBuffer<IndexT>   ext_bid;
        muda::DeviceBuffer<IndexT>   ext_cid;
        muda::DeviceBuffer<IndexT>   int_bid;
        muda::DeviceBuffer<IndexT>   int_cid;
        muda::DeviceBuffer<Node>     nodes;
        Config                       config;
    };

  private:
    muda::CBufferView<AABB>   m_aabbs;
    muda::CBufferView<IndexT> m_BIDs;
    muda::CBufferView<IndexT> m_CIDs;
    Impl                      m_impl;
};
}  // namespace uipc::backend::cuda

namespace muda
{
template <>
struct force_trivially_destructible<uipc::backend::cuda::InfoStacklessBVHV0::Node>
{
    constexpr static bool value = true;
};

template <>
struct force_trivially_constructible<uipc::backend::cuda::InfoStacklessBVHV0::Node>
{
    constexpr static bool value = true;
};

template <>
struct force_trivially_copy_constructible<uipc::backend::cuda::InfoStacklessBVHV0::Node>
{
    constexpr static bool value = true;
};

template <>
struct force_trivially_copy_assignable<uipc::backend::cuda::InfoStacklessBVHV0::Node>
{
    constexpr static bool value = true;
};
}  // namespace muda


#include "details/info_stackless_bvh_v0.inl"
