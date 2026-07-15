#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
#include <cuda_device/builtin.h>
#include <muda/launch.h>
#include <muda/cub/device/device_radix_sort.h>
#include <cub/block/block_reduce.cuh>
#include <thrust/device_ptr.h>
#define RAW_PTR(x) thrust::raw_pointer_cast((x).data())

namespace uipc::culbvh
{
using aabb          = uipc::backend::cuda::AABB;
using stacklessnode = uipc::backend::cuda::StacklessBVH::Node;
using Vector2i      = uipc::Vector2i;

using uint   = uint32_t;
using ullint = unsigned long long int;

constexpr int K_THREADS = 224;
constexpr int K_WARPS   = K_THREADS >> 5;

constexpr int K_REDUCTION_LAYER  = 5;
constexpr int K_REDUCTION_NUM    = 1 << K_REDUCTION_LAYER;
constexpr int K_REDUCTION_MODULO = K_REDUCTION_NUM - 1;

constexpr int    aabbBits  = 15;
constexpr int    aabbRes   = (1 << aabbBits) - 2;
constexpr int    indexBits = 64 - 3 * aabbBits;
constexpr int    offset3   = aabbBits * 3;
constexpr int    offset2   = aabbBits * 2;
constexpr int    offset1   = aabbBits * 1;
constexpr ullint indexMask = 0xFFFFFFFFFFFFFFFFu << offset3;
constexpr uint   aabbMask  = 0xFFFFFFFFu >> (32 - aabbBits);
constexpr uint   MaxIndex  = 0xFFFFFFFFFFFFFFFFu >> offset3;

constexpr uint MAX_CD_NUM_PER_VERT = 64;
constexpr int  MAX_RES_PER_BLOCK   = 1024;

static __global__ void kernel_fill_u32(uint32_t* data, int n, uint32_t value)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
        data[i] = value;
}

static __global__ void kernel_init_scene_box(aabb* scene_box)
{
    if(blockIdx.x == 0 && threadIdx.x == 0)
        *scene_box = aabb();
}

struct PlainAABB
{
    float3 _min, _max;
};

MUDA_HOST MUDA_DEVICE MUDA_INLINE PlainAABB toPlainAABB(const aabb& box)
{
    PlainAABB res;
    res._min = make_float3(box.min().x(), box.min().y(), box.min().z());
    res._max = make_float3(box.max().x(), box.max().y(), box.max().z());
    return res;
}

MUDA_HOST MUDA_DEVICE MUDA_INLINE aabb fromPlainAABB(const PlainAABB& box)
{
    aabb aabb;
    aabb.min() = Vector<float, 3>(box._min.x, box._min.y, box._min.z);
    aabb.max() = Vector<float, 3>(box._max.x, box._max.y, box._max.z);
    return aabb;
}

struct intAABB
{
    int3 _min, _max;

    MUDA_HOST MUDA_DEVICE MUDA_INLINE void convertFrom(const aabb& other, float3& origin, float3& delta)
    {
        _min.x = static_cast<int>((other.min().x() - origin.x) / delta.x);
        _min.y = static_cast<int>((other.min().y() - origin.y) / delta.y);
        _min.z = static_cast<int>((other.min().z() - origin.z) / delta.z);
        _max.x = static_cast<int>(ceilf((other.max().x() - origin.x) / delta.x));
        _max.y = static_cast<int>(ceilf((other.max().y() - origin.y) / delta.y));
        _max.z = static_cast<int>(ceilf((other.max().z() - origin.z) / delta.z));
    }
};

template <typename T>
MUDA_HOST MUDA_DEVICE MUDA_INLINE T __mm_min(T a, T b)
{
    return a > b ? b : a;
}

template <typename T>
MUDA_HOST MUDA_DEVICE MUDA_INLINE T __mm_max(T a, T b)
{
    return a > b ? a : b;
}

struct PlainAABBReduce
{
    MUDA_DEVICE MUDA_INLINE PlainAABB operator()(const PlainAABB& a,
                                                  const PlainAABB& b) const
    {
        PlainAABB out;
        out._min = make_float3(__mm_min(a._min.x, b._min.x),
                               __mm_min(a._min.y, b._min.y),
                               __mm_min(a._min.z, b._min.z));
        out._max = make_float3(__mm_max(a._max.x, b._max.x),
                               __mm_max(a._max.y, b._max.y),
                               __mm_max(a._max.z, b._max.z));
        return out;
    }
};

MUDA_DEVICE MUDA_INLINE float atomicMinf(float* addr, float value)
{
    float old;
    old = (value >= 0) ?
              __int_as_float(atomicMin((int*)addr, __float_as_int(value))) :
              __uint_as_float(atomicMax((unsigned int*)addr, __float_as_uint(value)));
    return old;
}

MUDA_DEVICE MUDA_INLINE float atomicMaxf(float* addr, float value)
{
    float old;
    old = (value >= 0) ?
              __int_as_float(atomicMax((int*)addr, __float_as_int(value))) :
              __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));
    return old;
}

static __global__ void kernel_reduce_scene_box(int         n,
                                                const aabb* boxes,
                                                aabb*       scene_box)
{
    using BlockReduce = cub::BlockReduce<PlainAABB, K_THREADS>;
    __shared__ typename BlockReduce::TempStorage storage;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    PlainAABB local;
    local._min = make_float3(CUDART_INF_F, CUDART_INF_F, CUDART_INF_F);
    local._max = make_float3(-CUDART_INF_F, -CUDART_INF_F, -CUDART_INF_F);
    if(i < n)
        local = toPlainAABB(boxes[i]);

    PlainAABB block_box = BlockReduce(storage).Reduce(local, PlainAABBReduce{});
    if(threadIdx.x == 0)
    {
        atomicMinf(&scene_box->min().x(), block_box._min.x);
        atomicMinf(&scene_box->min().y(), block_box._min.y);
        atomicMinf(&scene_box->min().z(), block_box._min.z);
        atomicMaxf(&scene_box->max().x(), block_box._max.x);
        atomicMaxf(&scene_box->max().y(), block_box._max.y);
        atomicMaxf(&scene_box->max().z(), block_box._max.z);
    }
}

MUDA_HOST MUDA_DEVICE MUDA_INLINE uint expandBits(uint v)
{  ///< Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

MUDA_HOST MUDA_DEVICE MUDA_INLINE uint morton3D(float x, float y, float z)
{  ///< Calculates a 30-bit Morton code for the given 3D point located within the unit cube [0,1].
    x       = ::fmin(::fmax(x * 1024.0f, 0.0f), 1023.0f);
    y       = ::fmin(::fmax(y * 1024.0f, 0.0f), 1023.0f);
    z       = ::fmin(::fmax(z * 1024.0f, 0.0f), 1023.0f);
    uint xx = expandBits((uint)x);
    uint yy = expandBits((uint)y);
    uint zz = expandBits((uint)z);
    return (xx * 4 + yy * 2 + zz);
}

// Custom comparison for int3 based on lexicographical ordering
MUDA_HOST MUDA_DEVICE MUDA_INLINE bool lessThan(const int3& a, const int3& b)
{
    if(a.x != b.x)
        return a.x < b.x;
    if(a.y != b.y)
        return a.y < b.y;
    return a.z < b.z;
}

MUDA_HOST MUDA_DEVICE MUDA_INLINE Vector2i to_eigen(int2 v)
{
    return Vector2i{v.x, v.y};
}

MUDA_HOST MUDA_DEVICE MUDA_INLINE int2 make_ordered_pair(int a, int b)
{
    if(a < b)
        return int2{a, b};
    else
        return int2{b, a};
}

MUDA_HOST MUDA_DEVICE MUDA_INLINE float3 operator-(const float3& v0, const float3& v1)
{
    return make_float3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z);
}

MUDA_DEVICE MUDA_INLINE Float corex_sqr(Float x)
{
    return x * x;
}

MUDA_DEVICE MUDA_INLINE Float point_box_distance2_lower_bound(const Vector3& p,
                                                              const Vector3& bmin,
                                                              const Vector3& bmax)
{
    Float d = 0;
    for(int k = 0; k < 3; ++k)
    {
        if(p[k] < bmin[k])
            d += corex_sqr(bmin[k] - p[k]);
        else if(p[k] > bmax[k])
            d += corex_sqr(p[k] - bmax[k]);
    }
    return d;
}

MUDA_DEVICE MUDA_INLINE Float box_box_distance2_lower_bound(const Vector3& amin,
                                                            const Vector3& amax,
                                                            const Vector3& bmin,
                                                            const Vector3& bmax)
{
    Float d = 0;
    for(int k = 0; k < 3; ++k)
    {
        if(amax[k] < bmin[k])
            d += corex_sqr(bmin[k] - amax[k]);
        else if(bmax[k] < amin[k])
            d += corex_sqr(amin[k] - bmax[k]);
    }
    return d;
}

MUDA_HOST MUDA_DEVICE MUDA_INLINE void SafeCopyTo(int2* sharedRes,
                                                  int   totalResInBlock,
                                                  Vector2i* globalRes,
                                                  int       globalIdx,
                                                  int       maxRes)
{
    if(globalIdx >= maxRes      // Out of memory for results.
       || totalResInBlock == 0  // No results to write
    )
        return;

    auto CopyCount = std::min(totalResInBlock, maxRes - globalIdx);

    // Copy full blocks
    int fullBlocks = (CopyCount - 1) / (int)blockDim.x;
    for(int i = 0; i < fullBlocks; i++)
    {
        int offset                    = i * blockDim.x + threadIdx.x;
        globalRes[globalIdx + offset] = to_eigen(sharedRes[offset]);
    }

    // Copy the rest
    int offset = fullBlocks * blockDim.x + threadIdx.x;
    if(offset < CopyCount)
        globalRes[globalIdx + offset] = to_eigen(sharedRes[offset]);
}

}  // namespace uipc::culbvh

namespace uipc::backend::cuda::corex_bvh
{
using AABB = uipc::backend::cuda::AABB;
using namespace uipc::culbvh;

static __global__ void kernel_calcMCs(int N, const AABB* boxes, const AABB* scene, uint32_t* codes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;
    auto bv = boxes[idx];
    auto center = bv.center();
    float3 c = make_float3(center.x(), center.y(), center.z());
    auto sceneMin = scene->min();
    float3 sceneMinVec = make_float3(sceneMin.x(), sceneMin.y(), sceneMin.z());
    float3 offset = c - sceneMinVec;
    auto sceneSize = scene->sizes();
    codes[idx] = morton3D(offset.x / sceneSize.x(), offset.y / sceneSize.y(), offset.z / sceneSize.z());
}

static __global__ void kernel_calcInverseMapping(int N, const int* sorted_id, int* primMap)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;
    primMap[sorted_id[idx]] = idx;
}

static __global__ void kernel_buildPrimitives(int N, const int* primMap, const AABB* boxes,
                                       int* ext_idx, AABB* ext_aabb)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;
    int newIdx = primMap[idx];
    ext_idx[newIdx] = idx;
    ext_aabb[newIdx] = boxes[idx];
}

static __global__ void kernel_calcSplitMetrics(int N, const uint32_t* codes, int* metrics)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;
    metrics[idx] = (idx != N - 1) ? (32 - __clz(codes[idx] ^ codes[idx + 1])) : 33;
}

static __global__ void kernel_calcIntNodeOrders(int N, const int* int_lc, const int* lcas,
                                          const uint32_t* depths, const uint32_t* offsets, int* tkMap)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;
    int node = lcas[idx];
    int depth = depths[idx];
    int id = offsets[idx];
    if(node != -1)
    {
        for(; depth--; node = int_lc[node])
        {
            tkMap[node] = id++;
        }
    }
}

static __global__ void kernel_updateBvhExtNodeLinks(int N, const int* mapTable, int* lcas, uint32_t* pars)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;
    pars[idx] = mapTable[pars[idx]];
    int ori = lcas[idx];
    if(ori != -1)
        lcas[idx] = mapTable[ori] << 1;
    else
        lcas[idx] = idx << 1 | 1;
}

static __global__ void kernel_reorderNode(int N, int intSize,
                                    const int* lcas, const AABB* lvs_box,
                                    const int* tkMap, const int* int_lc,
                                    const uint32_t* int_mark, const int* int_range_y,
                                    const AABB* int_aabb,
                                    stacklessnode* nodes,
                                    int* node_range_y)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;
    stacklessnode Node;
    Node.lc = -1;
    int escape = lcas[idx + 1];
    if(escape == -1)
    {
        Node.escape = -1;
    }
    else
    {
        int bLeaf = escape & 1;
        escape >>= 1;
        Node.escape = escape + (bLeaf ? intSize : 0);
    }
    Node.bound = lvs_box[idx];
    nodes[idx + intSize] = Node;
    node_range_y[idx + intSize] = idx;

    if(idx < intSize)
    {
        stacklessnode intNode;
        int newId = tkMap[idx];
        uint32_t mark = int_mark[idx];
        intNode.lc = (mark & 1) ? (int_lc[idx] + intSize) : tkMap[int_lc[idx]];
        intNode.bound = int_aabb[idx];
        int intEscape = lcas[int_range_y[idx] + 1];
        if(intEscape == -1)
        {
            intNode.escape = -1;
        }
        else
        {
            int bLeaf = intEscape & 1;
            intEscape >>= 1;
            intNode.escape = intEscape + (bLeaf ? intSize : 0);
        }
        nodes[newId] = intNode;
        node_range_y[newId] = int_range_y[idx];
    }
}

static __global__ void kernel_updateRefitNodeBounds(int            N,
                                                    int            intSize,
                                                    const AABB*    lvs_box,
                                                    const int*     tkMap,
                                                    const AABB*    int_aabb,
                                                    stacklessnode* nodes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;

    nodes[idx + intSize].bound = lvs_box[idx];

    if(idx < intSize)
    {
        int newId = tkMap[idx];
        nodes[newId].bound = int_aabb[idx];
    }
}

static __global__ void kernel_refitIntNodes(int             size,
                                            const uint32_t* ext_par,
                                            const AABB*     ext_aabb,
                                            const int*      int_lc,
                                            const int*      int_rc,
                                            const int*      int_par,
                                            const uint32_t* int_mark,
                                            AABB*           int_aabb,
                                            uint32_t*       flags)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size || size <= 1)
        return;

    int cur = static_cast<int>(ext_par[idx]);

    __threadfence();
    while(atomicAdd(&flags[cur], 1) == 1)
    {
        int      chl  = int_lc[cur];
        int      chr  = int_rc[cur];
        uint32_t mark = int_mark[cur];

        AABB box = (mark & 1) ? ext_aabb[chl] : int_aabb[chl];
        if(mark & 2)
            box.extend(ext_aabb[chr]);
        else
            box.extend(int_aabb[chr]);
        int_aabb[cur] = box;

        int par = int_par[cur];
        if(par == -1)
            break;
        __threadfence();
        cur = par;
    }
}

}  // namespace uipc::backend::cuda::corex_bvh

namespace uipc::backend::cuda
{
namespace
{
struct CorexVector2iLess
{
    MUDA_HOST MUDA_DEVICE bool operator()(const Vector2i& a, const Vector2i& b) const
    {
        return a(0) < b(0) || (a(0) == b(0) && a(1) < b(1));
    }
};

bool corex_sort_nomask_pairs_enabled()
{
    static const bool enabled = []
    {
        const char* env = std::getenv("UIPC_COREX_SORT_NOMASK_PAIRS");
        return env && env[0] != '\0' && env[0] != '0';
    }();
    return enabled;
}

inline void corex_sort_nomask_pairs(StacklessBVH::QueryBuffer& qbuffer, int pair_count)
{
    if(pair_count <= 1 || !corex_sort_nomask_pairs_enabled())
        return;
    corex_profile::ScopedPhase phase("bvh_query_detail", "nomask_sort_pairs");
    auto begin = thrust::device_pointer_cast(qbuffer.m_pairs.data());
    thrust::sort(thrust::device, begin, begin + pair_count, CorexVector2iLess{});
}
}  // namespace

MUDA_INLINE void StacklessBVH::Impl::calcMaxBVFromBox(muda::CBufferView<AABB> aabbs,
                                                      muda::VarView<AABB> scene_box)
{
    using namespace culbvh;

    auto numQuery = aabbs.size();
    auto BlockDim = K_THREADS;
    auto GridDim  = (numQuery + BlockDim - 1) / BlockDim;

    using namespace muda;

    kernel_init_scene_box<<<1, 1>>>(
        reinterpret_cast<AABB*>(scene_box.data()));
    checkCudaErrors(cudaGetLastError());

    kernel_reduce_scene_box<<<GridDim, K_THREADS>>>(
        static_cast<int>(numQuery),
        reinterpret_cast<const AABB*>(aabbs.data()),
        reinterpret_cast<AABB*>(scene_box.data()));
    checkCudaErrors(cudaGetLastError());
    return;

    Launch(GridDim, BlockDim)
        .file_line(__FILE__, __LINE__)
        .apply(
            [size = aabbs.size(),
             box  = aabbs.viewer().name("box"),
             _bv  = scene_box.viewer().name("_bv")] __device__()
            {
                int idx     = blockIdx.x * blockDim.x + threadIdx.x;
                int warpTid = threadIdx.x % 32;
                int warpId  = (threadIdx.x >> 5);
                int warpNum;
                if(idx >= size)
                    return;
                __shared__ PlainAABB aabbData[K_WARPS];

                PlainAABB temp = toPlainAABB(box(idx));
                __syncthreads();

                // Extract values for warp shuffle
                float tempMinX = temp._min.x;
                float tempMinY = temp._min.y;
                float tempMinZ = temp._min.z;
                float tempMaxX = temp._max.x;
                float tempMaxY = temp._max.y;
                float tempMaxZ = temp._max.z;

                for(int i = 1; i < 32; i = (i << 1))
                {
                    float otherMinX = __shfl_down_sync(0xffffffff, tempMinX, i);
                    float otherMinY = __shfl_down_sync(0xffffffff, tempMinY, i);
                    float otherMinZ = __shfl_down_sync(0xffffffff, tempMinZ, i);
                    float otherMaxX = __shfl_down_sync(0xffffffff, tempMaxX, i);
                    float otherMaxY = __shfl_down_sync(0xffffffff, tempMaxY, i);
                    float otherMaxZ = __shfl_down_sync(0xffffffff, tempMaxZ, i);
                    tempMinX        = __mm_min(tempMinX, otherMinX);
                    tempMinY        = __mm_min(tempMinY, otherMinY);
                    tempMinZ        = __mm_min(tempMinZ, otherMinZ);
                    tempMaxX        = __mm_max(tempMaxX, otherMaxX);
                    tempMaxY        = __mm_max(tempMaxY, otherMaxY);
                    tempMaxZ        = __mm_max(tempMaxZ, otherMaxZ);
                }

                if(blockIdx.x == gridDim.x - 1)
                {
                    warpNum = ((size - blockIdx.x * blockDim.x + 31) >> 5);
                }
                else
                {
                    warpNum = ((blockDim.x) >> 5);
                }

                if(warpTid == 0)
                {
                    // Reconstruct AABB from reduced values
                    aabbData[warpId]._min = make_float3(tempMinX, tempMinY, tempMinZ);
                    aabbData[warpId]._max = make_float3(tempMaxX, tempMaxY, tempMaxZ);
                }
                __syncthreads();
                if(threadIdx.x >= warpNum)
                    return;

                if(warpNum > 1)
                {
                    temp     = aabbData[threadIdx.x];
                    tempMinX = temp._min.x;
                    tempMinY = temp._min.y;
                    tempMinZ = temp._min.z;
                    tempMaxX = temp._max.x;
                    tempMaxY = temp._max.y;
                    tempMaxZ = temp._max.z;

                    for(int i = 1; i < warpNum; i = (i << 1))
                    {
                        float otherMinX = __shfl_down_sync(0xffffffff, tempMinX, i);
                        float otherMinY = __shfl_down_sync(0xffffffff, tempMinY, i);
                        float otherMinZ = __shfl_down_sync(0xffffffff, tempMinZ, i);
                        float otherMaxX = __shfl_down_sync(0xffffffff, tempMaxX, i);
                        float otherMaxY = __shfl_down_sync(0xffffffff, tempMaxY, i);
                        float otherMaxZ = __shfl_down_sync(0xffffffff, tempMaxZ, i);
                        tempMinX = __mm_min(tempMinX, otherMinX);
                        tempMinY = __mm_min(tempMinY, otherMinY);
                        tempMinZ = __mm_min(tempMinZ, otherMinZ);
                        tempMaxX = __mm_max(tempMaxX, otherMaxX);
                        tempMaxY = __mm_max(tempMaxY, otherMaxY);
                        tempMaxZ = __mm_max(tempMaxZ, otherMaxZ);
                    }
                }

                if(threadIdx.x == 0)
                {
                    atomicMinf(&_bv->min().x(), tempMinX);
                    atomicMinf(&_bv->min().y(), tempMinY);
                    atomicMinf(&_bv->min().z(), tempMinZ);
                    atomicMaxf(&_bv->max().x(), tempMaxX);
                    atomicMaxf(&_bv->max().y(), tempMaxY);
                    atomicMaxf(&_bv->max().z(), tempMaxZ);
                }
            });
}

MUDA_INLINE void StacklessBVH::Impl::calcMCsFromBox(muda::CBufferView<AABB> aabbs,
                                                    muda::CVarView<AABB> scene_box,
                                                    muda::BufferView<uint32_t> codes)
{
    using namespace culbvh;
    using namespace muda;
    int N = aabbs.size();
    if(N == 0) return;

    int block = 256;
    int grid = (N + block - 1) / block;
    corex_bvh::kernel_calcMCs<<<grid, block>>>(N, (const AABB*)aabbs.data(), (const AABB*)scene_box.data(), (uint32_t*)codes.data());
    checkCudaErrors(cudaGetLastError());
}

/// incoherent access, thus poor performance
MUDA_INLINE void StacklessBVH::Impl::calcInverseMapping()
{
    using namespace muda;
    int N = sorted_id.size();
    if(N == 0) return;

    int block = 256, grid = (N + block - 1) / block;
    corex_bvh::kernel_calcInverseMapping<<<grid, block>>>(N, RAW_PTR(sorted_id), RAW_PTR(primMap));
    checkCudaErrors(cudaGetLastError());
}

MUDA_INLINE void StacklessBVH::Impl::buildPrimitivesFromBox(muda::CBufferView<AABB> aabbs)
{
    using namespace muda;
    int N = aabbs.size();
    if(N == 0) return;

    int block = 256, grid = (N + block - 1) / block;
    corex_bvh::kernel_buildPrimitives<<<grid, block>>>(N, RAW_PTR(primMap), (const AABB*)aabbs.data(),
                                                        RAW_PTR(ext_idx), RAW_PTR(ext_aabb));
    checkCudaErrors(cudaGetLastError());
}


MUDA_INLINE void StacklessBVH::Impl::calcExtNodeSplitMetrics(
    muda::CBufferView<uint32_t> sorted_codes)
{
    using namespace muda;
    int N = sorted_codes.size();
    if(N == 0) return;

    int block = 256, grid = (N + block - 1) / block;
    corex_bvh::kernel_calcSplitMetrics<<<grid, block>>>(
        N, (const uint32_t*)sorted_codes.data(), RAW_PTR(metric));
    checkCudaErrors(cudaGetLastError());
}

MUDA_INLINE void StacklessBVH::Impl::buildIntNodes(int size)
{
    using namespace muda;

    auto GridDim  = (size + 255) / 256;
    auto BlockDim = 256;

    Launch(GridDim, BlockDim)
        .file_line(__FILE__, __LINE__)
        .apply(
            [size = size,
             // leaf nodes
             _depths     = count.viewer().name("_depths"),
             _lvs_lca    = ext_lca.viewer().name("_lvs_lca"),
             _lvs_metric = metric.viewer().name("_lvs_metric"),
             _lvs_par    = ext_par.viewer().name("_lvs_par"),
             _lvs_mark   = ext_mark.viewer().name("_lvs_mark"),
             _lvs_box    = ext_aabb.viewer().name("_lvs_box"),
             // internal nodes
             _tks_rc      = int_rc.viewer().name("_tks_rc"),
             _tks_lc      = int_lc.viewer().name("_tks_lc"),
             _tks_range_y = int_range_y.viewer().name("_tks_range_y"),
             _tks_range_x = int_range_x.viewer().name("_tks_range_x"),
             _tks_mark    = int_mark.viewer().name("_tks_mark"),
             _tks_box     = int_aabb.viewer().name("_tks_box"),
             _flag        = flags.viewer().name("_flag"),
             _tks_par     = int_par.viewer().name("_tks_par")] __device__()
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if(idx >= size)
                    return;

                _lvs_lca(idx) = -1, _depths(idx) = 0;
                int  l = idx - 1, r = idx;  ///< (l, r]
                bool mark;
                if(l >= 0)
                    mark = _lvs_metric(l) < _lvs_metric(r);  //determine direction
                else
                    mark = false;
                int cur = mark ? l : r;

                _lvs_par(idx) = cur;


                if(_flag.total_size() == 0)
                    // when we only have 1 external node
                    // there is no internal node to build
                    return;

                if(mark)
                {
                    _tks_rc(cur)      = idx;
                    _tks_range_y(cur) = idx;
                    atomicOr(&_tks_mark(cur), 0x00000002);
                    _lvs_mark(idx) = 0x00000007;
                }
                else
                {
                    _tks_lc(cur)      = idx;
                    _tks_range_x(cur) = idx;
                    atomicOr(&_tks_mark(cur), 0x00000001);
                    _lvs_mark(idx) = 0x00000003;
                }

                __threadfence();
                while(atomicAdd(&_flag(cur), 1) == 1)
                {
                    //_tks.update(cur, _lvs);	/// Update
                    //_tks.refit(cur, _lvs);	/// Refit
                    int      chl       = _tks_lc(cur);
                    int      chr       = _tks_rc(cur);
                    uint32_t temp_mark = _tks_mark(cur);
                    if(temp_mark & 1)
                    {
                        _tks_box(cur) = _lvs_box(chl);
                    }
                    else
                    {
                        _tks_box(cur) = _tks_box(chl);
                    }
                    if(temp_mark & 2)
                    {
                        _tks_box(cur).extend(_lvs_box(chr));
                    }
                    else
                    {
                        _tks_box(cur).extend(_tks_box(chr));
                    }

                    _tks_mark(cur) &= 0x00000007;

                    l               = _tks_range_x(cur) - 1;
                    r               = _tks_range_y(cur);
                    _lvs_lca(l + 1) = cur;
                    _depths(l + 1)++;
                    if(l >= 0)
                    {
                        mark = _lvs_metric(l) < _lvs_metric(r);  ///< true when right child, false otherwise
                    }
                    else
                    {
                        mark = false;
                    }

                    if(l + 1 == 0 && r == size - 1)
                    {
                        _tks_par(cur) = -1;
                        _tks_mark(cur) &= 0xFFFFFFFB;
                        break;
                    }

                    int par       = mark ? l : r;
                    _tks_par(cur) = par;
                    if(mark)
                    {
                        _tks_rc(par)      = cur;
                        _tks_range_y(par) = r;
                        atomicAnd(&_tks_mark(par), 0xFFFFFFFD);
                        _tks_mark(cur) |= 0x00000004;
                    }
                    else
                    {
                        _tks_lc(par)      = cur;
                        _tks_range_x(par) = l + 1;
                        atomicAnd(&_tks_mark(par), 0xFFFFFFFE);
                        _tks_mark(cur) &= 0xFFFFFFFB;
                    }
                    __threadfence();
                    cur = par;
                }
            });
}

MUDA_INLINE void StacklessBVH::Impl::calcIntNodeOrders(int size)
{
    using namespace muda;
    if(size == 0) return;

    int block = 256, grid = (size + block - 1) / block;
    corex_bvh::kernel_calcIntNodeOrders<<<grid, block>>>(size, RAW_PTR(int_lc), RAW_PTR(ext_lca),
                                                          RAW_PTR(count), RAW_PTR(offsetTable), RAW_PTR(tkMap));
    checkCudaErrors(cudaGetLastError());
}

MUDA_INLINE void StacklessBVH::Impl::updateBvhExtNodeLinks(int size)
{
    using namespace muda;

    if(flags.size() == 0)
        return;
    if(size == 0) return;

    int block = 256, grid = (size + block - 1) / block;
    corex_bvh::kernel_updateBvhExtNodeLinks<<<grid, block>>>(size, RAW_PTR(tkMap), RAW_PTR(ext_lca), RAW_PTR(ext_par));
    checkCudaErrors(cudaGetLastError());
}

MUDA_INLINE void StacklessBVH::Impl::reorderNode(int intSize)
{
    using namespace culbvh;
    using namespace muda;
    int N = intSize + 1;
    if(N == 0) return;

    int block = 256, grid = (N + block - 1) / block;
    corex_bvh::kernel_reorderNode<<<grid, block>>>(N, intSize,
        RAW_PTR(ext_lca), RAW_PTR(ext_aabb),
        RAW_PTR(tkMap), RAW_PTR(int_lc), RAW_PTR(int_mark), RAW_PTR(int_range_y),
        RAW_PTR(int_aabb), RAW_PTR(nodes), RAW_PTR(node_range_y));
    checkCudaErrors(cudaGetLastError());
}

MUDA_INLINE void StacklessBVH::Impl::updateRefitNodeBounds(int intSize)
{
    using namespace culbvh;
    using namespace muda;
    int N = intSize + 1;
    if(N == 0) return;

    int block = 256, grid = (N + block - 1) / block;
    corex_bvh::kernel_updateRefitNodeBounds<<<grid, block>>>(N,
                                                             intSize,
                                                             RAW_PTR(ext_aabb),
                                                             RAW_PTR(tkMap),
                                                             RAW_PTR(int_aabb),
                                                             RAW_PTR(nodes));
    checkCudaErrors(cudaGetLastError());
}

MUDA_INLINE bool StacklessBVH::Impl::can_refit(muda::CBufferView<AABB> aabbs) const
{
    auto numObjs = aabbs.size();
    if(numObjs == 0)
        return false;
    if(primMap.size() != numObjs)
        return false;
    if(nodes.size() != numObjs * 2 - 1)
        return false;
    return true;
}

inline void StacklessBVH::Impl::refit(muda::CBufferView<AABB> aabbs)
{
    using namespace muda;

    if(aabbs.size() == 0)
    {
        objs = aabbs;
        return;
    }

    UIPC_ASSERT(can_refit(aabbs),
                "StacklessBVH::refit requires a previous build with the same primitive count");

    objs         = aabbs;
    auto numObjs = static_cast<int>(aabbs.size());
    const int numInternalNodes = numObjs - 1;

    buildPrimitivesFromBox(aabbs);

    if(numInternalNodes > 0)
    {
        int block = 256;
        int flag_grid = (static_cast<int>(flags.size()) + block - 1) / block;
        corex_bvh::kernel_fill_u32<<<flag_grid, block>>>(
            RAW_PTR(flags), static_cast<int>(flags.size()), uint32_t{0});
        checkCudaErrors(cudaGetLastError());

        int grid  = (numObjs + block - 1) / block;
        corex_bvh::kernel_refitIntNodes<<<grid, block>>>(numObjs,
                                                         RAW_PTR(ext_par),
                                                         RAW_PTR(ext_aabb),
                                                         RAW_PTR(int_lc),
                                                         RAW_PTR(int_rc),
                                                         RAW_PTR(int_par),
                                                         RAW_PTR(int_mark),
                                                         RAW_PTR(int_aabb),
                                                         RAW_PTR(flags));
        checkCudaErrors(cudaGetLastError());
    }

    updateRefitNodeBounds(numInternalNodes);
}

inline void StacklessBVH::Impl::build(muda::CBufferView<AABB> aabbs)
{
    objs         = aabbs;
    auto numObjs = aabbs.size();

    if(aabbs.size() == 0)
        return;

    const unsigned int numInternalNodes = numObjs - 1;  // Total number of internal nodes
    const unsigned int numNodes = numObjs * 2 - 1;  // Total number of nodes


    mtcode.resize(numObjs);
    mtcode_sorted.resize(numObjs);
    sorted_id_input.resize(numObjs);
    sorted_id.resize(numObjs);
    primMap.resize(numObjs);
    ext_aabb.resize(numObjs);
    ext_idx.resize(numObjs);
    ext_lca.resize(numObjs + 1);
    ext_par.resize(numObjs);
    ext_mark.resize(numObjs);

    metric.resize(numObjs);
    tkMap.resize(numObjs);
    offsetTable.resize(numObjs);
    count.resize(numObjs);

    flags.resize(numInternalNodes);
    int_lc.resize(numInternalNodes);
    int_rc.resize(numInternalNodes);
    int_par.resize(numInternalNodes);
    int_range_x.resize(numInternalNodes);
    int_range_y.resize(numInternalNodes);
    int_mark.resize(numInternalNodes);
    int_aabb.resize(numInternalNodes);

    nodes.resize(numNodes);
    node_range_y.resize(numNodes);


    if(flags.size() > 0)
        checkCudaErrors(cudaMemsetAsync(RAW_PTR(flags), 0, sizeof(uint32_t) * flags.size()));
    if(ext_lca.size() > 0)
        checkCudaErrors(cudaMemsetAsync(RAW_PTR(ext_lca), 0, sizeof(int) * ext_lca.size()));
    if(ext_par.size() > 0)
        checkCudaErrors(cudaMemsetAsync(RAW_PTR(ext_par), 0, sizeof(uint32_t) * ext_par.size()));
    if(ext_mark.size() > 0)
    {
        constexpr int block = 256;
        int grid = (static_cast<int>(ext_mark.size()) + block - 1) / block;
        corex_bvh::kernel_fill_u32<<<grid, block>>>(
            RAW_PTR(ext_mark), static_cast<int>(ext_mark.size()), uint32_t{7});
        checkCudaErrors(cudaGetLastError());
    }

    {
        calcMaxBVFromBox(aabbs, scene_box.view());
    }

    {
        calcMCsFromBox(aabbs, scene_box.view(), mtcode.view());
    }

    auto null_stream = thrust::cuda::par.on(nullptr);

    {
        thrust::sequence(null_stream,
                         sorted_id_input.data(),
                         sorted_id_input.data() + sorted_id_input.size());
        muda::DeviceRadixSort().SortPairs(mtcode.data(),
                                          mtcode_sorted.data(),
                                          sorted_id_input.data(),
                                          sorted_id.data(),
                                          mtcode.size(),
                                          0,
                                          30);
    }

    {
        calcInverseMapping();
    }

    {
        buildPrimitivesFromBox(aabbs);
    }

    {
        calcExtNodeSplitMetrics(mtcode_sorted.view());
    }

    {
        buildIntNodes(numObjs);
    }

    {
        thrust::exclusive_scan(
            null_stream, count.data(), count.data() + count.size(), offsetTable.data());
    }

    {
        calcIntNodeOrders(numObjs);
    }

    {
        // fill the last ext_lca to -1
        thrust::fill(null_stream, ext_lca.data() + numObjs, ext_lca.data() + numObjs + 1, -1);
        updateBvhExtNodeLinks(numObjs);
    }

    {
        reorderNode(numInternalNodes);
    }
}

template <typename Pred>
void StacklessBVH::Impl::StacklessCDSharedSelf(Pred               pred,
                                               muda::VarView<int> cpNum,
                                               muda::BufferView<Vector2i> buffer)
{
    using namespace muda;
    using namespace culbvh;

    auto numQuery = static_cast<int>(ext_aabb.size());
    auto numObjs  = numQuery;
    auto BlockDim = K_THREADS;
    auto GridDim  = (numQuery + BlockDim - 1) / BlockDim;

    Launch(GridDim, BlockDim)
        .apply(
            [Size       = numQuery,
             _box       = objs.viewer().name("_box"),
             intSize    = numObjs - 1,
             numObjs    = numObjs,
             _lvs_idx   = ext_idx.viewer().name("_lvs_idx"),
             _nodes     = nodes.viewer().name("_nodes"),
             _node_range_y = node_range_y.viewer().name("_node_range_y"),
             resCounter = cpNum.viewer().name("resCounter"),
             res        = buffer.viewer().name("res"),
             pred] __device__()
            {
                int  tid    = blockIdx.x * blockDim.x + threadIdx.x;
                bool active = tid < Size;
                int  idx;
                AABB bv;
                if(active)
                {
                    idx = _lvs_idx(tid);
                    bv  = _box(idx);
                }

                __shared__ int2 sharedRes[MAX_RES_PER_BLOCK];
                __shared__ int sharedCounter;  // How many results are cached in shared memory
                __shared__ int sharedGlobalIdx;  // Where to write in global memory
                if(threadIdx.x == 0)
                    sharedCounter = 0;

                int  st = 0;
                Node node;
                // Upper bound of iterations to avoid infinite loop
                const int MaxIter = numObjs * 2;

                while(true)
                {
                    __syncthreads();
                    if(active)
                    {
                        int inner_I = 0;
                        for(; inner_I < MaxIter; inner_I++)
                        {
                            if(st == -1)
                                break;
                            // Load node data - Eigen::AlignedBox stores min and max as Vector3f members
                            node.lc     = _nodes(st).lc;
                            node.escape = _nodes(st).escape;
                            node.bound  = _nodes(st).bound;
                            if(_node_range_y(st) <= tid)
                            {
                                st = node.escape;
                                continue;
                            }
                            //node = _nodes[st];
                            if(node.bound.intersects(bv))
                            {
                                if(node.lc == -1)
                                {
                                    if(tid < st - intSize)
                                    {
                                        auto pair =
                                            make_ordered_pair(idx, _lvs_idx(st - intSize));
                                        if(pred(pair.x, pair.y))
                                        {
                                            int sIdx = atomicAdd(&sharedCounter, 1);
                                            if(sIdx >= MAX_RES_PER_BLOCK)
                                            {
                                                break;
                                            }

                                            sharedRes[sIdx] = pair;
                                        }
                                    }
                                    st = node.escape;
                                }
                                else
                                {
                                    st = node.lc;
                                }
                            }
                            else
                            {
                                st = node.escape;
                            }
                        }

                        MUDA_ASSERT(inner_I < MaxIter,
                                    "Exceeded max iteration in stackless traversal, %d (Max=%d), numObj=(%d)",
                                    inner_I,
                                    MaxIter,
                                    numObjs);
                    }
                    // Flush whatever we have
                    __syncthreads();

                    int totalResInBlock = min(sharedCounter, MAX_RES_PER_BLOCK);

                    if(threadIdx.x == 0)
                    {
                        // This Block Starts writing at sharedGlobalIdx
                        sharedGlobalIdx = atomicAdd(resCounter.data(), totalResInBlock);
                    }

                    __syncthreads();

                    // Make sure we dont write out of bounds
                    const int globalIdx = sharedGlobalIdx;

                    if(threadIdx.x == 0)
                        sharedCounter = 0;

                    // if there is at least one element empty
                    // it means we have found all collisions for this block
                    bool done = totalResInBlock < MAX_RES_PER_BLOCK;

                    SafeCopyTo(sharedRes,
                               totalResInBlock,
                               res.data(),
                               globalIdx,
                               static_cast<int>(res.total_size()));

                    if(done)
                        break;
                }
	        });
}

inline void StacklessBVH::Impl::StacklessCDSharedSelfEdgesNoMask(
    muda::CBufferView<Vector2i> edges,
    muda::CBufferView<IndexT>   vertex_to_body,
    muda::CBufferView<IndexT>   body_self_collision,
    muda::VarView<int>          cpNum,
    muda::BufferView<Vector2i>  buffer)
{
    using namespace muda;
    using namespace culbvh;

    auto numQuery = static_cast<int>(ext_aabb.size());
    auto numObjs  = numQuery;
    auto BlockDim = K_THREADS;
    auto GridDim  = (numQuery + BlockDim - 1) / BlockDim;

    Launch(GridDim, BlockDim)
        .apply(
            [Size       = numQuery,
             _box       = objs.viewer().name("_box"),
             intSize    = numObjs - 1,
             numObjs    = numObjs,
             _lvs_idx   = ext_idx.viewer().name("_lvs_idx"),
             _nodes     = nodes.viewer().name("_nodes"),
             _node_range_y = node_range_y.viewer().name("_node_range_y"),
             edges      = edges.viewer().name("edges"),
             v2b        = vertex_to_body.viewer().name("v2b"),
             body_self_collision = body_self_collision.viewer().name("body_self_collision"),
             resCounter = cpNum.viewer().name("resCounter"),
             res        = buffer.viewer().name("res")] __device__()
            {
                int  tid    = blockIdx.x * blockDim.x + threadIdx.x;
                bool active = tid < Size;
                int  idx;
                Vector2i E0;
                IndexT body_i = -1;
                AABB bv;
                if(active)
                {
                    idx    = _lvs_idx(tid);
                    bv     = _box(idx);
                    E0     = edges(idx);
                    body_i = v2b(E0[0]);
                }

                __shared__ int2 sharedRes[MAX_RES_PER_BLOCK];
                __shared__ int  sharedCounter;
                __shared__ int  sharedGlobalIdx;
                if(threadIdx.x == 0)
                    sharedCounter = 0;

                int  st = 0;
                Node node;
                const int MaxIter = numObjs * 2;

                while(true)
                {
                    __syncthreads();
                    if(active)
                    {
                        int inner_I = 0;
                        for(; inner_I < MaxIter; inner_I++)
                        {
                            if(st == -1)
                                break;

                            node.lc     = _nodes(st).lc;
                            node.escape = _nodes(st).escape;
                            node.bound  = _nodes(st).bound;
                            if(_node_range_y(st) <= tid)
                            {
                                st = node.escape;
                                continue;
                            }

                            if(node.bound.intersects(bv))
                            {
                                if(node.lc == -1)
                                {
                                    if(tid < st - intSize)
                                    {
                                        const int j  = _lvs_idx(st - intSize);
                                        const auto E1 = edges(j);

                                        const bool shared_vertex =
                                            E0[0] == E1[0] || E0[0] == E1[1]
                                            || E0[1] == E1[0] || E0[1] == E1[1];
                                        bool accept = !shared_vertex;
                                        if(accept)
                                        {
                                            const auto body_j = v2b(E1[0]);
                                            accept = !(body_i == body_j
                                                       && !body_self_collision(body_i));
                                        }
                                        if(accept)
                                        {
                                            auto pair = make_ordered_pair(idx, j);
                                            int  sIdx = atomicAdd(&sharedCounter, 1);
                                            if(sIdx >= MAX_RES_PER_BLOCK)
                                                break;

                                            sharedRes[sIdx] = pair;
                                        }
                                    }
                                    st = node.escape;
                                }
                                else
                                {
                                    st = node.lc;
                                }
                            }
                            else
                            {
                                st = node.escape;
                            }
                        }

                        MUDA_ASSERT(inner_I < MaxIter,
                                    "Exceeded max iteration in stackless traversal, %d (Max=%d), numObj=(%d)",
                                    inner_I,
                                    MaxIter,
                                    numObjs);
                    }

                    __syncthreads();
                    int totalResInBlock = min(sharedCounter, MAX_RES_PER_BLOCK);

                    if(threadIdx.x == 0)
                        sharedGlobalIdx = atomicAdd(resCounter.data(), totalResInBlock);

                    __syncthreads();

                    const int globalIdx = sharedGlobalIdx;

                    if(threadIdx.x == 0)
                        sharedCounter = 0;

                    bool done = totalResInBlock < MAX_RES_PER_BLOCK;

                    SafeCopyTo(sharedRes,
                               totalResInBlock,
                               res.data(),
                               globalIdx,
                               static_cast<int>(res.total_size()));

                    if(done)
                        break;
                }
            });
}

inline void StacklessBVH::Impl::StacklessCDSharedSelfEdgesActiveNoMask(
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
    muda::BufferView<IndexT>    selected_counts)
{
    using namespace muda;
    using namespace culbvh;

    auto numQuery = static_cast<int>(ext_aabb.size());
    auto numObjs  = numQuery;
    auto BlockDim = K_THREADS;
    auto GridDim  = (numQuery + BlockDim - 1) / BlockDim;
    const int pp_cap = static_cast<int>(out_PPs.size());
    const int pe_cap = static_cast<int>(out_PEs.size());
    const int ee_cap = static_cast<int>(out_EEs.size());

    Launch(GridDim, BlockDim)
        .apply(
            [Size       = numQuery,
             _box       = objs.viewer().name("_box"),
             intSize    = numObjs - 1,
             numObjs    = numObjs,
             _lvs_idx   = ext_idx.viewer().name("_lvs_idx"),
             _nodes     = nodes.viewer().name("_nodes"),
             _node_range_y = node_range_y.viewer().name("_node_range_y"),
             edges      = edges.viewer().name("edges"),
             Ps         = positions.viewer().name("positions"),
             dxs        = displacements.viewer().name("displacements"),
             rest_Ps    = rest_positions.viewer().name("rest_positions"),
             edge_thicknesses = edge_thicknesses.viewer().name("edge_thicknesses"),
             edge_d_hats = edge_d_hats.viewer().name("edge_d_hats"),
             alpha,
             v2b        = vertex_to_body.viewer().name("v2b"),
             body_self_collision = body_self_collision.viewer().name("body_self_collision"),
             out_PPs    = out_PPs.viewer().name("out_PPs"),
             out_PEs    = out_PEs.viewer().name("out_PEs"),
             out_EEs    = out_EEs.viewer().name("out_EEs"),
             counts     = selected_counts.viewer().name("selected_counts"),
             pp_cap,
             pe_cap,
             ee_cap] __device__()
            {
                int  tid    = blockIdx.x * blockDim.x + threadIdx.x;
                bool active = tid < Size;
                int  idx;
                Vector2i E0;
                IndexT body_i = -1;
                AABB bv;
                if(active)
                {
                    idx    = _lvs_idx(tid);
                    bv     = _box(idx);
                    E0     = edges(idx);
                    body_i = v2b(E0[0]);
                }

                int  st = 0;
                Node node;
                const int MaxIter = numObjs * 2;

                if(active)
                {
                    int inner_I = 0;
                    for(; inner_I < MaxIter; inner_I++)
                    {
                        if(st == -1)
                            break;

                        node.lc     = _nodes(st).lc;
                        node.escape = _nodes(st).escape;
                        node.bound  = _nodes(st).bound;
                        if(_node_range_y(st) <= tid)
                        {
                            st = node.escape;
                            continue;
                        }

                        if(node.bound.intersects(bv))
                        {
                            if(node.lc == -1)
                            {
                                if(tid < st - intSize)
                                {
                                    const int j  = _lvs_idx(st - intSize);
                                    const auto E1 = edges(j);

                                    const bool shared_vertex =
                                        E0[0] == E1[0] || E0[0] == E1[1]
                                        || E0[1] == E1[0] || E0[1] == E1[1];
                                    bool accept = !shared_vertex;
                                    if(accept)
                                    {
                                        const auto body_j = v2b(E1[0]);
                                        accept = !(body_i == body_j
                                                   && !body_self_collision(body_i));
                                    }
                                    if(accept)
                                    {
                                        const Vector4i vIs = {E0[0], E0[1], E1[0], E1[1]};
                                        Vector3 Ps_arr[] = {Ps(vIs(0)),
                                                            Ps(vIs(1)),
                                                            Ps(vIs(2)),
                                                            Ps(vIs(3))};
                                        Float thickness = edge_thicknesses(idx)
                                                        + edge_thicknesses(j);
                                        Float d_hat = (edge_d_hats(idx) + edge_d_hats(j))
                                                    * Float{0.5};
                                        Vector2 range = D_range(thickness, d_hat);
                                        Vector3 e0_min = Ps_arr[0].cwiseMin(Ps_arr[1]);
                                        Vector3 e0_max = Ps_arr[0].cwiseMax(Ps_arr[1]);
                                        Vector3 e1_min = Ps_arr[2].cwiseMin(Ps_arr[3]);
                                        Vector3 e1_max = Ps_arr[2].cwiseMax(Ps_arr[3]);
                                        if(box_box_distance2_lower_bound(e0_min,
                                                                         e0_max,
                                                                         e1_min,
                                                                         e1_max)
                                           >= range.y())
                                        {
                                            st = node.escape;
                                            continue;
                                        }
                                        if(alpha != static_cast<Float>(0))
                                        {
                                            Vector3 dE0_0 = alpha * dxs(E0[0]);
                                            Vector3 dE0_1 = alpha * dxs(E0[1]);
                                            Vector3 dE1_0 = alpha * dxs(E1[0]);
                                            Vector3 dE1_1 = alpha * dxs(E1[1]);
                                            if(!distance::edge_edge_ccd_broadphase(
                                                   Ps_arr[0],
                                                   Ps_arr[1],
                                                   Ps_arr[2],
                                                   Ps_arr[3],
                                                   dE0_0,
                                                   dE0_1,
                                                   dE1_0,
                                                   dE1_1,
                                                   d_hat + thickness))
                                            {
                                                st = node.escape;
                                                continue;
                                            }
                                        }
                                        Vector4i flag = distance::edge_edge_distance_flag(
                                            Ps_arr[0], Ps_arr[1], Ps_arr[2], Ps_arr[3]);
                                        Float D;
                                        distance::edge_edge_distance2(
                                            flag, Ps_arr[0], Ps_arr[1], Ps_arr[2], Ps_arr[3], D);
                                        if(D <= range.x())
                                        {
                                            IndexT dst = atomicAdd(&counts(3), IndexT{1});
                                            if(dst < ee_cap)
                                                out_EEs(dst) = vIs;
                                        }
                                        else if(is_active_D(range, D))
                                        {
                                            Vector4i offsets;
                                            offsets.setConstant(-1);
                                            auto dim = distance::degenerate_edge_edge(flag, offsets);
                                            if(dim == 4)
                                            {
                                                Float eps_x;
                                                distance::edge_edge_mollifier_threshold(
                                                    rest_Ps(vIs(0)),
                                                    rest_Ps(vIs(1)),
                                                    rest_Ps(vIs(2)),
                                                    rest_Ps(vIs(3)),
                                                    static_cast<Float>(1e-3),
                                                    eps_x);
                                                if(distance::need_mollify(Ps_arr[0],
                                                                          Ps_arr[1],
                                                                          Ps_arr[2],
                                                                          Ps_arr[3],
                                                                          eps_x))
                                                {
                                                    IndexT dst = atomicAdd(&counts(3), IndexT{1});
                                                    if(dst < ee_cap)
                                                        out_EEs(dst) = vIs;
                                                    st = node.escape;
                                                    continue;
                                                }
                                            }
                                            if(dim == 2)
                                            {
                                                IndexT dst = atomicAdd(&counts(0), IndexT{1});
                                                if(dst < pp_cap)
                                                    out_PPs(dst) = {vIs(offsets(0)), vIs(offsets(1))};
                                            }
                                            else if(dim == 3)
                                            {
                                                IndexT dst = atomicAdd(&counts(1), IndexT{1});
                                                if(dst < pe_cap)
                                                    out_PEs(dst) = {vIs(offsets(0)),
                                                                    vIs(offsets(1)),
                                                                    vIs(offsets(2))};
                                            }
                                            else if(dim == 4)
                                            {
                                                IndexT dst = atomicAdd(&counts(3), IndexT{1});
                                                if(dst < ee_cap)
                                                    out_EEs(dst) = vIs;
                                            }
                                        }
                                    }
                                }
                                st = node.escape;
                            }
                            else
                            {
                                st = node.lc;
                            }
                        }
                        else
                        {
                            st = node.escape;
                        }
                    }

                    MUDA_ASSERT(inner_I < MaxIter,
                                "Exceeded max iteration in stackless traversal, %d (Max=%d), numObj=(%d)",
                                inner_I,
                                MaxIter,
                                numObjs);
                }
            });
}

inline void StacklessBVH::Impl::StacklessCDSharedOtherPointsTrianglesNoMask(
    muda::CBufferView<AABB>     point_aabbs,
    muda::CBufferView<IndexT>   surf_vertices,
    muda::CBufferView<Vector3i> surf_triangles,
    muda::CBufferView<Vector3>  positions,
    muda::CBufferView<Vector3>  displacements,
    muda::CBufferView<Float>    thicknesses,
    muda::CBufferView<Float>    d_hats,
    Float                       alpha,
    muda::CBufferView<IndexT>   vertex_to_body,
    muda::CBufferView<IndexT>   body_self_collision,
    muda::VarView<int>          cpNum,
    muda::BufferView<Vector2i>  buffer)
{
    using namespace muda;
    using namespace culbvh;

    auto numQuery = static_cast<int>(point_aabbs.size());
    auto numObjs  = static_cast<int>(ext_aabb.size());
    auto BlockDim = K_THREADS;
    auto GridDim  = (numQuery + BlockDim - 1) / BlockDim;

    Launch(GridDim, BlockDim)
        .apply(
            [Size       = numQuery,
             _box       = point_aabbs.viewer().name("_box"),
             intSize    = numObjs - 1,
             numObjs    = numObjs,
             _lvs_idx   = ext_idx.viewer().name("_lvs_idx"),
             _nodes     = nodes.viewer().name("_nodes"),
             Vs         = surf_vertices.viewer().name("Vs"),
             Fs         = surf_triangles.viewer().name("Fs"),
             Ps         = positions.viewer().name("positions"),
             dxs        = displacements.viewer().name("displacements"),
             thicknesses = thicknesses.viewer().name("thicknesses"),
             d_hats     = d_hats.viewer().name("d_hats"),
             alpha,
             v2b        = vertex_to_body.viewer().name("v2b"),
             body_self_collision = body_self_collision.viewer().name("body_self_collision"),
             resCounter = cpNum.viewer().name("resCounter"),
             res        = buffer.viewer().name("res")] __device__()
            {
                int  tid    = blockIdx.x * blockDim.x + threadIdx.x;
                bool active = tid < Size;
                int  idx;
                AABB bv;
                if(active)
                {
                    idx = tid;
                    bv  = _box(idx);
                }

                __shared__ int2 sharedRes[MAX_RES_PER_BLOCK];
                __shared__ int  sharedCounter;
                __shared__ int  sharedGlobalIdx;
                if(threadIdx.x == 0)
                    sharedCounter = 0;

                int  st = 0;
                Node node;
                const int MaxIter = numObjs * 2;

                while(true)
                {
                    __syncthreads();
                    if(active)
                    {
                        int inner_I = 0;
                        for(; inner_I < MaxIter; inner_I++)
                        {
                            if(st == -1)
                                break;

                            node.lc     = _nodes(st).lc;
                            node.escape = _nodes(st).escape;
                            node.bound  = _nodes(st).bound;

                            if(node.bound.intersects(bv))
                            {
                                if(node.lc == -1)
                                {
                                    const int j = _lvs_idx(st - intSize);

                                    const auto V = Vs(idx);
                                    const auto F = Fs(j);
                                    bool accept =
                                        !(F[0] == V || F[1] == V || F[2] == V);
                                    if(accept)
                                    {
                                        const auto body_i = v2b(V);
                                        const auto body_j = v2b(F[0]);
                                        accept = !(body_i == body_j
                                                   && !body_self_collision(body_i));
                                    }
                                    if(accept)
                                    {
                                        Vector3 P  = Ps(V);
                                        Vector3 F0 = Ps(F[0]);
                                        Vector3 F1 = Ps(F[1]);
                                        Vector3 F2 = Ps(F[2]);

                                        Float thickness = PT_thickness(
                                            thicknesses(V),
                                            thicknesses(F[0]),
                                            thicknesses(F[1]),
                                            thicknesses(F[2]));
                                        Float d_hat = PT_d_hat(d_hats(V),
                                                               d_hats(F[0]),
                                                               d_hats(F[1]),
                                                               d_hats(F[2]));
                                        Float expand = d_hat + thickness;
                                        if(alpha == static_cast<Float>(0))
                                        {
                                            const auto max_p = P.array();
                                            const auto min_p = P.array();
                                            const auto max_tri =
                                                F0.array().max(F1.array()).max(F2.array());
                                            const auto min_tri =
                                                F0.array().min(F1.array()).min(F2.array());
                                            accept = !((min_p - max_tri > expand).any()
                                                       || (min_tri - max_p > expand).any());
                                        }
                                        else
                                        {
                                            Vector3 dP  = alpha * dxs(V);
                                            Vector3 dF0 = alpha * dxs(F[0]);
                                            Vector3 dF1 = alpha * dxs(F[1]);
                                            Vector3 dF2 = alpha * dxs(F[2]);
                                            accept = distance::point_triangle_ccd_broadphase(
                                                P, F0, F1, F2, dP, dF0, dF1, dF2, expand);
                                        }
                                    }
                                    if(accept)
                                    {
                                        int sIdx = atomicAdd(&sharedCounter, 1);
                                        if(sIdx >= MAX_RES_PER_BLOCK)
                                            break;

                                        sharedRes[sIdx] = int2{idx, j};
                                    }

                                    st = node.escape;
                                }
                                else
                                {
                                    st = node.lc;
                                }
                            }
                            else
                            {
                                st = node.escape;
                            }
                        }

                        MUDA_ASSERT(inner_I < MaxIter,
                                    "Exceeded max iteration in stackless traversal, %d (Max=%d), numObj=(%d)",
                                    inner_I,
                                    MaxIter,
                                    numObjs);
                    }

                    __syncthreads();
                    int totalResInBlock = min(sharedCounter, MAX_RES_PER_BLOCK);

                    if(threadIdx.x == 0)
                        sharedGlobalIdx = atomicAdd(resCounter.data(), totalResInBlock);

                    __syncthreads();

                    const int globalIdx = sharedGlobalIdx;

                    if(threadIdx.x == 0)
                        sharedCounter = 0;

                    __syncthreads();

                    bool done = totalResInBlock < MAX_RES_PER_BLOCK;

                    SafeCopyTo(sharedRes,
                               totalResInBlock,
                               res.data(),
                               globalIdx,
                               static_cast<int>(res.total_size()));

                    if(done)
                        break;
                }
            });
}

inline void StacklessBVH::Impl::StacklessCDSharedOtherPointsTrianglesActiveNoMask(
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
    muda::BufferView<IndexT>    selected_counts)
{
    using namespace muda;
    using namespace culbvh;

    auto numQuery = static_cast<int>(point_aabbs.size());
    auto numObjs  = static_cast<int>(ext_aabb.size());
    auto BlockDim = K_THREADS;
    auto GridDim  = (numQuery + BlockDim - 1) / BlockDim;
    const int pp_cap = static_cast<int>(out_PPs.size());
    const int pe_cap = static_cast<int>(out_PEs.size());
    const int pt_cap = static_cast<int>(out_PTs.size());

    Launch(GridDim, BlockDim)
        .apply(
            [Size       = numQuery,
             _box       = point_aabbs.viewer().name("_box"),
             intSize    = numObjs - 1,
             numObjs    = numObjs,
             _lvs_idx   = ext_idx.viewer().name("_lvs_idx"),
             _nodes     = nodes.viewer().name("_nodes"),
             Vs         = surf_vertices.viewer().name("Vs"),
             Fs         = surf_triangles.viewer().name("Fs"),
             Ps         = positions.viewer().name("positions"),
             dxs        = displacements.viewer().name("displacements"),
             thicknesses = thicknesses.viewer().name("thicknesses"),
             d_hats     = d_hats.viewer().name("d_hats"),
             triangle_thicknesses = triangle_thicknesses.viewer().name("triangle_thicknesses"),
             triangle_d_hats = triangle_d_hats.viewer().name("triangle_d_hats"),
             alpha,
             v2b        = vertex_to_body.viewer().name("v2b"),
             body_self_collision = body_self_collision.viewer().name("body_self_collision"),
             out_PPs    = out_PPs.viewer().name("out_PPs"),
             out_PEs    = out_PEs.viewer().name("out_PEs"),
             out_PTs    = out_PTs.viewer().name("out_PTs"),
             counts     = selected_counts.viewer().name("selected_counts"),
             pp_cap,
             pe_cap,
             pt_cap] __device__()
            {
                int  tid    = blockIdx.x * blockDim.x + threadIdx.x;
                bool active = tid < Size;
                int  idx;
                AABB bv;
                if(active)
                {
                    idx = tid;
                    bv  = _box(idx);
                }

                int  st = 0;
                Node node;
                const int MaxIter = numObjs * 2;

                if(active)
                {
                    int inner_I = 0;
                    for(; inner_I < MaxIter; inner_I++)
                    {
                        if(st == -1)
                            break;

                        node.lc     = _nodes(st).lc;
                        node.escape = _nodes(st).escape;
                        node.bound  = _nodes(st).bound;

                        if(node.bound.intersects(bv))
                        {
                            if(node.lc == -1)
                            {
                                const int j = _lvs_idx(st - intSize);

                                const auto V = Vs(idx);
                                const auto F = Fs(j);
                                bool accept = !(F[0] == V || F[1] == V || F[2] == V);
                                if(accept)
                                {
                                    const auto body_i = v2b(V);
                                    const auto body_j = v2b(F[0]);
                                    accept = !(body_i == body_j
                                               && !body_self_collision(body_i));
                                }
                                if(accept)
                                {
                                    const Vector4i vIs = {V, F(0), F(1), F(2)};
                                    Vector3 Ps_arr[] = {Ps(vIs(0)),
                                                        Ps(vIs(1)),
                                                        Ps(vIs(2)),
                                                        Ps(vIs(3))};
                                    Float thickness = thicknesses(V) + triangle_thicknesses(j);
                                    Float d_hat =
                                        (d_hats(V) + triangle_d_hats(j)) * Float{0.5};
                                    Vector2 range = D_range(thickness, d_hat);
                                    if(alpha != static_cast<Float>(0))
                                    {
                                        Vector3 dP  = alpha * dxs(V);
                                        Vector3 dF0 = alpha * dxs(F[0]);
                                        Vector3 dF1 = alpha * dxs(F[1]);
                                        Vector3 dF2 = alpha * dxs(F[2]);
                                        if(!distance::point_triangle_ccd_broadphase(
                                               Ps_arr[0],
                                               Ps_arr[1],
                                               Ps_arr[2],
                                               Ps_arr[3],
                                               dP,
                                               dF0,
                                               dF1,
                                               dF2,
                                               d_hat + thickness))
                                        {
                                            st = node.escape;
                                            continue;
                                        }
                                    }
                                    Vector3 tri_min = Ps_arr[1].array()
                                                          .min(Ps_arr[2].array())
                                                          .min(Ps_arr[3].array())
                                                          .matrix();
                                    Vector3 tri_max = Ps_arr[1].array()
                                                          .max(Ps_arr[2].array())
                                                          .max(Ps_arr[3].array())
                                                          .matrix();
                                    if(point_box_distance2_lower_bound(Ps_arr[0],
                                                                       tri_min,
                                                                       tri_max)
                                       < range.y())
                                    {
                                        Vector4i flag = distance::point_triangle_distance_flag(
                                            Ps_arr[0], Ps_arr[1], Ps_arr[2], Ps_arr[3]);
                                        Float D;
                                        distance::point_triangle_distance2(
                                            flag, Ps_arr[0], Ps_arr[1], Ps_arr[2], Ps_arr[3], D);
                                        if(is_active_D(range, D))
                                        {
                                            Vector4i offsets;
                                            offsets.setConstant(-1);
                                            auto dim = distance::degenerate_point_triangle(
                                                flag, offsets);
                                            if(dim == 2)
                                            {
                                                IndexT dst = atomicAdd(&counts(0), IndexT{1});
                                                if(dst < pp_cap)
                                                    out_PPs(dst) = {vIs(offsets(0)), vIs(offsets(1))};
                                            }
                                            else if(dim == 3)
                                            {
                                                IndexT dst = atomicAdd(&counts(1), IndexT{1});
                                                if(dst < pe_cap)
                                                    out_PEs(dst) = {vIs(offsets(0)),
                                                                    vIs(offsets(1)),
                                                                    vIs(offsets(2))};
                                            }
                                            else if(dim == 4)
                                            {
                                                IndexT dst = atomicAdd(&counts(2), IndexT{1});
                                                if(dst < pt_cap)
                                                    out_PTs(dst) = vIs;
                                            }
                                        }
                                    }
                                }

                                st = node.escape;
                            }
                            else
                            {
                                st = node.lc;
                            }
                        }
                        else
                        {
                            st = node.escape;
                        }
                    }

                    MUDA_ASSERT(inner_I < MaxIter,
                                "Exceeded max iteration in stackless traversal, %d (Max=%d), numObj=(%d)",
                                inner_I,
                                MaxIter,
                                numObjs);
                }
            });
}

template <typename Pred>
void StacklessBVH::Impl::StacklessCDSharedOther(Pred pred,
                                                muda::CBufferView<AABB> query_aabbs,
                                                muda::CBufferView<int> query_sorted_id,
                                                muda::VarView<int> cpNum,
                                                muda::BufferView<Vector2i> buffer)
{
    using namespace muda;
    using namespace culbvh;

    auto numQuery = static_cast<int>(query_aabbs.size());
    auto numObjs  = static_cast<int>(ext_aabb.size());
    auto BlockDim = K_THREADS;
    auto GridDim  = (numQuery + BlockDim - 1) / BlockDim;


    Launch(GridDim, BlockDim)
        .apply(
            [Size       = numQuery,
             _box       = query_aabbs.viewer().name("_box"),
             sortedIdx  = query_sorted_id.viewer().name("sortedIdx"),
             intSize    = numObjs - 1,
             numObjs    = numObjs,
             _lvs_idx   = ext_idx.viewer().name("_lvs_idx"),
             _nodes     = nodes.viewer().name("_nodes"),
             resCounter = cpNum.viewer().name("resCounter"),
             res        = buffer.viewer().name("res"),
             pred] __device__()
            {
                int  tid    = blockIdx.x * blockDim.x + threadIdx.x;
                bool active = tid < Size;
                int  idx;
                AABB bv;
                if(active)
                {
                    idx = sortedIdx(tid);
                    bv  = _box(idx);
                }

                __shared__ int2 sharedRes[MAX_RES_PER_BLOCK];
                __shared__ int sharedCounter;  // How many results are cached in shared memory
                __shared__ int sharedGlobalIdx;  // Where to write in global memory
                if(threadIdx.x == 0)
                    sharedCounter = 0;

                int  st = 0;
                Node node;

                // Upper bound of iterations to avoid infinite loop
                const int MaxIter = numObjs * 2;

                while(true)
                {
                    __syncthreads();
                    if(active)
                    {
                        int inner_I = 0;
                        for(; inner_I < MaxIter; inner_I++)
                        {
                            if(st == -1)
                                break;

                            node.lc     = _nodes(st).lc;
                            node.escape = _nodes(st).escape;
                            node.bound  = _nodes(st).bound;

                            //node = _nodes[st];
                            if(node.bound.intersects(bv))
                            {
                                if(node.lc == -1)
                                {
                                    auto pair = int2{idx, _lvs_idx(st - intSize)};
                                    if(pred(pair.x, pair.y))
                                    {
                                        int sIdx = atomicAdd(&sharedCounter, 1);

                                        if(sIdx >= MAX_RES_PER_BLOCK)
                                        {
                                            break;
                                        }

                                        sharedRes[sIdx] = pair;
                                    }

                                    st = node.escape;
                                }
                                else
                                {
                                    st = node.lc;
                                }
                            }
                            else
                            {
                                st = node.escape;
                            }
                        }

                        MUDA_ASSERT(inner_I < MaxIter,
                                    "Exceeded max iteration in stackless traversal, %d (Max=%d), numObj=(%d)",
                                    inner_I,
                                    MaxIter,
                                    numObjs);
                    }


                    // Flush whatever we have
                    __syncthreads();
                    int totalResInBlock = min(sharedCounter, MAX_RES_PER_BLOCK);

                    if(threadIdx.x == 0)
                    {
                        // This Block Starts writing at sharedGlobalIdx
                        sharedGlobalIdx = atomicAdd(resCounter.data(), totalResInBlock);
                    }

                    __syncthreads();

                    // Make sure we dont write out of bounds
                    const int globalIdx = sharedGlobalIdx;

                    if(threadIdx.x == 0)
                        sharedCounter = 0;

                    __syncthreads();

                    // if there is at least one element empty
                    // it means we have found all collisions for this block
                    bool done = totalResInBlock < MAX_RES_PER_BLOCK;

                    SafeCopyTo(sharedRes,
                               totalResInBlock,
                               res.data(),
                               globalIdx,
                               static_cast<int>(res.total_size()));

                    if(done)
                        break;
                }
            });
}

inline void StacklessBVH::build(muda::CBufferView<AABB> aabbs)
{
    m_impl.build(aabbs);
}

inline void StacklessBVH::refit(muda::CBufferView<AABB> aabbs)
{
    m_impl.refit(aabbs);
}

template <typename Pred>
void StacklessBVH::detect(Pred callback, QueryBuffer& qbuffer)
{
    using namespace muda;
    // Query the LBVH

    if(m_impl.objs.size() == 0)
    {
        qbuffer.m_size = 0;
        return;
    }

    if(qbuffer.m_pairs.size() == 0)
        qbuffer.m_pairs.resize(50 * 1024);

    auto do_query = [&]
    {
        // clear counter
        cudaMemsetAsync(qbuffer.m_cpNum.data(), 0, sizeof(int));

        m_impl.StacklessCDSharedSelf(
            callback, qbuffer.m_cpNum.view(), qbuffer.m_pairs.view());
    };

    do_query();

    // get total number of pairs
    int h_cp_num = qbuffer.m_cpNum;
    // if failed, resize and retry
    if(h_cp_num > qbuffer.m_pairs.size())
    {
        qbuffer.m_pairs.resize(h_cp_num * m_impl.config.reserve_ratio);
        do_query();
    }

    UIPC_ASSERT(h_cp_num >= 0, "fatal error");
    qbuffer.m_size = h_cp_num;
    corex_sort_nomask_pairs(qbuffer, h_cp_num);
}

inline bool StacklessBVH::detect_edges_no_mask_launch(
    muda::CBufferView<Vector2i> edges,
    muda::CBufferView<IndexT>   vertex_to_body,
    muda::CBufferView<IndexT>   body_self_collision,
    QueryBuffer&                qbuffer)
{
    using namespace muda;

    if(m_impl.objs.size() == 0)
    {
        qbuffer.m_size = 0;
        return false;
    }

    if(qbuffer.m_pairs.size() == 0)
        qbuffer.m_pairs.resize(128 * 1024);

    {
        {
            corex_profile::ScopedPhase phase("bvh_query_detail",
                                             "edge_nomask_memset_counter");
            cudaMemsetAsync(qbuffer.m_cpNum.data(), 0, sizeof(int));
        }

        {
            corex_profile::ScopedPhase phase("bvh_query_detail", "edge_nomask_kernel");
            m_impl.StacklessCDSharedSelfEdgesNoMask(edges,
                                                    vertex_to_body,
                                                    body_self_collision,
                                                    qbuffer.m_cpNum.view(),
                                                    qbuffer.m_pairs.view());
        }
    }

    return true;
}

inline void StacklessBVH::detect_edges_no_mask(
    muda::CBufferView<Vector2i> edges,
    muda::CBufferView<IndexT>   vertex_to_body,
    muda::CBufferView<IndexT>   body_self_collision,
    QueryBuffer&                qbuffer)
{
    auto launched = detect_edges_no_mask_launch(edges,
                                                vertex_to_body,
                                                body_self_collision,
                                                qbuffer);
    if(!launched)
        return;

    int h_cp_num = 0;
    {
        corex_profile::ScopedPhase phase("bvh_query_detail", "edge_nomask_count_readback");
        h_cp_num = qbuffer.m_cpNum;
    }
    if(h_cp_num > qbuffer.m_pairs.size())
    {
        qbuffer.m_pairs.resize(h_cp_num * m_impl.config.reserve_ratio);
        detect_edges_no_mask_launch(edges,
                                    vertex_to_body,
                                    body_self_collision,
                                    qbuffer);
    }

    UIPC_ASSERT(h_cp_num >= 0, "fatal error");
    qbuffer.m_size = h_cp_num;
    corex_sort_nomask_pairs(qbuffer, h_cp_num);
}

inline void StacklessBVH::detect_edges_active_no_mask(
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
    muda::BufferView<IndexT>    selected_counts)
{
    if(m_impl.objs.size() == 0)
        return;

    corex_profile::ScopedPhase phase("bvh_query_detail", "edge_active_nomask_kernel");
    m_impl.StacklessCDSharedSelfEdgesActiveNoMask(edges,
                                                  positions,
                                                  displacements,
                                                  rest_positions,
                                                  edge_thicknesses,
                                                  edge_d_hats,
                                                  alpha,
                                                  vertex_to_body,
                                                  body_self_collision,
                                                  out_PPs,
                                                  out_PEs,
                                                  out_EEs,
                                                  selected_counts);
}

inline void StacklessBVH::QueryBuffer::build(muda::CBufferView<AABB> aabbs)
{
    auto size = aabbs.size();
    m_queryMtCode.resize(size);
    m_querySortedId.resize(size);


    Impl::calcMaxBVFromBox(aabbs, m_querySceneBox);
    Impl::calcMCsFromBox(aabbs, m_querySceneBox, m_queryMtCode);

    auto d_querySceneBox = m_querySceneBox.data();
    auto d_queryMtCode   = m_queryMtCode.data();
    auto d_querySortedId = m_querySortedId.data();
    auto numQuery        = size;

    auto null_stream = thrust::cuda::par.on(nullptr);
    thrust::sequence(null_stream, d_querySortedId, d_querySortedId + numQuery);
    thrust::sort_by_key(null_stream, d_queryMtCode, d_queryMtCode + numQuery, d_querySortedId);
}

inline bool StacklessBVH::query_points_triangles_no_mask_launch(
    muda::CBufferView<AABB>     point_aabbs,
    muda::CBufferView<IndexT>   surf_vertices,
    muda::CBufferView<Vector3i> surf_triangles,
    muda::CBufferView<Vector3>  positions,
    muda::CBufferView<Vector3>  displacements,
    muda::CBufferView<Float>    thicknesses,
    muda::CBufferView<Float>    d_hats,
    Float                       alpha,
    muda::CBufferView<IndexT>   vertex_to_body,
    muda::CBufferView<IndexT>   body_self_collision,
    QueryBuffer&                qbuffer)
{
    if(point_aabbs.size() == 0 || m_impl.objs.size() == 0)
    {
        qbuffer.m_size = 0;
        return false;
    }

    using namespace muda;
    if(qbuffer.m_pairs.size() == 0)
        qbuffer.m_pairs.resize(50 * 1024);

    {
        {
            corex_profile::ScopedPhase phase("bvh_query_detail",
                                             "pt_nomask_memset_counter");
            cudaMemsetAsync(qbuffer.m_cpNum.data(), 0, sizeof(int));
        }

        {
            corex_profile::ScopedPhase phase("bvh_query_detail", "pt_nomask_kernel");
            m_impl.StacklessCDSharedOtherPointsTrianglesNoMask(point_aabbs,
                                                               surf_vertices,
                                                               surf_triangles,
                                                               positions,
                                                               displacements,
                                                               thicknesses,
                                                               d_hats,
                                                               alpha,
                                                               vertex_to_body,
                                                               body_self_collision,
                                                               qbuffer.m_cpNum.view(),
                                                               qbuffer.m_pairs.view());
        }
    }

    return true;
}

inline void StacklessBVH::query_points_triangles_no_mask(
    muda::CBufferView<AABB>     point_aabbs,
    muda::CBufferView<IndexT>   surf_vertices,
    muda::CBufferView<Vector3i> surf_triangles,
    muda::CBufferView<Vector3>  positions,
    muda::CBufferView<Vector3>  displacements,
    muda::CBufferView<Float>    thicknesses,
    muda::CBufferView<Float>    d_hats,
    Float                       alpha,
    muda::CBufferView<IndexT>   vertex_to_body,
    muda::CBufferView<IndexT>   body_self_collision,
    QueryBuffer&                qbuffer)
{
    auto launched = query_points_triangles_no_mask_launch(point_aabbs,
                                                          surf_vertices,
                                                          surf_triangles,
                                                          positions,
                                                          displacements,
                                                          thicknesses,
                                                          d_hats,
                                                          alpha,
                                                          vertex_to_body,
                                                          body_self_collision,
                                                          qbuffer);
    if(!launched)
        return;

    int h_cp_num = 0;
    {
        corex_profile::ScopedPhase phase("bvh_query_detail", "pt_nomask_count_readback");
        h_cp_num = qbuffer.m_cpNum;
    }
    if(h_cp_num > qbuffer.m_pairs.size())
    {
        qbuffer.m_pairs.resize(h_cp_num * m_impl.config.reserve_ratio);
        query_points_triangles_no_mask_launch(point_aabbs,
                                              surf_vertices,
                                              surf_triangles,
                                              positions,
                                              displacements,
                                              thicknesses,
                                              d_hats,
                                              alpha,
                                              vertex_to_body,
                                              body_self_collision,
                                              qbuffer);
    }

    UIPC_ASSERT(h_cp_num >= 0, "fatal error");
    qbuffer.m_size = h_cp_num;
}

inline void StacklessBVH::query_points_triangles_active_no_mask(
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
    muda::BufferView<IndexT>    selected_counts)
{
    if(point_aabbs.size() == 0 || m_impl.objs.size() == 0)
        return;

    corex_profile::ScopedPhase phase("bvh_query_detail", "pt_active_nomask_kernel");
    m_impl.StacklessCDSharedOtherPointsTrianglesActiveNoMask(point_aabbs,
                                                             surf_vertices,
                                                             surf_triangles,
                                                             positions,
                                                             displacements,
                                                             thicknesses,
                                                             d_hats,
                                                             triangle_thicknesses,
                                                             triangle_d_hats,
                                                             alpha,
                                                             vertex_to_body,
                                                             body_self_collision,
                                                             out_PPs,
                                                             out_PEs,
                                                             out_PTs,
                                                             selected_counts);
}

template <typename Pred>
void StacklessBVH::query(muda::CBufferView<AABB> aabbs, Pred callback, QueryBuffer& qbuffer)
{
    if(aabbs.size() == 0 || m_impl.objs.size() == 0)
    {
        qbuffer.m_size = 0;
        return;
    }

    using namespace muda;
    if(qbuffer.m_pairs.size() == 0)
        qbuffer.m_pairs.resize(50 * 1024);
    qbuffer.build(aabbs);

    auto do_query = [&]
    {
        // clear counter
        cudaMemsetAsync(qbuffer.m_cpNum.data(), 0, sizeof(int));

        m_impl.StacklessCDSharedOther(callback,
                                      aabbs,
                                      qbuffer.m_querySortedId.view(),
                                      qbuffer.m_cpNum.view(),
                                      qbuffer.m_pairs.view());
    };

    do_query();

    // get total number of pairs
    int h_cp_num = qbuffer.m_cpNum;
    // if failed, resize and retry
    if(h_cp_num > qbuffer.m_pairs.size())
    {
        qbuffer.m_pairs.resize(h_cp_num * m_impl.config.reserve_ratio);
        do_query();
    }

    UIPC_ASSERT(h_cp_num >= 0, "fatal error");
    qbuffer.m_size = h_cp_num;
}
}  // namespace uipc::backend::cuda
#else
#include <cuda_device/builtin.h>
#include <muda/launch.h>

namespace uipc::culbvh
{
using aabb          = uipc::backend::cuda::AABB;
using stacklessnode = uipc::backend::cuda::StacklessBVH::Node;
using Vector2i      = uipc::Vector2i;

using uint   = uint32_t;
using ullint = unsigned long long int;

constexpr int K_THREADS = 256;
constexpr int K_WARPS   = K_THREADS >> 5;

constexpr int K_REDUCTION_LAYER  = 5;
constexpr int K_REDUCTION_NUM    = 1 << K_REDUCTION_LAYER;
constexpr int K_REDUCTION_MODULO = K_REDUCTION_NUM - 1;

constexpr int    aabbBits  = 15;
constexpr int    aabbRes   = (1 << aabbBits) - 2;
constexpr int    indexBits = 64 - 3 * aabbBits;
constexpr int    offset3   = aabbBits * 3;
constexpr int    offset2   = aabbBits * 2;
constexpr int    offset1   = aabbBits * 1;
constexpr ullint indexMask = 0xFFFFFFFFFFFFFFFFu << offset3;
constexpr uint   aabbMask  = 0xFFFFFFFFu >> (32 - aabbBits);
constexpr uint   MaxIndex  = 0xFFFFFFFFFFFFFFFFu >> offset3;

constexpr uint MAX_CD_NUM_PER_VERT = 64;
constexpr int  MAX_RES_PER_BLOCK   = 1024;

struct PlainAABB
{
    float3 _min, _max;
};

MUDA_GENERIC MUDA_INLINE PlainAABB toPlainAABB(const aabb& box)
{
    PlainAABB res;
    res._min = make_float3(box.min().x(), box.min().y(), box.min().z());
    res._max = make_float3(box.max().x(), box.max().y(), box.max().z());
    return res;
}

MUDA_GENERIC MUDA_INLINE aabb fromPlainAABB(const PlainAABB& box)
{
    aabb aabb;
    aabb.min() = Vector<float, 3>(box._min.x, box._min.y, box._min.z);
    aabb.max() = Vector<float, 3>(box._max.x, box._max.y, box._max.z);
    return aabb;
}

struct intAABB
{
    int3 _min, _max;

    MUDA_GENERIC MUDA_INLINE void convertFrom(const aabb& other, float3& origin, float3& delta)
    {
        _min.x = static_cast<int>((other.min().x() - origin.x) / delta.x);
        _min.y = static_cast<int>((other.min().y() - origin.y) / delta.y);
        _min.z = static_cast<int>((other.min().z() - origin.z) / delta.z);
        _max.x = static_cast<int>(ceilf((other.max().x() - origin.x) / delta.x));
        _max.y = static_cast<int>(ceilf((other.max().y() - origin.y) / delta.y));
        _max.z = static_cast<int>(ceilf((other.max().z() - origin.z) / delta.z));
    }
};

template <typename T>
MUDA_GENERIC MUDA_INLINE T __mm_min(T a, T b)
{
    return a > b ? b : a;
}

template <typename T>
MUDA_GENERIC MUDA_INLINE T __mm_max(T a, T b)
{
    return a > b ? a : b;
}

MUDA_DEVICE MUDA_INLINE float atomicMinf(float* addr, float value)
{
    float old;
    old = (value >= 0) ?
              __int_as_float(atomicMin((int*)addr, __float_as_int(value))) :
              __uint_as_float(atomicMax((unsigned int*)addr, __float_as_uint(value)));
    return old;
}

MUDA_DEVICE MUDA_INLINE float atomicMaxf(float* addr, float value)
{
    float old;
    old = (value >= 0) ?
              __int_as_float(atomicMax((int*)addr, __float_as_int(value))) :
              __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));
    return old;
}

MUDA_GENERIC MUDA_INLINE uint expandBits(uint v)
{  ///< Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

MUDA_GENERIC MUDA_INLINE uint morton3D(float x, float y, float z)
{  ///< Calculates a 30-bit Morton code for the given 3D point located within the unit cube [0,1].
    x       = ::fmin(::fmax(x * 1024.0f, 0.0f), 1023.0f);
    y       = ::fmin(::fmax(y * 1024.0f, 0.0f), 1023.0f);
    z       = ::fmin(::fmax(z * 1024.0f, 0.0f), 1023.0f);
    uint xx = expandBits((uint)x);
    uint yy = expandBits((uint)y);
    uint zz = expandBits((uint)z);
    return (xx * 4 + yy * 2 + zz);
}

// Custom comparison for int3 based on lexicographical ordering
MUDA_GENERIC MUDA_INLINE bool lessThan(const int3& a, const int3& b)
{
    if(a.x != b.x)
        return a.x < b.x;
    if(a.y != b.y)
        return a.y < b.y;
    return a.z < b.z;
}

MUDA_GENERIC MUDA_INLINE Vector2i to_eigen(int2 v)
{
    return Vector2i{v.x, v.y};
}

MUDA_GENERIC MUDA_INLINE int2 make_ordered_pair(int a, int b)
{
    if(a < b)
        return int2{a, b};
    else
        return int2{b, a};
}

MUDA_GENERIC MUDA_INLINE float3 operator-(const float3& v0, const float3& v1)
{
    return make_float3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z);
}

MUDA_GENERIC MUDA_INLINE void SafeCopyTo(int2* sharedRes,
                                         int   totalResInBlock,

                                         Vector2i* globalRes,
                                         int       globalIdx,
                                         int       maxRes)
{
    if(globalIdx >= maxRes      // Out of memory for results.
       || totalResInBlock == 0  // No results to write
    )
        return;

    auto CopyCount = std::min(totalResInBlock, maxRes - globalIdx);

    // Copy full blocks
    int fullBlocks = (CopyCount - 1) / (int)blockDim.x;
    for(int i = 0; i < fullBlocks; i++)
    {
        int offset                    = i * blockDim.x + threadIdx.x;
        globalRes[globalIdx + offset] = to_eigen(sharedRes[offset]);
    }

    // Copy the rest
    int offset = fullBlocks * blockDim.x + threadIdx.x;
    if(offset < CopyCount)
        globalRes[globalIdx + offset] = to_eigen(sharedRes[offset]);
}

}  // namespace uipc::culbvh

namespace uipc::backend::cuda
{
MUDA_INLINE void StacklessBVH::Impl::calcMaxBVFromBox(muda::CBufferView<AABB> aabbs,
                                                      muda::VarView<AABB> scene_box)
{
    using namespace culbvh;

    auto numQuery = aabbs.size();
    auto BlockDim = K_THREADS;
    auto GridDim  = (numQuery + BlockDim - 1) / BlockDim;

    using namespace muda;

    Launch(GridDim, BlockDim)
        .file_line(__FILE__, __LINE__)
        .apply(
            [size = aabbs.size(),
             box  = aabbs.viewer().name("box"),
             _bv  = scene_box.viewer().name("_bv")] __device__()
            {
                int idx     = blockIdx.x * blockDim.x + threadIdx.x;
                int warpTid = threadIdx.x % 32;
                int warpId  = (threadIdx.x >> 5);
                int warpNum;
                if(idx >= size)
                    return;
                if(idx == 0)
                {
                    *_bv = AABB();
                }

                __shared__ PlainAABB aabbData[K_WARPS];

                PlainAABB temp = toPlainAABB(box(idx));
                __syncthreads();

                // Extract values for warp shuffle
                float tempMinX = temp._min.x;
                float tempMinY = temp._min.y;
                float tempMinZ = temp._min.z;
                float tempMaxX = temp._max.x;
                float tempMaxY = temp._max.y;
                float tempMaxZ = temp._max.z;

                for(int i = 1; i < 32; i = (i << 1))
                {
                    float otherMinX = __shfl_down_sync(0xffffffff, tempMinX, i);
                    float otherMinY = __shfl_down_sync(0xffffffff, tempMinY, i);
                    float otherMinZ = __shfl_down_sync(0xffffffff, tempMinZ, i);
                    float otherMaxX = __shfl_down_sync(0xffffffff, tempMaxX, i);
                    float otherMaxY = __shfl_down_sync(0xffffffff, tempMaxY, i);
                    float otherMaxZ = __shfl_down_sync(0xffffffff, tempMaxZ, i);
                    tempMinX        = __mm_min(tempMinX, otherMinX);
                    tempMinY        = __mm_min(tempMinY, otherMinY);
                    tempMinZ        = __mm_min(tempMinZ, otherMinZ);
                    tempMaxX        = __mm_max(tempMaxX, otherMaxX);
                    tempMaxY        = __mm_max(tempMaxY, otherMaxY);
                    tempMaxZ        = __mm_max(tempMaxZ, otherMaxZ);
                }

                if(blockIdx.x == gridDim.x - 1)
                {
                    warpNum = ((size - blockIdx.x * blockDim.x + 31) >> 5);
                }
                else
                {
                    warpNum = ((blockDim.x) >> 5);
                }

                if(warpTid == 0)
                {
                    // Reconstruct AABB from reduced values
                    aabbData[warpId]._min = make_float3(tempMinX, tempMinY, tempMinZ);
                    aabbData[warpId]._max = make_float3(tempMaxX, tempMaxY, tempMaxZ);
                }
                __syncthreads();
                if(threadIdx.x >= warpNum)
                    return;

                if(warpNum > 1)
                {
                    temp     = aabbData[threadIdx.x];
                    tempMinX = temp._min.x;
                    tempMinY = temp._min.y;
                    tempMinZ = temp._min.z;
                    tempMaxX = temp._max.x;
                    tempMaxY = temp._max.y;
                    tempMaxZ = temp._max.z;

                    for(int i = 1; i < warpNum; i = (i << 1))
                    {
                        float otherMinX = __shfl_down_sync(0xffffffff, tempMinX, i);
                        float otherMinY = __shfl_down_sync(0xffffffff, tempMinY, i);
                        float otherMinZ = __shfl_down_sync(0xffffffff, tempMinZ, i);
                        float otherMaxX = __shfl_down_sync(0xffffffff, tempMaxX, i);
                        float otherMaxY = __shfl_down_sync(0xffffffff, tempMaxY, i);
                        float otherMaxZ = __shfl_down_sync(0xffffffff, tempMaxZ, i);
                        tempMinX = __mm_min(tempMinX, otherMinX);
                        tempMinY = __mm_min(tempMinY, otherMinY);
                        tempMinZ = __mm_min(tempMinZ, otherMinZ);
                        tempMaxX = __mm_max(tempMaxX, otherMaxX);
                        tempMaxY = __mm_max(tempMaxY, otherMaxY);
                        tempMaxZ = __mm_max(tempMaxZ, otherMaxZ);
                    }
                }

                if(threadIdx.x == 0)
                {
                    atomicMinf(&_bv->min().x(), tempMinX);
                    atomicMinf(&_bv->min().y(), tempMinY);
                    atomicMinf(&_bv->min().z(), tempMinZ);
                    atomicMaxf(&_bv->max().x(), tempMaxX);
                    atomicMaxf(&_bv->max().y(), tempMaxY);
                    atomicMaxf(&_bv->max().z(), tempMaxZ);
                }
            });
}

MUDA_INLINE void StacklessBVH::Impl::calcMCsFromBox(muda::CBufferView<AABB> aabbs,
                                                    muda::CVarView<AABB> scene_box,
                                                    muda::BufferView<uint32_t> codes)
{
    using namespace culbvh;
    using namespace muda;

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(aabbs.size(),
               [box   = aabbs.viewer().name("box"),
                scene = scene_box.viewer().name("scene"),
                codes = codes.viewer().name("codes")] __device__(int idx)
               {
                   AABB bv = box(idx);

                   // Get center using Eigen API
                   auto   center = bv.center();
                   float3 c = make_float3(center.x(), center.y(), center.z());

                   // Get scene min
                   auto   sceneMin = scene->min();
                   float3 sceneMinVec =
                       make_float3(sceneMin.x(), sceneMin.y(), sceneMin.z());
                   const float3 offset = c - sceneMinVec;

                   // Get dimensions
                   auto sceneSize = scene->sizes();
                   codes(idx)     = morton3D(offset.x / sceneSize.x(),
                                         offset.y / sceneSize.y(),
                                         offset.z / sceneSize.z());
               });
}

/// incoherent access, thus poor performance
MUDA_INLINE void StacklessBVH::Impl::calcInverseMapping()
{
    using namespace muda;

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(sorted_id.size(),
               [map    = sorted_id.viewer().name("map"),
                invMap = primMap.viewer().name("invMap")] __device__(int idx)
               {
                   //
                   invMap(map(idx)) = idx;
               });
}

MUDA_INLINE void StacklessBVH::Impl::buildPrimitivesFromBox(muda::CBufferView<AABB> aabbs)
{  ///< update idx-th _bxs to idx-th leaf
    using namespace muda;
    ParallelFor().apply(aabbs.size(),
                        [_primIdx = ext_idx.viewer().name("primIdx"),
                         _primBox = ext_aabb.viewer().name("primBox"),
                         _primMap = primMap.viewer().name("primMap"),
                         box = aabbs.viewer().name("box")] __device__(int idx)
                        {
                            int  newIdx      = _primMap(idx);
                            AABB bv          = box(idx);
                            _primIdx(newIdx) = idx;
                            _primBox(newIdx) = bv;
                        });
}


MUDA_INLINE void StacklessBVH::Impl::calcExtNodeSplitMetrics()
{
    using namespace muda;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(mtcode.size(),
               [extsize  = mtcode.size(),
                _codes   = mtcode.viewer().name("_codes"),
                _metrics = metric.viewer().name("_metrics")] __device__(int idx)
               {
                   _metrics(idx) = idx != extsize - 1 ?
                                       32 - __clz(_codes(idx) ^ _codes(idx + 1)) :
                                       33;
               });
}

MUDA_INLINE void StacklessBVH::Impl::buildIntNodes(int size)
{
    using namespace muda;

    auto GridDim  = (size + 255) / 256;
    auto BlockDim = 256;

    Launch(GridDim, BlockDim)
        .file_line(__FILE__, __LINE__)
        .apply(
            [size = size,
             // leaf nodes
             _depths     = count.viewer().name("_depths"),
             _lvs_lca    = ext_lca.viewer().name("_lvs_lca"),
             _lvs_metric = metric.viewer().name("_lvs_metric"),
             _lvs_par    = ext_par.viewer().name("_lvs_par"),
             _lvs_mark   = ext_mark.viewer().name("_lvs_mark"),
             _lvs_box    = ext_aabb.viewer().name("_lvs_box"),
             // internal nodes
             _tks_rc      = int_rc.viewer().name("_tks_rc"),
             _tks_lc      = int_lc.viewer().name("_tks_lc"),
             _tks_range_y = int_range_y.viewer().name("_tks_range_y"),
             _tks_range_x = int_range_x.viewer().name("_tks_range_x"),
             _tks_mark    = int_mark.viewer().name("_tks_mark"),
             _tks_box     = int_aabb.viewer().name("_tks_box"),
             _flag        = flags.viewer().name("_flag"),
             _tks_par     = int_par.viewer().name("_tks_par")] __device__()
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if(idx >= size)
                    return;

                _lvs_lca(idx) = -1, _depths(idx) = 0;
                int  l = idx - 1, r = idx;  ///< (l, r]
                bool mark;
                if(l >= 0)
                    mark = _lvs_metric(l) < _lvs_metric(r);  //determine direction
                else
                    mark = false;
                int cur = mark ? l : r;

                _lvs_par(idx) = cur;


                if(_flag.total_size() == 0)
                    // when we only have 1 external node
                    // there is no internal node to build
                    return;

                if(mark)
                {
                    _tks_rc(cur)      = idx;
                    _tks_range_y(cur) = idx;
                    atomicOr(&_tks_mark(cur), 0x00000002);
                    _lvs_mark(idx) = 0x00000007;
                }
                else
                {
                    _tks_lc(cur)      = idx;
                    _tks_range_x(cur) = idx;
                    atomicOr(&_tks_mark(cur), 0x00000001);
                    _lvs_mark(idx) = 0x00000003;
                }

                __threadfence();
                while(atomicAdd(&_flag(cur), 1) == 1)
                {
                    //_tks.update(cur, _lvs);	/// Update
                    //_tks.refit(cur, _lvs);	/// Refit
                    int      chl       = _tks_lc(cur);
                    int      chr       = _tks_rc(cur);
                    uint32_t temp_mark = _tks_mark(cur);
                    if(temp_mark & 1)
                    {
                        _tks_box(cur) = _lvs_box(chl);
                    }
                    else
                    {
                        _tks_box(cur) = _tks_box(chl);
                    }
                    if(temp_mark & 2)
                    {
                        _tks_box(cur).extend(_lvs_box(chr));
                    }
                    else
                    {
                        _tks_box(cur).extend(_tks_box(chr));
                    }

                    _tks_mark(cur) &= 0x00000007;

                    l               = _tks_range_x(cur) - 1;
                    r               = _tks_range_y(cur);
                    _lvs_lca(l + 1) = cur;
                    _depths(l + 1)++;
                    if(l >= 0)
                    {
                        mark = _lvs_metric(l) < _lvs_metric(r);  ///< true when right child, false otherwise
                    }
                    else
                    {
                        mark = false;
                    }

                    if(l + 1 == 0 && r == size - 1)
                    {
                        _tks_par(cur) = -1;
                        _tks_mark(cur) &= 0xFFFFFFFB;
                        break;
                    }

                    int par       = mark ? l : r;
                    _tks_par(cur) = par;
                    if(mark)
                    {
                        _tks_rc(par)      = cur;
                        _tks_range_y(par) = r;
                        atomicAnd(&_tks_mark(par), 0xFFFFFFFD);
                        _tks_mark(cur) |= 0x00000004;
                    }
                    else
                    {
                        _tks_lc(par)      = cur;
                        _tks_range_x(par) = l + 1;
                        atomicAnd(&_tks_mark(par), 0xFFFFFFFE);
                        _tks_mark(cur) &= 0xFFFFFFFB;
                    }
                    __threadfence();
                    cur = par;
                }
            });
}

MUDA_INLINE void StacklessBVH::Impl::calcIntNodeOrders(int size)
{
    using namespace muda;

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(size,
               [_tks_lc  = int_lc.viewer().name("_tks_lc"),
                _lcas    = ext_lca.viewer().name("_lcas"),
                _depths  = count.viewer().name("_depths"),
                _offsets = offsetTable.viewer().name("_offsets"),
                _tkMap   = tkMap.viewer().name("_tkMap")] __device__(int idx)
               {
                   int node  = _lcas(idx);
                   int depth = _depths(idx);
                   int id    = _offsets(idx);

                   if(node != -1)
                   {
                       for(; depth--; node = _tks_lc(node))
                       {
                           _tkMap(node) = id++;
                       }
                   }
               });
}

MUDA_INLINE void StacklessBVH::Impl::updateBvhExtNodeLinks(int size)
{
    using namespace muda;

    if(flags.size() == 0)  // no internal nodes, thus no need to update
        return;

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(size,
               [_mapTable = tkMap.viewer().name("_mapTable"),
                _lcas     = ext_lca.viewer().name("_lcas"),
                _pars     = ext_par.viewer().name("_pars")] __device__(int idx)
               {
                   int ori;
                   _pars(idx) = _mapTable(_pars(idx));
                   if((ori = _lcas(idx)) != -1)
                       _lcas(idx) = _mapTable(ori) << 1;
                   else
                       _lcas(idx) = idx << 1 | 1;
               });
}

MUDA_INLINE void StacklessBVH::Impl::reorderNode(int intSize)
{
    using namespace culbvh;
    using namespace muda;

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(intSize + 1,
               [intSize,
                // leaf nodes
                _lvs_lca = ext_lca.viewer().name("_lvs_lca"),
                _lvs_box = ext_aabb.viewer().name("_lvs_box"),
                // internal nodes
                _tkMap           = tkMap.viewer().name("_tkMap"),
                _unorderedTks_lc = int_lc.viewer().name("_unorderedTks_lc"),
                _unorderedTks_mark = int_mark.viewer().name("_unorderedTks_mark"),
                _unorderedTks_rangey = int_range_y.viewer().name("_unorderedTks_rangey"),
                _unorderedTks_box = int_aabb.viewer().name("_unorderedTks_box"),
                // total nodes
                _nodes = nodes.viewer().name("_nodes")] __device__(int idx)
               {
                   stacklessnode Node;
                   Node.lc    = -1;
                   int escape = _lvs_lca(idx + 1);

                   if(escape == -1)
                   {
                       Node.escape = -1;
                   }
                   else
                   {
                       int bLeaf = escape & 1;
                       escape >>= 1;
                       Node.escape = escape + (bLeaf ? intSize : 0);
                   }
                   Node.bound = _lvs_box(idx);


                   _nodes(idx + intSize) = Node;

                   if(idx >= intSize)
                       return;

                   stacklessnode internalNode;
                   int           newId = _tkMap(idx);
                   uint32_t      mark  = _unorderedTks_mark(idx);

                   internalNode.lc = mark & 1 ? _unorderedTks_lc(idx) + intSize :
                                                _tkMap(_unorderedTks_lc(idx));
                   internalNode.bound = _unorderedTks_box(idx);

                   int internalEscape = _lvs_lca(_unorderedTks_rangey(idx) + 1);

                   if(internalEscape == -1)
                   {
                       internalNode.escape = -1;
                   }
                   else
                   {
                       int bLeaf = internalEscape & 1;
                       internalEscape >>= 1;
                       internalNode.escape = internalEscape + (bLeaf ? intSize : 0);
                   }
                   _nodes(newId) = internalNode;
               });
}

inline void StacklessBVH::Impl::build(muda::CBufferView<AABB> aabbs)
{
    objs         = aabbs;
    auto numObjs = aabbs.size();

    if(aabbs.size() == 0)
        return;

    const unsigned int numInternalNodes = numObjs - 1;  // Total number of internal nodes
    const unsigned int numNodes = numObjs * 2 - 1;  // Total number of nodes


    mtcode.resize(numObjs);
    sorted_id.resize(numObjs);
    primMap.resize(numObjs);
    ext_aabb.resize(numObjs);
    ext_idx.resize(numObjs);
    ext_lca.resize(numObjs + 1);
    ext_par.resize(numObjs);
    ext_mark.resize(numObjs);

    metric.resize(numObjs);
    tkMap.resize(numObjs);
    offsetTable.resize(numObjs);
    count.resize(numObjs);

    flags.resize(numInternalNodes);
    int_lc.resize(numInternalNodes);
    int_rc.resize(numInternalNodes);
    int_par.resize(numInternalNodes);
    int_range_x.resize(numInternalNodes);
    int_range_y.resize(numInternalNodes);
    int_mark.resize(numInternalNodes);
    int_aabb.resize(numInternalNodes);

    nodes.resize(numNodes);


    // Initialize flags to 0
    thrust::fill(flags.begin(), flags.end(), 0);
    thrust::fill(thrust::device, ext_mark.begin(), ext_mark.end(), 7);
    thrust::fill(thrust::device, ext_lca.begin(), ext_lca.end(), 0);
    thrust::fill(thrust::device, ext_par.begin(), ext_par.end(), 0);

    calcMaxBVFromBox(aabbs, scene_box.view());

    calcMCsFromBox(aabbs, scene_box.view(), mtcode.view());

    auto null_stream = thrust::cuda::par_nosync.on(nullptr);

    thrust::sequence(null_stream, sorted_id.begin(), sorted_id.end());
    thrust::sort_by_key(null_stream, mtcode.begin(), mtcode.end(), sorted_id.begin());

    calcInverseMapping();

    buildPrimitivesFromBox(aabbs);

    calcExtNodeSplitMetrics();

    buildIntNodes(numObjs);

    thrust::exclusive_scan(null_stream, count.begin(), count.end(), offsetTable.begin());

    calcIntNodeOrders(numObjs);

    // fill the last ext_lca to -1
    thrust::fill(null_stream, ext_lca.begin() + numObjs, ext_lca.begin() + numObjs + 1, -1);

    updateBvhExtNodeLinks(numObjs);

    reorderNode(numInternalNodes);
}

template <typename Pred>
void StacklessBVH::Impl::StacklessCDSharedSelf(Pred               pred,
                                               muda::VarView<int> cpNum,
                                               muda::BufferView<Vector2i> buffer)
{
    using namespace muda;
    using namespace culbvh;

    auto numQuery = static_cast<int>(ext_aabb.size());
    auto numObjs  = numQuery;
    auto BlockDim = K_THREADS;
    auto GridDim  = (numQuery + BlockDim - 1) / BlockDim;

    Launch(GridDim, BlockDim)
        .apply(
            [Size       = numQuery,
             _box       = objs.viewer().name("_box"),
             intSize    = numObjs - 1,
             numObjs    = numObjs,
             _lvs_idx   = ext_idx.viewer().name("_lvs_idx"),
             _nodes     = nodes.viewer().name("_nodes"),
             resCounter = cpNum.viewer().name("resCounter"),
             res        = buffer.viewer().name("res"),
             pred] __device__()
            {
                int  tid    = blockIdx.x * blockDim.x + threadIdx.x;
                bool active = tid < Size;
                int  idx;
                AABB bv;
                if(active)
                {
                    idx = _lvs_idx(tid);
                    bv  = _box(idx);
                }

                __shared__ int2 sharedRes[MAX_RES_PER_BLOCK];
                __shared__ int sharedCounter;  // How many results are cached in shared memory
                __shared__ int sharedGlobalIdx;  // Where to write in global memory
                if(threadIdx.x == 0)
                    sharedCounter = 0;

                int  st = 0;
                Node node;
                // Upper bound of iterations to avoid infinite loop
                const int MaxIter = numObjs * 2;

                while(true)
                {
                    __syncthreads();
                    if(active)
                    {
                        int inner_I = 0;
                        for(; inner_I < MaxIter; inner_I++)
                        {
                            if(st == -1)
                                break;
                            // Load node data - Eigen::AlignedBox stores min and max as Vector3f members
                            node.lc     = _nodes(st).lc;
                            node.escape = _nodes(st).escape;
                            node.bound  = _nodes(st).bound;
                            //node = _nodes[st];
                            if(node.bound.intersects(bv))
                            {
                                if(node.lc == -1)
                                {
                                    if(tid < st - intSize)
                                    {
                                        auto pair =
                                            make_ordered_pair(idx, _lvs_idx(st - intSize));
                                        if(pred(pair.x, pair.y))
                                        {
                                            int sIdx = atomicAdd(&sharedCounter, 1);
                                            if(sIdx >= MAX_RES_PER_BLOCK)
                                            {
                                                break;
                                            }

                                            sharedRes[sIdx] = pair;
                                        }
                                    }
                                    st = node.escape;
                                }
                                else
                                {
                                    st = node.lc;
                                }
                            }
                            else
                            {
                                st = node.escape;
                            }
                        }

                        MUDA_ASSERT(inner_I < MaxIter,
                                    "Exceeded max iteration in stackless traversal, %d (Max=%d), numObj=(%d)",
                                    inner_I,
                                    MaxIter,
                                    numObjs);
                    }
                    // Flush whatever we have
                    __syncthreads();

                    int totalResInBlock = min(sharedCounter, MAX_RES_PER_BLOCK);

                    if(threadIdx.x == 0)
                    {
                        // This Block Starts writing at sharedGlobalIdx
                        sharedGlobalIdx = atomicAdd(resCounter.data(), totalResInBlock);
                    }

                    __syncthreads();

                    // Make sure we dont write out of bounds
                    const int globalIdx = sharedGlobalIdx;

                    if(threadIdx.x == 0)
                        sharedCounter = 0;

                    // if there is at least one element empty
                    // it means we have found all collisions for this block
                    bool done = totalResInBlock < MAX_RES_PER_BLOCK;

                    SafeCopyTo(sharedRes,
                               totalResInBlock,
                               res.data(),
                               globalIdx,
                               static_cast<int>(res.total_size()));

                    if(done)
                        break;
                }
            });
}

template <typename Pred>
void StacklessBVH::Impl::StacklessCDSharedOther(Pred pred,
                                                muda::CBufferView<AABB> query_aabbs,
                                                muda::CBufferView<int> query_sorted_id,
                                                muda::VarView<int> cpNum,
                                                muda::BufferView<Vector2i> buffer)
{
    using namespace muda;
    using namespace culbvh;

    auto numQuery = static_cast<int>(query_aabbs.size());
    auto numObjs  = static_cast<int>(ext_aabb.size());
    auto BlockDim = K_THREADS;
    auto GridDim  = (numQuery + BlockDim - 1) / BlockDim;


    Launch(GridDim, BlockDim)
        .apply(
            [Size       = numQuery,
             _box       = query_aabbs.viewer().name("_box"),
             sortedIdx  = query_sorted_id.viewer().name("sortedIdx"),
             intSize    = numObjs - 1,
             numObjs    = numObjs,
             _lvs_idx   = ext_idx.viewer().name("_lvs_idx"),
             _nodes     = nodes.viewer().name("_nodes"),
             resCounter = cpNum.viewer().name("resCounter"),
             res        = buffer.viewer().name("res"),
             pred] __device__()
            {
                int  tid    = blockIdx.x * blockDim.x + threadIdx.x;
                bool active = tid < Size;
                int  idx;
                AABB bv;
                if(active)
                {
                    idx = sortedIdx(tid);
                    bv  = _box(idx);
                }

                __shared__ int2 sharedRes[MAX_RES_PER_BLOCK];
                __shared__ int sharedCounter;  // How many results are cached in shared memory
                __shared__ int sharedGlobalIdx;  // Where to write in global memory
                if(threadIdx.x == 0)
                    sharedCounter = 0;

                int  st = 0;
                Node node;

                // Upper bound of iterations to avoid infinite loop
                const int MaxIter = numObjs * 2;

                while(true)
                {
                    __syncthreads();
                    if(active)
                    {
                        int inner_I = 0;
                        for(; inner_I < MaxIter; inner_I++)
                        {
                            if(st == -1)
                                break;

                            node.lc     = _nodes(st).lc;
                            node.escape = _nodes(st).escape;
                            node.bound  = _nodes(st).bound;

                            //node = _nodes[st];
                            if(node.bound.intersects(bv))
                            {
                                if(node.lc == -1)
                                {
                                    auto pair = int2{idx, _lvs_idx(st - intSize)};
                                    if(pred(pair.x, pair.y))
                                    {
                                        int sIdx = atomicAdd(&sharedCounter, 1);

                                        if(sIdx >= MAX_RES_PER_BLOCK)
                                        {
                                            break;
                                        }

                                        sharedRes[sIdx] = pair;
                                    }

                                    st = node.escape;
                                }
                                else
                                {
                                    st = node.lc;
                                }
                            }
                            else
                            {
                                st = node.escape;
                            }
                        }

                        MUDA_ASSERT(inner_I < MaxIter,
                                    "Exceeded max iteration in stackless traversal, %d (Max=%d), numObj=(%d)",
                                    inner_I,
                                    MaxIter,
                                    numObjs);
                    }


                    // Flush whatever we have
                    __syncthreads();
                    int totalResInBlock = min(sharedCounter, MAX_RES_PER_BLOCK);

                    if(threadIdx.x == 0)
                    {
                        // This Block Starts writing at sharedGlobalIdx
                        sharedGlobalIdx = atomicAdd(resCounter.data(), totalResInBlock);
                    }

                    __syncthreads();

                    // Make sure we dont write out of bounds
                    const int globalIdx = sharedGlobalIdx;

                    if(threadIdx.x == 0)
                        sharedCounter = 0;

                    __syncthreads();

                    // if there is at least one element empty
                    // it means we have found all collisions for this block
                    bool done = totalResInBlock < MAX_RES_PER_BLOCK;

                    SafeCopyTo(sharedRes,
                               totalResInBlock,
                               res.data(),
                               globalIdx,
                               static_cast<int>(res.total_size()));

                    if(done)
                        break;
                }
            });
}

inline void StacklessBVH::build(muda::CBufferView<AABB> aabbs)
{
    m_impl.build(aabbs);
}

template <std::invocable<IndexT, IndexT> Pred>
void StacklessBVH::detect(Pred callback, QueryBuffer& qbuffer)
{
    using namespace muda;
    // Query the LBVH

    if(m_impl.objs.size() == 0)
    {
        qbuffer.m_size = 0;
        return;
    }

    auto do_query = [&]
    {
        // clear counter
        cudaMemsetAsync(qbuffer.m_cpNum.data(), 0, sizeof(int));

        m_impl.StacklessCDSharedSelf(
            callback, qbuffer.m_cpNum.view(), qbuffer.m_pairs.view());
    };

    do_query();

    // get total number of pairs
    int h_cp_num = qbuffer.m_cpNum;
    // if failed, resize and retry
    if(h_cp_num > qbuffer.m_pairs.size())
    {
        qbuffer.m_pairs.resize(h_cp_num * m_impl.config.reserve_ratio);
        do_query();
    }

    UIPC_ASSERT(h_cp_num >= 0, "fatal error");
    qbuffer.m_size = h_cp_num;
}

inline void StacklessBVH::QueryBuffer::build(muda::CBufferView<AABB> aabbs)
{
    auto size = aabbs.size();
    m_queryMtCode.resize(size);
    m_querySortedId.resize(size);


    Impl::calcMaxBVFromBox(aabbs, m_querySceneBox);
    Impl::calcMCsFromBox(aabbs, m_querySceneBox, m_queryMtCode);

    auto d_querySceneBox = m_querySceneBox.data();
    auto d_queryMtCode   = m_queryMtCode.data();
    auto d_querySortedId = m_querySortedId.data();
    auto numQuery        = size;

    auto null_stream = thrust::cuda::par_nosync.on(nullptr);
    thrust::sequence(null_stream, d_querySortedId, d_querySortedId + numQuery);
    thrust::sort_by_key(null_stream, d_queryMtCode, d_queryMtCode + numQuery, d_querySortedId);
}

template <std::invocable<IndexT, IndexT> Pred>
void StacklessBVH::query(muda::CBufferView<AABB> aabbs, Pred callback, QueryBuffer& qbuffer)
{
    if(aabbs.size() == 0 || m_impl.objs.size() == 0)
    {
        qbuffer.m_size = 0;
        return;
    }

    using namespace muda;
    qbuffer.build(aabbs);

    auto do_query = [&]
    {
        // clear counter
        cudaMemsetAsync(qbuffer.m_cpNum.data(), 0, sizeof(int));

        m_impl.StacklessCDSharedOther(callback,
                                      aabbs,
                                      qbuffer.m_querySortedId.view(),
                                      qbuffer.m_cpNum.view(),
                                      qbuffer.m_pairs.view());
    };

    do_query();

    // get total number of pairs
    int h_cp_num = qbuffer.m_cpNum;
    // if failed, resize and retry
    if(h_cp_num > qbuffer.m_pairs.size())
    {
        qbuffer.m_pairs.resize(h_cp_num * m_impl.config.reserve_ratio);
        do_query();
    }

    UIPC_ASSERT(h_cp_num >= 0, "fatal error");
    qbuffer.m_size = h_cp_num;
}
}  // namespace uipc::backend::cuda
#endif
