#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
#include <collision_detection/filters/stackless_bvh_simplex_trajectory_filter.h>
#include <muda/cub/device/device_select.h>
#include <muda/ext/eigen/log_proxy.h>
#include <sim_engine.h>
#include <kernel_cout.h>
#include <utils/distance/distance_flagged.h>
#include <utils/distance.h>
#include <utils/codim_thickness.h>
#include <utils/simplex_contact_mask_utils.h>
#include <uipc/common/zip.h>
#include <utils/primitive_d_hat.h>
#include <utils/corex_phase_profile.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <string>

namespace uipc::backend::cuda::corex_filter
{
using AABB = uipc::backend::cuda::AABB;

static __global__ void kernel_build_point_aabbs(
    int N, const IndexT* Vs, const Vector3* Ps, const Vector3* dxs,
    const Float* thicknesses, const Float* d_hats, Float alpha, AABB* aabbs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    auto vI = Vs[i];
    Float thickness = thicknesses[vI];
    Float d_hat_expansion = point_dcd_expansion(d_hats[vI]);
    const auto& pos = Ps[vI];
    Vector3 pos_t = pos + dxs[vI] * alpha;
    AABB aabb;
    aabb.extend(pos.cast<float>()).extend(pos_t.cast<float>());
    float expand = d_hat_expansion + thickness;
    aabb.min().array() -= expand;
    aabb.max().array() += expand;
    aabbs[i] = aabb;
}

static __global__ void kernel_build_edge_aabbs(
    int N, const Vector2i* Es, const Vector3* Ps, const Vector3* dxs,
    const Float* thicknesses, const Float* d_hats, Float alpha, AABB* aabbs,
    Float* edge_thicknesses, Float* edge_d_hats)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    auto eI = Es[i];
    Float thickness = edge_thickness(thicknesses[eI[0]], thicknesses[eI[1]]);
    Float d_hat_expansion = edge_dcd_expansion(d_hats[eI[0]], d_hats[eI[1]]);
    edge_thicknesses[i] = thicknesses[eI[0]];
    edge_d_hats[i] = d_hats[eI[0]];
    const auto& pos0 = Ps[eI[0]];
    const auto& pos1 = Ps[eI[1]];
    Vector3 pos0_t = pos0 + dxs[eI[0]] * alpha;
    Vector3 pos1_t = pos1 + dxs[eI[1]] * alpha;
    AABB aabb;
    aabb.extend(pos0.cast<float>()).extend(pos1.cast<float>())
        .extend(pos0_t.cast<float>()).extend(pos1_t.cast<float>());
    float expand = d_hat_expansion + thickness;
    aabb.min().array() -= expand;
    aabb.max().array() += expand;
    aabbs[i] = aabb;
}

static __global__ void kernel_build_triangle_aabbs(
    int N, const Vector3i* Fs, const Vector3* Ps, const Vector3* dxs,
    const Float* thicknesses, const Float* d_hats, Float alpha, AABB* aabbs,
    Float* triangle_thicknesses, Float* triangle_d_hats)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    auto fI = Fs[i];
    Float thickness = triangle_thickness(thicknesses[fI[0]], thicknesses[fI[1]], thicknesses[fI[2]]);
    Float d_hat_expansion = triangle_dcd_expansion(d_hats[fI[0]], d_hats[fI[1]], d_hats[fI[2]]);
    triangle_thicknesses[i] = thicknesses[fI[0]];
    triangle_d_hats[i] = d_hats[fI[0]];
    const auto& pos0 = Ps[fI[0]];
    const auto& pos1 = Ps[fI[1]];
    const auto& pos2 = Ps[fI[2]];
    Vector3 pos0_t = pos0 + dxs[fI[0]] * alpha;
    Vector3 pos1_t = pos1 + dxs[fI[1]] * alpha;
    Vector3 pos2_t = pos2 + dxs[fI[2]] * alpha;
    AABB aabb;
    aabb.extend(pos0.cast<float>()).extend(pos1.cast<float>()).extend(pos2.cast<float>())
        .extend(pos0_t.cast<float>()).extend(pos1_t.cast<float>()).extend(pos2_t.cast<float>());
    float expand = d_hat_expansion + thickness;
    aabb.min().array() -= expand;
    aabb.max().array() += expand;
    aabbs[i] = aabb;
}

static __global__ void kernel_filter_toi_PP(
    int N, const Vector2i* pairs, const IndexT* codim_vertices, const IndexT* surf_vertices,
    const Float* thicknesses, const Vector3* positions, const Vector3* displacements,
    const Float* d_hats, Float alpha, Float eta, SizeT max_iter, Float large_enough_toi,
    Float* out_tois)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    auto   indices = pairs[i];
    IndexT V0      = surf_vertices[indices(0)];
    IndexT V1      = codim_vertices[indices(1)];

    Float thickness = PP_thickness(thicknesses[V0], thicknesses[V1]);
    Float d_hat     = PP_d_hat(d_hats[V0], d_hats[V1]);

    Vector3 VP0  = positions[V0];
    Vector3 VP1  = positions[V1];
    Vector3 dVP0 = alpha * displacements[V0];
    Vector3 dVP1 = alpha * displacements[V1];

    Float toi = large_enough_toi;

    bool faraway = !distance::point_point_ccd_broadphase(
        VP0, VP1, dVP0, dVP1, d_hat + thickness);

    if(!faraway)
    {
        bool hit = distance::point_point_ccd(
            VP0, VP1, dVP0, dVP1, eta, thickness, static_cast<int>(max_iter), toi);
        if(!hit) toi = large_enough_toi;
    }
    out_tois[i] = toi;
}

static __global__ void kernel_filter_toi_PE(
    int N, const Vector2i* pairs, const IndexT* codim_vertices, const Vector2i* surf_edges,
    const Float* thicknesses, const Vector3* positions, const Vector3* displacements,
    const Float* d_hats, Float alpha, Float eta, SizeT max_iter, Float large_enough_toi,
    Float* out_tois)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    auto     indices = pairs[i];
    IndexT   V       = codim_vertices[indices(0)];
    Vector2i E       = surf_edges[indices(1)];

    Float thickness =
        PE_thickness(thicknesses[V], thicknesses[E(0)], thicknesses[E(1)]);
    Float d_hat = PE_d_hat(d_hats[V], d_hats[E(0)], d_hats[E(1)]);

    Vector3 VP  = positions[V];
    Vector3 dVP = alpha * displacements[V];

    Vector3 EP0  = positions[E[0]];
    Vector3 EP1  = positions[E[1]];
    Vector3 dEP0 = alpha * displacements[E[0]];
    Vector3 dEP1 = alpha * displacements[E[1]];

    Float toi = large_enough_toi;

    bool faraway = !distance::point_edge_ccd_broadphase(
        VP, EP0, EP1, dVP, dEP0, dEP1, d_hat + thickness);

    if(!faraway)
    {
        bool hit = distance::point_edge_ccd(
            VP, EP0, EP1, dVP, dEP0, dEP1, eta, thickness, static_cast<int>(max_iter), toi);
        if(!hit) toi = large_enough_toi;
    }
    out_tois[i] = toi;
}

static __global__ void kernel_filter_toi_PT(
    int N, const Vector2i* pairs, const IndexT* surf_vertices, const Vector3i* surf_triangles,
    const Float* thicknesses, const Vector3* positions, const Vector3* displacements,
    const Float* d_hats, Float alpha, Float eta, SizeT max_iter, Float large_enough_toi,
    Float* out_tois)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    auto     indices = pairs[i];
    IndexT   V       = surf_vertices[indices(0)];
    Vector3i F       = surf_triangles[indices(1)];

    Float thickness = PT_thickness(thicknesses[V],
                                 thicknesses[F(0)],
                                 thicknesses[F(1)],
                                 thicknesses[F(2)]);
    Float d_hat =
        PT_d_hat(d_hats[V], d_hats[F(0)], d_hats[F(1)], d_hats[F(2)]);

    Vector3 VP  = positions[V];
    Vector3 dVP = alpha * displacements[V];

    Vector3 FP0 = positions[F[0]];
    Vector3 FP1 = positions[F[1]];
    Vector3 FP2 = positions[F[2]];

    Vector3 dFP0 = alpha * displacements[F[0]];
    Vector3 dFP1 = alpha * displacements[F[1]];
    Vector3 dFP2 = alpha * displacements[F[2]];

    Float toi = large_enough_toi;

    bool faraway = !distance::point_triangle_ccd_broadphase(
        VP, FP0, FP1, FP2, dVP, dFP0, dFP1, dFP2, d_hat + thickness);

    if(!faraway)
    {
        bool hit = distance::point_triangle_ccd(
            VP, FP0, FP1, FP2, dVP, dFP0, dFP1, dFP2, eta, thickness, static_cast<int>(max_iter), toi);
        if(!hit) toi = large_enough_toi;
    }
    out_tois[i] = toi;
}

static __global__ void kernel_filter_toi_EE(
    int N, const Vector2i* pairs, const Vector2i* surf_edges, const Float* thicknesses,
    const Vector3* positions, const Vector3* displacements, const Float* d_hats, Float alpha,
    Float eta, SizeT max_iter, Float large_enough_toi, Float* out_tois)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    auto     indices = pairs[i];
    Vector2i E0      = surf_edges[indices(0)];
    Vector2i E1      = surf_edges[indices(1)];

    Float thickness = EE_thickness(thicknesses[E0(0)],
                                   thicknesses[E0(1)],
                                   thicknesses[E1(0)],
                                   thicknesses[E1(1)]);

    Float d_hat =
        EE_d_hat(d_hats[E0(0)], d_hats[E0(1)], d_hats[E1(0)], d_hats[E1(1)]);

    Vector3 EP0  = positions[E0[0]];
    Vector3 EP1  = positions[E0[1]];
    Vector3 dEP0 = alpha * displacements[E0[0]];
    Vector3 dEP1 = alpha * displacements[E0[1]];

    Vector3 EP2  = positions[E1[0]];
    Vector3 EP3  = positions[E1[1]];
    Vector3 dEP2 = alpha * displacements[E1[0]];
    Vector3 dEP3 = alpha * displacements[E1[1]];

    Float toi = large_enough_toi;

    bool faraway = !distance::edge_edge_ccd_broadphase(
        EP0, EP1, EP2, EP3, dEP0, dEP1, dEP2, dEP3, d_hat + thickness);

    if(!faraway)
    {
        bool hit = distance::edge_edge_ccd(
            EP0, EP1, EP2, EP3, dEP0, dEP1, dEP2, dEP3, eta, thickness, static_cast<int>(max_iter), toi);
        if(!hit) toi = large_enough_toi;
    }
    out_tois[i] = toi;
}

static inline void corex_filter_active_post_launch()
{
    cudaGetLastError();
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

MUDA_DEVICE MUDA_INLINE Vector3 min3(const Vector3& a, const Vector3& b, const Vector3& c)
{
    return a.cwiseMin(b).cwiseMin(c);
}

MUDA_DEVICE MUDA_INLINE Vector3 max3(const Vector3& a, const Vector3& b, const Vector3& c)
{
    return a.cwiseMax(b).cwiseMax(c);
}

// filter_active kernels

static __global__ void kernel_filter_active_PP(
    int N, const Vector2i* PCodimP_pairs, const IndexT* surf_vertices,
    const IndexT* codim_vertices, const Vector3* positions,
    const Float* thicknesses, const Float* d_hats, Vector2i* out_PPs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    out_PPs[i].setConstant(-1);
    Vector2i indices = PCodimP_pairs[i];
    IndexT P0 = surf_vertices[indices(0)];
    IndexT P1 = codim_vertices[indices(1)];
    const auto& V0 = positions[P0];
    const auto& V1 = positions[P1];
    Float thickness = PP_thickness(thicknesses[P0], thicknesses[P1]);
    Float d_hat = PP_d_hat(d_hats[P0], d_hats[P1]);
    Vector2 range = D_range(thickness, d_hat);
    Float D;
    distance::point_point_distance2(V0, V1, D);
    if(!is_active_D(range, D))
        return;
    out_PPs[i] = {P0, P1};
}

static __global__ void kernel_filter_active_CodimPE(
    int N, const Vector2i* CodimP_AllE_pairs, const IndexT* codim_vertices,
    const Vector2i* surf_edges, const Vector3* positions,
    const Float* thicknesses, const Float* d_hats,
    Vector2i* out_PPs, Vector3i* out_PEs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    out_PPs[i].setConstant(-1);
    out_PEs[i].setConstant(-1);
    Vector2i indices = CodimP_AllE_pairs[i];
    IndexT   V       = codim_vertices[indices(0)];
    Vector2i E       = surf_edges[indices(1)];
    Vector3i vIs = {V, E(0), E(1)};
    Vector3 Ps_arr[] = {positions[vIs(0)], positions[vIs(1)], positions[vIs(2)]};
    Float thickness = PE_thickness(thicknesses[V], thicknesses[E(0)], thicknesses[E(1)]);
    Float d_hat = PE_d_hat(d_hats[V], d_hats[E(0)], d_hats[E(1)]);
    Vector3i flag = distance::point_edge_distance_flag(Ps_arr[0], Ps_arr[1], Ps_arr[2]);
    Vector2 range = D_range(thickness, d_hat);
    Float D;
    distance::point_edge_distance2(flag, Ps_arr[0], Ps_arr[1], Ps_arr[2], D);
    if(!is_active_D(range, D))
        return;
    Vector3i offsets;
    auto dim = distance::degenerate_point_edge(flag, offsets);
    switch(dim)
    {
        case 2:
        {
            IndexT V0 = vIs(offsets(0));
            IndexT V1 = vIs(offsets(1));
            out_PPs[i] = {V0, V1};
        }
        break;
        case 3:
        {
            out_PEs[i] = vIs;
        }
        break;
        default:
            break;
    }
}

static __global__ void kernel_filter_active_PT(
    int N, const Vector2i* PT_pairs, const IndexT* surf_vertices,
    const Vector3i* surf_triangles, const Vector3* positions,
    const Float* thicknesses, const Float* d_hats,
    Float pt_pe_hyst_scale,
    Vector2i* out_PPs, Vector3i* out_PEs, Vector4i* out_PTs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    out_PPs[i].setConstant(-1);
    out_PEs[i].setConstant(-1);
    out_PTs[i].setConstant(-1);
    Vector2i indices = PT_pairs[i];
    IndexT   V       = surf_vertices[indices(0)];
    Vector3i F       = surf_triangles[indices(1)];
    Vector4i vIs  = {V, F(0), F(1), F(2)};
    Vector3  Ps_arr[] = {positions[vIs(0)], positions[vIs(1)], positions[vIs(2)], positions[vIs(3)]};
    Float thickness = PT_thickness(thicknesses[V], thicknesses[F(0)], thicknesses[F(1)], thicknesses[F(2)]);
    Float d_hat = PT_d_hat(d_hats[V], d_hats[F(0)], d_hats[F(1)], d_hats[F(2)]);
    Vector2 range = D_range(thickness, d_hat);
    Vector3 tri_min = min3(Ps_arr[1], Ps_arr[2], Ps_arr[3]);
    Vector3 tri_max = max3(Ps_arr[1], Ps_arr[2], Ps_arr[3]);
    if(point_box_distance2_lower_bound(Ps_arr[0], tri_min, tri_max) >= range.y())
        return;
    Vector4i flag = distance::point_triangle_distance_flag(Ps_arr[0], Ps_arr[1], Ps_arr[2], Ps_arr[3]);
    Float D;
    distance::point_triangle_distance2(flag, Ps_arr[0], Ps_arr[1], Ps_arr[2], Ps_arr[3], D);
    Vector4i offsets;
    offsets.setConstant(-1);
    auto dim = distance::degenerate_point_triangle(flag, offsets);
    bool active = is_active_D(range, D);
    if(!active && dim == 3)
    {
        // Keep near-threshold PT->PE transitions from dropping on one-frame float noise.
        Float slack = max(static_cast<Float>(1e-6),
                          (range.y() - range.x()) * pt_pe_hyst_scale);
        active = (D > range.x()) && (D < range.y() + slack);
    }
    if(!active)
        return;
    switch(dim)
    {
        case 2:
        {
            IndexT V0 = vIs(offsets(0));
            IndexT V1 = vIs(offsets(1));
            out_PPs[i] = {V0, V1};
        }
        break;
        case 3:
        {
            IndexT V0 = vIs(offsets(0));
            IndexT V1 = vIs(offsets(1));
            IndexT V2 = vIs(offsets(2));
            out_PEs[i] = {V0, V1, V2};
        }
        break;
        case 4:
        {
            out_PTs[i] = vIs;
        }
        break;
        default:
            break;
    }
}

static __global__ void kernel_filter_active_EE(
    int N, const Vector2i* EE_pairs, const Vector2i* surf_edges,
    const Vector3* positions, const Vector3* rest_positions,
    const Float* thicknesses, const Float* d_hats,
    Vector2i* out_PPs, Vector3i* out_PEs, Vector4i* out_EEs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    out_PPs[i].setConstant(-1);
    out_PEs[i].setConstant(-1);
    out_EEs[i].setConstant(-1);
    Vector2i indices = EE_pairs[i];
    Vector2i E0_edge = surf_edges[indices(0)];
    Vector2i E1_edge = surf_edges[indices(1)];
    Vector4i vIs  = {E0_edge(0), E0_edge(1), E1_edge(0), E1_edge(1)};
    Vector3  Ps_arr[] = {positions[vIs(0)], positions[vIs(1)], positions[vIs(2)], positions[vIs(3)]};
    Float thickness = EE_thickness(thicknesses[E0_edge(0)], thicknesses[E0_edge(1)],
                                   thicknesses[E1_edge(0)], thicknesses[E1_edge(1)]);
    Float d_hat = EE_d_hat(d_hats[E0_edge(0)], d_hats[E0_edge(1)],
                           d_hats[E1_edge(0)], d_hats[E1_edge(1)]);
    Vector2 range = D_range(thickness, d_hat);
    Vector3 e0_min = Ps_arr[0].cwiseMin(Ps_arr[1]);
    Vector3 e0_max = Ps_arr[0].cwiseMax(Ps_arr[1]);
    Vector3 e1_min = Ps_arr[2].cwiseMin(Ps_arr[3]);
    Vector3 e1_max = Ps_arr[2].cwiseMax(Ps_arr[3]);
    if(box_box_distance2_lower_bound(e0_min, e0_max, e1_min, e1_max) >= range.y())
        return;
    Vector4i flag = distance::edge_edge_distance_flag(Ps_arr[0], Ps_arr[1], Ps_arr[2], Ps_arr[3]);
    Float D;
    distance::edge_edge_distance2(flag, Ps_arr[0], Ps_arr[1], Ps_arr[2], Ps_arr[3], D);
    if(D <= range.x())
    {
        out_EEs[i] = vIs;
        return;
    }
    if(!is_active_D(range, D))
        return;
    Vector4i offsets;
    auto dim = distance::degenerate_edge_edge(flag, offsets);
    if(dim == 4)
    {
        Float eps_x;
        distance::edge_edge_mollifier_threshold(rest_positions[vIs(0)], rest_positions[vIs(1)],
                                                rest_positions[vIs(2)], rest_positions[vIs(3)],
                                                static_cast<Float>(1e-3), eps_x);
        if(distance::need_mollify(Ps_arr[0], Ps_arr[1], Ps_arr[2], Ps_arr[3], eps_x))
        {
            out_EEs[i] = vIs;
            return;
        }
    }

    switch(dim)
    {
        case 2:
        {
            IndexT V0 = vIs(offsets(0));
            IndexT V1 = vIs(offsets(1));
            out_PPs[i] = {V0, V1};
        }
        break;
        case 3:
        {
            IndexT V0 = vIs(offsets(0));
            IndexT V1 = vIs(offsets(1));
            IndexT V2 = vIs(offsets(2));
            out_PEs[i] = {V0, V1, V2};
        }
        break;
        case 4:
        {
            out_EEs[i] = vIs;
        }
        break;
        default:
            break;
    }
}

static __global__ void kernel_filter_active_PP_append(
    int N, const Vector2i* PCodimP_pairs, const IndexT* surf_vertices,
    const IndexT* codim_vertices, const Vector3* positions,
    const Float* thicknesses, const Float* d_hats, Vector2i* out_PPs,
    IndexT* pp_count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    Vector2i indices = PCodimP_pairs[i];
    IndexT P0 = surf_vertices[indices(0)];
    IndexT P1 = codim_vertices[indices(1)];
    const auto& V0 = positions[P0];
    const auto& V1 = positions[P1];
    Float thickness = PP_thickness(thicknesses[P0], thicknesses[P1]);
    Float d_hat = PP_d_hat(d_hats[P0], d_hats[P1]);
    Vector2 range = D_range(thickness, d_hat);
    Float D;
    distance::point_point_distance2(V0, V1, D);
    if(!is_active_D(range, D))
        return;
    IndexT dst = atomicAdd(pp_count, IndexT{1});
    out_PPs[dst] = {P0, P1};
}

static __global__ void kernel_filter_active_CodimPE_append(
    int N, const Vector2i* CodimP_AllE_pairs, const IndexT* codim_vertices,
    const Vector2i* surf_edges, const Vector3* positions,
    const Float* thicknesses, const Float* d_hats,
    Vector2i* out_PPs, Vector3i* out_PEs, IndexT* pp_count, IndexT* pe_count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    Vector2i indices = CodimP_AllE_pairs[i];
    IndexT   V       = codim_vertices[indices(0)];
    Vector2i E       = surf_edges[indices(1)];
    Vector3i vIs = {V, E(0), E(1)};
    Vector3 Ps_arr[] = {positions[vIs(0)], positions[vIs(1)], positions[vIs(2)]};
    Float thickness = PE_thickness(thicknesses[V], thicknesses[E(0)], thicknesses[E(1)]);
    Float d_hat = PE_d_hat(d_hats[V], d_hats[E(0)], d_hats[E(1)]);
    Vector3i flag = distance::point_edge_distance_flag(Ps_arr[0], Ps_arr[1], Ps_arr[2]);
    Vector2 range = D_range(thickness, d_hat);
    Float D;
    distance::point_edge_distance2(flag, Ps_arr[0], Ps_arr[1], Ps_arr[2], D);
    if(!is_active_D(range, D))
        return;
    Vector3i offsets;
    auto dim = distance::degenerate_point_edge(flag, offsets);
    if(dim == 2)
    {
        IndexT dst = atomicAdd(pp_count, IndexT{1});
        out_PPs[dst] = {vIs(offsets(0)), vIs(offsets(1))};
    }
    else if(dim == 3)
    {
        IndexT dst = atomicAdd(pe_count, IndexT{1});
        out_PEs[dst] = vIs;
    }
}

static __global__ void kernel_filter_active_PT_append(
    int N, const Vector2i* PT_pairs, const IndexT* surf_vertices,
    const Vector3i* surf_triangles, const Vector3* positions,
    const Float* thicknesses, const Float* d_hats,
    const Float* triangle_thicknesses, const Float* triangle_d_hats,
    Float pt_pe_hyst_scale,
    Vector2i* out_PPs, Vector3i* out_PEs, Vector4i* out_PTs,
    IndexT* pp_count, IndexT* pe_count, IndexT* pt_count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    Vector2i indices = PT_pairs[i];
    IndexT   V       = surf_vertices[indices(0)];
    Vector3i F       = surf_triangles[indices(1)];
    Vector4i vIs  = {V, F(0), F(1), F(2)};
    Vector3  Ps_arr[] = {positions[vIs(0)], positions[vIs(1)], positions[vIs(2)], positions[vIs(3)]};
    Float thickness = thicknesses[V] + triangle_thicknesses[indices(1)];
    Float d_hat = (d_hats[V] + triangle_d_hats[indices(1)]) * Float{0.5};
    Vector2 range = D_range(thickness, d_hat);
    Vector3 tri_min = min3(Ps_arr[1], Ps_arr[2], Ps_arr[3]);
    Vector3 tri_max = max3(Ps_arr[1], Ps_arr[2], Ps_arr[3]);
    if(point_box_distance2_lower_bound(Ps_arr[0], tri_min, tri_max) >= range.y())
        return;
    Vector4i flag = distance::point_triangle_distance_flag(Ps_arr[0], Ps_arr[1], Ps_arr[2], Ps_arr[3]);
    Float D;
    distance::point_triangle_distance2(flag, Ps_arr[0], Ps_arr[1], Ps_arr[2], Ps_arr[3], D);
    Vector4i offsets;
    offsets.setConstant(-1);
    auto dim = distance::degenerate_point_triangle(flag, offsets);
    bool active = is_active_D(range, D);
    if(!active && dim == 3)
    {
        Float slack = max(static_cast<Float>(1e-6),
                          (range.y() - range.x()) * pt_pe_hyst_scale);
        active = (D > range.x()) && (D < range.y() + slack);
    }
    if(!active)
        return;
    if(dim == 2)
    {
        IndexT dst = atomicAdd(pp_count, IndexT{1});
        out_PPs[dst] = {vIs(offsets(0)), vIs(offsets(1))};
    }
    else if(dim == 3)
    {
        IndexT dst = atomicAdd(pe_count, IndexT{1});
        out_PEs[dst] = {vIs(offsets(0)), vIs(offsets(1)), vIs(offsets(2))};
    }
    else if(dim == 4)
    {
        IndexT dst = atomicAdd(pt_count, IndexT{1});
        out_PTs[dst] = vIs;
    }
}

static __global__ void kernel_filter_active_EE_append(
    int N, const Vector2i* EE_pairs, const Vector2i* surf_edges,
    const Vector3* positions, const Vector3* rest_positions,
    const Float* edge_thicknesses, const Float* edge_d_hats,
    Vector2i* out_PPs, Vector3i* out_PEs, Vector4i* out_EEs,
    IndexT* pp_count, IndexT* pe_count, IndexT* ee_count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    Vector2i indices = EE_pairs[i];
    Vector2i E0_edge = surf_edges[indices(0)];
    Vector2i E1_edge = surf_edges[indices(1)];
    Vector4i vIs  = {E0_edge(0), E0_edge(1), E1_edge(0), E1_edge(1)};
    Vector3  Ps_arr[] = {positions[vIs(0)], positions[vIs(1)], positions[vIs(2)], positions[vIs(3)]};
    Float thickness = edge_thicknesses[indices(0)] + edge_thicknesses[indices(1)];
    Float d_hat = (edge_d_hats[indices(0)] + edge_d_hats[indices(1)]) * Float{0.5};
    Vector2 range = D_range(thickness, d_hat);
    Vector3 e0_min = Ps_arr[0].cwiseMin(Ps_arr[1]);
    Vector3 e0_max = Ps_arr[0].cwiseMax(Ps_arr[1]);
    Vector3 e1_min = Ps_arr[2].cwiseMin(Ps_arr[3]);
    Vector3 e1_max = Ps_arr[2].cwiseMax(Ps_arr[3]);
    if(box_box_distance2_lower_bound(e0_min, e0_max, e1_min, e1_max) >= range.y())
        return;
    Vector4i flag = distance::edge_edge_distance_flag(Ps_arr[0], Ps_arr[1], Ps_arr[2], Ps_arr[3]);
    Float D;
    distance::edge_edge_distance2(flag, Ps_arr[0], Ps_arr[1], Ps_arr[2], Ps_arr[3], D);
    if(D <= range.x())
    {
        IndexT dst = atomicAdd(ee_count, IndexT{1});
        out_EEs[dst] = vIs;
        return;
    }
    if(!is_active_D(range, D))
        return;
    Vector4i offsets;
    auto dim = distance::degenerate_edge_edge(flag, offsets);
    if(dim == 4)
    {
        Float eps_x;
        distance::edge_edge_mollifier_threshold(rest_positions[vIs(0)], rest_positions[vIs(1)],
                                                rest_positions[vIs(2)], rest_positions[vIs(3)],
                                                static_cast<Float>(1e-3), eps_x);
        if(distance::need_mollify(Ps_arr[0], Ps_arr[1], Ps_arr[2], Ps_arr[3], eps_x))
        {
            IndexT dst = atomicAdd(ee_count, IndexT{1});
            out_EEs[dst] = vIs;
            return;
        }
    }

    if(dim == 2)
    {
        IndexT dst = atomicAdd(pp_count, IndexT{1});
        out_PPs[dst] = {vIs(offsets(0)), vIs(offsets(1))};
    }
    else if(dim == 3)
    {
        IndexT dst = atomicAdd(pe_count, IndexT{1});
        out_PEs[dst] = {vIs(offsets(0)), vIs(offsets(1)), vIs(offsets(2))};
    }
    else if(dim == 4)
    {
        IndexT dst = atomicAdd(ee_count, IndexT{1});
        out_EEs[dst] = vIs;
    }
}

}  // namespace uipc::backend::cuda::corex_filter

namespace uipc::backend::cuda
{
constexpr bool PrintDebugInfo = false;
constexpr bool PrintKernelZeroDistance = false;

namespace
{
bool corex_selected_set_diag_enabled()
{
    const char* env = std::getenv("UIPC_COREX_SELECTED_SET_DIAG");
    return env && env[0] != '\0' && env[0] != '0';
}

bool corex_selected_set_hash_diag_enabled()
{
    const char* env = std::getenv("UIPC_COREX_SELECTED_SET_HASH_DIAG");
    return env && env[0] != '\0' && env[0] != '0';
}

bool corex_filter_view_slice_enabled()
{
    const char* env = std::getenv("UIPC_COREX_FILTER_VIEW_SLICE");
    return env && env[0] != '\0' && env[0] != '0';
}

int corex_filter_aabb_async_mask()
{
    const char* mask_env = std::getenv("UIPC_COREX_FILTER_AABB_ASYNC_MASK");
    if(mask_env && mask_env[0] != '\0')
    {
        char* end = nullptr;
        long  v   = std::strtol(mask_env, &end, 0);
        if(end != mask_env && v >= 0)
            return static_cast<int>(v);
    }

    const char* env = std::getenv("UIPC_COREX_FILTER_AABB_ASYNC");
    if(env && env[0] != '\0')
        return env[0] != '0' ? 0xF : 0;

    // Keep CoreX default synchronized. AABB async can change selected-set evolution
    // on this path, so it remains opt-in through the mask env.
    return 0;
}

void corex_filter_detect_sync_if_needed(int stage_bit)
{
    if((corex_filter_aabb_async_mask() & stage_bit) == 0)
        cudaDeviceSynchronize();
}

template <typename T>
void corex_filter_loose_resize(muda::DeviceBuffer<T>& buffer, SizeT size)
{
    if(size > buffer.capacity())
        buffer.reserve(static_cast<size_t>(static_cast<double>(size) * 1.1) + 1);
    buffer.resize(size);
}

static __global__ void kernel_pack_selected_counts(const IndexT* pp_count,
                                                   const IndexT* pe_count,
                                                   const IndexT* pt_count,
                                                   const IndexT* ee_count,
                                                   IndexT*       out_counts)
{
    out_counts[0] = *pp_count;
    out_counts[1] = *pe_count;
    out_counts[2] = *pt_count;
    out_counts[3] = *ee_count;
}

struct CorexSelectedHashStats
{
    unsigned int sum_lo[4];
    unsigned int sum_hi[4];
    unsigned int sum2_lo[4];
    unsigned int sum2_hi[4];
};

__device__ inline unsigned long long corex_mix_u64(unsigned long long x)
{
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

__device__ inline unsigned long long corex_hash_index(IndexT v, int lane)
{
    return corex_mix_u64(static_cast<unsigned long long>(static_cast<long long>(v))
                         ^ (static_cast<unsigned long long>(lane + 1) * 0x9e3779b97f4a7c15ull));
}

__device__ inline void corex_hash_update(CorexSelectedHashStats* stats,
                                         int                     type,
                                         unsigned long long      h)
{
    unsigned long long h2 = corex_mix_u64(h);
    atomicAdd(&stats->sum_lo[type], static_cast<unsigned int>(h & 0xffffffffull));
    atomicAdd(&stats->sum_hi[type], static_cast<unsigned int>(h >> 32));
    atomicAdd(&stats->sum2_lo[type], static_cast<unsigned int>(h2 & 0xffffffffull));
    atomicAdd(&stats->sum2_hi[type], static_cast<unsigned int>(h2 >> 32));
}

static __global__ void kernel_hash_selected_pp(int N, const Vector2i* values, CorexSelectedHashStats* stats)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N)
        return;
    auto v = values[i];
    unsigned long long h = corex_hash_index(v(0), 0) ^ corex_hash_index(v(1), 1);
    corex_hash_update(stats, 0, corex_mix_u64(h));
}

static __global__ void kernel_hash_selected_pe(int N, const Vector3i* values, CorexSelectedHashStats* stats)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N)
        return;
    auto v = values[i];
    unsigned long long h =
        corex_hash_index(v(0), 0) ^ corex_hash_index(v(1), 1) ^ corex_hash_index(v(2), 2);
    corex_hash_update(stats, 1, corex_mix_u64(h));
}

static __global__ void kernel_hash_selected_pt(int N, const Vector4i* values, CorexSelectedHashStats* stats)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N)
        return;
    auto v = values[i];
    unsigned long long h = corex_hash_index(v(0), 0) ^ corex_hash_index(v(1), 1)
                         ^ corex_hash_index(v(2), 2) ^ corex_hash_index(v(3), 3);
    corex_hash_update(stats, 2, corex_mix_u64(h));
}

static __global__ void kernel_hash_selected_ee(int N, const Vector4i* values, CorexSelectedHashStats* stats)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N)
        return;
    auto v = values[i];
    unsigned long long h = corex_hash_index(v(0), 0) ^ corex_hash_index(v(1), 1)
                         ^ corex_hash_index(v(2), 2) ^ corex_hash_index(v(3), 3);
    corex_hash_update(stats, 3, corex_mix_u64(h));
}

void corex_log_selected_hash(int frame,
                             int newton_iter,
                             IndexT PP_count,
                             IndexT PE_count,
                             IndexT PT_count,
                             IndexT EE_count,
                             const muda::DeviceBuffer<Vector2i>& PPs,
                             const muda::DeviceBuffer<Vector3i>& PEs,
                             const muda::DeviceBuffer<Vector4i>& PTs,
                             const muda::DeviceBuffer<Vector4i>& EEs)
{
    CorexSelectedHashStats* stats = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&stats), sizeof(CorexSelectedHashStats));
    cudaMemset(stats, 0, sizeof(CorexSelectedHashStats));

    constexpr int block = 256;
    if(PP_count > 0)
        kernel_hash_selected_pp<<<(static_cast<int>(PP_count) + block - 1) / block, block>>>(
            static_cast<int>(PP_count), PPs.data(), stats);
    if(PE_count > 0)
        kernel_hash_selected_pe<<<(static_cast<int>(PE_count) + block - 1) / block, block>>>(
            static_cast<int>(PE_count), PEs.data(), stats);
    if(PT_count > 0)
        kernel_hash_selected_pt<<<(static_cast<int>(PT_count) + block - 1) / block, block>>>(
            static_cast<int>(PT_count), PTs.data(), stats);
    if(EE_count > 0)
        kernel_hash_selected_ee<<<(static_cast<int>(EE_count) + block - 1) / block, block>>>(
            static_cast<int>(EE_count), EEs.data(), stats);

    cudaDeviceSynchronize();
    CorexSelectedHashStats h_stats{};
    cudaMemcpy(&h_stats, stats, sizeof(CorexSelectedHashStats), cudaMemcpyDeviceToHost);
    cudaFree(stats);

    auto combine = [](unsigned int hi, unsigned int lo) -> unsigned long long
    {
        return (static_cast<unsigned long long>(hi) << 32) | static_cast<unsigned long long>(lo);
    };
    unsigned long long sum_hash[4] = {
        combine(h_stats.sum_hi[0], h_stats.sum_lo[0]),
        combine(h_stats.sum_hi[1], h_stats.sum_lo[1]),
        combine(h_stats.sum_hi[2], h_stats.sum_lo[2]),
        combine(h_stats.sum_hi[3], h_stats.sum_lo[3]),
    };
    unsigned long long sum2_hash[4] = {
        combine(h_stats.sum2_hi[0], h_stats.sum2_lo[0]),
        combine(h_stats.sum2_hi[1], h_stats.sum2_lo[1]),
        combine(h_stats.sum2_hi[2], h_stats.sum2_lo[2]),
        combine(h_stats.sum2_hi[3], h_stats.sum2_lo[3]),
    };

    spdlog::info("[corex_selected_hash] frame={} newton={} "
                 "PP_count={} PE_count={} PT_count={} EE_count={} "
                 "PP_sum={:#018x} PE_sum={:#018x} PT_sum={:#018x} EE_sum={:#018x} "
                 "PP_sum2={:#018x} PE_sum2={:#018x} PT_sum2={:#018x} EE_sum2={:#018x}",
                 frame,
                 newton_iter,
                 PP_count,
                 PE_count,
                 PT_count,
                 EE_count,
                 sum_hash[0],
                 sum_hash[1],
                 sum_hash[2],
                 sum_hash[3],
                 sum2_hash[0],
                 sum2_hash[1],
                 sum2_hash[2],
                 sum2_hash[3]);
}
}  // namespace

REGISTER_SIM_SYSTEM(StacklessBVHSimplexTrajectoryFilter);

void StacklessBVHSimplexTrajectoryFilter::do_build(BuildInfo& info)
{
    auto& config = world().scene().config();
    auto  method = config.find<std::string>("collision_detection/method");
    if(method->view()[0] != "stackless_bvh")
    {
        throw SimSystemException("Stackless BVH unused");
    }
}

void StacklessBVHSimplexTrajectoryFilter::do_detect(DetectInfo& info)
{
    m_impl.detect(info, engine().frame(), engine().newton_iter());
}

void StacklessBVHSimplexTrajectoryFilter::do_filter_active(FilterActiveInfo& info)
{
    m_impl.filter_active(info, engine().frame(), engine().newton_iter());
}

void StacklessBVHSimplexTrajectoryFilter::do_filter_toi(FilterTOIInfo& info)
{
    m_impl.filter_toi(info);
}

muda::CBufferView<Vector2i> StacklessBVHSimplexTrajectoryFilter::candidate_PTs() const noexcept
{
    return m_impl.candidate_AllP_AllT_pairs.view();
}

muda::CBufferView<Vector2i> StacklessBVHSimplexTrajectoryFilter::candidate_EEs() const noexcept
{
    return m_impl.candidate_AllE_AllE_pairs.view();
}

muda::CBufferView<Float> StacklessBVHSimplexTrajectoryFilter::toi_PTs() const noexcept
{
    auto pp_size = m_impl.candidate_AllP_CodimP_pairs.size();
    auto pe_size = m_impl.candidate_CodimP_AllE_pairs.size();
    auto pt_size = m_impl.candidate_AllP_AllT_pairs.size();
    return m_impl.tois.view(pp_size + pe_size, pt_size);
}

muda::CBufferView<Float> StacklessBVHSimplexTrajectoryFilter::toi_EEs() const noexcept
{
    auto pp_size = m_impl.candidate_AllP_CodimP_pairs.size();
    auto pe_size = m_impl.candidate_CodimP_AllE_pairs.size();
    auto pt_size = m_impl.candidate_AllP_AllT_pairs.size();
    auto ee_size = m_impl.candidate_AllE_AllE_pairs.size();
    return m_impl.tois.view(pp_size + pe_size + pt_size, ee_size);
}

void StacklessBVHSimplexTrajectoryFilter::Impl::detect(DetectInfo& info,
                                                       SizeT       frame,
                                                       SizeT       newton_iter)
{
    using namespace muda;
    (void)frame;
    (void)newton_iter;

    auto alpha   = info.alpha();
    auto Ps      = info.positions();
    auto dxs     = info.displacements();
    auto codimVs = info.codim_vertices();
    auto Vs      = info.surf_vertices();
    auto Es      = info.surf_edges();
    auto Fs      = info.surf_triangles();

    auto contact_mask_extent  = info.contact_mask_tabular().extent();
    auto subscene_mask_extent = info.subscene_mask_tabular().extent();
    const int cm_h = static_cast<int>(contact_mask_extent.height());
    const int cm_w = static_cast<int>(contact_mask_extent.width());
    const int sm_h = static_cast<int>(subscene_mask_extent.height());
    const int sm_w = static_cast<int>(subscene_mask_extent.width());
    const IndexT* contact_mask_ptr =
        (cm_h > 0 && cm_w > 0) ? info.contact_mask_tabular().data(0) : nullptr;
    const IndexT* subscene_mask_ptr =
        (sm_h > 0 && sm_w > 0) ? info.subscene_mask_tabular().data(0) : nullptr;

    if(!mask_cache_valid || cached_contact_mask_ptr != contact_mask_ptr
       || cached_subscene_mask_ptr != subscene_mask_ptr || cached_contact_mask_h != cm_h
       || cached_contact_mask_w != cm_w || cached_subscene_mask_h != sm_h
       || cached_subscene_mask_w != sm_w)
    {
        auto table_all_enabled = [](const IndexT* ptr, int h, int w) -> bool
        {
            if(ptr == nullptr || h <= 0 || w <= 0)
                return true;
            std::vector<IndexT> host(static_cast<size_t>(h) * static_cast<size_t>(w));
            checkCudaErrors(cudaMemcpy(
                host.data(), ptr, sizeof(IndexT) * host.size(), cudaMemcpyDeviceToHost));
            for(int r = 0; r < h; ++r)
            {
                for(int c = 0; c < w; ++c)
                {
                    // ABD scenes commonly disable same-contact/self rows. The
                    // traversal still checks body self-collision before accepting
                    // a pair, so diagonal mask entries do not need per-pair reads.
                    if(r == c)
                        continue;
                    if(host[static_cast<size_t>(r) * static_cast<size_t>(w) + c] == 0)
                        return false;
                }
            }
            return true;
        };

        contact_mask_all_enabled  = table_all_enabled(contact_mask_ptr, cm_h, cm_w);
        subscene_mask_all_enabled = table_all_enabled(subscene_mask_ptr, sm_h, sm_w);

        cached_contact_mask_ptr = contact_mask_ptr;
        cached_subscene_mask_ptr = subscene_mask_ptr;
        cached_contact_mask_h = cm_h;
        cached_contact_mask_w = cm_w;
        cached_subscene_mask_h = sm_h;
        cached_subscene_mask_w = sm_w;
        mask_cache_valid = true;
    }

    const bool mask_fast_enabled = [] {
        const char* env = std::getenv("UIPC_COREX_FILTER_MASK_FAST");
        return !(env && env[0] != '\0' && env[0] == '0');
    }();
    const bool contact_mask_fast  = mask_fast_enabled && contact_mask_all_enabled;
    const bool subscene_mask_fast = mask_fast_enabled && subscene_mask_all_enabled;
    const bool trace_simplex_filter = (std::getenv("UIPC_COREX_TRACE_SIMPLEX_FILTER") != nullptr);

    if(trace_simplex_filter)
    {
        static int detect_call = 0;
        if(detect_call < 4 || (detect_call % 50 == 0))
        {
            int nv = (int)Ps.size();
            std::vector<Vector3> h_pos(nv);
            std::vector<Vector3> h_disp(nv);
            cudaMemcpy(h_pos.data(), Ps.data(), nv * sizeof(Vector3), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_disp.data(), dxs.data(), nv * sizeof(Vector3), cudaMemcpyDeviceToHost);
            for(int i = 0; i < nv; ++i)
                spdlog::info("[detect_pos] call={} alpha={} v{} pos=({},{},{}) disp=({},{},{})",
                    detect_call, alpha, i,
                    h_pos[i][0], h_pos[i][1], h_pos[i][2],
                    h_disp[i][0], h_disp[i][1], h_disp[i][2]);
        }

        if(detect_call < 2)
        {
            const int nv    = static_cast<int>(Ps.size());
            const int nVs   = static_cast<int>(Vs.size());
            const int nCoVs = static_cast<int>(codimVs.size());
            const int nEs   = static_cast<int>(Es.size());
            const int nFs   = static_cast<int>(Fs.size());

            std::vector<IndexT> h_vs(nVs);
            std::vector<IndexT> h_codim_vs(nCoVs);
            std::vector<IndexT> h_cids(nv);
            std::vector<IndexT> h_scids(nv);
            std::vector<IndexT> h_bids(nv);
            std::vector<IndexT> h_dims(nv);

            if(nVs > 0)
                cudaMemcpy(h_vs.data(), Vs.data(), sizeof(IndexT) * nVs, cudaMemcpyDeviceToHost);
            if(nCoVs > 0)
                cudaMemcpy(h_codim_vs.data(),
                           codimVs.data(),
                           sizeof(IndexT) * nCoVs,
                           cudaMemcpyDeviceToHost);
            if(nv > 0)
            {
                cudaMemcpy(h_cids.data(),
                           info.contact_element_ids().data(),
                           sizeof(IndexT) * nv,
                           cudaMemcpyDeviceToHost);
                cudaMemcpy(h_scids.data(),
                           info.subscene_element_ids().data(),
                           sizeof(IndexT) * nv,
                           cudaMemcpyDeviceToHost);
                cudaMemcpy(h_bids.data(),
                           info.v2b().data(),
                           sizeof(IndexT) * nv,
                           cudaMemcpyDeviceToHost);
                cudaMemcpy(h_dims.data(),
                           info.dimensions().data(),
                           sizeof(IndexT) * nv,
                           cudaMemcpyDeviceToHost);
            }

            auto contact_mask_extent = info.contact_mask_tabular().extent();
            auto subscene_mask_extent = info.subscene_mask_tabular().extent();
            const int cm_h = static_cast<int>(contact_mask_extent.height());
            const int cm_w = static_cast<int>(contact_mask_extent.width());
            const int sm_h = static_cast<int>(subscene_mask_extent.height());
            const int sm_w = static_cast<int>(subscene_mask_extent.width());
            std::vector<IndexT> h_contact_mask(cm_h * cm_w);
            std::vector<IndexT> h_subscene_mask(sm_h * sm_w);
            if(cm_h > 0 && cm_w > 0)
            {
                cudaMemcpy(h_contact_mask.data(),
                           info.contact_mask_tabular().data(0),
                           sizeof(IndexT) * h_contact_mask.size(),
                           cudaMemcpyDeviceToHost);
            }
            if(sm_h > 0 && sm_w > 0)
            {
                cudaMemcpy(h_subscene_mask.data(),
                           info.subscene_mask_tabular().data(0),
                           sizeof(IndexT) * h_subscene_mask.size(),
                           cudaMemcpyDeviceToHost);
            }

            spdlog::info(
                "[corex_trace][detect_input] call={} alpha={} Nv={} Vs={} codimVs={} Es={} Fs={} "
                "contactMask={}x{} subsceneMask={}x{}",
                detect_call,
                alpha,
                nv,
                nVs,
                nCoVs,
                nEs,
                nFs,
                cm_h,
                cm_w,
                sm_h,
                sm_w);

            int surf_neg_cid = 0;
            int codim_neg_cid = 0;
            int surf_neg_scid = 0;
            int codim_neg_scid = 0;
            std::vector<IndexT> surf_cids;
            std::vector<IndexT> codim_cids;
            std::vector<IndexT> surf_bodies;
            std::vector<IndexT> codim_bodies;
            surf_cids.reserve(nVs);
            codim_cids.reserve(nCoVs);
            surf_bodies.reserve(nVs);
            codim_bodies.reserve(nCoVs);

            for(auto v : h_vs)
            {
                if(v < 0 || v >= nv)
                    continue;
                auto cid  = h_cids[v];
                auto scid = h_scids[v];
                auto bid  = h_bids[v];
                if(cid < 0)
                    ++surf_neg_cid;
                else
                    surf_cids.push_back(cid);
                if(scid < 0)
                    ++surf_neg_scid;
                if(bid >= 0)
                    surf_bodies.push_back(bid);
            }
            for(auto v : h_codim_vs)
            {
                if(v < 0 || v >= nv)
                    continue;
                auto cid  = h_cids[v];
                auto scid = h_scids[v];
                auto bid  = h_bids[v];
                if(cid < 0)
                    ++codim_neg_cid;
                else
                    codim_cids.push_back(cid);
                if(scid < 0)
                    ++codim_neg_scid;
                if(bid >= 0)
                    codim_bodies.push_back(bid);
            }

            auto uniq_count = [](std::vector<IndexT>& values) -> size_t
            {
                if(values.empty())
                    return 0;
                std::sort(values.begin(), values.end());
                values.erase(std::unique(values.begin(), values.end()), values.end());
                return values.size();
            };

            auto surf_cid_unique = uniq_count(surf_cids);
            auto codim_cid_unique = uniq_count(codim_cids);
            auto surf_body_unique = uniq_count(surf_bodies);
            auto codim_body_unique = uniq_count(codim_bodies);

            spdlog::info(
                "[corex_trace][detect_input] surf_neg_cid={} codim_neg_cid={} "
                "surf_neg_scid={} codim_neg_scid={} surf_cid_unique={} codim_cid_unique={} "
                "surf_body_unique={} codim_body_unique={}",
                surf_neg_cid,
                codim_neg_cid,
                surf_neg_scid,
                codim_neg_scid,
                surf_cid_unique,
                codim_cid_unique,
                surf_body_unique,
                codim_body_unique);

            auto sample_count = std::min(nVs, 8);
            for(int i = 0; i < sample_count; ++i)
            {
                auto v = h_vs[i];
                if(v < 0 || v >= nv)
                    continue;
                spdlog::info(
                    "[corex_trace][detect_input] surf_sample[{}] v={} body={} dim={} cid={} scid={}",
                    i,
                    v,
                    h_bids[v],
                    h_dims[v],
                    h_cids[v],
                    h_scids[v]);
            }

            auto first_body_vertex = [&](IndexT bid) -> IndexT
            {
                for(auto v : h_vs)
                {
                    if(v >= 0 && v < nv && h_bids[v] == bid)
                        return v;
                }
                return -1;
            };
            auto v0 = first_body_vertex(0);
            auto v1 = first_body_vertex(1);
            if(v0 >= 0 && v1 >= 0)
            {
                auto cid0  = h_cids[v0];
                auto cid1  = h_cids[v1];
                auto scid0 = h_scids[v0];
                auto scid1 = h_scids[v1];
                auto contact_allow = -1;
                auto subscene_allow = -1;
                if(cid0 >= 0 && cid0 < cm_h && cid1 >= 0 && cid1 < cm_w)
                    contact_allow = h_contact_mask[cid0 * cm_w + cid1];
                if(scid0 >= 0 && scid0 < sm_h && scid1 >= 0 && scid1 < sm_w)
                    subscene_allow = h_subscene_mask[scid0 * sm_w + scid1];
                spdlog::info(
                    "[corex_trace][detect_input] body_pair_samples b0_v={} cid/scid=({},{}) "
                    "b1_v={} cid/scid=({},{}) contact_allow={} subscene_allow={}",
                    v0,
                    cid0,
                    scid0,
                    v1,
                    cid1,
                    scid1,
                    contact_allow,
                    subscene_allow);
            }

            for(int r = 0; r < std::min(cm_h, 4); ++r)
            {
                std::string row;
                for(int c = 0; c < std::min(cm_w, 4); ++c)
                {
                    if(!row.empty())
                        row += ",";
                    row += std::to_string(h_contact_mask[r * cm_w + c]);
                }
                spdlog::info("[corex_trace][detect_input] contact_mask_row{}={}", r, row);
            }
        }
        detect_call++;
    }

    //lbvh_E      = {};
    //lbvh_T      = {};
    //lbvh_CodimP = {};

    point_aabbs.resize(Vs.size());
    triangle_aabbs.resize(Fs.size());
    edge_aabbs.resize(Es.size());
    triangle_thicknesses.resize(Fs.size());
    triangle_d_hats.resize(Fs.size());
    edge_thicknesses.resize(Es.size());
    edge_d_hats.resize(Es.size());

    {
        corex_profile::ScopedPhase phase("contact_detect_detail", "build_aabbs");
        // build AABBs for codim vertices
        if(codimVs.size() > 0)
        {
            codim_point_aabbs.resize(codimVs.size());

            int block = 256, grid = ((int)codimVs.size() + block - 1) / block;
            corex_filter::kernel_build_point_aabbs<<<grid, block>>>(
                codimVs.size(), (const IndexT*)codimVs.data(), (const Vector3*)Ps.data(),
                (const Vector3*)dxs.data(), (const Float*)info.thicknesses().data(),
                (const Float*)info.d_hats().data(), alpha, codim_point_aabbs.data());
            corex_filter_detect_sync_if_needed(0x1);
        }

        // build AABBs for surf vertices (including codim vertices)
        if(Vs.size() > 0)
        {
            int block = 256, grid = ((int)Vs.size() + block - 1) / block;
            corex_filter::kernel_build_point_aabbs<<<grid, block>>>(
                Vs.size(), (const IndexT*)Vs.data(), (const Vector3*)Ps.data(),
                (const Vector3*)dxs.data(), (const Float*)info.thicknesses().data(),
                (const Float*)info.d_hats().data(), alpha, point_aabbs.data());
            corex_filter_detect_sync_if_needed(0x2);
        }

        // build AABBs for edges
        if(Es.size() > 0)
        {
            int block = 256, grid = ((int)Es.size() + block - 1) / block;
            corex_filter::kernel_build_edge_aabbs<<<grid, block>>>(
                Es.size(), (const Vector2i*)Es.data(), (const Vector3*)Ps.data(),
                (const Vector3*)dxs.data(), (const Float*)info.thicknesses().data(),
                (const Float*)info.d_hats().data(), alpha, edge_aabbs.data(),
                edge_thicknesses.data(), edge_d_hats.data());
            corex_filter_detect_sync_if_needed(0x4);
        }

        // build AABBs for triangles
        if(Fs.size() > 0)
        {
            int block = 256, grid = ((int)Fs.size() + block - 1) / block;
            corex_filter::kernel_build_triangle_aabbs<<<grid, block>>>(
                Fs.size(), (const Vector3i*)Fs.data(), (const Vector3*)Ps.data(),
                (const Vector3*)dxs.data(), (const Float*)info.thicknesses().data(),
                (const Float*)info.d_hats().data(), alpha, triangle_aabbs.data(),
                triangle_thicknesses.data(), triangle_d_hats.data());
            corex_filter_detect_sync_if_needed(0x8);
        }
    }

    {
        const bool can_refit_edge_tri =
            alpha > 0 && edge_tri_bvh_valid && edge_bvh_size == edge_aabbs.size()
            && tri_bvh_size == triangle_aabbs.size();
        corex_profile::ScopedPhase phase("contact_detect_detail",
                                         can_refit_edge_tri ? "bvh_refit_edge_tri"
                                                           : "bvh_build_edge_tri");
        if(can_refit_edge_tri)
        {
            if(edge_aabbs.size() > 0)
                lbvh_E.refit(edge_aabbs);
            if(triangle_aabbs.size() > 0)
                lbvh_T.refit(triangle_aabbs);
        }
        else
        {
            lbvh_E.build(edge_aabbs);
            lbvh_T.build(triangle_aabbs);
        }
        edge_tri_bvh_valid = edge_aabbs.size() > 0 || triangle_aabbs.size() > 0;
        edge_bvh_size      = edge_aabbs.size();
        tri_bvh_size       = triangle_aabbs.size();
    }

    if(codimVs.size() > 0)
    {
        {
            corex_profile::ScopedPhase phase("contact_detect_detail", "query_allp_codimp");
            // Use AllP to query CodimP
            lbvh_CodimP.build(codim_point_aabbs);

            muda::KernelLabel label{__FUNCTION__, __FILE__, __LINE__};
            lbvh_CodimP.query(
                point_aabbs,                                  // AllP
                [Vs      = Vs.viewer().name("Vs"),            // AllP
                 codimVs = codimVs.viewer().name("codimVs"),  // CodimP

                 Ps          = Ps.viewer().name("Ps"),
                 dxs         = dxs.viewer().name("dxs"),
                 thicknesses = info.thicknesses().viewer().name("thicknesses"),
                 dimensions  = info.dimensions().viewer().name("dimensions"),
                 contact_element_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                 contact_mask_tabular = info.contact_mask_tabular().viewer().name("contact_mask_tabular"),
                 subscene_element_ids = info.subscene_element_ids().viewer().name("subscene_element_ids"),
                 subscene_mask_tabular = info.subscene_mask_tabular().viewer().name("subscene_mask_tabular"),
                 v2b = info.v2b().viewer().name("v2b"),
                 body_self_collision = info.body_self_collision().viewer().name("body_self_collision"),
                 d_hats = info.d_hats().viewer().name("d_hats"),
                 alpha  = alpha] __device__(IndexT i, IndexT j)
                {
                    const auto& V      = Vs(i);
                    const auto& codimV = codimVs(j);

                    Vector2i cids = {contact_element_ids(V), contact_element_ids(codimV)};
                    Vector2i scids = {subscene_element_ids(V), subscene_element_ids(codimV)};

                    // discard if the contact is disabled
                    if(!allow_PP_contact(subscene_mask_tabular, scids))
                        return false;
                    if(!allow_PP_contact(contact_mask_tabular, cids))
                        return false;

                    bool V_is_codim = dimensions(V) <= 2;  // codim 0D vert and vert from codim 1D edge

                    if(V_is_codim && V >= codimV)  // avoid duplicate CodimP-CodimP pairs
                        return false;

                    auto body_i = v2b(V);
                    auto body_j = v2b(codimV);
                    // skip self-collision for the same body if self collision off
                    if(body_i == body_j && !body_self_collision(body_i))
                        return false;


                    Vector3 P0  = Ps(V);
                    Vector3 dP0 = alpha * dxs(V);

                    Vector3 P1  = Ps(codimV);
                    Vector3 dP1 = alpha * dxs(codimV);

                    Float thickness = PP_thickness(thicknesses(V), thicknesses(codimV));
                    Float d_hat = PP_d_hat(d_hats(V), d_hats(codimV));

                    Float expand = d_hat + thickness;

                    if(!distance::point_point_ccd_broadphase(P0, P1, dP0, dP1, expand))
                        return false;

                    return true;
                },
                candidate_AllP_CodimP_pairs);
        }

        {
            corex_profile::ScopedPhase phase("contact_detect_detail", "query_codimp_alle");
            // Use CodimP to query AllE
            muda::KernelLabel label{__FUNCTION__, __FILE__, __LINE__};
            lbvh_E.query(
                codim_point_aabbs,
                [codimVs     = codimVs.viewer().name("Vs"),
                 Es          = Es.viewer().name("Es"),
                 Ps          = Ps.viewer().name("Ps"),
                 dxs         = dxs.viewer().name("dxs"),
                 thicknesses = info.thicknesses().viewer().name("thicknesses"),
                 contact_element_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                 contact_mask_tabular = info.contact_mask_tabular().viewer().name("contact_mask_tabular"),
                 subscene_element_ids = info.subscene_element_ids().viewer().name("subscene_element_ids"),
                 subscene_mask_tabular = info.subscene_mask_tabular().viewer().name("subscene_mask_tabular"),
                 v2b = info.v2b().viewer().name("v2b"),
                 body_self_collision = info.body_self_collision().viewer().name("body_self_collision"),
                 d_hats = info.d_hats().viewer().name("d_hats"),
                 alpha  = alpha] __device__(IndexT i, IndexT j)
                {
                    const auto& codimV = codimVs(i);
                    const auto& E      = Es(j);

                    Vector3i cids = {contact_element_ids(codimV),
                                     contact_element_ids(E[0]),
                                     contact_element_ids(E[1])};

                    Vector3i scids = {subscene_element_ids(codimV),
                                      subscene_element_ids(E[0]),
                                      subscene_element_ids(E[1])};

                    // discard if the contact is disabled
                    if(!allow_PE_contact(subscene_mask_tabular, scids))
                        return false;
                    if(!allow_PE_contact(contact_mask_tabular, cids))
                        return false;

                    // discard if the vertex is on the edge
                    if(E[0] == codimV || E[1] == codimV)
                        return false;

                    auto body_i = v2b(codimV);
                    auto body_j = v2b(E[0]);
                    // skip self-collision for the same body if self collision off
                    if(body_i == body_j && !body_self_collision(body_i))
                        return false;

                    Vector3 E0  = Ps(E[0]);
                    Vector3 E1  = Ps(E[1]);
                    Vector3 dE0 = alpha * dxs(E[0]);
                    Vector3 dE1 = alpha * dxs(E[1]);

                    Vector3 P  = Ps(codimV);
                    Vector3 dP = alpha * dxs(codimV);

                    Float thickness = PE_thickness(thicknesses(codimV),
                                                   thicknesses(E[0]),
                                                   thicknesses(E[1]));
                    Float d_hat = PE_d_hat(d_hats(codimV), d_hats(E[0]), d_hats(E[1]));

                    Float expand = d_hat + thickness;

                    if(!distance::point_edge_ccd_broadphase(P, E0, E1, dP, dE0, dE1, expand))
                        return false;

                    return true;
                },
                candidate_CodimP_AllE_pairs);
        }
    }

    bool pending_fast_EE_count = false;
    bool pending_fast_PT_count = false;

    // Use AllE to query AllE
    if(Es.size() > 0)
    {
        corex_profile::ScopedPhase phase(
            "contact_detect_detail",
            contact_mask_fast && subscene_mask_fast
                ? (alpha > 0 ? "query_alle_alle_ccd_fast" : "query_alle_alle_dcd_fast")
                : "query_alle_alle");
        if(contact_mask_fast && subscene_mask_fast)
        {
            pending_fast_EE_count =
                lbvh_E.detect_edges_no_mask_launch(Es,
                                                   info.v2b(),
                                                   info.body_self_collision(),
                                                   candidate_AllE_AllE_pairs);
        }
        else
        {
            muda::KernelLabel label{__FUNCTION__, __FILE__, __LINE__};
            lbvh_E.detect(
            [Es          = Es.viewer().name("Es"),
             Ps          = Ps.viewer().name("Ps"),
             dxs         = dxs.viewer().name("dxs"),
             thicknesses = info.thicknesses().viewer().name("thicknesses"),
             contact_element_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
             contact_mask_tabular = info.contact_mask_tabular().viewer().name("contact_mask_tabular"),
             subscene_element_ids = info.subscene_element_ids().viewer().name("subscene_element_ids"),
             subscene_mask_tabular = info.subscene_mask_tabular().viewer().name("subscene_mask_tabular"),
             v2b = info.v2b().viewer().name("v2b"),
             body_self_collision = info.body_self_collision().viewer().name("body_self_collision"),
             d_hats = info.d_hats().viewer().name("d_hats"),
             alpha  = alpha] __device__(IndexT i, IndexT j)
            {
                const auto& E0 = Es(i);
                const auto& E1 = Es(j);

                Vector4i cids = {contact_element_ids(E0[0]),
                                 contact_element_ids(E0[1]),
                                 contact_element_ids(E1[0]),
                                 contact_element_ids(E1[1])};

                Vector4i scids = {subscene_element_ids(E0[0]),
                                  subscene_element_ids(E0[1]),
                                  subscene_element_ids(E1[0]),
                                  subscene_element_ids(E1[1])};

                // discard if the contact is disabled
                if(!allow_EE_contact(subscene_mask_tabular, scids))
                    return false;
                if(!allow_EE_contact(contact_mask_tabular, cids))
                    return false;

                // discard if the edges share same vertex
                if(E0[0] == E1[0] || E0[0] == E1[1] || E0[1] == E1[0] || E0[1] == E1[1])
                    return false;

                auto body_i = v2b(E0[0]);
                auto body_j = v2b(E1[0]);
                if(body_i == body_j && !body_self_collision(body_i))
                    return false;  // skip self-collision for the same body


                Vector3 E0_0  = Ps(E0[0]);
                Vector3 E0_1  = Ps(E0[1]);
                Vector3 dE0_0 = alpha * dxs(E0[0]);
                Vector3 dE0_1 = alpha * dxs(E0[1]);

                Vector3 E1_0  = Ps(E1[0]);
                Vector3 E1_1  = Ps(E1[1]);
                Vector3 dE1_0 = alpha * dxs(E1[0]);
                Vector3 dE1_1 = alpha * dxs(E1[1]);

                Float thickness = EE_thickness(thicknesses(E0[0]),
                                               thicknesses(E0[1]),
                                               thicknesses(E1[0]),
                                               thicknesses(E1[1]));

                Float d_hat =
                    EE_d_hat(d_hats(E0[0]), d_hats(E0[1]), d_hats(E1[0]), d_hats(E1[1]));

                Float expand = d_hat + thickness;

                if(!distance::edge_edge_ccd_broadphase(
                       E0_0, E0_1, E1_0, E1_1, dE0_0, dE0_1, dE1_0, dE1_1, expand))
                    return false;

                return true;
            },
            candidate_AllE_AllE_pairs);
        }
    }

    // Use AllP to query AllT
    if(Fs.size() > 0)
    {
        corex_profile::ScopedPhase phase(
            "contact_detect_detail",
            contact_mask_fast && subscene_mask_fast ? "query_allp_allt_fast"
                                                    : "query_allp_allt");
        if(contact_mask_fast && subscene_mask_fast)
        {
            pending_fast_PT_count =
                lbvh_T.query_points_triangles_no_mask_launch(point_aabbs,
                                                             Vs,
                                                             Fs,
                                                             Ps,
                                                             dxs,
                                                             info.thicknesses(),
                                                             info.d_hats(),
                                                             alpha,
                                                             info.v2b(),
                                                             info.body_self_collision(),
                                                             candidate_AllP_AllT_pairs);
        }
        else
        {
            muda::KernelLabel label{__FUNCTION__, __FILE__, __LINE__};
            lbvh_T.query(
                point_aabbs,
                [Vs          = Vs.viewer().name("Vs"),
                 Fs          = Fs.viewer().name("Fs"),
                 Ps          = Ps.viewer().name("Ps"),
                 dxs         = dxs.viewer().name("dxs"),
                 thicknesses = info.thicknesses().viewer().name("thicknesses"),
                 contact_element_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                 contact_mask_tabular = info.contact_mask_tabular().viewer().name("contact_mask_tabular"),
                 subscene_element_ids = info.subscene_element_ids().viewer().name("subscene_element_ids"),
                 subscene_mask_tabular = info.subscene_mask_tabular().viewer().name("subscene_mask_tabular"),
                 v2b = info.v2b().viewer().name("v2b"),
                 body_self_collision = info.body_self_collision().viewer().name("body_self_collision"),
                 d_hats = info.d_hats().viewer().name("d_hats"),
                 contact_mask_fast,
                 subscene_mask_fast,
                 alpha  = alpha] __device__(IndexT i, IndexT j)
                {
                    auto V = Vs(i);
                    auto F = Fs(j);

                    if(!subscene_mask_fast)
                    {
                        Vector4i scids = {subscene_element_ids(V),
                                          subscene_element_ids(F[0]),
                                          subscene_element_ids(F[1]),
                                          subscene_element_ids(F[2])};
                        if(!allow_PT_contact(subscene_mask_tabular, scids))
                            return false;
                    }

                    if(!contact_mask_fast)
                    {
                        Vector4i cids = {contact_element_ids(V),
                                         contact_element_ids(F[0]),
                                         contact_element_ids(F[1]),
                                         contact_element_ids(F[2])};
                        if(!allow_PT_contact(contact_mask_tabular, cids))
                            return false;
                    }

                    // discard if the point is on the triangle
                    if(F[0] == V || F[1] == V || F[2] == V)
                        return false;

                    auto body_i = v2b(V);
                    auto body_j = v2b(F[0]);
                    // skip self-collision for the same body if self collision off
                    if(body_i == body_j && !body_self_collision(body_i))
                        return false;

                    Vector3 P  = Ps(V);
                    Vector3 dP = alpha * dxs(V);

                    Vector3 F0 = Ps(F[0]);
                    Vector3 F1 = Ps(F[1]);
                    Vector3 F2 = Ps(F[2]);

                    Vector3 dF0 = alpha * dxs(F[0]);
                    Vector3 dF1 = alpha * dxs(F[1]);
                    Vector3 dF2 = alpha * dxs(F[2]);

                    Float thickness = PT_thickness(thicknesses(V),
                                                   thicknesses(F[0]),
                                                   thicknesses(F[1]),
                                                   thicknesses(F[2]));

                    Float d_hat =
                        PT_d_hat(d_hats(V), d_hats(F[0]), d_hats(F[1]), d_hats(F[2]));

                    Float expand = d_hat + thickness;

                    if(!distance::point_triangle_ccd_broadphase(
                           P, F0, F1, F2, dP, dF0, dF1, dF2, expand))
                        return false;

                    return true;
                },
                candidate_AllP_AllT_pairs);
        }
    }

    if(pending_fast_EE_count || pending_fast_PT_count)
    {
        int h_EE_cp_num = 0;
        int h_PT_cp_num = 0;
        {
            corex_profile::ScopedPhase phase("bvh_query_detail",
                                             "fast_nomask_count_readback");
            if(pending_fast_EE_count)
                h_EE_cp_num = candidate_AllE_AllE_pairs.m_cpNum;
            if(pending_fast_PT_count)
                h_PT_cp_num = candidate_AllP_AllT_pairs.m_cpNum;
        }

        if(pending_fast_EE_count)
        {
            UIPC_ASSERT(h_EE_cp_num >= 0, "fatal error");
            if(h_EE_cp_num > candidate_AllE_AllE_pairs.m_pairs.size())
            {
                candidate_AllE_AllE_pairs.m_pairs.resize(
                    static_cast<size_t>(h_EE_cp_num) * 2);
                lbvh_E.detect_edges_no_mask(Es,
                                            info.v2b(),
                                            info.body_self_collision(),
                                            candidate_AllE_AllE_pairs);
            }
            else
            {
                candidate_AllE_AllE_pairs.m_size = h_EE_cp_num;
            }
        }

        if(pending_fast_PT_count)
        {
            UIPC_ASSERT(h_PT_cp_num >= 0, "fatal error");
            if(h_PT_cp_num > candidate_AllP_AllT_pairs.m_pairs.size())
            {
                candidate_AllP_AllT_pairs.m_pairs.resize(
                    static_cast<size_t>(h_PT_cp_num) * 2);
                lbvh_T.query_points_triangles_no_mask(point_aabbs,
                                                      Vs,
                                                      Fs,
                                                      Ps,
                                                      dxs,
                                                      info.thicknesses(),
                                                      info.d_hats(),
                                                      alpha,
                                                      info.v2b(),
                                                      info.body_self_collision(),
                                                      candidate_AllP_AllT_pairs);
            }
            else
            {
                candidate_AllP_AllT_pairs.m_size = h_PT_cp_num;
            }
        }
    }

    if(trace_simplex_filter)
        spdlog::info("[corex_trace][detect] alpha={} PP_cands={} CodimPE_cands={} PT_cands={} EE_cands={}",
                     alpha,
                     (int)candidate_AllP_CodimP_pairs.size(),
                     (int)candidate_CodimP_AllE_pairs.size(),
                     (int)candidate_AllP_AllT_pairs.size(),
                     (int)candidate_AllE_AllE_pairs.size());
}

void StacklessBVHSimplexTrajectoryFilter::Impl::filter_active(FilterActiveInfo& info,
                                                              int frame,
                                                              int newton_iter)
{
    using namespace muda;
    const bool trace_filter_active_diag =
        (std::getenv("UIPC_COREX_TRACE_FILTER_ACTIVE_DIAG") != nullptr);
    const bool selected_set_diag = corex_selected_set_diag_enabled();
    const bool selected_set_hash_diag = corex_selected_set_hash_diag_enabled();
    constexpr bool view_slice_output = true;
    Float pt_pe_hyst_scale = static_cast<Float>(0.0);
    if(const char* env = std::getenv("UIPC_COREX_PTPE_HYST_SCALE"))
    {
        char* end = nullptr;
        double v = std::strtod(env, &end);
        if(end != env && v >= 0.0 && v <= 0.5)
            pt_pe_hyst_scale = static_cast<Float>(v);
    }
    // we will filter-out the active pairs
    auto positions = info.positions();

    SizeT N_PCoimP  = candidate_AllP_CodimP_pairs.size();
    SizeT N_CodimPE = candidate_CodimP_AllE_pairs.size();
    SizeT N_PTs     = candidate_AllP_AllT_pairs.size();
    SizeT N_EEs     = candidate_AllE_AllE_pairs.size();

    if(N_PCoimP + N_CodimPE + N_PTs + N_EEs == 0)
    {
        PPs.resize(0);
        PEs.resize(0);
        PTs.resize(0);
        EEs.resize(0);
        info.PPs(PPs);
        info.PEs(PEs);
        info.PTs(PTs);
        info.EEs(EEs);
        return;
    }

    {
        corex_profile::ScopedPhase phase("contact_filter_detail", "append_valid_all");

        IndexT PP_count = 0;
        IndexT PE_count = 0;
        IndexT PT_count = 0;
        IndexT EE_count = 0;

        SizeT PP_capacity = N_PCoimP + N_CodimPE + N_PTs + N_EEs;
        SizeT PE_capacity = N_CodimPE + N_PTs + N_EEs;
        SizeT PT_capacity = N_PTs;
        SizeT EE_capacity = N_EEs;

        if(view_slice_output)
        {
            corex_filter_loose_resize(PPs, PP_capacity);
            corex_filter_loose_resize(PEs, PE_capacity);
            corex_filter_loose_resize(PTs, PT_capacity);
            corex_filter_loose_resize(EEs, EE_capacity);
        }
        else
        {
            PPs.resize(PP_capacity);
            PEs.resize(PE_capacity);
            PTs.resize(PT_capacity);
            EEs.resize(EE_capacity);
        }

        checkCudaErrors(cudaMemsetAsync(selected_PP_count.data(), 0, sizeof(IndexT)));
        checkCudaErrors(cudaMemsetAsync(selected_PE_count.data(), 0, sizeof(IndexT)));
        checkCudaErrors(cudaMemsetAsync(selected_PT_count.data(), 0, sizeof(IndexT)));
        checkCudaErrors(cudaMemsetAsync(selected_EE_count.data(), 0, sizeof(IndexT)));

        constexpr int block = 256;
        if(N_PCoimP > 0)
        {
            int n = static_cast<int>(N_PCoimP);
            int grid = (n + block - 1) / block;
            corex_filter::kernel_filter_active_PP_append<<<grid, block>>>(
                n,
                (const Vector2i*)candidate_AllP_CodimP_pairs.view().data(),
                (const IndexT*)info.surf_vertices().data(),
                (const IndexT*)info.codim_vertices().data(),
                (const Vector3*)positions.data(),
                (const Float*)info.thicknesses().data(),
                (const Float*)info.d_hats().data(),
                PPs.data(),
                selected_PP_count.data());
            corex_filter::corex_filter_active_post_launch();
        }
        if(N_CodimPE > 0)
        {
            int n = static_cast<int>(N_CodimPE);
            int grid = (n + block - 1) / block;
            corex_filter::kernel_filter_active_CodimPE_append<<<grid, block>>>(
                n,
                (const Vector2i*)candidate_CodimP_AllE_pairs.view().data(),
                (const IndexT*)info.codim_vertices().data(),
                (const Vector2i*)info.surf_edges().data(),
                (const Vector3*)positions.data(),
                (const Float*)info.thicknesses().data(),
                (const Float*)info.d_hats().data(),
                PPs.data(),
                PEs.data(),
                selected_PP_count.data(),
                selected_PE_count.data());
            corex_filter::corex_filter_active_post_launch();
        }
        if(N_PTs > 0)
        {
            int n = static_cast<int>(N_PTs);
            int grid = (n + block - 1) / block;
            corex_filter::kernel_filter_active_PT_append<<<grid, block>>>(
                n,
                (const Vector2i*)candidate_AllP_AllT_pairs.view().data(),
                (const IndexT*)info.surf_vertices().data(),
                (const Vector3i*)info.surf_triangles().data(),
                (const Vector3*)positions.data(),
                (const Float*)info.thicknesses().data(),
                (const Float*)info.d_hats().data(),
                triangle_thicknesses.data(),
                triangle_d_hats.data(),
                pt_pe_hyst_scale,
                PPs.data(),
                PEs.data(),
                PTs.data(),
                selected_PP_count.data(),
                selected_PE_count.data(),
                selected_PT_count.data());
            corex_filter::corex_filter_active_post_launch();
        }
        if(N_EEs > 0)
        {
            int n = static_cast<int>(N_EEs);
            int grid = (n + block - 1) / block;
            corex_filter::kernel_filter_active_EE_append<<<grid, block>>>(
                n,
                (const Vector2i*)candidate_AllE_AllE_pairs.view().data(),
                (const Vector2i*)info.surf_edges().data(),
                (const Vector3*)positions.data(),
                (const Vector3*)info.rest_positions().data(),
                edge_thicknesses.data(),
                edge_d_hats.data(),
                PPs.data(),
                PEs.data(),
                EEs.data(),
                selected_PP_count.data(),
                selected_PE_count.data(),
                selected_EE_count.data());
            corex_filter::corex_filter_active_post_launch();
        }

        selected_counts.resize(4);
        kernel_pack_selected_counts<<<1, 1>>>(
            selected_PP_count.data(),
            selected_PE_count.data(),
            selected_PT_count.data(),
            selected_EE_count.data(),
            selected_counts.data());

        IndexT h_selected_counts[4] = {0, 0, 0, 0};
        {
            corex_profile::ScopedPhase count_phase("contact_filter_detail",
                                                   "append_count_readback");
            checkCudaErrors(cudaMemcpy(h_selected_counts,
                                       selected_counts.data(),
                                       sizeof(h_selected_counts),
                                       cudaMemcpyDeviceToHost));
        }
        PP_count = h_selected_counts[0];
        PE_count = h_selected_counts[1];
        PT_count = h_selected_counts[2];
        EE_count = h_selected_counts[3];

        if(selected_set_diag)
        {
            spdlog::info("[corex_selected_set] frame={} newton={} "
                         "cand_PP={} cand_CodimPE={} cand_PT={} cand_EE={} "
                         "temp_PP={} temp_PE={} temp_PT={} temp_EE={} "
                         "selected_PP={} selected_PE={} selected_PT={} selected_EE={}",
                         frame,
                         newton_iter,
                         static_cast<int>(N_PCoimP),
                         static_cast<int>(N_CodimPE),
                         static_cast<int>(N_PTs),
                         static_cast<int>(N_EEs),
                         static_cast<int>(PP_capacity),
                         static_cast<int>(PE_capacity),
                         static_cast<int>(PT_capacity),
                         static_cast<int>(EE_capacity),
                         PP_count,
                         PE_count,
                         PT_count,
                         EE_count);
        }

        if(selected_set_hash_diag)
        {
            corex_log_selected_hash(frame,
                                    newton_iter,
                                    PP_count,
                                    PE_count,
                                    PT_count,
                                    EE_count,
                                    PPs,
                                    PEs,
                                    PTs,
                                    EEs);
        }

        if(!view_slice_output)
        {
            PPs.resize(PP_count);
            PEs.resize(PE_count);
            PTs.resize(PT_count);
            EEs.resize(EE_count);
        }

        if(view_slice_output)
        {
            info.PPs(PPs.view(0, PP_count));
            info.PEs(PEs.view(0, PE_count));
            info.PTs(PTs.view(0, PT_count));
            info.EEs(EEs.view(0, EE_count));
        }
        else
        {
            info.PPs(PPs);
            info.PEs(PEs);
            info.PTs(PTs);
            info.EEs(EEs);
        }

        return;
    }

    // PT, EE, PT, PP can degenerate to PP
    if(view_slice_output)
        corex_filter_loose_resize(temp_PPs, N_PCoimP + N_CodimPE + N_PTs + N_EEs);
    else
        temp_PPs.resize(N_PCoimP + N_CodimPE + N_PTs + N_EEs);
    // PT, EE, PT can degenerate to PE
    if(view_slice_output)
        corex_filter_loose_resize(temp_PEs, N_CodimPE + N_PTs + N_EEs);
    else
        temp_PEs.resize(N_CodimPE + N_PTs + N_EEs);

    if(view_slice_output)
    {
        corex_filter_loose_resize(temp_PTs, N_PTs);
        corex_filter_loose_resize(temp_EEs, N_EEs);
    }
    else
    {
        temp_PTs.resize(N_PTs);
        temp_EEs.resize(N_EEs);
    }

    SizeT temp_PP_offset = 0;
    SizeT temp_PE_offset = 0;

    // AllP and CodimP
    if(N_PCoimP > 0)
    {
        corex_profile::ScopedPhase phase("contact_filter_detail", "filter_pp");
        auto PP_view = temp_PPs.view(temp_PP_offset, N_PCoimP);

        {
            int n = (int)candidate_AllP_CodimP_pairs.size();
            int block = 256, grid = (n + block - 1) / block;
            corex_filter::kernel_filter_active_PP<<<grid, block>>>(
                n,
                (const Vector2i*)candidate_AllP_CodimP_pairs.view().data(),
                (const IndexT*)info.surf_vertices().data(),
                (const IndexT*)info.codim_vertices().data(),
                (const Vector3*)positions.data(),
                (const Float*)info.thicknesses().data(),
                (const Float*)info.d_hats().data(),
                PP_view.data());
            corex_filter::corex_filter_active_post_launch();
        }

        temp_PP_offset += N_PCoimP;
    }
    // CodimP and AllE
    if(N_CodimPE > 0)
    {
        corex_profile::ScopedPhase phase("contact_filter_detail", "filter_codimpe");
        auto PP_view = temp_PPs.view(temp_PP_offset, N_CodimPE);
        auto PE_view = temp_PEs.view(temp_PE_offset, N_CodimPE);

        {
            int n = (int)candidate_CodimP_AllE_pairs.size();
            int block = 256, grid = (n + block - 1) / block;
            corex_filter::kernel_filter_active_CodimPE<<<grid, block>>>(
                n,
                (const Vector2i*)candidate_CodimP_AllE_pairs.view().data(),
                (const IndexT*)info.codim_vertices().data(),
                (const Vector2i*)info.surf_edges().data(),
                (const Vector3*)positions.data(),
                (const Float*)info.thicknesses().data(),
                (const Float*)info.d_hats().data(),
                PP_view.data(),
                PE_view.data());
            corex_filter::corex_filter_active_post_launch();
        }

        temp_PP_offset += N_CodimPE;
        temp_PE_offset += N_CodimPE;
    }

    // AllP and AllT
    {
        corex_profile::ScopedPhase phase("contact_filter_detail", "filter_pt");
        auto PP_view = temp_PPs.view(temp_PP_offset, N_PTs);
        auto PE_view = temp_PEs.view(temp_PE_offset, N_PTs);

        {
            int n = (int)candidate_AllP_AllT_pairs.size();
            if(n > 0)
            {
                int block = 256, grid = (n + block - 1) / block;
                corex_filter::kernel_filter_active_PT<<<grid, block>>>(
                    n,
                    (const Vector2i*)candidate_AllP_AllT_pairs.view().data(),
                    (const IndexT*)info.surf_vertices().data(),
                    (const Vector3i*)info.surf_triangles().data(),
                    (const Vector3*)positions.data(),
                    (const Float*)info.thicknesses().data(),
                    (const Float*)info.d_hats().data(),
                    pt_pe_hyst_scale,
                    PP_view.data(),
                    PE_view.data(),
                    temp_PTs.data());
                corex_filter::corex_filter_active_post_launch();
            }
        }

        temp_PP_offset += N_PTs;
        temp_PE_offset += N_PTs;
    }
    // AllE and AllE
    {
        corex_profile::ScopedPhase phase("contact_filter_detail", "filter_ee");
        auto PP_view = temp_PPs.view(temp_PP_offset, N_EEs);
        auto PE_view = temp_PEs.view(temp_PE_offset, N_EEs);

        {
            int n = (int)candidate_AllE_AllE_pairs.size();
            if(n > 0)
            {
                int block = 256, grid = (n + block - 1) / block;
                corex_filter::kernel_filter_active_EE<<<grid, block>>>(
                    n,
                    (const Vector2i*)candidate_AllE_AllE_pairs.view().data(),
                    (const Vector2i*)info.surf_edges().data(),
                    (const Vector3*)positions.data(),
                    (const Vector3*)info.rest_positions().data(),
                    (const Float*)info.thicknesses().data(),
                    (const Float*)info.d_hats().data(),
                    PP_view.data(),
                    PE_view.data(),
                    temp_EEs.data());
                corex_filter::corex_filter_active_post_launch();
            }
        }

        temp_PP_offset += N_EEs;
        temp_PE_offset += N_EEs;
    }

    UIPC_ASSERT(temp_PP_offset == temp_PPs.size(), "size mismatch");
    UIPC_ASSERT(temp_PE_offset == temp_PEs.size(), "size mismatch");

    if(trace_filter_active_diag)
    {
        static int filter_active_diag_call = 0;
        bool do_diag_log = (filter_active_diag_call < 3) || (filter_active_diag_call % 50 == 0);
        if((N_PTs > 0 || N_EEs > 0) && do_diag_log)
        {
            cudaDeviceSynchronize();
            int n_temp_pt = (int)temp_PTs.size();
            int n_temp_ee = (int)temp_EEs.size();
            int n_temp_pp = (int)temp_PPs.size();
            int n_temp_pe = (int)temp_PEs.size();
            std::vector<Vector4i> h_tpt(n_temp_pt);
            std::vector<Vector4i> h_tee(n_temp_ee);
            std::vector<Vector3i> h_tpe(n_temp_pe);
            if(n_temp_pt > 0)
                cudaMemcpy(h_tpt.data(), temp_PTs.data(), n_temp_pt * sizeof(Vector4i), cudaMemcpyDeviceToHost);
            if(n_temp_ee > 0)
                cudaMemcpy(h_tee.data(), temp_EEs.data(), n_temp_ee * sizeof(Vector4i), cudaMemcpyDeviceToHost);
            if(n_temp_pe > 0)
                cudaMemcpy(h_tpe.data(), temp_PEs.data(), n_temp_pe * sizeof(Vector3i), cudaMemcpyDeviceToHost);
            int pt_valid = 0, ee_valid = 0, pe_valid = 0;
            for(int i = 0; i < n_temp_pt; ++i)
                if(h_tpt[i](0) != -1) pt_valid++;
            for(int i = 0; i < n_temp_ee; ++i)
                if(h_tee[i](0) != -1) ee_valid++;
            for(int i = 0; i < n_temp_pe; ++i)
                if(h_tpe[i](0) != -1) pe_valid++;
            spdlog::info("[corex_trace][filter_active] temp_PT={} valid_PT={} temp_EE={} valid_EE={} temp_PP={} temp_PE={}",
                         n_temp_pt, pt_valid, n_temp_ee, ee_valid, n_temp_pp, n_temp_pe);
            for(int i = 0; i < std::min(n_temp_pt, 3); ++i)
                spdlog::info("[corex_trace][filter_active] temp_PT[{}] = ({},{},{},{})",
                             i, h_tpt[i](0), h_tpt[i](1), h_tpt[i](2), h_tpt[i](3));

            // Verify PT distance on host for first few pairs
            if(N_PTs > 0)
            {
                static int fa_host_diag = 0;
                const bool force_pt_drop_diag = (pt_valid == 0);
                if(fa_host_diag < 5 || force_pt_drop_diag)
                {
                    int nv = (int)positions.size();
                    std::vector<Vector3> h_pos(nv);
                    std::vector<Float> h_thick(nv), h_dhat(nv);
                    cudaMemcpy(h_pos.data(), positions.data(), nv * sizeof(Vector3), cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_thick.data(), info.thicknesses().data(), nv * sizeof(Float), cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_dhat.data(), info.d_hats().data(), nv * sizeof(Float), cudaMemcpyDeviceToHost);
                    std::vector<Vector2i> h_pt_pairs(N_PTs);
                    std::vector<IndexT> h_sverts(info.surf_vertices().size());
                    std::vector<Vector3i> h_stris(info.surf_triangles().size());
                    cudaMemcpy(h_pt_pairs.data(), candidate_AllP_AllT_pairs.view().data(), N_PTs * sizeof(Vector2i), cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_sverts.data(), info.surf_vertices().data(), h_sverts.size() * sizeof(IndexT), cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_stris.data(), info.surf_triangles().data(), h_stris.size() * sizeof(Vector3i), cudaMemcpyDeviceToHost);
                    spdlog::info("[corex_trace][fa_host_PT] mode={} N_PTs={} valid_PT={}",
                                 force_pt_drop_diag ? "pt_drop" : "warmup",
                                 (int)N_PTs,
                                 pt_valid);
                    for(int i = 0; i < std::min((int)N_PTs, 8); ++i)
                    {
                        auto indices = h_pt_pairs[i];
                        IndexT V = h_sverts[indices(0)];
                        Vector3i F = h_stris[indices(1)];
                        Vector4i vIs = {V, F(0), F(1), F(2)};
                        Vector3 Ps_arr[] = {h_pos[vIs(0)], h_pos[vIs(1)], h_pos[vIs(2)], h_pos[vIs(3)]};
                        Float thickness = PT_thickness(h_thick[V], h_thick[F(0)], h_thick[F(1)], h_thick[F(2)]);
                        Float d_hat = PT_d_hat(h_dhat[V], h_dhat[F(0)], h_dhat[F(1)], h_dhat[F(2)]);
                        Vector4i flag = distance::point_triangle_distance_flag(Ps_arr[0], Ps_arr[1], Ps_arr[2], Ps_arr[3]);
                        Vector2 range = D_range(thickness, d_hat);
                        Float D;
                        distance::point_triangle_distance2(flag, Ps_arr[0], Ps_arr[1], Ps_arr[2], Ps_arr[3], D);
                        Vector4i offsets;
                        offsets.setConstant(-1);
                        auto dim = distance::degenerate_point_triangle(flag, offsets);
                        bool active = (D > range.x() && D < range.y());
                        bool low = (D <= range.x());
                        bool high = (D >= range.y());
                        spdlog::info("[corex_trace][fa_host_PT] i={} V={} F=({},{},{}) P0=({},{},{}) D={} range=({},{}) active={} low={} high={} dim={} offsets=({},{},{},{}) flag=({},{},{},{})",
                                     i, V, F(0), F(1), F(2),
                                     Ps_arr[0][0], Ps_arr[0][1], Ps_arr[0][2],
                                     D, range.x(), range.y(), (int)active, (int)low, (int)high, (int)dim,
                                     offsets(0), offsets(1), offsets(2), offsets(3),
                                     flag(0), flag(1), flag(2), flag(3));
                    }
                    if(!force_pt_drop_diag)
                        fa_host_diag++;
                }
            }

            if(ee_valid > 0 && pe_valid == 0 && N_EEs > 0)
            {
                static int ee_host_diag = 0;
                if(ee_host_diag < 6)
                {
                    int nv = (int)positions.size();
                    std::vector<Vector3> h_pos(nv);
                    std::vector<Vector3> h_rest(nv);
                    std::vector<Float> h_thick(nv), h_dhat(nv);
                    std::vector<Vector2i> h_ee_pairs(N_EEs);
                    std::vector<Vector2i> h_sedges(info.surf_edges().size());
                    cudaMemcpy(h_pos.data(), positions.data(), nv * sizeof(Vector3), cudaMemcpyDeviceToHost);
                    cudaMemcpy(
                        h_rest.data(), info.rest_positions().data(), nv * sizeof(Vector3), cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_thick.data(), info.thicknesses().data(), nv * sizeof(Float), cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_dhat.data(), info.d_hats().data(), nv * sizeof(Float), cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_ee_pairs.data(),
                               candidate_AllE_AllE_pairs.view().data(),
                               N_EEs * sizeof(Vector2i),
                               cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_sedges.data(),
                               info.surf_edges().data(),
                               h_sedges.size() * sizeof(Vector2i),
                               cudaMemcpyDeviceToHost);
                    spdlog::info("[corex_trace][fa_host_EE] mode=ee_only_window N_EEs={} valid_EE={} valid_PE={}",
                                 (int)N_EEs,
                                 ee_valid,
                                 pe_valid);
                    for(int i = 0; i < std::min((int)N_EEs, 8); ++i)
                    {
                        auto pair = h_ee_pairs[i];
                        Vector2i e0 = h_sedges[pair(0)];
                        Vector2i e1 = h_sedges[pair(1)];
                        Vector4i vIs = {e0(0), e0(1), e1(0), e1(1)};
                        Vector3 Ps_arr[] = {h_pos[vIs(0)], h_pos[vIs(1)], h_pos[vIs(2)], h_pos[vIs(3)]};
                        Vector3 Rs_arr[] = {h_rest[vIs(0)], h_rest[vIs(1)], h_rest[vIs(2)], h_rest[vIs(3)]};
                        Float thickness = EE_thickness(
                            h_thick[vIs(0)], h_thick[vIs(1)], h_thick[vIs(2)], h_thick[vIs(3)]);
                        Float d_hat = EE_d_hat(h_dhat[vIs(0)], h_dhat[vIs(1)], h_dhat[vIs(2)], h_dhat[vIs(3)]);
                        Vector2 range = D_range(thickness, d_hat);
                        Vector4i flag = distance::edge_edge_distance_flag(
                            Ps_arr[0], Ps_arr[1], Ps_arr[2], Ps_arr[3]);
                        Float D;
                        distance::edge_edge_distance2(
                            flag, Ps_arr[0], Ps_arr[1], Ps_arr[2], Ps_arr[3], D);
                        Vector4i offsets;
                        offsets.setConstant(-1);
                        auto dim = distance::degenerate_edge_edge(flag, offsets);
                        Float eps_x;
                        distance::edge_edge_mollifier_threshold(Rs_arr[0],
                                                                Rs_arr[1],
                                                                Rs_arr[2],
                                                                Rs_arr[3],
                                                                static_cast<Float>(1e-3),
                                                                eps_x);
                        bool need_mollify =
                            distance::need_mollify(Ps_arr[0], Ps_arr[1], Ps_arr[2], Ps_arr[3], eps_x);
                        bool low = (D <= range.x());
                        bool high = (D >= range.y());
                        spdlog::info(
                            "[corex_trace][fa_host_EE] i={} e0=({},{}) e1=({},{}) D={} range=({},{}) "
                            "low={} high={} dim={} need_mollify={} eps_x={} offsets=({},{},{},{}) flag=({},{},{},{})",
                            i,
                            e0(0),
                            e0(1),
                            e1(0),
                            e1(1),
                            D,
                            range.x(),
                            range.y(),
                            (int)low,
                            (int)high,
                            (int)dim,
                            (int)need_mollify,
                            eps_x,
                            offsets(0),
                            offsets(1),
                            offsets(2),
                            offsets(3),
                            flag(0),
                            flag(1),
                            flag(2),
                            flag(3));
                    }
                    ee_host_diag++;
                }
            }
        }
        filter_active_diag_call++;
    }

    IndexT PP_count = 0;
    IndexT PE_count = 0;
    IndexT PT_count = 0;
    IndexT EE_count = 0;

    {  // select the valid ones
        corex_profile::ScopedPhase phase("contact_filter_detail", "select_valid_all");
        if(view_slice_output)
        {
            corex_filter_loose_resize(PPs, temp_PPs.size());
            corex_filter_loose_resize(PEs, temp_PEs.size());
            corex_filter_loose_resize(PTs, temp_PTs.size());
            corex_filter_loose_resize(EEs, temp_EEs.size());
        }
        else
        {
            PPs.resize(temp_PPs.size());
            PEs.resize(temp_PEs.size());
            PTs.resize(temp_PTs.size());
            EEs.resize(temp_EEs.size());
        }

        if(temp_PPs.size())
        {
            DeviceSelect().If(temp_PPs.data(),
                              PPs.data(),
                              selected_PP_count.data(),
                              temp_PPs.size(),
                              [] CUB_RUNTIME_FUNCTION(const Vector2i& PP)
                              { return PP(0) != -1; });
        }
        else
        {
            checkCudaErrors(cudaMemsetAsync(selected_PP_count.data(), 0, sizeof(IndexT)));
        }

        if(temp_PEs.size())
        {
            DeviceSelect().If(temp_PEs.data(),
                              PEs.data(),
                              selected_PE_count.data(),
                              temp_PEs.size(),
                              [] CUB_RUNTIME_FUNCTION(const Vector3i& PE)
                              { return PE(0) != -1; });
        }
        else
        {
            checkCudaErrors(cudaMemsetAsync(selected_PE_count.data(), 0, sizeof(IndexT)));
        }

        if(temp_PTs.size())
        {
            DeviceSelect().If(temp_PTs.data(),
                              PTs.data(),
                              selected_PT_count.data(),
                              temp_PTs.size(),
                              [] CUB_RUNTIME_FUNCTION(const Vector4i& PT)
                              { return PT(0) != -1; });
        }
        else
        {
            checkCudaErrors(cudaMemsetAsync(selected_PT_count.data(), 0, sizeof(IndexT)));
        }

        if(temp_EEs.size())
        {
            DeviceSelect().If(temp_EEs.data(),
                              EEs.data(),
                              selected_EE_count.data(),
                              temp_EEs.size(),
                              [] CUB_RUNTIME_FUNCTION(const Vector4i& EE)
                              { return EE(0) != -1; });
        }
        else
        {
            checkCudaErrors(cudaMemsetAsync(selected_EE_count.data(), 0, sizeof(IndexT)));
        }

        selected_counts.resize(4);
        kernel_pack_selected_counts<<<1, 1>>>(
            selected_PP_count.data(),
            selected_PE_count.data(),
            selected_PT_count.data(),
            selected_EE_count.data(),
            selected_counts.data());

        IndexT h_selected_counts[4] = {0, 0, 0, 0};
        {
            corex_profile::ScopedPhase count_phase("contact_filter_detail",
                                                   "select_count_readback");
            checkCudaErrors(cudaMemcpy(h_selected_counts,
                                       selected_counts.data(),
                                       sizeof(h_selected_counts),
                                       cudaMemcpyDeviceToHost));
        }
        PP_count = h_selected_counts[0];
        PE_count = h_selected_counts[1];
        PT_count = h_selected_counts[2];
        EE_count = h_selected_counts[3];

        if(selected_set_diag)
        {
            spdlog::info("[corex_selected_set] frame={} newton={} "
                         "cand_PP={} cand_CodimPE={} cand_PT={} cand_EE={} "
                         "temp_PP={} temp_PE={} temp_PT={} temp_EE={} "
                         "selected_PP={} selected_PE={} selected_PT={} selected_EE={}",
                         frame,
                         newton_iter,
                         static_cast<int>(N_PCoimP),
                         static_cast<int>(N_CodimPE),
                         static_cast<int>(N_PTs),
                         static_cast<int>(N_EEs),
                         static_cast<int>(temp_PPs.size()),
                         static_cast<int>(temp_PEs.size()),
                         static_cast<int>(temp_PTs.size()),
                         static_cast<int>(temp_EEs.size()),
                         PP_count,
                         PE_count,
                         PT_count,
                         EE_count);
        }

        if(selected_set_hash_diag)
        {
            corex_log_selected_hash(frame,
                                    newton_iter,
                                    PP_count,
                                    PE_count,
                                    PT_count,
                                    EE_count,
                                    PPs,
                                    PEs,
                                    PTs,
                                    EEs);
        }

        if(trace_filter_active_diag)
        {
            static int after_select_log_call = 0;
            if(after_select_log_call < 10 || (after_select_log_call % 50 == 0))
                spdlog::info("[corex_trace][filter_active] AFTER_SELECT PP={} PE={} PT={} EE={}",
                             PP_count, PE_count, PT_count, EE_count);
            after_select_log_call++;

            static bool first_contact_contract_dumped = false;
            if(!first_contact_contract_dumped && (PE_count > 0 || PT_count > 0 || EE_count > 0))
            {
                cudaDeviceSynchronize();

                const int nv = static_cast<int>(positions.size());
                std::vector<Vector3> h_pos(nv);
                std::vector<Float> h_thick(nv);
                std::vector<Float> h_dhat(nv);
                if(nv > 0)
                {
                    cudaMemcpy(h_pos.data(), positions.data(), nv * sizeof(Vector3), cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_thick.data(),
                               info.thicknesses().data(),
                               nv * sizeof(Float),
                               cudaMemcpyDeviceToHost);
                    cudaMemcpy(
                        h_dhat.data(), info.d_hats().data(), nv * sizeof(Float), cudaMemcpyDeviceToHost);
                }

                std::vector<IndexT> h_codim_vs(info.codim_vertices().size());
                std::vector<IndexT> h_surf_vs(info.surf_vertices().size());
                std::vector<Vector2i> h_sedges(info.surf_edges().size());
                std::vector<Vector3i> h_stris(info.surf_triangles().size());

                if(!h_codim_vs.empty())
                    cudaMemcpy(h_codim_vs.data(),
                               info.codim_vertices().data(),
                               h_codim_vs.size() * sizeof(IndexT),
                               cudaMemcpyDeviceToHost);
                if(!h_surf_vs.empty())
                    cudaMemcpy(h_surf_vs.data(),
                               info.surf_vertices().data(),
                               h_surf_vs.size() * sizeof(IndexT),
                               cudaMemcpyDeviceToHost);
                if(!h_sedges.empty())
                    cudaMemcpy(h_sedges.data(),
                               info.surf_edges().data(),
                               h_sedges.size() * sizeof(Vector2i),
                               cudaMemcpyDeviceToHost);
                if(!h_stris.empty())
                    cudaMemcpy(h_stris.data(),
                               info.surf_triangles().data(),
                               h_stris.size() * sizeof(Vector3i),
                               cudaMemcpyDeviceToHost);

                std::vector<Vector2i> h_pe_pairs(N_CodimPE);
                std::vector<Vector2i> h_pt_pairs(N_PTs);
                if(N_CodimPE > 0)
                    cudaMemcpy(h_pe_pairs.data(),
                               candidate_CodimP_AllE_pairs.view().data(),
                               N_CodimPE * sizeof(Vector2i),
                               cudaMemcpyDeviceToHost);
                if(N_PTs > 0)
                    cudaMemcpy(h_pt_pairs.data(),
                               candidate_AllP_AllT_pairs.view().data(),
                               N_PTs * sizeof(Vector2i),
                               cudaMemcpyDeviceToHost);

                spdlog::info(
                    "[corex_trace][first_contact_contract] selected PP={} PE={} PT={} EE={} candPE={} candPT={}",
                    PP_count,
                    PE_count,
                    PT_count,
                    EE_count,
                    static_cast<int>(N_CodimPE),
                    static_cast<int>(N_PTs));

                for(int i = 0; i < std::min(static_cast<int>(N_CodimPE), 8); ++i)
                {
                    auto pair = h_pe_pairs[i];
                    if(pair(0) < 0 || pair(1) < 0 || pair(0) >= static_cast<IndexT>(h_codim_vs.size())
                       || pair(1) >= static_cast<IndexT>(h_sedges.size()))
                        continue;

                    IndexT V = h_codim_vs[pair(0)];
                    auto   E = h_sedges[pair(1)];
                    if(V < 0 || E(0) < 0 || E(1) < 0 || V >= nv || E(0) >= nv || E(1) >= nv)
                        continue;

                    Vector3 p = h_pos[V];
                    Vector3 e0 = h_pos[E(0)];
                    Vector3 e1 = h_pos[E(1)];

                    const Float raw_t_p = h_thick[V];
                    const Float raw_t_e0 = h_thick[E(0)];
                    const Float raw_t_e1 = h_thick[E(1)];
                    const Float raw_d_p = h_dhat[V];
                    const Float raw_d_e0 = h_dhat[E(0)];
                    const Float raw_d_e1 = h_dhat[E(1)];

                    Float thickness = PE_thickness(raw_t_p, raw_t_e0, raw_t_e1);
                    Float d_hat = PE_d_hat(raw_d_p, raw_d_e0, raw_d_e1);
                    auto  flag = distance::point_edge_distance_flag(p, e0, e1);
                    auto  range = D_range(thickness, d_hat);
                    Float D;
                    distance::point_edge_distance2(flag, p, e0, e1, D);
                    Vector3i offsets;
                    offsets.setConstant(-1);
                    auto     dim = distance::degenerate_point_edge(flag, offsets);
                    bool     active = is_active_D(range, D);

                    spdlog::info(
                        "[corex_trace][first_contact_contract][PE] i={} pair=({},{}) V/E=({},{}:{}) "
                        "raw_t=({},{},{}) raw_d=({},{},{}) flag=({},{},{}) dim={} offsets=({},{},{}) "
                        "D={} range=({},{}) active={}",
                        i,
                        pair(0),
                        pair(1),
                        V,
                        E(0),
                        E(1),
                        raw_t_p,
                        raw_t_e0,
                        raw_t_e1,
                        raw_d_p,
                        raw_d_e0,
                        raw_d_e1,
                        flag(0),
                        flag(1),
                        flag(2),
                        dim,
                        offsets(0),
                        offsets(1),
                        offsets(2),
                        D,
                        range.x(),
                        range.y(),
                        static_cast<int>(active));
                }

                for(int i = 0; i < std::min(static_cast<int>(N_PTs), 8); ++i)
                {
                    auto pair = h_pt_pairs[i];
                    if(pair(0) < 0 || pair(1) < 0 || pair(0) >= static_cast<IndexT>(h_surf_vs.size())
                       || pair(1) >= static_cast<IndexT>(h_stris.size()))
                        continue;

                    IndexT V = h_surf_vs[pair(0)];
                    auto   F = h_stris[pair(1)];
                    if(V < 0 || F(0) < 0 || F(1) < 0 || F(2) < 0 || V >= nv || F(0) >= nv
                       || F(1) >= nv || F(2) >= nv)
                        continue;

                    Vector3 p = h_pos[V];
                    Vector3 t0 = h_pos[F(0)];
                    Vector3 t1 = h_pos[F(1)];
                    Vector3 t2 = h_pos[F(2)];

                    const Float raw_t_p = h_thick[V];
                    const Float raw_t_t0 = h_thick[F(0)];
                    const Float raw_t_t1 = h_thick[F(1)];
                    const Float raw_t_t2 = h_thick[F(2)];
                    const Float raw_d_p = h_dhat[V];
                    const Float raw_d_t0 = h_dhat[F(0)];
                    const Float raw_d_t1 = h_dhat[F(1)];
                    const Float raw_d_t2 = h_dhat[F(2)];

                    Float thickness = PT_thickness(raw_t_p, raw_t_t0, raw_t_t1, raw_t_t2);
                    Float d_hat = PT_d_hat(raw_d_p, raw_d_t0, raw_d_t1, raw_d_t2);
                    auto  flag = distance::point_triangle_distance_flag(p, t0, t1, t2);
                    auto  range = D_range(thickness, d_hat);
                    Float D;
                    distance::point_triangle_distance2(flag, p, t0, t1, t2, D);
                    Vector4i offsets;
                    offsets.setConstant(-1);
                    auto     dim = distance::degenerate_point_triangle(flag, offsets);
                    bool     active = is_active_D(range, D);

                    spdlog::info(
                        "[corex_trace][first_contact_contract][PT] i={} pair=({},{}) V/F=({},{}:{}:{}) "
                        "raw_t=({},{},{},{}) raw_d=({},{},{},{}) flag=({},{},{},{}) dim={} "
                        "offsets=({},{},{},{}) D={} range=({},{}) active={}",
                        i,
                        pair(0),
                        pair(1),
                        V,
                        F(0),
                        F(1),
                        F(2),
                        raw_t_p,
                        raw_t_t0,
                        raw_t_t1,
                        raw_t_t2,
                        raw_d_p,
                        raw_d_t0,
                        raw_d_t1,
                        raw_d_t2,
                        flag(0),
                        flag(1),
                        flag(2),
                        flag(3),
                        dim,
                        offsets(0),
                        offsets(1),
                        offsets(2),
                        offsets(3),
                        D,
                        range.x(),
                        range.y(),
                        static_cast<int>(active));
                }

                if(PE_count > 0)
                {
                    std::vector<Vector3i> h_selected_pes(PE_count);
                    cudaMemcpy(h_selected_pes.data(),
                               PEs.data(),
                               PE_count * sizeof(Vector3i),
                               cudaMemcpyDeviceToHost);

                    for(int i = 0; i < std::min(static_cast<int>(PE_count), 8); ++i)
                    {
                        auto pe = h_selected_pes[i];
                        if(pe(0) < 0 || pe(1) < 0 || pe(2) < 0 || pe(0) >= nv || pe(1) >= nv
                           || pe(2) >= nv)
                            continue;

                        Vector3 p = h_pos[pe(0)];
                        Vector3 e0 = h_pos[pe(1)];
                        Vector3 e1 = h_pos[pe(2)];
                        const Float raw_t_p = h_thick[pe(0)];
                        const Float raw_t_e0 = h_thick[pe(1)];
                        const Float raw_t_e1 = h_thick[pe(2)];
                        const Float raw_d_p = h_dhat[pe(0)];
                        const Float raw_d_e0 = h_dhat[pe(1)];
                        const Float raw_d_e1 = h_dhat[pe(2)];

                        Float thickness = PE_thickness(raw_t_p, raw_t_e0, raw_t_e1);
                        Float d_hat = PE_d_hat(raw_d_p, raw_d_e0, raw_d_e1);
                        auto  flag = distance::point_edge_distance_flag(p, e0, e1);
                        auto  range = D_range(thickness, d_hat);
                        Float D;
                        distance::point_edge_distance2(flag, p, e0, e1, D);
                        Vector3i offsets;
                        offsets.setConstant(-1);
                        auto dim = distance::degenerate_point_edge(flag, offsets);
                        bool active = is_active_D(range, D);

                        spdlog::info(
                            "[corex_trace][first_contact_contract][PE_selected] i={} pe=({},{},{}) "
                            "raw_t=({},{},{}) raw_d=({},{},{}) flag=({},{},{}) dim={} "
                            "offsets=({},{},{}) D={} range=({},{}) active={}",
                            i,
                            pe(0),
                            pe(1),
                            pe(2),
                            raw_t_p,
                            raw_t_e0,
                            raw_t_e1,
                            raw_d_p,
                            raw_d_e0,
                            raw_d_e1,
                            flag(0),
                            flag(1),
                            flag(2),
                            dim,
                            offsets(0),
                            offsets(1),
                            offsets(2),
                            D,
                            range.x(),
                            range.y(),
                            static_cast<int>(active));
                    }
                }

                first_contact_contract_dumped = true;
            }
        }

        if(!view_slice_output)
        {
            PPs.resize(PP_count);
            PEs.resize(PE_count);
            PTs.resize(PT_count);
            EEs.resize(EE_count);
        }
    }

    if(view_slice_output)
    {
        info.PPs(PPs.view(0, PP_count));
        info.PEs(PEs.view(0, PE_count));
        info.PTs(PTs.view(0, PT_count));
        info.EEs(EEs.view(0, EE_count));
    }
    else
    {
        info.PPs(PPs);
        info.PEs(PEs);
        info.PTs(PTs);
        info.EEs(EEs);
    }

    if constexpr(PrintDebugInfo)
    {
        std::vector<Vector2i> PPs_host;
        std::vector<Float>    PP_thicknesses_host;

        std::vector<Vector3i> PEs_host;
        std::vector<Float>    PE_thicknesses_host;

        std::vector<Vector4i> PTs_host;
        std::vector<Float>    PT_thicknesses_host;

        std::vector<Vector4i> EEs_host;
        std::vector<Float>    EE_thicknesses_host;

        PPs.copy_to(PPs_host);
        PEs.copy_to(PEs_host);
        PTs.copy_to(PTs_host);
        EEs.copy_to(EEs_host);

        std::cout << "filter result:" << std::endl;

        for(auto&& [PP, thickness] : zip(PPs_host, PP_thicknesses_host))
        {
            std::cout << "PP: " << PP.transpose() << " thickness: " << thickness << "\n";
        }

        for(auto&& [PE, thickness] : zip(PEs_host, PE_thicknesses_host))
        {
            std::cout << "PE: " << PE.transpose() << " thickness: " << thickness << "\n";
        }

        for(auto&& [PT, thickness] : zip(PTs_host, PT_thicknesses_host))
        {
            std::cout << "PT: " << PT.transpose() << " thickness: " << thickness << "\n";
        }

        for(auto&& [EE, thickness] : zip(EEs_host, EE_thicknesses_host))
        {
            std::cout << "EE: " << EE.transpose() << " thickness: " << thickness << "\n";
        }

        std::cout << std::flush;
    }
}

void StacklessBVHSimplexTrajectoryFilter::Impl::filter_toi(FilterTOIInfo& info)
{
    using namespace muda;

    auto toi_size = candidate_AllP_CodimP_pairs.size() + candidate_CodimP_AllE_pairs.size()
                    + candidate_AllP_AllT_pairs.size() + candidate_AllE_AllE_pairs.size();

    tois.resize(toi_size);

    auto offset  = 0;
    auto PP_tois = tois.view(offset, candidate_AllP_CodimP_pairs.size());
    offset += candidate_AllP_CodimP_pairs.size();
    auto PE_codim_tois = tois.view(offset, candidate_CodimP_AllE_pairs.size());
    offset += candidate_CodimP_AllE_pairs.size();
    auto PT_tois = tois.view(offset, candidate_AllP_AllT_pairs.size());
    offset += candidate_AllP_AllT_pairs.size();
    auto EE_tois = tois.view(offset, candidate_AllE_AllE_pairs.size());
    offset += candidate_AllE_AllE_pairs.size();

    UIPC_ASSERT(offset == toi_size, "size mismatch");

    constexpr Float eta = 0.1;
    constexpr SizeT max_iter = 1000;
    constexpr Float large_enough_toi = 1.1;

    // CoreX: same CCD formulas as the NVIDIA ParallelFor path, but explicit __global__
    // kernels in corex_filter (ParallelFor device lambdas are unreliable on CoreX).
    if(toi_size > 0)
    {
        constexpr int block_dim = 256;
        SizeT         toi_seg = 0;
        Float*        d_tois    = tois.data();

        const int n_pp = static_cast<int>(candidate_AllP_CodimP_pairs.size());
        if(n_pp > 0)
        {
            const int grid = (n_pp + block_dim - 1) / block_dim;
            corex_filter::kernel_filter_toi_PP<<<grid, block_dim>>>(
                n_pp,
                candidate_AllP_CodimP_pairs.view().data(),
                info.codim_vertices().data(),
                info.surf_vertices().data(),
                info.thicknesses().data(),
                info.positions().data(),
                info.displacements().data(),
                info.d_hats().data(),
                info.alpha(),
                eta,
                max_iter,
                large_enough_toi,
                d_tois + toi_seg);
        }
        toi_seg += static_cast<SizeT>(n_pp);

        const int n_pe_codim = static_cast<int>(candidate_CodimP_AllE_pairs.size());
        if(n_pe_codim > 0)
        {
            const int grid = (n_pe_codim + block_dim - 1) / block_dim;
            corex_filter::kernel_filter_toi_PE<<<grid, block_dim>>>(
                n_pe_codim,
                candidate_CodimP_AllE_pairs.view().data(),
                info.codim_vertices().data(),
                info.surf_edges().data(),
                info.thicknesses().data(),
                info.positions().data(),
                info.displacements().data(),
                info.d_hats().data(),
                info.alpha(),
                eta,
                max_iter,
                large_enough_toi,
                d_tois + toi_seg);
        }
        toi_seg += static_cast<SizeT>(n_pe_codim);

        const int n_pt = static_cast<int>(candidate_AllP_AllT_pairs.size());
        if(n_pt > 0)
        {
            const int grid = (n_pt + block_dim - 1) / block_dim;
            corex_filter::kernel_filter_toi_PT<<<grid, block_dim>>>(
                n_pt,
                candidate_AllP_AllT_pairs.view().data(),
                info.surf_vertices().data(),
                info.surf_triangles().data(),
                info.thicknesses().data(),
                info.positions().data(),
                info.displacements().data(),
                info.d_hats().data(),
                info.alpha(),
                eta,
                max_iter,
                large_enough_toi,
                d_tois + toi_seg);
        }
        toi_seg += static_cast<SizeT>(n_pt);

        const int n_ee = static_cast<int>(candidate_AllE_AllE_pairs.size());
        if(n_ee > 0)
        {
            const int grid = (n_ee + block_dim - 1) / block_dim;
            corex_filter::kernel_filter_toi_EE<<<grid, block_dim>>>(
                n_ee,
                candidate_AllE_AllE_pairs.view().data(),
                info.surf_edges().data(),
                info.thicknesses().data(),
                info.positions().data(),
                info.displacements().data(),
                info.d_hats().data(),
                info.alpha(),
                eta,
                max_iter,
                large_enough_toi,
                d_tois + toi_seg);
        }

        UIPC_ASSERT(static_cast<SizeT>(n_pp + n_pe_codim + n_pt + n_ee)
                        == static_cast<SizeT>(toi_size),
                    "filter_toi segment size mismatch");

        DeviceReduce().Min(tois.data(), info.toi().data(), tois.size());
    }
    else
    {
        info.toi().fill(large_enough_toi);
    }
}
}  // namespace uipc::backend::cuda
#else
#include <collision_detection/filters/stackless_bvh_simplex_trajectory_filter.h>
#include <muda/cub/device/device_select.h>
#include <muda/ext/eigen/log_proxy.h>
#include <sim_engine.h>
#include <kernel_cout.h>
#include <utils/distance/distance_flagged.h>
#include <utils/distance.h>
#include <utils/codim_thickness.h>
#include <utils/simplex_contact_mask_utils.h>
#include <uipc/common/zip.h>
#include <utils/primitive_d_hat.h>
#include <cstdio>

namespace uipc::backend::cuda
{
constexpr bool PrintDebugInfo = false;
constexpr bool PrintKernelZeroDistance = false;

REGISTER_SIM_SYSTEM(StacklessBVHSimplexTrajectoryFilter);

void StacklessBVHSimplexTrajectoryFilter::do_build(BuildInfo& info)
{
    auto& config = world().scene().config();
    auto  method = config.find<std::string>("collision_detection/method");
    if(method->view()[0] != "stackless_bvh")
    {
        throw SimSystemException("Stackless BVH unused");
    }
}

void StacklessBVHSimplexTrajectoryFilter::do_detect(DetectInfo& info)
{
    m_impl.detect(info);
}

void StacklessBVHSimplexTrajectoryFilter::do_filter_active(FilterActiveInfo& info)
{
    m_impl.filter_active(info);
}

void StacklessBVHSimplexTrajectoryFilter::do_filter_toi(FilterTOIInfo& info)
{
    m_impl.filter_toi(info);
}

muda::CBufferView<Vector2i> StacklessBVHSimplexTrajectoryFilter::candidate_PTs() const noexcept
{
    return m_impl.candidate_AllP_AllT_pairs.view();
}

muda::CBufferView<Vector2i> StacklessBVHSimplexTrajectoryFilter::candidate_EEs() const noexcept
{
    return m_impl.candidate_AllE_AllE_pairs.view();
}

muda::CBufferView<Float> StacklessBVHSimplexTrajectoryFilter::toi_PTs() const noexcept
{
    auto pp_size = m_impl.candidate_AllP_CodimP_pairs.size();
    auto pe_size = m_impl.candidate_CodimP_AllE_pairs.size();
    auto pt_size = m_impl.candidate_AllP_AllT_pairs.size();
    return m_impl.tois.view(pp_size + pe_size, pt_size);
}

muda::CBufferView<Float> StacklessBVHSimplexTrajectoryFilter::toi_EEs() const noexcept
{
    auto pp_size = m_impl.candidate_AllP_CodimP_pairs.size();
    auto pe_size = m_impl.candidate_CodimP_AllE_pairs.size();
    auto pt_size = m_impl.candidate_AllP_AllT_pairs.size();
    auto ee_size = m_impl.candidate_AllE_AllE_pairs.size();
    return m_impl.tois.view(pp_size + pe_size + pt_size, ee_size);
}

void StacklessBVHSimplexTrajectoryFilter::Impl::detect(DetectInfo& info)
{
    using namespace muda;

    auto alpha   = info.alpha();
    auto Ps      = info.positions();
    auto dxs     = info.displacements();
    auto codimVs = info.codim_vertices();
    auto Vs      = info.surf_vertices();
    auto Es      = info.surf_edges();
    auto Fs      = info.surf_triangles();

    //lbvh_E      = {};
    //lbvh_T      = {};
    //lbvh_CodimP = {};

    point_aabbs.resize(Vs.size());
    triangle_aabbs.resize(Fs.size());
    edge_aabbs.resize(Es.size());

    // build AABBs for codim vertices
    if(codimVs.size() > 0)
    {
        codim_point_aabbs.resize(codimVs.size());

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(codimVs.size(),
                   [codimVs = codimVs.viewer().name("codimVs"),
                    Ps      = Ps.viewer().name("Ps"),
                    dxs     = dxs.viewer().name("dxs"),
                    aabbs   = codim_point_aabbs.viewer().name("aabbs"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    d_hats = info.d_hats().viewer().name("d_hats"),
                    alpha  = alpha] __device__(int i) mutable
                   {
                       auto vI = codimVs(i);

                       Float thickness       = thicknesses(vI);
                       Float d_hat_expansion = point_dcd_expansion(d_hats(vI));

                       const auto& pos   = Ps(vI);
                       Vector3     pos_t = pos + dxs(vI) * alpha;

                       AABB aabb;
                       aabb.extend(pos.cast<float>()).extend(pos_t.cast<float>());

                       float expand = d_hat_expansion + thickness;

                       aabb.min().array() -= expand;
                       aabb.max().array() += expand;
                       aabbs(i) = aabb;
                   });
    }

    // build AABBs for surf vertices (including codim vertices)
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(Vs.size(),
               [Vs          = Vs.viewer().name("V"),
                dxs         = dxs.viewer().name("dx"),
                Ps          = Ps.viewer().name("Ps"),
                aabbs       = point_aabbs.viewer().name("aabbs"),
                thicknesses = info.thicknesses().viewer().name("thicknesses"),
                d_hats      = info.d_hats().viewer().name("d_hats"),
                alpha       = alpha] __device__(int i) mutable
               {
                   auto vI = Vs(i);

                   Float thickness       = thicknesses(vI);
                   Float d_hat_expansion = point_dcd_expansion(d_hats(vI));

                   const auto& pos   = Ps(vI);
                   Vector3     pos_t = pos + dxs(vI) * alpha;

                   AABB aabb;
                   aabb.extend(pos.cast<float>()).extend(pos_t.cast<float>());

                   float expand = d_hat_expansion + thickness;

                   aabb.min().array() -= expand;
                   aabb.max().array() += expand;
                   aabbs(i) = aabb;
               });

    // build AABBs for edges
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(Es.size(),
               [Es          = Es.viewer().name("E"),
                Ps          = Ps.viewer().name("Ps"),
                aabbs       = edge_aabbs.viewer().name("aabbs"),
                dxs         = dxs.viewer().name("dx"),
                thicknesses = info.thicknesses().viewer().name("thicknesses"),
                d_hats      = info.d_hats().viewer().name("d_hats"),
                alpha       = alpha] __device__(int i) mutable
               {
                   auto eI = Es(i);

                   Float thickness =
                       edge_thickness(thicknesses(eI[0]), thicknesses(eI[1]));
                   Float d_hat_expansion =
                       edge_dcd_expansion(d_hats(eI[0]), d_hats(eI[1]));

                   const auto& pos0   = Ps(eI[0]);
                   const auto& pos1   = Ps(eI[1]);
                   Vector3     pos0_t = pos0 + dxs(eI[0]) * alpha;
                   Vector3     pos1_t = pos1 + dxs(eI[1]) * alpha;

                   Vector3 max = pos0_t;
                   Vector3 min = pos0_t;

                   AABB aabb;

                   aabb.extend(pos0.cast<float>())
                       .extend(pos1.cast<float>())
                       .extend(pos0_t.cast<float>())
                       .extend(pos1_t.cast<float>());

                   float expand = d_hat_expansion + thickness;

                   aabb.min().array() -= expand;
                   aabb.max().array() += expand;
                   aabbs(i) = aabb;
               });

    // build AABBs for triangles
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(Fs.size(),
               [Fs          = Fs.viewer().name("F"),
                Ps          = Ps.viewer().name("Ps"),
                aabbs       = triangle_aabbs.viewer().name("aabbs"),
                dxs         = dxs.viewer().name("dx"),
                thicknesses = info.thicknesses().viewer().name("thicknesses"),
                d_hats      = info.d_hats().viewer().name("d_hats"),
                alpha       = alpha] __device__(int i) mutable
               {
                   auto fI = Fs(i);

                   Float thickness = triangle_thickness(thicknesses(fI[0]),
                                                        thicknesses(fI[1]),
                                                        thicknesses(fI[2]));
                   Float d_hat_expansion = triangle_dcd_expansion(
                       d_hats(fI[0]), d_hats(fI[1]), d_hats(fI[2]));

                   const auto& pos0   = Ps(fI[0]);
                   const auto& pos1   = Ps(fI[1]);
                   const auto& pos2   = Ps(fI[2]);
                   Vector3     pos0_t = pos0 + dxs(fI[0]) * alpha;
                   Vector3     pos1_t = pos1 + dxs(fI[1]) * alpha;
                   Vector3     pos2_t = pos2 + dxs(fI[2]) * alpha;

                   AABB aabb;

                   aabb.extend(pos0.cast<float>())
                       .extend(pos1.cast<float>())
                       .extend(pos2.cast<float>())
                       .extend(pos0_t.cast<float>())
                       .extend(pos1_t.cast<float>())
                       .extend(pos2_t.cast<float>());

                   float expand = d_hat_expansion + thickness;

                   aabb.min().array() -= expand;
                   aabb.max().array() += expand;
                   aabbs(i) = aabb;
               });

    lbvh_E.build(edge_aabbs);
    lbvh_T.build(triangle_aabbs);

    if(codimVs.size() > 0)
    {
        // Use AllP to query CodimP
        {
            lbvh_CodimP.build(codim_point_aabbs);

            muda::KernelLabel label{__FUNCTION__, __FILE__, __LINE__};
            lbvh_CodimP.query(
                point_aabbs,                                  // AllP
                [Vs      = Vs.viewer().name("Vs"),            // AllP
                 codimVs = codimVs.viewer().name("codimVs"),  // CodimP

                 Ps          = Ps.viewer().name("Ps"),
                 dxs         = dxs.viewer().name("dxs"),
                 thicknesses = info.thicknesses().viewer().name("thicknesses"),
                 dimensions  = info.dimensions().viewer().name("dimensions"),
                 contact_element_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                 contact_mask_tabular = info.contact_mask_tabular().viewer().name("contact_mask_tabular"),
                 subscene_element_ids = info.subscene_element_ids().viewer().name("subscene_element_ids"),
                 subscene_mask_tabular = info.subscene_mask_tabular().viewer().name("subscene_mask_tabular"),
                 v2b = info.v2b().viewer().name("v2b"),
                 body_self_collision = info.body_self_collision().viewer().name("body_self_collision"),
                 d_hats = info.d_hats().viewer().name("d_hats"),
                 alpha  = alpha] __device__(IndexT i, IndexT j)
                {
                    const auto& V      = Vs(i);
                    const auto& codimV = codimVs(j);

                    Vector2i cids = {contact_element_ids(V), contact_element_ids(codimV)};
                    Vector2i scids = {subscene_element_ids(V), subscene_element_ids(codimV)};

                    // discard if the contact is disabled
                    if(!allow_PP_contact(subscene_mask_tabular, scids))
                        return false;
                    if(!allow_PP_contact(contact_mask_tabular, cids))
                        return false;

                    bool V_is_codim = dimensions(V) <= 2;  // codim 0D vert and vert from codim 1D edge

                    if(V_is_codim && V >= codimV)  // avoid duplicate CodimP-CodimP pairs
                        return false;

                    auto body_i = v2b(V);
                    auto body_j = v2b(codimV);
                    // skip self-collision for the same body if self collision off
                    if(body_i == body_j && !body_self_collision(body_i))
                        return false;


                    Vector3 P0  = Ps(V);
                    Vector3 dP0 = alpha * dxs(V);

                    Vector3 P1  = Ps(codimV);
                    Vector3 dP1 = alpha * dxs(codimV);

                    Float thickness = PP_thickness(thicknesses(V), thicknesses(codimV));
                    Float d_hat = PP_d_hat(d_hats(V), d_hats(codimV));

                    Float expand = d_hat + thickness;

                    if(!distance::point_point_ccd_broadphase(P0, P1, dP0, dP1, expand))
                        return false;

                    return true;
                },
                candidate_AllP_CodimP_pairs);
        }

        // Use CodimP to query AllE
        {
            muda::KernelLabel label{__FUNCTION__, __FILE__, __LINE__};
            lbvh_E.query(
                codim_point_aabbs,
                [codimVs     = codimVs.viewer().name("Vs"),
                 Es          = Es.viewer().name("Es"),
                 Ps          = Ps.viewer().name("Ps"),
                 dxs         = dxs.viewer().name("dxs"),
                 thicknesses = info.thicknesses().viewer().name("thicknesses"),
                 contact_element_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                 contact_mask_tabular = info.contact_mask_tabular().viewer().name("contact_mask_tabular"),
                 subscene_element_ids = info.subscene_element_ids().viewer().name("subscene_element_ids"),
                 subscene_mask_tabular = info.subscene_mask_tabular().viewer().name("subscene_mask_tabular"),
                 v2b = info.v2b().viewer().name("v2b"),
                 body_self_collision = info.body_self_collision().viewer().name("body_self_collision"),
                 d_hats = info.d_hats().viewer().name("d_hats"),
                 alpha  = alpha] __device__(IndexT i, IndexT j)
                {
                    const auto& codimV = codimVs(i);
                    const auto& E      = Es(j);

                    Vector3i cids = {contact_element_ids(codimV),
                                     contact_element_ids(E[0]),
                                     contact_element_ids(E[1])};

                    Vector3i scids = {subscene_element_ids(codimV),
                                      subscene_element_ids(E[0]),
                                      subscene_element_ids(E[1])};

                    // discard if the contact is disabled
                    if(!allow_PE_contact(subscene_mask_tabular, scids))
                        return false;
                    if(!allow_PE_contact(contact_mask_tabular, cids))
                        return false;

                    // discard if the vertex is on the edge
                    if(E[0] == codimV || E[1] == codimV)
                        return false;

                    auto body_i = v2b(codimV);
                    auto body_j = v2b(E[0]);
                    // skip self-collision for the same body if self collision off
                    if(body_i == body_j && !body_self_collision(body_i))
                        return false;

                    Vector3 E0  = Ps(E[0]);
                    Vector3 E1  = Ps(E[1]);
                    Vector3 dE0 = alpha * dxs(E[0]);
                    Vector3 dE1 = alpha * dxs(E[1]);

                    Vector3 P  = Ps(codimV);
                    Vector3 dP = alpha * dxs(codimV);

                    Float thickness = PE_thickness(thicknesses(codimV),
                                                   thicknesses(E[0]),
                                                   thicknesses(E[1]));
                    Float d_hat = PE_d_hat(d_hats(codimV), d_hats(E[0]), d_hats(E[1]));

                    Float expand = d_hat + thickness;

                    if(!distance::point_edge_ccd_broadphase(P, E0, E1, dP, dE0, dE1, expand))
                        return false;

                    return true;
                },
                candidate_CodimP_AllE_pairs);
        }
    }

    // Use AllE to query AllE
    if(Es.size() > 0)
    {
        corex_profile::ScopedPhase phase(
            "contact_detect_detail",
            contact_mask_fast && subscene_mask_fast
                ? (alpha > 0 ? "query_alle_alle_ccd_fast" : "query_alle_alle_dcd_fast")
                : "query_alle_alle");
        if(contact_mask_fast && subscene_mask_fast)
        {
            lbvh_E.detect_edges_no_mask(Es,
                                        info.v2b(),
                                        info.body_self_collision(),
                                        candidate_AllE_AllE_pairs);
        }
        else
        {
            muda::KernelLabel label{__FUNCTION__, __FILE__, __LINE__};
            lbvh_E.detect(
            [Es          = Es.viewer().name("Es"),
             Ps          = Ps.viewer().name("Ps"),
             dxs         = dxs.viewer().name("dxs"),
             thicknesses = info.thicknesses().viewer().name("thicknesses"),
             contact_element_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
             contact_mask_tabular = info.contact_mask_tabular().viewer().name("contact_mask_tabular"),
             subscene_element_ids = info.subscene_element_ids().viewer().name("subscene_element_ids"),
             subscene_mask_tabular = info.subscene_mask_tabular().viewer().name("subscene_mask_tabular"),
             v2b = info.v2b().viewer().name("v2b"),
             body_self_collision = info.body_self_collision().viewer().name("body_self_collision"),
             d_hats = info.d_hats().viewer().name("d_hats"),
             alpha  = alpha,
             contact_mask_fast,
             subscene_mask_fast] __device__(IndexT i, IndexT j)
            {
                const auto& E0 = Es(i);
                const auto& E1 = Es(j);

                // discard if the contact is disabled
                if(!subscene_mask_fast)
                {
                    Vector4i scids = {subscene_element_ids(E0[0]),
                                      subscene_element_ids(E0[1]),
                                      subscene_element_ids(E1[0]),
                                      subscene_element_ids(E1[1])};
                    if(!allow_EE_contact(subscene_mask_tabular, scids))
                        return false;
                }
                if(!contact_mask_fast)
                {
                    Vector4i cids = {contact_element_ids(E0[0]),
                                     contact_element_ids(E0[1]),
                                     contact_element_ids(E1[0]),
                                     contact_element_ids(E1[1])};
                    if(!allow_EE_contact(contact_mask_tabular, cids))
                        return false;
                }

                // discard if the edges share same vertex
                if(E0[0] == E1[0] || E0[0] == E1[1] || E0[1] == E1[0] || E0[1] == E1[1])
                    return false;

                auto body_i = v2b(E0[0]);
                auto body_j = v2b(E1[0]);
                if(body_i == body_j && !body_self_collision(body_i))
                    return false;  // skip self-collision for the same body

                if(contact_mask_fast && subscene_mask_fast)
                    return true;

                Vector3 E0_0  = Ps(E0[0]);
                Vector3 E0_1  = Ps(E0[1]);
                Vector3 dE0_0 = alpha * dxs(E0[0]);
                Vector3 dE0_1 = alpha * dxs(E0[1]);

                Vector3 E1_0  = Ps(E1[0]);
                Vector3 E1_1  = Ps(E1[1]);
                Vector3 dE1_0 = alpha * dxs(E1[0]);
                Vector3 dE1_1 = alpha * dxs(E1[1]);

                Float thickness = EE_thickness(thicknesses(E0[0]),
                                               thicknesses(E0[1]),
                                               thicknesses(E1[0]),
                                               thicknesses(E1[1]));

                Float d_hat =
                    EE_d_hat(d_hats(E0[0]), d_hats(E0[1]), d_hats(E1[0]), d_hats(E1[1]));

                Float expand = d_hat + thickness;

                if(alpha == 0)
                {
                    if(!distance::edge_edge_cd_broadphase(E0_0, E0_1, E1_0, E1_1, expand))
                        return false;
                }
                else
                {
                    if(!distance::edge_edge_ccd_broadphase(
                           E0_0, E0_1, E1_0, E1_1, dE0_0, dE0_1, dE1_0, dE1_1, expand))
                        return false;
                }

                return true;
            },
            candidate_AllE_AllE_pairs);
        }
    }

    // Use AllP to query AllT
    if(Fs.size() > 0)
    {
        corex_profile::ScopedPhase phase(
            "contact_detect_detail",
            contact_mask_fast && subscene_mask_fast ? "query_allp_allt_fast"
                                                    : "query_allp_allt");
        muda::KernelLabel label{__FUNCTION__, __FILE__, __LINE__};
        lbvh_T.query(
            point_aabbs,
            [Vs          = Vs.viewer().name("Vs"),
             Fs          = Fs.viewer().name("Fs"),
             Ps          = Ps.viewer().name("Ps"),
             dxs         = dxs.viewer().name("dxs"),
             thicknesses = info.thicknesses().viewer().name("thicknesses"),
             contact_element_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
             contact_mask_tabular = info.contact_mask_tabular().viewer().name("contact_mask_tabular"),
             subscene_element_ids = info.subscene_element_ids().viewer().name("subscene_element_ids"),
             subscene_mask_tabular = info.subscene_mask_tabular().viewer().name("subscene_mask_tabular"),
             v2b = info.v2b().viewer().name("v2b"),
             body_self_collision = info.body_self_collision().viewer().name("body_self_collision"),
             d_hats = info.d_hats().viewer().name("d_hats"),
             alpha  = alpha,
             contact_mask_fast,
             subscene_mask_fast] __device__(IndexT i, IndexT j)
            {
                auto V = Vs(i);
                auto F = Fs(j);

                // discard if the contact is disabled
                if(!subscene_mask_fast)
                {
                    Vector4i scids = {subscene_element_ids(V),
                                      subscene_element_ids(F[0]),
                                      subscene_element_ids(F[1]),
                                      subscene_element_ids(F[2])};
                    if(!allow_PT_contact(subscene_mask_tabular, scids))
                        return false;
                }
                if(!contact_mask_fast)
                {
                    Vector4i cids = {contact_element_ids(V),
                                     contact_element_ids(F[0]),
                                     contact_element_ids(F[1]),
                                     contact_element_ids(F[2])};
                    if(!allow_PT_contact(contact_mask_tabular, cids))
                        return false;
                }

                // discard if the point is on the triangle
                if(F[0] == V || F[1] == V || F[2] == V)
                    return false;

                auto body_i = v2b(V);
                auto body_j = v2b(F[0]);
                // skip self-collision for the same body if self collision off
                if(body_i == body_j && !body_self_collision(body_i))
                    return false;


                Vector3 P  = Ps(V);
                Vector3 dP = alpha * dxs(V);

                Vector3 F0 = Ps(F[0]);
                Vector3 F1 = Ps(F[1]);
                Vector3 F2 = Ps(F[2]);

                Vector3 dF0 = alpha * dxs(F[0]);
                Vector3 dF1 = alpha * dxs(F[1]);
                Vector3 dF2 = alpha * dxs(F[2]);

                Float thickness = PT_thickness(thicknesses(V),
                                               thicknesses(F[0]),
                                               thicknesses(F[1]),
                                               thicknesses(F[2]));

                Float d_hat =
                    PT_d_hat(d_hats(V), d_hats(F[0]), d_hats(F[1]), d_hats(F[2]));

                Float expand = d_hat + thickness;

                if(alpha == 0)
                {
                    if(!distance::point_triangle_cd_broadphase(P, F0, F1, F2, expand))
                        return false;
                }
                else
                {
                    if(!distance::point_triangle_ccd_broadphase(
                           P, F0, F1, F2, dP, dF0, dF1, dF2, expand))
                        return false;
                }

                return true;
            },
            candidate_AllP_AllT_pairs);
    }
}

void StacklessBVHSimplexTrajectoryFilter::Impl::filter_active(FilterActiveInfo& info)
{
    using namespace muda;

    // we will filter-out the active pairs
    auto positions = info.positions();

    SizeT N_PCoimP  = candidate_AllP_CodimP_pairs.size();
    SizeT N_CodimPE = candidate_CodimP_AllE_pairs.size();
    SizeT N_PTs     = candidate_AllP_AllT_pairs.size();
    SizeT N_EEs     = candidate_AllE_AllE_pairs.size();

    // PT, EE, PT, PP can degenerate to PP
    temp_PPs.resize(N_PCoimP + N_CodimPE + N_PTs + N_EEs);
    // PT, EE, PT can degenerate to PE
    temp_PEs.resize(N_CodimPE + N_PTs + N_EEs);

    temp_PTs.resize(N_PTs);
    temp_EEs.resize(N_EEs);

    SizeT temp_PP_offset = 0;
    SizeT temp_PE_offset = 0;

    // AllP and CodimP
    if(N_PCoimP > 0)
    {
        auto PP_view = temp_PPs.view(temp_PP_offset, N_PCoimP);

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(candidate_AllP_CodimP_pairs.size(),
                   [positions = positions.viewer().name("positions"),
                    PCodimP_pairs = candidate_AllP_CodimP_pairs.viewer().name("PP_pairs"),
                    surf_vertices = info.surf_vertices().viewer().name("surf_vertices"),
                    codim_vertices = info.codim_vertices().viewer().name("codim_vertices"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    temp_PPs = PP_view.viewer().name("temp_PPs"),
                    d_hats = info.d_hats().viewer().name("d_hats")] __device__(int i) mutable
                   {
                       // default invalid
                       auto& PP = temp_PPs(i);
                       PP.setConstant(-1);

                       Vector2i indices = PCodimP_pairs(i);

                       IndexT P0 = surf_vertices(indices(0));
                       IndexT P1 = codim_vertices(indices(1));


                       const auto& V0 = positions(P0);
                       const auto& V1 = positions(P1);

                       Float thickness = PP_thickness(thicknesses(P0), thicknesses(P1));
                       Float d_hat = PP_d_hat(d_hats(P0), d_hats(P1));

                       Vector2 range = D_range(thickness, d_hat);

                       Float D;
                       distance::point_point_distance2(V0, V1, D);

                       if constexpr(PrintKernelZeroDistance)
                       {
                           if(D <= range.x())
                           {
                               printf("[SBVH][PP][low-dist] i=%d P=(%d,%d) D=%e range=(%e,%e) "
                                      "thickness=%e d_hat=%e\n",
                                      i,
                                      P0,
                                      P1,
                                      D,
                                      range.x(),
                                      range.y(),
                                      thickness,
                                      d_hat);
                           }
                       }

                       MUDA_ASSERT(D > range.x(),
                                   "Thickness Violated! D(%f) should be > D_range.x(%f), "
                                   "P=(%d,%d), thickness=%f, d_hat=%f",
                                   D,
                                   range.x(),
                                   P0,
                                   P1,
                                   thickness,
                                   d_hat);
                       if(!is_active_D(range, D))
                           return;  // early return

                       PP = {P0, P1};
                   });

        temp_PP_offset += N_PCoimP;
    }
    // CodimP and AllE
    if(N_CodimPE > 0)
    {
        auto PP_view = temp_PPs.view(temp_PP_offset, N_CodimPE);
        auto PE_view = temp_PEs.view(temp_PE_offset, N_CodimPE);

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(
                candidate_CodimP_AllE_pairs.size(),
                [positions = positions.viewer().name("positions"),
                 CodimP_AllE_pairs = candidate_CodimP_AllE_pairs.viewer().name("PE_pairs"),
                 codim_veritces = info.codim_vertices().viewer().name("codim_vertices"),
                 surf_edges  = info.surf_edges().viewer().name("surf_edges"),
                 thicknesses = info.thicknesses().viewer().name("thicknesses"),
                 temp_PPs    = PP_view.viewer().name("temp_PPs"),
                 temp_PEs    = PE_view.viewer().name("temp_PEs"),
                 d_hats = info.d_hats().viewer().name("d_hats")] __device__(int i) mutable
                {
                    auto& PP = temp_PPs(i);
                    PP.setConstant(-1);
                    auto& PE = temp_PEs(i);
                    PE.setConstant(-1);

                    Vector2i indices = CodimP_AllE_pairs(i);
                    IndexT   V       = codim_veritces(indices(0));
                    Vector2i E       = surf_edges(indices(1));

                    Vector3i vIs = {V, E(0), E(1)};
                    Vector3 Ps[] = {positions(vIs(0)), positions(vIs(1)), positions(vIs(2))};

                    Float thickness = PE_thickness(
                        thicknesses(V), thicknesses(E(0)), thicknesses(E(1)));

                    Float d_hat = PE_d_hat(d_hats(V), d_hats(E(0)), d_hats(E(1)));


                    Vector3i flag =
                        distance::point_edge_distance_flag(Ps[0], Ps[1], Ps[2]);

                    Vector2 range = D_range(thickness, d_hat);

                    Float D;
                    distance::point_edge_distance2(flag, Ps[0], Ps[1], Ps[2], D);

                    if constexpr(PrintKernelZeroDistance)
                    {
                        if(D <= range.x())
                        {
                            printf("[SBVH][PE][low-dist] i=%d V-E=(%d,%d,%d) flag=(%d,%d,%d) "
                                   "D=%e range=(%e,%e) thickness=%e d_hat=%e\n",
                                   i,
                                   vIs(0),
                                   vIs(1),
                                   vIs(2),
                                   flag(0),
                                   flag(1),
                                   flag(2),
                                   D,
                                   range.x(),
                                   range.y(),
                                   thickness,
                                   d_hat);
                        }
                    }

                    MUDA_ASSERT(D > range.x(),
                                "Thickness Violated! D(%f) should be > D_range.x(%f), "
                                "V-E=(%d,%d,%d), flag=(%d,%d,%d), thickness=%f, d_hat=%f",
                                D,
                                range.x(),
                                vIs(0),
                                vIs(1),
                                vIs(2),
                                flag(0),
                                flag(1),
                                flag(2),
                                thickness,
                                d_hat);
                       if(!is_active_D(range, D))
                        return;  // early return

                    Vector3i offsets;
                    auto dim = distance::degenerate_point_edge(flag, offsets);

                    switch(dim)
                    {
                        case 2:  // PP
                        {
                            IndexT V0 = vIs(offsets(0));
                            IndexT V1 = vIs(offsets(1));
                            PP        = {V0, V1};
                        }
                        break;
                        case 3:  // PE
                        {
                            PE = vIs;
                        }
                        break;
                        default: {
                            MUDA_ERROR_WITH_LOCATION("unexpected degenerate case dim=%d", dim);
                        }
                        break;
                    }
                });

        temp_PP_offset += N_CodimPE;
        temp_PE_offset += N_CodimPE;
    }

    // AllP and AllT
    {
        auto PP_view = temp_PPs.view(temp_PP_offset, N_PTs);
        auto PE_view = temp_PEs.view(temp_PE_offset, N_PTs);

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(
                candidate_AllP_AllT_pairs.size(),
                [positions = positions.viewer().name("Ps"),
                 PT_pairs = candidate_AllP_AllT_pairs.viewer().name("PT_pairs"),
                 surf_vertices = info.surf_vertices().viewer().name("surf_vertices"),
                 surf_triangles = info.surf_triangles().viewer().name("surf_triangles"),
                 thicknesses = info.thicknesses().viewer().name("thicknesses"),
                 temp_PPs    = PP_view.viewer().name("temp_PPs"),
                 temp_PEs    = PE_view.viewer().name("temp_PEs"),
                 temp_PTs    = temp_PTs.viewer().name("temp_PTs"),
                 d_hats = info.d_hats().viewer().name("d_hats")] __device__(int i) mutable
                {
                    auto& PP = temp_PPs(i);
                    PP.setConstant(-1);
                    auto& PE = temp_PEs(i);
                    PE.setConstant(-1);
                    auto& PT = temp_PTs(i);
                    PT.setConstant(-1);

                    Vector2i indices = PT_pairs(i);
                    IndexT   V       = surf_vertices(indices(0));
                    Vector3i F       = surf_triangles(indices(1));

                    Vector4i vIs  = {V, F(0), F(1), F(2)};
                    Vector3  Ps[] = {positions(vIs(0)),
                                     positions(vIs(1)),
                                     positions(vIs(2)),
                                     positions(vIs(3))};

                    Float thickness = PT_thickness(thicknesses(V),
                                                   thicknesses(F(0)),
                                                   thicknesses(F(1)),
                                                   thicknesses(F(2)));

                    Float d_hat =
                        PT_d_hat(d_hats(V), d_hats(F(0)), d_hats(F(1)), d_hats(F(2)));

                    Vector4i flag =
                        distance::point_triangle_distance_flag(Ps[0], Ps[1], Ps[2], Ps[3]);

                    Vector2 range = D_range(thickness, d_hat);

                    Float D;
                    distance::point_triangle_distance2(flag, Ps[0], Ps[1], Ps[2], Ps[3], D);

                    if constexpr(PrintKernelZeroDistance)
                    {
                        if(D <= range.x())
                        {
                            printf("[SBVH][PT][low-dist] i=%d V-F=(%d,%d,%d,%d) "
                                   "flag=(%d,%d,%d,%d) D=%e range=(%e,%e) thickness=%e d_hat=%e\n",
                                   i,
                                   vIs(0),
                                   vIs(1),
                                   vIs(2),
                                   vIs(3),
                                   flag(0),
                                   flag(1),
                                   flag(2),
                                   flag(3),
                                   D,
                                   range.x(),
                                   range.y(),
                                   thickness,
                                   d_hat);
                        }
                    }

                    MUDA_ASSERT(
                        D > 0.0, "D=%f, V F = (%d,%d,%d,%d)", D, vIs(0), vIs(1), vIs(2), vIs(3));

                    MUDA_ASSERT(D > range.x(),
                                "Thickness Violated! D(%f) should be > D_range.x(%f), "
                                "V-F=(%d,%d,%d,%d), flag=(%d,%d,%d,%d), thickness=%f, d_hat=%f",
                                D,
                                range.x(),
                                vIs(0),
                                vIs(1),
                                vIs(2),
                                vIs(3),
                                flag(0),
                                flag(1),
                                flag(2),
                                flag(3),
                                thickness,
                                d_hat);
                       if(!is_active_D(range, D))
                        return;  // early return

                    Vector4i offsets;
                    auto dim = distance::degenerate_point_triangle(flag, offsets);

                    switch(dim)
                    {
                        case 2:  // PP
                        {
                            IndexT V0 = vIs(offsets(0));
                            IndexT V1 = vIs(offsets(1));
                            PP        = {V0, V1};
                        }
                        break;
                        case 3:  // PE
                        {
                            IndexT V0 = vIs(offsets(0));
                            IndexT V1 = vIs(offsets(1));
                            IndexT V2 = vIs(offsets(2));
                            PE        = {V0, V1, V2};
                        }
                        break;
                        case 4:  // PT
                        {
                            PT = vIs;
                        }
                        break;
                        default: {
                            MUDA_ERROR_WITH_LOCATION("unexpected degenerate case dim=%d", dim);
                        }
                        break;
                    }
                });

        temp_PP_offset += N_PTs;
        temp_PE_offset += N_PTs;
    }
    // AllE and AllE
    {
        auto PP_view = temp_PPs.view(temp_PP_offset, N_EEs);
        auto PE_view = temp_PEs.view(temp_PE_offset, N_EEs);

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(
                candidate_AllE_AllE_pairs.size(),
                [positions = positions.viewer().name("Ps"),
                 rest_positions = info.rest_positions().viewer().name("rest_positions"),
                 EE_pairs = candidate_AllE_AllE_pairs.viewer().name("EE_pairs"),
                 surf_edges  = info.surf_edges().viewer().name("surf_edges"),
                 thicknesses = info.thicknesses().viewer().name("thicknesses"),
                 temp_PPs    = PP_view.viewer().name("temp_PPs"),
                 temp_PEs    = PE_view.viewer().name("temp_PEs"),
                 temp_EEs    = temp_EEs.viewer().name("temp_EEs"),
                 d_hats = info.d_hats().viewer().name("d_hats")] __device__(int i) mutable
                {
                    auto& PP = temp_PPs(i);
                    PP.setConstant(-1);
                    auto& PE = temp_PEs(i);
                    PE.setConstant(-1);
                    auto& EE = temp_EEs(i);
                    EE.setConstant(-1);

                    Vector2i indices = EE_pairs(i);
                    Vector2i E0      = surf_edges(indices(0));
                    Vector2i E1      = surf_edges(indices(1));

                    Vector4i vIs  = {E0(0), E0(1), E1(0), E1(1)};
                    Vector3  Ps[] = {positions(vIs(0)),
                                     positions(vIs(1)),
                                     positions(vIs(2)),
                                     positions(vIs(3))};

                    Float thickness = EE_thickness(thicknesses(E0(0)),
                                                   thicknesses(E0(1)),
                                                   thicknesses(E1(0)),
                                                   thicknesses(E1(1)));

                    Float d_hat = EE_d_hat(
                        d_hats(E0(0)), d_hats(E0(1)), d_hats(E1(0)), d_hats(E1(1)));

                    Vector2 range = D_range(thickness, d_hat);

                    Vector4i flag =
                        distance::edge_edge_distance_flag(Ps[0], Ps[1], Ps[2], Ps[3]);

                    Float D;
                    distance::edge_edge_distance2(flag, Ps[0], Ps[1], Ps[2], Ps[3], D);

                    if constexpr(PrintKernelZeroDistance)
                    {
                        if(D <= range.x())
                        {
                            printf("[SBVH][EE][low-dist] i=%d E-E=(%d,%d,%d,%d) "
                                   "flag=(%d,%d,%d,%d) D=%e range=(%e,%e) thickness=%e d_hat=%e\n",
                                   i,
                                   vIs(0),
                                   vIs(1),
                                   vIs(2),
                                   vIs(3),
                                   flag(0),
                                   flag(1),
                                   flag(2),
                                   flag(3),
                                   D,
                                   range.x(),
                                   range.y(),
                                   thickness,
                                   d_hat);
                        }
                    }
                    // Corner case: exact/near-zero EE distance may appear for degenerate or
                    // intersecting edge-edge candidates. Treat it as an active EE pair instead
                    // of hard-aborting in the trajectory filter stage.
                    if(D <= range.x())
                    {
                        EE = vIs;
                        return;
                    }
                       if(!is_active_D(range, D))
                        return;  // early return

                    Float eps_x;
                    distance::edge_edge_mollifier_threshold(rest_positions(vIs(0)),
                                                            rest_positions(vIs(1)),
                                                            rest_positions(vIs(2)),
                                                            rest_positions(vIs(3)),
                                                            static_cast<Float>(1e-3),
                                                            eps_x);

                    if(distance::need_mollify(Ps[0], Ps[1], Ps[2], Ps[3], eps_x))
                    {
                        EE = vIs;
                        return;
                    }
                    else  // classify to EE/PE/PP
                    {
                        Vector4i offsets;
                        auto dim = distance::degenerate_edge_edge(flag, offsets);

                        switch(dim)
                        {
                            case 2:  // PP
                            {
                                IndexT V0 = vIs(offsets(0));
                                IndexT V1 = vIs(offsets(1));
                                PP        = {V0, V1};
                            }
                            break;
                            case 3:  // PE
                            {
                                IndexT V0 = vIs(offsets(0));
                                IndexT V1 = vIs(offsets(1));
                                IndexT V2 = vIs(offsets(2));
                                PE        = {V0, V1, V2};
                            }
                            break;
                            case 4:  // EE
                            {
                                EE = vIs;
                            }
                            break;
                            default: {
                                MUDA_ERROR_WITH_LOCATION("unexpected degenerate case dim=%d", dim);
                            }
                            break;
                        }
                    }
                });

        temp_PP_offset += N_EEs;
        temp_PE_offset += N_EEs;
    }

    UIPC_ASSERT(temp_PP_offset == temp_PPs.size(), "size mismatch");
    UIPC_ASSERT(temp_PE_offset == temp_PEs.size(), "size mismatch");

    {  // select the valid ones
        PPs.resize(temp_PPs.size());
        PEs.resize(temp_PEs.size());
        PTs.resize(temp_PTs.size());
        EEs.resize(temp_EEs.size());

        DeviceSelect().If(temp_PPs.data(),
                          PPs.data(),
                          selected_PP_count.data(),
                          temp_PPs.size(),
                          [] CUB_RUNTIME_FUNCTION(const Vector2i& PP)
                          { return PP(0) != -1; });

        DeviceSelect().If(temp_PEs.data(),
                          PEs.data(),
                          selected_PE_count.data(),
                          temp_PEs.size(),
                          [] CUB_RUNTIME_FUNCTION(const Vector3i& PE)
                          { return PE(0) != -1; });

        DeviceSelect().If(temp_PTs.data(),
                          PTs.data(),
                          selected_PT_count.data(),
                          temp_PTs.size(),
                          [] CUB_RUNTIME_FUNCTION(const Vector4i& PT)
                          { return PT(0) != -1; });

        DeviceSelect().If(temp_EEs.data(),
                          EEs.data(),
                          selected_EE_count.data(),
                          temp_EEs.size(),
                          [] CUB_RUNTIME_FUNCTION(const Vector4i& EE)
                          { return EE(0) != -1; });

        IndexT PP_count = selected_PP_count;
        IndexT PE_count = selected_PE_count;
        IndexT PT_count = selected_PT_count;
        IndexT EE_count = selected_EE_count;

        PPs.resize(PP_count);
        PEs.resize(PE_count);
        PTs.resize(PT_count);
        EEs.resize(EE_count);
    }

    info.PPs(PPs);
    info.PEs(PEs);
    info.PTs(PTs);
    info.EEs(EEs);

    if constexpr(PrintDebugInfo)
    {
        std::vector<Vector2i> PPs_host;
        std::vector<Float>    PP_thicknesses_host;

        std::vector<Vector3i> PEs_host;
        std::vector<Float>    PE_thicknesses_host;

        std::vector<Vector4i> PTs_host;
        std::vector<Float>    PT_thicknesses_host;

        std::vector<Vector4i> EEs_host;
        std::vector<Float>    EE_thicknesses_host;

        PPs.copy_to(PPs_host);
        PEs.copy_to(PEs_host);
        PTs.copy_to(PTs_host);
        EEs.copy_to(EEs_host);

        std::cout << "filter result:" << std::endl;

        for(auto&& [PP, thickness] : zip(PPs_host, PP_thicknesses_host))
        {
            std::cout << "PP: " << PP.transpose() << " thickness: " << thickness << "\n";
        }

        for(auto&& [PE, thickness] : zip(PEs_host, PE_thicknesses_host))
        {
            std::cout << "PE: " << PE.transpose() << " thickness: " << thickness << "\n";
        }

        for(auto&& [PT, thickness] : zip(PTs_host, PT_thicknesses_host))
        {
            std::cout << "PT: " << PT.transpose() << " thickness: " << thickness << "\n";
        }

        for(auto&& [EE, thickness] : zip(EEs_host, EE_thicknesses_host))
        {
            std::cout << "EE: " << EE.transpose() << " thickness: " << thickness << "\n";
        }

        std::cout << std::flush;
    }
}

void StacklessBVHSimplexTrajectoryFilter::Impl::filter_toi(FilterTOIInfo& info)
{
    using namespace muda;

    auto toi_size =
        candidate_AllP_CodimP_pairs.size() + candidate_CodimP_AllE_pairs.size()
        + candidate_AllP_AllT_pairs.size() + candidate_AllE_AllE_pairs.size();

    tois.resize(toi_size);

    auto offset  = 0;
    auto PP_tois = tois.view(offset, candidate_AllP_CodimP_pairs.size());
    offset += candidate_AllP_CodimP_pairs.size();
    auto PE_tois = tois.view(offset, candidate_CodimP_AllE_pairs.size());
    offset += candidate_CodimP_AllE_pairs.size();
    auto PT_tois = tois.view(offset, candidate_AllP_AllT_pairs.size());
    offset += candidate_AllP_AllT_pairs.size();
    auto EE_tois = tois.view(offset, candidate_AllE_AllE_pairs.size());
    offset += candidate_AllE_AllE_pairs.size();

    UIPC_ASSERT(offset == toi_size, "size mismatch");


    // TODO: Now hard code the minimum separation coefficient
    // gap = eta * (dist2_cur - thickness * thickness) / (dist_cur + thickness);
    constexpr Float eta = 0.1;

    // TODO: Now hard code the maximum iteration
    constexpr SizeT max_iter = 1000;

    // large enough toi (>1)
    constexpr Float large_enough_toi = 1.1;

    // AllP and CodimP
    {
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(candidate_AllP_CodimP_pairs.size(),
                   [PP_tois = PP_tois.viewer().name("PP_tois"),
                    PCodimP_pairs = candidate_AllP_CodimP_pairs.viewer().name("PP_pairs"),
                    codim_vertices = info.codim_vertices().viewer().name("codim_vertices"),
                    surf_vertices = info.surf_vertices().viewer().name("surf_vertices"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    positions = info.positions().viewer().name("Ps"),
                    dxs       = info.displacements().viewer().name("dxs"),
                    d_hats    = info.d_hats().viewer().name("d_hats"),
                    alpha     = info.alpha(),

                    eta,
                    max_iter,
                    large_enough_toi] __device__(int i) mutable
                   {
                       auto   indices = PCodimP_pairs(i);
                       IndexT V0      = surf_vertices(indices(0));
                       IndexT V1      = codim_vertices(indices(1));

                       Float thickness = PP_thickness(thicknesses(V0), thicknesses(V1));
                       Float d_hat = PP_d_hat(d_hats(V0), d_hats(V1));

                       Vector3 VP0  = positions(V0);
                       Vector3 VP1  = positions(V1);
                       Vector3 dVP0 = alpha * dxs(V0);
                       Vector3 dVP1 = alpha * dxs(V1);

                       Float toi = large_enough_toi;

                       bool faraway = !distance::point_point_ccd_broadphase(
                           VP0, VP1, dVP0, dVP1, d_hat + thickness);

                       if(faraway)
                       {
                           PP_tois(i) = toi;
                           return;
                       }

                       bool hit = distance::point_point_ccd(
                           VP0, VP1, dVP0, dVP1, eta, thickness, max_iter, toi);

                       if(!hit)
                           toi = large_enough_toi;

                       PP_tois(i) = toi;
                   });
    }

    // CodimP and AllE
    {
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(candidate_CodimP_AllE_pairs.size(),
                   [PE_tois = PE_tois.viewer().name("PE_tois"),
                    CodimP_AllE_pairs = candidate_CodimP_AllE_pairs.viewer().name("PE_pairs"),
                    codim_vertices = info.codim_vertices().viewer().name("codim_vertices"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    surf_edges = info.surf_edges().viewer().name("surf_edges"),
                    Ps         = info.positions().viewer().name("Ps"),
                    dxs        = info.displacements().viewer().name("dxs"),
                    d_hats     = info.d_hats().viewer().name("d_hats"),
                    alpha      = info.alpha(),
                    eta,
                    max_iter,
                    large_enough_toi] __device__(int i) mutable
                   {
                       auto     indices = CodimP_AllE_pairs(i);
                       IndexT   V       = codim_vertices(indices(0));
                       Vector2i E       = surf_edges(indices(1));

                       Float thickness = PE_thickness(
                           thicknesses(V), thicknesses(E(0)), thicknesses(E(1)));
                       Float d_hat = PE_d_hat(d_hats(V), d_hats(E(0)), d_hats(E(1)));

                       Vector3 VP  = Ps(V);
                       Vector3 dVP = alpha * dxs(V);

                       Vector3 EP0  = Ps(E[0]);
                       Vector3 EP1  = Ps(E[1]);
                       Vector3 dEP0 = alpha * dxs(E[0]);
                       Vector3 dEP1 = alpha * dxs(E[1]);

                       Float toi = large_enough_toi;

                       bool faraway = !distance::point_edge_ccd_broadphase(
                           VP, EP0, EP1, dVP, dEP0, dEP1, d_hat + thickness);

                       if(faraway)
                       {
                           PE_tois(i) = toi;
                           return;
                       }

                       bool hit = distance::point_edge_ccd(
                           VP, EP0, EP1, dVP, dEP0, dEP1, eta, thickness, max_iter, toi);

                       if(!hit)
                           toi = large_enough_toi;

                       PE_tois(i) = toi;
                   });
    }

    // AllP and AllT
    {
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(candidate_AllP_AllT_pairs.size(),
                   [PT_tois = PT_tois.viewer().name("PT_tois"),
                    PT_pairs = candidate_AllP_AllT_pairs.viewer().name("PT_pairs"),
                    surf_vertices = info.surf_vertices().viewer().name("surf_vertices"),
                    surf_triangles = info.surf_triangles().viewer().name("surf_triangles"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    Ps     = info.positions().viewer().name("Ps"),
                    dxs    = info.displacements().viewer().name("dxs"),
                    d_hats = info.d_hats().viewer().name("d_hats"),
                    alpha  = info.alpha(),
                    eta,
                    max_iter,
                    large_enough_toi] __device__(int i) mutable
                   {
                       auto     indices = PT_pairs(i);
                       IndexT   V       = surf_vertices(indices(0));
                       Vector3i F       = surf_triangles(indices(1));

                       Float thickness = PT_thickness(thicknesses(V),
                                                      thicknesses(F(0)),
                                                      thicknesses(F(1)),
                                                      thicknesses(F(2)));
                       Float d_hat =
                           PT_d_hat(d_hats(V), d_hats(F(0)), d_hats(F(1)), d_hats(F(2)));

                       Vector3 VP  = Ps(V);
                       Vector3 dVP = alpha * dxs(V);

                       Vector3 FP0 = Ps(F[0]);
                       Vector3 FP1 = Ps(F[1]);
                       Vector3 FP2 = Ps(F[2]);

                       Vector3 dFP0 = alpha * dxs(F[0]);
                       Vector3 dFP1 = alpha * dxs(F[1]);
                       Vector3 dFP2 = alpha * dxs(F[2]);

                       Float toi = large_enough_toi;


                       bool faraway = !distance::point_triangle_ccd_broadphase(
                           VP, FP0, FP1, FP2, dVP, dFP0, dFP1, dFP2, d_hat + thickness);

                       if(faraway)
                       {
                           PT_tois(i) = toi;
                           return;
                       }

                       bool hit = distance::point_triangle_ccd(
                           VP, FP0, FP1, FP2, dVP, dFP0, dFP1, dFP2, eta, thickness, max_iter, toi);

                       if(!hit)
                           toi = large_enough_toi;

                       PT_tois(i) = toi;
                   });
    }

    // AllE and AllE
    {
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(candidate_AllE_AllE_pairs.size(),
                   [EE_tois = EE_tois.viewer().name("EE_tois"),
                    EE_pairs = candidate_AllE_AllE_pairs.viewer().name("EE_pairs"),
                    surf_edges = info.surf_edges().viewer().name("surf_edges"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    Ps     = info.positions().viewer().name("Ps"),
                    dxs    = info.displacements().viewer().name("dxs"),
                    d_hats = info.d_hats().viewer().name("d_hats"),
                    alpha  = info.alpha(),
                    eta,
                    max_iter,
                    large_enough_toi] __device__(int i) mutable
                   {
                       auto     indices = EE_pairs(i);
                       Vector2i E0      = surf_edges(indices(0));
                       Vector2i E1      = surf_edges(indices(1));

                       Float thickness = EE_thickness(thicknesses(E0(0)),
                                                      thicknesses(E0(1)),
                                                      thicknesses(E1(0)),
                                                      thicknesses(E1(1)));

                       Float d_hat = EE_d_hat(
                           d_hats(E0(0)), d_hats(E0(1)), d_hats(E1(0)), d_hats(E1(1)));


                       Vector3 EP0  = Ps(E0[0]);
                       Vector3 EP1  = Ps(E0[1]);
                       Vector3 dEP0 = alpha * dxs(E0[0]);
                       Vector3 dEP1 = alpha * dxs(E0[1]);

                       Vector3 EP2  = Ps(E1[0]);
                       Vector3 EP3  = Ps(E1[1]);
                       Vector3 dEP2 = alpha * dxs(E1[0]);
                       Vector3 dEP3 = alpha * dxs(E1[1]);

                       Float toi = large_enough_toi;

                       bool faraway = !distance::edge_edge_ccd_broadphase(
                           // position
                           EP0,
                           EP1,
                           EP2,
                           EP3,
                           // displacement
                           dEP0,
                           dEP1,
                           dEP2,
                           dEP3,
                           d_hat + thickness);

                       if(faraway)
                       {
                           EE_tois(i) = toi;
                           return;
                       }

                       bool hit = distance::edge_edge_ccd(
                           // position
                           EP0,
                           EP1,
                           EP2,
                           EP3,
                           // displacement
                           dEP0,
                           dEP1,
                           dEP2,
                           dEP3,
                           eta,
                           thickness,
                           max_iter,
                           toi);

                       if(!hit)
                           toi = large_enough_toi;

                       EE_tois(i) = toi;
                   });
    }

    if(tois.size())
    {
        // get min toi
        DeviceReduce().Min(tois.data(), info.toi().data(), tois.size());
    }
    else
    {
        info.toi().fill(large_enough_toi);
    }
}
}  // namespace uipc::backend::cuda
#endif
