#include <inter_primitive_effect_system/inter_primitive_constitution.h>
#include <inter_primitive_effect_system/constitutions/soft_vertex_triangle_stitch_function.h>
#include <finite_element/fem_utils.h>
#include <finite_element/matrix_utils.h>
#include <uipc/builtin/attribute_name.h>
#include <utils/matrix_assembler.h>
#include <utils/make_spd.h>
#include <Eigen/LU>

namespace uipc::backend::cuda
{
class SoftVertexTriangleStitch : public InterPrimitiveConstitution
{
  public:
    static constexpr U64   ConstitutionUID = 30;
    static constexpr SizeT StencilSize     = 4;
    static constexpr SizeT HalfHessianSize = StencilSize * (StencilSize + 1) / 2;

    using InterPrimitiveConstitution::InterPrimitiveConstitution;

    U64 get_uid() const noexcept override { return ConstitutionUID; }

    void do_build(BuildInfo& info) override {}

    muda::DeviceBuffer<Vector4i>  topos;
    muda::DeviceBuffer<Float>     mus;
    muda::DeviceBuffer<Float>     lambdas;
    muda::DeviceBuffer<Matrix3x3> Dm_invs;
    muda::DeviceBuffer<Float>     rest_volumes;

    vector<Vector4i>  h_topos;
    vector<Float>     h_mus;
    vector<Float>     h_lambdas;
    vector<Matrix3x3> h_Dm_invs;
    vector<Float>     h_rest_volumes;

    muda::CBufferView<Float>              energies;
    muda::CDoubletVectorView<Float, 3>    gradients;
    muda::CTripletMatrixView<Float, 3, 3> hessians;

    void do_init(FilteredInfo& info) override
    {
        list<Vector4i>  topo_buffer;
        list<Float>     mu_buffer;
        list<Float>     lambda_buffer;
        list<Matrix3x3> Dm_inv_buffer;
        list<Float>     rest_volume_buffer;

        auto geo_slots    = world().scene().geometries();
        using ForEachInfo = InterPrimitiveConstitutionManager::ForEachInfo;
        info.for_each(
            geo_slots,
            [&](const ForEachInfo& I, geometry::Geometry& geo)
            {
                auto topo = geo.instances().find<Vector4i>(builtin::topo);
                UIPC_ASSERT(topo, "SoftVertexTriangleStitch requires attribute `topo` on instances()");
                auto geo_ids = geo.meta().find<Vector2i>("geo_ids");
                UIPC_ASSERT(geo_ids, "SoftVertexTriangleStitch requires attribute `geo_ids` on meta()");
                auto mu_slot = geo.instances().find<Float>("mu");
                UIPC_ASSERT(mu_slot, "SoftVertexTriangleStitch requires attribute `mu` on instances()");
                auto lambda_slot = geo.instances().find<Float>("lambda");
                UIPC_ASSERT(lambda_slot, "SoftVertexTriangleStitch requires attribute `lambda` on instances()");

                Vector2i ids         = geo_ids->view()[0];
                auto     l_slot      = info.geo_slot(ids[0]);
                auto     r_slot      = info.geo_slot(ids[1]);
                auto     l_rest_slot = info.rest_geo_slot(ids[0]);
                auto     r_rest_slot = info.rest_geo_slot(ids[1]);
                UIPC_ASSERT(l_rest_slot,
                            "SoftVertexTriangleStitch requires rest geometry for slot id {}",
                            ids[0]);
                UIPC_ASSERT(r_rest_slot,
                            "SoftVertexTriangleStitch requires rest geometry for slot id {}",
                            ids[1]);

                auto l_geo = l_slot->geometry().as<geometry::SimplicialComplex>();
                UIPC_ASSERT(l_geo,
                            "SoftVertexTriangleStitch requires simplicial complex geometry, but got {} ({})",
                            l_slot->geometry().type(),
                            l_slot->id());
                auto r_geo = r_slot->geometry().as<geometry::SimplicialComplex>();
                UIPC_ASSERT(r_geo,
                            "SoftVertexTriangleStitch requires simplicial complex geometry, but got {} ({})",
                            r_slot->geometry().type(),
                            r_slot->id());
                auto l_rest_geo =
                    l_rest_slot->geometry().as<geometry::SimplicialComplex>();
                UIPC_ASSERT(l_rest_geo,
                            "SoftVertexTriangleStitch requires rest simplicial complex for id {}",
                            ids[0]);
                auto r_rest_geo =
                    r_rest_slot->geometry().as<geometry::SimplicialComplex>();
                UIPC_ASSERT(r_rest_geo,
                            "SoftVertexTriangleStitch requires rest simplicial complex for id {}",
                            ids[1]);

                auto l_offset = l_geo->meta().find<IndexT>(builtin::global_vertex_offset);
                UIPC_ASSERT(l_offset,
                            "SoftVertexTriangleStitch requires attribute `global_vertex_offset` on meta() of geometry {} ({})",
                            l_slot->geometry().type(),
                            l_slot->id());
                IndexT l_offset_v = l_offset->view()[0];
                auto r_offset = r_geo->meta().find<IndexT>(builtin::global_vertex_offset);
                UIPC_ASSERT(r_offset,
                            "SoftVertexTriangleStitch requires attribute `global_vertex_offset` on meta() of geometry {} ({})",
                            r_slot->geometry().type(),
                            r_slot->id());
                IndexT r_offset_v = r_offset->view()[0];

                auto rest0_pos = l_rest_geo->positions().view();
                auto rest1_pos = r_rest_geo->positions().view();

                Transform l_transform(l_rest_geo->transforms().view()[0]);
                Transform r_transform(r_rest_geo->transforms().view()[0]);

                auto min_sep_slot = geo.instances().find<Float>("min_separate_distance");
                UIPC_ASSERT(min_sep_slot, "SoftVertexTriangleStitch requires per-instance attribute `min_separate_distance`");
                auto min_sep_view = min_sep_slot->view();

                auto  topo_view   = topo->view();
                auto  mu_view     = mu_slot->view();
                auto  lambda_view = lambda_slot->view();
                SizeT n           = topo_view.size();

                for(SizeT i = 0; i < n; ++i)
                {
                    const Vector4i& t = topo_view[i];
                    IndexT  v_id = t(0), tri0 = t(1), tri1 = t(2), tri2 = t(3);
                    Vector3 x0 = l_transform * rest0_pos[v_id];
                    Vector3 x1 = r_transform * rest1_pos[tri0];
                    Vector3 x2 = r_transform * rest1_pos[tri1];
                    Vector3 x3 = r_transform * rest1_pos[tri2];

                    Float   d  = min_sep_view[i];
                    Vector3 e1 = x2 - x1, e2 = x3 - x1;
                    Vector3 normal = e1.cross(e2);
                    constexpr Float geo_degeneracy_tol = 1e-12;

                    Float   nrm    = normal.norm();
                    UIPC_ASSERT(nrm >= geo_degeneracy_tol,
                                "SoftVertexTriangleStitch: triangle ({},{},{}) is degenerate",
                                tri0,
                                tri1,
                                tri2);
                    normal /= nrm;

                    Float signed_dist = normal.dot(x0 - x1);
                    if(std::abs(signed_dist) < d)
                    {
                        Float sign = (signed_dist >= 0) ? 1.0 : -1.0;
                        x0         = x0 + (sign * d - signed_dist) * normal;
                    }

                    Matrix3x3 Dm;
                    Dm.col(0)      = x1 - x0;
                    Dm.col(1)      = x2 - x0;
                    Dm.col(2)      = x3 - x0;
                    Float rest_vol = (1.0 / 6.0) * std::abs(Dm.determinant());

                    topo_buffer.push_back(Vector4i{t(0) + l_offset_v,
                                                   t(1) + r_offset_v,
                                                   t(2) + r_offset_v,
                                                   t(3) + r_offset_v});
                    mu_buffer.push_back(mu_view[i]);
                    lambda_buffer.push_back(lambda_view[i]);
                    Dm_inv_buffer.push_back(Dm.inverse());
                    rest_volume_buffer.push_back(rest_vol);
                }
            });

        h_topos.assign(topo_buffer.begin(), topo_buffer.end());
        h_mus.assign(mu_buffer.begin(), mu_buffer.end());
        h_lambdas.assign(lambda_buffer.begin(), lambda_buffer.end());
        h_Dm_invs.assign(Dm_inv_buffer.begin(), Dm_inv_buffer.end());
        h_rest_volumes.assign(rest_volume_buffer.begin(), rest_volume_buffer.end());

        topos.copy_from(h_topos);
        mus.copy_from(h_mus);
        lambdas.copy_from(h_lambdas);
        Dm_invs.copy_from(h_Dm_invs);
        rest_volumes.copy_from(h_rest_volumes);
    }

    void do_report_energy_extent(EnergyExtentInfo& info) override
    {
        info.energy_count(topos.size());
    }

    void do_compute_energy(ComputeEnergyInfo& info) override
    {
        using namespace muda;
        namespace SVTS = sym::soft_vertex_triangle_stitch;

        energies = info.energies();

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(topos.size(),
                   [topos     = topos.cviewer().name("topos"),
                    xs        = info.positions().cviewer().name("xs"),
                    mus       = mus.cviewer().name("mus"),
                    lambdas   = lambdas.cviewer().name("lambdas"),
                    Dm_invs   = Dm_invs.cviewer().name("Dm_invs"),
                    rest_vols = rest_volumes.cviewer().name("rest_volumes"),
                    Es        = info.energies().viewer().name("Es"),
                    dt        = info.dt()] __device__(int I)
                   {
                       const Vector4i& tet = topos(I);
                       Vector3         x0  = xs(tet(0));
                       Vector3         x1  = xs(tet(1));
                       Vector3         x2  = xs(tet(2));
                       Vector3         x3  = xs(tet(3));

                       Matrix3x3 F    = fem::F(x0, x1, x2, x3, Dm_invs(I));
                       Vector9   VecF = flatten(F);

                       Float E_val;
                       SVTS::E(E_val, mus(I), lambdas(I), VecF);
                       Es(I) = E_val * dt * dt * rest_vols(I);
                   });
    }

    void do_report_gradient_hessian_extent(GradientHessianExtentInfo& info) override
    {
        info.gradient_count(StencilSize * topos.size());

        if(info.gradient_only())
            return;

        info.hessian_count(HalfHessianSize * topos.size());
    }

    void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        using namespace muda;
        namespace SVTS = sym::soft_vertex_triangle_stitch;

        const int       n         = static_cast<int>(topos.size());
        const Vector4i* ptopos    = topos.data();
        const Vector3*  pxs       = info.positions().data();
        const Float*    pmus      = mus.data();
        const Float*    plambdas  = lambdas.data();
        const Matrix3x3* pDm      = Dm_invs.data();
        const Float*    prest_vol = rest_volumes.data();
        const Float     dt        = info.dt();
        const int       grad_only = info.gradient_only() ? 1 : 0;

        const auto gl = info.gradients().device_layout_mut();
        const auto hl = info.hessians().device_layout_mut();

        const int  gl_tsc  = gl.total_segment_count;
        const int  gl_dio  = gl.doublet_index_offset;
        const int  gl_dc   = gl.doublet_count;
        const int  gl_tdc  = gl.total_doublet_count;
        const int  gl_so   = gl.subvector_offset;
        const int  gl_se   = gl.subvector_extent;
        int* const gl_si   = gl.segment_indices;
        Vector3* const gl_sv = gl.segment_values;

        const int        hl_tr  = hl.total_rows;
        const int        hl_tc  = hl.total_cols;
        const int        hl_tio = hl.triplet_index_offset;
        const int        hl_tic = hl.triplet_count;
        const int        hl_ttc = hl.total_triplet_count;
        const int        hl_sox = hl.submatrix_offset.x;
        const int        hl_soy = hl.submatrix_offset.y;
        const int        hl_sex = hl.submatrix_extent.x;
        const int        hl_sey = hl.submatrix_extent.y;
        int* const       hl_ri = hl.row_indices;
        int* const       hl_ci = hl.col_indices;
        Matrix3x3* const hl_va = hl.values;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(n,
                   [ptopos,
                    pxs,
                    pmus,
                    plambdas,
                    pDm,
                    prest_vol,
                    dt,
                    grad_only,
                    gl_tsc,
                    gl_dio,
                    gl_dc,
                    gl_tdc,
                    gl_so,
                    gl_se,
                    gl_si,
                    gl_sv,
                    hl_tr,
                    hl_tc,
                    hl_tio,
                    hl_tic,
                    hl_ttc,
                    hl_sox,
                    hl_soy,
                    hl_sex,
                    hl_sey,
                    hl_ri,
                    hl_ci,
                    hl_va] __device__(int I)
                   {
                       DoubletVectorViewer<Float, 3> G3s{gl_tsc,
                                                       gl_dio,
                                                       gl_dc,
                                                       gl_tdc,
                                                       gl_so,
                                                       gl_se,
                                                       gl_si,
                                                       gl_sv};

                       TripletMatrixViewer<Float, 3, 3> H3x3s{hl_tr,
                                                             hl_tc,
                                                             hl_tio,
                                                             hl_tic,
                                                             hl_ttc,
                                                             int2{hl_sox, hl_soy},
                                                             int2{hl_sex, hl_sey},
                                                             hl_ri,
                                                             hl_ci,
                                                             hl_va};

                       const Vector4i& tet = ptopos[I];
                       Vector3         x0  = pxs[tet(0)];
                       Vector3         x1  = pxs[tet(1)];
                       Vector3         x2  = pxs[tet(2)];
                       Vector3         x3  = pxs[tet(3)];

                       Matrix3x3 F    = fem::F(x0, x1, x2, x3, pDm[I]);
                       Vector9   VecF = flatten(F);

                       Float Vdt2 = prest_vol[I] * dt * dt;

                       Vector9 dEdVecF;
                       SVTS::dEdVecF(dEdVecF, pmus[I], plambdas[I], VecF);
                       dEdVecF *= Vdt2;

                       Matrix9x12 dFdx = fem::dFdx(pDm[I]);
                       Vector12   G;
                       for(int r = 0; r < 12; ++r)
                       {
                           Float s = 0;
                           for(int k = 0; k < 9; ++k)
                               s += dFdx(k, r) * dEdVecF(k);
                           G(r) = s;
                       }

                       DoubletVectorAssembler VA{G3s};
                       VA.template segment<4>(I * 4).write(tet, G);

                       if(grad_only)
                           return;

                       Matrix9x9 ddEddVecF;
                       SVTS::ddEddVecF(ddEddVecF, pmus[I], plambdas[I], VecF);
                       ddEddVecF *= Vdt2;
                       make_spd(ddEddVecF);

                       Matrix9x12 T_mid;
                       for(int r = 0; r < 9; ++r)
                           for(int c = 0; c < 12; ++c)
                           {
                               Float s = 0;
                               for(int k = 0; k < 9; ++k)
                                   s += ddEddVecF(r, k) * dFdx(k, c);
                               T_mid(r, c) = s;
                           }

                       Matrix12x12 H;
                       for(int i = 0; i < 12; ++i)
                           for(int j = 0; j < 12; ++j)
                           {
                               Float s = 0;
                               for(int k = 0; k < 9; ++k)
                                   s += dFdx(k, i) * T_mid(k, j);
                               H(i, j) = s;
                           }
                       TripletMatrixAssembler MA{H3x3s};
                       MA.template half_block<4>(I * 10).write(tet, H);
                   });
    }
};

REGISTER_SIM_SYSTEM(SoftVertexTriangleStitch);
}  // namespace uipc::backend::cuda
