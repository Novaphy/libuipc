#include <finite_element/fem_3d_constitution.h>
#include <finite_element/constitutions/arap_function.h>
#include <finite_element/fem_utils.h>
#include <kernel_cout.h>
#include <muda/ext/eigen/log_proxy.h>
#include <Eigen/Dense>
#include <utils/make_spd.h>
#include <utils/matrix_assembler.h>

namespace uipc::backend::cuda
{
class ARAP3D final : public FEM3DConstitution
{
  public:
    // Constitution UID by libuipc specification
    static constexpr U64   ConstitutionUID = 9;
    static constexpr SizeT StencilSize     = 4;
    static constexpr SizeT HalfHessianSize = StencilSize * (StencilSize + 1) / 2;

    using FEM3DConstitution::FEM3DConstitution;

    vector<Float> h_kappas;

    muda::DeviceBuffer<Float> kappas;

    virtual U64 get_uid() const noexcept override { return ConstitutionUID; }

    virtual void do_build(BuildInfo& info) override {}

    virtual void do_report_extent(ReportExtentInfo& info) override
    {
        info.energy_count(kappas.size());
        info.gradient_count(kappas.size() * StencilSize);
        if(info.gradient_only())
            return;
        info.hessian_count(kappas.size() * HalfHessianSize);
    }

    virtual void do_init(FiniteElementMethod::FilteredInfo& info) override
    {
        using ForEachInfo = FiniteElementMethod::ForEachInfo;

        auto geo_slots = world().scene().geometries();

        auto N = info.primitive_count();

        h_kappas.resize(N);

        info.for_each(
            geo_slots,
            [](geometry::SimplicialComplex& sc) -> auto
            {
                auto kappa = sc.tetrahedra().find<Float>("kappa");
                UIPC_ASSERT(kappa, "Can't find attribute `kappa` on tetrahedra, why can it happen?");
                return kappa->view();
            },
            [&](const ForEachInfo& I, Float kappa)
            { h_kappas[I.global_index()] = kappa; });

        kappas.resize(N);
        kappas.view().copy_from(h_kappas.data());
    }

    virtual void do_compute_energy(ComputeEnergyInfo& info) override
    {
        using namespace muda;
        namespace ARAP = sym::arap_3d;

        // Minimal capture: device pointers + scalars only (no DenseViewer / .name()).
        const int         n     = static_cast<int>(info.indices().size());
        const Float*      pk    = kappas.data();
        Float*            pe    = info.energies().data();
        const Vector4i*   pidx  = info.indices().data();
        const Vector3*    pxs   = info.xs().data();
        const Matrix3x3* pDm  = info.Dm_invs().data();
        const Float*      pvol = info.rest_volumes().data();
        const Float       dt   = info.dt();

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(n,
                   [pk, pe, pidx, pxs, pDm, pvol, dt] __device__(int I)
                   {
                       const Vector4i&  tet    = pidx[I];
                       const Matrix3x3& Dm_inv = pDm[I];

                       const Vector3& x0 = pxs[tet(0)];
                       const Vector3& x1 = pxs[tet(1)];
                       const Vector3& x2 = pxs[tet(2)];
                       const Vector3& x3 = pxs[tet(3)];

                       auto F = fem::F(x0, x1, x2, x3, Dm_inv);

                       Float E;

                       ARAP::E(E, pk[I] * dt * dt, pvol[I], F);
                       pe[I] = E;
                   });
    }

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        using namespace muda;
        namespace ARAP = sym::arap_3d;

        const int       n     = static_cast<int>(info.indices().size());
        const Float*    pk    = kappas.data();
        const Vector4i* pidx  = info.indices().data();
        const Vector3*  pxs   = info.xs().data();
        const Matrix3x3* pDm  = info.Dm_invs().data();
        const Float*    pvol  = info.rest_volumes().data();
        const Float     dt    = info.dt();
        const int       grad_only_i = info.gradient_only() ? 1 : 0;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(n,
                   [pk,
                    pidx,
                    pxs,
                    pDm,
                    pvol,
                    dt,
                    grad_only_i,
                    G3s = info.gradients().viewer().name("gradients"),
                    H3s = info.hessians().viewer().name("hessians")] __device__(int I) mutable
                   {
                       const Vector4i&  tet    = pidx[I];
                       const Matrix3x3& Dm_inv = pDm[I];

                       const Vector3& x0 = pxs[tet(0)];
                       const Vector3& x1 = pxs[tet(1)];
                       const Vector3& x2 = pxs[tet(2)];
                       const Vector3& x3 = pxs[tet(3)];

                       auto F = fem::F(x0, x1, x2, x3, Dm_inv);

                       auto kt2 = pk[I] * dt * dt;
                       auto v   = pvol[I];

                       Vector9 dEdF;
                       ARAP::dEdF(dEdF, kt2, v, F);

                       Matrix9x12 dFdx = fem::dFdx(Dm_inv);

                       // Corex: avoid Eigen transpose / chained expr in device lambda
                       Vector12 G12;
                       for(int r = 0; r < 12; ++r)
                       {
                           Float s = 0;
                           for(int k = 0; k < 9; ++k)
                               s += dFdx(k, r) * dEdF(k);
                           G12(r) = s;
                       }

                       DoubletVectorAssembler DVA{G3s};
                       DVA.template segment<4>(I * 4).write(tet, G12);

                       if(grad_only_i)
                           return;

                       Matrix9x9 ddEddF;
                       ARAP::ddEddF(ddEddF, kt2, v, F);
                       make_spd(ddEddF);

                       Matrix9x12 T_mid;
                       for(int r = 0; r < 9; ++r)
                           for(int c = 0; c < 12; ++c)
                           {
                               Float s = 0;
                               for(int k = 0; k < 9; ++k)
                                   s += ddEddF(r, k) * dFdx(k, c);
                               T_mid(r, c) = s;
                           }

                       Matrix12x12 H12x12;
                       for(int i = 0; i < 12; ++i)
                           for(int j = 0; j < 12; ++j)
                           {
                               Float s = 0;
                               for(int k = 0; k < 9; ++k)
                                   s += dFdx(k, i) * T_mid(k, j);
                               H12x12(i, j) = s;
                           }

                       TripletMatrixAssembler TMA{H3s};
                       TMA.template half_block<4>(I * 10).write(tet, H12x12);
                   });
    }
};

REGISTER_SIM_SYSTEM(ARAP3D);
}  // namespace uipc::backend::cuda
