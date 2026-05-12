#include <finite_element/fem_3d_constitution.h>
#include <finite_element/constitutions/stable_neo_hookean_3d_function.h>
#include <finite_element/fem_utils.h>
#include <kernel_cout.h>
#include <muda/ext/eigen/log_proxy.h>
#include <Eigen/Dense>
#include <muda/ext/eigen/evd.h>
#include <utils/make_spd.h>
#include <utils/matrix_assembler.h>

namespace uipc::backend::cuda
{
class StableNeoHookean3D final : public FEM3DConstitution
{
  public:
    // Constitution UID by libuipc specification
    static constexpr U64   ConstitutionUID = 10;
    static constexpr SizeT StencilSize     = 4;
    static constexpr SizeT HalfHessianSize = StencilSize * (StencilSize + 1) / 2;

    using FEM3DConstitution::FEM3DConstitution;

    vector<Float> h_mus;
    vector<Float> h_lambdas;

    muda::DeviceBuffer<Float> mus;
    muda::DeviceBuffer<Float> lambdas;

    virtual U64 get_uid() const noexcept override { return ConstitutionUID; }

    virtual void do_build(BuildInfo& info) override {}

    virtual void do_report_extent(ReportExtentInfo& info) override
    {
        info.energy_count(mus.size());
        info.gradient_count(mus.size() * StencilSize);

        if(info.gradient_only())
            return;

        info.hessian_count(mus.size() * HalfHessianSize);
    }

    virtual void do_init(FiniteElementMethod::FilteredInfo& info) override
    {
        using ForEachInfo = FiniteElementMethod::ForEachInfo;

        auto geo_slots = world().scene().geometries();

        auto N = info.primitive_count();

        h_mus.resize(N);
        h_lambdas.resize(N);

        info.for_each(
            geo_slots,
            [](geometry::SimplicialComplex& sc) -> auto
            {
                auto mu     = sc.tetrahedra().find<Float>("mu");
                auto lambda = sc.tetrahedra().find<Float>("lambda");

                return zip(mu->view(), lambda->view());
            },
            [&](const ForEachInfo& I, auto mu_and_lambda)
            {
                auto&& [mu, lambda] = mu_and_lambda;

                auto vI = I.global_index();

                h_mus[vI]     = mu;
                h_lambdas[vI] = lambda;
            });

        mus.resize(N);
        mus.view().copy_from(h_mus.data());

        lambdas.resize(N);
        lambdas.view().copy_from(h_lambdas.data());
    }

    virtual void do_compute_energy(ComputeEnergyInfo& info) override
    {
        using namespace muda;
        namespace SNH = sym::stable_neo_hookean_3d;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.indices().size(),
                   [mus      = mus.cviewer().name("mus"),
                    lambdas  = lambdas.cviewer().name("lambdas"),
                    energies = info.energies().viewer().name("energies"),
                    indices  = info.indices().viewer().name("indices"),
                    xs       = info.xs().viewer().name("xs"),
                    Dm_invs  = info.Dm_invs().viewer().name("Dm_invs"),
                    volumes  = info.rest_volumes().viewer().name("volumes"),
                    dt       = info.dt()] __device__(int I)
                   {
                       const Vector4i&  tet    = indices(I);
                       const Matrix3x3& Dm_inv = Dm_invs(I);
                       Float            mu     = mus(I);
                       Float            lambda = lambdas(I);

                       const Vector3& x0 = xs(tet(0));
                       const Vector3& x1 = xs(tet(1));
                       const Vector3& x2 = xs(tet(2));
                       const Vector3& x3 = xs(tet(3));

                       auto F = fem::F(x0, x1, x2, x3, Dm_inv);

                       auto J = F.determinant();

                       //auto VecF = flatten(F);

                       Float E;

                       SNH::E(E, mu, lambda, F);
                       E *= dt * dt * volumes(I);
                       energies(I) = E;
                   });
    }

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        using namespace muda;
        namespace SNH = sym::stable_neo_hookean_3d;

        const int       n            = static_cast<int>(info.indices().size());
        const Float*    pmus         = mus.data();
        const Float*    plambdas     = lambdas.data();
        const Vector4i* pidx         = info.indices().data();
        const Vector3*  pxs          = info.xs().data();
        const Matrix3x3* pDm         = info.Dm_invs().data();
        const Float*    pvol         = info.rest_volumes().data();
        const Float     dt           = info.dt();
        const int       grad_only_i  = info.gradient_only() ? 1 : 0;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(n,
                   [pmus,
                    plambdas,
                    pidx,
                    pxs,
                    pDm,
                    pvol,
                    dt,
                    grad_only_i,
                    G3s   = info.gradients().viewer().name("gradients"),
                    H3x3s = info.hessians().viewer().name("hessians")] __device__(int I) mutable
                   {
                       const Vector4i&  tet    = pidx[I];
                       const Matrix3x3& Dm_inv = pDm[I];
                       Float            mu     = pmus[I];
                       Float            lambda = plambdas[I];

                       const Vector3& x0 = pxs[tet(0)];
                       const Vector3& x1 = pxs[tet(1)];
                       const Vector3& x2 = pxs[tet(2)];
                       const Vector3& x3 = pxs[tet(3)];

                       auto F = fem::F(x0, x1, x2, x3, Dm_inv);

                       auto Vdt2 = pvol[I] * dt * dt;

                       Matrix3x3 dEdF;
                       SNH::dEdVecF(dEdF, mu, lambda, F);
                       auto VecdEdF = flatten(dEdF);
                       VecdEdF *= Vdt2;

                       Matrix9x12 dFdx = fem::dFdx(Dm_inv);
                       Vector12   G;
                       for(int r = 0; r < 12; ++r)
                       {
                           Float s = 0;
                           for(int k = 0; k < 9; ++k)
                               s += dFdx(k, r) * VecdEdF(k);
                           G(r) = s;
                       }

                       DoubletVectorAssembler DVA{G3s};
                       DVA.template segment<4>(I * 4).write(tet, G);

                       if(grad_only_i)
                           return;

                       Matrix9x9 ddEddF;
                       SNH::ddEddVecF(ddEddF, mu, lambda, F);
                       ddEddF *= Vdt2;
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

                       Matrix12x12 H;
                       for(int i = 0; i < 12; ++i)
                           for(int j = 0; j < 12; ++j)
                           {
                               Float s = 0;
                               for(int k = 0; k < 9; ++k)
                                   s += dFdx(k, i) * T_mid(k, j);
                               H(i, j) = s;
                           }

                       TripletMatrixAssembler TMA{H3x3s};
                       TMA.template half_block<4>(I * 10).write(tet, H);
                   });
    }
};

REGISTER_SIM_SYSTEM(StableNeoHookean3D);
}  // namespace uipc::backend::cuda
