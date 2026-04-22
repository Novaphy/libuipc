#pragma once
#include <type_define.h>
#include <contact_system/contact_coeff.h>
#include <contact_system/contact_models/codim_ipc_contact_function.h>

namespace uipc::backend::cuda
{

#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
namespace corex_numgrad
{
    template <size_t N>
    inline __device__ Float finite_diff_step(const Vector3 (&verts)[N], Float base_step)
    {
        Float scale = Float(1);
        for(size_t i = 0; i < N; ++i)
        {
            Float vertex_scale = verts[i].cwiseAbs().maxCoeff();
            if(vertex_scale > scale)
                scale = vertex_scale;
        }
        return base_step * scale;
    }

    // Numerical gradient of PT squared distance via central differences.
    // Only relies on the scalar distance2 function (which is numerically robust
    // on CoreX), avoiding the analytical gradient that suffers from
    // catastrophic cancellation with float-level double precision.
    inline __device__ void PT_distance2_numgrad(
        const Vector4i& flag,
        const Vector3& P, const Vector3& T0, const Vector3& T1, const Vector3& T2,
        Float& D_out, Vector12& GradD)
    {
        using namespace distance;
        point_triangle_distance2(flag, P, T0, T1, T2, D_out);

        Vector3 verts[4] = {P, T0, T1, T2};
        const Float eps = finite_diff_step(verts, Float(1e-4));
        for(int v = 0; v < 4; ++v)
        {
            for(int d = 0; d < 3; ++d)
            {
                Vector3 v_p[4] = {verts[0], verts[1], verts[2], verts[3]};
                Vector3 v_m[4] = {verts[0], verts[1], verts[2], verts[3]};
                v_p[v](d) += eps;
                v_m[v](d) -= eps;
                Float Dp, Dm;
                point_triangle_distance2(flag, v_p[0], v_p[1], v_p[2], v_p[3], Dp);
                point_triangle_distance2(flag, v_m[0], v_m[1], v_m[2], v_m[3], Dm);
                GradD(v * 3 + d) = (Dp - Dm) / (Float(2) * eps);
            }
        }
    }

    inline __device__ void PT_distance2_numhess(
        const Vector4i& flag,
        const Vector3& P, const Vector3& T0, const Vector3& T1, const Vector3& T2,
        Matrix12x12& HessD)
    {
        Vector3 verts[4] = {P, T0, T1, T2};
        const Float eps = finite_diff_step(verts, Float(1e-3));
        for(int j = 0; j < 12; ++j)
        {
            int vj = j / 3, dj = j % 3;
            Vector3 verts_p[4] = {verts[0], verts[1], verts[2], verts[3]};
            Vector3 verts_m[4] = {verts[0], verts[1], verts[2], verts[3]};
            verts_p[vj](dj) += eps;
            verts_m[vj](dj) -= eps;
            Float Dp_dummy;
            Vector12 Gp, Gm;
            PT_distance2_numgrad(flag, verts_p[0], verts_p[1], verts_p[2], verts_p[3], Dp_dummy, Gp);
            PT_distance2_numgrad(flag, verts_m[0], verts_m[1], verts_m[2], verts_m[3], Dp_dummy, Gm);
            HessD.col(j) = (Gp - Gm) / (Float(2) * eps);
        }
        HessD = (HessD + HessD.transpose()) * 0.5;
    }

    inline __device__ void EE_distance2_numgrad(
        const Vector4i& flag,
        const Vector3& Ea0, const Vector3& Ea1, const Vector3& Eb0, const Vector3& Eb1,
        Float& D_out, Vector12& GradD)
    {
        using namespace distance;
        edge_edge_distance2(flag, Ea0, Ea1, Eb0, Eb1, D_out);

        Vector3 verts[4] = {Ea0, Ea1, Eb0, Eb1};
        const Float eps = finite_diff_step(verts, Float(1e-4));
        for(int v = 0; v < 4; ++v)
        {
            for(int d = 0; d < 3; ++d)
            {
                Vector3 v_p[4] = {verts[0], verts[1], verts[2], verts[3]};
                Vector3 v_m[4] = {verts[0], verts[1], verts[2], verts[3]};
                v_p[v](d) += eps;
                v_m[v](d) -= eps;
                Float Dp, Dm;
                edge_edge_distance2(flag, v_p[0], v_p[1], v_p[2], v_p[3], Dp);
                edge_edge_distance2(flag, v_m[0], v_m[1], v_m[2], v_m[3], Dm);
                GradD(v * 3 + d) = (Dp - Dm) / (Float(2) * eps);
            }
        }
    }

    inline __device__ void EE_distance2_numhess(
        const Vector4i& flag,
        const Vector3& Ea0, const Vector3& Ea1, const Vector3& Eb0, const Vector3& Eb1,
        Matrix12x12& HessD)
    {
        Vector3 verts[4] = {Ea0, Ea1, Eb0, Eb1};
        const Float eps = finite_diff_step(verts, Float(1e-3));
        for(int j = 0; j < 12; ++j)
        {
            int vj = j / 3, dj = j % 3;
            Vector3 verts_p[4] = {verts[0], verts[1], verts[2], verts[3]};
            Vector3 verts_m[4] = {verts[0], verts[1], verts[2], verts[3]};
            verts_p[vj](dj) += eps;
            verts_m[vj](dj) -= eps;
            Float Dp_dummy;
            Vector12 Gp, Gm;
            EE_distance2_numgrad(flag, verts_p[0], verts_p[1], verts_p[2], verts_p[3], Dp_dummy, Gp);
            EE_distance2_numgrad(flag, verts_m[0], verts_m[1], verts_m[2], verts_m[3], Dp_dummy, Gm);
            HessD.col(j) = (Gp - Gm) / (Float(2) * eps);
        }
        HessD = (HessD + HessD.transpose()) * 0.5;
    }
    inline __device__ void PE_distance2_numgrad(
        const Vector3i& flag,
        const Vector3& P, const Vector3& E0, const Vector3& E1,
        Float& D_out, Vector9& GradD)
    {
        using namespace distance;
        point_edge_distance2(flag, P, E0, E1, D_out);

        Vector3 verts[3] = {P, E0, E1};
        const Float eps = finite_diff_step(verts, Float(1e-4));
        for(int v = 0; v < 3; ++v)
        {
            for(int d = 0; d < 3; ++d)
            {
                Vector3 v_p[3] = {verts[0], verts[1], verts[2]};
                Vector3 v_m[3] = {verts[0], verts[1], verts[2]};
                v_p[v](d) += eps;
                v_m[v](d) -= eps;
                Float Dp, Dm;
                point_edge_distance2(flag, v_p[0], v_p[1], v_p[2], Dp);
                point_edge_distance2(flag, v_m[0], v_m[1], v_m[2], Dm);
                GradD(v * 3 + d) = (Dp - Dm) / (Float(2) * eps);
            }
        }
    }

    inline __device__ void PE_distance2_numhess(
        const Vector3i& flag,
        const Vector3& P, const Vector3& E0, const Vector3& E1,
        Matrix9x9& HessD)
    {
        Vector3 verts[3] = {P, E0, E1};
        const Float eps = finite_diff_step(verts, Float(1e-3));
        for(int j = 0; j < 9; ++j)
        {
            int vj = j / 3, dj = j % 3;
            Vector3 verts_p[3] = {verts[0], verts[1], verts[2]};
            Vector3 verts_m[3] = {verts[0], verts[1], verts[2]};
            verts_p[vj](dj) += eps;
            verts_m[vj](dj) -= eps;
            Float Dp_dummy;
            Vector9 Gp, Gm;
            PE_distance2_numgrad(flag, verts_p[0], verts_p[1], verts_p[2], Dp_dummy, Gp);
            PE_distance2_numgrad(flag, verts_m[0], verts_m[1], verts_m[2], Dp_dummy, Gm);
            HessD.col(j) = (Gp - Gm) / (Float(2) * eps);
        }
        HessD = (HessD + HessD.transpose()) * 0.5;
    }

    inline __device__ void PP_distance2_numgrad(
        const Vector2i& flag,
        const Vector3& P0, const Vector3& P1,
        Float& D_out, Vector6& GradD)
    {
        using namespace distance;
        point_point_distance2(flag, P0, P1, D_out);

        Vector3 verts[2] = {P0, P1};
        const Float eps = finite_diff_step(verts, Float(1e-4));
        for(int v = 0; v < 2; ++v)
        {
            for(int d = 0; d < 3; ++d)
            {
                Vector3 v_p[2] = {verts[0], verts[1]};
                Vector3 v_m[2] = {verts[0], verts[1]};
                v_p[v](d) += eps;
                v_m[v](d) -= eps;
                Float Dp, Dm;
                point_point_distance2(flag, v_p[0], v_p[1], Dp);
                point_point_distance2(flag, v_m[0], v_m[1], Dm);
                GradD(v * 3 + d) = (Dp - Dm) / (Float(2) * eps);
            }
        }
    }

    inline __device__ void PP_distance2_numhess(
        const Vector2i& flag,
        const Vector3& P0, const Vector3& P1,
        Matrix6x6& HessD)
    {
        Vector3 verts[2] = {P0, P1};
        const Float eps = finite_diff_step(verts, Float(1e-3));
        for(int j = 0; j < 6; ++j)
        {
            int vj = j / 3, dj = j % 3;
            Vector3 verts_p[2] = {verts[0], verts[1]};
            Vector3 verts_m[2] = {verts[0], verts[1]};
            verts_p[vj](dj) += eps;
            verts_m[vj](dj) -= eps;
            Float Dp_dummy;
            Vector6 Gp, Gm;
            PP_distance2_numgrad(flag, verts_p[0], verts_p[1], Dp_dummy, Gp);
            PP_distance2_numgrad(flag, verts_m[0], verts_m[1], Dp_dummy, Gm);
            HessD.col(j) = (Gp - Gm) / (Float(2) * eps);
        }
        HessD = (HessD + HessD.transpose()) * 0.5;
    }
}  // namespace corex_numgrad
#endif

namespace sym::codim_ipc_simplex_contact
{
    inline __device__ Float PT_kappa(const muda::CDense2D<ContactCoeff>& table,
                                     const Vector4i&                     cids)
    {
        Float kappa = 0.0;
        for(int j = 1; j < 4; ++j)
        {
            ContactCoeff coeff = table(cids[0], cids[j]);
            kappa += coeff.kappa;
        }
        return kappa / Float(3);
    }

    inline __device__ Float EE_kappa(const muda::CDense2D<ContactCoeff>& table,
                                     const Vector4i&                     cids)
    {
        Float kappa = 0.0;
        for(int j = 0; j < 2; ++j)
        {
            for(int k = 2; k < 4; ++k)
            {
                ContactCoeff coeff = table(cids[j], cids[k]);
                kappa += coeff.kappa;
            }
        }
        return kappa / Float(4);
    }

    inline __device__ Float PE_kappa(const muda::CDense2D<ContactCoeff>& table,
                                     const Vector3i&                     cids)
    {
        Float kappa = 0.0;
        for(int j = 1; j < 3; ++j)
        {
            ContactCoeff coeff = table(cids[0], cids[j]);
            kappa += coeff.kappa;
        }
        return kappa / Float(2);
    }

    inline __device__ Float PP_kappa(const muda::CDense2D<ContactCoeff>& table,
                                     const Vector2i&                     cids)
    {
        ContactCoeff coeff = table(cids[0], cids[1]);
        return coeff.kappa;
    }


    inline __device__ Float PT_barrier_energy(Float          kappa,
                                              Float          d_hat,
                                              Float          thickness,
                                              const Vector3& P,
                                              const Vector3& T0,
                                              const Vector3& T1,
                                              const Vector3& T2)
    {
        using namespace codim_ipc_contact;
        using namespace distance;
        Float D;
        point_triangle_distance2(P, T0, T1, T2, D);
        Float B;
        KappaBarrier(B, kappa, D, d_hat, thickness);
        return B;
    }

    inline __device__ Float PT_barrier_energy(const Vector4i& flag,
                                              Float           kappa,
                                              Float           d_hat,
                                              Float           thickness,
                                              const Vector3&  P,
                                              const Vector3&  T0,
                                              const Vector3&  T1,
                                              const Vector3&  T2)
    {
        using namespace codim_ipc_contact;
        using namespace distance;
        Float D;
        point_triangle_distance2(flag, P, T0, T1, T2, D);
        Float B;
        KappaBarrier(B, kappa, D, d_hat, thickness);
        return B;
    }

    inline __device__ void PT_barrier_gradient_hessian(Vector12&      G,
                                                       Matrix12x12&   H,
                                                       Float          kappa,
                                                       Float          d_hat,
                                                       Float          thickness,
                                                       const Vector3& P,
                                                       const Vector3& T0,
                                                       const Vector3& T1,
                                                       const Vector3& T2)
    {
        using namespace codim_ipc_contact;
        using namespace distance;

        Float D;
        point_triangle_distance2(P, T0, T1, T2, D);

        Vector12 GradD;
        point_triangle_distance2_gradient(P, T0, T1, T2, GradD);

        Float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        //tex:
        //$$
        // G = \frac{\partial B}{\partial D} \frac{\partial D}{\partial x}
        //$$
        G = dBdD * GradD;

        Float ddBddD;
        ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness);

        Matrix12x12 HessD;
        point_triangle_distance2_hessian(P, T0, T1, T2, HessD);

        //tex:
        //$$
        // H = \frac{\partial^2 B}{\partial D^2} \frac{\partial D}{\partial x} \frac{\partial D}{\partial x}^T + \frac{\partial B}{\partial D} \frac{\partial^2 D}{\partial x^2}
        //$$
        H = ddBddD * GradD * GradD.transpose() + dBdD * HessD;
    }

    inline __device__ void PT_barrier_gradient(Vector12&       G,
                                               const Vector4i& flag,
                                               Float           kappa,
                                               Float           d_hat,
                                               Float           thickness,
                                               const Vector3&  P,
                                               const Vector3&  T0,
                                               const Vector3&  T1,
                                               const Vector3&  T2)
    {
        using namespace codim_ipc_contact;
        using namespace distance;

        Float D;
        Vector12 GradD;
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT && \
    !(defined(UIPC_FLOAT_SCALAR) && UIPC_FLOAT_SCALAR)
        corex_numgrad::PT_distance2_numgrad(flag, P, T0, T1, T2, D, GradD);
#else
        point_triangle_distance2(flag, P, T0, T1, T2, D);
        point_triangle_distance2_gradient(flag, P, T0, T1, T2, GradD);
#endif

        Float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        G = dBdD * GradD;
    }

    inline __device__ void PT_barrier_gradient_hessian(Vector12&       G,
                                                       Matrix12x12&    H,
                                                       const Vector4i& flag,
                                                       Float           kappa,
                                                       Float           d_hat,
                                                       Float          thickness,
                                                       const Vector3& P,
                                                       const Vector3& T0,
                                                       const Vector3& T1,
                                                       const Vector3& T2)
    {
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT && \
    !(defined(UIPC_FLOAT_SCALAR) && UIPC_FLOAT_SCALAR)
        PT_barrier_gradient(G, flag, kappa, d_hat, thickness, P, T0, T1, T2);

        Vector3 verts[4] = {P, T0, T1, T2};
        const Float eps  = corex_numgrad::finite_diff_step(verts, Float(1e-3));
        for(int j = 0; j < 12; ++j)
        {
            int vj = j / 3, dj = j % 3;
            Vector3 vp[4] = {verts[0], verts[1], verts[2], verts[3]};
            Vector3 vm[4] = {verts[0], verts[1], verts[2], verts[3]};
            vp[vj](dj) += eps;
            vm[vj](dj) -= eps;
            Vector12 Gp, Gm;
            PT_barrier_gradient(Gp, flag, kappa, d_hat, thickness, vp[0], vp[1], vp[2], vp[3]);
            PT_barrier_gradient(Gm, flag, kappa, d_hat, thickness, vm[0], vm[1], vm[2], vm[3]);
            H.col(j) = (Gp - Gm) / (Float(2) * eps);
        }
        H = (H + H.transpose()) * Float(0.5);
#else
        using namespace codim_ipc_contact;
        using namespace distance;

        Float D;
        point_triangle_distance2(flag, P, T0, T1, T2, D);

        Vector12 GradD;
        point_triangle_distance2_gradient(flag, P, T0, T1, T2, GradD);

        Float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);
        G = dBdD * GradD;

        Float ddBddD;
        ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness);

        Matrix12x12 HessD;
        point_triangle_distance2_hessian(flag, P, T0, T1, T2, HessD);
        H = ddBddD * GradD * GradD.transpose() + dBdD * HessD;
#endif
    }


    inline __device__ Float mollified_EE_barrier_energy(const Vector4i& flag,
                                                        Float           kappa,
                                                        Float           d_hat,
                                                        Float thickness,
                                                        const Vector3& t0_Ea0,
                                                        const Vector3& t0_Ea1,
                                                        const Vector3& t0_Eb0,
                                                        const Vector3& t0_Eb1,
                                                        const Vector3& Ea0,
                                                        const Vector3& Ea1,
                                                        const Vector3& Eb0,
                                                        const Vector3& Eb1)
    {
        // using mollifier to improve the smoothness of the edge-edge barrier
        using namespace codim_ipc_contact;
        using namespace distance;
        Float D;
        edge_edge_distance2(flag, Ea0, Ea1, Eb0, Eb1, D);
        Float B;
        KappaBarrier(B, kappa, D, d_hat, thickness);

        Float eps_x;
        edge_edge_mollifier_threshold(
            t0_Ea0, t0_Ea1, t0_Eb0, t0_Eb1, static_cast<Float>(1e-3), eps_x);

        Float ek;
        edge_edge_mollifier(Ea0, Ea1, Eb0, Eb1, eps_x, ek);

        return ek * B;
    }

    inline __device__ void mollified_EE_barrier_gradient(Vector12&       G,
                                                         const Vector4i& flag,
                                                         Float           kappa,
                                                         Float           d_hat,
                                                         Float           thickness,
                                                         const Vector3&  t0_Ea0,
                                                         const Vector3&  t0_Ea1,
                                                         const Vector3&  t0_Eb0,
                                                         const Vector3&  t0_Eb1,
                                                         const Vector3&  Ea0,
                                                         const Vector3&  Ea1,
                                                         const Vector3&  Eb0,
                                                         const Vector3&  Eb1)
    {
        using namespace codim_ipc_contact;
        using namespace distance;

        Float D;
        Vector12 GradD;
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT && \
    !(defined(UIPC_FLOAT_SCALAR) && UIPC_FLOAT_SCALAR)
        corex_numgrad::EE_distance2_numgrad(flag, Ea0, Ea1, Eb0, Eb1, D, GradD);
#else
        edge_edge_distance2(flag, Ea0, Ea1, Eb0, Eb1, D);
        edge_edge_distance2_gradient(flag, Ea0, Ea1, Eb0, Eb1, GradD);
#endif

        Float B;
        KappaBarrier(B, kappa, D, d_hat, thickness);

        Float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        Vector12 GradB = dBdD * GradD;

        Float eps_x;
        edge_edge_mollifier_threshold(
            t0_Ea0, t0_Ea1, t0_Eb0, t0_Eb1, static_cast<Float>(1e-3), eps_x);

        Float ek;
        edge_edge_mollifier(Ea0, Ea1, Eb0, Eb1, eps_x, ek);

        Vector12 Gradek;
        edge_edge_mollifier_gradient(Ea0, Ea1, Eb0, Eb1, eps_x, Gradek);

        G = Gradek * B + ek * GradB;
    }

    inline __device__ void mollified_EE_barrier_gradient_hessian(Vector12&    G,
                                                                 Matrix12x12& H,
                                                                 const Vector4i& flag,
                                                                 Float kappa,
                                                                 Float d_hat,
                                                                 Float thickness,
                                                                 const Vector3& t0_Ea0,
                                                                 const Vector3& t0_Ea1,
                                                                 const Vector3& t0_Eb0,
                                                                 const Vector3& t0_Eb1,
                                                                 const Vector3& Ea0,
                                                                 const Vector3& Ea1,
                                                                 const Vector3& Eb0,
                                                                 const Vector3& Eb1)
    {
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT && \
    !(defined(UIPC_FLOAT_SCALAR) && UIPC_FLOAT_SCALAR)
        mollified_EE_barrier_gradient(G, flag, kappa, d_hat, thickness,
            t0_Ea0, t0_Ea1, t0_Eb0, t0_Eb1, Ea0, Ea1, Eb0, Eb1);

        Vector3 verts[4] = {Ea0, Ea1, Eb0, Eb1};
        const Float eps  = corex_numgrad::finite_diff_step(verts, Float(1e-3));
        for(int j = 0; j < 12; ++j)
        {
            int vj = j / 3, dj = j % 3;
            Vector3 vp[4] = {verts[0], verts[1], verts[2], verts[3]};
            Vector3 vm[4] = {verts[0], verts[1], verts[2], verts[3]};
            vp[vj](dj) += eps;
            vm[vj](dj) -= eps;
            Vector12 Gp, Gm;
            mollified_EE_barrier_gradient(Gp, flag, kappa, d_hat, thickness,
                t0_Ea0, t0_Ea1, t0_Eb0, t0_Eb1, vp[0], vp[1], vp[2], vp[3]);
            mollified_EE_barrier_gradient(Gm, flag, kappa, d_hat, thickness,
                t0_Ea0, t0_Ea1, t0_Eb0, t0_Eb1, vm[0], vm[1], vm[2], vm[3]);
            H.col(j) = (Gp - Gm) / (Float(2) * eps);
        }
        H = (H + H.transpose()) * Float(0.5);
#else
        using namespace codim_ipc_contact;
        using namespace distance;

        Float D;
        edge_edge_distance2(flag, Ea0, Ea1, Eb0, Eb1, D);

        Vector12 GradD;
        edge_edge_distance2_gradient(flag, Ea0, Ea1, Eb0, Eb1, GradD);

        Matrix12x12 HessD;
        edge_edge_distance2_hessian(flag, Ea0, Ea1, Eb0, Eb1, HessD);

        Float B;
        KappaBarrier(B, kappa, D, d_hat, thickness);

        Float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        Float ddBddD;
        ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness);

        Vector12 GradB = dBdD * GradD;
        Matrix12x12 HessB = ddBddD * GradD * GradD.transpose() + dBdD * HessD;

        Float eps_x;
        edge_edge_mollifier_threshold(
            t0_Ea0, t0_Ea1, t0_Eb0, t0_Eb1, static_cast<Float>(1e-3), eps_x);

        Float ek;
        edge_edge_mollifier(Ea0, Ea1, Eb0, Eb1, eps_x, ek);

        Vector12 Gradek;
        edge_edge_mollifier_gradient(Ea0, Ea1, Eb0, Eb1, eps_x, Gradek);

        Matrix12x12 Hessek;
        edge_edge_mollifier_hessian(Ea0, Ea1, Eb0, Eb1, eps_x, Hessek);

        G = Gradek * B + ek * GradB;
        H = Hessek * B + Gradek * GradB.transpose() + GradB * Gradek.transpose() + ek * HessB;
#endif
    }

    inline __device__ Float PE_barrier_energy(const Vector3i& flag,
                                              Float           kappa,
                                              Float           d_hat,
                                              Float           thickness,
                                              const Vector3&  P,
                                              const Vector3&  E0,
                                              const Vector3&  E1)
    {
        using namespace codim_ipc_contact;
        using namespace distance;
        Float D = 0.0;
        point_edge_distance2(flag, P, E0, E1, D);
        Float E = 0.0;
        KappaBarrier(E, kappa, D, d_hat, thickness);
        return E;
    }

    inline __device__ void PE_barrier_gradient(Vector9&        G,
                                               const Vector3i& flag,
                                               Float           kappa,
                                               Float           d_hat,
                                               Float           thickness,
                                               const Vector3&  P,
                                               const Vector3&  E0,
                                               const Vector3&  E1)
    {
        using namespace codim_ipc_contact;
        using namespace distance;

        Float D;
        Vector9 GradD;
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT && \
    !(defined(UIPC_FLOAT_SCALAR) && UIPC_FLOAT_SCALAR)
        corex_numgrad::PE_distance2_numgrad(flag, P, E0, E1, D, GradD);
#else
        D = 0.0;
        point_edge_distance2(flag, P, E0, E1, D);
        point_edge_distance2_gradient(flag, P, E0, E1, GradD);
#endif

        Float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        G = dBdD * GradD;
    }

    inline __device__ void PE_barrier_gradient_hessian(Vector9&        G,
                                                       Matrix9x9&      H,
                                                       const Vector3i& flag,
                                                       Float           kappa,
                                                       Float           d_hat,
                                                       Float          thickness,
                                                       const Vector3& P,
                                                       const Vector3& E0,
                                                       const Vector3& E1)
    {
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT && \
    !(defined(UIPC_FLOAT_SCALAR) && UIPC_FLOAT_SCALAR)
        PE_barrier_gradient(G, flag, kappa, d_hat, thickness, P, E0, E1);

        Vector3 verts[3] = {P, E0, E1};
        const Float eps  = corex_numgrad::finite_diff_step(verts, Float(1e-3));
        for(int j = 0; j < 9; ++j)
        {
            int vj = j / 3, dj = j % 3;
            Vector3 vp[3] = {verts[0], verts[1], verts[2]};
            Vector3 vm[3] = {verts[0], verts[1], verts[2]};
            vp[vj](dj) += eps;
            vm[vj](dj) -= eps;
            Vector9 Gp, Gm;
            PE_barrier_gradient(Gp, flag, kappa, d_hat, thickness, vp[0], vp[1], vp[2]);
            PE_barrier_gradient(Gm, flag, kappa, d_hat, thickness, vm[0], vm[1], vm[2]);
            H.col(j) = (Gp - Gm) / (Float(2) * eps);
        }
        H = (H + H.transpose()) * Float(0.5);
#else
        using namespace codim_ipc_contact;
        using namespace distance;

        Float D = 0.0;
        point_edge_distance2(flag, P, E0, E1, D);

        Vector9 GradD;
        point_edge_distance2_gradient(flag, P, E0, E1, GradD);

        Matrix9x9 HessD;
        point_edge_distance2_hessian(flag, P, E0, E1, HessD);

        Float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);
        G = dBdD * GradD;

        Float ddBddD;
        ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness);
        H = ddBddD * GradD * GradD.transpose() + dBdD * HessD;
#endif
    }

    inline __device__ Float PP_barrier_energy(const Vector2i& flag,
                                              Float           kappa,
                                              Float           d_hat,
                                              Float           thickness,
                                              const Vector3&  P0,
                                              const Vector3&  P1)
    {
        using namespace codim_ipc_contact;
        using namespace distance;
        Float D = 0.0;
        point_point_distance2(flag, P0, P1, D);
        Float E = 0.0;
        KappaBarrier(E, kappa, D, d_hat, thickness);
        return E;
    }

    inline __device__ void PP_barrier_gradient(Vector6&        G,
                                               const Vector2i& flag,
                                               Float           kappa,
                                               Float           d_hat,
                                               Float           thickness,
                                               const Vector3&  P0,
                                               const Vector3&  P1)
    {
        using namespace codim_ipc_contact;
        using namespace distance;

        Float D;
        Vector6 GradD;
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT && \
    !(defined(UIPC_FLOAT_SCALAR) && UIPC_FLOAT_SCALAR)
        corex_numgrad::PP_distance2_numgrad(flag, P0, P1, D, GradD);
#else
        D = 0.0;
        point_point_distance2(flag, P0, P1, D);
        point_point_distance2_gradient(flag, P0, P1, GradD);
#endif

        Float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        G = dBdD * GradD;
    }

    inline __device__ void PP_barrier_gradient_hessian(Vector6&        G,
                                                       Matrix6x6&      H,
                                                       const Vector2i& flag,
                                                       Float           kappa,
                                                       Float           d_hat,
                                                       Float          thickness,
                                                       const Vector3& P0,
                                                       const Vector3& P1)
    {
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT && \
    !(defined(UIPC_FLOAT_SCALAR) && UIPC_FLOAT_SCALAR)
        PP_barrier_gradient(G, flag, kappa, d_hat, thickness, P0, P1);

        Vector3 verts[2] = {P0, P1};
        const Float eps  = corex_numgrad::finite_diff_step(verts, Float(1e-3));
        for(int j = 0; j < 6; ++j)
        {
            int vj = j / 3, dj = j % 3;
            Vector3 vp[2] = {verts[0], verts[1]};
            Vector3 vm[2] = {verts[0], verts[1]};
            vp[vj](dj) += eps;
            vm[vj](dj) -= eps;
            Vector6 Gp, Gm;
            PP_barrier_gradient(Gp, flag, kappa, d_hat, thickness, vp[0], vp[1]);
            PP_barrier_gradient(Gm, flag, kappa, d_hat, thickness, vm[0], vm[1]);
            H.col(j) = (Gp - Gm) / (Float(2) * eps);
        }
        H = (H + H.transpose()) * Float(0.5);
#else
        using namespace codim_ipc_contact;
        using namespace distance;

        Float D = 0.0;
        point_point_distance2(flag, P0, P1, D);

        Vector6 GradD;
        point_point_distance2_gradient(flag, P0, P1, GradD);

        Matrix6x6 HessD;
        point_point_distance2_hessian(flag, P0, P1, HessD);

        Float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);
        G = dBdD * GradD;

        Float ddBddD;
        ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness);
        H = ddBddD * GradD * GradD.transpose() + dBdD * HessD;
#endif
    }
}  // namespace sym::codim_ipc_simplex_contact
}  // namespace uipc::backend::cuda
