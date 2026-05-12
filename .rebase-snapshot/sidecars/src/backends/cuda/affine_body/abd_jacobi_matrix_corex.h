#pragma once
#include <affine_body/type_define.h>
#include <cuda_runtime_api.h>
#include <muda/muda_def.h>
namespace uipc::backend::cuda
{
//tex: $$ \mathbf{J}_{3 \times 12} $$ or $$ (\mathbf{J}^T)_{12 \times 3} $$
class ABDJacobi  // for every point
{
  public:
    class ABDJacobiT
    {
        const ABDJacobi& m_j;

      public:
        explicit MUDA_HOST MUDA_DEVICE ABDJacobiT(const ABDJacobi& j)
            : m_j(j)
        {
        }
        MUDA_HOST MUDA_DEVICE friend Vector12 operator*(const ABDJacobiT& j, const Vector3& g);

        MUDA_HOST MUDA_DEVICE const auto& J() const { return m_j; }
    };
    MUDA_HOST MUDA_DEVICE ABDJacobi(const Vector3& x_bar)
        : m_x_bar(x_bar)
    {
    }

    MUDA_HOST MUDA_DEVICE ABDJacobi()
        : m_x_bar(Vector3::Zero())
    {
    }

    MUDA_HOST MUDA_DEVICE friend Vector3 operator*(const ABDJacobi& j, const Vector12& q);
    MUDA_HOST MUDA_DEVICE friend Vector12 operator*(const ABDJacobi::ABDJacobiT& j, const Vector3& g);

    MUDA_HOST MUDA_DEVICE Vector3 point_from_affine(const Vector12& q)
    {
        return (*this) * q;
    }

    MUDA_HOST MUDA_DEVICE Vector3 point_x(const Vector12& q) const
    {
        return (*this) * q;
    };

    // without translation, only rotation and scaling
    MUDA_HOST MUDA_DEVICE Vector3 vec_x(const Vector12& q) const;

    MUDA_HOST MUDA_DEVICE Matrix3x12 to_mat() const;

    MUDA_HOST MUDA_DEVICE ABDJacobiT T() const { return ABDJacobiT(*this); }

    MUDA_HOST MUDA_DEVICE const Vector3& x_bar() const { return m_x_bar; }

    //tex: $$ \mathbf{J}^T\mathbf{H}\mathbf{J} $$
    static MUDA_HOST MUDA_DEVICE Matrix12x12 JT_H_J(const ABDJacobiT& lhs_J_T,
                                                    const Matrix3x3&  Hessian,
                                                    const ABDJacobi&  rhs_J);

  private:
    //tex: $$ \bar{\mathbf{x}} $$
    Vector3 m_x_bar;
};

template <size_t N>
class ABDJacobiStack
{
  protected:
    const ABDJacobi* m_jacobis[N];

  public:
    class ABDJacobiStackT
    {
        const ABDJacobiStack& m_origin;

      public:
        MUDA_HOST MUDA_DEVICE ABDJacobiStackT(const ABDJacobiStack& j)
            : m_origin(j)
        {
        }
        MUDA_HOST MUDA_DEVICE Vector12 operator*(const Vector<Float, 3 * N>& g) const;
    };

    MUDA_HOST MUDA_DEVICE Vector<Float, 3 * N> operator*(const Vector12& q) const;

    MUDA_HOST MUDA_DEVICE Matrix<Float, 3 * N, 12> to_mat() const;

    MUDA_HOST MUDA_DEVICE ABDJacobiStackT T() const { return ABDJacobiStackT(*this); }
};

class ABDJacobiStack2 : public ABDJacobiStack<2>
{
  public:
    MUDA_HOST MUDA_DEVICE ABDJacobiStack2(const ABDJacobi& j1, const ABDJacobi& j2)
    {
        m_jacobis[0] = &j1;
        m_jacobis[1] = &j2;
    }
};

class ABDJacobiStack3 : public ABDJacobiStack<3>
{
  public:
    MUDA_HOST MUDA_DEVICE ABDJacobiStack3(const ABDJacobi& j1, const ABDJacobi& j2, const ABDJacobi& j3)
    {
        m_jacobis[0] = &j1;
        m_jacobis[1] = &j2;
        m_jacobis[2] = &j3;
    }
};

class ABDJacobiStack4 : public ABDJacobiStack<4>
{
  public:
    MUDA_HOST MUDA_DEVICE ABDJacobiStack4(const ABDJacobi& j1,
                                          const ABDJacobi& j2,
                                          const ABDJacobi& j3,
                                          const ABDJacobi& j4)
    {
        m_jacobis[0] = &j1;
        m_jacobis[1] = &j2;
        m_jacobis[2] = &j3;
        m_jacobis[3] = &j4;
    }
};
//tex:
// $$
//\mathbf{g}^{\text{Affine}}_k = \sum_{i\in \mathscr{C}_k \cap \mathscr{A}}
//\mathbf{J}_i^T \frac{\partial B}{\partial\mathbf{x}_i}
//= \sum_{i\in \mathscr{C}_k \cap \mathscr{A}}
//
//\begin{bmatrix}
//g_{1}\\
//g_{2}\\
//g_{3}\\
//\hline
//
//\bar{x}_1 g_{1}\\
//\bar{x}_2 g_{1}\\
//\bar{x}_3 g_{1}\\
//\hdashline
//
//\bar{x}_1 g_{2}\\
//\bar{x}_2 g_{2}\\
//\bar{x}_3 g_{2}\\
//\hdashline
//
//\bar{x}_1 g_{3}\\
//\bar{x}_2 g_{3}\\
//\bar{x}_3 g_{3}
//
//\end{bmatrix}_{i}
//
//=
//\sum_{i\in \mathscr{C}_k \cap \mathscr{A}}
//
//\begin{bmatrix}
//\mathbf{g}\\
//\hline
//
//g_{1} \bar{\mathbf{x}}\\
//\hdashline
//
//g_{2} \bar{\mathbf{x}}\\
//\hdashline
//
//g_{3} \bar{\mathbf{x}}\\
//
//\end{bmatrix}_{i}
// $$

//tex:
// where $\mathscr{C}_k$ is the $k$-th contact pair, and $\mathscr{A}$ represents the point set of all affine bodies.
//

//tex: $$\mathbf{J}^T\mathbf{M}_i\mathbf{J} $$
class ABDJacobiDyadicMass
{
  public:
    MUDA_HOST MUDA_DEVICE ABDJacobiDyadicMass()
        : m_mass(0)
        , m_mass_times_x_bar(Vector3::Zero())
        , m_mass_times_dyadic_x_bar(Matrix3x3::Zero())
    {
    }

    MUDA_HOST MUDA_DEVICE static ABDJacobiDyadicMass from_dyadic_mass(
        Float            sum_m,
        const Vector3&   sum_m_x_bar,
        const Matrix3x3& sum_m_x_bar_x_bar)
    {
        ABDJacobiDyadicMass ret;
        ret.m_mass                    = sum_m;
        ret.m_mass_times_x_bar        = sum_m_x_bar;
        ret.m_mass_times_dyadic_x_bar = sum_m_x_bar_x_bar;
        return ret;
    }

    MUDA_HOST MUDA_DEVICE ABDJacobiDyadicMass(Float node_mass, const Vector3& x_bar)
        : m_mass(node_mass)
        , m_mass_times_x_bar(node_mass * x_bar)
        , m_mass_times_dyadic_x_bar((node_mass * x_bar) * x_bar.transpose())
    {
    }

    MUDA_HOST MUDA_DEVICE friend Vector12 operator*(const ABDJacobiDyadicMass& mJTJ,
                                                    const Vector12&            p);

    MUDA_HOST MUDA_DEVICE ABDJacobiDyadicMass& operator+=(const ABDJacobiDyadicMass& rhs);

    MUDA_HOST MUDA_DEVICE void add_to(Matrix12x12& h) const
    {
        h(0, 0) += m_mass;
        h.block<1, 3>(0, 3) += m_mass_times_x_bar.transpose();
        h.block<3, 1>(3, 0) += m_mass_times_x_bar;

        h(1, 1) += m_mass;
        h.block<1, 3>(1, 6) += m_mass_times_x_bar.transpose();
        h.block<3, 1>(6, 1) += m_mass_times_x_bar;

        h(2, 2) += m_mass;
        h.block<1, 3>(2, 9) += m_mass_times_x_bar.transpose();
        h.block<3, 1>(9, 2) += m_mass_times_x_bar;

        h.block<3, 3>(3, 3) += m_mass_times_dyadic_x_bar;
        h.block<3, 3>(6, 6) += m_mass_times_dyadic_x_bar;
        h.block<3, 3>(9, 9) += m_mass_times_dyadic_x_bar;
    }

    MUDA_HOST MUDA_DEVICE Matrix12x12 to_mat() const
    {
        Matrix12x12 h = Matrix12x12::Zero();
        add_to(h);
        return h;
    }

    MUDA_HOST MUDA_DEVICE Float mass() const { return m_mass; }

    /**
     * @brief Inertia tensor about center of mass (3x3).
     * Derived from second moment about origin: I_cm = I^O - m(|c|^2 I_3 - c c^T),
     * I^O = tr(S) I_3 - S, with c = m_x_bar/m and S = m_x_bar_x_bar.
     * Returns zero matrix if mass is zero.
     */
    MUDA_HOST MUDA_DEVICE Matrix3x3 inertia_tensor_cm() const;

    static MUDA_HOST MUDA_DEVICE auto zero() { return ABDJacobiDyadicMass{}; }

    static MUDA_DEVICE ABDJacobiDyadicMass atomic_add(ABDJacobiDyadicMass& dst,
                                                      const ABDJacobiDyadicMass& src);

  private:
    Float m_mass;
    //tex: $$ m\bar{\mathbf{x}} $$
    Vector3 m_mass_times_x_bar;
    //tex: $$ m\bar{\mathbf{x}} \otimes \bar{\mathbf{x}} $$
    Matrix3x3 m_mass_times_dyadic_x_bar;
};
}  // namespace uipc::backend::cuda

namespace muda
{
template <>
struct force_trivially_destructible<uipc::backend::cuda::ABDJacobi>
{
    constexpr static bool value = true;
};

template <>
struct force_trivially_constructible<uipc::backend::cuda::ABDJacobi>
{
    constexpr static bool value = true;
};

template <>
struct force_trivially_copy_constructible<uipc::backend::cuda::ABDJacobi>
{
    constexpr static bool value = true;
};

template <>
struct force_trivially_copy_assignable<uipc::backend::cuda::ABDJacobi>
{
    constexpr static bool value = true;
};

template <>
struct force_trivially_destructible<uipc::backend::cuda::ABDJacobiDyadicMass>
{
    constexpr static bool value = true;
};

template <>
struct force_trivially_constructible<uipc::backend::cuda::ABDJacobiDyadicMass>
{
    constexpr static bool value = true;
};

template <>
struct force_trivially_copy_constructible<uipc::backend::cuda::ABDJacobiDyadicMass>
{
    constexpr static bool value = true;
};

template <>
struct force_trivially_copy_assignable<uipc::backend::cuda::ABDJacobiDyadicMass>
{
    constexpr static bool value = true;
};
}  // namespace muda

#include "details/abd_jacobi_matrix.inl"