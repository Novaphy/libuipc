namespace uipc::backend::cuda
{
template <size_t N>
MUDA_HOST MUDA_DEVICE Vector<Float, 3 * N> ABDJacobiStack<N>::operator*(const Vector12& q) const
{
    Vector<Float, 3 * N> ret;
#pragma unroll
    for(size_t i = 0; i < N; ++i)
    {
        ret.segment<3>(3 * i) = (*m_jacobis[i]) * q;
    }
    return ret;
}

template <size_t N>
MUDA_HOST MUDA_DEVICE Matrix<Float, 3 * N, 12> ABDJacobiStack<N>::to_mat() const
{
    Matrix<Float, 3 * N, 12> ret;
    for(size_t i = 0; i < N; ++i)
    {
        ret.block<3, 12>(3 * i, 0) = m_jacobis[i]->to_mat();
    }
    return ret;
}

template <size_t N>
MUDA_HOST MUDA_DEVICE Vector12
ABDJacobiStack<N>::ABDJacobiStackT::operator*(const Vector<Float, 3 * N>& g) const
{
    Vector12 ret = Vector12::Zero();
#pragma unroll
    for(size_t i = 0; i < N; ++i)
    {
        const ABDJacobi* jacobi = m_origin.m_jacobis[i];
        ret += jacobi->T() * g.segment<3>(3 * i);
    }
    return ret;
}

// ---------------------------------------------------------------------------
// ABDJacobi / dyadic mass: definitions in this .inl so Corex+clang (no RDC) emits device
// code in every translation unit that includes abd_jacobi_matrix.h.
// ---------------------------------------------------------------------------
MUDA_HOST MUDA_DEVICE inline Vector12 operator*(const ABDJacobi::ABDJacobiT& j, const Vector3& g)
{
    Vector12    g12;
    const auto& x     = j.m_j.m_x_bar;
    g12.segment<3>(0) = g;
    g12.segment<3>(3) = x * g.x();
    g12.segment<3>(6) = x * g.y();
    g12.segment<3>(9) = x * g.z();
    return g12;
}

MUDA_HOST MUDA_DEVICE inline Vector3 operator*(const ABDJacobi& j, const Vector12& q)
{
    const auto& t  = q.segment<3>(0);
    const auto& a1 = q.segment<3>(3);
    const auto& a2 = q.segment<3>(6);
    const auto& a3 = q.segment<3>(9);
    const auto& x  = j.m_x_bar;
    return Vector3{x.dot(a1), x.dot(a2), x.dot(a3)} + t;
}

MUDA_HOST MUDA_DEVICE inline Vector3 ABDJacobi::vec_x(const Vector12& q) const
{
    const auto& a1 = q.segment<3>(3);
    const auto& a2 = q.segment<3>(6);
    const auto& a3 = q.segment<3>(9);
    const auto& x  = m_x_bar;
    return Vector3{x.dot(a1), x.dot(a2), x.dot(a3)};
}

MUDA_HOST MUDA_DEVICE inline Matrix3x12 ABDJacobi::to_mat() const
{
    Matrix3x12  ret       = Matrix3x12::Zero();
    const auto& x         = m_x_bar;
    ret(0, 0)             = 1;
    ret(1, 1)             = 1;
    ret(2, 2)             = 1;
    ret.block<1, 3>(0, 3) = x.transpose();
    ret.block<1, 3>(1, 6) = x.transpose();
    ret.block<1, 3>(2, 9) = x.transpose();
    return ret;
}

MUDA_HOST MUDA_DEVICE inline Matrix12x12 ABDJacobi::JT_H_J(const ABDJacobiT& lhs_J_T,
                                                           const Matrix3x3&  Hessian,
                                                           const ABDJacobi&  rhs_J)
{
    Matrix12x12 ret       = Matrix12x12::Zero();
    auto        x         = lhs_J_T.J().x_bar();
    auto        y         = rhs_J.x_bar();
    ret.block<3, 3>(0, 0) = Hessian;

    ret.block<3, 3>(0, 3) = Hessian.block<3, 1>(0, 0) * y.transpose();
    ret.block<3, 3>(0, 6) = Hessian.block<3, 1>(0, 1) * y.transpose();
    ret.block<3, 3>(0, 9) = Hessian.block<3, 1>(0, 2) * y.transpose();

    ret.block<3, 3>(3, 0) = x * Hessian.block<1, 3>(0, 0);
    ret.block<3, 3>(6, 0) = x * Hessian.block<1, 3>(1, 0);
    ret.block<3, 3>(9, 0) = x * Hessian.block<1, 3>(2, 0);

    Matrix3x3 x_y = x * y.transpose();

    ret.block<3, 3>(3, 3) = x_y * Hessian(0, 0);
    ret.block<3, 3>(3, 6) = x_y * Hessian(0, 1);
    ret.block<3, 3>(3, 9) = x_y * Hessian(0, 2);

    ret.block<3, 3>(6, 3) = x_y * Hessian(1, 0);
    ret.block<3, 3>(6, 6) = x_y * Hessian(1, 1);
    ret.block<3, 3>(6, 9) = x_y * Hessian(1, 2);

    ret.block<3, 3>(9, 3) = x_y * Hessian(2, 0);
    ret.block<3, 3>(9, 6) = x_y * Hessian(2, 1);
    ret.block<3, 3>(9, 9) = x_y * Hessian(2, 2);

    return ret;
}

MUDA_HOST MUDA_DEVICE inline Vector12 operator*(const ABDJacobiDyadicMass& JTJ, const Vector12& p)
{
    Vector12    ret;
    const auto& m = JTJ.m_mass;
    const auto& D = JTJ.m_mass_times_dyadic_x_bar;
    const auto& x = JTJ.m_mass_times_x_bar;

    const auto& p_p  = p.segment<3>(0);
    const auto& p_a1 = p.segment<3>(3);
    const auto& p_a2 = p.segment<3>(6);
    const auto& p_a3 = p.segment<3>(9);

    ret(0) = x.dot(p_a1) + m * p_p.x();
    ret(1) = x.dot(p_a2) + m * p_p.y();
    ret(2) = x.dot(p_a3) + m * p_p.z();

    ret.segment<3>(3) = D * p_a1 + x * p_p.x();
    ret.segment<3>(6) = D * p_a2 + x * p_p.y();
    ret.segment<3>(9) = D * p_a3 + x * p_p.z();

    return ret;
}

MUDA_HOST MUDA_DEVICE inline ABDJacobiDyadicMass& ABDJacobiDyadicMass::operator+=(
    const ABDJacobiDyadicMass& rhs)
{
    m_mass += rhs.m_mass;
    m_mass_times_x_bar += rhs.m_mass_times_x_bar;
    m_mass_times_dyadic_x_bar += rhs.m_mass_times_dyadic_x_bar;
    return *this;
}

MUDA_HOST MUDA_DEVICE inline Matrix3x3 ABDJacobiDyadicMass::inertia_tensor_cm() const
{
    const Float m = m_mass;
    if(m <= 0)
        return Matrix3x3::Zero();
    const Vector3   c = m_mass_times_x_bar / m;
    const Matrix3x3 S = m_mass_times_dyadic_x_bar;
    const Float     trS      = S.trace();
    const Matrix3x3 I_origin = trS * Matrix3x3::Identity() - S;
    const Float     c2       = c.squaredNorm();
    return I_origin - m * (c2 * Matrix3x3::Identity() - c * c.transpose());
}

// ABDJacobiDyadicMass::add_to/to_mat are inline in abd_jacobi_matrix.h; atomic_add stays in .cu
// to provide a concrete exported symbol used by host-side paths.
}  // namespace uipc::backend::cuda