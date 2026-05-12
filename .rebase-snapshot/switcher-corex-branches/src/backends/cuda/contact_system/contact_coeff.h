#pragma once
#include <type_define.h>

namespace uipc::backend::cuda
{
class ContactCoeff
{
  public:
    // normal stiffness
    Float kappa = 0.0;
    // friction coefficient
    Float mu = 0.0;
};
}  // namespace uipc::backend::cuda

namespace muda
{
template <>
struct force_trivially_destructible<uipc::backend::cuda::ContactCoeff>
{
    constexpr static bool value = true;
};

template <>
struct force_trivially_constructible<uipc::backend::cuda::ContactCoeff>
{
    constexpr static bool value = true;
};

template <>
struct force_trivially_copy_constructible<uipc::backend::cuda::ContactCoeff>
{
    constexpr static bool value = true;
};

template <>
struct force_trivially_copy_assignable<uipc::backend::cuda::ContactCoeff>
{
    constexpr static bool value = true;
};
}  // namespace muda
